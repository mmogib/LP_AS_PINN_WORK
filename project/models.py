import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from colorama import just_fix_windows_console, init, Fore, Style
from typing import List, Callable
from prettytable import PrettyTable, ALL


class PINN_B(nn.Module):
    def __init__(self, out_dim=5):
        super(PINN, self).__init__()
        # Input dimension becomes 2: one from t and one from the pooled b.
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, out_dim)
        self.activation = nn.Tanh()
        self.out_dim = out_dim

    def forward(self, t, b):
        """
        Forward pass for inputs t and b.
          - t: tensor of shape (N, 1) (or scalar, which will be unsqueezed)
          - b: tensor of variable length. For a batch, we expect shape (N, L)
               where L can vary across different calls but should be >0.

        We average b over its variable dimension to get a fixed-size (N, 1) tensor,
        then concatenate it with t. The output is modulated as:
            ŷ(t) = (1 - exp(-t)) * NN([t, pooled_b])
        to enforce ŷ(0)=0.
        """
        # Ensure t is batched (if given as a scalar, make it (1, 1))
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # Ensure b is batched:
        if b.dim() == 1:
            b = b.unsqueeze(0)  # now shape (1, L)

        # Average b over its last dimension (variable-length dimension)
        b_avg = torch.mean(b, dim=1, keepdim=True)  # shape: (N, 1)

        # Concatenate t and b_avg along feature dimension
        input_tensor = torch.cat([t, b_avg], dim=1)  # shape: (N, 2)

        x = self.activation(self.fc1(input_tensor))
        out = self.fc2(x)
        # The modulation factor uses t; note that t has shape (N, 1)
        return (1 - torch.exp(-t)) * out


# ------------------------------
# Define the PINN model (moved to GPU)
# ------------------------------
class PINN(nn.Module):
    def __init__(self, out_dim=5):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, out_dim)
        self.activation = nn.Tanh()
        self.out_dim = out_dim

    def forward(self, t):
        """
        Forward pass for a scalar (or batch) time input.
        We assume t is a tensor of shape (N, 1) or a scalar tensor.
        The network output is modulated as:
            ŷ(t) = (1 - exp(-t)) * NN(t)
        to enforce ŷ(0) = 0.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        x = self.activation(self.fc1(t))
        out = self.fc2(x)
        return (1 - torch.exp(-t)) * out


# ------------------------------
# Define the vectorized loss function using functorch
# ------------------------------
def create_loss(
    model: PINN,
    ts: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
    *,
    bs: torch.Tensor | None = None,
):

    def loss_t():
        # ts: shape (N,) ; we need to work with scalar inputs, so we keep ts as a 1D tensor.
        # Evaluate the PINN on all collocation points.
        # The model expects input shape (N,1), so unsqueeze ts.
        ts_var = ts.clone().detach().requires_grad_(True)  # shape (N,)
        y_hat = model(ts_var.unsqueeze(1))

        def model_single(t):
            return model(t.unsqueeze(0)).squeeze(0)

        dy_dt = torch.vmap(torch.func.jacrev(model_single))(ts_var)  # shape: (N, 5)

        phi_y = torch.vmap(phi)(y_hat)

        residuals = dy_dt - phi_y

        loss = torch.mean(torch.sum(residuals**2, dim=1))
        return loss

    def loss_b():
        N = ts.shape[0]  # number of time points
        M = bs.shape[0]  # number of parameter sets

        # Expand ts and bs to create (N*M) pairs
        ts_expanded = ts.repeat_interleave(M)  # shape: (N*M,)
        bs_expanded = bs.repeat(N, 1)  # shape: (N*M, L)

        # Unsqueeze ts_expanded to shape (N*M, 1) for model input
        ts_var = ts_expanded.unsqueeze(1).clone().detach().requires_grad_(True)

        # Evaluate the model -> shape (N*M, out_dim)
        y_hat = model(ts_var, bs_expanded)

        # Compute derivative wrt t for each sample in a loop (simpler to read)
        # For better performance, you might use functorch's jacrev + vmap if supported.
        out_dim = y_hat.shape[1]
        dy_dt = torch.zeros_like(y_hat)

        for i in range(N * M):
            # Single sample
            t_val = ts_var[i : i + 1]  # shape (1,1)
            b_val = bs_expanded[i : i + 1]  # shape (1, L)

            # Define a single-sample closure for autograd
            def single_sample(t_s):
                return model(t_s, b_val).squeeze(0)  # shape (out_dim,)

            # Use autograd to compute derivative wrt t
            _, back_fn = torch.autograd.functional._jvp(
                single_sample,  # function
                (t_val,),  # primal
                (torch.ones_like(t_val),),  # tangent
            )
            # back_fn is the directional derivative wrt t, shape (out_dim,)
            dy_dt[i] = back_fn

            # Apply phi -> shape (N*M, out_dim)
            phi_y = phi(y_hat, bs_expanded)

            # Residual = dy_dt - phi_y
            residuals = dy_dt - phi_y

            # Mean of sum of squares -> effectively 1/(N*M) * sum ||residuals||^2
            sq_norm = torch.sum(residuals**2, dim=1)  # shape: (N*M,)
            loss_val = torch.mean(sq_norm)
            return loss_val

    loss_fn = loss_t if bs is None else loss_b
    return loss_fn


# Save the model's state dictionary to a file


# Get the current date in YYYY_MM_DD format
def save_model(model: PINN, name: str = "pinn_model"):
    current_date = datetime.date.today().strftime("%Y_%m_%d")

    # Create the filename with the date stamp
    filename = f"{name}_{current_date}_out_dim_{model.out_dim}.pt"

    # Save the model state dictionary
    torch.save(model.state_dict(), filename)

    print(f"Model saved to {filename}")


def train_model(
    phi,
    objective_fun,
    ts_batches,
    b_list,
    device,
    T=10.0,
    lr=0.001,
    epochs=1000,
    tol=1e-5,
    out_dim=5,
):
    just_fix_windows_console()
    init()

    model = PINN(out_dim=out_dim).to(device)
    # ------------------------------
    # Training Loop using Adam (lr = 0.001) on the GPU
    # ------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(Fore.RED + "Starting training..." + Style.RESET_ALL)
    batches = len(ts_batches)
    loss_list = []
    MOV = 0
    no_of_bs = len(b_list)
    for rhs, b in enumerate(b_list, start=1):
        for batch, ts in enumerate(ts_batches, start=1):
            compute_loss_vectorized = create_loss(model, ts, phi(b))
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                loss_val = compute_loss_vectorized()
                loss_val.backward()
                optimizer.step()

                loss_valued = loss_val.item()

                loss_list.append(loss_valued)

                # Print loss every 10 epochs.
                if epoch % 100 == 0:
                    print(
                        Fore.MAGENTA
                        + f"(RHS: {rhs}) Batch {batch}/{batches},  Epoch {epoch}/{epochs}: Loss = {loss_valued:.6f}"
                        + Style.RESET_ALL
                    )
                # Stop if loss is below the threshold.
                if loss_valued < tol:
                    print(
                        Fore.GREEN
                        + f"Stopping training at epoch {epoch} with loss = {loss_valued:.6f}"
                        + Style.RESET_ALL
                    )
                    break
            y_pred = model(
                torch.tensor(T, dtype=torch.float32, device=device, requires_grad=True)
            )
            val = objective_fun(y_pred)
            MOV = MOV + val.item()

    print(Fore.BLUE + "Training complete.\n" + Style.RESET_ALL)
    return model, loss_list, MOV / (batches * len(b_list))


def load_model(model, device: torch.DeviceObjType, filename):
    model.load_state_dict(torch.load(filename, map_location=device))
    return model


def test_model(
    model: PINN,
    test_times: List[int],
    *,
    dev: torch.DeviceObjType = torch.device("cpu"),
):
    table = PrettyTable()
    table.field_names = ["t", "ŷ(t)"]
    # Set horizontal rules between every row
    table.hrules = ALL
    # Left-align each column
    table.align["t"] = "l"
    table.align["ŷ(t)"] = "l"
    for t in test_times:
        t_tensor = torch.tensor(t, dtype=torch.float32, device=dev, requires_grad=True)
        y_pred = model(t_tensor)
        # Flatten the tensor and convert to a formatted string
        y_pred_str = str(y_pred.detach().cpu().numpy().flatten())
        table.add_row([f"{t:6.2f}", y_pred_str])

    # Print the entire table with color formatting
    print(Fore.MAGENTA + table.get_string() + Style.RESET_ALL)
