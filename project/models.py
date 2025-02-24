from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.utils.data import Dataset, DataLoader
import datetime
from colorama import just_fix_windows_console, init, Fore, Style
from typing import List, Callable, Optional, Tuple
from prettytable import PrettyTable, ALL

from ode import createObjectiveFun, createPhi


class PINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=5):
        super(PINN, self).__init__()
        # Input dimension becomes 2: one from t and one from the pooled b.
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, out_dim)
        self.activation = nn.Tanh()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, tb):
        # If tb is a scalar, convert it to a 2D tensor with shape (1, in_dim)
        if tb.dim() == 0:
            tb = tb.unsqueeze(0).unsqueeze(1)
        elif tb.dim() == 1:
            tb = tb.unsqueeze(1)
        t = tb[:, 0].unsqueeze(1)  # Now safe because tb is at least 2D.
        x = self.activation(self.fc1(tb))
        out = self.fc2(x)
        return (1 - torch.exp(-t)) * out


# ------------------------------
# Define the vectorized loss function using functorch
# ------------------------------
def create_loss(
    model: PINN,
    tb: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
):

    def loss_t():
        # ts: shape (N,) ; we need to work with scalar inputs, so we keep ts as a 1D tensor.
        # Evaluate the PINN on all collocation points.
        # The model expects input shape (N,1), so unsqueeze ts.
        ts_var = tb.clone().detach().requires_grad_(True)  # shape (N,)
        y_hat = model(ts_var.unsqueeze(1))

        def model_single(t):
            return model(t.unsqueeze(0)).squeeze(0)

        dy_dt = torch.vmap(torch.func.jacrev(model_single))(ts_var)  # shape: (N, 5)

        phi_y = torch.vmap(phi)(y_hat.squeeze(0))

        residuals = dy_dt - phi_y

        loss = torch.mean(torch.sum(residuals**2, dim=1))
        return loss

    def loss_b():
        """
        Compute the mean squared loss:
            loss = mean( || d/dt y_hat(tb) - phi(y_hat(tb), b) ||^2 )
        where tb is of shape (N, L) with L>=2:
        - tb[:, 0] contains time t (shape (N,1))
        - tb[:, 1:] contains b (shape (N, L-1))

        The function phi, when called as phi(b) with b of shape (L-1,), returns a function
        that accepts a single sample y (of shape (out_dim,)) and returns a tensor of shape (out_dim,).
        """
        # Ensure tb is differentiable.
        tb_var = tb.clone().detach().requires_grad_(True)  # shape: (N, L)

        # Evaluate model: y_hat will have shape (N, out_dim)
        y_hat = model(tb_var)
        # print("y_hat shape:", y_hat.shape)
        # print("y_hat:", y_hat)

        N = tb.shape[0]
        dy_dt = torch.zeros_like(y_hat)  # to store the derivative with respect to t

        # For each sample, compute the derivative of the model output with respect to t.
        for i in range(N):
            # Extract t: first column of tb_var for sample i.
            # t_val: shape (1, 1)
            t_val = tb_var[i, 0].unsqueeze(0).unsqueeze(1)
            # Extract b: remaining columns of tb_var for sample i.
            # b_val: shape (1, L-1)
            b_val = tb_var[i, 1:].unsqueeze(0)

            def single_sample(t_s):
                # t_s has shape (1,1). Concatenate with fixed b_val to form a full sample of shape (1, L)
                sample_input = torch.cat([t_s, b_val], dim=1)
                # Call the model and remove the batch dimension.
                return model(sample_input).squeeze(0)  # shape: (out_dim,)

            # Compute the directional derivative (Jacobian-vector product) with respect to t.
            # Here we compute d/dt at t_val in the direction of 1.
            _, jvp = torch.autograd.functional.jvp(
                single_sample, (t_val,), (torch.ones_like(t_val),)
            )
            dy_dt[i] = jvp  # shape (out_dim,)

        # Now, for each sample, apply the physics operator.
        phi_y_list = []
        for i in range(N):
            # For sample i, b_i is the parameter vector: shape (L-1,)
            b_i = tb_var[i, 1:]
            # Get the phi operator for this b.
            phi_func = phi(b_i)
            # Apply it to the model output for sample i.
            phi_y_list.append(phi_func(y_hat[i]))
        # Stack the results: shape (N, out_dim)
        phi_y = torch.stack(phi_y_list, dim=0)

        # Compute the residual and then the mean squared error.
        residuals = dy_dt - phi_y
        loss_val = torch.mean(torch.sum(residuals**2, dim=1))
        return loss_val

    loss_fn = loss_t if tb.shape[1] == 1 else loss_b
    return loss_fn


# Save the model's state dictionary to a file


def create_batches(
    *,
    batch_size: int = 128,
    no_batches: int = 1,
    tspan: Tuple[float, float] = (0.0, 10.0),
    in_dim: int = 5,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    # Create a list of time batches: each tensor has shape (batch_size, 1)
    ts_batches = [
        torch.empty(batch_size, dtype=torch.float32, device=device)
        .uniform_(*tspan)
        .unsqueeze(1)
        for _ in range(no_batches)
    ]

    if in_dim == 1:
        # Return a list of tuples (ts, None)
        return [(ts, None) for ts in ts_batches]
    else:
        # Create a list of b batches: each tensor has shape (batch_size, in_dim - 1)
        b_list = [
            torch.empty(
                (batch_size, in_dim - 1), dtype=torch.float32, device=device
            ).uniform_(*tspan)
            for _ in range(no_batches)
        ]

        # Zip the two lists together into a list of tuples (ts, b)
        return list(zip(ts_batches, b_list))


def load_model(model, device: torch.DeviceObjType, filename):
    model.load_state_dict(torch.load(filename, map_location=device))
    return model


def train_model(A, b, D, name, tspan, rhs, epochs, batch_size, batches, device):

    b_raw = b.tolist()[0]
    phi = createPhi(D, A)
    out_dim = sum(A.shape)
    if rhs == 0:
        in_dim = 1
        model, loss_list, mov_list = _train_model(
            phi(b),
            createObjectiveFun(D),
            tspan=tspan,
            in_dim=in_dim,
            out_dim=out_dim,
            batch_size=batch_size,
            no_batches=batches,
            epochs=epochs,
            lr=0.001,
            tol=1e-3,
            training_name=name,
            device=device,
        )
    else:
        in_dim = len(b_raw) + 1
        model, loss_list, mov_list = _train_model(
            phi,
            createObjectiveFun(D),
            tspan=tspan,
            in_dim=in_dim,
            out_dim=out_dim,
            batch_size=batch_size,
            no_batches=batches,
            epochs=epochs,
            lr=0.001,
            tol=1e-3,
            training_name=name,
            device=device,
            testing_tb=[tspan[1], *b_raw],
        )
    return model, loss_list, mov_list


def _train_model(
    phi,  # physics operator generator: phi(b) returns a function to apply to model output.
    objective_fun,  # function that computes a scalar objective from model output.
    *,
    tspan: Tuple[float, float] = (0.0, 10.0),
    in_dim: int = 1,
    out_dim: int = 5,
    batch_size: int = 128,
    no_batches: int = 1,
    epochs: int = 1000,
    lr: float = 0.001,
    tol: float = 1e-5,
    training_name: str = "",
    device: torch.device = torch.device("cpu"),
    testing_tb: List[
        float
    ] = None,  # required when in_dim > 1; list of floats: first element is t, rest are b.
):
    just_fix_windows_console()
    init()
    T = tspan[1]
    loss_list = []
    mov_list = []

    batched_data = create_batches(
        batch_size=batch_size,
        no_batches=no_batches,
        tspan=tspan,
        in_dim=in_dim,
        device=device,
    )
    if in_dim == 1:
        t_eval = torch.tensor(
            [[T]], dtype=torch.float32, device=device, requires_grad=True
        )
    else:
        # testing_tb should be provided as a list: first element is t, rest are b.
        t_eval = torch.tensor(
            [testing_tb], dtype=torch.float32, device=device, requires_grad=True
        )
    print("Batched data created.")
    # Create the model using the provided in_dim.
    model = PINN(in_dim=in_dim, out_dim=out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(Fore.RED + f"Starting training {training_name}" + Style.RESET_ALL)
    bb = torch.empty(
        (batch_size, in_dim - 1), dtype=torch.float32, device=device
    ).uniform_(*tspan)
    epoch = 1
    while True:
        epoch_loss = 0.0
        # Use tqdm over ts_batches; we also need the index so we use enumerate.
        pbar = tqdm(
            batched_data,
            total=no_batches,
            desc=f"Epoch {epoch}/{epochs} training {no_batches} batches ",
            leave=False,
        )
        for ts, b in pbar:
            tb = ts if b is None else torch.cat([ts, bb], dim=1)
            compute_loss_vectorized = create_loss(model, tb, phi)
            optimizer.zero_grad()
            loss_val = compute_loss_vectorized()
            loss_val.backward()
            optimizer.step()
            loss_item = loss_val.item()
            epoch_loss += loss_item
            loss_list.append(loss_item)
            pbar.set_postfix(loss=f"{loss_item:.6f}")
            y_pred = model(t_eval)
            val = objective_fun(y_pred[0])
            mov_list.append(val.item())

        epoch_loss /= no_batches

        if epoch % 100 == 0:
            print(
                Fore.MAGENTA
                + f"{training_name} | {no_batches} batches, Epoch {epoch}/{epochs}: Loss = {epoch_loss:.6f}"
                + Style.RESET_ALL
            )
        if epoch_loss < tol:
            print(
                Fore.GREEN
                + f"{training_name} | Stopping training at epoch {epoch} with loss = {epoch_loss:.6f}"
                + Style.RESET_ALL
            )
            break
        if epoch == epochs:
            print(
                Fore.GREEN
                + f"{training_name} | Max iteration {epochs} reached: stopping training loss = {epoch_loss:.6f}"
                + Style.RESET_ALL
            )
            break

        epoch += 1

    # Evaluation step:

    print(Fore.BLUE + "Training complete.\n" + Style.RESET_ALL)
    return model, loss_list, mov_list


def test_model(model, dev, b_raw, rhs, T, name, epochs):
    # ------------------------------
    # Evaluate the trained model at select time points
    # ------------------------------
    name = name.lower().replace(" ", "_")
    filename = f"saved_models/{name}"
    test_times = [i * (T / 5) for i in range(6)]
    if rhs == 0:
        _test_model(model, test_times, dev=dev)
        filename = f"{filename}_with_same_rhs_pinn_{epochs}"
    else:
        b_tests = [b_raw for _ in range(6)]
        _test_model(model, test_times, dev=dev, b_tests=b_tests)
        filename = f"{filename}_with_random_rhs_pinn_{epochs}"

    return filename


def _test_model(
    model: PINN,
    test_times: List[float],
    *,
    dev: torch.DeviceObjType = torch.device("cpu"),
    b_tests: List[List[float]] | None = None,
):
    table = PrettyTable()
    table.field_names = ["t", "ŷ(t)"]
    # Set horizontal rules between every row
    table.hrules = ALL
    # Left-align each column
    table.align["t"] = "l"
    table.align["ŷ(t)"] = "l"

    for i, t in enumerate(test_times, start=1):
        if b_tests is not None:
            b = b_tests[i - 1]

            t_tensor = torch.tensor(
                [t, *b], dtype=torch.float32, device=dev, requires_grad=True
            ).unsqueeze(0)

            y_pred = model(t_tensor)
        else:
            t_tensor = torch.tensor(
                t, dtype=torch.float32, device=dev, requires_grad=True
            )
            y_pred = model(t_tensor)
        # Flatten the tensor and convert to a formatted string
        y_pred_str = str(y_pred.detach().cpu().numpy().flatten())
        table.add_row([f"{t:6.2f}", y_pred_str])

    # Print the entire table with color formatting
    print(Fore.MAGENTA + table.get_string() + Style.RESET_ALL)


# Get the current date in YYYY_MM_DD format
def save_model(model: PINN, name: str = "pinn_model"):
    current_date = datetime.date.today().strftime("%Y_%m_%d")

    # Create the filename with the date stamp
    filename = f"{name}_{current_date}_out_dim_{model.out_dim}.pt"

    # Save the model state dictionary
    torch.save(model.state_dict(), filename)

    print(f"Model saved to {filename}")
