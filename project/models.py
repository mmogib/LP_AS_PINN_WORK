import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from colorama import just_fix_windows_console, init, Fore, Style
from typing import List, Callable
from prettytable import PrettyTable, ALL


# ------------------------------
# Define the PINN model (moved to GPU)
# ------------------------------
class PINN(nn.Module):
    def __init__(self, out_vars=5):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, out_vars)
        self.activation = nn.Tanh()
        self.out_vars = out_vars

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
    model: PINN, ts: torch.Tensor, phi: Callable[[torch.Tensor], torch.Tensor]
):
    def compute_loss_vectorized():
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

    return compute_loss_vectorized


# Save the model's state dictionary to a file


# Get the current date in YYYY_MM_DD format
def save_model(model: PINN, name: str = "pinn_model"):
    current_date = datetime.date.today().strftime("%Y_%m_%d")

    # Create the filename with the date stamp
    filename = f"{name}_{current_date}_out_vars_{model.out_vars}.pt"

    # Save the model state dictionary
    torch.save(model.state_dict(), filename)

    print(f"Model saved to {filename}")


def train_model(phi, ts, device, lr=0.001, epochs=1000, out_vars=5):
    just_fix_windows_console()
    init()

    model = PINN(out_vars=out_vars).to(device)
    compute_loss_vectorized = create_loss(model, ts, phi)
    # ------------------------------
    # Training Loop using Adam (lr = 0.001) on the GPU
    # ------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(Fore.RED + "Starting training..." + Style.RESET_ALL)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_val = compute_loss_vectorized()
        loss_val.backward()
        optimizer.step()

        # Print loss every 10 epochs.
        if epoch % 10 == 0:
            print(
                Fore.MAGENTA
                + f"Epoch {epoch}/{epochs}: Loss = {loss_val.item():.6f}"
                + Style.RESET_ALL
            )

        # Stop if loss is below the threshold.
        if loss_val.item() < 1e-4:
            print(
                Fore.GREEN
                + f"Stopping training at epoch {epoch} with loss = {loss_val.item():.6f}"
                + Style.RESET_ALL
            )
            break

    print(Fore.BLUE + "Training complete.\n" + Style.RESET_ALL)
    return model


def load_model(model, device, filename):
    model.load_state_dict(torch.load(filename, map_location=device))
    return model


def test_model(model: "PINN", test_times: List[int], device=torch.device("cpu")):
    table = PrettyTable()
    table.field_names = ["t", "ŷ(t)"]
    # Set horizontal rules between every row
    table.hrules = ALL
    # Left-align each column
    table.align["t"] = "l"
    table.align["ŷ(t)"] = "l"
    for t in test_times:
        t_tensor = torch.tensor(
            t, dtype=torch.float32, device=device, requires_grad=True
        )
        y_pred = model(t_tensor)
        # Flatten the tensor and convert to a formatted string
        y_pred_str = str(y_pred.detach().cpu().numpy().flatten())
        table.add_row([f"{t:6.2f}", y_pred_str])

    # Print the entire table with color formatting
    print(Fore.MAGENTA + table.get_string() + Style.RESET_ALL)
