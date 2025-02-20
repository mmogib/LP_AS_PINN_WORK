import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian
from functorch import vmap, jacrev
import matplotlib.pyplot as plt


# ------------------------------
# Set up the device: GPU if available, otherwise CPU
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Define ODE constants and the ODE function φ on the selected device
# ------------------------------
D = torch.tensor([-9.54, -8.16, -4.26, -11.43], dtype=torch.float32, device=device)
A = torch.tensor([[3.18, 2.72, 1.42, 3.81]], dtype=torch.float32, device=device)
b = torch.tensor([[7.81]], dtype=torch.float32, device=device)


def createPhi(D, A, b):
    _, n = A.shape

    def phi(y):

        x = y[:n]
        u = y[n:]

        m = torch.clamp(u + A @ x - b, min=0.0)

        top = -(D.unsqueeze(1) + A.t() @ m)  # shape: (4,1)

        bottom = m - u

        return torch.cat((top, bottom), dim=0).squeeze(1)

    return phi


phi = createPhi(D, A, b)


# ------------------------------
# Define the PINN model (moved to GPU)
# ------------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 5)
        self.activation = nn.Tanh()

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


model = PINN().to(device)

# ------------------------------
# Set up collocation points on the device
# ------------------------------
ts = torch.linspace(0, 10, 128, dtype=torch.float32, device=device)


# ------------------------------
# Define the vectorized loss function using functorch
# ------------------------------
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


# ------------------------------
# Training Loop using Adam (lr = 0.001) on the GPU
# ------------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500000

print("Starting training...")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss_val = compute_loss_vectorized()
    loss_val.backward()
    optimizer.step()

    # Print loss every 10 epochs.
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss_val.item():.6f}")

    # Stop if loss is below the threshold.
    if loss_val.item() < 1e-4:
        print(f"Stopping training at epoch {epoch} with loss = {loss_val.item():.6f}")
        break

print("Training complete.\n")

# ------------------------------
# Evaluate the trained model at select time points
# ------------------------------
test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
for t in test_times:
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device, requires_grad=True)
    y_pred = model(t_tensor)
    print(f"t = {t:4.1f}, ŷ(t) = {y_pred.detach().cpu().numpy().flatten()}")
