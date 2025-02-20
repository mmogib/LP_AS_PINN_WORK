import torch
from device import get_device
from ode import createPhi
from models import save_model, train_model, test_model

device = get_device()


# ------------------------------
# (E1) Define ODE constants and the ODE function φ on the selected device
# ------------------------------
D1 = torch.tensor([-9.54, -8.16, -4.26, -11.43], dtype=torch.float32, device=device)
A1 = torch.tensor([[3.18, 2.72, 1.42, 3.81]], dtype=torch.float32, device=device)
b1 = torch.tensor([[7.81]], dtype=torch.float32, device=device)

# Objective: minimize D^T x, with D = [-3, -1, -3]^T
D2 = torch.tensor([-3, -1, -3], dtype=torch.float32, device=device)
A2 = torch.tensor(
    [[2, 1, 1], [1, 2, 3], [2, 2, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    dtype=torch.float32,
    device=device,
)
b2 = torch.tensor([2, 5, 6, 0, 0, 0], dtype=torch.float32, device=device)

# Example 3
D3 = torch.tensor(
    [-1.0, -4.0, -3.0],
    dtype=torch.float32,
    device=device,
)
A3 = torch.tensor(
    [
        [2.0, 2.0, 1.0],  # 2x1 + 2x2 + x3 <= 4
        [1.0, 2.0, 2.0],  # x1 + 2x2 + 2x3 <= 6
        [-1.0, 0.0, 0.0],  # -x1 <= 0
        [0.0, -1.0, 0.0],  # -x2 <= 0
        [0.0, 0.0, -1.0],  # -x3 <= 0
    ],
    dtype=torch.float32,
    device=device,
)

b3 = torch.tensor(
    [4.0, 6.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
    device=device,
)
# # ------------------------------
# # Set up collocation points on the device
# # ------------------------------
ts = torch.linspace(0, 10, 128, dtype=torch.float32, device=device)

phi1 = createPhi(D1, A1, b1)
phi2 = createPhi(D2, A2, b2)
phi3 = createPhi(D3, A3, b3)
epochs = 1000
out_vars = sum(A3.shape)
model = train_model(phi3, ts, device, lr=0.001, epochs=epochs, out_vars=out_vars)

# ------------------------------
# Evaluate the trained model at select time points
# ------------------------------
test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
test_model(model, test_times, device=device)

save_model(model, f"pinn_{epochs}")


# import numpy as np
# from scipy.optimize import linprog

# # Define the objective vector D, constraint matrix A, and RHS vector b
# D = np.array([-1.0, -4.0, -3.0])
# A = np.array(
#     [
#         [2.0, 2.0, 1.0],  # 2x1 + 2x2 + x3 <= 4
#         [1.0, 2.0, 2.0],  # x1 + 2x2 + 2x3 <= 6
#         [-1.0, 0.0, 0.0],  # -x1 <= 0  (i.e., x1 >= 0)
#         [0.0, -1.0, 0.0],  # -x2 <= 0  (i.e., x2 >= 0)
#         [0.0, 0.0, -1.0],  # -x3 <= 0  (i.e., x3 >= 0)
#     ]
# )
# b = np.array([4.0, 6.0, 0.0, 0.0, 0.0])

# # Solve the LP using scipy.optimize.linprog
# result = linprog(c=D, A_ub=A, b_ub=b, method="highs-ds")

# # Check if the optimization was successful and print the results
# if result.success:
#     print("Optimal solution:", result.x)
#     print("Optimal objective value:", result.fun)
#     print("Optimal objective status:", result.status)
#     print("Optimal objective slack:", result.slack)
# else:
#     print("Optimization failed:", result.message)
