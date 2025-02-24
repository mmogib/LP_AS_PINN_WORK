# from torch.utils.data import Dataset, DataLoader

# class PINN_B(nn.Module):
#     def __init__(self, in_dim=2, out_dim=5):
#         super(PINN_B, self).__init__()
#         # Input dimension becomes 2: one from t and one from the pooled b.
#         self.fc1 = nn.Linear(in_dim, 100)
#         self.fc2 = nn.Linear(100, out_dim)
#         self.activation = nn.Tanh()
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#     def forward(self, tb):
#         """
#         Forward pass for inputs t and b.
#           - t: tensor of shape (N, 1) (or scalar, which will be unsqueezed)
#           - b: tensor of variable length. For a batch, we expect shape (N, L)
#                where L can vary across different calls but should be >0.
#         We average b over its last dimension to get a fixed-size (N, 1) tensor,
#         then concatenate it with t. The output is modulated as:
#             ŷ(t) = (1 - exp(-t)) * NN([t, pooled_b])
#         to enforce ŷ(0)=0.
#         """
#         # Average b over its last dimension (variable-length dimension)

#         # Now both t (N,1) and b_avg (N,1) have matching batch dimensions.
#         # input_tensor = torch.cat(tb, dim=1)  # shape: (N, 2)
#         t = tb[:, 0].unsqueeze(1)
#         x = self.activation(self.fc1(tb))
#         out = self.fc2(x)
#         return (1 - torch.exp(-t)) * out

# class TsBatchDataset(Dataset):
#     def __init__(self, ts_batches):
#         self.ts_batches = ts_batches

#     def __len__(self):
#         return len(self.ts_batches)

#     def __getitem__(self, idx):
#         return self.ts_batches[idx]


# ------------------------------
# Define the PINN model (moved to GPU)
# ------------------------------
# class PINN(nn.Module):
#     def __init__(self, out_dim=5):
#         super(PINN, self).__init__()
#         self.fc1 = nn.Linear(1, 100)
#         self.fc2 = nn.Linear(100, out_dim)
#         self.activation = nn.Tanh()
#         self.out_dim = out_dim

#     def forward(self, t):
#         """
#         Forward pass for a scalar (or batch) time input.
#         We assume t is a tensor of shape (N, 1) or a scalar tensor.
#         The network output is modulated as:
#             ŷ(t) = (1 - exp(-t)) * NN(t)
#         to enforce ŷ(0) = 0.
#         """
#         if t.dim() == 0:
#             t = t.unsqueeze(0)
#         x = self.activation(self.fc1(t))
#         out = self.fc2(x)
#         return (1 - torch.exp(-t)) * out

# def train_model_case_1(
#     phi,
#     objective_fun,
#     ts_batches,
#     *,
#     device=torch.device("cpu"),
#     T=10.0,
#     lr=0.001,
#     epochs=1000,
#     tol=1e-5,
#     out_dim=5,
#     trainig_name: str = "",
#     batch_size: int = 128,
#     in_dim=1,
# ):
#     just_fix_windows_console()
#     init()
#     batches = len(ts_batches)
#     # dataset = TsBatchDataset(ts_batches)
#     # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     model = PINN(in_dim=1, out_dim=out_dim).to(device)
#     # ------------------------------
#     # Training Loop using Adam (lr = 0.001) on the GPU
#     # ------------------------------
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     print(Fore.RED + f"Starting training {trainig_name}" + Style.RESET_ALL)
#     loss_list = []
#     MOV = 0
#     for epoch in range(1, epochs + 1):
#         epoch_loss = 0
#         pbar = tqdm(
#             ts_batches,
#             desc=f"Epoch {epoch}/{epochs} training {batches} batches ",
#             leave=False,
#         )
#         for batch, ts in enumerate(pbar, start=1):
#             compute_loss_vectorized = create_loss(model, ts, phi)
#             optimizer.zero_grad()
#             loss_val = compute_loss_vectorized()
#             loss_val.backward()
#             optimizer.step()

#             loss_valued = loss_val.item()
#             epoch_loss += loss_valued
#             loss_list.append(loss_valued)
#             pbar.set_postfix(loss=f"{batch} -- {loss_valued:.6f}")
#         epoch_loss /= batches
#         # Print loss every 10 epochs.
#         if epoch % 100 == 0:
#             print(
#                 Fore.MAGENTA
#                 + f"{trainig_name} | (RHS: FIXED) {batches} batches,  Epoch {epoch}/{epochs}: Loss = {epoch_loss:.6f}"
#                 + Style.RESET_ALL
#             )
#         # Stop if loss is below the threshold.
#         if epoch_loss < tol:
#             print(
#                 Fore.GREEN
#                 + f"{trainig_name} | Stopping training at epoch {epoch} with loss = {epoch_loss:.6f}"
#                 + Style.RESET_ALL
#             )
#             break
#         y_pred = model(
#             torch.tensor(T, dtype=torch.float32, device=device, requires_grad=True)
#         )
#         val = objective_fun(y_pred)
#         MOV = MOV + val.item()

#     print(Fore.BLUE + "Training complete.\n" + Style.RESET_ALL)
#     return model, loss_list, MOV / (batches)


# was in example 1
# ts_batches = [
#     torch.empty(batch_size, dtype=torch.float32, device=device)
#     .uniform_(*tspan)
#     .unsqueeze(1)
#     for _ in range(batches)
# ]

#  phi,  # physics operator generator: phi(b) returns a function to apply to model output.
# objective_fun,  # function that computes a scalar objective from model output.
# *,
# tspan: Tuple[float, float] = (0.0, 10.0),
# in_dim: int = 1,
# out_dim: int = 5,
# batch_size: int = 128,
# no_batches: int = 1,
# epochs: int = 1000,
# lr: float = 0.001,
# tol: float = 1e-5,
# training_name: str = "",
# device: torch.device = torch.device("cpu"),
# testing_tb: List[
#     float
# ] = None,  # required when in_dim > 1; list of floats: first element is t, rest are b.
# b_list = [
#             torch.empty(batch_size, dtype=torch.float32, device=device)
#             .uniform_(*tspan)
#             .unsqueeze(1)
#             for _ in range(batches)
# #         ]
#  b_list = [
#             torch.empty(
#                 (batch_size, in_dim - 1), dtype=torch.float32, device=device
#             ).uniform_(*tspan)
#             for _ in range(batches)
#         ]


# model, loss_list, mov = train_model_case_2(
#     phi,
#     createObjectiveFun(D),
#     ts_batches,
#     b_list=b_list,
#     in_dim=in_dim,
#     out_dim=out_dim,
#     batch_size=batch_size,
#     # no_batches=batches,
#     epochs=epochs,
#     lr=0.001,
#     tol=1e-3,
#     trainig_name=name,
#     device=device,
#     testing_tb=[tspan[1], *b_raw],
# )


# def train_model_case_2(
#     phi,
#     objective_fun,
#     ts_batches,
#     *,
#     b_list: List[torch.Tensor] = [
#         torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
#     ],
#     device=torch.device("cpu"),
#     T=10.0,
#     lr=0.001,
#     epochs=1000,
#     tol=1e-5,
#     out_dim=5,
#     in_dim=2,
#     testing_tb: List[float],
#     trainig_name: str = "",
#     batch_size: int = 128,
# ):
#     just_fix_windows_console()
#     init()
#     model = PINN(in_dim=in_dim, out_dim=out_dim).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     print(Fore.RED + f"Starting training {trainig_name}" + Style.RESET_ALL)

#     batches = len(ts_batches)
#     loss_list = []
#     MOV = 0
#     t_eval = torch.tensor(
#         [testing_tb], dtype=torch.float32, device=device, requires_grad=True
#     )
#     for epoch in range(1, epochs + 1):
#         epoch_loss = 0
#         pbar = tqdm(
#             ts_batches,
#             desc=f"Epoch {epoch}/{epochs} training {batches} batches ",
#             leave=False,
#         )
#         for batch, ts in enumerate(pbar, start=1):
#             # Get corresponding b for this batch.
#             b = b_list[batch - 1]
#             tb = torch.cat([ts, b], dim=1)
#             # For the current batch, assume ts is shape (N,)
#             # Create bs of shape (N, L) by repeating b for each time point.
#             compute_loss_vectorized = create_loss(model, tb, phi)
#             # Define compute_loss using our externally defined loss_b function.

#             optimizer.zero_grad()
#             loss_val = compute_loss_vectorized()
#             loss_valued = loss_val.item()
#             epoch_loss += loss_valued
#             loss_val.backward()
#             optimizer.step()
#             loss_list.append(loss_val.item())
#             pbar.set_postfix(loss=f"{batch} -- {loss_valued:.6f}")

#             # pbar.set_postfix(loss=f"{batch} -- {loss_valued:.6f}")

#         epoch_loss /= batches
#         if epoch % 100 == 0:
#             print(
#                 Fore.MAGENTA
#                 + f"{trainig_name} | (RHS: NOT FIXED) {batches} batches, Epoch {epoch}/{epochs}: Loss = {epoch_loss:.6f}"
#                 + Style.RESET_ALL
#             )
#         if epoch_loss < tol:
#             print(
#                 Fore.GREEN
#                 + f"Stopping training at epoch {epoch} with loss = {epoch_loss:.6f}"
#                 + Style.RESET_ALL
#             )
#             break

#         # Evaluate model at t=T using the same b.

#         y_pred = model(t_eval)
#         val = objective_fun(y_pred[0])
#         MOV += val.item()

#     print(Fore.BLUE + "Training complete.\n" + Style.RESET_ALL)
#     return model, loss_list, MOV / batches


# def example_1(
#     batches: int = 1,
#     batch_size: int = 128,
#     epochs: int = 1000,
#     rhs: int = 0,
# ):
#     device = get_device()
#     name = "Example 1"
#     D = torch.tensor([-9.54, -8.16, -4.26, -11.43], dtype=torch.float32, device=device)
#     A = torch.tensor([[3.18, 2.72, 1.42, 3.81]], dtype=torch.float32, device=device)
#     b_raw = [7.81]
#     b = torch.tensor([b_raw], dtype=torch.float32, device=device)
#     tspan = (0.0, 10.0)
#     phi = createPhi(D, A)
#     out_dim = sum(A.shape)

#     if rhs == 0:
#         in_dim = 1
#         model, loss_list, mov = train_model(
#             phi(b),
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#         )
#     else:
#         in_dim = len(b_raw) + 1
#         model, loss_list, mov = train_model(
#             phi,
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#             testing_tb=[tspan[1], *b_raw],
#         )

#     # ------------------------------
#     # Evaluate the trained model at select time points
#     # ------------------------------
#     test_times = [i * (tspan[1] / 5) for i in range(6)]
#     if rhs == 0:
#         test_model(model, test_times, dev=device)
#         filename = f"saved_models/example_1_with_same_rhs_pinn_{epochs}"
#     else:
#         b_tests = [b_raw for _ in range(6)]
#         test_model(model, test_times, dev=device, b_tests=b_tests)
#         filename = f"saved_models/example_1_with_random_rhs_pinn_{epochs}"
#     save_model(model, filename)

#     print(f"MOV: {mov}")


# # Example 2: Solving one LP with 3 variables and 3 constraints
# # write as a linear program:
# # minimize -3x1 - x2 - 3x3
# # subject to 2x1 + x2 + x3 <= 2
# #            x1 + 2x2 + 3x3 <= 5
# #            2x1 + 2x2 + x3 <= 6
# # The optimal value is obtained at t* = 10.0


# def example_2(
#     batches: int = 1,
#     batch_size: int = 128,
#     epochs: int = 1000,
#     rhs: int = 0,
# ):
#     device = get_device()
#     name = "Example 2"
#     D = torch.tensor([-3, -1, -3], dtype=torch.float32, device=device)
#     A = torch.tensor(
#         [[2, 1, 1], [1, 2, 3], [2, 2, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
#         dtype=torch.float32,
#         device=device,
#     )
#     b_raw = [2, 5, 6, 0, 0, 0]
#     b = torch.tensor([b_raw], dtype=torch.float32, device=device)
#     tspan = (0.0, 30.0)
#     phi = createPhi(D, A)
#     out_dim = sum(A.shape)

#     if rhs == 0:
#         in_dim = 1
#         model, loss_list, mov = train_model(
#             phi(b),
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#         )
#     else:

#         in_dim = len(b_raw) + 1
#         model, loss_list, mov = train_model(
#             phi,
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#             testing_tb=[tspan[1], *b_raw],
#         )

#     # ------------------------------
#     # Evaluate the trained model at select time points
#     # ------------------------------
#     test_times = [i * (tspan[1] / 5) for i in range(6)]
#     if rhs == 0:
#         test_model(model, test_times, dev=device)
#         filename = f"saved_models/example_2_with_same_rhs_pinn_{epochs}"
#     else:
#         b_tests = [b_raw for _ in range(6)]
#         test_model(model, test_times, dev=device, b_tests=b_tests)
#         filename = f"saved_models/example_2_with_random_rhs_pinn_{epochs}"
#     save_model(model, filename)

#     print(f"MOV: {mov}")


# Example 3: Solving one LP with 3 variables and 3 constraints
# write as a linear program:
# minimize -x1 - 4x2 - 3x3
# subject to 2x1 + 2x2 + x3 <= 4
#            x1 + 2x2 + 2x3 <= 6
#            -x1 <= 0
#            -x2 <= 0
#            -x3 <= 0
# The optimal value is obtained at t* = 50.0


# def example_3(
#     batches: int = 1,
#     batch_size: int = 128,
#     epochs: int = 1000,
#     rhs: int = 0,
# ):
#     device = get_device()
#     name = "Example 3"
#     D = torch.tensor(
#         [-1.0, -4.0, -3.0],
#         dtype=torch.float32,
#         device=device,
#     )
#     A = torch.tensor(
#         [
#             [2.0, 2.0, 1.0],  # 2x1 + 2x2 + x3 <= 4
#             [1.0, 2.0, 2.0],  # x1 + 2x2 + 2x3 <= 6
#             [-1.0, 0.0, 0.0],  # -x1 <= 0
#             [0.0, -1.0, 0.0],  # -x2 <= 0
#             [0.0, 0.0, -1.0],  # -x3 <= 0
#         ],
#         dtype=torch.float32,
#         device=device,
#     )
#     b_raw = [4.0, 6.0, 0.0, 0.0, 0.0]
#     b = torch.tensor(
#         [b_raw],
#         dtype=torch.float32,
#         device=device,
#     )
#     tspan = (0.0, 50.0)
#     phi = createPhi(D, A)
#     out_dim = sum(A.shape)

#     if rhs == 0:
#         in_dim = 1
#         model, loss_list, mov = train_model(
#             phi(b),
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#         )
#     else:
#         in_dim = len(b_raw) + 1
#         model, loss_list, mov = train_model(
#             phi,
#             createObjectiveFun(D),
#             tspan=tspan,
#             in_dim=in_dim,
#             out_dim=out_dim,
#             batch_size=batch_size,
#             no_batches=batches,
#             epochs=epochs,
#             lr=0.001,
#             tol=1e-3,
#             training_name=name,
#             device=device,
#             testing_tb=[tspan[1], *b_raw],
#         )

#     # ------------------------------
#     # Evaluate the trained model at select time points
#     # ------------------------------
#     test_times = [i * (tspan[1] / 5) for i in range(6)]
#     if rhs == 0:
#         test_model(model, test_times, dev=device)
#         filename = f"saved_models/example_3_with_same_rhs_pinn_{epochs}"
#     else:
#         b_tests = [b_raw for _ in range(6)]
#         test_model(model, test_times, dev=device, b_tests=b_tests)
#         filename = f"saved_models/example_3_with_random_rhs_pinn_{epochs}"
#     save_model(model, filename)

#     print(f"MOV: {mov}")
