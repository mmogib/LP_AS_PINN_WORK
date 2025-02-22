import torch
import argparse


from device import get_device
from ode import createPhi, createObjectiveFun
from models import save_model, train_model, test_model
from utils import read_mps

torch.manual_seed(2025)


def example_1(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    D = torch.tensor([-9.54, -8.16, -4.26, -11.43], dtype=torch.float32, device=device)
    A = torch.tensor([[3.18, 2.72, 1.42, 3.81]], dtype=torch.float32, device=device)
    b = torch.tensor([[7.81]], dtype=torch.float32, device=device)
    phi = createPhi(D, A)
    out_dim = sum(A.shape)
    ts_batches = [
        torch.empty(batch_size, dtype=torch.float32, device=device).uniform_(0.0, 10.1)
        for _ in range(batches)
    ]
    b_list = (
        [b]
        if rhs == 0
        else [
            torch.empty(1, dtype=torch.float32, device=device)
            .uniform_(0.0, 10.1)
            .unsqueeze(1)
            for _ in range(batches)
        ]
    )
    model, loss_list, mov = train_model(
        phi,
        createObjectiveFun(D),
        ts_batches,
        b_list,
        device,
        lr=0.001,
        epochs=epochs,
        out_dim=out_dim,
        T=10.0,
    )

    # ------------------------------
    # Evaluate the trained model at select time points
    # ------------------------------
    test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    test_model(model, test_times, dev=device)
    save_model(model, f"saved_models/example_1_pinn_{epochs}")

    print(f"MOV: {mov}")


def example_2(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    D = torch.tensor([-3, -1, -3], dtype=torch.float32, device=device)
    A = torch.tensor(
        [[2, 1, 1], [1, 2, 3], [2, 2, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        dtype=torch.float32,
        device=device,
    )
    b = torch.tensor([2, 5, 6, 0, 0, 0], dtype=torch.float32, device=device)
    phi = createPhi(D, A)
    out_dim = sum(A.shape)
    ts_batches = [
        torch.empty(batch_size, dtype=torch.float32, device=device).uniform_(0.0, 10.1)
        for _ in range(batches)
    ]
    b_list = (
        [b]
        if rhs == 0
        else [
            torch.empty(6, dtype=torch.float32, device=device).uniform_(0.0, 10.1)
            for _ in range(batches)
        ]
    )
    print(b_list)
    model, loss_list, mov = train_model(
        phi,
        createObjectiveFun(D),
        ts_batches,
        b_list,
        device,
        lr=0.001,
        epochs=epochs,
        out_dim=out_dim,
        T=10.0,
    )

    # ------------------------------
    # Evaluate the trained model at select time points
    # ------------------------------
    test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    test_model(model, test_times, dev=device)
    save_model(model, f"saved_models/example_1_pinn_{epochs}")

    print(f"MOV: {mov}")


def example_3(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    D = torch.tensor(
        [-1.0, -4.0, -3.0],
        dtype=torch.float32,
        device=device,
    )
    A = torch.tensor(
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

    b = torch.tensor(
        [4.0, 6.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    phi = createPhi(D, A)
    out_dim = sum(A.shape)
    ts_batches = [
        torch.empty(batch_size, dtype=torch.float32, device=device).uniform_(0.0, 10.1)
        for _ in range(batches)
    ]
    b_list = (
        [b]
        if rhs == 0
        else [
            torch.empty(5, dtype=torch.float32, device=device).uniform_(0.0, 10.1)
            for _ in range(batches)
        ]
    )
    model, loss_list, mov = train_model(
        phi,
        createObjectiveFun(D),
        ts_batches,
        b_list,
        device,
        lr=0.001,
        epochs=epochs,
        out_dim=out_dim,
        T=10.0,
    )

    # ------------------------------
    # Evaluate the trained model at select time points
    # ------------------------------
    test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    test_model(model, test_times, dev=device)
    save_model(model, f"saved_models/example_1_pinn_{epochs}")

    print(f"MOV: {mov}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the PINN model using named arguments"
    )
    parser.add_argument("--no-action", help="No action needed", action="store_true")

    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batches", type=int, default=1, help="Number of training batches"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--random_rhs", type=int, default=0, help="Batch size")
    parser.add_argument(
        "--example",
        type=str,
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which example to run (1, 2, 3, or all)",
    )
    args = parser.parse_args()
    if args.no_action:
        print("No action flag is set.")
        read_mps("mps_files/problem2.mps")
        exit(0)
    device = get_device()

    print(f"Running example {args.example} for {args.epochs} epochs.")
    if args.example == "all":
        example_1(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
        example_2(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
        example_3(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
    elif args.example == "1":
        example_1(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
    elif args.example == "2":
        example_2(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
    elif args.example == "3":
        example_3(
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
