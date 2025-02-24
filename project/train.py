import torch
import argparse
import numpy as np

# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt


from models import (
    train_model,
    save_model,
    test_model,
)
from utils import read_mps

torch.manual_seed(2025)


def plot_loss():
    # Assume loss_list is a list of float loss values collected during training
    # Load loss_list from the text file.
    loss_array = np.loadtxt("saved_models/example_1_with_random_rhs_pinn_100_loss.txt")
    # Convert to a Python list if needed:
    loss_list = loss_array.tolist()
    print(loss_list)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_lists_to_file(loss_list, mov_list, filename):
    np_loss = np.array(loss_list)
    loss_filename = f"{filename}_loss.txt"
    print(f"saving... loss list in {loss_filename}")
    np.savetxt(loss_filename, np_loss, fmt="%.6f")

    movs_filename = f"{filename}_mov.txt"
    np_movs = np.array(mov_list)
    print(f"saving... movs list in {movs_filename}")
    np.savetxt(movs_filename, np_movs, fmt="%.6f")


# Example 1: Solving one LP with 4 variables and 1 constraint
# write as a linear program:
# minimize -9.54x1 - 8.16x2 - 4.26x3 - 11.43x4
# subject to 3.18x1 + 2.72x2 + 1.42x3 + 3.81x4 <= 7.81
# The optimal value is obtained at t* = 10.0


def example_1(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "Example 1"

    D = torch.tensor([-9.54, -8.16, -4.26, -11.43], dtype=torch.float32, device=device)
    A = torch.tensor([[3.18, 2.72, 1.42, 3.81]], dtype=torch.float32, device=device)
    b_raw = [7.81]
    b = torch.tensor([b_raw], dtype=torch.float32, device=device)
    tspan = (0.0, 10.0)

    model, loss_list, mov_list = train_model(
        A, b, D, name, tspan, rhs, epochs, batch_size, batches, device
    )

    filename = test_model(model, device, b_raw, rhs, tspan[1], name, epochs)
    save_model(model, filename)

    save_lists_to_file(loss_list, mov_list, filename)


# Example 2: Solving one LP with 3 variables and 3 constraints
# write as a linear program:
# minimize -3x1 - x2 - 3x3
# subject to 2x1 + x2 + x3 <= 2
#            x1 + 2x2 + 3x3 <= 5
#            2x1 + 2x2 + x3 <= 6
# The optimal value is obtained at t* = 10.0


def example_2(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "Example 2"
    D = torch.tensor([-3, -1, -3], dtype=torch.float32, device=device)
    A = torch.tensor(
        [[2, 1, 1], [1, 2, 3], [2, 2, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        dtype=torch.float32,
        device=device,
    )
    b_raw = [2, 5, 6, 0, 0, 0]
    b = torch.tensor([b_raw], dtype=torch.float32, device=device)
    tspan = (0.0, 30.0)

    model, loss_list, mov_list = train_model(
        A, b, D, name, tspan, rhs, epochs, batch_size, batches, device
    )

    filename = test_model(model, device, b_raw, rhs, tspan[1], name, epochs)
    save_model(model, filename)

    save_lists_to_file(loss_list, mov_list, filename)


# Example 3: Solving one LP with 3 variables and 3 constraints
# write as a linear program:
# minimize -x1 - 4x2 - 3x3
# subject to 2x1 + 2x2 + x3 <= 4
#            x1 + 2x2 + 2x3 <= 6
#            -x1 <= 0
#            -x2 <= 0
#            -x3 <= 0
# The optimal value is obtained at t* = 50.0


def example_3(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    rhs: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "Example 3"
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
    b_raw = [4.0, 6.0, 0.0, 0.0, 0.0]
    b = torch.tensor(
        [b_raw],
        dtype=torch.float32,
        device=device,
    )
    tspan = (0.0, 50.0)
    model, loss_list, mov_list = train_model(
        A, b, D, name, tspan, rhs, epochs, batch_size, batches, device
    )

    filename = test_model(model, device, b_raw, rhs, tspan[1], name, epochs)
    save_model(model, filename)

    save_lists_to_file(loss_list, mov_list, filename)


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
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default="1",
        help="Which example to run (1, 2, 3, ...)",
    )
    args = parser.parse_args()
    if args.no_action:
        print("No action flag is set.")
        # read_mps("mps_files/problem2.mps")
        # plot_loss()
        exit(0)
    examples = [example_1, example_2, example_3]

    print(f"Running example {args.example} for {args.epochs} epochs.")
    for example in args.example:
        examples[example - 1](
            batches=args.batches,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rhs=args.random_rhs,
        )
