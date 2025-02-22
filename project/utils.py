import pulp
import torch
import numpy as np


# Load the MPS file into a PuLP model.
# This assumes your MPS file (e.g. "problem.mps") is formatted correctly.
def read_mps(filename: str):
    # Load the MPS file into a PuLP model.
    # This assumes your MPS file (e.g. "problem.mps") is formatted correctly.
    variables, prob = pulp.LpProblem.fromMPS(filename)
    print(prob)
    print(prob.objective.to_dict())
    # Get the list of variables (the order will be used for constructing matrices)
    var_names = list(map(lambda v: v, variables))
    n = len(variables)

    # --- Extract Objective ---
    # Use map to extract coefficients for each variable (defaulting to 0 if missing)
    obj_coeffs = list(map(lambda v: v["value"], prob.objective.to_dict()))

    # Convert to a NumPy array and then to a torch tensor.
    obj_np = np.array(obj_coeffs, dtype=np.float32)
    obj_tensor = torch.tensor(obj_np, dtype=torch.float32)
    print("Objective tensor:")
    print(obj_tensor)

    # --- Extract Constraints ---
    def extract_constraint(cons):
        # Build a row of zeros for the coefficients corresponding to each variable.
        row = [0.0] * n
        # cons is an LpConstraint (an LpAffineExpression): its items provide (variable, coefficient) pairs.
        for var, coeff in cons.items():
            # Use var_names.index(var.name) to locate the index for this variable.
            idx = var_names.index(var.name)
            row[idx] = coeff
        # The right-hand side is defined as -constant.
        return row, -cons.constant

    # Use map to process all constraints.
    rows_b = list(map(extract_constraint, prob.constraints.values()))
    A_rows = [row for row, b_val in rows_b]
    b_rows = [b_val for row, b_val in rows_b]

    # Convert A and b to NumPy arrays and then torch tensors.
    A_np = np.array(A_rows, dtype=np.float32)
    b_np = np.array(b_rows, dtype=np.float32)
    A_tensor = torch.tensor(A_np, dtype=torch.float32)
    b_tensor = torch.tensor(b_np, dtype=torch.float32)

    print("Constraint matrix A:")
    print(A_tensor)
    print("Right-hand side b:")
    print(b_tensor)
