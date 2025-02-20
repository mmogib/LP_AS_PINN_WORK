from models import PINN, load_model, test_model
from device import get_device
import torch

dev = get_device()

model = PINN(out_vars=8).to(dev)

filename = "pinn_model.pt"
filename = "pinn_1000_2025_02_20.pt"

model_reloaded = load_model(model, dev, filename)


# ------------------------------
# Evaluate the trained model at select time points
# ------------------------------
test_times = [0.0, 2.5, 5.0, 7.5, 10.0, 10.1]
test_model(model_reloaded, test_times, device=dev)
