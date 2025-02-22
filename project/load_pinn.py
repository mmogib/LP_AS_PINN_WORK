from models import PINN, load_model, test_model
from device import get_device

dev = get_device()

model = PINN(out_dim=5).to(dev)

filename = "saved_models/example_1_pinn_5000_2025_02_21_out_dim_5.pt"

model_reloaded = load_model(model, dev, filename)


# ------------------------------
# Evaluate the trained model at select time points
# ------------------------------
test_times = [0.0, 2.5, 5.0, 7.5, 10.0, 10.1]
test_model(model_reloaded, test_times, dev=dev)
