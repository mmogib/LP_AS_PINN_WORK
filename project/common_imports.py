import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian
from functorch import vmap, jacrev
import matplotlib.pyplot as plt
