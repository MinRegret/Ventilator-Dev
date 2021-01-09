import math
import torch
from collections import OrderedDict


class SNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, dropout_prob=0.0):
        super().__init__()
        layers = OrderedDict()
        for i in range(n_layers - 1):
            if i == 0:
                layers[f"fc{i}"] = torch.nn.Linear(in_dim, hidden_dim, bias=False)
            else:
                layers[f"fc{i}"] = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            layers[f"selu_{i}"] = torch.nn.SELU()
            layers[f"dropout_{i}"] = torch.nn.AlphaDropout(p=dropout_prob)
        layers[f"fc_{i+1}"] = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.network = torch.nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x):
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network:
            if not isinstance(layer, torch.nn.Linear):
                continue
            torch.nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(layer.bias, -bound, bound)

    def track_layer_activations(self, x):
        activations = []
        for layer in self.network:
            x = layer.forward(x)
            if isinstance(layer, torch.nn.SELU):
                activations.append(x.data.flatten())
        return activations


class ShallowBoundaryModel(torch.nn.Module):  # Paula's boundary model
    def __init__(self, in_dim, hidden_dim=80, out_dim=1):
        super().__init__()
        layers = OrderedDict()
        layers["fc1"] = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        layers["tanh_1"] = torch.nn.Tanh()
        layers["fc_2"] = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.model = torch.nn.Sequential(layers)

    def forward(self, x):
        out = self.model(x)
        return out


class ConstantModel(torch.nn.Module):  # boundary model to predict first pressure with no features
    def __init__(self, const_out, trainable=True):
        super().__init__()

        self.const_out = torch.tensor(const_out, dtype=torch.float)
        if trainable:
            self.const_out = torch.nn.parameter.Parameter(self.const_out)

    def forward(self, x=None):
        return self.const_out


class InspiratoryModel(
    torch.nn.Module
):  # manager of a family of trained boundary models + general model
    def __init__(self, default_model, boundary_dict):
        super().__init__()

        self.default_model = default_model
        self.boundary_dict = torch.nn.ModuleDict({str(i): m for i, m in boundary_dict.items()})

    def forward(self, t, features):  # placeholder for stitching
        if t in self.boundary_dict:
            return self.boundary_dict[str(t)](features)
        return self.default_model(features)
