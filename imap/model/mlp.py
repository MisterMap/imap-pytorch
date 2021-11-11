import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dimension, output_dimension=4, hidden_dimensions=(256, 256, 256, 256)):
        super().__init__()
        self.first_layer = nn.Sequential(*self.make_layers(input_dimension, hidden_dimensions[:2]))
        input_dimension = input_dimension + hidden_dimensions[1]
        models = self.make_layers(input_dimension, hidden_dimensions[2:])
        models.append(nn.Linear(hidden_dimensions[-1], output_dimension))
        # models[-1].weight.data[3, :] *= 0.1
        self.last_layers = nn.Sequential(*models)

    def forward(self, x):
        before_last_layer = self.first_layer(x)
        x = torch.cat([before_last_layer, x], dim=1)
        return self.last_layers(x)

    @staticmethod
    def make_layers(input_dimension, hidden_dimensions):
        models = []
        for hidden_dimension in hidden_dimensions:
            models.append(nn.Linear(input_dimension, hidden_dimension))
            models.append(nn.ReLU())
            input_dimension = hidden_dimension
        return models
