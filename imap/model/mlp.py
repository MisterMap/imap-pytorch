import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimensions=(256, 256, 256, 256)):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.ReLU(),
        )
        models = []
        input_dimension = input_dimension + hidden_dimensions[0]
        for hidden_dimension in hidden_dimensions[1:]:
            models.append(nn.Linear(input_dimension, hidden_dimension))
            models.append(nn.ReLU())
            input_dimension = hidden_dimension
        models.append(nn.Linear(input_dimension, output_dimension))
        self.last_layers = nn.Sequential(*models)

    def forward(self, x):
        before_last_layer = self.first_layer(x)
        x = torch.cat([before_last_layer, x], dim=1)
        return self.last_layers(x)

