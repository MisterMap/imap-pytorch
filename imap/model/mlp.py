import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(self, input_dimension, output_dimension, hidden_dimensions=(256, 256, 256, 256)):
        models = []
        for hidden_dimension in hidden_dimensions:
            models.append(nn.Linear(input_dimension, hidden_dimension))
            models.append(nn.ReLU())
            input_dimension = hidden_dimension
        models.append(nn.Linear(input_dimension, output_dimension))
        super().__init__(*models)
