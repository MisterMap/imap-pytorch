import torch
import torch.nn as nn

from .base_lightning_model import BaseLightningModule
from .mlp import MLP
from .gaussian_positional_encoding import GaussianPositionalEncoding
from ..utils.torch_math import back_project_pixel


class NERF(BaseLightningModule):
    def __init__(self, parameters, data_module):
        super().__init__(parameters)
        self._mlp = MLP(parameters.encoding_dimension, 4)
        self._positional_encoding = GaussianPositionalEncoding(encoding_dimension=parameters.encoding_dimension)
        self._inverted_camera_matrix = torch.tensor(data_module.get_inverted_camera_matrix())
        self._default_color = torch.tensor(data_module.get_default_color())
        self._default_depth = torch.tensor(data_module.get_default_depth())
        self._loss = nn.L1Loss()

    def forward(self, pixel, camera_position):
        bins_count = self.hparams.sample_bins
        points, depths = self.sample_along_direction(pixel, camera_position, bins_count)

        encodings = self._positional_encoding(points)
        prediction = self._mlp(encodings)

        colors = prediction[:, :3].reshape(bins_count, -1, 3)
        density = prediction[:, 3].reshape(bins_count, -1)
        depths = depths.reshape(bins_count, -1)
        weights = self.calculate_weights(density, depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth)
        return reconstructed_color, reconstructed_depths

    def sample_along_direction(self, pixels, camera_positions, bins_count):
        depths = self.get_random_depths(pixels.shape[0], pixels.device, bins_count)
        pixels = self.repeat_tensor(pixels, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, depths, camera_positions, self._inverted_camera_matrix)
        return back_projected_points, depths

    def get_random_depths(self, batch_size, device, bins_count):
        result = torch.rand((bins_count, batch_size), device=device) * self._default_depth
        result = torch.sort(result, dim=0).values
        result = result.reshape(-1)
        return result

    @staticmethod
    def repeat_tensor(tensor, bins_count):
        result = torch.repeat_interleave(tensor[None], bins_count)
        result = result.reshape(-1, *tensor.shape[1:])
        return result

    @staticmethod
    def calculate_weights(densities, depths):
        weights = []
        previous_depth = 0
        product = 1
        for density, depth in zip(densities, depths):
            depth_delta = depth - previous_depth
            previous_depth = depth
            hit_probability = 1 - torch.exp(-density * depth_delta)
            weights.append(hit_probability * product)
            product = product * (1 - hit_probability)
        weights.append(product)
        return torch.stack(weights, dim=0)

    @staticmethod
    def reconstruct_color(colors, weights, default_color):
        return torch.sum(colors * weights[:-1, :, None], dim=0
                         ) + default_color.to(colors.device)[None] * weights[-1, :, None]

    @staticmethod
    def reconstruct_depth(depths, weights, default_depth):
        return torch.sum(depths * weights[:-1, :], dim=0) + default_depth.to(depths.device)[None] * weights[-1]

    def loss(self, batch):
        output = self.forward(batch["pixel"], batch["camera_position"])
        image_loss = self._loss(output[0], batch["color"])
        depth_loss = self._loss(output[1], batch["depth"])
        loss = image_loss + self.hparams.depth_loss_koef * depth_loss
        losses = {
            "image_loss": image_loss,
            "depth_loss": depth_loss,
            "loss": loss
        }
        return output, losses
