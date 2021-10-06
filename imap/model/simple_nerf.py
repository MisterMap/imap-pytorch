import torch
import torch.nn as nn

from .base_lightning_model import BaseLightningModule
from .mlp import MLP
from .gaussian_positional_encoding import GaussianPositionalEncoding
from ..utils.torch_math import back_project_pixel


class NERF(BaseLightningModule):
    def __init__(self, parameters, data_module):
        super().__init__(parameters)
        self._fine_mlp = MLP(parameters.encoding_dimension, 4)
        self._course_mlp = MLP(parameters.encoding_dimension, 4)
        self._positional_encoding = GaussianPositionalEncoding(encoding_dimension=parameters.encoding_dimension,
                                                               sigma=parameters.encoding_sigma)
        self._inverted_camera_matrix = torch.tensor(data_module.get_inverted_camera_matrix())
        self._default_color = torch.tensor(data_module.get_default_color())
        self._default_depth = torch.tensor(data_module.get_default_depth())
        self._loss = nn.L1Loss()

    def forward(self, pixel, camera_position):
        course_sampled_depths = self.stratified_sample_depths(
            pixel.shape[0],
            pixel.device,
            self.hparams.course_sample_bins)
        course_color, course_depths, course_weights = self.reconstruct_color_and_depths(
            course_sampled_depths,
            pixel,
            camera_position,
            self._course_mlp)
        fine_sampled_depths = self.hierarchical_sample_depths(
            course_weights,
            pixel.shape[0],
            pixel.device,
            self.hparams.fine_sample_bins,
            course_sampled_depths)
        fine_sampled_depths = torch.cat([fine_sampled_depths, course_sampled_depths], dim=0)
        fine_color, fine_depths, fine_weights = self.reconstruct_color_and_depths(
            fine_sampled_depths,
            pixel,
            camera_position,
            self._fine_mlp)
        return course_color, course_depths, fine_color, fine_depths

    def reconstruct_color_and_depths(self, sampled_depths, pixels, camera_positions, mlp_model):
        bins_count = sampled_depths.shape[0]
        sampled_depths = torch.sort(sampled_depths, dim=0).values
        sampled_depths = sampled_depths.reshape(-1)
        pixels = self.repeat_tensor(pixels, bins_count)
        camera_positions = self.repeat_tensor(camera_positions, bins_count)
        back_projected_points = back_project_pixel(pixels, sampled_depths, camera_positions,
                                                   self._inverted_camera_matrix)
        encodings = self._positional_encoding(back_projected_points)
        prediction = mlp_model(encodings)

        colors = prediction[:, :3].reshape(bins_count, -1, 3)
        density = prediction[:, 3].reshape(bins_count, -1)
        depths = sampled_depths.reshape(bins_count, -1)
        weights = self.calculate_weights(density, depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth)
        return reconstructed_color, reconstructed_depths, weights

    def stratified_sample_depths(self, batch_size, device, bins_count):
        result = torch.rand((bins_count, batch_size), device=device)
        result = (torch.arange(bins_count, device=device)[:, None] + result) * self._default_depth / bins_count
        return result

    @staticmethod
    def hierarchical_sample_depths(weights, batch_size, device, bins_count, bins):
        weights = weights.transpose(1, 0)[:, :-1] + 1e-5
        pdf = weights / torch.sum(weights, dim=1)[:, None]
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
        bins = bins.transpose(1, 0)
        bins = torch.cat([torch.zeros_like(bins[:, :1]), bins], 1)

        uniform = torch.rand((batch_size, bins_count), device=device).contiguous()
        indexes = torch.searchsorted(cdf, uniform, right=True)
        index_below = torch.max(torch.zeros_like(indexes), indexes - 1)
        index_above = torch.min((cdf.shape[1] - 1) * torch.ones_like(indexes), indexes)

        denominator = torch.gather(cdf, 1, index_above) - torch.gather(cdf, 1, index_below)
        denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator) * 1e-5, denominator)
        t = (uniform - torch.gather(cdf, 1, index_below)) / denominator
        hierarchical_sample = torch.gather(bins, 1, index_below) + t * (
                torch.gather(bins, 1, index_above) - torch.gather(bins, 1, index_below))
        return hierarchical_sample.transpose(1, 0)

    @staticmethod
    def repeat_tensor(tensor, bins_count):
        result = torch.repeat_interleave(tensor[None], bins_count, dim=0)
        result = result.reshape(-1, *tensor.shape[1:])
        return result

    @staticmethod
    def calculate_weights(densities, depths):
        weights = []
        previous_depth = 0
        product = 1
        densities = torch.logsumexp(torch.cat([torch.zeros_like(densities)[None], densities[None]], dim=0), dim=0)
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
        course_image_loss = self._loss(output[0], batch["color"])
        course_depth_loss = self._loss(output[1], batch["depth"])
        fine_image_loss = self._loss(output[2], batch["color"])
        fine_depth_loss = self._loss(output[3], batch["depth"])
        image_loss = course_image_loss + fine_image_loss
        depth_loss = course_depth_loss + fine_depth_loss
        loss = image_loss + self.hparams.depth_loss_koef * depth_loss
        losses = {
            "course_image_loss": course_image_loss,
            "course_depth_loss": course_depth_loss,
            "fine_image_loss": fine_image_loss,
            "fine_depth_loss": fine_depth_loss,
            "loss": loss
        }
        return output, losses
