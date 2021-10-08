import torch
import torch.nn as nn

from .base_lightning_model import BaseLightningModule
from .mlp import MLP
from .gaussian_positional_encoding import GaussianPositionalEncoding
from ..utils.torch_math import back_project_pixel


class NERF(BaseLightningModule):
    def __init__(self, parameters, camera_info):
        super().__init__(parameters)
        self._mlp = MLP(parameters.encoding_dimension, 4)
        self._positional_encoding = GaussianPositionalEncoding(encoding_dimension=parameters.encoding_dimension,
                                                               sigma=parameters.encoding_sigma)
        self._inverted_camera_matrix = torch.tensor(camera_info.get_inverted_camera_matrix())
        self._default_color = torch.tensor(camera_info.get_default_color())
        self._default_depth = torch.tensor(camera_info.get_default_depth())
        self._loss = nn.L1Loss()
        self._positions = nn.Parameter(torch.zeros(0, 4, 4), requires_grad=parameters.optimize_positions)

    def forward(self, pixel, camera_position):
        course_sampled_depths = self.stratified_sample_depths(
            pixel.shape[0],
            pixel.device,
            self.hparams.course_sample_bins,
            not self.training)
        course_color, course_depths, course_weights, course_depth_variance = self.reconstruct_color_and_depths(
            course_sampled_depths,
            pixel,
            camera_position,
            self._mlp)
        fine_sampled_depths = self.hierarchical_sample_depths(
            course_weights,
            pixel.shape[0],
            pixel.device,
            self.hparams.fine_sample_bins,
            course_sampled_depths,
            not self.training)
        fine_sampled_depths = torch.cat([fine_sampled_depths, course_sampled_depths], dim=0)
        fine_color, fine_depths, fine_weights, fine_depth_variance = self.reconstruct_color_and_depths(
            fine_sampled_depths,
            pixel,
            camera_position,
            self._mlp)
        return course_color, course_depths, fine_color, fine_depths, course_depth_variance, fine_depth_variance

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
        reconstructed_depth_variance = self.reconstruct_depth_variance(depths, weights, reconstructed_depths,
                                                                       self._default_depth)
        return reconstructed_color, reconstructed_depths, weights, reconstructed_depth_variance

    def stratified_sample_depths(self, batch_size, device, bins_count, deterministic=False):
        if deterministic:
            uniform = torch.ones((bins_count, batch_size), device=device) * 0.5
        else:
            uniform = torch.rand((bins_count, batch_size), device=device)
        result = (torch.arange(bins_count, device=device)[:, None] + uniform) * self._default_depth / bins_count
        return result

    @staticmethod
    def hierarchical_sample_depths(weights, batch_size, device, bins_count, bins, deterministic=False):
        weights = weights.transpose(1, 0)[:, :-1] + 1e-5
        pdf = weights / torch.sum(weights, dim=1)[:, None]
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
        bins = bins.transpose(1, 0)
        bins = torch.cat([torch.zeros_like(bins[:, :1]), bins], 1)

        if deterministic:
            uniform = torch.arange(bins_count, device=device) / bins_count + 1. / 2 / bins_count
            uniform = torch.repeat_interleave(uniform[None], batch_size, dim=0)
        else:
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

    @staticmethod
    def reconstruct_depth_variance(depths, weights, mean_depths, default_depth):
        return torch.sum((depths - mean_depths[None]) ** 2 * weights[:-1], dim=0
                         ) + (default_depth.to(depths.device)[None] - mean_depths) ** 2 * weights[-1]

    def loss(self, batch):
        camera_position = self.positions_from_batch(batch)
        output = self.forward(batch["pixel"], camera_position)
        course_image_loss = self._loss(output[0], batch["color"])
        course_depth_weights = 1. / (torch.sqrt(output[4]) + 1e-5)
        course_depth_loss = self._loss(output[1] * course_depth_weights, batch["depth"] * course_depth_weights)
        fine_image_loss = self._loss(output[2], batch["color"])
        fine_depth_weights = 1. / (torch.sqrt(output[5]) + 1e-5)
        fine_depth_loss = self._loss(output[3] * fine_depth_weights, batch["depth"] * fine_depth_weights)
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

    def positions_from_batch(self, batch):
        if "camera_position" in batch.keys():
            return batch["camera_position"]
        indexes = batch["model_index"]
        return self._positions[indexes]

    def add_tracked_position(self, position):
        previous_positions = self._positions.data
        new_positions = torch.cat([previous_positions, torch.tensor(position)], dim=0)
        self._positions = nn.Parameter(new_positions, requires_grad=self.hparams.optimize_positions)

    def last_position(self):
        return self._positions[-1]
