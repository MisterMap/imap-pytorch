import torch
import torch.nn as nn

from .base_lightning_model import BaseLightningModule
from .gaussian_positional_encoding import GaussianPositionalEncoding
from .mlp import MLP
from ..utils.torch_math import back_project_pixel, matrix_from_9d_position


class NERF(BaseLightningModule):
    def __init__(self, parameters, camera_info):
        super().__init__(parameters)
        self._mlp = MLP(parameters.encoding_dimension, 4)
        self._positional_encoding = GaussianPositionalEncoding(encoding_dimension=parameters.encoding_dimension,
                                                               sigma=parameters.encoding_sigma)
        self._inverted_camera_matrix = torch.tensor(camera_info.get_inverted_camera_matrix())
        self._default_color = torch.tensor(camera_info.get_default_color())
        self._default_depth = torch.tensor(camera_info.get_default_depth())
        self._loss = nn.L1Loss(reduction="none")
        self._positions = None

    def forward(self, pixel, camera_position):
        with torch.no_grad():
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
        with torch.no_grad():
            fine_sampled_depths = self.hierarchical_sample_depths(
                course_weights,
                pixel.shape[0],
                pixel.device,
                self.hparams.fine_sample_bins,
                course_sampled_depths,
                not self.training)
        fine_sampled_depths = torch.cat([fine_sampled_depths, course_sampled_depths], dim=0)
        # fine_sampled_depths = torch.cat([course_sampled_depths], dim=0)
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

        colors = torch.sigmoid(prediction[:, :3]).reshape(bins_count, -1, 3)
        density = prediction[:, 3].reshape(bins_count, -1)
        depths = sampled_depths.reshape(bins_count, -1)
        weights = self.calculate_weights(density, depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth)
        with torch.no_grad():
            reconstructed_depth_variance = self.reconstruct_depth_variance(depths, weights, reconstructed_depths,
                                                                           self._default_depth)
        return reconstructed_color, reconstructed_depths, weights, reconstructed_depth_variance

    def stratified_sample_depths(self, batch_size, device, bins_count, deterministic=False):
        if deterministic:
            depth_delta = (self._default_depth.item() - self.hparams.minimal_depth) / bins_count
            result = torch.arange(self.hparams.minimal_depth, self._default_depth.item(), depth_delta, device=device)
            result = torch.repeat_interleave(result[:, None], batch_size, dim=1)
            return result
        uniform = torch.rand((bins_count, batch_size), device=device)
        uniform[0] = 1
        result = (torch.arange(bins_count, device=device)[:, None] + uniform - 1
                  ) * (self._default_depth - self.hparams.minimal_depth) / (bins_count - 1) + self.hparams.minimal_depth
        return result

    def hierarchical_sample_depths(self, weights, batch_size, device, bins_count, bins, deterministic=False):
        weights = weights.transpose(1, 0)[:, :-1] + 1e-10
        pdf = weights / torch.sum(weights, dim=1)[:, None]
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
        minimal_bin = bins[0]
        bins = (torch.roll(bins, 1) + bins) / 2
        bins[0] = minimal_bin
        bins = bins.transpose(1, 0)

        if deterministic:
            uniform = torch.arange(bins_count, device=device) / bins_count + 1. / 2 / bins_count
            uniform = torch.repeat_interleave(uniform[None], batch_size, dim=0)
        else:
            uniform = torch.rand((batch_size, bins_count), device=device).contiguous()
        indexes = torch.searchsorted(cdf, uniform, right=True)
        index_below = self.clip_indexes(indexes - 1, 0, bins.shape[1] - 1)
        index_above = self.clip_indexes(indexes, 0, bins.shape[1] - 1)

        denominator = torch.gather(cdf, 1, index_above) - torch.gather(cdf, 1, index_below)
        denominator = torch.where(denominator < 1e-10, torch.ones_like(denominator), denominator)
        t = (uniform - torch.gather(cdf, 1, index_below)) / denominator
        bins_below = torch.gather(bins, 1, index_below)
        bins_above = torch.gather(bins, 1, index_above)
        hierarchical_sample = bins_below + t * (bins_above - bins_below)
        return hierarchical_sample.transpose(1, 0)

    @staticmethod
    def clip_indexes(indexes, minimal, maximal):
        result = torch.max(minimal * torch.ones_like(indexes), indexes)
        result = torch.min(maximal * torch.ones_like(indexes), result)
        return result

    @staticmethod
    def repeat_tensor(tensor, bins_count):
        result = torch.repeat_interleave(tensor[None], bins_count, dim=0)
        result = result.reshape(-1, *tensor.shape[1:])
        return result

    def calculate_weights(self, densities, depths):
        weights = []
        product = 1
        densities = torch.logsumexp(torch.cat([torch.zeros_like(densities)[None], densities[None]], dim=0), dim=0)
        for i in range(len(depths)):
            if i < len(depths) - 1:
                depth_delta = depths[i + 1] - depths[i]
            else:
                depth_delta = self._default_depth - depths[i]
            hit_probability = 1 - torch.exp(-densities[i] * depth_delta)
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

    def loss(self, batch, reduction=True):
        camera_position = self.positions_from_batch(batch)
        output = self.forward(batch["pixel"], camera_position)
        mask = (batch["depth"] > 1e-12) & (batch["depth"] < self._default_depth)
        course_image_loss = torch.mean(self._loss(output[0], batch["color"]), dim=1)
        course_depth_weights = 1. / (torch.sqrt(output[4]) + 1e-10) * mask
        course_depth_loss = self._loss(output[1] * course_depth_weights, batch["depth"] * course_depth_weights)
        fine_image_loss = torch.mean(self._loss(output[2], batch["color"]), dim=1)
        fine_depth_weights = 1. / (torch.sqrt(output[5]) + 1e-10) * mask
        fine_depth_loss = self._loss(output[3] * fine_depth_weights, batch["depth"] * fine_depth_weights)
        # image_loss = course_image_loss + fine_image_loss
        # depth_loss = course_depth_loss + fine_depth_loss
        image_loss = fine_image_loss
        depth_loss = fine_depth_loss
        loss = self.hparams.color_loss_koef * image_loss + self.hparams.depth_loss_koef * depth_loss
        if reduction:
            course_depth_loss = torch.mean(course_depth_loss)
            course_image_loss = torch.mean(course_image_loss)
            fine_depth_loss = torch.mean(fine_depth_loss)
            fine_image_loss = torch.mean(fine_image_loss)
            loss = torch.mean(loss)
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
        indexes = batch["frame_index"]
        return matrix_from_9d_position(self._positions[indexes])

    def set_positions(self, position):
        self._positions = position
