import numpy as np
import torch


class IMAPDataLoader(object):
    def __init__(self, batch_count, points_per_image, camera_info, device="cuda"):
        self._batch_count = batch_count
        self._points_per_image = points_per_image
        self._depth_images = []
        self._color_images = []
        self._camera_info = camera_info
        self._index = 0
        self._device = device
        self._region_weights = None

    def __len__(self):
        return self._batch_count

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == self._batch_count:
            raise StopIteration
        self._index += 1
        return self._prepare_batch()

    def get_batch(self):
        return self._prepare_batch()

    def _prepare_batch(self):
        batch_parts = [self._prepare_frame(x) for x in range(len(self._color_images))]
        batch = self._concatenate_batch(batch_parts)
        return self._make_tensor(batch)

    # noinspection PyTypeChecker
    def _prepare_frame(self, frame_index):
        if self._region_weights is not None:
            x, y = self.sample_pixels_with_region_weights(self._color_images[frame_index].shape[:2],
                                                          self._region_weights[frame_index])
        else:
            x, y = self.sample_pixels(self._color_images[frame_index].shape[:2])
        return {
            "pixel": np.array([x, y], dtype=np.float32).T,
            "color": self._color_images[frame_index][y, x],
            "depth": self._depth_images[frame_index][y, x],
            "frame_index": (np.ones(len(x)) * frame_index).astype(np.int64)
        }

    def sample_pixels(self, image_shape):
        x = np.random.randint(image_shape[1], size=self._points_per_image)
        y = np.random.randint(image_shape[0], size=self._points_per_image)
        return x, y

    def sample_pixels_with_region_weights(self, image_shape, region_weights):
        region_height = image_shape[0] // region_weights.shape[0]
        region_width = image_shape[1] // region_weights.shape[1]
        region_y, region_x = np.meshgrid(range(region_weights.shape[0]), range(region_weights.shape[1]))
        region_left = region_x.reshape(-1)
        region_bottom = region_y.reshape(-1)
        normalized_weights = region_weights.reshape(-1) / np.sum(region_weights)
        region_indices = np.random.choice(len(normalized_weights), size=self._points_per_image,
                                          replace=True, p=normalized_weights)
        x = region_left[region_indices] * region_width + np.random.randint(
            region_width, size=self._points_per_image)
        y = region_bottom[region_indices] * region_height + np.random.randint(
            region_height, size=self._points_per_image)
        return x, y

    @staticmethod
    def _concatenate_batch(batch_parts):
        batch = batch_parts[0]
        for part in batch_parts[1:]:
            batch = {x: np.concatenate([batch[x], part[x]]) for x in batch.keys()}
        return batch

    def _make_tensor(self, batch):
        return {x: torch.tensor(batch[x], device=self._device) for x in batch.keys()}

    def update_frames(self, frames, region_weights=None):
        self._color_images = self._camera_info.process_color_image(np.array([x.image for x in frames]))
        self._depth_images = self._camera_info.process_depth_image(np.array([x.depth for x in frames]))
        self._region_weights = region_weights
