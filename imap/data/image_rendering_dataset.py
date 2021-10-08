from torch.utils import data
import numpy as np


class ImageRenderingDataset(data.Dataset):
    def __init__(self, color_images, depth_images, positions, camera_info):
        print(color_images.shape)
        self._camera_info = camera_info
        camera_info.update_color_normalization_parameters(color_images)
        self._color_images = camera_info.process_color_image(color_images)
        self._depth_images = camera_info.process_depth_image(depth_images)
        self._positions = positions

    def __len__(self):
        color_image_shape = self._color_images.shape
        return color_image_shape[0] * color_image_shape[1] * color_image_shape[2]

    def __getitem__(self, index):
        image_count, height, width = self._color_images.shape[:3]
        image_index = index // (width * height)
        y = (index % (width * height)) // width
        x = (index % (width * height)) % width
        return {
            "pixel": np.array([x, y], dtype=np.float32),
            "color": self._color_images[image_index, y, x],
            "depth": self._depth_images[image_index, y, x],
            "camera_position": self._positions[image_index]
        }

    def camera_info(self):
        return self._camera_info
