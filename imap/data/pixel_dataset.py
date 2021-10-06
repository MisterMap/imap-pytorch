from torch.utils import data
import cv2
import numpy as np


class PixelDataset(data.Dataset):
    def __init__(self, color_image_path, depth_image_path, clip_distance_threshold=4.):
        color_image = cv2.imread(color_image_path).astype(np.float32)
        self._color_image, self._color_mean, self._color_std = self.normalize(color_image)
        depth_image = cv2.imread(depth_image_path, -1).astype(np.float32) / 1000
        self._clip_distance_threshold = clip_distance_threshold
        self._depth_image = np.clip(depth_image, 0, clip_distance_threshold)

        self._position = np.eye(4, dtype=np.float32)
        camera_matrix = np.array([[525., 0, 320],
                                  [0, 525., 240],
                                  [0, 0, 1.]], dtype=np.float32)
        self._inverted_camera_matrix = np.linalg.inv(camera_matrix)

    @staticmethod
    def normalize(image):
        mean = np.mean(image.reshape(-1, 3), axis=0)
        std = np.std(image.reshape(-1, 3), axis=0)
        return (image - mean) / std, mean, std

    def __len__(self):
        color_image_shape = self._color_image.shape
        return color_image_shape[0] * color_image_shape[1]

    def __getitem__(self, index):
        height, width = self._color_image.shape[:2]
        y = index // width
        x = index % width
        return {
            "pixel": np.array([x, y], dtype=np.float32),
            "color": self._color_image[y, x],
            "depth": self._depth_image[y, x],
            "camera_position": self._position
        }

    def get_inverted_camera_matrix(self):
        return self._inverted_camera_matrix

    def get_default_color(self):
        return (np.array([255., 255., 255.], dtype=np.float32) - self._color_mean) / self._color_std

    def get_default_depth(self):
        return self._clip_distance_threshold
