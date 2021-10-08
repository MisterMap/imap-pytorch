import numpy as np


class CameraInfo(object):
    def __init__(self, clip_depth_distance_threshold):
        camera_matrix = np.array([[525., 0, 320],
                                     [0, 525., 240],
                                     [0, 0, 1.]], dtype=np.float32)
        self._inverted_camera_matrix = np.linalg.inv(camera_matrix)
        self._clip_depth_distance_threshold = clip_depth_distance_threshold
        self._color_mean = np.ones(3) * 127.
        self._color_std = np.ones(3)

    def process_depth_image(self, depth_image):
        depth_image = self.convert_depths(depth_image)
        return np.clip(depth_image, 0, self._clip_depth_distance_threshold)

    def convert_depths(self, depth_image):
        y, x = np.meshgrid(range(depth_image.shape[1]), range(depth_image.shape[2]))
        y = y.reshape(-1)
        x = y.reshape(-1)
        homogeneous_pixel = np.array([x, y, np.ones(y.shape[0])])
        homogeneous_keypoints = self._inverted_camera_matrix @ homogeneous_pixel
        koefs = np.linalg.norm(homogeneous_keypoints.T, axis=1).reshape(depth_image.shape[1], depth_image.shape[2])
        return depth_image * koefs[None, :, :]

    def process_color_image(self, color_image):
        return (color_image - self._color_mean) / self._color_std

    def get_inverted_camera_matrix(self):
        return self._inverted_camera_matrix

    def get_default_color(self):
        return (np.array([255., 255., 255.], dtype=np.float32) - self._color_mean) / self._color_std

    def get_default_depth(self):
        return self._clip_depth_distance_threshold

    def update_color_normalization_parameters(self, data):
        self._color_mean = np.mean(data.reshape(-1, 3), axis=0)
        self._color_std = np.mean(data.reshape(-1, 3), axis=0)
