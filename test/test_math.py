import unittest

import torch

from imap.utils.math import *
from imap.utils.torch_math import *


class TestMath(unittest.TestCase):
    def setUp(self) -> None:
        translation = np.array([1., 2., 3.])
        rotation = np.array([0.1, 0.2, 0.3])
        matrix_position = np.eye(4)
        matrix_position[:3, 3] = translation
        matrix_position[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
        self._matrix_position = matrix_position

    def test_back_project_pixel(self):
        points3d = np.array([[4., 4., 4.], [5., 4., 4.], [4., 5., 4.], [6., 6., 7.]], dtype=float)
        camera_matrix = np.array([[525., 0, 320], [0, 525., 240], [0, 0, 1.]])
        invert_position = invert_positions(self._matrix_position[None])[0]
        local_point3d = invert_position[:3, :3] @ points3d.T + invert_position[:3, 3:4]
        homogeneous_keypoints = (camera_matrix @ (invert_position[:3, :3] @ points3d.T + invert_position[:3, 3:4])).T
        keypoints = homogeneous_keypoints[:, :2] / homogeneous_keypoints[:, 2][:, None]
        distances = np.linalg.norm(local_point3d, axis=0)

        inverted_camera_matrix = torch.tensor(np.linalg.inv(camera_matrix))
        depth = torch.tensor(distances)
        pixel = torch.tensor(keypoints)
        camera_position = torch.repeat_interleave(torch.tensor(self._matrix_position)[None], 4, dim=0)
        reconstructed_points3d = back_project_pixel(pixel, depth, camera_position, inverted_camera_matrix)
        reconstructed_points3d = reconstructed_points3d.detach().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(reconstructed_points3d - points3d), 0)

    def test_9d_position_converter(self):
        position9d = position_9d_from_matrix(torch.tensor([self._matrix_position[:3, :]])).detach().cpu().numpy()
        expected_position = np.array([[1., 2., 3., 0.9363, -0.2751, 0.2184, 0.2896, 0.9564, -0.0370]])
        self.assertAlmostEqual(np.linalg.norm(position9d - expected_position), 0, 3)
        position9d[3:6] *= 2
        calculated_position = matrix_from_9d_position(torch.tensor(position9d)).detach().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(calculated_position - self._matrix_position[:3, :]), 0, 3)
