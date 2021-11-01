import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .image_rendering_dataset import ImageRenderingDataset
from .camera_info import CameraInfo

DEFAULT_CAMERA_MATRIX = np.array([[517.3, 0, 318.6],
                                 [0, 516.5, 255.3],
                                 [0, 0, 1.]], dtype=np.float32)


class TUMDatasetFactory(object):
    @staticmethod
    def make_dataset(dataset_path, scene_name, association_file_name, frame_indices, camera_matrix=DEFAULT_CAMERA_MATRIX):
        sequence_directory = Path(dataset_path) / scene_name
        association_file = sequence_directory / association_file_name
        print(f"Reading {association_file}")

        associations = pd.read_csv(association_file, names=[i for i in range(12)], sep=' ')
        positions = associations.iloc[:, 5:].values
        positions = [TUMDatasetFactory.tum_position_to_matrix(positions[i]) for i in frame_indices]
        color_image_paths = [str(sequence_directory / associations.iloc[i, 1]) for i in frame_indices]
        depth_image_paths = [str(sequence_directory / associations.iloc[i, 3]) for i in frame_indices]

        positions = np.array(positions, dtype=np.float32)
        color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        depth_images = np.array([cv2.imread(x, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000 for x in depth_image_paths])
        camera_info = CameraInfo(clip_depth_distance_threshold=4., camera_matrix=camera_matrix)
        return ImageRenderingDataset(color_images, depth_images, positions, camera_info)

    @staticmethod
    def tum_position_to_matrix(tum_position):
        """
        Convert TUM position format to matrix form
        :param tum_position: [tx ty tz qx qy qz qw]
        :return:
        """
        matrix_form = np.eye(4)
        rotation = R.from_quat(tum_position[3:])
        matrix_form[:3, :3] = rotation.as_matrix()
        matrix_form[:3, 3] = tum_position[:3]
        return matrix_form
