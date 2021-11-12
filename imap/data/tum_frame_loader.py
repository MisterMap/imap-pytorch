from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .tum_dataset_factory import TUMDatasetFactory, DEFAULT_CAMERA_MATRIX
from ..slam.frame import Frame


class TUMFrameLoader(object):
    def __init__(self, dataset_path, scene_name, association_file_name, frame_indices):
        sequence_directory = Path(dataset_path) / scene_name
        association_file = sequence_directory / association_file_name

        print(f"Reading {association_file}")

        associations = pd.read_csv(association_file, names=[i for i in range(12)], sep=' ')
        positions = associations.iloc[:, 5:].values
        positions = [TUMDatasetFactory.tum_position_to_matrix(positions[i]) for i in frame_indices]
        color_image_paths = [str(sequence_directory / associations.iloc[i, 1]) for i in frame_indices]
        depth_image_paths = [str(sequence_directory / associations.iloc[i, 3]) for i in frame_indices]
        self._positions = np.array(positions, dtype=np.float32)
        self._color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        self._depth_images = np.array(
            [cv2.imread(x, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000 for x in depth_image_paths])
        self._frame_indices = frame_indices

    def __getitem__(self, index):
        return Frame(self._color_images[index],
                     self._depth_images[index],
                     self._positions[index],
                     self._frame_indices[index])

    def __len__(self):
        return len(self._color_images)
