import os
from pathlib import Path

import cv2
import numpy as np

from ..slam.frame import Frame


class SevenScenesFrameLoader(object):
    def __init__(self, dataset_path, scene, sequence, frame_indices):
        sequence_directory = Path(dataset_path) / scene / sequence
        positions = [np.loadtxt(os.path.join(sequence_directory, 'frame-{:06d}.pose.txt'.format(i))
                                ) for i in frame_indices]
        color_image_paths = [os.path.join(sequence_directory, 'frame-{:06d}.color.png'.format(i)
                                          ) for i in frame_indices]
        depth_image_paths = [os.path.join(sequence_directory, 'frame-{:06d}.depth.png'.format(i)
                                          ) for i in frame_indices]
        self._positions = np.array(positions, dtype=np.float32)
        self._color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        self._depth_images = np.array([cv2.imread(x, -1).astype(np.float32) / 1000 for x in depth_image_paths])
        self._frame_indices = frame_indices

    def __getitem__(self, index):
        return Frame(self._color_images[index],
                     self._depth_images[index],
                     self._positions[index],
                     self._frame_indices[index])

    def __len__(self):
        return len(self._color_images)
