import os
import cv2
import numpy as np
from pathlib import Path
from .camera_info import CameraInfo
from .image_rendering_dataset import ImageRenderingDataset

DEFAULT_CAMERA_MATRIX = np.array([[525., 0, 319.5],
                                 [0, 525., 239.5],
                                 [0, 0, 1.]], dtype=np.float32)


class SevenScenesDatasetFactory(object):
    @staticmethod
    def make_dataset(dataset_path, scene_name, sequence, frame_indices, camera_matrix=DEFAULT_CAMERA_MATRIX):
        sequence_directory = str(Path(dataset_path) / scene_name / sequence)
        print(f"Reading {sequence_directory} dataset")

        positions = [np.loadtxt(os.path.join(sequence_directory, 'frame-{:06d}.pose.txt'.format(i))
                                ) for i in frame_indices]
        color_image_paths = [os.path.join(sequence_directory, 'frame-{:06d}.color.png'.format(i)
                                          ) for i in frame_indices]
        depth_image_paths = [os.path.join(sequence_directory, 'frame-{:06d}.depth.png'.format(i)
                                          ) for i in frame_indices]

        positions = np.array(positions, dtype=np.float32)
        color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        depth_images = np.array([cv2.imread(x, -1).astype(np.float32) / 1000 for x in depth_image_paths])
        camera_info = CameraInfo(clip_depth_distance_threshold=4., camera_matrix=camera_matrix)
        return ImageRenderingDataset(color_images, depth_images, positions, camera_info)
