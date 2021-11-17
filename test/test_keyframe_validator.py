import unittest

import numpy as np
import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_dataset_factory import DEFAULT_CAMERA_MATRIX
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.model.nerf import NERF
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.keyframe_validator import KeyframeValidator
from imap.slam.optimized_frame import OptimizedFrame
from imap.utils import UniversalFactory


class TestKeyframeValidator(unittest.TestCase):
    def setUp(self) -> None:
        scene = "fire"
        sequence = "seq-01"
        dataset_path = "../test_datasets/7scenes"
        parameters = AttributeDict(
            name="NERF",
            optimizer=AttributeDict(),
            encoding_dimension=93,
            course_sample_bins=32,
            fine_sample_bins=12,
            maximal_distance=4,
            depth_loss_koef=0.5,
            color_loss_koef=1,
            encoding_sigma=25,
            optimize_positions=False,
            minimal_depth=0.001,
        )
        factory = UniversalFactory([NERF])
        camera_info = CameraInfo(4., camera_matrix=DEFAULT_CAMERA_MATRIX)
        self._model = factory.make_from_parameters(parameters, camera_info=camera_info)
        self._data_loader = IMAPDataLoader(2, 10, camera_info)
        initial_position = np.eye(4)
        self._frames = [OptimizedFrame(SevenScenesFrameLoader(dataset_path, scene, sequence, [109])[0],
                                       initial_position)]
        self._keyframe_validator = KeyframeValidator(0.1, 0.9)
        self._index = 0

    def test_sample_keyframes(self):
        self._data_loader.update_frames([x.frame for x in self._frames])
        self._model.set_positions(torch.stack([x.position for x in self._frames]))
        batch = None
        for x in self._data_loader:
            batch = x
        output, losses = self._model.loss(batch)
        self._keyframe_validator.validate_keyframe(output, batch, self._index)
