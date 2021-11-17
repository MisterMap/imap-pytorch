import unittest

import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_dataset_factory import DEFAULT_CAMERA_MATRIX
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.model.nerf import NERF
from imap.slam.active_sampler import ActiveSampler
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.imap_map_builder import IMAPMapBuilder
from imap.slam.keyframe_validator import KeyframeValidator
from imap.slam.optimized_frame import OptimizedFrame
from imap.utils import UniversalFactory


class TestIMAPMapBuilder(unittest.TestCase):
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

        data_loader = IMAPDataLoader(20, 200, camera_info)

        sampler_data_loader = IMAPDataLoader(2, 10, camera_info)
        sampler = ActiveSampler(sampler_data_loader, 3, 1)
        keyframe_validator = KeyframeValidator(0.1, 0.9)
        self._builder = IMAPMapBuilder(self._model, data_loader, sampler, keyframe_validator)
        initial_position = np.eye(4)[:3, :]
        self._frames = [OptimizedFrame(x, initial_position) for x in SevenScenesFrameLoader(
            dataset_path, scene, sequence, [1, 81, 109, 266, 303, 406])]

    def test_builder(self):
        self._builder.set_current_frame(self._frames[0])
        self._builder.step()
        self._builder.set_current_frame(self._frames[1])
        self._builder.step()
        self._builder.set_current_frame(self._frames[2])
        self._builder.step()
