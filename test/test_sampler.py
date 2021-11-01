import unittest

import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.data.seven_scenes_dataset_factory import DEFAULT_CAMERA_MATRIX
from imap.model.nerf import NERF
from imap.slam.active_sampler import ActiveSampler
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.posed_frame import PosedFrame
from imap.utils import UniversalFactory


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestSampler(unittest.TestCase):
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
            encoding_sigma=25,
            optimize_positions=False,
        )
        factory = UniversalFactory([NERF])
        camera_info = CameraInfo(4., camera_matrix=DEFAULT_CAMERA_MATRIX)
        self._model = factory.make_from_parameters(parameters, camera_info=camera_info)
        data_loader = IMAPDataLoader(2, 10, camera_info)
        initial_position = np.eye(4)
        self._frames = [PosedFrame(x, initial_position) for x in SevenScenesFrameLoader(
            dataset_path, scene, sequence, [1, 81, 109, 266, 303, 406])]
        self._active_sampler = ActiveSampler(data_loader, 3, 1)

    def test_sample_keyframes(self):
        active_keyframes = self._active_sampler.sample_keyframes(self._frames, self._model)
        self.assertEqual(len(active_keyframes), 4)
