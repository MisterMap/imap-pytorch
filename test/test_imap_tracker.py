import unittest

import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_dataset_factory import DEFAULT_CAMERA_MATRIX
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.model.nerf import NERF
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.imap_tracker import IMAPTracker
from imap.slam.optimized_frame import OptimizedFrame
from imap.utils import UniversalFactory


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestIMAPTracker(unittest.TestCase):
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
        tracker_data_loader = IMAPDataLoader(2, 10, camera_info)
        self._tracker = IMAPTracker(self._model, tracker_data_loader)
        self._tracker.update_model(self._model)
        initial_position = np.eye(4)
        self._frame = [OptimizedFrame(x, initial_position) for x in SevenScenesFrameLoader(
            dataset_path, scene, sequence, [1])][0]

    def test_tracker_step(self):
        self._frame.position.data = self._frame.position.detach().clone().requires_grad_(True)
        self._tracker.update_model(self._model)
        self._tracker.track(self._frame)
