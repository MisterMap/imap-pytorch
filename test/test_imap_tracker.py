import unittest

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.model.nerf import NERF
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.imap_tracker import IMAPTracker
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
            encoding_sigma=25,
            optimize_positions=False,
        )
        factory = UniversalFactory([NERF])
        camera_info = CameraInfo(4.)
        self._model = factory.make_from_parameters(parameters, camera_info=camera_info)
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        tracker_data_loader = IMAPDataLoader(2, 10, camera_info)
        self._tracker = IMAPTracker(trainer, tracker_data_loader)
        self._tracker.update_model(self._model)
        initial_position = np.eye(4)
        self._tracker.set_initial_position(initial_position)
        self._frame = SevenScenesFrameLoader(dataset_path, scene, sequence, [109])[0]

    def test_tracker_step(self):
        self._tracker.track(self._frame)
