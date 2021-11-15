import unittest

import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.camera_info import CameraInfo
from imap.data.seven_scenes_frame_loader import SevenScenesFrameLoader
from imap.data.seven_scenes_dataset_factory import DEFAULT_CAMERA_MATRIX
from imap.model.nerf import NERF
from imap.slam.imap_data_loader import IMAPDataLoader
from imap.slam.keyframe_validator import KeyframeValidator
from imap.slam.posed_frame import PosedFrame
from imap.utils import UniversalFactory


# noinspection PyTypeChecker,PyUnresolvedReferences
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
        data_loader = IMAPDataLoader(2, 10, camera_info)
        initial_position = np.eye(4)
        self._frame = PosedFrame(SevenScenesFrameLoader(dataset_path, scene, sequence, [109])[0], initial_position)
        self._keyframe_validator = KeyframeValidator(0.1, 0.9, data_loader)

    def test_sample_keyframes(self):
        self._keyframe_validator.validate_keyframe(self._frame, self._model)
