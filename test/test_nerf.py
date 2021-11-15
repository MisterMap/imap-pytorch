import unittest

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.parsing import AttributeDict

from imap.data.image_rendering_data_module import ImageRenderingDataModule
from imap.model.nerf import NERF
from imap.utils import UniversalFactory


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestNERF(unittest.TestCase):
    def setUp(self) -> None:
        dataset_name = '7scenes'
        dataset_params = {
            'dataset_path': "../test_datasets/7scenes",
            'scene_name': 'fire',
            'sequence': "seq-01",
            'frame_indices': [1, 109]
        }
        self._data_module = ImageRenderingDataModule(dataset_name, **dataset_params)
        parameters = AttributeDict(
            name="NERF",
            optimizer=AttributeDict(),
            encoding_dimension=93,
            course_sample_bins=5,
            fine_sample_bins=5,
            maximal_distance=4,
            depth_loss_koef=0.5,
            color_loss_koef=1,
            encoding_sigma=25,
            optimize_positions=False,
            minimal_depth=0.001,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1,
                                   limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
        factory = UniversalFactory([NERF])
        self._model = factory.make_from_parameters(parameters, camera_info=self._data_module.camera_info())

    def test_training(self):
        self._trainer.fit(self._model, self._data_module)

    def test_testing(self):
        self._trainer.test(self._model, self._data_module.test_dataloader())
