import unittest
from imap.data.pixel_data_module import PixelDataModule
from imap.model.simple_nerf import NERF
from imap.utils import UniversalFactory
from pytorch_lightning.utilities.parsing import AttributeDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestNERF(unittest.TestCase):
    def setUp(self) -> None:
        depth_image_path = "/media/mikhail/Data3T/7scenes/chess/seq-01/frame-000001.depth.png"
        color_image_path = "/media/mikhail/Data3T/7scenes/chess/seq-01/frame-000001.color.png"
        self._data_module = PixelDataModule(color_image_path, depth_image_path)
        parameters = AttributeDict(
            name="NERF",
            optimizer=AttributeDict(),
            encoding_dimension=93,
            course_sample_bins=32,
            fine_sample_bins=12,
            maximal_distance=4,
            depth_loss_koef=0.5,
            encoding_sigma=25,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1,
                                   limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
        factory = UniversalFactory([NERF])
        self._model = factory.make_from_parameters(parameters, data_module=self._data_module)

    def test_training(self):
        self._trainer.fit(self._model, self._data_module)

    def test_testing(self):
        self._trainer.test(self._model, self._data_module.test_dataloader())