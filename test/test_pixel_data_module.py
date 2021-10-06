import unittest
from imap.data.pixel_data_module import PixelDataModule
import torch


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        depth_image_path = "/media/mikhail/Data3T/7scenes/chess/seq-01/frame-000001.depth.png"
        color_image_path = "/media/mikhail/Data3T/7scenes/chess/seq-01/frame-000001.color.png"
        self._data_module = PixelDataModule(color_image_path, depth_image_path)

    def test_load(self):
        self.assertEqual(len(self._data_module._train_dataset), 301056)
        self.assertEqual(len(self._data_module._test_dataset), 6144)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["pixel"].shape, torch.Size([4096, 2]))
            self.assertEqual(batch["color"].shape, torch.Size([4096, 3]))
            self.assertEqual(batch["depth"].shape, torch.Size([4096]))
            self.assertEqual(batch["camera_position"].shape, torch.Size([4096, 4, 4]))
            break
