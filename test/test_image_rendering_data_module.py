import unittest

import torch

from imap.data.image_rendering_data_module import ImageRenderingDataModule


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        scene = "fire"
        sequence = "seq-01"
        dataset_path = "/media/mikhail/Data3T/7scenes"
        frame_indices = [1, 2]
        self._data_module = ImageRenderingDataModule(dataset_path, scene, sequence, frame_indices)

    def test_load(self):
        self.assertEqual(len(self._data_module._dataset), 614400)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["pixel"].shape, torch.Size([4096, 2]))
            self.assertEqual(batch["color"].shape, torch.Size([4096, 3]))
            self.assertEqual(batch["depth"].shape, torch.Size([4096]))
            self.assertEqual(batch["camera_position"].shape, torch.Size([4096, 4, 4]))
            break
