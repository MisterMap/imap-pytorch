import unittest

import torch

from imap.data.image_rendering_data_module import ImageRenderingDataModule


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        dataset_name = '7scenes'
        dataset_params = {
            'dataset_path': "../test_datasets/7scenes",
            'scene_name': 'fire',
            'sequence': "seq-01",
            'frame_indices': [1, 81, 109, 266, 303, 406]
        }
        self._data_module = ImageRenderingDataModule(dataset_name, **dataset_params)

    def test_load(self):
        self.assertEqual(len(self._data_module._dataset), 1843200)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["pixel"].shape, torch.Size([4096, 2]))
            self.assertEqual(batch["color"].shape, torch.Size([4096, 3]))
            self.assertEqual(batch["depth"].shape, torch.Size([4096]))
            self.assertEqual(batch["camera_position"].shape, torch.Size([4096, 4, 4]))
            break


# noinspection PyTypeChecker,PyUnresolvedReferences
class TestTumDataModule(unittest.TestCase):
    def setUp(self) -> None:
        dataset_name = 'tum'
        dataset_params = {
            'dataset_path': "../test_datasets/tum rgbd/",
            'scene_name': "rgbd_dataset_freiburg1_desk",
            'association_file_name': "data_association_file.txt",
            'frame_indices': [131, 257, 325, 407, 455]
        }
        self._data_module = ImageRenderingDataModule(dataset_name, **dataset_params)

    def test_load(self):
        self.assertEqual(len(self._data_module._dataset), 1536000)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["pixel"].shape, torch.Size([4096, 2]))
            self.assertEqual(batch["color"].shape, torch.Size([4096, 3]))
            self.assertEqual(batch["depth"].shape, torch.Size([4096]))
            self.assertEqual(batch["camera_position"].shape, torch.Size([4096, 4, 4]))
            break
