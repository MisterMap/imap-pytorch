import pytorch_lightning as pl
import torch
from .pixel_dataset import PixelDataset
from torch.utils import data


class PixelDataModule(pl.LightningDataModule):
    def __init__(self, color_image_path, depth_image_path, split=0.98, seed=0, batch_size=4096, num_workers=4):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataset = PixelDataset(color_image_path, depth_image_path)
        dataset_length = len(self._dataset)
        train_length = int(split * dataset_length)
        val_length = dataset_length - train_length
        self._train_dataset, self._test_dataset = data.random_split(self._dataset, [train_length, val_length])

        print(f"[ToyDataModule] - train subset size {len(self._train_dataset)}")
        print(f"[ToyDataModule] - validation dataset size {len(self._test_dataset)}")

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._train_dataset, self._batch_size, shuffle=True, pin_memory=False,
                                           num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, shuffle=False, pin_memory=False,
                                           num_workers=self._num_workers)

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, shuffle=False, pin_memory=False,
                                           num_workers=self._num_workers)

    def get_inverted_camera_matrix(self):
        return self._dataset.get_inverted_camera_matrix()

    def get_default_color(self):
        return self._dataset.get_default_color()

    def get_default_depth(self):
        return self._dataset.get_default_depth()
