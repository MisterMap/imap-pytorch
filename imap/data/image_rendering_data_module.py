import pytorch_lightning as pl
import torch
from torch.utils import data
from .tum_dataset_factory import TUMDatasetFactory
from .seven_scenes_dataset_factory import SevenScenesDatasetFactory


class ImageRenderingDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size=4096, num_workers=4, **dataset_params):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        if dataset_name == '7scenes':
            self._dataset = SevenScenesDatasetFactory().make_dataset(**dataset_params)
        elif dataset_name == 'tum':
            self._dataset = TUMDatasetFactory().make_dataset(**dataset_params)
        print(f"[ToyDataModule] - train subset size {len(self._dataset)}")

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._dataset, self._batch_size, shuffle=True, pin_memory=False,
                                           num_workers=self._num_workers)

    def camera_info(self):
        return self._dataset.camera_info()
