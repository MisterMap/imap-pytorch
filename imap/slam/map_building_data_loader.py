import pytorch_lightning as pl
import numpy as np


class MapBuildingDataLoader(object):
    def __init__(self, sample_points_per_image, batch_count):
        self._sample_points_per_image = sample_points_per_image
        self._batch_count = batch_count
        self._frames = []

    def __len__(self):
        return self._batch_count

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(self._batch_count):
            yield self.prepare_batch()

    def prepare_batch(self):
        batch = [self.prepare_frame(x) for x in self._frames]
        return self.concatenate_dictionary(batch)

    def prepare_frame(self, frame):
        sample_pixels = self.sample_pixels(frame.image.shape, frame.region_sample_weights)

    def sample_pixels(self, image_shape, region_sample_weights=None):
        pass

    def concatenate_dictionary(self, dictionary):
        pass

    def set_frames(self, frames):
        self._frames = frames
