import numpy as np


class ActiveSampler(object):
    def __init__(self, data_loader, active_keyframe_count=4, keep_keyframe_count=1):
        self._active_keyframe_count = active_keyframe_count
        self._keep_keyframe_count = keep_keyframe_count
        self._data_loader = data_loader

    def sample_keyframes(self, keyframes):
        if len(keyframes) <= self._keep_keyframe_count + self._active_keyframe_count:
            return keyframes
        frames = keyframes[:-self._keep_keyframe_count]
        weights = np.array([x.weight for x in frames])
        active_keyframes = self._sample(frames, weights)
        return active_keyframes + keyframes[-self._keep_keyframe_count:]

    def _sample(self, keyframes, weights):
        normalized_weights = weights / np.sum(weights)
        indices = np.random.choice(len(keyframes), size=self._active_keyframe_count,
                                   replace=False, p=normalized_weights)
        return [keyframes[x] for x in indices]

    @staticmethod
    def update_frame_weights(frames, batch, losses):
        losses = losses["loss"].cpu().detach().numpy()
        for index, frame in enumerate(frames):
            mask = (batch["frame_index"] == index).cpu().detach().numpy()
            frame.weight = np.mean(losses[mask])

    def update_region_frame_weights(self, frames, batch, losses):
        losses = losses["loss"].cpu().detach().numpy()
        pixels = batch["pixel"].cpu().detach().numpy()
        for index, frame in enumerate(frames):
            mask = (batch["frame_index"] == index).cpu().detach().numpy().astype(np.bool)
            frame.region_weights = self.calculate_region_frame_weight(frame, losses[mask], pixels[mask])

    # noinspection PyUnresolvedReferences
    @staticmethod
    def calculate_region_frame_weight(frame, losses, pixels):
        region_height = frame.shape[0] // frame.region_weights.shape[0]
        region_width = frame.shape[1] // frame.region_weights.shape[1]
        region_y, region_x = np.meshgrid(range(frame.region_weights.shape[0]), range(frame.region_weights.shape[1]))
        region_left = region_x.reshape(-1) * region_width
        region_right = (region_x.reshape(-1) + 1) * region_width
        region_top = (region_y.reshape(-1) + 1) * region_height
        region_bottom = region_y.reshape(-1) * region_height
        weights = []
        x = pixels[:, 0]
        y = pixels[:, 1]
        for i in range(len(region_left)):
            mask = (x > region_left[i]) & (x < region_right[i]) & (y > region_bottom[i]) & (y < region_top[i])
            if np.count_nonzero(mask) > 0:
                weights.append(np.mean(losses[mask]))
            else:
                weights.append(np.mean(losses))
        return np.array(weights).reshape(frame.region_weights.shape)
