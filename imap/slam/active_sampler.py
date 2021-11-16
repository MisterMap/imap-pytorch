import numpy as np


class ActiveSampler(object):
    def __init__(self, data_loader, active_keyframe_count=4, keep_keyframe_count=1):
        self._active_keyframe_count = active_keyframe_count
        self._keep_keyframe_count = keep_keyframe_count
        self._data_loader = data_loader

    def sample_keyframes(self, keyframes, model):
        if len(keyframes) <= self._keep_keyframe_count + self._active_keyframe_count:
            return keyframes
        # weights = self._get_sample_weights(keyframes[:-self._keep_keyframe_count], model)
        weights = np.ones(len(keyframes) - self._keep_keyframe_count)
        active_keyframes = self._sample(keyframes[:-self._keep_keyframe_count], weights)
        return active_keyframes + keyframes[-self._keep_keyframe_count:]

    def _get_sample_weights(self, frames, model):
        self._data_loader.update_frames([x.frame for x in frames])
        model.eval()
        model.set_positions(np.array([x.position for x in frames]))
        image_indices = np.zeros(0)
        loss = np.zeros(0)
        for batch in self._data_loader:
            output, losses = model.loss(batch, False)
            image_indices = np.concatenate([image_indices, batch["frame_index"].cpu().detach().numpy()])
            loss = np.concatenate([loss, losses["loss"].cpu().detach().numpy()])
        result = []
        for i in range(len(frames)):
            result.append(np.mean(loss[image_indices == i]))
        return np.array(result)

    def _sample(self, keyframes, weights):
        normalized_weights = weights / np.sum(weights)
        indices = np.random.choice(len(keyframes), size=self._active_keyframe_count,
                                   replace=False, p=normalized_weights)
        return [keyframes[x] for x in indices]
