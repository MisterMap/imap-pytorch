import torch

from .imap_trainer import IMAPTrainer


class IMAPMapBuilder(IMAPTrainer):
    def __init__(self, model, data_loader, active_sampler, keyframe_validator, device="cuda"):
        super().__init__(model, data_loader, device)
        self._active_sampler = active_sampler
        self._keyframe_validator = keyframe_validator
        self._keyframes = []
        self._current_frame = None
        self._keep_current_frame = True

    def set_current_frame(self, current_frame):
        self._current_frame = current_frame

    def step(self):
        print("Map builder step")
        with torch.no_grad():
            active_keyframes = self._active_sampler.sample_keyframes(self._keyframes)
            active_keyframes = active_keyframes + [self._current_frame]
            self._update_keyframe_region_weights(active_keyframes)
        output, losses, batch = self.fit(active_keyframes)
        with torch.no_grad():
            self._keep_current_frame = self._keyframe_validator.validate_keyframe(
                output, batch, len(active_keyframes) - 1)
            output, losses = self._model.loss(batch, False)
            self._active_sampler.update_frame_weights(active_keyframes, batch, losses)
        self._update_keyframes()

    def _update_keyframe_region_weights(self, frames):
        self._data_loader.update_frames([x.frame for x in frames])
        self._model.set_positions(torch.stack([x.position for x in frames]))
        batch = self._data_loader.get_batch()
        output, losses = self._model.loss(batch, False)
        self._active_sampler.update_region_frame_weights(frames, batch, losses)

    def _update_keyframes(self):
        if self._current_frame is None:
            return
        if self._current_frame in self._keyframes:
            return
        if len(self._keyframes) > 1 and not self._keep_current_frame:
            return
        self._keyframes.append(self._current_frame)
