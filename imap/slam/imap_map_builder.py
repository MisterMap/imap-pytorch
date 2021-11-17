import torch

from .imap_trainer import IMAPTrainer


class IMAPMapBuilder(IMAPTrainer):
    def __init__(self, model, data_loader, active_sampler, keyframe_validator):
        super().__init__(model, data_loader)
        self._active_sampler = active_sampler
        self._keyframe_validator = keyframe_validator
        self._keyframes = []
        self._current_frame = None
        self._keep_current_frame = True

    def set_current_frame(self, current_frame):
        self._current_frame = current_frame

    def step(self):
        print("Map builder step")
        active_keyframes = self._active_sampler.sample_keyframes(self._keyframes, self._model)
        active_keyframes = active_keyframes + [self._current_frame]
        output, losses, batch = self.fit(active_keyframes)
        with torch.no_grad():
            self._keep_current_frame = self._keyframe_validator.validate_keyframe(
                output, batch, len(active_keyframes) - 1)
        self._update_keyframes()

    def _update_keyframes(self):
        if self._current_frame is None:
            return
        if self._current_frame in self._keyframes:
            return
        if len(self._keyframes) > 1 and not self._keep_current_frame:
            return
        self._keyframes.append(self._current_frame)
