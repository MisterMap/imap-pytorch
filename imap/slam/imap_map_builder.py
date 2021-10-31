import numpy as np


class IMAPMapBuilder(object):
    def __init__(self, trainer, model, map_builder_data_loader, active_sampler, keyframe_validator):
        self._trainer = trainer
        self._model = model
        self._map_builder_data_loader = map_builder_data_loader
        self._active_sampler = active_sampler
        self._keyframe_validator = keyframe_validator
        self._keyframes = []
        self._current_frame = None
        self._keep_current_frame = True

    def init(self, frame):
        self._current_frame = frame

    def step(self):
        active_keyframes = self._active_sampler.sample_keyframes(self._keyframes, self._model)
        active_keyframes = active_keyframes + [self._current_frame]
        self._map_builder_data_loader.update_frames([x.frame for x in active_keyframes])
        self._update_model_positions(active_keyframes)
        self._trainer.fit(self._model, self._map_builder_data_loader)
        self._update_keyframe_positions(active_keyframes)
        self._keep_current_frame = self._keyframe_validator.validate_keyframe(self._current_frame, self._model)

    def add_current_frame(self, current_frame):
        if len(self._keyframes) < 2 and self._keep_current_frame and self._current_frame is not None:
            self._keyframes.append(self._current_frame)
        self._current_frame = current_frame

    def _update_model_positions(self, keyframes):
        positions = np.array([x.position for x in keyframes])
        self._model.set_positions(positions)

    def _update_keyframe_positions(self, keyframes):
        positions = self._model.get_positions()
        for frame, position in zip(keyframes, positions):
            frame.positions = position
