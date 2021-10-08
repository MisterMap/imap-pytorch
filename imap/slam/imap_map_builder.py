class IMAPMapBuilder(object):
    def __init__(self, trainer, model, map_builder_data_loader, active_sampler, keyframe_validator):
        self._trainer = trainer
        self._model = model
        self._map_builder_data_loader = map_builder_data_loader
        self._keyframes = []
        self._active_sampler = active_sampler
        self._keyframe_validator = keyframe_validator
        self._keep_previous_frame = True

    def init(self, frame):
        self.update_keyframes(frame)

    def step(self):
        active_keyframes = self._active_sampler.sample_keyframes(self._keyframes, self._model)
        self._map_builder_data_loader.set_frames(active_keyframes)
        self._trainer.fit(self._model, self._map_builder_data_loader)
        self._keep_previous_frame = self._keyframe_validator.validate_keyframe(self._keyframes[-1], self._model)

    def add_current_frame(self, current_frame):
        self.update_keyframes(current_frame)
        current_frame.model_index = self._model.add_tracked_position(current_frame.position)

    def update_keyframes(self, current_frame):
        if len(self._keyframes) >= 2 and not self._keep_previous_frame:
            self._keyframes.pop(len(self._keyframes) - 1)
        self._keyframes.append(current_frame)
