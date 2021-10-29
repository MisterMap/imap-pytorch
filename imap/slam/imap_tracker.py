import copy


class IMAPTracker(object):
    def __init__(self, trainer, tracker_data_loader):
        self._model = None
        self._trainer = trainer
        self._tracker_data_loader = tracker_data_loader
        self._initial_position = None

    def update_model(self, model):
        if self._model is not None:
            del self._model
        self._model = copy.deepcopy(model)
        self._model.freeze_model()
        self._model.unfreeze_positions()

    def set_initial_position(self, initial_position):
        self._initial_position = initial_position

    def track(self, frame):
        self._model.set_positions([self._initial_position])
        self._tracker_data_loader.update_frames([frame])
        self._trainer.fit(self._model, self._tracker_data_loader)
        self._initial_position = self._model.get_positions()[0]
        return self._initial_position
