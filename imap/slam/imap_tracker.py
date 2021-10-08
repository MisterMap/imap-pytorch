class IMAPTracker(object):
    def __init__(self, trainer, tracker_data_loader):
        self._model = None
        self._trainer = None
        self._tracker_data_loader = None

    def update_model(self, model):
        if self._model is not None:
            del self._model
        self._model = model.clone()

    def track(self, frame):
        self.freeze_model()
        frame.model_index = self._model.add_tracked_position(self.initial_position())
        self._trainer.fit(self._model, self._tracker_data_loader)

    def initial_position(self):
        return self._model.last_position().detach().cpu().numpy()

    def freeze_model(self):
        pass
