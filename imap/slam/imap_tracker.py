import copy

from .imap_trainer import IMAPTrainer


class IMAPTracker(IMAPTrainer):
    def __init__(self, model, data_loader):
        model = copy.deepcopy(model)
        super(IMAPTracker, self).__init__(model, data_loader, optimize_model=False)

    def update_model(self, model):
        self._model.load_state_dict(model.state_dict())
        self._model.requires_grad_(False)

    def track(self, frame):
        print("Tracker step")
        self.fit([frame])
        return frame

