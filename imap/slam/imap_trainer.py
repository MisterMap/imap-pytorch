import torch


class IMAPTrainer(object):
    def __init__(self, model, data_loader, device="cuda"):
        self._model = model.to(device)
        self._data_loader = data_loader
        self._optimizer = self._model.configure_optimizers()

    def fit(self, optimized_frames):
        output = None
        losses = None
        batch = None
        self._data_loader.update_frames([x.frame for x in optimized_frames])
        print("Start training")
        for batch in self._data_loader:
            self._model.set_positions(torch.stack([x.position for x in optimized_frames]))
            [x.optimizer.zero_grad() for x in optimized_frames]
            self._optimizer.zero_grad()
            output, losses = self._model.loss(batch)
            losses["loss"].backward()
            self._optimizer.step()
            [x.optimizer.step() for x in optimized_frames]
        print(f"Final loss = {losses['loss'].item()}")
        return output, losses, batch

    def postprocessing(self, output):
        pass
