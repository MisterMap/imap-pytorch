import torch


class IMAPTrainer(object):
    def __init__(self, model, data_loader, device="cuda", optimize_model=True):
        self._model = model.to(device)
        self._data_loader = data_loader
        self._optimizer = self._model.configure_optimizers()
        self._optimize_model = optimize_model

    def fit(self, optimized_frames):
        output = None
        losses = None
        batch = None
        self._data_loader.update_frames([x.frame for x in optimized_frames])
        print("Start training")
        for batch in self._data_loader:
            self._model.set_positions(torch.stack([x.position for x in optimized_frames]))
            [x.optimizer.zero_grad() for x in optimized_frames]
            if self._optimize_model:
                self._optimizer.zero_grad()
            output, losses = self._model.loss(batch)
            losses["loss"].backward()
            if self._optimize_model:
                self._optimizer.step()
            [x.optimizer.step() for x in optimized_frames]
        print(f"Final loss = {losses['loss'].item()}")
        print(f"Final image loss = {losses['fine_image_loss'].item()}")
        return output, losses, batch
