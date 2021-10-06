import pytorch_lightning as pl
import torch


def add_prefix_to_keys(dictionary, prefix):
    result = {}
    for key, value in dictionary.items():
        result[f"{prefix}_{key}"] = value
    return result


class BaseLightningModule(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__()
        self.save_hyperparameters(parameters)
        self._figure_reporter = None

    def set_figure_reporter(self, figure_reporter):
        self._figure_reporter = figure_reporter

    def metrics(self):
        return {}

    def on_train_epoch_end(self, unused=None) -> None:
        metrics = self.metrics()
        metrics = add_prefix_to_keys(metrics, "train")
        self.log_dict(metrics)

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics()
        self.log_dict(metrics)

    def on_test_epoch_end(self) -> None:
        metrics = self.metrics()
        metrics = add_prefix_to_keys(metrics, "test")
        self.log_dict(metrics)

    def training_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "val")

    def test_step(self, batch, batch_index):
        return self.learning_step(batch, batch_index, "test")

    def learning_step(self, batch, batch_index, prefix="train"):
        output, losses = self.loss(batch)
        logged_losses = {}
        for key, value in losses.items():
            logged_losses[f"{prefix}_{key}"] = value
        self.log_dict(logged_losses)
        if self.need_report_figures(batch_index, prefix):
            self.report_figures(batch, output)
        return losses["loss"]

    def loss(self, batch):
        raise NotImplementedError()

    def need_report_figures(self, batch_index, prefix):
        return (self._figure_reporter is not None) and (batch_index == 0) and (prefix == "val")

    def report_figures(self, batch, output):
        pass

    # noinspection PyUnresolvedReferences
    def configure_optimizers(self):
        if "betas" in self.hparams.optimizer.keys():
            beta1 = float(self.hparams.optimizer.betas.split(" ")[0])
            beta2 = float(self.hparams.optimizer.betas.split(" ")[1])
            self.hparams.optimizer.betas = (beta1, beta2)
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        if "scheduler" in self.hparams.keys():
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler)
            return [optimizer], [scheduler]
        return optimizer
