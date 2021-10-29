import numpy as np


class KeyframeValidator(object):
    def __init__(self, td, tp, data_loader):
        self._data_loader = data_loader
        self._td = td
        self._tp = tp

    def validate_keyframe(self, keyframe, model):
        self._data_loader.update_frames([keyframe.frame])
        model.eval()
        model.set_positions(np.array([keyframe.position]))
        depth = np.zeros(0)
        depth_variance = np.zeros(0)
        ground_truth_depth = np.zeros(0)
        for batch in self._data_loader:
            output, losses = model.loss(batch)
            depth = np.concatenate([depth, output[3].cpu().detach().numpy()])
            depth_variance = np.concatenate([depth_variance, output[5].cpu().detach().numpy()])
            ground_truth_depth = np.concatenate([ground_truth_depth, batch["depth"].cpu().detach().numpy()])
        koefs = np.abs(depth - ground_truth_depth) / depth_variance
        criterion = np.mean(np.where(koefs < self._td, 1, 0))
        return criterion < self._tp
