import numpy as np


class KeyframeValidator(object):
    def __init__(self, td, tp):
        self._td = td
        self._tp = tp

    def validate_keyframe(self, output, batch, index):
        mask = (batch["frame_index"] == index).cpu().detach().numpy()
        mask &= batch["depth"].cpu().detach().numpy() > 0
        depth = output[3].cpu().detach().numpy()[mask]
        depth_variance = output[5].cpu().detach().numpy()[mask]
        ground_truth_depth = batch["depth"].cpu().detach().numpy()[mask]
        koefs = np.abs(depth - ground_truth_depth) / (np.sqrt(depth_variance) + 1e-10)
        criterion = np.mean(np.where(koefs < self._td, 1, 0))
        print(f"Keyframe validator criterion = {criterion}")
        print(f"Keyframe is validated = {criterion < self._tp}")
        return criterion < self._tp
