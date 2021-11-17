import torch

from ..utils.torch_math import matrix_from_9d_position, position_9d_from_matrix


class OptimizedFrame(object):
    def __init__(self, frame, position, lr=1e-3, device="cuda"):
        self.frame = frame
        with torch.no_grad():
            position9d = position_9d_from_matrix(torch.tensor(position, device=device)[None])[0]
        self.position = position9d.clone().detach().requires_grad_(True)
        self.optimizer = torch.optim.Adam(params=[self.position], lr=lr)

    @property
    def matrix_position(self):
        return matrix_from_9d_position(self.position[None])[0].cpu().detach().numpy()
