import torch
import torch.nn.functional


def back_project_pixel(pixel, depth, camera_position, inverted_camera_matrix):
    batch_size = pixel.shape[0]
    inverted_camera_matrix = inverted_camera_matrix.to(pixel.device)
    if len(inverted_camera_matrix.shape) != 3:
        inverted_camera_matrix = torch.repeat_interleave(inverted_camera_matrix[None], batch_size, dim=0)
    homogeneous_pixel = torch.cat([pixel, torch.ones((batch_size, 1), device=pixel.device)], dim=1)
    homogeneous_keypoints = inverted_camera_matrix @ homogeneous_pixel[:, :, None]
    local_keypoints = (torch.nn.functional.normalize(homogeneous_keypoints, dim=1)) * depth[:, None, None]
    result = (camera_position[:, :3, :3] @ local_keypoints + camera_position[:, :3, 3:4])[:, :, 0]
    return result
