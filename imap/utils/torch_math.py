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


def matrix_from_9d_position(position):
    matrix = torch.zeros(position.shape[0], 3, 4, device=position.device)
    matrix[:, :3, 3] = position[:, :3]
    matrix[:, :3, :3] = rotation_matrix_from_6d_parametrization(position[:, 3:9])
    return matrix


def rotation_matrix_from_6d_parametrization(parametrization):
    x = torch.nn.functional.normalize(parametrization[:, :3])
    z = torch.nn.functional.normalize(torch.cross(parametrization[:, :3], parametrization[:, 3:]), dim=1)
    y = torch.cross(z, x)
    matrix = torch.zeros(parametrization.shape[0], 3, 3, device=parametrization.device)
    matrix[:, 0, :3] = x
    matrix[:, 1, :3] = y
    matrix[:, 2, :3] = z
    return matrix


def position_9d_from_matrix(matrix):
    position_9d = torch.zeros((matrix.shape[0], 9), device=matrix.device)
    position_9d[:, :3] = matrix[:, :3, 3]
    position_9d[:, 3:6] = matrix[:, 0, :3]
    position_9d[:, 6:9] = matrix[:, 1, :3]
    return position_9d
