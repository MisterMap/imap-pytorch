import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def calculate_position_errors(predicted_trajectory, truth_trajectory, show_statistic=True):
    position_errors = np.linalg.norm(truth_trajectory - predicted_trajectory, axis=1)
    if show_statistic:
        print("Mean position error: {}".format(np.mean(position_errors)))
        print("Median position error: {}".format(np.median(position_errors)))
    return position_errors


def invert_positions(positions):
    result = np.zeros_like(positions)
    result[:, :3, :3] = np.transpose(positions[:, :3, :3], (0, 2, 1))
    result[:, :3, 3] = -np.einsum('ijk, ik->ij', result[:, :3, :3], positions[:, :3, 3])
    result[:, 3, 3] = 1
    return result


def get_quaternion(positions):
    rotations = Rotation.from_matrix(positions[:, :3, :3])
    return rotations.as_quat()


def calculate_rotation_errors(predicted_rotation, truth_rotation, show_statistic=True):
    rotation_errors = [quaternion_angular_error(q1, q2) for q1, q2 in zip(truth_rotation, predicted_rotation)]
    rotation_errors = np.array(rotation_errors)
    if show_statistic:
        print("Mean rotation error: {}".format(np.mean(rotation_errors)))
        print("Median rotation error: {}".format(np.median(rotation_errors)))
    return rotation_errors


def calculate_errors(predicted_positions, truth_positions, show_statistic=True):
    position_errors = calculate_position_errors(predicted_positions[:, :3, 3], truth_positions[:, :3, 3],
                                                show_statistic)
    predicted_rotation = get_quaternion(predicted_positions)
    truth_rotation = get_quaternion(truth_positions)
    rotation_errors = calculate_rotation_errors(predicted_rotation, truth_rotation, show_statistic)
    return position_errors, rotation_errors


def pnp_position(keypoints3d, keypoints, camera_matrix, dist_coef, flags=cv2.SOLVEPNP_EPNP):
    found, rvec, tvec = cv2.solvePnP(keypoints3d, keypoints, camera_matrix,
                                     dist_coef, flags=flags)
    rotation = cv2.Rodrigues(rvec)[0]
    translation = -np.matrix(rotation).T * np.matrix(tvec)
    translation = np.array(translation)[:, 0]
    recovered_camera_position = np.zeros((4, 4))
    recovered_camera_position[:3, :3] = rotation.T
    recovered_camera_position[:3, 3] = translation
    recovered_camera_position[3, 3] = 1
    return recovered_camera_position
