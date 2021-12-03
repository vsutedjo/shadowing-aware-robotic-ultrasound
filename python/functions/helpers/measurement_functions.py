import numpy as np
from numpy import shape


def measure_voxel_values_for_voxel(vox_location: tuple, ground_truth):
    # The measurement is the ground truth plus a random noise.
    return ground_truth[vox_location[0]][vox_location[1]][vox_location[2]] + np.random.normal(0, 0.05)


def get_avg_error_measurements(measurements, ground_truth):
    ms = measurements.ravel()
    gs = ground_truth.ravel()
    assert (shape(ms) == shape(gs))
    diffs = [abs(g - m) for g, m in zip(gs, ms)]
    return sum(diffs) / len(diffs)


def get_std_dev_measurements(measurements, avg_error):
    ms = measurements.ravel()
    variance = sum([(m - avg_error) ** 2 for m in ms]) / len(ms)
    return np.sqrt(variance)
