import numpy as np
import pickle

def volume_coverage(dims, volume_coverage_voxel_counts_paths):
    """
    This function computes the average volume coverage of the US sweep
    :param dims: The dimensions (rows, cols and planes) of the grid.
    :param volume_coverage_voxel_counts_path: file path to the volume coverage voxel counts pkl file
    :type: string
    :return: average voxel coverage
    :rtype: float
    """

    if not isinstance(volume_coverage_voxel_counts_paths, list):
        volume_coverage_voxel_counts_paths = [volume_coverage_voxel_counts_paths]

    volume_coverage_voxel_array = None
    [nr_rows, nr_cols, nr_planes] = dims
    all_voxel = nr_rows * nr_cols * nr_planes

    for volume_coverage_voxel_counts_path in volume_coverage_voxel_counts_paths:
        # Getting the seen voxels in an array
        with open(volume_coverage_voxel_counts_path, 'rb') as f:
            if volume_coverage_voxel_array is None:
                volume_coverage_voxel_array = pickle.load(f)
            else:
                volume_coverage_voxel_array += pickle.load(f)

    seen_voxel = np.count_nonzero(volume_coverage_voxel_array >= 1)
    avg_volume_coverage = seen_voxel / all_voxel
    avg_volume_coverage = round(avg_volume_coverage, 5)

    # print("Avg. Volume Coverage:", avg_volume_coverage)

    return avg_volume_coverage

