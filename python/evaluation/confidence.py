from functions.helpers.imfusion.imfusion_helpers import get_confidence_means_from_file
import json
import numpy as np


def get_confidence_matrix(dims, confidence_means_paths):

    """
    This functions puts the maximum mean confidence value for each voxel into a 3D numpy array.

    :param dims: The dimensions (rows, cols and planes) of the grid.
    :param confidence_means_paths: file path to the confidence means .json file
    :type: string
    :return: matrix of max. confidence mean per voxel
    :rtype: 3D numpy array
    """

    if not isinstance(confidence_means_paths, list):
        confidence_means_paths = [confidence_means_paths]

    [nr_rows, nr_cols, nr_planes] = dims

    confidence_matrix = np.zeros((nr_rows, nr_cols, nr_planes))

    for confidence_means_path in confidence_means_paths:
        confidence_means = get_confidence_means_from_file(confidence_means_path)

        for optimized_pose in confidence_means.keys():
            voxels = confidence_means[str(optimized_pose)].keys()
            for voxel in voxels:
                row = int(voxel[1])
                col = int(voxel[4])
                plane = int(voxel[7])
                if confidence_matrix[row][col][plane] == 0:
                    confidence_matrix[row][col][plane] = confidence_means[str(optimized_pose)][str(voxel)]
                elif confidence_matrix[row][col][plane] <= confidence_means[str(optimized_pose)][str(voxel)]:
                    confidence_matrix[row][col][plane] = confidence_means[str(optimized_pose)][str(voxel)]
                else:
                    confidence_matrix[row][col][plane] = confidence_matrix[row][col][plane]

    return confidence_matrix


def confidence(confidence_matrix, threshold):

    """
    This function calculates the amount of voxels above a certain threshold.
    :param confidence_matrix: matrix of max. confidence mean per voxel
    :type 3D numpy array
    :param threshold: threshold value for accepted mean confidence values
    :return: voxel count above threshold
    :rtype: int
    """

    confidence_matrix[confidence_matrix < threshold] = 0
    voxel_count_above_threshold = np.count_nonzero(confidence_matrix)

    total_voxel_amount = len(confidence_matrix[0]) * len(confidence_matrix[1]) * len(confidence_matrix[2])
    avg_confidence_count = voxel_count_above_threshold / total_voxel_amount
    avg_confidence_count  = round(avg_confidence_count,5)
    # print("The amount of voxels above the threshold of ", threshold, " amount to: ", voxel_count_above_threshold, "voxels")
    # print("Avg. confidence:", avg_confidence_count)
    return voxel_count_above_threshold, avg_confidence_count










