import numpy as np
from scipy.optimize import minimize
import pickle

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

import math


def print_array(arr, title=""):
    assert len(arr.shape) == 1

    to_print = title
    for i in range(arr.shape[0]):
        to_print += "{0:.2f}, ".format(arr[i])

    print(to_print)

def flatten_index(tup, dims):
    """ Get from a pose tuple to its index in a large array."""
    nr_rows, nr_cols, nr_planes = dims
    (angle, row, col) = tup
    return col + (row * nr_cols) + (angle * (nr_rows * nr_cols))


def E(w, d, dims, angles, occluded_voxels=None):
    """ The cost function for volume coverage."""
    sig_w = sigmoid(w)
    cost = 0
    nr_rows, nr_cols, nr_planes = dims

    voxels = np.zeros((nr_rows, nr_cols, nr_planes))

    # Maximize coverage with minimal sweeps
    cov_term = 0
    # Iterate through all poses
    for pose in d.keys():
        (angle, pos_row, pos_col) = pose
        pose_voxels = d[pose]
        # Add weighted voxels to voxel coverage
        voxels += pose_voxels * sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)]

        sum_all_voxels = 0
        if occluded_voxels is None:
            # Iterate through all voxels for that pose
            sum_all_voxels = sum(sum(sum(pose_voxels)))
        else:
            # Sum only over the voxels that are occluded, if occlusion is present
            # Voxels that are not occluded will not contribute to the term, so the optimizer will favor
            # poses that have many voxels > 0
            for occ_vox in occluded_voxels:
                x, y, z = occ_vox
                sum_all_voxels += pose_voxels[x][y][z]

        weighted_sum_all_voxels = sum_all_voxels * sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)]
        cov_term += weighted_sum_all_voxels
    cov_term = math.exp(-cov_term)
    cov_term += sum(sig_w)

    # Minimize std deviation for voxels
    # Calculate average coverage: for each voxel, how often it was touched
    avg_cov = 0
    if occluded_voxels is None:
        avg_cov = sum(sum(sum(voxels))) / (nr_rows * nr_cols * nr_planes)
    else:
        # Only calculate the std dev for the occluded voxels and ignore the rest
        for vox in occluded_voxels:
            x, y, z = vox
            avg_cov += voxels[x][y][z]
        avg_cov /= len(occluded_voxels)

    avg_term = 0
    if occluded_voxels is None:
        # Iterate over all voxels
        for row in range(nr_rows):
            for col in range(nr_cols):
                for plane in range(nr_planes):
                    # For each voxel, iterate over all poses
                    weighted = 0
                    for pose in d.keys():
                        (angle, pos_row, pos_col) = pose
                        # Calculate the weighted coverage of this voxel
                        weighted += d[pose][row, col, plane] * sig_w[
                            flatten_index((angles.index(angle), pos_row, pos_col), dims)]
                    # Subtract the average coverage to get the deviation
                    weighted -= avg_cov

                    # Multiply by weight for "importance" of the voxel
                    avg_term += abs(weighted)
    else:
        # Iterate over all occluded voxels
        for vox in occluded_voxels:
            x, y, z = vox
            weighted = 0

            for pose in d.keys():
                (angle, pos_row, pos_col) = pose
                # Calculate the weighted coverage of this voxel
                weighted += d[pose][x, y, z] * sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)]
            # Subtract the average coverage to get the deviation
            weighted -= avg_cov

            # Multiply by weight for "importance" of the voxel
            avg_term += abs(weighted)

    # Calculate cost with differently weighted terms
    cost += cov_term * 1.0 + avg_term

    return cost

def coverage_loss(w, d, occluded_voxels = None):

    w_sig = sigmoid(w)
    v = np.expand_dims(w_sig, axis=(1, 2, 3))
    voxels_torch = v * d  # N_poses x 3 x 3 x 3

    if occluded_voxels is not None:
        sum_all_voxels_torch = np.sum(np.multiply(np.expand_dims(occluded_voxels, 0), d), axis=(1, 2, 3))
    else:
        sum_all_voxels_torch = np.sum(d, axis=(1, 2, 3))

    cov_term = np.sum(np.multiply(sum_all_voxels_torch, w_sig))  # cov_term

    cov_term = math.exp(-cov_term)
    cov_term = cov_term + np.sum(w_sig)

    weighted_torch = np.sum(voxels_torch, axis=0)  # weighted for each voxel

    if occluded_voxels is not None:
        avg_cov_torch = np.mean(weighted_torch[occluded_voxels>0])
    else:
        avg_cov_torch = np.mean(weighted_torch)

    weighted_torch = weighted_torch - avg_cov_torch

    if occluded_voxels is not None:
        avg_term_torch = np.sum(np.abs(weighted_torch[occluded_voxels > 0]))
    else:
        avg_term_torch = np.sum(np.abs(weighted_torch))

    loss = cov_term * 1.0 + avg_term_torch

    return loss

def main():
    d_path = "tests/test_data/test_E_penalty/d.pickle"
    nr_rows = 3
    nr_cols = 3
    nr_angles = 2
    dims = (3, 3, 3)
    angles = [0, 45]
    # 1. Load d and initialize weights

    with open(d_path, 'rb') as handle:
        d = pickle.load(handle)

    # 2. Check the functions yield same results
    d_array = np.array([d[key] for key in d.keys()])

    # 3. Check the optimization - what is different?
    cost_fun_args = (dims, angles)

    for _ in range(10):
        w = np.ones(nr_rows * nr_cols * nr_angles)
        w = w / len(w)

        w_0 = w + (np.random.random(nr_rows * nr_cols * nr_angles) - 0.5)/1000
        w_0_rearranged = np.reshape(w_0, [len(angles), nr_cols, nr_rows])
        w_0_rearranged = np.transpose(w_0_rearranged, [1, 2, 0])
        w_0_rearranged = w_0_rearranged.flatten()

        print(E(w_0, d, *cost_fun_args))
        print(coverage_loss(w_0_rearranged, d_array))

        w1 = minimize(E, w_0, args=(d, *cost_fun_args), method=None)["x"]
        print_array(w1, "w_old: ")

        w2 = minimize(coverage_loss, w_0_rearranged, args=(d_array), method=None)["x"]

        w_reshaped = np.reshape(w2, [nr_rows, nr_cols, nr_angles])
        w = np.transpose(w_reshaped, [2, 0, 1])
        w2 = w.flatten()
        print_array(w2, "w_new: ")

        print_array(np.argsort(w1), "sorted weights old: ")
        print_array(np.argsort(w2), "sorted weights new: ")

main()