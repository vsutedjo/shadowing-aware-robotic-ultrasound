import math

import numpy as np
from numpy import shape

from functions.helpers.helpers import flatten_index, sigmoid
from functions.scan_calculators import d_and_measurements_for_pose
from functions.helpers.measurement_functions import measure_voxel_values_for_voxel


def print_array(arr, title=""):
    assert len(arr.shape) == 1

    to_print = title
    for i in range(arr.shape[0]):
        to_print += "{0:.2f} ".format(arr[i])

    print(to_print)


def E_penalty(w, d, args):
    """ The cost function for occlusion prevention."""
    updated_d, gauss, seen_voxels = args

    sim_penalty_term = 0
    sig_w = sigmoid(w)

    # Preparation of the factors
    seen_penalty_term = 0

    # Calculate the additional cost function terms
    tmp = []
    for pose in d.keys():
        (angle, pos_row, pos_col) = pose

        # Penalize poses which see voxels that have been seen sufficiently before
        weights, _, __, ___ = d_and_measurements_for_pose((nr_rows, nr_cols, nr_planes), transducer_width,
                                                          (pos_row, pos_col), focal_dist, angle)
        # print("seen weight is ", weights * seen_voxels)
        seen_penalty = sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)] * sum(
            sum(sum(weights * seen_voxels)))
        # print("penalty = ", seen_penalty)
        seen_penalty_term += seen_penalty

        # Penalize poses which are too similar to the occluding pose

        sim_penalty = gauss[pos_row][pos_col]
        tmp.append(sim_penalty)
        sim_penalty_term += sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)] + sim_penalty

    print_array(np.array(tmp))
    #cost = E(w, d, ()) + 5.0 * sim_penalty_term + 5.0 * seen_penalty_term

    print("Original function -- seen_penalty_term: {} -- sim_penalty_term: {} ".format(seen_penalty_term, sim_penalty_term))
    print(seen_penalty_term)
    return sim_penalty_term


def E_penalty_optimiezed(w, d, args, rescanning_weights):
    """ The cost function for occlusion prevention."""
    updated_d, gauss, seen_voxels = args

    sim_penalty_term = 0
    sig_w = sigmoid(w)

    # Preparation of the factors
    seen_penalty_term = 0

    seen_penalty = np.multiply(sig_w, rescanning_weights)

    seen_penalty_term = np.sum(seen_penalty)

    sim_penalty_term = np.sum(sig_w + gauss_rep)

    print("Optimized function -- seen_penalty_term: {} -- sim_penalty_term: {} ".format(seen_penalty_term,
                                                                                       sim_penalty_term))

    # Calculate the additional cost function terms

    #cost = E(w, d, ()) + 5.0 * sim_penalty_term + 5.0 * seen_penalty_term

    return sim_penalty_term


# 1 compute d_and_measurements_for_pose once for all
import pickle

data_path = "C:\\Repo\\optimization_python\\python\\tests\\test_data\\test_E_penalty\\"
with open(data_path + "w.pickle", 'rb') as handle:
    w = pickle.load(handle)

with open(data_path + "d.pickle", 'rb') as handle:
    d = pickle.load(handle)

with open(data_path + "args.pickle", 'rb') as handle:
    args = pickle.load(handle)

nr_rows = 3
nr_cols = 3
nr_planes = 3
nr_angles = 2
angles = [0, 45]
dims = (3, 3, 3)
transducer_width = 1
focal_dist = 3.5

weights = []
for pose in d.keys():
    (angle, pos_row, pos_col) = pose

    # Penalize poses which see voxels that have been seen sufficiently before
    weight, _, __, ___ = d_and_measurements_for_pose((nr_rows, nr_cols, nr_planes), transducer_width,
                                                      (pos_row, pos_col), focal_dist, angle)
    weights.append(weight)

weights = np.array(weights)
updated_d, gauss, seen_voxels = args
seen_voxels = np.expand_dims(seen_voxels, 0)

# reshaping gauss as it should also be a function of the angle. Also reshaping in a way that
# the dim changin faster is angle, col, row
# now is [row, cols]]
gauss_rep = np.repeat(np.expand_dims(gauss, -1), axis=-1, repeats=nr_angles)
gauss_rep = gauss_rep.flatten()

# preparation to get weights to assign to voxels which are already seen. It can be computed once, as it
# is not dependent from w.
rescanning_weights = np.sum(np.multiply(weights, seen_voxels), axis=(1, 2, 3))

# rearrange weights so to match d arrangement
w_torch = w.copy()
w_torch = np.reshape(w_torch, [nr_angles, nr_rows, nr_cols])
w_torch = np.transpose(w_torch, [1, 2, 0])
w_flatten = w_torch.flatten()


ret_val = E_penalty(w, d, args)

E_penalty_optimiezed(w_flatten, d, args, rescanning_weights)