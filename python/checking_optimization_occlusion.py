import pickle
import numpy as np
import math
from functions.helpers.calculate_weights import d_and_measurements_for_pose
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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

def E_penalty(w, _, updated_d, gauss, seen_voxels, dims, angles, transducer_width, focal_dist, occluding_poses,
              occluded_voxels):
    """ The cost function for occlusion prevention."""

    nr_rows, nr_cols, nr_planes = dims
    sim_penalty_term = 0
    sig_w = sigmoid(w)

    # Preparation of the factors
    seen_penalty_term = 0

    # Calculate the additional cost function terms
    for pose in updated_d.keys():
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
        sim_penalty_term += sig_w[flatten_index((angles.index(angle), pos_row, pos_col), dims)] + sim_penalty

    cost = E(w, updated_d, dims, angles, occluded_voxels) + 2.0 * sim_penalty_term + 1.0 * seen_penalty_term

    return cost


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


def occlusion_loss(w, _, d, rescanning_weights, gauss_rep, occluded_voxels):
    """ The cost function for occlusion prevention."""

    sig_w = sigmoid(w)
    seen_penalty = np.multiply(sig_w, rescanning_weights)
    seen_penalty_term = np.sum(seen_penalty)
    sim_penalty_term = np.sum( sig_w + gauss_rep)

    cost = coverage_loss(w, d, occluded_voxels) + 2.0 * sim_penalty_term + 1.0 * seen_penalty_term

    return cost


def coverage_loss(w, d, occluded_voxels = None):
    """
    Implement the loss function to be minimized during the optimization.
    :param w: The initialized weights vector. The vector is a 1-d vector containing the weights for each pose
    parameters, and has shape (n_poses,)
    :param d: The plane-to-voxel distance for between each voxel in the volume grid and each pose.
    It has shape (n_poses, volume_grid_rows, volume_grid_cols, volume_grid_planes).
    It is CRUCIAL that the order with which the poses are ordered in w and d is consistent.

    Example. Considering to have 3 degrees of freedom for the pose, being one rotation angle and translation along
    rows and columns of the voxel grid,  and considering that there are 2 possible angles (discrete angle space
    has size 2), 3 possible position on the volume rows and 4 possible positions on the volume columns, i.e.
    nr_angles = 2, nr_rows = 3, nr_cols = 4. If we assume d to be ordered as:

        d is saved as d = [array(volume_grid_size)(angle_0, row_0, col_0),
                           array(volume_grid_size)(angle_1, row_0, col_0),
                           array(volume_grid_size)(angle_0, row_0, col_1),
                           array(volume_grid_size)(angle_1, row_0, col_1),
                           array(volume_grid_size)(angle_0, row_0, col_2),
                           array(volume_grid_size)(angle_1, row_0, col_2),
                           array(volume_grid_size)(angle_0, row_0, col_3),
                           array(volume_grid_size)(angle_1, row_0, col_3),
                           array(volume_grid_size)(angle_0, row_1, col_0),
                           array(volume_grid_size)(angle_1, row_1, col_0),
                           array(volume_grid_size)(angle_0, row_1, col_1),
                           array(volume_grid_size)(angle_1, row_1, col_1),
                           array(volume_grid_size)(angle_0, row_1, col_2),
                           array(volume_grid_size)(angle_1, row_1, col_2),
                           array(volume_grid_size)(angle_0, row_1, col_3),
                           array(volume_grid_size)(angle_1, row_1, col_3),
                           array(volume_grid_size)(angle_0, row_2, col_0),
                           array(volume_grid_size)(angle_1, row_2, col_0),
                           array(volume_grid_size)(angle_0, row_2, col_1),
                           array(volume_grid_size)(angle_1, row_2, col_1),
                           array(volume_grid_size)(angle_0, row_2, col_2),
                           array(volume_grid_size)(angle_1, row_2, col_2),
                           array(volume_grid_size)(angle_0, row_2, col_3),
                           array(volume_grid_size)(angle_1, row_2, col_3)
                           ]
        with array(volume_grid_size)(angle_i, row_j, col_k) being the an array with size of the volume voxel grid
        expressing voxel distances from the plane generated by (angle_i, row_j, col_k) pose parameters.

        THEN w must/will be ordered consistently as:

        w = [w(angle_0, row_0, col_0),
             w(angle_1, row_0, col_0),
             w(angle_0, row_0, col_1)
             w(angle_1, row_0, col_1)
             w(angle_0, row_0, col_2),
             w(angle_1, row_0, col_2),
             w(angle_0, row_0, col_3),
             w(angle_1, row_0, col_3),
             w(angle_0, row_1, col_0),
             w(angle_1, row_1, col_0),
             w(angle_0, row_1, col_1),
             w(angle_1, row_1, col_1),
             w(angle_0, row_1, col_2),
             w(angle_1, row_1, col_2),
             w(angle_0, row_1, col_3),
             w(angle_1, row_1, col_3),
             w(angle_0, row_2, col_0),
             w(angle_1, row_2, col_0),
             w(angle_0, row_2, col_1),
             w(angle_1, row_2, col_1),
             w(angle_0, row_2, col_2),
             w(angle_1, row_2, col_2),
             w(angle_0, row_2, col_3),
             w(angle_1, row_2, col_3)]

        :return: The volume coverage loss
    """

    w_sig = sigmoid(w)
    v = np.expand_dims(w_sig, axis=(1, 2, 3))

    # todo: multiply by occluded voxels [N_poses x 3 x 3 x 3] x [1 x 3 x 3 x 3] if

    voxels_torch = v * d  # N_poses x 3 x 3 x 3

    if occluded_voxels is not None:
        sum_all_voxels_torch = np.sum(np.multiply(np.expand_dims(occluded_voxels, 0), d), axis=(1, 2, 3))
    else:
        sum_all_voxels_torch = np.sum(d, axis=(1, 2, 3))

    cov_term = np.sum(np.multiply(sum_all_voxels_torch, w_sig))  # cov_term

    cov_term = math.exp(-cov_term)
    cov_term = cov_term + np.sum(w_sig)

    weighted_torch = np.sum(voxels_torch, axis=0)  # weighted for each voxel
    avg_cov_torch = np.mean(weighted_torch[occluded_voxels>0])

    weighted_torch = weighted_torch - avg_cov_torch

    if occluded_voxels is not None:
        avg_term_torch = np.sum(np.abs(weighted_torch[occluded_voxels > 0]))
    else:
        avg_term_torch = np.sum(np.abs(weighted_torch))

    loss = cov_term * 1.0 + avg_term_torch

    return loss



def main():
    # old optimization
    data_path = "C:\\Repo\\optimization_python\\python\\tests\\test_data\\test_optimization_occlusion\\old_function\\"
    with open(data_path + 'updated_d.pickle', 'rb') as handle:
        updated_d = pickle.load(handle)

    with open(data_path + 'seen_voxels.pickle', 'rb') as handle:
        seen_voxels = pickle.load(handle)

    with open(data_path + 'gauss.pickle', 'rb') as handle:
        gauss = pickle.load(handle)

    with open(data_path + 'dims.pickle', 'rb') as handle:
        dims = pickle.load(handle)

    with open(data_path + 'angles.pickle', 'rb') as handle:
        angles = pickle.load(handle)

    with open(data_path + 'transducer_width.pickle', 'rb') as handle:
        transducer_width = pickle.load(handle)

    with open(data_path + 'focal_dist.pickle', 'rb') as handle:
        focal_dist = pickle.load(handle)

    with open(data_path + 'occluded_voxels.pickle', 'rb') as handle:
        occluded_voxels = pickle.load(handle)

    nr_rows = 3
    nr_cols = 4
    nr_planes = 3
    # angles = [0, 15]
    angles = [-45, -15, 0, 15, 45]  # Write the angles from smallest to largest
    nr_angles = len(angles)
    w = np.ones(nr_rows * nr_cols * nr_angles)
    w = w / len(w)

    # new optimization
    data_path = "C:\\Repo\\optimization_python\\python\\tests\\test_data\\test_optimization_occlusion\\new_function\\"

    with open(data_path + 'occluded_voxels.pickle', 'rb') as handle:
        occluded_voxels_n = pickle.load(handle)

    with open(data_path + 'd.pickle', 'rb') as handle:
        d_n = pickle.load(handle)

    with open(data_path + 'rescanning_weights.pickle', 'rb') as handle:
        rescanning_weights_n = pickle.load(handle)

    with open(data_path + 'gauss_rep.pickle', 'rb') as handle:
        gauss_rep_n = pickle.load(handle)

    for _ in range(10):
        w = np.ones(nr_rows * nr_cols * nr_angles)
        w = w / len(w)

        w_0 = w + (np.random.random(nr_rows * nr_cols * nr_angles) - 0.5)/1000
        w_0_rearranged = np.reshape(w_0, [len(angles), nr_cols, nr_rows])
        w_0_rearranged = np.transpose(w_0_rearranged, [1, 2, 0])
        w_0_rearranged = w_0_rearranged.flatten()

        print(occlusion_loss(w, None, d_n, rescanning_weights_n, gauss_rep_n, occluded_voxels_n))
        print(E_penalty(w, None, updated_d, gauss, seen_voxels, dims, angles, transducer_width, focal_dist, None,
                        occluded_voxels))

        # minimize old function
        cost_fun_args = (updated_d, gauss, seen_voxels, dims, angles, transducer_width, focal_dist, None,
                        occluded_voxels)

        w1 = minimize(E_penalty, w_0, args=(updated_d, *cost_fun_args), method=None)["x"]
        print_array(w1, "w_old: ")

        # minimize new function
        cost_fun_args = (d_n, rescanning_weights_n, gauss_rep_n, occluded_voxels_n)
        w2 = minimize(occlusion_loss, w_0_rearranged, args=(updated_d, *cost_fun_args), method=None)["x"]

        w_reshaped = np.reshape(w2, [nr_rows, nr_cols, nr_angles])
        w = np.transpose(w_reshaped, [2, 0, 1])
        w2 = w.flatten()
        print_array(w2, "w_new: ")

        print_array(np.argsort(w1), "sorted weights old: ")
        print_array(np.argsort(w2), "sorted weights new: ")


    occluding_poses = ([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 3), (0, 1, 2), (0, 1, 1), (0, 1, 0), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)])

main()