import h5py
import numpy as np
from matplotlib import pyplot as plt
from numba.core.serialize import pickle
from numpy import shape
from scipy.spatial.transform import Rotation as R

from functions.helpers.helpers import tracking_to_pose, print_trajectory_mat
from functions.scan_calculators import calculate_distances


def read_h5_file(filename) -> dict:
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        data = {}
        for key in f.keys():
            data[key] = np.array(f[key])
        return data


def plot_sweep(filename):
    sweep = read_h5_file(filename)
    for i in range(0, sweep[4].size() - 1):
        plt.imshow(sweep[:, :, 0, i])


def plot_compounding(filename):
    comp = read_h5_file(filename)
    minimal_comp = comp[0, 0, :, :, :]
    comp_shape = shape(minimal_comp)
    non_zero_comp = np.nonzero(minimal_comp)
    zero_comp = np.zeros(comp_shape)
    xs = np.zeros((comp_shape[0]))
    ys = np.zeros((comp_shape[0]))
    zs = np.zeros((comp_shape[0]))

    for i in range(shape(non_zero_comp)[1]):
        zero_comp[non_zero_comp[0][i], non_zero_comp[1][i], non_zero_comp[2][i]] = 1
        xs[non_zero_comp[0][i]] = 1
        ys[non_zero_comp[1][i]] = 1
        zs[non_zero_comp[2][i]] = 1

    points = np.zeros((shape(xs)[0], 3))
    for j in range(shape(xs)[0]):
        points[j, 0] = xs[j]
        points[j, 1] = ys[j]
        points[j, 2] = zs[j]
    ax = plt.axes(projection="3d")
    ax.scatter(non_zero_comp[0], non_zero_comp[1], non_zero_comp[0])
    plt.show()


def write_data_to_file(name, data):
    print("writing data to file...", name)
    a_file = open(name, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    print("done writing to file!")


# @Viviana with weights you mean distances here, correct?
def get_weights_from_file(name, params):
    angles, nr_rows, nr_cols, nr_planes, focal_dist, transducer_width = params
    print("Getting from file ", name)
    try:
        data = pickle.load(open(name, "rb"))
    except FileNotFoundError:
        print("File not found, writing weights to file...")
        dist = calculate_distances(angles, nr_rows, nr_cols, nr_planes, focal_dist=focal_dist,
                                   transducer_width=transducer_width, plot=False)
        write_data_to_file(name, dist)
        data = pickle.load(open(name, "rb"))

    print("Got data from file.")
    return data


def poses_from_tracking_file(filename, dims, roi: tuple, print_trajectory: bool = False) -> [(float, int, int)]:
    """ Open a trajectory file and read the values in."""
    with open("imfusion_files/" + filename + ".ts", 'r') as fid:
        lines = fid.readlines()

    poses = []
    for line in lines:
        split_line = line.split("\t")
        split_line = [item.replace(" ", "") for item in split_line]
        array_line = [float(item) for item in split_line]
        reshaped_mat = np.reshape(array_line[0:-2], [4, 4])
        reshaped_mat = np.transpose(reshaped_mat)

        if print_trajectory:
            print_trajectory_mat(reshaped_mat)

        # Calculate the euler angle
        rot_mat = [list(x[:3]) for x in reshaped_mat[:3]]
        r = R.from_matrix(rot_mat)

        # TODO Flip the pose by 180 degrees as it is originally flipped
        # flip_matrix = R.from_euler("Y", (0, 180, 0), True)
        # r = R.from_matrix(flip_matrix.apply(r))

        tilt_angle = r.as_euler('zyx', degrees=True)

        # The x,y,z translation
        x, y, z = float(reshaped_mat[0][3]), float(reshaped_mat[1][3]), float(reshaped_mat[2][3])

        poses.append(
            tracking_to_pose((x, y, z, float(tilt_angle[0]), float(tilt_angle[1]), float(tilt_angle[2])), dims, roi))

    return poses
