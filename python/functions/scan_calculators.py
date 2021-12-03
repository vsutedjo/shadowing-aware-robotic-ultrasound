import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
from tqdm import tqdm

from functions.helpers.calculate_weights import d_and_measurements_for_pose
from functions.cutoff_functions import all_voxels_under_thresh, CutoffState
from functions.helpers.helpers import clean_occlusions
from functions.helpers.plot_functions import plot_transducer, plot_voxel_measurements, plot_occlusion, setup_plot, \
    get_intersected_voxels
from functions.helpers.measurement_functions import get_avg_error_measurements, get_std_dev_measurements


def calculate_distances(angles: list, nr_rows: int, nr_cols: int, nr_planes: int, transducer_width: float = 1,
                        focal_dist: float = 0.0, plot: bool = True) -> {(float, float, float): [float]}:
    """ A function that calculates the d function.
    This function can plot the transducer poses and the corresponding scanned voxels while calculating.
    It returns the distances from a pose to a scanned voxel,
    or 0 if the voxel has not been intersected by this transducer."""
    # Init voxels for plotting with ones
    voxels = np.ones((nr_rows, nr_cols, nr_planes))
    ax = None
    if plot:
        _, fig, ax = setup_plot(nr_cols, voxels, plot_multiple=False)
    d = {}
    for i in tqdm(range(nr_rows)):
        for j in tqdm(range(nr_cols)):
            for angle in tqdm(angles):
                # We are looking at one specific pose.
                d[(angle, i, j)], _, _, _ = d_and_measurements_for_pose(shape(voxels), transducer_width, [i, j + 0.5],
                                                                        focal_dist,
                                                                        angle,
                                                                        plt_meta=(ax, 0.02))
                if plot:
                    td = plot_transducer(nr_planes, transducer_width, [i, j + 0.5], angle, ax)
                    plt.draw()
                    plt.pause(0.05)
                    td.remove()
    if plot:
        plt.draw()
        plt.pause(0.5)
        plt.close("all")
    return d


def measure_optimization_results(transducer_positions: list, nr_rows: int, nr_cols: int, nr_planes: int,
                                 cutoff_function=all_voxels_under_thresh, transducer_width: float = 1,
                                 focal_dist: float = 0.0,
                                 plot: bool = False, occlusion_volume=None, old_measurements: list = None,
                                 print_poses: bool = False,
                                 block_plot: bool = False, synthetic: bool = False) -> (
        (list, list, {(float, float, float): [(int, int, int)]}), {(float, float, float): [(int, int, int)]},
        [(float, int, int)]):
    """ This function takes a sorted list of poses and executes the scan while measuring the poses.
    The measurement includes information about the occluded poses/voxels and the noisy weights."""

    # For re-computation of optimization, save all poses that have occlusions
    occlusions = {}

    # Setup all vars
    measurements = np.zeros((nr_rows, nr_cols, nr_planes))
    voxel_counts = np.zeros((nr_rows, nr_cols, nr_planes))
    ground_truth = np.ones((nr_rows, nr_cols, nr_planes))
    poses = []
    old_occlusions = {}
    if old_measurements is not None:
        measurements, voxel_counts, old_occlusions = old_measurements

    # If we don't specify any transducer positions, we can immediately return.
    if transducer_positions is None:
        return (measurements, voxel_counts, occlusions), occlusions, []

    # Init voxels for plotting with ones
    voxels = np.ones((nr_rows, nr_cols, nr_planes))

    ax = None
    fig = None
    cumul_err_ax = None
    measurement_ax = None

    # Setup the plot init
    if plot:
        spec, fig, ax = setup_plot(nr_cols, voxels, plot_multiple=synthetic)
        if synthetic:
            # Set up the additional subplots
            cumul_err_ax = fig.add_subplot(223)
            measurement_ax = fig.add_subplot(spec[:, 1], projection="3d")

            # Disable the grids
            ax.grid(False)
            measurement_ax.grid(False)

        # This pause is here so we can prepare for screen recording the following poses.
        plt.pause(1)

    # The calculation of cumulative coverage and the cutoff value.
    avg_error_list = []
    cov_list = [0]
    std_dev_list = []
    voxel_cumul_dists = np.zeros((nr_rows, nr_cols, nr_planes))
    cov = 0
    nr_pos = 0

    # Iterate through all positions. Plot the position and measure the value it sees in the voxels.
    for pos in transducer_positions:
        nr_pos += 1

        # Find out if we want to cut off now.
        cutoff_args = (voxel_cumul_dists, occlusions, old_occlusions, voxel_counts, nr_pos)
        cutoff_state = cutoff_function(cutoff_args)
        # If we should cut off, stop the loop
        if cutoff_state == CutoffState.STOP:
            break
        # If we should skip this pose because of the local maximum, continue
        elif cutoff_state == CutoffState.SKIP:
            continue

        poses.append(pos)
        if print_poses:
            print(pos)

        (angle, row, col) = pos
        d_pose, measurements, voxel_counts, occ_vox = d_and_measurements_for_pose(shape(voxels), transducer_width,
                                                                                  [row, col + 0.5],
                                                                                  focal_dist, angle,
                                                                                  measurements=measurements,
                                                                                  voxel_counts=voxel_counts,
                                                                                  plt_meta=(ax, 0.1),
                                                                                  occlusion=occlusion_volume,
                                                                                  ground_truth=ground_truth)

        # If this pose had an occlusion, we should save it for re-computation later.
        if len(occ_vox) > 0:
            occlusions[pos] = occ_vox

        # Calculate the metrics for visualization.
        voxel_cumul_dists += d_pose
        # divided_measurements = np.divide(weights, voxel_counts, where=voxel_counts > 0)
        # avg_err = get_avg_error_measurements(divided_measurements, ground_truth)
        # std_dev_err = get_std_dev_measurements(divided_measurements, avg_err)

        avg_err = 0.0
        std_dev_err = 0.0
        std_dev_list.append(std_dev_err)
        avg_error_list.append(avg_err)
        w = sum(sum(sum(d_pose)))
        cov += w
        # cutoff_list.append(w - cov_list[-1])
        cov_list.append(w)
        # avg_error_list.append(cov)

        if plot:
            if synthetic:
                label = "avg_err: " + str(round(avg_err, 4)) + ", std_dev_err: " + str(round(std_dev_err, 4))
                fig.suptitle(label)
                cumul_err_ax.cla()
                measurement_ax.cla()
                # cumul_ax.plot(cumul_cov_list)
                cumul_err_ax.plot(avg_error_list)
                cumul_err_ax.set_xlabel("nr. sweeps")
                cumul_err_ax.set_ylabel("avg.error")
                # secondary_ax.plot(cutoff_list)
                measurement_ax.grid(False)
                measurement_ax.voxels(voxels, facecolors=[0.6, 0.6, 0.6, 0.1], edgecolor=[0.5, 0.5, 0.5, 0.2])
                # plot_voxel_measurements(divided_measurements, measurement_ax)

            # measurement_ax.set_xlabel(label, fontsize=10)
            td = plot_transducer(nr_planes, transducer_width, [row, col + 0.5], angle, ax)
            plt.draw()
            plt.pause(1)
            td.remove()
        # print(voxel_counts)

    print("Done measuring, returning...")
    if plot:
        plt.draw()
        # plt.pause(5)
        if block_plot:
            plt.show(block=True)
        plt.clf()

    # Remove occluded voxels and poses which have been covered by other poses during this scan
    occlusions = clean_occlusions(occlusions, voxel_counts)

    # Return the measured values and the occlusions
    return (measurements, voxel_counts, occlusions), occlusions, poses
