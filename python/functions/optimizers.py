import pickle

import numpy as np
from numpy import shape
from scipy.optimize import minimize

from functions.cutoff_functions import all_voxels_under_thresh, occluded_cutoff
from functions.helpers.gaussians import bivariate_gauss
from functions.helpers.helpers import map_weights_to_poses, get_std_name, calculate_possible_poses
from functions.helpers.imfusion.imfusion_main import get_occlusions_from_simulation, IterationType
from functions.helpers.plot_functions import plot_occluded_voxels, prepare_poses_list
from functions.helpers.read_from_file import get_weights_from_file, write_data_to_file
from functions.helpers.random_trajectories import get_random_poses
from functions.scan_calculators import measure_optimization_results
from functions.cost_functions import occlusion_prevention_pre_processing, coverage_loss, post_process_weights, \
    coverage_loss_preparation, occlusion_loss, occlusion_loss_preparation


class OptimizersOptions:
    def __init__(self, pre_processing_function, post_processing_function, optimization_function,
                 method=None, framework='scipy'):
        self.pre_processing_function = pre_processing_function
        self.post_processing_function = post_processing_function
        self.method = method
        self.framework = framework
        self.optimization_function = optimization_function


class AcquisitionGeometryParams:
    def __init__(self, angles, nr_rows, nr_cols, nr_planes, focal_dist, transducer_width, dims, model_name, material_type, occ_threshold):
        self.angles = angles
        self.nr_rows = nr_rows
        self.nr_cols = nr_cols
        self.nr_planes = nr_planes
        self.focal_dist = focal_dist
        self.transducer_width = transducer_width
        self.dims = dims
        self.model_name = model_name
        self.material_type = material_type
        self.occ_threshold = occ_threshold

    def get_params(self):
        return self.angles, self.nr_rows, self.nr_cols, self.nr_planes, self.focal_dist, self.transducer_width


def random_scan(acquisition_params, synthetic: bool = False, show_occluded_voxels_after_simulation: bool = True,
                nr_scans=None,
                visualize_scans=True):
    """
    A random trajectory for comparison.
    """
    poses = get_random_poses(acquisition_params.nr_rows, acquisition_params.nr_cols, acquisition_params.angles,
                             nr_poses=nr_scans)
    measurements, voxel_counts, occlusions = measure_optimization_results(poses, acquisition_params.nr_rows,
                                                                          acquisition_params.nr_cols,
                                                                          acquisition_params.nr_planes,
                                                                          focal_dist=acquisition_params.focal_dist,
                                                                          transducer_width=acquisition_params.transducer_width)
    if not synthetic:
        occlusions, voxel_counts, vc_path = get_occlusions_from_simulation(acquisition_params.dims,
                                                                           poses,
                                                                           acquisition_params.model_name,
                                                                           acquisition_params.material_type,
                                                                           acquisition_params.transducer_width,
                                                                           acquisition_params.angles,
                                                                           iteration_type=IterationType.RANDOM,
                                                                           old_voxel_counts=None,
                                                                           occ_threshold=acquisition_params.occ_threshold,
                                                                           plot_simulation=visualize_scans,
                                                                           init=True)
        measurements = None
        if show_occluded_voxels_after_simulation:
            plot_occluded_voxels(occlusions, acquisition_params.dims)  # Plot the measured occlusions
        print("After running random simulation, occlusions are: ", occlusions)

    return measurements, voxel_counts, occlusions, poses, vc_path


def perpendicular_scan(acquisition_params, synthetic: bool = False, show_occluded_voxels_after_simulation: bool = True,
                       visualize_scans: bool = False):
    """
    A state-of-the-art perpendicular scan for comparison with the pipeline.
    :param synthetic: whether the desired scan is purely synthetic or US simulation.
    :type synthetic: bool
    """
    poses = calculate_possible_poses(acquisition_params.nr_rows, acquisition_params.nr_cols, [0])

    measurements, voxel_counts, occlusions = measure_optimization_results(poses, acquisition_params.nr_rows,
                                                                          acquisition_params.nr_cols,
                                                                          acquisition_params.nr_planes,
                                                                          focal_dist=acquisition_params.focal_dist,
                                                                          transducer_width=acquisition_params.transducer_width)
    if not synthetic:
        occlusions, voxel_counts, vc_path = get_occlusions_from_simulation(acquisition_params.dims,
                                                                           poses,
                                                                           acquisition_params.model_name,
                                                                           acquisition_params.material_type,
                                                                           acquisition_params.transducer_width,
                                                                           acquisition_params.angles,
                                                                           iteration_type=IterationType.PERPENDICULAR,
                                                                           old_voxel_counts=None,
                                                                           occ_threshold=acquisition_params.occ_threshold,
                                                                           plot_simulation=visualize_scans,
                                                                           init=True)
        measurements = None
        if show_occluded_voxels_after_simulation:
            plot_occluded_voxels(occlusions, acquisition_params.dims)  # Plot the measured occlusions
        print("After running perpendicular simulation, occlusions are: ", occlusions)

    return (measurements, voxel_counts, occlusions), poses, vc_path


def optimize(d: {(float, float, float): float},
             cost_fun, cost_fun_args,
             measured_arrs,
             cutoff_fun,
             acquisition_params,
             block_plot: bool = False,
             old_poses: list = [],
             preparation_fun=None,
             post_processing_function=None,
             method=None,
             optimization_framework='scipy') -> (
        (list, list, {(float, float, float): [(int, int, int)]}), {(float, float, float): [(int, int, int)]},
        [(float, int, int)]):
    """
    The actual optimizer, can be flexibly used with different kinds of cost functions, params etc.
    :param d: the distance function result
    :type d: dict
    :param cost_fun: the cost function we are optimizing on.
    :type cost_fun: function
    :param cost_fun_args: additional arguments for the cost function
    :type cost_fun_args: dynamic
    :param measured_arrs: (weights, voxel counts, occlusions)
    :type measured_arrs: tuple
    :param cutoff_fun: the cutoff function after which to halt the execution.
    :type cutoff_fun: function
    :param block_plot: when plotting, whether to block the pipeline when plotting.
    :type block_plot: bool
    :param print_poses: whether to print out the resulting poses list in console.
    :type print_poses: bool
    :param preparation_fun: a function to run on the cost function args before the cost function.
    :type preparation_fun: function
    :param visualize_scans: a boolean
    :type visualize_scans: bool
    :return: (weights, voxel counts, occlusions)
    :rtype: tuple
    """
    print("Starting optimization now...")
    # d[(angle,row,col)] = array that is [0,1] for each voxel that is touched
    # Minimize the cost function

    w = np.ones(acquisition_params.nr_rows * acquisition_params.nr_cols * shape(acquisition_params.angles)[0])
    w = w / len(w)

    if preparation_fun is not None:
        cost_fun_args = preparation_fun(d, *cost_fun_args)

    # w, d, updated_d, gauss, seen_voxels, dims, angles, transducer_width, focal_dist)

    print("Starting minimization...")
    if optimization_framework == 'scipy':
        w = minimize(cost_fun, w, args=(d, *cost_fun_args), method=method)["x"]
    elif optimization_framework == 'torch':
        # todo: implement this
        pass

    if post_processing_function is not None:
        w = post_processing_function(w, len(acquisition_params.angles), acquisition_params.nr_cols,
                                     acquisition_params.nr_rows)

    print("w: ", np.argsort(w))

    print("Done minimizing, starting map...")
    # Get a map of pose tuple (angle,row,col) to weight for that pose
    poses = map_weights_to_poses(w, acquisition_params.dims, acquisition_params.angles)

    # Reverse order by weight
    poses = dict(sorted(poses.items(), key=lambda item: item[1], reverse=True))

    # Remove poses that are not touching any occlusions
    if measured_arrs is not None and len(measured_arrs) > 2:
        poses_list = prepare_poses_list(list(poses.keys()), measured_arrs[2], old_poses, acquisition_params.dims)
    else:
        poses_list = list(poses.keys())

    print("Done calculating prioritization, starting measurement...")
    # Plot the result
    return measure_optimization_results(poses_list, acquisition_params.nr_rows, acquisition_params.nr_cols,
                                        acquisition_params.nr_planes, focal_dist=acquisition_params.focal_dist,
                                        plot=False,
                                        cutoff_function=cutoff_fun,
                                        transducer_width=acquisition_params.transducer_width,
                                        old_measurements=measured_arrs, block_plot=block_plot, print_poses=False)


def volume_coverage_optimizer(acquisition_params,
                              preparation_fun=coverage_loss_preparation,
                              visualize_scans: bool = False,
                              post_processing_function=post_process_weights,
                              show_occluded_voxels_after_simulation: bool = False,
                              optimization_method=None,
                              optimization_function=coverage_loss,
                              optimization_framework='scipy') -> (
        (list, list, {(float, float, float): [(int, int, int)]}, str)):
    """
    Optimize for volume coverage.
    :param show_occluded_voxels_after_simulation:
    :type show_occluded_voxels_after_simulation:
    :param params: The metadata for the optimizer.
    :type params: tuple
    :return: the tuple (weights, voxel_counts, occlusions)
    :rtype: (list, list, {(float, float, float): [(int, int, int)]})
    """

    params = acquisition_params.get_params()
    measured_arrs, occlusions, poses = optimize(
        d=get_weights_from_file(get_std_name("weights", "pkl", acquisition_params.dims, acquisition_params.focal_dist,
                                             acquisition_params.angles), params),
        cost_fun=optimization_function,
        cost_fun_args=(acquisition_params.dims, acquisition_params.angles),
        measured_arrs=None,
        cutoff_fun=all_voxels_under_thresh,
        acquisition_params=acquisition_params,
        preparation_fun=preparation_fun,
        post_processing_function=post_processing_function,
        method=optimization_method,
        optimization_framework=optimization_framework)

    weights, voxel_counts, occlusions = measured_arrs

    print("finished running optimizer, now running a simulation for evaluation for poses ", poses)

    # Run an evaluation simulation, if there are any poses
    if len(poses) > 0:
        occlusions, voxel_counts, vc_path = get_occlusions_from_simulation(acquisition_params.dims, poses,
                                                                           acquisition_params.model_name,
                                                                           acquisition_params.material_type,
                                                                           acquisition_params.transducer_width,
                                                                           acquisition_params.angles,
                                                                           occ_threshold=acquisition_params.occ_threshold,
                                                                           iteration_type=IterationType.VOLUMECOVERAGE,
                                                                           plot_simulation=visualize_scans,
                                                                           init=True)

        if show_occluded_voxels_after_simulation:
            plot_occluded_voxels(occlusions, acquisition_params.dims)  # Plot the measured occlusions
        print("After running evaluation simulation for volume coverage, occlusions are: ", occlusions)
    return (weights, voxel_counts, occlusions), poses, vc_path


def occlusion_prevention_optimizer(measured_arrs,
                                   acquisition_params,
                                   preparation_fun=occlusion_loss_preparation,
                                   old_poses=None,
                                   visualize_scans: bool = False,
                                   show_occluded_voxels_after_simulation: bool = False,
                                   post_processing_function=post_process_weights,
                                   old_voxel_counts_filepath: str = None,
                                   optimization_function=occlusion_loss,
                                   optimization_method=None,
                                   optimization_framework='scipy') -> (tuple, list, str):
    """
    This optimization iteration optimizeses for occlusion prevention.
    Namely, there has to be a previous measurement which gives us the occluding poses and voxels to avoid.
    :param measured_arrs: (weights, voxel_counts, occlusions)
    :type measured_arrs: tuple
    :param params: The metadata for the optimizer.
    :type params: tuple
    :return:
    :rtype:
    """
    if old_poses is None:
        old_poses = []
    params = acquisition_params.get_params()
    occlusions = measured_arrs[2]

    if len(occlusions) > 0:
        print("Recalculating optimization...")
        # Recalculate the bivariate gauss.
        gauss = bivariate_gauss(acquisition_params.nr_rows, acquisition_params.nr_cols, occlusions.keys())

        # Re-optimize the poses.
        measured_arrs, occlusions, poses = optimize(
            d=get_weights_from_file(get_std_name("inv_distances", "pkl", acquisition_params.dims,
                                                 acquisition_params.focal_dist, acquisition_params.angles), params),
            cost_fun=optimization_function,
            cost_fun_args=(gauss, occlusions, acquisition_params.dims, acquisition_params.angles,
                           acquisition_params.transducer_width, acquisition_params.focal_dist),
            measured_arrs=measured_arrs,
            cutoff_fun=occluded_cutoff,
            old_poses=old_poses,
            preparation_fun=preparation_fun,
            post_processing_function=post_processing_function,
            method=optimization_method,
            acquisition_params=acquisition_params,
            optimization_framework=optimization_framework)

        weights, voxel_counts, occlusions = measured_arrs

        print("finished running optimizer, now running a simulation for evaluation for poses ", poses)

        # Run an evaluation simulation, if there are any poses
        if len(poses) > 0:
            vc = None
            if old_voxel_counts_filepath is not None:
                vc = pickle.load(open(old_voxel_counts_filepath, "rb"))
            occlusions, voxel_counts, vc_path = get_occlusions_from_simulation(acquisition_params.dims, poses,
                                                                               acquisition_params.model_name,
                                                                               acquisition_params.material_type,
                                                                               acquisition_params.transducer_width,
                                                                               acquisition_params.angles,
                                                                               iteration_type=IterationType.OCCLUSIONPREVENTION,
                                                                               old_voxel_counts=vc,
                                                                               occ_threshold=acquisition_params.occ_threshold,
                                                                               plot_simulation=visualize_scans,
                                                                               init=False)

            if show_occluded_voxels_after_simulation:
                plot_occluded_voxels(occlusions, acquisition_params.dims)  # Plot the measured occlusions
            print("After running evaluation simulation for occlusion prevention, occlusions are: ", occlusions)
        return (weights, voxel_counts, occlusions), poses, vc_path
    return None, []


def perpendicular_occlusion_prevention_optimizer(acquisition_params,
                                                 occlusion_optimizer_options: OptimizersOptions,
                                                 visualize_scans: bool = False,
                                                 show_occluded_voxels_after_simulation: bool = True):
    """First perpendicular, then occlusion prevention."""
    # Run a perpendicular scan
    measured_arrs, perpendicular_poses, vc_path = perpendicular_scan(
        acquisition_params=acquisition_params,
        show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
        visualize_scans=visualize_scans,
        synthetic=False)

    occlusion_prevention_optimizer(measured_arrs=measured_arrs,
                                   acquisition_params=acquisition_params,
                                   old_poses=perpendicular_poses,
                                   old_voxel_counts_filepath=vc_path,
                                   visualize_scans=visualize_scans,
                                   show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
                                   preparation_fun=occlusion_optimizer_options.pre_processing_function,
                                   optimization_function=occlusion_optimizer_options.optimization_function,
                                   post_processing_function=occlusion_optimizer_options.post_processing_function,
                                   optimization_method=occlusion_optimizer_options.method)


def random_occlusion_prevention_optimizer(acquisition_params,
                                          occlusion_optimizer_options: OptimizersOptions,
                                          visualize_scans: bool = False,
                                          show_occluded_voxels_after_simulation: bool = True):
    """First random, then occlusion prevention."""
    # Run a random scan
    measurements, random_voxel_counts, occlusions, perpendicular_poses, vc_path = random_scan(
        acquisition_params=acquisition_params,
        show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
        visualize_scans=False,
        synthetic=False)

    occlusion_prevention_optimizer(measured_arrs=(measurements, random_voxel_counts, occlusions),
                                   acquisition_params=acquisition_params,
                                   old_poses=perpendicular_poses,
                                   old_voxel_counts_filepath=vc_path,
                                   preparation_fun=occlusion_optimizer_options.pre_processing_function,
                                   optimization_function=occlusion_optimizer_options.optimization_function,
                                   post_processing_function=occlusion_optimizer_options.post_processing_function,
                                   visualize_scans=visualize_scans,
                                   show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
                                   optimization_method=occlusion_optimizer_options.method)


def volume_coverage_occlusion_prevention_optimizer(acquisition_params: AcquisitionGeometryParams,
                                                   volume_coverage_optimizer_options: OptimizersOptions,
                                                   occlusion_optimizer_options: OptimizersOptions,
                                                   show_occluded_voxels_after_simulation: bool = False,
                                                   synthetic: bool = False, visualize_scans: bool = False):
    """First VC, then OP."""

    measured_arrs, poses, vc_path = volume_coverage_optimizer(acquisition_params=acquisition_params,
                                                              preparation_fun=volume_coverage_optimizer_options.pre_processing_function,
                                                              post_processing_function=volume_coverage_optimizer_options.post_processing_function,
                                                              optimization_function=volume_coverage_optimizer_options.optimization_function,
                                                              optimization_method=volume_coverage_optimizer_options.method,
                                                              visualize_scans=visualize_scans,
                                                              show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
                                                              optimization_framework=volume_coverage_optimizer_options.framework)

    # Run an evaluation simulation, if there are any poses
    if not synthetic and len(poses) > 0:
        occlusion_prevention_optimizer(measured_arrs=measured_arrs,
                                       acquisition_params=acquisition_params,
                                       old_poses=poses,
                                       visualize_scans=visualize_scans,
                                       show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
                                       preparation_fun=occlusion_optimizer_options.pre_processing_function,
                                       post_processing_function=occlusion_optimizer_options.post_processing_function,
                                       optimization_function=occlusion_optimizer_options.optimization_function,
                                       optimization_method=occlusion_optimizer_options.method,
                                       optimization_framework=occlusion_optimizer_options.framework)
