import json
import os
import pickle
import sys
from enum import Enum

import numpy as np

from functions.helpers.helpers import clean_occlusions, sort_transducer_poses, prepare_results_folders, IterationType, \
    get_iter_prefix_filepath
from functions.helpers.imfusion.imfusion_helpers import get_algorithm_properties_from_iws, \
    occlusions_from_confidence, get_confidence_map_from_simulation, save_data, cleanup_and_save_occlusions
from functions.helpers.read_from_file import write_data_to_file

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def get_occlusions_from_simulation(dims: (int, int, int), poses: [(float, int, int)], model_name: str, material_type: str,
                                   transducer_width: float, angles: [float], occ_threshold: float, iteration_type: IterationType,
                                   old_voxel_counts: list = None, plot_simulation: bool = False,
                                   init: bool = False) -> (dict, list):
    """
    Run a hybrid ultrasound simulation and calculate the confidence map. Turn confidence map into occlusions map.
    :param iteration_type: Whether it's perpendicular, random, or one of the two modules.
    :type iteration_type: IterationType
    :param angles: the list of possible pose angles
    :type angles: list(float)
    :param old_voxel_counts: the voxel count of the previous iteration, if existant.
    :type old_voxel_counts: list
    :param poses: the list of optimized poses (angle, x,y,)
    :type poses: [(float,int,int)]
    :param dims: (nr rows, nr cols, nr planes)
    :type dims: (int,int,int)
    :param transducer_width:
    :type transducer_width:
    :param model_name: the file name of the 3d model we want to evaluate on.
    :type model_name: str
    :param init: whether this is the first time we initialize imfusion
    :type init: bool
    :return: the occlusions map {pose: [occluded_voxels]}, and the voxel_counts
    :rtype:dict
    """

    # Sort the transducer poses
    poses = sort_transducer_poses(poses)  # Sort the trajectory for robot usage

    # Append material type to model name
    model_name = model_name+"_"+material_type

    # Get the iteration filepath for saving this iteration
    iter_filepath_prefix = get_iter_prefix_filepath(model_name, iteration_type)

    # Check if folders for saving the weights exists, if not, create them
    prepare_results_folders(model_name, iter_filepath_prefix)

    # Save poses for iteration
    with open(iter_filepath_prefix + "/poses.json", "w") as f:
        json.dump(poses, f)

    # Run simulation and get the confidence map.
    confidence_map, data = get_confidence_map_from_simulation(model_name, material_type, dims, poses, transducer_width,
                                                              iter_filepath_prefix,
                                                              init=init)

    # Calculate occlusions from confidence map and return.
    occlusions, voxel_counts = occlusions_from_confidence(model_name, confidence_map=confidence_map,
                                                          dims=dims,
                                                          data=data,
                                                          transducer_width=transducer_width,
                                                          occ_threshold=occ_threshold,
                                                          iter_filepath_prefix=iter_filepath_prefix,
                                                          angles=angles, poses=poses, plot_simulation=plot_simulation)

    # Clean up occlusions that have been seen in previous runs or in this run.
    occlusions = cleanup_and_save_occlusions(occlusions, old_voxel_counts, voxel_counts, dims, iter_filepath_prefix)


    # Save voxel_counts
    vc_path = iter_filepath_prefix+"/voxel_counts.pkl"
    write_data_to_file(vc_path, voxel_counts)

    return occlusions, voxel_counts, vc_path
