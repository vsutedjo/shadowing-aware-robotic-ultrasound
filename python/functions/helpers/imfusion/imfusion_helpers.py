import json
import math
import os
import xml.etree.ElementTree as ET
from shutil import copy

import numpy as np
from matplotlib import pyplot as plt, gridspec

from functions.helpers.helpers import splines_from_poses, get_probe_width, prepare_results_folders, \
    get_imfusion_savepath_name, get_all_intersections, iteration_name_dict, IterationType, clean_occlusions
from functions.helpers.imfusion.ImfusionTrackedImage import ImfusionTrackedImage
from functions.helpers.imfusion.spatial_transformation_helpers import get_pixels_inside_voxel
from functions.helpers.plot_functions import get_intersected_voxels, plot_intersected_voxels_with_occlusions
from functions.helpers.read_from_file import write_data_to_file

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def stl_to_labelmap(model_name: str, material_type: str):
    """
    This function converts a stl 3d model into the kind of label map we need for US simulations.
    The output file is saved in the 3dModels folder as .imf
    :param model_name: the name of the stl file we want to convert.
    :type model_name:str
    :return: void
    :rtype:
    """
    imfusion.init()
    data = imfusion.open("imfusion_files/3dModels/" + model_name + ".stl")
    labelmap_param = \
        get_algorithm_properties_from_iws("imfusion_files/ws_" + material_type + ".iws", 'Convert To Label Map')
    labelmap = imfusion.executeAlgorithm('Convert To Label Map', data, labelmap_param)

    imfusion.executeAlgorithm('IO;ImFusionFile', labelmap,
                              imfusion.Properties({'location': "imfusion_files/3dModels/" + model_name + ".imf"}))


def cast_param(text):
    if text is None:
        return ""

    text_list = text.split(" ")

    if len(text_list) == 1:
        try:
            casted_value = float(text)
            return casted_value

        except:
            return text

    else:
        try:
            converted_list = [float(item) for item in text_list]
            return np.array(converted_list)
        except:
            return text


def get_block(parent_block, block_name):
    for child in parent_block:
        if child.attrib["name"] == block_name:
            return child


def parse_alg_to_dict(param_dict, block, prefix=""):
    for item in block:
        if item.tag == 'param':
            param_dict[prefix + item.attrib["name"]] = cast_param(item.text)
        elif item.tag == 'property':
            sub_block = get_block(block, item.attrib["name"])
            parse_alg_to_dict(param_dict, sub_block, prefix + item.attrib["name"] + "/")

    return param_dict


def read_roi_from_json(model_name: str):
    with open("imfusion_files/3dModels/" + model_name + "_ROI.json") as f:
        data = json.load(f)
        min_x_loc = data["min_x_loc"]
        max_x_loc = data["max_x_loc"]
        min_y_loc = data["min_y_loc"]
        max_y_loc = data["max_y_loc"]
        max_z_loc = data["max_z_loc"]
        min_z_loc = data["min_z_loc"]
        return min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc


def write_poses_output(poses: list, ws_filepath: str, dims: (int, int), transducer_width: float, model_name: str):
    roi = read_roi_from_json(model_name)
    location, direction = splines_from_poses(poses, dims, roi)
    tree = ET.parse(ws_filepath)
    root = tree.getroot()

    # Write the stl file name to workspace so we can also try out in imfusion
    for x in root[3].findall("*/[@name='Mesh File']/param[@name='location']"):
        x.text = str("3dModels/" + model_name + ".stl")

    # Write the frame count
    for x in root[3].findall("*/[@name='Hybrid Ultrasound Simulation']/param[@name='frameCount']"):
        x.text = str(5)  # str(dims[0] * dims[1] * 5)  # On average 5 frames for each transducer location "voxel"

    # Write probe width
    for x in root[3].findall("*/[@name='Hybrid Ultrasound Simulation']/param[@name='probeWidth']"):
        x.text = str(get_probe_width(dims[0],
                                     transducer_width, roi))

    # Write location and direction points
    for x in root[3].find("*/property/.[@name='SplineTransducer']"):
        if x.attrib["name"] == "points":
            x.text = location
            break
    for x in root[3].find("*/property/.[@name='SplineDirection']"):
        if x.attrib["name"] == "points":
            x.text = direction
            break
    for child in root.findall("property/.[@name='Annotations']/"):
        if child.attrib["name"] == "GlSpline":
            is_td_spline = False
            for x in child:
                if x.attrib["name"] == "name":
                    is_td_spline = x.text == "Transducer Spline"
                if x.attrib["name"] == "points":
                    if is_td_spline:
                        x.text = location
                    else:
                        x.text = direction

    tree.write(ws_filepath)


def get_algorithm_properties_from_iws(iws_filepath, algorithm_name):
    """
    This function parse a iws file and saves all the parameters referred to the <algorithm_name> algorithm in a dict,
    that can be then used to create the algorithm properties.
    Every field of the dict can be overwritten with custom values. E.g. if one wants to overwrite Transducer and
    Direction spline in the dictionary obtained by passing algorithm_name='Hybrid Ultrasound Simulation', they can simply
    do:

    default_hybrid_simulation_params = get_algorithm_properties_from_iws(iws_filepath)
    default_hybrid_simulation_params[SplineDirection/points] = np.ndarray([...])
    default_hybrid_simulation_params[SplineTransducer/points] = np.ndarray([...])

    :param iws_filepath: The path to the iws file. It MUST contain the <algorithm_name> property!!
    :param algorithm_name: The algorithm for which you want to extract the parameters
    :return: A dictionary containing the <algorithm_name> parameters
    """

    tree = ET.parse(iws_filepath)
    root = tree.getroot()

    algorithms = get_block(root, "Algorithms")
    us_sim = get_block(algorithms, algorithm_name)

    p_dict = dict()
    parse_alg_to_dict(p_dict, us_sim)

    imfusion_properties = imfusion.Properties(p_dict)

    return imfusion_properties


def get_confidence_map_from_simulation(model_name: str, material_type: str, dims: (int, int, int),
                                       poses: [(float, int, int)],
                                       transducer_width: float, iter_filepath_prefix: str, init: bool = False) -> (
        dict, imfusion.SharedImage):
    """
    This function will run a US simulation, calculate the confidence map and save both the sweep and
    the confidence map in imfusion_files. Before calling this function, make sure the workspace file
    is loaded with your params, e.g. you have run write_poses_output() before to update the points.
    :param iter_filepath_prefix:
    :type iter_filepath_prefix:
    :param dims:
    :type dims:
    :param transducer_width:
    :type transducer_width:
    :param poses:
    :type poses:
    :param model_name: the name of the 3d model you want to run the simulation with
    :type model_name: str
    :param init: whether to initialize imfusion
    :type init: bool
    :return: confidence map dict and coordinate system of confidence map?
    :rtype:dict
    """
    if init:
        imfusion.init()

    model_path = "imfusion_files/3dModels/" + model_name + ".imf"

    # Check if label map exists and if not, create it
    if not os.path.isfile(model_path):
        print("imf file does not exist yet, creating labelmap from stl...")
        stl_to_labelmap(model_name, material_type)

    # Load the model volume file into imfusion.
    # This is typically the label map with 13 (bone) for inside and 4 (water) for outside.
    # If you don't have the imf file yet, use stl_to_labelmap(filename) to create it.
    data = imfusion.open(model_path)

    # Return immediately if no poses are given.
    if len(poses) < 1:
        return {}

    # For each pose, calculate the corresponding confidence map.
    confidence_maps = {}
    for i, pose in enumerate(poses):
        print("________________\nlooking at pose ", str(pose), "with index ", i)
        confidence_maps[pose] = run_us_sweep_for_pose(pose, data, dims, transducer_width, model_name, material_type,
                                                      iter_filepath_prefix, i)

    return confidence_maps, data[0]

def get_confidence_decay():

    if not os.path.exists("imfusion_files/target_cm_soft_tissues.imf"):
        print("Target cm for intensity decay correction not found - using decay = 1")
        return 1

    cm = imfusion.open(
        "imfusion_files/target_cm_soft_tissues.imf")[0][0]
    cm_array = np.squeeze(np.array(cm))

    decay = np.mean(cm_array, axis=1)
    decay[decay < 0.1] = 0.1

    plt.plot(1/decay)
    plt.show()

    return 1/decay

def run_us_sweep_for_pose(pose: (float, int, int), data, dims: (int, int, int), transducer_width: float,
                          model_name: str, material_type: str, iter_filepath_prefix: str, idx=0):
    nr_rows, nr_cols, nr_planes = dims

    original_pose = pose

    sweep_path = "imfusion_files/results/" + model_name + "/general/sweeps/" + get_imfusion_savepath_name(
        "simulated_sweep", original_pose, dims) + ".imf"

    tracking_path = "imfusion_files/results/" + model_name + "/general/tracking/" + get_imfusion_savepath_name(
        "tracking", original_pose, dims) + ".ts"

    cm_path = "imfusion_files/results/" + model_name + "/general/confidence_maps/" + get_imfusion_savepath_name(
        "confidence_map", original_pose, dims) + ".imf"

    sweep_path_iter = iter_filepath_prefix + "/sweeps/" + get_imfusion_savepath_name(
        "simulated_sweep", original_pose, dims) + ".imf"

    tracking_path_iter = iter_filepath_prefix + "/tracking/" + get_imfusion_savepath_name(
        "tracking", original_pose, dims) + ".ts"

    cm_path_iter = iter_filepath_prefix + "/confidence_maps/" + get_imfusion_savepath_name(
        "confidence_map", original_pose, dims) + ".imf"

    ws_filepath = "imfusion_files/ws_" + material_type + ".iws"

    # Find out if we need to run the sweep
    if not os.path.isfile(sweep_path):
        print("Calculating simulated sweep...")
        poses = []

        # Add a small area around the sweep
        for i in range(4):
            poses.append((original_pose[0], original_pose[1], original_pose[2] + i * 0.01))
        write_poses_output(poses, ws_filepath, (nr_rows, nr_cols), transducer_width, model_name)

        # Run a hybrid US simulation on the volume.
        hybrid_us_param = \
            get_algorithm_properties_from_iws(ws_filepath, 'Hybrid Ultrasound Simulation')
        simulated_sweep = imfusion.executeAlgorithm('Ultrasound;Hybrid Ultrasound Simulation', data, hybrid_us_param)

        # Save the sweep.
        imfusion.executeAlgorithm('IO;ImFusionFile', simulated_sweep,
                                  imfusion.Properties({'location': sweep_path}))
    else:
        print("Sweep ", str(pose), "exists, skipping simulation...")

    # Copy data to iter folder
    if not os.path.isfile(sweep_path_iter):
        copy(sweep_path, sweep_path_iter)

        # Find out if we need to run the tracking path extraction
    if not os.path.isfile(tracking_path):
        print("Extracting tracking path...")

        # Open simulated sweep
        simulated_sweep = imfusion.open(sweep_path)

        # Save the tracking stream.
        tracking_stream = imfusion.executeAlgorithm('Tracking;Extract Tracking Stream', simulated_sweep,
                                                    imfusion.Properties({}))
        imfusion.executeAlgorithm('IO;Tracking Stream', tracking_stream,
                                  imfusion.Properties(
                                      {
                                          'location': tracking_path}))
    else:
        print("Tracking path for ", str(pose), "exists, skipping simulation...")

    # Copy data to iter folder
    if not os.path.isfile(tracking_path_iter):
        copy(tracking_path, tracking_path_iter)

    # Find out if we need to run the confidence map
    if not os.path.isfile(cm_path):
        print("Calculating the confidence map...")
        # Open simulated sweep
        simulated_sweep = imfusion.open(sweep_path)

        # Inverting sweep as confidence maps are computed bottom to top
        flipped_sweeps = imfusion.executeAlgorithm('Basic Processing', simulated_sweep,
                                                   imfusion.Properties(
                                                       {'mode': 2,
                                                        'flip': 1}))

        # Run the confidence map calculation and save the map.
        confidence_maps_params = \
            get_algorithm_properties_from_iws(ws_filepath, 'Compute Confidence Maps')
        confidence_maps = imfusion.executeAlgorithm('Ultrasound;Compute Confidence Maps', flipped_sweeps,
                                                    confidence_maps_params)

        # Sweep confidence maps back to recover correct orientation
        imageset = imfusion.executeAlgorithm('Basic Processing', confidence_maps,
                                                    imfusion.Properties(
                                                        {'mode': 2,
                                                         'flip': 1}))[0]

        image = imageset[0]
        confidence_array = np.array(image, copy=False)

        if "none" in material_type:

            decay = get_confidence_decay()
            confidence_array[...] = np.multiply(confidence_array, np.expand_dims(decay, axis=(0, -1, -2)))

        image.setDirtyMem()

        out_data = imfusion.SharedImageSet()
        out_data.add(image)

        imfusion.executeAlgorithm('IO;ImFusionFile', [out_data],
                                  imfusion.Properties(
                                      {'location': cm_path}))

    else:
        print("Confidence Map for ", str(pose), "exists, skipping simulation...")

    # Copy data to iter folder
    if not os.path.isfile(cm_path_iter):
        copy(cm_path, cm_path_iter)

    # Open the sweeps/maps
    confidence_maps = imfusion.open(cm_path)
    simulated_sweep = imfusion.open(sweep_path)
    print("Opened the confidence maps and sweeps.")

    if len(confidence_maps[0]) == 1:
        first_confidence_map = confidence_maps[0]
    else:
        first_confidence_map = imfusion.executeAlgorithm('Split Images', [confidence_maps[0]])[0]
        # first_confidence_map = first_confidence_map[0]

    confidence_map = ImfusionTrackedImage(imfusion_shared_image_set=first_confidence_map,
                                          matrix=simulated_sweep[0].matrix(0))  # matrix in imfusion coordinates
    return confidence_map


def occlusions_from_confidence(model_name: str,
                               confidence_map: {((float, float, float), int, int): ImfusionTrackedImage},
                               dims: (int, int, int),
                               data,
                               transducer_width: float,
                               occ_threshold: float,
                               iter_filepath_prefix: str,
                               angles: [float], poses: list, plot_simulation: bool = False, ) -> (dict, list):
    """

    :param poses:
    :type poses:
    :param plot_simulation:
    :type plot_simulation:
    :param model_name:
    :type model_name:
    :param confidence_map:
    :param dims: (nr_rows, nr_cols, nr_planes) - this are the dimensions along python axes
    :param data:
    :param transducer_width:
    :param angles:
    :return:
    """
    """ Calculate the new occlusion dict from the imfusion parsed confidence map.
    Returns the occlusions and the voxel_counts"""

    print("Calculating occlusions from confidences...")
    occluding_poses_map = {}
    voxel_counts = np.zeros(dims)

    # 1. Reshape the imfusion volume to match the considered volume grid
    # properties = imfusion.Properties({'resamplingMode': 0,
    #                                   'targetDimensions': dims})
    # data = imfusion.executeAlgorithm('Image Resampling', [data], properties)[0]

    # using roi selection
    min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc = read_roi_from_json(model_name)

    rows, cols, planes = dims
    spacing = [(max_x_loc - min_x_loc) / rows, (max_y_loc - min_y_loc) / cols, (max_z_loc - min_z_loc) / planes]
    T_world2vol = np.eye(4, 4)
    T_world2vol[0:3, -1] = -np.array([min_x_loc + (max_x_loc - min_x_loc) / 2,
                                      min_y_loc + (max_y_loc - min_y_loc) / 2,
                                      min_z_loc + (max_z_loc - min_z_loc) / 2])

    arr = np.random.randint(1, 10, (dims[2], dims[1], dims[0])).astype(np.uint8)
    arr[0, 0, 0] = 0
    shared_image = imfusion.SharedImage(np.expand_dims(arr, axis=-1))
    shared_image.matrix = T_world2vol
    shared_image.spacing = spacing
    data = imfusion.SharedImageSet()
    data.add(shared_image)
    imfusion.executeAlgorithm('IO;ImFusionFile', [data],
                              imfusion.Properties({'location': "imfusion_files/voxelized_volume.imf"}))

    # # 3. Getting the volume info
    T_vol2world = np.linalg.inv(data[0].matrix)  # data.matrix = world to data coordinates
    vol_spacing = data[0].spacing  # spacing in imfusion axes
    vol_shape = [data[0].width, data[0].height, data[0].slices]  # shape in imfusion axes

    confidence_means_map = {}  # The general confidence means
    confidence_means_iter = {}  # The confidence means only for this iteration

    # Use the pre-computed confidence means if they exist
    rows, cols, planes = dims
    general_means_path = "imfusion_files/results/" + model_name + "/general/confidence_means/dims" + str(rows) + str(
        cols) + str(
        planes) + "_confidence_means.json"
    means_path_iter = iter_filepath_prefix + "/confidence_means/dims" + str(rows) + str(cols) + str(
        planes) + "_confidence_means.json"

    if os.path.isfile(general_means_path):
        confidence_means_map = get_confidence_means_from_file(general_means_path)

    for pose in confidence_map.keys():

        confidence_means = {}

        if str(pose) in confidence_means_map:
            print("Confidence means for pose ", str(pose), " are pre-computed.")
            confidence_means = confidence_means_map[str(pose)]

        occluded_voxels = []
        confidence = confidence_map[pose]

        # Getting ultrasound info in imfusion axes
        T_us2world = confidence.matrix
        confidence_shape = confidence.get_shape()
        confidence_spacing = confidence.get_spacing()

        confidence_array = confidence.get_python_array()

        intersected_voxels = get_intersected_voxels(dims, pose)  # , plot=True)
        # print("\npose: ", str(pose), ":")
        # print(intersected_voxels)

        for vox in intersected_voxels:
            x, y, z = vox
            if str((x, y, z)) in confidence_means:
                confidence_mean = confidence_means[str((x, y, z))]
                # print("confidence mean exists for voxel ", str((x, y, z)), ": ", round(confidence_mean, 5))

            else:
                # convert the values to ints so we can use them as approximating indices in the voxel grid
                proximal_pixels = get_pixels_inside_voxel(P_voxel=[x, y, z],
                                                          volume_spacing=vol_spacing,
                                                          volume_size_in_voxel=vol_shape,
                                                          T_vol2world=T_vol2world,
                                                          image_spacing=confidence_spacing,
                                                          image_size_in_pixels=confidence_shape,
                                                          T_img2world=T_us2world)

                if len(proximal_pixels) != 0:
                    confidence_values = [confidence_array[item[1], item[0]] for item in proximal_pixels]
                    confidence_mean = np.mean(confidence_values)
                    confidence_means[str((x, y, z))] = float(confidence_mean)
                else:
                    # confidence_means[str((x, y, z))] = 0.0
                    # confidence_mean = 0
                    continue  # todo: change this!

            # todo: change this!
            is_occluded = confidence_mean <= occ_threshold

            if is_occluded:
                occluded_voxels.append([x, y, z])
            else:
                voxel_counts[x][y][z] = 1

        confidence_means_map[str(pose)] = confidence_means
        # If this pose has occluded voxels, we save it as occluding pose.
        if len(occluded_voxels) > 0:
            occluding_poses_map[pose] = occluded_voxels

        if plot_simulation:
            plot_intersected_voxels_with_occlusions(intersected_voxels, occluded_voxels, dims, pose)

    # Overwrite the confidence means with the new confidence means.
    save_data(general_means_path, confidence_means_map)

    # Save only confidence means for this iteration that were part of this iteration
    poses = [str(p) for p in poses]
    for pose in poses:
        confidence_means_iter[pose] = confidence_means_map[pose]
    save_data(means_path_iter, confidence_means_iter)

    # saving voxelized volume for visualiztion
    vol_array = np.array(data[0], copy=False)
    data.setDirtyMem()
    i = 0
    for x in range(vol_shape[0]):
        for y in range(vol_shape[1]):
            for z in range(vol_shape[2]):
                vol_array[z, y, x, :] = i
                i += 1

    return occluding_poses_map, voxel_counts


def cleanup_and_save_occlusions(occlusions, old_voxel_counts, voxel_counts, dims, iter_filepath_prefix):
    occlusions = clean_occlusions(occlusions, voxel_counts)

    # Cleanup
    if old_voxel_counts is not None:
        occlusions = clean_occlusions(occlusions, old_voxel_counts)

    occluded_voxels = np.zeros_like(voxel_counts)
    for occluded_voxs in occlusions.values():
        for vox in occluded_voxs:
            x, y, z = vox
            occluded_voxels[x][y][z] = 1

    occluded_voxels = (occluded_voxels >= 1)
    # Save the occluded voxels
    rows, cols, planes = dims
    save_data(iter_filepath_prefix + "/dims" + str(rows) + str(cols) + str(
        planes) + "_occluded_voxels.json", occluded_voxels.tolist())

    return occlusions


def save_data(fname: str, data):
    """
    Save the confidence mean values in a json
    :param fname:
    :type fname:
    :param data: a dict of poses to dicts of voxels and their confidence means.
    :type data: {(float,int,int):{(int,int,int):float}}
    :return: void
    :rtype:
    """
    print("Writing data to json file ", fname, "...")
    if os.path.isfile(fname):
        with open(fname, "w") as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)
    else:
        with open(fname, "a") as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)
    print("Done writing.")


def get_confidence_means_from_file(file_path: str):
    # print("Opening confidence means from file.")
    with open(file_path) as infile:
        confidence_means = json.load(infile)
        return confidence_means
