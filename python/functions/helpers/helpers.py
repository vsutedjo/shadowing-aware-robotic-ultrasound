import math
import os
from datetime import datetime
from enum import Enum

import numpy as np
from PIL import Image
from numpy import shape


def print_trajectory_mat(matrix):
    for line in range(matrix.shape[0]):

        formatted_string = ""
        for item in range(matrix.shape[1]):
            formatted_string = formatted_string + "{:.3f} ".format(round(matrix[line, item], 2))
        print(formatted_string)

    print("\n")

def clean_occlusions(occlusions: {tuple: list}, voxel_counts: list) -> {(float, float, float): [(int, int, int)]}:
    """ We only want to keep occlusions that are truly occluded.
    If the voxel has been covered by another non-occluding pose, we don't keep it"""
    new_occs = {}
    for pos in occlusions.keys():
        voxels = occlusions[pos]

        # Remove poses that have empty voxel lists.
        if len(voxels) < 1:
            continue

        # Only add poses that have true occluded voxels.
        new_voxels = []
        for vox in voxels:
            x, y, z = vox
            if voxel_counts[x][y][z] <= 0 and vox not in new_voxels:
                new_voxels.append(vox)
        if len(new_voxels) > 0:
            new_occs[pos] = new_voxels
    return new_occs


def poses_distance(pos1: (float, float, float), pos2: (float, float, float)) -> float:
    """ Calculates the distance between two poses, normally between an occluding pose and another pose."""
    angle1, x1, y1 = pos1
    angle2, x2, y2 = pos2
    # dist = (abs(angle1 - angle2) / 45) * np.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2)
    # dist /= 2
    dist = (abs(angle1 - angle2) / 45)
    return dist


def arr_to_str(arr: list) -> str:
    res = ""
    for elem in arr:
        res += str(elem) + "\n"
    return res


def sort_alternating_cols(forwards: bool, pose: (float, int, int)):
    if forwards:
        return pose[2]
    else:
        return -pose[2]


def sort_transducer_poses(poses: [float, int, int]) -> [float, int, int]:
    sorted_map = {}
    for pose in poses:
        if pose[1] not in sorted_map:
            sorted_map[pose[1]] = []
        sorted_map[pose[1]].append(pose)

    for row_key in sorted_map.keys():
        col_list = sorted_map[row_key]
        col_list = sorted(sorted(col_list, key=lambda p: p[0]),
                          key=lambda p: sort_alternating_cols(row_key % 2 == 0, p))  # Sort poses (col/ang)
        sorted_map[row_key] = col_list

    sorted_list = []
    for row_key in sorted(sorted_map.keys()):
        sorted_list.extend(sorted_map[row_key])
    print("Sorted poses: ", sorted_list)
    return sorted_list


def get_probe_width(nr_rows: int, transducer_width: float, roi: tuple):
    min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc = roi
    volume_width = max_x_loc - min_x_loc
    voxel_width = volume_width / nr_rows
    return transducer_width * voxel_width


def confidence_map_resize(confidence_map: [[float]], dims: (int, int, int), transducer_width: float):
    nr_touched_voxels = math.ceil(transducer_width)
    nr_rows, nr_cols, nr_planes = dims
    img = Image.fromarray(np.array(confidence_map))
    i = img.resize((nr_touched_voxels, nr_planes))
    arr = np.asarray(i)
    return arr


def tracking_to_pose(tracking_point: (float, float, float, float, float, float), dims: (int, int, int), roi: tuple) -> (
        float, int, int):
    """
    For a single tracking point, calculate the corresponding voxel pose and angle
    :param roi: min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc
    :type roi: tuple
    :param tracking_point: (x,y,z, angle)
    :type tracking_point: (float,float,float,float)
    :param dims: (nr_rows,nr_cols, nr_planes)
    :type dims: (int,int,int)
    :return: (tilt_x, tilt_y, tilt_z, x,y)
    :rtype: tuple
    """
    min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc = roi

    # Read in the params
    x, y, z, angle_x, angle_y, angle_z = tracking_point
    nr_rows, nr_cols, nr_planes = dims

    # Shift the translation to voxel size
    delta_row = (max_x_loc - min_x_loc) / (nr_rows - 1)  # Calculate the voxel length
    delta_col = (max_y_loc - min_y_loc) / (nr_cols - 1)  # Calculate the voxel width
    x_shifted = int(round((x - min_x_loc) / delta_row, 0))  # Shift translation back to voxel grid
    y_shifted = int(round((y - min_y_loc) / delta_col, 0))
    return angle_x, angle_y, angle_z, x_shifted, y_shifted


def splines_from_poses(poses: list, dims: (int, int), roi: tuple) -> (str, str):
    min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc = roi
    rows, cols = dims
    delta_row = (max_x_loc - min_x_loc) / (rows)  # Calculate the voxel length
    delta_col = (max_y_loc - min_y_loc) / (cols)  # Calculate the voxel width
    delta_plane = max_z_loc - min_z_loc  # The entire height of the volume
    str_loc = ""
    str_dir = ""
    for p in poses:
        (angle, x, y) = p
        x_td = min_x_loc + x * delta_row + delta_row / 2
        y_td = min_y_loc + y * delta_col + delta_col / 2
        loc_pose = (x_td, y_td, max_z_loc)
        str_loc += str(loc_pose[0]) + " " + str(loc_pose[1]) + " " + str(loc_pose[2]) + " "

        # Calculate the direction spline
        slope = 0
        if angle != 0:
            slope = round(math.tan(math.radians(angle)), 2)
        x_dir = x_td
        y_dir = y_td + delta_plane * slope
        dir_pose = (x_dir, y_dir, max_z_loc - delta_plane)  # we are going downwards, so z must be subtracted
        str_dir += str(dir_pose[0]) + " " + str(dir_pose[1]) + " " + str(dir_pose[2]) + " "
    return str_loc, str_dir


def get_all_intersections(dims, origin_xy, tilt_angle, slope, y_spacing=1, z_spacing=1):
    rows, cols, planes = dims
    vertical_intersections = get_vertical_intersections(cols, planes, origin_xy[0], origin_xy, tilt_angle, slope,
                                                        y_spacing, z_spacing)
    horizontal_intersections = get_horizontal_intersections(planes, origin_xy[0], origin_xy, tilt_angle, slope,
                                                            y_spacing, z_spacing)
    v_ints = []
    h_ints = []
    for intersection in vertical_intersections:
        if intersection[0] <= rows and intersection[1] <= cols and intersection[2] <= planes:
            v_ints.append(intersection)
    for intersection in horizontal_intersections:
        if intersection[0] <= rows and intersection[1] <= cols and intersection[2] <= planes:
            h_ints.append(intersection)
    return v_ints, h_ints


def get_vertical_intersections(cols, planes, row, origin_xy, tilt_angle, slope, y_spacing=1, z_spacing=1):
    # If we go straight down we dont have side intersections

    if tilt_angle == 0:
        return []
    else:
        slope = 1 / round(math.tan(math.radians(tilt_angle)), 2)

    intersections = []
    for y in range(cols + 1):
        y_dist = (y - origin_xy[1]) * y_spacing
        z = planes * z_spacing - slope * y_dist
        if 0 <= z <= planes * z_spacing:
            intersection_point = [row, y, z / z_spacing]
            intersections.append(intersection_point)
            # ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2])
    return intersections


def get_horizontal_intersections(planes, row, origin_xy, tilt_angle, slope, y_spacing=1, z_spacing=1):
    intersections = np.zeros((planes + 1, 3))
    slope = round(math.tan(math.radians(tilt_angle)), 2)

    for z in range(planes + 1):
        z_dist = (planes - z) * z_spacing
        y = origin_xy[1] * y_spacing + z_dist * slope
        intersection_point = [row, y / y_spacing, z]
        intersections[z] = intersection_point
        # ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2])
    return intersections


def calculate_possible_poses(nr_rows: int, nr_cols: int, angles: [float]):
    """
    Calculate all possible poses.
    :param nr_rows:
    :type nr_rows:
    :param nr_cols:
    :type nr_cols:
    :param angles:
    :type angles:
    :return:
    :rtype:
    """
    poses = []
    for row in range(nr_rows):
        for col in range(nr_cols):
            for angle in angles:
                poses.append((angle, row, col))
    return poses


def calculate_occluding_poses(occluding_voxels: list, possible_poses: [(float, int, int)], pose_intersection_map):
    """ Calculate the poses:voxels dict for all occlusions"""
    occlusions = {}
    for pose in possible_poses:
        intersected_occluded_voxels = []
        if pose not in pose_intersection_map.keys():
            continue
        for intersection in pose_intersection_map[pose]:
            ix, iy, iz = intersection
            if occluding_voxels[ix][iy][iz] > 0:
                # We have an occlusion for this pose at this voxel
                intersected_occluded_voxels.append(intersection)
        if len(intersected_occluded_voxels) > 0:
            occlusions[pose] = intersected_occluded_voxels
    return occlusions


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def flatten_index(tup, dims):
    """ Get from a pose tuple to its index in a large array."""
    nr_rows, nr_cols, nr_planes = dims
    (angle, row, col) = tup
    return col + (row * nr_cols) + (angle * (nr_rows * nr_cols))


def unflatten_index(i, dims, angles):
    nr_rows, nr_cols, nr_planes = dims
    """ Get from a flat index to the pose."""
    angle = angles[int(np.floor(i / (nr_rows * nr_cols)))]
    i = i % (nr_rows * nr_cols)
    row = int(np.floor(i / nr_cols))
    col = i % nr_cols
    return angle, row, col


class IterationType(Enum):
    VOLUMECOVERAGE = 1
    OCCLUSIONPREVENTION = 2
    PERPENDICULAR = 3
    RANDOM = 4


iteration_name_dict = {IterationType.VOLUMECOVERAGE: "volume_coverage",
                       IterationType.OCCLUSIONPREVENTION: "occlusion_prevention",
                       IterationType.PERPENDICULAR: "perpendicular",
                       IterationType.RANDOM: "random"}

def get_iter_prefix_filepath(model_name: str, iter_type: IterationType):
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m_%H%M_")
    return "imfusion_files/results/" + model_name + "/" + dt_string + iteration_name_dict[iter_type]


def prepare_results_folders(model_name: str, iter_filepath_prefix: str):
    # Check if folder for saving the sweeps/maps exists, and if not, create it
    model_folder_path = "imfusion_files/results/" + model_name
    if not os.path.isdir(model_folder_path):
        os.mkdir(model_folder_path)

    general_folder_path = model_folder_path + "/general"
    if not os.path.isdir(general_folder_path):
        os.mkdir(general_folder_path)

    sweeps_folder_path = general_folder_path + "/sweeps"
    if not os.path.isdir(sweeps_folder_path):
        os.mkdir(sweeps_folder_path)

    confidence_maps_folder_path = general_folder_path + "/confidence_maps"
    if not os.path.isdir(confidence_maps_folder_path):
        os.mkdir(confidence_maps_folder_path)

    tracking_folder_path = general_folder_path + "/tracking"
    if not os.path.isdir(tracking_folder_path):
        os.mkdir(tracking_folder_path)

    tracking_folder_path = general_folder_path + "/confidence_means"
    if not os.path.isdir(tracking_folder_path):
        os.mkdir(tracking_folder_path)

    if iter_filepath_prefix is not None:
        # Create folders for this specific iteration
        if not os.path.isdir(iter_filepath_prefix):
            os.mkdir(iter_filepath_prefix)

        sweeps_folder_path_i = iter_filepath_prefix + "/sweeps"
        if not os.path.isdir(sweeps_folder_path_i):
            os.mkdir(sweeps_folder_path_i)

        confidence_maps_folder_path_i = iter_filepath_prefix + "/confidence_maps"
        if not os.path.isdir(confidence_maps_folder_path_i):
            os.mkdir(confidence_maps_folder_path_i)

        tracking_folder_path_i = iter_filepath_prefix + "/tracking"
        if not os.path.isdir(tracking_folder_path_i):
            os.mkdir(tracking_folder_path_i)

        tracking_folder_path_i = iter_filepath_prefix + "/confidence_means"
        if not os.path.isdir(tracking_folder_path_i):
            os.mkdir(tracking_folder_path_i)


# Returns the default name following a pattern for a pose
def get_imfusion_savepath_name(file_name: str, pose: (float, int, int), dims: (int, int, int)):
    angle, x, y = pose
    rows, cols, planes = dims
    return file_name + "_dims" + str(rows) + str(cols) + str(planes) + "_pose" + str(round(angle, 0)) + "_" + str(
        x) + "_" + str(y)


def get_std_name(folder: str, filetype: str, dims: tuple, focal_dist: float, angles: list):
    """ The std name for a distance file."""
    nr_rows, nr_cols, nr_planes = dims
    name = folder + "/foc_" + str(focal_dist) + "_rows" + str(nr_rows) + "_cols" + str(nr_cols) + "_planes" \
           + str(nr_planes) + "_angles_"
    for angle in angles:
        name += str(angle) + "_"
    return name + "." + filetype


def map_weights_to_poses(w, dims, angles):
    """ Re-assign each weight to its pose."""
    poses = {}
    for i in range(shape(w)[0]):
        tup = unflatten_index(i, dims, angles)
        poses[tup] = w[i]
    return poses


def get_occluded_voxels_from_occlusions(occlusions):
    occluded_voxels = []
    for voxels in occlusions.values():
        occluded_voxels.extend(voxels)
    return occluded_voxels
