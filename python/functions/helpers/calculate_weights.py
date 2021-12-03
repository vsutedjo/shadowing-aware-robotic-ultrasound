import math

import numpy as np
from matplotlib import pyplot as plt
from numpy import shape

from functions.helpers.helpers import get_horizontal_intersections, get_vertical_intersections
from functions.helpers.measurement_functions import measure_voxel_values_for_voxel
from functions.helpers.plot_functions import setup_plot, plot_transducer, color_cube


def d_and_measurements_for_pose(dims: (int, int, int), transducer_width: float, origin_xy: (int, int),
                                foc_depth_percent: float, tilt_angle: float, plt_meta: (any, float) = (None, None),
                                measurements: [float] = None, voxel_counts: [int] = None, ground_truth: [float] = None,
                                occlusion: tuple = None) -> (
        [float], [float], [int], [(int, int, int)]):
    [rows, cols, planes] = dims
    ax, cube_intensity = plt_meta

    left_x = origin_xy[0]

    # Return for this measurement whether it was occluded.
    occluded_voxels = []

    # The rightmost voxel we should look at.
    # If the transducer is exactly at the edge, it should not count as intersection.
    # Hence, we deduce a small amount on that side.
    right_x = min(rows - 1, int(np.floor(origin_xy[0] + transducer_width - 0.00001)))
    m = np.zeros((rows, cols, planes))
    for row in range(left_x, right_x + 1):
        slope = 0
        if tilt_angle != 0:
            slope = 1 / round(math.tan(math.radians(tilt_angle)), 2)

        horizontal_intersections = get_horizontal_intersections(planes, row, origin_xy, tilt_angle, slope)
        vertical_intersections = get_vertical_intersections(cols, planes, row, origin_xy, tilt_angle, slope)

        focal_point = get_focal_point((origin_xy[0], origin_xy[1], planes), foc_depth_percent, slope)

        for v_y in range(cols):
            for v_z in range(planes):
                if (v_z == planes - 1 and v_y == np.floor(origin_xy[1])) or voxel_is_intersected_top_or_bottom(
                        [origin_xy[0], v_y, v_z], horizontal_intersections) or voxel_is_intersected_sides(
                        [origin_xy[0], v_y, v_z], tilt_angle, vertical_intersections):

                    # Check if this voxel is occluded
                    # TODO add left/right of the transducer
                    curr_vox = (row, v_y, v_z)
                    dist = 0
                    if occlusion is not None and ray_occludes_voxel(planes, origin_xy, curr_vox, occlusion, tilt_angle):
                        occluded_voxels.append(curr_vox)

                        # Color this cube grey if it is a true occlusion
                        if ax is not None and voxel_counts is not None and voxel_counts[row][v_y][v_z] == 0:
                            color_cube(curr_vox, ax, dims, 0.1, color=(0.1, 0.1, 0.1))
                    else:
                        # Measure the value in this voxel and add it to our weights
                        if measurements is not None and ground_truth is not None:
                            measurement = measure_voxel_values_for_voxel(curr_vox, ground_truth=ground_truth)
                            measurements[row][v_y][v_z] += measurement

                        # Add the voxel count
                        if voxel_counts is not None:
                            voxel_counts[row][v_y][v_z] += 1

                        # Calculate the dist for this voxel
                        dist = calculate_dist(focal_point, (origin_xy[0], v_y + 0.5, v_z + 0.5))

                        # Color the cube intensity according to its distance
                        if ax is not None:
                            color_cube(curr_vox, ax, dims, cube_intensity * dist)

                    # Save the calculated dist (or 0 if occluded) to m
                    m[row, v_y, v_z] = dist

    m = normalize_array(m)
    return m, measurements, voxel_counts, occluded_voxels


def ray_occludes_voxel(planes, origin_xy, vox_location, occlusion, tilt_angle):
    if occlusion is None:
        return False
    occ_z = occlusion[2][0][0]  # Assuming z is constant / occlusion is untilted plane
    vox_x, vox_y, vox_z = vox_location

    # The voxel is occluded if the occlusion is over the voxel
    smallest_occlusion_x = min(occlusion[0][0])
    largest_occlusion_x = max(occlusion[0][0])
    smallest_occlusion_y = min(occlusion[1][0])
    largest_occlusion_y = min(occlusion[1][-1])

    occluded = (smallest_occlusion_x <= vox_x <= largest_occlusion_x
                or smallest_occlusion_x <= vox_x + 1 <= largest_occlusion_x) \
               and (smallest_occlusion_y <= vox_y <= largest_occlusion_y
                    or smallest_occlusion_y <= vox_y + 1 <= largest_occlusion_y) \
               and vox_z < occ_z

    # Calculate the intersection of the transducer to the plane.
    slope = round(math.tan(math.radians(tilt_angle)), 2)
    z_dist = planes - occ_z
    y = origin_xy[1] + z_dist * slope
    # The voxel is only occluded if the transducer actually crosses the plane.
    occluded &= smallest_occlusion_y <= y <= largest_occlusion_y
    return occluded


def get_focal_point(p, dist, slope):
    if slope == 0:
        return p[0], p[1], p[2] - dist
    v = [0, 1, -slope]

    if slope < 0:
        v = [0, -1, slope]
    delta = dist / np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    delta_v = (v[0] * delta, v[1] * delta, v[2] * delta)

    focal_point = (p[0] + delta_v[0], p[1] + delta_v[1], p[2] + delta_v[2])
    return focal_point


def normalize_array(a):
    return (a - a.min()) / (a.max() - a.min())


def calculate_dist(p1, p2):
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    # Inverse distance => small distance get a higher weight
    if d != 0:
        d = 1 / np.sqrt(d)
    else:
        d = 3
    return d


def voxel_is_intersected_sides(voxel_coordinates, tilt_angle, points):
    for p in points:
        (row, col, plane) = p
        is_in_voxel_plane = voxel_coordinates[2] <= plane <= voxel_coordinates[2] + 1
        is_left_of_voxel = voxel_coordinates[1] - 1 == col - 1
        is_right_of_voxel = voxel_coordinates[1] == col - 1
        if is_in_voxel_plane and (is_right_of_voxel or is_left_of_voxel):
            return True

    return False


def voxel_is_intersected_top_or_bottom(voxel_coordinates, points):
    if len(points) < 1:
        return False
    p_bottom = points[voxel_coordinates[2]]
    bottom_contained = voxel_coordinates[1] <= p_bottom[1] <= voxel_coordinates[
        1] + 1 and voxel_coordinates[2] <= p_bottom[2] <= voxel_coordinates[2] + 1

    top_contained = False
    # If there exists a point on top of us
    if voxel_coordinates[2] < shape(points)[1] - 1:
        p_top = points[voxel_coordinates[2]]
        top_contained = voxel_coordinates[1] <= p_top[1] <= voxel_coordinates[
            1] + 1 and voxel_coordinates[2] <= p_top[2] <= voxel_coordinates[2] + 1

    return bottom_contained or top_contained
