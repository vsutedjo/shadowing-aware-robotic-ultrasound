import numpy as np
from matplotlib import pyplot as plt, gridspec
from numpy import shape
from numpy.lib import math

from functions.helpers.helpers import get_all_intersections


def prepare_poses_list(all_poses: [(float, int, int)], old_occlusions, old_poses, dims):
    # Hard remove all poses that are not touching any occluded voxels
    # Hard remove all poses that have been executed before

    occluded_voxels = []
    for occ_list in old_occlusions.values():
        occluded_voxels.extend(occ_list)
    occluded_voxels = [tuple(v) for v in occluded_voxels]
    poses = []
    for pose in all_poses:
        # Ignore poses that have been executed before
        if pose in old_poses:
            continue

        # Only add poses that intersect with the occluded voxels
        intersected_voxels = get_intersected_voxels(dims, pose)
        if any([v in occluded_voxels for v in intersected_voxels]):
            poses.append(pose)

    return poses


def color_cube(location: (int, int, int), ax, dims, cube_intensity, color: (float, float, float) = (0.5, 0, 0.5)):
    x, y, z = location
    other_color = np.zeros(dims)
    other_color[x, y, z] = 1
    ax.voxels(other_color, facecolors=[color[0], color[1], color[2], cube_intensity])


def setup_plot(nr_cols, voxels, occlusion_volume=None, plot_multiple: bool = False, freeze: bool = False):
    # Interactive mode on for pycharm
    plt.ion()
    fig = plt.figure(0)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Set up the main axis
    subplot_config = 111
    if plot_multiple:
        subplot_config = 221
    ax = fig.add_subplot(subplot_config, projection="3d")
    ax.set_ylim(0, nr_cols)
    ax.voxels(voxels, facecolors=[0.6, 0.6, 0.6, 0.1], edgecolor=[0.5, 0.5, 0.5, 0.2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot the occlusion volume
    if occlusion_volume is not None:
        plot_occlusion(occlusion_volume, ax)

    if freeze:
        plt.show(block=True)
    return spec, fig, ax


def plot_transducer(nr_planes, transducer_width, origin_xy, tilt_angle, ax):
    slope = math.tan(math.radians(tilt_angle))
    [x, y] = origin_xy
    xs = [[x, x + transducer_width], [x, x + transducer_width]]
    ys = [[y, y], [y + nr_planes * slope, y + nr_planes * slope]]
    return ax.plot_surface(xs, ys, np.array([[nr_planes, nr_planes], [0, 0]]), color=[0.6, 0.6, 0.6, 0.8])


def is_vox_valid(vox, dims):
    """ Whether or not a voxel is within the dimensions."""
    for i in range(3):
        v = vox[i]
        d = dims[i]
        if v < 0 or v >= d:
            # The voxel has an overflow or underflow at one dimension
            return False
    return True


def plot_intersected_voxels_with_occlusions(voxels, occluded_voxels, dims, pose):
    rows, cols, planes = dims
    origin_xy = pose[1], pose[2] + 0.5
    angle = pose[0]

    _, __, ax = setup_plot(cols, np.ones(dims))
    ax.view_init(20, 0)

    plot_transducer(planes, 1, origin_xy, angle, ax)
    for vox in voxels:
        color_cube(vox, ax, dims, 0.1, color=(0.1, 0.1, 0.1))
    for occ_vox in occluded_voxels:
        color_cube(occ_vox, ax, dims, 0.2, color=(0.8, 0.0, 0.0))

    # Uncomment if you want to also plot the intersections
    # slope = 0
    # if pose[0] != 0:
    #     slope = 1 / round(math.tan(math.radians(pose[0])), 2)

    # h_ints, v_ints = get_all_intersections(dims, (pose[1] + 0.5, pose[2] + 0.5),
    #                                        pose[0],
    #                                        slope)
    # ints = h_ints
    # ints.extend(v_ints)
    # for ints in ints:
    #     ax.scatter(ints[0],ints[1],ints[2])
    plt.draw()
    plt.pause(0.5)


def get_intersected_voxels(dims: (int, int, int), pose, plot: bool = False):
    slope = 0
    if pose[0] != 0:
        slope = 1 / round(math.tan(math.radians(pose[0])), 2)

    vertical_intersections, horizontal_intersections = get_all_intersections(dims, (pose[1] + 0.5, pose[2] + 0.5),
                                                                             pose[0],
                                                                             slope)
    voxs = []
    rows, cols, planes = dims
    origin_xy = pose[1], pose[2] + 0.5
    angle = pose[0]

    for h_int in horizontal_intersections:
        x, y, z = h_int
        upper_vox = int(x), math.floor(y), int(z)
        lower_vox = int(x), math.floor(y), int(z - 1)

        if is_vox_valid(upper_vox, dims):
            voxs.append(upper_vox)
        if is_vox_valid(lower_vox, dims):
            voxs.append(lower_vox)

    for v_int in vertical_intersections:
        x, y, z = v_int
        front_vox = int(x), math.floor(y), math.floor(z)
        back_vox = int(x), math.floor(y - 1), math.floor(z)

        if is_vox_valid(front_vox, dims):
            voxs.append(front_vox)
        if is_vox_valid(back_vox, dims):
            voxs.append(back_vox)

    if plot:
        _, __, ax = setup_plot(cols, np.ones(dims))
        plot_transducer(planes, 1, origin_xy, angle, ax)
        for vox in voxs:
            color_cube(vox, ax, dims, 0.5)
        plt.draw()
        plt.pause(0.5)

    return set(voxs)


def plot_voxel_measurements(measurements, ax):
    rows, cols, planes = shape(measurements)
    ax.set_xlim(0, rows)
    ax.set_ylim(0, cols)
    ax.set_zlim(0, planes)
    for row in range(rows):
        for col in range(cols):
            for plane in range(planes):
                ax.text(row + 0.25, col + 0.25, plane + 0.5, round(measurements[row][col][plane], 3))


def plot_occlusion(occlusion_volume, ax):
    ax.plot_surface(X=occlusion_volume[0], Y=occlusion_volume[1], Z=occlusion_volume[2])


def plot_occluded_voxels(occlusions: dict, dims):
    _, fig, ax = setup_plot(dims[1], np.ones(dims), plot_multiple=False)
    plt.title("Occluded voxels from US simulation, in voxel coordinates")
    for voxels in occlusions.values():
        for voxel in voxels:
            color_cube(voxel, ax, dims, 0.5, color=(0.9, 0.3, 0.3))
    plt.show(block=True)


def plot_voxel_counts(voxel_counts: list, dims):
    _, fig, ax = setup_plot(dims[1], voxel_counts, plot_multiple=False)
    plt.title("Voxel counts")

    plt.show(block=True)
