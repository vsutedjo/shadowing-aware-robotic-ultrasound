"""
These scripts are not proper unit test, but are rather meant to fastly check the functioning of single functions
in the code, with simple data and simple visualization
"""

import os
import numpy as np
from functions.helpers.imfusion.imfusion_helpers_maria import get_voxel_confidence_at_pose

from functions.helpers.imfusion.ImfusionTrackedImage import ImfusionTrackedImage

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
import imfusion


def test_script_get_pixels_inside_voxel():

    imfusion.init()

    # TEST 1 on first image:
    us_sweep = imfusion.open("test_data/simulated_sweep1.imf")[0]
    data = imfusion.open("test_data/colored_voxels_volume.imf")[0]

    # 3. Getting the volume info
    T_vol2world = np.linalg.inv(data[0].matrix)  # data.matrix = world to data coordinates
    vol_spacing = data[0].spacing  # spacing in imfusion axes
    vol_shape = [data[0].width, data[0].height, data[0].slices]  # shape in imfusion axes

    # Getting ultrasound info in imfusion axes

    T_us2world = us_sweep.matrix(0)
    confidence_shape = [us_sweep[0].width, us_sweep[0].height]
    confidence_spacing = us_sweep[0].spacing

    vol_array = np.array(data[0])
    # get the us sweep data without creating a copy, for each array in the us sweep
    us_arrays = []
    for _, img in enumerate(us_sweep):
        us_arrays.append(np.array(img, copy=False))

    us_sweep.setDirtyMem()

    print(T_us2world)

    i = 0
    for x in range(vol_shape[0]):
        for y in range(vol_shape[1]):
            for z in range(vol_shape[2]):
                proximal_pixels = get_voxel_confidence_at_pose(P_voxel=[x, y, z],
                                                               volume_spacing=vol_spacing,
                                                               volume_size_in_voxel=vol_shape,
                                                               T_vol2world=T_vol2world,
                                                               image_spacing=confidence_spacing,
                                                               image_size_in_pixels=confidence_shape,
                                                               T_img2world=T_us2world)

                # color the us sweep pixels corresponding to the volume x, y, z, with the same color

                color = vol_array[z, y, x, :]
                for item in proximal_pixels:
                    for k in range(len(us_arrays)):
                        us_arrays[k][item[1], item[0], :] = color

                if x == 0 and y == 0 and z == 0:
                    print(proximal_pixels)

                i += 1

    imfusion.executeAlgorithm('IO;ImFusionFile', [us_sweep],
                              imfusion.Properties({'location': "out_data/us_modified.imf"}))


test_script_get_pixels_inside_voxel()
