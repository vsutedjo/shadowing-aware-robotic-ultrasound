import math
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from numpy import shape

from functions.helpers.helpers import calculate_possible_poses, sort_transducer_poses
from functions.helpers.imfusion.spatial_transformation_helpers import get_pixels_inside_voxel

from functions.helpers.imfusion.ImfusionTrackedImage import ImfusionTrackedImage

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
import imfusion


def main():
    imfusion.init()

    # settings
    dims = [3, 3, 3]
    transducer_width = 1
    angles = [0]
    model_name = "square"

    # load poses
    poses = sort_transducer_poses(calculate_possible_poses(dims[0], dims[1], angles))
    data = imfusion.open("imfusion_files/3dModels/" + model_name + ".imf")[0]

    confidence_maps_dict = {}

    for i in range(len(poses)):
        confidence_maps = imfusion.open("imfusion_files/confidence_maps/confidence_map" + str(i) + ".imf")
        simulated_sweep = imfusion.open("imfusion_files/sweeps/simulated_sweep" + str(i) + ".imf")
        first_confidence_map = imfusion.executeAlgorithm('Split Images', [confidence_maps[0]])[0]
        confidence_map = ImfusionTrackedImage(imfusion_shared_image_set=first_confidence_map,
                                              # Taking the first element
                                              matrix=simulated_sweep[0].matrix(0))  # matrix in imfusion coordinates

        confidence_maps_dict[poses[i]] = confidence_map

    # TEST 1 on first image:

    us_Sweep = imfusion.open("imfusion_files/sweeps/simulated_sweep" + str(0) + ".imf")[0]

    properties = imfusion.Properties({'resamplingMode': 0,
                                      'targetDimensions': dims})
    data = imfusion.executeAlgorithm('Image Resampling', [data], properties)[0]

    # 3. Getting the volume info
    T_vol2world = np.linalg.inv(data[0].matrix)  # data.matrix = world to data coordinates
    vol_spacing = data[0].spacing  # spacing in imfusion axes
    vol_shape = [data[0].width, data[0].height, data[0].slices]  # shape in imfusion axes

    confidence = confidence_maps_dict[poses[0]]

    # Getting ultrasound info in imfusion axes
    T_us2world = confidence.matrix
    confidence_shape = confidence.get_shape()
    confidence_spacing = confidence.get_spacing()

    vol_array = np.array(data[0], copy=False)

    us_arrays = []
    for _, img in enumerate(us_Sweep):
        us_arrays.append(np.array(img, copy=False))

    data.setDirtyMem()
    us_Sweep.setDirtyMem()

    i = 0
    for x in range(vol_shape[0]):
        for y in range(vol_shape[1]):
            for z in range(vol_shape[2]):
                proximal_pixels = get_pixels_inside_voxel(P_voxel=[x, y, z],
                                                          volume_spacing=vol_spacing,
                                                          volume_size_in_voxel=vol_shape,
                                                          T_vol2world=T_vol2world,
                                                          image_spacing=confidence_spacing,
                                                          image_size_in_pixels=confidence_shape,
                                                          T_img2world=T_us2world)

                for item in proximal_pixels:
                    for k in range(len(us_arrays)):
                        us_arrays[k][item[1], item[0], :] = i

                if x == 0 and y == 0 and z == 0:
                    print(proximal_pixels)

                if x == 0 and y == 1 and z == 0:
                    print(proximal_pixels)

                if x == 1 and y == 1 and z == 0:
                    print(proximal_pixels)

                vol_array[z, y, x, :] = i

                i += 1

    vol_slice = imfusion.executeAlgorithm('Convert To Volume', [us_Sweep],
                                          imfusion.Properties({'Slice Thickness': 0.5}))[0]

    vol_slice[0].matrix = np.linalg.inv(T_us2world)
    vol_slice[0].modality = imfusion.Data.Modality.LABEL

    imfusion.executeAlgorithm('IO;ImFusionFile', [vol_slice],
                              imfusion.Properties({'location': "imfusion_files/vol_slice.imf"}))

    imfusion.executeAlgorithm('IO;ImFusionFile', [data],
                              imfusion.Properties({'location': "imfusion_files/data_modified.imf"}))
    imfusion.executeAlgorithm('IO;ImFusionFile', [us_Sweep],
                              imfusion.Properties({'location': "imfusion_files/us_modified.imf"}))


main()
