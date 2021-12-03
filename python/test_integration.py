import math
import os

from functions.helpers.helpers import get_iter_prefix_filepath, IterationType
from functions.helpers.imfusion.ImfusionTrackedImage import ImfusionTrackedImage
from functions.helpers.imfusion.imfusion_helpers import occlusions_from_confidence

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def main():
    imfusion.init()
    # load poses
    poses = [(0, 0, 0), (45, 0, 0), (0, 0, 1), (45, 0, 1), (0, 1, 1), (45, 1, 1), (0, 1, 0), (45, 1, 0), (0, 2, 0),
             (45, 2, 0), (0, 2, 1), (45, 2, 1)]
    model_name = "BigA"
    data = imfusion.open("imfusion_files/3dModels/" + model_name + ".imf")[0]

    dims = [3, 3, 3]
    transducer_width = 1
    angles = [0, 45]
    confidence_maps_dict = {}
    iteration_type = IterationType.VOLUMECOVERAGE  # This one doesn't matter here

    for i in range(12):
        confidence_maps = imfusion.open("imfusion_files/confidence_maps/confidence_map" + str(i) + ".imf")
        simulated_sweep = imfusion.open("imfusion_files/sweeps/simulated_sweep" + str(i) + ".imf")
        first_confidence_map = imfusion.executeAlgorithm('Split Images', [confidence_maps[0]])[0]

        confidence_map = ImfusionTrackedImage(imfusion_shared_image_set=first_confidence_map,
                                              # Taking the first element
                                              matrix=simulated_sweep[0].matrix(0))  # matrix in imfusion coordinates

        confidence_maps_dict[poses[i]] = confidence_map
    iter_filepath_prefix = get_iter_prefix_filepath(model_name, iteration_type)
    occlusions_from_confidence(model_name, confidence_maps_dict, dims, data, transducer_width, iter_filepath_prefix,
                               angles=angles, poses=poses, plot_simulation=False)


main()
