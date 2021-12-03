import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib import pyplot as plt, gridspec

from functions.helpers.helpers import splines_from_poses, get_probe_width, prepare_results_folders, \
    get_imfusion_savepath_name
from functions.helpers.imfusion.ImfusionTrackedImage import ImfusionTrackedImage
from functions.helpers.imfusion.spatial_transformation_helpers import get_pixels_inside_voxel
from functions.helpers.plot_functions import setup_plot

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def main(model_name, out_folder):
    imfusion.init()

    cm_path = "imfusion_files/results/" + model_name + "/confidence_maps/"
    tracking_path = "imfusion_files/results/" + model_name + "/tracking/"

    cm_list = os.listdir(cm_path)

    for cm_name in cm_list:
        tracking_file = cm_name.replace("confidence_map", "tracking")
        tracking_file = tracking_file.replace(".imf", ".ts")

        cm = imfusion.open(os.path.join(cm_path, cm_name))

        cm[0].modality = imfusion.Data.Modality.ULTRASOUND
        tracked_cm = imfusion.executeAlgorithm("Convert to Sweep", cm,
                                       imfusion.Properties({'Tracking stream file': tracking_path + tracking_file}) )

        imfusion.executeAlgorithm('IO;ImFusionFile', tracked_cm,
                                  imfusion.Properties({'location': out_folder + cm_name}))

model_name = "square"
out_folder = "imfusion_files/results/" + model_name + "tracked_confidence"

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

main(model_name, out_folder)
