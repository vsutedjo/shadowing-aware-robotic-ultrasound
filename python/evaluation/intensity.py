import numpy as np
import os
import json
import matplotlib.pyplot as plt

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def intensity(comp_method: str, compounded_volume, intensity_mask, num_iterations):
    """
    This function computes the average pixel intensity in an occlusion area based on the US sweep compounding.
    :param model_name: Model name of the phantom that is being evaluated.
    :type: string
    :param compounding_path: path to the compounding .imf file
    :type: string
    :return: averaged intensity value in occlusion area
    """
    # imfusion.init()

    avg_intensity = np.mean(compounded_volume[intensity_mask > 0]) / num_iterations
    avg_intensity = round(avg_intensity, 5)
    # print("avg intensity with "+comp_method+" is: ", avg_intensity)

    return avg_intensity
