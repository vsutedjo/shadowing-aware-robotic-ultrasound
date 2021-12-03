import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def get_decay(confidence_path):
    imfusion.init()
    cm = imfusion.open(confidence_path)[0][0]
    cm_array = np.squeeze(np.array(cm))

    decay = np.mean(cm_array, axis=1)
    decay[decay < 0.1] = 0.1

    plt.plot(decay)
    plt.show()

    return decay


def adjust_confidence(file_paths, decay):
    for file in file_paths:

        cm = imfusion.open(file)[0][0]
        cm_array = np.squeeze(np.array(cm))

        adjusted_confidence = np.multiply(cm_array, np.expand_dims(1/decay, -1))

        plt.imshow(adjusted_confidence, cmap='gray')
        plt.show()


cm_folder = "imfusion_files/results/two_beams_nonparallel_soft_tissue/general/confidence_maps/"

decay = get_decay("imfusion_files/results/two_beams_nonparallel_soft_tissue/general/confidence_maps/confidence_map_dims8106_pose0_0_0.imf")

filelist = [os.path.join(cm_folder, item) for item in os.listdir(cm_folder)]
adjust_confidence(filelist, decay)