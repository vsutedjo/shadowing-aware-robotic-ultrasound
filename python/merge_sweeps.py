import os

import numpy as np
from functions.helpers.helpers import sort_transducer_poses

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def get_sorted_files(data_path):

    prefix = os.listdir(data_path)[0].split("pose")[0]
    poses = [item.split("pose")[-1] for item in os.listdir(data_path)]
    poses = [item.replace(".imf", "") for item in poses]
    poses = [(int(item.split("_")[0]), int(item.split("_")[1]), int(item.split("_")[2])) for item in poses]

    sorted_poses = sort_transducer_poses(poses)

    sorted_poses_files = [os.path.join(data_path, prefix+ "pose" +
                                       str(item[0]) + "_" + str(item[1]) + "_" + str(item[2]) + ".imf") for item in sorted_poses]
    return sorted_poses_files

def main(sweeps_folder, output_path):

    imfusion.init()

    sweep_paths = get_sorted_files(sweeps_folder)

    shared_image_set = imfusion.SharedImageSet()
    tracking_stream = imfusion.TrackingStream()
    for sweep_path in sweep_paths:
        sweep = imfusion.open(os.path.join(sweeps_folder, sweep_path))[0]
        us_image = np.array(sweep[0])
        shared_us_image = imfusion.SharedImage(us_image)
        shared_us_image.spacing = sweep[0].spacing

        tracking_data = sweep.matrix(0)

        shared_image_set.add(shared_us_image)
        tracking_stream.add(tracking_data)

    shared_image_set.modality = imfusion.Data.Modality.ULTRASOUND

    imfusion.executeAlgorithm('IO;Tracking Stream', [tracking_stream],
                              imfusion.Properties(
                                  {
                                      'location': "tracking_stream.ts"}))

    tracked_cm = imfusion.executeAlgorithm("Convert to Sweep", [shared_image_set],
                                           imfusion.Properties({'Tracking stream file': "tracking_stream.ts"}))

    imfusion.executeAlgorithm('IO;ImFusionFile', tracked_cm,
                              imfusion.Properties({'location': output_path}))


if __name__ == "__main__":

    sweep_path = "C:/UniMatter/Master/MasterThesis/optimization_python/python/imfusion_files/results/two_beams_nonparallel_water/0709_0007_occlusion_prevention/sweeps"
    outpath = "imfusion_files/results/two_beams_nonparallel_water/combined_occlusion_reduction.imf"

    imfusion.init()
    main(sweep_path, outpath)
