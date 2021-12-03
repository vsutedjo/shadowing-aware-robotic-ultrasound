import json

from evaluation.volume_coverage import volume_coverage
from evaluation.confidence import get_confidence_matrix
from evaluation.confidence import confidence
from evaluation.intensity import intensity
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion

imfusion.init()

nr_rows = 8
nr_cols = 10
nr_planes = 6
angles = [-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]  # Write the angles from smallest to largest
model_name = "two_beams_nonparallel"
material_type = "water"  # "soft_tissue"
dims = (nr_rows, nr_cols, nr_planes)

# iteration_folders_names = ["0109_1141_perpendicular"]
# iteration_folders_names = ["0209_2250_random", "0209_2256_occlusion_prevention"]

# Put all the evaluation runs directly here, it will print them as a table
all_folders = [["0209_2328_random"], ["0209_2328_random", "0209_2339_occlusion_prevention"],
               ["0209_2347_random"], ["0209_2347_random", "0209_2356_occlusion_prevention"],
               ["0309_0001_random"], ["0309_0001_random", "0309_0011_occlusion_prevention"],
               ["0309_0014_random"], ["0309_0014_random", "0309_0022_occlusion_prevention"],
               ["0309_0024_random"], ["0309_0024_random", "0309_0035_occlusion_prevention"],
               ["0309_0826_random"], ["0309_0826_random", "0309_0831_occlusion_prevention"],
               ["0309_0833_random"], ["0309_0833_random", "0309_0838_occlusion_prevention"],
               ["0309_1003_random"], ["0309_1003_random", "0309_1009_occlusion_prevention"],
               ["0309_1012_random"], ["0309_1012_random", "0309_1022_occlusion_prevention"],
               ["0309_1024_random"], ["0309_1024_random", "0309_1029_occlusion_prevention"], ]

def line_ray_casting(line):
    ray_casted_line = np.zeros(line.shape)
    idx = np.argwhere(line > 0)

    if len(idx) != 0:
        try:
            idx = idx.flatten()[-1]
        except:
            print()
        ray_casted_line[0:idx] = 1  # from 0 to idx as we used flip in imfusion

    return ray_casted_line


def get_under_occlusion_area(imfusion_model, foreground_value=13, save_path="", save_result=True):
    volume_array = np.array(imfusion_model)

    model_array = np.squeeze(np.array(volume_array[0]))
    model_array[model_array != foreground_value] = 0
    model_array[model_array > 0] = 1

    flatten_columns = np.reshape(model_array, [model_array.shape[0], -1])
    ray_casted_columns = np.apply_along_axis(line_ray_casting, axis=0, arr=flatten_columns)

    ray_casted_volume = np.reshape(ray_casted_columns, model_array.shape)

    if save_result:
        shared_image = imfusion.SharedImage(np.expand_dims(ray_casted_volume, -1))
        shared_image.spacing = imfusion_model[0].spacing
        shared_image.matrix = imfusion_model[0].matrix

        out_set = imfusion.SharedImageSet()
        out_set.add(shared_image)

        imfusion.executeAlgorithm('IO;ImFusionFile', [out_set],
                                  imfusion.Properties({'location': save_path}))

    return ray_casted_volume


def get_compounded_volume(experiment_folder, save_result=True):
    if not os.path.exists(os.path.join(experiment_folder, "sweep_compounding")):
        os.mkdir(os.path.join(experiment_folder, "sweep_compounding"))

    compounding_path = os.path.join(experiment_folder, "sweep_compounding", "compounding.imf")
    if os.path.exists(compounding_path):
        compounded_volume, = imfusion.open(compounding_path)
        return compounded_volume

    sweeps_folder = os.path.join(experiment_folder, "sweeps")
    sweeps = [imfusion.open(os.path.join(sweeps_folder, item))[0] for item in os.listdir(sweeps_folder)]

    compounded_volume, = imfusion.executeAlgorithm('Sweep Compounding', sweeps,
                                                   imfusion.Properties({'backgroundIntensity': 0,
                                                                        'individualCompoundings': 0}))

    if save_result:
        imfusion.executeAlgorithm('IO;ImFusionFile', [compounded_volume],
                                  imfusion.Properties({'location': compounding_path}))

    return compounded_volume


def compound_from_different_folders(folder_list):
    sweeps_paths = []
    existing_items = []
    for folder in folder_list:
        new_items = [item for item in os.listdir(folder) if item not in existing_items]
        sweeps_paths.extend(os.path.join(folder, item) for item in new_items)
        existing_items.extend(new_items)

    sweeps = [imfusion.open(item)[0] for item in sweeps_paths]
    compounded_volume, = imfusion.executeAlgorithm('Sweep Compounding', sweeps,
                                                   imfusion.Properties({'backgroundIntensity': 0,
                                                                        'individualCompoundings': 0}))
    return compounded_volume


def add_volumes(volume_list):
    assert len(volume_list) > 0

    if len(volume_list) == 1:
        return volume_list[0]

    volume = volume_list[0]
    for item in volume_list[1::]:
        volume += item
    return volume


def add_imfusion_volumes(volume_list):
    properties = imfusion.Properties({'evalString': "(imgset0) + (imgset1)",
                                      'inPlace': 0,
                                      'mode': 0,
                                      'rightHandSide': 1,
                                      'channelIndices': 0})
    combined_volumes, = imfusion.executeAlgorithm('Arithmetic Operators', volume_list, properties)

    return combined_volumes


def resize_imfusion_volume(imfusion_volume, new_size):
    properties = imfusion.Properties({'resamplingMode': 0,
                                      'targetDimensions': (new_size[0],
                                                           new_size[1],
                                                           new_size[2])})
    resized_model, = imfusion.executeAlgorithm('Image Resampling', imfusion_volume, properties)
    return resized_model


def align_volumes(volume_list):
    if len(volume_list) == 1:
        return volume_list

    properties = imfusion.Properties({'cloneDeformation': 1,
                                      'preserveExtent': 1,
                                      'reductionMode': 1,
                                      'interpolationMode': 1})
    imfusion.executeAlgorithm('Image Resampling', volume_list, properties)
    return volume_list


def save_imfusion_volume(vol, filepath):
    imfusion.executeAlgorithm('IO;ImFusionFile', [vol],
                              imfusion.Properties({'location': filepath}))


def evaluation():
    # Run an evaluation on the processed data.
    model_name_whole = model_name + "_" + material_type
    rows, cols, planes = dims
    print("            iteration        || #pos. ||  avg cov ||  conf.  || inten avg  ||intensity comb.  ")

    for iteration_folders_names in all_folders:

        iteration_folders = ["../imfusion_files/results/" + model_name_whole + "/" + item
                             for item in iteration_folders_names]

        it_name = iteration_folders_names[0][10:]
        if len(iteration_folders_names) > 1:
            it_name += " + " + iteration_folders_names[1][10:]
        else:
            it_name += "                       "
        print(it_name, end='  ||  ')

        poses_paths = [item + "/poses.json" for item in iteration_folders]
        nr_poses = 0
        for path in poses_paths:
            with open(path) as f:
                data = json.load(f)
                nr_poses += len(list(data))
        print(nr_poses, end='  ||  ')


        volume_coverage_voxel_counts_path = [item + "/voxel_counts.pkl" for item in iteration_folders]

        confidence_filepath = "\\confidence_means\\dims" + str(rows) + str(cols) + str(
            planes) + "_confidence_means.json"
        confidence_means_path = [item + confidence_filepath for item in iteration_folders]

        imfusion_model, = imfusion.open("../imfusion_files/3dModels/" + model_name_whole + ".imf")
        occlusion_mask = get_under_occlusion_area(imfusion_model, foreground_value=13, save_result=True,
                                                  save_path="../imfusion_files/3dModels/" + model_name_whole + "mask.imf")

        # Run a volume coverage evaluation
        vol_cov = volume_coverage(dims, volume_coverage_voxel_counts_path)
        print(str(vol_cov).replace(".", ","), end='  ||  ')

        # Run a confidence maps evaluation
        threshold = 0.95
        if material_type is "soft_tissue":
            threshold = 0.4
        confidence_matrix = get_confidence_matrix(dims, confidence_means_path)
        _, avg_confidence = confidence(confidence_matrix, threshold)
        print(str(avg_confidence).replace(".", ","), end='  ||  ')

        if material_type is not "water":
            # Run both intensity calculations [averaged, combined compounding]
            for i in range(2):
                combine_after_compounding = i is 0
                combined_imfusion_volume = add_imfusion_volumes(align_volumes([get_compounded_volume(item) for item in
                                                                               iteration_folders])) if combine_after_compounding else \
                    compound_from_different_folders([os.path.join(item, "sweeps") for item in iteration_folders])
                # resampling and positioning the compounding onto the map
                combined_imfusion_volume_aligned = align_volumes([combined_imfusion_volume, imfusion_model])[0]
                combined_volume_aligned = np.squeeze(combined_imfusion_volume_aligned)

                # save_imfusion_volume(combined_imfusion_volume_aligned, "../imfusion_files/combined_resampled_on_model.imf")

                # Run an intensity evaluation
                if combine_after_compounding:
                    avg_intensity = intensity("averaged compounding", combined_volume_aligned, occlusion_mask,
                                              len(iteration_folders_names))
                    print(str(avg_intensity).replace(".", ","))

                else:
                    comb_intensity = intensity("combined compounding", combined_volume_aligned, occlusion_mask, 1)
                    print(str(comb_intensity).replace(".", ","))
        else:
            print("       -         ||           -        ")


if __name__ == '__main__':
    imfusion.init()
    evaluation()  # Running the evaluation for all metrics.
