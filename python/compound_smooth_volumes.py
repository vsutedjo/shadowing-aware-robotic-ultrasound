import os
import numpy as np
import xml.etree.ElementTree as ET
import os
from functions.helpers.helpers import sort_transducer_poses
from functions.helpers.imfusion.imfusion_helpers import get_algorithm_properties_from_iws
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
import imfusion

imfusion.init()

sweep_path = ""


def get_pose_from_us_sweep(data_path):
    sweep = imfusion.open(data_path)[0]
    matrix = sweep.matrix(0)
    return matrix

def create_shared_image(template_data):
    data_array = np.array(template_data)
    data = imfusion.SharedImage(data_array)
    data.spacing = template_data.spacing
    return data


def interpolate_trajectories():
    return []


def create_single_sweep(data_dir, save_path):
    data_list = [os.path.join(data_dir, item) for item in os.listdir(data_dir)]

    shared_image_set = imfusion.SharedImageSet()
    tracking_stream = imfusion.TrackingStream()
    for data_path in data_list:
        us_sweep = imfusion.open(data_path)[0][0]
        tracking_stream.add(get_pose_from_us_sweep(data_path))
        shared_image = create_shared_image(us_sweep)
        shared_image_set.add(shared_image)

    shared_image_set.modality = imfusion.Data.Modality.ULTRASOUND

    imfusion.executeAlgorithm('IO;Tracking Stream', [tracking_stream],
                              imfusion.Properties(
                                  {
                                      'location': "tracking_stream.ts"}))

    tracked_sweep = imfusion.executeAlgorithm("Convert to Sweep", [shared_image_set],
                                           imfusion.Properties({'Tracking stream file':"tracking_stream.ts"}))

    imfusion.executeAlgorithm('IO;ImFusionFile', tracked_sweep,
                              imfusion.Properties({'location': save_path + "sweep.imf"}))


def get_simulation_splines(matrices, image_height, interpolated_points = 10):

    transducer_splines_list = []
    direction_splines_list = []
    for i in range(len(matrices) - 1):
        interpolated_timestamps = np.linspace(0, 1, interpolated_points)
        translations = np.array([matrices[i][0:3, -1], matrices[i+1][0:3, -1]])

        rotations = [R.from_matrix(matrices[i][0:3, 0:3]), R.from_matrix(matrices[i+1][0:3, 0:3])]
        x_rotations = np.array([item.as_euler('zyx', degrees=True)[-1] for item in rotations])

        timestamps = np.array([0, 1])

        f = interp1d(timestamps, translations, axis=0)
        interpolated_translations = f(interpolated_timestamps)

        f = interp1d(timestamps, x_rotations, axis=0)
        interpolated_x_rotations = f(interpolated_timestamps)

        str_tr = ""
        str_dir = ""
        for j in range(interpolated_points):
            # getting the matrix defining the new pose
            [xc, yc, zc] = interpolated_translations[j]
            T_us2world = np.eye(4)
            T_us2world[0:3, -1] = interpolated_translations[j]
            T_us2world[0:3, 0:3] = R.from_euler('x', interpolated_x_rotations[j], degrees=True).as_matrix()

            tr_spline_pt = np.matmul(T_us2world, np.array([0, -image_height/2, 0, 1]))
            dir_spline_pt = np.matmul(T_us2world, np.array([0, +image_height / 2, 0, 1]))

            str_tr += ' '.join(["{0:0.2f}".format(i) for i in tr_spline_pt[0:-1]]) + " "
            str_dir += ' '.join(["{0:0.2f}".format(i) for i in dir_spline_pt[0:-1]]) + " "

        transducer_splines_list.append(str_tr)
        direction_splines_list.append(str_dir)


    return transducer_splines_list, direction_splines_list


def simulated_interpolated_sweeps(us_matrices, transducer_width: float, us_height: float, model_name: str, material_type: str, interpolated_points):
    location, direction = get_simulation_splines(us_matrices, us_height, interpolated_points = interpolated_points)

    model_path = "imfusion_files/3dModels/" + model_name + ".imf"
    data = imfusion.open(model_path)

    hybrid_us_param = \
        get_algorithm_properties_from_iws("imfusion_files/ws_"+material_type+".iws", 'Hybrid Ultrasound Simulation')
    hybrid_us_param["probeWidth"] = transducer_width

    shared_image_set = imfusion.SharedImageSet()
    tracking_stream = imfusion.TrackingStream()
    for transducer_spline, direction_spline in zip(location, direction):

        hybrid_us_param["frameCount"] = interpolated_points
        hybrid_us_param['SplineTransducer/points'] = transducer_spline
        hybrid_us_param['SplineDirection/points'] = direction_spline

        simulated_sweep = imfusion.executeAlgorithm('Ultrasound;Hybrid Ultrasound Simulation', data, hybrid_us_param)[0]

        for j, us_image in enumerate(simulated_sweep):

            us_array = np.array(us_image)
            shared_image = imfusion.SharedImage(us_array)
            shared_image.spacing = us_image.spacing
            shared_image_set.add(shared_image)
            tracking_stream.add(simulated_sweep.matrix(j))

    shared_image_set.modality = imfusion.Data.Modality.ULTRASOUND

    return shared_image_set, tracking_stream


def get_interpolated_trajectory(matrices, interpolated_points = 10):

    timestamps = np.linspace(0, 1, len(matrices))
    translations = np.array([item[0:3, -1] for item in matrices])
    rotations = [R.from_matrix(item[0:3, 0:3]) for item in matrices]
    x_rotations = np.array([item.as_euler('zyx', degrees=True)[-1] for item in rotations])
    print(x_rotations)

    interpolated_timestamps = np.linspace(0, 1, interpolated_points)

    f = interp1d(timestamps, translations, axis=0)
    interpolated_translations = f(interpolated_timestamps)

    f = interp1d(timestamps, x_rotations, axis=0)
    interpolated_x_rotations = f(interpolated_timestamps)
    print(interpolated_x_rotations)

    for item in matrices:
        print(item)

    interpolated_matrices = []

    for i in range(interpolated_points):
        matrix = np.eye(4)
        matrix[0:3, -1] = interpolated_translations[i]
        matrix[0:3, 0:3] = R.from_euler('x', interpolated_x_rotations[i], degrees=True).as_matrix()
        interpolated_matrices.append(matrix)

    return interpolated_matrices


def get_sorted_files(data_path):

    prefix = os.listdir(data_path)[0].split("pose")[0]
    poses = [item.split("pose")[-1] for item in os.listdir(data_path)]
    poses = [item.replace(".imf", "") for item in poses]
    poses = [(int(item.split("_")[0]), int(item.split("_")[1]), int(item.split("_")[2])) for item in poses]

    sorted_poses = sort_transducer_poses(poses)

    sorted_poses_files = [os.path.join(data_path, prefix+ "pose" +
                                       str(item[0]) + "_" + str(item[1]) + "_" + str(item[2]) + ".imf") for item in sorted_poses]
    return sorted_poses_files


def main(data_path, save_path, model_name, material_type="water", interpolated_points = 10):

    sorted_files = get_sorted_files(data_path + 'sweeps')
    us_image = imfusion.open(sorted_files[0])[0][0]
    image_height = us_image.height*us_image.spacing[1]
    transducer_width = us_image.width*us_image.spacing[0]

    us_matrices = [imfusion.open(item)[0].matrix(0) for item in sorted_files]

    shared_image_set, tracking_stream = simulated_interpolated_sweeps(us_matrices, transducer_width, image_height,
                                                                      model_name, material_type, interpolated_points)

    imfusion.executeAlgorithm('IO;Tracking Stream', [tracking_stream],
                              imfusion.Properties(
                                  {
                                      'location': save_path + "/tracking_stream.ts"}))

    tracked_sweep = imfusion.executeAlgorithm("Convert to Sweep", [shared_image_set],
                                              imfusion.Properties({'Tracking stream file': save_path + "/tracking_stream.ts"}))

    imfusion.executeAlgorithm('IO;ImFusionFile', tracked_sweep,
                              imfusion.Properties({'location': save_path + "interpolated_sweep.imf"}))


main(data_path="imfusion_files/results/two_beams_nonparallel_water/3108_1043_perpendicular/",
     save_path="imfusion_files/results/two_beams_nonparallel_water/3108_1043_perpendicular/",
     model_name="two_beams_nonparallel_water",
     material_type="water")