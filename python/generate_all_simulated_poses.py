import json
import math
import os
import xml.etree.ElementTree as ET
from shutil import copy

import numpy as np
from matplotlib import pyplot as plt, gridspec

from functions.helpers.helpers import prepare_results_folders
from functions.helpers.imfusion.imfusion_helpers import read_roi_from_json, splines_from_poses, get_probe_width

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def get_all_possible_poses(nr_rows, nr_cols, angles):

    poses = []
    for angle in angles:
        for col in range(nr_cols):
            for row in range(nr_rows):
                poses.append((angle, row, col))

    return poses


def run_us_sweep_for_pose(poses, dims: (int, int, int), transducer_width: float,
                          model_name: str, batch_file_path: str, out_folder: str):
    nr_rows, nr_cols, nr_planes = dims
    roi = read_roi_from_json(model_name)

    print("roi: ", roi)
    transducer_width = str(get_probe_width(nr_rows,
                                     transducer_width, roi))

    batch_file = open(batch_file_path, "w")

    batch_file.write("INPUDIRECTION;INPUTTRANSDUCER;OUTPUTFILE;PROBEWIDTH")

    for i, pose in enumerate(poses):
        # Find out if we need to run the sweep
        poses = []
        original_pose = pose

        # Add a small area around the sweep
        for i in range(4):
            poses.append((original_pose[0], original_pose[1], original_pose[2] + i * 0.01))

        location, direction = splines_from_poses(poses, (nr_rows, nr_cols), roi)
        outfile = out_folder + "simulated_sweep_dims" + str(nr_rows) + str(nr_cols) + str(nr_planes) + "_pose" + \
                  str(pose[0]) + "_" + str(pose[1]) + "_" + str(pose[2]) + ".imf"

        if pose == (0, 0, 8):
            print(direction + ";" + location + ";" + outfile + ";" + transducer_width)

        if os.path.exists("imfusion_files/" + outfile):
            continue


        batch_file.write("\n" + direction + ";" + location + ";" + outfile + ";" + transducer_width)

    batch_file.close()


def main():

    model_name = "two_beams_nonparallel"
    material_type = "soft_tissue"
    nr_rows = 8
    nr_cols = 10
    nr_planes = 6
    angles = [-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]  # Write the angles from smallest to
    # angles = [0]

    # nr_rows = 3
    # nr_cols = 4
    # nr_planes = 6
    # angles = [0, 5]  # Write the angles from smallest to
    transducer_width = 1

    model_name = model_name + "_" + material_type
    poses = get_all_possible_poses(nr_rows, nr_cols, angles)
    dims = (nr_rows, nr_cols, nr_planes)

    prepare_results_folders(model_name, None)


    out_folder = 'results/' + model_name + "/general/sweeps/"

    run_us_sweep_for_pose(poses, dims, transducer_width, model_name, "imfusion_files/batch.txt", out_folder)

main()
