import os
from scipy.spatial.transform import Rotation as R
import numpy as np
os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
import matplotlib.pyplot as plt

import imfusion
imfusion.init()


def main(P_vg, us_image_idx):
    """

    :param P_vg: The point expressed in the volume grid )
    :return:
    """

    # Convert to imfusion axes for convenience, as we work with imfusion data
    P_vg = [P_vg[2], P_vg[1], P_vg[0]]

    # 1. Loading the data
    imf_volume = imfusion.open("viviana_test/volume.imf")[0][0]
    us_sweep = imfusion.open("viviana_test/us_sweep.imf")[0]
    label_sweep = imfusion.open("viviana_test/sweep_label.imf")[0]

    us_image_idx = 4


    # 2.
    # For ultrasound sweep, imfusion saves the data to world matrix, that is the data (image) coordinate system
    # expressed in the world coordinate system, coinciding with the transformation to apply to points expressed in the
    # data coordinate system to express them in the world coordinate system.
    # For convenience, we also read the us image shape and spacing
    T_ig2world = us_sweep.matrix(us_image_idx)
    us_image = np.squeeze(np.array(us_sweep[us_image_idx]))
    us_shape = [us_sweep[us_image_idx].width, us_sweep[us_image_idx].height]  # shape in imfusion format
    us_spacing = us_sweep[us_image_idx].spacing

    # 3. Reading the pose matrix for the volume data.
    # For volume data, imfusion saves the world to data matrix, that is the world coordinate system
    # expressed in the data coordinate system, coinciding with the transformation to apply to points expressed in the
    # world coordinate system to express them in the data coordinate system.
    # Since we are interested in the transformation to be applied to points in the data coordinate system to express
    # them in the world coordinate system, we have to invert this matrix
    T_vg2w = np.linalg.inv(imf_volume.matrix)
    volume = np.squeeze(np.array(imf_volume))
    vol_spacing = imf_volume.spacing
    vol_shape = [imf_volume.width, imf_volume.height, imf_volume.slices] # shape in imfusion format

    # Given the voxel coordinates of a voxel in the volume, we have to get the physical coordinates of the point,
    # expressed with respect to the volume center
    P_vg_phys = [(P_vg[0] - int(vol_shape[0]/2)) * vol_spacing[0],
                 (P_vg[1] - int(vol_shape[1]/2)) * vol_spacing[1],
                 (P_vg[2] - int(vol_shape[2]/2)) * vol_spacing[2],
                 1]

    # Then, the coordinate of the image in physical space are given by:
    # inv(T_ig2w) * T_vg2w * P_vg_phys
    T_w2ig = np.linalg.inv(T_ig2world)
    P_image_phys = np.matmul(T_w2ig, np.matmul(T_vg2w, P_vg_phys))

    # To obtain the point in pixel coordinates we write:

    # Make sure that the point is in (or close enough) to the image plane
    print(P_image_phys[2])
    if abs(P_image_phys[2]) > 3:
        print("You're out of the image plane!")

    print(P_image_phys)

    # getting the point in python coordinates
    # P_pixel = center_position_in_pixel + P_image_phys_in_pixel
    P_image_pixel = [round( (us_shape[1]/2) + (P_image_phys[1] / us_spacing[1]) ),
                     round( (us_shape[0]/2) + (P_image_phys[0] / us_spacing[0]))]

    print(P_image_pixel)
    return P_image_pixel

# In ImFusion the voxel coordinates are: [43, 40, 121], where the axes 0, 1, 2 correspondes to x, y, z coordinates.
# Imfusion x, y, z (thus 0, 1, 2) axis correspond to python 2, 1, 0 axis, therefore
# P_vg = [121, 40, 43]

# Setting inputs (check what it corresponds to in python)
# Reading the pose matrix for the last frame in the sweep, which is the one indicated by the red arrow
#     # in the us_sweep.png file

p_1 = [121, 40, 43]
us_idx_1 = -1

p_2 = [95, 83, 73]
us_idx_2 = 4


main(p_2, us_idx_2)

