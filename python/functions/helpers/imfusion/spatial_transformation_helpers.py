import os
import numpy as np


os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']


import imfusion


def get_pixels_inside_voxel(P_voxel, volume_spacing, volume_size_in_voxel, T_vol2world, image_spacing,
                            image_size_in_pixels, T_img2world):
    """
    The function detects the pixels of an tracked image that "falls" inside a certain volume voxel
    :param P_voxel: The coordinate of the voxel expressed in voxel wrt to the coordinate system centered in the
        top-left corner of the volume.
    :param volume_spacing: The voxel spacing of the volume
    :param volume_size_in_voxel: The volume size in voxel
    :param T_vol2world: The matrix mapping the image coordinate system in the world coordinate system, where the image
        coordinate system has origin in the center of the image
    :param image_spacing: The pixel spacing of the image
    :param image_size_in_pixels: The image size in pixels
    :param T_img2world: The matrix mapping the image coordinate system in the world coordinate system with origin
        in the center of the image
    """

    assert len(volume_spacing) == 3 and len(volume_size_in_voxel) == 3 and len(P_voxel) == 3

    # Point expressed in physical coordinates wrt to top left corner of the volume
    P_phys_vol_apex = [P_voxel[i] * volume_spacing[i] for i in range(3)]

    # # Adding half voxel, assuming that we consider voxel centers - This is only needed for very large voxels,
    # # otherwise is negligible
    P_phys_vol_apex = [P_phys_vol_apex[i] + volume_spacing[i]/2 for i in range(3)]

    volume_size_physical = [volume_size_in_voxel[i] * volume_spacing[i] for i in range(3)]

    # Point expressed in physical coordinates wrt to the volume center
    P_phys_vol_center = [P_phys_vol_apex[i] - volume_size_physical[i]/2 for i in range(3)]

    P_phys_vol_center.extend([1])
    P_phys_vol_center = np.array(P_phys_vol_center)

    P_phys_world = np.matmul(T_vol2world, P_phys_vol_center)

    distance_threshold = volume_spacing/2

    proximal_pixels = []

    for x in range(image_size_in_pixels[0]):
        for y in range(image_size_in_pixels[1]):

            Q_phys_img_apex = [x * image_spacing[0], y * image_spacing[1]]
            Q_phys_img_apex = [Q_phys_img_apex[i] + image_spacing[i] / 2 for i in range(2)]

            image_size_physical = [image_size_in_pixels[i] * image_spacing[i] for i in range(2)]
            Q_phys_img_center = [Q_phys_img_apex[i] - image_size_physical[i] / 2 for i in range(2)]

            Q_phys_img_center.extend([0, 1])  # expressing the point in homogeneous coordinates
            Q_phys_img_center = np.array(Q_phys_img_center)
            Q_phys_world = np.matmul(T_img2world, Q_phys_img_center)[0:3]

            if abs(P_phys_world[0] - Q_phys_world[0]) < distance_threshold[0] and \
                    abs(P_phys_world[1] - Q_phys_world[1]) < distance_threshold[1] and \
                    abs(P_phys_world[2] - Q_phys_world[2]) < distance_threshold[2]:
                proximal_pixels.append((x, y))

    return proximal_pixels


def get_image_pixel_from_voxel(P_voxel, volume_spacing, volume_size_in_voxel, T_vol2world, image_spacing,
                               image_size_in_pixels, T_img2world):
    """
    The function computes the pixel coordinate of an image, given the voxel coordinate of a volume, assuming the
    volume intersects the image at that voxel.

    IT IS CRUCIAL, that all the parameters are given considering consistent axes orientation. That is,
    if you decide to use the imfusion axes convention, the transformation matrices as well as the voxel coordinates,
    spacings and sizes must be given according to this convention, if you decide to use the python axes convention,
    the transformation matrices as well as the voxel coordinates, spacings and sizes must be given according to this
    convention.
    :param P_voxel: The coordinate of the voxel expressed in voxel wrt to the coordinate system centered in the
        top-left corner of the volume.
    :param volume_spacing: The voxel spacing of the volume
    :param volume_size_in_voxel: The volume size in voxel
    :param T_vol2world: The matrix mapping the image coordinate system in the world coordinate system, where the image
        coordinate system has origin in the center of the image
    :param image_spacing: The pixel spacing of the image
    :param image_size_in_pixels: The image size in pixels
    :param T_img2world: The matrix mapping the image coordinate system in the world coordinate system with origin
        in the center of the image
    :return:
    """

    assert len(volume_spacing) == 3 and len(volume_size_in_voxel) == 3 and len(P_voxel) == 3

    # Point expressed in physical coordinates wrt to top left corner of the volume
    P_phys_vol_apex = [P_voxel[i] * volume_spacing[i] for i in range(3)]

    # # Adding half voxel, assuming that we consider voxel centers - This is only needed for very large voxels,
    # # otherwise is negligible
    #P_phys_vol_apex = [P_phys_vol_apex[i] + volume_spacing[i]/2 for i in range(3)]

    volume_size_physical = [volume_size_in_voxel[i] * volume_spacing[i] for i in range(3)]

    # Point expressed in physical coordinates wrt to the volume center
    P_phys_vol_center = [P_phys_vol_apex[i] - volume_size_physical[i]/2 for i in range(3)]
    # print("phys. vol center:",P_phys_vol_center)

    # Then, the coordinate of the image in physical space are given by:
    # inv(T_ig2w) * T_vg2w * P_vg_phys
    T_world2img = np.linalg.inv(T_img2world)
    P_phys_vol_center.extend([1])  # expressing the point in homogeneous coordinates
    P_phys_vol_center = np.array(P_phys_vol_center)
    P_phys_img_center = np.matmul(T_world2img, np.matmul(T_vol2world, P_phys_vol_center))

    # To obtain the point in pixel coordinates we write:

    # Make sure that the point is in (or close enough) to the image plane
    if abs(P_phys_img_center[2]) > 3:
        print("You're out of the image plane!")

    print("Distance from the image plane", P_phys_img_center[2])

    # getting the point expressed wrt the image top-left corner
    image_physical_size = [image_size_in_pixels[i] * image_spacing[i] for i in range(2)]
    P_phys_img_apex = [P_phys_img_center[0] + image_physical_size[0]/2,
                       P_phys_img_center[1] + image_physical_size[1] / 2,
                       P_phys_img_center[2]]

    # P_pixel = center_position_in_pixel + P_image_phys_in_pixel
    P_image_pixel = [round(P_phys_img_apex[0]/image_spacing[0]),
                     round(P_phys_img_apex[1]/image_spacing[1])]

    return P_image_pixel
