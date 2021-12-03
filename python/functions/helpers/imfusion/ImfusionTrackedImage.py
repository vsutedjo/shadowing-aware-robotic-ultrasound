import os
import numpy as np

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


class ImfusionTrackedImage:
    def __init__(self, imfusion_shared_image_set, matrix):

        # todo: add check on data types

        self.imf_image_set = imfusion_shared_image_set  # imfusion data
        self.matrix = matrix  # the matrix expressed in imfusion axes

        # todo: maybe add function to get pose paramters in python coordinates from the matrix as a check

    def get_spacing(self):
        """
        :return: The spacing in imfusion axes
        """
        return self.imf_image_set[0].spacing

    def get_shape(self):
        """
        :return: The shape in imfusion axes
        """
        return [self.imf_image_set[0].width, self.imf_image_set[0].height]

    def get_python_array(self):
        """
        returns the image array in python axes (transposed compared to imfusion)
        :return:
        """

        return np.squeeze(np.array(self.imf_image_set[0]))

    def resize(self, new_size):

        if len(new_size) == 2:
            new_size.extend([1])

        properties = imfusion.Properties({'resamplingMode': 0,
                                          'targetDimensions': new_size})
        self.imf_image_set = imfusion.executeAlgorithm('Image Resampling', [self.imf_image_set], properties)[0]




