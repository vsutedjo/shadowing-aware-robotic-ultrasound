import os
import numpy as np

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']

import imfusion


def save_tracked_image_as_volume(tracked_images):

    image = tracked_images.get_python_array()
    volume = np.stack([np.expand_dims(image, 0), np.expand_dims(image, 0)])

    spacing = image.spacing

    # frames x slices x height x width x channels
    volume_array = np.reshape(tracked_images, [1, 2, tracked_images.shape[1], tracked_images.shape[0], 1])

    volume_array = np.zeros([1, 2, 10, 10, 1])

    imageset = imfusion.SharedImageSet(volume_array, dtype='uint8')

    properties = imfusion.Properties({'spacing': [spacing[0], spacing[1], 0.5]})
    data = imfusion.executeAlgorithm('Set Spacing', [imageset[0]], properties)[0]

    print()


