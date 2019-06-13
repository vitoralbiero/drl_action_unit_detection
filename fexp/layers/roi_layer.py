from keras.layers.merge import Concatenate
from .roi_crop_layer import RoiCropLayer


class RoiLayer(object):

    _regions = None

    def __init__(self):
        pass

    def split(self, layer, landmarks,
              original_size, region_height, region_width):
        layer_shape = layer.get_shape().as_list()

        if len(layer_shape) != 4:
            raise ValueError('Input tensor must be 4D.')

        self._landmarks = landmarks

        (height, width) = layer_shape[1:3]

        width_scale = float(width) / float(original_size[1])
        height_scale = float(height) / float(original_size[0])

        self._regions = []

        for landmark_num in range(self._landmarks.shape[1]):
            region = RoiCropLayer(region_height,
                                  region_width,
                                  height_scale,
                                  width_scale,
                                  landmark_num)([layer, landmarks])

            self._regions.append(region)

    def concatenate_fully_connected(self):
        regions = []
        for i in range(self._landmarks.shape[1]):
            regions.append(self._regions[i])

        return Concatenate(axis=1)(regions)

    def add(self, operation):
        for (i, region) in enumerate(self._regions):
            self._regions[i] = operation(region, i)
