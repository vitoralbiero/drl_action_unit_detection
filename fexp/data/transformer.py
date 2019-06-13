import cv2
import numpy as np
from keras_vggface import utils


class Transformer(object):

    _mean = 0.0
    _scale = 1.0
    _new_image_size = None
    _size_of_minimum_side = None
    _grayscale = False

    def __init__(self, new_image_size=None, grayscale=False):
        if new_image_size:
            if type(new_image_size) == int:
                self._size_of_minimum_side = new_image_size
            elif len(new_image_size) != 2:
                raise ValueError('The argument "size" should have 2 values.')
            else:
                self._new_image_size = new_image_size

        self._grayscale = grayscale

    def set_mean(self, mean):
        self._mean = mean

    def set_scale(self, scale):
        self._scale = scale

    def preprocess(self, image):
        new_size = self._new_image_size
        if new_size and new_size != tuple(image.shape[:2]):
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        new_size = self._size_of_minimum_side
        if new_size and new_size != np.min(image.shape[:2]):
            new_size = self._compute_size(new_size, image.shape[:2])
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        if self._grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, :, np.newaxis]

        image = np.float32(image)
        image = (image - self._mean) * self._scale

        # if self._mean == 0 and self._scale == 1:
        #    image = utils.preprocess_input(image, version=1)

        return image

    def _compute_size(self, size_of_minimum_side, shape):
        (height, width) = shape
        height = float(height)
        width = float(width)
        if height < width:
            width = width * (size_of_minimum_side / height)
            height = size_of_minimum_side
        else:
            height = height * (size_of_minimum_side / width)
            width = size_of_minimum_side

        return (int(width), int(height))
