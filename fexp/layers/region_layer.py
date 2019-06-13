from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import itertools


class RegionLayer(object):

    _regions = None

    def __init__(self):
        pass

    def split(self, layer, n_cols, n_rows):
        self._n_rows = int(n_rows)
        self._n_cols = int(n_cols)
        layer_shape = layer.get_shape().as_list()

        if len(layer_shape) != 4:
            raise ValueError('Input tensor must be 4D.')

        (height, width) = layer_shape[1:3]

        if height % self._n_rows > 0:
            raise ValueError(
                'Invalid image height size for the number of rows.')

        if width % self._n_cols > 0:
            raise ValueError(
                'Invalid image widht size for the number of collumns.')

        # (region_height, region_width) = height
        region_height = height // self._n_rows
        region_width = width // self._n_cols

        self._regions = []
        indices = itertools.product(range(n_rows), range(n_cols))

        for (i, j) in indices:
            i_begin = i * region_height
            j_begin = j * region_width
            rectangle = (i_begin, j_begin, region_width, region_height)
            region = Lambda(self._crop_region(rectangle))(layer)
            self._regions.append(region)

    def _crop_region(self, rectangle):
        (top, left, width, height) = rectangle
        bottom = top + height
        right = left + width
        return lambda x: x[:, top:bottom, left:right, :]

    def concatenate_convolution(self):
        rows = []
        for i in range(self._n_rows):
            row = []
            for j in range(self._n_cols):
                index = i * self._n_cols + j
                row.append(self._regions[index])
            rows.append(Concatenate(axis=2)(row))

        return Concatenate(axis=1)(rows)

    def concatenate_fully_connected(self):
        regions = []
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                index = i * self._n_cols + j
                regions.append(self._regions[index])
        return Concatenate(axis=1)(regions)

    def add(self, operation):
        for (i, region) in enumerate(self._regions):
            self._regions[i] = operation(region)
