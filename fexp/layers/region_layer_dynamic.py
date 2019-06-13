from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import itertools
from keras import backend as K


class RegionLayerDynamic(object):
    '''
    Different from the RegionLayer that uses the layer shape to do the
    operations, this layer uses the tensor shape for that.
     '''

    _regions = None

    def __init__(self):
        pass

    def _error(self):
        raise ValueError('Invalid image height size for the number of rows.')

    def _pass(self):
        pass

    def split(self, layer, n_cols, n_rows):
        layer_shape = K.shape(layer)
        height, width = layer_shape[1], layer_shape[2]

        self._n_rows = int(n_rows)
        self._n_cols = int(n_cols)

        if layer_shape[3] is None:
            raise ValueError('Input tensor must be 4D.')

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

    def concatenate(self):
        rows = []
        for i in range(self._n_rows):
            row = []
            for j in range(self._n_cols):
                index = i * self._n_cols + j
                row.append(self._regions[index])
            rows.append(Concatenate(axis=2)(row))

        return Concatenate(axis=1)(rows)

    def add(self, operation):
        for (i, region) in enumerate(self._regions):
            self._regions[i] = operation(region)
