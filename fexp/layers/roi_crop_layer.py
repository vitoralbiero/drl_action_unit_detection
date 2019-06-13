from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import Lambda


class RoiCropLayer(Layer):

    def __init__(self, region_height=1, region_width=1,
                 height_scale=1, width_scale=1, landmark_num=1, **kwargs):
        self._region_width = region_width
        self._region_height = region_height
        self._landmark_num = landmark_num
        self._width_scale = width_scale
        self._height_scale = height_scale

        super(RoiCropLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RoiCropLayer, self).build(input_shape)

    def call(self, inputs):

        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('RoiCropLayer must be called on a list of ' +
                            'tensors: (layer, landmarks)')

        landmarks = self._adjust_landmarks(inputs[1], inputs[0].shape)

        outputs = K.tf.map_fn(self._crop_region,
                              elems=[inputs[0], landmarks],
                              parallel_iterations=256)
        return outputs[0]

    def _adjust_landmarks(self, landmarks, shape):
        lefts, tops = K.tf.split(landmarks, 2, axis=2)
        lefts = Lambda(lambda x: x * self._width_scale)(lefts)
        tops = Lambda(lambda x: x * self._height_scale)(tops)

        tops = K.tf.cast(tops, K.tf.int32) - (self._region_height // 2)
        tops = K.tf.clip_by_value(tops, 0, shape[1] - self._region_height)

        lefts = K.tf.cast(lefts, K.tf.int32) - (self._region_width // 2)
        lefts = K.tf.clip_by_value(lefts, 0, shape[2] - self._region_width)

        return K.tf.concat([lefts, tops], axis=2)

    def _crop_region(self, inputs):
        top = inputs[1][self._landmark_num, 1]
        bottom = top + self._region_height
        left = inputs[1][self._landmark_num, 0]
        right = left + self._region_width

        region = inputs[0][top:bottom, left:right, :]

        return [region, inputs[1]]

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self._region_height,
                self._region_width, input_shape[0][3])
