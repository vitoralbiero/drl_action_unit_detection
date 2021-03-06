from keras.utils.np_utils import to_categorical
import cv2
import logging
import numpy as np
import pandas as pd

INPUT_NAME = 'images'
POSE_OUTPUT_NAME = 'pose'
INPUT_INTENSITY_WEIGHTS = 'intensity_weights'
AU_OCCURRENCE_OUTPUT_NAME = 'au_occurrence'
AU_INTENSITY_OUTPUT_NAME = 'au_intensity'

N_AUS_INTENSITY = 7
N_AUS_OCCURRENCE = 10
N_POSES = 9


class DefaultDataLoader(object):

    batch_size = 0
    iterations_per_epoch = 0
    total_count = 0
    data_shape = None
    _indices = None
    _data = None
    _transformer = None

    def __init__(self, data_file_path, transformer, batch_size, data_shape):
        self._transformer = transformer
        self.batch_size = batch_size
        self.data_shape = data_shape
        self._data = pd.read_csv(data_file_path, delimiter=' ', header=None)

        self.total_count = len(self._data)
        self._indices = list(range(self.total_count))
        np.random.shuffle(self._indices)

        iterations_per_epoch = float(self.total_count) / float(batch_size)
        self.iterations_per_epoch = int(np.ceil(iterations_per_epoch))
        # self.iterations_per_epoch = 10

        self._data = self._data.iloc

    def batch_generator(self):
        index = -1
        while True:
            batch = self._empty_batch()
            for i in range(self.batch_size):
                index += 1
                index = index % self.total_count
                self._fill_batch_at(batch,
                                    batch_index=i,
                                    data_index=index)

            yield self._prepare_batch(batch)

    def _empty_batch(self):
        return [
            np.zeros(shape=(self.batch_size,) + self.data_shape),
            np.zeros(shape=(self.batch_size, N_POSES)),
            np.zeros(shape=(self.batch_size, N_AUS_OCCURRENCE)),
            np.zeros(shape=(self.batch_size, N_AUS_INTENSITY))
        ]

    def _prepare_batch(self, batch_data):
        inputs = {
            INPUT_NAME: batch_data[0]
        }
        outputs = {
            POSE_OUTPUT_NAME: batch_data[1],
            AU_OCCURRENCE_OUTPUT_NAME: batch_data[2],
            AU_INTENSITY_OUTPUT_NAME: batch_data[3]
        }
        return (inputs, outputs)

    def _fill_batch_at(self, batch_data, batch_index, data_index):
        index = self._indices[data_index]

        image_path = self._data[index, 0]
        image = cv2.imread(image_path)
        image = self._transformer.preprocess(image)
        batch_data[0][batch_index] = image

        pose = to_categorical(int(self._data[index, 1]), num_classes=9)
        batch_data[1][batch_index] = pose

        occurrence = [int(v) for v in self._data[index, 2:12].tolist()]
        batch_data[2][batch_index] = occurrence

        intensity = [float(v) for v in self._data[index, 12:19].tolist()]
        intensity = np.asarray(intensity)
        intensity[intensity == 9] = 0
        batch_data[3][batch_index] = intensity

    def shuffle_data(self):
        logging.info('Shuffling Network data indices.')
        np.random.shuffle(self._indices)
