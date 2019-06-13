import cv2
import logging
import numpy as np
import pandas as pd
from os import path
from fexp.helper import rectangle
import csv

INPUT_IMAGE = 'input_1'
INPUT_LANDMARKS = 'landmarks'
AU_OCCURRENCE_OUTPUT_NAME = 'au_occurrence'

N_AUS_OCCURRENCE = 10
N_LANDMARKS = (20, 2)

# DEFAULT
LABEL_LANDMARKS = [18, 21, 37, 38, 41, 22, 25, 43, 44, 46,
                   50, 51, 52, 61, 63, 48, 58, 57, 56, 54]


class RoiLoader(object):

    batch_size = 0
    iterations_per_epoch = 0
    total_count = 0
    data_shape = None
    _indices = None
    _data = None
    _transformer = None

    def __init__(self, data_file_path, transformer,
                 bbox_file, landmarks_path,
                 batch_size, data_shape):
        self._transformer = transformer
        self.batch_size = batch_size
        self.data_shape = data_shape
        self._data = pd.read_csv(data_file_path, delimiter=' ', header=None)
        self.landmarks_path = landmarks_path

        self._bboxes = {}
        with open(bbox_file, 'r+') as file:
            for line in file:
                line = line.split(',')
                key = line[0]
                values = [np.float64(v) for v in line[1:]]
                self._bboxes[key] = np.asarray(values)

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
            np.zeros(shape=(self.batch_size,) + N_LANDMARKS),
            np.zeros(shape=(self.batch_size, N_AUS_OCCURRENCE))
        ]

    def _prepare_batch(self, batch_data):
        inputs = {
            INPUT_IMAGE: batch_data[0],
            INPUT_LANDMARKS: batch_data[1]
        }
        outputs = {
            AU_OCCURRENCE_OUTPUT_NAME: batch_data[2]
        }
        return (inputs, outputs)

    def _fill_batch_at(self, batch_data, batch_index, data_index):
        index = self._indices[data_index]

        image_path = self._data[index, 0]
        image = cv2.imread(image_path)
        image = self._transformer.preprocess(image)
        batch_data[0][batch_index] = image
        batch_data[1][batch_index] = self.adjust_landmarks(image_path)

        occurrence = [int(v) for v in self._data[index, 2:12].tolist()]
        batch_data[2][batch_index] = occurrence

    def adjust_landmarks(self, image_path):
        image_name = path.basename(path.splitext(image_path)[0])

        bbox = rectangle.bbox_to_square_center(self._bboxes[image_name])
        left = bbox[0]
        top = bbox[1]
        size = bbox[2]
        scale = float(self.data_shape[0]) / float(size)

        landmarks_file = path.join(self.landmarks_path,
                                   image_name + '_landmarks.csv')

        raw_landmarks = []
        with open(landmarks_file, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                raw_landmarks.append(row)

        landmarks = np.asarray(raw_landmarks).astype('float64')

        # deslocation
        landmarks[41, 0] = np.mean(landmarks[[41, 48], 0])
        landmarks[41, 1] += (landmarks[48, 1] - landmarks[41, 1]) * 0.66

        landmarks[46, 0] = np.mean(landmarks[[46, 54], 0])
        landmarks[46, 1] += (landmarks[54, 1] - landmarks[46, 1]) * 0.66

        landmarks[58, :] = np.mean(landmarks[[58, 7], :], axis=0)
        landmarks[56, :] = np.mean(landmarks[[56, 9], :], axis=0)

        landmarks = landmarks[LABEL_LANDMARKS, 0:2]

        # scaling
        landmarks[:, 0] -= left
        landmarks[:, 1] -= top
        landmarks *= scale

        return landmarks

    def shuffle_data(self):
        logging.info('Shuffling Network data indices.')
        np.random.shuffle(self._indices)
