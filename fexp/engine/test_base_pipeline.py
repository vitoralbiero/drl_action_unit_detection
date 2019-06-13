from __future__ import print_function
import numpy as np
import cv2
import itertools
from fexp.cache import CacheFactory
from fexp.detector import CachedFaceDetector
from os import listdir, makedirs, path
from timeit import default_timer as timer
from fexp.helper import rectangle
import csv

OCCURRENCE_AUS = [1, 4, 6, 7, 10, 12, 14, 15, 17, 23]
LABEL_LANDMARKS = [18, 21, 37, 38, 41, 22, 25, 43, 44, 46,
                   50, 51, 52, 61, 63, 48, 58, 57, 56, 54]

LEFT_EYES = [36, 37, 38, 39, 40, 41]
RIGHT_EYES = [42, 43, 44, 45, 46, 47]


class TestBasePipeline(object):

    _network_model = None
    _model = None
    _model_training_data = ''
    _model_validation_data = ''
    _output_directory = ''
    _epochs = 0

    def __init__(self, model, weights, data_shape, test_data_path,
                 database_name, cache_folder, output_directory,
                 transformer, bbox_transfomer, landmarks_path=None,
                 extract_features=False):

        self._model = model
        self._weights = weights
        self._test_data_path = test_data_path
        self._cache_folder = cache_folder
        self._output_directory = output_directory
        self._database_name = database_name
        self._landmarks_path = landmarks_path
        self._data_shape = data_shape
        self._size = max(self._data_shape[0], self._data_shape[1])
        self._transformer = transformer
        self._bbox_transfomer = bbox_transfomer
        self._align = False
        self._extract_features = extract_features

    def init_cache(self):
        if self._cache_folder is None:
            return CacheFactory('', no_cache=True)

        database = self._database_name.lower()
        database_cache_directory = path.join(self._cache_folder, database)

        database_cache_factory = CacheFactory(database_cache_directory,
                                              no_cache=False)

        if not path.exists(database_cache_directory):
            makedirs(database_cache_directory)

        return database_cache_factory

    def video_has_score(self, occurrence_folder, video_name):
        if self._extract_features:
            occurrence_file = self.feature_file_path_full(occurrence_folder,
                                                          video_name)
        else:
            occurrence_file = self.occurrence_file_path(occurrence_folder,
                                                        video_name)
        if not path.exists(occurrence_file):
            return False

        return True

    def concatenate_results(self, values, new_values, index):
        new_values = np.append(index, new_values)
        if values is None:
            return new_values

        return np.vstack((values, new_values))

    def save_features(self, output_directory, video_name, features):
        output_file_path = self.feature_file_path(output_directory,
                                                  video_name)

        np.save(output_file_path, features)

    def save_occurrences(self, output_directory, video_name, aus_occurrences):
        header = np.array([0] + OCCURRENCE_AUS)
        aus_occurrences = np.vstack((header, aus_occurrences))

        output_file_path = self.occurrence_file_path(output_directory,
                                                     video_name)

        np.savetxt(output_file_path, aus_occurrences, fmt='%d', delimiter=',')

    def occurrence_file_path(self, output_directory, video_name):
        output_file = path.splitext(video_name)[0] + '.csv'
        return path.join(output_directory, output_file)

    def feature_file_path(self, output_directory, video_name):
        output_file = path.splitext(video_name)[0]
        return path.join(output_directory, output_file)

    def feature_file_path_full(self, output_directory, video_name):
        output_file = path.splitext(video_name)[0] + '.npy'
        return path.join(output_directory, output_file)

    def load_landmarks(self, landmark_name):
        landmarks_file = path.join(self._landmarks_path,
                                   landmark_name + '_landmarks.csv')

        if not path.exists(landmarks_file):
            return None

        raw_landmarks = []
        with open(landmarks_file, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                raw_landmarks.append(row)

        landmarks = np.asarray(raw_landmarks).astype('float64')

        return landmarks

    def adjust_landmarks(self, image_name, bbox, data_shape):
        bbox = rectangle.bbox_to_square_center(bbox)
        left = bbox[0]
        top = bbox[1]
        size = bbox[2]
        scale = float(data_shape[0]) / float(size)

        landmarks = self.load_landmarks(image_name)

        if landmarks is None:
            return None

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

    def run(self):
        self._model.load_weights(self._weights)

        if self._extract_features:
            self._model.layers.pop()
            self._model.outputs = [self._model.layers[-1].output]
            self._model.output_layers = [self._model.layers[-1]]
            self._model.layers[-1].outbound_nodes = []

        self._model.summary()

        occurrence_output_folder = path.join(
            self._output_directory, 'occurrence')

        if not path.exists(occurrence_output_folder):
            makedirs(occurrence_output_folder)

        database_cache_factory = self.init_cache()

        face_detector = None
        face_detector = CachedFaceDetector(face_detector, database_cache_factory)

        for video_name in listdir(self._test_data_path):

            if self.video_has_score(occurrence_output_folder,
                                    video_name):
                print('Video {0} already has score. Skipping.'.
                      format(video_name))
                continue

            print(video_name)
            video_path = path.join(self._test_data_path, video_name)
            video = cv2.VideoCapture(video_path)

            aus_occurrence = None
            features = []

            start = timer()
            for i in itertools.count():
                (success, frame) = video.read()
                if not success:
                    break

                frame_id = path.splitext(video_name)[0] + '_{}'.format(i)

                face_bbox = face_detector.detect_face(frame, frame_id)

                face = self._bbox_transfomer(face_bbox, frame, self._size)

                if self._landmarks_path is not None:
                    new_landmarks = self.adjust_landmarks(frame_id,
                                                          face_bbox,
                                                          face.shape)
                face = self._transformer.preprocess(face)

                if self._landmarks_path is not None:

                    if new_landmarks is not None:
                        landmarks = new_landmarks

                    if landmarks.ndim == 2:
                        landmarks = landmarks[np.newaxis, :, :]

                if face.ndim == 3:
                    face = face[np.newaxis, :, :, :]

                if self._landmarks_path is not None:
                    predictions = self._model.predict([face, landmarks])
                else:
                    predictions = self._model.predict(face)

                if self._extract_features:
                    features.append(predictions)
                else:
                    occurrences = np.round(predictions[0])
                    aus_occurrence = self.concatenate_results(aus_occurrence,
                                                              occurrences,
                                                              i + 1)

                print(i, end='\r')

            end = timer()
            print('{} ~{}s for {} frames'.format(
                video_name, end - start, i + 1))

            video.release()

            if self._extract_features:
                self.save_features(occurrence_output_folder,
                                   video_name, features)
            else:
                self.save_occurrences(occurrence_output_folder,
                                      video_name, aus_occurrence)
