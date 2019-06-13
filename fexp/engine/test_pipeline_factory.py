from .test_base_pipeline import TestBasePipeline
from .test_base_pipeline_lstm import TestBasePipelineLstm
import fexp.model.baseline as baseline
import fexp.model.roi as roi
import fexp.model.vggface as vggface
import fexp.model.lstm as lstm
from fexp.data import Transformer
import logging
from fexp.helper import bbox_transfomer


class TestPipelineFactory(object):

    def create(self, pipeline_name, weights, testing_data_path,
               cache_folder, database_name, output_directory,
               landmarks_path, extract_features):
        pipeline_name = pipeline_name.lower()
        logging.info('Creating pipeline {}.'.format(pipeline_name))

        if pipeline_name == 'drl':
            return self._drl(weights, testing_data_path,
                             cache_folder, database_name,
                             landmarks_path, output_directory,
                             extract_features)

        elif pipeline_name == 'roi_aligned':
            return self._roi_aligned(weights, testing_data_path,
                                     cache_folder, database_name,
                                     landmarks_path, output_directory)

        elif pipeline_name == 'vggface':
            return self._vggface(weights, testing_data_path,
                                 cache_folder, database_name,
                                 output_directory)

        elif pipeline_name == 'lstm':
            return self._lstm(weights, testing_data_path, output_directory)

        raise ValueError('Unknown pipeline "{}".'.format(pipeline_name))

    def _baseline(self, weights, testing_data_path,
                  cache_folder, database_name, output_directory):
        data_shape = (132, 132, 3)
        model = baseline.model(data_shape)
        transformer = Transformer()
        transformer.set_mean(128.0)
        transformer.set_scale(1 / 128.0)

        return TestBasePipeline(model,
                                weights,
                                data_shape,
                                testing_data_path,
                                database_name,
                                cache_folder,
                                output_directory,
                                transformer,
                                bbox_transfomer.squared_center_bbox)

    def _roi(self, weights, testing_data_path,
             cache_folder, database_name,
             landmarks_path, output_directory, extract_features):
        data_shape = (224, 224, 3)
        model = roi.model(data_shape, None)
        transformer = Transformer()
        transformer.set_mean(128.0)
        transformer.set_scale(1 / 128.0)

        return TestBasePipeline(model,
                                weights,
                                data_shape,
                                testing_data_path,
                                database_name,
                                cache_folder,
                                output_directory,
                                transformer,
                                bbox_transfomer.squared_center_bbox,
                                landmarks_path,
                                extract_features)

    def _roi_aligned(self, weights, testing_data_path,
                     cache_folder, database_name,
                     landmarks_path, output_directory):
        data_shape = (224, 224, 3)
        model = roi.model(data_shape, None)
        transformer = Transformer()
        transformer.set_mean(0)
        transformer.set_scale(1)

        return TestBasePipeline(model,
                                weights,
                                data_shape,
                                testing_data_path,
                                database_name,
                                cache_folder,
                                output_directory,
                                transformer,
                                bbox_transfomer.squared_center_bbox,
                                landmarks_path,
                                True)

    def _vggface(self, weights, testing_data_path,
                 cache_folder, database_name, output_directory):
        data_shape = (224, 224, 3)
        model = vggface.model(data_shape, None)
        transformer = Transformer()
        transformer.set_mean(128.0)
        transformer.set_scale(1 / 128.0)

        return TestBasePipeline(model,
                                weights,
                                data_shape,
                                testing_data_path,
                                database_name,
                                cache_folder,
                                output_directory,
                                transformer,
                                bbox_transfomer.squared_center_bbox)

    def _lstm(self, weights, testing_data_path, output_directory):
        num_sequences = 500
        num_features = 2000
        model = lstm.model(num_sequences, num_features)

        return TestBasePipelineLstm(model,
                                    weights,
                                    num_sequences,
                                    testing_data_path,
                                    output_directory)
