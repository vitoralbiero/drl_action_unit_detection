from .training import Training
from .validation import Validation
import fexp.model.aumpnet as aumpnet
import fexp.model.vggface as vggface
import fexp.model.drl as drl
from fexp.data import Transformer
from fexp.data import DefaultDataLoader
from fexp.data import BaselineLoader
from fexp.data import DRLLoader
from fexp.model import metrics as validation_metrics
from fexp.tensorboard import TensorboardWriter
import logging
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from os import path


class EngineRunner(object):

    def run(self, engine_name, training_data_path, validation_data_path,
            output_directory, tensorboard_log_directory,
            bbox_path, train_landmarks_path, valid_landmarks_path,
            weights):

        engine_name = engine_name.lower()
        logging.info('Creating engine {}.'.format(engine_name))

        if engine_name == 'aumpnet':
            self._run_aumpnet(training_data_path,
                              validation_data_path,
                              output_directory,
                              tensorboard_log_directory)

        elif engine_name == 'drl':
            self._run_drl(training_data_path,
                          validation_data_path,
                          output_directory,
                          tensorboard_log_directory,
                          bbox_path,
                          train_landmarks_path,
                          valid_landmarks_path,
                          weights)

        elif engine_name == 'vggface':
            self._run_vggface(training_data_path,
                              validation_data_path,
                              output_directory,
                              tensorboard_log_directory)

        else:
            raise ValueError('Unknown engine "{}".'.format(engine_name))

    def _run_drl(self, training_data_path, validation_data_path,
                 output_directory, tensorboard_log_directory,
                 bbox_path, train_landmarks_path, valid_landmarks_path,
                 weights):

        if tensorboard_log_directory is None:
            tensorboard_log_directory = path.join(output_directory,
                                                  'tensorboard_logs')
        writer = TensorboardWriter(tensorboard_log_directory)

        data_shape = (224, 224, 3)

        if weights is not None:
            model = drl.model(data_shape, None)
        else:
            model = drl.model(data_shape)

        transformer = Transformer((224, 224))
        transformer.set_mean(128.0)
        transformer.set_scale(1.0 / 128.0)
        epochs = 30
        batch_size = 64

        if weights is not None:
            model.load_weights(weights)

        drl.compile(model, writer)

        # log hyper parameters to tensorboard
        writer.log_hyper_param('batch_size', batch_size)
        writer.log_hyper_param('data_shape',
                               '{0}x{1}x{2}'.format(data_shape[0],
                                                    data_shape[1],
                                                    data_shape[2]))

        model_training_data = DRLLoader(training_data_path,
                                        transformer,
                                        bbox_path,
                                        train_landmarks_path,
                                        batch_size=batch_size,
                                        data_shape=data_shape)

        model_validation_data = DRLLoader(validation_data_path,
                                          transformer,
                                          bbox_path,
                                          valid_landmarks_path,
                                          batch_size=90,
                                          data_shape=data_shape)

        engine = Training(model_training_data, model_validation_data,
                          output_directory, epochs, writer)

        metrics = {'au_occurrence': validation_metrics.f1_score}

        engine = Validation(engine, model_validation_data, metrics, writer)

        callbacks = [self._tensorboard_callback(tensorboard_log_directory),
                     ReduceLROnPlateau(monitor='val_loss',
                                       patience=2,
                                       epsilon=0.1,
                                       verbose=1,
                                       mode='min')]
        engine.run(model, callbacks)

    def _run_aumpnet(self, training_data_path, validation_data_path,
                     output_directory, tensorboard_log_directory):
        if tensorboard_log_directory is None:
            tensorboard_log_directory = path.join(output_directory,
                                                  'tensorboard_logs')
        writer = TensorboardWriter(tensorboard_log_directory)

        data_shape = (132, 132, 3)
        model = aumpnet.model(data_shape)
        transformer = Transformer((132, 132))
        transformer.set_mean(128.0)
        transformer.set_scale(1.0 / 128.0)
        epochs = 10
        batch_size = 128
        aumpnet.compile(model, writer)

        # log hyper parameters to tensorboard
        writer.log_hyper_param('batch_size', batch_size)
        writer.log_hyper_param('data_shape',
                               '{0}x{1}x{2}'.format(data_shape[0],
                                                    data_shape[1],
                                                    data_shape[2]))

        model_training_data = DefaultDataLoader(training_data_path,
                                                transformer,
                                                batch_size=batch_size,
                                                data_shape=data_shape)
        engine = Training(model_training_data, output_directory,
                          epochs, writer)

        model_validation_data = DefaultDataLoader(validation_data_path,
                                                  transformer,
                                                  batch_size=90,
                                                  data_shape=data_shape)

        metrics = {'au_occurrence': validation_metrics.f1_score}

        engine = Validation(engine, model_validation_data, metrics, writer)

        callbacks = [self._tensorboard_callback(tensorboard_log_directory),
                     ReduceLROnPlateau(monitor='loss',
                                       patience=0,
                                       epsilon=0.1,
                                       verbose=1),
                     EarlyStopping(monitor='val_f1',
                                   min_delta=0.001,
                                   patience=5,
                                   verbose=1,
                                   mode='max')]

        engine.run(model, callbacks)

    def _run_vggface(self, training_data_path, validation_data_path,
                     output_directory, tensorboard_log_directory):
        if tensorboard_log_directory is None:
            tensorboard_log_directory = path.join(output_directory,
                                                  'tensorboard_logs')
        writer = TensorboardWriter(tensorboard_log_directory)

        data_shape = (224, 224, 3)
        model = vggface.model(data_shape)
        transformer = Transformer((224, 224))
        transformer.set_mean(128.0)
        transformer.set_scale(1.0 / 128.0)
        epochs = 100
        batch_size = 64
        vggface.compile(model, writer)

        # log hyper parameters to tensorboard
        writer.log_hyper_param('batch_size', batch_size)
        writer.log_hyper_param('data_shape',
                               '{0}x{1}x{2}'.format(data_shape[0],
                                                    data_shape[1],
                                                    data_shape[2]))

        model_training_data = BaselineLoader(training_data_path,
                                             transformer,
                                             batch_size=batch_size,
                                             data_shape=data_shape)

        model_validation_data = BaselineLoader(validation_data_path,
                                               transformer,
                                               batch_size=90,
                                               data_shape=data_shape)

        engine = Training(model_training_data, model_validation_data,
                          output_directory, epochs, writer)

        metrics = {'au_occurrence': validation_metrics.f1_score}

        engine = Validation(engine, model_validation_data, metrics, writer)

        callbacks = [self._tensorboard_callback(tensorboard_log_directory),
                     ReduceLROnPlateau(monitor='val_loss',
                                       patience=2,
                                       epsilon=0.1,
                                       verbose=1)]

        engine.run(model, callbacks)

    def _log_model_config(self, model):
        config = model.to_yaml()
        model.summary()
        logging.debug('## Begin model config. ##')
        logging.debug(config)
        logging.debug('## End model config. ##')

    def _tensorboard_callback(self, output_directory):
        log_directory = output_directory
        return TensorBoard(log_dir=log_directory,
                           histogram_freq=0,
                           write_graph=True)
