from keras.callbacks import CSVLogger, LambdaCallback, ModelCheckpoint
import logging
from os import path
from keras import backend as K


class Training(object):

    _model_training_data = None
    _output_directoty = ''
    _epochs = 0

    def __init__(self, model_training_data, validation_data,
                 output_directory, epochs, writer):
        self._model_training_data = model_training_data
        self._epochs = epochs
        self._output_directoty = output_directory
        self._writer = writer
        self._validation_data = validation_data

    def run(self, model, callbacks=[]):
        callbacks = self._build_callbacks(callbacks, model)

        training_data = self._model_training_data.batch_generator()
        iterations_per_epoch = self._model_training_data.iterations_per_epoch

        # validation_data = self._validation_data.batch_generator()
        # validation_iter_per_epoch =
        # self._validation_data.iterations_per_epoch

        logging.info('Fitting model using generator.')

        model.fit_generator(training_data,
                            steps_per_epoch=iterations_per_epoch,
                            epochs=self._epochs,
                            callbacks=callbacks,
                            verbose=1)

    def _build_callbacks(self, callbacks, model):
        results_filepath = path.join(self._output_directoty,
                                     'epoch_outputs.csv')

        log_results = CSVLogger(results_filepath, append=True)
        # log_batches = LambdaCallback(on_batch_end=self._on_batch_end)
        reset_sequences = LambdaCallback(on_batch_begin=self._on_batch_end)

        shuffle_data = LambdaCallback(on_epoch_end=self._on_batch_begin(model))

        # change save_weights_only to False
        # but needs to solve https://github.com/fchollet/keras/issues/2790
        filepath = path.join(self._output_directoty, 'net_epoch_{epoch}.hdf5')
        checkpoint = ModelCheckpoint(filepath,
                                     save_best_only=False,
                                     save_weights_only=True,
                                     period=1)

        return callbacks + [reset_sequences, shuffle_data,
                            log_results, checkpoint]

    def _on_epoch_end(self, model):
        return lambda epoch, logs: self._epoch_end(model, epoch, logs)

    def _epoch_end(self, model, epoch, logs):
        if logs is not None:
            lr = float(K.get_value(model.optimizer.lr))
            self._writer.log_scalar('learning_rate', lr, epoch)

        self._model_training_data.shuffle_data()

    def _on_batch_end(self, batch, logs):
        logging.info('Batch {}, loss = {}.'.format(batch, logs['loss']))

    def _on_batch_begin(self, model, batch, logs):
        if self._model_training_data.frame_index == 0:
            model.reset_states()
