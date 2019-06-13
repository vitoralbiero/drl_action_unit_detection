from keras.callbacks import LambdaCallback
import logging
import numpy as np
from keras import backend as K


class Validation(object):

    _base_engine = None
    _metrics = None
    _validation_data = None

    def __init__(self, base_engine, validation_data, metrics, writer):
        self._validation_data = validation_data
        self._base_engine = base_engine
        self._metrics = metrics
        self._writer = writer

        self.patience = 5
        self.wait = 1
        self.current_score = -1
        self.best_score = -1
        self.reduce_rate = 0.1
        self.current_reduce_nb = 0
        self.early_stop = 5

    def run(self, model, callbacks=[]):
        callbacks = [self._validate_callback(model)] + callbacks
        self._base_engine.run(model, callbacks)

    def _validate_callback(self, model):
        def callback(epoch, logs):
            return self._validate_model(model, epoch, logs)

        return LambdaCallback(on_epoch_end=callback)

    def _reduce_learning_rate_cl(self, model, epoch, logs):
        if epoch % 10 == 0 and epoch != 0 and epoch < 30:
            lr = float(K.get_value(model.optimizer.lr))
            new_lr = lr * self.reduce_rate
            K.set_value(model.optimizer.lr, new_lr)

    def _reduce_learning_rate(self, model, epoch, logs):
        current_score = np.round(self.current_score, 3)
        best_score = np.round(self.best_score, 3)

        if current_score - best_score >= 0.001:
            self.best_score = self.current_score
            self.wait = 1
            self.current_reduce_nb = 0
        else:
            if self.wait >= self.patience:
                if self.current_reduce_nb < self.early_stop:
                    lr = float(K.get_value(model.optimizer.lr))
                    new_lr = lr * self.reduce_rate
                    K.set_value(model.optimizer.lr, new_lr)

                    logging.debug('Epoch {0}: reducing lr to {1}'.format(
                                  epoch, new_lr))

                    self.current_reduce_nb += 1
                    self.wait = 0
                else:
                    logging.debug('Epoch {0}: early stopping'.format(epoch))
                    model.stop_training = True
            else:
                self.wait += 1

    def _early_stop(self, model, epoch, logs):
        current_score = np.round(self.current_score, 3)
        best_score = np.round(self.best_score, 3)

        if current_score - best_score >= 0.001:
            self.best_score = self.current_score
            self.wait = 1
        else:
            if self.wait >= self.early_stop:
                logging.debug('Epoch {0}: early stopping'.format(epoch))
                model.stop_training = True
            else:
                self.wait += 1

    def _validate_model(self, model, epoch, logs):
        logging.info('Epoch {} ended, validating model.'.format(epoch))
        data = self._compute_predictions(model)
        (ground_truth_names, ground_truth, output_names, predictions) = data

        for (j, name) in enumerate(output_names):
            data = self._prepare_predictions_and_ground_truth(predictions,
                                                              ground_truth,
                                                              j, name)
            (output_predictions, output_ground_truth) = data
            self._compute_metric(output_predictions,
                                 output_ground_truth, name, epoch)

        self._reduce_learning_rate(model, epoch, logs)
        self._early_stop(model, epoch, logs)

    def _compute_predictions(self, model):
        self._validation_data.shuffle_data()
        validation_data = self._validation_data.batch_generator()
        ground_truth_names = None
        ground_truth = []
        predictions = []
        # get all predictions and output names
        for i in range(self._validation_data.iterations_per_epoch):
            (inputs, batch_ground_truth) = next(validation_data)
            batch_predictions = model.predict_on_batch(inputs)

            if ground_truth_names is None:
                ground_truth_names = list(batch_ground_truth.keys())

            ground_truth.append(batch_ground_truth)
            if not isinstance(batch_predictions, list):
                batch_predictions = [batch_predictions]
            predictions.append(batch_predictions)

        output_names = [o.name for o in model.outputs]
        output_names = [name[:name.index('/')] for name in output_names]

        return (ground_truth_names, ground_truth, output_names, predictions)

    def _prepare_predictions_and_ground_truth(self, predictions, ground_truth,
                                              index, output_name):
        logging.debug('Formatting predictions for {}.'.format(output_name))
        output_predictions = None
        output_ground_truth = None

        for i in range(len(ground_truth)):
            if output_predictions is None:
                output_predictions = predictions[i][index]
                output_ground_truth = ground_truth[i][output_name]
            else:
                output_predictions = np.vstack([output_predictions,
                                                predictions[i][index]])
                output_ground_truth = np.vstack([output_ground_truth,
                                                 ground_truth[i][output_name]])

        if output_predictions.ndim == 3:
            output_predictions = output_predictions.reshape(
                -1, output_predictions.shape[2])
            output_ground_truth = output_ground_truth.reshape(
                -1, output_ground_truth.shape[2])

        return (output_predictions, output_ground_truth)

    def _compute_metric(self, output_predictions, output_ground_truth,
                        output_name, epoch):
        metric = self._metrics[output_name]
        logging.debug('Computing metric for {}.'.format(output_name))

        scores = metric(output_ground_truth, output_predictions)
        logging.debug('The computed scores for {} are:'.format(output_name))

        for (index, score) in enumerate(scores):
            logging.debug('==> score[{}]={}.'.format(index, score))
            # self._writer(output_name + '_{}'.format(index), score, epoch)

        mean_score = scores.mean()
        self.current_score = mean_score
        logging.debug('Epoch {}, mean score for {}={}.'.format(
            epoch, output_name, mean_score))

        if self._writer is not None:
            self._writer.log_scalar(output_name + '_val', mean_score, epoch)
