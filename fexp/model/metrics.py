import logging
import numpy as np
import sklearn.metrics as sklearn_metrics
from keras import backend as K


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def f1_score_lstm(y, y_hat):
    logging.info('Computing F1 score.')
    scores = []

    for j in range(y.shape[2]):
        target = y[:, :, j]
        predictions = np.round(y_hat[:, :, j])
        (target, predictions) = filter_scores(target, predictions)
        scores.append(sklearn_metrics.f1_score(target, predictions))
    return np.asarray(scores)


def f1_score(y, y_hat):
    logging.info('Computing F1 score.')
    scores = []

    for j in range(y.shape[1]):
        target = y[:, j]
        predictions = np.round(y_hat[:, j])
        (target, predictions) = filter_scores(target, predictions)
        scores.append(sklearn_metrics.f1_score(target, predictions))
    return np.asarray(scores)


def mean_squared_error(y, y_hat):
    logging.info('Computing MSE score.')
    scores = []

    for j in range(y.shape[1]):
        target = y[:, j]
        predictions = y_hat[:, j]
        (target, predictions) = filter_scores(target, predictions)
        score = np.square(target - predictions).mean()
        scores.append(score)
    return np.asarray(scores)


def accuracy(y, y_hat):
    logging.info('Computing accuracy score.')
    targets = []
    predictions = []
    for i in range(y.shape[0]):
        targets.append(np.argmax(y[i, :]))
        predictions.append(np.argmax(y_hat[i, :]))

    return np.asarray([sklearn_metrics.accuracy_score(targets, predictions)])


def filter_scores(y, y_hat):
    indices = y != 9.0
    return (y[indices], y_hat[indices])


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))
