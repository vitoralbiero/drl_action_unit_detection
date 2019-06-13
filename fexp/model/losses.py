from keras import backend as K
import tensorflow as tf
import numpy as np


def weighted_mean_squared_error(ignore_label=-1):
    def loss(y, y_hat):

        return K.mean(tf.multiply(K.square(y - y_hat),
                                  tf.cast(tf.not_equal(y, ignore_label),
                                          tf.float32)),
                      axis=-1)

    return loss


def weighted_binary_crossentropy(y_true, y_pred):
    pos_weight = np.array([3.731, 3.935, 1.170, 0.822, 0.683,
                           0.779, 1.148, 4.905, 1.913, 5.046])

    loss = K.binary_crossentropy(y_true, y_pred)

    weights = ((pos_weight - 1) * y_true) + 1

    weighted_loss = tf.losses.compute_weighted_loss(loss, weights)

    return K.mean(weighted_loss)


def binary_crossentropy(target, output):
    # transform back to logits
    _epsilon = K._to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)
