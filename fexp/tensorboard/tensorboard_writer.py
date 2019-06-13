from keras import backend as K


class TensorboardWriter(object):
    _writer = None

    def __init__(self, tensorboard_log_directory):
        self._writer = K.tf.summary.FileWriter(tensorboard_log_directory)

    def log_scalar(self, tag, value_1, value_2):
        summary = K.tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value_1
        summary_value.tag = tag

        K.tf.summary.merge_all()

        self._writer.add_summary(summary, value_2)
        self._writer.flush()

    def log_hyper_param(self, tag, value):
        summary = K.tf.summary.text(tag, K.tf.convert_to_tensor(str(value)))

        with K.tf.Session() as sess:
            text = sess.run(summary)

        self._writer.add_summary(text)
        self._writer.flush()
