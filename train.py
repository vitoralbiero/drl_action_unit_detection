from argparse import ArgumentParser
import numpy as np
import logging
from os import makedirs, path
# import tensorflow as tf
# import random as rn

# import os
# os.environ['PYTHONHASHSEED'] = '0'


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--model', '-m', required=True, type=str,
                    help='Network name to be used for training.')

    ap.add_argument('--training_data_path', '-tdp', required=True, type=str,
                    help='Path to training file with ground truth.')

    ap.add_argument('--validation_data_path', '-vdp', required=True, type=str,
                    help='Path to validation file with ground truth.')

    ap.add_argument('--output_directory', '-od', required=True, type=str,
                    help='Path to write outputs/models.')

    ap.add_argument('--tensorboard_log_directory', '-tb',
                    required=False, type=str,
                    help='Separted path for tensorboard logs.')

    ap.add_argument('--bbox_path', '-bp',
                    required=False, type=str,
                    help='Path to file containing the bboxes.')

    ap.add_argument('--train_landmarks_path', '-lt',
                    required=False, type=str,
                    help='Path to the files containing the train landmarks.')

    ap.add_argument('--valid_landmarks_path', '-lv',
                    required=False, type=str,
                    help='Path to the files containing the valid landmarks.')

    ap.add_argument('--weights', '-w', required=False, type=str,
                    help='Network weights.')

    ap.add_argument('--seed', required=False, type=int, default=42,
                    help='Seed for random numbers [42].')

    return ap.parse_args()


def init_log(output_directory):
    log_filepath = path.join(output_directory, 'train.log')
    logging.basicConfig(filename=log_filepath,
                        format='%(asctime)s |%(levelname)s| %(message)s',
                        level=logging.DEBUG)


if __name__ == '__main__':
    args = parse_args()
    if not path.exists(args.output_directory):
        makedirs(args.output_directory)

    np.random.seed(args.seed)
    '''
    rn.seed(12345)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    '''

    from fexp.engine import EngineRunner
    init_log(args.output_directory)

    logging.info('Training with {}.'.format(args))
    runner = EngineRunner()
    logging.info('Running engine!')
    model = runner.run(args.model,
                       args.training_data_path,
                       args.validation_data_path,
                       args.output_directory,
                       args.tensorboard_log_directory,
                       args.bbox_path,
                       args.train_landmarks_path,
                       args.valid_landmarks_path,
                       args.weights)
