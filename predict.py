from argparse import ArgumentParser
import numpy as np
from os import makedirs, path


def parse_args():
    ap = ArgumentParser()

    ap.add_argument('--model', '-m', required=True, type=str,
                    help='Network name to be used for testing.')

    ap.add_argument('--weights', '-w', required=True, type=str,
                    help='Network weights.')

    ap.add_argument('--test_data_path', '-tdp', required=True, type=str,
                    help='Path to test file with ground truth.')

    ap.add_argument('--cache_folder', '-cf', required=False, type=str,
                    help='Path to cache folder.')

    ap.add_argument('--database_name', '-db', required=False,
                    help='Name of the database to detect objects.')

    ap.add_argument('--landmarks_path', '-lt',
                    required=False, type=str,
                    help='Path to the files containing the landmarks.')

    ap.add_argument('--output_directory', '-od', required=True, type=str,
                    help='Path to write outputs/models.')

    ap.add_argument('--seed', required=False, type=int, default=42,
                    help='Seed for random numbers [42].')

    ap.add_argument('--extract_features', '-ef',
                    action="store_true", default=False,
                    help='Extract features instead of predicting labels.')

    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not path.exists(args.output_directory):
        makedirs(args.output_directory)

    np.random.seed(args.seed)
    from fexp.engine import TestPipelineFactory

    factory = TestPipelineFactory()
    model = factory.create(args.model,
                           args.weights,
                           args.test_data_path,
                           args.cache_folder,
                           args.database_name,
                           args.output_directory,
                           args.landmarks_path,
                           args.extract_features)

    model.run()
