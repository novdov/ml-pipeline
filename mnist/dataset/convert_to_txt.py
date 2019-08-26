import argparse
import gzip
import os

import tensorflow as tf
from tensorflow.python.data import Dataset

from mnist.dataset import utils

tf.logging.set_verbosity("INFO")

TRAIN_IMAGES = "train-images-idx3-ubyte"
TRAIN_LABELS = "train-labels-idx1-ubyte"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


def get_parser(_=None):
    def _bool_str(string):
        if string not in {"false", "true"}:
            raise ValueError("Not a valid boolean string")
        return string == "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="directory for output data.")
    parser.add_argument("--train_size", type=int, default=55000, help="train size.")
    parser.add_argument("--valid_size", type=int, default=5000, help="validation size.")
    parser.add_argument(
        "--split_valid",
        type=_bool_str,
        default=False,
        help="whether split train into train/valid.",
    )
    return parser


def write_to_txt(dataset: Dataset, output_dir: str, name: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    filename = f"{output_dir}/mnist.{name}.txt.gz"
    tf.logging.info(f"Write txt into {filename}")
    total_samples = 0
    with gzip.open(filename, "wt") as f:
        with tf.Session() as sess:
            while True:
                try:
                    image, label = sess.run(batch)
                    label = label[0]
                    image = image[0].astype("<U16")

                    contents = str(label) + ":" + ",".join(list(image))
                    f.write(f"{contents}\n")
                    total_samples += 1
                except tf.errors.OutOfRangeError:
                    tf.logging.info(
                        f"Finished conversion. Total samples: {total_samples}"
                    )
                    break


def convert_dataset(
    directory: str, train_size: int, valid_size: int, split_valid: bool = True
):
    raw_directory = os.path.join(directory, "raw_data")
    txt_directory = os.path.join(directory, "txt")

    train_dataset = utils.dataset(raw_directory, TRAIN_IMAGES, TRAIN_LABELS)
    test_dataset = utils.dataset(raw_directory, TEST_IMAGES, TEST_LABELS)

    if split_valid:
        train_dataset, valid_dataset = utils.split_train_valid(
            train_dataset, train_size, valid_size, shuffle=True
        )
        names = ["train", "valid", "test"]
        dataset_list = [train_dataset, valid_dataset, test_dataset]
    else:
        names = ["train", "test"]
        dataset_list = [train_dataset, test_dataset]

    for name, dataset in zip(names, dataset_list):
        write_to_txt(dataset, txt_directory, name)


def main(directory, train_size, valid_size, split_valid):
    convert_dataset(directory, train_size, valid_size, split_valid)


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()
    main(
        os.path.expanduser(args.directory),
        args.train_size,
        args.valid_size,
        args.split_valid,
    )
