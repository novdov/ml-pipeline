import argparse
import os

import tensorflow as tf
from tensorflow.python.data import Dataset

from pipeline.dataset import utils

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
    parser.add_argument("--directory", type=str, help="directory for output data")
    parser.add_argument("--train_size", type=int, default=55000, help="train size")
    parser.add_argument("--valid_size", type=int, default=5000, help="validation size")
    parser.add_argument(
        "--split_valid",
        type=_bool_str,
        default=True,
        help="whether split train into train/valid",
    )
    return parser


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(tf_dataset: Dataset, directory: str, name: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    tf_dataset = tf_dataset.batch(batch_size=1)
    batches = tf_dataset.make_one_shot_iterator().get_next()

    filename = os.path.join(directory, name + ".tfrecords")
    tf.logging.info(f"Write tfrecords into {filename}")
    writer = tf.python_io.TFRecordWriter(filename)

    total_samples = 0
    with tf.Session() as sess:
        while True:
            try:
                image, label = sess.run(batches)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "label": _int64_feature(label),
                            "image_raw": _bytes_feature(image.tostring()),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                total_samples += 1
            except tf.errors.OutOfRangeError:
                tf.logging.info(f"Finished conversion. Total samples: {total_samples}")
                break


def convert_dataset(
    directory: str, train_size: int, valid_size: int, split_valid: bool = True
):
    raw_directory = os.path.join(directory, "raw_data")
    record_directory = os.path.join(directory, "tfrecords")

    train_dataset = utils.dataset(raw_directory, TRAIN_IMAGES, TRAIN_LABELS)
    test_dataset = utils.dataset(raw_directory, TEST_IMAGES, TEST_LABELS)

    if split_valid:
        train_dataset, valid_dataset = utils.split_train_valid(
            train_dataset, train_size, valid_size, shuffle=True
        )
        names = ["train", "valid", "test"]
        for name, dataset in zip(names, [train_dataset, valid_dataset, test_dataset]):
            convert_to(dataset, record_directory, name)

    else:
        names = ["train", "test"]
        for name, dataset in zip(names, [train_dataset, test_dataset]):
            convert_to(dataset, record_directory, name)


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
