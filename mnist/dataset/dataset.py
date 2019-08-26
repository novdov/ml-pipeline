import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from mnist import SEED
from mnist.dataset.gc import get_client

random.seed(SEED)

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

data_dir = Path(__file__).resolve().parents[2].joinpath("data/tfrecords")


class MNISTDataset:
    """MNIST dataset."""

    QUERIES = {
        TRAIN: "SELECT * FROM `mnist-pipeline.mnist.train`",
        EVAL: "SELECT * FROM `mnist-pipeline.mnist.test`",
        PREDICT: "SELECT * FROM `mnist-pipeline.mnist.test`",
    }

    def __init__(self, mode: tf.estimator.ModeKeys, unflatten: bool = True):
        self.mode = mode
        self.unflatten = unflatten
        self.client = get_client("bigquery")

        # self.data = self.get_data()
        if mode == TRAIN:
            self.data, _ = tf.keras.datasets.mnist.load_data()
        else:
            _, self.data = tf.keras.datasets.mnist.load_data()

    def get_data(self):
        mode = self.mode

        def _get_rows():
            query_job = self.client.query(self.QUERIES[mode], location="US")
            return query_job

        def _string_to_float(_raw_image: str):
            return np.asarray(_raw_image.split(","), dtype=np.float32)

        print(f"Mode: {mode} -> Load data from table")
        rows = _get_rows()

        images, labels = [], []
        for row in rows:
            label, raw_image = row.values()
            label = np.asarray(label, dtype=np.int32)
            image = _string_to_float(raw_image)

            if self.unflatten:
                image = image.reshape([28, 28])

            images.append(image)
            labels.append(label)

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        data = list(zip(images, labels))
        random.shuffle(data)

        images, labels = zip(*data)

        return images, labels

    # def data_generator(self):
    #     def _gen():
    #         for image, label in zip(*self.data):
    #             yield image, label
    #     return _gen

    # def get_input_fn(
    #     self, batch_size: int, shuffle=False
    # ):
    #     def _preprocess(image, label):
    #         image = image / 255.0
    #         image = tf.reshape(image, [28, 28, 1])
    #         return {"image": image}, label
    #
    #     def _get_input_fn():
    #         output_types = (tf.float32, tf.int32)
    #         output_shapes = ([28, 28] if self.unflatten else [784], [])
    #
    #         dataset = tf.data.Dataset.from_generator(
    #             self.data_generator(), output_types, output_shapes)
    #
    #         if self.mode == TRAIN:
    #             dataset = dataset.repeat()
    #         if shuffle:
    #             dataset = dataset.shuffle(batch_size * 10)
    #
    #         dataset = dataset.map(_preprocess)
    #         dataset = dataset.batch(batch_size).prefetch(8)
    #         iterator = dataset.make_one_shot_iterator()
    #         features, labels = iterator.get_next()
    #         return features, labels
    #     return _get_input_fn

    def get_input_fn(self, batch_size: int, shuffle=False):

        if self.mode == TRAIN:
            filenames = data_dir.joinpath("train.tfrecords")
        elif self.mode == EVAL:
            filenames = data_dir.joinpath("valid.tfrecords")
        else:
            filenames = data_dir.joinpath("test.tfrecords")

        def _parser(record):
            features = {
                # "label": tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string)
            }
            parsed_record = tf.parse_single_example(record, features)
            image = tf.decode_raw(parsed_record["image_raw"], tf.float32)

            label = tf.cast(parsed_record["label"], tf.int32)
            return {"image": image}, label

        def _input_fn():
            dataset = tf.data.TFRecordDataset(filenames).map(_parser)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10_000)

            if self.mode == TRAIN:
                dataset = dataset.repeat(None)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        return _input_fn
