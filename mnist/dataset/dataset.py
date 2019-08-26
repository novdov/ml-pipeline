import random

import numpy as np
import tensorflow as tf
import tqdm

from mnist import SEED
from mnist.dataset.gc import get_client

random.seed(SEED)

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT


class MNISTDataset:
    """MNIST dataset."""

    TABLES = {TRAIN: "train", EVAL: "valid", PREDICT: "test"}

    def __init__(
        self, mode: tf.estimator.ModeKeys, dataset_id: str, unflatten: bool = True
    ):
        self.mode = mode
        self.dataset_id = dataset_id
        self.unflatten = unflatten
        self.client = get_client("bigquery")

        self.data = self.get_data()

    def get_data(self):
        def _string_to_float(_raw_image: str):
            arr = np.asarray(_raw_image.split(","), "float")
            if self.unflatten:
                return arr.reshape([28, 28])
            return arr

        mode = self.mode
        dataset_ref = self.client.dataset(self.dataset_id)

        print(f"Mode: {mode} -> Load data from table")
        rows = self.client.list_rows(dataset_ref.table(self.TABLES[self.mode]))
        arrows = rows.to_arrow()
        arrows_dict = arrows.to_pydict()

        labels, images = arrows_dict["key"], arrows_dict["image"]
        labels = np.array(labels)
        images = np.array([_string_to_float(image) for image in tqdm.tqdm(images)], "float")

        # Result from BigQuery are sorted by key.
        data = list(zip(images, labels))
        random.shuffle(data)
        images, labels = zip(*data)
        return np.array(images, "float"), np.array(labels, "int")

    def data_generator(self):
        def _gen():
            for image, label in zip(*self.data):
                yield image, label

        return _gen

    def get_input_fn(self, batch_size: int, shuffle=False):
        def _preprocess(image, label):
            image = image / 255.0
            if self.unflatten:
                image = tf.reshape(image, [28, 28, 1])
            return {"image": image}, label

        def _get_input_fn():
            output_types = (tf.float32, tf.int32)
            output_shapes = ([28, 28] if self.unflatten else [784], [])

            dataset = tf.data.Dataset.from_generator(
                self.data_generator(), output_types, output_shapes
            )

            if self.mode == TRAIN:
                dataset = dataset.repeat()
            if shuffle:
                dataset = dataset.shuffle(batch_size * 10)

            dataset = dataset.map(_preprocess)
            dataset = dataset.batch(batch_size).prefetch(8)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        return _get_input_fn
