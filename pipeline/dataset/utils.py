#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

# https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py

import gzip
import os
import shutil
import tempfile
import urllib
from typing import Tuple

import numpy as np
import tensorflow as tf

from pipeline import SEED


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, "rb") as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError(
                "Invalid magic number %d in MNIST file %s" % (magic, f.name)
            )
        if rows != 28 or cols != 28:
            raise ValueError(
                "Invalid MNIST file %s: Expected 28x28 images, found %dx%d"
                % (f.name, rows, cols)
            )


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, "rb") as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError(
                "Invalid magic number %d in MNIST file %s" % (magic, f.name)
            )


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename + ".gz"
    _, zipped_filepath = tempfile.mkstemp(suffix=".gz")
    print(f"Downloading {url} to {zipped_filepath}")
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, "rb") as f_in, tf.gfile.Open(
        filepath, "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file, normalize=True):
    """Download and parse MNIST dataset."""

    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        if normalize:
            return image / 255.0
        return image

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.cast(label, tf.int32)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16
    ).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(
        decode_label
    )
    return tf.data.Dataset.zip((images, labels))


def split_train_valid(
    tf_dataset: tf.data.Dataset,
    train_size: int = 55000,
    valid_size: int = 5000,
    shuffle=True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    tf_dataset = tf_dataset.shuffle(train_size + valid_size, seed=SEED)

    train_dataset = tf_dataset.take(train_size)
    valid_dataset = tf_dataset.skip(train_size)

    if shuffle:
        train_dataset = train_dataset.shuffle(train_size, seed=SEED)
        valid_dataset = valid_dataset.shuffle(valid_size, seed=SEED)

    return train_dataset, valid_dataset
