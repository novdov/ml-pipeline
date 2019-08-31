import tensorflow as tf
from adanet.examples import simple_dnn

from mnist import SEED


class DNNGenerator:
    """DNN subnetwork Generator using `adanet.examples.simple_dnn.Generator`"""

    FEATURE_KEY = "image"

    def __init__(self, learning_rate: int):
        self.learning_rate = learning_rate

    def build_subnetwork_generator(self):
        feature_columns = [
            tf.feature_column.numeric_column(self.FEATURE_KEY, shape=[28, 28, 1])
        ]

        return simple_dnn.Generator(
            feature_columns=feature_columns,
            optimizer=tf.train.AdamOptimizer(self.learning_rate),
            seed=SEED,
        )
