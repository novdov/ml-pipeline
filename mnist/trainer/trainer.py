import adanet
import tensorflow as tf
from adanet import Evaluator


class AdanetTrainer:
    """Adanet Trainer."""

    N_CLASSES = 10

    def __init__(self, train_steps: int):
        self.head = self._head()
        self.train_steps = train_steps

    def _head(self):
        return tf.contrib.estimator.multi_class_head(
            self.N_CLASSES, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

    def create_estimator(
        self,
        subnetwork_generator,
        adanet_iterations: int,
        evaluator: Evaluator,
        config: tf.estimator.RunConfig,
    ) -> tf.estimator.Estimator:
        return adanet.Estimator(
            head=self.head,
            subnetwork_generator=subnetwork_generator,
            max_iteration_steps=self.train_steps // adanet_iterations,
            evaluator=evaluator,
            config=config,
        )

    def train_and_evaluate(
        self, estimator: tf.estimator.Estimator, train_input_fn, eval_input_fn
    ):
        results, _ = tf.estimator.train_and_evaluate(
            estimator,
            train_spec=self.get_train_spec(train_input_fn),
            eval_spec=self.get_eval_spec(eval_input_fn),
        )
        return results

    def get_train_spec(self, train_input_fn):
        return tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=self.train_steps
        )

    @staticmethod
    def get_eval_spec(eval_input_fn):
        return tf.estimator.EvalSpec(
            input_fn=eval_input_fn, steps=None, start_delay_secs=1, throttle_secs=1
        )
