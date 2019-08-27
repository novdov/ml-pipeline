import argparse
import json
import os

import adanet
import tensorflow as tf

from mnist.dataset import MNISTDataset
from mnist.model_generator.dnn import DNNGenerator
from mnist.trainer import AdanetTrainer
from mnist.trainer.utils import make_config

MODEL_MAPPER = {"dnn": DNNGenerator}

tf.compat.v1.logging.set_verbosity("INFO")


def get_parser(_=None):
    parser = argparse.ArgumentParser("Adanet for MNIST [Training]")
    parser.add_argument("--model_dir", type=str, help="output directory for model")
    parser.add_argument("--hparams_path", type=str, help="path of json format hparams")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--dataset_id", type=str, help="dataset id on google BigQuery")
    return parser


def train_and_evaluate(
    experiment_name: str, model_dir: str, hparam_path: str, dataset_id: str
):
    hparams = tf.contrib.training.HParams(**json.load(open(hparam_path)))
    batch_size = hparams.batch_size

    train_dataset = MNISTDataset(
        tf.estimator.ModeKeys.TRAIN, dataset_id, hparams.unflatten
    )
    eval_dataset = MNISTDataset(
        tf.estimator.ModeKeys.EVAL, dataset_id, hparams.unflatten
    )

    trainer = AdanetTrainer(hparams.train_steps)

    network_generator_cls = MODEL_MAPPER[experiment_name]
    network_generator = network_generator_cls(hparams.learning_rate)

    estimator = trainer.create_estimator(
        network_generator.build_subnetwork_generator(hparams.unflatten),
        hparams.adanet_iterations,
        adanet.Evaluator(input_fn=train_dataset.get_input_fn(batch_size)),
        make_config(
            experiment_name,
            model_dir,
            hparams.summary_steps,
            hparams.ckpt_steps,
            hparams.max_ckpt,
        ),
    )

    results = trainer.train_and_evaluate(
        estimator,
        train_dataset.get_input_fn(batch_size, shuffle=True),
        eval_dataset.get_input_fn(batch_size),
    )

    return {"accuracy": results["accuracy"], "loss": results["average_loss"]}


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    results = train_and_evaluate(
        args.experiment_name, model_dir, args.hparams_path, args.dataset_id
    )
