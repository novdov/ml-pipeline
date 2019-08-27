import argparse
import json
import os
from datetime import datetime

import adanet
import numpy as np
import tensorflow as tf

from mnist.trainer import AdanetTrainer
from mnist.inference.api import InferAPI
from mnist.bin import train
from mnist.dataset import MNISTDataset


def get_parser(_=None):
    parser = argparse.ArgumentParser("Adanet for MNIST [Training/Evaluation]")
    parser.add_argument("--model_dir", type=str, help="output directory for model")
    parser.add_argument("--hparams_path", type=str, help="path of json format hparams")
    parser.add_argument("--experiment_name", type=str, help="experiment name")
    parser.add_argument("--dataset_id", type=str, help="dataset id on google BigQuery")
    parser.add_argument("--project_id", type=str, help="project id on GCP")
    parser.add_argument("--model_name", type=str, help="model name for inference")
    parser.add_argument("--version", type=str, help="model version")
    parser.add_argument("--max_request", type=int, help="max number of requests at once",
                        default=100)
    parser.add_argument(
        "--model_logging_path",
        type=str,
        help="path for logging model result",
        default="logging/model_results.txt",
    )
    return parser


def logging_to_file(logging_file, name, metric):
    with open(logging_file) as f:
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        f.write(f"{timestamp}\t{name}\t{metric}")


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()

    root_path = os.path.dirname(os.path.abspath(__file__))
    logging_filename = os.path.join(root_path, args.model_logging_path)

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    hparams = tf.contrib.training.HParams(**json.load(open(args.hparam_path)))
    results = train.train_and_evaluate(
        args.experiment_name, model_dir, args.hparams_path, args.dataset_id
    )

    print("Training Done.")
    print("==" * 20)
    print("Accuracy: ", results["accuracy"])
    print("    Loss: ", results["loss"])
    print("==" * 20)

    tf.compat.v1.logging.info("Start to evaluate model with test data.")
    tf.compat.v1.logging.info("Restore model.")

    network_generator_cls = train.MODEL_MAPPER[args.experiment_name]
    network_generator = network_generator_cls(hparams.learning_rate)
    estimator = adanet.Estimator(
        head=AdanetTrainer(1000).head,
        max_iteration_steps=500,
        subnetwork_generator=network_generator.build_subnetwork_generator(hparams.unflatten),
        model_dir=args.model_dir
    )

    test_dataset = MNISTDataset(
        tf.estimator.ModeKeys.PREDICT, args.dataset_id, hparams.unflatten
    )

    tf.compat.v1.logging.info("Prepare test data.")
    test_predictions = estimator.predict(test_dataset.get_input_fn(hparams.batch_size))
    test_predictions = [pred for pred in test_predictions]

    class_ids = np.array([
        r["class_ids"].item() for r in test_predictions
    ])
    answers = test_dataset.data[1]

    test_accuracy = (class_ids == answers) / len(answers)
    print("Test Result [Current Model]")
    print("==" * 20)
    print("Accuracy: ", test_accuracy)
    print("==" * 20)

    logging_to_file(logging_filename, args.experiment_name, test_accuracy)

    print("Performance of served model.")
    api = InferAPI(args.project_id, args.model_name, args.version)
    images = [
        image.reshape([28, 28, 1]).tolist() for image in test_dataset.data[0] / 255.0
    ]
    result_dict = api.predict(images, args.max_request)
    accuracy = (np.array(result_dict["class_ids"]) == answers) / len(answers)
    print("==" * 20)
    print("Accuracy: ", result_dict)
    print("==" * 20)
