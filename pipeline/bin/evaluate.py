from typing import List

import numpy as np

from pipeline.inference.api import InferAPI


def get_parser(_=None):
    parser = argparse.ArgumentParser("Adanet for MNIST [Evaluation]")
    parser.add_argument("--project_id", type=str, help="project id on GCP")
    parser.add_argument("--model_name", type=str, help="model name for inference")
    parser.add_argument("--version", type=str, help="model version")
    parser.add_argument("--dataset_id", type=str, help="dataset id on google BigQuery")
    parser.add_argument("--hparams_path", type=str, help="path of json format hparams")
    parser.add_argument(
        "--max_request", type=int, help="max number of requests at once", default=100
    )
    return parser


if __name__ == "__main__":
    import argparse
    import json

    import tensorflow as tf

    from pipeline.dataset import MNISTDataset

    args, _ = get_parser().parse_known_args()

    api = InferAPI(args.project_id, args.model_name, args.version)

    hparams = tf.contrib.training.HParams(**json.load(open(args.hparams_path)))
    test_dataset = MNISTDataset(tf.estimator.ModeKeys.PREDICT, args.dataset_id)

    images = [
        image.reshape([28, 28, 1]).tolist() for image in test_dataset.data[0] / 255.0
    ]
    result_dict = api.predict(images, args.max_request)
    answers = test_dataset.data[1]
    accuracy = sum(np.array(result_dict["class_ids"]) == answers) / len(answers)
    print("==" * 20)
    print("Accuracy: ", accuracy)
    print("==" * 20)
