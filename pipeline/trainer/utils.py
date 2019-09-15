import os

import tensorflow as tf

from pipeline import SEED


def make_config(
    experiment_name: str,
    model_dir: str,
    summary_steps: int = 1000,
    ckpt_steps: int = 1000,
    max_ckpt: int = 10,
) -> tf.estimator.RunConfig:
    return tf.estimator.RunConfig(
        model_dir=os.path.join(model_dir, experiment_name),
        save_summary_steps=summary_steps,
        save_checkpoints_steps=ckpt_steps,
        keep_checkpoint_max=max_ckpt,
        tf_random_seed=SEED,
    )
