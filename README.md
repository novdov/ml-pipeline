# ml-pipeline
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Pipeline for data, training, evaluating, and serving ML model. Inspired by this [post](https://www.facebook.com/groups/TensorFlowKR/permalink/971390023202056/).

- Current data/task: MNIST
- Data Input: Google BigQuery
- Training/Evaluation: AutoML (using [adanet](https://github.com/tensorflow/adanet))
- Serving: Serveless API (using Google Cloud Functions)
  - Train and evaluate model, and compare it with served model.


## Requirements
```text
python >= 3.6
adanet
tensorflow >= 1.14, < 2.0
pyarrow
google-cloud-storage
google-api-python-client
```

## Usage
### Prepare Data

Data preparation contains two steps.
1. Download, parse, and convert data into .gz format
2. Upload converted files to BigQuery

```bash
# Raw data -> tfrecords -> .gz
PYTHONPATH="." python3 mnist/dataset/convert_to_txt.py --directory="[output directory]"
```

```bash
# Upload files to BigQuery
PYTHONPATH="." python3 mnist/dataset/upload_to_bigquery.py \
--data_dir="[directory where source files are stored]" \
--dataset_id="[BigQuery dataset id]" \
--client_type="bigquery"
```

### Training/Evaluation

Currently, only DNN is supported. (CNN is in progress.)

```bash
PYTHONPATH="." python3 mnist/bin/train.py \
--model_dir="[output directory of model weights]" \
--experiment_name="[experiment name (e.g.) dnn)]" \
--hparams_path="hparams.json"
```

### Serving

Serving uses Google Cloud Functions.
1. Upload model to GCP AI Platform. ([See this](https://cloud.google.com/blog/products/ai-machine-learning/empower-your-ai-platform-trained-serverless-endpoints-with-machine-learning-on-google-cloud-functions))
2. Request to model API (See `mnist/inference/api.py`)

After first serving, train and evaluate model, and compare it with served model.
```bash
PYTHONPATH="." python3 mnist/bin/train_and_evaluate.py \
--model_dir="[output directory of model weights]" \
--hparams_path="hparams.json" \
--experiment_name="[experiment name (e.g.) dnn)]" \
--dataset_id="[BigQuery dataset id]" \
--project_id="[Project id]" \
--model_name="[model name on AI platform]" \
--version="[model version]" \
--max_request="[max number of instances to request at once]" \
```
