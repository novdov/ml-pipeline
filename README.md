# ml-pipeline
Pipeline for data, training, evaluating, and serving ML model.

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
--dataset_id="[Your Dataset Id of BigQuery]" \
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

- Now in progress
