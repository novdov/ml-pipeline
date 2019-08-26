# ml-pipeline
Pipeline for data, training, evaluating, and serving ML model.

- Current data/task: MNIST
- Data Input: Google BigQuery (current: tfrecords from local)
  - Curretn Issue: Low accuracy when using data from BigQuery (0.96 vs 0.78)
- Training/Evaluation: AutoML (using [adanet](https://github.com/tensorflow/adanet))
- Serving: Serveless API (using Google Cloud Functions)
  - Train and evaluate model, and compare it with served model.


## Usage
### Prepare Data
- Using tfrecords (local)
```bash
PYTHONPATH="." python3 mnist/dataset/convert_to_records.py --directory="[output directory]"
```

- Using BigQuery
  - Convert mnist data into .gz format and upload to BigQuery
```bash
PYTHONPATH="." python3 mnist/dataset/convert_to_txt.py --directory="[output directory]"
```

```bash
PYTHONPATH="." python3 mnist/dataset/upload_to_bigquery.py \
--data_dir="[directory where source files are stored]" \
--dataset_id="[Your Dataset Id of BigQuery]" \
--client_type="bigquery"
```

### Training/Evaluation
```bash
PYTHONPATH="." python3 mnist/bin/train.py \
--model_dir="[output directory of model weights]" \
--experiment_name="[experiment name (e.g.) dnn)]" \
--hparams_path="hparams.json"
```

### Serving
Now in progress
