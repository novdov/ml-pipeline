# ml-pipeline
Pipeline for data, training, evaluating, and serving ML model.

- Current data/task: MNIST
- Data Input: Google BigQuery (current: tfrecords from local)
  - Curretn Issue: Low accuracy when using data from BigQuery
- Training/Evaluation: AutoML (using [adanet](https://github.com/tensorflow/adanet))
- Serving: Serveless API (using Google Cloud Functions)
  - Train and evaluate model, and compare it with served model.
