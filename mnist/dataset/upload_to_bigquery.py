import argparse
import os

from google.cloud import bigquery

from mnist.dataset.gc import get_client


def get_parser(_=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory for source files")
    parser.add_argument(
        "--client_type", type=str, help="client type on gcp (storage or bigquery)"
    )
    parser.add_argument("--dataset_id", type=str, help="dataset id on gcp")
    return parser


def get_table_ref(dataset_id, table_id):
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    return table_ref


def get_job_config():
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.field_delimiter = ":"
    job_config.schema = [
        bigquery.schema.SchemaField("key", "integer"),
        bigquery.schema.SchemaField("image", "string"),
    ]
    job_config.autodetect = True
    return job_config


def upload_file_to_bigquery(
    filename: str, table_ref, job_config, dataset_id: str, table_id: str
):
    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(
            source_file, table_ref, location="US", job_config=job_config
        )
    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_id}:{table_id}.")


if __name__ == "__main__":
    args, _ = get_parser().parse_known_args()
    data_dir = os.path.expanduser(args.data_dir)

    client = get_client(args.client_type)
    filenames = [
        f"{data_dir}/{fname}" for fname in os.listdir(data_dir) if fname.endswith(".gz")
    ]
    table_ids = [fname.split("/")[-1].split(".")[1] for fname in filenames]

    for filename, table_id in zip(filenames, table_ids):
        print(f"Loading {filename} rows into {args.dataset_id}:{table_id}.")
        table_ref = get_table_ref(args.dataset_id, table_id)
        upload_file_to_bigquery(
            filename, table_ref, get_job_config(), args.dataset_id, table_id
        )
