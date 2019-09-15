from google.cloud import bigquery, storage


def get_client(client_type="bigquery"):
    if client_type == "storage":
        return storage.Client()
    elif client_type == "bigquery":
        return bigquery.Client()
    else:
        raise ValueError(f"{client_type} is not supported.")
