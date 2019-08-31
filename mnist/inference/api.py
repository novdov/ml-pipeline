from typing import List, Mapping, Optional

import googleapiclient.discovery
import tqdm

SERVICE_NAME = "ml"


class InferAPI:
    """Serverless Inference API."""

    def __init__(
        self,
        project_id: str,
        model_name: str,
        version: Optional[str] = "v1",
        service_name: str = SERVICE_NAME,
    ):
        self._project_id = project_id
        self._model_name = model_name
        self._version = version
        self.service = googleapiclient.discovery.build(
            serviceName=service_name, version=version
        )

    def predict(self, images: List[List[float]], batch_size: int = 100) -> Mapping:
        def _parse_results(list_of_dict: List[Mapping], key: str):
            return [res[key].pop() for res in list_of_dict]

        response = {"class_ids": []}

        for start in tqdm.tqdm(range(0, len(images), batch_size)):
            end = min(start + batch_size, len(images))
            _response = self.request(images[start:end]).execute()

            predictions = _response["predictions"]
            for key in response.keys():
                response[key].extend(_parse_results(predictions, key))
        return response

    def request(self, images: List[List[float]]):
        def _generate_payloads(_images):
            return {"instances": [{"image": image} for image in _images]}

        return self.service.projects().predict(
            name=f"projects/{self.project_id}/models/{self.model_name}",
            body=_generate_payloads(images),
        )

    def get_model_meta(self):
        name = f"projects/{self.project_id}/models/{self.model_name}"
        if self.version is not None:
            name += f"/versions/{self.version}"
            response = (
                self.service.projects().models().versions().get(name=name).execute()
            )
            return response
        response = self.service.projects().models().get(name=name).execute()
        return response["defaultVersion"]

    @property
    def project_id(self):
        return self._project_id

    @property
    def model_name(self):
        return self._model_name

    @property
    def version(self):
        return self._version
