import json
from pathlib import Path
from typing import List, Optional

import yaml
from ray.job_submission import JobSubmissionClient

from surfer.core import constants
from surfer.core.models import SubmitExperimentRequest, ExperimentSummary, ExperimentDetails
from surfer.core.schemas import SurferConfig
from surfer.log import logger
from surfer.storage.clients import StorageClient


class ExperimentService:
    def __init__(self, storage_client: StorageClient, job_client: JobSubmissionClient):
        self.storage_client = storage_client
        self.job_client = job_client

    async def submit(self, req: SubmitExperimentRequest):
        pass

    async def delete(self, experiment_name: str):
        pass

    async def stop(self, experiment_name: str):
        pass

    async def list(self) -> List[ExperimentSummary]:
        # Fetch experiments data from storage
        paths = await self.storage_client.list(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/*")
        summaries = []
        for path in paths:
            try:
                summaries.append(ExperimentSummary.from_path(path))
            except ValueError:
                logger.warn(f"Found invalid path in experiments storage: {path}")
        # Fetch experiment running status TODO
        # client = JobSubmissionClient("http://127.0.0.1:8265")
        # for j in client.list_jobs():
        #     j.status
        return summaries

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        return None


class Factory:
    @staticmethod
    def new_experiment_service(config: SurferConfig) -> ExperimentService:
        storage_client = StorageClient.from_config(config.storage)
        job_client = JobSubmissionClient(address=config.ray_address)
        return ExperimentService(storage_client=storage_client, job_client=job_client)


class SurferConfigManager:
    def __init__(
            self,
            base_path: Path = constants.SURFER_CONFIG_BASE_DIR_PATH,
            config_file_name=constants.SURFER_CONFIG_FILE_NAME,
    ):
        self.config_file_path = base_path / config_file_name

    def config_exists(self) -> bool:
        return self.config_file_path.exists()

    def save_config(self, config: SurferConfig):
        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file_path, "w") as f:
            config_dict = json.loads(config.json())
            f.write(yaml.dump(config_dict))

    def load_config(self) -> Optional[SurferConfig]:
        if not self.config_exists():
            return None
        try:
            with open(self.config_file_path) as f:
                config_dict = yaml.safe_load(f.read())
                return SurferConfig.parse_obj(config_dict)
        except Exception as e:
            raise Exception(f"Error parsing CloudSurfer config at {self.config_file_path}: {e}")
