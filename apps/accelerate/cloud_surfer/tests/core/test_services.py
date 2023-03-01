import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from unittest.mock import AsyncMock, MagicMock

from ray.job_submission import JobDetails
from ray.job_submission import JobStatus, JobType

from surfer.core import constants
from surfer.core.models import ExperimentStatus, ExperimentSummary
from surfer.core.schemas import SurferConfig
from surfer.core.services import SurferConfigManager, ExperimentService
from surfer.storage.aws import AWSStorageConfig
from surfer.storage.azure import AzureStorageConfig
from surfer.storage.gcp import GCPStorageConfig
from surfer.storage.models import StorageConfig, StorageProvider


class MockedStorageConfig(StorageConfig):
    provider = StorageProvider.AZURE


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_config_exists__not_exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        self.assertFalse(manager.config_exists())

    def test_config_exists__exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        manager.config_file_path.touch()
        self.assertTrue(manager.config_exists())

    def test_load_config__not_exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        self.assertIsNone(manager.load_config())

    def test_load_config__invalid_config(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        with open(manager.config_file_path, "w") as f:
            f.write("foo: bar")
        with self.assertRaises(Exception):
            manager.load_config()

    def test_save_config(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        cluster_file_path = Path(self.tmp_dir.name, "cluster.yaml")
        with open(cluster_file_path, "w") as f:
            f.write("")
        config = SurferConfig(
            cluster_file=cluster_file_path,
            storage=AzureStorageConfig(sas_url="https://myaccount.blob.core.windows.net/pictures"),
        )
        manager.save_config(config)
        self.assertEqual(config.json(), manager.load_config().json())

    def test_save_config__storage_serialization(self):
        # Init config file
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        cluster_file_path = Path(self.tmp_dir.name, "cluster.yaml")
        with open(cluster_file_path, "w") as f:
            f.write("")
        # Available storage configs
        storage_configs = [
            AzureStorageConfig(
                sas_url="https://myaccount.blob.core.windows.net/pictures",
            ),
            GCPStorageConfig(
                project="my-project",
                bucket="my-bucket",
            ),
            AWSStorageConfig()
        ]
        # For each storage config, test whether it gets serialized/deserialized correctly
        for c in storage_configs:
            surfer_config = SurferConfig(
                cluster_file=cluster_file_path,
                storage=c,
            )
            manager.save_config(surfer_config)
            loaded_config = manager.load_config()
            self.assertEqual(c, loaded_config.storage)


class TestExperimentService(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _get_job_details(*statuses: JobStatus, metadata: Dict = None) -> list[JobDetails]:
        if metadata is None:
            metadata = {}
        return [JobDetails(type=JobType.SUBMISSION, entrypoint="", status=s, metadata=metadata) for s in statuses]

    def test_get_experiment_status__empty_list(self):
        self.assertEqual(ExperimentStatus.UNKNOWN, ExperimentService._get_experiment_status([]))

    def test_get_experiment_status__running(self):
        # If any job is running, the experiment is running
        jobs = self._get_job_details(
            JobStatus.RUNNING,
            JobStatus.FAILED,
            JobStatus.PENDING,
        )
        self.assertEqual(ExperimentStatus.RUNNING, ExperimentService._get_experiment_status(jobs))

    def test_get_experiment_status__failed(self):
        # No jobs are running, no jobs pending, at least one job failed
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.FAILED,
        )
        self.assertEqual(ExperimentStatus.FAILED, ExperimentService._get_experiment_status(jobs))

    def test_get_experiment_status__succeeded(self):
        # All jobs succeeded
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.SUCCEEDED,
            JobStatus.SUCCEEDED,
        )
        self.assertEqual(ExperimentStatus.SUCCEEDED, ExperimentService._get_experiment_status(jobs))

    def test_get_experiment_status__pending(self):
        # No jobs are running, at least one job is pending
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.PENDING,
            JobStatus.FAILED,
        )
        self.assertEqual(ExperimentStatus.PENDING, ExperimentService._get_experiment_status(jobs))

    async def test_list__no_experiments(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client
        )
        # Setup mocks
        storage_client.list.return_value = []
        # Run
        experiments = await service.list()
        self.assertEqual(0, len(experiments))
        job_client.list_jobs.assert_not_called()

    async def test_list__no_jobs(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client
        )
        # Setup mocks
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        exp_2 = ExperimentSummary(
            name="exp-2",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [exp_1.get_path(), exp_2.get_path()]
        # Run
        experiments = await service.list()
        self.assertEqual(2, len(experiments))
        # No jobs are returned, so all experiments should have status UNKNOWN
        for e in experiments:
            self.assertEqual(ExperimentStatus.UNKNOWN, e.status)

    async def test_list__wrong_paths(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client
        )
        # Setup mocks
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [exp_1.get_path(), Path("foo/bar")]
        # Run
        experiments = await service.list()
        self.assertEqual(1, len(experiments))  # invalid paths should be ignored

    async def test_list__jobs_should_update_status(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        exp_2 = ExperimentSummary(
            name="exp-2",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [exp_1.get_path(), exp_2.get_path()]
        # Setup job client mock
        job_client.list_jobs.return_value = [
            *self._get_job_details(JobStatus.RUNNING, metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name}),
            *self._get_job_details(JobStatus.SUCCEEDED, metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name}),
            *self._get_job_details(JobStatus.SUCCEEDED, metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_2.name}),
            *self._get_job_details(JobStatus.FAILED, metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_2.name}),
        ]

        # Run
        experiments = await service.list()
        self.assertEqual(2, len(experiments))
        # Check if the status is updated correctly
        for e in experiments:
            if e.name == exp_1.name:
                self.assertEqual(ExperimentStatus.RUNNING, e.status)
            if e.name == exp_2.name:
                self.assertEqual(ExperimentStatus.FAILED, e.status)
