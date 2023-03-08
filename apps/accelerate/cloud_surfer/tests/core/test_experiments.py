import datetime
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import yaml
from ray.job_submission import JobDetails
from ray.job_submission import JobStatus, JobType

import surfer.common
import surfer.core
from surfer.common import constants
from surfer.common.exceptions import InternalError, NotFoundError
from surfer.common.schemas import SurferConfig, ExperimentConfig
from surfer.core.experiments import (
    ExperimentService,
    ExperimentStatus,
    SubmitExperimentRequest,
    ExperimentPath,
    ExperimentSummary,
)
from surfer.storage.providers.azure import AzureStorageConfig
from tests.test_utils import MockedStorageConfig


class TestExperimentService(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _get_job_details(
        *statuses: JobStatus,
        metadata: Dict = None,
    ) -> List[JobDetails]:
        if metadata is None:
            metadata = {}
        return [
            JobDetails(
                type=JobType.SUBMISSION,
                entrypoint="",
                status=s,
                metadata=metadata,
            )
            for s in statuses
        ]

    def setUp(self) -> None:
        self.paths_to_cleanup = []

    def tearDown(self) -> None:
        for p in self.paths_to_cleanup:
            p.unlink()

    def test_get_experiment_status__empty_list(self):
        self.assertEqual(
            ExperimentStatus.UNKNOWN,
            ExperimentService._get_experiment_status([]),
        )

    def test_get_experiment_status__stopped(self):
        jobs = self._get_job_details(
            JobStatus.STOPPED,
            JobStatus.STOPPED,
        )
        self.assertEqual(
            ExperimentStatus.STOPPED,
            ExperimentService._get_experiment_status(jobs),
        )

    def test_get_experiment_status__running(self):
        # If any job is running, the experiment is running
        jobs = self._get_job_details(
            JobStatus.RUNNING,
            JobStatus.FAILED,
            JobStatus.PENDING,
        )
        self.assertEqual(
            ExperimentStatus.RUNNING,
            ExperimentService._get_experiment_status(jobs),
        )

    def test_get_experiment_status__failed(self):
        # No jobs are running, no jobs pending, at least one job failed
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.FAILED,
        )
        self.assertEqual(
            ExperimentStatus.FAILED,
            ExperimentService._get_experiment_status(jobs),
        )

    def test_get_experiment_status__succeeded(self):
        # All jobs succeeded
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.SUCCEEDED,
            JobStatus.SUCCEEDED,
        )
        self.assertEqual(
            ExperimentStatus.SUCCEEDED,
            ExperimentService._get_experiment_status(jobs),
        )

    def test_get_experiment_status__pending(self):
        # No jobs are running, at least one job is pending
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.PENDING,
            JobStatus.FAILED,
        )
        self.assertEqual(
            ExperimentStatus.PENDING,
            ExperimentService._get_experiment_status(jobs),
        )

    async def test_list__no_experiments(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
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
            job_client=job_client,
            surfer_config=MagicMock(),
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
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
            ExperimentPath(
                experiment_name=exp_2.name,
                experiment_creation_time=exp_2.created_at,
            ).as_path(),
        ]
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
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup mocks
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
            Path("foo/bar"),
        ]
        # Run
        experiments = await service.list()
        self.assertEqual(1, len(experiments))  # invalid paths should be ignored

    async def test_list__jobs_should_update_status(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
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
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
            ExperimentPath(
                experiment_name=exp_2.name,
                experiment_creation_time=exp_2.created_at,
            ).as_path(),
        ]
        # Setup job client mock
        job_client.list_jobs.return_value = [
            *self._get_job_details(
                JobStatus.RUNNING,
                JobStatus.SUCCEEDED,
                metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name},
            ),
            *self._get_job_details(
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                metadata={constants.JOB_METADATA_EXPERIMENT_NAME: exp_2.name},
            ),
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

    async def test_get__experiment_not_found(self):
        # Init
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        storage_client.list.return_value = []
        self.assertIsNone(await service.get("not-found"))
        job_client.list_jobs.assert_not_called()

    async def test_get__experiment_not_succeeded_should_not_include_results(
        self,
    ):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
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
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
            ExperimentPath(
                experiment_name=exp_2.name,
                experiment_creation_time=exp_2.created_at,
            ).as_path(),
        ]
        # Setup job client mock
        job_client.list_jobs.return_value = self._get_job_details(
            JobStatus.RUNNING,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )

        details = await service.get(exp_1.name)
        self.assertEqual(ExperimentStatus.RUNNING, details.status)
        self.assertEqual(exp_1.created_at, details.created_at)
        self.assertIsNone(details.result)

    async def test_get__experiment_succeeded_but_results_are_missing(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        storage_client.get.return_value = None
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.SUCCEEDED,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs

        details = await service.get(exp_1.name)
        self.assertEqual(ExperimentStatus.SUCCEEDED, details.status)
        self.assertEqual(exp_1.created_at, details.created_at)
        self.assertEqual(len(jobs), len(details.jobs))
        self.assertIsNone(details.result)

    async def test_get__experiment_results_wrong_format(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        storage_client.get.return_value = "invalid"
        # Setup job client mock
        job_client.list_jobs.return_value = self._get_job_details(
            JobStatus.SUCCEEDED,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        # Run
        with self.assertRaises(InternalError):
            await service.get(exp_1.name)

    async def test_delete___job_is_running(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        storage_client.get.return_value = None
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.RUNNING,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        # Run
        with self.assertRaises(InternalError):
            await service.delete(exp_1.name)

    async def test_delete___job_is_pending(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        storage_client.get.return_value = None
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.PENDING,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        # Run
        with self.assertRaises(InternalError):
            await service.delete(exp_1.name)

    async def test_delete___experiment_not_found(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = []
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.STOPPED,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        # Run
        with self.assertRaises(NotFoundError):
            await service.delete(exp_1.name)

    async def test_delete___experiment_single_blob_not_found_should_be_ignored(
        self,
    ):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        storage_client.delete.side_effect = FileNotFoundError
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.SUCCEEDED,
            JobStatus.STOPPED,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        # Run
        await service.delete(exp_1.name)

    async def test_stop__experiment_not_found(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        storage_client.list.return_value = []
        with self.assertRaises(NotFoundError):
            await service.stop("exp-1")

    async def test_stop__experiment_cannot_be_stopped(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.STOPPED,
            JobStatus.STOPPED,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        with self.assertRaises(ValueError):
            await service.stop("exp-1")

    async def test_stop__experiment_does_not_have_jobs(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        # Setup job client mock
        job_client.list_jobs.return_value = []
        with self.assertRaises(ValueError):
            await service.stop("exp-1")

    async def test_stop__success(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name="exp-1",
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        # Setup job client mock
        jobs = self._get_job_details(
            JobStatus.STOPPED,
            JobStatus.RUNNING,
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: exp_1.name,
            },
        )
        job_client.list_jobs.return_value = jobs
        await service.stop("exp-1")

    async def test_submit__experiment_already_exist(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=MagicMock(),
        )
        exp_name = "exp-1"
        # Setup storage client mock
        exp_1 = ExperimentSummary(
            name=exp_name,
            created_at=datetime(2021, 1, 1, 0, 0, 0),
        )
        storage_client.list.return_value = [
            ExperimentPath(
                experiment_name=exp_1.name,
                experiment_creation_time=exp_1.created_at,
            ).as_path(),
        ]
        # Run
        req = SubmitExperimentRequest(
            config=MagicMock(),
            name=exp_name,
        )
        with self.assertRaises(ValueError):
            await service.submit(req)

    async def test_submit_experiment__success(self):
        storage_client = AsyncMock()
        job_client = MagicMock()
        _, name = mkstemp()
        cluster_file_path = Path(name)
        self.paths_to_cleanup.append(cluster_file_path)
        service = ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
            surfer_config=SurferConfig(
                cluster_file=Path(cluster_file_path),
                storage=MockedStorageConfig(),
            ),
        )
        # Setup storage client mock
        storage_client.list.return_value = []
        # Run
        with TemporaryDirectory() as cluster_file_path:
            model_loader = Path(cluster_file_path, "model_loader.py")
            model_loader.touch()
            data_loader = Path(cluster_file_path, "data_loader.py")
            data_loader.touch()
            model_evaluator = Path(cluster_file_path, "model_evaluator.py")
            model_evaluator.touch()
            req = SubmitExperimentRequest(
                config=ExperimentConfig(
                    data_loader=data_loader,
                    model_loader=model_loader,
                    model_evaluator=model_evaluator,
                ),
                name="exp-1",
            )
            await service.submit(req)
        # Asserts
        storage_client.upload_content.assert_called_once()


class TestExperimentPath(unittest.TestCase):
    def test_from_path__invalid_prefix_path(self):
        with self.assertRaises(ValueError):
            surfer.core.experiments.ExperimentPath.from_path(Path("invalid/path"))

    def test_from_path__valid_prefix_wrong_format(self):
        with self.assertRaises(ValueError):
            surfer.core.experiments.ExperimentPath.from_path(
                Path(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/invalid/path"),
            )

    def test_from_path__valid_path(self):
        experiment_name = "test"
        creation_date = datetime.now()
        path = surfer.core.experiments.ExperimentPath(
            experiment_name=experiment_name,
            experiment_creation_time=creation_date,
        ).as_path()
        experiment_path = surfer.core.experiments.ExperimentPath.from_path(path)
        self.assertEqual(experiment_name, experiment_path.experiment_name)
        self.assertEqual(creation_date, experiment_path.experiment_creation_time)


class TestJobWorkingDir(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.paths_to_cleanup = []

    def tearDown(self) -> None:
        for path in self.paths_to_cleanup:
            path.unlink()

    async def test_all_files_are_copied(self):
        _, cluster_file = mkstemp()
        cluster_file_path = Path(cluster_file)
        self.paths_to_cleanup.append(cluster_file_path)

        _, data_loader_file = mkstemp()
        data_loader_path = Path(data_loader_file)
        self.paths_to_cleanup.append(data_loader_path)

        _, model_loader_file = mkstemp()
        model_loader_path = Path(model_loader_file)
        self.paths_to_cleanup.append(model_loader_path)

        _, model_evaluator_file = mkstemp()
        model_evaluator_path = Path(model_evaluator_file)
        self.paths_to_cleanup.append(model_evaluator_path)

        surfer_config = surfer.common.schemas.SurferConfig(
            cluster_file=cluster_file_path,
            storage=AzureStorageConfig(
                sas_url="https://test.blob.core.windows.net",
            ),
        )
        original_surfer_config = surfer_config.copy()
        experiment_config = surfer.common.schemas.ExperimentConfig(
            data_loader=data_loader_path,
            model_loader=model_loader_path,
            model_evaluator=model_evaluator_path,
        )

        async with surfer.core.experiments.job_working_dir(
            surfer_config,
            experiment_config,
        ) as workdir:
            self.assertTrue((workdir.base / workdir.surfer_config_path).exists())

            self.assertTrue((workdir.base / workdir.data_loader_path).exists())
            self.assertNotEqual(data_loader_path, workdir.data_loader_path)

            self.assertTrue((workdir.base / workdir.model_loader_path).exists())
            self.assertNotEqual(model_loader_path, workdir.model_loader_path)

            self.assertTrue((workdir.base / workdir.model_evaluator_path).exists())
            self.assertNotEqual(model_evaluator_path, workdir.model_evaluator_path)

            # Check cluster file has been copied
            with open(workdir.base / workdir.surfer_config_path) as f:
                obj = yaml.safe_load(f.read())
                loaded = surfer.common.schemas.SurferConfig.parse_obj(obj)
                self.assertTrue(loaded.cluster_file.exists())
                self.assertNotEqual(loaded.cluster_file, cluster_file_path)

        # Check surfer config is not modified
        self.assertEqual(original_surfer_config, surfer_config)
