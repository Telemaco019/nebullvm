import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch, MagicMock

from google.cloud import exceptions as gcp_exceptions
from google.cloud import storage

from surfer.storage.providers import gcp


@patch("surfer.storage.providers.gcp.storage.Client")
class TestGCSBucketClient(unittest.IsolatedAsyncioTestCase):
    async def test_list__empty(self, mocked_gcp_client):
        mocked_gcp_client.list_blobs.return_value = []
        client = gcp.GCSBucketClient(MagicMock())
        self.assertEqual(0, len(await client.list()))

    async def test_list__multiple_results(self, _):
        client = gcp.GCSBucketClient(MagicMock())
        mock_1 = MagicMock()
        mock_1.name = "blob-1"
        mock_2 = MagicMock()
        mock_2.name = "blob-2"
        mock_3 = MagicMock()
        mock_3.name = "blob-3"
        client.gcs_client.list_blobs.return_value = [
            mock_1,
            mock_2,
            mock_3,
        ]
        results = await client.list()
        self.assertEqual(3, len(results))
        self.assertEqual(Path(mock_1.name), results[0])

    async def test_list__filter(self, _):
        client = gcp.GCSBucketClient(MagicMock())
        filter_str = "filter"
        results = await client.list(filter_str)
        self.assertEqual(0, len(results))
        client.gcs_client.list_blobs.assert_called_with(
            bucket_or_name=mock.ANY,
            prefix=filter_str,
        )

    @patch.object(storage.Bucket, "get_blob", return_value=None)
    async def test_get__not_found(self, *_):
        client = gcp.GCSBucketClient(MagicMock())
        res = await client.get(Path("invalid"))
        self.assertIsNone(res)

    @patch.object(storage.Bucket, "get_blob")
    async def test_get__found(self, *_):
        client = gcp.GCSBucketClient(MagicMock())
        res = await client.get(Path("valid"))
        self.assertIsNotNone(res)

    @patch.object(storage.Bucket, "list_blobs", return_value=[])
    async def test_delete__path_not_found(self, *_):
        client = gcp.GCSBucketClient(MagicMock())
        with self.assertRaises(FileNotFoundError):
            await client.delete(Path("invalid"))

    async def test_delete__dir(self, *_):
        client = gcp.GCSBucketClient(MagicMock())
        dir_blobs = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        client.gcs_client.list_blobs.return_value = dir_blobs
        await client.delete(Path("dir"))
        for blob_mock in dir_blobs:
            blob_mock.delete.assert_called_once()

    async def test_delete__blob_not_found(self, *_):
        client = gcp.GCSBucketClient(MagicMock())
        not_found_blob = MagicMock()
        not_found_blob.delete.side_effect = gcp_exceptions.NotFound(
            "Not found"
        )
        dir_blobs = [
            MagicMock(),
            not_found_blob,
            MagicMock(),
        ]
        client.gcs_client.list_blobs.return_value = dir_blobs
        with self.assertRaises(FileNotFoundError):
            await client.delete(Path("dir"))
