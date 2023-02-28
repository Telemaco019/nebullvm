import unittest

from surfer.core import storage
from surfer.core.storage import SignedURL


class TestSignedURL(unittest.TestCase):
    def test_valid_azure_url(self):
        url = "https://storageaccount.blob.core.windows.net/container"
        signed_url = SignedURL(url)
        self.assertEqual(storage.Provider.AZURE, signed_url.provider)

    def test_valid_gcp_url(self):
        url = "https://storage.googleapis.com/bucket"
        signed_url = SignedURL(url)
        self.assertEqual(storage.Provider.GCP, signed_url.provider)

    def test_valid_aws_url(self):
        url = "https://s3.amazonaws.com/bucket"
        signed_url = SignedURL(url)
        self.assertEqual(storage.Provider.AWS, signed_url.provider)

    def test_invalid_url__not_an_url(self):
        url = "not an url"
        with self.assertRaises(ValueError):
            SignedURL(url)

    def test_invalid_url__not_storage(self):
        url = "https://www.google.com"
        with self.assertRaises(ValueError):
            SignedURL(url)

    def test_invalid_url__not_https(self):
        url = "http://storageaccount.blob.core.windows.net/container"  # noqa
        with self.assertRaises(ValueError):
            SignedURL(url)
