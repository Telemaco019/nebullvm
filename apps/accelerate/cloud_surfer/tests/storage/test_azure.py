import unittest

from rich.prompt import InvalidResponse

from surfer.storage.providers.azure import SignedURL, URLPrompt


class TestSignedURL(unittest.TestCase):
    def test_valid_azure_url(self):
        url = "https://storageaccount.blob.core.windows.net/container"
        signed_url = SignedURL(url)
        self.assertEqual(url, signed_url.url)

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


class TestURLPrompt(unittest.TestCase):
    def test_process_response__valid(self):
        val = "https://storageaccount.blob.core.windows.net/container"
        processed = URLPrompt().process_response(val)
        self.assertEqual(val, processed)

    def test_process_response__invalid(self):
        with self.assertRaises(InvalidResponse):
            URLPrompt().process_response("not an url")
