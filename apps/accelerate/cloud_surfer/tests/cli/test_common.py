import unittest
from unittest.mock import patch

import typer

from surfer.cli.commands.common import must_load_config
from surfer.common.schemas import SurferConfig


class TestMustLoadConfig(unittest.TestCase):
    @patch("surfer.cli.commands.common.config_manager")
    def test_validation_error(self, config_manager_mock):
        def __validation_error():
            SurferConfig()

        config_manager_mock.load_config.side_effect = __validation_error
        with self.assertRaises(typer.Exit):
            must_load_config()
