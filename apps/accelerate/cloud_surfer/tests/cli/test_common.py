import unittest
from unittest.mock import patch

import typer

from surfer.cli.commands.common import (
    must_load_config,
    format_float,
    format_rate,
)
from surfer.common.schemas import SurferConfig


class TestMustLoadConfig(unittest.TestCase):
    @patch("surfer.cli.commands.common.config_manager")
    def test_validation_error(self, config_manager_mock):
        def __validation_error():
            SurferConfig()

        config_manager_mock.load_config.side_effect = __validation_error
        with self.assertRaises(typer.Exit):
            must_load_config()


class TestFormatFloat(unittest.TestCase):
    def test_format_float(self):
        self.assertEqual(format_float(1.23456789), "1.23")

    def test_format_float__precision(self):
        self.assertEqual(format_float(1.23456789, precision=4), "1.2346")


class TestFormatRate(unittest.TestCase):
    def test_format_rate(self):
        self.assertEqual(format_rate(1.23456789), "1.2x")
