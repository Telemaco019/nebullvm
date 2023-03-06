import logging
from unittest.mock import patch

from surfer.cli import util


@patch("surfer.cli.util.logger")
def test_configure_debug_mode__set_debug_true(logger_mock):
    # Given
    debug = True
    # When
    util.configure_debug_mode(debug)
    # Then
    assert logger_mock.level == logging.DEBUG


@patch("surfer.cli.util.logger")
def test_configure_debug_mode__set_debug_false(logger_mock):
    # Given
    debug = False
    original_logger_level = 1
    logger_mock.level = original_logger_level
    # When
    util.configure_debug_mode(debug)
    # Then
    assert logger_mock.level == original_logger_level
