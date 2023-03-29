import logging

import surfer.log
from surfer import log


def test_configure_debug_mode__set_debug_true():
    original_level = log.level
    # Given
    debug = True
    # When
    surfer.log.setup_logger(debug)
    # Then
    assert log.level == logging.DEBUG
    log.level = original_level


def test_configure_debug_mode__set_debug_false():
    # Given
    original_level = log.level
    # When
    surfer.log.setup_logger(debug=False)
    # Then
    assert log.level == original_level
