from datetime import datetime

from surfer.common import constants


def format_datetime_ui(date: datetime) -> str:
    return date.strftime(constants.DISPLAYED_DATETIME_FORMAT)


def format_datetime(dt: datetime) -> str:
    return dt.strftime(constants.INTERNAL_DATETIME_FORMAT)
