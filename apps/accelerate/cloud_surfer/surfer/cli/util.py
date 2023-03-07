from datetime import datetime

from surfer.core import constants


def format_datetime_ui(date: datetime) -> str:
    return date.strftime(constants.DISPLAYED_DATETIME_FORMAT)
