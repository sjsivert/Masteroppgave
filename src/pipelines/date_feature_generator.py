import math
from datetime import datetime


def calculate_season(date: datetime) -> float:
    """
    Takes a number between 0 and 11 and returns a number between -1 and 1
    If the number is a month number close to christmas the number will be close to 1,
    if the number is close to summer, the number will be close to -1.
    """
    month = date.month
    return math.cos(math.pi * (month % 12) / 6)
