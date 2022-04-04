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


def calculate_day_of_the_week(date: datetime) -> int:
    """
    Takes a date and returns a number between 0 and 6
              0 = Sunday
              1 = Monday
              2 = Tuesday
              3 = Wednesday
              4 = Thursday
              5 = Friday
              6 = Saturday
    """
    return (
        year_code(date)
        + month_code(date)
        + century_code(date)
        + date_number(date)
        - leap_year_code(date)
    ) % 7


# ---- Helper functions calculate day of the week----
year_code = lambda date: (int(str(date.year)[2:]) + (int(str(date.year)[2:]) // 4)) % 7


def century_code(date):
    year = str(date.year)[:2]
    code = 4
    if year == "17":
        code = 4
    elif year == "18":
        code = 2
    elif year == "19":
        code = 0
    elif year == "20":
        code = 6
    elif year == "21":
        code = 4
    elif year == "22":
        code = 2
    elif year == "23":
        code = 0
    return code


month_codes = [
    0,
    3,
    3,
    6,
    1,
    4,
    6,
    2,
    5,
    0,
    3,
    5,
]
month_code = lambda date: month_codes[date.month - 1]
date_number = lambda date: date.day


def leap_year_code(date):
    year = date.year
    month = date.month
    code = 0
    is_divisible_by_4_but_not_100 = year % 4 == 0 and year % 100 != 0
    if is_divisible_by_4_but_not_100:
        code = 1
    elif year % 400 == 0:
        code = 1

    is_january_or_february = month == 1 or month == 2
    if not is_january_or_february:
        code = 0

    return code
