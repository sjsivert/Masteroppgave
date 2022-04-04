import datetime
from datetime import datetime

import numpy as np
import pandas as pd
from expects import be_true, equal, expect
from genpipes import compose
from genpipes.compose import Pipeline
from mamba import _it, before, description, included_context, it, shared_context
from pandas import DataFrame, Timestamp
from src.pipelines.date_feature_generator import calculate_day_of_the_week, calculate_season

with description("date_feature_generator", "this") as self:
    with it("can calculate the season"):
        months = [(datetime(2000, x % 12 + 1, 1)) for x in range(0, 24)]
        result = list(map(lambda month: calculate_season(month), months))
        expect(max(result)).to(equal(1))
        expect(min(result)).to(equal(-1))
        expect(calculate_season(datetime(2000, 12, 1))).to(equal(1))
        expect(calculate_season(datetime(2000, 6, 1))).to(equal(-1))

    with it("can calculate the day of the week"):
        """
        0 = Sunday
        1 = Monday
        2 = Tuesday
        3 = Wednesday
        4 = Thursday
        5 = Friday
        6 = Saturday
        """
        calculate_day_of_the_week(datetime(2000, 1, 1)) == 6
        calculate_day_of_the_week(datetime(1969, 7, 20)) == 0
