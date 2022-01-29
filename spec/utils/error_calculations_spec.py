import os

import pytest
from confuse.exceptions import NotFoundError
from expects import be_true, expect, equal
from mamba import description, it
from pandas import DataFrame
from src.utils.error_calculations import *

with description("error_calculations", "unit") as self:

    with it("can calculate MAE"):
        data_set = DataFrame([1, 2, 3, 4])
        proposed_data = DataFrame([1, 2, 4, 4])
        expected_mae = 0.25
        mae = calculate_mae(data_set, proposed_data)
        expect(mae).to(equal(expected_mae))

    with it("can calculate MSE"):
        data_set = DataFrame([1, 2, 3, 4])
        proposed_data = DataFrame([1, 2, 4, 4])
        expected_mse = 0.25
        mse = calculate_mse(data_set, proposed_data)
        expect(mse).to(equal(expected_mse))

    with it("can calculate errors"):
        data_set = DataFrame([1, 2, 3, 4])
        proposed_data = DataFrame([1, 2, 4, 4])
        errors = calculate_error(data_set, proposed_data)
        expect(("MAE" and "MSE") in errors).to(be_true)
