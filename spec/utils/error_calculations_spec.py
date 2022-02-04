import os

import pytest
from confuse import Configuration
from confuse.exceptions import NotFoundError
from expects import be_true, equal, expect
from mamba import after, before, description, included_context, it, shared_context
from mockito import unstub, verify, when
from pandas import DataFrame
from spec.mock_config import init_mock_config
from src.utils import config_parser
from src.utils.error_calculations import *

with description("error_calculations", "unit") as self:

    with before.each:
        init_mock_config()
        config["experiment"].set({"error_metrics": ["MSE", "MAE"]})

    with shared_context("mock_dataset"):
        data_set = DataFrame([1, 2, 3, 4])
        proposed_data = DataFrame([1, 2, 4, 4])

    with it("can calculate MAE"):
        with included_context("mock_dataset"):
            expected_mae = 0.25
            mae = calculate_mae(data_set, proposed_data)
            expect(mae).to(equal(expected_mae))

    with it("can calculate MSE"):
        with included_context("mock_dataset"):
            expected_mse = 0.25
            mse = calculate_mse(data_set, proposed_data)
            expect(mse).to(equal(expected_mse))

    with it("can calculate RMSE"):
        with included_context("mock_dataset"):
            expected_rmse = 0.5
            rmse = calculate_rmse(data_set, proposed_data)
            expect(rmse).to(equal(expected_rmse))

    with it("can calculate mape"):
        with included_context("mock_dataset"):
            expected_mape = 0.0833
            mape = calculate_mape(data_set, proposed_data)
            expect(round(mape, 4)).to(equal(expected_mape))

    with it("can caluclate MASE"):
        with included_context("mock_dataset"):
            expected_mase = 0.25
            mase = calculate_mase(data_set, proposed_data)
            expect(round(mase, 4)).to(equal(expected_mase))

    with it("can caluclate SMAPE"):
        with included_context("mock_dataset"):
            expected_smape = 0.071
            smape = calculate_smape(data_set, proposed_data)
            expect(round(smape, 4)).to(equal(expected_smape))

    with it("can calculate errors"):
        with included_context("mock_dataset"):
            errors = calculate_error(data_set, proposed_data)
            expect(("MAE" and "MSE") in errors).to(be_true)

    with it("can ignore wrong configuration"):
        with included_context("mock_dataset"):
            config["experiment"].set({"error_metrics": ["not a valid metric", "MAE"]})
            errors = calculate_error(data_set, proposed_data)
            expect("MAE" in errors).to(be_true)
            expect(len(errors.keys())).to(equal(1))
