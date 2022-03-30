import os

import pytest
import torch
from confuse import Configuration
from confuse.exceptions import NotFoundError
from expects import be_true, equal, expect
from mamba import after, before, description, included_context, it, shared_context
from mockito import unstub, verify, when
from pandas import DataFrame
from spec.mock_config import init_mock_config
from src.utils import config_parser
from src.utils.pytorch_error_calculations import *

with description("error_calculations", "unit") as self:
    with before.each:
        init_mock_config()
        config["experiment"].set({"error_metrics": ["MSE", "MAE"]})

    with shared_context("mock_dataset"):
        targets = torch.tensor([[[2], [4], [1]]], dtype=torch.float64)
        predictions = torch.tensor([[[3], [2], [1]]], dtype=torch.float64)

    with it("can calculate MAE"):
        with included_context("mock_dataset"):
            expected_mae = 1.0
            mae = calculate_mae(targets, predictions)
            expect(round(mae.item(), 4)).to(equal(expected_mae))

    with it("can calculate MSE"):
        with included_context("mock_dataset"):
            expected_mse = round(5 / 3, 4)
            mse = calculate_mse(targets, predictions)
            expect(round(mse.item(), 4)).to(equal(expected_mse))

    with it("can caluclate MASE"):
        with included_context("mock_dataset"):
            expected_mase = 0.6
            mase = calculate_mase(targets, predictions)
            expect(round(mase.item(), 4)).to(equal(expected_mase))

    with it("can caluclate SMAPE"):
        with included_context("mock_dataset"):
            expected_smape = 0.356
            smape = calculate_smape(targets, predictions)
            expect(round(smape.item(), 3)).to(equal(expected_smape))

    with it("can calculate errors"):
        with included_context("mock_dataset"):
            errors = calculate_errors(targets, predictions)
            expect(("MAE" and "MSE") in errors).to(be_true)

    with it("can ignore wrong configuration"):
        with included_context("mock_dataset"):
            config["experiment"].set({"error_metrics": ["not a valid metric", "MAE"]})
            errors = calculate_errors(targets, predictions)
            expect("MAE" in errors).to(be_true)
            expect(len(errors.keys())).to(equal(1))
