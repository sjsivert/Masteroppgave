import pandas as pd
from expects import be_true, equal, expect
from genpipes import compose
from mamba import (_it, before, description, included_context, it,
                   shared_context)
from pandas import DataFrame, Timestamp
from spec.test_logger import init_test_logging
from spec.utils.mock_pipeline import create_mock_pipeline
from spec.utils.test_data import mock_data, test_data
from src.pipelines import market_insight_processing as p

with description("Market insight prosessing pipeline", "unit") as self:
    with before.all:
        init_test_logging()

    with shared_context("mock_pipeline"):
        pipeline = create_mock_pipeline()
    with it("It can load data"):
        # noinspection PyTypeChecker
        result = compose.Pipeline(steps=[("load data", test_data, {})]).run()
        expect(result.equals(pd.DataFrame(mock_data))).to(be_true)

    with it("Can run all processing steps in a complete pipeline"):
        # arrange
        with included_context("mock_pipeline"):
            expected_result = {
                "id_x": {0: 0, 1: 1, 2: 2},
                "product_id": {0: 34817620, 1: 34796949, 2: 34763798},
                "manufacturer_id": {0: 211757, 1: 211757, 2: 211757},
                "cat_id": {0: 722, 1: 722, 2: 722},
                "id_y": {0: "internettkabel", 1: "internettkabel", 2: "internettkabel"},
                "date": {
                    0: Timestamp("2021-11-29 04:01:40.409000+0000", tz="UTC"),
                    1: Timestamp("2021-11-29 04:01:40.409000+0000", tz="UTC"),
                    2: Timestamp("2021-11-29 04:01:40.409000+0000", tz="UTC"),
                },
            }
            result = pipeline.run()
            expect(result.to_dict()).to(equal(expected_result))

    with it("can pivot transform the data with date as index"):
        with included_context("mock_pipeline"):
            # fmt: off
            pipeline_with_pivot = compose.Pipeline(
                steps=pipeline.steps + [
                    ("pivot transform with date as index and cat_id as column", p.pivot_transform,
                     {"index": ["date"], "columns": ["cat_id"]})
                ]
            )
            result = pipeline.run()

            expected_result = {'id_x': {0: 0, 1: 1, 2: 2}, 'product_id': {0: 34817620, 1: 34796949, 2: 34763798},
                               'manufacturer_id': {0: 211757, 1: 211757, 2: 211757}, 'cat_id': {0: 722, 1: 722, 2: 722},
                               'id_y': {0: 'internettkabel', 1: 'internettkabel', 2: 'internettkabel'},
                               'date': {0: Timestamp('2021-11-29 04:01:40.409000+0000', tz='UTC'),
                                        1: Timestamp('2021-11-29 04:01:40.409000+0000', tz='UTC'),
                                        2: Timestamp('2021-11-29 04:01:40.409000+0000', tz='UTC')}}
            expect(result.to_dict()).to(equal(expected_result))
