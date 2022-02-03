from random import randint

from genpipes import declare
from pandas import DataFrame

mock_data = [
    {
        "id_x": 0,
        "product_id": 34817620,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z",
    },
    {
        "id_x": 1,
        "product_id": 34796949,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z",
    },
    {
        "id_x": 2,
        "product_id": 34763798,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z ",
    },
]


@declare.generator()
def test_data() -> DataFrame:
    return DataFrame(mock_data)


@declare.generator()
def random_data_loader() -> DataFrame:
    return DataFrame([randint(1, 40) for _ in range(50)])
