import batch

from datetime import datetime
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual_result = batch.prepare_data(df, columns)

    expected_result = {'PUlocationID': {0: '-1', 1: '1'}, 'DOlocationID': {0: '-1', 1: '1'},
                       'pickup_datetime': {0: '1609462920000000000', 1: '1609462920000000000'},
                       'dropOff_datetime': {0: '1609463400000000000', 1: '1609463400000000000'},
                       'duration': {0: 8.000000000000002, 1: 8.000000000000002}}

    assert actual_result.to_dict() == expected_result
