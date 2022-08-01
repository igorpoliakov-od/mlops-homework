import pandas as pd

import batch

from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df = pd.DataFrame(data, columns=columns)


year = 2021
month = 1

output_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"


options = {
    'client_kwargs': {
        'endpoint_url': "https://localhost.localstack.cloud:4566"
    }
}

# print(df_output)
print(df)

df.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)