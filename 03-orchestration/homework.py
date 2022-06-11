from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle

from datetime import date, datetime, timedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(x: str = None):
    '''
    * a. The flow will take in a parameter called `date` which will be a datetime. `date` should default to `None`.
    * b. If `date` is `None`, use the current day. Use the data from 2 months back as the training data and the data from the previous month as validation data.
    * c. If a `date` value is supplied, get 2 months before the `date` as the training data, and the previous month as validation data.
    * d. As a concrete example, if the date passed is "2021-03-15", the training data should be "fhv_tripdata_2021-01.parquet" and the validation file will be "fhv_trip_data_2021-02.parquet".

    >>> get_paths(None)
    ('../data/fhv_tripdata_2022-04.parquet',
    '../data/fhv_tripdata_2022-05.parquet')

    >>> get_paths("2021-03-15")
    ('../data/fhv_tripdata_2021-01.parquet',
    '../data/fhv_tripdata_2021-02.parquet')

    '''

    assert x is None or type(x) == str

    if not x:
        x = date.today()
    else:
        x = datetime.strptime(x, '%Y-%m-%d')

    m1 = x - relativedelta(months=2, day=1)
    m2 = x - relativedelta(months=1, day=1)
    train_path = f'../data/fhv_tripdata_{format(m1, "%Y-%m")}.parquet'
    val_path = f'../data/fhv_tripdata_{format(m2, "%Y-%m")}.parquet'

    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(report_date:str = None):
    train_path, val_path = get_paths(report_date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()


    with open(f"models/model-{report_date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"models/dv-{report_date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


# main(report_date="2021-08-15")

DeploymentSpec(
  flow=main,
  name="model_training",
  schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
  # flow_runner=SubprocessFlowRunner(),
  tags=["ml"]
)
