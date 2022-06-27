FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv 

COPY "Pipfile" .
COPY "Pipfile.lock" .

RUN pipenv install --system --deploy

COPY "starter.py" .