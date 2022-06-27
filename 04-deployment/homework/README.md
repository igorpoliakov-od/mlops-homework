To build docker container:

docker build -t ride-duration-prediction-service:v2 .

To run docker container with prediction:

 docker run -it --rm ride-duration-prediction-service:v2 python starter.py 2021 04