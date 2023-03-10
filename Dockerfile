FROM python:3.10.6-buster


RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY linguistic_dna/api/api.py api.py
COPY linguistic_dna/ml_dna/preprocessor.py ml_dna/preprocessor.py
COPY cnn_model.h5 cnn_model.h5


CMD uvicorn api:app --host 0.0.0.0 --port $PORT
