FROM python:3.10.6-buster


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY api.py api.py
COPY preproc.py preproc.py
COPY cnn_model.h5 cnn_model.h5


CMD uvicorn api:app --host 0.0.0.0 --port $PORT
