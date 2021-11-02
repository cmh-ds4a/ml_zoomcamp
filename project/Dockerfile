FROM python:3.8.12-slim

RUN pip install pipenv

RUN pip install gunicorn

RUN pip install flask

RUN pip install xgboost==1.4.2

RUN pip install scikit-learn==0.24.2

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "model_xgb.bin", "./"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]