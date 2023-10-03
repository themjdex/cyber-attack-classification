FROM python:3.9

RUN apt-get update && apt-get install nano

WORKDIR /ctr_app
COPY . /ctr_app

RUN pip install --timeout 1000 -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# original airlow image
#ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
#FROM ${AIRFLOW_BASE_IMAGE}
#
#RUN pip install --user --no-cache-dir \
#    apache-airflow-providers-docker==2.6.0
#
#USER root
#ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages