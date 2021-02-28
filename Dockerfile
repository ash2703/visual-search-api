FROM python:3.6-slim
LABEL maintainer="Aayush Singh <aayushsingh@gofynd.com>"

RUN apt update && apt install -y git

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

COPY . /
WORKDIR /

ENV LC_ALL C.UTF-8  
ENV LANG C.UTF-8

RUN pip install -r requirements.txt

RUN dvc remote modify vs-static access_key_id ${AWS_ACCESS_KEY_ID}
RUN dvc remote modify vs-static secret_access_key ${AWS_SECRET_ACCESS_KEY}
RUN dvc pull

CMD uvicorn server:app --host 0.0.0.0 --port 8000
