FROM python:3.6-slim
LABEL maintainer="Aayush Singh <aayushsingh@gofynd.com>"
COPY . /
WORKDIR /
ENV LC_ALL C.UTF-8  
ENV LANG C.UTF-8
RUN pip install -r requirements.txt
CMD uvicorn server:app --host 0.0.0.0 --port 8000
