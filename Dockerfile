FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    curl \
    tar \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements-dev.txt ./
RUN pip install --upgrade pip && pip install -r requirements-dev.txt
COPY . .
RUN pip install -e .

CMD ["bash", "-c", "coverage run --source=src -m unittest discover -s tests && coverage report --fail-under=80"]