FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt && \
    pip install .
    # pip install -e .

CMD ["bash", "-c", "coverage run -m unittest discover -s tests && coverage report --fail-under=80"]