FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl util-linux && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pytest==8.4.1 \
        pytest-json-ctrf==0.3.5 \
        numpy==2.1.3 \
        scipy==1.14.1 \
        netCDF4==1.7.2 \
        joblib==1.4.2

COPY . /tests/

RUN mkdir -p /app/output /logs/verifier && \
    chown -R root:root /tests && \
    chmod 755 /tests && \
    chmod -R go-rwx /tests/hidden_truth /tests/trusted_forward \
        /tests/thresholds.json /tests/verifier /tests/test_science.py
