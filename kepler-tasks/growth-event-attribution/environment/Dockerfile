FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git netcdf-bin && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        scipy==1.14.1 \
        netCDF4==1.7.2 \
        h5netcdf==1.4.1 \
        xarray==2024.11.0 \
        pandas==2.2.3 \
        joblib==1.4.2 \
        matplotlib==3.9.2 \
        emcee==3.1.6

COPY data /app/data
RUN mkdir -p /app/output
WORKDIR /app
