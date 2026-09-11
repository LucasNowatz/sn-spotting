FROM python:3.13-slim-bookworm

# netcdf-bin gives the agent ncdump, which is what anyone would reach for first
# when opening an unfamiliar model archive.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        netcdf-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.2.6 \
        netCDF4==1.7.2 \
        xarray==2025.1.2 \
        pandas==2.2.3

WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
