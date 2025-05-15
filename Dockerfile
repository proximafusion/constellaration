FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install constellaration

WORKDIR /workspace
