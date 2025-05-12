FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/proximafusion/constellaration.git

WORKDIR /workspace
