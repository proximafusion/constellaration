FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin the version to guarantee consistent problem evaluation
RUN pip install constellaration==0.2.1

COPY evaluation.py evaluation.py

ENTRYPOINT ["python", "evaluation.py"]
