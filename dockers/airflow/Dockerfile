# FROM apache/airflow:2.9.3-python3.11

# USER root
# # Install system libs for ML and SeaweedFS (curl)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Fix permissions for the requirements file
# USER airflow
# COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

# # Install ML stack - This will be large (4GB+)
# # We do not use --index-url here to ensure we get the CUDA-enabled version
# RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

FROM apache/airflow:2.9.3-python3.10

USER root

# System deps (only if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt
