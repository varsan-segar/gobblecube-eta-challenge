# Reference Dockerfile for the ETA Challenge.
# Target total image size: ≤ 2.5 GB.
#
# Build:
#   docker build -t my-eta .
# Test the grader pathway:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Submission surface
COPY predict.py grade.py features.py ./
COPY model.pkl ./

# Features.py needs the static zone centroids
RUN mkdir -p /app/data
COPY data/zone_centroids.csv ./data/zone_centroids.csv

ENTRYPOINT ["python", "grade.py"]
