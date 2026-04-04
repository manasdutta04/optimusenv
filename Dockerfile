FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory for torchvision dataset caching
RUN mkdir -p /app/data && chmod 777 /app/data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV TORCHVISION_DATASETS=/app/data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
