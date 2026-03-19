FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Expose API and Dashboard ports
EXPOSE 8000 8050

ENV PYTHONUNBUFFERED=1

CMD ["python", "app/app.py"]
