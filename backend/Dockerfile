FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY start.sh ./start.sh
RUN chmod +x start.sh && mkdir -p app/data model

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV USE_MOCK=false

CMD ["bash", "start.sh"]
