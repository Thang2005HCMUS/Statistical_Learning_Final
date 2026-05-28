# server/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện đồ họa lõi của Linux để OpenCV chạy mượt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mở cổng 8000
EXPOSE 8000

# Chạy bằng uvicorn với port 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]