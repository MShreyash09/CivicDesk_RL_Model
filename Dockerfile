FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Force Hugging Face's port into the environment variables
EXPOSE 7860
ENV PORT=7860

# Run the app (app.py will find the ENV PORT automatically)
CMD ["python", "server/app.py"]
