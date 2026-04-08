FROM python:3.10-slim

WORKDIR /app

# Install system essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Force Hugging Face's port into the environment
EXPOSE 7860
ENV PORT=7860

CMD ["python", "server/app.py", "--port", "7860"]
