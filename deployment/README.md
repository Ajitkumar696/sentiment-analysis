# 🚀 Deployment Directory (`deployment/`)

This directory contains containerization and infrastructure assets for deploying the sentiment analysis inference API into production environments.

---

## 📂 Subdirectories & Files

### 1. `deployment/docker/`
* **`Dockerfile`**: Defines the lightweight container image with Python, installed dependencies, and the FastAPI application.
* **`docker-compose.yml`**: Configures multi-service deployment (e.g. inference API alongside an optional monitoring or logging container).

---

## 🐳 Docker Deployment Instructions

### 1. Build the Docker Image
```bash
docker build -f deployment/docker/Dockerfile -t sentiment-api:latest .
```

### 2. Run the Container
```bash
docker run -d --name sentiment_service -p 8000:8000 sentiment-api:latest
```

### 3. Test Health Endpoint
```bash
curl http://localhost:8000/health
```
