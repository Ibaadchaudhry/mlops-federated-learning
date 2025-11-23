# Docker Deployment Guide for Federated Learning MLOps

## Overview
This Docker setup containerizes the entire federated learning MLOps pipeline with the following services:

- **fl-server**: Federated learning server and clients
- **model-api**: FastAPI service for model predictions  
- **dashboard**: Streamlit monitoring dashboard
- **data-prep**: One-time data preparation service

## Quick Start

### 1. Prepare Client Data (First Time Only)
```bash
# Run data preparation to create client_datasets.pkl
docker-compose --profile data-prep up data-prep
```

### 2. Start All Services
```bash
# Start FL server, API, and dashboard
docker-compose up -d
```

### 3. Access Services
- **Dashboard**: http://localhost:8501
- **Model API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **FL Server**: localhost:8080 (internal)

## Service Details

### Federated Learning Server (fl-server)
- **Port**: 8080
- **Purpose**: Orchestrates federated training across clients
- **Volumes**: 
  - `./models` - Saves trained models
  - `./drift_reports` - Stores drift detection results
  - `./client_datasets.pkl` - Client data (read-only)

### Model API Service (model-api)
- **Port**: 8000
- **Purpose**: REST API for model predictions
- **Endpoints**:
  - `GET /health` - Service health check
  - `GET /model-info` - Current model information
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `POST /reload-model` - Reload latest model
  - `GET /features` - List available features

### Dashboard Service (dashboard)
- **Port**: 8501
- **Purpose**: Web interface for monitoring and predictions
- **Features**:
  - System overview and metrics
  - Client data inspection
  - Drift monitoring visualization
  - Model management
  - Interactive prediction interface

## Usage Examples

### API Prediction Example
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "age": 35,
         "hours-per-week": 40,
         "education.num": 13,
         "capital.gain": 0,
         "capital.loss": 0
       }
     }'

# Check model info
curl http://localhost:8000/model-info

# Health check
curl http://localhost:8000/health
```

### Training New Models
```bash
# Run federated learning training
docker-compose exec fl-server python fl_server.py

# Reload API with new model
curl -X POST http://localhost:8000/reload-model
```

## Docker Commands

### Build Services
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build model-api
```

### Service Management
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f model-api
docker-compose logs -f dashboard

# Scale services (if needed)
docker-compose up -d --scale model-api=2
```

### Development Mode
```bash
# Run with code reload (development)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Access container shell
docker-compose exec model-api bash
docker-compose exec fl-server bash
```

## Volumes and Persistence

### Data Persistence
- **models/**: Trained model artifacts
- **drift_reports/**: Drift detection CSV files  
- **client_datasets.pkl**: Federated client data

### Volume Management
```bash
# List volumes
docker volume ls

# Remove volumes (caution: deletes data)
docker-compose down -v
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check if models directory exists and has files
   docker-compose exec model-api ls -la /app/models/
   ```

2. **API not responding**
   ```bash
   # Check API health
   curl http://localhost:8000/health
   
   # View logs
   docker-compose logs model-api
   ```

3. **Dashboard connection issues**
   ```bash
   # Check if client_datasets.pkl exists
   docker-compose exec dashboard ls -la /app/client_datasets.pkl
   ```

### Service Health Checks
```bash
# Check all service status
docker-compose ps

# Test individual service health
curl http://localhost:8000/health       # API
curl http://localhost:8501/_stcore/health  # Dashboard
```

## Production Deployment

### Environment Variables
Create `.env` file:
```env
# Production settings
PYTHONUNBUFFERED=1
MODEL_RELOAD_INTERVAL=300
LOG_LEVEL=info

# Security (for production)
API_KEY=your-secure-api-key
ALLOWED_HOSTS=your-domain.com
```

### Security Considerations
- Add authentication to API endpoints
- Use HTTPS in production
- Restrict network access with firewall rules
- Use secrets management for sensitive data

### Monitoring
- Health checks are configured for all services
- Logs are available via `docker-compose logs`
- Add Prometheus/Grafana for advanced monitoring

## Network Architecture
```
Internet → Load Balancer → Docker Network (fl-network)
                          ├── fl-server:8080
                          ├── model-api:8000  
                          └── dashboard:8501
```

All services communicate through the `fl-network` Docker network for isolation and security.