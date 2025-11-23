#!/bin/bash
# deploy.sh - Deployment script for Federated Learning MLOps

set -e  # Exit on error

echo "🚀 Deploying Federated Learning MLOps System"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# Check if client_datasets.pkl exists
if [ ! -f "client_datasets.pkl" ]; then
    print_warning "client_datasets.pkl not found. Preparing client data..."
    
    # Run data preparation
    docker-compose --profile data-prep up --build data-prep
    
    if [ -f "client_datasets.pkl" ]; then
        print_status "Client data prepared successfully"
    else
        print_error "Failed to prepare client data"
        exit 1
    fi
else
    print_status "Client data already exists"
fi

# Check if models exist
if [ ! -d "models" ] || [ ! "$(ls -A models)" ]; then
    print_warning "No trained models found. You may need to run federated learning first."
    mkdir -p models
fi

# Build and start services
echo ""
echo "🔨 Building Docker images..."
docker-compose build

print_status "Docker images built successfully"

echo ""
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."

# Function to wait for service health
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            print_status "$service_name is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to become ready"
    return 1
}

# Wait for API service
if wait_for_service "Model API" "http://localhost:8000/health"; then
    echo ""
else
    echo ""
    print_error "Model API service failed to start"
    docker-compose logs model-api
fi

# Wait for dashboard
if wait_for_service "Dashboard" "http://localhost:8501/_stcore/health"; then
    echo ""
else
    echo ""
    print_warning "Dashboard service may not be fully ready (this is sometimes normal)"
fi

# Show service status
echo ""
echo "📊 Service Status:"
echo "=================="
docker-compose ps

# Show access URLs
echo ""
echo "🌐 Access URLs:"
echo "==============="
echo "Dashboard:      http://localhost:8501"
echo "Model API:      http://localhost:8000"
echo "API Docs:       http://localhost:8000/docs"
echo ""

# Test API endpoint
echo "🧪 Testing API endpoint..."
if curl -s -f "http://localhost:8000/health" > /dev/null; then
    print_status "API is responding"
    
    # Show model info if available
    model_info=$(curl -s "http://localhost:8000/model-info" 2>/dev/null || echo "null")
    if [ "$model_info" != "null" ]; then
        echo "Current model info:"
        echo "$model_info" | python -m json.tool 2>/dev/null || echo "$model_info"
    fi
else
    print_warning "API not responding yet (may need more time)"
fi

echo ""
print_status "Deployment completed!"

echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Open dashboard: http://localhost:8501"
echo "2. Check API docs: http://localhost:8000/docs"
echo "3. Run federated learning if no models exist:"
echo "   docker-compose exec fl-server python fl_server.py"
echo "4. Monitor logs with: docker-compose logs -f"
echo ""

echo "🔧 Management Commands:"
echo "======================"
echo "Stop services:    docker-compose down"
echo "View logs:        docker-compose logs -f [service]"
echo "Restart service:  docker-compose restart [service]"
echo "Update code:      docker-compose up --build -d"
echo ""

print_status "Ready for MLOps! 🎉"