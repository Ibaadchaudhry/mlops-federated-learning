# GitHub Actions CI/CD Guide for MLOps Federated Learning

## Overview

This repository includes a comprehensive CI/CD pipeline for automated MLOps workflows with federated learning. The pipeline includes code quality checks, automated testing, Docker builds, model training automation, deployment, and continuous monitoring.

## Workflows

### 1. Main CI/CD Pipeline (`mlops-ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual dispatch with environment selection

**Jobs:**
- **Code Quality & Security**: Black, isort, Flake8, Bandit, Safety checks
- **Unit Tests**: Multi-version Python testing with coverage
- **Docker Build & Scan**: Container builds with Trivy security scanning
- **Model Training & Validation**: Automated FL training with quality gates
- **Deployment**: Environment-specific deployment (staging/production)
- **Notification**: Status reporting

### 2. Model Training Automation (`model-training.yml`)

**Triggers:**
- Changes to model/training code
- Changes to data pipeline
- Manual dispatch with parameters
- Weekly schedule (Sundays 2 AM UTC)

**Features:**
- **Smart Training Triggers**: Only retrain when needed (model age, drift, performance)
- **Configurable Parameters**: Rounds, clients, data refresh options
- **Quality Validation**: Performance gates and model validation
- **Artifact Management**: Model versioning and backup

### 3. Performance Monitoring (`monitoring.yml`)

**Triggers:**
- Every 6 hours (scheduled)
- After deployments
- Manual dispatch with monitoring options

**Monitoring Types:**
- **Health Monitoring**: Service availability and responsiveness
- **Performance Monitoring**: Model accuracy, AUC, trend analysis
- **Drift Monitoring**: Feature-wise drift detection and alerting
- **Alert Management**: Automated responses to performance/drift issues

## Setup Instructions

### 1. Repository Secrets

Configure the following secrets in your GitHub repository:

```bash
# Repository Settings > Secrets and variables > Actions

# Required secrets:
GITHUB_TOKEN          # Automatically provided by GitHub
```

### 2. Branch Protection

Configure branch protection for `main`:

```yaml
# Settings > Branches > Add rule
Branch name pattern: main
Restrictions:
  - Require pull request reviews before merging
  - Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Include administrators
```

### 3. Environment Configuration

Set up environments in your repository:

```yaml
# Settings > Environments

staging:
  - Protection rules: None
  - Environment secrets: (none required)

production:
  - Protection rules: Required reviewers
  - Environment secrets: (production-specific if needed)
```

## Usage Guide

### Manual Workflow Triggers

#### 1. Main CI/CD Pipeline
```bash
# Via GitHub UI: Actions > MLOps Federated Learning CI/CD > Run workflow
# Options:
# - force_retrain: true/false
# - environment: staging/production
```

#### 2. Model Training
```bash
# Via GitHub UI: Actions > Model Training Automation > Run workflow
# Options:
# - training_rounds: "10" (default)
# - client_count: "3" (default)
# - force_data_refresh: true/false
```

#### 3. Performance Monitoring
```bash
# Via GitHub UI: Actions > Performance Monitoring & Alerts > Run workflow
# Options:
# - check_type: full/performance/drift/health
# - alert_threshold: "70" (accuracy percentage)
```

### Automated Triggers

#### Code Changes
- **Push to main**: Full CI/CD pipeline
- **Pull Request**: Code quality + testing only
- **Model/Data changes**: Automatic retraining evaluation

#### Scheduled Operations
- **Weekly Training**: Sundays 2 AM UTC (if conditions met)
- **Continuous Monitoring**: Every 6 hours
- **Health Checks**: After each deployment

### Quality Gates

#### Code Quality
- **Black formatting**: Must pass
- **Import sorting (isort)**: Must pass
- **Linting (Flake8)**: Must pass
- **Security (Bandit)**: Warnings only
- **Dependencies (Safety)**: Warnings only

#### Testing
- **Unit test coverage**: 70% minimum
- **Multi-version testing**: Python 3.9, 3.10, 3.11
- **Integration tests**: Client data and model validation

#### Model Quality
- **Accuracy threshold**: 65% minimum (configurable)
- **AUC threshold**: 0.70 minimum
- **Performance stability**: Low variance check
- **Trend analysis**: Declining performance detection

#### Security
- **Container scanning**: Trivy vulnerability assessment
- **Dependency checking**: Known vulnerability detection
- **Code security**: Bandit static analysis

### Monitoring & Alerting

#### Performance Alerts
Triggered when:
- Model accuracy < 70% (configurable)
- Performance declining trend (> -0.01 per round)
- High performance variance (std > 0.05)

#### Drift Alerts
Triggered when:
- Feature drift ratio > 20% (configurable)
- No drift reports available
- Significant distribution changes detected

#### Automatic Responses
- **High drift (>30%)**: Automatic retraining trigger
- **Performance issues**: Alert generation and reporting
- **Service health**: Health check reporting

### Artifact Management

#### Training Artifacts
- **Models**: `models/global_model_round_X.pt`
- **Metrics**: `models/metrics_history.json`
- **Drift Reports**: `drift_reports/drift_round_X_client_Y.csv`
- **Retention**: 30 days for training runs

#### Reports
- **Performance Reports**: Model metrics and trends
- **Drift Reports**: Feature-wise drift analysis
- **Training Summaries**: Training run details
- **Monitoring Summaries**: System health overview

### Development Workflow

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-functionality

# Make changes
# ... code changes ...

# Run local tests
pytest tests/

# Create pull request
# GitHub Actions will run code quality + tests
```

#### 2. Model Updates
```bash
# Update model architecture or training
# Changes to: model.py, fl_server.py, fl_client.py, train_utils.py

# Push to main
git push origin main

# Automatic evaluation for retraining will trigger
```

#### 3. Production Deployment
```bash
# Manual deployment via GitHub Actions UI
# Actions > MLOps Federated Learning CI/CD > Run workflow
# Select environment: production
```

## Troubleshooting

### Common Issues

#### 1. Tests Failing
```bash
# Check test logs in Actions tab
# Run locally: pytest tests/ -v
# Fix failing tests and push updates
```

#### 2. Docker Build Issues
```bash
# Check Dockerfile syntax
# Verify requirements.txt dependencies
# Check build logs in Actions tab
```

#### 3. Model Training Failures
```bash
# Check available resources
# Verify client_datasets.pkl exists
# Review training logs for memory/timeout issues
```

#### 4. Deployment Issues
```bash
# Check service health endpoints
# Verify Docker images are built
# Review deployment logs
```

### Performance Optimization

#### 1. Faster Builds
- Use Docker layer caching
- Optimize requirements.txt (pin versions)
- Use smaller base images

#### 2. Efficient Testing
- Run tests in parallel
- Use test data fixtures
- Cache Python dependencies

#### 3. Training Optimization
- Adjust timeout limits for large models
- Use appropriate resource allocation
- Monitor training convergence

## Best Practices

### 1. Code Quality
- Write comprehensive unit tests
- Follow PEP 8 style guidelines
- Add type hints to functions
- Document complex algorithms

### 2. Model Development
- Version control model architectures
- Track hyperparameters
- Validate model performance
- Monitor for overfitting

### 3. Deployment
- Use environment-specific configurations
- Implement proper health checks
- Monitor resource usage
- Plan for rollback scenarios

### 4. Security
- Regularly update dependencies
- Scan for vulnerabilities
- Use secrets management
- Implement access controls

## Monitoring Dashboard

Access the monitoring dashboard through:
- **Staging**: http://localhost:8601
- **Production**: http://localhost:8501
- **API**: http://localhost:8000/docs

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review troubleshooting section
3. Create GitHub issues for bugs
4. Check model performance in dashboard