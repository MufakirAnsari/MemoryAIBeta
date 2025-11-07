# MemoryAI Enterprise - Local Air-Gap Deployment Guide

## 🚀 One-Command Deployment

Deploy the entire MemoryAI Enterprise stack locally in under 4 minutes:

```bash
# Clone and deploy
git clone https://github.com/memoryai/memoryai-enterprise.git
cd memoryai-enterprise

# Air-gap deployment (fully functional offline)
docker compose -f memoryai.local.yml up -d

# Verify deployment
docker compose -f memoryai.local.yml ps
```

## 📋 Prerequisites

- **Docker** 24.0+ with BuildKit
- **Docker Compose** 2.20+
- **8GB RAM** minimum (16GB recommended)
- **10GB disk space**
- **Architecture**: M1/x86_64/RISC-V

## 🔧 Quick Start

1. **Download models** (first time only):
```bash
./memoryaictl download-models --quantization q4_0
```

2. **Deploy stack**:
```bash
docker compose -f memoryai.local.yml up -d
```

3. **Access services**:
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8082
- **Monitoring**: http://localhost:3001 (admin/memoryai_admin)
- **Metrics**: http://localhost:9090

4. **Verify health**:
```bash
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8082/health
```

## 🔒 Security Features

- **Post-quantum signatures**: Dilithium-3 on all binaries
- **Immutable audit logs**: Tamper-proof operation history
- **Differential privacy**: ε≤0.5 for all user data
- **Zero-knowledge proofs**: Routing decision verification
- **GDPR/COPPA compliance**: Built-in privacy controls

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   Guardian      │    │   Federated     │
│   (Halo/Focus)  │    │   (Safety)      │    │   Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Suggest       │
                    │   Router        │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Core      │    │   Memory        │    │   PostgreSQL    │
│   (Mythic MP-30)│    │   RAG Fusion    │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Monitoring

- **Health checks**: 30s intervals
- **Metrics**: Prometheus + Grafana
- **Logs**: Centralized logging
- **Alerts**: Automatic failover

## 🛠️ Development

```bash
# Hot-reload development
docker compose -f memoryai.local.yml -f memoryai.dev.yml up

# Run tests
./memoryaictl test --all

# Generate SBOM
./memoryaictl sbom --format cyclonedx

# DP audit
./memoryaictl audit --epsilon 0.5
```

## 🔧 Troubleshooting

**Container won't start**:
```bash
docker logs memoryai-llm
docker system prune -f
```

**Out of memory**:
```bash
docker update --memory 8g memoryai-llm
```

**Model download issues**:
```bash
./memoryaictl download-models --force
```

## 📞 Support

- **Documentation**: /docs
- **Issues**: GitHub Issues
- **Security**: security@memoryai.ai
- **Enterprise**: enterprise@memoryai.ai

## 📄 License

AGPL-3.0 with commercial licensing available.