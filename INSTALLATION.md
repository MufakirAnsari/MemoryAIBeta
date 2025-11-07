# MemoryAI Enterprise Installation Guide

## Quick Start (All Devices)

### Prerequisites
- Docker Desktop 4.0+
- Git
- 8GB RAM minimum, 16GB recommended
- 10GB free disk space

### One-Command Installation
```bash
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
./scripts/install.sh
```

## Platform-Specific Installation

### 🐧 Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install -y docker.io docker-compose git curl

# Clone and run
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
sudo docker-compose -f memoryai.local.yml up -d
```

### 🍎 macOS (Intel/Apple Silicon)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker and dependencies
brew install docker docker-compose git

# Clone and run
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
docker-compose -f memoryai.local.yml up -d
```

### 🪟 Windows 10/11
```powershell
# Install Docker Desktop for Windows
# https://docs.docker.com/desktop/install/windows-install/

# Clone repository
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise

# Run with Docker Compose
docker-compose -f memoryai.local.yml up -d
```

### 📱 Mobile Devices

#### Android (via Termux)
```bash
# Install Termux from F-Droid or Play Store
pkg update && pkg upgrade
pkg install git docker-compose

# Clone and run
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
docker-compose -f memoryai.local.yml up -d
```

#### iOS (via iSH)
```bash
# Install iSH from App Store
apk add git docker-compose

# Clone and run (limited functionality on iOS)
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
```

## Advanced Installation Options

### Kubernetes Deployment
```bash
# For production Kubernetes clusters
kubectl apply -f k8s/
helm install memoryai ./charts/memoryai
```

### Air-Gap Deployment
```bash
# For offline environments
./scripts/airgap-package.sh
# Transfer memoryai-airgap.tar.gz to target system
./scripts/airgap-install.sh memoryai-airgap.tar.gz
```

### Development Mode
```bash
# For developers contributing to MemoryAI
git clone https://github.com/your-org/memoryai-enterprise.git
cd memoryai-enterprise
./scripts/dev-setup.sh
```

## Verification

After installation, verify the system is running:
```bash
# Check service status
./memoryaictl status

# Test AI capabilities
./memoryaictl test

# Access UI
open http://localhost:3000
```

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change ports in `memoryai.local.yml`
2. **Memory issues**: Increase Docker memory allocation
3. **Permission errors**: Run with `sudo` on Linux

### Support
- 📧 Email: support@memoryai.com
- 💬 Discord: https://discord.gg/memoryai
- 📚 Docs: https://docs.memoryai.com

## Next Steps
1. Configure your first AI assistant
2. Set up privacy preferences
3. Install VS Code extension
4. Connect mobile app