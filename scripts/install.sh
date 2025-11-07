#!/bin/bash

# MemoryAI Enterprise Universal Installer
# Supports Linux, macOS, Windows (via WSL), and mobile devices

set -e

echo "🚀 MemoryAI Enterprise Installation Starting..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Platform detection
OS="$(uname -s)"
ARCH="$(uname -m)"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check RAM
    if [[ "$OS" == "Linux" ]]; then
        RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        RAM_GB=$((RAM_KB / 1024 / 1024))
    elif [[ "$OS" == "Darwin" ]]; then
        RAM_BYTES=$(sysctl -n hw.memsize)
        RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    fi
    
    if [[ $RAM_GB -lt 8 ]]; then
        print_warning "System has only ${RAM_GB}GB RAM. 8GB minimum recommended"
    fi
    
    # Check disk space
    DISK_AVAIL=$(df . | tail -1 | awk '{print $4}')
    DISK_GB=$((DISK_AVAIL / 1024 / 1024))
    
    if [[ $DISK_GB -lt 10 ]]; then
        print_error "Insufficient disk space. 10GB minimum required"
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Install Docker and dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    case "$OS" in
        Linux*)
            if ! command -v docker &> /dev/null; then
                print_status "Installing Docker for Linux..."
                curl -fsSL https://get.docker.com -o get-docker.sh
                sh get-docker.sh
                sudo usermod -aG docker $USER
                rm get-docker.sh
            fi
            ;;
        Darwin*)
            if ! command -v docker &> /dev/null; then
                print_status "Please install Docker Desktop for macOS from:"
                print_status "https://docs.docker.com/desktop/install/mac-install/"
                exit 1
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            print_status "Please install Docker Desktop for Windows from:"
            print_status "https://docs.docker.com/desktop/install/windows-install/"
            exit 1
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    # Install docker-compose if not present
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_status "Installing docker-compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    print_success "Dependencies installed"
}

# Pull and start services
start_services() {
    print_status "Starting MemoryAI services..."
    
    # Create necessary directories
    mkdir -p data/{models,memory,logs,config}
    
    # Set proper permissions
    chmod -R 755 data/
    
    # Pull latest images
    if docker compose version &> /dev/null; then
        docker compose -f memoryai.local.yml pull
        docker compose -f memoryai.local.yml up -d
    else
        docker-compose -f memoryai.local.yml pull
        docker-compose -f memoryai.local.yml up -d
    fi
    
    print_success "Services started successfully"
}

# Build CLI tool
build_cli() {
    print_status "Building memoryaictl CLI tool..."
    
    if ! command -v go &> /dev/null; then
        print_warning "Go not found. CLI tool will not be built."
        print_status "Please install Go from https://golang.org/"
        return 0
    fi
    
    go build -o memoryaictl memoryaictl.go
    chmod +x memoryaictl
    sudo mv memoryaictl /usr/local/bin/
    
    print_success "CLI tool installed"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for health checks
    for i in {1..30}; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            print_success "Services are ready!"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    print_warning "Services may still be starting. Check logs with:"
    print_status "docker-compose -f memoryai.local.yml logs"
}

# Display access information
show_access_info() {
    print_success "MemoryAI Enterprise installation complete!"
    echo ""
    echo "🌐 Access URLs:"
    echo "  • Web UI: http://localhost:3000"
    echo "  • API: http://localhost:8080"
    echo "  • Monitoring: http://localhost:9090"
    echo ""
    echo "🔧 Management:"
    echo "  • CLI tool: memoryaictl --help"
    echo "  • Logs: docker-compose -f memoryai.local.yml logs -f"
    echo "  • Stop: docker-compose -f memoryai.local.yml down"
    echo ""
    echo "📱 Mobile Setup:"
    echo "  • Download MemoryAI app from app store"
    echo "  • Connect to: http://$(hostname -I | awk '{print $1}'):3000"
    echo ""
    echo "📚 Documentation:"
    echo "  • User Guide: https://docs.memoryai.com"
    echo "  • Developer Docs: https://dev.memoryai.com"
    echo "  • Support: support@memoryai.com"
}

# Main installation flow
main() {
    echo "MemoryAI Enterprise Universal Installer"
    echo "======================================"
    echo "Platform: $OS ($ARCH)"
    echo ""
    
    check_requirements
    install_dependencies
    start_services
    build_cli
    wait_for_services
    show_access_info
}

# Run main function
main "$@"