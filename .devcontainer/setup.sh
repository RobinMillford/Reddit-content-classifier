#!/bin/bash

# Reddit Content Classifier - Dev Container Setup Script
# This script sets up the development environment for the multi-label ML pipeline

echo "🚀 Setting up Reddit Content Classifier development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install additional system dependencies for ML development
echo "🔧 Installing system dependencies..."
sudo apt-get install -y -qq \
    build-essential \
    curl \
    wget \
    unzip \
    tree \
    htop \
    vim \
    nano

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Ensure Git LFS is properly configured
echo "📁 Configuring Git LFS..."
git lfs install
git config --global init.defaultBranch main

# Pull any existing LFS files
echo "⬇️ Pulling Git LFS files..."
if git lfs ls-files | grep -q .; then
    git lfs pull
    echo "✅ Model files downloaded from Git LFS"
else
    echo "ℹ️ No LFS files found (this is normal for fresh clones)"
fi

# Create useful aliases
echo "⚡ Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Reddit Content Classifier Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Project-specific aliases
alias run-app='streamlit run app.py --server.port 8501 --server.address 0.0.0.0'
alias train-models='python src/train.py'
alias ingest-data='python src/ingest_data.py'
alias check-models='ls -lh *.pkl *.joblib 2>/dev/null || echo "No model files found"'
alias show-git-lfs='git lfs ls-files'

# Quick development commands
alias dev-status='echo "=== Git Status ===" && git status && echo -e "\n=== Model Files ===" && ls -lh *.pkl *.joblib 2>/dev/null || echo "No model files found"'
alias dev-info='echo "Reddit Content Classifier Dev Environment" && echo "Python: $(python3 --version)" && echo "Git LFS: $(git lfs version)" && echo "Streamlit: $(streamlit version)"'

EOF

# Set proper permissions
chmod +x ~/.bashrc

# Create development workspace structure if it doesn't exist
echo "📂 Setting up workspace structure..."
mkdir -p {logs,temp,notebooks}

# Create a quick development info file
cat > DEV_QUICK_START.md << 'EOF'
# 🚀 Reddit Content Classifier - Quick Development Guide

## 🏃‍♂️ Quick Start Commands

```bash
# Run the Streamlit application
run-app

# Train new models
train-models

# Ingest fresh Reddit data (requires API credentials)
ingest-data

# Check model files status
check-models

# Show Git LFS tracked files
show-git-lfs

# Development status overview
dev-status

# Environment information
dev-info
```

## 📁 Important Files

- `app.py` - Main Streamlit application
- `src/train.py` - Model training pipeline
- `src/ingest_data.py` - Reddit data ingestion
- `champion_model.pkl` - Binary classification model (Git LFS)
- `multi_label_model.pkl` - Multi-label classification model (Git LFS)
- `vectorizer.joblib` - Text vectorizer (Git LFS)
- `model_metadata.joblib` - Model metadata (Git LFS)

## 🔧 Development Workflow

1. **Local Development**: Make code changes and test with `run-app`
2. **Model Training**: Use `train-models` to retrain models locally
3. **Testing**: Run tests with `python -m pytest` (if test files exist)
4. **Deployment**: Push to main branch triggers GitHub Actions MLOps pipeline

## 🌐 Access Points

- **Streamlit App**: http://localhost:8501
- **VS Code**: Available in the dev container
- **Terminal**: Full access to all development tools

## 📚 Documentation

- Project README: `README.md`
- MLOps Pipeline: `.github/workflows/main.yml`
- Multi-label Implementation: See `src/` directory

Happy coding! 🎉
EOF

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  • Run 'run-app' to start the Streamlit application"
echo "  • Run 'dev-info' to see environment details"
echo "  • Run 'dev-status' to check project status"
echo "  • See 'DEV_QUICK_START.md' for detailed development guide"
echo ""
echo "🌐 Streamlit will be available at: http://localhost:8501"
echo ""