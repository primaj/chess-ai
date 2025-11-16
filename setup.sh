#!/bin/bash
# Setup script for Chess Transformer AI

set -e

echo "Setting up Chess Transformer AI..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.10..."
    # Try python3.10 first, fall back to python3 if not available
    if command -v python3.10 &> /dev/null; then
        python3.10 -m venv venv
    elif command -v /opt/homebrew/bin/python3.10 &> /dev/null; then
        /opt/homebrew/bin/python3.10 -m venv venv
    else
        echo "Warning: python3.10 not found. Using default python3."
        echo "Please install Python 3.10 for full compatibility."
        python3 -m venv venv
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

