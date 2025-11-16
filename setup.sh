#!/bin/bash
# Setup script for Chess Transformer AI

set -e

echo "Setting up Chess Transformer AI..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
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

