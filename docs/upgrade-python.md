# Upgrading to Python 3.10

This guide will help you upgrade your project to use Python 3.10.

## Prerequisites

First, check if Python 3.10 is installed on your system:

```bash
python3.10 --version
```

If Python 3.10 is not installed, install it:

### macOS (using Homebrew)
```bash
brew install python@3.10
```

### macOS (using pyenv - Recommended)
```bash
# Install pyenv if you don't have it
brew install pyenv

# Install Python 3.10
pyenv install 3.10.13

# Set it as local version for this project
cd /Users/johnprimavesi/chess-ai
pyenv local 3.10.13
```

### Linux (Ubuntu/Debian and derivatives)

**Pop!_OS, Linux Mint, etc.** — Use the same instructions as Ubuntu; these distros are Ubuntu-based and support the same PPAs and packages.

**Option A: Default repositories (Ubuntu 22.04+)**  
If your distro ships Python 3.10:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**Option B: When Python 3.10 is not available (Ubuntu 20.04, some Debian, or minimal installs)**  
Use the deadsnakes PPA (Ubuntu) or pyenv.

**Ubuntu — deadsnakes PPA:**
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**Any Linux — pyenv (recommended if you need multiple Python versions):**
```bash
# Install pyenv (see https://github.com/pyenv/pyenv#installation)
curl https://pyenv.run | bash
# Restart shell or: source ~/.bashrc  (or ~/.zshrc)

pyenv install 3.10.13
cd /path/to/chess-ai
pyenv local 3.10.13
# Then: python3 -m venv venv  (uses pyenv’s 3.10)
```

**Note:** `python3.10-pip` is not required; use `python3.10 -m pip` or create a venv and use `pip` inside it.

## Upgrade Steps

### 1. Deactivate Current Virtual Environment

If you have a virtual environment active, deactivate it:

```bash
deactivate
```

### 2. Remove Old Virtual Environment

```bash
cd /Users/johnprimavesi/chess-ai
rm -rf venv
```

### 3. Create New Virtual Environment with Python 3.10

**Option A: Using python3.10 directly**
```bash
python3.10 -m venv venv
```

**Option B: Using pyenv (if you installed via pyenv)**
```bash
python3 -m venv venv
# This will use the pyenv version (3.10.13)
```

**Option C: Using the updated setup script**
```bash
./setup.sh
# The script has been updated to use Python 3.10
```

### 4. Activate New Virtual Environment

```bash
source venv/bin/activate
```

### 5. Verify Python Version

```bash
python --version
# Should output: Python 3.10.x
```

### 6. Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 7. Verify Installation

```bash
python -c "import torch; import gradio; import chess; print('All dependencies installed successfully!')"
```

## Troubleshooting

### Python 3.10 Not Found

If `python3.10` is not found, you may need to:

1. **Check Homebrew installation path:**
   ```bash
   /opt/homebrew/bin/python3.10 --version
   # or
   /usr/local/bin/python3.10 --version
   ```

2. **Create an alias or symlink:**
   ```bash
   # Add to ~/.zshrc or ~/.bashrc
   alias python3.10=/opt/homebrew/bin/python3.10
   ```

3. **Use pyenv (recommended for managing multiple Python versions):**
   ```bash
   brew install pyenv
   pyenv install 3.10.13
   pyenv local 3.10.13
   ```

### Virtual Environment Issues

If you encounter issues with the virtual environment:

1. Make sure you're using the correct Python version:
   ```bash
   which python3.10
   python3.10 -m venv --help
   ```

2. Try creating the venv with explicit path:
   ```bash
   /opt/homebrew/bin/python3.10 -m venv venv
   ```

### Dependency Installation Issues

If some packages fail to install:

1. Make sure pip is up to date:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Try installing packages individually to identify issues:
   ```bash
   pip install torch
   pip install gradio
   pip install gradio-chessboard
   # etc.
   ```

## Notes

- **gradio-chessboard** requires Python >=3.10, so upgrading is necessary for the UI to work
- All other dependencies should work fine with Python 3.10
- Your existing model checkpoints and data will work with the new Python version
- No code changes are needed - Python 3.10 is backward compatible with Python 3.9 code

## After Upgrade

Once upgraded, you can:

1. Run the training script:
   ```bash
   python src/train.py --pgn_file data/your_games.pgn
   ```

2. Launch the interactive UI:
   ```bash
   python src/ui.py
   ```

3. Use the Jupyter notebook:
   ```bash
   jupyter notebook train_notebook.ipynb
   ```


