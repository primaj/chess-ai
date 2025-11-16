# Chess Transformer AI - Proof of Concept

A transformer-based chess AI that learns to predict moves and evaluate positions from PGN game data.

## Overview

This project implements a `MiniChessTransformer` - a neural network that uses transformer architecture to:
- **Predict moves** (policy head): Given a chess position, predict the probability distribution over all possible moves
- **Evaluate positions** (value head): Estimate the expected game outcome from a given position

The model is trained on human chess games in PGN format, learning patterns from real gameplay.

## Architecture

The model consists of:
- **SquareEmbedding**: Encodes each of the 64 squares with piece type, square position, and side to move
- **TransformerBackbone**: Multi-layer transformer encoder that learns global dependencies across the board
- **PolicyHead**: Predicts move probabilities (logits over ~4000 possible UCI moves)
- **ValueHead**: Predicts position evaluation (scalar between -1 and +1)

**Model Configuration (POC):**
- Hidden dimension: 256
- Layers: 6
- Attention heads: 8
- Move vocabulary: 4096

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Quick setup (macOS/Linux):**
```bash
./setup.sh
```

This will create a virtual environment, activate it, and install all dependencies.

## Usage

### Training

Place your PGN files in the `data/` directory, then run:

```bash
python src/train.py --pgn_file data/your_games.pgn --epochs 5 --batch_size 32
```

Additional options:
- `--max_games`: Limit number of games to parse
- `--min_rating`: Filter games by minimum player rating
- `--hidden_dim`: Model hidden dimension (default: 256)
- `--n_layers`: Number of transformer layers (default: 6)
- `--lr`: Learning rate (default: 1e-4)
- `--val_split`: Validation split ratio (default: 0.1, i.e., 10%)
- `--num_workers`: Number of DataLoader workers (default: auto-detect, min(8, cpu_count))
- `--parse_workers`: Number of parallel workers for PGN parsing (default: 1, sequential)
- `--no_cache`: Disable caching of parsed PGN data (caching is enabled by default)
- `--cache_dir`: Directory for cache files (default: "cache")

The script will:
- Parse PGN games into training samples (cached for faster subsequent runs)
- Train the transformer model
- Save checkpoints to `models/` (best model and per-epoch checkpoints)

**Note:** Parsed PGN data is automatically cached to speed up subsequent training runs. Cache files are stored in the `cache/` directory and are keyed by the PGN file path and filtering parameters.

### Inference

Use the inference utilities to predict moves or evaluate positions:

```python
from src.inference import load_model, predict_move_from_legal, evaluate_position
import chess

model, info = load_model('models/minichess_transformer.pt')
board = chess.Board()

# Get move encoder from checkpoint (automatically loaded)
move_encoder = info.get('move_encoder')

# Predict top moves
top_moves = predict_move_from_legal(model, board, move_encoder=move_encoder, top_k=5)
for move, prob in top_moves:
    print(f"{move.uci()}: {prob:.4f}")

# Evaluate position
value = evaluate_position(model, board)
print(f"Position evaluation: {value:.4f} (from White's perspective)")
```

Or use the interactive CLI:

```bash
python src/inference.py --checkpoint models/minichess_transformer.pt --interactive
```

## Data

The model trains on PGN (Portable Game Notation) files. You can obtain chess games from:
- [Lichess Database](https://database.lichess.org/) - Downloads are in `.pgn.zst` format (Zstandard compressed)
- [Chess.com](https://www.chess.com/)
- [FICS](https://www.freechess.org/)

### Downloading from Lichess

Use the download utility to fetch PGN files directly:

```bash
# Download by date (YYYY-MM format)
python src/download.py 2025-10

# Download by full URL
python src/download.py https://database.lichess.org/standard/lichess_db_standard_rated_2025-10.pgn.zst

# Specify output directory
python src/download.py 2025-10 --output data/
```

The download utility supports:
- Resume interrupted downloads (automatically resumes if file exists)
- Progress bar with download speed
- Date-based shortcuts (e.g., `2025-10`)

**Note:** The code automatically handles both `.pgn` and `.pgn.zst` (compressed) files. You can use Lichess database files directly without decompressing them.

Place PGN files in the `data/` directory before training.

## Project Structure

```
chess-ai/
├── src/
│   ├── model.py              # MiniChessTransformer architecture
│   ├── data.py               # PGN parsing and dataset utilities
│   ├── train.py              # Training script
│   ├── inference.py          # Inference/evaluation utilities
│   └── download.py           # Lichess database download utility
├── data/                     # PGN files directory
├── models/                   # Saved model checkpoints
├── cache/                    # Cached parsed PGN data (auto-generated)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Training Details

The model uses supervised learning with two loss components:
- **Policy loss**: Cross-entropy between predicted move distribution and actual move
- **Value loss**: Mean squared error between predicted value and game result

Total loss = Policy Loss + 0.5 × Value Loss

**Improvements in v0.2.0:**
- **Validation Split**: Training now includes a validation set (10% by default) for proper model evaluation
- **Game Phase Filtering**: Value learning uses game phase filtering - early positions use neutral value to improve signal quality
- **Model Portability**: Checkpoints now include `MoveEncoder` state, making models fully portable

## Future Enhancements

- Self-play reinforcement learning (AlphaZero-style)
- Move history conditioning
- Multi-modal training with commentary
- MCTS integration for stronger play
- Larger model variants

## License

MIT

