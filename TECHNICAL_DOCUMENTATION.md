# Chess Transformer AI - Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Training Mechanism](#training-mechanism)
5. [What the Model Learns](#what-the-model-learns)
6. [Inference Process](#inference-process)
7. [Critical Analysis](#critical-analysis)
8. [Potential Issues](#potential-issues)
9. [Areas for Improvement](#areas-for-improvement)

---

## Project Overview

This project implements a transformer-based chess AI that learns to predict moves and evaluate positions through supervised learning on human chess games. The model uses a dual-head architecture (policy + value) similar to AlphaZero, but with a transformer backbone instead of convolutional networks.

**Key Characteristics:**
- **Model Type**: Transformer encoder with dual output heads
- **Training Method**: Supervised learning on PGN game data
- **Input**: Chess board positions (64 squares)
- **Outputs**: Move probability distribution (policy) and position evaluation (value)
- **Model Size**: ~2-3M parameters (configurable, default: 256 hidden dim, 6 layers)

---

## Architecture Deep Dive

### Overall Structure

The model follows a three-stage architecture:

```
Input Board → Embedding Layer → Transformer Encoder → Dual Heads
                                                      ├─ Policy Head (move prediction)
                                                      └─ Value Head (position evaluation)
```

### Component Breakdown

#### 1. SquareEmbedding (`src/model.py:25-54`)

**Purpose**: Converts discrete board representation into continuous embeddings.

**Input Format**:
- `piece_ids`: `[batch, 64]` tensor where each element is an integer (0-13)
  - 0 = empty square
  - 1-6 = white pieces (P, N, B, R, Q, K)
  - 7-12 = black pieces (P, N, B, R, Q, K)
- `side_to_move`: `[batch]` tensor (0 = white, 1 = black)

**Embedding Strategy**:
The embedding combines three components additively:
1. **Piece Embedding**: Learned embeddings for each piece type (14 total: 12 pieces + empty + padding)
2. **Square Embedding**: Learned positional embeddings for each of the 64 squares
3. **Side Embedding**: Learned embedding for which side is to move (broadcasted to all squares)

**Mathematical Representation**:
```
E(square_i) = E_piece(piece_i) + E_square(i) + E_side(side_to_move)
```

**Critical Observation**: The square embeddings are **learned** rather than using chess-specific positional encodings (e.g., rank/file coordinates). This means the model must learn spatial relationships from scratch, which may be inefficient.

**Output**: `[batch, 64, hidden_dim]` tensor where each square has a dense representation.

#### 2. TransformerBackbone (`src/model.py:60-83`)

**Purpose**: Learns global dependencies and patterns across the board using self-attention.

**Architecture**:
- Standard PyTorch `TransformerEncoder` with `TransformerEncoderLayer`
- Default: 6 layers, 8 attention heads, 256 hidden dimension
- Feed-forward dimension: `hidden_dim * 4 = 1024`
- Activation: GELU
- Dropout: 0.1

**Attention Mechanism**:
The transformer applies self-attention over the 64 square tokens, allowing each square to attend to all other squares. This enables the model to learn:
- Piece coordination patterns
- Attack/defense relationships
- Long-range dependencies (e.g., rook on a-file seeing pieces on h-file)

**Key Limitation**: The transformer treats the board as a **sequence of 64 tokens** with no inherent 2D structure. While attention can learn spatial relationships, it lacks explicit geometric inductive bias that would be natural for chess (e.g., knight moves, diagonals).

#### 3. PolicyHead (`src/model.py:89-109`)

**Purpose**: Predicts the probability distribution over all possible moves.

**Architecture**:
- Mean pooling over all 64 square embeddings → `[batch, hidden_dim]`
- LayerNorm
- Linear layer: `hidden_dim → move_vocab_size`

**Output**: `[batch, move_vocab_size]` logits over moves

**Critical Issue**: The move vocabulary is **dynamically built** during data parsing (`MoveEncoder` class). This means:
- Different training runs may have different move vocabularies
- Move IDs are not consistent across models
- The vocabulary size depends on what moves appear in the training data
- Default `move_vocab=4096` is a placeholder that gets overridden

**Move Encoding**: Moves are encoded as UCI strings (e.g., "e2e4") and mapped to integer IDs. The vocabulary grows as new moves are encountered.

#### 4. ValueHead (`src/model.py:115-138`)

**Purpose**: Predicts position evaluation (expected game outcome from White's perspective).

**Architecture**:
- Mean pooling over all 64 square embeddings → `[batch, hidden_dim]`
- LayerNorm
- Two-layer MLP: `hidden_dim → hidden_dim/2 → 1`
- Tanh activation (output range: [-1, 1])
  - +1 = White wins
  - 0 = Draw
  - -1 = Black wins

**Output**: `[batch, 1]` scalar value

---

## Data Processing Pipeline

### PGN Parsing (`src/data.py:211-297`)

**Input**: PGN (Portable Game Notation) files, optionally compressed with Zstandard (.zst)

**Processing Steps**:

1. **File Reading**: Supports both `.pgn` and `.pgn.zst` files via `open_pgn_file()`
   - Zstandard files are decompressed on-the-fly
   - Uses context managers for proper resource cleanup

2. **Game Parsing**: Iterates through games using `chess.pgn.read_game()`
   - Filters by `max_games` and `min_rating` if specified
   - Extracts game result from headers

3. **Position Extraction**: For each game:
   - Starts with initial board position
   - For each move in the game:
     - Converts current board to tensor (`board_to_tensor()`)
     - Encodes the move played (`MoveEncoder.encode()`)
     - Records side to move (0 = white, 1 = black)
     - Records game outcome (1.0 = white wins, -1.0 = black wins, 0.0 = draw)
     - Pushes move to board for next position

4. **Sample Format**: Each training sample is a tuple:
   ```python
   (piece_ids: torch.Tensor[64], side: int, move_id: int, outcome: float)
   ```

**Critical Observation**: The outcome is **the same for all positions in a game**. This means:
- Early game positions get the same label as endgame positions
- A position from a game where White eventually won is always labeled +1.0, even if it's a losing position
- This is a **major limitation** - the value target is not position-specific

### Board Encoding (`src/data.py:74-88`)

**Process**:
1. Creates a `[64]` tensor of zeros
2. Iterates through all 64 squares (in chess.SQUARES order: a1, a2, ..., h8)
3. Maps each piece to an integer ID:
   - Empty square → 0
   - White pieces → 1-6 (P, N, B, R, Q, K)
   - Black pieces → 7-12 (P, N, B, R, Q, K)

**Square Ordering**: Uses `chess.SQUARES` which is a flat list. The model must learn that squares are arranged in an 8x8 grid through attention, as there's no explicit 2D structure.

### Move Encoding (`src/data.py:45-68`)

**Dynamic Vocabulary Building**:
- `MoveEncoder` builds vocabulary on-the-fly during parsing
- Each unique UCI move string gets assigned a sequential integer ID
- Vocabulary size depends on training data (typically 1000-4000 moves)

**Issues**:
1. **Inconsistency**: Different training runs produce different move vocabularies
2. **No Standard Mapping**: Move IDs don't correspond to any canonical ordering
3. **Inference Limitation**: Requires the same `MoveEncoder` instance used during training to decode predictions

### Caching (`src/data.py:132-205`)

**Purpose**: Avoid re-parsing PGN files on subsequent training runs.

**Implementation**:
- Cache key includes: PGN file path, `max_games`, `min_rating`
- Uses MD5 hash of cache key for filename
- Stores: samples list and `MoveEncoder` state (pickled)
- Cache files saved in `cache/` directory

**Benefits**: Significantly speeds up subsequent training runs (parsing can take hours for large files).

---

## Training Mechanism

### Loss Function (`src/train.py:39-77`)

The model uses a **multi-task learning** approach with two loss components:

1. **Policy Loss**: Cross-entropy between predicted move distribution and actual move
   ```python
   loss_policy = CrossEntropyLoss(policy_logits, actual_move_id)
   ```

2. **Value Loss**: Mean squared error between predicted value and game outcome
   ```python
   loss_value = MSELoss(value_pred, game_outcome)
   ```

3. **Combined Loss**:
   ```python
   total_loss = loss_policy + 0.5 * loss_value
   ```

The 0.5 weighting on value loss is arbitrary and not tuned. This suggests value learning may be under-emphasized.

### Training Loop (`src/train.py:80-136`)

**Process**:
1. For each epoch:
   - Iterate through batches
   - Forward pass: `policy_logits, value_pred = model(piece_ids, side)`
   - Compute losses
   - Backward pass with gradient clipping (max_norm=1.0)
   - Update parameters via AdamW optimizer

2. **Checkpointing**:
   - Saves checkpoint after each epoch
   - Saves "best" model if validation loss improves (but validation is currently disabled - `val_loader = None`)

**Optimizer**: AdamW with learning rate 1e-4, weight decay 1e-5

**Gradient Clipping**: Applied with max_norm=1.0 to prevent exploding gradients

### Critical Training Issues

1. **No Validation Split**: The code sets `val_loader = None` (line 259), meaning:
   - No validation during training
   - No early stopping
   - No model selection based on validation performance
   - "Best model" saving is never triggered

2. **Value Target Problem**: All positions from a game share the same outcome label:
   - A position where White is clearly losing but eventually wins still gets +1.0 label
   - This teaches the model incorrect position evaluations
   - Should use position-specific evaluations (e.g., from Stockfish) or at least filter by game phase

3. **No Data Augmentation**: 
   - No board rotations/flips
   - No position mirroring
   - Could double training data with minimal effort

4. **Move Vocabulary Mismatch**: 
   - Model is initialized with `move_vocab=4096` (default)
   - Actual vocabulary size is determined during parsing
   - Model is recreated with correct size, but this is inefficient

---

## What the Model Learns

### Policy Learning (Move Prediction)

**Objective**: Learn to predict which move a human player would make in a given position.

**What It Learns**:
- Opening patterns (e.g., e4, d4, Nf3 are common first moves)
- Tactical patterns (captures, checks, threats)
- Positional patterns (development, central control)
- Endgame patterns (if endgame positions are in training data)

**Limitations**:
- Learns from **human play**, not optimal play
- May learn suboptimal patterns if training data contains weak players
- No explicit chess knowledge (rules, piece values, etc.) - must learn everything from data
- Cannot reason about move legality - outputs distribution over all moves, including illegal ones

**Training Signal**: The model sees millions of (position, move) pairs and learns: "In positions similar to this, humans typically play this move."

### Value Learning (Position Evaluation)

**Objective**: Learn to estimate the expected game outcome from a position.

**What It Learns**:
- Positional advantages (material, piece activity, king safety)
- Strategic patterns (weak squares, pawn structure)
- Tactical patterns (threats, combinations)

**Critical Problem**: The training signal is **game outcome**, not position evaluation:
- A position from move 5 of a game where White eventually won gets label +1.0
- But that position might actually be equal or even slightly worse for White
- The model learns: "Positions from games White won tend to be good for White" (which is true on average but wrong for individual positions)

**Better Approach**: Should use:
- Position-specific evaluations (e.g., Stockfish scores)
- Or at least filter by game phase (early/mid/endgame)
- Or use temporal difference learning (predict outcome from current position, not from game start)

### Attention Patterns

The transformer's attention mechanism learns to:
- Connect related squares (e.g., pieces attacking the same square)
- Identify piece coordination (e.g., rooks on same rank/file)
- Recognize patterns (e.g., castling structure, pawn chains)

However, without explicit 2D structure, the model must learn these relationships from scratch, which is inefficient compared to architectures with spatial inductive bias (e.g., CNNs or graph neural networks).

---

## Inference Process

### Model Loading (`src/inference.py:19-71`)

**Process**:
1. Load checkpoint file (PyTorch `.pt` format)
2. Extract model configuration from checkpoint
3. Recreate model architecture with saved config
4. Load model weights
5. Set model to evaluation mode

**Device Selection**: Automatically detects CUDA, MPS (Apple Silicon), or falls back to CPU.

### Move Prediction (`src/inference.py:77-173`)

**Process**:
1. Convert board to tensor: `board_to_tensor(board)` → `[64]`
2. Add batch dimension: `[1, 64]`
3. Get side to move: `0` (white) or `1` (black)
4. Forward pass: `policy_logits, _ = model(piece_ids, side)`
5. Apply softmax: `policy_probs = softmax(policy_logits)`
6. Get top-k moves: `topk(policy_probs, k)`

**Critical Limitation**: The `predict_move_from_legal()` function (lines 128-173) has a **major bug**:
- It gets policy logits from the model
- But then **ignores them** and returns uniform probabilities over legal moves
- The comment says "This is a placeholder - real implementation needs move_encoder"
- This means move prediction **doesn't actually work** without the training `MoveEncoder`

**To Fix**: Need to:
1. Save `MoveEncoder` with model checkpoint
2. Load it during inference
3. Map move IDs to UCI strings
4. Filter to legal moves only
5. Renormalize probabilities

### Position Evaluation (`src/inference.py:179-208`)

**Process**:
1. Convert board to tensor
2. Forward pass: `_, value_pred = model(piece_ids, side)`
3. Extract scalar value: `value = value_pred[0, 0]`
4. Adjust for perspective: If Black to move, flip sign (since model predicts from side-to-move perspective)

**Output**: Float in [-1, 1] representing expected outcome from White's perspective.

---

## Critical Analysis

### Strengths

1. **Clean Architecture**: Well-structured code with clear separation of concerns
2. **Caching**: Efficient caching system for parsed data
3. **Compression Support**: Handles Zstandard-compressed PGN files
4. **Device Flexibility**: Supports CUDA, MPS, and CPU
5. **Modular Design**: Easy to modify and extend

### Weaknesses

1. **No Validation**: Training has no validation split, making it impossible to:
   - Detect overfitting
   - Select best model
   - Tune hyperparameters properly

2. **Value Learning Signal**: Using game outcome as position evaluation is fundamentally flawed:
   - Early positions get same label as endgame positions
   - Model learns average outcome, not position-specific evaluation
   - Should use position-specific evaluations or filtered data

3. **Move Vocabulary Issues**:
   - Dynamic vocabulary makes model portability difficult
   - Inference requires training-time `MoveEncoder`
   - No standard move representation

4. **Inference Broken**: `predict_move_from_legal()` doesn't actually use model predictions

5. **No Spatial Inductive Bias**: 
   - Treats board as flat sequence
   - Must learn 2D relationships from scratch
   - Less efficient than architectures with spatial awareness

6. **No Move Legality Checking**: 
   - Model outputs distribution over all moves (including illegal)
   - Must filter to legal moves post-hoc
   - Wastes model capacity on impossible moves

7. **Limited Model Size**: 
   - Default 256 hidden dim, 6 layers is quite small
   - May limit learning capacity for complex patterns
   - But appropriate for POC

8. **No Data Augmentation**: 
   - Missing opportunity to double training data
   - No board symmetries exploited

9. **Batch Size**: 
   - Default 32 is conservative
   - Could likely increase for better GPU utilization

10. **Loss Weighting**: 
    - 0.5 weight on value loss is arbitrary
    - Should be tuned or learned

---

## Potential Issues

### 1. Memory Issues with Large Datasets

**Problem**: Loading all samples into memory (`samples` list) can cause OOM errors with large PGN files.

**Current Behavior**: `pgn_to_samples()` loads all samples into a Python list before creating the dataset. For 500k positions, this could be several GB of RAM.

**Solution**: Use streaming dataset that reads from PGN on-the-fly, or use memory-mapped arrays.

### 2. Move Vocabulary Explosion

**Problem**: If training on diverse data, move vocabulary can grow very large (10k+ moves), making the policy head huge.

**Current Behavior**: Vocabulary grows unbounded during parsing.

**Solution**: 
- Pre-define vocabulary (all legal UCI moves ≈ 4000)
- Or use move decomposition (from-square, to-square, promotion)

### 3. Inconsistent Move IDs

**Problem**: Different training runs produce different move vocabularies, making model comparison difficult.

**Solution**: Use canonical move ordering (e.g., all possible UCI moves in lexicographic order).

### 4. Value Head Learning Wrong Signal

**Problem**: Value head learns "average outcome of games containing this position" rather than "evaluation of this position."

**Impact**: Model will be poor at position evaluation, especially in early/mid-game.

**Solution**: Use position-specific evaluations or filter training data by game phase.

### 5. No Handling of Special Positions

**Problem**: Model doesn't explicitly handle:
- Check/checkmate
- Stalemate
- Draw by repetition/50-move rule
- Castling rights
- En passant

**Current Behavior**: These are implicitly encoded in the board state, but model may not learn them well.

### 6. Inference Requires Training MoveEncoder

**Problem**: Can't use a trained model without the `MoveEncoder` from training, making model portability difficult.

**Solution**: Save `MoveEncoder` with checkpoint, or use canonical move vocabulary.

### 7. No Gradient Accumulation

**Problem**: With small batch sizes, gradient updates may be noisy.

**Solution**: Implement gradient accumulation for effective larger batch sizes.

### 8. MPS Compatibility

**Problem**: MPS (Apple Silicon GPU) may have compatibility issues with certain operations.

**Current Status**: Code checks for MPS but hasn't been tested extensively.

---

## Areas for Improvement

### High Priority

1. **Fix Inference**: Implement proper move prediction that uses model outputs
2. **Add Validation Split**: Implement proper train/val split and validation loop
3. **Fix Value Learning**: Use position-specific evaluations or filter by game phase
4. **Save MoveEncoder**: Include `MoveEncoder` in model checkpoints
5. **Add Move Legality**: Filter illegal moves in policy head or use legal-move-only vocabulary

### Medium Priority

6. **Spatial Inductive Bias**: 
   - Add 2D positional encodings (rank/file coordinates)
   - Or use Vision Transformer-style patch embeddings
   - Or use graph neural network for piece relationships

7. **Data Augmentation**: 
   - Board rotations/flips
   - Color inversion
   - Could double training data

8. **Canonical Move Vocabulary**: 
   - Pre-define all legal UCI moves
   - Makes models portable and comparable

9. **Larger Model**: 
   - Increase hidden dimension to 512 or 1024
   - Add more layers (8-12)
   - Use more attention heads

10. **Better Loss Weighting**: 
    - Tune value loss weight
    - Or use learned weighting
    - Or separate optimizers for policy/value

### Low Priority

11. **Mixed Precision Training**: Use FP16/BF16 for faster training
12. **Gradient Accumulation**: For effective larger batch sizes
13. **Learning Rate Scheduling**: Cosine annealing or warmup
14. **Regularization**: Add dropout tuning, weight decay tuning
15. **Monitoring**: Add TensorBoard logging, better metrics
16. **Distributed Training**: Multi-GPU support for larger models
17. **MCTS Integration**: Use model as policy/value network in MCTS search
18. **Self-Play**: Add reinforcement learning component (AlphaZero-style)

### Architectural Improvements

19. **Move History**: Add recent move sequence as input (for opening/plan awareness)
20. **Game Phase Encoding**: Explicit encoding of opening/middlegame/endgame
21. **Piece-Square Tables**: Add learnable piece-square value embeddings
22. **Attention Visualization**: Tools to see what patterns model learns
23. **Multi-Scale Features**: Combine local (piece-level) and global (board-level) features

---

## Conclusion

This is a solid proof-of-concept implementation of a transformer-based chess AI. The architecture is clean and the code is well-structured. However, there are several critical issues that limit its effectiveness:

1. **Value learning uses wrong signal** (game outcome vs. position evaluation)
2. **Inference is broken** (move prediction doesn't use model outputs)
3. **No validation** (can't detect overfitting or select best model)
4. **Move vocabulary issues** (inconsistent, requires training-time encoder)

The model will likely learn basic move patterns and opening theory, but will struggle with:
- Accurate position evaluation (especially in early/mid-game)
- Tactical calculation
- Endgame play
- Long-term planning

For a POC, this is acceptable, but for a production chess engine, significant improvements are needed, particularly around the value learning signal and inference implementation.

The transformer architecture is interesting for chess, but may not be optimal compared to:
- CNNs with spatial inductive bias (like AlphaZero)
- Graph neural networks that explicitly model piece relationships
- Hybrid architectures combining transformers with chess-specific components

Nonetheless, this project provides a good foundation for experimentation and can be improved incrementally by addressing the issues outlined above.

