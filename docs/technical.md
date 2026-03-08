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

**Positional Encoding Options**:
- **Default (learned)**: Uses learned embeddings for each of the 64 squares
- **2D Positional Encodings** (enabled with `use_2d_pos_encoding=True`):
  - Replaces learned square embeddings with explicit rank/file coordinate encodings
  - Options: `'learned'` (learned rank/file embeddings), `'sinusoidal'` (sinusoidal encodings), or `'2d_coords'` (explicit coordinates)
  - Formula: `E(square) = E_piece + E_rank(rank) + E_file(file) + E_side`
  - Provides explicit 2D structure awareness

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

**Spatial Inductive Bias Options**:
The model now supports three approaches to add spatial awareness (all optional, can be combined):
1. **2D Positional Encodings**: See `SquareEmbedding` above
2. **Patch Embeddings**: Vision Transformer-style convolutional processing (see below)
3. **Graph Neural Network**: Explicit piece relationship modeling (see below)

#### 2a. PatchEmbedding (`src/model.py:157-234`) - Optional

**Purpose**: Adds Vision Transformer-style spatial processing using 2D convolutions.

**Architecture**:
- Reshapes board from `[batch, 64, hidden_dim]` to `[batch, 8, 8, hidden_dim]`
- Applies 2D convolutions with configurable kernel size (default: 3x3)
- Three-layer conv network: `1 → hidden_dim//4 → hidden_dim//2 → hidden_dim`
- Uses residual connection: `output = input + conv(input)`

**Benefits**:
- Captures local spatial patterns (knight moves, diagonals, adjacent squares)
- Explicit 2D convolution operations
- Natural for chess geometry

**Configuration**: Enable with `--use_patch_embeddings` flag. Options: `--patch_size`, `--conv_kernel`

#### 2b. ChessGNN (`src/model.py:236-310`) - Optional

**Purpose**: Models board as graph where nodes are squares and edges represent spatial relationships.

**Graph Construction**:
- Nodes: 64 squares
- Edges connect squares that are:
  - On same rank (row)
  - On same file (column)
  - On same diagonal
  - Within knight move distance
- Normalized adjacency matrix for message passing

**Architecture**:
- Multiple GNN layers (configurable, default: 2)
- Each layer: message passing → linear transformation → GELU → LayerNorm
- Residual connections between layers

**Benefits**:
- Explicitly models piece relationships (ranks, files, diagonals, knight moves)
- Natural for chess (pieces interact through spatial relationships)
- Can capture long-range dependencies through graph edges

**Configuration**: Enable with `--use_gnn` flag. Options: `--gnn_layers`

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

## Spatial Inductive Bias Features

The model now supports three complementary approaches to add spatial awareness, addressing the limitation that transformers treat the board as a flat sequence. All features are optional and can be combined.

### 1. 2D Positional Encodings

**Implementation**: `SquareEmbedding` class supports rank/file coordinate encodings.

**Usage**: Enable with `--use_2d_pos_encoding` and choose encoding type:
- `--pos_encoding_type learned`: Learned embeddings for rank (0-7) and file (0-7)
- `--pos_encoding_type sinusoidal`: Sinusoidal positional encodings for rank/file
- `--pos_encoding_type 2d_coords`: Explicit coordinate embeddings

**Benefits**:
- Explicit encoding of 2D structure (ranks and files)
- Model immediately understands rank/file relationships
- Minimal architecture change
- Backward compatible (default: learned square embeddings)

**Example**:
```bash
python src/train.py --pgn_file data/games.pgn --use_2d_pos_encoding --pos_encoding_type learned
```

### 2. Vision Transformer-Style Patch Embeddings

**Implementation**: `PatchEmbedding` class applies 2D convolutions to capture local spatial patterns.

**Usage**: Enable with `--use_patch_embeddings`
- `--patch_size`: Patch size (default: 2, not used with conv)
- `--conv_kernel`: Convolution kernel size (default: 3)

**Architecture**:
- Reshapes board to 8x8 spatial layout
- Applies 3-layer 2D convolution network
- Uses residual connection to combine with original embeddings

**Benefits**:
- Explicit 2D convolution operations capture local patterns
- Natural for capturing knight moves, diagonals, adjacent squares
- Can be combined with other approaches

**Example**:
```bash
python src/train.py --pgn_file data/games.pgn --use_patch_embeddings --conv_kernel 3
```

### 3. Graph Neural Network

**Implementation**: `ChessGNN` class models board as graph with spatial relationships.

**Usage**: Enable with `--use_gnn`
- `--gnn_layers`: Number of GNN layers (default: 2)

**Graph Structure**:
- Nodes: 64 squares
- Edges: Connect squares on same rank, file, diagonal, or knight move distance
- Normalized adjacency matrix for message passing

**Benefits**:
- Explicitly models piece relationships (ranks, files, diagonals, knight moves)
- Natural for chess (pieces interact through spatial relationships)
- Can capture long-range dependencies through graph edges

**Example**:
```bash
python src/train.py --pgn_file data/games.pgn --use_gnn --gnn_layers 2
```

### Combining Approaches

All three approaches can be combined for maximum spatial awareness:

```bash
python src/train.py --pgn_file data/games.pgn \
    --use_2d_pos_encoding --pos_encoding_type learned \
    --use_patch_embeddings --conv_kernel 3 \
    --use_gnn --gnn_layers 2
```

**Processing Order**:
1. Initial embedding with optional 2D positional encodings
2. Optional patch embeddings (convolutional processing)
3. Optional GNN processing (graph message passing)
4. Transformer backbone (global attention)
5. Policy and value heads

**Backward Compatibility**: All features are disabled by default. Existing checkpoints load correctly with default settings.

---

## Data Processing Pipeline

### PGN Parsing (`src/data.py`)

**Input**: PGN (Portable Game Notation) files, optionally compressed with Zstandard (.zst)

**Processing Steps**:

1. **File Reading**: Supports both `.pgn` and `.pgn.zst` files via `open_pgn_file()`
   - Zstandard files are decompressed on-the-fly
   - Uses context managers for proper resource cleanup

2. **Game Parsing**: Two parsing modes depending on `num_parse_workers`:
   - **Sequential** (`_parse_sequential`): Single-pass — reads and parses from the PGN file in one go. No intermediate PGN string storage.
   - **Parallel** (`_parse_parallel`): Two-phase — reads all games as PGN strings first, then distributes chunks to worker processes via `multiprocessing.Pool`. Workers return raw samples (numpy arrays + UCI strings) without move encoding, eliminating the need for per-worker vocabularies and the old merge/remap step.

3. **Position Extraction**: For each game:
   - Starts with initial board position
   - For each move in the game:
     - Converts current board to numpy int8 array via `board_to_array()` (uses `piece_map()` to iterate only occupied squares)
     - Records move as UCI string (move encoding is deferred)
     - Records side to move (0 = white, 1 = black) as numpy int8
     - Records game outcome as numpy float32 (with game-phase filtering)
     - Pushes move to board for next position

4. **Tensorisation** (`_tensorise`): After all parsing is complete, raw samples are converted into 4 contiguous tensors in a single pass:
   ```python
   piece_ids: torch.Tensor[N, 64]  # int64 — ready for embedding lookup
   sides:     torch.Tensor[N]      # int64
   moves:     torch.Tensor[N]      # int64 — encoded via MoveEncoder
   values:    torch.Tensor[N]      # float32
   ```
   This replaces the old list-of-tuples format which created millions of individual Python objects and separate tensor allocations.

5. **Return Format**: `pgn_to_samples()` returns `(tensor_dict, MoveEncoder)` where `tensor_dict` is a dict of the 4 contiguous tensors above.

**Critical Observation**: The outcome is **the same for all positions in a game**. This means:
- Early game positions get the same label as endgame positions (mitigated by game-phase filtering which sets early positions to 0.0)
- A position from a game where White eventually won is always labeled +1.0, even if it's a losing position
- This is a **major limitation** - the value target is not position-specific

### Board Encoding (`src/data.py`)

Two encoding functions are provided:

- **`board_to_array()`** (training): Returns a numpy int8 array. Uses `board.piece_map()` to iterate only occupied squares (~16-32) rather than all 64. Much faster for bulk parsing and avoids per-sample `torch.Tensor` allocation overhead.
- **`board_to_tensor()`** (inference): Returns a `torch.Tensor[64]`. Retained for inference compatibility where single-position overhead is negligible.

**Piece Mapping**:
- Empty square → 0
- White pieces → 1-6 (P, N, B, R, Q, K)
- Black pieces → 7-12 (P, N, B, R, Q, K)

**Square Ordering**: Uses `chess.SQUARES` which is a flat list. The model must learn that squares are arranged in an 8x8 grid through attention, as there's no explicit 2D structure.

### Move Encoding (`src/data.py`)

**Dynamic Vocabulary Building**:
- `MoveEncoder` builds vocabulary on-the-fly during the tensorisation step
- `encode_uci()` encodes UCI strings directly without constructing `chess.Move` objects
- Each unique UCI move string gets assigned a sequential integer ID
- Vocabulary size depends on training data (typically 1000-4000 moves)

**Issues**:
1. **Inconsistency**: Different training runs produce different move vocabularies
2. **No Standard Mapping**: Move IDs don't correspond to any canonical ordering
3. **Inference Limitation**: Requires the same `MoveEncoder` instance used during training to decode predictions

### Caching (`src/data.py`)

**Purpose**: Avoid re-parsing PGN files on subsequent training runs.

**Implementation**:
- Cache key includes: cache version (`v2`), PGN file path, `max_games`, `min_rating`
- Uses MD5 hash of cache key for filename
- Stores: 4 numpy arrays (converted from contiguous tensors) and `MoveEncoder` state
- Uses pickle protocol 5 for efficient out-of-band serialisation of large arrays
- Cache files saved in `cache/` directory
- Old v1 caches are automatically bypassed (different hash due to version prefix)

**Benefits**: Significantly speeds up subsequent training runs (parsing can take hours for large files).

### Dataset and Batch Loading (`src/data.py`)

**`ChessDataset`**: Backed by contiguous tensors. `__getitem__` is a single tensor index operation (returns a view, no allocation). Retained for backward compatibility but no longer used directly by the training loop.

**`TensorBatchLoader`**: Custom iterable that replaces PyTorch's `DataLoader` for in-memory tensor datasets. Standard `DataLoader` calls `__getitem__` once per sample then collates — with batch_size=2048 that's 2048 Python dict creations, 6144 `.long()` type conversions, and a `torch.stack` per field per batch. `TensorBatchLoader` instead does a single `tensor[batch_indices]` per field, making batch preparation orders of magnitude faster.

- Train/val splits share the same underlying tensors via separate index arrays (no data duplication)
- Shuffle via `torch.randperm` on the index array each epoch
- `pin_memory=True` when CUDA is available (enables async CPU→GPU transfer)
- `drop_last=True` for training (avoids small final batches)
- `non_blocking=True` on all `.to(device)` calls in training/validation loops
- Compact storage dtypes (int8 for piece_ids/sides, int32 for moves) with per-batch `.long()` upcast — reduces tensor memory by 5-8x

**`CUDAPrefetcher`**: Wraps `TensorBatchLoader` (or any batch iterator) and overlaps CPU→GPU transfer with GPU compute. A separate CUDA stream transfers the next batch while the current batch is being processed on the default stream. Combined with `pin_memory=True`, the transfer is a true async DMA copy, hiding transfer latency almost entirely.

**Why not DataLoader workers?** On Windows, `multiprocessing` uses `spawn` which pickles the entire `Dataset` to each worker process. With 318M positions (~23 GB), 12 workers would require 276 GB — exceeding the 128 GB system RAM. Since the data is already in memory as contiguous tensors, there is no disk I/O to parallelise, so workers provide zero benefit. See `docs/briefs/dataloader-bottleneck-brief.md` for full analysis.

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

### Training Loop (`src/train.py`)

**Process**:
1. For each epoch:
   - Iterate through batches under optional bf16 autocast (`torch.amp.autocast`)
   - Forward pass: `policy_logits, value_pred = model(piece_ids, side)`
   - Compute losses
   - Scale loss by `1 / grad_accum_steps` and call backward
   - Every `grad_accum_steps` batches: clip gradients, step optimizer, step LR scheduler, zero gradients

2. **Checkpointing**:
   - Saves checkpoint after each epoch (includes model state, optimizer state, config, and move encoder)
   - Saves "best" model if validation loss improves (validation is now enabled with proper train/val split)

**Optimizer**: AdamW with learning rate 3e-4, weight decay 1e-5

**Gradient Clipping**: Applied with max_norm=1.0 to prevent exploding gradients

**Mixed Precision**: bf16 autocast enabled by default on CUDA (disable with `--no_amp`). bf16 is preferred over fp16 as it shares fp32's exponent range, eliminating the need for `GradScaler`.

**LR Schedule**: Linear warmup (default 1000 steps) followed by cosine decay to zero over the remaining training steps. Scheduler steps per optimizer update.

**Gradient Accumulation**: Configurable via `--grad_accum_steps` (default: 1). Effective batch size equals `batch_size * grad_accum_steps`.

**Direct Cache Loading**: The `--cache_file` argument loads a pre-built `.cache` file directly, bypassing PGN parsing and cache-key computation. Useful when the original PGN file is no longer on disk.

**Multi-GPU**: Two strategies available:
- **DistributedDataParallel (preferred)**: Launch with `torchrun --nproc_per_node=N src/train.py`. Each rank trains on a non-overlapping shard of the data. Gradient allreduce overlaps with backward pass. A fixed seed ensures all ranks agree on the train/val split.
- **DataParallel (fallback)**: Used automatically when multiple GPUs are detected and the script is not launched via torchrun. Works in a single process but has GIL contention and scatter/gather overhead.
- Disable both with `--no_multi_gpu`.

**`torch.compile`**: Opt-in via `--compile`. Fuses kernels, eliminates Python overhead in the forward/backward pass, and reduces memory traffic. Particularly effective for transformer models. Requires PyTorch 2.0+.

### Training Improvements (Implemented)

1. **Validation Split**: ✅ **FIXED** - Now implements proper train/val split:
   - Uses `random_split` to create train/validation datasets
   - Default validation split: 10% (configurable via `--val_split`)
   - Validation loop runs during training
   - Best model saving based on validation loss now works
   - Enables proper model selection and overfitting detection

2. **Value Target Problem**: ✅ **IMPROVED** - Game phase filtering implemented:
   - Early game positions (first 30% of moves) use neutral value (0.0) instead of game outcome
   - Mid-to-endgame positions use actual game outcome
   - Reduces incorrect value signals from early positions
   - Value head should learn more accurate position-specific evaluations
   - Note: Still uses game outcome, but filtered by phase for better signal quality

3. **Parallel PGN Parsing**: ✅ **ADDED** - Multiprocessing support for faster parsing:
   - Uses `multiprocessing.Pool` to parse games in parallel
   - Only activates for datasets > 100 games with `--parse_workers > 1`
   - Games serialized as PGN strings for pickleability
   - Move encoders merged after parallel processing
   - **Expected speedup**: 2-4x on multi-core CPUs

4. **DataLoader Optimizations**: ✅ **ADDED** - GPU-optimized data loading:
   - Auto-detects optimal number of workers: `min(8, cpu_count())`
   - `pin_memory=True` for CUDA (faster CPU→GPU transfers)
   - `persistent_workers=True` to avoid worker restart overhead
   - **Expected speedup**: 1.5-2x for data loading, 10-20% for GPU transfers

5. **No Data Augmentation**: 
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

**Training Signal**: The training signal is **game outcome**, filtered by game phase:
- **Early game (first 30% of moves)**: Uses neutral value (0.0) to avoid learning incorrect evaluations
- **Mid-to-endgame (last 70% of moves)**: Uses actual game outcome
- This improves signal quality by reducing noise from early positions where outcome is less correlated with position quality
- The model still learns: "Positions from games White won tend to be good for White" but only for positions where this correlation is stronger

**Remaining Limitation**: Still uses game outcome rather than position-specific evaluations. For further improvement, could use:
- Position-specific evaluations (e.g., Stockfish scores)
- Temporal difference learning (predict outcome from current position)
- But current phase filtering is a significant improvement

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

**Implementation**: ✅ **FIXED** - The `predict_move_from_legal()` function now works correctly:
- Gets policy logits from the model
- Uses `MoveEncoder` (loaded from checkpoint) to map move IDs to UCI strings
- Filters to legal moves only
- Extracts probabilities for legal moves from model output
- Renormalizes probabilities over legal moves
- Handles moves not in vocabulary gracefully (assigns 0.0 probability)
- Returns top-k moves sorted by probability

**Requirements**: 
- Model checkpoint must include `MoveEncoder` state (now saved automatically)
- If `MoveEncoder` is missing, falls back to uniform distribution with warning

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

5. ✅ **No Spatial Inductive Bias**: ✅ **ADDRESSED** - Three approaches now available:
   - ✅ 2D positional encodings (rank/file coordinates)
   - ✅ Vision Transformer-style patch embeddings
   - ✅ Graph neural network for piece relationships
   - All are optional and can be combined
   - See "Spatial Inductive Bias Features" section for details

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

9. ✅ **Batch Size**: ✅ **IMPROVED** - Default increased to 2048, tuned for multi-GPU setups with large VRAM

10. **Loss Weighting**: 
    - 0.5 weight on value loss is arbitrary
    - Should be tuned or learned

---

## Potential Issues

### 1. Memory Usage with Large Datasets

**Status**: ✅ **IMPROVED** — Data is stored as contiguous tensors with compact dtypes (int8 for piece_ids/sides, int32 for moves, float32 for values). For 318M positions this is ~22 GiB vs ~157 GiB with the old int64 layout. Training uses `TensorBatchLoader` which shares the same underlying tensors between train/val splits via index arrays, avoiding data duplication.

**Windows-specific concern**: PyTorch's `DataLoader` with `num_workers > 0` uses `spawn` multiprocessing on Windows, which pickles the entire dataset to each worker. With 318M positions (~23 GB) and 12 workers, this required 276 GB — exceeding 128 GB RAM. `TensorBatchLoader` avoids this entirely by running in the main process with batch-level tensor indexing. See `docs/briefs/dataloader-bottleneck-brief.md` for details.

**Remaining Consideration**: All data is still held in RAM. With 128 GB and compact dtypes this supports ~500M+ positions comfortably, but for truly enormous corpora a streaming or memory-mapped approach may be needed. Under DDP, each rank loads the full dataset independently (data sharing between processes would require shared memory tensors).

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

**Status**: ✅ **FIXED** - `MoveEncoder` is now saved with all checkpoints:
- Checkpoints include: `move_to_id`, `id_to_move`, and `next_move_id`
- Model loading automatically reconstructs `MoveEncoder` from checkpoint
- Enables proper inference without requiring original training data
- Models are now portable and self-contained

### 7. No Gradient Accumulation

**Status**: ✅ **FIXED** - Gradient accumulation implemented via `--grad_accum_steps`. Loss is scaled by `1 / grad_accum_steps` and optimizer steps every N batches. Effective batch size = `batch_size * grad_accum_steps`.

### 8. MPS Compatibility

**Problem**: MPS (Apple Silicon GPU) may have compatibility issues with certain operations.

**Current Status**: Code checks for MPS but hasn't been tested extensively.

---

## Areas for Improvement

### High Priority

1. ✅ **Fix Inference**: ✅ **COMPLETED** - Move prediction now properly uses model outputs with `MoveEncoder`
2. ✅ **Add Validation Split**: ✅ **COMPLETED** - Proper train/val split implemented with validation loop
3. ✅ **Fix Value Learning**: ✅ **IMPROVED** - Game phase filtering implemented (early positions use neutral value)
4. ✅ **Save MoveEncoder**: ✅ **COMPLETED** - `MoveEncoder` included in all checkpoints
5. ✅ **Add Move Legality**: ✅ **COMPLETED** - Inference filters to legal moves and renormalizes probabilities

### Medium Priority

6. ✅ **Spatial Inductive Bias**: ✅ **COMPLETED** - Three approaches implemented:
   - ✅ 2D positional encodings (rank/file coordinates) - `--use_2d_pos_encoding`
   - ✅ Vision Transformer-style patch embeddings - `--use_patch_embeddings`
   - ✅ Graph neural network for piece relationships - `--use_gnn`
   - All approaches are configurable and can be combined
   - See "Spatial Inductive Bias Features" section below for details

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

11. ✅ **Mixed Precision Training**: ✅ **COMPLETED** - bf16 autocast enabled by default on CUDA (`--no_amp` to disable)
12. ✅ **Gradient Accumulation**: ✅ **COMPLETED** - `--grad_accum_steps` for larger effective batches
13. ✅ **Learning Rate Scheduling**: ✅ **COMPLETED** - Linear warmup + cosine annealing (`--warmup_steps`)
14. **Regularization**: Add dropout tuning, weight decay tuning
15. **Monitoring**: Add TensorBoard logging, better metrics
16. ✅ **Distributed Training**: ✅ **COMPLETED** - DDP via torchrun with DataParallel fallback. `torch.compile` available via `--compile`. `CUDAPrefetcher` overlaps CPU batch prep with GPU compute.
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

