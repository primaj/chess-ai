# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Resume from checkpoint**: `--resume path/to/epoch_N.pt` continues training from a saved epoch. Requires `--cache_file` and `--epochs` (total desired epochs). Restores model, optimizer, and LR scheduler; preserves best validation loss so `_best.pt` is not overwritten by a worse run. Older checkpoints without scheduler state are supported (scheduler is fast-forwarded).
- **`run_parse.py` CLI**: Script now accepts `--date`, `--max_games`, `--min_rating`, and `--output` so download-and-parse can be run with e.g. `python scripts/run_parse.py --date 2016-01 --min_rating 1500`.
- **Direct Cache Loading**: New `--cache_file` argument loads a pre-built `.cache` file directly, bypassing PGN path and hash computation. `--pgn_file` is no longer required when `--cache_file` is provided.
- **bf16 Mixed Precision**: Training and validation now run under `torch.amp.autocast` with bfloat16 by default on CUDA. Disable with `--no_amp`. No `GradScaler` needed (bf16 shares fp32's exponent range).
- **LR Scheduler**: Cosine annealing with linear warmup. Configurable via `--warmup_steps` (default: 1000). Steps per optimizer update, not per batch.
- **Gradient Accumulation**: New `--grad_accum_steps` argument (default: 1). Enables larger effective batch sizes without extra VRAM.
- **`TensorBatchLoader`**: Custom batch loader in `src/data.py` that replaces `DataLoader` for in-memory tensor datasets. Does batch-level tensor indexing (one op per field) instead of per-sample `__getitem__` calls, eliminating collation and per-sample type conversion overhead. Train/val splits share the same underlying tensors via separate index arrays.
- **`CUDAPrefetcher`**: Wraps any batch iterator and transfers the next batch to GPU on a separate CUDA stream while the current batch is being computed, hiding CPU-to-GPU transfer latency.
- **DistributedDataParallel support**: Launch with `torchrun --nproc_per_node=N src/train.py` for DDP multi-GPU training. Falls back to DataParallel when not launched via torchrun.
- **`--compile` flag**: Enables `torch.compile` for kernel fusion (requires PyTorch 2.0+).

### Changed
- **Default batch size**: Increased from 64 to 2048 (tuned for multi-GPU setups with ample VRAM).
- **Default learning rate**: Increased from 1e-4 to 3e-4 (better for larger batch sizes).
- **`--pgn_file`**: No longer required; either `--pgn_file` or `--cache_file` must be provided.
- **Data loading**: Replaced `DataLoader` + `random_split` with `TensorBatchLoader` + direct index splitting. Eliminates per-sample Python overhead and the Windows `spawn` memory duplication issue with multi-worker DataLoader.
- **Compact tensor dtypes**: `_tensorise` now stores piece_ids as int8 and moves as int32 (matching the parallel parsing path), reducing in-memory tensor footprint by 5-8x. The `.long()` upcast happens per-batch in `TensorBatchLoader`.
- **Cached coordinate buffers**: `SquareEmbedding` now registers rank/file coordinate tensors as buffers (computed once at init) instead of rebuilding them on every forward pass.
- **GNN adjacency matrix**: `ChessGNN` now computes and registers the normalised adjacency matrix as a buffer in `__init__`, replacing the fragile `hasattr`-based cache that did not survive `model.to(device)` or serialisation.
- **`--no_multi_gpu`**: Now disables all multi-GPU parallelism (DataParallel and DDP).
- **Checkpoint format**: Per-epoch checkpoints now include `scheduler_state_dict` and `best_val_loss` for correct resume behaviour.
- **LR display**: When learning rate is 0 (warmup), the epoch header now shows e.g. `lr=0.00e+00 warmup -> 3.00e-04` to avoid confusion.

### Removed
- **`--num_workers` CLI arg**: DataLoader workers are no longer used (in-memory tensor datasets don't benefit from multi-worker loading, and on Windows `spawn` clones the entire dataset to each worker).
- **`--prefetch_factor` CLI arg**: No longer applicable without DataLoader workers.

## [0.4.0] - 2026-03-07

### Changed
- **Contiguous Tensor Storage**: All training data is now stored as 4 contiguous tensors instead of a list of individual Python tuples. Eliminates millions of per-sample object allocations and reduces memory overhead by ~5-10x.
- **Numpy Board Encoding**: New `board_to_array()` uses `piece_map()` to iterate only occupied squares (~16-32) instead of all 64, and returns a numpy array instead of allocating a torch.Tensor per position.
- **Simplified Parallel Parsing**: Worker processes now return raw samples (numpy + UCI strings) without per-worker move vocabularies. Eliminates the merge/remap step entirely.
- **Sequential Parsing Path**: Single-pass through the PGN file (read + parse in one go) instead of storing both PGN strings and game objects in memory.
- **Cache Format v2**: Stores contiguous numpy arrays via pickle protocol 5 (out-of-band large buffer serialisation). Old v1 caches are automatically bypassed.
- **Dataset**: `ChessDataset` is now backed by contiguous tensors — `__getitem__` is a zero-allocation tensor index.
- **Default batch size**: Increased from 32 to 64.
- **DataLoader workers**: Auto-detect ceiling raised from 8 to 12.

### Added
- **`board_to_array()`**: Fast numpy-based board encoding for training (original `board_to_tensor()` retained for inference).
- **`MoveEncoder.encode_uci()`**: Encode UCI strings directly without constructing chess.Move objects.
- **`get_optimal_workers()`**: Exported utility for auto-detecting DataLoader worker count.
- **`--prefetch_factor` CLI arg**: Control how many batches each DataLoader worker prefetches (default: 4).
- **`non_blocking=True`**: All `.to(device)` calls in training and validation loops now use asynchronous CPU→GPU transfer.
- **`optimizer.zero_grad(set_to_none=True)`**: Faster gradient clearing (sets to None instead of filling with zeros).
- **`drop_last=True`**: Training DataLoader drops the final incomplete batch for consistent batch sizes.
- **`pin_memory=True`**: Enabled by default when CUDA is available.
- **`persistent_workers=True`**: Enabled by default when `num_workers > 0`.
- **`prefetch_factor=4`**: Each DataLoader worker prefetches 4 batches ahead to keep GPUs fed.

### Performance
- **Memory**: ~5-10x reduction in Python object overhead for training data storage
- **Parsing**: Faster board encoding via numpy + piece_map(); no double-storage of games
- **Data Loading**: Zero-allocation `__getitem__`, pin_memory, prefetching, and non_blocking transfers
- **GPU Utilisation**: Async transfers + prefetching minimise GPU idle time on multi-GPU setups

## [0.3.0] - 2024-11-15

### Added
- **Parallel PGN Parsing**: Multiprocessing support for parsing PGN files (2-4x speedup on multi-core CPUs)
- **DataLoader Optimizations**: 
  - Auto-detection of optimal number of workers (default: min(8, cpu_count))
  - `pin_memory=True` for faster CPU→GPU transfers when using CUDA
  - `persistent_workers=True` to avoid worker restart overhead
- **Lichess Download Utility**: New `src/download.py` script to download PGN files directly from Lichess database
  - Support for date-based shortcuts (e.g., `2025-10`)
  - Resume interrupted downloads
  - Progress bar with download speed
- **CLI Arguments**: 
  - `--num_workers`: Control DataLoader workers (default: auto-detect)
  - `--parse_workers`: Control parallel parsing workers (default: 1)

### Changed
- **PGN Parsing**: Now supports parallel processing for datasets > 100 games
- **DataLoader**: Default workers changed from 0 to auto-detected value
- **Memory Usage**: Parallel parsing uses more RAM but significantly faster

### Performance Improvements
- **Parsing**: 2-4x faster with multiprocessing (on multi-core CPUs)
- **Data Loading**: 1.5-2x faster with multiple workers
- **GPU Transfer**: 10-20% faster with pin_memory (CUDA only)
- **Overall**: 30-50% faster data pipeline for GPU training

## [0.2.0] - 2024-11-15

### Fixed
- **Inference**: Fixed `predict_move_from_legal()` to properly use model outputs instead of returning uniform probabilities
- **Move Encoder**: `MoveEncoder` is now saved with all model checkpoints, enabling proper inference without training data
- **Move Prediction**: Inference now correctly maps model outputs to legal moves using the saved `MoveEncoder`

### Added
- **Validation Split**: Implemented proper train/validation split with configurable ratio (default: 10%)
- **Validation Loop**: Validation now runs during training with proper metrics tracking
- **Best Model Saving**: Best model is now saved based on validation loss (previously disabled)
- **Game Phase Filtering**: Value learning now filters by game phase - early positions (first 30%) use neutral value to improve signal quality
- **Move Legality Filtering**: Inference properly filters to legal moves only and renormalizes probabilities

### Changed
- **Value Learning Signal**: Early game positions now use neutral value (0.0) instead of game outcome to reduce incorrect training signals
- **Checkpoint Format**: Checkpoints now include `MoveEncoder` state for model portability
- **Training Output**: Training now displays validation metrics and train/val split information

### Technical Details
- Validation split uses `random_split` from PyTorch
- Game phase is calculated as `move_count / total_moves` with threshold at 0.3 (30%)
- Move encoder state includes: `move_to_id`, `id_to_move`, and `next_move_id`
- Inference handles moves not in vocabulary gracefully (assigns 0.0 probability)

## [0.1.0] - 2024-11-15

### Added
- Initial proof-of-concept implementation
- Transformer-based chess AI with policy and value heads
- PGN parsing with Zstandard compression support
- Caching system for parsed PGN data
- MPS (Apple Silicon) GPU support
- Training and inference utilities
- Interactive CLI for testing

[Unreleased]: https://github.com/yourusername/chess-ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/chess-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/chess-ai/releases/tag/v0.1.0

