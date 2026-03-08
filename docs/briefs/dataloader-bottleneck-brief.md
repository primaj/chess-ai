# Fix: DataLoader Bottleneck on Windows with Large In-Memory Datasets

## System

- **GPUs**: 2x NVIDIA RTX 4090 (24 GB each)
- **RAM**: 128 GB
- **OS**: Windows 10
- **Python**: 3.10
- **Dataset**: `cache/lichess_db_standard_rated_2016-01_6e7615a0.cache` (21.6 GiB, 318M positions)

## What happened

Training was launched with the recommended command:

```powershell
python src/train.py --cache_file cache/lichess_db_standard_rated_2016-01_6e7615a0.cache \
    --use_2d_pos_encoding --pos_encoding_type learned \
    --hidden_dim 512 --n_layers 8 --n_heads 8 \
    --batch_size 2048 --lr 3e-4 --epochs 10
```

Two issues surfaced in sequence:

### Issue 1: Training stuck at 0% — DataLoader workers cloning the entire dataset

`get_optimal_workers()` returned 12, so PyTorch's `DataLoader` spawned 12 worker processes. On Windows, `multiprocessing` uses the `spawn` start method, which means each worker receives a pickled copy of the `Dataset` object. The dataset contains ~23 GB of contiguous tensors (318M positions). With 12 workers, each deserialising its own copy:

```
12 workers × 23 GB = 276 GB  →  exceeds 128 GB RAM
```

The system hit swap immediately. The `tqdm` progress bar showed `0/139918 [00:00<?, ?it/s]` — no batches ever completed. Task Manager showed 11 Python processes each consuming ~10-12 GB and climbing.

**Attempted fix**: Calling `.share_memory_()` on the tensors in `ChessDataset.__init__` to place them in a shared memory segment. This did not work reliably on Windows with numpy-backed tensors created via `torch.from_numpy()`.

**Actual fix**: Set `get_optimal_workers()` to return 0. Multi-worker DataLoader is unnecessary when the dataset is already resident in RAM as contiguous tensors — `__getitem__` is a single tensor index operation with no disk I/O to parallelise.

### Issue 2: Training moving but extremely slow (1-4 it/s)

With `num_workers=0`, training progressed but at only 1-4 iterations/second — far below expected throughput for dual 4090s. The bottleneck was CPU-side per-sample overhead in `DataLoader`:

1. **2048 individual `__getitem__` calls per batch** — each creates a Python dict and calls `.long()` on 3 tensors
2. **Default collate** — `torch.stack` on 2048 individual tensors per field
3. **Type conversion** — `.long()` called 2048 × 3 = 6144 times per batch on individual scalars/rows

With a 26M-parameter model on 64-token sequences, GPU compute per batch is only ~25-50 ms, but the DataLoader CPU overhead was 200-900 ms — completely dominating wall-clock time.

**Fix**: Replaced `DataLoader` with `TensorBatchLoader` (`src/data.py`), a custom iterable that does batch-level tensor indexing:

- One `tensor[batch_indices]` call per field instead of 2048 individual `__getitem__` calls
- `.long()` conversion once on the batched tensor instead of 6144 times on scalars
- No collate step (batch is already formed)
- No DataLoader framework overhead
- Train/val splits share the same underlying tensors via separate index arrays (no data duplication)

## Root cause analysis

The fundamental mismatch is that PyTorch's `DataLoader` is designed for datasets where `__getitem__` involves meaningful work (disk reads, decoding images, tokenisation). For in-memory contiguous tensor datasets, all that machinery — worker processes, IPC, per-sample fetching, collation — adds pure overhead with zero benefit.

The problem was masked during earlier development with smaller datasets (hundreds of thousands of positions) where the overhead was negligible relative to GPU compute. At 318M positions with a small-but-fast model, the CPU overhead became the dominant cost.

## Changes made

### `src/data.py`

- Added `TensorBatchLoader` class: batch-level tensor indexing, `pin_memory` support, shuffle via `torch.randperm`, shared underlying tensors between train/val splits
- `get_optimal_workers()` now returns 0 (DataLoader workers provide no benefit for in-memory tensor datasets and cause memory explosion on Windows)

### `src/train.py`

- Replaced `DataLoader` + `random_split` with `TensorBatchLoader` + direct index splitting via `torch.randperm`
- Removed `--num_workers` and `--prefetch_factor` CLI arguments (no longer applicable)
- Removed `DataLoader` import

## Key takeaway

For in-memory tensor datasets with cheap `__getitem__`, bypass `DataLoader` entirely and do batch-level tensor indexing. The per-sample Python overhead of DataLoader (dict creation, type conversion, collation) dominates when GPU compute is fast relative to batch preparation.
