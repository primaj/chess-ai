"""
Training Script for MiniChessTransformer
-----------------------------------------
Trains the transformer model on PGN chess game data.

Supports bf16 mixed precision, cosine-annealing LR with warmup,
gradient accumulation, torch.compile, and multi-GPU via DDP
(with DataParallel fallback when not launched through torchrun).
"""

import os
# Must be set before any torch.distributed import so the TCPStore
# falls back to the non-libuv implementation on Windows builds
# that were compiled without libuv support.
os.environ.setdefault('USE_LIBUV', '0')

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import argparse

from model import MiniChessTransformer, create_model


# ============================================================
#  CONFIG
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 5
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8


# ============================================================
#  DISTRIBUTED HELPERS
# ============================================================
def _setup_distributed():
    """Initialise DDP when launched via torchrun / torch.distributed.launch.

    Returns (rank, world_size).  When not in a distributed launch the
    process group is left uninitialised and (0, 1) is returned.
    """
    if 'RANK' not in os.environ:
        return 0, 1

    backend = 'nccl' if dist.is_nccl_available() else 'gloo'
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ============================================================
#  TRAINING FUNCTIONS
# ============================================================
def _unwrap_model(model):
    """Return the underlying model, stripping compile / DataParallel / DDP wrappers."""
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    if isinstance(model, (nn.DataParallel, DDP)):
        model = model.module
    return model


def train_epoch(model, loader, optimizer, criterion_policy, criterion_value,
                device, use_amp=False, grad_accum_steps=1, scheduler=None,
                verbose=True):
    """Train for one epoch with optional AMP and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False, disable=not verbose)):
        piece_ids = batch["piece_ids"].to(device, non_blocking=True)
        side = batch["side"].to(device, non_blocking=True)
        move = batch["move"].to(device, non_blocking=True)
        result = batch["result"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            policy_logits, value_pred = model(piece_ids, side)
            loss_policy = criterion_policy(policy_logits, move)
            loss_value = criterion_value(value_pred.squeeze(), result)
            loss = loss_policy + 0.5 * loss_value

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_policy_loss += loss_policy.item()
        total_value_loss += loss_value.item()
        num_batches += 1

    if num_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        'total_loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }


def train(model, train_loader, val_loader, optimizer, n_epochs, device,
          save_dir="models", save_prefix="minichess_transformer",
          move_encoder_state=None, use_amp=False, grad_accum_steps=1,
          scheduler=None, verbose=True):
    """Main training loop.

    Args:
        verbose: When False, suppresses prints and tqdm bars (non-main DDP ranks).
    """
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    base_model = _unwrap_model(model)

    if verbose:
        os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{n_epochs}  (lr={current_lr:.2e})")
            print(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion_policy, criterion_value, device,
            use_amp=use_amp, grad_accum_steps=grad_accum_steps,
            scheduler=scheduler, verbose=verbose,
        )

        if verbose:
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Policy: {train_metrics['policy_loss']:.4f}, "
                  f"Value: {train_metrics['value_loss']:.4f})")

        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion_policy,
                                  criterion_value, device, use_amp=use_amp,
                                  verbose=verbose)
            if verbose:
                print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                      f"(Policy: {val_metrics['policy_loss']:.4f}, "
                      f"Value: {val_metrics['value_loss']:.4f})")

            if verbose and val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                save_path = os.path.join(save_dir, f"{save_prefix}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': base_model.config,
                    'val_loss': best_val_loss,
                    'move_encoder': move_encoder_state,
                }, save_path)
                print(f"Saved best model to {save_path}")

        if verbose:
            checkpoint_path = os.path.join(save_dir, f"{save_prefix}_epoch_{epoch+1}.pt")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': base_model.config,
                'train_loss': train_metrics['total_loss'],
            }
            if move_encoder_state is not None:
                checkpoint_data['move_encoder'] = move_encoder_state
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def validate(model, loader, criterion_policy, criterion_value, device,
             use_amp=False, verbose=True):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False, disable=not verbose):
            piece_ids = batch["piece_ids"].to(device, non_blocking=True)
            side = batch["side"].to(device, non_blocking=True)
            move = batch["move"].to(device, non_blocking=True)
            result = batch["result"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                policy_logits, value_pred = model(piece_ids, side)
                loss_policy = criterion_policy(policy_logits, move)
                loss_value = criterion_value(value_pred.squeeze(), result)
                loss = loss_policy + 0.5 * loss_value

            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            num_batches += 1

    return {
        'total_loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }


# ============================================================
#  MAIN
# ============================================================
def main():
    rank, world_size = _setup_distributed()
    is_distributed = world_size > 1
    verbose = _is_main_process()

    parser = argparse.ArgumentParser(description="Train MiniChessTransformer")

    # Data source (one of --pgn_file or --cache_file is required)
    parser.add_argument("--pgn_file", type=str, default=None,
                       help="Path to PGN file")
    parser.add_argument("--cache_file", type=str, default=None,
                       help="Path to pre-built .cache file (bypasses --pgn_file)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                       help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps (default: 1, no accumulation)")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Linear LR warmup steps before cosine decay (default: 1000)")
    parser.add_argument("--no_amp", action="store_true",
                       help="Disable bf16 mixed-precision training (enabled by default on CUDA)")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM,
                       help=f"Hidden dimension (default: {DEFAULT_HIDDEN_DIM})")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS,
                       help=f"Number of transformer layers (default: {DEFAULT_N_LAYERS})")
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS,
                       help=f"Number of attention heads (default: {DEFAULT_N_HEADS})")

    # Data filtering
    parser.add_argument("--max_games", type=int, default=None,
                       help="Maximum number of games to parse (default: all)")
    parser.add_argument("--min_rating", type=int, default=None,
                       help="Minimum player rating to include (default: all)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1)")

    # Saving
    parser.add_argument("--save_dir", type=str, default="models",
                       help="Directory to save models (default: models)")
    parser.add_argument("--save_prefix", type=str, default="minichess_transformer",
                       help="Prefix for saved model files (default: minichess_transformer)")

    # Caching / data loading
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Use cache for parsed PGN data (default: True)")
    parser.add_argument("--no_cache", dest="use_cache", action="store_false",
                       help="Disable cache for parsed PGN data")
    parser.add_argument("--cache_dir", type=str, default="cache",
                       help="Directory to store cache files (default: cache)")
    parser.add_argument("--parse_workers", type=int, default=1,
                       help="Number of parallel workers for PGN parsing (default: 1, sequential)")

    # Spatial inductive bias options
    parser.add_argument("--use_2d_pos_encoding", action="store_true",
                       help="Enable 2D positional encodings (rank/file coordinates)")
    parser.add_argument("--pos_encoding_type", type=str, default="learned",
                       choices=["learned", "sinusoidal", "2d_coords"],
                       help="Type of positional encoding: learned, sinusoidal, or 2d_coords (default: learned)")
    parser.add_argument("--use_patch_embeddings", action="store_true",
                       help="Enable Vision Transformer-style patch embeddings")
    parser.add_argument("--patch_size", type=int, default=2,
                       help="Patch size for patch embeddings (default: 2)")
    parser.add_argument("--conv_kernel", type=int, default=3,
                       help="Convolution kernel size for patch embeddings (default: 3)")
    parser.add_argument("--use_gnn", action="store_true",
                       help="Enable Graph Neural Network for piece relationships")
    parser.add_argument("--gnn_layers", type=int, default=2,
                       help="Number of GNN layers (default: 2)")
    parser.add_argument("--no_multi_gpu", action="store_true",
                       help="Disable multi-GPU parallelism even when multiple GPUs are available")

    # Performance options
    parser.add_argument("--compile", action="store_true",
                       help="Enable torch.compile for kernel fusion (requires PyTorch 2.0+)")

    args = parser.parse_args()

    if args.pgn_file is None and args.cache_file is None:
        parser.error("Either --pgn_file or --cache_file is required")

    # Resolve device — DDP pins each rank to its local GPU
    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f"cuda:{local_rank}"
    else:
        device = DEVICE

    use_amp = (device.startswith("cuda") and not args.no_amp)

    if verbose:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"  DDP: {world_size} processes (rank {rank})")
        print(f"Training configuration:")
        if args.cache_file:
            print(f"  Cache file: {args.cache_file}")
        else:
            print(f"  PGN file: {args.pgn_file}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Grad accum steps: {args.grad_accum_steps}"
              f"  (effective batch: {args.batch_size * args.grad_accum_steps})")
        print(f"  Learning rate: {args.lr}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  AMP (bf16): {'enabled' if use_amp else 'disabled'}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Layers: {args.n_layers}")
        print(f"  Heads: {args.n_heads}")
        if not args.cache_file:
            print(f"  Parse workers: {args.parse_workers}")
        if args.use_2d_pos_encoding:
            print(f"  2D Positional Encoding: {args.pos_encoding_type}")
        if args.use_patch_embeddings:
            print(f"  Patch Embeddings: enabled (patch_size={args.patch_size}, kernel={args.conv_kernel})")
        if args.use_gnn:
            print(f"  GNN: enabled ({args.gnn_layers} layers)")
        if args.compile:
            print(f"  torch.compile: enabled")

    # ------------------------------------------------------------------
    #  Load data
    # ------------------------------------------------------------------
    if verbose:
        print("\nLoading data...")
    from data import pgn_to_samples, load_cache, ChessDataset, TensorBatchLoader, CUDAPrefetcher

    if args.cache_file:
        result = load_cache(args.cache_file)
        if result is None:
            raise RuntimeError(f"Failed to load cache file: {args.cache_file}")
        tensor_dict, move_encoder = result
    else:
        if verbose:
            if args.use_cache:
                print(f"Cache enabled (cache directory: {args.cache_dir})")
            else:
                print("Cache disabled")

        tensor_dict, move_encoder = pgn_to_samples(
            args.pgn_file,
            max_games=args.max_games,
            min_rating=args.min_rating,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
            num_parse_workers=args.parse_workers,
        )

    move_vocab_size = move_encoder.get_vocab_size()
    if verbose:
        print(f"Move vocabulary size: {move_vocab_size}")

    full_dataset = ChessDataset(tensor_dict)
    del tensor_dict

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    # Fixed seed so all DDP ranks agree on the train/val split
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_size, generator=generator)
    train_indices = perm[val_size:]
    val_indices = perm[:val_size]
    del perm

    # Under DDP, each rank trains on a non-overlapping shard
    if is_distributed:
        per_rank = len(train_indices) // world_size
        shard_start = rank * per_rank
        shard_end = shard_start + per_rank
        train_indices = train_indices[shard_start:shard_end]
        train_size = len(train_indices)

    pin_memory = device.startswith("cuda")

    train_loader = TensorBatchLoader(
        full_dataset.piece_ids, full_dataset.sides,
        full_dataset.moves, full_dataset.values,
        indices=train_indices, batch_size=args.batch_size,
        shuffle=True, drop_last=True, pin_memory=pin_memory,
    )

    val_loader = TensorBatchLoader(
        full_dataset.piece_ids, full_dataset.sides,
        full_dataset.moves, full_dataset.values,
        indices=val_indices, batch_size=args.batch_size,
        shuffle=False, pin_memory=pin_memory,
    )

    # Wrap with CUDA prefetcher to overlap CPU batch prep with GPU compute
    if pin_memory:
        train_loader = CUDAPrefetcher(train_loader, device)
        val_loader = CUDAPrefetcher(val_loader, device)

    if verbose:
        print(f"Train samples: {train_size:,}, Val samples: {val_size:,}")

    # ------------------------------------------------------------------
    #  Create model
    # ------------------------------------------------------------------
    if verbose:
        print("\nCreating model...")
    model = create_model(
        vocab_size=14,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        move_vocab=move_vocab_size,
        use_2d_pos_encoding=args.use_2d_pos_encoding,
        use_patch_embeddings=args.use_patch_embeddings,
        use_gnn=args.use_gnn,
        pos_encoding_type=args.pos_encoding_type,
        patch_size=args.patch_size,
        conv_kernel=args.conv_kernel,
        gnn_layers=args.gnn_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {num_params:,}")

    # Multi-GPU: prefer DDP (via torchrun), fall back to DataParallel
    n_gpus = torch.cuda.device_count()
    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = DDP(model, device_ids=[local_rank])
        if verbose:
            print(f"Using DistributedDataParallel across {world_size} GPUs")
    elif device.startswith("cuda") and n_gpus > 1 and not args.no_multi_gpu:
        if verbose:
            print(f"Wrapping model in DataParallel across {n_gpus} GPUs: "
                  + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))
            print("  Tip: launch with torchrun for better multi-GPU scaling (DDP)")
        model = nn.DataParallel(model)

    if args.compile and hasattr(torch, 'compile'):
        if verbose:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ------------------------------------------------------------------
    #  LR Scheduler: linear warmup then cosine decay
    # ------------------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    if verbose:
        print(f"  LR schedule: {warmup_steps} warmup steps, "
              f"{total_steps} total steps (cosine decay)")

    # Prepare move encoder state for saving
    move_encoder_state = {
        'move_to_id': move_encoder.move_to_id,
        'id_to_move': move_encoder.id_to_move,
        'next_move_id': move_encoder.next_move_id,
    }

    # ------------------------------------------------------------------
    #  Train
    # ------------------------------------------------------------------
    if verbose:
        print("\nStarting training...")
    train(
        model, train_loader, val_loader, optimizer, args.epochs, device,
        save_dir=args.save_dir, save_prefix=args.save_prefix,
        move_encoder_state=move_encoder_state,
        use_amp=use_amp, grad_accum_steps=args.grad_accum_steps,
        scheduler=scheduler, verbose=verbose,
    )

    if verbose:
        print("\nTraining complete!")

    _cleanup_distributed()


if __name__ == "__main__":
    main()

