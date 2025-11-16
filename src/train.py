"""
Training Script for MiniChessTransformer
-----------------------------------------
Trains the transformer model on PGN chess game data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from model import MiniChessTransformer, create_model
from data import create_dataloader
from torch.utils.data import DataLoader


# ============================================================
#  CONFIG
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 5
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8


# ============================================================
#  TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, criterion_policy, criterion_value, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        piece_ids = batch["piece_ids"].to(device)
        side = batch["side"].to(device)
        move = batch["move"].to(device)
        result = batch["result"].to(device)
        
        # Forward pass
        policy_logits, value_pred = model(piece_ids, side)
        
        # Compute losses
        loss_policy = criterion_policy(policy_logits, move)
        loss_value = criterion_value(value_pred.squeeze(), result)
        loss = loss_policy + 0.5 * loss_value
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_policy_loss += loss_policy.item()
        total_value_loss += loss_value.item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }


def train(model, train_loader, val_loader, optimizer, n_epochs, device, 
          save_dir="models", save_prefix="minichess_transformer", move_encoder_state=None):
    """Main training loop."""
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, 
            criterion_policy, criterion_value, device
        )
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Policy: {train_metrics['policy_loss']:.4f}, "
              f"Value: {train_metrics['value_loss']:.4f})")
        
        # Validation (if validation loader provided)
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion_policy, 
                                  criterion_value, device)
            print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Policy: {val_metrics['policy_loss']:.4f}, "
                  f"Value: {val_metrics['value_loss']:.4f})")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                save_path = os.path.join(save_dir, f"{save_prefix}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model.config,
                    'val_loss': best_val_loss,
                    'move_encoder': move_encoder_state,
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"{save_prefix}_epoch_{epoch+1}.pt")
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config,
            'train_loss': train_metrics['total_loss'],
        }
        if move_encoder_state is not None:
            checkpoint_data['move_encoder'] = move_encoder_state
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def validate(model, loader, criterion_policy, criterion_value, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            piece_ids = batch["piece_ids"].to(device)
            side = batch["side"].to(device)
            move = batch["move"].to(device)
            result = batch["result"].to(device)
            
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
    parser = argparse.ArgumentParser(description="Train MiniChessTransformer")
    parser.add_argument("--pgn_file", type=str, required=True,
                       help="Path to PGN file")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                       help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM,
                       help=f"Hidden dimension (default: {DEFAULT_HIDDEN_DIM})")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS,
                       help=f"Number of transformer layers (default: {DEFAULT_N_LAYERS})")
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS,
                       help=f"Number of attention heads (default: {DEFAULT_N_HEADS})")
    parser.add_argument("--max_games", type=int, default=None,
                       help="Maximum number of games to parse (default: all)")
    parser.add_argument("--min_rating", type=int, default=None,
                       help="Minimum player rating to include (default: all)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument("--save_dir", type=str, default="models",
                       help="Directory to save models (default: models)")
    parser.add_argument("--save_prefix", type=str, default="minichess_transformer",
                       help="Prefix for saved model files (default: minichess_transformer)")
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Use cache for parsed PGN data (default: True)")
    parser.add_argument("--no_cache", dest="use_cache", action="store_false",
                       help="Disable cache for parsed PGN data")
    parser.add_argument("--cache_dir", type=str, default="cache",
                       help="Directory to store cache files (default: cache)")
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Training configuration:")
    print(f"  PGN file: {args.pgn_file}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Heads: {args.n_heads}")
    
    # Load data
    print("\nLoading data...")
    if args.use_cache:
        print(f"Cache enabled (cache directory: {args.cache_dir})")
    else:
        print("Cache disabled")
    
    # Load data and get move encoder
    from data import pgn_to_samples, ChessDataset, collate_fn
    samples, move_encoder = pgn_to_samples(
        args.pgn_file,
        max_games=args.max_games,
        min_rating=args.min_rating,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir
    )
    
    # Get move vocabulary size
    move_vocab_size = move_encoder.get_vocab_size()
    print(f"Move vocabulary size: {move_vocab_size}")
    
    # Create dataset
    full_dataset = ChessDataset(samples)
    
    # Create validation split
    from torch.utils.data import random_split
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=14,  # Fixed: 12 pieces + empty + padding
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        move_vocab=move_vocab_size
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Prepare move encoder state for saving
    move_encoder_state = {
        'move_to_id': move_encoder.move_to_id,
        'id_to_move': move_encoder.id_to_move,
        'next_move_id': move_encoder.next_move_id
    }
    
    # Train
    print("\nStarting training...")
    train(
        model, train_loader, val_loader, optimizer, args.epochs, DEVICE,
        save_dir=args.save_dir, save_prefix=args.save_prefix, move_encoder_state=move_encoder_state
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

