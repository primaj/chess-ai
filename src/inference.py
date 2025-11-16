"""
Inference Utilities for MiniChessTransformer
---------------------------------------------
Functions for loading models and making predictions.
"""

import torch
import chess
from typing import List, Tuple, Dict
import numpy as np

from model import MiniChessTransformer
from data import board_to_tensor, MoveEncoder


# ============================================================
#  MODEL LOADING
# ============================================================
def load_model(checkpoint_path: str, device: str = None) -> Tuple[MiniChessTransformer, Dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on (None for auto-detect)
    
    Returns:
        model: Loaded MiniChessTransformer instance
        checkpoint_info: Dictionary with checkpoint metadata
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {
        'vocab_size': 14,
        'hidden_dim': 256,
        'n_layers': 6,
        'n_heads': 8,
        'move_vocab': 4096
    })
    
    # Create model with saved config
    model = MiniChessTransformer(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        move_vocab=config['move_vocab']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load move encoder if available
    move_encoder = None
    if 'move_encoder' in checkpoint:
        from data import MoveEncoder
        move_encoder = MoveEncoder()
        move_encoder.move_to_id = checkpoint['move_encoder']['move_to_id']
        move_encoder.id_to_move = checkpoint['move_encoder']['id_to_move']
        move_encoder.next_move_id = checkpoint['move_encoder']['next_move_id']
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'config': config,
        'move_encoder': move_encoder
    }
    
    return model, checkpoint_info


# ============================================================
#  MOVE PREDICTION
# ============================================================
def predict_move(model: MiniChessTransformer, board: chess.Board, 
                move_encoder: MoveEncoder = None, top_k: int = 5,
                device: str = None) -> List[Tuple[str, float]]:
    """
    Predict top moves for a given board position.
    
    Args:
        model: Trained MiniChessTransformer model
        board: chess.Board instance
        move_encoder: MoveEncoder instance (if None, will try to decode from model)
        top_k: Number of top moves to return
        device: Device to run inference on (None for auto-detect)
    
    Returns:
        List of (move_uci, probability) tuples, sorted by probability
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Convert board to tensor
    piece_ids = board_to_tensor(board).unsqueeze(0).to(device)
    side = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.long).to(device)
    
    # Get legal moves
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return []
    
    # Forward pass
    with torch.no_grad():
        policy_logits, _ = model(piece_ids, side)
        policy_probs = torch.softmax(policy_logits, dim=-1)
    
    # Get top moves (we need to map move IDs back to UCI)
    # For now, return top logits (this is a limitation - we'd need the move_encoder)
    top_probs, top_indices = torch.topk(policy_probs[0], k=min(top_k, len(legal_moves)))
    
    # If we have a move encoder, decode the moves
    if move_encoder is not None:
        results = []
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
            move_uci = move_encoder.decode(int(idx))
            if move_uci and chess.Move.from_uci(move_uci) in legal_moves:
                results.append((move_uci, float(prob)))
        return results
    else:
        # Return indices and probabilities (user can map these)
        return [(f"move_{idx}", float(prob)) for idx, prob in 
                zip(top_indices.cpu().numpy(), top_probs.cpu().numpy())]


def predict_move_from_legal(model: MiniChessTransformer, board: chess.Board,
                           move_encoder: MoveEncoder = None, top_k: int = 5, 
                           device: str = None) -> List[Tuple[chess.Move, float]]:
    """
    Predict top moves from legal moves only (more practical version).
    
    Args:
        model: Trained MiniChessTransformer model
        board: chess.Board instance
        move_encoder: MoveEncoder instance (required for proper move prediction)
        top_k: Number of top moves to return
        device: Device to run inference on (None for auto-detect)
    
    Returns:
        List of (chess.Move, probability) tuples, sorted by probability
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Convert board to tensor
    piece_ids = board_to_tensor(board).unsqueeze(0).to(device)
    side = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.long).to(device)
    
    # Get legal moves
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return []
    
    # Forward pass
    with torch.no_grad():
        policy_logits, _ = model(piece_ids, side)
        policy_probs = torch.softmax(policy_logits, dim=-1)
    
    # Map legal moves to their probabilities using move_encoder
    if move_encoder is None:
        # Fallback: return uniform distribution if no move_encoder
        move_scores = [(move, 1.0 / len(legal_moves)) for move in legal_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[:top_k]
    
    # Get probabilities for legal moves
    move_scores = []
    for move in legal_moves:
        move_uci = move.uci()
        if move_uci in move_encoder.move_to_id:
            move_id = move_encoder.move_to_id[move_uci]
            if move_id < policy_probs.shape[1]:
                prob = policy_probs[0, move_id].item()
                move_scores.append((move, prob))
            else:
                # Move ID out of range (shouldn't happen, but handle gracefully)
                move_scores.append((move, 0.0))
        else:
            # Move not in vocabulary (unseen during training)
            move_scores.append((move, 0.0))
    
    # Sort by probability and return top_k
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Renormalize probabilities over legal moves only
    total_prob = sum(prob for _, prob in move_scores)
    if total_prob > 0:
        move_scores = [(move, prob / total_prob) for move, prob in move_scores]
    
    return move_scores[:top_k]


# ============================================================
#  POSITION EVALUATION
# ============================================================
def evaluate_position(model: MiniChessTransformer, board: chess.Board,
                      device: str = None) -> float:
    """
    Evaluate a chess position.
    
    Args:
        model: Trained MiniChessTransformer model
        board: chess.Board instance
        device: Device to run inference on (None for auto-detect)
    
    Returns:
        Position evaluation (float between -1 and +1, from White's perspective)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Convert board to tensor
    piece_ids = board_to_tensor(board).unsqueeze(0).to(device)
    side = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        _, value_pred = model(piece_ids, side)
    
    # Return value from White's perspective
    value = value_pred[0, 0].item()
    if board.turn == chess.BLACK:
        value = -value  # Flip perspective if it's Black's turn
    
    return value


# ============================================================
#  CLI INTERFACE
# ============================================================
def interactive_inference(checkpoint_path: str):
    """
    Interactive CLI for testing model inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
    """
    print(f"Loading model from {checkpoint_path}...")
    model, info = load_model(checkpoint_path)
    print(f"Model loaded (epoch {info['epoch']}, loss: {info.get('train_loss', 'N/A')})")
    
    board = chess.Board()
    
    print("\nInteractive Chess Inference")
    print("Commands:")
    print("  <move> - Make a move (e.g., 'e2e4')")
    print("  eval - Evaluate current position")
    print("  moves - Show top predicted moves")
    print("  reset - Reset to starting position")
    print("  quit - Exit")
    print()
    
    while True:
        print(f"\n{board}")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        cmd = input("\n> ").strip().lower()
        
        if cmd == "quit":
            break
        elif cmd == "reset":
            board = chess.Board()
            print("Board reset")
        elif cmd == "eval":
            value = evaluate_position(model, board)
            print(f"Position evaluation: {value:.4f} (from White's perspective)")
        elif cmd == "moves":
            move_encoder = info.get('move_encoder')
            if move_encoder is None:
                print("Warning: No move encoder found. Move predictions may be inaccurate.")
            moves = predict_move_from_legal(model, board, move_encoder=move_encoder, top_k=5)
            print("Top predicted moves:")
            for i, (move, prob) in enumerate(moves, 1):
                print(f"  {i}. {move.uci()}: {prob:.4f}")
        else:
            # Try to parse as a move
            try:
                move = chess.Move.from_uci(cmd)
                if move in board.legal_moves:
                    board.push(move)
                    print(f"Move played: {move.uci()}")
                else:
                    print("Illegal move")
            except ValueError:
                print("Invalid command or move")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with MiniChessTransformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive inference mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_inference(args.checkpoint)
    else:
        # Simple test
        model, info = load_model(args.checkpoint)
        board = chess.Board()
        
        print("Testing inference...")
        value = evaluate_position(model, board)
        print(f"Starting position evaluation: {value:.4f}")
        
        move_encoder = info.get('move_encoder')
        moves = predict_move_from_legal(model, board, move_encoder=move_encoder, top_k=3)
        print("Top 3 moves:")
        for move, prob in moves:
            print(f"  {move.uci()}: {prob:.4f}")

