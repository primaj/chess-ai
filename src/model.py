"""
MiniChessTransformer - Transformer-based Chess AI Model
--------------------------------------------------------
Implements a transformer architecture for chess move prediction and position evaluation.
"""

import torch
import torch.nn as nn


# ============================================================
#  CONFIG
# ============================================================
BOARD_TOKENS = 64       # one per square
VOCAB_SIZE = 14         # 12 pieces + empty + padding
HIDDEN_DIM = 256
N_LAYERS = 6
N_HEADS = 8
MOVE_VOCAB = 4096       # UCI move tokens (approximate)


# ============================================================
#  SQUARE EMBEDDING
# ============================================================
class SquareEmbedding(nn.Module):
    """Embeds chess board squares with piece type, position, and side to move."""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.piece_embed = nn.Embedding(vocab_size, hidden_dim)
        self.square_embed = nn.Embedding(BOARD_TOKENS, hidden_dim)
        self.side_embed = nn.Embedding(2, hidden_dim)  # white / black

    def forward(self, piece_ids, side_to_move):
        """
        Args:
            piece_ids: [batch, 64] tensor of piece IDs
            side_to_move: [batch] tensor of 0 (white) or 1 (black)
        
        Returns:
            [batch, 64, hidden_dim] embedded board representation
        """
        # Embed pieces
        x = self.piece_embed(piece_ids)
        
        # Add square positional embeddings
        square_positions = torch.arange(BOARD_TOKENS, device=piece_ids.device)
        x = x + self.square_embed(square_positions)
        
        # Add side to move embedding (broadcasted across all squares)
        side_vec = self.side_embed(side_to_move).unsqueeze(1)  # [batch, 1, hidden_dim]
        x = x + side_vec
        
        return x


# ============================================================
#  TRANSFORMER BACKBONE
# ============================================================
class TransformerBackbone(nn.Module):
    """Transformer encoder that learns global dependencies across the board."""
    
    def __init__(self, hidden_dim, n_layers, n_heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='gelu',
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        Args:
            x: [batch, 64, hidden_dim] embedded board
        
        Returns:
            [batch, 64, hidden_dim] encoded board representation
        """
        return self.encoder(x)


# ============================================================
#  POLICY HEAD
# ============================================================
class PolicyHead(nn.Module):
    """Predicts move probability distribution over all possible moves."""
    
    def __init__(self, hidden_dim, move_vocab):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, move_vocab)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 64, hidden_dim] encoded board
        
        Returns:
            [batch, move_vocab] logits over moves
        """
        # Pool across all squares (mean pooling)
        pooled = x.mean(dim=1)  # [batch, hidden_dim]
        return self.fc(pooled)  # [batch, move_vocab]


# ============================================================
#  VALUE HEAD
# ============================================================
class ValueHead(nn.Module):
    """Predicts position evaluation (expected game outcome)."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # value ∈ [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 64, hidden_dim] encoded board
        
        Returns:
            [batch, 1] position evaluation
        """
        # Pool across all squares (mean pooling)
        pooled = x.mean(dim=1)  # [batch, hidden_dim]
        return self.fc(pooled)  # [batch, 1]


# ============================================================
#  FULL MODEL
# ============================================================
class MiniChessTransformer(nn.Module):
    """Complete transformer-based chess AI model."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, 
                 n_layers=N_LAYERS, n_heads=N_HEADS, move_vocab=MOVE_VOCAB):
        super().__init__()
        self.embed = SquareEmbedding(vocab_size, hidden_dim)
        self.backbone = TransformerBackbone(hidden_dim, n_layers, n_heads)
        self.policy = PolicyHead(hidden_dim, move_vocab)
        self.value = ValueHead(hidden_dim)
        
        # Store config for model loading
        self.config = {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'move_vocab': move_vocab
        }

    def forward(self, piece_ids, side_to_move):
        """
        Args:
            piece_ids: [batch, 64] tensor of piece IDs
            side_to_move: [batch] tensor of 0 (white) or 1 (black)
        
        Returns:
            policy_logits: [batch, move_vocab] move prediction logits
            value_pred: [batch, 1] position evaluation
        """
        x = self.embed(piece_ids, side_to_move)
        x = self.backbone(x)
        policy_logits = self.policy(x)
        value_pred = self.value(x)
        return policy_logits, value_pred


# ============================================================
#  MODEL FACTORY
# ============================================================
def create_model(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, 
                 n_layers=N_LAYERS, n_heads=N_HEADS, move_vocab=MOVE_VOCAB):
    """Factory function to create a MiniChessTransformer."""
    return MiniChessTransformer(vocab_size, hidden_dim, n_layers, n_heads, move_vocab)

