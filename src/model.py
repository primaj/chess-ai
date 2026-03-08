"""
MiniChessTransformer - Transformer-based Chess AI Model
--------------------------------------------------------
Implements a transformer architecture for chess move prediction and position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
#  UTILITY FUNCTIONS
# ============================================================
def square_to_coords(square_index: int) -> tuple:
    """
    Convert a square index (0-63) to (rank, file) coordinates.
    
    Args:
        square_index: Integer from 0-63 (a1=0, b1=1, ..., h8=63)
    
    Returns:
        Tuple of (rank, file) where both are 0-7
    """
    rank = square_index // 8
    file = square_index % 8
    return (rank, file)


def get_square_coords_tensor(device: torch.device) -> torch.Tensor:
    """
    Get a tensor of (rank, file) coordinates for all 64 squares.
    
    Args:
        device: Device to create tensor on
    
    Returns:
        [64, 2] tensor where each row is [rank, file]
    """
    coords = torch.zeros(64, 2, dtype=torch.long, device=device)
    for i in range(64):
        rank, file = square_to_coords(i)
        coords[i] = torch.tensor([rank, file], dtype=torch.long)
    return coords


# ============================================================
#  SQUARE EMBEDDING
# ============================================================
class SquareEmbedding(nn.Module):
    """Embeds chess board squares with piece type, position, and side to move."""
    
    def __init__(self, vocab_size, hidden_dim, use_2d_pos_encoding=False, pos_encoding_type='learned'):
        """
        Args:
            vocab_size: Number of piece types (14)
            hidden_dim: Hidden dimension size
            use_2d_pos_encoding: If True, use rank/file coordinate encodings instead of learned square embeddings
            pos_encoding_type: 'learned', 'sinusoidal', or '2d_coords'
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_2d_pos_encoding = use_2d_pos_encoding
        self.pos_encoding_type = pos_encoding_type
        
        self.piece_embed = nn.Embedding(vocab_size, hidden_dim)
        self.side_embed = nn.Embedding(2, hidden_dim)  # white / black
        
        if use_2d_pos_encoding:
            coords = get_square_coords_tensor(torch.device('cpu'))
            self.register_buffer('_square_ranks', coords[:, 0])
            self.register_buffer('_square_files', coords[:, 1])

            if pos_encoding_type == '2d_coords':
                rank_dim = hidden_dim // 2
                file_dim = hidden_dim - rank_dim
                self.rank_embed = nn.Embedding(8, rank_dim)
                self.file_embed = nn.Embedding(8, file_dim)
            elif pos_encoding_type == 'sinusoidal':
                self.register_buffer('rank_pe', self._create_sinusoidal_pe(8, hidden_dim // 2))
                self.register_buffer('file_pe', self._create_sinusoidal_pe(8, hidden_dim - hidden_dim // 2))
            else:  # learned 2D
                rank_dim = hidden_dim // 2
                file_dim = hidden_dim - rank_dim
                self.rank_embed = nn.Embedding(8, rank_dim)
                self.file_embed = nn.Embedding(8, file_dim)
        else:
            # Original learned square embeddings
            self.square_embed = nn.Embedding(BOARD_TOKENS, hidden_dim)
    
    def _create_sinusoidal_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, piece_ids, side_to_move):
        """
        Args:
            piece_ids: [batch, 64] tensor of piece IDs
            side_to_move: [batch] tensor of 0 (white) or 1 (black)
        
        Returns:
            [batch, 64, hidden_dim] embedded board representation
        """
        batch_size = piece_ids.shape[0]
        device = piece_ids.device
        
        # Embed pieces
        x = self.piece_embed(piece_ids)
        
        # Add positional embeddings
        if self.use_2d_pos_encoding:
            ranks = self._square_ranks   # [64] — pre-computed buffer
            files = self._square_files   # [64]
            
            if self.pos_encoding_type == 'sinusoidal':
                rank_pe = self.rank_pe[ranks]  # [64, rank_dim]
                file_pe = self.file_pe[files]  # [64, file_dim]
                pos_encoding = torch.cat([rank_pe, file_pe], dim=-1)  # [64, hidden_dim]
            else:
                # Learned embeddings
                rank_emb = self.rank_embed(ranks)  # [64, rank_dim]
                file_emb = self.file_embed(files)  # [64, file_dim]
                pos_encoding = torch.cat([rank_emb, file_emb], dim=-1)  # [64, hidden_dim]
            
            # Broadcast to batch
            pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 64, hidden_dim]
            x = x + pos_encoding
        else:
            # Original learned square embeddings
            square_positions = torch.arange(BOARD_TOKENS, device=device)
            x = x + self.square_embed(square_positions).unsqueeze(0)
        
        # Add side to move embedding (broadcasted across all squares)
        side_vec = self.side_embed(side_to_move).unsqueeze(1)  # [batch, 1, hidden_dim]
        x = x + side_vec
        
        return x


# ============================================================
#  PATCH EMBEDDING (Vision Transformer Style)
# ============================================================
class PatchEmbedding(nn.Module):
    """Vision Transformer-style patch embeddings for 8x8 chess board."""
    
    def __init__(self, hidden_dim, patch_size=2, use_conv=True, conv_kernel=3):
        """
        Args:
            hidden_dim: Hidden dimension size
            patch_size: Size of patches (2x2 or 4x4). If None, uses full 8x8 with conv
            use_conv: If True, use 2D convolutions; if False, use patch embeddings
            conv_kernel: Convolution kernel size (3 or 5)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.use_conv = use_conv
        
        if use_conv:
            # Use 2D convolutions to capture local spatial patterns
            self.conv = nn.Sequential(
                nn.Conv2d(1, hidden_dim // 4, kernel_size=conv_kernel, padding=conv_kernel//2),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=conv_kernel, padding=conv_kernel//2),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=conv_kernel, padding=conv_kernel//2),
            )
        else:
            # Patch embedding: divide 8x8 into patches
            if patch_size is None:
                patch_size = 8
            num_patches = (8 // patch_size) ** 2
            self.patch_embed = nn.Linear(patch_size * patch_size, hidden_dim)
            self.num_patches = num_patches
    
    def forward(self, x):
        """
        Args:
            x: [batch, 64, hidden_dim] embedded board
        
        Returns:
            [batch, 64, hidden_dim] processed board representation
        """
        batch_size = x.shape[0]
        
        if self.use_conv:
            # Reshape to 8x8: [batch, 64, hidden_dim] -> [batch, 8, 8, hidden_dim]
            x_2d = x.view(batch_size, 8, 8, self.hidden_dim)
            
            # Project to single channel for convolution input
            # Use mean pooling across hidden_dim to get spatial representation
            x_spatial = x_2d.mean(dim=-1, keepdim=True)  # [batch, 8, 8, 1]
            x_spatial = x_spatial.permute(0, 3, 1, 2)  # [batch, 1, 8, 8]
            
            # Apply convolution
            x_conv = self.conv(x_spatial)  # [batch, hidden_dim, 8, 8]
            
            # Reshape back: [batch, hidden_dim, 8, 8] -> [batch, 8, 8, hidden_dim] -> [batch, 64, hidden_dim]
            x_conv = x_conv.permute(0, 2, 3, 1)  # [batch, 8, 8, hidden_dim]
            x_conv = x_conv.reshape(batch_size, 64, self.hidden_dim)
            
            # Combine with original (residual connection)
            return x + x_conv
        else:
            # Patch embedding approach
            x_2d = x.view(batch_size, 8, 8, self.hidden_dim)
            patches = []
            for i in range(0, 8, self.patch_size):
                for j in range(0, 8, self.patch_size):
                    patch = x_2d[:, i:i+self.patch_size, j:j+self.patch_size, :]
                    patch_flat = patch.reshape(batch_size, -1)  # [batch, patch_size*patch_size*hidden_dim]
                    # For simplicity, take mean of hidden_dim and project
                    patch_mean = patch_flat.mean(dim=-1, keepdim=True)  # Simplified
                    patches.append(patch_mean)
            # This is a simplified version - full implementation would reshape properly
            return x  # Placeholder - would need proper patch processing


# ============================================================
#  GRAPH NEURAL NETWORK
# ============================================================
class ChessGNN(nn.Module):
    """Graph Neural Network for modeling piece relationships."""
    
    def __init__(self, hidden_dim, num_layers=2):
        """
        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.register_buffer('adj', self._build_adjacency_matrix())
    
    @staticmethod
    def _build_adjacency_matrix() -> torch.Tensor:
        """Build normalised adjacency matrix for the chess board graph.

        Connects squares on the same rank, file, diagonal, or within
        knight-move distance.  Includes self-connections and row-normalisation.

        Returns:
            [1, 64, 64] tensor (leading dim for batch broadcasting)
        """
        adj = torch.zeros(64, 64)
        
        for i in range(64):
            rank_i, file_i = square_to_coords(i)
            for j in range(64):
                rank_j, file_j = square_to_coords(j)
                
                if rank_i == rank_j or file_i == file_j:
                    adj[i, j] = 1.0
                
                if abs(rank_i - rank_j) == abs(file_i - file_j):
                    adj[i, j] = 1.0
                
                rank_diff = abs(rank_i - rank_j)
                file_diff = abs(file_i - file_j)
                if (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2):
                    adj[i, j] = 1.0
        
        adj = adj + torch.eye(64)
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-8)
        
        return adj.unsqueeze(0)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 64, hidden_dim] embedded board
        
        Returns:
            [batch, 64, hidden_dim] processed board representation
        """
        batch_size = x.shape[0]
        
        for layer in self.gnn_layers:
            x_neighbors = torch.bmm(self.adj.expand(batch_size, -1, -1), x)
            x = layer(x_neighbors)
            x = x + x_neighbors
        
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
    """Complete transformer-based chess AI model with spatial inductive bias options."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, 
                 n_layers=N_LAYERS, n_heads=N_HEADS, move_vocab=MOVE_VOCAB,
                 use_2d_pos_encoding=False, use_patch_embeddings=False, use_gnn=False,
                 pos_encoding_type='learned', patch_size=2, conv_kernel=3, gnn_layers=2):
        """
        Args:
            vocab_size: Number of piece types (14)
            hidden_dim: Hidden dimension size
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            move_vocab: Size of move vocabulary
            use_2d_pos_encoding: Enable 2D positional encodings (rank/file)
            use_patch_embeddings: Enable Vision Transformer-style patch embeddings
            use_gnn: Enable Graph Neural Network for piece relationships
            pos_encoding_type: 'learned', 'sinusoidal', or '2d_coords'
            patch_size: Patch size for patch embeddings (if not using conv)
            conv_kernel: Convolution kernel size for patch embeddings
            gnn_layers: Number of GNN layers
        """
        super().__init__()
        
        # Embedding layer with optional 2D positional encodings
        self.embed = SquareEmbedding(
            vocab_size, hidden_dim, 
            use_2d_pos_encoding=use_2d_pos_encoding,
            pos_encoding_type=pos_encoding_type
        )
        
        # Optional patch embeddings (ViT-style)
        self.use_patch_embeddings = use_patch_embeddings
        if use_patch_embeddings:
            self.patch_embed = PatchEmbedding(
                hidden_dim, 
                patch_size=patch_size,
                use_conv=True,
                conv_kernel=conv_kernel
            )
        
        # Optional GNN layer
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = ChessGNN(hidden_dim, num_layers=gnn_layers)
        
        # Transformer backbone
        self.backbone = TransformerBackbone(hidden_dim, n_layers, n_heads)
        
        # Output heads
        self.policy = PolicyHead(hidden_dim, move_vocab)
        self.value = ValueHead(hidden_dim)
        
        # Store config for model loading
        self.config = {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'move_vocab': move_vocab,
            'use_2d_pos_encoding': use_2d_pos_encoding,
            'use_patch_embeddings': use_patch_embeddings,
            'use_gnn': use_gnn,
            'pos_encoding_type': pos_encoding_type,
            'patch_size': patch_size,
            'conv_kernel': conv_kernel,
            'gnn_layers': gnn_layers
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
        # Initial embedding with optional 2D positional encodings
        x = self.embed(piece_ids, side_to_move)
        
        # Optional patch embeddings (ViT-style)
        if self.use_patch_embeddings:
            x = self.patch_embed(x)
        
        # Optional GNN processing
        if self.use_gnn:
            x = self.gnn(x)
        
        # Transformer backbone
        x = self.backbone(x)
        
        # Output heads
        policy_logits = self.policy(x)
        value_pred = self.value(x)
        
        return policy_logits, value_pred


# ============================================================
#  MODEL FACTORY
# ============================================================
def create_model(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, 
                 n_layers=N_LAYERS, n_heads=N_HEADS, move_vocab=MOVE_VOCAB,
                 use_2d_pos_encoding=False, use_patch_embeddings=False, use_gnn=False,
                 pos_encoding_type='learned', patch_size=2, conv_kernel=3, gnn_layers=2):
    """
    Factory function to create a MiniChessTransformer.
    
    Args:
        vocab_size: Number of piece types (14)
        hidden_dim: Hidden dimension size
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        move_vocab: Size of move vocabulary
        use_2d_pos_encoding: Enable 2D positional encodings (rank/file)
        use_patch_embeddings: Enable Vision Transformer-style patch embeddings
        use_gnn: Enable Graph Neural Network for piece relationships
        pos_encoding_type: 'learned', 'sinusoidal', or '2d_coords'
        patch_size: Patch size for patch embeddings
        conv_kernel: Convolution kernel size for patch embeddings
        gnn_layers: Number of GNN layers
    """
    return MiniChessTransformer(
        vocab_size, hidden_dim, n_layers, n_heads, move_vocab,
        use_2d_pos_encoding=use_2d_pos_encoding,
        use_patch_embeddings=use_patch_embeddings,
        use_gnn=use_gnn,
        pos_encoding_type=pos_encoding_type,
        patch_size=patch_size,
        conv_kernel=conv_kernel,
        gnn_layers=gnn_layers
    )

