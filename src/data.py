"""
Data Processing Utilities for Chess Transformer
------------------------------------------------
Handles PGN parsing, board encoding, and dataset creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import zstandard as zstd
import io
import pickle
import os
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count


# ============================================================
#  PIECE ENCODING
# ============================================================
PIECE_TO_ID = {
    None: 0,
    chess.Piece.from_symbol('P'): 1,
    chess.Piece.from_symbol('N'): 2,
    chess.Piece.from_symbol('B'): 3,
    chess.Piece.from_symbol('R'): 4,
    chess.Piece.from_symbol('Q'): 5,
    chess.Piece.from_symbol('K'): 6,
}

# Add black pieces (offset by +6)
for k in list(PIECE_TO_ID.keys()):
    if k is not None:
        PIECE_TO_ID[chess.Piece(k.piece_type, chess.BLACK)] = PIECE_TO_ID[k] + 6

ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}


# ============================================================
#  MOVE ENCODING
# ============================================================
class MoveEncoder:
    """Encodes and decodes chess moves to/from integer IDs."""
    
    def __init__(self):
        self.move_to_id: Dict[str, int] = {}
        self.id_to_move: Dict[int, str] = {}
        self.next_move_id = 0
    
    def encode(self, move: chess.Move) -> int:
        """Convert a chess.Move to an integer ID."""
        uci = move.uci()
        if uci not in self.move_to_id:
            self.move_to_id[uci] = self.next_move_id
            self.id_to_move[self.next_move_id] = uci
            self.next_move_id += 1
        return self.move_to_id[uci]
    
    def decode(self, move_id: int) -> str:
        """Convert an integer ID back to UCI move string."""
        return self.id_to_move.get(move_id, "")
    
    def get_vocab_size(self) -> int:
        """Get the current move vocabulary size."""
        return self.next_move_id


# ============================================================
#  BOARD ENCODING
# ============================================================
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to a tensor of shape [64] with piece IDs.
    
    Args:
        board: chess.Board instance
    
    Returns:
        [64] tensor of piece IDs
    """
    tensor = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tensor[square] = PIECE_TO_ID[piece]
    return tensor


# ============================================================
#  FILE HANDLING
# ============================================================
def open_pgn_file(pgn_path: str):
    """
    Open a PGN file, handling both compressed (.zst) and uncompressed files.
    
    Args:
        pgn_path: Path to PGN file (may be .pgn or .pgn.zst)
    
    Returns:
        Context manager that yields a file-like object for reading
    """
    if pgn_path.endswith('.zst'):
        # Open compressed file
        class ZstdFileWrapper:
            def __init__(self, path):
                self.path = path
                self.file = None
                self.wrapper = None
            
            def __enter__(self):
                dctx = zstd.ZstdDecompressor()
                self.file = open(self.path, 'rb')
                decompressed = dctx.stream_reader(self.file)
                self.wrapper = io.TextIOWrapper(decompressed, encoding='utf-8')
                return self.wrapper
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.wrapper:
                    self.wrapper.close()
                if self.file:
                    self.file.close()
                return False
        
        return ZstdFileWrapper(pgn_path)
    else:
        # Open regular text file
        return open(pgn_path, 'r', encoding='utf-8')


# ============================================================
#  CACHE MANAGEMENT
# ============================================================
def get_cache_path(pgn_path: str, max_games: Optional[int] = None, 
                   min_rating: Optional[int] = None, cache_dir: str = "cache") -> str:
    """
    Generate a cache file path based on PGN path and parameters.
    
    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games (for cache key)
        min_rating: Minimum rating (for cache key)
        cache_dir: Directory to store cache files
    
    Returns:
        Path to cache file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a hash from the PGN path and parameters
    cache_key = f"{pgn_path}_{max_games}_{min_rating}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # Get base filename from PGN path
    pgn_name = Path(pgn_path).stem
    if pgn_name.endswith('.pgn'):
        pgn_name = pgn_name[:-4]
    
    cache_filename = f"{pgn_name}_{cache_hash[:8]}.cache"
    return os.path.join(cache_dir, cache_filename)


def save_cache(cache_path: str, samples: List[Tuple], move_encoder: MoveEncoder):
    """Save parsed samples and move encoder to cache file."""
    cache_data = {
        'samples': samples,
        'move_encoder': {
            'move_to_id': move_encoder.move_to_id,
            'id_to_move': move_encoder.id_to_move,
            'next_move_id': move_encoder.next_move_id
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Cache saved to {cache_path}")


def load_cache(cache_path: str) -> Optional[Tuple[List[Tuple], MoveEncoder]]:
    """
    Load parsed samples and move encoder from cache file.
    
    Returns:
        (samples, move_encoder) if cache exists, None otherwise
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Reconstruct MoveEncoder
        move_encoder = MoveEncoder()
        move_encoder.move_to_id = cache_data['move_encoder']['move_to_id']
        move_encoder.id_to_move = cache_data['move_encoder']['id_to_move']
        move_encoder.next_move_id = cache_data['move_encoder']['next_move_id']
        
        samples = cache_data['samples']
        print(f"Loaded {len(samples)} samples from cache: {cache_path}")
        return samples, move_encoder
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


# ============================================================
#  PGN PARSING (HELPER FUNCTIONS)
# ============================================================
def parse_single_game(game: chess.pgn.Game, min_rating: Optional[int] = None) -> Optional[List[Tuple]]:
    """
    Parse a single game into samples.
    
    Args:
        game: chess.pgn.Game instance
        min_rating: Minimum player rating to include (None for all)
    
    Returns:
        List of (piece_ids, side, move_id, outcome) tuples, or None if filtered out
    """
    # Filter by rating if specified
    if min_rating is not None:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if white_elo < min_rating or black_elo < min_rating:
                return None
        except ValueError:
            return None
    
    # Parse game result
    result = game.headers.get("Result", "*")
    if result == "1-0":
        outcome = 1.0  # White wins
    elif result == "0-1":
        outcome = -1.0  # Black wins
    elif result == "1/2-1/2":
        outcome = 0.0  # Draw
    else:
        outcome = 0.0  # Unknown result
    
    # Extract positions and moves
    board = game.board()
    mainline_moves = list(game.mainline_moves())
    total_moves = len(mainline_moves)
    move_count = 0
    samples = []
    
    for move in mainline_moves:
        move_count += 1
        piece_ids = board_to_tensor(board)
        side = 0 if board.turn == chess.WHITE else 1
        
        # Filter value learning by game phase
        game_phase = move_count / max(total_moves, 1)  # 0.0 to 1.0
        
        if game_phase < 0.3:  # First 30% of game
            value_target = 0.0
        else:
            value_target = outcome
        
        # Store move UCI for later encoding (to avoid encoder conflicts in parallel)
        samples.append((piece_ids, side, move.uci(), value_target))
        board.push(move)
    
    return samples if samples else None


def parse_games_chunk(args: Tuple) -> Tuple[List[Tuple], Dict[str, int]]:
    """
    Parse a chunk of games in parallel.
    
    Args:
        args: Tuple of (games_list, min_rating)
    
    Returns:
        Tuple of (samples, move_vocab_dict) where move_vocab_dict maps UCI to sequential IDs
    """
    games_list, min_rating = args
    samples = []
    move_vocab = {}  # Local move vocabulary for this chunk
    next_id = 0
    
    for game in games_list:
        game_samples = parse_single_game(game, min_rating)
        if game_samples is None:
            continue
        
        # Encode moves with local vocabulary
        encoded_samples = []
        for piece_ids, side, move_uci, value_target in game_samples:
            if move_uci not in move_vocab:
                move_vocab[move_uci] = next_id
                next_id += 1
            move_id = move_vocab[move_uci]
            encoded_samples.append((piece_ids, side, move_id, value_target))
        
        samples.extend(encoded_samples)
    
    return samples, move_vocab


def merge_move_encoders(move_vocabs: List[Dict[str, int]]) -> Tuple[MoveEncoder, Dict[int, int]]:
    """
    Merge multiple move vocabularies into a single MoveEncoder.
    
    Args:
        move_vocabs: List of dictionaries mapping UCI moves to local IDs
    
    Returns:
        Tuple of (merged MoveEncoder, mapping from old IDs to new IDs for each vocab)
    """
    merged_encoder = MoveEncoder()
    id_mappings = []  # For each vocab, maps old_id -> new_id
    
    for vocab in move_vocabs:
        old_to_new = {}
        for move_uci, old_id in vocab.items():
            # Encode in merged encoder (will assign new ID if not seen)
            new_id = merged_encoder.encode(chess.Move.from_uci(move_uci))
            old_to_new[old_id] = new_id
        id_mappings.append(old_to_new)
    
    return merged_encoder, id_mappings


# ============================================================
#  PGN PARSING
# ============================================================
def pgn_to_samples(pgn_path: str, max_games: Optional[int] = None, 
                   min_rating: Optional[int] = None, 
                   use_cache: bool = True, cache_dir: str = "cache",
                   num_parse_workers: int = 1) -> Tuple[List[Tuple], MoveEncoder]:
    """
    Parse PGN file into training samples.
    Supports both .pgn and .pgn.zst (Zstandard compressed) files.
    Caches parsed results for faster subsequent loads.
    
    Args:
        pgn_path: Path to PGN file (may be .pgn or .pgn.zst)
        max_games: Maximum number of games to parse (None for all)
        min_rating: Minimum player rating to include (None for all)
        use_cache: Whether to use cache (default: True)
        cache_dir: Directory to store cache files (default: "cache")
        num_parse_workers: Number of parallel workers for parsing (1 = sequential)
    
    Returns:
        samples: List of (piece_ids, side, move_id, outcome) tuples
        move_encoder: MoveEncoder instance with learned vocabulary
    """
    # Check cache first
    if use_cache:
        cache_path = get_cache_path(pgn_path, max_games, min_rating, cache_dir)
        cached = load_cache(cache_path)
        if cached is not None:
            return cached
    
    # Read all games first (needed for parallel processing)
    games = []
    with open_pgn_file(pgn_path) as f:
        game_count = 0
        pbar = tqdm(desc="Reading games")
        while True:
            if max_games is not None and game_count >= max_games:
                break
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
            game_count += 1
            pbar.update(1)
        pbar.close()
    
    print(f"Read {len(games)} games, parsing...")
    
    # Parse games (parallel or sequential)
    if num_parse_workers > 1 and len(games) > 100:  # Only parallelize for larger datasets
        # Split games into chunks for parallel processing
        chunk_size = max(100, len(games) // (num_parse_workers * 2))
        game_chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]
        
        # Parse chunks in parallel
        with Pool(processes=num_parse_workers) as pool:
            chunk_args = [(chunk, min_rating) for chunk in game_chunks]
            results = list(tqdm(
                pool.imap(parse_games_chunk, chunk_args),
                total=len(game_chunks),
                desc="Parsing games"
            ))
        
        # Extract samples and vocabularies
        all_samples = []
        move_vocabs = []
        for chunk_samples, chunk_vocab in results:
            all_samples.append(chunk_samples)
            move_vocabs.append(chunk_vocab)
        
        # Merge move encoders
        move_encoder, id_mappings = merge_move_encoders(move_vocabs)
        
        # Remap move IDs in samples to merged encoder IDs
        final_samples = []
        for chunk_idx, chunk_samples in enumerate(all_samples):
            id_mapping = id_mappings[chunk_idx]
            for piece_ids, side, old_move_id, value_target in chunk_samples:
                new_move_id = id_mapping[old_move_id]
                final_samples.append((piece_ids, side, new_move_id, value_target))
        
        samples = final_samples
    else:
        # Sequential parsing (original method)
        samples = []
        move_encoder = MoveEncoder()
        
        for game in tqdm(games, desc="Parsing games"):
            game_samples = parse_single_game(game, min_rating)
            if game_samples is None:
                continue
            
            for piece_ids, side, move_uci, value_target in game_samples:
                move_id = move_encoder.encode(chess.Move.from_uci(move_uci))
                samples.append((piece_ids, side, move_id, value_target))
    
    print(f"Parsed {len(games)} games, {len(samples)} positions")
    print(f"Move vocabulary size: {move_encoder.get_vocab_size()}")
    
    # Save to cache
    if use_cache:
        cache_path = get_cache_path(pgn_path, max_games, min_rating, cache_dir)
        save_cache(cache_path, samples, move_encoder)
    
    return samples, move_encoder


# ============================================================
#  DATASET
# ============================================================
class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions."""
    
    def __init__(self, samples: List[Tuple]):
        """
        Args:
            samples: List of (piece_ids, side, move_id, outcome) tuples
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, side, move, result = self.samples[idx]
        return {
            "piece_ids": x,
            "side": torch.tensor(side, dtype=torch.long),
            "move": torch.tensor(move, dtype=torch.long),
            "result": torch.tensor(result, dtype=torch.float)
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "piece_ids": torch.stack([b["piece_ids"] for b in batch]),
        "side": torch.stack([b["side"] for b in batch]),
        "move": torch.stack([b["move"] for b in batch]),
        "result": torch.stack([b["result"] for b in batch]),
    }


# ============================================================
#  DATA LOADER UTILITIES
# ============================================================
def create_dataloader(pgn_path: str, batch_size: int = 32, 
                     max_games: Optional[int] = None, min_rating: Optional[int] = None,
                     shuffle: bool = True, num_workers: int = 0,
                     use_cache: bool = True, cache_dir: str = "cache",
                     return_samples: bool = False,
                     num_parse_workers: int = 1) -> Tuple[DataLoader, MoveEncoder]:
    """
    Create a DataLoader from a PGN file.
    
    Args:
        pgn_path: Path to PGN file
        batch_size: Batch size for training
        max_games: Maximum number of games to parse
        min_rating: Minimum player rating to include
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use cache (default: True)
        cache_dir: Directory to store cache files (default: "cache")
    
    Returns:
        DataLoader and MoveEncoder instance
    """
    samples, move_encoder = pgn_to_samples(pgn_path, max_games, min_rating, use_cache, cache_dir, num_parse_workers)
    dataset = ChessDataset(samples)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    return loader, move_encoder

