"""
Data Processing Utilities for Chess Transformer
------------------------------------------------
Handles PGN parsing, board encoding, and dataset creation.

Optimised for high-throughput training on multi-GPU setups with large RAM.
All training data is stored as contiguous tensors — eliminates per-sample
Python object overhead and enables zero-copy DataLoader indexing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import numpy as np
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

for k in list(PIECE_TO_ID.keys()):
    if k is not None:
        PIECE_TO_ID[chess.Piece(k.piece_type, chess.BLACK)] = PIECE_TO_ID[k] + 6

ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

_PIECE_MAP_LOOKUP = {}
for piece_obj, pid in PIECE_TO_ID.items():
    if piece_obj is not None:
        _PIECE_MAP_LOOKUP[(piece_obj.piece_type, piece_obj.color)] = pid


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

    def encode_uci(self, uci: str) -> int:
        """Encode a UCI string directly without constructing a chess.Move."""
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
    Retained for inference compatibility — use board_to_array for training.
    """
    tensor = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tensor[square] = PIECE_TO_ID[piece]
    return tensor


def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to a numpy int8 array of shape [64].
    Uses piece_map() to iterate only occupied squares — faster than
    scanning all 64, and avoids per-sample torch.Tensor allocation.
    """
    arr = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        arr[square] = _PIECE_MAP_LOOKUP[(piece.piece_type, piece.color)]
    return arr


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
        return open(pgn_path, 'r', encoding='utf-8')


# ============================================================
#  CACHE MANAGEMENT
# ============================================================
_CACHE_VERSION = "v3"


def get_cache_path(pgn_path: str, max_games: Optional[int] = None,
                   min_rating: Optional[int] = None, cache_dir: str = "cache") -> str:
    """
    Generate a cache file path based on PGN path and parameters.
    Includes cache version in the key so old caches are bypassed automatically.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{_CACHE_VERSION}_{pgn_path}_{max_games}_{min_rating}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    pgn_name = Path(pgn_path).stem
    if pgn_name.endswith('.pgn'):
        pgn_name = pgn_name[:-4]

    cache_filename = f"{pgn_name}_{cache_hash[:8]}.cache"
    return os.path.join(cache_dir, cache_filename)


def save_cache(cache_path: str, tensor_dict: Dict[str, torch.Tensor],
               move_encoder: MoveEncoder):
    """Save contiguous tensor data and move encoder to cache using pickle protocol 5.
    Protocol 5 serialises large numpy/torch buffers out-of-band for speed."""
    cache_data = {
        'version': 3,
        'piece_ids': tensor_dict['piece_ids'].numpy(),
        'sides': tensor_dict['sides'].numpy(),
        'moves': tensor_dict['moves'].numpy(),
        'values': tensor_dict['values'].numpy(),
        'move_encoder': {
            'move_to_id': move_encoder.move_to_id,
            'id_to_move': move_encoder.id_to_move,
            'next_move_id': move_encoder.next_move_id,
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=5)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cache saved to {cache_path} ({size_mb:.1f} MB)")


def load_cache(cache_path: str) -> Optional[Tuple[Dict[str, torch.Tensor], MoveEncoder]]:
    """Load cached tensor data and move encoder."""
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        move_encoder = MoveEncoder()
        enc = cache_data['move_encoder']
        move_encoder.move_to_id = enc['move_to_id']
        move_encoder.id_to_move = enc['id_to_move']
        move_encoder.next_move_id = enc['next_move_id']

        tensor_dict = {
            'piece_ids': torch.from_numpy(cache_data['piece_ids']),
            'sides': torch.from_numpy(cache_data['sides']),
            'moves': torch.from_numpy(cache_data['moves']),
            'values': torch.from_numpy(cache_data['values']),
        }

        n = tensor_dict['piece_ids'].shape[0]
        print(f"Loaded {n:,} samples from cache: {cache_path}")
        return tensor_dict, move_encoder
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


# ============================================================
#  PGN PARSING (HELPER FUNCTIONS)
# ============================================================
def parse_single_game(game: chess.pgn.Game,
                      min_rating: Optional[int] = None) -> Optional[List[Tuple]]:
    """
    Parse a single game into raw samples.

    Returns list of (piece_ids_np, side_int8, move_uci_str, value_float32)
    tuples, or None if filtered out. Move encoding is deferred to the
    tensorisation step so worker processes don't need their own vocabularies.
    """
    if min_rating is not None:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if white_elo < min_rating or black_elo < min_rating:
                return None
        except ValueError:
            return None

    result = game.headers.get("Result", "*")
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    elif result == "1/2-1/2":
        outcome = 0.0
    else:
        outcome = 0.0

    board = game.board()
    mainline_moves = list(game.mainline_moves())
    total_moves = len(mainline_moves)
    move_count = 0
    samples = []

    for move in mainline_moves:
        move_count += 1
        piece_ids = board_to_array(board)
        side = np.int8(0 if board.turn == chess.WHITE else 1)

        game_phase = move_count / max(total_moves, 1)
        value_target = np.float32(0.0 if game_phase < 0.3 else outcome)

        samples.append((piece_ids, side, move.uci(), value_target))
        board.push(move)

    return samples if samples else None


def parse_games_chunk(args: Tuple) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Parse a chunk of PGN strings in a worker process.

    Returns packed arrays (piece_ids int8, sides int8, move UCI strings,
    values float32) instead of a list of Python tuples.  Packing inside
    the worker keeps IPC payloads compact and avoids accumulating hundreds
    of millions of individual Python objects in the main process.
    """
    pgn_strings_list, min_rating = args
    samples = []

    for pgn_string in pgn_strings_list:
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_string))
            if game is None:
                continue
        except Exception:
            continue

        game_samples = parse_single_game(game, min_rating)
        if game_samples is not None:
            samples.extend(game_samples)

    if not samples:
        return (np.empty((0, 64), dtype=np.int8),
                np.empty(0, dtype=np.int8),
                [],
                np.empty(0, dtype=np.float32))

    return (np.array([s[0] for s in samples], dtype=np.int8),
            np.array([s[1] for s in samples], dtype=np.int8),
            [s[2] for s in samples],
            np.array([s[3] for s in samples], dtype=np.float32))


# ============================================================
#  TENSORISATION
# ============================================================
def _tensorise(raw_samples: List[Tuple],
               move_encoder: MoveEncoder) -> Dict[str, torch.Tensor]:
    """
    Convert raw parsed samples into contiguous tensors.

    This is the key optimisation: instead of N individual Python tuples each
    holding a separate tensor, all data lives in exactly 4 contiguous tensors.
    __getitem__ becomes a single pointer-arithmetic index with zero allocation.
    """
    n = len(raw_samples)

    piece_ids_np = np.empty((n, 64), dtype=np.int8)
    sides_np = np.empty(n, dtype=np.int8)
    moves_np = np.empty(n, dtype=np.int32)
    values_np = np.empty(n, dtype=np.float32)

    for i, (p, s, m_uci, v) in enumerate(raw_samples):
        piece_ids_np[i] = p
        sides_np[i] = s
        moves_np[i] = move_encoder.encode_uci(m_uci)
        values_np[i] = v

    return {
        'piece_ids': torch.from_numpy(piece_ids_np),
        'sides': torch.from_numpy(sides_np),
        'moves': torch.from_numpy(moves_np),
        'values': torch.from_numpy(values_np),
    }


# ============================================================
#  PGN PARSING (MAIN ENTRY POINT)
# ============================================================
def _parse_sequential(pgn_path: str, max_games: Optional[int],
                      min_rating: Optional[int]) -> Tuple[List[Tuple], int]:
    """Single-pass sequential parsing — reads and parses in one go."""
    raw_samples = []
    game_count = 0

    with open_pgn_file(pgn_path) as f:
        pbar = tqdm(desc="Parsing games")
        while True:
            if max_games is not None and game_count >= max_games:
                break
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_samples = parse_single_game(game, min_rating)
            if game_samples is not None:
                raw_samples.extend(game_samples)
            game_count += 1
            pbar.update(1)
        pbar.close()

    return raw_samples, game_count


def _read_raw_pgn_games(pgn_path: str,
                        max_games: Optional[int] = None) -> List[str]:
    """Bulk-read raw PGN game strings without any chess parsing.

    For zst files: decompress the entire stream then split on '[Event '
    boundaries.  For plain PGN: read the whole file then split.
    This is orders of magnitude faster than chess.pgn.read_game() +
    StringExporter because it's pure string operations — no game-tree
    construction, no move parsing, no re-serialisation.
    """
    import time as _time
    t0 = _time.time()

    if pgn_path.endswith('.zst'):
        print("Decompressing .zst file (bulk)...")
        dctx = zstd.ZstdDecompressor()
        with open(pgn_path, 'rb') as f:
            raw = dctx.stream_reader(f)
            text = io.TextIOWrapper(raw, encoding='utf-8').read()
    else:
        print("Reading PGN file...")
        with open(pgn_path, 'r', encoding='utf-8') as f:
            text = f.read()

    size_gb = len(text) / (1024 ** 3)
    print(f"Decompressed in {_time.time() - t0:.1f}s ({size_gb:.2f} GB)")

    print("Splitting into individual games...")
    parts = text.split('\n[Event ')
    del text

    games: List[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i > 0:
            part = '[Event ' + part
        games.append(part)
        if max_games is not None and len(games) >= max_games:
            break
    del parts

    print(f"Found {len(games):,} games in {_time.time() - t0:.1f}s")
    return games


def _parse_parallel(pgn_path: str, max_games: Optional[int],
                    min_rating: Optional[int],
                    num_workers: int
                    ) -> Tuple[Dict[str, torch.Tensor], MoveEncoder, int]:
    """Fast parallel parsing with incremental tensorisation.

    Workers return packed numpy arrays per chunk.  The main process
    encodes moves (building the vocabulary) and accumulates compact
    arrays.  At the end a single np.concatenate per field produces
    the final contiguous tensors — no giant List[Tuple] intermediate.
    """
    pgn_strings = _read_raw_pgn_games(pgn_path, max_games)
    game_count = len(pgn_strings)

    num_workers = min(num_workers, 16)
    print(f"Distributing {game_count:,} games to {num_workers} workers...")

    chunk_size = 1000
    pgn_chunks = [
        pgn_strings[i:i + chunk_size]
        for i in range(0, game_count, chunk_size)
    ]
    del pgn_strings

    num_chunks = len(pgn_chunks)

    def _chunk_args():
        for i in range(num_chunks):
            chunk = pgn_chunks[i]
            pgn_chunks[i] = None
            yield (chunk, min_rating)

    move_encoder = MoveEncoder()
    pieces_acc: List[np.ndarray] = []
    sides_acc: List[np.ndarray] = []
    moves_acc: List[np.ndarray] = []
    values_acc: List[np.ndarray] = []

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(parse_games_chunk, _chunk_args()),
            total=num_chunks,
            desc="Parsing games (parallel)"
        ):
            chunk_pieces, chunk_sides, chunk_moves_uci, chunk_values = result
            del result

            if len(chunk_moves_uci) == 0:
                continue

            encoded = np.array(
                [move_encoder.encode_uci(m) for m in chunk_moves_uci],
                dtype=np.int32,
            )
            del chunk_moves_uci

            pieces_acc.append(chunk_pieces)
            sides_acc.append(chunk_sides)
            moves_acc.append(encoded)
            values_acc.append(chunk_values)

    print("Concatenating results...")

    piece_ids_cat = np.concatenate(pieces_acc); del pieces_acc
    sides_cat = np.concatenate(sides_acc);     del sides_acc
    moves_cat = np.concatenate(moves_acc);     del moves_acc
    values_cat = np.concatenate(values_acc);   del values_acc

    tensor_dict = {
        'piece_ids': torch.from_numpy(piece_ids_cat),
        'sides': torch.from_numpy(sides_cat),
        'moves': torch.from_numpy(moves_cat),
        'values': torch.from_numpy(values_cat),
    }

    return tensor_dict, move_encoder, game_count


def pgn_to_samples(pgn_path: str, max_games: Optional[int] = None,
                   min_rating: Optional[int] = None,
                   use_cache: bool = True, cache_dir: str = "cache",
                   num_parse_workers: int = 1
                   ) -> Tuple[Dict[str, torch.Tensor], MoveEncoder]:
    """
    Parse PGN file into training tensors.

    Returns:
        tensor_dict: Dict with keys 'piece_ids' [N,64], 'sides' [N],
                     'moves' [N], 'values' [N]
        move_encoder: MoveEncoder instance with learned vocabulary
    """
    if use_cache:
        cache_path = get_cache_path(pgn_path, max_games, min_rating, cache_dir)
        cached = load_cache(cache_path)
        if cached is not None:
            return cached

    if num_parse_workers > 1:
        tensor_dict, move_encoder, game_count = _parse_parallel(
            pgn_path, max_games, min_rating, num_parse_workers
        )
    else:
        raw_samples, game_count = _parse_sequential(
            pgn_path, max_games, min_rating
        )
        print(f"Parsed {game_count:,} games -> {len(raw_samples):,} positions, tensorising...")
        move_encoder = MoveEncoder()
        tensor_dict = _tensorise(raw_samples, move_encoder)
        del raw_samples

    n = tensor_dict['piece_ids'].shape[0]
    print(f"Parsed {game_count:,} games -> {n:,} positions")
    print(f"Move vocabulary size: {move_encoder.get_vocab_size():,}")
    mem_mb = sum(t.element_size() * t.nelement() for t in tensor_dict.values()) / (1024 * 1024)
    print(f"Tensor memory: {mem_mb:.1f} MB ({n:,} samples)")

    if use_cache:
        cache_path = get_cache_path(pgn_path, max_games, min_rating, cache_dir)
        save_cache(cache_path, tensor_dict, move_encoder)

    return tensor_dict, move_encoder


# ============================================================
#  DATASET
# ============================================================
class ChessDataset(Dataset):
    """PyTorch Dataset backed by contiguous tensors.

    Each __getitem__ call is a single tensor index operation (no allocation,
    no Python object creation). Combined with pin_memory and prefetching,
    this keeps dual-GPU setups fully fed.
    """

    def __init__(self, tensor_dict: Dict[str, torch.Tensor]):
        self.piece_ids = tensor_dict['piece_ids']
        self.sides = tensor_dict['sides']
        self.moves = tensor_dict['moves']
        self.values = tensor_dict['values']

    def __len__(self):
        return self.piece_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "piece_ids": self.piece_ids[idx].long(),
            "side": self.sides[idx].long(),
            "move": self.moves[idx].long(),
            "result": self.values[idx],
        }


def collate_fn(batch):
    """Custom collate (kept for backward compatibility).
    PyTorch's default collate handles dict-of-tensors identically."""
    return {
        "piece_ids": torch.stack([b["piece_ids"] for b in batch]),
        "side": torch.stack([b["side"] for b in batch]),
        "move": torch.stack([b["move"] for b in batch]),
        "result": torch.stack([b["result"] for b in batch]),
    }


# ============================================================
#  FAST BATCH LOADER (replaces DataLoader for in-memory data)
# ============================================================
class TensorBatchLoader:
    """Batch loader that does batch-level tensor indexing.

    DataLoader calls __getitem__ once per sample then collates, adding
    significant Python overhead for large batches. This class instead
    indexes the full tensor with a batch of indices in a single op,
    which is orders of magnitude faster for in-memory datasets.

    Shares the underlying tensors between train/val splits via an
    ``indices`` array -- no data duplication.
    """

    def __init__(self, piece_ids: torch.Tensor, sides: torch.Tensor,
                 moves: torch.Tensor, values: torch.Tensor,
                 indices: torch.Tensor = None, batch_size: int = 2048,
                 shuffle: bool = True, drop_last: bool = False,
                 pin_memory: bool = False):
        self.piece_ids = piece_ids
        self.sides = sides
        self.moves = moves
        self.values = values
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self._n = len(indices) if indices is not None else piece_ids.shape[0]

    def __len__(self):
        if self.drop_last:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self._n)
            order = self.indices[perm] if self.indices is not None else perm
        else:
            order = self.indices if self.indices is not None else torch.arange(self._n)

        for start in range(0, self._n, self.batch_size):
            end = start + self.batch_size
            if end > self._n:
                if self.drop_last:
                    return
                end = self._n
            idx = order[start:end]
            batch = {
                "piece_ids": self.piece_ids[idx].long(),
                "side": self.sides[idx].long(),
                "move": self.moves[idx].long(),
                "result": self.values[idx],
            }
            if self.pin_memory:
                batch = {k: v.pin_memory() for k, v in batch.items()}
            yield batch


# ============================================================
#  CUDA PREFETCHER (overlaps CPU batch prep with GPU compute)
# ============================================================
class CUDAPrefetcher:
    """Prefetches batches to GPU on a separate CUDA stream.

    Wraps any batch iterator (e.g. TensorBatchLoader) and transfers the
    *next* batch to the GPU while the current batch is being computed,
    hiding the CPU->GPU transfer latency almost entirely.

    When the loader already uses ``pin_memory=True``, the non-blocking
    transfer on the prefetch stream is a true async DMA copy.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        it = iter(self.loader)
        try:
            batch = next(it)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        for next_batch in it:
            torch.cuda.current_stream().wait_stream(self.stream)
            yield batch
            batch = next_batch
            with torch.cuda.stream(self.stream):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        torch.cuda.current_stream().wait_stream(self.stream)
        yield batch


# ============================================================
#  DATA LOADER UTILITIES
# ============================================================
def get_optimal_workers() -> int:
    """Auto-detect optimal DataLoader workers for the current machine.

    Returns 0 for in-memory tensor datasets because __getitem__ is a
    single tensor index (nanoseconds) and multi-worker spawn on Windows
    would copy the entire dataset to each worker process.
    """
    return 0


def create_dataloader(pgn_path: str, batch_size: int = 32,
                      max_games: Optional[int] = None,
                      min_rating: Optional[int] = None,
                      shuffle: bool = True,
                      num_workers: Optional[int] = None,
                      use_cache: bool = True, cache_dir: str = "cache",
                      return_samples: bool = False,
                      num_parse_workers: int = 1,
                      pin_memory: Optional[bool] = None,
                      prefetch_factor: int = 4,
                      persistent_workers: Optional[bool] = None,
                      drop_last: bool = False
                      ) -> Tuple[DataLoader, MoveEncoder]:
    """
    Create an optimised DataLoader from a PGN file.

    Defaults are tuned for multi-GPU CUDA setups with ample RAM:
      - pin_memory=True when CUDA is available
      - persistent_workers=True when num_workers > 0
      - prefetch_factor=4 to keep GPUs fed
      - num_workers auto-detected (up to 12)
    """
    tensor_dict, move_encoder = pgn_to_samples(
        pgn_path, max_games, min_rating, use_cache, cache_dir, num_parse_workers
    )
    dataset = ChessDataset(tensor_dict)

    if num_workers is None:
        num_workers = get_optimal_workers()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'drop_last': drop_last,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, move_encoder
