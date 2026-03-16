"""
Headless game runner for chess matches.
---------------------------------------
Plays full games between two players (model, random, or UCI engine).
Used by scripts/estimate_elo.py for ELO estimation.
"""

import math
import random
import chess
import chess.engine
from typing import Optional, Tuple, Callable, Any

from .inference import predict_move_from_legal
from .model import MiniChessTransformer
from .data import MoveEncoder


MAX_MOVES = 500


def play_game(
    white_player: Callable[[chess.Board], Optional[chess.Move]],
    black_player: Callable[[chess.Board], Optional[chess.Move]],
    start_fen: Optional[str] = None,
) -> Tuple[str, int, str]:
    """
    Play one game between two players.

    Args:
        white_player: Callable that takes the current board and returns a legal move (or None to resign).
        black_player: Same for Black.
        start_fen: Starting FEN (default: standard initial position).

    Returns:
        Tuple of (result, move_count, termination).
        result: "1-0", "0-1", or "1/2-1/2".
        move_count: Number of full moves played.
        termination: Short description (e.g. "checkmate", "stalemate", "50 moves", "resignation").
    """
    board = chess.Board(fen=start_fen) if start_fen else chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < MAX_MOVES:
        player = white_player if board.turn == chess.WHITE else black_player
        move = player(board)
        if move is None:
            # Resignation: opponent wins
            if board.turn == chess.WHITE:
                return "0-1", move_count, "resignation"
            return "1-0", move_count, "resignation"
        if move not in board.legal_moves:
            # Illegal move treated as resignation
            if board.turn == chess.WHITE:
                return "0-1", move_count, "illegal"
            return "1-0", move_count, "illegal"
        board.push(move)
        move_count += 1

    if board.is_game_over():
        result = board.result()
        term = "checkmate" if board.is_checkmate() else "stalemate" if board.is_stalemate() else "draw"
        return result, move_count, term
    # Max moves exceeded — treat as draw
    return "1/2-1/2", move_count, "max_moves"


def make_model_player(
    model: MiniChessTransformer,
    move_encoder: Optional[MoveEncoder],
    temperature: float = 0.0,
) -> Callable[[chess.Board], Optional[chess.Move]]:
    """
    Build a player that uses the transformer model to choose moves.

    Args:
        model: Loaded MiniChessTransformer.
        move_encoder: Move encoder from checkpoint (required for correct move decoding).
        temperature: If > 0, sample from policy; if 0, play top move.

    Returns:
        A callable(board) -> move.
    """

    def player(board: chess.Board) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        top = predict_move_from_legal(
            model, board, move_encoder=move_encoder, top_k=min(len(legal), 10)
        )
        if not top:
            return random.choice(legal)
        if temperature <= 0:
            return top[0][0]
        moves, probs = zip(*top)
        probs = [max(p, 1e-9) for p in probs]
        if temperature != 1:
            logits = [math.log(p) for p in probs]
            inv_t = 1.0 / temperature
            probs = [math.exp(inv_t * x) for x in logits]
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(moves, weights=probs, k=1)[0]

    return player


def make_random_player() -> Callable[[chess.Board], Optional[chess.Move]]:
    """Build a player that picks a random legal move."""

    def player(board: chess.Board) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        return random.choice(legal)

    return player


def make_uci_engine_player(
    engine_path: str,
    elo: Optional[int] = None,
    time_limit: float = 0.1,
) -> Tuple[Callable[[chess.Board], Optional[chess.Move]], Any]:
    """
    Spawn a UCI engine (e.g. Stockfish) and build a player.

    Args:
        engine_path: Path to engine binary.
        elo: If set, configure UCI_LimitStrength and UCI_Elo (Stockfish).
        time_limit: Time per move in seconds.

    Returns:
        (player_callable, engine). Caller must close the engine when done.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    if elo is not None:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def player(board: chess.Board) -> Optional[chess.Move]:
        try:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move
        except Exception:
            return None

    return player, engine


def find_stockfish() -> Optional[str]:
    """Try to find Stockfish binary in PATH or common locations."""
    import shutil
    path = shutil.which("stockfish")
    if path:
        return path
    for candidate in ["/usr/bin/stockfish", "/usr/local/bin/stockfish"]:
        import os
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None
