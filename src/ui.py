"""
Interactive Chess UI with Gradio
---------------------------------
Web-based interface for playing chess against the AI or watching AI vs AI games.
"""

import gradio as gr
import chess
import chess.pgn
import torch
from typing import Optional, Tuple, List
import os
from pathlib import Path
import time
import threading

from gradio_chessboard import Chessboard
from inference import load_model, predict_move_from_legal, evaluate_position
from model import MiniChessTransformer
from data import MoveEncoder


# Global state
current_model: Optional[MiniChessTransformer] = None
current_move_encoder: Optional[MoveEncoder] = None
current_board: Optional[chess.Board] = None
model_info: str = "No model loaded"
ai_vs_ai_running: bool = False
ai_vs_ai_thread: Optional[threading.Thread] = None


def load_model_from_file(file_path: str) -> Tuple[str, str]:
    """
    Load a model checkpoint from file.
    
    Args:
        file_path: Path to .pt checkpoint file
    
    Returns:
        Tuple of (status message, model info)
    """
    global current_model, current_move_encoder, model_info
    
    if not file_path or not os.path.exists(file_path):
        return "Please select a valid model file.", "No model loaded"
    
    try:
        model, checkpoint_info = load_model(file_path)
        current_model = model
        current_move_encoder = checkpoint_info.get('move_encoder')
        
        # Extract model info
        epoch = checkpoint_info.get('epoch', 'Unknown')
        val_loss = checkpoint_info.get('val_loss', 'N/A')
        config = checkpoint_info.get('config', {})
        
        model_info = f"Epoch: {epoch} | Val Loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"Epoch: {epoch}"
        if config:
            model_info += f" | Config: {config.get('hidden_dim', 'N/A')}d, {config.get('n_layers', 'N/A')} layers"
        
        return f"✅ Model loaded successfully from {os.path.basename(file_path)}", model_info
    except Exception as e:
        return f"❌ Error loading model: {str(e)}", "No model loaded"


def reset_game() -> Tuple[str, str]:
    """
    Reset the game to starting position.
    
    Returns:
        Tuple of (FEN string, status message)
    """
    global current_board
    current_board = chess.Board()
    return current_board.fen(), "Game reset. White to move."


def get_game_status() -> str:
    """
    Get current game status and analysis.
    
    Returns:
        Status string with turn, evaluation, and top moves
    """
    global current_board, current_model, current_move_encoder
    
    if current_board is None:
        return "No game in progress. Click 'New Game' to start."
    
    # Check game end conditions
    if current_board.is_checkmate():
        winner = "Black" if current_board.turn == chess.WHITE else "White"
        return f"🏁 Checkmate! {winner} wins."
    elif current_board.is_stalemate():
        return "🏁 Stalemate! Game drawn."
    elif current_board.is_insufficient_material():
        return "🏁 Insufficient material! Game drawn."
    elif current_board.is_seventyfive_moves():
        return "🏁 75-move rule! Game drawn."
    elif current_board.is_fivefold_repetition():
        return "🏁 Fivefold repetition! Game drawn."
    
    # Get turn
    turn_str = "White" if current_board.turn == chess.WHITE else "Black"
    status = f"{turn_str} to move"
    
    # Get evaluation and top moves if model is loaded
    if current_model is not None and current_move_encoder is not None:
        try:
            # Evaluate position
            eval_score = evaluate_position(current_model, current_board)
            status += f" | Eval: {eval_score:+.3f}"
            
            # Get top moves
            top_moves = predict_move_from_legal(
                current_model, current_board, 
                move_encoder=current_move_encoder, 
                top_k=5
            )
            
            if top_moves:
                moves_str = " | Top moves: "
                moves_list = []
                for move, prob in top_moves:
                    moves_list.append(f"{move.uci()}({prob:.2f})")
                status += moves_str + ", ".join(moves_list)
        except Exception as e:
            status += f" | Error: {str(e)}"
    
    return status


def make_ai_move(fen: str) -> Tuple[str, str, str]:
    """
    Make an AI move on the current board.
    
    Args:
        fen: Current FEN string
    
    Returns:
        Tuple of (new FEN, status message, move history)
    """
    global current_board, current_model, current_move_encoder
    
    if current_model is None or current_move_encoder is None:
        return fen, "❌ No model loaded. Please load a model first.", ""
    
    if current_board is None:
        current_board = chess.Board(fen)
    
    # Check if game is over
    if current_board.is_game_over():
        status = get_game_status()
        move_history = get_move_history()
        return current_board.fen(), status, move_history
    
    # Get AI move
    try:
        top_moves = predict_move_from_legal(
            current_model, current_board,
            move_encoder=current_move_encoder,
            top_k=1
        )
        
        if not top_moves:
            return current_board.fen(), "❌ No legal moves available.", get_move_history()
        
        # Play best move
        ai_move = top_moves[0][0]
        current_board.push(ai_move)
        
        status = get_game_status()
        move_history = get_move_history()
        
        return current_board.fen(), status, move_history
    except Exception as e:
        return current_board.fen(), f"❌ Error making AI move: {str(e)}", get_move_history()


def handle_user_move(fen: str) -> Tuple[str, str, str]:
    """
    Handle a move made by the user on the chessboard.
    
    Args:
        fen: New FEN string after user move
    
    Returns:
        Tuple of (FEN, status, move history)
    """
    global current_board
    
    if current_board is None:
        current_board = chess.Board()
    
    # Update board from FEN
    try:
        new_board = chess.Board(fen)
        current_board = new_board
    except Exception as e:
        return current_board.fen() if current_board else chess.Board().fen(), f"❌ Invalid position: {str(e)}", ""
    
    status = get_game_status()
    move_history = get_move_history()
    
    return current_board.fen(), status, move_history


def undo_move() -> Tuple[str, str, str]:
    """
    Undo the last move.
    
    Returns:
        Tuple of (FEN, status, move history)
    """
    global current_board
    
    if current_board is None or len(current_board.move_stack) == 0:
        return chess.Board().fen(), "No moves to undo.", ""
    
    current_board.pop()
    status = get_game_status()
    move_history = get_move_history()
    
    return current_board.fen(), status, move_history


def get_move_history() -> str:
    """
    Get move history as PGN or move list.
    
    Returns:
        Move history string
    """
    global current_board
    
    if current_board is None or len(current_board.move_stack) == 0:
        return "No moves yet."
    
    # Create a game from the board
    game = chess.pgn.Game()
    game.headers["White"] = "Player"
    game.headers["Black"] = "AI"
    
    node = game
    temp_board = chess.Board()
    
    for move in current_board.move_stack:
        node = node.add_variation(move)
        temp_board.push(move)
    
    # Get PGN string
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter)
    
    # Format as numbered moves
    moves = pgn_string.strip().split('\n')[-1] if pgn_string else ""
    if moves:
        move_list = moves.split()
        formatted = []
        for i in range(0, len(move_list), 2):
            move_num = (i // 2) + 1
            white_move = move_list[i] if i < len(move_list) else ""
            black_move = move_list[i + 1] if i + 1 < len(move_list) else ""
            formatted.append(f"{move_num}. {white_move} {black_move}")
        return "\n".join(formatted)
    
    return moves if moves else "No moves yet."


def ai_vs_ai_step(fen: str, delay: float) -> Tuple[str, str, str, bool]:
    """
    Execute one step of AI vs AI game.
    
    Args:
        fen: Current FEN string
        delay: Delay between moves (not used in single step)
    
    Returns:
        Tuple of (FEN, status, move history, continue flag)
    """
    global current_board, current_model, current_move_encoder, ai_vs_ai_running
    
    if current_model is None or current_move_encoder is None:
        return fen, "❌ No model loaded.", "", False
    
    if current_board is None:
        current_board = chess.Board(fen)
    
    # Check if game is over
    if current_board.is_game_over():
        ai_vs_ai_running = False
        status = get_game_status()
        move_history = get_move_history()
        return current_board.fen(), status, move_history, False
    
    # Make AI move
    try:
        top_moves = predict_move_from_legal(
            current_model, current_board,
            move_encoder=current_move_encoder,
            top_k=1
        )
        
        if not top_moves:
            ai_vs_ai_running = False
            return current_board.fen(), "❌ No legal moves available.", get_move_history(), False
        
        ai_move = top_moves[0][0]
        current_board.push(ai_move)
        
        status = get_game_status()
        move_history = get_move_history()
        
        # Continue if game not over
        continue_game = not current_board.is_game_over() and ai_vs_ai_running
        
        return current_board.fen(), status, move_history, continue_game
    except Exception as e:
        ai_vs_ai_running = False
        return current_board.fen(), f"❌ Error: {str(e)}", get_move_history(), False


def toggle_ai_vs_ai(fen: str, enabled: bool, delay: float) -> Tuple[str, str, str, bool]:
    """
    Toggle AI vs AI mode.
    
    Args:
        fen: Current FEN string
        enabled: Whether AI vs AI is enabled
        delay: Delay between moves in seconds
    
    Returns:
        Tuple of (FEN, status, move history, enabled state)
    """
    global ai_vs_ai_running, ai_vs_ai_thread
    
    if enabled and not ai_vs_ai_running:
        ai_vs_ai_running = True
        status = "AI vs AI started. Game will play automatically."
        return fen, status, get_move_history(), True
    elif not enabled and ai_vs_ai_running:
        ai_vs_ai_running = False
        status = "AI vs AI stopped."
        return fen, status, get_move_history(), False
    else:
        return fen, get_game_status(), get_move_history(), enabled


def ai_vs_ai_continuous(fen: str, enabled: bool, delay: float) -> Tuple[str, str, str, bool]:
    """
    Continuous AI vs AI gameplay.
    
    Args:
        fen: Current FEN string
        enabled: Whether AI vs AI is enabled
        delay: Delay between moves in seconds
    
    Returns:
        Tuple of (FEN, status, move history, continue flag)
    """
    global ai_vs_ai_running
    
    if not enabled or not ai_vs_ai_running:
        return fen, get_game_status(), get_move_history(), False
    
    # Make one move
    new_fen, status, history, continue_flag = ai_vs_ai_step(fen, delay)
    
    # If game continues and AI vs AI is still enabled, return True to trigger next update
    if continue_flag and ai_vs_ai_running:
        time.sleep(delay)
        return new_fen, status, history, True
    else:
        ai_vs_ai_running = False
        return new_fen, status, history, False


def create_chess_ui() -> gr.Blocks:
    """
    Create the main Gradio chess UI interface.
    
    Returns:
        Gradio Blocks interface
    """
    global current_board
    
    # Initialize board
    current_board = chess.Board()
    
    with gr.Blocks(title="Chess AI - Interactive Playground") as demo:
        gr.Markdown("# ♟️ Chess AI - Interactive Playground")
        gr.Markdown("Load a trained model and play chess against the AI, or watch it play against itself!")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model selection
                gr.Markdown("## Model Selection")
                model_file = gr.File(
                    label="Select Model Checkpoint",
                    file_types=[".pt"],
                    value=None
                )
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False
                )
                model_info_display = gr.Textbox(
                    label="Model Info",
                    value="No model loaded",
                    interactive=False
                )
                
                # Chess board
                gr.Markdown("## Chess Board")
                chessboard = Chessboard(
                    label="Board",
                    game_mode=True,
                    value=current_board.fen()
                )
                
                # Game controls
                gr.Markdown("## Game Controls")
                with gr.Row():
                    new_game_btn = gr.Button("🔄 New Game", variant="primary")
                    undo_btn = gr.Button("↩️ Undo Move")
                    ai_move_btn = gr.Button("🤖 AI Move")
                
                # AI vs AI controls
                gr.Markdown("## AI vs AI Mode")
                with gr.Row():
                    ai_vs_ai_toggle = gr.Checkbox(
                        label="Enable AI vs AI",
                        value=False
                    )
                    ai_vs_ai_delay = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Delay between moves (seconds)"
                    )
                
            with gr.Column(scale=1):
                # Status and analysis
                gr.Markdown("## Game Status")
                game_status = gr.Textbox(
                    label="Status",
                    value="White to move",
                    interactive=False,
                    lines=5
                )
                
                # Move history
                gr.Markdown("## Move History")
                move_history = gr.Textbox(
                    label="Moves",
                    value="No moves yet.",
                    interactive=False,
                    lines=10
                )
        
        # Event handlers
        model_file.change(
            fn=load_model_from_file,
            inputs=[model_file],
            outputs=[model_status, model_info_display]
        )
        
        new_game_btn.click(
            fn=reset_game,
            outputs=[chessboard, game_status]
        )
        
        undo_btn.click(
            fn=undo_move,
            outputs=[chessboard, game_status, move_history]
        )
        
        ai_move_btn.click(
            fn=make_ai_move,
            inputs=[chessboard],
            outputs=[chessboard, game_status, move_history]
        )
        
        # Handle user moves on chessboard
        chessboard.move(
            fn=handle_user_move,
            inputs=[chessboard],
            outputs=[chessboard, game_status, move_history]
        )
        
        # AI vs AI toggle
        ai_vs_ai_toggle.change(
            fn=toggle_ai_vs_ai,
            inputs=[chessboard, ai_vs_ai_toggle, ai_vs_ai_delay],
            outputs=[chessboard, game_status, move_history, ai_vs_ai_toggle]
        )
        
        # AI vs AI play button for manual stepping
        ai_vs_ai_play_btn = gr.Button("▶️ Play Next Move (AI vs AI)", visible=True)
        
        def play_ai_vs_ai_move(fen: str, delay: float):
            """Play one AI vs AI move."""
            global ai_vs_ai_running
            if not ai_vs_ai_running:
                ai_vs_ai_running = True
            new_fen, status, history, continue_flag = ai_vs_ai_step(fen, delay)
            # Update toggle state based on continue flag
            return new_fen, status, history, continue_flag
        
        ai_vs_ai_play_btn.click(
            fn=play_ai_vs_ai_move,
            inputs=[chessboard, ai_vs_ai_delay],
            outputs=[chessboard, game_status, move_history, ai_vs_ai_toggle]
        )
        
        # Auto-play AI vs AI with timer
        def auto_play_check():
            """Check if AI vs AI should continue and make a move automatically."""
            global ai_vs_ai_running, current_board
            
            if not ai_vs_ai_running or current_board is None:
                return (
                    current_board.fen() if current_board else chess.Board().fen(),
                    get_game_status(),
                    get_move_history(),
                    False
                )
            
            if current_board.is_game_over():
                ai_vs_ai_running = False
                return (
                    current_board.fen(),
                    get_game_status(),
                    get_move_history(),
                    False
                )
            
            # Make a move
            delay = 1.0  # Use default delay for auto-play
            new_fen, status, history, continue_flag = ai_vs_ai_step(current_board.fen(), delay)
            
            return new_fen, status, history, continue_flag
        
        # Set up periodic auto-play when AI vs AI is enabled
        # This will check every 2 seconds and make moves if enabled
        demo.load(
            fn=auto_play_check,
            outputs=[chessboard, game_status, move_history, ai_vs_ai_toggle],
            every=2.0  # Check every 2 seconds
        )
    
    return demo


def main():
    """Launch the chess UI."""
    demo = create_chess_ui()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

