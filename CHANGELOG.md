# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-11-15

### Fixed
- **Inference**: Fixed `predict_move_from_legal()` to properly use model outputs instead of returning uniform probabilities
- **Move Encoder**: `MoveEncoder` is now saved with all model checkpoints, enabling proper inference without training data
- **Move Prediction**: Inference now correctly maps model outputs to legal moves using the saved `MoveEncoder`

### Added
- **Validation Split**: Implemented proper train/validation split with configurable ratio (default: 10%)
- **Validation Loop**: Validation now runs during training with proper metrics tracking
- **Best Model Saving**: Best model is now saved based on validation loss (previously disabled)
- **Game Phase Filtering**: Value learning now filters by game phase - early positions (first 30%) use neutral value to improve signal quality
- **Move Legality Filtering**: Inference properly filters to legal moves only and renormalizes probabilities

### Changed
- **Value Learning Signal**: Early game positions now use neutral value (0.0) instead of game outcome to reduce incorrect training signals
- **Checkpoint Format**: Checkpoints now include `MoveEncoder` state for model portability
- **Training Output**: Training now displays validation metrics and train/val split information

### Technical Details
- Validation split uses `random_split` from PyTorch
- Game phase is calculated as `move_count / total_moves` with threshold at 0.3 (30%)
- Move encoder state includes: `move_to_id`, `id_to_move`, and `next_move_id`
- Inference handles moves not in vocabulary gracefully (assigns 0.0 probability)

## [0.1.0] - 2024-11-15

### Added
- Initial proof-of-concept implementation
- Transformer-based chess AI with policy and value heads
- PGN parsing with Zstandard compression support
- Caching system for parsed PGN data
- MPS (Apple Silicon) GPU support
- Training and inference utilities
- Interactive CLI for testing

[Unreleased]: https://github.com/yourusername/chess-ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/chess-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/chess-ai/releases/tag/v0.1.0

