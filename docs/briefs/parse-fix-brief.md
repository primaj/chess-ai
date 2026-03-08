# Fix: `_parse_parallel` in `src/data.py` — Memory + Progress Issues

## System

- **CPU**: 32 logical cores
- **RAM**: 128 GB
- **OS**: Windows 10
- **Python**: 3.10 (venv at `.venv\Scripts\activate`)
- **File being parsed**: `data/lichess_db_standard_rated_2016-01.pgn.zst` (831 MB compressed, 4.25 GB decompressed, 4,770,357 games)

## What happened

I ran the parse with `PARSE_WORKERS = 30` (auto-detected as `cpu_count() - 2`). The bulk decompression + text splitting phase worked great (16 seconds for 4.77M games). But during the parallel parsing phase:

1. **RAM hit 99%** — 30 worker processes each hold copies of their PGN string chunks (sent via pickle IPC) plus all their parsed results. The main process also collects ALL results via `list(pool.imap(...))` before proceeding. With ~4 GB of PGN strings duplicated across processes plus ~286M parsed position tuples, this blows past 128 GB.

2. **Progress bar stuck at 0%** — Each chunk is ~79,000 games (`game_count // (num_workers * 2)`), so a single worker takes 4+ minutes per chunk. tqdm only updates when a chunk completes, so it appears frozen.

## What needs fixing

The function `_parse_parallel` in `src/data.py` (line ~412). Here's the current code:

```python
def _parse_parallel(pgn_path: str, max_games: Optional[int],
                    min_rating: Optional[int],
                    num_workers: int) -> Tuple[List[Tuple], int]:
    """Fast parallel parsing: bulk-read raw text, then distribute to workers."""
    pgn_strings = _read_raw_pgn_games(pgn_path, max_games)
    game_count = len(pgn_strings)
    print(f"Distributing {game_count:,} games to {num_workers} workers...")

    chunk_size = max(100, game_count // (num_workers * 2))
    pgn_chunks = [
        pgn_strings[i:i + chunk_size]
        for i in range(0, game_count, chunk_size)
    ]
    del pgn_strings

    with Pool(processes=num_workers) as pool:
        chunk_args = [(chunk, min_rating) for chunk in pgn_chunks]
        results = list(tqdm(
            pool.imap(parse_games_chunk, chunk_args),
            total=len(pgn_chunks),
            desc="Parsing games (parallel)"
        ))

    raw_samples = []
    for chunk_samples in results:
        raw_samples.extend(chunk_samples)
    del results

    return raw_samples, game_count
```

The worker function it dispatches to (`parse_games_chunk`, line ~282):

```python
def parse_games_chunk(args: Tuple) -> List[Tuple]:
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
    return samples
```

Each sample tuple returned is `(np.ndarray[64, int8], np.int8, str, np.float32)` — a board array, side to move, UCI move string, and value.

## Requirements for the fix

1. **Memory**: Don't hold all PGN strings AND all results in memory simultaneously. Process and free incrementally. Reduce concurrent worker count to ~16 max — each Python worker process uses significant baseline RAM.

2. **Progress bar**: Use much smaller chunks (e.g. 500-2000 games per chunk instead of 79,000) so tqdm updates frequently. With 500-game chunks and 4.77M games, that's ~9,500 chunks — tqdm will update every few seconds.

3. **Keep the same return signature**: `_parse_parallel` must still return `(List[Tuple], int)` — the list of raw sample tuples and the game count. Downstream code (`pgn_to_samples`) depends on this.

4. **Keep `_read_raw_pgn_games` as-is** — the bulk decompression + split is fast and correct. Only `_parse_parallel` needs changing.

5. **Use `imap_unordered`** instead of `imap` — order doesn't matter for training data, and unordered returns results as soon as workers finish rather than waiting for sequential order.

6. **Free chunk data progressively** — once a chunk's PGN strings have been dispatched, they should be eligible for GC. Once a chunk's results have been extended into `raw_samples`, the chunk result list should be freed.

## How to test

```powershell
.venv\Scripts\activate
python run_parse.py
```

The file `data/lichess_db_standard_rated_2016-01.pgn.zst` is already downloaded (831 MB). The script (`run_parse.py`) calls `pgn_to_samples` with `num_parse_workers=30`. You may want to cap that to 16 in the fix.

Expected output: ~4.77M games parsed into ~280-320M positions, tensorised and cached to `cache/`. RAM should stay well under 128 GB. Progress bar should update every few seconds.
