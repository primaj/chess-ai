"""Download and parse a significant Lichess dataset."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from src.download import construct_lichess_url, download_file
    from src.data import pgn_to_samples

    # --- Config ---
    DATE = "2016-01"
    MAX_GAMES = None
    MIN_RATING = None
    PARSE_WORKERS = max(1, os.cpu_count() - 2)  # leave 2 cores for OS / main thread

    os.makedirs("data", exist_ok=True)
    filename = f"lichess_db_standard_rated_{DATE}.pgn.zst"
    pgn_path = os.path.join("data", filename)

    # --- Download ---
    if os.path.exists(pgn_path):
        size_mb = os.path.getsize(pgn_path) / (1024 * 1024)
        print(f"File already exists: {pgn_path} ({size_mb:.0f} MB)")
    else:
        url = construct_lichess_url(DATE)
        print(f"Downloading: {url}")
        print(f"Output: {pgn_path}")
        success = download_file(url, pgn_path, resume=True)
        if not success:
            print("Download failed. You can resume by running again.")
            return

    # --- Parse ---
    print()
    print("=" * 60)
    mg = f"{MAX_GAMES:,}" if MAX_GAMES else "all"
    print(f"Parsing: max_games={mg}, min_rating={MIN_RATING}, "
          f"workers={PARSE_WORKERS}")
    print("=" * 60)

    t0 = time.time()
    tensor_dict, move_encoder = pgn_to_samples(
        pgn_path,
        max_games=MAX_GAMES,
        min_rating=MIN_RATING,
        use_cache=True,
        cache_dir="cache",
        num_parse_workers=PARSE_WORKERS,
    )
    elapsed = time.time() - t0

    n = tensor_dict["piece_ids"].shape[0]
    mem_mb = sum(
        t.element_size() * t.nelement() for t in tensor_dict.values()
    ) / (1024 * 1024)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Positions:    {n:,}")
    print(f"  Move vocab:   {move_encoder.get_vocab_size():,}")
    print(f"  Tensor RAM:   {mem_mb:.1f} MB")
    print(f"  Parse time:   {elapsed:.1f}s")
    print(f"  Cache dir:    cache/")
    print()
    print("Done! Cached and ready for training.")


if __name__ == "__main__":
    main()
