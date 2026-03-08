"""Quick test: load cached data, verify shapes, run a DataLoader batch."""
import sys
import os
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader

if __name__ == "__main__":
    cache_files = glob.glob("cache/*.cache")
    if not cache_files:
        raise SystemExit("No cache files found")
    cache_path = cache_files[0]
    print(f"Loading: {cache_path}")

    t0 = time.time()
    from src.data import load_cache, ChessDataset
    td, me = load_cache(cache_path)
    t1 = time.time()

    n = td["piece_ids"].shape[0]
    print(f"Cache loaded in {t1 - t0:.1f}s")
    print(f"Positions:  {n:,}")
    print(f"Move vocab: {me.get_vocab_size()}")
    print(f"Dtypes:     piece_ids={td['piece_ids'].dtype}, "
          f"sides={td['sides'].dtype}, "
          f"moves={td['moves'].dtype}, "
          f"values={td['values'].dtype}")

    ds = ChessDataset(td)
    assert len(ds) == n
    s = ds[0]
    assert s["piece_ids"].shape == (64,)
    assert s["side"].shape == ()
    assert s["move"].shape == ()
    assert s["result"].shape == ()
    print(f"Dataset[0]: piece_ids={s['piece_ids'].shape}, "
          f"side={s['side'].shape}, move={s['move'].shape}, result={s['result'].shape}")

    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2,
                        persistent_workers=True, pin_memory=True)
    batch = next(iter(loader))
    assert batch["piece_ids"].shape == (64, 64)
    assert batch["move"].shape == (64,)
    assert batch["result"].shape == (64,)
    print(f"Batch:      piece_ids={batch['piece_ids'].shape}, "
          f"move={batch['move'].shape}, result={batch['result'].shape}")

    print("\nALL TESTS PASSED")
