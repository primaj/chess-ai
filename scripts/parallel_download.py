"""Parallel chunk downloader to work around per-connection throttling."""
import os
import sys
import time
import requests
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-01.pgn.zst"
OUTPUT = "data/lichess_db_standard_rated_2016-01.pgn.zst"
NUM_CONNECTIONS = 8
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
}


def get_file_size(url):
    r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers.get("Content-Length", 0))


def download_chunk(url, start, end, chunk_id, parts_dir, progress):
    part_path = os.path.join(parts_dir, f"part_{chunk_id:02d}")
    existing = 0
    if os.path.exists(part_path):
        existing = os.path.getsize(part_path)
        if existing >= (end - start + 1):
            progress[chunk_id] = end - start + 1
            return
        start += existing

    headers = {**HEADERS, "Range": f"bytes={start}-{end}"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        mode = "ab" if existing > 0 else "wb"
        with open(part_path, mode) as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
                    progress[chunk_id] = existing + f.tell()
    except Exception as e:
        print(f"\nChunk {chunk_id} error: {e}")


def merge_parts(parts_dir, output, num_parts):
    with open(output, "wb") as out:
        for i in range(num_parts):
            part_path = os.path.join(parts_dir, f"part_{i:02d}")
            with open(part_path, "rb") as inp:
                while True:
                    data = inp.read(4 * 1024 * 1024)
                    if not data:
                        break
                    out.write(data)


def main():
    os.makedirs("data", exist_ok=True)
    parts_dir = "data/parts_2016-01"
    os.makedirs(parts_dir, exist_ok=True)

    if os.path.exists(OUTPUT):
        size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        print(f"Already exists: {OUTPUT} ({size_mb:.0f} MB)")
        return

    print(f"Getting file size...")
    total = get_file_size(URL)
    total_mb = total / (1024 * 1024)
    print(f"File size: {total_mb:.0f} MB")
    print(f"Using {NUM_CONNECTIONS} parallel connections\n")

    chunk_size = total // NUM_CONNECTIONS
    ranges = []
    for i in range(NUM_CONNECTIONS):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < NUM_CONNECTIONS - 1 else total - 1
        ranges.append((start, end))

    progress = [0] * NUM_CONNECTIONS
    threads = []
    t0 = time.time()

    for i, (start, end) in enumerate(ranges):
        t = threading.Thread(target=download_chunk,
                             args=(URL, start, end, i, parts_dir, progress))
        t.daemon = True
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        time.sleep(5)
        downloaded = sum(progress)
        elapsed = time.time() - t0
        speed_kbs = downloaded / 1024 / max(elapsed, 1)
        pct = downloaded / total * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        dl_mb = downloaded / (1024 * 1024)
        eta_s = (total - downloaded) / max(downloaded / max(elapsed, 1), 1)
        eta_m = eta_s / 60
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}%  {dl_mb:.0f}/{total_mb:.0f} MB  "
                         f"{speed_kbs:.0f} kB/s  ETA {eta_m:.0f}min  ")
        sys.stdout.flush()

    for t in threads:
        t.join()

    downloaded = sum(progress)
    elapsed = time.time() - t0
    speed = downloaded / 1024 / max(elapsed, 1)
    print(f"\n\nDownload complete: {downloaded / 1024 / 1024:.0f} MB in {elapsed:.0f}s "
          f"(avg {speed:.0f} kB/s)")

    print("Merging parts...")
    merge_parts(parts_dir, OUTPUT, NUM_CONNECTIONS)
    final_size = os.path.getsize(OUTPUT)
    print(f"Merged: {OUTPUT} ({final_size / 1024 / 1024:.0f} MB)")

    import shutil
    shutil.rmtree(parts_dir)
    print("Cleaned up parts. Done!")


if __name__ == "__main__":
    main()
