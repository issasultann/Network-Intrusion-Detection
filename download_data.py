"""
CICIDS2017 Fast Downloader
---------------------------
Downloads the dataset from a Hugging Face mirror (much faster than UNB server).

Files (5 CSVs, ~810 MB total):
  monday.csv     146 MB    BENIGN traffic
  tuesday.csv    125 MB    BENIGN + FTP-Patator + SSH-Patator
  wednesday.csv  207 MB    BENIGN + DoS Hulk + GoldenEye + Slowloris + SlowHTTP + Heartbleed
  thursday.csv   132 MB    BENIGN + Web Attacks + Infiltration
  friday.csv     200 MB    BENIGN + Bot + PortScan + DDoS

Source: https://huggingface.co/datasets/bvk/CICIDS-2017
"""
import os
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Hugging Face LFS direct-download URLs
FILES = {
    "monday.csv":    ("https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/monday.csv",    146_143_601),
    "tuesday.csv":   ("https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/tuesday.csv",   125_393_583),
    "wednesday.csv": ("https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/wednesday.csv", 206_984_560),
    "thursday.csv":  ("https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/thursday.csv",  132_500_204),
    "friday.csv":    ("https://huggingface.co/datasets/bvk/CICIDS-2017/resolve/main/friday.csv",    199_983_721),
}

TOTAL_BYTES = sum(sz for _, sz in FILES.values())
_lock_progress = {}


def _format_bytes(b):
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:6.1f} {unit}"
        b /= 1024
    return f"{b:6.1f} TB"


def download_file(name: str, url: str, expected_size: int, dest: Path) -> dict:
    """Download with progress, retries, and resumability."""
    path = dest / name

    # Skip if already exists with correct size
    if path.exists() and path.stat().st_size == expected_size:
        print(f"  [SKIP] {name} already downloaded ({_format_bytes(expected_size).strip()})")
        return {"name": name, "ok": True, "skipped": True}

    print(f"  [START] {name} ({_format_bytes(expected_size).strip()})")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(path, "wb") as f:
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB
                last_print = time.time()
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 3.0:
                        pct = 100 * downloaded / expected_size
                        speed = downloaded / max(now - t0, 0.1) / 1024 / 1024
                        print(f"  [..] {name:<14} {pct:5.1f}%  "
                              f"{_format_bytes(downloaded)} @ {speed:5.1f} MB/s")
                        last_print = now

        elapsed = time.time() - t0
        actual_size = path.stat().st_size
        ok = actual_size == expected_size
        speed = actual_size / max(elapsed, 0.1) / 1024 / 1024
        status = "OK" if ok else "SIZE MISMATCH"
        print(f"  [{status}] {name:<14} {_format_bytes(actual_size).strip()} "
              f"in {elapsed:.0f}s ({speed:.1f} MB/s)")
        return {"name": name, "ok": ok, "skipped": False, "elapsed": elapsed}
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return {"name": name, "ok": False, "error": str(e)}


def main():
    print("=" * 65)
    print(" CICIDS2017 Fast Downloader (Hugging Face mirror)")
    print("=" * 65)
    print(f" Target dir : {DATA_DIR.resolve()}")
    print(f" Total size : {_format_bytes(TOTAL_BYTES).strip()}")
    print(f" Files      : {len(FILES)} CSVs (Mon/Tue/Wed/Thu/Fri)")
    print("=" * 65)
    print("\nDownloading 3 files in parallel ...\n")

    t_start = time.time()
    results = []

    # 3 parallel downloads is the sweet spot for HF CDN
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(download_file, name, url, sz, DATA_DIR): name
                for name, (url, sz) in FILES.items()}
        for fut in as_completed(futs):
            results.append(fut.result())

    total_time = time.time() - t_start
    n_ok = sum(1 for r in results if r["ok"])

    print("\n" + "=" * 65)
    print(f" Completed: {n_ok}/{len(FILES)} files in {total_time:.0f}s "
          f"(avg {TOTAL_BYTES/total_time/1024/1024:.1f} MB/s)")
    print("=" * 65)

    if n_ok == len(FILES):
        print("\n SUCCESS - all files downloaded.")
        print("\nNext step: open pipeline.ipynb and 'Run All Cells'")
        print("(DEMO_MODE is already set to False in the config)")
    else:
        print("\n Some files failed. Re-run this script to retry "
              "(it skips already-downloaded files).")
        sys.exit(1)


if __name__ == "__main__":
    main()
