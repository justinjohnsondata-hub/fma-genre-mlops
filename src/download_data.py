"""Download and verify the FMA metadata archive.

Fetches the pinned ``fma_metadata.zip`` from the SWITCH object store, verifies its
integrity against the published SHA1 *before* extracting, then unzips the CSVs into
``data/raw/``. Integrity-first on purpose: a truncated or swapped download would
silently poison every downstream artifact, so we refuse to extract on a hash
mismatch rather than build on bad bytes.

Dataset: FMA (Free Music Archive), Defferrard et al., ISMIR 2017. Metadata is
licensed CC BY 4.0 (see README for attribution).

Usage:
    python src/download_data.py                # download + verify + extract
    python src/download_data.py --force        # re-download even if the zip exists
    python src/download_data.py --no-extract   # fetch + verify only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests

# --- Pinned source (do not float) ---
FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
DEFAULT_DEST = Path("data/raw")
CHUNK = 1 << 20  # 1 MiB streaming chunks


def sha1_of(path: Path, chunk: int = CHUNK) -> str:
    """Streaming SHA1 so we never hold 342 MiB in memory at once."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream the archive to disk with coarse MiB progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({pct:4.1f}%)",
                          end="", flush=True)
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download + verify + extract FMA metadata.")
    p.add_argument("--url", default=FMA_URL)
    p.add_argument("--sha1", default=FMA_SHA1)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="extraction dir")
    p.add_argument("--force", action="store_true", help="re-download even if the zip exists")
    p.add_argument("--no-extract", action="store_true", help="fetch + verify only, skip unzip")
    args = p.parse_args(argv)

    zip_path = args.dest / "fma_metadata.zip"

    # 1. Download (idempotent: skip if a copy is already on disk)
    if zip_path.exists() and not args.force:
        print(f"zip already present: {zip_path}")
    else:
        print(f"downloading {args.url}")
        download(args.url, zip_path)

    # 2. Verify integrity BEFORE trusting the bytes
    print("verifying SHA1 ...")
    got = sha1_of(zip_path)
    if got != args.sha1:
        print(f"SHA1 MISMATCH\n  expected {args.sha1}\n  got      {got}", file=sys.stderr)
        print("refusing to extract a corrupt/altered archive.", file=sys.stderr)
        return 1
    print(f"SHA1 OK: {got}")

    # 3. Extract (the zip unpacks a fma_metadata/ subdir of CSVs)
    if args.no_extract:
        print("--no-extract set; done.")
        return 0
    print(f"extracting into {args.dest} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.dest)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
