#!/usr/bin/env python3
"""Fetch pinned external assets with streamed downloads and SHA-256 verification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    {
        "filename": "stories15M.bin",
        "url": "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin",
        "size": 60816028,
        "sha256": "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a",
    },
    {
        "filename": "tokenizer.bin",
        "url": "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin",
        "size": 433869,
        "sha256": "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
    },
)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verified(path, asset):
    return (
        path.is_file()
        and path.stat().st_size == asset["size"]
        and digest(path) == asset["sha256"]
    )


def fetch(directory, asset, verify_only=False):
    destination = directory / asset["filename"]
    if verified(destination, asset):
        print(f"Verified {destination.name} ({asset['size']:,} bytes)")
        return
    if verify_only:
        raise ValueError(f"Missing or invalid asset: {destination}")
    # Never truncate an existing checkpoint until its replacement is verified.
    temporary = directory / (asset["filename"] + f".{os.getpid()}.part")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "TensorTrace/0.1"})
    checksum, received = hashlib.sha256(), 0
    try:
        print(f"Downloading {asset['filename']}…", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > asset["size"]:
                    raise ValueError("Download exceeds the pinned asset size")
                out.write(chunk)
                checksum.update(chunk)
        if received != asset["size"] or checksum.hexdigest() != asset["sha256"]:
            raise ValueError(f"Asset checksum/size mismatch: {asset['filename']}")
        temporary.replace(destination)
        print(f"Verified {destination.name}: {asset['sha256']}")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "models")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    try:
        for asset in ASSETS:
            fetch(args.dir, asset, args.verify_only)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Model setup failed: {error}\n")
    if not args.verify_only:
        (args.dir / "provenance.json").write_text(json.dumps({"assets": ASSETS}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
