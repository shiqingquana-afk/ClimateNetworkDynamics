#!/usr/bin/env python3
"""Download the original ChaosNexus checkpoint from the authors' public Dropbox."""
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path


# Public share identifiers from the upstream README, not account credentials.
SHARE = "i2r3ds0emg9sdcqe7vm9q"
SHARE_KEY = "6ckgecajs35zbhek6hmpe0fel"
FILES = [
    ("config.json", "AMViBpI0XDPLeVyhhCI1uN0", 1661,
     "f6d5e3b3f740b380494e1789ab25d1f79be0282a4cac6e73219eda871f77b8ba"),
    ("training_info.json", "AA69UslOF-ljfnhCGFIoB50", 15694,
     "8d62d887fab6f4600319cc972c114b28539087ee0eac894d23ce834f344e028b"),
    ("model.safetensors", "AI_6_KoKfPCaLFI2SQ_rJus", 232434392,
     "14d230d89209923c868278a8ca6042c7077c582f916e8ce578bf3c5ffd9ab55e"),
]


def verify(path: Path, size: int, checksum: str) -> None:
    if path.stat().st_size != size:
        raise ValueError(f"Wrong file size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != checksum:
        raise ValueError(f"SHA256 mismatch: {path}. File retained for inspection; do not use it as a checkpoint.")


def download(path: Path, urls: list[str], size: int, checksum: str,
             retries: int = 8, timeout: float = 60) -> None:
    if path.exists():
        verify(path, size, checksum)
        print(f"Already verified: {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    failures, request_count = 0, 0
    last_progress = time.monotonic()
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise ValueError(f"Oversized partial file: {partial}; inspect it before retrying")
        url = urls[request_count % len(urls)]
        request_count += 1
        request = urllib.request.Request(url, headers={
            "User-Agent": "ChaosNexus-checkpoint-downloader/1.0",
            "Range": f"bytes={offset}-{size - 1}", "Cache-Control": "no-cache",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("The server returned an HTML page instead of weights")
                if response.status == 206:
                    expected = f"bytes {offset}-"
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(expected) or not content_range.endswith(f"/{size}"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                elif response.status != 200 or offset != 0:
                    raise ValueError("The server did not honor resume; existing bytes were kept")
                with partial.open("ab") as stream:
                    # read1 preserves received bytes even on slow/interrupted connections.
                    while True:
                        block = response.read1(256 * 1024)
                        if not block:
                            break
                        if stream.tell() + len(block) > size:
                            raise ValueError("Response exceeds expected file size")
                        stream.write(block)
                        if time.monotonic() - last_progress >= 10:
                            print(f"{path.name}: {stream.tell() / size:.1%}", flush=True)
                            last_progress = time.monotonic()
            if partial.stat().st_size == offset:
                raise OSError("No download progress")
            failures = 0
        except (OSError, ValueError, urllib.error.URLError) as exc:
            current = partial.stat().st_size if partial.exists() else 0
            failures = 0 if current > offset else failures + 1
            if failures >= retries:
                raise RuntimeError(f"Download interrupted: {exc}. Re-run the same command to resume {partial}") from exc
            print(f"Continuing {path.name} from {current:,} bytes: {exc}", flush=True)
            time.sleep(min(failures + 1, 5))
    verify(partial, size, checksum)
    # Preserve a complete checkpoint if another download finished concurrently.
    if path.exists():
        verify(path, size, checksum)
        partial.unlink()
    else:
        partial.rename(path)
    print(f"Downloaded and verified: {path.name} ({size:,} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "checkpoints/ChaosNexus/checkpoint-final")
    parser.add_argument("--retries", type=int, default=8, help="Maximum consecutive failures without progress")
    parser.add_argument("--timeout", type=float, default=60, help="Timeout per network operation, in seconds")
    args = parser.parse_args()
    if args.retries < 1 or args.timeout <= 0:
        parser.error("retries and timeout must be positive")
    try:
        for name, item_id, size, checksum in FILES:
            urls = [f"https://{host}/scl/fo/{SHARE}/{item_id}/ChaosNexus/checkpoint-final/{name}?rlkey={SHARE_KEY}&dl=1"
                    for host in ["dl.dropboxusercontent.com", "www.dropbox.com"]]
            download(args.output_dir / name, urls, size, checksum, args.retries, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(f"Checkpoint ready: {args.output_dir.resolve()}\nNext: python high_dim_eval/quickstart.py --profile smoke", flush=True)


if __name__ == "__main__":
    main()
