"""Download the free, Pi-sized default GGUF conversational model."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
)
MINIMUM_SIZE = 1_000_000_000


def valid_gguf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MINIMUM_SIZE:
        return False
    with path.open("rb") as model:
        return model.read(4) == b"GGUF"


def download() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if valid_gguf(MODEL_PATH):
        print(f"Present: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1_000_000:.0f} MB)")
        return
    if MODEL_PATH.exists():
        raise RuntimeError(f"invalid existing model; move or remove it before retrying: {MODEL_PATH}")

    partial = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "AURUS-local-model-setup/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming at {existing / 1_000_000:.0f} MB …")
    else:
        print("Downloading Qwen2.5 1.5B Instruct Q4_K_M (about 1.12 GB) …")

    request = urllib.request.Request(MODEL_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = int(getattr(response, "status", 200)) == 206
        mode = "ab" if existing and resumed else "wb"
        downloaded = existing if mode == "ab" else 0
        expected = int(response.headers.get("Content-Length", "0")) + downloaded
        last_report = time.monotonic()
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 5.0:
                    total = f"/{expected / 1_000_000:.0f}" if expected else ""
                    print(f"  {downloaded / 1_000_000:.0f}{total} MB")
                    last_report = time.monotonic()

    if not valid_gguf(partial):
        raise RuntimeError(f"downloaded file is incomplete or not GGUF: {partial}")
    partial.replace(MODEL_PATH)
    print(f"Ready: {MODEL_PATH}")


def main() -> int:
    try:
        download()
        return 0
    except Exception as exc:
        print(f"LLM setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
