"""Download free runtime models from their official distribution locations."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FILES = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Present: {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} …")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"unsafe archive entry: {member.filename}")
    archive.extractall(destination)


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        download(url, MODELS / filename)
    vosk_dir = MODELS / "vosk-model-small-en-us-0.15"
    if not vosk_dir.exists():
        archive_path = MODELS / "vosk-model-small-en-us-0.15.zip"
        download(VOSK_URL, archive_path)
        print("Extracting Vosk model …")
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, MODELS)
        archive_path.unlink(missing_ok=True)
    print("Vision and speech models are ready.")
    print("For Piper: python -m piper.download_voices en_US-lessac-medium --data-dir models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
