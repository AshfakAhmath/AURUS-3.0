"""Read-only Raspberry Pi environment gate for the evaluation build."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "os_release": command_output(["cat", "/etc/os-release"]) if platform.system() == "Linux" else platform.system(),
        "camera_cli": command_output(["rpicam-hello", "--list-cameras"]) if shutil.which("rpicam-hello") else "unavailable",
        "microphones": command_output(["arecord", "-l"]) if shutil.which("arecord") else "unavailable",
        "speakers": command_output(["aplay", "-l"]) if shutil.which("aplay") else "unavailable",
        "modules": {
            name: module(name)
            for name in ("flask", "flask_socketio", "numpy", "cv2", "picamera2", "pyaudio", "vosk", "piper", "groq")
        },
        "models": {
            "yunet": (ROOT / "models" / "face_detection_yunet_2023mar.onnx").exists(),
            "sface": (ROOT / "models" / "face_recognition_sface_2021dec.onnx").exists(),
            "vosk": (ROOT / "models" / "vosk-model-small-en-us-0.15").is_dir(),
        },
        "executables": {name: bool(shutil.which(name)) for name in ("espeak-ng", "aplay")},
    }
    if module("cv2"):
        import cv2

        result["opencv"] = {
            "version": cv2.__version__,
            "yunet_api": hasattr(cv2, "FaceDetectorYN"),
            "sface_api": hasattr(cv2, "FaceRecognizerSF"),
        }
    print(json.dumps(result, indent=2))
    required = result["modules"]
    core_ok = all(required[name] for name in ("flask", "flask_socketio", "numpy", "cv2"))
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
