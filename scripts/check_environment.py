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
            for name in ("flask", "flask_socketio", "numpy", "cv2", "picamera2", "pyaudio", "vosk", "sherpa_onnx", "piper", "llama_cpp", "groq")
        },
        "models": {
            "yunet": (ROOT / "models" / "face_detection_yunet_2023mar.onnx").exists(),
            "sface": (ROOT / "models" / "face_recognition_sface_2021dec.onnx").exists(),
            "vosk": (ROOT / "models" / "vosk-model-small-en-us-0.15").is_dir(),
            "wake_word": (ROOT / "models" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01" / "aurus_keywords.txt").is_file(),
            "silero_vad": (ROOT / "models" / "silero_vad.onnx").is_file(),
            "piper": all(
                (ROOT / "models" / name).is_file()
                for name in ("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json")
            ),
            "local_llm": (ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf").is_file(),
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
    result["voice_ready"] = all(
        result["modules"].get(name, False) for name in ("pyaudio", "vosk", "sherpa_onnx", "piper")
    ) and all(result["models"].get(name, False) for name in ("vosk", "wake_word", "silero_vad", "piper"))
    result["assistant_ready"] = bool(
        result["modules"].get("llama_cpp", False) and result["models"].get("local_llm", False)
    )
    print(json.dumps(result, indent=2))
    required = result["modules"]
    core_ok = all(required[name] for name in ("flask", "flask_socketio", "numpy", "cv2"))
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
