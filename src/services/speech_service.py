"""Offline microphone recognition with Vosk and gated wake-phrase support."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

try:
    import pyaudio
except ImportError:  # pragma: no cover - dependency is Pi-specific
    pyaudio = None

try:
    from vosk import KaldiRecognizer, Model
except ImportError:  # pragma: no cover - dependency is optional
    KaldiRecognizer = None
    Model = None


class SpeechService:
    RATE = 16000
    CHUNK = 4000

    def __init__(
        self,
        model_path: str | Path,
        microphone_index: int | None = None,
        record_seconds: float = 4.0,
        wake_enabled: bool = False,
        wake_test_passes: int = 0,
        wake_test_total: int = 0,
    ):
        self.model_path = Path(model_path)
        self.microphone_index = microphone_index
        self.record_seconds = record_seconds
        self.wake_requested = wake_enabled
        self.wake_gate_passed = wake_test_total >= 20 and wake_test_passes >= 18
        self._model = None
        self._lock = threading.RLock()
        self._busy = False
        self._stop = threading.Event()
        self._wake_thread: threading.Thread | None = None
        self.error = "not initialized"

    @property
    def healthy(self) -> bool:
        return self._model is not None and pyaudio is not None

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def wake_active(self) -> bool:
        return bool(self.wake_requested and self.wake_gate_passed and self._wake_thread and self._wake_thread.is_alive())

    def start(self, wake_callback: Callable[[str], None] | None = None) -> None:
        if pyaudio is None or Model is None or KaldiRecognizer is None:
            self.error = "PyAudio or Vosk is not installed"
            return
        if not self.model_path.exists():
            self.error = f"Vosk model missing: {self.model_path}"
            return
        try:
            self._model = Model(str(self.model_path))
            self.error = ""
        except Exception as exc:
            self.error = f"Vosk model failed: {exc}"
            return

        if self.wake_requested and self.wake_gate_passed and wake_callback:
            self._stop.clear()
            self._wake_thread = threading.Thread(
                target=self._wake_loop, args=(wake_callback,), name="wake-listener", daemon=True
            )
            self._wake_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._wake_thread and self._wake_thread.is_alive():
            self._wake_thread.join(timeout=2.0)

    def listen_once(self, callback: Callable[[str, str | None], None]) -> bool:
        with self._lock:
            if self._busy or not self.healthy:
                return False
            self._busy = True

        def worker():
            try:
                callback(self._record_and_recognize(), None)
            except Exception as exc:
                callback("", str(exc))
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=worker, name="push-to-talk", daemon=True).start()
        return True

    def _open_stream(self, audio):
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.microphone_index,
        )

    def _record_and_recognize(self) -> str:
        if not self.healthy:
            raise RuntimeError(self.error or "speech recognition unavailable")
        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = self._open_stream(audio)
            recognizer = KaldiRecognizer(self._model, self.RATE)
            end = time.monotonic() + self.record_seconds
            while time.monotonic() < end and not self._stop.is_set():
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                recognizer.AcceptWaveform(data)
            result = json.loads(recognizer.FinalResult())
            return str(result.get("text", "")).strip()
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    def _wake_loop(self, callback: Callable[[str], None]) -> None:
        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = self._open_stream(audio)
            recognizer = KaldiRecognizer(self._model, self.RATE, json.dumps(["aurus", "[unk]"]))
            while not self._stop.is_set():
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "")
                    if "aurus" in text.lower():
                        stream.stop_stream()
                        stream.close()
                        stream = None
                        callback(self._record_and_recognize())
                        stream = self._open_stream(audio)
                        recognizer = KaldiRecognizer(self._model, self.RATE, json.dumps(["aurus", "[unk]"]))
        except Exception as exc:
            self.error = f"wake listener failed: {exc}"
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()
