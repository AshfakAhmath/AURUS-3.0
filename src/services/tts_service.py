"""Serialized local TTS with Piper, eSpeak, and visual-only fallback."""

from __future__ import annotations

import os
from pathlib import Path
import importlib.util
import queue
import shutil
import subprocess
import tempfile
import threading
import wave


class TTSService:
    def __init__(
        self,
        piper_model: str | Path | None = None,
        playback_active: threading.Event | None = None,
    ):
        self.piper_model = Path(piper_model) if piper_model else None
        self.playback_active = playback_active or threading.Event()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._piper_voice = None
        self.backend = self._select_backend()
        self.error = "" if self.backend != "visual" else "No local TTS executable available"

    def _select_backend(self) -> str:
        if importlib.util.find_spec("piper") and self.piper_model and self.piper_model.exists():
            return "piper"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        return "visual"

    @property
    def healthy(self) -> bool:
        return self.backend != "visual"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tts-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.playback_active.clear()

    def speak(self, text: str) -> bool:
        clean = " ".join(text.strip().split())[:400]
        if not clean:
            return False
        try:
            self._queue.put_nowait(clean)
            return True
        except queue.Full:
            self.error = "TTS queue full"
            return False

    def _play_wav(self, path: str) -> None:
        if shutil.which("aplay"):
            subprocess.run(["aplay", "-q", path], check=True, timeout=30)
        elif os.name == "nt":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME)
        else:
            raise RuntimeError("no WAV player available")

    def _speak_piper(self, text: str) -> None:
        if self._piper_voice is None:
            from piper import PiperVoice

            self._piper_voice = PiperVoice.load(str(self.piper_model))
        handle = tempfile.NamedTemporaryFile(prefix="aurus-", suffix=".wav", delete=False)
        handle.close()
        try:
            with wave.open(handle.name, "wb") as wav_file:
                self._piper_voice.synthesize_wav(text, wav_file)
            self._play_wav(handle.name)
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def _speak(self, text: str) -> None:
        if self.backend == "piper":
            self._speak_piper(text)
        elif self.backend in ("espeak-ng", "espeak"):
            subprocess.run([self.backend, text], check=True, timeout=30)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            self.playback_active.set()
            try:
                self._speak(text)
                self.error = ""
            except Exception as exc:
                self.error = str(exc)[:200]
                fallback = "espeak-ng" if shutil.which("espeak-ng") else "espeak" if shutil.which("espeak") else None
                if self.backend == "piper" and fallback:
                    self.backend = fallback
                    self._piper_voice = None
                    try:
                        self._speak(text)
                    except Exception:
                        self.backend = "visual"
                else:
                    self.backend = "visual"
            finally:
                self.playback_active.clear()
