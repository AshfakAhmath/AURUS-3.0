"""Explicit composition root and application service for AURUS."""

from __future__ import annotations

import os
from pathlib import Path
import re
import threading
import time
from typing import Callable

from src.core.models import RobotMode
from src.hardware.motors import MecanumDriver
from src.hardware.sensors import ProximitySensor
from src.memory.repository import MemoryRepository
from src.services.behavior_controller import BehaviorController
from src.services.conversation_service import ConversationReply, ConversationService
from src.services.identity_service import IdentityService
from src.services.motion_arbiter import MotionArbiter
from src.services.sensor_sampler import SensorSampler
from src.services.speech_service import SpeechService
from src.services.tts_service import TTSService
from src import config
from src.services.vision_service import VisionService
from src.services.mcp_agent_service import MCPAgentService


class RobotRuntime:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.driver = MecanumDriver()
        self.sensor = ProximitySensor(self.driver)
        self.repository = MemoryRepository(self.project_root / "aurus_memory.db")
        self.identity = IdentityService(self.repository)
        self.sensor_sampler = SensorSampler(self.sensor)
        self.arbiter = MotionArbiter(self.driver, self.sensor_sampler)
        self.vision = VisionService(self.identity, self.project_root / "models")

        microphone_index = os.getenv("MIC_INDEX")
        try:
            microphone_index_value = (
                int(microphone_index)
                if microphone_index and microphone_index.strip().lower() not in ("none", "null")
                else None
            )
        except ValueError:
            microphone_index_value = None

        def model_path(variable: str, default: Path) -> Path:
            raw = os.getenv(variable, "").strip()
            configured = Path(raw) if raw else default
            return configured if configured.is_absolute() else self.project_root / configured

        def float_setting(variable: str, default: float) -> float:
            try:
                return float(os.getenv(variable, str(default)))
            except ValueError:
                return default

        audio_playback = threading.Event()
        keyword_model = model_path(
            "KWS_MODEL_PATH",
            Path("models") / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
        )
        self.speech = SpeechService(
            model_path=model_path(
                "VOSK_MODEL_PATH", Path("models") / "vosk-model-small-en-us-0.15"
            ),
            keyword_model_path=keyword_model,
            keywords_path=model_path("KWS_KEYWORDS_PATH", keyword_model / "aurus_keywords.txt"),
            vad_model_path=model_path("VAD_MODEL_PATH", Path("models") / "silero_vad.onnx"),
            microphone_index=microphone_index_value,
            record_seconds=config.AURUS_RECORD_SECONDS,
            speech_start_timeout=float_setting("AURUS_SPEECH_START_TIMEOUT", 4.0),
            wake_enabled=os.getenv("AURUS_WAKE_ENABLED", "false").lower() == "true",
            playback_active=audio_playback,
        )
        self.tts = TTSService(
            model_path("PIPER_MODEL_PATH", Path("models") / "en_US-lessac-medium.onnx"),
            playback_active=audio_playback,
        )
        self.conversation = ConversationService(
            model_path=model_path(
                "LOCAL_LLM_MODEL_PATH",
                Path("models") / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            ),
            server_url=os.getenv("LOCAL_LLM_SERVER_URL", ""),
            model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct"),
            timeout=float_setting("LOCAL_LLM_TIMEOUT", 20.0),
            context_size=int(float_setting("LOCAL_LLM_CONTEXT", 2048)),
            max_tokens=int(float_setting("LOCAL_LLM_MAX_TOKENS", 96)),
            threads=int(float_setting("LOCAL_LLM_THREADS", min(4, os.cpu_count() or 2))),
        )
        self.behavior = BehaviorController(
            self.arbiter, self.sensor_sampler, self.vision, self._handle_behavior_event
        )
        self.mcp_agent = MCPAgentService(self)
        self._event_sink: Callable[[str, dict], None] = lambda event, payload: None
        self._started = False
        self._lock = threading.RLock()
        self._last_manual_sequence = -1

    def set_event_sink(self, callback: Callable[[str, dict], None]) -> None:
        self._event_sink = callback

    def emit(self, event: str, payload: dict) -> None:
        try:
            self._event_sink(event, payload)
        except Exception:
            pass

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        try:
            self.sensor_sampler.sample_once()
            self.sensor_sampler.start()
            self.arbiter.start()
            self.vision.start()
            self.tts.start()
            self.speech.start(lambda text: self.handle_text(text, source="wake"))
            self.behavior.start()
            self.conversation.start()
            self.repository.log_event("runtime", "AURUS evaluation runtime started")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        errors = []
        shutdown_steps = (
            ("emergency stop", lambda: self.arbiter.emergency_stop("runtime shutdown")),
            ("behavior", self.behavior.stop),
            ("speech", self.speech.stop),
            ("local LLM", self.conversation.stop),
            ("vision", self.vision.stop),
            ("arbiter", self.arbiter.stop),
            ("sensor sampler", self.sensor_sampler.stop),
            ("tts", self.tts.stop),
            ("motor driver", self.driver.cleanup),
        )
        for component, action in shutdown_steps:
            try:
                action()
            except Exception as exc:
                errors.append(f"{component}: {str(exc)[:120]}")
        description = "AURUS evaluation runtime stopped"
        if errors:
            description += "; shutdown warnings: " + " | ".join(errors)
        try:
            self.repository.log_event("runtime", description)
        except Exception:
            pass

    def dashboard_connected(self, connected: bool) -> None:
        if connected:
            with self._lock:
                self._last_manual_sequence = -1
        self.arbiter.set_dashboard_connected(connected)

    def set_mode(self, value: str) -> bool:
        aliases = {
            "follow": RobotMode.FOLLOWING,
            "following": RobotMode.FOLLOWING,
            "explore": RobotMode.EXPLORING,
            "exploring": RobotMode.EXPLORING,
            "manual": RobotMode.MANUAL,
            "idle": RobotMode.IDLE,
        }
        mode = aliases.get(value.lower())
        if mode is None:
            return False
        return self.arbiter.set_mode(mode)

    def manual_drive(self, vx: float, vy: float, omega: float, sequence: int) -> bool:
        with self._lock:
            if sequence <= self._last_manual_sequence:
                return False
            self._last_manual_sequence = sequence
        if self.arbiter.mode != RobotMode.MANUAL:
            return False
        return self.arbiter.command("dashboard-manual", vx, vy, omega, ttl=0.25, priority=50)

    def emergency_stop(self) -> None:
        self.arbiter.emergency_stop()
        self.repository.log_event("estop", "Emergency stop activated")
        self.emit("system_health", self.health())

    def clear_estop(self) -> None:
        self.arbiter.clear_estop()
        self.repository.log_event("estop", "Emergency stop cleared")

    def start_enrollment(self, name: str):
        snapshot = self.vision.get_snapshot()
        if snapshot.backend == "yunet+sface":
            status = self.identity.begin_enrollment(name)
        elif snapshot.backend in ("haar-session-fallback", "video-only-fallback"):
            status = self.identity.begin_session_identity(name)
        else:
            raise ValueError("Vision is not ready for enrollment yet.")
        self.emit("enrollment_progress", status.as_dict())
        return status

    def remember_fact(self, text: str) -> tuple[bool, str]:
        identity = self.vision.get_snapshot().identity
        if identity.status != "known" or identity.user_id is None:
            return False, "Please enroll or recognize a person before storing a memory."
        self.repository.remember(identity.user_id, text)
        return True, f"I will remember that {text}."

    def start_listening(self) -> bool:
        if not self.speech.healthy:
            self.emit("conversation", {"source": "voice", "transcript": "", "response": "Microphone recognition is unavailable. Please use text input.", "provider": "local", "fallback_reason": self.speech.error})
            return False
        if not self.arbiter.set_mode(RobotMode.LISTENING):
            self.emit("conversation", {
                "source": "voice",
                "transcript": "",
                "response": "Listening is unavailable while the emergency stop is latched.",
                "provider": "local",
                "fallback_reason": "emergency stop latched",
            })
            return False
        started = self.speech.listen_once(self._speech_result)
        if not started:
            self.arbiter.set_mode(RobotMode.IDLE)
        return started

    def _speech_result(self, text: str, error: str | None) -> None:
        if error or not text:
            response = f"I could not understand that. {error or 'Please try again.'}"
            self._announce(response, "voice", text, "local", error or "empty transcript")
            self.arbiter.set_mode(RobotMode.IDLE)
            return
        self.handle_text(text, source="voice")
        if self.arbiter.mode == RobotMode.LISTENING:
            self.arbiter.set_mode(RobotMode.IDLE)

    def _current_person(self):
        identity = self.vision.get_snapshot().identity
        if identity.status == "known" and identity.user_id is not None:
            return identity
        current = self.identity.current()
        return current if current.status == "known" else identity

    def _announce(
        self,
        response: str,
        source: str,
        transcript: str,
        provider: str,
        fallback_reason: str = "",
    ) -> dict:
        self.tts.speak(response)
        payload = {
            "source": source,
            "transcript": transcript,
            "response": response,
            "provider": provider,
            "fallback_reason": fallback_reason,
        }
        self.emit("conversation", payload)
        person = self._current_person()
        try:
            self.repository.log_interaction(person.user_id, source, transcript, response, provider)
        except Exception:
            pass
        return payload

    def handle_text(self, text: str, source: str = "text") -> dict:
        clean = " ".join((text or "").strip().split())[:1000]
        if not clean:
            return self._announce("I did not receive a command.", source, clean, "local")
        lower = clean.lower()

        if re.search(r"\bclear(?:\s+the)?\s+(?:emergency\s+stop|e[\s-]?stop)\b", lower):
            self.clear_estop()
            return self._announce("Emergency stop cleared. I am idle.", source, clean, "local")
        if "emergency stop" in lower or re.search(r"\be[\s-]?stop\b", lower):
            self.emergency_stop()
            return self._announce("Emergency stop activated and latched.", source, clean, "local")
        if lower in ("stop", "halt", "freeze") or "stop following" in lower or "stop exploring" in lower:
            self.arbiter.halt("voice-stop")
            if not self.arbiter.estopped:
                self.arbiter.set_mode(RobotMode.IDLE)
            return self._announce("Stopped safely.", source, clean, "local")

        name_match = re.search(r"(?:my name is|call me)\s+([a-z][a-z .'-]{0,60})", clean, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip(" .")
            status = self.start_enrollment(name)
            response = (
                f"Hello {name}. Hold still while I learn your face."
                if status.active
                else f"Hello {name}. I have linked your name to this session."
            )
            return self._announce(response, source, clean, "local")

        if any(phrase in lower for phrase in ("what do you remember", "what do you know about me", "remember about me", "do you remember me")):
            person = self._current_person()
            if person.user_id is None:
                return self._announce("I do not know who you are yet. Please tell me your name.", source, clean, "local")
            facts = self.repository.memories_for(person.user_id)
            if facts:
                response = f"{person.name}, I remember: " + "; ".join(facts[:4])
            else:
                response = f"I recognize you as {person.name}, but you have not taught me a fact yet."
            return self._announce(response, source, clean, "local")

        remember_match = re.search(r"(?:remember that|remember)\s+(.+)", clean, re.IGNORECASE)
        if remember_match:
            ok, response = self.remember_fact(remember_match.group(1).strip())
            return self._announce(response, source, clean, "local", "" if ok else "identity required")

        if "follow me" in lower or lower == "follow":
            if self.arbiter.set_mode(RobotMode.FOLLOWING):
                return self._announce("Follow mode engaged. I will keep you centered and respect my safety sensors.", source, clean, "local")
        if "explore" in lower:
            if self.arbiter.set_mode(RobotMode.EXPLORING):
                return self._announce("Exploration mode engaged. My ultrasonic safety layer remains in control.", source, clean, "local")
        if "manual mode" in lower or "take control" in lower:
            self.arbiter.set_mode(RobotMode.MANUAL)
            return self._announce("Manual mode ready.", source, clean, "local")
        if any(phrase in lower for phrase in ("show me what you can do", "showcase", "dance")):
            started = self.behavior.start_showcase()
            return self._announce("Beginning the mecanum showcase." if started else "A showcase is already running.", source, clean, "local")
        if "status" in lower or "how are your systems" in lower:
            sensors = self.sensor_sampler.get_snapshot()
            response = f"Systems checked. Front clearance is {sensors.front_min:.0f} centimetres. Current mode is {self.arbiter.mode.value}."
            return self._announce(response, source, clean, "local")
        if any(word in lower for word in ("hello", "hi aurus", "hey aurus")):
            person = self._current_person()
            name = f", {person.name}" if person.name else ""
            return self._announce(f"Hello{name}! My local systems are ready.", source, clean, "local")

        person = self._current_person()
        memories = self.repository.memories_for(person.user_id) if person.user_id else []
        reply: ConversationReply = self.conversation.reply(
            clean,
            user_name=person.name,
            memories=memories,
            recent_interactions=self.repository.recent_interactions(),
        )
        return self._announce(reply.text, source, clean, reply.provider, reply.fallback_reason)

    def _handle_behavior_event(self, event: str, payload: dict) -> None:
        if event == "greeting":
            self._announce(f"Welcome back, {payload['name']}! I recognize you.", "vision", "", "local")
        elif event == "behavior_notice":
            self._announce(payload.get("text", ""), "behavior", "", "local")
        else:
            self.emit(event, payload)

    def health(self) -> dict:
        sensors = self.sensor_sampler.get_snapshot()
        vision = self.vision.get_snapshot()
        return {
            "runtime": self._started,
            "motors": True,
            "sensors": sensors.healthy and not sensors.is_stale(),
            "camera": vision.healthy,
            "vision_backend": vision.backend,
            "microphone": self.speech.healthy,
            "microphone_error": self.speech.error,
            "stt_backend": self.speech.stt_backend,
            "vad_backend": self.speech.vad_backend,
            "vad_error": self.speech.vad_error,
            "wake_phrase": self.speech.wake_active,
            "wake_backend": self.speech.wake_backend,
            "wake_error": self.speech.wake_error,
            "wake_gate_passed": self.speech.wake_gate_passed,
            "tts": self.tts.healthy,
            "tts_backend": self.tts.backend,
            "tts_error": self.tts.error,
            "audio_playback": self.tts.playback_active.is_set(),
            "database": True,
            "llm": self.conversation.ready,
            "llm_loading": self.conversation.loading,
            "llm_backend": self.conversation.backend,
            "llm_error": self.conversation.last_error,
            "cloud": False,
            "cloud_error": "local LLM selected for dialogue",
            "mcp_agent": self.mcp_agent.ready,
            "estop": self.arbiter.estopped,
        }

    def telemetry(self) -> dict:
        return {
            "timestamp": time.time(),
            "mode": self.arbiter.mode.value,
            "sensors": self.sensor_sampler.get_snapshot().as_dict(),
            "motion": self.arbiter.get_decision().as_dict(),
            "vision": self.vision.get_snapshot().as_dict(),
            "enrollment": self.identity.get_enrollment().as_dict(),
            "health": self.health(),
        }
