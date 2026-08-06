import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from src.core.models import IdentityResult, MotionCommand, RobotMode, SensorSnapshot
from src.memory.repository import MemoryRepository
from src.runtime import RobotRuntime
from src.services.conversation_service import ConversationService, _speech_text
from src.services.identity_service import IdentityService
from src.services.mcp_agent_service import MCPAgentService
from src.services.motion_arbiter import MotionArbiter
from src.services.sensor_sampler import SensorSampler
from src.services.speech_service import AdaptiveEnergyVad
from src.services.tts_service import TTSService
from src.web.app import create_app


class FakeDriver:
    def __init__(self):
        self.commands = []

    def drive(self, vx, vy, omega):
        self.commands.append((vx, vy, omega))

    def stop(self):
        self.commands.append((0.0, 0.0, 0.0))


class FakeSensor:
    def __init__(self, values=None):
        self.values = values or {"fl": 100, "f": 100, "fr": 100, "rl": 100, "rr": 100}

    def read_all(self):
        return dict(self.values)


class FakeSampler:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_snapshot(self):
        return self.snapshot


def snapshot(**changes):
    values = dict(timestamp=time.monotonic(), fl=100, f=100, fr=100, rl=100, rr=100, healthy=True, error="")
    values.update(changes)
    return SensorSnapshot(**values)


class SensorSamplerTests(unittest.TestCase):
    def test_sample_once_publishes_all_five_values(self):
        sampler = SensorSampler(FakeSensor())
        result = sampler.sample_once()
        self.assertTrue(result.healthy)
        self.assertEqual(result.front_min, 100)
        self.assertEqual(result.rear_min, 100)

    def test_non_finite_sensor_value_marks_snapshot_unhealthy(self):
        sampler = SensorSampler(FakeSensor({"fl": 100, "f": float("nan"), "fr": 100, "rl": 100, "rr": 100}))
        result = sampler.sample_once()
        self.assertFalse(result.healthy)
        self.assertIn("non-finite", result.error)


class MotionArbiterTests(unittest.TestCase):
    def build(self, sensor_snapshot):
        driver = FakeDriver()
        arbiter = MotionArbiter(driver, FakeSampler(sensor_snapshot))
        arbiter.set_dashboard_connected(True)
        arbiter.set_mode(RobotMode.MANUAL)
        return driver, arbiter

    def test_front_hard_stop_rejects_forward_motion(self):
        _, arbiter = self.build(snapshot(f=15))
        arbiter.command("test", 0.5, 0, 0)
        decision = arbiter.tick()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.final_vx, 0)
        self.assertIn("front obstacle", decision.reason)

    def test_caution_zone_reduces_forward_speed(self):
        _, arbiter = self.build(snapshot(f=30, fl=50, fr=50))
        arbiter.command("test", 0.5, 0, 0)
        decision = arbiter.tick()
        self.assertTrue(decision.allowed)
        self.assertAlmostEqual(decision.final_vx, 0.175)

    def test_stale_sensor_data_stops_motion(self):
        _, arbiter = self.build(snapshot(timestamp=time.monotonic() - 1.0))
        arbiter.command("test", 0.5, 0, 0)
        self.assertFalse(arbiter.tick().allowed)

    def test_expired_command_is_deadman_stopped(self):
        _, arbiter = self.build(snapshot())
        arbiter.submit(MotionCommand("test", 0.5, 0, 0, expires_at=time.monotonic() - 0.01))
        decision = arbiter.tick()
        self.assertFalse(decision.allowed)
        self.assertIn("expired", decision.reason)

    def test_estop_latches_until_explicit_clear(self):
        _, arbiter = self.build(snapshot())
        arbiter.emergency_stop()
        self.assertFalse(arbiter.command("test", 0.5, 0, 0))
        self.assertEqual(arbiter.mode, RobotMode.ESTOP)
        arbiter.clear_estop()
        self.assertEqual(arbiter.mode, RobotMode.IDLE)

    def test_dashboard_disconnect_stops_motion(self):
        _, arbiter = self.build(snapshot())
        arbiter.command("test", 0.5, 0, 0)
        arbiter.set_dashboard_connected(False)
        self.assertFalse(arbiter.tick().allowed)

    def test_non_finite_sensor_data_stops_motion(self):
        _, arbiter = self.build(snapshot(f=float("nan")))
        arbiter.command("test", 0.5, 0, 0)
        decision = arbiter.tick()
        self.assertFalse(decision.allowed)
        self.assertIn("invalid", decision.reason)

    def test_dashboard_disconnect_rejects_new_motion(self):
        _, arbiter = self.build(snapshot())
        arbiter.set_dashboard_connected(False)
        self.assertFalse(arbiter.command("test", 0.5, 0, 0))


class MemoryAndIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = MemoryRepository(Path(self.temp.name) / "memory.db")

    def tearDown(self):
        self.temp.cleanup()

    def test_name_fact_and_embedding_survive_repository_reload(self):
        user_id = self.repo.ensure_user("Humaidh")
        self.repo.remember(user_id, "likes robotics")
        self.repo.save_embedding(user_id, np.array([1.0, 0.0], dtype=np.float32), 20)
        second = MemoryRepository(Path(self.temp.name) / "memory.db")
        self.assertEqual(second.memories_for(user_id), ["likes robotics"])
        embeddings = second.load_embeddings()
        self.assertEqual(embeddings[0][1], "Humaidh")

    def test_enrollment_requires_samples_and_confirmation_frames(self):
        identity = IdentityService(self.repo, threshold=0.4, required_samples=3, confirmation_frames=2)
        identity.begin_enrollment("Evaluator")
        for _ in range(3):
            result = identity.process_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result.status, "known")
        identity.clear_current()
        first = identity.process_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        second = identity.process_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(first.status, "uncertain")
        self.assertEqual(second.name, "Evaluator")


class ConversationFallbackTests(unittest.TestCase):
    def test_no_api_key_returns_local_reply(self):
        service = ConversationService(api_key="")
        reply = service.reply("Tell me something")
        self.assertEqual(reply.provider, "local")
        self.assertTrue(reply.fallback_reason)

    def test_local_python_model_loads_and_returns_speech_text(self):
        class FakeLlama:
            def __init__(self, **settings):
                self.settings = settings

            def create_chat_completion(self, **_request):
                return iter(
                    [
                        {"choices": [{"delta": {"content": "Hello "}}]},
                        {"choices": [{"delta": {"content": "from *AURUS*."}}]},
                    ]
                )

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"GGUF")
            service = ConversationService(model_path=model, llm_factory=FakeLlama)
            service.start()
            deadline = time.monotonic() + 1.0
            while not service.ready and time.monotonic() < deadline:
                time.sleep(0.01)
            try:
                reply = service.reply("Who are you?")
                self.assertEqual(reply.provider, "llama-cpp-python")
                self.assertEqual(reply.text, "Hello from AURUS.")
            finally:
                service.stop()

    def test_local_server_backend_uses_openai_compatible_endpoint(self):
        class FakeResponse:
            status = 200

            def __init__(self, body=b"{}"):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.body

        urls = []

        def fake_urlopen(request, timeout):
            del timeout
            urls.append(request.full_url)
            if request.full_url.endswith("/v1/models"):
                return FakeResponse()
            return FakeResponse(
                json.dumps({"choices": [{"message": {"content": "A local answer."}}]}).encode()
            )

        service = ConversationService(server_url="http://127.0.0.1:8080/v1", urlopen=fake_urlopen)
        service.start()
        deadline = time.monotonic() + 1.0
        while not service.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            reply = service.reply("Hello")
            self.assertEqual(reply.provider, "llama.cpp-server")
            self.assertEqual(reply.text, "A local answer.")
            self.assertEqual(urls[-1], "http://127.0.0.1:8080/v1/chat/completions")
        finally:
            service.stop()

    def test_model_output_is_bounded_and_reasoning_is_removed(self):
        raw = "<think>private chain of thought</think> " + " ".join(["word"] * 70)
        clean = _speech_text(raw)
        self.assertNotIn("private", clean)
        self.assertLessEqual(len(clean.split()), 45)


class AudioPipelineTests(unittest.TestCase):
    def test_adaptive_vad_ends_after_voiced_audio_and_silence(self):
        vad = AdaptiveEnergyVad(sample_rate=16000, frame_samples=512, max_seconds=3.0)
        silence = np.zeros(512, dtype=np.int16).tobytes()
        voice = np.full(512, 2200, dtype=np.int16).tobytes()
        for _ in range(5):
            self.assertIsNone(vad.accept(silence))
        for _ in range(8):
            self.assertIsNone(vad.accept(voice))
        utterance = None
        for _ in range(20):
            utterance = vad.accept(silence)
            if utterance is not None:
                break
        self.assertIsNotNone(utterance)
        self.assertGreater(len(utterance), len(voice) * 8)

    def test_adaptive_vad_does_not_emit_silence(self):
        vad = AdaptiveEnergyVad(sample_rate=16000, frame_samples=512, max_seconds=1.0)
        silence = np.zeros(512, dtype=np.int16).tobytes()
        self.assertTrue(all(vad.accept(silence) is None for _ in range(40)))
        self.assertIsNone(vad.flush())

    def test_tts_marks_playback_active_while_speaking(self):
        playback = threading.Event()
        observed = threading.Event()
        completed = threading.Event()
        service = TTSService(playback_active=playback)

        def fake_speak(_text):
            if playback.is_set():
                observed.set()
            completed.set()

        service._speak = fake_speak
        service.start()
        try:
            self.assertTrue(service.speak("pipeline check"))
            self.assertTrue(completed.wait(1.0))
            self.assertTrue(observed.is_set())
            deadline = time.monotonic() + 1.0
            while playback.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(playback.is_set())
        finally:
            service.stop()


class FakeRuntimeArbiter:
    def __init__(self, mode=RobotMode.IDLE, estopped=False):
        self.mode = mode
        self.estopped = estopped
        self.mode_changes = []
        self.halts = []
        self.commands = []

    def set_mode(self, mode):
        if self.estopped and mode != RobotMode.ESTOP:
            return False
        self.mode = mode
        self.mode_changes.append(mode)
        return True

    def command(self, source, vx, vy, omega, ttl=0.3, priority=10):
        self.commands.append((source, vx, vy, omega, ttl, priority))
        return False

    def halt(self, source="system"):
        self.halts.append(source)


class RuntimeHarness(RobotRuntime):
    def __init__(self):
        self.arbiter = FakeRuntimeArbiter()
        self.actions = []
        self.repository = self

    def _announce(self, response, source, transcript, provider, fallback_reason=""):
        return {
            "response": response,
            "source": source,
            "transcript": transcript,
            "provider": provider,
            "fallback_reason": fallback_reason,
        }

    def _current_person(self):
        return IdentityResult("known", 7, "Evaluator", 0.9)

    def memories_for(self, user_id, limit=8):
        return ["likes reliable robots"]

    def clear_estop(self):
        self.actions.append("clear")

    def emergency_stop(self):
        self.actions.append("estop")

    def remember_fact(self, text):
        self.actions.append(("remember", text))
        return True, f"I will remember that {text}."


class RuntimeCommandTests(unittest.TestCase):
    def test_clear_estop_is_not_misread_as_estop(self):
        runtime = RuntimeHarness()
        response = runtime.handle_text("clear the e-stop")
        self.assertEqual(runtime.actions, ["clear"])
        self.assertIn("cleared", response["response"].lower())

    def test_recall_phrase_is_not_stored_as_a_new_fact(self):
        runtime = RuntimeHarness()
        response = runtime.handle_text("do you remember me")
        self.assertEqual(runtime.actions, [])
        self.assertIn("likes reliable robots", response["response"])

    def test_voice_result_returns_to_idle_after_non_mode_command(self):
        runtime = RuntimeHarness()
        runtime.arbiter.mode = RobotMode.LISTENING
        runtime._speech_result("hello", None)
        self.assertEqual(runtime.arbiter.mode, RobotMode.IDLE)


class DriverMustNotBeCalled:
    is_simulation = True

    def __getattr__(self, name):
        raise AssertionError(f"MCP animation bypassed the arbiter via driver.{name}")


class MissingUserRepository:
    def get_user(self, user_id):
        return None

    def remember(self, user_id, fact):
        raise AssertionError("A fact was stored for an unknown user")


class MCPRuntimeHarness:
    def __init__(self):
        self.arbiter = FakeRuntimeArbiter()
        self.driver = DriverMustNotBeCalled()
        self.repository = MissingUserRepository()

    def emit(self, event, payload):
        pass


class MCPAgentSafetyTests(unittest.TestCase):
    def test_animation_reports_blocked_when_arbiter_rejects_it(self):
        service = MCPAgentService(MCPRuntimeHarness(), api_key="")
        result = json.loads(service._dispatch_tool("perform_animation", {"name": "spin"}))
        self.assertEqual(result["status"], "blocked")
        self.assertIn("rejected", result["reason"])

    def test_invalid_direction_does_not_default_to_forward(self):
        service = MCPAgentService(MCPRuntimeHarness(), api_key="")
        result = json.loads(service._dispatch_tool("move_rover", {"direction": "sideways-ish"}))
        self.assertEqual(result["status"], "error")

    def test_stop_tool_leaves_autonomous_mode_idle(self):
        runtime = MCPRuntimeHarness()
        runtime.arbiter.mode = RobotMode.FOLLOWING
        service = MCPAgentService(runtime, api_key="")
        result = json.loads(service._dispatch_tool("stop_rover", {}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(runtime.arbiter.mode, RobotMode.IDLE)

    def test_memory_tool_rejects_unknown_user_without_writing(self):
        service = MCPAgentService(MCPRuntimeHarness(), api_key="")
        result = json.loads(service._dispatch_tool("remember_fact", {"fact": "likes robots"}))
        self.assertEqual(result["status"], "error")
        self.assertIn("enroll", result["message"])


class WebRuntimeHarness:
    def set_event_sink(self, callback):
        self.event_sink = callback

    def dashboard_connected(self, connected):
        self.connected = connected

    def health(self):
        return {}

    def telemetry(self):
        return {}

    def manual_drive(self, vx, vy, omega, sequence):
        return False


class WebAdapterTests(unittest.TestCase):
    def test_malformed_manual_payload_is_rejected_without_handler_failure(self):
        app, socketio, _, _ = create_app(WebRuntimeHarness())
        client = socketio.test_client(app)
        client.get_received()
        client.emit("manual_drive", None)
        errors = [event for event in client.get_received() if event["name"] == "command_error"]
        self.assertEqual(len(errors), 1)
        client.disconnect()


if __name__ == "__main__":
    unittest.main()
