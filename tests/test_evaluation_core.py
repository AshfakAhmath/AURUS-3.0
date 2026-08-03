import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.core.models import MotionCommand, RobotMode, SensorSnapshot
from src.memory.repository import MemoryRepository
from src.services.conversation_service import ConversationService
from src.services.identity_service import IdentityService
from src.services.motion_arbiter import MotionArbiter
from src.services.sensor_sampler import SensorSampler


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


if __name__ == "__main__":
    unittest.main()
