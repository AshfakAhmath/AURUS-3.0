"""Deterministic high-level rover behavior state machine."""

from __future__ import annotations

import threading
import time
from typing import Callable

from src.core.models import RobotMode


class BehaviorController:
    def __init__(self, arbiter, sensor_sampler, vision_service, event_callback: Callable[[str, dict], None]):
        self.arbiter = arbiter
        self.sensor_sampler = sensor_sampler
        self.vision_service = vision_service
        self.event_callback = event_callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._showcase_thread: threading.Thread | None = None
        self._lost_since: float | None = None
        self._last_greeted: dict[int, float] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="behavior-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.arbiter.halt("behavior-shutdown")

    def start_showcase(self) -> bool:
        if self._showcase_thread and self._showcase_thread.is_alive():
            return False
        if not self.arbiter.set_mode(RobotMode.PERFORMING):
            return False
        self._showcase_thread = threading.Thread(target=self._showcase, name="showcase-routine", daemon=True)
        self._showcase_thread.start()
        return True

    def _emit_greeting_if_needed(self, vision) -> None:
        identity = vision.identity
        if identity.status != "known" or identity.user_id is None or not identity.name:
            return
        now = time.monotonic()
        if now - self._last_greeted.get(identity.user_id, 0.0) < 60.0:
            return
        self._last_greeted[identity.user_id] = now
        self.event_callback(
            "greeting",
            {"user_id": identity.user_id, "name": identity.name, "confidence": identity.confidence},
        )

    def _follow(self) -> None:
        vision = self.vision_service.get_snapshot()
        sensors = self.sensor_sampler.get_snapshot()
        now = time.monotonic()
        if not vision.healthy or vision.age > 0.5 or vision.primary_center_x is None:
            if self._lost_since is None:
                self._lost_since = now
            elapsed = now - self._lost_since
            if elapsed <= 0.5:
                self.arbiter.halt("follow-target-lost")
            elif elapsed <= 5.5:
                self.arbiter.command("follow-search", 0.0, 0.0, 0.12, ttl=0.2, priority=30)
            else:
                self.arbiter.halt("follow-search-timeout")
                self.arbiter.set_mode(RobotMode.IDLE)
                self.event_callback("behavior_notice", {"text": "I lost sight of you, so I stopped safely."})
            return

        self._lost_since = None
        error = vision.primary_center_x - 0.5
        if abs(error) > 0.12:
            omega = max(-0.4, min(0.4, error * 0.9))
            self.arbiter.command("follow-align", 0.0, 0.0, omega, ttl=0.2, priority=30)
            return
        if sensors.f > 70.0:
            self.arbiter.command("follow-distance", 0.28, 0.0, error * 0.35, ttl=0.2, priority=30)
        elif sensors.f < 45.0:
            self.arbiter.command("follow-distance", -0.20, 0.0, 0.0, ttl=0.2, priority=30)
        else:
            self.arbiter.halt("follow-distance-reached")

    def _explore(self) -> None:
        sensors = self.sensor_sampler.get_snapshot()
        if sensors.front_min < 35.0:
            omega = -0.28 if sensors.fl < sensors.fr else 0.28
            self.arbiter.command("explore-avoid", 0.0, 0.0, omega, ttl=0.2, priority=20)
        else:
            self.arbiter.command("explore-forward", 0.24, 0.0, 0.0, ttl=0.2, priority=20)

    def _showcase(self) -> None:
        self.event_callback("behavior_notice", {"text": "Showcase routine engaged. Observe the mecanum choreography!"})
        sequence = [
            (0.0, 0.85, 0.0, 0.55),
            (0.0, -0.85, 0.0, 0.55),
            (0.0, 0.0, 0.80, 0.75),
            (0.0, 0.0, -0.80, 0.75),
            (0.70, 0.70, 0.0, 0.45),
            (-0.70, -0.70, 0.0, 0.45),
        ]
        try:
            for vx, vy, omega, duration in sequence:
                if self._stop.is_set() or self.arbiter.estopped or self.arbiter.mode != RobotMode.PERFORMING:
                    break
                end = time.monotonic() + duration
                while time.monotonic() < end and not self._stop.is_set():
                    self.arbiter.command("showcase", vx, vy, omega, ttl=0.15, priority=40)
                    time.sleep(0.08)
                self.arbiter.halt("showcase-step")
                time.sleep(0.12)
        finally:
            self.arbiter.halt("showcase-complete")
            if not self.arbiter.estopped:
                self.arbiter.set_mode(RobotMode.IDLE)
                self.event_callback("behavior_notice", {"text": "Showcase complete. All motion stopped."})

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                vision = self.vision_service.get_snapshot()
                self._emit_greeting_if_needed(vision)
                mode = self.arbiter.mode
                if mode == RobotMode.FOLLOWING:
                    self._follow()
                elif mode == RobotMode.EXPLORING:
                    self._explore()
            except Exception as exc:
                self.arbiter.halt("behavior-error")
                self.event_callback("system_error", {"component": "behavior", "error": str(exc)[:200]})
