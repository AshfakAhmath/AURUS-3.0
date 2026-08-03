"""Picamera2/OpenCV vision with YuNet detection and SFace recognition."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import numpy as np

from src.core.models import IdentityResult, VisionSnapshot

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on dependency-free hosts
    cv2 = None

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - normal away from Raspberry Pi
    Picamera2 = None


class VisionService:
    def __init__(
        self,
        identity_service,
        model_dir: str | Path,
        camera_index: int = 0,
        process_size: tuple[int, int] = (320, 240),
        rate_hz: float = 10.0,
    ):
        self.identity_service = identity_service
        self.model_dir = Path(model_dir)
        self.camera_index = camera_index
        self.width, self.height = process_size
        self.period = 1.0 / max(1.0, rate_hz)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera = None
        self._camera_kind = "none"
        self._detector = None
        self._recognizer = None
        self._haar = None
        self._snapshot = VisionSnapshot()
        self._jpeg: bytes | None = None
        self._backend = "unavailable"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._close_camera()

    def get_snapshot(self) -> VisionSnapshot:
        with self._lock:
            return self._snapshot

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _configure_models(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed")
        yunet = self.model_dir / "face_detection_yunet_2023mar.onnx"
        sface = self.model_dir / "face_recognition_sface_2021dec.onnx"
        if yunet.exists() and sface.exists() and hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"):
            self._detector = cv2.FaceDetectorYN.create(str(yunet), "", (self.width, self.height), 0.85, 0.3, 5000)
            self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
            self._backend = "yunet+sface"
            return

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(str(cascade_path))
        if self._haar.empty():
            raise RuntimeError("YuNet/SFace models and Haar fallback are unavailable")
        self._backend = "haar-session-fallback"

    def _open_camera(self) -> None:
        if Picamera2 is not None and os.name != "nt":
            try:
                camera = Picamera2()
                try:
                    config = camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
                    camera.configure(config)
                except Exception:
                    # Fallback to default video configuration if RGB888 format is rejected by driver
                    config = camera.create_video_configuration(main={"size": (640, 480)})
                    camera.configure(config)
                camera.start()
                time.sleep(0.5)
                self._camera = camera
                self._camera_kind = "picamera2"
                return
            except Exception as exc:
                print(f"[VisionService] Picamera2 failed: {exc}. Trying OpenCV fallback...")

        if cv2 is None:
            raise RuntimeError("No camera backend available")
        camera = cv2.VideoCapture(self.camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"could not open camera {self.camera_index}")
        self._camera = camera
        self._camera_kind = "opencv"

    def _close_camera(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        try:
            if self._camera_kind == "picamera2":
                camera.stop()
                camera.close()
            else:
                camera.release()
        except Exception:
            pass

    def _read_frame(self):
        if self._camera_kind == "picamera2":
            rgb = self._camera.capture_array("main")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, frame = self._camera.read()
        return frame if ok else None

    def _detect(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        detections = []
        if self._detector is not None:
            self._detector.setInputSize((self.width, self.height))
            _, faces = self._detector.detect(resized)
            if faces is not None:
                for face in faces:
                    x, y, w, h = [int(value) for value in face[:4]]
                    detections.append((x, y, w, h, float(face[-1]), face))
        else:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for x, y, w, h in faces:
                detections.append((int(x), int(y), int(w), int(h), 1.0, None))
        return resized, detections

    def _embedding(self, frame, raw_face):
        if self._recognizer is None or raw_face is None:
            return None
        aligned = self._recognizer.alignCrop(frame, raw_face)
        return np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)

    def process_frame(self, frame) -> VisionSnapshot:
        resized, detections = self._detect(frame)
        detections.sort(key=lambda item: item[2] * item[3], reverse=True)
        identity = IdentityResult()
        center_x = None
        primary_width = None
        embedding = None

        if detections:
            x, y, w, h, _, raw = detections[0]
            center_x = (x + w / 2) / self.width
            primary_width = w / self.width
            embedding = self._embedding(resized, raw)
            if self._recognizer is not None:
                identity = self.identity_service.process_embedding(embedding)
            else:
                identity = self.identity_service.current()
        else:
            self.identity_service.clear_current()

        faces_payload = []
        for index, (x, y, w, h, score, _) in enumerate(detections):
            faces_payload.append(
                {
                    "box": [x / self.width, y / self.height, w / self.width, h / self.height],
                    "score": round(score, 3),
                    "primary": index == 0,
                }
            )
            colour = (57, 217, 138) if index == 0 else (73, 159, 255)
            cv2.rectangle(resized, (x, y), (x + w, y + h), colour, 2)

        label = "No face"
        if detections:
            label = identity.name if identity.status == "known" and identity.name else identity.status.title()
            if identity.confidence:
                label += f" {identity.confidence:.2f}"
            x, y, _, _, _, _ = detections[0]
            cv2.putText(resized, label, (max(0, x), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        snapshot = VisionSnapshot(
            timestamp=time.monotonic(),
            healthy=True,
            faces=tuple(faces_payload),
            primary_center_x=center_x,
            primary_width=primary_width,
            identity=identity,
            backend=self._backend,
            error="",
        )
        with self._lock:
            self._snapshot = snapshot
            self._jpeg = encoded.tobytes() if ok else None
        return snapshot

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = VisionSnapshot(
                timestamp=time.monotonic(), healthy=False, backend=self._backend, error=message[:200]
            )

    def _run(self) -> None:
        try:
            self._configure_models()
            self._open_camera()
        except Exception as exc:
            self._set_error(str(exc))
            self._close_camera()
            return

        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = self._read_frame()
                if frame is None:
                    self._set_error("camera returned no frame")
                    self._stop.wait(0.2)
                    continue
                try:
                    self.process_frame(frame)
                except Exception as exc:
                    self._set_error(f"vision processing failed: {exc}")
                self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))
        finally:
            self._close_camera()
