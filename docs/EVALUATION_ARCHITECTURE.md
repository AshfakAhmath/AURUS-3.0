# AURUS Evaluation Edition Architecture

**Status:** implementation baseline  
**Target hardware:** Raspberry Pi 4B (8 GB), Camera Module V2, USB microphone, speakers, five HC-SR04 sensors, four mecanum wheels, two L298N drivers  
**Primary constraint:** the verified motor/sensor wiring, pins, and kinematics are preserved

## 1. Product definition

AURUS is a persistent recognizing companion rover. It detects and enrolls a person, remembers their name and facts, recognizes them later, follows them with camera and ultrasonic safety, speaks locally, and exposes its real decisions through a live dashboard.

The system is offline-first. Internet access improves open-ended dialogue but is never required for motion, safety, identity, memory, speech recognition, TTS, or the evaluation showcase.

### Showcase sequence

1. A person enters the camera view.
2. AURUS enrolls or recognizes them and greets them by name.
3. The person teaches AURUS a fact; SQLite stores it.
4. AURUS recalls the fact without cloud access.
5. Follow mode aligns to the face and maintains distance using the front sensor.
6. A physical obstacle causes a visible safety override.
7. Push-to-talk accepts a local command or an optional Groq question.
8. A deterministic mecanum showcase ends with all motors stopped.

## 2. Technology decisions

| Layer | Selected technology | Reliability rationale |
|---|---|---|
| OS/runtime | Existing Raspberry Pi OS + Python virtual environment | Avoids a risky OS reimage; Picamera2 remains installed through `apt`/system packages. |
| GPIO | Existing `rpi-lgpio`-compatible motor and sensor modules | Wiring and behavior are already verified and are not rewritten. |
| Camera | Picamera2 | Raspberry Pi's supported Python camera API for Camera Module V2. |
| Vision | OpenCV YuNet + SFace | Small detection model, persistent embeddings, explicit unknown/uncertain states. |
| Speech recognition | Vosk small English model | Fully local, streaming-capable, designed to run on Raspberry Pi. |
| TTS | Piper; eSpeak NG fallback | Local neural speech with a dependable process fallback. |
| Dialogue | Groq `openai/gpt-oss-20b`; local fallback | One bounded optional request; never used for motor decisions. |
| Persistence | SQLite | Local, transactional, zero external service dependency. |
| Web | Flask-SocketIO with standard threading | Simple process model; browser assets are served locally. |

References:

- [Raspberry Pi camera software and Picamera2](https://www.raspberrypi.com/documentation/computers/camera_software.html)
- [OpenCV YuNet/SFace tutorial](https://docs.opencv.org/master/d0/dd4/tutorial_dnn_face.html)
- [Vosk offline speech recognition](https://alphacephei.com/vosk/)
- [Piper local TTS](https://github.com/OHF-voice/piper1-gpl)
- [Flask-SocketIO async choices](https://flask-socketio.readthedocs.io/en/stable/intro.html)
- [Groq free-plan limits](https://console.groq.com/docs/rate-limits)

No free hosted service can promise permanent availability. AURUS therefore treats cloud dialogue as optional enrichment rather than a reliability dependency.

## 3. Runtime architecture

```mermaid
flowchart TD
    Dashboard[Minimal dashboard] --> Runtime[RobotRuntime]
    Microphone[USB microphone] --> Speech[Vosk SpeechService]
    Speech --> Runtime
    Runtime --> Conversation[Local intents and optional Groq]
    Runtime --> TTS[Piper or eSpeak]

    Camera[Picamera2] --> Vision[VisionService]
    Vision --> Identity[IdentityService]
    Identity --> DB[(SQLite)]
    Vision --> Behavior[BehaviorController]

    Ultrasonic[Five ultrasonic sensors] --> Sampler[SensorSampler]
    Sampler --> Behavior
    Sampler --> Arbiter[MotionArbiter]
    Behavior --> Arbiter
    Dashboard --> Arbiter
    Arbiter --> Motors[Existing MecanumDriver]

    Runtime --> Dashboard
```

### Ownership invariants

- Only `SensorSampler` calls `ProximitySensor.read_all()` after startup.
- Only `MotionArbiter` calls `MecanumDriver.drive()` during runtime.
- Only `VisionService` owns the camera.
- TTS requests are serialized through one queue.
- SQLite operations use short-lived connections protected by a repository lock.
- The cloud model returns dialogue text only. It cannot select modes or movement.
- Importing a module never starts hardware or background threads.

## 4. Control and safety

`MotionCommand` contains `source`, `vx`, `vy`, `omega`, `priority`, and `expires_at`. The arbiter evaluates the newest eligible command at 20 Hz.

Safety order:

1. Latched E-STOP rejects all movement.
2. Dashboard disconnection stops the current movement.
3. Expired commands produce a deadman stop.
4. Unhealthy or older-than-300-ms sensor data produces a stop.
5. Forward, reverse, and side movement are checked against the relevant sensors.
6. Distances below 20 cm are rejected; 20–35 cm limits translational speed.
7. Only the safety-adjusted vector reaches the existing motor driver.

Modes are `IDLE`, `MANUAL`, `LISTENING`, `FOLLOWING`, `EXPLORING`, `PERFORMING`, `ESTOP`, and `DEGRADED`. Mode changes stop the previous movement first. E-STOP requires an explicit clear operation.

Because the chassis has no wheel encoders, AURUS does not claim odometry, SLAM, or precise path navigation. Explore mode is reactive, and following uses current perception rather than an estimated global pose.

## 5. Recognition and memory

YuNet detects faces at 320×240 processing resolution. SFace aligns the primary face and produces an embedding. Enrollment accepts 20 samples, normalizes their mean, and stores it as a float32 BLOB. Recognition uses cosine similarity with a default `0.40` threshold and requires three consecutive matches.

Recognition states:

- `known`: threshold and temporal confirmation passed.
- `uncertain`: a plausible score that has not passed confirmation.
- `unknown`: no stored identity passes the threshold.

If YuNet/SFace is unavailable, Haar detection remains functional and enrollment becomes session-only. The dashboard reports this degradation rather than pretending persistent recognition succeeded.

The database contains `schema_version`, `Users`, `FaceEmbeddings`, `Memories`, `Interactions`, and `Events`. Existing legacy databases receive a one-time `.legacy-backup.db` copy before schema initialization. Raw face images are not stored by default.

## 6. Behavior and interaction

Follow mode:

- Face horizontal error outside ±0.12 commands a bounded rotation.
- When centered, front distance above 70 cm allows slow forward movement.
- Distance below 45 cm requests slow reverse movement, subject to rear safety.
- A 45–70 cm range is treated as reached.
- Loss of the target stops after 500 ms, then allows a slow five-second search before returning to idle.

Local commands include stop, E-STOP, clear E-STOP, follow, explore, manual mode, status, enrollment, remember, recall, greeting, and showcase. All are processed before cloud dialogue.

Groq configuration uses one request, no SDK retry, a four-second timeout, and a maximum short answer. HTTP 429, timeout, missing key, invalid response, or network failure immediately produces a local personality response.

Push-to-talk is always exposed. Always-on wake listening is activated only when `AURUS_WAKE_ENABLED=true`, `WAKE_TEST_TOTAL>=20`, and `WAKE_TEST_PASSES>=18`.

## 7. Public interface

Socket commands:

- `manual_drive {vx, vy, omega, sequence}`
- `stop {}`
- `clear_estop {}`
- `set_mode {mode}`
- `start_listening {}`
- `send_text {text}`
- `enroll_person {name}`
- `remember_fact {text}`
- `perform_showcase {}`

Server events:

- `telemetry`: mode, sensor snapshot, requested/final motion, vision, enrollment, and health.
- `conversation`: transcript, response, provider, and fallback reason.
- `enrollment_progress`: sample progress and completion/error.
- `system_health`: component availability without secrets.

HTTP endpoints are `/`, `/video_feed`, and `/health`.

## 8. Setup

First record the existing environment; do not perform a major OS upgrade before evaluation:

```bash
cat /etc/os-release
uname -m
python3 --version
rpicam-hello --list-cameras
arecord -l
aplay -l
```

Install system packages and create a system-package-aware virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-pyaudio portaudio19-dev espeak-ng
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
python scripts/setup_models.py
python -m piper.download_voices en_US-lessac-medium --data-dir models
cp .env.example .env
python scripts/check_environment.py
```

Set `GROQ_API_KEY` and `PIPER_MODEL_PATH` in `.env`, then run:

```bash
python run.py
```

Open `http://<raspberry-pi-ip>:5000`.

Only after manual launch and repeated showcase testing succeed, adjust the user/path in `deploy/aurus.service`, copy it to `/etc/systemd/system/`, and enable it with `sudo systemctl enable --now aurus`.

## 9. Graceful degradation

| Failure | Result |
|---|---|
| Internet/Groq unavailable | Local commands, personality, memory, vision, and motion continue. |
| Piper unavailable | eSpeak NG is selected; if that also fails, dashboard text remains. |
| Vosk/microphone unavailable | Text input and dashboard controls remain. |
| SFace model unavailable | Haar face detection and session identity remain; persistent recognition is marked unavailable. |
| Camera unavailable | Manual and ultrasonic modes remain; follow is unavailable. |
| Database error | Motion safety remains independent; identity/memory report degraded. |
| Dashboard disconnect | Active motion stops. |
| Sensor stale/error | Motion stops. |

## 10. Acceptance criteria

- Ten consecutive showcase runs without process restart.
- E-STOP produces zero motion within 250 ms and remains latched.
- Sensor staleness, target loss, dashboard disconnect, and expired commands stop motion.
- A known person is recognized after process restart; unknown people are not assigned a stored name.
- Follow mode respects the 20 cm hard-stop boundary and stops within 500 ms of losing the face.
- Names and facts survive restart and are recalled offline.
- Cloud timeout, quota rejection, malformed output, and no internet produce a local response without affecting control.
- Camera failure leaves manual/ultrasonic operation available.
- Audio failure leaves text interaction available.
- The browser receives no API key.
- Dashboard and core health become available within 30 seconds of launch.
