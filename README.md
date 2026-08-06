# AURUS Evaluation Edition

AURUS is an offline-first recognizing companion rover built for Raspberry Pi 4B. It combines persistent face identity, local memory, safe person-following, local voice/TTS, optional Groq dialogue, real sensor telemetry, and expressive mecanum movement.

The existing verified motor directions, GPIO pin assignments, ultrasonic wiring, and mecanum kinematics are intentionally preserved.

## Evaluation capabilities

- Enroll and recognize a person with OpenCV YuNet/SFace.
- Persist names, face embeddings, memories, interactions, and events in SQLite.
- Follow a face while enforcing five-sensor proximity safety.
- Stop on stale sensors, lost targets, expired commands, E-STOP, or dashboard disconnection.
- Recognize commands locally with Vosk and speak with Piper/eSpeak.
- Use Groq only for optional open-ended conversation; robot control remains deterministic.
- Show live camera, radar, identity, health, and requested-versus-final motion in a local dashboard.
- **NEW**: Run a live, autonomous **Groq MCP Agent** directly inside the dashboard to execute complex commands (like navigation, sensing, and speaking) using lightning-fast AI tool calling.

## Quick start on Raspberry Pi

Do not reimage or upgrade the OS immediately before evaluation. Confirm the existing environment first:

```bash
cat /etc/os-release
uname -m
python3 --version
rpicam-hello --list-cameras
arecord -l
aplay -l
```

Install runtime packages:

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

Set `GROQ_API_KEY` and the downloaded `PIPER_MODEL_PATH` in `.env`, then:

```bash
python run.py
```

Open `http://<raspberry-pi-ip>:5000`.

After the manual launch has passed the full showcase repeatedly, copy `deploy/aurus.service` to `/etc/systemd/system/`, adjust its user/path if needed, then enable it with `sudo systemctl enable --now aurus`.

## Tests

```bash
python -m unittest tests.test_suite tests.test_evaluation_core -v
```

`tests/test_ultrasonic_voltage.py` is an interactive Raspberry Pi hardware utility and is intentionally not included in desktop test discovery.

## Documentation

See [docs/EVALUATION_ARCHITECTURE.md](docs/EVALUATION_ARCHITECTURE.md) for architecture, interfaces, setup, degradation behavior, and acceptance criteria.
