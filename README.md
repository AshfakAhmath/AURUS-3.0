# AURUS Evaluation Edition

AURUS is an offline-first recognizing companion rover built for Raspberry Pi 4B. It combines persistent face identity, local memory, safe person-following, a free local LLM voice assistant, real sensor telemetry, and expressive mecanum movement.

The existing verified motor directions, GPIO pin assignments, ultrasonic wiring, and mecanum kinematics are intentionally preserved.

## Evaluation capabilities

- Enroll and recognize a person with OpenCV YuNet/SFace.
- Persist names, face embeddings, memories, interactions, and events in SQLite.
- Follow a face while enforcing five-sensor proximity safety.
- Stop on stale sensors, lost targets, expired commands, E-STOP, or dashboard disconnection.
- Detect “AURUS” locally with sherpa-onnx, endpoint speech with Silero VAD, transcribe with Vosk, and speak with Piper/eSpeak.
- Answer open-ended questions locally with Qwen2.5 1.5B through llama.cpp; robot control remains deterministic.
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
sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-pyaudio portaudio19-dev espeak-ng build-essential cmake libopenblas-dev
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install -r requirements-pi.txt
python scripts/setup_models.py
python scripts/setup_llm.py
cp .env.example .env
python scripts/check_environment.py
```

`scripts/setup_models.py` installs the voice and vision assets. `scripts/setup_llm.py` downloads the approximately 1.12 GB Apache-2.0 Qwen GGUF. Set `MIC_INDEX` if the default input device is not the USB microphone. The assistant needs no API key or internet after setup. Then:

```bash
python run.py
```

Open `http://<raspberry-pi-ip>:5000`.

Say “AURUS”, pause briefly, then ask a question. Deterministic commands such as stop, follow, explore, remember, and status are always handled before the LLM.

After the manual launch has passed the full showcase repeatedly, copy `deploy/aurus.service` to `/etc/systemd/system/`, adjust its user/path if needed, then enable it with `sudo systemctl enable --now aurus`.

## Tests

```bash
python -m unittest tests.test_suite tests.test_evaluation_core -v
```

`tests/test_ultrasonic_voltage.py` is an interactive Raspberry Pi hardware utility and is intentionally not included in desktop test discovery.

## Documentation

See [docs/EVALUATION_ARCHITECTURE.md](docs/EVALUATION_ARCHITECTURE.md) for architecture, interfaces, setup, degradation behavior, and acceptance criteria.
