"""AURUS Evaluation Edition entry point."""

from __future__ import annotations

import atexit
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.runtime import RobotRuntime
from src.web.app import create_app


def configure_logging(root: Path) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(log_dir / "aurus.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def main() -> None:
    root = Path(__file__).resolve().parent
    configure_logging(root)
    logger = logging.getLogger("aurus")
    runtime = RobotRuntime(root)
    app, socketio, start_web_tasks, stop_web_tasks = create_app(runtime)
    stopped = False

    def shutdown():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        stop_web_tasks()
        runtime.stop()

    atexit.register(shutdown)
    runtime.start()
    start_web_tasks()
    logger.info(
        "AURUS Evaluation Edition ready on %s:%s",
        os.getenv("AURUS_HOST", "0.0.0.0"),
        os.getenv("AURUS_PORT", "5000"),
    )
    try:
        socketio.run(
            app,
            host=os.getenv("AURUS_HOST", "0.0.0.0"),
            port=int(os.getenv("AURUS_PORT", "5000")),
            debug=False,
            allow_unsafe_werkzeug=True,
        )
    finally:
        shutdown()


if __name__ == "__main__":
    main()
