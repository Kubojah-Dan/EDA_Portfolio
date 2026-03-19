"""
app.py
Convenience launcher: starts FastAPI backend and Dash frontend in separate processes.
Usage: python app/app.py
"""
import os
import sys
import subprocess
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config
from src.utils.logger import get_logger

log = get_logger("app")


def main():
    cfg = load_config()
    python = sys.executable

    api_port = cfg["api"]["port"]
    dash_port = cfg["dashboard"]["port"]

    log.info(f"Starting FastAPI backend on port {api_port}...")
    backend = subprocess.Popen([
        python, "-m", "uvicorn",
        "app.backend.main:app",
        "--host", cfg["api"]["host"],
        "--port", str(api_port),
        "--reload",
    ])

    time.sleep(2)

    log.info(f"Starting Dash frontend on port {dash_port}...")
    frontend = subprocess.Popen([python, "app/frontend/dashboard.py"])

    log.info(f"✅ API docs:      http://localhost:{api_port}/docs")
    log.info(f"✅ Dashboard:     http://localhost:{dash_port}")
    log.info("Press Ctrl+C to stop both services.")

    def shutdown(sig, frame):
        log.info("Shutting down...")
        backend.terminate()
        frontend.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    backend.wait()
    frontend.wait()


if __name__ == "__main__":
    main()

