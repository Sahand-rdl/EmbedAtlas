"""
EmbedAtlas — CLI entry point
Invoked when the user runs `embedatlas` after `pip install`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "embedatlas" / "app.py"

    if not app_path.exists():
        print(f"[EmbedAtlas] Could not find app.py at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
        "--server.port",
        "8501",
        "--theme.base",
        "dark",
    ]

    print(f"\n🗺️  Starting EmbedAtlas …  (http://localhost:8501)\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[EmbedAtlas] Stopped.")
    except subprocess.CalledProcessError as e:
        print(f"[EmbedAtlas] Error: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
