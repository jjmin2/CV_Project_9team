"""이메일 입력이나 시작 안내 없이 환영 문구만 표시하고 앱을 실행한다."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    print("Welcome to Streamlit!", flush=True)
    try:
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
             "--server.headless=true", "--logger.hideWelcomeMessage=true"],
            cwd=root,
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
