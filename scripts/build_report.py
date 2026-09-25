"""Build docs/MILESTONE1_REPORT.md from docs/MILESTONE1_REPORT.template.md.

Every line ``<!-- run: <command> -->`` in the template is replaced by that command's
stdout (normally ``python scripts/summarize.py ...``), so tables are regenerated from
the result artifacts rather than typed by hand.

    python scripts/build_report.py
"""
from pathlib import Path
import re
import shlex
import subprocess
import sys

TEMPLATE = Path("docs/MILESTONE1_REPORT.template.md")
OUTPUT = Path("docs/MILESTONE1_REPORT.md")


def run(command: str) -> str:
    argv = shlex.split(command)
    if argv[0] == "python":
        argv[0] = sys.executable
    return subprocess.run(argv, check=True, capture_output=True, text=True, encoding="utf-8").stdout.rstrip()


def main():
    text = TEMPLATE.read_text(encoding="utf-8")
    text = re.sub(r"^<!-- run: (.+?) -->$", lambda m: run(m.group(1)), text, flags=re.MULTILINE)
    OUTPUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
