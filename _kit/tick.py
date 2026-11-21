import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    lesson, step, note = sys.argv[1], sys.argv[2], sys.argv[3]
    path = ROOT / "TASKS.md"
    text = path.read_text(encoding="utf-8")
    start = text.index(f"### {lesson}. ")
    end = text.find("\n### ", start + 1)
    end = len(text) if end < 0 else end
    block = text[start:end]
    line = re.search(rf"^- \[ \] {step} .*$", block, re.M)
    if not line:
        raise SystemExit(f"no open step {step} under lesson {lesson}")
    done = f"- [x] {line.group(0)[6:].split(' (')[0]} ({note})"
    path.write_text(text[:start] + block.replace(line.group(0), done, 1) + text[end:], encoding="utf-8")
    print(done)


if __name__ == "__main__":
    main()
