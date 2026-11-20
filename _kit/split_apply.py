import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    "1 prose and sheet",
    "2 gates: book check (PDF agents), sheet resync, greps",
    "3 review stop: user approved the prose",
    "4 figures",
    "5 page",
    "6 learner judge: ready",
    "7 verification",
    "8 delivery: series.md, index.html",
    "9 concept check: mastery confirmed",
    "10 end-of-chapter problems done",
]


def section(n: str, title: str, folder: str, note: str) -> str:
    lines = [f"### {n}. {title}", "", f"Folder: `{folder}/`", ""]
    for i, step in enumerate(STEPS):
        lines.append(f"- [x] {step} ({note})" if i == 0 else f"- [ ] {step}")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    n, title_a, question_a, title_b, question_b, date = sys.argv[1:7]
    num = int(n)
    old = next(ROOT.glob(f"{num:02d}-*/"))
    a = next(ROOT.glob(f"{num:02d}a-*/"))
    b = next(ROOT.glob(f"{num:02d}b-*/"))
    (ROOT / "_archive").mkdir(exist_ok=True)
    subprocess.run(["git", "mv", str(old), str(ROOT / "_archive" / old.name)], cwd=ROOT, check=True)

    series = ROOT / "series.md"
    text = series.read_text(encoding="utf-8")
    row = re.compile(rf"^\| {num} \| [^\n]*$", re.M)
    assert row.search(text), f"no course-plan row for {num}"
    text = row.sub(f"| {num}a | {title_a} | {question_a} |\n| {num}b | {title_b} | {question_b} |", text, count=1)
    series.write_text(text, encoding="utf-8")

    tasks = ROOT / "TASKS.md"
    text = tasks.read_text(encoding="utf-8")
    start = text.index(f"### {num}. ")
    end = text.index("\n### ", start + 1) + 1
    note = f"split from lesson {num} on {date}; original in `_archive/{old.name}`; clean-context review carried over"
    text = text[:start] + section(f"{num}a", title_a, a.name, note) + section(f"{num}b", title_b, b.name, note) + text[end:]
    tasks.write_text(text, encoding="utf-8")
    print(f"{old.name} -> _archive; {a.name}, {b.name} in series.md and TASKS.md")


if __name__ == "__main__":
    main()
