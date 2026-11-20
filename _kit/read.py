import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parent
LADDER = Path.home() / ".claude/skills/klay-learning-ladder"
sys.path.insert(0, str(KIT))

import highlight  # noqa: E402

DOCS = {"1-prose.md": "Lesson text (draft)", "1-sheet.md": "Teacher's sheet"}
SLOT = re.compile(r"^\[\[([\w:-]+)\]\]\s*$", re.M)
LABEL = re.compile(r"^[A-Z][\w'’ -]{1,30}: ")


def lesson_title(folder: Path) -> str:
    prose = folder / "1-prose.md"
    if prose.exists():
        first = next((l for l in prose.read_text(encoding="utf-8").splitlines() if l.startswith("# ")), "")
        if first:
            return first[2:].strip()
    return folder.name


def sectioned(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if LABEL.match(line) and lines and lines[-1].strip():
            lines.append("")
        lines.append(line)
    text = "\n".join(lines).strip("\n")
    head, sep, rest = text.partition("\n## ")
    if text.startswith("## "):
        return text + "\n"
    if not sep:
        return "## Overview\n" + text + "\n"
    return "## Overview\n" + head + "\n\n## " + rest + "\n"


def render(md: Path) -> str:
    text = sectioned(md.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp) / md.name
        flat.write_text(text, encoding="utf-8")
        args = []
        for name in SLOT.findall(text):
            slot = Path(tmp) / f"{name.replace(':', '-')}.html"
            slot.write_text(f'<p class="pending">[{html.escape(name)}: built on the lesson page]</p>', encoding="utf-8")
            args += ["--slot", f"{name}={slot}"]
        body = subprocess.run([sys.executable, str(LADDER / "scripts/prose.py"), str(flat), *args],
                              capture_output=True, text=True, check=True).stdout
    return highlight.highlight(body)


def page(folder: Path, md: Path, body: str) -> str:
    number = folder.name.split("-", 1)[0].lstrip("0")
    title = f"{DOCS[md.name]}: {lesson_title(folder)}"
    other = "1-sheet.html" if md.name == "1-prose.md" else "1-prose.html"
    other_label = DOCS["1-sheet.md" if md.name == "1-prose.md" else "1-prose.md"]
    kit_css = (KIT / "page.css").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="../tokens.css">
<link rel="stylesheet" href="../lesson.css">
<style>
{kit_css}
.pending {{ color: var(--ink-soft); font-family: var(--font-ui); font-size: var(--text-small); }}
</style>
<div class="wrap">
<nav class="crumbs" aria-label="Course"><a href="../index.html">The Mail Room</a><span aria-hidden="true">/</span><span>Lesson {html.escape(number)}</span><span aria-hidden="true">/</span><span>{html.escape(DOCS[md.name])}</span></nav>
<header><h1>{html.escape(lesson_title(folder))}</h1><p class="links">{html.escape(DOCS[md.name])} · <a href="{other}">{html.escape(other_label)}</a></p></header>
{body}
</div>
</html>
"""


def build(folder: Path) -> list[Path]:
    built = []
    for name in DOCS:
        md = folder / name
        if not md.exists():
            continue
        out = folder / name.replace(".md", ".html")
        if out.exists() and out.stat().st_mtime >= md.stat().st_mtime:
            built.append(out)
            continue
        src = folder / name.replace(".md", ".src.html")
        src.write_text(page(folder, md, render(md)), encoding="utf-8")
        run = subprocess.run(["node", str(LADDER / "scripts/math.mjs"), str(src), str(out), "--css", "../katex/katex.min.css"],
                             capture_output=True, text=True)
        errors = run.stderr.count("math error")
        if errors:
            print(f"{md.relative_to(ROOT)}: {errors} formula(s) failed to render", file=sys.stderr)
        if not out.exists():
            raise SystemExit(run.stderr)
        src.unlink()
        built.append(out)
    return built


def main() -> None:
    folders = [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.glob("[0-3][0-9]*-*/") if p.is_dir())
    for folder in folders:
        for out in build(folder.resolve()):
            print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
