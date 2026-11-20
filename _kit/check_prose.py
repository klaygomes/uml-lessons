import re
import subprocess
import sys
from pathlib import Path

BOOK = re.compile(r"(?i)\bthe book\b|\bbook's|\bchapter [0-9]|\bsection [0-9]|\b(theorem|corollary|lemma|definition|exercise) [0-9]")


def check(folder):
    src = (Path(folder) / "1-prose.md").read_text(encoding="utf-8")
    fences = re.findall(r"```(\w*)\n(.*?)```", src, re.S)
    failing = []
    for i, (lang, code) in enumerate(fences, 1):
        if lang == "python" and subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=600).returncode:
            failing.append(i)
    long = sum(len(line) > 66 for _, code in fences for line in code.splitlines())
    words = len(src.split())
    print(f"{Path(folder).name}: {words} words, {sum(l == 'python' for l, _ in fences)} python fences, failing {failing}, "
          f"lines>66 {long}, book refs {len(BOOK.findall(src))}, em dashes {src.count(chr(0x2014))}, "
          f"\\epsilon {len(re.findall(r'\\\\epsilon(?![a-zA-Z])', src))}")


for folder in sys.argv[1:]:
    check(folder)
