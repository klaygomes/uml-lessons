import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = 10

STATES = {
    "ready": "Ready",
    "page": "Page built",
    "drafted": "Text drafted",
    "writing": "Being written",
    "planned": "Planned",
}


def plan():
    rows = []
    text = (ROOT / "series.md").read_text(encoding="utf-8")
    section = text.split("## Course plan", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        m = re.match(r"\| (\d+[ab]?) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            rows.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return rows


def progress():
    found = {}
    parts = re.split(r"^### (\d+[ab]?)\. .*$", (ROOT / "TASKS.md").read_text(encoding="utf-8"), flags=re.M)
    for n, body in zip(parts[1::2], parts[2::2]):
        steps = {}
        for m in re.finditer(r"^- \[([ x])\] (\d+) (.*)$", body, flags=re.M):
            steps[int(m.group(2))] = (m.group(1) == "x", m.group(3))
        found[n] = steps
    return found


def state(n, steps):
    number, part = (n[:-1], n[-1]) if n[-1] in "ab" else (n, "")
    folder = next(iter(sorted(ROOT.glob(f"{int(number):02d}{part}-*/"))), None)
    done = lambda k: steps.get(k, (False, ""))[0]
    page = folder / "3-page/3-page.html" if folder else None
    prose = folder / "1-prose.md" if folder else None
    if done(8) and page and page.exists():
        return "ready", page
    if done(5) and page and page.exists():
        return "page", page
    if done(1) and prose and prose.exists():
        return "drafted", prose
    if "drafting" in steps.get(1, (False, ""))[1]:
        return "writing", None
    return "planned", None


def item(n, title, question, steps):
    key, target = state(n, steps)
    ticked = sum(1 for k in range(1, STEPS + 1) if steps.get(k, (False, ""))[0])
    links = []
    if key in ("ready", "page"):
        label = "Open the lesson" if key == "ready" else "Open the page (draft)"
        links.append((target, label))
    if target:
        folder = target.parent.parent if target.parent.name == "3-page" else target.parent
        for name, label in (("1-prose.html", "Lesson text"), ("1-sheet.html", "Teacher's sheet")):
            if (folder / name).exists():
                links.append((folder / name, label))
    action = "".join(f' <a href="{html.escape(path.relative_to(ROOT).as_posix())}">{html.escape(label)}</a>' for path, label in links)
    return (
        f'<li class="is-{key}"><span class="n">{n}</span><div>'
        f"<h2>{html.escape(title)}</h2><p>{html.escape(question)}</p>"
        f'<p class="links"><span class="state">{STATES[key]}</span>'
        f'<span class="steps" aria-label="{ticked} of {STEPS} steps done">'
        f'<meter min="0" max="{STEPS}" value="{ticked}"></meter> {ticked}/{STEPS}</span>{action}</p>'
        f"</div></li>"
    )


def build():
    steps = progress()
    rows = plan()
    keys = [state(n, steps.get(n, {}))[0] for n, _, _ in rows]
    summary = " · ".join(f"{keys.count(k)} {STATES[k].lower()}" for k in STATES if keys.count(k))
    items = "\n".join(item(n, t, q, steps.get(n, {})) for n, t, q in rows)
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Mail Room</title>
<link rel="stylesheet" href="tokens.css">
<link rel="stylesheet" href="lesson.css">
<style>
.course {{ list-style: none; padding: 0; display: grid; gap: var(--space-5); }}
.course li {{ display: grid; grid-template-columns: var(--space-10) 1fr; gap: var(--space-3); border-top: var(--line-hair) solid var(--rule); padding-top: var(--space-4); }}
.course .n {{ font-family: var(--font-mono); font-size: var(--text-h3); color: var(--ink-soft); }}
.course h2 {{ margin: 0 0 var(--space-1); font-size: var(--text-h3); }}
.course p {{ margin: 0 0 var(--space-1); }}
.course .links {{ display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3); font-family: var(--font-ui); font-size: var(--text-small); }}
.course .state {{ padding: 0 var(--space-2); border-radius: var(--radius-sm); background: var(--paper-2); color: var(--ink-soft); }}
.course .is-ready .state {{ background: var(--ok-soft); color: var(--ok); }}
.course .is-page .state, .course .is-drafted .state {{ background: var(--input-soft); color: var(--input); }}
.course .is-writing .state {{ background: var(--accent-soft); color: var(--accent-ink); }}
.course .steps {{ color: var(--ink-soft); font-family: var(--font-mono); }}
.course meter {{ width: var(--space-20); vertical-align: middle; }}
.course .is-planned h2 {{ color: var(--ink-soft); }}
.summary {{ font-family: var(--font-ui); font-size: var(--text-small); color: var(--ink-soft); }}
</style>
<div class="wrap">
<header>
<h1>The Mail Room</h1>
<p class="question">You are building the learner behind a company's mail filter. Every lesson opens on a bug your users notice, and ends when you fix it yourself in a lab. Thirty-one lessons, from the first question of learning theory to its deepest bounds.</p>
<p class="summary">{html.escape(summary)}. Each lesson has {STEPS} steps, from the draft text to your concept check and problems.</p>
</header>
<ol class="course">
{items}
</ol>
<footer><span>Lesson text, figures and pages: CC BY-NC 4.0. Code: MIT. See NOTICE.</span></footer>
</div>
</html>
"""
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    print(ROOT / "index.html")
    print(summary)


if __name__ == "__main__":
    build()
