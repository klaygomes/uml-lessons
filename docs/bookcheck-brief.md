You are doing the last text pass on one lesson of a story-driven machine learning theory course, /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}/. The prose has been through a clean-context review. Your two jobs are a book check and a sheet resync.

**Source of truth, never memory:** the PDF "/Users/cleiton.loiola/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/understanding-machine-learning-theory-algorithms.pdf". Read it with `pdftotext -layout -f N -l M "<pdf>" -`. book/map.md gives the chapter's pages; the PDF page number equals the printed page number.

**1. Book check.**
- Read the chapter's pages.
- In 1-sheet.md, the Assumptions list and any "book error", "typo" or "slip" notes record claims that the book is wrong somewhere. TASKS.md lists more under this lesson's "Writer's claims to verify". Check each claim against the PDF and decide:
  - **confirmed:** the book really is wrong, and the lesson's fix is right;
  - **writer wrong:** the book is right, so fix the lesson's prose and sheet to follow the book;
  - **a choice:** the book is not wrong, the lesson just does it differently; keep it if it is valid, and say so.
- Then check every theorem, definition and constant the prose states against the PDF. Mark each one correct, sharper and valid (keep it, and say why in the sheet), or wrong (fix it). Run a Python check for any constant you change.

**2. Sheet resync.** 1-sheet.md must match the reviewed 1-prose.md exactly:
- every number;
- the story details (vocabulary sizes, budgets, names, worlds);
- the key words;
- the `[[diagram:...]]` figure list;
- the lab spec;
- the concept-check answers;
- the problem numbering.

Recompute any number with a script and paste the real output. Also fix LaTeX that does not render in KaTeX: for example `j^\*` should be `j^*`, and `\epsilon` should be `\varepsilon`. Check with: `node ~/.claude/skills/klay-learning-ladder/scripts/math.mjs <in.html> <out.html>` on a scratch HTML file holding the sheet's formulas, or simply run `python3 -I /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/_kit/read.py /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}` and read its stderr. It must report no "failed to render" lines; if it does, fix the LaTeX in that file.

**Keep the prose's voice and rules:**
- Change prose only to fix a real error.
- No "the book", and no chapter, section or theorem numbers in the prose. Book page references belong in the sheet only.
- Formula walkthroughs stay.
- No em dashes; curly quotes.

**Check:** `python3 -I /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/_kit/check_prose.py /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}` must stay clean, with only build-it starters failing.

**Limits:**
- Edit only that lesson's 1-prose.md and 1-sheet.md.
- Do not commit.
- No agent-voice, no subagents, no browsers.
- Put scratch files in /private/tmp/claude-502/-Users-cleiton-loiola-personal/10412219-6ffb-4586-aa42-9a1b7eaa0636/scratchpad/check{N}/.

**Final report,** under 150 words:
- each writer claim with its verdict (confirmed, writer wrong, or a choice);
- prose errors fixed;
- sheet items resynced;
- LaTeX fixed;
- the checker line;
- whether `read.py` rendered every formula.
