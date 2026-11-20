You are a fresh reader and editor for one lesson of a course on machine learning theory. You have no other context, on purpose. Read only these files:

- the lesson: /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}/1-prose.md
- the course conventions and glossary: /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/series.md
- for terms the lesson recaps from earlier lessons, you may open the earlier lesson's 1-prose.md to check what it taught, but judge this lesson as its reader meets it.

Do not read the teacher's sheet (1-sheet.md), TASKS.md, or any other notes before your first full read: you must see only what the learner sees.

**Who reads it.** An experienced software engineer with no machine learning background, who understands math best through Python code, has done every earlier lesson, and reads this one top to bottom, once.

**Pass 1, judge.** Read the lesson top to bottom as that reader and list, with line numbers:

1. Every term, symbol, notation or result used before it is explained, even inline, even once (concept before symbol). A recap of an earlier lesson's term counts as an explanation only if it says what the term means.
2. Every formula, display or dense inline, that is not followed by a walkthrough: each symbol named in words, read from the inside out, the whole formula as one plain sentence in bold, one tiny number from the story pushed through it, and a Python snippet that mirrors it with its real output (reusing an existing snippet is fine when the prose links them).
3. Every step of a proof or derivation where you had to stop and work something out the prose did not say.
4. Every place where the idea arrives before its intuition: a formula or definition with no plain-language idea and number before it.
5. Anything wrong: a number that does not match its snippet's output, a snippet that fails to run (other than the build-it starter, which fails by design), a claim that does not follow.

**Pass 2, improve.** Fix every item yourself in 1-prose.md:

- Move or add explanations so each idea comes first in words with a number, then the formula, then the walkthrough.
- Add missing walkthroughs and snippets. Snippets use the standard library, about 15 lines at most, every line at most 66 characters, no comments, and names that read as the math. Run each with `python3 -I` and paste its real output.
- Fill the gaps in proofs with the missing step, spelled out.

Keep the story, the voice (a natural teacher talking: no robotic short sentences, no em dashes, curly quotes), the headings, the slots (`[[diagram:...]]`, `[[lab]]`, `[[lab-act2]]`) and the build-it starter. Use `\varepsilon`, never `\epsilon`. Never write "the book", or chapter, section or theorem numbers. Do not bloat: one clear sentence beats a paragraph. Where the problem is the order, move text rather than repeat it.

**Check.** Run `python3 -I /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/_kit/check_prose.py /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}`. You are done when every check is clean:

- lines>66 is 0;
- book refs is 0;
- em dashes is 0;
- \epsilon is 0;
- only the build-it starter fails.

**Limits.** No agent-voice, no subagents, no browsers. Put scratch files in /private/tmp/claude-502/-Users-cleiton-loiola-personal/10412219-6ffb-4586-aa42-9a1b7eaa0636/scratchpad/review{N}/. Edit only that lesson's 1-prose.md. Do not commit.

**Final report**, under 150 words:

- the count of findings per kind (1 to 5) and the three worst;
- what you changed;
- new snippets and their outputs;
- the checker line;
- a verdict: ready, or what is still open.
