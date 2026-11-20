You are splitting one long lesson of a story-driven machine learning theory course into two complete lessons, {N}a and {N}b. The learner asked for it so each sitting stays manageable.

**Source.** /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{FOLDER}/1-prose.md and 1-sheet.md. The prose has been through a clean-context review: keep its wording and its numbers.

**The split point,** proposed by that review: {SPLIT}

**Read first:**
- /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/series.md (world, conventions, glossary).
- The "Learner's rules" in /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/TASKS.md.
- ~/.claude/skills/klay-learning-ladder/references/writing.md (lesson shape and sheet format).
- 01-learning-needs-a-bias/1-sheet.md, as the model for a sheet.

**Write two folders**, each a complete lesson:

- /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{NN}a-{SLUG_A}/
- /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/{NN}b-{SLUG_B}/

Each folder gets 1-prose.md and 1-sheet.md, and copies of the series lesson.css and tokens.css.

**Each half must stand alone as a lesson:**
- It opens on a bug in the mail room.
  - {N}a keeps the original opening.
  - {N}b gets a new short opening bug that grows out of {N}a's ending, plus a short recap of what it needs from {N}a.
- It ends when the learner fixes that bug in a two-act lab:
  - `[[lab]]` explores;
  - `[[lab-act2]]` fixes the bug, and succeeds only when the lesson's guarantee is met, never on a lucky run.
- Where the original lab served only one half, specify the other half's lab in its sheet, in the same format as lesson 1's sheet. Compute its numbers with a script, and paste real outputs.
- It has its own:
  - "Fix the bug" section;
  - recap;
  - concept check of 2 or 3 questions, with answers only in the sheet;
  - problems: distribute the original problems, and add one only if a half has none, with a hint ladder in the sheet.
- It ends with a cliffhanger:
  - {N}a's leads into {N}b;
  - {N}b keeps the original cliffhanger into lesson {NEXT}.
- Its sheet holds everything the original sheet held for its half: objective, bug and fix, key words, terms, misconception, numbers, figure specs for its `[[diagram:...]]` slots, the lab spec with its iwrzwr study, concept-check answers, problems with hint ladders, and Assumptions. Every number in the sheet must match its prose.

**Keep all the rules:**
- Every formula is read aloud: symbols in words, read inside out, one bold plain sentence, a tiny story number, and a Python mirror.
- Concept before symbol: an idea comes in words with a number before its formula.
- Natural teacher voice; no em dashes; curly quotes; `\varepsilon`, never `\epsilon`.
- No "the book", and no chapter, section or theorem numbers in the prose.
- Snippets use the standard library, with every line at most 66 characters. Run each with `python3 -I` and paste its real output.
- A symbol that {N}b uses from {N}a gets a one-line reminder at first use.
- Do not shorten walkthroughs to save words. The split is the length fix.

**Check:** run `python3 -I /Users/cleiton.loiola/personal/lessons/uml-shalev-shwartz/_kit/check_prose.py` on both folders. It must report:
- lines>66 0;
- book refs 0;
- em dashes 0;
- \epsilon 0;
- only build-it starters failing.

**Limits:**
- Do not edit the original folder, series.md, TASKS.md or any other lesson.
- Do not commit.
- No agent-voice, no subagents, no browsers.
- Put scratch files in /private/tmp/claude-502/-Users-cleiton-loiola-personal/10412219-6ffb-4586-aa42-9a1b7eaa0636/scratchpad/split{N}/.

**Final report,** under 200 words:
- both folder paths;
- both titles, one line each;
- the opening question of each half, one sentence each, for the course index;
- word counts;
- the checker lines;
- the new bug, lab and fix for each half;
- anything new you computed, with its outputs.
