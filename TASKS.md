# Tasks

Where the course build stands, so work can resume in a new session on any day. Built with the `klay-learning-ladder` skill. Pages ship as files in this repo; never publish them as Artifacts. No videos in this series.

To resume: read **Session handoff** near the end first (every skill, tool, reference and rule the user gave), then continue from the first unticked box under **Status**. Work skipped on purpose (gates, reviews, judges) is collected under **Deferred (do later)** near the end: check it every session. Read `series.md` first (world, glossary, next bug) and update it last.

## Source rule

Every claim about the book comes from the PDF, never from memory: `~/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/understanding-machine-learning-theory-algorithms.pdf`. `book/map.md` gives each chapter's PDF pages, sections and exercises; `book/notation.md` gives the notation. Extract a range with `pdftotext -f FIRST -l LAST "<pdf>" -`. For each lesson, spawn parallel agents that each read one section range of the PDF and check the prose against it (step 2 below).

## Visual tools

All cloned locally under `~/.local/share/klay-learning-ladder/` (2026-11-20). Pick the tool by what the idea needs, and brief every figure or page builder with this table.

| Tool | Local path | Commit | Use it for |
|---|---|---|---|
| diagram-design | `diagram-design/` (skill in `skills/diagram-design/`) | 0cf73c3 | rule and relationship figures: charts, flows, geometric pictures, proof sketches (the default) |
| impeccable | `impeccable/` (skill in `skill/`, detector `skill/scripts/impeccable detect --json --scope type <page>`, then `--scope layout`) | b49177d | layout and typography passes on every page: Read mode, refinement of the series world (fonts and palette are committed); run the detector before and after |
| stack-diagram | `stack-diagram/` (skill in `skills/stack-diagram/SKILL.md`) | 5889b8e | system and pipeline pictures: the mail filter's architecture (users label, training sequence, learner, hypothesis, inbox), SGD and boosting loops drawn as systems, online learning as a service. Swap its JetBrains Mono for Geist Mono and its colors for `lesson.css` tokens. |
| iwrzwr visual archive | `iwrzwr-visual-archive/` | 794af32 | canvas drawing in labs (the kit's `lab.js` already adapts its Sample Hold Ladder study; MIT notice kept) |
| three.js | `threejs/` (sparse: `build`, `src`, `examples/jsm`, `manual`, `docs`) | 0245b6d | 3D only where the idea is 3D: halfspaces and margins in \(\mathbb R^3\) (ch. 9, 15), loss surfaces and gradient paths (ch. 12, 14), PCA projections (ch. 23), kernel feature maps lifting data (ch. 16). Load from a CDN with an exact version on the page; the clone is for reading examples and docs. |
| img2threejs | `img2threejs/` | 3e7099e | a physical object the learner must see in 3D (rare) |

## The learner's rules (from the syllabus prompt)

- Learner: experienced software engineer, learning ML theory from the ground up.
- Define every concept before using it. Full definitions, lemmas, theorems and step-by-step proofs, citing Appendices A, B and C where a step needs them.
- Connect each idea to an algorithm and its time and space cost.
- Notation of the book's Section 1.6: \(\mathcal X, \mathcal Y, \mathcal H\), \(S = (z_1, \dots, z_m) \sim \mathcal D^m\), \(\ell: \mathcal H \times Z \to \mathbb R_+\), \(L_{\mathcal D}(h)\), \(L_S(h)\), \(\|x\|\), \(O, \Theta, \tilde O\).
- Writing: clear and natural, a teacher talking. No robotic short-sentence style, no limited vocabulary. Easy to read, exact on concepts.
- Read every formula aloud for a learner meeting it for the first time (user, 2026-11-20: "everytime you show complex math, you have to explain as i'm learning it"). After each display formula, and any dense inline one: name every symbol and piece of notation in words (sets and set-builder braces, the colon as "such that", \(|\cdot|\), \([m]\), \(\sim\), \(\mathbb P\), \(\mathbb E\), subscripts, sums, sup, argmin, and so on), read it from the inside out, give the whole formula as one plain sentence in bold, work one tiny number from the story through it, and add a Python snippet that mirrors the formula term by term with its real output. A symbol explained in an earlier lesson still gets a one-line reminder the first time it appears in each lesson. Model: lesson 1, "The formal statements", the empirical and true risk.
- Concept before symbol (the klay-learning-ladder rule "every term defined before use", "show every idea working before you name it"): introduce each idea in words with a small number from the story before its formula appears, and never use a term, symbol or result, even inline, before it has been explained. The walkthrough after a formula reads the notation; it does not replace explaining the idea first.
- Make every class playful (user, 2026-11-20: "dont forget using the iwrzwr visual archive for lab drawing code and diagram-design to make classes playful"): every new concept gets a diagram-design figure (animated when the idea is a process, still under reduced motion), and every lab's drawing code adapts a named study from the iwrzwr visual archive (the sheet names the study, the code keeps its MIT notice). The learner should poke, drag and watch things move, not read static boxes.
- Never advance to the next unit until the learner has answered the concept check and confirmed mastery.

## How each lesson is made

Every lesson runs the same loop. Each lesson under **Status** has one checkbox per step; tick it, with the date, as soon as the step is done.

```mermaid
flowchart TD
  P[1. Prose + sheet] --> RV[1b. Clean-context reviewer: judge, then improve]
  RV --> G1[2. Read-aloud pass + fresh-reader judge]
  G1 --> R1{3. User reviews prose}
  R1 -->|changes| P
  R1 -->|ok| F[4. Figures: 2-figures/figures.mjs]
  F --> PG[5. Page: 3-page/ shell, lab, build.py]
  PG --> J[6. Learner judge until ready]
  J --> V[7. Cross-check prose, figures, page]
  V --> D[8. Deliver: series.md, index.html]
  D --> CC{9. Learner answers concept check in chat}
  CC -->|gaps| CC
  CC -->|mastery| X[10. End-of-chapter problems, hints only]
  X --> N[Next lesson]
```

1. **Prose and sheet.** `1-prose.md` follows the five-step loop: the problem and motivation (opened as a bug in the mail room), formal definitions and proofs, the algorithm with pseudocode, complexity and a build-it challenge, a concept check of 2 or 3 questions. `1-sheet.md` holds terms, numbers, the lab spec and the concept-check answers. Every number is computed by a script, never guessed.
1b. **Clean-context reviewer** (user, 2026-11-20: "include a reviewer with clean context to judge and improve"). A fresh agent with only `docs/reviewer-brief.md` reads the prose as the learner, lists every term or symbol used before it is explained, every formula without a walkthrough, every proof gap and every wrong number, then fixes them in `1-prose.md`. Runs after the writer and the formula-walkthrough pass, before the user's review. The lead reruns `_kit/check_prose.py` before ticking.
2. **Gates.** Book check: parallel agents, one per section range of the PDF, mark every claim correct, not in the book, or wrong; `agent-voice speak` read-aloud pass per section; a fresh-reader subagent with no context; grep the key words for synonyms; grep for em dashes.
3. **Review stop.** Show the user the prose and the sheet. Nothing else gets built before this.
4. **Figures.** One per new concept, through diagram-design (never hand-drawn SVG), animated when the idea is a process, colors from `lesson.css`. `node _kit/build-figures.mjs NN-slug/2-figures/figures.mjs`, then `self_check.py` and `verify-geometry.py`; render once per theme.
5. **Page.** Lab drawing code adapted from a named iwrzwr visual archive study (list them in `~/.local/share/klay-learning-ladder/iwrzwr-visual-archive/`, pick the one whose motion matches the idea, keep the MIT notice); `3-page/` shell, `lab.html`, `lab-act2.html`, `lab.js`, `build.py`; `python3 3-page/build.py` exits 0.
6. **Learner judge.** A fresh agent walks the page in its own Chrome; fix and judge again until "ready".
7. **Verification.** Prose, figures and page teach one mental model; fix each mismatch at the earliest rung that has it.
8. **Delivery.** Mark the row in `series.md`, add glossary terms and the next bug, run `python3 _kit/index.py`.
9. **Concept check.** The learner answers in chat; critique rigorously; repeat until mastery is confirmed.
10. **End-of-chapter problems.** Pick 2 or 3 of the book's exercises (standard and starred once ready). Hints only, never the solution.

## Status

Tick a box the moment its step is done, and add the date. The first unticked box in the file is where work resumes.

### 0. Series setup

- [x] folder, `_kit/` (copied from dsp-oppenheim, video parts removed), `lesson.css`, README, NOTICE, `.gitignore`, `series.md`, `TASKS.md`, git init (2026-11-20)
- [x] `book/notation.md`: the book's notation with page numbers (2026-11-20)
- [x] `book/map.md`: page, section and exercise map of all 31 chapters; PDF page equals printed page throughout; 122 exercises (2026-11-20)
- [ ] first commit (when the user asks)
- [x] index lists all 31 lessons with status (user asked, 2026-11-20): titles from the series.md "Course plan" table, status and n/10 steps read from this file's boxes; rerun `python3 -I _kit/index.py` after every tick (2026-11-20)
- [x] (2026-11-20, uncommitted in ~/dotfiles) update the `klay-learning-ladder` skill with this session's findings (user asked, 2026-11-20): source from the PDF with parallel agents, TASKS.md with tick boxes, design tokens, impeccable passes, fonts and palette, layout rules, lab guarantee rule

### 1. Ch. 1, §1.1 to 1.4: what learning is, inductive bias, when ML is needed, kinds of learning, related fields

Folder: `01-learning-needs-a-bias/`

- [x] 1 prose and sheet (2026-11-20)
- [x] 2 gates (2026-11-20)
  - [x] fresh reader: random-pick learner, Proposition 2 independence spelled out, failure modes named, overclaims softened
  - [x] book check, three PDF agents (§1.1; §1.2 to 1.4; notation and §1.5 to 1.6): five sections rewritten to the book
  - [x] read-aloud pass (`read-aloud.log`): long sentences split
  - [x] key-word and em-dash greps clean
- [x] 3 review stop: user said go ahead; kept as one lesson, since the user set §1.1 to 1.4 as the unit (2026-11-20)
- [x] 4 figures: built by agent, checks rerun by lead, all 8 files pass self_check and verify-geometry (2026-11-20)
  - [x] messages (16-message grid)
  - [x] funnel (65,536 to 256 vs bias to 1)
  - [x] flip (the map φ pairing hypotheses)
  - [x] taxonomy (the book's four dimensions)
- [x] 5 page: built by agent, rebuilt by lead, `build.py` exits 0, 183 formulas, 6 slots (2026-11-20)
  - [x] lab act 1: memorizer vs one-feature learner, slider for m
  - [x] lab act 2 (budget 20 labels; one feature of ten 99% at 10 labels, 55 class 97% at 20): spammers change feature, pick class and label budget
  - [x] `python3 3-page/build.py` exits 0
- [x] 5b typography pass with impeccable (Read mode, refinement): measure 35em (about 70 characters), serif section headings with more space above than below, eyebrow replaced by a breadcrumb, curly quotes, themed selection and focus, canvas and figure text at least 12px. Scan: 26 type findings down to 4 (3 are the series' committed fonts, kept; 1 is the funnel phone figure, sent back to the figure builder) (2026-11-20)
- [x] 5c layout: one centred reading column, figures and labs break out evenly, tables and formulas fit at 360 px, no sideways scroll at 1280, 390 or 360 (2026-11-20)
- [x] 5d design tokens: `tokens.css` holds every colour, font, size, spacing, layout and motion value; `lesson.css`, `_kit/page.css`, `_kit/fig.css`, `_kit/lab.js` and the shell read only tokens; documented in `docs/design-tokens.md` (2026-11-20)
- [x] 5e fonts: Literata (reading), Rubik (headings and interface), IBM Plex Mono (code, data, figure labels), chosen by the user (2026-11-20)
- [x] 5f impeccable critique, dual-agent: 28/40 (Good); priority issues listed in the open items (2026-11-20)
- [x] 5g code highlighting (build-time, Python via the standard tokenizer, pseudocode by keyword, colours as tokens), Python snippets beside Propositions 1 and 2 with their real output, all book references removed from the page, figures and index (2026-11-20)
- [x] 6 learner judge: three judges, every finding addressed; final rebuild passes fit.mjs, no em dashes, no book references, no raw colours (2026-11-20) (judge 1 and 2: ready with polish. Judge 2 fixes: prose done 2026-11-20 (55-class hint corrected with its own bound and snippet, first-three class introduced, impostor defined before act 1, varphi, quotes, 4.6% scoped to the tiny world); lab fixes with the page builder (budget back to 20, fix must be earned, act 1 text, class text, Try another reset, inbox visual, subscripts); dark --input-soft strengthened. Judge 3 (2026-11-20): ready with polish; prose, figure and CSS fixes done by the lead (|X| instead of N in Proposition 1, lesson numbers in the taxonomy figure, captions centred, no smooth scroll under reduced motion); lab fixes with the page builder (feature 7 as a hypothetical, no fix credit for a lucky narrow class, budget 12, m = 3 wording, hide the answer before the prediction, reduced-motion success). No further judges without asking.)
- [x] 7 verification: judges 2 and 3 cross-checked every number across prose, snippets, labs and figures; lead reran all snippets (2026-11-20)
- [x] 8 delivery: series.md row and next bug, index.html rebuilt (2026-11-20)
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems: Chapter 1 has none in the book; its problems come with lesson 2

### 2. Ch. 1 §1.6 notation and Chapter 1 problems; Ch. 2 a gentle start (statistical framework, ERM, overfitting, finite classes, realizability, i.i.d., union bound)

Folder: `02-a-gentle-start/`

- [x] 1 prose and sheet (8,134 words, all Python snippets run by the lead, no book references, 2026-11-20)
- [ ] 2 gates: PDF self-check by the writer, number scripts and greps done; read-aloud partial and fresh reader not run (CPU rule: ask first); code reflowed to 66 characters and outputs rounded, rechecked by the lead (2026-11-20)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 3. Ch. 3 a formal learning model (PAC, sample complexity, agnostic PAC, Bayes optimal, general losses)

Folder: `03-a-formal-learning-model/`

- [x] 1 prose and sheet (7,576 words, all Python snippets run by the lead, no book references, 2026-11-20)
- [ ] 2 gates: PDF self-check by the writer, number scripts and greps done; read-aloud partial and fresh reader not run (CPU rule: ask first); code reflowed to 66 characters and outputs rounded, rechecked by the lead (2026-11-20)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 4. Ch. 4 uniform convergence (ε-representative samples, Hoeffding, discretization)

Folder: `04-uniform-convergence/`

- [x] 1 prose and sheet (7,479 words, 9 snippets run by the lead, the 9th is the build-it stub, no book references, 2026-11-20)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 5. Ch. 5 bias-complexity tradeoff (no free lunch, approximation vs estimation error)

Folder: `05-no-free-lunch/`

- [x] 1 prose and sheet (8,233 words, 8 snippets run by the lead, the failing fence is the build-it stub, no book references, 2026-11-20)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 6. Ch. 6 VC dimension (shattering, growth function, Sauer, fundamental theorem)

Folder: `06-vc-dimension/`

- [x] 1 prose and sheet (about 10,400 words with code, 13 snippets plus the build-it stub run by the lead, no book references, 2026-11-20; long: consider a 6a/6b split)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule); verify the writer's claimed book error (Hoeffding constant in the VC uniform-convergence proof)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 7. Ch. 7 nonuniform learnability (SRM, MDL, Kraft, Occam, consistency)

Folder: `07-nonuniform-learnability/`

- [x] 1 prose and sheet (about 9,580 words, snippets run by the lead, no book references, 2026-11-20)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule); verify the writer's own constructions (decision-list size, Elias gamma code, SRM stopping rule, the (1/10, 1/10) variant of the characterization proof)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 8. Ch. 8 the runtime of learning

Folder: `08-runtime-of-learning/`

- [x] 1 prose and sheet (10,310 words, 12 snippets plus 4 build-it stubs run by the lead, no book references; opening aligned with lesson 7's cliffhanger (1.55e28 lists at size 22), rechecked by the lead, 2026-11-20)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 9. Ch. 9 linear predictors (halfspaces, LP, Perceptron, least squares, logistic regression)

Folder: `09-linear-predictors/`

- [x] 1 prose and sheet (about 12,800 words with code, 15 snippets plus 2 build-it stubs run by the lead, no book references, 2026-11-20; long: the sheet lists three sections that could be cut; recheck its assumptions about lessons 4 to 8, which it could not read)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 10. Ch. 10 boosting (weak learnability, decision stumps, AdaBoost, Viola-Jones)

Folder: `10-boosting/`

- [x] 1 prose and sheet (about 10,900 words with code, 11 snippets plus 2 build-it stubs run by the lead, no book references, 2026-11-20; the writer reports three book slips it repaired: Example 10.1's outer rays, w_1 = 0.5 only for even T, and the VC dimension of T-stump combinations; verify them)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 11. Ch. 11 model selection and validation

Folder: `11-model-selection/`

- [x] 1 prose and sheet (8,580 words, 9 snippets plus the build-it stub run by the lead, no book references; check its boosting terms against lesson 10 when it lands, 2026-11-20)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 12. Ch. 12 convex learning problems

Folder: `12-convex-learning/`

- [x] 1 prose and sheet (about 13,300 words with code, 14 snippets run by the lead; the failing fence is the build-it stub; no book references, 2026-11-20; long like lesson 9). Writer's claims to verify: a radius typo in the book (β/2 should be √(β/2)); several proofs added by the writer for results the book states without proof
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 13. Ch. 13 regularization and stability

Folder: `13-regularization-stability/`

- [x] 1 prose and sheet (about 14,600 words with code, 18 snippets run by the lead; the failing fence is the build-it stub; no book references; written with the formula walkthrough rule, 19 plain-words readings; 2026-11-20). Writer's claims to verify: smooth stability rate 24β/(λm) against the book's 48 (sharper, implies the book's); strong-convexity modulus renamed σ; several proofs the writer's own
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 14. Ch. 14 stochastic gradient descent

Folder: `14-stochastic-gradient-descent/`

- [x] 1 prose and sheet (about 12,600 words with code, 13 snippets run by the lead; the failing fence is the build-it stub; no book references, 2026-11-20; long like lesson 9, the sheet names three cuttable sections). Writer's claims to verify: a proof needing 0 in H and ε ≤ 1, unstated in the book; the regularized-SGD constant 2ρ² against the book's 4ρ². Align terms and opening with lessons 12 and 13 when they land
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 15. Ch. 15 support vector machines

Folder: `15-support-vector-machines/`

- [x] 1 prose and sheet (about 12,700 words with code, 15 snippets run by the lead; the failing fence is the build-it stub; no book references, 2026-11-20; long like lesson 9). Writer's claims to verify: an off-by-one in the book's SGD-for-soft-SVM algorithm and a label-set typo in a corollary; own proof of the support-vector theorem. Align terms with lessons 12 to 14
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 16. Ch. 16 kernel methods

Folder: `16-kernel-methods/`

- [x] 1 prose and sheet (about 10,100 words with code, 12 snippets run by the lead; the failing fence is the build-it stub; no book references, 2026-11-20). Writer's claims to verify: exercise 1's "norm 1, margin 1" looks unattainable with 0/1 substring features (prose uses norm 2); filled gaps in the kernel characterization proof. Align recaps with lessons 12 to 15
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 17. Ch. 17 multiclass, ranking, structured output

Folder: `17-multiclass-ranking/`

- [x] 1 prose and sheet (about 24,500 words with code, 65 snippets run by the lead; the failing fences are build-it stubs; no book references; written with the formula walkthrough and concept-before-symbol rules; 2026-11-20). Too long for one sitting: the writer recommends splitting into 17a (multiclass, structured output) and 17b (ranking) at "Ranking the inbox"; ask the user. Writer's claims to verify: the zero-one formula (b+d) typo, the printed λ in the SVM corollary, the Kendall rewrite holding only as ≤
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 18. Ch. 18 decision trees

Folder: `18-decision-trees/`

- [x] 1 prose and sheet (about 9,700 words with code, 10 snippets run by the lead; the 2 failing fences are build-it stubs, by design; no book references, 2026-11-20). Writer's claims to verify: the book's threshold indexing is off by one, and its ID3 pseudocode has a check that never triggers; opening still to align with lesson 17's cliffhanger
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 19. Ch. 19 nearest neighbor

Folder: `19-nearest-neighbor/`

- [x] 1 prose and sheet (about 9,050 words with code, 13 snippets run by the lead; the failing fence is the build-it stub; no book references, 2026-11-20). Writer's claims to verify: the 1-NN proof's parameter choice made rigorous; a possible book error (exercise 4 gives 2k, not k, unless m^{1/(d+1)}/2 is whole); the weighted-average formula looks like a typo. Align opening with lesson 18's cliffhanger
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred (CPU rule)
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 20. Ch. 20 neural networks

Folder: `20-neural-networks/`

- [x] 1 prose and sheet (about 13,000 words with code, 20 snippets run by the lead; the failing fence is the build-it stub; no book references; written with the walkthrough and concept-before-symbol rules; 2026-11-20). Long: the sheet lists what could move to a 20b. Writer's claims to verify: a sharper exponential-size bound than the book's; own explicit VC constant; constant neurons output 1
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 21. Ch. 21 online learning

Folder: `21-online-learning/`

- [x] 1 prose and sheet (about 18,200 words with code, 25 snippets run by the lead; the failing fence is the build-it stub; no book references; written with the walkthrough and concept-before-symbol rules; 2026-11-20). Split candidate. Writer's notes to verify: a mismatch between the book's prose and its tree definition; the regret lower bound stated without proof; own recursion for the Littlestone dimension
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 22. Ch. 22 clustering

Folder: `22-clustering/`

- [x] 1 prose and sheet (about 13,400 words with code, 13 snippets run by the lead; the failing fences are build-it stubs; no book references; written with the walkthrough and concept-before-symbol rules; 2026-11-20). Writer's claims to verify: the book contradicts itself on whether k-means satisfies consistency (prose keeps 'fails', softened; no small counterexample found, the lab must not claim one); own termination proof and tie rules. Align opening with lesson 21's cliffhanger
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 23. Ch. 23 dimensionality reduction

Folder: `23-dimensionality-reduction/`

- [x] 1 prose and sheet (about 17,400 words with code, 31 snippets run by the lead; the 3 failing fences are build-it stubs; no book references; written with both rules; 2026-11-20). Split candidate. Writer's claims to verify: the chi-squared upper tail is false for large ε (lesson proves ε ≤ 1/2); RIP theorem constant 216 instead of the book's 100; covering lemma by packing. Page builder must ship the act-2 table (0.243 at n = 500) or update the prose
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 24. Ch. 24 generative models

Folder: `24-generative-models/`

- [x] 1 prose and sheet (about 18,300 words with code, 17 snippets run by the lead; the 3 failing fences are build-it stubs; no book references; written with the walkthrough and concept-before-symbol rules; 2026-11-20). Split candidate: 24a (likelihood, pseudo-counts, naive Bayes) and 24b (LDA, EM). Writer's claims to verify: five book slips (Bayesian prior not integrating to 1, a variance typo, strict vs non-strict inequality, an inequality that is an equality, row vs column form of w)
- [ ] 2 gates: writer's PDF self-check, scripts and greps done; book-check agents, read-aloud and fresh reader deferred
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 25. Ch. 25 feature selection and generation

Folder: `25-feature-selection/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 26. Ch. 26 Rademacher complexities

Folder: `26-rademacher-complexity/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 27. Ch. 27 covering numbers

Folder: `27-covering-numbers/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 28. Ch. 28 proof of the fundamental theorem

Folder: `28-fundamental-theorem-proof/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 29. Ch. 29 multiclass learnability

Folder: `29-multiclass-learnability/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 30. Ch. 30 compression bounds

Folder: `30-compression-bounds/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

### 31. Ch. 31 PAC-Bayes

Folder: `31-pac-bayes/`

- [ ] 1 prose and sheet (drafting by agent, started 2026-11-20; brief in the session scratchpad, template docs/writer-brief.md)
- [ ] 2 gates: book check (PDF agents), read-aloud, fresh reader, greps
- [ ] 3 review stop: user approved the prose
- [ ] 4 figures
- [ ] 5 page
- [ ] 6 learner judge: ready
- [ ] 7 verification
- [ ] 8 delivery: series.md, index.html
- [ ] 9 concept check: mastery confirmed
- [ ] 10 end-of-chapter problems done

Appendices A (technical lemmas), B (measure concentration) and C (linear algebra) are not lessons of their own: each lesson cites and proves what it needs from them, the first time it needs it.

Lesson numbers match chapter numbers. A chapter too big for one lesson splits into `NNa-`, `NNb-` folders; record the split here.

## Session handoff (read this after any context reset)

Everything the user gave this series, so no session loses it:

- **Skills to load:** `klay-learning-ladder` (this series' method; skill folder `~/.claude/skills/klay-learning-ladder`, symlinked from `~/dotfiles/claude/skills/klay-learning-ladder`, changed this session and not yet committed). `artifact-design` only for the page contract. Never `klay-artifact-style` on lessons.
- **Source:** the book PDF (path in Source rule above), read with `pdftotext`; `book/map.md` (chapter pages, sections, all 122 exercises) and `book/notation.md`. Lessons never cite the book.
- **Visual tools, all cloned in `~/.local/share/klay-learning-ladder/` and pinned in the skill's `scripts/ensure.sh`:** diagram-design, stack-diagram (caru-ini), iwrzwr visual archive, three.js, impeccable (pbakaus; layout, typography and critique playbooks, `skill/scripts/impeccable detect`), anatomy, img2threejs.
- **Look:** `tokens.css` (single source of colours, fonts, sizes, spacing, layout), documented in `docs/design-tokens.md`. Fonts chosen by the user: Literata (reading) and Rubik (headings and interface), with IBM Plex Mono for code. Warm off-white palette, blue input, amber accent and wrong values, green success only. Avoid anything that looks like Claude's own interface.
- **Writing rules:** natural teacher voice, easy to read, no robotic style; every term defined first; rigorous proofs; Python snippets beside hard math (the user learns math through code), run and pasted; code lines at most 66 characters; `\varepsilon` for epsilon; no em dashes; curly quotes; no book references in anything the learner sees.
- **Process rules:** tick boxes here with dates as work is done; spawn writers freely; ask before judges, browsers or agent-voice (CPU); the user answers concept checks in chat; no videos in this series; commit only when the user asks.
- **Checks:** `python3 -I _kit/check_prose.py <lesson-folder>` (snippets run, book references, long code lines, em dashes, `\epsilon`), `node ~/.claude/skills/klay-learning-ladder/scripts/fit.mjs <page>` (no sideways scroll), the impeccable detector.
- **Writer brief:** `docs/writer-brief.md` with `{N}`, `{PREV}`, `{CHAPTER}`, `{STORY}`, `{FOLDER}` to fill; each lesson's story is fixed by the previous lesson's cliffhanger.
- **Memories saved:** lessons from the PDF, not memory; writers yes, judges no; the user learns math through Python.

## Deferred (do later)

Skipped on purpose to keep writing and save CPU (user, 2026-11-20: "keep writing, tick so we remember doing it later"). Tick each when done.

- [ ] Lesson 1: the learner's concept-check answers (step 9), then Chapter 1's problems come with lesson 2 (step 10)
- [ ] Lessons 2 to 19: read-aloud pass (`scripts/readaloud.py`), ask before running (CPU)
- [ ] Lessons 2 to 19: fresh-reader judge on the prose, ask before running (CPU)
- [ ] Lessons 2 to 19: book check by parallel agents per section range (writers self-checked against the PDF)
- [ ] Lessons 2 to 19: the user's review stop (step 3) before figures and pages
- [ ] Lessons 2 to 19: figures (step 4) and pages (step 5), then the impeccable pass and `fit.mjs`
- [ ] Lessons 2 to 19: learner judges (step 6), ask before running (CPU)
- [ ] Lesson 3: align key words with lesson 2 (the lesson 3 writer could not see lesson 2's sheet); lesson 3's Bayes-optimality proof is the writer's own, verify it
- [ ] Lesson 6: decide whether to split into 06a/06b (about 10,400 words), and verify the claimed book error in the uniform-convergence proof's Hoeffding constant
- [ ] Lessons 2 to 19 (and any draft written before 2026-11-20 evening): formula walkthrough pass, every formula read aloud as in lesson 1's risks (the learner's rule above); revision writers, then rerun `_kit/check_prose.py`. Lesson 1 done 2026-11-20; lessons 2 to 12, 14 and 15 started (2026-11-20, brief in the session scratchpad as brief-walkthrough.md, a copy of the rule is in docs/writer-brief.md); lessons 5 to 19 next. Writers for 13 and 17, 20 to 25 were told the rule mid-draft
- [ ] Lessons 1 to 31: clean-context reviewer (step 1b, `docs/reviewer-brief.md`) after each lesson's writer and walkthrough pass. Lesson 4 done 2026-11-20 (walkthrough 7,479 to 12,396; review: 11 findings fixed incl. a wrong 9% that is 6.6% (float error), verdict ready). Lesson 11 done 2026-11-20 (review: 12 findings fixed incl. two overstated claims, verdict ready). Lesson 22 reviewed 2026-11-20 (9 findings fixed incl. a false "no ties" note in the k-means trace, verdict ready). Lesson 8 done 2026-11-20 (review: 17 findings fixed incl. a one-to-one assumption missing from the trapdoor proof, verdict ready). Lesson 5 done 2026-11-20 (review: 9 findings fixed incl. two false claims about the ladder table, verdict ready). Lesson 20 reviewed 2026-11-20 (10 findings fixed, verdict ready). Walkthrough passes done: 10 (10,918 to 15,924 words, 11 snippets), 9 (12,775 to 20,382 words, 22 snippets; split candidate), 11 (8,580 to 11,380 words, 7 snippets), 6 (10,439 to 17,275 words, 18 snippets), 7 (9,581 to 15,765 words, 19 snippets), 5 (8,233 to 13,021 words, 11 snippets) and 8 (10,346 to 14,624 words, 15 snippets); their reviews running. Lesson 2 done 2026-11-20 (walkthrough 8,156 to 13,102 words; review: 12 findings fixed, repeats cut to 12,790 words, verdict ready). Lesson 3 done 2026-11-20 (walkthrough pass 7,595 to 11,140 words; review: 6 findings fixed incl. the realizability definition, verdict ready). Lesson 1 done 2026-11-20 (13 findings fixed: 6 used before explained, 3 formulas without walkthrough, 3 proof gaps, 1 formula before intuition; page rebuilt, fit OK)
- [ ] Lessons 20 to 31: prose drafts (12 to 19 drafting) (writers; the brief template is `docs/writer-brief.md`)
- [x] Commit the series repo and publish it: github.com/klaygomes/uml-lessons, Pages at http://www.estacouveflor.com/uml-lessons/ (user asked, 2026-11-20). Push with the klaygomes gh account (`gh auth switch -u klaygomes`), then switch back to cleiton-loiola
- [ ] Commit the skill changes in ~/dotfiles (when the user asks)

## Open items

- Paused 2026-11-20 by the usage limit, resumed after reset in waves (load rule). Stopped mid-task and resumable by message: walkthrough passes 10 and 11; writers 20 to 27 (20, 22, 24 had partial prose on disk). Resumed first: reviews 2 and 4, walkthroughs 5 to 9. If a later session finds these unfinished, restart each from its brief (scratchpad briefs may be gone: rebuild from docs/writer-brief.md, docs/reviewer-brief.md and the walkthrough rule in the learner's rules).

- Load rule (user, 2026-11-20: "you can verify cpu load and manage as you go"): before spawning agents run `python3 -I _kit/load.py`; spawn only on OK (1-minute load under two thirds of the cores and at least 30% memory free), otherwise wait for a running agent to finish. Judges and reviewers are allowed under this rule.

- CPU budget (user, 2026-11-20): spawn writers freely, but no judge agents (learner, diagram or fresh-reader) and no agent-voice read-aloud without asking; they use too much CPU. Learner judge 3 on lesson 1 was already running when this was said.

- Lesson 1: prose and sheet done, all gates run (fresh reader, read-aloud, three PDF book checks, key-word and em-dash greps); findings applied. Approved 2026-11-20, kept as one lesson (about 6,000 words). Figures and page being built by two builder agents in parallel; then `build.py`, learner judge.
- `book/notation.md` and `book/map.md` done. Chapters 26, 27, 28 and 30 have no exercises in the book; for those lessons, step 10 uses problems written for the lesson, marked as such.
- Running in parallel (2026-11-20): lesson 1 figures builder, lesson 1 page builder, lesson 2 prose drafter, lesson 3 prose drafter. Lessons 2 and 3 are prepared ahead; the learner still sees lesson 2 only after lesson 1's concept check.
