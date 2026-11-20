# Learning Needs a Bias

Objective: after this, you can explain why a learner with no restriction on its hypotheses cannot say anything about unseen messages (and prove it by counting), define an inductive bias as a hypothesis class fixed in advance, compute how fast coincidences die in a small finite class, and place a learning problem in the book's taxonomy (supervised or not, active or passive, kind of teacher, online or batch).

Book: Chapter 1, Sections 1.1 (What is learning: bait shyness, pigeon superstition, inductive bias), 1.2 (when do we need machine learning: complexity, adaptivity), 1.3 (types of learning), 1.4 (relations to other fields). Sections 1.5 and 1.6 open lesson 2. The book's Chapter 1 has no exercises; end-of-chapter problems for Chapter 1 come with lesson 2, after Section 1.6.

Bug: version one of the filter is a memorizer. It is right on 100% of the messages users labelled and right about half the time on unseen ones.
Fix: act 2 of the lab: the spammers' junk feature changes every round. The learner chooses a hypothesis class (anything goes, one feature out of ten, one feature out of the first three, one or two features) and a number of labelled examples within a budget, and reaches true risk under 1%.

Key words: message (not "email", "mail item"; "mail room" is a place name), feature (not "clue", "attribute"), junk / keep (not "spam" / "ham"; "spam" appears once, in the adaptivity section's general claims), example (not "data point", "sample" as a noun for one item), training sequence (not "training set", "dataset"), hypothesis (not "rule" after its definition, not "model", "predictor", "classifier"; the book's synonyms named once), learner (not "algorithm" when meaning the learner, not "model"), hypothesis class (not "model class", "family"), consistent (not "fits" in formal statements; "fits" allowed in plain prose before the definition), empirical risk (not "training error"), true risk (not "test error", "generalization error"), inductive bias (not "prior" as a noun), memorizer, consistent-feature, survive (a hypothesis survives an example when it agrees with it), impostor (a wrong hypothesis that is still consistent).

Terms:
- feature: one yes/no property of a message (F, L, U, N in the tiny world).
- domain set \(\mathcal X\): all possible messages; \(\{0,1\}^4\), 16 messages.
- example: a message and its label, \(z = (x, y)\).
- training sequence: the \(m\) examples the learner sees, \(S = (z_1, \dots, z_m)\).
- hypothesis: a function \(h: \mathcal X \to \mathcal Y\) labelling every message.
- learner: an algorithm from training sequences to hypotheses.
- empirical risk \(L_S(h)\): fraction of the examples \(h\) gets wrong.
- true risk \(L_{\mathcal D}(h)\): probability that \(h\) is wrong on a fresh message drawn from \(\mathcal D\).
- consistent: \(L_S(h) = 0\).
- hypothesis class \(\mathcal H\): the set of hypotheses a learner may return, fixed before seeing data.
- inductive bias: prior knowledge a learner brings, built in by choosing \(\mathcal H\).
- i.i.d.: each message drawn from the same \(\mathcal D\), independently of the others.
- supervised / unsupervised / reinforcement; passive / active; indifferent / helpful / adversarial teacher; batch / online.

Misconception: "consistent on the training data means it works". Why wrong: by Proposition 1, for each unseen message exactly half of the consistent hypotheses are wrong; the memorizer is consistent with true risk 0.25.
Two failure modes, named in the prose after the survivor table: the pigeon's (class too large for the examples, a coincidence wins; cured by more examples or a smaller class) and the rat's (class misses the truth; no number of examples helps). Chapter 5's estimation and approximation errors.
Second misconception: "a bias is a flaw to remove". Why wrong: without one the examples cannot separate the consistent hypotheses; the danger is a wrong bias, not a bias.

Worked examples (all computed by `check.py` in the session scratchpad; reproduce with the build-it challenge):
1. Tiny world, \(f(x) = x_F\), training sequence 1111 j, 0000 k, 1110 j, 1011 j, 0001 k, 0100 k, 1101 j, 0010 k.
   - Unseen: 0011, 0101, 0110, 0111, 1000, 1001, 1010, 1100. The memorizer (default keep) is wrong on 1000, 1001, 1010, 1100: 4 of 8 unseen, \(L_S = 0\), \(L_{\mathcal D} = 4/16 = 0.25\).
   - All hypotheses: \(2^{16} = 65{,}536\); consistent after 8 examples: \(2^8 = 256\); for each unseen message exactly 128 say junk (exhaustive check). Mean number of unseen errors of a consistent hypothesis: 4.0; distribution 1, 8, 28, 56, 70, 56, 28, 8, 1.
2. One-feature class \(\mathcal H_1 = \{h_F, h_L, h_U, h_N\}\), same sequence, survivors after each example: FLUN, FLUN, FLU, FU, FU, FU, F, F. All-hypothesis survivors: 32,768, 16,384, 8,192, 4,096, 2,048, 1,024, 512, 256. \(L_{\mathcal D}(h_L) = L_{\mathcal D}(h_U) = L_{\mathcal D}(h_N) = 8/16\).

Derivation:
- Proposition 1: \(2^{N-k}\) consistent hypotheses for \(k\) distinct labelled messages; the flip map at an unseen \(x'\) is an involution on the consistent set swapping the two label groups. Seed of the no-free-lunch theorem (Ch. 5).
- Proposition 2: in the tiny world with \(S \sim \mathcal D^m\), \(\mathbb P[h_j \text{ survives}] = 2^{-m}\), \(\mathbb P[\text{some impostor survives}] = 1 - (1 - 2^{-m})^3 \le 3 \cdot 2^{-m}\), via independence of the agreement bits and \((1-a)^3 \ge 1 - 3a\). Exact values: m = 6: 0.0461; m = 8: 0.01167, bound 0.01172. Simulation (200,000 trials): 0.0457 and 0.0118. Preview of the finite-class bound of Ch. 2 (union bound), which this lesson names but does not prove.

Algorithms: Memorize (hash map: build \(O(md)\) time and space, predict \(O(d)\) expected) and ConsistentFeature (keeps the survivors, returns one uniformly at random, since the learner cannot know which feature is real; \(O(md)\) time, \(O(d)\) space; brute-force ERM over finite \(\mathcal H\) is \(O(m|\mathcal H|)\) evaluations). Build-it challenge: `memorize`, `consistent_feature`, `true_risk` in standard-library Python; reproduce the tables; simulate survival for d = 4 and d = 10.

Figures:
- [[diagram:messages]] (still): the 16 messages as a 4 by 4 grid (rows by the F and L bits, columns by U and N), each cell showing its four bits. Junk cells (F = 1) hatched or marked with a filled square, keep cells plain; the eight training messages outlined solid with their order number, the eight unseen ones dashed. The four unseen junk cells (1000, 1001, 1010, 1100) carry a small cross for "the memorizer says keep: wrong". Reads in grayscale.
- [[diagram:flip]] (still): the flip map \(\phi\) at one unseen message. Two columns of consistent hypotheses drawn as rows of 16 cells (eight fixed, shared, shown faint; eight free); left column \(h(1000) = 1\), right column \(h(1000) = 0\); paired rows joined by a two-headed arrow labelled "\(\phi\): flip the label at 1000, nothing else". Under each column "128". Shows three pairs and an ellipsis.
- [[diagram:funnel]] (animates): two columns from the same top box, "all hypotheses: 65,536". Left column, "anything goes": eight halving steps, one per example, the count dropping 65,536 to 32,768 to ... to 256, ending in a box "256 consistent; for each unseen message, 128 say junk, 128 say keep" with a balanced tick-free scale. Right column, "bias first: one-feature hypotheses": one big cut to 4 before any example, then the examples knock out N (example 3), L (example 4), U (example 7), ending at 1, "\(h_F\), true risk 0". Each column steps in sync with the examples. Still frame: both columns complete, the halving counts listed. Tool: diagram-design, custom figure on its primitives (a two-lane funnel is not one of its standard types; the lanes must line up example by example).
- [[diagram:taxonomy]] (still): the book's four dimensions (§1.3, pp. 22 to 24) as four rows, each with its settings and the book's example under each: supervised (spam filter) / reinforcement (chess, between) / unsupervised (anomaly detection, clustering); passive (spam filter) / active (learner picks or composes messages); helpful (child, student) / indifferent nature (scientist; "statistical") / adversarial (spam, fraud; also the worst case); batch (data miner) / online (stockbroker). The book's main focus (supervised, statistical, batch, passive) is tinted in each row; pointers: online Ch. 21, clustering Ch. 22. Tool: diagram-design, a matrix or a row of labelled toggles drawn as static labels (never control-looking).
- No anatomy figure: no mechanism here is better shown as a machine.

Lab:
- World: \(d = 10\) features, 1,024 messages, \(\mathcal D\) uniform, a hidden junk feature \(j^\*\), \(f(x) = x_{j^\*}\). Training sequences drawn i.i.d. with a seeded generator.
- [[lab]] act 1, explore. Prediction first, input locked until answered: "The memorizer has seen 100 examples and is consistent with all of them. About what fraction of fresh messages will it get wrong?" Options: about 0%, about 10%, about 45%, about 90%. Answer: about 45% (expected distinct messages seen 95.3, true risk 0.5 × (1 − 95.3/1024) = 0.453). Then a slider for \(m\) from 1 to 200, with readouts per learner: number of consistent hypotheses in its class (one-feature: 1 to 10; anything goes: \(2^{1024 - k}\), shown as a power of 2), the hypothesis it returns, its true risk computed exactly over all 1,024 messages. Regions for the readout: m ≤ 3, several impostors usually alive; m around 6 to 10, an impostor sometimes alive (exact chance 13.2% at m = 6, 0.88% at m = 10); m large, one-feature learner at 0 while the memorizer stays near 0.5 (0.495 at 10, 0.476 at 50, 0.411 at 200). "Resample" redraws S at the same m, so the reader sees the impostor appear and vanish. "Pin this" freezes one run, dashed in --input.
- [[lab-act2]] act 2, fix the bug. Story: the spammers changed feature; each "Try another" draws a new \(j^\*\) from the ten, sometimes one of features 4 to 10. Controls: class menu (anything goes / one feature of ten / one feature of the first three / one or two features: \(h_j\) and \(h_j \wedge h_k\), 55 hypotheses) and a labels slider, budget 12. Prediction before the first try (an outcome, not the fix): "You pick one feature of the first three, and this week's junk feature is feature 7. What will the learner return?" Options: the truth; an impostor; no consistent hypothesis; the memorizer's answer. Answer: no consistent hypothesis, once enough examples arrive. Feedback names the mistake without giving the answer:
  - anything goes: "Consistent with all your examples, wrong on X% of fresh messages: half the consistent hypotheses say junk for every message it has not seen."
  - narrow class, truth outside: "No hypothesis in this class is consistent with the examples: the real junk feature is not one this class can see."
  - right class, too few labels, impostor picked: "k hypotheses are consistent, and the one returned is wrong on X% of fresh messages."
  - success: true risk under 1%, within budget: the inbox empties of junk.
  The default setting (anything goes, 5 labels) must not pass. Exact act-2 tuning (budget, the 55-class pair risks) is checked by script at page time.

Concept check, answers (never shown to the learner; critique against these):
1. A world where sickness is caused by something the rat's class excludes (a sound, a light, a place, or a delayed shock). The rat's class contains no hypothesis close to the truth, so it either blames the food wrongly or learns nothing; a learner with no bias would at least memorize. This is the "wrong bias" failure, a class that does not contain the truth: approximation error, formalized in Ch. 5. Contrast with the pigeon's failure: a class so large that coincidences survive (estimation error).
2. \(\phi\) changes only the label of \(x'\), which is not in \(S\), so every example's label is untouched and \(L_S\) stays 0. \(\phi \circ \phi = \mathrm{id}\), so \(\phi\) is its own inverse, hence a bijection; it maps \(\{h(x') = 1\}\) into \(\{h(x') = 0\}\) and back. If \(S\) contains one message with both labels, no function is consistent: the count is 0 (the proposition assumes each message has one label; realizability fails). Look for the learner saying "involution" or showing injectivity and surjectivity explicitly.
3. Each wrong one-feature hypothesis has \(L_{\mathcal D} = 1/2\), so survives with probability \(2^{-m}\); the 19 agreement bits per message are independent, so \(\mathbb P = 1 - (1 - 2^{-m})^{19} \le 19 \cdot 2^{-m}\). Need \(2^m \ge 19{,}000\): \(m = 15\) (\(2^{14} = 16{,}384\) is too small; bound at 15 is 0.00058). The memorizer has seen at most 15 of \(2^{20} = 1{,}048{,}576\) messages, about 0.0014%. Accept the union-bound-style argument without independence for the inequality.

Next bug: the real system has 30 features and labels are expensive (a few hundred). Which hypothesis classes can you afford, and how sure can you be that a consistent hypothesis has small true risk? Chapter 2: ERM, overfitting, finite classes, the union bound.

Act 2 guarantee (computed exactly over 1,024 messages): one of ten needs 10 labels (0.88%); the 55 class has 9 pair impostors with true risk 1/4 and 45 impostors with true risk 1/2, so the union bound 9(3/4)^m + 45·2^-m is 0.5508 at m = 10, 0.0286 at m = 20 and 0.0090 at m = 24; with a budget of 12 only one of ten (needs 10) can be certified; the narrow first-three class is never awarded a fix, because it cannot be guaranteed to contain next week's feature.

Assumptions:
- Every claim about the book was checked against the PDF (pp. 19 to 29, and 33 to 38 for Chapter 2 notation) by three parallel agents on 2026-11-20. Fixes applied: the book's nausea cue is a sound, not a light; the pigeon behaviours are the book's (pecking, turning the head); the book first compares memorization with the rat; the rat's blindness to sounds is its correct bias, wrong only in a hypothetical world; §1.2 to 1.4 rewritten to the book's lists (the four 1.3 dimensions with the book's examples, the 1.4 fields and its four contrasts with statistics); true risk written \(L_{(\mathcal D, f)}\), shortened to \(L_{\mathcal D}\).
- Not in the book, kept as true background: none. Removed: "light or click", the light-shock pairing, first names of Garcia and Koelling, the 1948 date, "active learners need fewer labels".
- The book's bibliography dates Garcia and Koelling 1996 (an anthology reprint of the 1966 paper); the lesson gives no year.
- \(\mathcal D\) is uniform in the tiny world so true risk is a count out of 16; the prose says Chapter 2 drops this.
- The book writes \(L_{(\mathcal D, f)}(h)\) for the realizable setting in Chapter 2; this lesson writes \(L_{\mathcal D}(h)\) with \(f\) fixed, to match the syllabus notation.
- Proposition 2 is specific to the uniform tiny world, where exact independence holds; the general finite-class bound is left to lesson 2.
- Learner-facing pages ask the concept check as open questions answered in chat, not as auto-graded options.
