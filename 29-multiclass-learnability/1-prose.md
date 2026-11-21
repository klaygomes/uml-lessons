# Multiclass Learnability

## Three hundred folders and one hungry folder

The auditor signed, and the product team got its new plan. The filter no longer answers junk or keep: it files every message into one of 300 folders, invoices, newsletters, travel, receipts, one for each team’s project, and junk. The learner behind it is lesson 17a’s four-folder filter grown up, a linear multiclass predictor with one weight vector per folder, trained by ERM on messages described by 30 features. It works. And then the launch review asks the question every launch since lesson 2 has had to answer: how many labelled messages does this need, and how sure can you be?

You reach for the fundamental theorem that lesson 28 has just finished proving: the labels a class needs grow with its VC dimension. But open the definition of the VC dimension and look at what it measures: how many points a class can label *in every possible way with two labels*. Junk or keep. Shattering asks for all \(2^{|C|}\) yes/no labellings of a set \(C\) of \(|C|\) messages. Your hypotheses don’t answer yes or no; they answer “folder 17” or “folder 243”. The VC dimension simply isn’t defined for them, and the review is stuck.

There is a second, stranger problem. Last week the team shipped **auto-folders**: a user can gather a group of senders, and all their mail goes into a folder named after that group. Mail from everyone else stays in the inbox. Right now no user has made a folder, so the true rule is “everything stays in the inbox”. The learner is ERM over all possible auto-folder rules, and its training error is zero. Yet users are furious: mail from people they have never written to before is vanishing into a mysterious folder.

Here is what happened. All 60 labelled messages said “inbox”, and many rules fit that. The team’s ERM picked the one that makes a folder out of *every sender it has not seen*. It is consistent: every labelled message really did come from a sender it has seen, and those stay in the inbox. On fresh mail it is a disaster. In the simulation below, 90% of the company’s mail comes from one sender (the all-staff announcements) and the rest is spread over 999 other senders:

```python
import random

rng, n, m = random.Random(0), 1000, 60
p = [0.9] + [0.1 / (n - 1)] * (n - 1)
senders = rng.choices(range(n), weights=p, k=m)
seen = set(senders)
L_bad = sum(p[x] for x in range(n) if x not in seen)
L_good = 0.0
print(len(seen), round(L_bad, 4), L_good)
```

It prints `9 0.0992 0.0`. The 60 labels came from only 9 distinct senders, so the team’s rule hides the mail of the other 991, about 9.9% of all mail, while the rule “everything in the inbox”, which is just as consistent, is perfect. Both are ERM. One learns, one doesn’t.

So you have two bugs. You cannot say how many labels a 300-folder rule needs, because the tool you would use only speaks in two labels. And you have an ERM over a class of folder rules that learns badly, even though lesson 6a promised that, for two labels, *every* ERM learns whenever anything does. This lesson fixes both.

## What carries over from two labels

Let’s set the stage with the words we already have. A **multiclass** problem is one where each label is one of \(k\) values. We write the label set as \([k] = \{1, 2, \dots, k\}\), the numbers from 1 to \(k\), so with 300 folders a label is a number from 1 to 300. A hypothesis is still a function \(h\) from the domain \(\mathcal X\) (all possible messages) to the labels, now written \(h: \mathcal X \to [k]\), and a hypothesis class \(\mathcal H\) is a set of them. We write \([k]^{\mathcal X}\) for the set of *all* functions from \(\mathcal X\) to \([k]\), so “\(\mathcal H \subseteq [k]^{\mathcal X}\)” just says that every member of \(\mathcal H\) is such a function.

The loss is the **0-1 loss** of lesson 3: a hypothesis pays 1 for a wrong folder and 0 for the right one, whichever wrong folder it picks. So the empirical risk \(L_S(h)\) is still the fraction of the training sequence \(S\) that \(h\) files wrongly, and the true risk \(L_{\mathcal D}(h)\) is the chance that it files the next message wrongly, when messages and their folders are drawn from the distribution \(\mathcal D\). Recall also from lesson 2 that **ERM** (empirical risk minimization) is any learner that returns a hypothesis in \(\mathcal H\) with the smallest empirical risk, and that when several hypotheses tie, which one it returns is up to the particular ERM. That freedom will matter a great deal at the end of this lesson.

The definitions of lesson 3 carry over word for word, because they never mentioned how many labels there are. A class is **PAC learnable** when some learner, given \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples labelled by some hypothesis in the class (the realizable case), returns a hypothesis with true risk at most \(\varepsilon\) with probability at least \(1 - \delta\). Here \(\varepsilon\) is the accuracy you tolerate, \(\delta\) the chance of failure you tolerate, \(m\) the number of examples, and \(m_{\mathcal H}\) the **sample complexity**, the smallest number of examples that suffices. It is **agnostic PAC learnable** when, with no assumption that any hypothesis is perfect, some learner gets within \(\varepsilon\) of the best hypothesis in the class. And it has the **uniform convergence property** (lesson 4) when enough examples make every hypothesis’s empirical risk close to its true risk at once; lesson 4 showed that this makes every ERM a successful learner.

What does *not* carry over is the yardstick. Lesson 6a measured a class by its **restriction** to a finite set \(C = \{c_1, \dots, c_r\}\), the set of label vectors the class produces there,

\[ \mathcal H_C = \{ (h(c_1), \dots, h(c_r)) : h \in \mathcal H \}, \]

which now lives inside \([k]^r\), the vectors of \(r\) folder numbers. Read it as before: one vector of answers per hypothesis, collected in braces, duplicates kept once. If two hypotheses file two messages as (1, 2) and a third as (1, 3), then \(\mathcal H_C = \{(1, 2), (1, 3)\}\); the first snippet of the next section computes one. For two labels, \(\mathcal H\) **shatters** \(C\) when \(\mathcal H_C\) contains all \(2^r\) vectors, and the **VC dimension** is the size of the largest shattered set.

Lesson 6a stated the **fundamental theorem of statistical learning**, and lesson 28 proved it with explicit constants. For two labels, a class with VC dimension \(d\) is learnable exactly when \(d\) is finite, and any ERM learns it. With noisy labels, no learner can do with fewer than \((d + \log(1/\delta))/(1024\,\varepsilon^2)\) labels (for \(\varepsilon \le 1/(8\sqrt2)\) and \(\delta \le 1/8\)), and ERM succeeds with order \((d\log(1/\varepsilon) + \log(1/\delta))/\varepsilon^2\), the theorem’s ceiling up to that extra \(\log(1/\varepsilon)\). With clean labels, ERM succeeds with order \((d\log(1/\varepsilon) + \log(1/\delta))/\varepsilon\). The engine of the ceilings was **Sauer’s lemma**: a class of VC dimension \(d\) produces at most \(\sum_{i=0}^{d}\binom{r}{i}\) labellings on \(r\) points, at most \((er/d)^d\) once \(r \ge d\), a polynomial in \(r\) rather than \(2^r\).

So the question is: what should “shatter” mean when there are 300 labels?

## Two choices per message

Let’s work it out on a class small enough to hold in your head. Suppose the scoring service from lesson 6a now returns an **urgency score** between 0 and 1, and the filter files each message into one of \(k = 3\) **urgency folders**: 1 (low), 2 (normal), 3 (urgent). A rule picks two cut points, \(t_1 \le t_2\), and files a message as low if its score is at most \(t_1\), normal if it is above \(t_1\) but at most \(t_2\), and urgent if it is above \(t_2\). Call these rules **urgency levels**. With cut points 0.5 and 1.0, a message scoring 0.3 is low and one scoring 0.7 is normal.

Now take two messages, scoring 0.3 and 0.7. A first idea would be to ask for all \(3^2 = 9\) ways to file them, by analogy with \(2^2\). But urgency levels can never file 0.3 as urgent and 0.7 as low, because a higher score never gets a lower folder. Asking for all \(k^{|C|}\) labellings is so demanding that hardly any interesting class would shatter anything, and the measure would say nothing.

Here is a gentler question, and it is the right one. For each message, *you* pick two candidate folders. Then ask: can the class produce every mixture, every way of choosing one of the two candidates for each message? For 0.3, pick low or normal; for 0.7, pick normal or urgent. That is four mixtures, the two-label question in disguise: on each message there is a binary choice, “first candidate or second”. Here are the four, tried against urgency levels (in the code, a 1 in `B` means that message takes its first candidate):

```python
from itertools import product

k, C = 3, [0.3, 0.7]
f0, f1 = (1, 2), (2, 3)

def levels(t):
    return lambda x: 1 + sum(x > tj for tj in t)

cuts = [0.0, 0.5, 1.0]
H = [levels(t) for t in product(cuts, repeat=k - 1)
     if list(t) == sorted(t)]
H_C = {tuple(h(c) for c in C) for h in H}
for B in product([0, 1], repeat=len(C)):
    mix = tuple(f0[i] if B[i] else f1[i] for i in range(len(C)))
    print(B, mix, mix in H_C)
```

It prints

```
(0, 0) (2, 3) True
(0, 1) (2, 2) True
(1, 0) (1, 3) True
(1, 1) (1, 2) True
```

All four mixtures are achieved. In the new sense, urgency levels shatter the two messages. Notice that the choice of candidates was part of the work: had we offered “low or urgent” for 0.7, the mixture “normal, low” would have been demanded, and no urgency level files a higher score lower.

[[diagram:two-choices]]

What about three messages? Try scores 0.2, 0.5 and 0.8 with any candidates you like, and you will always find a mixture the class cannot produce. Here is why, in one line of reasoning. Call the two candidates of message \(i\) its lower one, \(\mathrm{lo}_i\), and its upper one, \(\mathrm{hi}_i\), so \(\mathrm{lo}_i < \mathrm{hi}_i\). The mixture that takes the upper candidate for message 1 and the lower one for message 2 must be achievable, and urgency levels never file a higher score lower, so \(\mathrm{hi}_1 \le \mathrm{lo}_2\). The same goes for messages 2 and 3. So \(\mathrm{lo}_1 < \mathrm{hi}_1 \le \mathrm{lo}_2 < \mathrm{hi}_2 \le \mathrm{lo}_3 < \mathrm{hi}_3\), a chain with three strict steps up from \(\mathrm{lo}_1 \ge 1\), which forces \(\mathrm{hi}_3 \ge 4\). There is no folder 4. With three folders, two messages is the most this class can shatter.

## The Natarajan dimension

Now let’s write the idea down precisely. Each message \(x\) of the set \(C\) gets two candidate folders, which we collect into two functions: \(f_0(x)\), the first candidate, and \(f_1(x)\), the second. A mixture is described by the set \(B\) of messages that take their first candidate; the rest take their second.

**Definition (shattering, multiclass).** A finite set \(C \subseteq \mathcal X\) is **shattered** by a class \(\mathcal H \subseteq [k]^{\mathcal X}\) if there exist two functions \(f_0, f_1: C \to [k]\) such that \(f_0(x) \ne f_1(x)\) for every \(x \in C\), and for every subset \(B \subseteq C\) there is a hypothesis \(h \in \mathcal H\) with

\[ h(x) = f_0(x) \text{ for all } x \in B \quad \text{and} \quad h(x) = f_1(x) \text{ for all } x \in C \setminus B. \]

Read it piece by piece.

- \(f_0, f_1: C \to [k]\) are the two lists of candidates: each assigns a folder to every message of \(C\). The condition \(f_0(x) \ne f_1(x)\) says the two candidates for a message are genuinely different folders.
- \(B \subseteq C\) is any subset of \(C\), including the empty set and \(C\) itself; there are \(2^{|C|}\) of them, one per mixture.
- \(C \setminus B\) (“\(C\) minus \(B\)”) is the set of messages of \(C\) that are not in \(B\).
- The display asks for one hypothesis \(h\) that gives the first candidate on every message of \(B\) and the second candidate on every other message of \(C\).

In plain words: **\(\mathcal H\) shatters \(C\) when you can pick two different candidate folders for each message of \(C\) so that the class achieves every way of choosing between them.** In our example, \(C = \{0.3, 0.7\}\), \(f_0 = (\text{low}, \text{normal})\), \(f_1 = (\text{normal}, \text{urgent})\), and \(B = \{0.3\}\) asks for low on 0.3 and urgent on 0.7, which the cut points 0.5 and 0.5 deliver.

**Definition (Natarajan dimension).** The **Natarajan dimension** of \(\mathcal H\), written \(\mathrm{Ndim}(\mathcal H)\), is the largest size of a set \(C \subseteq \mathcal X\) that \(\mathcal H\) shatters in the multiclass sense, and \(\infty\) if there are shattered sets of every finite size. It is named after Natarajan, who introduced it in 1989.

In plain words: **the Natarajan dimension is the most messages on which the class can play every two-way choice, with candidates of your choosing.** For three urgency folders it is 2. Two things are worth saying about how to check it. First, a lower bound \(\mathrm{Ndim}(\mathcal H) \ge r\) needs just one set of size \(r\) and one good choice of candidates; an upper bound needs an argument that defeats every set of size \(r + 1\) and every choice of candidates, as the chain argument did. Second, swapping the two candidates of a message only swaps whether it sits in \(B\), and \(B\) ranges over all subsets anyway, so for each message only the *unordered* pair of candidates matters.

That second remark makes a brute-force checker short. For a finite domain, store each hypothesis as the tuple of its folders, try every set \(C\) and every choice of one unordered pair of folders per message, and test all mixtures against the restriction:

```python
from itertools import combinations, product

def shatters(H, C, k):
    H_C = {tuple(h[x] for x in C) for h in H}
    pairs = list(combinations(range(1, k + 1), 2))
    return any(all(tuple(p[b] for p, b in zip(fs, B)) in H_C
                   for B in product([0, 1], repeat=len(C)))
               for fs in product(pairs, repeat=len(C)))

def ndim(H, X, k):
    sizes = range(len(X) + 1)
    return max(r for r in sizes if any(
        shatters(H, C, k) for C in combinations(X, r)))

def levels(n, k):
    return [h for h in product(range(1, k + 1), repeat=n)
            if list(h) == sorted(h)]

print([ndim(levels(5, k), range(5), k) for k in (2, 3, 4)])
```

It prints `[1, 2, 3]`. Here the domain is five scores in increasing order, `levels(5, k)` lists every way urgency levels can file them (every non-decreasing tuple of folders), `pairs` are the unordered candidate pairs, `B` runs over all mixtures, and `ndim` returns the largest size of a shattered set. With two, three and four urgency folders the Natarajan dimension is 1, 2 and 3. It looks like \(k - 1\), and it is.

**Proposition (urgency levels).** For \(k \ge 2\) urgency folders on scores in \([0, 1]\), \(\mathrm{Ndim} = k - 1\).

Formally, the class is \(\{h_{\mathbf t}\}\) over all cut points \(t_1 \le \dots \le t_{k-1}\), with

\[ h_{\mathbf t}(x) = 1 + \left|\{ j \in [k-1] : x > t_j \}\right|. \]

Inside out: \(\{ j \in [k-1] : x > t_j \}\) is the set of cut points that the score \(x\) is above (the colon reads “such that”); the bars count them; adding 1 turns “above no cut point” into folder 1. In plain words: **a message’s folder is one more than the number of cut points below its score.** With cut points 0.5 and 1.0, the score 0.7 is above one cut point, so it goes to folder 2. The `levels` function of the first snippet is this formula, `1 + sum(x > tj for tj in t)`.

*Proof.* Lower bound: take scores \(0 < c_1 < \dots < c_{k-1}\) and the candidates \(f_0(c_i) = i\), \(f_1(c_i) = i + 1\). Any mixture gives \(c_i\) a folder in \(\{i, i+1\}\), so the folders never go down as the score goes up: message \(i\) gets at most \(i + 1\) and message \(i + 1\) gets at least \(i + 1\). And every non-decreasing filing \(\ell_1 \le \dots \le \ell_r\) of scores \(c_1 < \dots < c_r\) is some urgency level: put the cut point \(t_j\) at the largest score whose folder is at most \(j\), or at 0, below all the scores, if there is none. A score with folder \(\ell_i\) is then above exactly the cut points \(t_1, \dots, t_{\ell_i - 1}\) (for \(j < \ell_i\), the score at \(t_j\) has a smaller folder, so it is lower; for \(j \ge \ell_i\), \(t_j\) is at or above \(c_i\)), and the formula gives \(1 + (\ell_i - 1) = \ell_i\). So a set of \(k - 1\) messages is shattered.

Upper bound: suppose \(r\) scores \(c_1 < \dots < c_r\) are shattered, with candidates \(\mathrm{lo}_i < \mathrm{hi}_i\). For each \(i < r\), the mixture taking \(\mathrm{hi}_i\) on \(c_i\) and \(\mathrm{lo}_{i+1}\) on \(c_{i+1}\) is achieved by some urgency level, which never files a higher score lower, so \(\mathrm{hi}_i \le \mathrm{lo}_{i+1}\). Chaining, \(1 \le \mathrm{lo}_1 < \mathrm{hi}_1 \le \mathrm{lo}_2 < \dots < \mathrm{hi}_r\), which is \(r\) strict steps, so \(\mathrm{hi}_r \ge r + 1\). Since \(\mathrm{hi}_r \le k\), we get \(r \le k - 1\). \(\blacksquare\)

With \(k = 2\) folders, urgency levels are lesson 6a’s thresholds, whose VC dimension is 1. That is no coincidence.

**Proposition (two labels).** If \(k = 2\), then \(\mathrm{Ndim}(\mathcal H) = \mathrm{VCdim}(\mathcal H)\).

*Proof.* With only two labels, \(f_0(x) \ne f_1(x)\) means that \(\{f_0(x), f_1(x)\}\) is the whole label set at every point. So the mixtures are exactly all \(2^{|C|}\) labellings of \(C\), and \(\mathcal H\) shatters \(C\) in the multiclass sense precisely when it produces all of them, which is lesson 6a’s shattering. Conversely, if \(\mathcal H\) shatters \(C\) in lesson 6a’s sense, take \(f_0\) constantly the first label and \(f_1\) constantly the second. The two notions of shattering coincide, so their largest shattered sets do too. \(\blacksquare\)

Python can check this on 300 random classes of two-label functions on four points, every subset \(C\), comparing the new test with lesson 6a’s count \(|\mathcal H_C| = 2^{|C|}\):

```python
import random
from itertools import combinations, product

def shatters(H, C, k):
    H_C = {tuple(h[x] for x in C) for h in H}
    pairs = list(combinations(range(1, k + 1), 2))
    return any(all(tuple(p[b] for p, b in zip(fs, B)) in H_C
                   for B in product([0, 1], repeat=len(C)))
               for fs in product(pairs, repeat=len(C)))

def vc_shatters(H, C):
    return len({tuple(h[x] for x in C) for h in H}) == 2 ** len(C)

rng, every = random.Random(1), list(product([1, 2], repeat=4))
sets = [C for r in range(5) for C in combinations(range(4), r)]
classes = [rng.sample(every, rng.randint(1, 16))
           for _ in range(300)]
print(all(shatters(H, C, 2) == vc_shatters(H, C)
          for H in classes for C in sets))
```

It prints `True`. So the Natarajan dimension is a genuine generalization: on two labels it *is* the VC dimension.

Before the formulas take over, play with it. In the lab below, drop messages on the urgency line, choose two candidate folders for each, and watch the grid of mixtures light up as the class achieves them. Then switch to two folders and see lesson 6a come back.

[[lab]]

## Counting what a class can do: Natarajan’s lemma

Lesson 6b’s proofs ran on one engine: a class of small VC dimension produces few labellings, polynomially many in the number of points instead of exponentially many. Does a small Natarajan dimension buy the same thing?

Before the general statement, a number. The class of *all* filings of four messages into three folders has \(3^4 = 81\) members and Natarajan dimension 4. A class of Natarajan dimension 1 on four messages and three folders should be much smaller. The lemma below says it has at most \(4^1 \cdot 3^2 = 36\) members.

**Natarajan’s lemma.** Let \(\mathcal X\) be finite and \(\mathcal H \subseteq [k]^{\mathcal X}\). Then

\[ |\mathcal H| \le |\mathcal X|^{\mathrm{Ndim}(\mathcal H)} \cdot k^{2\,\mathrm{Ndim}(\mathcal H)}. \]

Read it from the outside in. \(|\mathcal H|\) is the number of hypotheses; \(|\mathcal X|\) is the number of messages; \(\mathrm{Ndim}(\mathcal H)\), say \(d\), sits in both exponents; and \(k^{2d}\) is \((k^2)^d\), one factor \(k^2\) per dimension, the number of candidate pairs. In plain words: **a class of Natarajan dimension \(d\) on \(n\) messages has at most \(n^d k^{2d}\) members, a polynomial in \(n\) instead of \(k^n\).** For three urgency folders on five scores, \(d = 2\), the lemma allows \(5^2 \cdot 3^4 = 2{,}025\); the true count is \(\binom{7}{2} = 21\) non-decreasing tuples. Loose, but polynomial, and that is all the proofs need.

The proof is an induction that removes one message at a time, so first the counting step it rests on. Fix one message \(x_0\), and let \(\mathcal X' = \mathcal X \setminus \{x_0\}\) be the other messages. Every hypothesis, cut down to \(\mathcal X'\), gives a **trace**: a function \(g\) on the other messages. Write \(\mathcal H_{\mathcal X'}\) for the set of traces. Several hypotheses may leave the same trace, differing only at \(x_0\). For two folders \(i < j\), let \(\mathcal H_{ij}\) be the traces that can be completed *both* with \(x_0 \mapsto i\) and with \(x_0 \mapsto j\) inside \(\mathcal H\) (here \(x_0 \mapsto i\) reads “\(x_0\) is filed in folder \(i\)”).

**Splitting step.**

\[ |\mathcal H| \le |\mathcal H_{\mathcal X'}| + \sum_{1 \le i < j \le k} |\mathcal H_{ij}|. \]

The sum runs over every pair of folders \(i < j\), adding up the sizes of the sets \(\mathcal H_{ij}\). In plain words: **the class is at most its traces, plus one extra for every trace and every pair of folders it can put at \(x_0\).** Take a trace that the class completes in \(t\) ways, with \(t\) different folders at \(x_0\). It contributes \(t\) to the left side. On the right it contributes 1 to the first term, plus 1 for each of the \(t(t-1)/2\) pairs among its \(t\) folders. And \(t \le 1 + t(t-1)/2\) for every whole number \(t \ge 1\), because the difference is \((t-1)(t-2)/2\), which is 0 for \(t = 1\) and \(t = 2\) and positive beyond. A trace with three completions contributes 3 on the left and \(1 + 3 = 4\) on the right. Summing over all traces gives the inequality. Here it is on 1,000 random classes of filings of four messages into three folders, with \(x_0\) the first message:

```python
import random
from itertools import combinations, product

rng, n, k = random.Random(2), 4, 3
every = list(product(range(1, k + 1), repeat=n))
holds = []
for _ in range(1000):
    H = set(rng.sample(every, rng.randint(1, 40)))
    rest = {h[1:] for h in H}
    tails = {i: {h[1:] for h in H if h[0] == i}
             for i in range(1, k + 1)}
    both = sum(len(tails[i] & tails[j])
               for i, j in combinations(range(1, k + 1), 2))
    holds.append(len(H) <= len(rest) + both)
print(all(holds), len(holds))
```

It prints `True 1000`: every one of the 1,000 classes satisfies the splitting step. `rest` is \(\mathcal H_{\mathcal X'}\), `tails[i]` is the set of traces completed with folder \(i\), and the intersection `tails[i] & tails[j]` is \(\mathcal H_{ij}\).

[[diagram:natarajan-split]]

*Proof of Natarajan’s lemma.* We prove a slightly more flexible statement: for every whole number \(d \ge 0\), every class on \(n \ge 1\) messages with \(\mathrm{Ndim} \le d\) has at most \(n^d k^{2d}\) members. The lemma is the case \(d = \mathrm{Ndim}(\mathcal H)\). (If \(\mathcal X\) is empty, there is only one function on it, and the lemma reads \(1 \le 1\) with the convention \(0^0 = 1\).)

*Case \(d = 0\).* If two hypotheses differed at some message \(x\), the single message \(\{x\}\) would be shattered with \(f_0(x)\) and \(f_1(x)\) the two folders they give it. So all hypotheses agree everywhere, \(|\mathcal H| \le 1 = n^0 k^0\).

*Case \(n = 1\), \(d \ge 1\).* A class on one message has at most \(k\) members, and \(k \le k^{2d}\).

*Induction step, \(n \ge 2\), \(d \ge 1\).* Assume the statement for all classes on \(n - 1\) messages, and apply the splitting step. Two facts about the pieces:

1. \(\mathrm{Ndim}(\mathcal H_{\mathcal X'}) \le d\). A set \(C \subseteq \mathcal X'\) shattered by the traces, with candidates \(f_0, f_1\), is shattered by \(\mathcal H\) with the same candidates, because each trace comes from a hypothesis that agrees with it on \(C\).
2. \(\mathrm{Ndim}(\mathcal H_{ij}) \le d - 1\). Suppose \(\mathcal H_{ij}\) shatters \(C \subseteq \mathcal X'\) with candidates \(f_0, f_1\). Extend the candidates to \(C \cup \{x_0\}\) by \(f_0(x_0) = i\) and \(f_1(x_0) = j\), which differ. Take any \(B \subseteq C \cup \{x_0\}\). Some trace \(g \in \mathcal H_{ij}\) gives the required mixture on \(C\) (for the set \(B \setminus \{x_0\}\)). Since \(g\) is in \(\mathcal H_{ij}\), both completions \(g\) with \(x_0 \mapsto i\) and \(g\) with \(x_0 \mapsto j\) are in \(\mathcal H\): use the first if \(x_0 \in B\) and the second otherwise. So \(\mathcal H\) shatters \(C \cup \{x_0\}\), a set one larger than \(C\). Since \(\mathcal H\) shatters nothing larger than \(d\), \(|C| + 1 \le d\).

By the induction hypothesis on \(n - 1\) messages, \(|\mathcal H_{\mathcal X'}| \le (n-1)^d k^{2d}\) and each \(|\mathcal H_{ij}| \le (n-1)^{d-1} k^{2(d-1)}\). There are \(k(k-1)/2 \le k^2\) pairs \(i < j\). So

\[ |\mathcal H| \le (n-1)^d k^{2d} + k^2 (n-1)^{d-1} k^{2d-2} = k^{2d} (n-1)^{d-1} \big( (n - 1) + 1 \big) = k^{2d}\, n\, (n-1)^{d-1} \le k^{2d} n^d. \]

Read the chain left to right: the first term is the bound for the traces; the second is \(k^2\) pairs times the bound for each \(\mathcal H_{ij}\); factoring out \(k^{2d}(n-1)^{d-1}\) leaves \((n - 1) + 1 = n\); and \((n-1)^{d-1} \le n^{d-1}\) finishes it. In plain words: **the traces and the pairs together never outgrow \(n^d k^{2d}\).** With \(n = 4\), \(d = 1\), \(k = 3\): \(k^{2d}\, n\, (n-1)^{d-1} = 9 \cdot 4 \cdot 1 = 36 = n^d k^{2d}\), so the last step is an equality there. \(\blacksquare\)

Now check the lemma itself on 200 random classes of filings of four messages into three folders, computing each class’s Natarajan dimension by brute force and the worst ratio \(|\mathcal H| / (n^d k^{2d})\) for each \(d\):

```python
import random
from itertools import combinations as comb, product

def ndim(H, n, k):
    pairs = list(comb(range(1, k + 1), 2))
    def sh(C):
        H_C = {tuple(h[x] for x in C) for h in H}
        return any(all(tuple(p[b] for p, b in zip(fs, B)) in H_C
                       for B in product((0, 1), repeat=len(C)))
                   for fs in product(pairs, repeat=len(C)))
    return max(r for r in range(n + 1)
               if any(sh(C) for C in comb(range(n), r)))

rng, n, k = random.Random(2), 4, 3
every = list(product(range(1, k + 1), repeat=n))
worst = {}
for _ in range(200):
    H = rng.sample(every, rng.randint(1, 30))
    d = ndim(H, n, k)
    r = len(H) / (n ** d * k ** (2 * d))
    worst[d] = round(max(worst.get(d, 0), r), 4)
print(sorted(worst.items()))
```

It prints `[(0, 1.0), (1, 0.1667), (2, 0.017), (3, 0.0006)]`. Every ratio is at most 1, as the lemma promises. The ratio 1 at \(d = 0\) is a one-hypothesis class, where the lemma is exact; for larger \(d\) the bound is generous.

## The multiclass fundamental theorem

With a dimension and a counting lemma in hand, the whole of lesson 6a’s theorem comes back, with one new ingredient: the number of folders, \(k\), appears, but only inside a logarithm.

**The multiclass fundamental theorem.** There are absolute constants \(C_1, C_2 > 0\) such that, for every class \(\mathcal H \subseteq [k]^{\mathcal X}\) with \(\mathrm{Ndim}(\mathcal H) = d\):

1. \(\mathcal H\) has the uniform convergence property with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon^2} \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d \log k + \log(1/\delta)}{\varepsilon^2}. \]
2. \(\mathcal H\) is agnostic PAC learnable with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon^2} \le m_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d \log k + \log(1/\delta)}{\varepsilon^2}. \]
3. \(\mathcal H\) is PAC learnable (assuming realizability) with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon} \le m_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d \log\left( kd/\varepsilon \right) + \log(1/\delta)}{\varepsilon}. \]

All three have the same anatomy, so read the second from the inside out.

- \(d\) is the Natarajan dimension, playing exactly the role the VC dimension played in lesson 6a.
- \(\log(1/\delta)\) is the price of confidence; \(\log\) is the natural logarithm throughout.
- \(d \log k\) is new: the dimension, multiplied by the logarithm of the number of folders. Going from 300 folders to 3,000 multiplies \(\log k\) by about 1.4, not by 10.
- Dividing by \(\varepsilon^2\) means halving the tolerance quadruples the labels; in the third line, where some hypothesis is perfect, the division is by \(\varepsilon\), and the extra \(\log(kd/\varepsilon)\) is the logarithm of the folder count times the dimension over the tolerance.
- \(C_1\) and \(C_2\) are **absolute constants**: fixed numbers, the same for every class, every \(k\), every distribution. The theorem says they exist but not what they are; as in lesson 28, the floors hold for small enough \(\varepsilon\) and \(\delta\).
- \(m^{\mathrm{UC}}_{\mathcal H}\) is lesson 4’s uniform-convergence sample size, the examples needed for every hypothesis’s empirical risk to be within \(\varepsilon\) of its true risk with probability at least \(1 - \delta\); \(m_{\mathcal H}\) is the learning one.

In plain words: **a multiclass class is learnable exactly when its Natarajan dimension is finite, and the labels it needs grow with that dimension, plus the log of one over the failure chance, divided by the tolerance squared, with a logarithmic factor such as \(\log k\) between the floor and the ceiling.** For your 300-folder filter, the next section will show that \(d\) is at most \(300 \times 30 = 9{,}000\). Without the unknown constants, at \(\varepsilon = \delta = 0.05\):

```python
from math import log

k, n, eps, delta = 300, 30, 0.05, 0.05
d = k * n
lower = (d + log(1 / delta)) / eps ** 2
upper = (d * log(k) + log(1 / delta)) / eps ** 2
pac_low = (d + log(1 / delta)) / eps
pac_up = (d * log(k * d / eps) + log(1 / delta)) / eps
print(d, round(lower), round(upper), round(upper / lower, 2))
print(round(pac_low), round(pac_up), round(pac_up / pac_low, 2))
```

It prints `9000 3601198 20534815 5.7`, then `180060 3204869 17.8`. In the agnostic case, the floor is a constant times 3.6 million and the ceiling a constant times 20.5 million, a gap of the factor \(\log 300 \approx 5.7\). In the realizable case the floor is a constant times 180,060 and the ceiling a constant times 3.2 million. These are the shapes the review needed. For a concrete budget we’ll need numbers without unknown constants; the last sections of this lesson produce some.

[[diagram:theorem-bounds]]

### Why the upper bounds hold

Before lesson 28, lesson 6b proved a weaker form of the theorem (with \(1/\delta\) where \(\log(1/\delta)\) belongs, the price of Markov’s inequality) through a theorem we can reuse almost untouched: **uniform convergence from a small growth function.** It said that with probability at least \(1 - \delta\), every \(h\) has \(|L_{\mathcal D}(h) - L_S(h)| \le (4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/(\delta\sqrt{2m})\), where the **growth function** \(\tau_{\mathcal H}(2m)\) is the largest restriction \(|\mathcal H_C|\) over sets \(C\) of \(2m\) points. Look back at Step 4 of its proof. The only thing it used about the class was that, once the \(2m\) training and ghost examples are fixed, each example’s 0-1 loss depends on \(h\) only through \(h\)’s label on that example’s message, so the number of distinct loss vectors is at most \(|\mathcal H_C|\). That is just as true with 300 folders: whether \(h\) files message \(x_i\) correctly depends only on \(h(x_i)\). So the same theorem holds for multiclass classes, with \(\tau_{\mathcal H}(2m)\) counting folder vectors.

Now apply Natarajan’s lemma to the restriction \(\mathcal H_C\), a class on the at most \(2m\) messages of \(C\). It cannot shatter more than \(\mathcal H\) does, so its Natarajan dimension is at most \(d\), and \(|\mathcal H_C| \le (2m)^d k^{2d}\). Taking logs, \(\log \tau_{\mathcal H}(2m) \le d \log(2m) + 2d \log k\), and with probability at least \(1 - \delta\), for every \(h \in \mathcal H\),

\[ |L_{\mathcal D}(h) - L_S(h)| \le \frac{4 + 2\sqrt{d \log(2m) + 2d \log k}}{\delta\sqrt{2m}}. \]

From the inside out: \(d\log(2m) + 2d\log k\) is the log of Natarajan’s count on \(2m\) messages; its square root, doubled and plus 4, is the numerator from lesson 6b; \(\delta\sqrt{2m}\) underneath shrinks the bound as labels grow. In plain words: **every hypothesis’s training score is within a quantity that shrinks like \(\sqrt{d(\log m + \log k)/m}\) of its true score, with probability at least \(1 - \delta\).** For three urgency folders (\(d = 2\), \(k = 3\)) at \(\delta = 0.1\):

```python
from math import log, sqrt

def uc_bound(m, d, k, delta):
    log_tau = d * log(2 * m) + 2 * d * log(k)
    return (4 + 2 * sqrt(log_tau)) / (delta * sqrt(2 * m))

d, k, delta = 2, 3, 0.1
for m in (10 ** 3, 10 ** 5, 10 ** 7):
    print(m, round(uc_bound(m, d, k, delta), 4))
```

It prints `1000 2.8741`, `100000 0.3295` and `10000000 0.0365`. Useless at a thousand labels, but it goes to zero as \(m\) grows, which is all a uniform convergence property needs: for any \(\varepsilon\) there is an \(m\) beyond which the bound is under \(\varepsilon\). So finite Natarajan dimension and finitely many folders give uniform convergence, and by lesson 4 any ERM is then a successful agnostic learner. The sharper rates come from running lesson 28’s proofs with Natarajan’s \((2m)^d k^{2d}\) in the place of Sauer’s \((2em/d)^d\): its \(\varepsilon\)-net argument gives the realizable ceiling, \(d\log(kd/\varepsilon)\) and all, and its Rademacher argument gives an agnostic ceiling with \(d\log(kd/\varepsilon)\) where the theorem has \(d\log k\) (Problem 3). Removing that extra \(\log(d/\varepsilon)\) needs the chaining that lesson 28 left unproved for two labels too.

### Why the lower bounds hold

The floors come from the two-label theorem by a disguise. Suppose \(\mathrm{Ndim}(\mathcal H) = d\), witnessed by a set \(C\) of \(d\) messages and candidates \(f_0, f_1\). Picture a two-label problem on \(C\): each message’s label is 1 (“it belongs in its first candidate”) or 0 (“in its second”). Any multiclass learner for \(\mathcal H\) can be turned into a two-label learner for the class of *all* yes/no functions on \(C\), whose VC dimension is \(d\), and a learner for that class cannot beat lesson 28’s two-label floor, \((d + \log(1/\delta))/(1024\,\varepsilon^2)\) for \(\varepsilon \le 1/(8\sqrt2)\) and \(\delta \le 1/8\).

**Proposition (the disguise).** Fix \(C, f_0, f_1\) as above and let \(A\) be any multiclass learner. Define a two-label learner \(A'\): given yes/no examples \((x, b)\) with \(x \in C\), relabel each as \((x, f_0(x))\) if \(b = 1\) and \((x, f_1(x))\) if \(b = 0\), run \(A\) to get \(h\), and return \(g(x) = \mathbb 1[h(x) = f_0(x)]\). Then for every distribution \(\mathcal D'\) on \(C \times \{0, 1\}\), with \(\mathcal D\) the relabelled distribution, \(L_{\mathcal D'}(g) - \min_{b'} L_{\mathcal D'}(b') \le L_{\mathcal D}(h) - \min_{h' \in \mathcal H} L_{\mathcal D}(h')\), where \(b'\) ranges over all yes/no functions on \(C\).

Here \(\mathbb 1[\cdot]\) is the indicator, 1 when the statement inside holds and 0 otherwise, so \(g\) says yes exactly when \(h\) picked the first candidate; \(\min\) is the smallest true risk in a class. In plain words: **the disguised learner’s excess risk is never more than the multiclass learner’s.**

*Proof.* Two pointwise facts. First, \(g\) errs only where \(h\) errs. On an example whose label is the first candidate, \(g\) is wrong exactly when \(h(x) \ne f_0(x)\), that is, exactly when \(h\) is wrong. On an example whose label is the second candidate, \(g\) is wrong when \(h(x) = f_0(x)\), and then \(h(x) \ne f_1(x)\), so \(h\) is wrong too. Taking expectations, \(L_{\mathcal D'}(g) \le L_{\mathcal D}(h)\).

Second, the best yes/no function is no better than the best multiclass hypothesis. For each yes/no function \(b'\) on \(C\), shattering gives an \(h' \in \mathcal H\) taking \(f_0\) where \(b' = 1\) and \(f_1\) where \(b' = 0\). Since \(h'(x)\) is always one of the two candidates, \(h'\) is wrong on an example exactly when \(b'\) is, so \(L_{\mathcal D}(h') = L_{\mathcal D'}(b')\), and therefore \(\min_{h'} L_{\mathcal D}(h') \le \min_{b'} L_{\mathcal D'}(b')\). Subtracting the second fact from the first gives the claim. \(\blacksquare\)

Here are both pointwise facts on 100,000 random examples with five folders:

```python
import random

rng, k, n = random.Random(0), 5, 100_000
g_errs = h_errs = g_only = 0
for _ in range(n):
    f0, f1 = rng.sample(range(1, k + 1), 2)
    y, h = rng.choice([f0, f1]), rng.randint(1, k)
    g_wrong = (h == f0) != (y == f0)
    g_errs, h_errs = g_errs + g_wrong, h_errs + (h != y)
    g_only += g_wrong and h == y
print(round(g_errs / n, 4), round(h_errs / n, 4), g_only)
```

It prints `0.5014 0.8014 0`: the disguised learner errs on about half the examples and the multiclass one on about 80%, and the last number counts examples where \(g\) erred but \(h\) did not. There are none.

So if \(A\) agnostically learns \(\mathcal H\) with \(m\) examples, then \(A'\) agnostically learns a two-label class of VC dimension \(d\) with \(m\) examples, and lesson 28’s floor forces \(m \ge (d + \log(1/\delta))/(1024\,\varepsilon^2)\), so \(C_1 = 1/1024\) works here too. In the realizable case the same disguise works: if \(\mathcal D'\) is labelled by some \(b'\), then \(\mathcal D\) is labelled by the matching \(h'\), and \(L_{\mathcal D'}(g) \le L_{\mathcal D}(h)\), so any two-label realizable floor carries over; the one lesson 6a stated has \(\varepsilon\) in place of \(\varepsilon^2\) (lesson 28 left only its \(\log(1/\delta)\) part, as a problem). For uniform convergence, recall from lesson 4 that an \(\varepsilon/2\)-representative sample makes any ERM \(\varepsilon\)-accurate, so \(m_{\mathcal H}(\varepsilon, \delta) \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\). Writing \(\varepsilon'\) for \(\varepsilon/2\), \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon', \delta) \ge m_{\mathcal H}(2\varepsilon', \delta) \ge C_1 (d + \log(1/\delta))/(4\varepsilon'^2)\): the first floor, with the constant divided by 4.

[[diagram:disguise]]

## Calculating the Natarajan dimension

A dimension is only useful if you can compute it, or at least bound it, for the classes you actually use. Here is the pattern you will see three times: the Natarajan dimension tends to be about the number of parameters it takes to write a hypothesis down.

### One-versus-all

Recall lesson 17a’s **one-versus-all**: train one yes/no filter per folder, “is it folder \(i\) or not?”, and file a message in a folder whose filter says yes, breaking ties (several yeses, or none) by the smallest folder number. Lesson 17a wrote the answers as \(+1\) and \(-1\); here yes is 1 and no is 0, which picks the same folder. If each yes/no filter comes from a class \(\mathcal H_{\mathrm{bin}}\) of VC dimension \(d\), the multiclass class is

\[ \mathcal H^{\mathrm{OvA}, k}_{\mathrm{bin}} = \left\{ T(h_1, \dots, h_k) : h_1, \dots, h_k \in \mathcal H_{\mathrm{bin}} \right\}, \qquad T(h_1, \dots, h_k)(x) = \operatorname*{argmax}_{i \in [k]} h_i(x). \]

Inside out: \(h_i(x)\) is filter \(i\)’s answer, 1 or 0; \(\operatorname{argmax}_{i \in [k]}\) is the folder number whose answer is largest, with ties to the smallest number; \(T(h_1, \dots, h_k)\) is the folder rule built from the \(k\) filters; and the braces collect it over every choice of \(k\) filters from \(\mathcal H_{\mathrm{bin}}\). In plain words: **the class of every filer you can assemble from \(k\) yes/no filters, one per folder.** With filters answering 0, 1, 1, 0 for a message, the largest answer is 1, shared by folders 2 and 3, and the message goes to folder 2.

How big should its Natarajan dimension be? A hypothesis takes \(k\) filters to write down, each with “\(d\) parameters’ worth” of freedom, so about \(kd\). That is the right size, up to a logarithm. To prove it, we will use Sauer’s lemma in a form that holds for every \(c\), not only \(c \ge d\): a class of VC dimension \(d\) produces at most \(\sum_{i \le d}\binom{c}{i} \le (c + 1)^d\) labellings of \(c\) points. (Each \(\binom{c}{i} \le c^i \le \binom{d}{i} c^i\), and the binomial theorem sums \(\sum_{i \le d} \binom{d}{i} c^i\) to \((c + 1)^d\).)

### General reductions

One-versus-all is one of many **multiclass-to-binary reductions**: train \(l\) yes/no filters, and let a fixed rule \(r: \{0,1\}^l \to [k]\) turn their \(l\) answers into a folder. All-pairs from lesson 17a is another, with \(l = \binom{k}{2}\). The class is

\[ \mathcal H^{r}_{\mathrm{bin}} = \left\{ R(h_1, \dots, h_l) : h_1, \dots, h_l \in \mathcal H_{\mathrm{bin}} \right\}, \qquad R(h_1, \dots, h_l)(x) = r\big(h_1(x), \dots, h_l(x)\big). \]

Inside out: the \(l\) filters answer, \(r\) reads the vector of answers and names a folder, and the braces collect every such rule. In plain words: **every filer you can build by asking \(l\) yes/no filters and combining their answers with a fixed rule.** One-versus-all is the case \(l = k\), with \(r\) returning the position of the first 1, or 1 if there is none.

**Lemma (reductions).** If \(\mathrm{VCdim}(\mathcal H_{\mathrm{bin}}) = d\) and \(ld \ge 4\), then \(\mathrm{Ndim}(\mathcal H^{r}_{\mathrm{bin}}) \le 3\, l d \log(l d)\). In particular, \(\mathrm{Ndim}(\mathcal H^{\mathrm{OvA},k}_{\mathrm{bin}}) \le 3kd\log(kd)\) when \(kd \ge 4\).

In plain words: **a reduction that asks \(l\) filters of VC dimension \(d\) has Natarajan dimension at most about \(3ld\log(ld)\).** For 300 one-versus-all filters from a class of VC dimension \(d = 30\), that is \(3 \times 9{,}000 \times \log 9{,}000 \approx 245{,}800\).

*Proof.* Let \(C\) be shattered, and count the folder vectors \(|(\mathcal H^r_{\mathrm{bin}})_C|\) two ways.

From below: the \(2^{|C|}\) mixtures are all different vectors on \(C\), because \(f_0(x) \ne f_1(x)\) at every message, so two different sets \(B\) give vectors that differ somewhere. Each is achieved, so \(|(\mathcal H^r_{\mathrm{bin}})_C| \ge 2^{|C|}\).

From above: a rule’s folders on \(C\) are determined by its \(l\) filters’ answers on \(C\), which are \(l\) labellings from \((\mathcal H_{\mathrm{bin}})_C\). There are at most \(|(\mathcal H_{\mathrm{bin}})_C|^l\) ways to choose them, and Sauer gives \(|(\mathcal H_{\mathrm{bin}})_C| \le (|C| + 1)^d\). Together,

\[ 2^{|C|} \le \left|(\mathcal H^r_{\mathrm{bin}})_C\right| \le \left|(\mathcal H_{\mathrm{bin}})_C\right|^{l} \le (|C| + 1)^{ld}. \]

Left to right: the mixtures, the folder vectors, the choices of \(l\) labellings, and Sauer raised to the \(l\)-th power. In plain words: **a shattered set needs \(2^{|C|}\) folder vectors, but \(l\) filters can only produce polynomially many.** Write \(c = |C|\) and \(D = ld\); taking logs, \(c \log 2 \le D \log(c + 1)\).

It remains to see that this forces \(c \le 3D\log D\) when \(D \ge 4\). Let \(\phi(c) = c\log 2 - D\log(c+1)\); we know \(\phi(c) \le 0\), and we show \(\phi(c) > 0\) for every \(c > 3D\log D\). The slope of \(\phi\) is \(\log 2 - D/(c + 1)\), positive once \(c + 1 > D/\log 2 \approx 1.44D\); and \(3D\log D \ge 3D\log 4 > 4D\), so \(\phi\) is increasing beyond \(c_0 = 3D\log D\). At \(c_0\), \(\phi(c_0) \ge 0\) says \(2^{c_0} \ge (c_0 + 1)^D\). The left side is \(e^{3D\log D \cdot \log 2} = (D^{3\log 2})^D\), so taking \(D\)-th roots it is the inequality \(D^{3\log 2} \ge 3D\log D + 1\). With \(3\log 2 \approx 2.079\), at \(D = 4\) the left side is 17.9 and the right side 17.6; and the left side’s slope, \(2.079 D^{1.079}\), beats the right side’s, \(3\log D + 3\), for every \(D \ge 4\): at \(D = 4\) it is 9.3 against 7.2, and from there the left slope rises faster, since the slopes’ own rates of change are \(2.24 D^{0.079} \ge 2.24\) against \(3/D \le 0.75\). So \(\phi(c) > 0\) for \(c > c_0\), hence \(c \le 3D \log D\). \(\blacksquare\)

The condition \(D \ge 4\) is real, not decoration: for \(D = 2\) and \(3\), the inequality \(c\log 2 \le D\log(c+1)\) allows sets larger than \(3D\log D\). Here is the largest \(c\) it allows, next to \(3D\log D\), and the last step checked numerically:

```python
from math import log

def largest_c(D):
    c = 0
    while (c + 1) * log(2) <= D * log(c + 2):
        c += 1
    return c

for D in (2, 3, 4, 10, 300, 9000):
    print(D, largest_c(D), round(3 * D * log(D), 1))
print(all(D ** (3 * log(2)) >= 3 * D * log(D) + 1
          for D in range(4, 10 ** 5)))
```

It prints

```
2 5 4.2
3 10 9.9
4 16 16.6
10 59 69.1
300 3536 5133.4
9000 155193 245834.5
True
```

From \(D = 4\) on, the counting argument never allows more than \(3D\log D\); at 2 and 3 it does, which is why the lemma asks for \(ld \ge 4\).

How tight is it? Take one-versus-all with \(k = 4\) folders over lesson 6a’s thresholds (VC dimension 1) on six scores. Python builds every rule from every choice of four thresholds and computes the Natarajan dimension by brute force:

```python
from itertools import combinations as comb, product

def ndim(H, n, k):
    pairs = list(comb(range(1, k + 1), 2))
    def sh(C):
        H_C = {tuple(h[x] for x in C) for h in H}
        return any(all(tuple(p[b] for p, b in zip(fs, B)) in H_C
                       for B in product((0, 1), repeat=len(C)))
                   for fs in product(pairs, repeat=len(C)))
    return max(r for r in range(n + 1)
               if any(sh(C) for C in comb(range(n), r)))

n, k = 6, 4
H_bin = [tuple(int(x > t) for x in range(n))
         for t in range(-1, n)]

def T(hs):
    return tuple(1 + max(range(k), key=lambda i: (hs[i][x], -i))
                 for x in range(n))

H_ova = {T(hs) for hs in product(H_bin, repeat=k)}
print(len(H_bin), len(H_bin) ** k, len(H_ova), ndim(H_ova, n, k))
```

It prints `7 2401 204 4`. Seven threshold labellings of six scores, \(7^4 = 2{,}401\) ways to choose four filters, but only 204 distinct folder rules: different choices often file every message the same way. The Natarajan dimension is 4, equal to \(kd\), and well under the lemma’s \(3 \cdot 4 \cdot \log 4 \approx 16.6\). For some binary classes one-versus-all is far smaller than \(kd\) (Problem 1), and for halfspaces it really is of order \(kd\) (Problem 2).

[[diagram:ova-count]]

### Linear multiclass predictors

Your 300-folder filter is not a reduction; it is lesson 17a’s **linear multiclass predictor**, the class \(\mathcal H_{\Psi, W}\) with every weight vector allowed, \(W = \mathbb R^d\). In this section, as in lesson 17a, \(d\) is the length of the feature vectors, not a dimension of a class. Recall: a **class-sensitive feature mapping** \(\Psi: \mathcal X \times [k] \to \mathbb R^d\) turns a message \(x\) and a candidate folder \(i\) into a vector \(\Psi(x, i)\) of \(d\) numbers; a weight vector \(\mathbf w \in \mathbb R^d\) scores the pair by the **inner product** \(\langle \mathbf w, \Psi(x, i) \rangle\) (multiply matching entries and add); and the predictor files the message in the folder with the highest score. The class is

\[ \mathcal H_\Psi = \left\{ x \mapsto \operatorname*{argmax}_{i \in [k]} \langle \mathbf w, \Psi(x, i) \rangle \;:\; \mathbf w \in \mathbb R^d \right\}. \]

Inside out: for a message \(x\), compute all \(k\) scores \(\langle \mathbf w, \Psi(x, i)\rangle\); \(\operatorname{argmax}\) picks the folder with the largest score, and as with one-versus-all we break ties by the smallest folder number; \(x \mapsto \dots\) (“\(x\) goes to …”) names the resulting function; the braces collect one such function per weight vector \(\mathbf w\), where \(\mathbb R^d\) is the set of all vectors of \(d\) real numbers. In plain words: **one hypothesis per weight vector, each filing a message in its highest-scoring folder.** Each hypothesis is written down with \(d\) numbers, so you might guess \(\mathrm{Ndim} \le d\). That guess is exactly right.

**Theorem (linear multiclass predictors).** \(\mathrm{Ndim}(\mathcal H_\Psi) \le d\).

The idea is to turn each shattered message into a vector, the *difference* between its two candidates’ feature vectors, and to notice that “the first candidate wins” is a halfspace question about that difference. Then a lemma about halfspaces finishes the job.

**Lemma (no \(d + 1\) vectors).** No \(d + 1\) vectors \(\mathbf u_1, \dots, \mathbf u_{d+1} \in \mathbb R^d\) are shattered by the class of yes/no rules \(\mathbf v \mapsto \mathbb 1[\langle \mathbf w, \mathbf v \rangle \ge 0]\), \(\mathbf w \in \mathbb R^d\).

*Proof.* Any \(d + 1\) vectors in \(\mathbb R^d\) are linearly dependent: there are numbers \(a_1, \dots, a_{d+1}\), not all zero, with \(\sum_i a_i \mathbf u_i = \mathbf 0\). Flipping the signs of all the \(a_i\) if necessary, the set \(I = \{i : a_i > 0\}\) is not empty; let \(J = \{i : a_i < 0\}\). Suppose some \(\mathbf w\) labels every \(\mathbf u_i\) with \(i \in I\) as 0 (so \(\langle \mathbf w, \mathbf u_i \rangle < 0\)) and every \(\mathbf u_j\) with \(j \in J\) as 1 (so \(\langle \mathbf w, \mathbf u_j \rangle \ge 0\)). Then

\[ 0 = \Big\langle \mathbf w, \sum_i a_i \mathbf u_i \Big\rangle = \sum_{i \in I} a_i \langle \mathbf w, \mathbf u_i \rangle + \sum_{j \in J} a_j \langle \mathbf w, \mathbf u_j \rangle < 0, \]

because the first sum has at least one term, each positive times negative, and the second has terms negative times non-negative. Here \(\sum_{i \in I}\) adds over the indices in \(I\). In plain words: **the dependency forbids one particular labelling, so the vectors are not shattered.** \(\blacksquare\)

For \(\mathbf u_1 = (1, 0)\), \(\mathbf u_2 = (0, 1)\), \(\mathbf u_3 = (1, 1)\), the dependency is \(\mathbf u_1 + \mathbf u_2 - \mathbf u_3 = \mathbf 0\), so the forbidden labelling is “0, 0, 1”. A hundred thousand random \(\mathbf w\) agree:

```python
import random

u = [(1, 0), (0, 1), (1, 1)]
rng, seen = random.Random(0), set()
for _ in range(100_000):
    w = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    seen.add(tuple(int(w[0] * a + w[1] * b >= 0) for a, b in u))
print(len(seen), sorted(set(
    (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
    - seen))
```

It prints `6 [(0, 0, 1), (1, 1, 0)]`: six of the eight labellings appear, and the two missing are the one the proof forbids and its mirror, which comes from the same dependency with the signs flipped.

*Proof of the theorem.* Let \(C\) be shattered by \(\mathcal H_\Psi\) with candidates \(f_0, f_1\). For each \(x \in C\) define the **difference vector**

\[ \rho(x) = \Psi(x, f_0(x)) - \Psi(x, f_1(x)), \]

the feature vector of the first candidate minus that of the second. Then \(\langle \mathbf w, \rho(x) \rangle\) is the first candidate’s score minus the second’s: positive means the first candidate is ahead.

Now the ties. Split \(C\) into \(C_<\), the messages whose first candidate has the smaller folder number, and \(C_>\), the rest. Take a hypothesis with weights \(\mathbf w\) that picks one of the two candidates at \(x\).

- If \(x \in C_<\): picking \(f_0(x)\) means its score is the largest, so \(\langle \mathbf w, \rho(x) \rangle \ge 0\). Picking \(f_1(x)\) means \(f_1(x)\) has the largest score and no smaller folder number ties it; \(f_0(x)\) has a smaller number, so its score is strictly lower: \(\langle \mathbf w, \rho(x) \rangle < 0\).
- If \(x \in C_>\): by the same reasoning with the roles swapped, picking \(f_1(x)\) means \(\langle \mathbf w, -\rho(x) \rangle \ge 0\), and picking \(f_0(x)\) means \(\langle \mathbf w, -\rho(x) \rangle < 0\).

So set \(\mathbf u(x) = \rho(x)\) on \(C_<\) and \(\mathbf u(x) = -\rho(x)\) on \(C_>\). The yes/no rule \(\mathbb 1[\langle \mathbf w, \mathbf u(x) \rangle \ge 0]\) says yes exactly on the messages of \(C_<\) where \(f_0\) was picked and the messages of \(C_>\) where \(f_1\) was picked. As \(B\) runs over all subsets of \(C\), that yes-set, \((B \cap C_<) \cup (C_> \setminus B)\), also runs over all subsets of \(C\) (the map flips membership on \(C_>\) and keeps it on \(C_<\), and doing it twice gives \(B\) back). So the vectors \(\mathbf u(x)\) are shattered by the yes/no rules of the lemma. They are also all different: two messages with the same vector would always get the same answer, and could not be split. By the lemma there are at most \(d\) of them, so \(|C| \le d\). \(\blacksquare\)

[[diagram:difference-vector]]

For your filter, \(\Psi\) is the **multivector construction** of lesson 17a: with messages \(\mathbf x \in \mathbb R^n\) and \(k\) folders, \(\Psi(\mathbf x, y) \in \mathbb R^{kn}\) is \(k\) blocks of \(n\) entries, all zero except block \(y\), which holds \(\mathbf x\):

\[ \Psi(\mathbf x, y) = \big[\, \underbrace{0, \dots, 0}_{(y-1)n},\ \underbrace{x_1, \dots, x_n}_{n},\ \underbrace{0, \dots, 0}_{(k-y)n} \,\big]. \]

The braces underneath count the entries: \((y-1)n\) zeros for the blocks before \(y\), the \(n\) features, then \((k-y)n\) zeros. Writing \(\mathbf w\) as \(k\) blocks \(\mathbf w_1, \dots, \mathbf w_k\), the score of folder \(y\) is \(\langle \mathbf w_y, \mathbf x \rangle\). In plain words: **one weight vector per folder, and each folder scores the message with its own.** With \(k = 300\) and \(n = 30\), \(\Psi\) has \(9{,}000\) entries.

**Corollary (multivector).** For the multivector construction with \(n\) features and \(k\) folders,

\[ (k - 1)(n - 1) \le \mathrm{Ndim}(\mathcal H_\Psi) \le kn. \]

In plain words: **the Natarajan dimension of one-weight-vector-per-folder is about the number of weights.** For your filter, between \(299 \times 29 = 8{,}671\) and \(9{,}000\).

*Proof.* The upper bound is the theorem with \(d = kn\). For the lower bound we shatter \((k-1)(n-1)\) messages explicitly. Use the message \(\mathbf x(i, j)\), for \(i \in [k-1]\) and \(j \in [n-1]\), with a 1 in feature \(j\), the number \(i\) in the last feature, and zeros elsewhere. Candidates: \(f_0 = i\) and \(f_1 = i + 1\).

Fix a column \(j\) and look at the scores of the messages \(\mathbf x(1, j), \dots, \mathbf x(k-1, j)\). Folder \(y\) scores \(\mathbf x(i, j)\) as \(w_{y,j} + i\, w_{y,n}\): a straight line in \(i\), with intercept \(a_y = w_{y,j}\) and slope \(w_{y,n}\). Set every slope to \(w_{y,n} = y\), shared by all columns. Lines \(y\) and \(y + 1\) cross where \(a_y + iy = a_{y+1} + i(y+1)\), at \(i = a_y - a_{y+1}\); call that crossing \(s_y\). For \(i < s_y\) line \(y\) is strictly higher, for \(i > s_y\) line \(y + 1\) is. Now choose, for each \(i\), \(s_i = i + 0.4\) if the mixture wants folder \(i\) at \(\mathbf x(i, j)\), and \(s_i = i - 0.4\) if it wants \(i + 1\); then set \(a_1 = 0\) and \(a_{y+1} = a_y - s_y\). The crossings increase strictly, \(s_i \le i + 0.4 < i + 0.6 \le s_{i+1}\).

Take the message \(\mathbf x(i, j)\) and suppose the mixture wants folder \(i\), so \(i < s_i\); also \(s_{i-1} \le i - 0.6 < i\). Comparing neighbours, line \(i\) beats \(i + 1\) (as \(i < s_i\)), which beats \(i + 2\) (as \(i < s_{i+1}\)), and so on up; and line \(i\) beats \(i - 1\) (as \(i > s_{i-1}\)), which beats \(i - 2\), and so on down. All strictly, so folder \(i\) wins with no tie. The case where the mixture wants \(i + 1\) is the same argument with \(s_i < i < s_{i+1}\). Each column’s intercepts are chosen independently, through the entries \(w_{y,j}\), so every mixture over all \((k-1)(n-1)\) messages is achieved. \(\blacksquare\)

Python builds the weights for every one of the \(2^6\) mixtures with \(k = 4\) folders and \(n = 3\) features, and checks that each message lands in the wanted folder with no tie:

```python
from itertools import product

k, n = 4, 3
pts = [(i, j) for i in range(1, k) for j in range(1, n)]

def x(i, j):
    return [int(c == j) for c in range(1, n)] + [i]

def realize(want):
    w = {y: [0.0] * (n - 1) + [y] for y in range(1, k + 1)}
    for (i, j), label in want.items():
        t = i + 0.4 if label == i else i - 0.4
        w[i + 1][j - 1] = w[i][j - 1] - t
    return w

ok = 0
for B in product([0, 1], repeat=len(pts)):
    want = {p: p[0] + b for p, b in zip(pts, B)}
    w = realize(dict(sorted(want.items())))
    s = {p: [sum(a * b for a, b in zip(w[y], x(*p)))
             for y in range(1, k + 1)] for p in pts}
    ok += all(s[p].index(max(s[p])) + 1 == want[p]
              and s[p].count(max(s[p])) == 1 for p in pts)
print(len(pts), ok, 2 ** len(pts))
```

It prints `6 64 64`: six messages, all 64 mixtures achieved.

So your 300-folder filter has Natarajan dimension between 8,671 and 9,000, and the multiclass fundamental theorem gives the review its answer, in the shape of the numbers printed earlier. One bug down.

## Good and bad ERMs

Now the hungry folder. Let’s build the auto-folder class carefully, because it hides a surprise that the two-label world never showed us.

The domain \(\mathcal X\) is the set of senders (each message is filed by its sender), finite or even infinite. A user’s folder is a group of senders \(A\). If \(\mathcal X\) is infinite we allow only groups that are **finite or cofinite**: either \(A\) is finite or everything outside \(A\) is. Write \(\mathcal P_f(\mathcal X)\) for the collection of such groups (with a finite \(\mathcal X\), every group qualifies). The labels are the groups themselves, together with one special label \(\ast\), “inbox”. For each group \(A\), the rule

\[ h_A(x) = \begin{cases} A & \text{if } x \in A, \\ \ast & \text{if } x \notin A, \end{cases} \]

files mail from senders in \(A\) into the folder named \(A\), and everything else into the inbox. The class is \(\mathcal H_{\mathrm{auto}} = \{h_A : A \in \mathcal P_f(\mathcal X)\}\). The big brace picks one of two lines depending on the condition on the right. In plain words: **each rule names a group of senders, files their mail under that group’s name, and leaves the rest in the inbox.** With \(A = \{\text{Ana}, \text{Bo}\}\), mail from Bo goes to the folder “\(\{\text{Ana}, \text{Bo}\}\)” and mail from Cy to the inbox. The rule \(h_\emptyset\), with the empty group \(\emptyset\), keeps everything in the inbox.

With 1,000 senders there are \(2^{1000}\) groups, so the folder count \(k\) is astronomical: \(2^{1000} - 1\) nonempty group names plus the inbox, \(2^{1000}\) labels in all. And yet the class is tiny in the Natarajan sense.

**Proposition.** If \(\mathcal X\) has at least one sender, \(\mathrm{Ndim}(\mathcal H_{\mathrm{auto}}) = 1\).

*Proof.* One sender \(x\) is shattered with candidates \(\{x\}\) and \(\ast\): \(h_{\{x\}}\) and \(h_\emptyset\). Now suppose two senders \(x_1, x_2\) were shattered. Every rule gives each sender either the inbox or the name of its own group, and a rule that names a group at both senders names the *same* group at both. At least one candidate at \(x_1\) is a group name, say \(A\) (the two candidates differ, so they are not both \(\ast\)). Likewise \(x_2\) has a group-name candidate \(A'\). The mixture choosing \(A\) at \(x_1\) and \(A'\) at \(x_2\) needs a rule naming two groups at once, so \(A = A'\), and the rule is \(h_A\) with both senders in \(A\). But then the mixture choosing \(A\) at \(x_1\) and the *other* candidate at \(x_2\) needs a rule naming \(A\) at \(x_1\), which must be \(h_A\), and \(h_A\) gives \(x_2\) the label \(A\) too, not the other candidate. No rule achieves it, so no two senders are shattered. \(\blacksquare\)

A brute-force check with four senders and all 16 groups:

```python
from itertools import combinations as comb, product

n = 4
groups = [frozenset(g) for r in range(1, n + 1)
          for g in comb(range(n), r)] + [frozenset()]
H = [tuple(A if x in A else "*" for x in range(n))
     for A in groups]
labels = list(groups[:-1]) + ["*"]

def shatters(C):
    H_C = {tuple(h[x] for x in C) for h in H}
    return any(all(tuple(p[b] for p, b in zip(fs, B)) in H_C
                   for B in product((0, 1), repeat=len(C)))
               for fs in product(comb(labels, 2), repeat=len(C)))

sizes = [r for r in (1, 2)
         if any(shatters(C) for C in comb(range(n), r))]
print(len(H), len(labels), sizes)
```

It prints `16 16 [1]`: 16 rules, 16 labels in use, and the only shattered size among 1 and 2 is 1.

### Every ERM, almost

Suppose the true rule is \(h_A\), and the training sequence \(S\) contains a message labelled with a group name. The only group name any rule can output for it is its own, so that name must be \(A\), and only \(h_A\) outputs it. Any ERM must return a hypothesis with empirical risk 0, and \(h_A\) is the only one that fits that example: so *every* ERM returns the truth, with true risk 0. The ERMs can differ only on training sequences where every label is \(\ast\), all inbox:

\[ S = \big( (x_1, \ast), \dots, (x_m, \ast) \big). \]

There, any \(h_B\) whose group \(B\) avoids every training sender is consistent. Two choices stand out.

- \(A_{\mathrm{good}}(S) = h_\emptyset\): when in doubt, keep everything in the inbox.
- \(A_{\mathrm{bad}}(S) = h_{\{x_1, \dots, x_m\}^c}\): make a folder out of every sender not seen. Here \(\{x_1, \dots, x_m\}^c\) is the **complement** of the seen senders, everyone else. It is cofinite, so the rule is in the class, and it is consistent because every training sender is outside it.

Both are ERMs: on sequences containing a group name both return \(h_A\), and on all-inbox sequences both return a consistent rule. Your team shipped \(A_{\mathrm{bad}}\).

**Claim (the good ERM).** Let \(\varepsilon, \delta \in (0, 1)\), let \(\mathcal D\) be any distribution over senders, and let the truth be any \(h_A \in \mathcal H_{\mathrm{auto}}\). If

\[ m \ge \frac{\log(1/\delta)}{\varepsilon}, \]

then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A_{\mathrm{good}}(S)) \le \varepsilon\).

Read it: \(\log(1/\delta)\) is the log of one over the failure chance, divided by the tolerance; no \(k\), no number of senders, no dimension appears. \(S \sim \mathcal D^m\) means the \(m\) senders are drawn independently from \(\mathcal D\). In plain words: **the good ERM needs only about \(\log(1/\delta)/\varepsilon\) labels, whatever the senders and folders.** At \(\varepsilon = \delta = 0.05\) that is \(\log 20/0.05 \approx 59.9\), so 60 labels.

*Proof.* \(A_{\mathrm{good}}\) returns either \(h_A\) (true risk 0) or \(h_\emptyset\). It returns \(h_\emptyset\) only when no training sender is in \(A\), and \(h_\emptyset\) errs exactly on mail from senders in \(A\), so \(L_{\mathcal D}(h_\emptyset) = \mathcal D(A)\), the chance that a message comes from \(A\). So a failure needs two things: \(\mathcal D(A) > \varepsilon\), and all \(m\) independent draws land outside \(A\). If \(\mathcal D(A) \le \varepsilon\), failure is impossible. Otherwise each draw lands outside \(A\) with probability \(1 - \mathcal D(A) < 1 - \varepsilon\), and by independence all \(m\) do so with probability less than \((1 - \varepsilon)^m\). Finally, \(1 - \varepsilon \le e^{-\varepsilon}\) (the line \(1 - u\) lies below the curve \(e^{-u}\), which it touches at \(u = 0\)), so

\[ (1 - \varepsilon)^m \le e^{-\varepsilon m} \le e^{-\log(1/\delta)} = \delta, \]

where the middle step uses \(m \ge \log(1/\delta)/\varepsilon\). \(\blacksquare\)

The worst case puts exactly \(\varepsilon\) of the mail on the true group. Simulating it at \(\varepsilon = \delta = 0.05\):

```python
import random
from math import ceil, exp, log

eps, delta = 0.05, 0.05
m = ceil(log(1 / delta) / eps)
rng, trials = random.Random(0), 100_000
fails = sum(all(rng.random() >= eps for _ in range(m))
            for _ in range(trials))
print(m, round((1 - eps) ** m, 4), round(exp(-eps * m), 4),
      round(fails / trials, 4))
```

It prints `60 0.0461 0.0498 0.046`: with 60 labels the exact failure chance \((0.95)^{60}\) is 4.6%, the bound \(e^{-3}\) is 5.0%, and 100,000 simulated training sequences fail 4.6% of the time, under \(\delta = 5\%\).

**Claim (the bad ERM).** Let \(\mathcal X = \{x_0, x_1, \dots, x_{n-1}\}\) have \(n \ge 2\) senders and let \(0 < \varepsilon \le 1/2\). Give the first sender most of the mail and spread the rest evenly:

\[ \mathcal D(x_0) = 1 - 2\varepsilon, \qquad \mathcal D(x_i) = \frac{2\varepsilon}{n - 1} \ \text{ for } 1 \le i \le n - 1, \]

and let the truth be \(h_\emptyset\). Then for every \(m \le (n-1)/2\), and in particular every \(m \le (n-1)/6\), *every* training sequence of size \(m\) leads \(A_{\mathrm{bad}}\) to a true risk of at least \(\varepsilon\). With \(m \le (n-1)/6\) it is at least \(5\varepsilon/3\).

\(\mathcal D(x_i)\) is the share of mail from sender \(x_i\); the shares add to \((1 - 2\varepsilon) + (n-1) \cdot 2\varepsilon/(n-1) = 1\). In plain words: **when most mail comes from one sender and the rest is spread thin, the bad ERM fails on every training sequence shorter than about half the number of senders.** With \(n = 1{,}000\) and \(\varepsilon = 0.05\), that is the all-staff sender with 90% of the mail and 999 senders at 0.01% each, the opening’s simulation.

*Proof.* With the truth \(h_\emptyset\), every label is \(\ast\), so \(A_{\mathrm{bad}}\) returns the folder of all unseen senders and errs on every message from an unseen sender. At most \(m\) of \(x_1, \dots, x_{n-1}\) can appear in \(m\) examples, so at least \(n - 1 - m\) of them are unseen, and

\[ L_{\mathcal D}(A_{\mathrm{bad}}(S)) \ge (n - 1 - m)\frac{2\varepsilon}{n - 1} = 2\varepsilon - \frac{2\varepsilon m}{n-1}. \]

With \(m \le (n-1)/2\) the right side is at least \(\varepsilon\); with \(m \le (n-1)/6\) it is at least \(2\varepsilon - \varepsilon/3 = 5\varepsilon/3\). This holds for every training sequence, so with probability 1. \(\blacksquare\)

Here it is with 1,000 senders, \(\varepsilon = 0.05\) and \(m = 166\), the largest whole number below \(999/6\), over 2,000 random training sequences:

```python
import random

n, eps = 1000, 0.05
p = [1 - 2 * eps] + [2 * eps / (n - 1)] * (n - 1)
m_max = (n - 1) // 6
rng, worst = random.Random(0), 1.0
for _ in range(2000):
    seen = set(rng.choices(range(n), weights=p, k=m_max))
    L_bad = sum(p[x] for x in range(n) if x not in seen)
    worst = min(worst, L_bad)
print(m_max, round(worst, 4), round(5 * eps / 3, 4))
```

It prints `166 0.0969 0.0833`: the best the bad ERM ever did was a true risk of 9.7%, above the proven floor of 8.3% and far above \(\varepsilon = 5\%\). The good ERM needed 60 labels; the bad one fails with 166, and with more senders it fails with more. Its sample complexity grows with the number of senders, while the good ERM’s does not depend on it at all.

[[diagram:good-bad-erm]]

### What the hungry folder teaches

Three lessons hide in those two claims, and they are worth slowing down for.

**Different ERMs can need wildly different numbers of labels.** In the two-label world this never happens: the fundamental theorem makes every ERM a learner with the same rate. With many labels, the choice of tie-breaking is part of the learner, and it can cost you a factor of the number of senders.

**Learnable does not mean learnable by every ERM.** Let \(\mathcal X\) be infinite, say all possible email addresses. The good ERM still learns with \(\lceil \log(1/\delta)/\varepsilon \rceil\) labels (the brackets round up), so the class is PAC learnable. But for any \(m\), pick \(n\) senders with \((n-1)/6 \ge m\) and the distribution of the bad-ERM claim on them: the bad ERM fails on every training sequence of size \(m\). So no number of labels makes it a learner. And this rules out uniform convergence too: by lesson 4, uniform convergence would make *every* ERM a learner. So here is a class that is learnable, by a perfectly natural ERM, and yet has no uniform convergence property. For two labels, lesson 6b proved that impossible: learnability, uniform convergence and “every ERM learns” were one property. With many labels they come apart, and they come apart even with the plain 0-1 loss.

**The \(\log k\) in the theorem is not a ghost.** The auto-folder class has Natarajan dimension 1, but with \(n\) senders it uses \(k = 2^n\) labels, so \(\log k = n\log 2\). The theorem’s realizable ceiling, \(C_2(d\log(kd/\varepsilon) + \log(1/\delta))/\varepsilon\), grows with \(n\), and the bad ERM really does need on the order of \(n\) labels. And with infinitely many senders the label set is infinite, \(\log k = \infty\), and the theorem says nothing, which is just as well, since the class lacks uniform convergence. The theorem covers every ERM, so it must pay for the worst one.

Can one always find a good ERM that avoids the \(\log k\)? It has been conjectured that yes: that every class \(\mathcal H \subseteq [k]^{\mathcal X}\) has realizable sample complexity

\[ m_{\mathcal H}(\varepsilon, \delta) = \tilde O\!\left( \frac{\mathrm{Ndim}(\mathcal H)}{\varepsilon} \right), \]

where \(\tilde O\) (“soft O”) means “at most a constant times this, up to factors that are powers of logarithms of \(1/\varepsilon\), \(1/\delta\) and \(\mathrm{Ndim}(\mathcal H)\)”, and, crucially, no factor of \(k\) at all. In plain words: **some learner should need only about the Natarajan dimension over the tolerance, however many folders there are.** For the auto-folder class that is about \(1/\varepsilon\) labels up to logs, and the good ERM’s 60 labels at \(\varepsilon = \delta = 0.05\) fit it. This lesson leaves it as an open question.

### Certificates for the auto-folder filter

Back to the launch. You want a label budget backed by a proof, with numbers and no unknown constants. Two proofs are available.

The first covers every ERM at once. With \(n\) senders the class is finite, so lesson 2’s **finite-class guarantee** applies: in the realizable case, every ERM has true risk at most \(\varepsilon\) with probability at least \(1 - \delta\) once \(m \ge \log(|\mathcal H|/\delta)/\varepsilon\). Natarajan’s lemma bounds \(|\mathcal H|\) by the dimension, so

\[ m \ge \frac{d \log n + 2d \log k + \log(1/\delta)}{\varepsilon} \]

certifies every ERM. Inside out: \(d\log n + 2d\log k\) is the log of Natarajan’s \(n^d k^{2d}\), \(\log(1/\delta)\) the price of confidence, and the division by \(\varepsilon\) the realizable rate. In plain words: **Natarajan’s count, put into the finite-class guarantee, certifies any ERM at all.** The second proof covers only the good ERM, at \(\log(1/\delta)/\varepsilon\). Here are both, for 1,000 senders, \(d = 1\), \(\varepsilon = \delta = 0.05\), along with the finite-class guarantee using the exact count \(|\mathcal H| = 2^{1000}\):

```python
from math import ceil, log

n, d, eps, delta = 1000, 1, 0.05, 0.05
log_k = n * log(2)
natarajan = d * log(n) + 2 * d * log_k
every_erm = ceil((natarajan + log(1 / delta)) / eps)
direct = ceil((n * log(2) + log(1 / delta)) / eps)
good = ceil(log(1 / delta) / eps)
print(round(natarajan, 1), every_erm, direct, good)
```

It prints `1393.2 27924 13923 60`. The log of Natarajan’s count is 1,393.2, so certifying every ERM costs 27,924 labels (13,923 with the exact count); certifying the good ERM costs 60. The difference is the price of not choosing your ERM.

## The algorithms as code

Two algorithms came out of this lesson: computing the Natarajan dimension of a finite class by brute force, and the two ERMs for auto-folders.

```
Shatters(H, C, labels):
  H_C ← { (h(x) for x in C) : h in H }
  for each choice of one unordered pair {a_x, b_x}
        of labels per message x in C:
    if every B ⊆ C has (a_x if x in B else b_x)_x in H_C:
      return true
  return false

Ndim(H, X, labels):
  d ← 0
  for r = 1, 2, ..., |X|:
    if no C ⊆ X with |C| = r has Shatters(H, C, labels):
      return d
    d ← r
  return d

GoodERM(S):
  if some (x, y) in S has y ≠ *: return h_y
  return h_∅

BadERM(S):
  if some (x, y) in S has y ≠ *: return h_y
  return h_A with A = senders not in S
```

Stopping at the first size with no shattered set is safe: a subset of a shattered set is shattered with the same candidates, so if no set of size \(r\) is shattered, none larger is either.

**Cost of Shatters.** Building \(\mathcal H_C\) takes \(O(|\mathcal H|\, r)\) time and space for a set of \(r\) messages (hashing tuples of length \(r\)). With \(k\) labels there are \(\binom{k}{2}\) unordered pairs per message, so \(\binom{k}{2}^r\) candidate choices, each checked against \(2^r\) mixtures at \(O(r)\) per lookup: \(O(|\mathcal H|\, r + \binom{k}{2}^r 2^r r)\) time in the worst case. **Ndim** calls it for every set of each size up to \(d + 1\): \(\sum_{r \le d+1} \binom{n}{r}\) sets, exponential in \(d\). That is fine for the tiny classes of this lesson and hopeless for your 300-folder filter, which is why the bounds of the previous section matter: for real classes we compute the dimension by proof, not by search.

**Cost of the ERMs.** Both scan \(S\) once, \(O(m)\) time. GoodERM stores nothing beyond the answer; BadERM stores the set of seen senders, \(O(m)\) space, and its returned rule is described by that set. Same cost, same training error, very different futures.

**Build it yourself.** Using only the Python standard library:

```python
from itertools import combinations, product

def levels(n, k):
    return [h for h in product(range(1, k + 1), repeat=n)
            if list(h) == sorted(h)]

def auto_folders(n):
    groups = [frozenset(g) for r in range(n + 1)
              for g in combinations(range(n), r)]
    return [tuple(A if x in A else "*" for x in range(n))
            for A in groups]

def shatters(H, C, labels): ...
def ndim(H, n, labels): ...
def good_erm(S, n): ...
def bad_erm(S, n): ...

if __name__ == "__main__":
    assert shatters(levels(3, 3), (0, 2), [1, 2, 3])
    assert not shatters(levels(3, 3), (0, 1, 2), [1, 2, 3])
    assert ndim(levels(5, 3), 5, [1, 2, 3]) == 2
    assert ndim(levels(5, 2), 5, [1, 2]) == 1
    H = auto_folders(4)
    labels = sorted({v for h in H for v in h}, key=str)
    assert ndim(H, 4, labels) == 1
    A = frozenset({1, 3})
    S = [(0, "*"), (2, "*"), (0, "*")]
    assert good_erm(S, 4) == ("*",) * 4
    assert bad_erm(S, 4) == ("*", A, "*", A)
    assert good_erm(S + [(3, A)], 4) == ("*", A, "*", A)
    assert bad_erm(S + [(3, A)], 4) == ("*", A, "*", A)
    print("ok")
```

1. Implement `shatters`, `ndim` (labels passed explicitly), `good_erm` and `bad_erm`. A hypothesis is a tuple of labels, one per message; a group label is a `frozenset`; `"*"` is the inbox. The ERMs return the rule as a tuple over senders `0..n-1`.
2. Reproduce the Natarajan dimensions 1, 2, 3 of urgency levels with two, three and four folders, and 1 for auto-folders on four senders.
3. Simulate both ERMs with 1,000 senders and the distribution of the bad-ERM claim at \(\varepsilon = 0.05\), for \(m = 60, 166, 500, 2{,}000\). At what \(m\) does the bad ERM first reach true risk \(\varepsilon\)? Compare with \((n-1)/2\).
4. Build one-versus-all over thresholds for \(k = 3\) on seven scores and compute its Natarajan dimension. Is it \(kd\)?

## Fix the bug

Two bugs, two fixes. The review gets its answer from the multiclass fundamental theorem: your 300-folder filter has Natarajan dimension between 8,671 and 9,000, so its sample complexity is sandwiched between constant multiples of the shapes printed earlier, with \(\log 300\) between floor and ceiling.

The auto-folder filter is yours to fix in the lab. There are 1,000 senders, the all-staff sender sends 90% of the mail, and your users have a budget of 120 labels. You must hit \(\varepsilon = \delta = 0.05\), and not by luck: you have to pick a label budget and an ERM whose true risk this lesson has *proved* is at most \(\varepsilon\) with probability at least \(1 - \delta\). First you will compute the Natarajan dimension of the auto-folder class, because the every-ERM certificate needs it. Then choose your ERM and your budget, and watch the folders.

[[lab-act2]]

## What people get wrong

**“With \(k\) labels, shattering means producing all \(k^{|C|}\) filings.”** That is far too strict: urgency levels never achieve “high score, low folder”, yet they are a perfectly learnable class. Shattering asks only for every mixture of *two* candidates per message, candidates you choose.

**“More folders means proportionally more labels.”** The number of folders enters the theorem only through \(\log k\), and for a good ERM perhaps not at all. Going from 300 folders to 3,000 multiplies \(\log k\) by about 1.4.

**“Any ERM will do.”** For two labels, yes, and that is lesson 6a’s theorem. With many labels, two ERMs with the same training error can differ by a factor of the number of senders in the labels they need, and with infinitely many labels one can learn while the other never does.

**“If a class is learnable, it has uniform convergence.”** The auto-folder class with infinitely many senders is learnable, by the good ERM, and has no uniform convergence property. The equivalence was a two-label fact.

## Recap and the next bug

1. With many labels, a set \(C\) is **shattered** when you can choose two different candidate folders for each message so that the class achieves every mixture of them. The **Natarajan dimension** is the size of the largest shattered set; with two labels it is the VC dimension.
2. **Natarajan’s lemma**: a class of Natarajan dimension \(d\) on \(n\) messages has at most \(n^d k^{2d}\) members, playing the role Sauer’s lemma played.
3. **The multiclass fundamental theorem**: finite Natarajan dimension \(d\) (with finitely many labels \(k\)) means learnable, with sample complexity between constant multiples of \((d + \log(1/\delta))/\varepsilon^2\) and \((d\log k + \log(1/\delta))/\varepsilon^2\) in the agnostic case, and similar shapes over \(\varepsilon\) in the realizable case.
4. **Calculations**: one-versus-all and every reduction with \(l\) filters of VC dimension \(d\) have Natarajan dimension at most \(3ld\log(ld)\); linear multiclass predictors in \(\mathbb R^d\) at most \(d\); the multivector construction between \((k-1)(n-1)\) and \(kn\). Roughly, the number of parameters.
5. **Good and bad ERMs**: for auto-folders, keeping everything in the inbox learns with \(\log(1/\delta)/\varepsilon\) labels, while foldering the unseen senders needs labels in proportion to the number of senders; with infinitely many senders the class is learnable but has no uniform convergence.

The filter is certified, and the auto-folders behave. Then the auditor from compliance stops by with a question you didn’t expect. She has been reading the archive of old filters, and she noticed something about the tightest-box filter, the one that draws the smallest axis-aligned rectangle around the junk examples. “You told me the box is learned from all your labelled messages,” she says. “But I can rebuild it from just four of them: the leftmost, rightmost, highest and lowest junk messages. The other examples don’t matter at all.” Then she asks: if a rule can be compressed to a handful of its own training examples, does that alone guarantee that it generalizes, with no VC dimension anywhere in sight? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Let \(\mathcal X\) have at least two messages and \(k \ge 2\). What is the Natarajan dimension of the class of *constant* rules, those that file every message in the same folder? Prove it, being explicit about which mixture fails for two messages. Then: what is the Natarajan dimension of the class of *all* functions \(\mathcal X \to [k]\) when \(|\mathcal X| = n\)?
2. In the proof of Natarajan’s lemma, explain in your own words why \(\mathrm{Ndim}(\mathcal H_{ij}) \le d - 1\). Which two candidates does the proof give the extra message \(x_0\), and why must they be different folders? What would break if \(\mathcal H_{ij}\) were defined as the traces completable with \(x_0 \mapsto i\) *or* with \(x_0 \mapsto j\)?
3. The company acquires a smaller one: now there are 600 senders, and you want \(\varepsilon = 0.05\) and \(\delta = 0.01\). (a) How many labels does the good-ERM claim require? (b) Up to what training size does the bad-ERM claim prove that the bad ERM fails on every training sequence, under its distribution, with a true risk of at least \(5\varepsilon/3\)? (c) How many labels does the finite-class guarantee with the exact count \(|\mathcal H| = 2^{600}\) require to certify every ERM?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (one-versus-all can be as small as one filter).** Let \(d, k > 0\). Show that there is a binary class \(\mathcal H_{\mathrm{bin}}\) of VC dimension \(d\) whose one-versus-all class \(\mathcal H^{\mathrm{OvA},k}_{\mathrm{bin}}\) has Natarajan dimension exactly \(d\).
- **Problem 2 (one-versus-all over halfspaces really is of order \(kd\)).** Let \(\mathcal H\) be the class of (non-homogeneous) halfspaces in \(\mathbb R^d\). Prove that \(\mathrm{Ndim}(\mathcal H^{\mathrm{OvA},k}) \ge (d-1)(k-1)\), in three steps. (a) Let \(\mathcal H_{\mathrm{discrete}}\) be the class of all functions \(f: [k-1] \times [d-1] \to \{0, 1\}\) for which there is some \(i_0\) such that, for every \(j \in [d-1]\), \(f(i, j) = 1\) for all \(i < i_0\) and \(f(i, j) = 0\) for all \(i > i_0\). Show that \(\mathrm{Ndim}(\mathcal H^{\mathrm{OvA},k}_{\mathrm{discrete}}) = (d-1)(k-1)\). (b) Show that \(\mathcal H_{\mathrm{discrete}}\) can be realized by halfspaces: there is a map \(\psi: [k-1] \times [d-1] \to \mathbb R^d\) with \(\mathcal H_{\mathrm{discrete}} \subseteq \{h \circ \psi : h \in \mathcal H\}\). (c) Conclude.
- **Problem 3 (the agnostic ceiling).** Adapt lesson 28’s proof of the binary fundamental theorem, with Natarajan’s lemma in place of Sauer’s lemma, to prove that for some universal constant \(C > 0\) and every class of Natarajan dimension \(d\), the agnostic sample complexity satisfies \(m_{\mathcal H}(\varepsilon, \delta) \le C \big(d\log(kd/\varepsilon) + \log(1/\delta)\big)/\varepsilon^2\).
