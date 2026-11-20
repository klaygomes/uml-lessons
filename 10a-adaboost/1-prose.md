# AdaBoost

## A shelf of weak rules

The mail room has moved on from yes/no features. The team now counts words. For every message the filter knows how many times it says “free”, “win”, “click”, “offer”, “cash”, “now”, “prize” and “urgent”, and, for good measure, “meeting” and “lunch”. Each count is capped at 5, so a message is a vector of ten whole numbers between 0 and 5.

You have twenty thousand labelled messages, and a deadline. Lesson 8 taught you what the deadline means: for most rich hypothesis classes, finding the hypothesis with the smallest empirical risk is computationally hopeless. So the team trained the only kind of rule it can train fast, a rule that looks at a single count and compares it with a threshold. “Junk if ‘prize’ appears more than three times.” “Junk if ‘cash’ appears more than three times.” There are a hundred or so of these rules, and the learner finds the best one in a second.

The best one is also not very good. On your twenty thousand messages, the best single rule is right 63% of the time. On fresh mail it does no better: its true risk is 0.375, so it is right five times in eight. Every other rule on the shelf is in the same range, except the ones built on “meeting” and “lunch”, which are no better than always saying keep. The inbox is still full of junk.

Here is the puzzle. You own a machine that can, in a second, produce a rule a little better than a coin. You need a rule that is right nearly always. Can a pile of rules that are each barely better than guessing be turned into one that is excellent? And if it can, what does the result cost you on fresh mail? This lesson answers the first, and the answer is a surprisingly confident yes. The second waits for lesson 10b.

## Decision stumps, and ten messages

The rules on the team’s shelf have a name. A **decision stump** looks at one feature, compares it with a threshold, and gives one label above the threshold and the other label at or below it.

Before we write one down, one change of convention. Until now a label was 1 for junk and 0 for keep. In this lesson it is convenient to write **junk as \(+1\) and keep as \(-1\)**, so \(\mathcal Y = \{-1, +1\}\). The reason will be clear soon: with labels \(\pm 1\), a hypothesis \(h\) is right on an example \((x, y)\) exactly when \(y\, h(x) = +1\) and wrong exactly when \(y\, h(x) = -1\). That single product does a lot of work in the proofs.

We also need a sign function that never returns 0. It takes a number and answers only “positive” (\(+1\)) or “not positive” (\(-1\)): 3 gives \(+1\), \(-2\) gives \(-1\), and 0 itself gives \(-1\). Throughout this lesson,
\[ \operatorname{sign}(u) = \begin{cases} +1 & \text{if } u > 0, \\ -1 & \text{if } u \le 0. \end{cases} \]
The big brace lists the cases: read each row as “the value on the left, if the condition on the right holds”. In plain words: **sign gives \(+1\) to a positive number and \(-1\) to everything else, zero included.**

A stump is a sign with a shift. “Junk if ‘free’ appears more than 0.5 times” subtracts 0.5 from the “free” count and asks for the sign: a message with one “free” gets \(\operatorname{sign}(1 - 0.5) = +1\), junk; a message with none gets \(\operatorname{sign}(-0.5) = -1\), keep.

**Definition (decision stump).** Let \(\mathcal X = \mathbb R^d\). For a feature index \(j \in [d]\), a threshold \(\theta \in \mathbb R\) and a sign \(b \in \{\pm 1\}\), the decision stump \(h_{j, \theta, b}\) is
\[ h_{j, \theta, b}(x) = b \cdot \operatorname{sign}(x_j - \theta). \]

- \(\mathbb R^d\) is the set of lists of \(d\) real numbers; a message \(x\) is such a list, one count per word, so here \(d\) is the number of words.
- \([d]\) is shorthand for the positions \(\{1, \dots, d\}\), and \(j \in [d]\) reads “\(j\) is one of them”: which word the stump looks at.
- \(x_j\), with the subscript, is entry \(j\) of the message: its count of word \(j\).
- \(\theta\) (theta) is the threshold, any real number; \(x_j - \theta\) is positive exactly when the count exceeds it.
- \(b \in \{\pm 1\}\) is a direction, \(+1\) or \(-1\), and multiplying by it either keeps the answer or flips it.
- The subscripts on \(h_{j, \theta, b}\) say which of the many stumps we mean: one for each choice of the three.

In plain words: **look at count \(j\), say \(b\) if it exceeds \(\theta\), and say \(-b\) otherwise.** With \(b = +1\) it says junk when count \(j\) exceeds \(\theta\) and keep otherwise; with \(b = -1\) it says the opposite. The class of all of them is \(\mathcal H_{DS} = \{h_{j, \theta, b} : j \in [d],\ \theta \in \mathbb R,\ b \in \{\pm 1\}\}\): the braces collect every stump such that (the colon) \(j\), \(\theta\) and \(b\) range over their allowed values.

Take message \(x = (1, 1)\), one “free” and one “click” (word 1 is “free”, word 2 is “click”). The stump \(h_{1, 0.5, +1}\) computes \(+1 \cdot \operatorname{sign}(1 - 0.5) = +1\): junk. The stump \(h_{2, 3.5, +1}\) computes \(\operatorname{sign}(1 - 3.5) = -1\): keep. Python counts positions from 0, so word 1 is `x[0]` below:

```python
def sign(u):
    return 1 if u > 0 else -1

def h(j, theta, b, x):
    return b * sign(x[j] - theta)

x4, x6 = (1, 1), (1, 4)
print(h(0, 0.5, 1, x4), h(1, 3.5, 1, x4), h(1, 3.5, 1, x6))
print(h(1, 3.5, -1, x6), sign(0))
```

It prints `1 -1 1` and `-1 -1`: the “click” stump keeps \((1, 1)\) and flags \((1, 4)\), flipping \(b\) flips its answer, and the sign of 0 is \(-1\).

[[diagram:stump]]

To have something we can follow with a pencil, take just two counts, “free” and “click”, and ten messages. In this toy world a message is junk exactly when its two counts add up to at least 4. The learner doesn’t know that.

| i | free | click | label |
|---|---|---|---|
| 1 | 0 | 0 | keep |
| 2 | 0 | 1 | keep |
| 3 | 0 | 3 | keep |
| 4 | 1 | 1 | keep |
| 5 | 2 | 0 | keep |
| 6 | 1 | 4 | junk |
| 7 | 2 | 4 | junk |
| 8 | 3 | 4 | junk |
| 9 | 4 | 0 | junk |
| 10 | 4 | 1 | junk |

No single stump gets all ten right, because the junk messages sit along a diagonal and a stump can only cut straight across one axis. How well can a stump do? Try “junk if free \(> 0.5\)”. It says junk for messages 4 to 10, so it is wrong on 4 and 5, which are keep. Two mistakes out of ten.

Soon we will score a stump not by counting mistakes, but by adding up how much each mistaken example *matters*. Give each of the ten messages a share of one unit of importance: say 0.25 each to messages 4 and 5, and \(1/16 = 0.0625\) each to the other eight, so that the shares add up to \(0.5 + 0.5 = 1\). Then “junk if free \(> 0.5\)”, which is wrong only on 4 and 5, scores \(0.25 + 0.25 = 0.5\) instead of 0.2: the two messages it misses now count for half of everything.

A **distribution over the examples** is a vector \(D = (D_1, \dots, D_m)\) with every \(D_i \ge 0\) and \(\sum_i D_i = 1\) (the capital sigma \(\sum\) means “add up”, here over every example \(i\)). As before, \(m\) is the number of examples, and example \(i\) is the pair \((x_i, y_i)\) of a message and its label. We call \(D_i\) the **weight** of example \(i\). The **weighted risk** of a hypothesis is the total weight of the examples it gets wrong:
\[ L_D(h) = \sum_{i=1}^m D_i\, \mathbb 1[h(x_i) \neq y_i]. \]

- \(h(x_i) \neq y_i\) asks whether \(h\) gets example \(i\) wrong.
- \(\mathbb 1[\cdot]\) is the indicator: it turns a yes/no question into a number, 1 for yes and 0 for no.
- \(D_i\, \mathbb 1[\dots]\) is therefore the weight of example \(i\) if \(h\) gets it wrong, and 0 if \(h\) gets it right.
- \(\sum_{i=1}^m\) adds that up over the examples, from \(i = 1\) to \(i = m\).

In plain words: **the total weight of the examples \(h\) gets wrong.** In the numbers above, the terms for messages 4 and 5 are \(0.25 \cdot 1\) each and every other term is \(0.0625 \cdot 0\), so \(L_D(h) = 0.5\). Recall the empirical risk \(L_S(h)\) from lesson 1, the fraction of the examples that \(h\) gets wrong. When every weight is \(1/m\), the weighted risk is just that, \(L_D(h) = L_S(h)\). (Here \(D\) lives on the \(m\) examples, not on all messages; we keep the letter because the weighted risk is literally the true risk under the distribution that picks example \(i\) with probability \(D_i\).)

With equal weights of 0.1, here is the weighted risk of every stump with \(b = +1\) and a threshold between two observed counts. The function `weighted_risk` mirrors the formula: it adds `D[i]` for exactly the examples where `h(X[i]) != y[i]`.

```python
X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
D = [0.1] * 10

def stump(j, theta, b):
    return lambda x: b if x[j] > theta else -b

def weighted_risk(h):
    return sum(D[i] for i in range(10) if h(X[i]) != y[i])

for theta in (0.5, 1.5, 2.5, 3.5):
    risks = [weighted_risk(stump(j, theta, 1)) for j in (0, 1)]
    print(theta, [round(r, 1) for r in risks])
```

It prints

```
0.5 [0.2, 0.4]
1.5 [0.2, 0.3]
2.5 [0.2, 0.3]
3.5 [0.3, 0.2]
```

The left number in each pair is the stump on “free”, the right one the stump on “click”. Four stumps tie at 0.2, and none does better. So on these ten messages, the best stump is right 80% of the time and can never be right 100% of the time. That is the bug in miniature.

## Weak learners

Recall what we asked of a learner in lesson 3. Fresh messages are drawn from a distribution \(\mathcal D\) and labelled by the true labelling \(f\); the true risk \(L_{(\mathcal D, f)}(h)\) is the chance that \(h\) disagrees with \(f\) on a fresh message; realizability means some hypothesis in \(\mathcal H\) has true risk 0; and the examples are **i.i.d.**, drawn independently from the same \(\mathcal D\). A class \(\mathcal H\) is PAC learnable if some algorithm, given enough such examples, at least a sample complexity \(m_{\mathcal H}(\varepsilon, \delta)\), returns a hypothesis with true risk at most \(\varepsilon\), with probability at least \(1 - \delta\), for every accuracy \(\varepsilon\), confidence \(\delta\), \(\mathcal D\) and \(f\) with realizability. In this lesson we call that **strong learning**, because \(\varepsilon\) can be made as small as you like.

Lesson 6’s fundamental theorem told us which classes are PAC learnable, and that ERM will do: recall from lesson 2 that ERM, empirical risk minimization, returns a hypothesis with the smallest empirical risk. But it said nothing about time, and lesson 8 showed that for some classes ERM is computationally hard even when realizability holds. So here is a tempting trade. What if we ask for much less accuracy, in return for speed? Not “true risk at most \(\varepsilon\)”, but merely “a little better than a coin”.

Here is that idea with the mail room’s numbers. A coin that says junk or keep at random is wrong half the time. The team’s best stump has true risk 0.375, so it beats the coin by \(0.5 - 0.375 = 0.125\). A weak learner is a learner that can always beat the coin by some fixed amount like that, called \(\gamma\) (gamma), whatever mail it is trained on.

**Definition (\(\gamma\)-weak learnability).** Let \(\gamma > 0\).

- A learning algorithm \(A\) is a **\(\gamma\)-weak learner** for a class \(\mathcal H\) if there is a function \(m_{\mathcal H}: (0, 1) \to \mathbb N\) such that for every \(\delta \in (0, 1)\), every distribution \(\mathcal D\) over \(\mathcal X\) and every labelling function \(f: \mathcal X \to \{\pm 1\}\), if realizability holds with respect to \(\mathcal H, \mathcal D, f\), then when \(A\) runs on \(m \ge m_{\mathcal H}(\delta)\) i.i.d. examples drawn from \(\mathcal D\) and labelled by \(f\), it returns a hypothesis \(h\) such that, with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(h) \le \tfrac12 - \gamma. \]
- A class \(\mathcal H\) is **\(\gamma\)-weak learnable** if a \(\gamma\)-weak learner for it exists.

The new pieces: \(m_{\mathcal H}: (0, 1) \to \mathbb N\) says the sample size is a function that takes a confidence \(\delta\) strictly between 0 and 1 and returns a whole number (\(\mathbb N\) is \(1, 2, 3, \dots\)); \(f: \mathcal X \to \{\pm 1\}\) says the labelling takes a message and returns junk or keep. The display reads: the true risk of the returned \(h\) is at most one half minus \(\gamma\). In plain words: **the hypothesis is wrong on fresh mail at most \(\gamma\) less often than a coin.** With the team’s stump, \(\gamma = 0.125\) and the promise is \(0.375 \le 0.5 - 0.125\), which holds with no room to spare.

Everything is as in strong learning except one thing: the accuracy is no longer yours to choose. It is pinned at \(\tfrac12 - \gamma\). The number \(\gamma\) is called the learner’s **edge**.

[[diagram:weak]]

Is weak learning statistically easier than strong learning? Surprisingly, no. Recall the VC dimension from lesson 6: the size of the largest set of points that a class can shatter, that is, label in every possible way. Lesson 6’s fundamental theorem gives a lower bound on the sample complexity of PAC learning a class of VC dimension \(d\), of the form \(m_{\mathcal H}(\varepsilon, \delta) \ge C_1 \frac{d + \log(1/\delta)}{\varepsilon}\) for an absolute constant \(C_1\) (here, and whenever it is a VC dimension, \(d\) is not the number of words). Read it from the inside: \(\log\) is the natural logarithm (as everywhere in this course), so \(\log(1/\delta)\) grows slowly as you ask for more confidence; the class’s VC dimension \(d\) is added to it; and dividing by \(\varepsilon\) makes the requirement grow as the accuracy gets finer. “Absolute constant” means a fixed number that depends on nothing else. In plain words: **you need at least about \((d + \log(1/\delta))/\varepsilon\) examples, up to a fixed factor.** With \(C_1 = 1\) for illustration, \(d = 3\) and \(\delta = 0.05\), the weak target \(\varepsilon = 0.375\) asks for at least 16 examples, the strong target \(\varepsilon = 0.01\) for about 600:

```python
import math

gamma = 0.5 - 0.375
print(gamma, round(0.5 - 1 / 12, 4))
C1, d, delta = 1, 3, 0.05
for eps in (0.5 - gamma, 0.01):
    m = C1 * (d + math.log(1 / delta)) / eps
    print(eps, round(m, 1))
```

It prints `0.125 0.4167` (the team’s edge, and the target \(\tfrac12 - \tfrac1{12}\) we will meet in the next section), then `0.375 16.0` and `0.01 599.6`. Weak learning needs fewer examples, but only by a constant factor. Now put \(\varepsilon = \tfrac12 - \gamma\) into the bound. If \(d = \infty\), no finite number of examples suffices even for that modest accuracy, so a class with infinite VC dimension is not \(\gamma\)-weak learnable either. If we ignore computation, weak learning is characterized by the VC dimension, exactly like strong learning.

The hope lies elsewhere. Maybe there is a weak learner that is *fast*, for a class where every strong learner is slow. A natural recipe is to pick a simple **base class** \(B\), such as decision stumps, and run ERM over \(B\) as the weak learner. (\(\mathrm{ERM}_B\), with the subscript, searches only the class \(B\), and \(\mathrm{ERM}_B(S)\) is the hypothesis it returns on the training sequence \(S\). With weights, it minimizes the weighted risk instead.) That works for \(\mathcal H\) if two things hold:

- \(\mathrm{ERM}_B\) can be computed efficiently, and
- for every training sequence labelled by some hypothesis in \(\mathcal H\), every \(\mathrm{ERM}_B\) hypothesis has risk at most \(\tfrac12 - \gamma\).

Notice that \(B\) and \(\mathcal H\) are different classes. The weak learner returns stumps, but the labels come from something richer. Let’s see this work on a small example before we ask the big question.

## A weak learner for three-piece rules

Leave the mail room for a moment and put messages on a line, \(\mathcal X = \mathbb R\), so a message is a single number, say its “free” count. Picture the rule “junk if the count is below 2 or above 5, keep from 2 to 5”: two cuts, three pieces, and the label flips at each cut. A count of 1 is junk, 3.5 is keep, 6 is junk. A **three-piece hypothesis** cuts the line at two points \(\theta_1 < \theta_2\) and alternates labels:
\[ h_{\theta_1, \theta_2, b}(x) = \begin{cases} +b & \text{if } x < \theta_1 \text{ or } x > \theta_2, \\ -b & \text{if } \theta_1 \le x \le \theta_2. \end{cases} \]
Reading the two rows: outside the interval from \(\theta_1\) to \(\theta_2\) the label is \(b\); inside it, cuts included, the label is \(-b\). In plain words: **one label on the two outer pieces, the other label on the middle piece.** The example above is \(\theta_1 = 2\), \(\theta_2 = 5\), \(b = +1\):

```python
from fractions import Fraction as Fr

def three_piece(t1, t2, b):
    return lambda x: -b if t1 <= x <= t2 else b

h = three_piece(2, 5, 1)
print([h(x) for x in (1, 2, 3.5, 5, 6)])
print(Fr(1, 3) + Fr(1, 24) + Fr(1, 24), Fr(1, 2) - Fr(1, 12))
```

It prints `[1, -1, -1, -1, 1]`: the cuts themselves belong to the middle piece. (The second line, `5/12 5/12`, is the arithmetic at the end of the proof below.)

Let \(\mathcal H\) be the class of all three-piece hypotheses, and let the base class be the stumps on the line, \(B = \{x \mapsto b \cdot \operatorname{sign}(x - \theta) : \theta \in \mathbb R,\ b \in \{\pm 1\}\}\). The arrow \(x \mapsto \dots\) reads “the function that sends \(x\) to \(\dots\)”, so \(B\) is the set of all functions \(b \cdot \operatorname{sign}(x - \theta)\), one per threshold and direction: the stumps of the last section with only one feature. A stump can’t express a three-piece rule. Yet:

**Claim.** \(\mathrm{ERM}_B\) is a \(\gamma\)-weak learner for \(\mathcal H\) with \(\gamma = 1/12\).

[[diagram:three-piece]]

*Proof.* Fix \(\mathcal D\) and \(f\) with realizability. Then some \(h^\star = h_{\theta_1, \theta_2, b} \in \mathcal H\) has \(L_{(\mathcal D, f)}(h^\star) = 0\), so \(f = h^\star\) except on a set of probability zero, and for any stump \(g\), \(L_{(\mathcal D, f)}(g) = \mathcal D(\{x : g(x) \neq h^\star(x)\})\). Here \(\mathcal D(A)\), for a set \(A\) of points, is the probability that a point drawn from \(\mathcal D\) lands in \(A\), and the set is every \(x\) such that \(g\) and \(h^\star\) disagree: the true risk of a stump is the weight of the region where it disagrees with the truth.

**Step 1: some stump is within \(1/3\) of perfect.** The two cuts split the line into three regions: \(R_1 = (-\infty, \theta_1)\), \(R_2 = [\theta_1, \theta_2]\) and \(R_3 = (\theta_2, \infty)\) (a round bracket leaves its endpoint out, a square one keeps it in, and \(\infty\) means the piece runs on forever). They are disjoint and cover the line, so \(\mathcal D(R_1) + \mathcal D(R_2) + \mathcal D(R_3) = 1\), and the smallest of the three weights is at most \(1/3\) (if all three exceeded \(1/3\), their sum would exceed 1). We now show that, for each region \(R_k\) and each \(\eta > 0\), some stump disagrees with \(h^\star\) on a set of probability at most \(\mathcal D(R_k) + \eta\).

- Region \(R_1\). The stump \(g = b \cdot \operatorname{sign}(x - \theta_2)\) gives \(+b\) for \(x > \theta_2\) and \(-b\) for \(x \le \theta_2\). It agrees with \(h^\star\) on \(R_2\) and \(R_3\), so it disagrees only on \(R_1\).
- Region \(R_3\). Take \(g = -b \cdot \operatorname{sign}(x - \theta)\) with \(\theta < \theta_1\). It gives \(+b\) for \(x \le \theta\) and \(-b\) for \(x > \theta\), so it disagrees with \(h^\star\) on \(R_3\) and on the sliver \((\theta, \theta_1)\). As \(\theta\) rises to \(\theta_1\), the slivers shrink to the empty set, and the probability of sets that shrink to nothing tends to 0 (continuity of probability). So some \(\theta\) makes the sliver weigh at most \(\eta\).
- Region \(R_2\). Take \(g = b \cdot \operatorname{sign}(x - \theta)\) with \(\theta\) far to the left. It gives \(+b\) except on \((-\infty, \theta]\), so it disagrees with \(h^\star\) on \(R_2\) and on the part of \((-\infty, \theta]\) outside \(R_2\). As \(\theta \to -\infty\) the sets \((-\infty, \theta]\) shrink to the empty set, so again some \(\theta\) makes their weight at most \(\eta\).

Applying this to the lightest region, for every \(\eta > 0\) there is a stump with \(L_{(\mathcal D, f)}(g) \le \tfrac13 + \eta\). (The \(\eta\) is needed because a stump on the whole line is never constant: it can only push its threshold further and further out.)

**Step 2: ERM over stumps finds one nearly that good.** The class \(B\) has VC dimension 2, which you can check the way lesson 6 checked thresholds. By lesson 6’s fundamental theorem, a class of finite VC dimension is agnostic PAC learnable by ERM. Recall from lesson 3 that agnostic means we no longer assume some hypothesis in the class is perfect; we only ask to come close to the best one the class has. We need the agnostic version here, because the labels come from \(h^\star\), which is usually not a stump. So there is a sample complexity \(m_B(\varepsilon, \delta)\) such that, on \(m \ge m_B(\varepsilon, \delta)\) examples, with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(\mathrm{ERM}_B(S)) \le \min_{g \in B} L_{(\mathcal D, f)}(g) + \varepsilon. \]
On the left is the true risk of the stump ERM returns on \(S\). On the right, \(\min_{g \in B}\) means “the smallest value over every stump \(g\) in \(B\)”: the true risk of the best stump there is. In plain words: **the stump ERM finds is at most \(\varepsilon\) worse on fresh points than the best stump.**

Now push numbers through it: set \(\varepsilon = 1/24\) and \(\eta = 1/24\). The minimum over \(B\) is at most \(\tfrac13 + \tfrac1{24}\), so with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(\mathrm{ERM}_B(S)) \le \tfrac13 + \tfrac1{24} + \tfrac1{24} = \tfrac{5}{12} = \tfrac12 - \tfrac1{12}. \]
The three terms are the lightest region, the sliver \(\eta\) and ERM’s slack \(\varepsilon\); the fractions printed `5/12 5/12` in the snippet above.
That is the definition of a \(\tfrac1{12}\)-weak learner, with \(m_{\mathcal H}(\delta) = m_B(1/24, \delta)\). \(\blacksquare\)

You can watch Step 1 happen. Below, each trial puts 30 random points on a line, gives them random weights, labels them with a random three-piece hypothesis, and finds the best stump by brute force. On a finite set of points, a threshold below all of them is a perfectly good constant stump, so here the \(\eta\) disappears.

```python
import random

rng, worst = random.Random(0), 0.0
for _ in range(2000):
    xs = sorted(rng.uniform(0, 10) for _ in range(30))
    D = [rng.random() for _ in xs]
    D = [d / sum(D) for d in D]
    t1, t2 = sorted(rng.uniform(0, 10) for _ in range(2))
    y = [-1 if t1 <= x <= t2 else 1 for x in xs]
    stumps = [(th, b) for th in [-1] + xs for b in (1, -1)]
    best = min(sum(d for x, d, l in zip(xs, D, y)
                   if (b if x > th else -b) != l)
               for th, b in stumps)
    worst = max(worst, best)
print(round(worst, 4))
```

It prints `0.3242`. In two thousand random worlds, the best stump never did worse than about a third, just as the proof promises.

## Finding the best stump fast

The weak learner has to be fast, or the whole plan is pointless. So how do you find the stump with the smallest weighted risk? Let’s make it concrete for \(b = +1\), the stumps that say junk when count \(j\) exceeds \(\theta\). A stump like that can go wrong in two ways. It lets junk through when a junk message’s count is at or below the threshold, and it flags a keep message when its count is above. Take “junk if free \(> 2.5\)” with the weights from before (0.25 on messages 4 and 5, \(1/16\) on the rest). It lets through junk messages 6 and 7, whose “free” counts are 1 and 2, and flags no keep message, since none has more than two “free”s. Its weighted risk is \(1/16 + 1/16 = 0.125\). As a formula,
\[ F_j(\theta) = \sum_{i : y_i = +1} D_i\, \mathbb 1[x_{i,j} \le \theta] + \sum_{i : y_i = -1} D_i\, \mathbb 1[x_{i,j} > \theta]. \]

- \(x_{i,j}\), with two subscripts, is count \(j\) of message \(i\): the message first, the word second.
- \(\sum_{i : y_i = +1}\) adds up over the junk examples only (the colon again reads “such that”), and \(\sum_{i : y_i = -1}\) over the keep examples only.
- In the first sum, \(\mathbb 1[x_{i,j} \le \theta]\) is 1 when a junk example is at or below the threshold, so the stump calls it keep: a miss.
- In the second sum, \(\mathbb 1[x_{i,j} > \theta]\) is 1 when a keep example is above the threshold, so the stump calls it junk: a false alarm.
- \(F_j\) is named after the feature \(j\) and read as a function of the threshold \(\theta\).

In plain words: **the weight of the junk the stump lets through plus the weight of the keep mail it flags.** What about \(b = -1\)? That stump flips every prediction, so it is wrong exactly where the \(b = +1\) stump is right, and its weighted risk is \(1 - F_j(\theta)\), because the weights add up to 1. For “free” at 2.5 that is \(1 - 0.125 = 0.875\). Here is \(F_j\), one sum per line:

```python
X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [-1] * 5 + [1] * 5
D = [1 / 16] * 3 + [1 / 4] * 2 + [1 / 16] * 5

def F(j, th):
    let = sum(d for x, l, d in zip(X, y, D)
              if l == 1 and x[j] <= th)
    flag = sum(d for x, l, d in zip(X, y, D)
               if l == -1 and x[j] > th)
    return let + flag

print(F(0, 2.5), 1 - F(0, 2.5), F(0, 0.5), F(0, 1.5))
```

It prints `0.125 0.875 0.5 0.3125`. The third number is “junk if free \(> 0.5\)”, round 1’s stump, now flagging the two heavy keep messages; the last one returns below. So one function covers both signs, and we want
\[ \min_{j \in [d]} \ \min_{\theta \in \mathbb R} \ \min\{F_j(\theta),\, 1 - F_j(\theta)\}. \]
Read it from the inside out: for one feature and one threshold, take the better of the two directions; then the best threshold for that feature; then the best feature. In plain words: **the smallest weighted risk of any stump at all.**

**There are only \(m + 1\) thresholds worth trying.** Fix \(j\) and sort the examples so that \(x_{1,j} \le x_{2,j} \le \dots \le x_{m,j}\). A threshold \(\theta\) affects the predictions only through the set of examples with \(x_{i,j} \le \theta\), and that set is always a prefix of the sorted list. On the “free” counts of the ten messages, sorted as 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, the thresholds 2.2 and 2.7 both put the first seven at or below, so they make the same predictions. Any two thresholds that give the same prefix give the same predictions on \(S\). So it is enough to try one threshold per prefix:
\[ \Theta_j = \{x_{1,j} - 1\} \cup \left\{ \tfrac{x_{i,j} + x_{i+1,j}}{2} : i \in [m-1] \right\} \cup \{x_{m,j} + 1\}. \]
The symbol \(\cup\) is “union”, putting sets together. The first set is one threshold below the smallest value, the middle set is the midpoint between each pair of neighbours in the sorted list (for \(i\) from 1 to \(m - 1\)), and the last is one threshold above the largest. In plain words: **one threshold below everything, one between each pair of neighbours, and one above everything.**

```python
X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [-1] * 5 + [1] * 5
D = [1 / 16] * 3 + [1 / 4] * 2 + [1 / 16] * 5

def F(j, th):
    return sum(d for x, l, d in zip(X, y, D)
               if (1 if x[j] > th else -1) != l)

def Theta(j):
    v = sorted(x[j] for x in X)
    mids = [(a + b) / 2 for a, b in zip(v, v[1:])]
    return sorted({v[0] - 1, *mids, v[-1] + 1})

print(Theta(0))
print(min(min(F(j, t), 1 - F(j, t))
          for j in (0, 1) for t in Theta(j)))
```

It prints `[-1, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5]` and `0.125`. Equal neighbours give a midpoint equal to their value, which only repeats a prefix that another threshold already covers, and the set keeps each value once. The best stump under these weights has weighted risk 0.125, the one round 2 of AdaBoost will pick below.

Trying every \(\theta \in \Theta_j\) for every \(j\), and summing over \(m\) examples each time, costs \(O(dm^2)\). (Recall the big-O notation from lesson 2: \(O(dm^2)\) means at most a fixed multiple of \(d \cdot m^2\) steps.) There is a neat way to save a factor of \(m\).

**Each step costs \(O(1)\).** Suppose we know \(F = F_j(\theta)\) for a threshold just below \(x_{i,j}\), and now move to a threshold \(\theta'\) just above it. Only example \(i\) changes sides: it was above the threshold, predicted junk, and now it is at or below it, predicted keep. If \(y_i = +1\), it was right and is now wrong, so \(F\) grows by \(D_i\). If \(y_i = -1\), it was wrong and is now right, so \(F\) shrinks by \(D_i\). Both cases in one line:
\[ F_j(\theta') = F_j(\theta) + y_i D_i. \]
The label \(y_i\) is \(+1\) or \(-1\), so the product \(y_i D_i\) is the weight with a sign: added for junk, subtracted for keep. In plain words: **sliding the threshold past one example changes the risk by that example’s weight, up for junk and down for keep.** Under the round-2 weights, slide “free” from 0.5 to 1.5: messages 4 (keep, 0.25) and 6 (junk, \(1/16\)) cross, so \(F\) goes from 0.5 to \(0.5 - 0.25 + 0.0625 = 0.3125\), the last number the \(F_j\) snippet printed.

[[diagram:sweep]]

The sweep starts with a threshold below everything, where every example is predicted junk and \(F\) is the total weight of the keep examples, and then walks up the sorted list. Here the sweep and brute force meet on 200 random examples of one feature. The line `F += ...` is the update \(F_j(\theta') = F_j(\theta) + y_i D_i\), applied by `groupby` to a whole group of equal counts at once, since ties cross the threshold together.

```python
import itertools, random

def risk(th, X, y, D):
    return sum(d for x, l, d in zip(X, y, D)
               if (1 if x > th else -1) != l)

rng = random.Random(1)
X = [rng.randint(0, 9) for _ in range(200)]
y = [rng.choice((-1, 1)) for _ in X]
D = [1 / 200] * 200
brute = min(risk(th, X, y, D) for th in range(-1, 10))
F = best = risk(-1, X, y, D)
rows = sorted(zip(X, y, D))
for x, group in itertools.groupby(rows, key=lambda r: r[0]):
    F += sum(l * d for _, l, d in group)
    best = min(best, F)
print(round(brute, 3), round(best, 3))
```

It prints `0.44 0.44`. Random labels, so the best stump is barely better than a coin, and both methods find the same one.

Here is the whole weak learner.

```
ERM-Stumps(S, D):
  input: training sequence S = (x_1, y_1), ..., (x_m, y_m)
         distribution vector D
  goal:  j*, θ*, b* minimizing the weighted risk
  F* ← ∞
  for j = 1, ..., d:
    sort S by coordinate j, so x_{1,j} ≤ ... ≤ x_{m,j},
      and set x_{m+1,j} ← x_{m,j} + 1
    F ← Σ_{i : y_i = −1} D_i
    if min(F, 1 − F) < F*:
      F* ← min(F, 1 − F), θ* ← x_{1,j} − 1, j* ← j
      b* ← +1 if F ≤ 1 − F else −1
    for i = 1, ..., m:
      F ← F + y_i D_i
      if min(F, 1 − F) < F* and x_{i,j} ≠ x_{i+1,j}:
        F* ← min(F, 1 − F)
        θ* ← (x_{i,j} + x_{i+1,j}) / 2, j* ← j
        b* ← +1 if F ≤ 1 − F else −1
  output j*, θ*, b*
```

**Cost.** Sorting each coordinate costs \(O(m \log m)\), so one call costs \(O(dm \log m)\) time. But notice that the sorted order of coordinate \(j\) does not depend on \(D\). Boosting will call this learner again and again on the same examples with different weights, so you sort once, keep the \(d\) sorted index lists, and every later call costs only \(O(dm)\) time. The price is \(O(dm)\) space for the index lists.

**Build it yourself (part 1).** Using only the Python standard library:

```python
X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [1 if a + b >= 4 else -1 for a, b in X]

def erm_stump(X, y, D): ...

if __name__ == "__main__":
    eps, stump = erm_stump(X, y, [0.1] * 10)
    assert abs(eps - 0.2) < 1e-9 and stump == (0, 0.5, 1)
    D1 = [1 / 16] * 3 + [1 / 4] * 2 + [1 / 16] * 5
    eps, stump = erm_stump(X, y, D1)
    assert abs(eps - 0.125) < 1e-9 and stump == (0, 2.5, 1)
```

1. Implement `erm_stump`, returning the weighted risk and `(j, theta, b)` with the stump’s convention from this lesson. Break ties by keeping the first candidate found, in the order of the pseudocode.
2. Write a brute-force version that tries every \(j\), every \(\theta \in \Theta_j\) and both signs, and assert that the two agree on 300 random examples with three features and counts from 0 to 9.
3. Time both for \(m\) = 1,000, 2,000 and 4,000. Predict the ratios before you run them.

## Reweighting the mail

Now the big idea. Run the stump learner once and you get a stump with weighted risk 0.2 on the ten messages. Run it again on the same messages with the same weights and you get the same stump. Useless. But run it again with *different* weights, ones that make the first stump’s mistakes expensive, and the learner is forced to look somewhere else.

That is **AdaBoost**, short for adaptive boosting. It works in **rounds**. In round \(t\) it holds a distribution \(D^{(t)}\) over the examples, hands \(D^{(t)}\) and \(S\) to the weak learner, and gets back a hypothesis \(h_t\) with weighted risk
\[ \varepsilon_t = L_{D^{(t)}}(h_t) = \sum_{i=1}^m D_i^{(t)}\, \mathbb 1[h_t(x_i) \neq y_i]. \]
The superscript \((t)\) in \(D^{(t)}\) and \(D_i^{(t)}\) is a round number, not a power: \(D_i^{(t)}\) is the weight of example \(i\) in round \(t\). The subscript \(t\) on \(\varepsilon_t\) and \(h_t\) also names the round. The formula is the weighted risk from the stumps section, with this round’s weights: in round 1 every weight is 0.1 and the first stump misses two messages, so \(\varepsilon_1 = 0.2\).

A stump with a small weighted risk should count for more in the final decision than one that is barely better than a coin. With \(\varepsilon_1 = 0.2\), the stump is right four times as often as it is wrong, so \(1/\varepsilon_1 - 1 = 4\) measures that; a coin with \(\varepsilon = 0.5\) gets \(1/0.5 - 1 = 1\). AdaBoost gives \(h_t\) a **vote**
\[ w_t = \tfrac12 \log\!\left(\frac{1}{\varepsilon_t} - 1\right). \]
Inside the brackets is that odds ratio, right versus wrong. The natural logarithm \(\log\) turns ratio 1 into 0 and makes larger ratios positive; the \(\tfrac12\) is a scale that the proof will explain. In plain words: **half the log of how many times more often \(h_t\) is right than wrong.** Round 1 gives \(\tfrac12 \log 4 = \log 2 \approx 0.693\), and a coin gets \(\tfrac12 \log 1 = 0\). So the vote is large when \(\varepsilon_t\) is small and shrinks to 0 as \(\varepsilon_t\) approaches \(\tfrac12\). (Two words, two things: a *weight* belongs to an example, a *vote* belongs to a hypothesis.)

Then it moves weight onto the examples \(h_t\) got wrong. Here \(e\) is Euler’s number, about 2.718, and \(e^{u}\), also written \(\exp(u)\), is always positive, above 1 for positive \(u\) and below 1 for negative \(u\). An example that \(h_t\) gets right has \(y_i h_t(x_i) = +1\), so its weight is multiplied by \(e^{-w_t}\) and shrinks. An example it gets wrong has \(y_i h_t(x_i) = -1\), so its weight is multiplied by \(e^{w_t}\) and grows. Dividing by the total keeps it a distribution. In round 1, \(e^{-0.693} = 1/2\) and \(e^{0.693} = 2\). After \(T\) rounds, the final hypothesis is a weighted vote of everything it has collected. In the pseudocode, \(\Sigma_j\) in the denominator is the same total, written with another letter because \(i\) is already in use.

```
AdaBoost(S, WL, T):
  input: training sequence S = (x_1, y_1), ..., (x_m, y_m)
         weak learner WL
         number of rounds T
  initialize D^(1) = (1/m, ..., 1/m)
  for t = 1, ..., T:
    h_t ← WL(D^(t), S)
    ε_t ← Σ_i D_i^(t) · 1[y_i ≠ h_t(x_i)]
    w_t ← (1/2) log(1/ε_t − 1)
    for i = 1, ..., m:
      D_i^(t+1) ← D_i^(t) exp(−w_t y_i h_t(x_i))
                  / Σ_j D_j^(t) exp(−w_t y_j h_t(x_j))
  output h_s(x) = sign(Σ_{t=1}^T w_t h_t(x))
```

Why hand the weak learner a distribution over \(S\) rather than fresh examples? Because a distribution over the examples is a perfectly good \(\mathcal D\): the weak learner could draw i.i.d. examples from it if it wanted, and its guarantee applies to every distribution. For stumps it doesn’t need to draw at all, since it can minimize the weighted risk exactly.

Let’s run three rounds on the ten messages by hand. Watch what happens to messages 4 and 5, and then to 6 and 7.

**Round 1.** All weights are 0.1. The learner returns “junk if free \(> 0.5\)” (the first of the four tied stumps). It is wrong on messages 4 and 5, so \(\varepsilon_1 = 0.2\) and \(w_1 = \tfrac12 \log 4 = \log 2 \approx 0.693\). The eight correct messages are multiplied by \(e^{-w_1} = 1/2\) and the two wrong ones by \(e^{w_1} = 2\). That gives \(0.05\) eight times and \(0.2\) twice, which add up to 0.8, so dividing by 0.8 gives the new weights: \(1/16\) for the eight, \(1/4\) for messages 4 and 5.

**Round 2.** Under the new weights, “junk if free \(> 0.5\)” would now have weighted risk \(1/2\). The learner returns “junk if free \(> 2.5\)” instead. It gets 4 and 5 right but misses 6 and 7, which are junk with low “free” counts. So \(\varepsilon_2 = 2/16 = 0.125\) and \(w_2 = \tfrac12 \log 7 \approx 0.973\).

**Round 3.** The update multiplies 6 and 7 by \(e^{w_2} = \sqrt 7\) and the rest by \(1/\sqrt 7\). After normalizing, 6 and 7 carry \(1/4\) each, messages 4 and 5 carry \(1/7\) each, and the other six carry \(1/28\) each (check: \(1/2 + 2/7 + 6/28 = 1\)). Now 6 and 7 are heavy, and the learner returns “junk if click \(> 3.5\)”, which misses only 9 and 10. Their weights are \(1/28\) each, so \(\varepsilon_3 = 1/14 \approx 0.071\) and \(w_3 = \tfrac12 \log 13 \approx 1.282\).

The vote after three rounds is
\[ h_s(x) = \operatorname{sign}\big(0.693\, h_1(x) + 0.973\, h_2(x) + 1.282\, h_3(x)\big), \]
and it gets all ten messages right. Read it from the inside out: each \(h_t(x)\) is one stump’s answer, \(+1\) or \(-1\); multiplying by its vote and adding gives a score; the sign of the score is the label. In plain words: **each stump votes junk or keep with its own strength, and the side with more total strength wins.** Message 6, \((1, 4)\), gets \(+1, -1, +1\) from the three stumps, so its score is \(0.693 - 0.973 + 1.282 = 1.003 > 0\): junk, correctly. Message 4, \((1, 1)\), gets \(+1, -1, -1\) and the score \(-1.562\): keep.

```python
import math

w = [math.log(2), math.log(7) / 2, math.log(13) / 2]
hs = [lambda x: 1 if x[0] > 0.5 else -1,
      lambda x: 1 if x[0] > 2.5 else -1,
      lambda x: 1 if x[1] > 3.5 else -1]
for x in [(1, 4), (1, 1)]:
    psi = [ht(x) for ht in hs]
    s = sum(wt * p for wt, p in zip(w, psi))
    print(x, psi, round(s, 3), 1 if s > 0 else -1)
```

It prints `(1, 4) [1, -1, 1] 1.003 1` and `(1, 1) [1, -1, -1] -1.562 -1`. Three stumps, each wrong on two of ten messages, combine into a rule that is wrong on none. Here is the whole run, printing \(\varepsilon_t\), \(w_t\), the empirical risk of the vote so far, and the weight of message 4 after the update; `u` holds the weights times \(e^{-w_t y_i h_t(x_i)}\), and dividing by `sum(u)` normalizes:

```python
import math

X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [1 if a + b >= 4 else -1 for a, b in X]
D, f = [0.1] * 10, [0.0] * 10
for j, theta in [(0, 0.5), (0, 2.5), (1, 3.5)]:
    h = [1 if x[j] > theta else -1 for x in X]
    eps = sum(d for d, hi, yi in zip(D, h, y) if hi != yi)
    w = 0.5 * math.log(1 / eps - 1)
    f = [fi + w * hi for fi, hi in zip(f, h)]
    u = [d * math.exp(-w * yi * hi) for d, yi, hi in zip(D, y, h)]
    D = [ui / sum(u) for ui in u]
    L_S = sum((1 if fi > 0 else -1) != yi for fi, yi in zip(f, y))
    print(round(eps, 3), round(w, 3), L_S / 10, round(D[3], 4))
```

It prints

```
0.2 0.693 0.2 0.25
0.125 0.973 0.2 0.1429
0.071 1.282 0.0 0.0769
```

Two things deserve a second look. First, the empirical risk did not go down after round 2: it stayed at 0.2. The vote of the first two stumps happens to say keep for messages 4 to 7, getting 4 and 5 right but now 6 and 7 wrong. AdaBoost’s empirical risk is not guaranteed to fall every round; what is guaranteed to fall is a smooth quantity sitting above it, the heart of the proof. Second, look at round 2’s weights: right after the update, messages 4 and 5, the ones round 1 got wrong, carry exactly half the total weight. Coincidence? Problem 1 at the end asks you to prove it never is.

[[diagram:reweight]]

Now play with it yourself. In the lab, you step AdaBoost round by round, first on the ten messages and then on the twenty thousand, and watch the weights move.

[[lab]]

## Why the empirical risk falls exponentially

Here is the theorem that makes boosting worth doing. First the idea in numbers. On the ten messages, the worst round had \(\varepsilon_1 = 0.2\), so every round beat the coin by at least \(\gamma = 0.3\). The theorem says that each such round multiplies a ceiling on the empirical risk by at most \(e^{-2 \cdot 0.3^2} = e^{-0.18} \approx 0.835\), starting from 1. Three rounds give a ceiling of \(0.835^3 \approx 0.583\).

**Theorem (AdaBoost drives the empirical risk down exponentially).** Let \(S\) be a training sequence, and suppose that in every round of AdaBoost the weak learner returns a hypothesis with \(\varepsilon_t \le \tfrac12 - \gamma\). Then the hypothesis \(h_s\) that AdaBoost outputs after \(T\) rounds satisfies
\[ L_S(h_s) = \frac1m \sum_{i=1}^m \mathbb 1[h_s(x_i) \neq y_i] \le e^{-2\gamma^2 T}. \]

The middle is the empirical risk written out: the indicator counts each mistake of the vote \(h_s\) as 1, the sum adds them, and \(\tfrac1m\) turns the count into a fraction. On the right, the exponent \(-2\gamma^2 T\) is minus twice the squared edge times the number of rounds, so the bound is \(e^{-2\gamma^2}\) multiplied by itself \(T\) times. In plain words: **if every round beats a coin by \(\gamma\), the fraction of examples the vote gets wrong is at most \(e^{-2\gamma^2 T}\), which shrinks by a fixed factor every round.**

```python
import math

gamma, T, m = 0.5 - 0.2, 3, 10
print(round(math.exp(-2 * gamma ** 2 * T), 3))
T0 = math.floor(math.log(m) / (2 * gamma ** 2)) + 1
print(T0, round(math.exp(-2 * gamma ** 2 * T0), 4))
```

It prints `0.583`, the ceiling after three rounds, and `13 0.0963`: by round 13 the ceiling would be below \(1/10\). The empirical risk is a multiple of \(1/m\), so once the bound drops below \(1/m\) it must be exactly 0: on ten messages, zero mistakes. (Our run got there in three, because its later rounds had bigger edges than 0.3.) Notice how strong this is: an edge of \(\gamma\) per round, on whatever weights the learner is handed, buys an exponential fall.

We also need the rounds to be well defined. If \(\varepsilon_t = 0\), the vote \(w_t\) would be infinite; in that case the weak learner has found a hypothesis with no mistakes at all, and AdaBoost can simply stop and return it. So assume \(0 < \varepsilon_t \le \tfrac12 - \gamma\), which makes every vote finite and positive.

*Proof.* We follow the score of the vote as it builds up: after one round on the ten messages it is \(0.693\, h_1(x)\), after two \(0.693\, h_1(x) + 0.973\, h_2(x)\). For each \(t\), let
\[ f_t = \sum_{p \le t} w_p h_p, \]
where \(\sum_{p \le t}\) adds the voted stumps of rounds \(p = 1\) up to \(t\), so \(f_t(x)\) is the number whose sign is the vote. In plain words: **\(f_t\) is the running total of the voted stumps.** The symbol \(\equiv\) means “equal for every input”, so \(f_0 \equiv 0\) says no rounds give score 0 everywhere, and AdaBoost outputs \(h_s = \operatorname{sign}(f_T)\).

Now the smooth quantity. The product \(y_i f_t(x_i)\) is positive when the score points the right way and negative when it points the wrong way, and its size says how confidently. Feed it through \(e^{-u}\): a confidently right example contributes a small number, a wrong one a number above 1. After round 1, the eight messages the stump gets right have \(y_i f_1(x_i) = 0.693\) and contribute \(e^{-0.693} = 0.5\); messages 4 and 5 have \(-0.693\) and contribute 2. Define
\[ Z_t = \frac1m \sum_{i=1}^m e^{-y_i f_t(x_i)}. \]
In plain words: **the average, over the examples, of \(e\) to the minus signed score.** After round 1 it is \((8 \cdot 0.5 + 2 \cdot 2)/10 = 0.8\):

```python
import math

X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [-1] * 5 + [1] * 5
w1 = math.log(2)
f1 = [w1 * (1 if x[0] > 0.5 else -1) for x in X]
e = [math.exp(-yi * fi) for yi, fi in zip(y, f1)]
print([round(v, 2) for v in e])
print(round(sum(e) / 10, 3), [v / sum(e) for v in e[2:5]])
```

It prints `[0.5, 0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5]` and `0.8 [0.0625, 0.25, 0.25]`. The second list, each term divided by the total, is round 2’s weights for messages 3, 4 and 5; Step 3 below proves that this is no accident. The plan has four steps: show \(Z_T\) sits above the empirical risk, split \(Z_T\) into a product of per-round ratios, compute each ratio exactly, and bound it.

**Step 1: the empirical risk is at most \(Z_T\).** Take one example. If \(h_s\) gets it wrong, then \(\operatorname{sign}(f_T(x_i)) \neq y_i\). If \(y_i = +1\), that means \(f_T(x_i) \le 0\); if \(y_i = -1\), it means \(f_T(x_i) > 0\). Either way \(y_i f_T(x_i) \le 0\), so \(e^{-y_i f_T(x_i)} \ge 1\). If \(h_s\) gets it right, the indicator is 0, and an exponential is always positive. So for every example
\[ \mathbb 1[h_s(x_i) \neq y_i] \le e^{-y_i f_T(x_i)}. \]
In plain words: **each example’s mistake count, 0 or 1, is at most its term in \(Z_T\).** After round 1, message 4 is a mistake, \(1 \le 2\), and message 1 is not, \(0 \le 0.5\). Average over \(i\): \(L_S(h_s) \le Z_T\). (This also settles any worry about ties: our sign sends 0 to \(-1\), but the inequality holds whichever label a tie gets, because \(e^0 = 1\).) So it suffices to show \(Z_T \le e^{-2\gamma^2 T}\).

**Step 2: a telescoping product.** Since \(f_0 \equiv 0\), every term of \(Z_0\) is \(e^0 = 1\), so \(Z_0 = 1\). All the \(Z_t\) are positive, so we may write
\[ Z_T = \frac{Z_T}{Z_0} = \frac{Z_T}{Z_{T-1}} \cdot \frac{Z_{T-1}}{Z_{T-2}} \cdots \frac{Z_1}{Z_0}. \]
Each intermediate \(Z\) appears once on top and once below, and cancels; the dots \(\cdots\) stand for the factors in between. In plain words: **the final \(Z\) is the product of how much \(Z\) shrank in each round.** On the ten messages, the three factors will turn out to be 0.8, 0.6614 and 0.5151, and their product is \(Z_3 \approx 0.273\). So it suffices to show that for every round,
\[ \frac{Z_{t+1}}{Z_t} \le e^{-2\gamma^2}, \]
that every round shrinks \(Z\) by a factor of at most \(e^{-2\gamma^2}\) (about 0.835 for \(\gamma = 0.3\)), because multiplying \(T\) such factors gives \(e^{-2\gamma^2 T}\).

[[diagram:telescope]]

**Step 3: the weights are the exponentials, normalized.** We claim that for every \(t \ge 0\) and every \(i\),
\[ D_i^{(t+1)} = \frac{e^{-y_i f_t(x_i)}}{\sum_{j=1}^m e^{-y_j f_t(x_j)}}. \]
The top is example \(i\)’s term in \(Z_t\); the bottom adds the same term over all examples (with letter \(j\), since \(i\) is taken). In plain words: **the next round’s weight of an example is its share of the total \(e^{-y f_t}\).** That is what the snippet above printed: \(0.5/4 = 0.0625\) and \(2/4 = 0.25\). We prove it by induction on \(t\). For \(t = 0\), \(f_0 \equiv 0\), so the right side is \(1/m\), which is \(D^{(1)}\). Now suppose it holds for \(t - 1\), that is, \(D_i^{(t)} = e^{-y_i f_{t-1}(x_i)} / N_{t-1}\), where \(N_{t-1}\) is the sum in the denominator. AdaBoost’s update multiplies \(D_i^{(t)}\) by \(e^{-w_t y_i h_t(x_i)}\) and renormalizes. Since \(f_t = f_{t-1} + w_t h_t\),
\[ D_i^{(t)} e^{-w_t y_i h_t(x_i)} = \frac{e^{-y_i f_{t-1}(x_i)}\, e^{-y_i w_t h_t(x_i)}}{N_{t-1}} = \frac{e^{-y_i f_t(x_i)}}{N_{t-1}}. \]
So the unnormalized new weights are proportional to \(e^{-y_i f_t(x_i)}\), with the same constant \(1/N_{t-1}\) for every \(i\). Normalizing them to sum to 1 cancels that constant and divides by \(\sum_j e^{-y_j f_t(x_j)}\), which is the claim for \(t\).

**Step 4: compute one ratio exactly.** Use \(f_{t+1} = f_t + w_{t+1} h_{t+1}\), then Step 3:
\[
\begin{aligned}
\frac{Z_{t+1}}{Z_t}
&= \frac{\sum_{i} e^{-y_i f_{t+1}(x_i)}}{\sum_{j} e^{-y_j f_t(x_j)}}
 = \frac{\sum_{i} e^{-y_i f_t(x_i)}\, e^{-y_i w_{t+1} h_{t+1}(x_i)}}{\sum_{j} e^{-y_j f_t(x_j)}} \\
&= \sum_{i=1}^m D_i^{(t+1)}\, e^{-y_i w_{t+1} h_{t+1}(x_i)}.
\end{aligned}
\]
The \(1/m\) in \(Z_{t+1}\) and \(Z_t\) cancelled in the first line. Now split the sum by whether \(h_{t+1}\) is right (\(y_i h_{t+1}(x_i) = 1\)) or wrong (\(y_i h_{t+1}(x_i) = -1\)) on example \(i\). By definition, the total weight under \(D^{(t+1)}\) of the wrong ones is \(\varepsilon_{t+1}\), so the right ones weigh \(1 - \varepsilon_{t+1}\):
\[ \frac{Z_{t+1}}{Z_t} = e^{-w_{t+1}} \sum_{i:\, y_i h_{t+1}(x_i) = 1} D_i^{(t+1)} + e^{w_{t+1}} \sum_{i:\, y_i h_{t+1}(x_i) = -1} D_i^{(t+1)} = e^{-w_{t+1}}(1 - \varepsilon_{t+1}) + e^{w_{t+1}}\varepsilon_{t+1}. \]
In plain words: **\(Z\) shrinks by the right examples’ weight times \(e^{-w}\) plus the wrong examples’ weight times \(e^{w}\).** In round 1 that is \(0.5 \cdot 0.8 + 2 \cdot 0.2 = 0.8\), the ratio \(Z_1/Z_0\). Now plug in the vote. From \(w = \tfrac12 \log(1/\varepsilon - 1)\) we get \(e^{w} = \sqrt{1/\varepsilon - 1} = \sqrt{(1 - \varepsilon)/\varepsilon}\), and \(e^{-w} = \sqrt{\varepsilon/(1 - \varepsilon)}\). Writing \(\varepsilon\) for \(\varepsilon_{t+1}\),
\[ \frac{Z_{t+1}}{Z_t} = \sqrt{\frac{\varepsilon}{1 - \varepsilon}}\,(1 - \varepsilon) + \sqrt{\frac{1 - \varepsilon}{\varepsilon}}\,\varepsilon = \sqrt{\varepsilon(1 - \varepsilon)} + \sqrt{\varepsilon(1 - \varepsilon)} = 2\sqrt{\varepsilon_{t+1}(1 - \varepsilon_{t+1})}. \]
In plain words: **with AdaBoost’s vote, a round with weighted risk \(\varepsilon\) shrinks \(Z\) by exactly \(2\sqrt{\varepsilon(1 - \varepsilon)}\).** For round 1, \(2\sqrt{0.2 \cdot 0.8} = 2 \cdot 0.4 = 0.8\) again.

Here are Steps 3 and 4 on the ten messages. This run never uses AdaBoost’s multiplicative update: `D = ...` computes the weights straight from Step 3’s formula, so if that formula were wrong, the risks would differ from the earlier run’s. It prints each ratio \(Z_{t+1}/Z_t\) beside Step 4’s `root`, \(2\sqrt{\varepsilon(1 - \varepsilon)}\).

```python
import math

X = [(0, 0), (0, 1), (0, 3), (1, 1), (2, 0),
     (1, 4), (2, 4), (3, 4), (4, 0), (4, 1)]
y = [1 if a + b >= 4 else -1 for a, b in X]
D, f, Z = [0.1] * 10, [0.0] * 10, 1.0
for j, theta in [(0, 0.5), (0, 2.5), (1, 3.5)]:
    h = [1 if x[j] > theta else -1 for x in X]
    eps = sum(d for d, hi, yi in zip(D, h, y) if hi != yi)
    w = 0.5 * math.log(1 / eps - 1)
    f = [fi + w * hi for fi, hi in zip(f, h)]
    e = [math.exp(-yi * fi) for yi, fi in zip(y, f)]
    D = [ei / sum(e) for ei in e]
    root = 2 * math.sqrt(eps * (1 - eps))
    Z, ratio = sum(e) / 10, sum(e) / 10 / Z
    print(round(eps, 3), round(ratio, 4), round(root, 4))
```

It prints

```
0.2 0.8 0.8
0.125 0.6614 0.6614
0.071 0.5151 0.5151
```

Same risks, and each ratio matches the formula to four places. Their product, \(Z_3 \approx 0.273\), sits well above the actual empirical risk of 0.

**Step 5: bound the ratio.** Let \(g(a) = a(1 - a)\). Its derivative is \(1 - 2a \ge 0\) on \([0, \tfrac12]\), so \(g\) is increasing there. Since \(0 < \varepsilon_{t+1} \le \tfrac12 - \gamma \le \tfrac12\),
\[ 2\sqrt{\varepsilon_{t+1}(1 - \varepsilon_{t+1})} \le 2\sqrt{\left(\tfrac12 - \gamma\right)\left(\tfrac12 + \gamma\right)} = 2\sqrt{\tfrac14 - \gamma^2} = \sqrt{1 - 4\gamma^2}. \]
In plain words: **the more risk a round is allowed, the less it shrinks \(Z\), so the worst allowed round, \(\varepsilon = \tfrac12 - \gamma\), gives the largest factor.** With \(\gamma = 0.3\), that worst factor is \(\sqrt{1 - 0.36} = 0.8\), exactly round 1’s ratio, because round 1 sat right at \(\varepsilon = 0.2 = \tfrac12 - 0.3\). Finally, use \(1 - a \le e^{-a}\), which holds for every real \(a\). (Lesson 2 used it too. To see it, let \(\phi(a) = e^{-a} - 1 + a\). Then \(\phi(0) = 0\) and \(\phi'(a) = 1 - e^{-a}\), which is negative for \(a < 0\) and positive for \(a > 0\), so \(\phi\) has its minimum at 0 and is never negative.) With \(a = 4\gamma^2\), and because the square root is increasing,
\[ \sqrt{1 - 4\gamma^2} \le \sqrt{e^{-4\gamma^2}} = e^{-2\gamma^2}. \]
(The square root of \(e^{-4\gamma^2}\) halves the exponent.) For \(\gamma = 0.3\) it reads \(0.8 \le 0.835\). That is the per-round bound of Step 2, and the proof is complete. \(\blacksquare\)

How fast is “exponentially”? Here is the per-round factor \(\sqrt{1 - 4\gamma^2}\), the bound \(e^{-2\gamma^2}\) that replaced it, and the number of rounds after which \(e^{-2\gamma^2 T}\) is at most 1%. That last number solves \(e^{-2\gamma^2 T} \le 1/100\) for \(T\): taking logs, \(T \ge \log(100)/(2\gamma^2)\), and `math.ceil` rounds up to a whole round:

```python
import math

for gamma in (0.05, 0.1, 0.2, 0.3, 0.45):
    per_round = math.sqrt(1 - 4 * gamma ** 2)
    bound = math.exp(-2 * gamma ** 2)
    assert per_round <= bound
    T = math.ceil(math.log(100) / (2 * gamma ** 2))
    print(gamma, round(per_round, 4), round(bound, 4), T)
```

It prints

```
0.05 0.995 0.995 922
0.1 0.9798 0.9802 231
0.2 0.9165 0.9231 58
0.3 0.8 0.8353 26
0.45 0.4359 0.667 12
```

A stump learner that is right 60% of the time on every reweighting (\(\gamma = 0.1\)) gets the empirical risk below 1% within 231 rounds. With a small edge the last step costs little; with a large edge, \(e^{-2\gamma^2}\) is a generous overestimate.

**Where the vote comes from.** The proof shows something more. Look at the ratio before the vote was plugged in: \(e^{-w}(1 - \varepsilon) + e^{w}\varepsilon\). Which \(w\) makes it smallest? Its derivative in \(w\) is \(-e^{-w}(1 - \varepsilon) + e^{w}\varepsilon\), which is zero exactly when \(e^{2w} = (1 - \varepsilon)/\varepsilon\), that is, at \(w = \tfrac12 \log(1/\varepsilon - 1)\). The second derivative, \(e^{-w}(1 - \varepsilon) + e^{w}\varepsilon\), is positive, so it is a minimum. AdaBoost’s vote is not a magic formula: it is the vote that shrinks \(Z\) the most in this round.

```python
import math

eps = 0.2
def ratio(w):
    return math.exp(-w) * (1 - eps) + math.exp(w) * eps

grid = [k / 1000 for k in range(3000)]
best_w = min(grid, key=ratio)
print(best_w, round(0.5 * math.log(1 / eps - 1), 3))
print(round(ratio(best_w), 4), 2 * math.sqrt(eps * (1 - eps)))
```

It prints `0.693 0.693` and `0.8 0.8`: a grid search lands on \(\log 2\), round 1’s vote, and the smallest ratio is round 1’s 0.8.

**A sharper reading.** Step 4 is an equality, so the proof actually shows
\[ L_S(h_s) \le Z_T = \prod_{t=1}^T 2\sqrt{\varepsilon_t(1 - \varepsilon_t)}, \]
whatever the \(\varepsilon_t\) are. The capital pi \(\prod_{t=1}^T\) is to multiplication what \(\sum\) is to addition: multiply the factor for every round from 1 to \(T\). In plain words: **the empirical risk is at most the product of every round’s exact shrink factor.** Writing \(\gamma_t = \tfrac12 - \varepsilon_t\) for the edge of round \(t\), Step 5 applied round by round gives \(Z_T \le e^{-2 \sum_t \gamma_t^2}\). You don’t need the same edge every round; each round contributes its own. On the ten messages the edges are 0.3, 0.375 and about 0.429:

```python
import math

eps = [0.2, 0.125, 1 / 14]
Z = math.prod(2 * math.sqrt(e * (1 - e)) for e in eps)
g = [0.5 - e for e in eps]
print(round(Z, 3), round(math.exp(-2 * sum(v * v for v in g)), 3))
```

It prints `0.273 0.437`: the exact product, and the looser bound from the edges, which is still far below the constant-edge 0.583. This product is what the lab draws as “the bound”.

## When the weak learner fails

The theorem assumed that the weak learner succeeded in every round. A \(\gamma\)-weak learner only promises success with probability at least \(1 - \delta\), and AdaBoost calls it \(T\) times. By the union bound (the chance that at least one of several events happens is at most the sum of their chances), the probability that it fails in at least one of the \(T\) rounds is at most \(\delta T\), so the theorem’s conclusion holds with probability at least \(1 - \delta T\). With \(\delta = 0.001\) per call and \(T = 100\) rounds, that is \(1 - 0.1 = 0.9\). That is not a real obstacle: a learner’s sample complexity can always be made to depend on \(\delta\) only through \(\log(1/\delta)\) (run it on several independent chunks and keep the one that does best on a final chunk), so a tiny \(\delta\) per call is cheap. Better still, ERM over stumps minimizes the weighted risk exactly and never fails: \(\delta = 0\).

What it cannot do is promise an edge that isn’t there. The theorem needs an edge of at least \(\gamma\) on *every* reweighting AdaBoost produces. In the lab, on the twenty thousand messages, the weighted risks \(\varepsilon_t\) creep up toward \(\tfrac12\) as the rounds go on: around 0.37 in the first rounds, around 0.45 by round 100. The edge shrinks, so the constant-\(\gamma\) bound is pessimistic, and the product above is the honest one.

## AdaBoost as code

**Cost.** Each round does \(O(m)\) work to compute \(\varepsilon_t\) and update the weights, plus one call to the weak learner. So \(T\) rounds cost \(O(T \cdot (m + c_{WL}))\), where \(c_{WL}\) is the cost of one call. With stumps, presorted once in \(O(dm \log m)\), each call is \(O(dm)\), so training costs \(O(dm \log m + Tdm)\) in total. Space is \(O(dm)\) for the sorted index lists, \(O(m)\) for the weights and \(O(T)\) for the votes and stumps. Predicting a fresh message costs \(O(T)\): evaluate \(T\) stumps and add. That is the whole computational point: an efficient weak learner gives an efficient strong one, with only a factor of \(T\) on top.

**Build it yourself (part 2).** Extend part 1:

```python
import math

def adaboost(X, y, T): ...
def predict(hypotheses, x): ...

if __name__ == "__main__":
    hs = adaboost(X, y, 3)
    votes = [round(w, 3) for w, _ in hs]
    assert votes == [0.693, 0.973, 1.282]
    assert all(predict(hs, x) == t for x, t in zip(X, y))
```

1. Implement `adaboost` with your `erm_stump` as the weak learner, returning a list of `(vote, stump)` pairs. Stop early, returning the perfect stump with any vote, if a round has \(\varepsilon_t = 0\).
2. On the ten messages, print \(Z_t\) after every round and check that it never goes below the empirical risk of the vote.
3. Generate the lab’s mail: ten counts, each uniform on 0 to 5, junk exactly when the first eight add up to at least 21. Train on 20,000 messages and measure the empirical risk after 1, 10, 40 and 100 rounds. Then measure the error on 50,000 fresh messages. Before you look, predict how close the two will be.

## Fix the bug

Back to the inbox. Boosting has done what the opening asked for. In the lab, forty rounds of stumps, each barely better than a coin on the weights it was handed, add up to a vote that is wrong on 3.3% of the twenty thousand labelled messages. A pile of weak rules really did become a strong one.

But your users have found the junk that still gets through, and it has a pattern: it arrives at odd hours. One shared inbox, the support desk, is hit hardest. Its genuine mail arrives inside a window of working hours, and everything sent before the window opens or after it closes is junk. The trouble is that the desk’s hours change from week to week. On a 24-hour clock, the window opens somewhere between 5 and 10 and closes somewhere between 16 and 22, and nobody tells the filter which.

So the team adds one number to every message: the hour it was sent, from 0 to 24, with minutes as a fraction, so 9:30 is 9.5. Remember feature N from lesson 1, “sent at night”? This is its grown-up version. On the hour alone, the desk’s rule is a three-piece hypothesis with \(b = +1\): junk before \(\theta_1\), keep from \(\theta_1\) to \(\theta_2\), junk after \(\theta_2\). That is exactly the class from “A weak learner for three-piece rules”, and a single stump on the hour can’t express it.

Here is the job. Every night, a scheduled run of AdaBoost, with stumps on the hour as its weak learner, trains on that day’s 300 labelled messages from the desk. Nobody watches it. The number of rounds \(T\) is written into the schedule once, before any night runs, and the manager wants a promise attached to that number: **every night, whatever the window and whatever the mix of hours, the vote is right on every one of that day’s 300 labelled messages.**

You already own both halves of the argument. Before reading on, ask yourself: what edge does the weak learner have here, and on which weights?

The first half is the edge. In the three-piece section, the two cuts split the line into three regions, and for each region some stump disagrees with the truth on that region alone. The lightest region weighs at most \(1/3\). On a finite set of messages there is no sliver to pay for, because a threshold below every message gives a constant stump; that was the \(\eta\) that disappeared in the simulation, whose worst best stump weighed 0.3242. And nothing in that argument cared what the weights were. So in every round, under whatever weights AdaBoost hands it, ERM over stumps returns a stump with \(\varepsilon_t \le \tfrac13\). That is an edge of \(\gamma = \tfrac12 - \tfrac13 = \tfrac16\), guaranteed in every round of every night. (If a round ever finds \(\varepsilon_t = 0\), AdaBoost stops with a perfect stump, which is even better.)

The second half is the theorem. Recall that \(L_S(h_s)\) is the empirical risk of the vote, the fraction of the night’s labelled messages it gets wrong. With \(\gamma = \tfrac16\), the exponent \(-2\gamma^2 T\) becomes \(-2 \cdot \tfrac1{36} \cdot T = -T/18\), so
\[ L_S(h_s) \le e^{-2\gamma^2 T} = e^{-T/18}. \]

- \(\gamma = \tfrac16\) is the guaranteed edge, so \(\gamma^2 = \tfrac1{36}\) and \(2\gamma^2 = \tfrac1{18}\).
- \(T\) is the number of rounds in the schedule.
- \(e^{-T/18}\) is \(e\) to the power minus \(T\) over 18; each extra round multiplies it by \(e^{-1/18} \approx 0.946\).

In plain words: **on any night, after \(T\) rounds, the vote gets at most a fraction \(e^{-T/18}\) of that night’s labelled messages wrong.** With \(T = 36\), the ceiling is \(e^{-2} \approx 0.135\), which still allows 40 of the 300 messages to be wrong:

```python
import math

gamma, m, T = 1 / 6, 300, 36
ceiling = math.exp(-2 * gamma ** 2 * T)
print(round(2 * gamma ** 2, 4), round(ceiling, 4))
print(math.floor(ceiling * m))
```

It prints `0.0556 0.1353` and `40`. The last piece you met on the ten messages: the empirical risk on 300 messages is a whole number of mistakes divided by 300, so a ceiling below \(1/300\) leaves room for exactly zero.

In the lab you set \(T\) and run nights, one after another. You see tonight’s empirical risk after every round, tonight’s product bound \(Z_T\), the theorem’s ceiling \(e^{-T/18}\), and the line at \(1/300\). Before you settle on a number, watch a night closely. Suppose tonight’s vote is perfect after three rounds. Is three rounds a schedule? Run it and see. Then find the smallest \(T\) the manager can sign for every night, and schedule it.

[[lab-act2]]

## What people get wrong

**“AdaBoost’s empirical risk goes down every round.”** It doesn’t have to. On the ten messages it stayed at 0.2 after round 2, and on the twenty thousand it went up between rounds 80 and 100. On the desk’s first night in the lab it was 0 after round 3 and back to 0.16 after round 4, which is 48 of the 300 messages. What goes down every round is \(Z_t\), the average of \(e^{-y_i f_t(x_i)}\), which sits above the empirical risk.

**“A weak learner is a bad learner, so boosting is a trick.”** The weak learner promises an edge of \(\gamma\) on *every* distribution over the examples, not just the uniform one: a strong assumption in disguise, and exactly what the proof uses each round. A stump that is right 63% of the time on the original mail but can’t beat a coin once the weights move gives AdaBoost nothing to work with. The desk’s schedule worked only because the three-piece argument holds for every weighting.

**“Tonight’s run was perfect after three rounds, so three rounds is a schedule.”** It isn’t, twice over. On the lab’s first night, round 4 undid it. In 2,000 simulated nights, every vote was perfect after round 3, about four nights in ten slipped back at round 4, and all of them were perfect from round 5 on. That is still not a promise: the simulated nights are our guess at the desk, and the promise has to hold for every window and every mix of hours. Tonight’s product bound \(Z_T\) does fall below \(1/300\), somewhere between round 21 and round 25 in those nights, and that proves tonight’s vote perfect. But it exists only after tonight has run. The one number available before any night runs, and true for every night, is the theorem’s ceiling \(e^{-T/18}\), and it needs 103 rounds. The extra rounds are the price of a promise.

## Where boosting came from

Boosting began as a purely theoretical question. In 1988 Kearns and Valiant asked whether an efficient weak learner can always be “boosted” into an efficient strong learner. Robert Schapire, then a graduate student at MIT, answered yes in 1990, but his construction was not practical. In 1995 Freund and Schapire proposed AdaBoost, the first truly practical boosting algorithm, and it became hugely popular: a rare case of a question asked for its own sake turning into a tool everyone uses.

The question also runs in reverse. If strong learning of a class is computationally hard, then so is weak learning of it, since a weak learner could be boosted. That makes boosting useful for proving hardness results, too.

## Recap and the next bug

1. A **decision stump** compares one feature with a threshold. ERM over stumps, for any distribution over the examples, runs in \(O(dm)\) time per call after sorting each feature once, using the update \(F \leftarrow F + y_i D_i\).
2. A **\(\gamma\)-weak learner** only promises true risk at most \(\tfrac12 - \gamma\). Statistically that is no easier than strong learning (infinite VC dimension rules out both), but it can be much easier computationally.
3. **AdaBoost** reweights the examples each round so that the last hypothesis’s mistakes carry more weight, and gives each hypothesis the vote \(w_t = \tfrac12\log(1/\varepsilon_t - 1)\), the vote that shrinks \(Z_t\) the most.
4. If every round has an edge of \(\gamma\), the empirical risk after \(T\) rounds is at most \(e^{-2\gamma^2 T}\). The proof bounds the empirical risk by \(Z_T\), splits \(Z_T\) into a product of per-round ratios, and computes each ratio exactly as \(2\sqrt{\varepsilon_t(1 - \varepsilon_t)}\).
5. With an edge that is guaranteed on every weighting, the theorem becomes a schedule. Stumps have an edge of \(\tfrac16\) on every weighting of three-piece data, so more than \(18 \log m\) rounds make the vote right on all \(m\) labelled messages, every night: 103 rounds for \(m = 300\).

The desk’s night job ships with its promise, and the word-count filter ships beside it after 100 rounds, wrong on 0.9% of its twenty thousand labelled messages. Then a customer asks to put “wrong on 0.9% of messages” into a contract, and someone asks the awkward question: which messages? Every guarantee this lesson proved about the vote is about the labelled messages. The theorem drives the empirical risk down and says nothing at all about the next message to arrive, and lesson 1’s memorizer was perfect on its labels and a coin on everything else. Worse, a vote of 100 stumps is not a stump. It lives in a far bigger class than the one you started from, and lessons 4 to 6 taught you that a bigger class makes the empirical risk a worse witness for the true risk. How big is the class of votes, and how many rounds can you run before the extra expressive power costs more than it buys on fresh mail? That is lesson 10b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A weak learner hands AdaBoost a hypothesis with \(\varepsilon_t = 0.7\). What vote \(w_t\) does AdaBoost give it, and what happens to that hypothesis inside the final vote? Does \(Z\) still shrink this round, and by what factor? Then answer the same questions for \(\varepsilon_t = 0.5\), and say what AdaBoost should do when \(\varepsilon_t = 0\).
2. In the proof that the empirical risk falls exponentially, the formula \(w_t = \tfrac12\log(1/\varepsilon_t - 1)\) is used in exactly one step. Which one? Suppose AdaBoost used the vote \(w_t = 1\) in every round instead. For a round with \(\varepsilon_t = 0.25\), compute the ratio \(Z_{t+1}/Z_t\) with that vote and with AdaBoost’s vote. Explain why AdaBoost’s vote can never do worse in any round.
3. The desk’s night job now gets 1,000 labelled messages a night instead of 300. How many rounds must the schedule book for the same promise? The window and the mix of hours change every night: which step of the argument makes the promise independent of both?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (AdaBoost forgets its last hypothesis).** Show that the hypothesis \(h_t\) of round \(t\) has weighted risk exactly \(\tfrac12\) under the next round’s weights: for every \(t \in [T]\), \(\sum_{i=1}^m D_i^{(t+1)}\, \mathbb 1[y_i \neq h_t(x_i)] = \tfrac12\). This is the rigorous version of “the weights force the weak learner to look at the problematic examples”.
