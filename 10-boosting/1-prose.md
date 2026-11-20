# Boosting

## A shelf of weak rules

The mail room has moved on from yes/no features. The team now counts words. For every message the filter knows how many times it says “free”, “win”, “click”, “offer”, “cash”, “now”, “prize” and “urgent”, and, for good measure, “meeting” and “lunch”. Each count is capped at 5, so a message is a vector of ten whole numbers between 0 and 5.

You have twenty thousand labelled messages, and a deadline. Lesson 8 taught you what the deadline means: for most rich hypothesis classes, finding the hypothesis with the smallest empirical risk is computationally hopeless. So the team trained the only kind of rule it can train fast, a rule that looks at a single count and compares it with a threshold. “Junk if ‘prize’ appears more than three times.” “Junk if ‘cash’ appears more than three times.” There are a hundred or so of these rules, and the learner finds the best one in a second.

The best one is also not very good. On your twenty thousand messages, the best single rule is right 63% of the time. On fresh mail it does no better: its true risk is 0.375, so it is right five times in eight. Every other rule on the shelf is in the same range, except the ones built on “meeting” and “lunch”, which are no better than always saying keep. The inbox is still full of junk.

Here is the puzzle. You own a machine that can, in a second, produce a rule a little better than a coin. You need a rule that is right nearly always. Can a pile of rules that are each barely better than guessing be turned into one that is excellent? And if it can, what does the result cost you on fresh mail? This lesson answers both, and the answer to the first is a surprisingly confident yes.

## Decision stumps, and ten messages

The rules on the team’s shelf have a name. A **decision stump** looks at one feature, compares it with a threshold, and gives one label above the threshold and the other label at or below it.

Before we write one down, one change of convention. Until now a label was 1 for junk and 0 for keep. In this lesson it is convenient to write **junk as \(+1\) and keep as \(-1\)**, so \(\mathcal Y = \{-1, +1\}\). The reason will be clear soon: with labels \(\pm 1\), a hypothesis \(h\) is right on an example \((x, y)\) exactly when \(y\, h(x) = +1\) and wrong exactly when \(y\, h(x) = -1\). That single product does a lot of work in the proofs.

We also need a sign function that never returns 0. Throughout this lesson,
\[ \operatorname{sign}(u) = \begin{cases} +1 & \text{if } u > 0, \\ -1 & \text{if } u \le 0. \end{cases} \]

**Definition (decision stump).** Let \(\mathcal X = \mathbb R^d\). For a feature index \(j \in [d]\), a threshold \(\theta \in \mathbb R\) and a sign \(b \in \{\pm 1\}\), the decision stump \(h_{j, \theta, b}\) is
\[ h_{j, \theta, b}(x) = b \cdot \operatorname{sign}(x_j - \theta). \]
With \(b = +1\) it says junk when count \(j\) exceeds \(\theta\) and keep otherwise; with \(b = -1\) it says the opposite. The class of all of them is \(\mathcal H_{DS} = \{h_{j, \theta, b} : j \in [d],\ \theta \in \mathbb R,\ b \in \{\pm 1\}\}\).

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

Soon we will need to score a stump not by counting mistakes, but by adding up how much each mistaken example *matters*. So let’s set that up now. A **distribution over the examples** is a vector \(D = (D_1, \dots, D_m)\) with every \(D_i \ge 0\) and \(\sum_i D_i = 1\). We call \(D_i\) the **weight** of example \(i\). The **weighted risk** of a hypothesis is the total weight of the examples it gets wrong:
\[ L_D(h) = \sum_{i=1}^m D_i\, \mathbb 1[h(x_i) \neq y_i]. \]
When every weight is \(1/m\), the weighted risk is just the empirical risk, \(L_D(h) = L_S(h)\). (Here \(D\) lives on the \(m\) examples, not on all messages; we keep the letter because the weighted risk is literally the true risk under the distribution that picks example \(i\) with probability \(D_i\).)

With equal weights of 0.1, here is the weighted risk of every stump with \(b = +1\) and a threshold between two observed counts:

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

Recall what we asked of a learner in lesson 3. A class \(\mathcal H\) is PAC learnable if there are a sample complexity \(m_{\mathcal H}(\varepsilon, \delta)\) and an algorithm such that, for every accuracy \(\varepsilon\), every confidence \(\delta\), every distribution \(\mathcal D\) and every labelling function \(f\) for which realizability holds, after \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) examples it returns a hypothesis with true risk at most \(\varepsilon\), with probability at least \(1 - \delta\). In this lesson we call that **strong learning**, because \(\varepsilon\) can be made as small as you like.

Lesson 6’s fundamental theorem told us which classes are PAC learnable, and that any ERM learner will do. But it said nothing about time, and lesson 8 showed that for some classes ERM is computationally hard even when realizability holds. So here is a tempting trade. What if we ask for much less accuracy, in return for speed? Not “true risk at most \(\varepsilon\)”, but merely “a little better than a coin”.

**Definition (\(\gamma\)-weak learnability).** Let \(\gamma > 0\).

- A learning algorithm \(A\) is a **\(\gamma\)-weak learner** for a class \(\mathcal H\) if there is a function \(m_{\mathcal H}: (0, 1) \to \mathbb N\) such that for every \(\delta \in (0, 1)\), every distribution \(\mathcal D\) over \(\mathcal X\) and every labelling function \(f: \mathcal X \to \{\pm 1\}\), if realizability holds with respect to \(\mathcal H, \mathcal D, f\), then when \(A\) runs on \(m \ge m_{\mathcal H}(\delta)\) i.i.d. examples drawn from \(\mathcal D\) and labelled by \(f\), it returns a hypothesis \(h\) such that, with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(h) \le \tfrac12 - \gamma. \]
- A class \(\mathcal H\) is **\(\gamma\)-weak learnable** if a \(\gamma\)-weak learner for it exists.

Read it next to the definition of strong learning. Everything is the same except one thing: the accuracy is no longer yours to choose. It is pinned at \(\tfrac12 - \gamma\). A hypothesis that answers by flipping a fair coin has true risk exactly \(\tfrac12\), so a weak learner only promises to beat the coin by a fixed margin \(\gamma\). The number \(\gamma\) is called the learner’s **edge**.

[[diagram:weak]]

Is weak learning statistically easier than strong learning? Surprisingly, no. Lesson 6’s fundamental theorem gives a lower bound on the sample complexity of PAC learning a class of VC dimension \(d\), of the form \(m_{\mathcal H}(\varepsilon, \delta) \ge C_1 \frac{d + \log(1/\delta)}{\varepsilon}\) for an absolute constant \(C_1\). Put \(\varepsilon = \tfrac12 - \gamma\) into it. If \(d = \infty\), no finite number of examples suffices even for that modest accuracy, so a class with infinite VC dimension is not \(\gamma\)-weak learnable either. If we ignore computation, weak learning is characterized by the VC dimension, exactly like strong learning.

The hope lies elsewhere. Maybe there is a weak learner that is *fast*, for a class where every strong learner is slow. A natural recipe is to pick a simple **base class** \(B\), such as decision stumps, and run ERM over \(B\) as the weak learner. That works for \(\mathcal H\) if two things hold:

- \(\mathrm{ERM}_B\) can be computed efficiently, and
- for every training sequence labelled by some hypothesis in \(\mathcal H\), every \(\mathrm{ERM}_B\) hypothesis has risk at most \(\tfrac12 - \gamma\).

Notice that \(B\) and \(\mathcal H\) are different classes. The weak learner returns stumps, but the labels come from something richer. Let’s see this work on a small example before we ask the big question.

## A weak learner for three-piece rules

Leave the mail room for a moment and put messages on a line, \(\mathcal X = \mathbb R\). A **three-piece hypothesis** cuts the line at two points \(\theta_1 < \theta_2\) and alternates labels:
\[ h_{\theta_1, \theta_2, b}(x) = \begin{cases} +b & \text{if } x < \theta_1 \text{ or } x > \theta_2, \\ -b & \text{if } \theta_1 \le x \le \theta_2. \end{cases} \]
Let \(\mathcal H\) be the class of all three-piece hypotheses, and let the base class be the stumps on the line, \(B = \{x \mapsto b \cdot \operatorname{sign}(x - \theta) : \theta \in \mathbb R,\ b \in \{\pm 1\}\}\). A stump can’t express a three-piece rule. Yet:

**Claim.** \(\mathrm{ERM}_B\) is a \(\gamma\)-weak learner for \(\mathcal H\) with \(\gamma = 1/12\).

[[diagram:three-piece]]

*Proof.* Fix \(\mathcal D\) and \(f\) with realizability. Then some \(h^\star = h_{\theta_1, \theta_2, b} \in \mathcal H\) has \(L_{(\mathcal D, f)}(h^\star) = 0\), so \(f = h^\star\) except on a set of probability zero, and for any stump \(g\), \(L_{(\mathcal D, f)}(g) = \mathcal D(\{x : g(x) \neq h^\star(x)\})\).

**Step 1: some stump is within \(1/3\) of perfect.** The two cuts split the line into three regions: \(R_1 = (-\infty, \theta_1)\), \(R_2 = [\theta_1, \theta_2]\) and \(R_3 = (\theta_2, \infty)\). They are disjoint and cover the line, so \(\mathcal D(R_1) + \mathcal D(R_2) + \mathcal D(R_3) = 1\), and the smallest of the three weights is at most \(1/3\) (if all three exceeded \(1/3\), their sum would exceed 1). We now show that, for each region \(R_k\) and each \(\eta > 0\), some stump disagrees with \(h^\star\) on a set of probability at most \(\mathcal D(R_k) + \eta\).

- Region \(R_1\). The stump \(g = b \cdot \operatorname{sign}(x - \theta_2)\) gives \(+b\) for \(x > \theta_2\) and \(-b\) for \(x \le \theta_2\). It agrees with \(h^\star\) on \(R_2\) and \(R_3\), so it disagrees only on \(R_1\).
- Region \(R_3\). Take \(g = -b \cdot \operatorname{sign}(x - \theta)\) with \(\theta < \theta_1\). It gives \(+b\) for \(x \le \theta\) and \(-b\) for \(x > \theta\), so it disagrees with \(h^\star\) on \(R_3\) and on the sliver \((\theta, \theta_1)\). As \(\theta\) rises to \(\theta_1\), the slivers shrink to the empty set, and by continuity of probability their weight tends to 0. So some \(\theta\) makes the sliver weigh at most \(\eta\).
- Region \(R_2\). Take \(g = b \cdot \operatorname{sign}(x - \theta)\) with \(\theta\) far to the left. It gives \(+b\) except on \((-\infty, \theta]\), so it disagrees with \(h^\star\) on \(R_2\) and on the part of \((-\infty, \theta]\) outside \(R_2\). As \(\theta \to -\infty\) the sets \((-\infty, \theta]\) shrink to the empty set, so again some \(\theta\) makes their weight at most \(\eta\).

Applying this to the lightest region, for every \(\eta > 0\) there is a stump with \(L_{(\mathcal D, f)}(g) \le \tfrac13 + \eta\). (The \(\eta\) is needed because a stump on the whole line is never constant: it can only push its threshold further and further out.)

**Step 2: ERM over stumps finds one nearly that good.** The class \(B\) has VC dimension 2, which you can check the way lesson 6 checked thresholds. By lesson 6’s fundamental theorem, a class of finite VC dimension is agnostic PAC learnable by ERM. We need the agnostic version here, because the labels come from \(h^\star\), which is usually not a stump. So there is a sample complexity \(m_B(\varepsilon, \delta)\) such that, on \(m \ge m_B(\varepsilon, \delta)\) examples, with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(\mathrm{ERM}_B(S)) \le \min_{g \in B} L_{(\mathcal D, f)}(g) + \varepsilon. \]
Set \(\varepsilon = 1/24\) and \(\eta = 1/24\). The minimum over \(B\) is at most \(\tfrac13 + \tfrac1{24}\), so with probability at least \(1 - \delta\),
\[ L_{(\mathcal D, f)}(\mathrm{ERM}_B(S)) \le \tfrac13 + \tfrac1{24} + \tfrac1{24} = \tfrac{5}{12} = \tfrac12 - \tfrac1{12}. \]
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

The weak learner has to be fast, or the whole plan is pointless. So how do you find the stump with the smallest weighted risk? Let’s make it concrete for \(b = +1\), the stumps that say junk when count \(j\) exceeds \(\theta\). Their weighted risk is
\[ F_j(\theta) = \sum_{i : y_i = +1} D_i\, \mathbb 1[x_{i,j} \le \theta] + \sum_{i : y_i = -1} D_i\, \mathbb 1[x_{i,j} > \theta], \]
the junk examples it lets through plus the keep examples it flags. What about \(b = -1\)? That stump flips every prediction, so it is wrong exactly where the \(b = +1\) stump is right, and its weighted risk is \(1 - F_j(\theta)\), because the weights add up to 1. So one function covers both signs, and we want
\[ \min_{j \in [d]} \ \min_{\theta \in \mathbb R} \ \min\{F_j(\theta),\, 1 - F_j(\theta)\}. \]

**There are only \(m + 1\) thresholds worth trying.** Fix \(j\) and sort the examples so that \(x_{1,j} \le x_{2,j} \le \dots \le x_{m,j}\). A threshold \(\theta\) affects the predictions only through the set of examples with \(x_{i,j} \le \theta\), and that set is always a prefix of the sorted list. Any two thresholds that give the same prefix give the same predictions on \(S\). So it is enough to try one threshold per prefix:
\[ \Theta_j = \{x_{1,j} - 1\} \cup \left\{ \tfrac{x_{i,j} + x_{i+1,j}}{2} : i \in [m-1] \right\} \cup \{x_{m,j} + 1\}. \]

Trying every \(\theta \in \Theta_j\) for every \(j\), and summing over \(m\) examples each time, costs \(O(dm^2)\). There is a neat way to save a factor of \(m\).

**Each step costs \(O(1)\).** Suppose we know \(F = F_j(\theta)\) for a threshold just below \(x_{i,j}\), and now move to a threshold \(\theta'\) just above it. Only example \(i\) changes sides: it was above the threshold, predicted junk, and now it is at or below it, predicted keep. If \(y_i = +1\), it was right and is now wrong, so \(F\) grows by \(D_i\). If \(y_i = -1\), it was wrong and is now right, so \(F\) shrinks by \(D_i\). Both cases in one line:
\[ F_j(\theta') = F_j(\theta) + y_i D_i. \]

[[diagram:sweep]]

The sweep starts with a threshold below everything, where every example is predicted junk and \(F\) is the total weight of the keep examples, and then walks up the sorted list. Here is the check that the sweep and brute force agree on 200 random examples of one feature. The `groupby` handles ties: examples with equal counts must cross the threshold together, which is exactly why \(\Theta_j\) has no threshold between two equal values.

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
It gives \(h_t\) a **vote**
\[ w_t = \tfrac12 \log\!\left(\frac{1}{\varepsilon_t} - 1\right), \]
which is large when \(\varepsilon_t\) is small and shrinks to 0 as \(\varepsilon_t\) approaches \(\tfrac12\). (Two words, two things: a *weight* belongs to an example, a *vote* belongs to a hypothesis.) Then it moves weight onto the examples \(h_t\) got wrong. An example that \(h_t\) gets right has \(y_i h_t(x_i) = +1\), so its weight is multiplied by \(e^{-w_t}\) and shrinks. An example it gets wrong has \(y_i h_t(x_i) = -1\), so its weight is multiplied by \(e^{w_t}\) and grows. Dividing by the total keeps it a distribution. After \(T\) rounds, the final hypothesis is a weighted vote of everything it has collected.

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

**Round 3.** Now 6 and 7 are heavy, and the learner returns “junk if click \(> 3.5\)”, which misses only 9 and 10. Their weights are \(1/28\) each, so \(\varepsilon_3 = 1/14 \approx 0.071\) and \(w_3 = \tfrac12 \log 13 \approx 1.282\).

The vote after three rounds is
\[ h_s(x) = \operatorname{sign}\big(0.693\, h_1(x) + 0.973\, h_2(x) + 1.282\, h_3(x)\big), \]
and it gets all ten messages right. Three stumps, each wrong on two of ten messages, combine into a rule that is wrong on none. Here is the same run in Python, printing \(\varepsilon_t\), \(w_t\), the empirical risk of the vote so far, and the weight of message 4 after the update:

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

Two things deserve a second look. First, the empirical risk did not go down after round 2: it stayed at 0.2. The vote of the first two stumps happens to say keep for messages 4 to 7, getting 4 and 5 right but now 6 and 7 wrong. The empirical risk of AdaBoost is not guaranteed to fall every round. What *is* guaranteed to fall is a smooth quantity sitting above it, and that is the heart of the proof. Second, look at round 2’s weights: right after the update, messages 4 and 5, the ones round 1 got wrong, carry exactly half the total weight. Coincidence? Problem 1 at the end asks you to prove it never is.

[[diagram:reweight]]

Now play with it yourself. In the lab, you step AdaBoost round by round, first on the ten messages and then on the twenty thousand, and watch the weights move.

[[lab]]

## Why the empirical risk falls exponentially

Here is the theorem that makes boosting worth doing.

**Theorem (AdaBoost drives the empirical risk down exponentially).** Let \(S\) be a training sequence, and suppose that in every round of AdaBoost the weak learner returns a hypothesis with \(\varepsilon_t \le \tfrac12 - \gamma\). Then the hypothesis \(h_s\) that AdaBoost outputs after \(T\) rounds satisfies
\[ L_S(h_s) = \frac1m \sum_{i=1}^m \mathbb 1[h_s(x_i) \neq y_i] \le e^{-2\gamma^2 T}. \]

Before the proof, notice how strong this is. The weak learner only promises to beat a coin by \(\gamma\), each round, on whatever weights it is handed. Yet the empirical risk of the vote falls exponentially in the number of rounds. And because the empirical risk is a multiple of \(1/m\), the moment the bound drops below \(1/m\) the empirical risk must be exactly 0.

We also need the rounds to be well defined. If \(\varepsilon_t = 0\), the vote \(w_t\) would be infinite; in that case the weak learner has found a hypothesis with no mistakes at all, and AdaBoost can simply stop and return it. So assume \(0 < \varepsilon_t \le \tfrac12 - \gamma\), which makes every vote finite and positive.

*Proof.* We follow the vote as it builds up. For each \(t\), let
\[ f_t = \sum_{p \le t} w_p h_p, \]
so \(f_0 \equiv 0\) and AdaBoost outputs \(h_s = \operatorname{sign}(f_T)\). Define
\[ Z_t = \frac1m \sum_{i=1}^m e^{-y_i f_t(x_i)}. \]
This \(Z_t\) is the smooth quantity. The plan has four steps: show \(Z_T\) sits above the empirical risk, split \(Z_T\) into a product of per-round ratios, compute each ratio exactly, and bound it.

**Step 1: the empirical risk is at most \(Z_T\).** Take one example. If \(h_s\) gets it wrong, then \(\operatorname{sign}(f_T(x_i)) \neq y_i\). If \(y_i = +1\), that means \(f_T(x_i) \le 0\); if \(y_i = -1\), it means \(f_T(x_i) > 0\). Either way \(y_i f_T(x_i) \le 0\), so \(e^{-y_i f_T(x_i)} \ge 1\). If \(h_s\) gets it right, the indicator is 0, and an exponential is always positive. So for every example
\[ \mathbb 1[h_s(x_i) \neq y_i] \le e^{-y_i f_T(x_i)}. \]
Average over \(i\): \(L_S(h_s) \le Z_T\). (This also settles any worry about ties: our sign sends 0 to \(-1\), but the inequality holds whichever label a tie gets, because \(e^0 = 1\).) So it suffices to show \(Z_T \le e^{-2\gamma^2 T}\).

**Step 2: a telescoping product.** Since \(f_0 \equiv 0\), every term of \(Z_0\) is \(e^0 = 1\), so \(Z_0 = 1\). All the \(Z_t\) are positive, so we may write
\[ Z_T = \frac{Z_T}{Z_0} = \frac{Z_T}{Z_{T-1}} \cdot \frac{Z_{T-1}}{Z_{T-2}} \cdots \frac{Z_1}{Z_0}. \]
Each intermediate \(Z\) appears once on top and once below, and cancels. So it suffices to show that for every round,
\[ \frac{Z_{t+1}}{Z_t} \le e^{-2\gamma^2}, \]
because multiplying \(T\) such factors gives \(e^{-2\gamma^2 T}\).

[[diagram:telescope]]

**Step 3: the weights are the exponentials, normalized.** We claim that for every \(t \ge 0\) and every \(i\),
\[ D_i^{(t+1)} = \frac{e^{-y_i f_t(x_i)}}{\sum_{j=1}^m e^{-y_j f_t(x_j)}}. \]
We prove it by induction on \(t\). For \(t = 0\), \(f_0 \equiv 0\), so the right side is \(1/m\), which is \(D^{(1)}\). Now suppose it holds for \(t - 1\), that is, \(D_i^{(t)} = e^{-y_i f_{t-1}(x_i)} / N_{t-1}\), where \(N_{t-1}\) is the sum in the denominator. AdaBoost’s update multiplies \(D_i^{(t)}\) by \(e^{-w_t y_i h_t(x_i)}\) and renormalizes. Since \(f_t = f_{t-1} + w_t h_t\),
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
Now plug in the vote. From \(w = \tfrac12 \log(1/\varepsilon - 1)\) we get \(e^{w} = \sqrt{1/\varepsilon - 1} = \sqrt{(1 - \varepsilon)/\varepsilon}\), and \(e^{-w} = \sqrt{\varepsilon/(1 - \varepsilon)}\). Writing \(\varepsilon\) for \(\varepsilon_{t+1}\),
\[ \frac{Z_{t+1}}{Z_t} = \sqrt{\frac{\varepsilon}{1 - \varepsilon}}\,(1 - \varepsilon) + \sqrt{\frac{1 - \varepsilon}{\varepsilon}}\,\varepsilon = \sqrt{\varepsilon(1 - \varepsilon)} + \sqrt{\varepsilon(1 - \varepsilon)} = 2\sqrt{\varepsilon_{t+1}(1 - \varepsilon_{t+1})}. \]

Both Step 3 and Step 4 are easy to check on the ten messages. This version of the run never uses AdaBoost’s multiplicative update: it computes the weights straight from Step 3’s formula. If the formula were wrong, the weak learner’s risks would come out different from the previous run’s 0.2, 0.125 and 0.071. Then it prints each ratio \(Z_{t+1}/Z_t\) beside \(2\sqrt{\varepsilon(1 - \varepsilon)}\).

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

Same risks, and each ratio matches the formula to four places. Multiply the three ratios and you get \(Z_3 \approx 0.273\), comfortably above the actual empirical risk of 0.

**Step 5: bound the ratio.** Let \(g(a) = a(1 - a)\). Its derivative is \(1 - 2a \ge 0\) on \([0, \tfrac12]\), so \(g\) is increasing there. Since \(0 < \varepsilon_{t+1} \le \tfrac12 - \gamma \le \tfrac12\),
\[ 2\sqrt{\varepsilon_{t+1}(1 - \varepsilon_{t+1})} \le 2\sqrt{\left(\tfrac12 - \gamma\right)\left(\tfrac12 + \gamma\right)} = 2\sqrt{\tfrac14 - \gamma^2} = \sqrt{1 - 4\gamma^2}. \]
Finally, use \(1 - a \le e^{-a}\), which holds for every real \(a\). (Lesson 2 used it too. To see it, let \(\phi(a) = e^{-a} - 1 + a\). Then \(\phi(0) = 0\) and \(\phi'(a) = 1 - e^{-a}\), which is negative for \(a < 0\) and positive for \(a > 0\), so \(\phi\) has its minimum at 0 and is never negative.) With \(a = 4\gamma^2\), and because the square root is increasing,
\[ \sqrt{1 - 4\gamma^2} \le \sqrt{e^{-4\gamma^2}} = e^{-2\gamma^2}. \]
That is the per-round bound of Step 2, and the proof is complete. \(\blacksquare\)

How fast is “exponentially”? Here is the per-round factor \(\sqrt{1 - 4\gamma^2}\), the bound \(e^{-2\gamma^2}\) that replaced it, and the number of rounds after which \(e^{-2\gamma^2 T}\) is at most 1%:

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
whatever the \(\varepsilon_t\) are. Writing \(\gamma_t = \tfrac12 - \varepsilon_t\) for the edge of round \(t\), Step 5 applied round by round gives \(Z_T \le e^{-2 \sum_t \gamma_t^2}\). You don’t need the same edge every round; each round contributes its own. This product is what the lab draws as “the bound”.

## When the weak learner fails

The theorem assumed that the weak learner succeeded in every round. A \(\gamma\)-weak learner only promises success with probability at least \(1 - \delta\), and AdaBoost calls it \(T\) times. By the union bound, the probability that it fails in at least one of the \(T\) rounds is at most \(\delta T\), so the theorem’s conclusion holds with probability at least \(1 - \delta T\). That is not a real obstacle. The sample complexity of a learner can always be made to depend on \(\delta\) only through \(\log(1/\delta)\) (run the learner on several independent chunks of examples and keep the one that does best on a final chunk), so asking each call for a tiny \(\delta\) is cheap. Better still, the weak learner here only ever sees distributions over the training sequence, and it can often be built to never fail at all. ERM over stumps is such a learner: it minimizes the weighted risk exactly, so it has \(\delta = 0\).

What it cannot do is promise an edge that isn’t there. The theorem needs an edge of at least \(\gamma\) on *every* reweighting AdaBoost produces. In the lab, on the twenty thousand messages, the weighted risks \(\varepsilon_t\) creep up toward \(\tfrac12\) as the rounds go on: around 0.37 in the first rounds, around 0.45 by round 100. The edge shrinks, so the constant-\(\gamma\) bound is pessimistic, and the product above is the honest one.

## AdaBoost as code

**Cost.** Each round does \(O(m)\) work to compute \(\varepsilon_t\) and update the weights, plus one call to the weak learner. So \(T\) rounds cost \(O(T \cdot (m + c_{WL}))\), where \(c_{WL}\) is the cost of one call. With stumps, presorted once in \(O(dm \log m)\), each call is \(O(dm)\), so training costs \(O(dm \log m + Tdm)\) in total. Space is \(O(dm)\) for the sorted index lists, \(O(m)\) for the weights and \(O(T)\) for the votes and stumps. Predicting a fresh message costs \(O(T)\): evaluate \(T\) stumps and add.

That is the whole computational point. If the weak learner is efficient, as ERM over stumps is, the strong hypothesis is efficient to train too, with only a factor of \(T\) on top.

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

## What the vote can express

AdaBoost does not return a stump. It returns a vote of \(T\) stumps, and that vote lives in a bigger class. For a base class \(B\), define
\[ L(B, T) = \left\{ x \mapsto \operatorname{sign}\!\left( \sum_{t=1}^T w_t h_t(x) \right) : w \in \mathbb R^T,\ h_1, \dots, h_T \in B \right\}. \]
Read it in two stages. First, the \(T\) base hypotheses turn a message \(x\) into a vector of \(T\) labels, \(\psi(x) = (h_1(x), \dots, h_T(x)) \in \{\pm 1\}^T\). Then a homogeneous halfspace from lesson 9, with weight vector \(w\), labels that vector. A vote of stumps is a halfspace on top of stumps.

How much richer is the vote than one stump? On the line, take the stumps \(B = \{x \mapsto b \cdot \operatorname{sign}(x - \theta)\}\), and compare \(L(B, T)\) with **piecewise constant** hypotheses: cut the line at \(\theta_1 < \theta_2 < \dots < \theta_{r-1}\), set \(\theta_0 = -\infty\) and \(\theta_r = +\infty\), and give each piece \((\theta_{i-1}, \theta_i]\) its own label \(\alpha_i \in \{\pm 1\}\):
\[ g_r(x) = \sum_{i=1}^r \alpha_i\, \mathbb 1[x \in (\theta_{i-1}, \theta_i]]. \]
Call \(G_r\) the class of such hypotheses with at most \(r\) pieces. The main case is the alternating one, \(\alpha_i = (-1)^i\), where every cut flips the label. Here is how to write it as a vote of \(T\) stumps:
\[ h(x) = \operatorname{sign}\!\left( \sum_{t=1}^T w_t \operatorname{sign}(x - \theta_{t-1}) \right), \qquad w_1 = \tfrac{(-1)^T}{2}, \quad w_t = (-1)^t \text{ for } t \ge 2. \]
The first term uses \(\theta_0 = -\infty\), so it is a constant \(+1\) that acts like an offset. On the whole line no stump is constant, but on any finite set of points (and for word counts, which are never negative) a stump with its threshold below every point is constant, which is all we will need. Problem 2 asks you to prove that \(h = g_T\); before you do, here is a check for \(T = 1\) to 7, with cuts at 2, 4, 6, and so on:

```python
def sign(u):
    return 1 if u > 0 else -1

def staircase(T, thetas):
    w = [(-1) ** T / 2] + [(-1) ** t for t in range(2, T + 1)]
    th = [-1] + thetas
    return lambda x: sign(sum(w[t] * sign(x - th[t])
                              for t in range(T)))

for T in range(1, 8):
    thetas = [2 * t for t in range(1, T)]
    h = staircase(T, thetas)
    piece = lambda x: 1 + sum(x > th for th in thetas)
    assert all(h(x) == (-1) ** piece(x) for x in range(2 * T))
print("h equals g for T = 1, ..., 7")
```

It prints `h equals g for T = 1, ..., 7`. (The sign of \(w_1\) matters. With \(w_1 = +\tfrac12\) for every \(T\), the vote is right for even \(T\) and says \(+1\) everywhere for odd \(T\). Try it.)

[[diagram:staircase]]

So \(T\) stumps can draw a staircase with \(T\) steps. That has a consequence for the VC dimension (lesson 6: the size of the largest set the class can shatter, that is, label in every possible way). Take any \(T\) distinct points on the line. Every one of the \(2^T\) labellings of them is a piecewise constant hypothesis with at most \(T\) pieces, one piece per point, so on those points it is a vote of \(T\) stumps. Hence
\[ \operatorname{VCdim}(L(B, T)) \ge T. \]
The exact value turns out to depend on the parity of \(T\): it is \(T + 1\) when \(T\) is odd and \(T\) when \(T\) is even. You will see why in a moment, and it is a nice puzzle on its own. What matters is the trend: every extra round buys the vote more expressive power.

That power is a double-edged sword, and lesson 5 named both edges. A richer class lowers the **approximation error**, the true risk of the best hypothesis in the class. A richer class also raises the **estimation error**, the gap between that best and what the learner finds from finitely many examples. AdaBoost gives you a single dial for this tradeoff: the number of rounds \(T\). The empirical risk falls as \(T\) grows. How fast does the class grow?

## How big the class gets

**Lemma (the VC dimension of a vote).** Let \(B\) be a base class with \(d = \operatorname{VCdim}(B)\), and assume \(T \ge 3\) and \(d \ge 3\). Then
\[ \operatorname{VCdim}(L(B, T)) \le T(d + 1)\big(3 \log(T(d + 1)) + 2\big). \]
Up to the logarithm, the VC dimension of the vote is at most \(T\) times that of the base class.

We need two tools from earlier lessons. **Sauer’s lemma** (lesson 6): if a class has VC dimension \(d\), then on any set of \(m > d + 1\) points it induces at most \((em/d)^d\) different labellings. And **homogeneous halfspaces** in \(\mathbb R^T\) have VC dimension \(T\) (lesson 9), so on \(m > T + 1\) points of \(\mathbb R^T\) they induce at most \((em/T)^T\) labellings.

*Proof.* Let \(C = \{x_1, \dots, x_m\}\) be a set that \(L(B, T)\) shatters. We count the labellings of \(C\) that \(L(B, T)\) can produce, and compare with \(2^m\).

**A small case first.** The right side of the lemma is larger than both \(d + 1\) and \(T + 1\): with \(d, T \ge 3\) we have \(T(d+1) \ge 3(d + 1)\) and \(T(d+1) \ge 4T\), and the factor \(3\log(T(d+1)) + 2\) is at least \(3 \log 12 + 2 > 1\). So if \(m \le \max(d, T) + 1\), the lemma already holds. Assume from now on that \(m > d + 1\) and \(m > T + 1\), so both versions of Sauer’s lemma apply.

**Count the labellings.** A hypothesis in \(L(B, T)\) labels \(C\) in two stages. First it picks \(h_1, \dots, h_T \in B\). What matters for \(C\) is only how each \(h_t\) labels \(C\), and by Sauer’s lemma \(B\) has at most \((em/d)^d\) labellings of \(C\). Choosing \(T\) of them, in order, can be done in at most \((em/d)^{dT}\) ways. Second, once the \(T\) labellings are fixed, the vectors \(\psi(x_1), \dots, \psi(x_m) \in \mathbb R^T\) are fixed, and the halfspace with weights \(w\) labels those \(m\) vectors in at most \((em/T)^T\) ways. Every labelling of \(C\) by \(L(B, T)\) arises from some choice in stage one followed by some choice in stage two, so the number of labellings is at most
\[ (em/d)^{dT}(em/T)^T. \]

**Simplify.** Since \(d \ge 3 > e\) we have \(e/d < 1\), so \((em/d)^{dT} < m^{dT}\). Likewise \(T \ge 3\) gives \((em/T)^T < m^T\). So the count is less than \(m^{(d+1)T}\).

**Use the shattering.** \(C\) is shattered, so all \(2^m\) labellings occur, and
\[ 2^m < m^{(d+1)T}. \]
Taking logarithms, \(m \log 2 < (d + 1)T \log m\), that is,
\[ m < a \log m, \qquad a = \frac{(d + 1)T}{\log 2}. \]

**Turn \(m < a \log m\) into a bound on \(m\).** We use a small fact about logarithms: for \(a > 0\), if \(x \ge 2a \log a\) then \(x \ge a \log x\). Here is why. If \(a \le \sqrt e\), the function \(x - a \log x\) has its minimum at \(x = a\) (its derivative \(1 - a/x\) changes sign there), where it equals \(a(1 - \log a) \ge 0\); so \(x \ge a \log x\) for every \(x > 0\). If \(a > \sqrt e\), then \(2 \log a > 1\), so \(2a \log a > a\), and \(x - a\log x\) is increasing for \(x > a\). At \(x = 2a\log a\) it equals \(a \log a - a \log(2 \log a) = a \log\!\big(a / (2 \log a)\big)\), which is nonnegative because \(a \ge 2 \log a\) for every \(a > 0\). So for \(x \ge 2a\log a\) the function is at least its value there, which is at least 0.

Read it backwards. Since \(m < a \log m\), the conclusion fails for \(x = m\), so the hypothesis must fail too: \(m < 2a \log a\). Now unpack \(a\):
\[ 2a \log a = \frac{2}{\log 2}(d+1)T \left( \log((d+1)T) + \log\frac{1}{\log 2} \right). \]
With \(2/\log 2 \approx 2.885 \le 3\) and \(\log(1/\log 2) \approx 0.367\), this is at most \((d + 1)T\big(2.885 \log((d+1)T) + 1.06\big) \le (d + 1)T\big(3 \log((d+1)T) + 2\big)\). So every shattered set has fewer elements than the bound. \(\blacksquare\)

Here are the numbers for \(d = 3\) and \(T = 10\): the largest \(m\) for which the count still allows \(2^m\) labellings, the intermediate bound \(2a\log a\), and the lemma’s bound. Then the lemma’s bound at \(T = 40\) and 100.

```python
import math

def log_count(m, d, T):
    return d * T * math.log(math.e * m / d) + T * math.log(
        math.e * m / T)

d, T = 3, 10
m = max(m for m in range(T + 2, 2000)
        if m * math.log(2) <= log_count(m, d, T))
a = (d + 1) * T / math.log(2)
lemma = T * (d + 1) * (3 * math.log(T * (d + 1)) + 2)
print(m, round(2 * a * math.log(a)), round(lemma))
for T in (40, 100):
    print(T, round(T * 4 * (3 * math.log(T * 4) + 2)))
```

It prints `307 468 523`, then `40 2756` and `100 7990`. Each step of the proof loosens the bound a little, and the shape survives: roughly linear in \(T\).

Now the parity puzzle from the last section, which this proof style explains in one line. Put \(T\) stumps on the line, all with genuine thresholds between \(T + 1\) points. To the left of every threshold, every stump says \(-1\), so the sum is \(-\sum_t w_t\). To the right of every threshold, every stump says \(+1\), so the sum is \(+\sum_t w_t\). The two ends of the line always get opposite signs (or both get \(-1\), if the votes sum to zero). That forbids some labellings. With two stumps and three points \(x_1 < x_2 < x_3\), the labelling \(+, -, +\) needs two sign changes, so both thresholds must sit between the points, and then it needs both ends positive, which is impossible; and three points are not shattered. With an odd number of stumps the two ends of an alternating labelling have opposite signs anyway, which is why the parity of \(T\) decides the exact VC dimension.

## A promise about fresh mail

The empirical risk falls with \(T\). The class grows with \(T\). To turn that into a promise about fresh mail, we need a bound with real numbers in it, and our mail room has a feature that makes one easy: there are only finitely many possible messages. Ten counts, each from 0 to 5, give \(|\mathcal X| = 6^{10} = 60{,}466{,}176\) messages.

On a finite domain, every class is finite once you count functions rather than formulas. Two hypotheses that agree on every possible message have the same true risk and the same empirical risk, so only the distinct functions matter. How many functions does \(L(B, T)\) contain? Exactly the count from the proof above, with \(C = \mathcal X\):

- A stump on one count from 0 to 5 can cut in 5 genuine places, with 2 signs, and there are 10 counts. Add the two constant stumps (a threshold below 0 or above 5), and \(B\) contains \(2 \cdot 10 \cdot 5 + 2 = 102\) distinct functions on \(\mathcal X\).
- Choosing \(T\) of them, in order: at most \(102^T\) ways.
- For each choice, a homogeneous halfspace labels the \(|\mathcal X|\) vectors \(\psi(x)\) in at most \((e|\mathcal X|/T)^T\) ways, by Sauer’s lemma.

So \(|L(B, T)| \le 102^T (e|\mathcal X|/T)^T\) as a set of functions on \(\mathcal X\).

Now bring in lesson 4. For a finite class \(\mathcal H\) and \(m\) i.i.d. examples, with probability at least \(1 - \delta\), every \(h \in \mathcal H\) satisfies
\[ |L_{\mathcal D}(h) - L_S(h)| \le \sqrt{\frac{\log(2|\mathcal H|/\delta)}{2m}}. \]
That was uniform convergence for finite classes, from Hoeffding’s inequality and the union bound. One more detail. We will try every \(T\) from 1 to 100 and keep the best, so we need the promise to hold for all hundred classes at once. Give each class confidence \(\delta/100\); by the union bound, all hundred promises hold together with probability at least \(1 - \delta\). That is the same trick lesson 7 used to spread confidence over a sequence of classes. So, with probability at least \(1 - \delta\), for every \(T \le 100\) and every vote \(h\) of \(T\) stumps,
\[ L_{\mathcal D}(h) \le L_S(h) + \operatorname{gap}(T), \qquad \operatorname{gap}(T) = \sqrt{\frac{\log|L(B, T)| + \log(200/\delta)}{2m}}. \]
We call \(L_S(h) + \operatorname{gap}(T)\) the **certified risk**: a number you compute from the training sequence alone, which the true risk exceeds only with probability at most \(\delta\).

With \(m\) = 20,000 and \(\delta = 0.05\):

```python
import math

m, delta, k, K = 20_000, 0.05, 10, 5
log_B = math.log(2 * k * K + 2)
log_X = k * math.log(K + 1)

def gap(T):
    log_H = T * (log_B + 1 + log_X - math.log(T))
    log_terms = log_H + math.log(2 * 100 / delta)
    return math.sqrt(log_terms / (2 * m))

for T in (1, 10, 20, 40, 60, 100):
    print(T, round(gap(T), 3))
```

It prints

```
1 0.028
10 0.074
20 0.102
40 0.142
60 0.171
100 0.218
```

The gap grows like \(\sqrt{T}\), give or take a logarithm. The empirical risk shrinks with \(T\). Their sum must have a sweet spot.

Here is one run of AdaBoost with stumps on 20,000 messages from the lab’s mail, where junk means the first eight counts add up to at least 21. The last column is the error on 50,000 fresh messages, which a real mail room would not have; the lab can afford it.

| rounds \(T\) | \(L_S(h_s)\) | bound \(Z_T\) | gap\((T)\) | certified risk | fresh error |
|---|---|---|---|---|---|
| 1 | 0.368 | 0.965 | 0.028 | 0.397 | 0.377 |
| 10 | 0.158 | 0.599 | 0.074 | 0.233 | 0.155 |
| 20 | 0.106 | 0.446 | 0.102 | 0.208 | 0.103 |
| 30 | 0.067 | 0.349 | 0.124 | 0.191 | 0.069 |
| 40 | 0.033 | 0.290 | 0.142 | 0.175 | 0.032 |
| 60 | 0.023 | 0.233 | 0.171 | 0.195 | 0.024 |
| 80 | 0.006 | 0.196 | 0.196 | 0.203 | 0.006 |
| 100 | 0.009 | 0.173 | 0.218 | 0.227 | 0.009 |

[[diagram:tradeoff]]

Read it column by column. The empirical risk falls fast and then crawls; it even goes up between 80 and 100 rounds, the same wobble as on the ten messages. The bound \(Z_T\) sits above it every time, and falls every time, as the proof promised. The gap climbs steadily. The certified risk, their sum, falls to about 0.175 near 40 rounds and climbs again. In this run, any \(T\) from 23 to 64 certifies a true risk of at most 0.20.

And the fresh error? It keeps falling all the way to 80 rounds. The certificate is conservative: it is a promise that holds for *every* distribution and *every* vote in the class, so it charges you for the class’s size even when, as here, the extra rounds happen not to hurt. The certificate is what you can promise in writing. It is not a prediction of what will happen.

## Fix the bug

Back to the inbox. Your manager wants the filter shipped with a written promise: with probability at least 95%, its true risk on fresh mail is at most 0.20. You have the twenty thousand labelled messages and the stump learner. The only dial is the number of rounds.

In the lab you set \(T\), from 1 to 100, and train. You see the empirical risk, the bound \(Z_T\) falling under it, the gap growing, and the certified risk. Too few rounds, and the vote can’t express the true labelling: the empirical risk alone is above 0.20. Too many, and the class is so large that the gap alone eats your margin. Find a \(T\) whose certified risk is at most 0.20, and ship.

[[lab-act2]]

## What people get wrong

**“AdaBoost’s empirical risk goes down every round.”** It doesn’t have to. On the ten messages it stayed at 0.2 after round 2, and on the twenty thousand it went up between rounds 80 and 100. What goes down every round is \(Z_t\), the average of \(e^{-y_i f_t(x_i)}\), which sits above the empirical risk.

**“More rounds means a better filter.”** More rounds means a lower empirical risk and a bigger class. The VC dimension of the vote grows roughly linearly with \(T\), and so does the price on fresh mail. In the lab’s run, 100 rounds have a smaller empirical risk than 40, and a certified risk of 0.227 instead of 0.175.

**“A weak learner is a bad learner, so boosting is a trick.”** The weak learner’s promise is precise: an edge of \(\gamma\) on *every* distribution over the examples, not just on the uniform one. That is a strong assumption in disguise, and it is exactly what the proof uses each round. A stump that is right 63% of the time on the original mail but can’t beat a coin once the weights move gives AdaBoost nothing to work with.

**“The theorem says AdaBoost learns.”** The theorem is about the empirical risk only. It says nothing about fresh mail. That part needs the size of \(L(B, T)\), and the two together make the tradeoff you just navigated.

## Faces: boosting in the wild

The same recipe, counts plus stumps plus boosting, made one of the best-known detectors in computer vision: the face detector of Viola and Jones. The task is to tell whether a small grey image, 24 by 24 pixels, shows a human face. So \(\mathcal X\) is the set of \(24 \times 24\) real matrices of grey levels, and the goal is a hypothesis \(h: \mathcal X \to \{\pm 1\}\).

Each base hypothesis has the form \(h(x) = f(g(x))\), where \(f\) is a decision stump on a single number and \(g\) turns the image into that number. Every \(g\) is described by two things:

- an axis-aligned rectangle \(R\) inside the image, and
- a type, one of four masks, A to D.

To compute \(g\), stretch the mask to fill \(R\), add up the pixel values in each of the mask’s regions, and take a difference. Types A and B split the rectangle into two equal halves, side by side or one above the other, and subtract one sum from the other. Type C splits it into three strips and subtracts the two outer sums from the centre one. Type D splits it into four quarters and takes the difference between the diagonal pairs.

[[diagram:haar]]

How many functions \(g\) are there? A rectangle is fixed by its two horizontal and two vertical edges, so there are at most \(24^4\) rectangles and \(4 \cdot 24^4\) functions. Counting more carefully, the edges sit on 25 grid lines in each direction:

```python
rectangles = (25 * 24 // 2) ** 2
print(rectangles, 24 ** 4, 4 * rectangles)
```

It prints `90000 331776 360000`. So the weak learner is the stump learner of this lesson, run over a few hundred thousand features: compute every \(g\) on every training image once, then find the best stump on the weighted images in each round. The first step is made fast by a preprocessing trick, the **integral image**, which lets every rectangle sum be computed in constant time. Problem 3 asks you to invent it.

And what does AdaBoost pick? The first feature it chooses compares the band across the eyes with the band across the upper cheeks just below: the eye region is often darker than the cheeks. The second compares the eye regions with the bridge of the nose between them. Nobody told the learner that faces have eyes. It found the two cheapest rules of thumb that beat a coin, then the next one, on the faces the first two missed.

## Where boosting came from

Boosting began as a purely theoretical question. In 1988 Kearns and Valiant asked whether an efficient weak learner can always be “boosted” into an efficient strong learner. Robert Schapire, then a graduate student at MIT, answered yes in 1990, but his construction was not practical. In 1995 Freund and Schapire proposed AdaBoost, the first truly practical boosting algorithm, and it became hugely popular: a rare case of a question asked for its own sake turning into a tool everyone uses.

The question also runs in reverse. If strong learning of a class is computationally hard, then so is weak learning of it, since a weak learner could be boosted. That makes boosting useful for proving hardness results, too.

## Recap and the next bug

1. A **decision stump** compares one feature with a threshold. ERM over stumps, for any distribution over the examples, runs in \(O(dm)\) time per call after sorting each feature once, using the update \(F \leftarrow F + y_i D_i\).
2. A **\(\gamma\)-weak learner** only promises true risk at most \(\tfrac12 - \gamma\). Statistically that is no easier than strong learning (infinite VC dimension rules out both), but it can be much easier computationally.
3. **AdaBoost** reweights the examples each round so that the last hypothesis’s mistakes carry more weight, and gives each hypothesis the vote \(w_t = \tfrac12\log(1/\varepsilon_t - 1)\), the vote that shrinks \(Z_t\) the most.
4. If every round has an edge of \(\gamma\), the empirical risk after \(T\) rounds is at most \(e^{-2\gamma^2 T}\). The proof bounds the empirical risk by \(Z_T\), splits \(Z_T\) into a product of per-round ratios, and computes each ratio exactly as \(2\sqrt{\varepsilon_t(1 - \varepsilon_t)}\).
5. AdaBoost returns a member of \(L(B, T)\), a halfspace on top of \(T\) base hypotheses. Its VC dimension is at least \(T\) for stumps on the line, and at most about \(T\) times the base class’s, up to a logarithm. The number of rounds is a dial for the tradeoff between approximation and estimation error.

Your filter ships with its promise, and the inbox is quiet. Then the questions start. Why stumps, and not small trees two levels deep? Why these ten words, and not fifty? Why 40 rounds? The certificate you used covered your choice of \(T\), because you paid for all hundred values up front. But for every other knob you turned, you looked at the same twenty thousand labels, tried several settings, and kept whichever looked best on them. Each choice tuned on the same labels makes those labels look better than fresh mail will. Pay for every knob up front, and the certificate becomes too loose to say anything. So how do you choose the rounds, the depth and the features without fooling yourself? That is the next lesson: model selection and validation.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A weak learner hands AdaBoost a hypothesis with \(\varepsilon_t = 0.7\). What vote \(w_t\) does AdaBoost give it, and what happens to that hypothesis inside the final vote? Does \(Z\) still shrink this round, and by what factor? Then answer the same questions for \(\varepsilon_t = 0.5\), and say what AdaBoost should do when \(\varepsilon_t = 0\).
2. In the proof that the empirical risk falls exponentially, the formula \(w_t = \tfrac12\log(1/\varepsilon_t - 1)\) is used in exactly one step. Which one? Suppose AdaBoost used the vote \(w_t = 1\) in every round instead. For a round with \(\varepsilon_t = 0.25\), compute the ratio \(Z_{t+1}/Z_t\) with that vote and with AdaBoost’s vote. Explain why AdaBoost’s vote can never do worse in any round.
3. Your stump learner has an edge of \(\gamma = 0.05\) on every reweighting, and you have \(m\) = 20,000 examples. After how many rounds does the theorem guarantee that the empirical risk is exactly 0? For that number of rounds, evaluate the bound on \(\operatorname{VCdim}(L(B, T))\) from this lesson with \(d = 3\), and compare it with \(m\). What does the comparison tell you about using a VC-dimension argument to promise anything about fresh mail at that point?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (AdaBoost forgets its last hypothesis).** Show that the hypothesis \(h_t\) of round \(t\) has weighted risk exactly \(\tfrac12\) under the next round’s weights: for every \(t \in [T]\), \(\sum_{i=1}^m D_i^{(t+1)}\, \mathbb 1[y_i \neq h_t(x_i)] = \tfrac12\). This is the rigorous version of “the weights force the weak learner to look at the problematic examples”.
- **Problem 2 (staircases from stumps).** Prove that the vote \(h(x) = \operatorname{sign}\big(\sum_{t=1}^T w_t \operatorname{sign}(x - \theta_{t-1})\big)\), with \(\theta_0 = -\infty\), \(w_1 = (-1)^T/2\) and \(w_t = (-1)^t\) for \(t \ge 2\), equals the piecewise constant hypothesis with labels \(\alpha_i = (-1)^i\) on the pieces \((\theta_{i-1}, \theta_i]\).
- **Problem 3 (the integral image).** Let \(A\) be a \(24 \times 24\) matrix of grey levels. Its integral image \(I(A)\) is the matrix \(B\) with \(B_{i,j} = \sum_{i' \le i,\ j' \le j} A_{i',j'}\). Show that \(I(A)\) can be computed from \(A\) in time linear in the size of \(A\), and that, given \(I(A)\), every Viola and Jones feature can be computed in constant time, that is, in a time that does not depend on the size of its rectangle.
