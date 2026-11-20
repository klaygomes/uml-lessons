# The VC Dimension

## A threshold with no guarantee

The data team has shipped something you have wanted for a while. Their new scoring service reads a message, every word of it, and returns one number between 0 and 1: the message’s **spam score**. A high score means the message looks like junk. Your learner no longer has to choose among features. It has a single decision to make, and it is about where to draw a line. The new filter says:

> junk if the spam score is above \(t\).

Your users label messages as before, and the learner picks \(t\). In testing it behaves beautifully. With a few dozen labels, the line lands within a whisker of where your users would have drawn it themselves.

Then comes the launch review. Since lesson 2, every launch has needed a guarantee: how many labels, and how sure can you be? So you open the finite-class guarantee from lesson 2,

\[ m \ge \frac{\log(|\mathcal H|/\delta)}{\varepsilon}, \]

and you stop. How many hypotheses are in your class? There is one threshold rule for every real number \(t\), so \(|\mathcal H|\) is infinite, and so is \(\log |\mathcal H|\). The agnostic version from lesson 4 needs \(\log(2|\mathcal H|/\delta)/(2\varepsilon^2)\) examples: infinite again. Every guarantee you own says “you need infinitely many labels”, which is a polite way of saying nothing.

You try the trick from lesson 4. A computer stores \(t\) as a 64-bit number, so in practice the class has at most \(2^{64}\) members, and the finite-class guarantee applies after all:

```python
from math import ceil, log

eps, delta = 0.05, 0.05
by_bits = ceil((64 * log(2) + log(1 / delta)) / eps)
by_threshold_argument = ceil(log(2 / delta) / eps)
print(by_bits, by_threshold_argument)
```

It prints `948 74`. Ignore the second number for now; it is where this lesson is heading. The first says 948 labels buy a true risk of at most 5% with probability at least 95%.

The reviewer is not impressed, for two reasons. The first: if the machine moved to 128-bit numbers, your guarantee would double, while the rule, the users and the messages stayed exactly the same. A number that changes with the hardware is measuring the hardware, not the problem. The second reason is sharper. Lesson 5 proved that the class of all functions on an infinite domain cannot be learned at all, by any learner, from any number of examples. That class is infinite too. So infinitely many hypotheses *can* make learning impossible, and you have no tool that tells your infinite class apart from that one. Launch blocked, until you can say why your class is different.

Yet you have watched thresholds learn. So something about a class matters other than the number of hypotheses in it. What is it? This lesson finds it, gives it a name, and proves that it decides, completely, which classes can be learned.

## Thresholds learn anyway

First let’s show that the reviewer is wrong about thresholds in particular, by proving a guarantee directly.

Here is the setting, in the language of lessons 2 and 3. A message is now represented by its spam score, so the domain set is \(\mathcal X = [0, 1]\). For each real number \(t\), the **threshold** hypothesis is

\[ h_t(x) = \mathbb 1[x > t], \]

which says junk (1) when the score is above \(t\) and keep (0) otherwise. The class of thresholds is \(\mathcal H_{\mathrm{thr}} = \{ h_t : t \in \mathbb R \}\). We assume realizability: the true labelling is \(f = h_{t^\star}\) for some unknown \(t^\star\). Recall from lesson 3 that a class is **PAC learnable** when some learner, given at least \(m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples from any distribution, labelled by any \(f\) the class realizes, returns a hypothesis with true risk at most \(\varepsilon\) with probability at least \(1 - \delta\). We need one mild assumption about the scores: no single score value carries positive probability. A score computed from a whole message is a continuous quantity, so this is a fair assumption, and it lets us cut the score line into pieces of exactly the probability we want.

**Claim (thresholds are PAC learnable).** Let \(\varepsilon, \delta \in (0, 1)\). Every ERM learner for \(\mathcal H_{\mathrm{thr}}\), given

\[ m \ge \frac{\log(2/\delta)}{\varepsilon} \]

examples, returns a hypothesis with \(L_{\mathcal D}(h_S) \le \varepsilon\) with probability at least \(1 - \delta\). So \(m_{\mathcal H_{\mathrm{thr}}}(\varepsilon, \delta) \le \lceil \log(2/\delta)/\varepsilon \rceil\).

Before the proof, a picture of where the mistakes can live. Draw the score line with \(t^\star\) marked. Just to the left of \(t^\star\) lies a strip of scores that carries probability exactly \(\varepsilon\); just to the right lies another. If the training sequence puts at least one example in each strip, the learner’s threshold is trapped between them, and every mistake it can make lies inside one strip.

[[diagram:strips]]

*Proof.* Choose \(a_0 < t^\star\) so that \(\mathbb P_{x \sim \mathcal D}[x \in (a_0, t^\star]] = \varepsilon\), and \(a_1 > t^\star\) so that \(\mathbb P[x \in (t^\star, a_1]] = \varepsilon\). Such points exist because no single score carries probability, so the probability of \((a, t^\star]\) moves continuously as \(a\) slides. If all the scores below \(t^\star\) together carry probability at most \(\varepsilon\), set \(a_0 = -\infty\) instead, and similarly \(a_1 = +\infty\).

Now look at what ERM returns. Let \(b_0\) be the largest score among the keep examples in \(S\), and \(b_1\) the smallest score among the junk examples (\(b_0 = -\infty\) if there is no keep example, \(b_1 = +\infty\) if there is no junk example). Under realizability, \(h_{t^\star}\) has empirical risk 0, so ERM returns some consistent \(h_t\), and \(h_t\) is consistent exactly when every keep score is at most \(t\) and every junk score is above \(t\): that is, \(b_0 \le t < b_1\).

Where can \(h_t\) be wrong? It disagrees with the truth exactly on the scores between \(t\) and \(t^\star\): on \((t, t^\star]\) if \(t < t^\star\), and on \((t^\star, t]\) if \(t \ge t^\star\). Suppose \(S\) has an example in each strip. An example in \((a_0, t^\star]\) is a keep example, so \(b_0 > a_0\), so \(t > a_0\), and when \(t < t^\star\) the mistakes lie in \((a_0, t^\star]\), probability at most \(\varepsilon\). An example in \((t^\star, a_1]\) is a junk example, so \(b_1 \le a_1\), so \(t < a_1\), and when \(t \ge t^\star\) the mistakes lie in \((t^\star, a_1]\), probability at most \(\varepsilon\). (If a strip was set to infinity, the same conclusion holds for free, since all the scores on that side carry at most \(\varepsilon\).) So \(L_{\mathcal D}(h_S) > \varepsilon\) can only happen if some strip is empty.

What is the chance that the left strip receives no example? Each of the \(m\) scores misses it with probability \(1 - \varepsilon\), and they are drawn independently, so all of them miss it with probability \((1 - \varepsilon)^m \le e^{-\varepsilon m}\), using \(1 - \varepsilon \le e^{-\varepsilon}\) from lesson 2. The same holds for the right strip. By the union bound,

\[ \mathbb P_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(h_S) > \varepsilon \right] \le 2 e^{-\varepsilon m}, \]

and \(2e^{-\varepsilon m} \le \delta\) exactly when \(m \ge \log(2/\delta)/\varepsilon\). \(\blacksquare\)

Let’s watch it happen, and take the worst learner the claim allows: among all the consistent thresholds, it always returns the one with the largest true risk. With \(t^\star = 0.6\), uniform scores and \(\varepsilon = \delta = 0.1\):

```python
import math
import random

eps, delta, t_star = 0.1, 0.1, 0.6
m = math.ceil(math.log(2 / delta) / eps)

def worst_risk(S):
    b0 = max([x for x, y in S if y == 0], default=0.0)
    b1 = min([x for x, y in S if y == 1], default=1.0)
    return max(t_star - b0, b1 - t_star)

rng, trials, fails = random.Random(0), 100_000, 0
for _ in range(trials):
    xs = [rng.random() for _ in range(m)]
    fails += worst_risk([(x, int(x > t_star)) for x in xs]) > eps
print(m, fails / trials, round(2 * math.exp(-eps * m), 4))
```

It prints `30 0.08507 0.0996`. Thirty labels; even the most unlucky ERM learner fails about 8.5% of the time (the exact figure is \(2(0.9)^{30} - (0.8)^{30} \approx 8.35\%\)); and the bound \(2e^{-3} \approx 9.96\%\) sits just above, under \(\delta = 10\%\). With \(\varepsilon = \delta = 0.05\), the same claim asks for 74 labels: the second number the first snippet printed, against 948 from counting floating-point numbers.

Notice what the proof never did. It never counted hypotheses. It used one fact about thresholds: two well-placed examples, one on each side of \(t^\star\), pin the threshold down. That is a fact about how thresholds can *label* points, and it is the thread we pull next.

## What a class can do to a few messages

Take a small set of scores, say \(C = \{0.3, 0.7\}\), and ask: in how many different ways can thresholds label these two messages? Each hypothesis gives a pair of labels, one per score. Different thresholds may give the same pair, and we only care about the distinct pairs.

**Definition (restriction).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\{0, 1\}\), and let \(C = \{c_1, \dots, c_m\} \subset \mathcal X\). The **restriction** of \(\mathcal H\) to \(C\) is the set of label vectors that hypotheses in \(\mathcal H\) produce on \(C\):

\[ \mathcal H_C = \{ (h(c_1), \dots, h(c_m)) : h \in \mathcal H \} \subseteq \{0, 1\}^m. \]

Each vector in \(\mathcal H_C\) is a **labelling** of \(C\). There are \(2^m\) possible labellings of \(m\) points, so \(|\mathcal H_C| \le 2^m\), however many hypotheses \(\mathcal H\) has. Here it is in Python for thresholds:

```python
def threshold(t):
    return lambda x: int(x > t)

def restriction(H, C):
    return sorted({tuple(h(c) for c in C) for h in H})

def cuts(C):
    p = sorted(C)
    mids = [(a + b) / 2 for a, b in zip(p, p[1:])]
    return [p[0] - 1] + mids + [p[-1]]

for C in ([0.3], [0.3, 0.7]):
    print(C, restriction([threshold(t) for t in cuts(C)], C))
```

It prints

```
[0.3] [(0,), (1,)]
[0.3, 0.7] [(0, 0), (0, 1), (1, 1)]
```

Why is it enough to try a handful of thresholds when there are infinitely many? Because the labelling \(h_t\) gives to \(C\) depends only on which points of \(C\) lie above \(t\). Two thresholds in the same gap between consecutive points give the same labelling, so one threshold per gap (below everything, between each neighbouring pair, at or above the top) produces every labelling there is. An infinite class, seen through two points, is just three labellings.

And look at which one is missing: \((1, 0)\), junk at 0.3 and keep at 0.7. No threshold can do that, because a threshold that calls 0.3 junk calls everything above 0.3 junk too.

**Definition (shattering).** A class \(\mathcal H\) **shatters** a finite set \(C \subset \mathcal X\) when its restriction to \(C\) contains every labelling: \(|\mathcal H_C| = 2^{|C|}\).

Thresholds shatter every single point \(\{c\}\): take \(t = c - 1\) for junk and \(t = c + 1\) for keep. They shatter no set of two points \(c_1 < c_2\), because \((1, 0)\) is always missing.

[[diagram:restriction]]

Why should shattering matter? Think back to the No-Free-Lunch theorem of lesson 5. Its adversary took a set \(C\) of \(2m\) messages and looked at all \(2^{2m}\) ways to label them. For each labelling it built a distribution, uniform on \(C\) and labelled that way. A learner that has seen at most \(m\) of the \(2m\) messages knows nothing about the labels of the others, and the adversary chose the labelling that made the learner pay. The conclusion: some distribution makes the learner’s true risk at least \(1/8\) with probability at least \(1/7\).

In the PAC model, though, the adversary is restricted: it may only use labellings that some \(h \in \mathcal H\) achieves with risk 0. On \(C\), that means labellings in \(\mathcal H_C\). Now suppose \(\mathcal H\) shatters \(C\). Then every labelling of \(C\) is in \(\mathcal H_C\), the restriction costs the adversary nothing, and the whole argument goes through inside the PAC model.

**Corollary (shattering defeats every learner).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\{0, 1\}\), and \(m\) a training sequence size. If some set \(C \subset \mathcal X\) of size \(2m\) is shattered by \(\mathcal H\), then for every learner \(A\) there exist a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\) and a hypothesis \(h \in \mathcal H\) with \(L_{\mathcal D}(h) = 0\), such that with probability at least \(1/7\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \ge 1/8\).

*Proof idea.* Run the No-Free-Lunch construction on the shattered set \(C\). Each of its labellings \(f_i: C \to \{0,1\}\) equals the restriction of some \(h_i \in \mathcal H\), because \(\mathcal H\) shatters \(C\), and \(h_i\) has true risk 0 under the distribution built from \(f_i\). So every distribution the adversary uses is realizable by \(\mathcal H\), and the theorem’s conclusion applies to it unchanged. \(\blacksquare\)

There is an old saying behind this: an explanation that can explain anything explains nothing. If the class can label half the set any way at all and the other half any way at all, then the labels you have seen say nothing about the labels you have not.

## The VC dimension

So the question to ask of a class is not “how many hypotheses?” but “how large a set can it shatter?”

**Definition (VC dimension).** The **VC dimension** of a class \(\mathcal H\), written \(\mathrm{VCdim}(\mathcal H)\), is the largest size of a set \(C \subset \mathcal X\) that \(\mathcal H\) shatters. If \(\mathcal H\) shatters sets of every finite size, its VC dimension is infinite.

The name honours Vladimir Vapnik and Alexey Chervonenkis, who discovered this characterization in 1970. An immediate consequence of the corollary:

**Theorem (infinite VC dimension means not PAC learnable).** If \(\mathrm{VCdim}(\mathcal H) = \infty\), then \(\mathcal H\) is not PAC learnable.

*Proof.* Suppose some learner \(A\) PAC learns \(\mathcal H\), and pick \(\varepsilon < 1/8\) and \(\delta < 1/7\). Let \(m = m_{\mathcal H}(\varepsilon, \delta)\). Since the VC dimension is infinite, \(\mathcal H\) shatters some set of size \(2m\). By the corollary there is a realizable distribution on which, with probability at least \(1/7 > \delta\), \(A\) returns a hypothesis with true risk at least \(1/8 > \varepsilon\). That contradicts the PAC guarantee. \(\blacksquare\)

The rest of the lesson proves the converse: a finite VC dimension guarantees learnability. But first let’s measure some classes. To prove \(\mathrm{VCdim}(\mathcal H) = d\), you need two things, and they are not symmetric:

1. **Some** set of size \(d\) is shattered by \(\mathcal H\). One example is enough.
2. **Every** set of size \(d + 1\) fails to be shattered. One example is not enough; you need an argument that works for any set.

For thresholds, step 1 is any single point, and step 2 is the missing labelling \((1, 0)\), which we showed is missing for every pair \(c_1 < c_2\). So \(\mathrm{VCdim}(\mathcal H_{\mathrm{thr}}) = 1\). An infinite class with VC dimension 1. Keep that in mind; it is what you will tell the reviewer.

## Four classes, measured

**Intervals.** Suppose junk is not “above a line” but “inside a band of scores”. The class of **intervals** is \(\mathcal H_{\mathrm{int}} = \{ h_{a,b} : a \le b \}\) with

\[ h_{a,b}(x) = \mathbb 1[a \le x \le b]. \]

(Whether the endpoints belong to the interval changes nothing below.) Two points \(\{1, 2\}\) are shattered: an interval can cover neither, either one, or both. Three points \(c_1 < c_2 < c_3\) are never shattered, because the labelling \((1, 0, 1)\) would need an interval containing \(c_1\) and \(c_3\) but not \(c_2\), which sits between them. So \(\mathrm{VCdim}(\mathcal H_{\mathrm{int}}) = 2\). Python agrees, trying intervals whose ends sit at the points themselves (and one past the end for the empty labelling), which again covers every labelling:

```python
def interval(a, b):
    return lambda x: int(a <= x <= b)

def restriction(H, C):
    return {tuple(h(c) for c in C) for h in H}

def all_intervals(C):
    ends = sorted(C) + [max(C) + 1]
    return [interval(a, b) for a in ends for b in ends if a <= b]

for C in ([1, 2], [1, 2, 3]):
    HC = restriction(all_intervals(C), C)
    print(C, len(HC), "of", 2 ** len(C))
print("missing:", {(1, 0, 1)} - HC)
```

It prints `[1, 2] 4 of 4`, then `[1, 2, 3] 7 of 8`, then `missing: {(1, 0, 1)}`.

**Axis-aligned rectangles.** Now give each message two scores: its spam score \(x_1\) and a second score \(x_2\) for its sender’s reputation. A rectangle rule says junk when both scores fall in a range:

\[ h_{(a_1, a_2, b_1, b_2)}(x_1, x_2) = \mathbb 1[a_1 \le x_1 \le a_2 \text{ and } b_1 \le x_2 \le b_2], \]

for \(a_1 \le a_2\) and \(b_1 \le b_2\). Call this class \(\mathcal H_{\mathrm{rec}}\). Its VC dimension is 4.

For step 1, place four points in a diamond: top, right, bottom, left. Whatever subset you want labelled junk, a rectangle can wrap exactly those points, because each point of a diamond sticks out in its own direction and can be left out by pulling in one edge.

For step 2, take any five points. Pick a leftmost one (smallest first coordinate), a rightmost one, a lowest one and a highest one; that is at most four points, so at least one point, call it \(c_5\), was not picked. Now ask for the labelling “junk on the picked points, keep on \(c_5\)”. Any rectangle containing the picked points spans from the leftmost to the rightmost first coordinate and from the lowest to the highest second coordinate. The coordinates of \(c_5\) lie inside both ranges, so the rectangle contains \(c_5\) too. That labelling is impossible, so no five points are shattered.

[[diagram:rectangles]]

To check this by brute force, use one fact: if any rectangle achieves a labelling, so does the *tightest* rectangle around the junk points, since it sits inside the other one and so contains no extra points. So a labelling is achievable exactly when the tightest rectangle around its junk points contains none of its keep points.

```python
from itertools import product
import random

def realizable(C, ys):
    pos = [c for c, y in zip(C, ys) if y] or [(9, 9)]
    xs, zs = [p[0] for p in pos], [p[1] for p in pos]
    return all((min(xs) <= x <= max(xs) and
                min(zs) <= z <= max(zs)) == bool(y)
               for (x, z), y in zip(C, ys))

def shatters(C):
    ys = product([0, 1], repeat=len(C))
    return all(realizable(C, y) for y in ys)

diamond = [(0, 1), (1, 0), (0, -1), (-1, 0)]
r = random.Random(0).random
fives = [[(r(), r()) for _ in range(5)] for _ in range(10**4)]
print(shatters(diamond), sum(map(shatters, fives)))
```

It prints `True 0`: the diamond is shattered, and not one of ten thousand random five-point sets is. (The point \((9, 9)\) stands in for “no junk points”: a tiny rectangle far away contains nothing.) The random sets are evidence, not proof; the proof is the leftmost, rightmost, lowest, highest argument.

**Finite classes.** If \(\mathcal H\) is finite, then \(|\mathcal H_C| \le |\mathcal H|\) for every \(C\), since each labelling comes from at least one hypothesis. Shattering \(C\) needs \(2^{|C|}\) labellings, so it is impossible once \(2^{|C|} > |\mathcal H|\). Hence \(\mathrm{VCdim}(\mathcal H) \le \log_2 |\mathcal H|\). This bound can be very loose. Take the scores \(\{1, 2, \dots, 16\}\) and thresholds on them:

```python
from itertools import combinations
from math import log2

def vc_dim(H, X):
    d = 0
    for size in range(1, len(X) + 1):
        for C in combinations(X, size):
            HC = {tuple(h(c) for c in C) for h in H}
            if len(HC) == 2 ** size:
                d = size
    return d

k = 16
X = list(range(1, k + 1))
H = [lambda x, t=t: int(x > t) for t in range(k + 1)]
print(len(H), vc_dim(H, X), round(log2(len(H)), 2))
```

It prints `17 1 4.09`: seventeen distinct thresholds, VC dimension 1, while \(\log_2 |\mathcal H| \approx 4.09\). With \(k\) scores the class has \(k + 1\) members and VC dimension 1, so the gap between \(\log_2 |\mathcal H|\) and the VC dimension grows without limit. The finite-class guarantee pays for \(\log |\mathcal H|\); we are about to see that the right price is the VC dimension.

**Unions of intervals.** Here is a class that is too rich. Let \(\mathcal H_{\cup}\) be every finite union of intervals: junk if the score falls in any one of finitely many bands, as many as you like. Given any finite set \(C\) and any labelling of it, put a tiny interval around each junk point and nothing around the keep points. That realizes the labelling, so \(\mathcal H_{\cup}\) shatters every finite set:

```python
from itertools import product
import random

def bands_around(points, w=1e-9):
    return lambda x: int(any(abs(x - p) <= w for p in points))

rng = random.Random(0)
C = sorted(rng.random() for _ in range(12))
realized = set()
for ys in product([0, 1], repeat=len(C)):
    h = bands_around([c for c, y in zip(C, ys) if y])
    realized.add(tuple(h(c) for c in C))
print(len(realized), 2 ** len(C))
```

It prints `4096 4096`: all \(2^{12}\) labellings of twelve random scores. So \(\mathrm{VCdim}(\mathcal H_{\cup}) = \infty\), and by the theorem above, \(\mathcal H_{\cup}\) is not PAC learnable. It is the memorizer of lesson 1, dressed up for the score line: it can wrap every junk example in its own little band and say nothing useful about the rest.

**VC dimension and the number of parameters.** In these examples the VC dimension equals the number of numbers that define a hypothesis: one for a threshold, two for an interval, four for a rectangle. That is often the case, and a handy first guess. It is not a law. The class \(\{ x \mapsto \lceil 0.5 \sin(\theta x) \rceil : \theta \in \mathbb R \}\) on the real line has one parameter and infinite VC dimension: for every \(d\) there are \(d\) points it shatters. (Proving it is a good exercise in binary expansions.)

Now it’s your turn to hunt for labellings no class can make.

[[lab]]

## The fundamental theorem

We have shown one direction: infinite VC dimension rules out PAC learning. The converse is also true, and together they form the central result of this part of the course. It ties together every idea from lessons 2 to 5: learnability, ERM, uniform convergence and now the VC dimension. Recall from lesson 4 that a class has the **uniform convergence property** when there is a function \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) such that, for every distribution \(\mathcal D\), a training sequence of at least that many i.i.d. examples is \(\varepsilon\)-representative (every \(h \in \mathcal H\) has \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\)) with probability at least \(1 - \delta\).

**The fundamental theorem of statistical learning.** Let \(\mathcal H\) be a class of functions from a domain \(\mathcal X\) to \(\{0, 1\}\), with the 0-1 loss. The following are equivalent:

1. \(\mathcal H\) has the uniform convergence property.
2. Any ERM rule is a successful agnostic PAC learner for \(\mathcal H\).
3. \(\mathcal H\) is agnostic PAC learnable.
4. \(\mathcal H\) is PAC learnable.
5. Any ERM rule is a successful PAC learner for \(\mathcal H\).
6. \(\mathcal H\) has a finite VC dimension.

Pause on how much this says. Six properties that look quite different, one statistical (uniform convergence), three about learners, one combinatorial, are the same property. In particular, if a binary classification problem can be learned at all, by any clever learner, then plain ERM learns it, and uniform convergence is the reason.

The VC dimension does more than decide learnability. It also sets the price.

**The fundamental theorem, quantitative version.** Let \(\mathcal H\) be as above with \(\mathrm{VCdim}(\mathcal H) = d < \infty\). There are absolute constants \(C_1, C_2\) such that:

1. \(\mathcal H\) has the uniform convergence property with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon^2} \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d + \log(1/\delta)}{\varepsilon^2}. \]
2. \(\mathcal H\) is agnostic PAC learnable with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon^2} \le m_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d + \log(1/\delta)}{\varepsilon^2}. \]
3. \(\mathcal H\) is PAC learnable with
\[ C_1 \frac{d + \log(1/\delta)}{\varepsilon} \le m_{\mathcal H}(\varepsilon, \delta) \le C_2 \frac{d \log(1/\varepsilon) + \log(1/\delta)}{\varepsilon}. \]

Compare with lessons 2 and 4: wherever \(\log |\mathcal H|\) stood, \(d\) now stands. That is the answer to the reviewer in one line. This quantitative version is proved in a later lesson. Its proof needs tools we don’t have yet; the proof below gives a weaker bound with the same message.

One caution about scope. Both versions are about binary classification with the 0-1 loss. Similar results hold for some other problems, such as regression with the absolute or squared loss, but not for every learning problem: there are problems that can be learned even though uniform convergence fails, and problems where ERM fails while some other learner succeeds. Later lessons meet both.

[[diagram:theorem-map]]

## The easy arrows

Most of the proof is a chain of short steps. Let’s walk around the circle.

**1 implies 2.** This is lesson 4. If \(S\) is \(\varepsilon/2\)-representative, then any ERM hypothesis has true risk within \(\varepsilon\) of the best in the class, and uniform convergence makes \(S\) \(\varepsilon/2\)-representative with probability at least \(1 - \delta\) once \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\).

**2 implies 3.** An ERM rule is a learner. If it succeeds, the class is agnostic PAC learnable.

**3 implies 4, and 2 implies 5.** PAC learning is the special case of agnostic PAC learning in which some \(h \in \mathcal H\) has true risk 0. In that case \(\min_{h \in \mathcal H} L_{\mathcal D}(h) = 0\), and the agnostic promise “true risk at most \(\min_h L_{\mathcal D}(h) + \varepsilon\)” becomes the PAC promise “true risk at most \(\varepsilon\)”. The same learner, with the same sample complexity, keeps it.

**4 implies 6, and 5 implies 6.** Read them backwards. If the VC dimension were infinite, the theorem in “The VC dimension” says no learner PAC learns \(\mathcal H\), so 4 fails. And 5 implies 4, since a successful ERM rule is a PAC learner, so 5 fails too.

**6 implies 1.** This is the hard one, and the rest of the lesson’s theory is its proof. The plan has two parts. First, **Sauer’s lemma**: a class of VC dimension \(d\), however large, produces only polynomially many labellings of \(m\) points, roughly \(m^d\) instead of \(2^m\). Second, a generalization of lesson 4: uniform convergence holds whenever a class has few labellings in this sense, which we will call a small **effective size**.

## Counting labellings: the growth function

We need a name for “how many labellings of \(m\) points, at most”.

**Definition (growth function).** The **growth function** of \(\mathcal H\) is \(\tau_{\mathcal H}: \mathbb N \to \mathbb N\),

\[ \tau_{\mathcal H}(m) = \max_{C \subset \mathcal X : |C| = m} |\mathcal H_C|. \]

In words: the largest number of distinct labellings that \(\mathcal H\) produces on any set of \(m\) points. Always \(\tau_{\mathcal H}(m) \le 2^m\), and if \(m \le \mathrm{VCdim}(\mathcal H)\), then some set of size \(m\) is shattered, so \(\tau_{\mathcal H}(m) = 2^m\). The interesting question is what happens past the VC dimension. Let’s just count, for thresholds and intervals. On the line, only the order of the points matters, so the points \(0, 1, \dots, m-1\) are as good as any \(m\) distinct scores. The last columns compute a sum of binomial coefficients that will turn out to be the answer:

```python
from math import comb

def tau(H, m):
    C = list(range(m))
    return len({tuple(h(c) for c in C) for h in H})

def sauer(m, d):
    return sum(comb(m, i) for i in range(d + 1))

for m in range(1, 9):
    cuts = [c - 0.5 for c in range(m + 2)]
    thr = [lambda x, t=t: x > t for t in cuts]
    itv = [lambda x, a=a, b=b: a < x < b
           for a in cuts for b in cuts if a < b]
    print(m, tau(thr, m), sauer(m, 1),
          tau(itv, m), sauer(m, 2), 2 ** m)
```

It prints

```
1 2 2 2 2 2
2 3 3 4 4 4
3 4 4 7 7 8
4 5 5 11 11 16
5 6 6 16 16 32
6 7 7 22 22 64
7 8 8 29 29 128
8 9 9 37 37 256
```

Read the columns as: \(m\), \(\tau_{\mathrm{thr}}(m)\), \(\sum_{i \le 1} \binom{m}{i}\), \(\tau_{\mathrm{int}}(m)\), \(\sum_{i \le 2} \binom{m}{i}\), \(2^m\). Thresholds produce \(m + 1\) labellings (a run of keeps, then a run of junks, split at one of \(m + 1\) places). Intervals produce \(\binom{m}{2} + m + 1\) (one block of junk, chosen by its two ends, or a single point, or nothing). Both grow like a polynomial of degree equal to the VC dimension, while \(2^m\) runs away. And both match the binomial sums exactly. That is no coincidence.

[[diagram:growth]]

## Sauer’s lemma

**Sauer’s lemma.** Let \(\mathcal H\) be a class with \(\mathrm{VCdim}(\mathcal H) \le d < \infty\). Then for every \(m\),

\[ \tau_{\mathcal H}(m) \le \sum_{i=0}^{d} \binom{m}{i}. \]

In particular, if \(1 \le d \le m\), then \(\tau_{\mathcal H}(m) \le (em/d)^d\).

It is also called the Sauer-Shelah-Perles lemma: Sauer, Shelah with Perles, and Vapnik and Chervonenkis each found it independently.

The proof is a beautiful piece of counting, and the trick is to prove something stronger. Instead of counting labellings against a formula, count them against the shattered subsets.

**Stronger claim.** For every class \(\mathcal H\) (non-empty) and every finite set \(C = \{c_1, \dots, c_m\}\),

\[ |\mathcal H_C| \le \left| \{ B \subseteq C : \mathcal H \text{ shatters } B \} \right|. \]

The empty set counts as shattered, since its single labelling (the empty one) is produced by any hypothesis.

*Why the stronger claim gives Sauer’s lemma.* If \(\mathrm{VCdim}(\mathcal H) \le d\), no subset of size above \(d\) is shattered, so the shattered subsets of \(C\) are among the subsets of size \(0, 1, \dots, d\), of which there are \(\sum_{i=0}^{d} \binom{m}{i}\). Apply this to the set \(C\) that achieves the maximum in \(\tau_{\mathcal H}(m)\).

*Proof of the stronger claim,* by induction on \(m\), for all classes at once.

*Base case, \(m = 1\).* Either every hypothesis gives \(c_1\) the same label, so \(|\mathcal H_C| = 1\) and the only shattered subset is \(\emptyset\); or both labels occur, so \(|\mathcal H_C| = 2\) and both \(\emptyset\) and \(\{c_1\}\) are shattered. Equality either way.

*Inductive step.* Assume the claim for every class and every set of fewer than \(m\) points. Fix \(\mathcal H\) and \(C = \{c_1, \dots, c_m\}\), and let \(C' = \{c_2, \dots, c_m\}\), the set without its first point. Split the labellings in \(\mathcal H_C\) by their **tail**, the labels on \(C'\):

\[ Y_0 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ or } (1, y_2, \dots, y_m) \in \mathcal H_C \}, \]

\[ Y_1 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ and } (1, y_2, \dots, y_m) \in \mathcal H_C \}. \]

So \(Y_0\) holds every tail that occurs, and \(Y_1\) the tails that occur with both labels on \(c_1\).

*Step 1: \(|\mathcal H_C| = |Y_0| + |Y_1|\).* Group the labellings in \(\mathcal H_C\) by tail. A tail in \(Y_0\) but not \(Y_1\) occurs with exactly one label on \(c_1\), so it accounts for one labelling. A tail in \(Y_1\) accounts for two. Since \(Y_1 \subseteq Y_0\), the total is \(|Y_0 \setminus Y_1| + 2|Y_1| = |Y_0| + |Y_1|\).

*Step 2: bound \(Y_0\).* \(Y_0\) is exactly the restriction of \(\mathcal H\) to \(C'\): the tails that occur are the labellings of \(C'\). By the induction hypothesis on \(C'\),

\[ |Y_0| = |\mathcal H_{C'}| \le |\{ B \subseteq C' : \mathcal H \text{ shatters } B \}| = |\{ B \subseteq C : c_1 \notin B,\ \mathcal H \text{ shatters } B \}|. \]

*Step 3: bound \(Y_1\) through a smaller class.* Let \(\mathcal H'\) be the hypotheses that have a **partner**: another hypothesis in \(\mathcal H\) that agrees with it on \(C'\) and disagrees on \(c_1\),

\[ \mathcal H' = \{ h \in \mathcal H : \exists h' \in \mathcal H,\ h'(c_1) = 1 - h(c_1),\ h'(c_i) = h(c_i) \text{ for } i = 2, \dots, m \}. \]

A tail is in \(Y_1\) exactly when it is the tail of a hypothesis with a partner, so \(Y_1 = \mathcal H'_{C'}\). By the induction hypothesis, now applied to the class \(\mathcal H'\) on \(C'\),

\[ |Y_1| = |\mathcal H'_{C'}| \le |\{ B \subseteq C' : \mathcal H' \text{ shatters } B \}|. \]

*Step 4: each set shattered by \(\mathcal H'\) grows by \(c_1\).* Suppose \(\mathcal H'\) shatters \(B \subseteq C'\). Take any labelling of \(B \cup \{c_1\}\). Some \(g \in \mathcal H'\) produces its part on \(B\). The partner \(g'\) of \(g\) agrees with \(g\) on all of \(C'\), hence on \(B\), and gives \(c_1\) the other label; and \(g'\) is itself in \(\mathcal H'\), with \(g\) as its partner. One of \(g, g'\) gives \(c_1\) the label we want. So \(\mathcal H'\) shatters \(B \cup \{c_1\}\), and since \(\mathcal H' \subseteq \mathcal H\), so does \(\mathcal H\). The map \(B \mapsto B \cup \{c_1\}\) is one-to-one from subsets of \(C'\) to subsets of \(C\) that contain \(c_1\), so

\[ |Y_1| \le |\{ B \subseteq C : c_1 \in B,\ \mathcal H \text{ shatters } B \}|. \]

*Step 5: add.* The shattered subsets of \(C\) split into those without \(c_1\) (Step 2) and those with it (Step 4), so

\[ |\mathcal H_C| = |Y_0| + |Y_1| \le |\{ B \subseteq C : \mathcal H \text{ shatters } B \}|. \qquad \blacksquare \]

[[diagram:sauer-split]]

Why does Step 3 switch to \(\mathcal H'\) instead of using \(\mathcal H\) again? Because \(\mathcal H\) itself may shatter sets of \(C'\) that it cannot extend by \(c_1\). The partner class keeps only the hypotheses that come in flip pairs at \(c_1\), and those can always extend.

The stronger claim says something about every set of labellings, not only those from tidy classes, so we can test it on random ones. Each trial picks a random set \(\mathcal H_C\) of labellings of five points and checks Step 1 and the claim:

```python
from itertools import combinations, product
import random

def shattered(HC, B):
    patterns = {tuple(v[i] for i in B) for v in HC}
    return len(patterns) == 2 ** len(B)

rng, cube = random.Random(1), list(product([0, 1], repeat=5))
for trial in range(2000):
    HC = set(rng.sample(cube, rng.randint(1, 32)))
    Y0 = {v[1:] for v in HC}
    Y1 = {v[1:] for v in HC if (1 - v[0],) + v[1:] in HC}
    n_sh = sum(shattered(HC, B) for k in range(6)
               for B in combinations(range(5), k))
    assert len(HC) == len(Y0) + len(Y1) <= n_sh
print(len(HC), len(Y0), len(Y1), n_sh)
```

All 2,000 assertions hold, and it prints the last trial, `23 14 9 26`: 23 labellings, which split into 14 tails plus 9 doubled tails, against 26 shattered subsets.

**The polynomial bound.** It remains to show \(\sum_{i=0}^{d} \binom{m}{i} \le (em/d)^d\) when \(1 \le d \le m\). Since \(d/m \le 1\), raising it to a smaller power makes it bigger, so \((d/m)^d \le (d/m)^i\) for every \(i \le d\). Hence

\[ \left(\frac{d}{m}\right)^d \sum_{i=0}^{d} \binom{m}{i} \le \sum_{i=0}^{d} \binom{m}{i} \left(\frac{d}{m}\right)^i \le \sum_{i=0}^{m} \binom{m}{i} \left(\frac{d}{m}\right)^i = \left(1 + \frac{d}{m}\right)^m \le e^d. \]

The second step adds non-negative terms, the third is the binomial theorem, and the last uses \(1 + u \le e^u\). Multiplying both sides by \((m/d)^d\) gives the bound. A quick check over every \(1 \le d \le m < 200\), and a look at the sizes involved for \(d = 2\):

```python
from math import comb, e

def sauer(m, d):
    return sum(comb(m, i) for i in range(d + 1))

ratio = max(sauer(m, d) / (e * m / d) ** d
            for m in range(1, 200) for d in range(1, m + 1))
print(round(ratio, 4))
for m in (10, 100, 1000):
    poly = round((e * m / 2) ** 2)
    print(m, sauer(m, 2), poly, len(str(2 ** m)))
```

It prints `0.7358`, so the sum never exceeds the bound, and then `10 56 185 4`, `100 5051 18473 31`, `1000 500501 1847264 302`. At \(m = 1{,}000\), a class of VC dimension 2 produces at most 500,501 labellings, under the bound of about 1.8 million, while \(2^{1000}\) has 302 digits. That gap is the whole story.

## Uniform convergence from a small effective size

Now the second part of the plan. Lesson 4 proved uniform convergence for a finite class by a union bound over its \(|\mathcal H|\) hypotheses, paying \(\log |\mathcal H|\). The idea here is to pay only for the labellings, \(\log \tau_{\mathcal H}\). Throughout, \(\mathcal D\) is a distribution over examples \(z = (x, y) \in \mathcal X \times \{0, 1\}\), the loss is \(\ell(h, z) = \mathbb 1[h(x) \ne y]\), and we assume \(\mathcal X\) has at least \(2m\) points (true for the score line).

**Theorem (uniform convergence from a small growth function).** For every class \(\mathcal H\), every distribution \(\mathcal D\) and every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H\) satisfies

\[ |L_{\mathcal D}(h) - L_S(h)| \le \frac{4 + 2\sqrt{\log \tau_{\mathcal H}(2m)}}{\delta \sqrt{2m}}. \]

The proof bounds the expected worst gap, \(\mathbb E_{S \sim \mathcal D^m}\left[\sup_{h \in \mathcal H} |L_{\mathcal D}(h) - L_S(h)|\right]\), by \((4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/\sqrt{2m}\), and then converts expectation into probability. It is the deepest proof in the lesson, so let’s take it in seven steps, each saying what changes and why.

*Step 1: from expectation to probability.* Let \(Z = \sup_{h} |L_{\mathcal D}(h) - L_S(h)|\), a non-negative random variable. **Markov’s inequality** says that for a non-negative \(Z\) and any \(a > 0\), \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\). (Proof: \(Z \ge a \cdot \mathbb 1[Z \ge a]\) pointwise; take expectations.) With \(a = \mathbb E[Z]/\delta\), the probability that \(Z \ge \mathbb E[Z]/\delta\) is at most \(\delta\). So a bound on \(\mathbb E[Z]\), divided by \(\delta\), holds with probability at least \(1 - \delta\). (If \(\mathbb E[Z] = 0\), then \(Z = 0\) with probability 1 and there is nothing to prove.) It remains to bound \(\mathbb E[Z]\).

*Step 2: a ghost sample.* The trouble with \(L_{\mathcal D}(h)\) is that it involves the whole distribution. Replace it with something that looks like \(L_S\). Imagine a second, independent training sequence \(S' = (z'_1, \dots, z'_m) \sim \mathcal D^m\), a **ghost sample** that nobody ever collects. Each \(\ell(h, z'_i)\) has mean \(L_{\mathcal D}(h)\), so by linearity of expectation \(L_{\mathcal D}(h) = \mathbb E_{S'}[L_{S'}(h)]\). Then

\[ |L_{\mathcal D}(h) - L_S(h)| = \left| \mathbb E_{S'}\left[ L_{S'}(h) - L_S(h) \right] \right| \le \mathbb E_{S'}\left| L_{S'}(h) - L_S(h) \right| \le \mathbb E_{S'}\left[ \sup_{h' \in \mathcal H} |L_{S'}(h') - L_S(h')| \right]. \]

The first inequality is “the absolute value of an average is at most the average of absolute values”. The second holds because the quantity inside the expectation can only grow when we replace it by its supremum over the class. The right side no longer depends on \(h\), so it also bounds the supremum over \(h\) of the left side. Taking \(\mathbb E_S\) of both sides,

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

*Step 3: random signs.* The \(2m\) examples \(z_1, \dots, z_m, z'_1, \dots, z'_m\) are i.i.d. So if we swap \(z_i\) with \(z'_i\) for some indices \(i\), the joint distribution of the \(2m\) examples does not change, and neither does the expectation of any function of them. Swapping at index \(i\) flips the sign of the \(i\)-th term \(\ell(h, z'_i) - \ell(h, z_i)\). So for every sign vector \(\sigma \in \{\pm 1\}^m\),

\[ \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right] = \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

The right side has the same value for all \(2^m\) sign vectors, so it also equals their average. Draw \(\sigma\) uniformly, each \(\sigma_i\) an independent fair \(\pm 1\) coin. Averaging over the finitely many \(\sigma\) and swapping the order of the two averages (linearity of expectation),

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\, \mathbb E_{\sigma}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

[[diagram:ghost-sample]]

*Step 4: the infinite class becomes finite.* Now fix \(S\) and \(S'\), and let \(C\) be the set of instances appearing in them, at most \(2m\) points. Each loss \(\ell(h, z_i) = \mathbb 1[h(x_i) \ne y_i]\) depends on \(h\) only through its labels on \(C\). So the expression inside the supremum depends on \(h\) only through its labelling in \(\mathcal H_C\), and the supremum over the infinite class is a maximum over at most \(|\mathcal H_C|\) distinct values. How large can \(|\mathcal H_C|\) be? If \(|C| < 2m\), add points of \(\mathcal X\) to reach a set \(C^+\) of exactly \(2m\) points; each labelling of \(C\) is the shortening of some labelling of \(C^+\), so \(|\mathcal H_C| \le |\mathcal H_{C^+}| \le \tau_{\mathcal H}(2m)\). This is where the growth function enters, and it is the heart of the matter: once \(S\) and \(S'\) are fixed, an infinite class acts like a finite one.

*Step 5: Hoeffding and the union bound.* For each labelling, pick one hypothesis \(h\) that produces it, and write

\[ \theta_h = \frac{1}{m} \sum_{i=1}^{m} \sigma_i c_i, \qquad c_i = \ell(h, z'_i) - \ell(h, z_i) \in \{-1, 0, 1\}. \]

With \(S, S'\) fixed, the \(c_i\) are fixed numbers and only the signs are random. The terms \(\sigma_i c_i\) are independent, each has mean 0, and each lies in \([-1, 1]\). **Hoeffding’s inequality** from lesson 4 says that an average of \(m\) independent variables with values in \([a, b]\) is more than \(\rho\) from its mean with probability at most \(2\exp(-2m\rho^2/(b - a)^2)\). (Lesson 4 stated it for identically distributed variables, but its proof multiplies one bound per variable and never uses that they share a distribution, so it holds for independent ones.) Here \(b - a = 2\), so

\[ \mathbb P_\sigma\left[ |\theta_h| > \rho \right] \le 2 \exp\left( -\frac{m \rho^2}{2} \right). \]

A union bound over the at most \(|\mathcal H_C|\) chosen hypotheses gives

\[ \mathbb P_\sigma\left[ \max_{h} |\theta_h| > \rho \right] \le 2 |\mathcal H_C| \exp\left( -\frac{m \rho^2}{2} \right) \quad \text{for every } \rho > 0. \]

*Step 6: from a tail bound to an expectation.* We need \(\mathbb E_\sigma[\max_h |\theta_h|]\). Here is a small lemma that turns a tail bound of this shape into one.

**Tail lemma.** Let \(X \ge 0\) be a random variable, and suppose \(\mathbb P[X > \rho] \le 2b\, e^{-\rho^2/a^2}\) for all \(\rho \ge 0\), with \(a > 0\) and \(b \ge e\). Then \(\mathbb E[X] \le a(1 + \sqrt{\log b})\).

*Proof.* For a non-negative variable, \(\mathbb E[X] = \int_0^\infty \mathbb P[X > \rho]\, d\rho\). Split at \(\rho_0 = a\sqrt{\log b}\). Below \(\rho_0\), bound the probability by 1, which contributes \(\rho_0\). Above \(\rho_0\), use the tail bound together with \(1 \le \rho/\rho_0\):

\[ \int_{\rho_0}^\infty 2b\, e^{-\rho^2/a^2}\, d\rho \le \frac{2b}{\rho_0} \int_{\rho_0}^\infty \rho\, e^{-\rho^2/a^2}\, d\rho = \frac{2b}{\rho_0} \cdot \frac{a^2}{2} e^{-\rho_0^2/a^2} = \frac{2b}{\rho_0} \cdot \frac{a^2}{2b} = \frac{a}{\sqrt{\log b}} \le a, \]

where the last step uses \(\log b \ge 1\). Adding the two parts gives \(a\sqrt{\log b} + a\). \(\blacksquare\)

Apply it with \(a^2 = 2/m\) and \(b = \max(|\mathcal H_C|, e)\); the tail bound of Step 5 still holds with this larger \(b\). Then \(\sqrt{\log b} \le 1 + \sqrt{\log |\mathcal H_C|}\), and \(a = \sqrt{2/m} = 2/\sqrt{2m}\), so

\[ \mathbb E_\sigma\left[ \max_h |\theta_h| \right] \le \frac{2}{\sqrt{2m}} \left( 2 + \sqrt{\log |\mathcal H_C|} \right) \le \frac{4 + 2\sqrt{\log \tau_{\mathcal H}(2m)}}{\sqrt{2m}}. \]

*Step 7: put it together.* The bound of Step 6 holds for every fixed \(S, S'\), so it survives the outer expectation in Step 3: \(\mathbb E_S[Z] \le (4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/\sqrt{2m}\). Step 1 divides by \(\delta\). \(\blacksquare\)

Steps 4 to 6 are easy to watch for thresholds. Draw \(2m = 400\) scores, label them by \(t^\star = 0.6\), split them into \(S\) and \(S'\), and estimate \(\mathbb E_\sigma[\max_h |\theta_h|]\) by drawing random signs:

```python
import math
import random

rng, m, t_star = random.Random(0), 200, 0.6
x = [rng.random() for _ in range(2 * m)]
loss_at = lambda t: [int((xi > t) != (xi > t_star)) for xi in x]
HC = {tuple(loss_at(t)) for t in sorted(x) + [-1.0]}

def max_theta(sigma):
    return max(abs(sum(s * (v[m + i] - v[i])
                       for i, s in enumerate(sigma))) / m
               for v in HC)

draws = [max_theta([rng.choice((-1, 1)) for _ in range(m)])
         for _ in range(300)]
bound = math.sqrt(2 / m) * (2 + math.sqrt(math.log(len(HC))))
mean = sum(draws) / len(draws)
print(len(HC), round(mean, 4), round(bound, 4))
```

It prints `401 0.0845 0.4448`. An infinite class, but on these 400 scores only \(401 = \tau_{\mathrm{thr}}(400)\) distinct loss vectors, exactly as Step 4 promised. The average largest \(|\theta_h|\) is about 0.085, comfortably under the bound of 0.445. The bound is valid, and loose.

## Closing the circle: finite VC dimension gives uniform convergence

Now combine the two halves. Let \(\mathrm{VCdim}(\mathcal H) = d\). If \(d = 0\), no single point is shattered, so all hypotheses agree on every point: \(\mathcal H\) is one function, a finite class, and lesson 4 already gives uniform convergence. So let \(d \ge 1\). For \(2m \ge d\), Sauer’s lemma gives \(\tau_{\mathcal H}(2m) \le (2em/d)^d\), and the theorem gives, with probability at least \(1 - \delta\), for all \(h\),

\[ |L_S(h) - L_{\mathcal D}(h)| \le \frac{4 + 2\sqrt{d \log(2em/d)}}{\delta \sqrt{2m}}. \]

Write \(u = d \log(2em/d)\). When \(u \ge 4\), we have \(4 \le 2\sqrt u\), so the numerator is at most \(4\sqrt u\) and the bound is at most \(\frac{1}{\delta}\sqrt{8u/m}\). This is at most \(\varepsilon\) when

\[ m \ge \frac{8 d \log(2em/d)}{(\delta\varepsilon)^2} = \alpha \log m + \alpha \log(2e/d), \qquad \alpha = \frac{8d}{(\delta\varepsilon)^2}. \]

The unknown \(m\) sits on both sides. A small inequality, which we’ll call the **log lemma**, unties it: if \(\alpha \ge 1\) and \(\beta \ge 0\), then \(m \ge 4\alpha \log(2\alpha) + 2\beta\) implies \(m \ge \alpha \log m + \beta\).

*Proof of the log lemma.* First, for any \(\gamma > 0\), \(x \ge 2\gamma \log \gamma\) implies \(x \ge \gamma \log x\). If \(\gamma \le e\), then \(x - \gamma \log x\) is smallest at \(x = \gamma\), where it equals \(\gamma(1 - \log \gamma) \ge 0\), so the conclusion holds for every \(x > 0\). If \(\gamma > e\), then \(x - \gamma \log x\) increases for \(x > \gamma\), and \(2\gamma\log\gamma > \gamma\); at \(x = 2\gamma \log \gamma\) it equals \(\gamma(\log \gamma - \log(2\log\gamma)) \ge 0\), because \(\gamma \ge 2 \log \gamma\) for every \(\gamma > 0\). Now apply this with \(\gamma = 2\alpha\): \(m \ge 4\alpha \log(2\alpha)\) gives \(m \ge 2\alpha \log m\), that is, \(m/2 \ge \alpha \log m\). Also \(m \ge 2\beta\), that is, \(m/2 \ge \beta\), because \(4\alpha\log(2\alpha) \ge 0\) when \(\alpha \ge 1\). Add the two halves. \(\blacksquare\)

With \(\beta = \alpha \log(2e/d)\) when \(d \le 5\), and \(\beta = 0\) when \(d \ge 6\) (then \(\log(2e/d) < 0\) and that term only helps), we get:

\[ m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le \frac{32 d}{(\delta\varepsilon)^2} \log\left( \frac{16 d}{(\delta\varepsilon)^2} \right) + \frac{16 d}{(\delta\varepsilon)^2} \max\left\{ \log\frac{2e}{d},\, 0 \right\}. \]

Two loose ends. Such an \(m\) is far larger than \(d\), so \(2m \ge d\) holds. And \(m/d \ge 32 \log 16 > 88\), so \(2em/d > 478\), \(\log(2em/d) > 6\), and \(u \ge 4\) holds too. So every class of finite VC dimension has the uniform convergence property, which is \(6 \Rightarrow 1\), and the circle of equivalences is closed. \(\blacksquare\)

How good is this bound in practice? For thresholds we know the growth function exactly, \(\tau(2m) = 2m + 1\), so let’s ask the theorem directly for the smallest \(m\) that brings its bound under \(\varepsilon = 0.1\) at \(\delta = 0.1\):

```python
from math import log, sqrt

def theorem_bound(m, delta, tau_2m):
    return (4 + 2 * sqrt(log(tau_2m))) / (delta * sqrt(2 * m))

eps = delta = 0.1
m = 1
while theorem_bound(m, delta, 2 * m + 1) > eps:
    m *= 2
lo, hi = m // 2, m
while hi - lo > 1:
    mid = (lo + hi) // 2
    ok = theorem_bound(mid, delta, 2 * mid + 1) <= eps
    lo, hi = (lo, mid) if ok else (mid, hi)
print(hi)
```

It prints `662296`. The direct argument for thresholds needed 30. The general proof pays twice: Markov’s inequality puts \(1/\delta\) where a \(\log(1/\delta)\) would do, and the symmetrization constants are generous. That is why the quantitative version, proved in a later lesson, matters for budgets. But the proof above answers the yes-or-no question completely, and that is what the reviewer asked: *is there any guarantee at all?* For any class of finite VC dimension, yes.

## ERM on the score line, as code

The fundamental theorem says ERM is the learner to run. On the score line it is fast, and it handles the agnostic case, where labels may be noisy and no hypothesis is consistent. Here is ERM for thresholds and for intervals. Each takes a training sequence \(S\) of \(m\) examples \((x_i, y_i)\).

```
ERM-Threshold(S):
  group S by score, sorted ascending: (x, junk count, keep count)
  t ← -∞;  errors ← number of keep examples;  best ← errors
  for each group (x, j, k) in order:
    errors ← errors + j - k
    if errors < best: best ← errors;  t ← x
  return h_t, best

ERM-Interval(S):
  group S by score, sorted ascending: (x, junk count, keep count)
  best_gain ← 0;  best ← empty interval
  run_gain ← 0;  run_start ← none
  for each group (x, j, k) in order:
    w ← j - k
    if run_start is none or run_gain ≤ 0:
      run_gain ← w;  run_start ← x
    else:
      run_gain ← run_gain + w
    if run_gain > best_gain:
      best_gain ← run_gain;  best ← [run_start, x]
  return best, (number of junk examples) - best_gain
```

**ERM-Threshold** starts with the threshold below every score, where every keep example is a mistake, and slides it up one distinct score at a time. Passing a group of scores turns its junk examples into mistakes and its keep examples into correct answers, so the mistake count changes by \(j - k\). Grouping equal scores matters: a threshold cannot separate two examples with the same score.

**ERM-Interval** uses a different trick. An interval \([a, b]\) makes a mistake on every junk example outside it and every keep example inside it. So its mistake count is (all junk) minus (junk inside minus keep inside). Give each group the weight \(w = j - k\); minimizing mistakes means finding the run of consecutive groups with the largest total weight, or the empty interval if every run is negative. That is the classic maximum-subarray scan: keep extending the current run while its total is positive, and restart when it is not.

Both cost \(O(m \log m)\) time for the sort and \(O(m)\) for the scan, with \(O(m)\) space. In the realizable case there is an even simpler ERM for intervals: return the **tightest interval**, from the smallest junk score to the largest, in \(O(m)\) time without sorting. Compare that with lesson 2’s brute-force ERM over a finite class, \(O(m |\mathcal H|)\): here \(|\mathcal H|\) is infinite, and the algorithm works anyway, for the same reason the theory does. Only the \(O(m)\) and \(O(m^2)\) labellings of the training scores matter, and the scan visits the right ones in order.

**Build it yourself.** Using only the Python standard library:

```python
import random

S = [(0.05, 0), (0.12, 0), (0.20, 0), (0.33, 0), (0.41, 1),
     (0.47, 0), (0.52, 1), (0.58, 1), (0.63, 1), (0.70, 1),
     (0.74, 0), (0.81, 1), (0.88, 1), (0.95, 0)]

def erm_threshold(S): ...
def erm_interval(S): ...

def errors(h, S):
    return sum(h(x) != y for x, y in S)

def brute_threshold(S):
    ts = [-1.0] + [x for x, _ in S]
    return min(errors(lambda x: x > t, S) for t in ts)

def brute_interval(S):
    xs = [x for x, _ in S]
    hs = [lambda x: False] + [lambda x, a=a, b=b: a <= x <= b
                              for a in xs for b in xs if a <= b]
    return min(errors(h, S) for h in hs)

if __name__ == "__main__":
    t, e = erm_threshold(S)
    assert e == 3 == errors(lambda x: x > t, S)
    (a, b), e = erm_interval(S)
    assert e == 2 == errors(lambda x: a <= x <= b, S)
    rng = random.Random(0)
    for _ in range(2000):
        R = [(round(rng.random(), 1), rng.randint(0, 1))
             for _ in range(rng.randint(1, 25))]
        assert erm_threshold(R)[1] == brute_threshold(R)
        assert erm_interval(R)[1] == brute_interval(R)
```

1. Implement `erm_threshold`, returning `(t, mistakes)`, and `erm_interval`, returning `((a, b), mistakes)`, or `(None, mistakes)` for the empty interval. Make both run in \(O(m \log m)\). The random training sequences round scores to one decimal on purpose, so equal scores are common: handle them.
2. Explain why `brute_threshold` and `brute_interval` only need to try the training scores as endpoints, using the argument from “What a class can do to a few messages”.
3. Add the realizable tightest-interval learner and measure, over 10,000 random training sequences with uniform scores and true interval \([0.55, 0.9]\), how often its true risk exceeds \(\varepsilon = 0.05\) at \(m = 148\). The next section predicts at most 5%.

## Fix the bug

You take the threshold guarantee to the reviewer: thresholds have VC dimension 1, so by the fundamental theorem they are learnable, and the direct argument says 74 labels buy \(\varepsilon = \delta = 0.05\). The reviewer signs. Then, the week before launch, the inbox changes. Marketing starts sending the company’s own newsletters, which are crammed with links and the word “FREE”, and the scoring service gives them the highest scores of all. Your users keep them. So junk is no longer “above a line”: it is an interval of scores, \([a^\star, b^\star]\), with the top scores marked keep.

Three classes are on the table, and the budget is 160 labels. The target: true risk at most \(\varepsilon = 5\%\), certified with confidence \(1 - \delta = 95\%\), and certified means proved, not observed in one lucky run.

- **Thresholds**, VC dimension 1, certified at 74 labels. But this week the truth is not a threshold.
- **Intervals**, VC dimension 2. Does a certificate fit the budget?
- **Unions of intervals**, which contain every truth you could meet. VC dimension infinite.

The VC dimension sorts these before any arithmetic. Unions of intervals have infinite VC dimension, so by the fundamental theorem no number of labels certifies them; the learner can wrap every junk example in its own band. Thresholds have finite VC dimension, but their certificate assumes the truth is a threshold, and it is not: that is the rat’s failure from lesson 1, a class that misses the truth. Intervals have VC dimension 2, so some finite budget works. How many labels, exactly?

**Claim (intervals, tightest-interval learner).** Assume realizability by an interval \([a^\star, b^\star]\) and no single score carrying probability. The tightest-interval learner, given \(m \ge 2\log(2/\delta)/\varepsilon\) examples, has \(L_{\mathcal D}(h_S) \le \varepsilon\) with probability at least \(1 - \delta\).

*Proof.* The learner returns \([x_{\min}, x_{\max}]\), the smallest and largest junk scores in \(S\), or the empty interval if there are none. Every junk score lies in \([a^\star, b^\star]\), so the learned interval sits inside the true one, and the learner errs only on true junk it leaves out: \(L_{\mathcal D}(h_S) = \mathcal D([a^\star, x_{\min})) + \mathcal D((x_{\max}, b^\star])\), or \(\mathcal D([a^\star, b^\star])\) if there are no junk examples. If \(\mathcal D([a^\star, b^\star]) \le \varepsilon\), every outcome has risk at most \(\varepsilon\). Otherwise, choose a strip \([a^\star, \ell]\) at the left end of the true interval and a strip \([r, b^\star]\) at its right end, each with probability exactly \(\varepsilon/2\); they do not overlap, since together they hold \(\varepsilon\) and the interval holds more. If some example lands in the left strip, then \(x_{\min} \le \ell\) and the left error is at most \(\varepsilon/2\); the same holds on the right. So failure needs an empty strip. Each strip is empty with probability \((1 - \varepsilon/2)^m \le e^{-\varepsilon m/2}\), and the union bound gives \(\mathbb P[L_{\mathcal D}(h_S) > \varepsilon] \le 2e^{-\varepsilon m/2} \le \delta\) once \(m \ge 2\log(2/\delta)/\varepsilon\). \(\blacksquare\)

```python
from math import ceil, log
import random

eps = delta = 0.05
m_thr = ceil(log(2 / delta) / eps)
m_itv = ceil(2 * log(2 / delta) / eps)
a, b = 0.55, 0.9

def tightest_risk(xs):
    junk = [x for x in xs if a <= x <= b]
    if not junk:
        return b - a
    return (min(junk) - a) + (b - max(junk))

rng = random.Random(0)
runs = [tightest_risk([rng.random() for _ in range(m_itv)])
        for _ in range(100_000)]
print(m_thr, m_itv, sum(r > eps for r in runs) / len(runs))
```

It prints `74 148 0.00433`: thresholds would need 74, intervals need 148, and at 148 labels the tightest-interval learner missed the target in 0.4% of 100,000 runs, well inside the certified 5%. A hundred and forty-eight fits a budget of 160.

Notice the division of labour. The VC dimension decides whether any budget can work: finite for thresholds and intervals, infinite for unions of intervals. For these two classes on the line, a direct argument then pins the number, because the general proof’s constants are far too generous for a budget.

In the lab below you face the same choice. Each round the hidden truth changes. You choose the class and the number of labels, and the lab tells you whether your choice is certified before it trains. Watch what happens when you pick a class that is too rich, or one too poor.

[[lab-act2]]

## What people get wrong

**“An infinite class can’t be learned.”** Thresholds are infinite and learnable from 74 labels at \(\varepsilon = \delta = 0.05\). What decides learnability is the VC dimension, a property of how the class labels points, not how many hypotheses it lists.

**“To show the VC dimension is at least \(d\), I need every set of \(d\) points to be shattered.”** One shattered set of size \(d\) is enough. Thresholds on \([0, 1]\) with \(t \in [0, 1]\) cannot shatter \(\{0\}\), and their VC dimension is still 1. The asymmetry runs the other way for the upper bound: to show it is less than \(d + 1\), *every* set of \(d + 1\) points must fail, and one failing set proves nothing.

**“The VC dimension is the number of parameters.”** Often, not always. One parameter, \(\theta\), gives the class \(\lceil 0.5 \sin(\theta x) \rceil\) an infinite VC dimension.

**“The proven bound tells me how many labels to buy.”** The proof in this lesson certifies that some finite number exists, and its number, 662,296 for thresholds at \(\varepsilon = \delta = 0.1\), is a wild overestimate of the 30 a direct argument gives. Use a direct argument where you have one, and the quantitative version of the fundamental theorem for the shape of the dependence.

## Recap and the next bug

1. The finite-class guarantees pay \(\log |\mathcal H|\), which is useless for an infinite class. Yet thresholds on the score line are PAC learnable with \(m \le \lceil \log(2/\delta)/\varepsilon \rceil\), by a two-strip argument that never counts hypotheses.
2. The **restriction** \(\mathcal H_C\) is the set of labellings a class produces on \(C\). A class **shatters** \(C\) when it produces all \(2^{|C|}\) of them, and its **VC dimension** is the size of the largest set it shatters. If a class shatters \(2m\) points, the No-Free-Lunch argument defeats every learner with \(m\) examples, so infinite VC dimension means not PAC learnable.
3. Thresholds have VC dimension 1, intervals 2, axis-aligned rectangles 4, a finite class at most \(\log_2 |\mathcal H|\), unions of intervals infinity.
4. **The fundamental theorem of statistical learning**: for binary classification with the 0-1 loss, uniform convergence, agnostic PAC learnability by ERM, agnostic PAC learnability, PAC learnability, PAC learnability by ERM and a finite VC dimension are all equivalent, and the sample complexity scales with \(d\) where \(\log |\mathcal H|\) used to stand.
5. The hard direction rests on two pieces: **Sauer’s lemma**, \(\tau_{\mathcal H}(m) \le \sum_{i \le d} \binom{m}{i} \le (em/d)^d\), and uniform convergence from a small growth function, through a ghost sample, random signs and Hoeffding’s inequality over the finitely many labellings.

Your filter ships with a certificate. Then a teammate comes to you with a better idea. Scores are blunt, she says; what you want are **decision lists**: rules like “if the score is above 0.9 and the sender is unknown, junk; else if the message has no link, keep; else if ...; else keep”. Short lists for simple weeks, long lists when the spammers get clever, and the list can grow as long as it needs to. You check: lists of length at most \(k\) form a class of finite VC dimension, for each \(k\). But she wants no cap on \(k\), and the class of all lists, of every length, shatters sets of every size. Its VC dimension is infinite, and today’s theorem says no learner can PAC learn it. Yet a learner that prefers short lists and accepts a long one only when the data demand it seems perfectly sensible. Can a class like that be learned at all, in some weaker but still honest sense? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Consider the class with just two hypotheses on the score line: always keep and always junk. What is its VC dimension? Now consider thresholds \(h_t(x) = \mathbb 1[x > t]\) on \(\mathcal X = [0, 1]\), but with \(t\) allowed only in \([0, 1]\). Which single points does this class shatter, and what is its VC dimension?
2. In the proof of Sauer’s lemma, explain in your own words why \(|\mathcal H_C| = |Y_0| + |Y_1|\), and why every set \(B \subseteq C'\) shattered by the partner class \(\mathcal H'\) gives a shattered set \(B \cup \{c_1\}\). What would go wrong in Step 4 if we used \(\mathcal H\) in place of \(\mathcal H'\)?
3. Let \(\mathcal H_2\) be “junk if the score lies in one of at most two intervals”. Find \(\mathrm{VCdim}(\mathcal H_2)\), with both halves of the proof. Then compute \(\tau_{\mathcal H_2}(5)\) and \(\tau_{\mathcal H_2}(10)\) and compare them with Sauer’s bound \(\sum_{i \le d} \binom{m}{i}\) and with \((em/d)^d\).
