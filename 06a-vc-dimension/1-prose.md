# The VC Dimension

## A threshold with no guarantee

The data team has shipped something you have wanted for a while. Their new scoring service reads a message, every word of it, and returns one number between 0 and 1: the message’s **spam score**. A high score means the message looks like junk. Your learner no longer has to choose among features. It has a single decision to make, and it is about where to draw a line. The new filter says:

> junk if the spam score is above \(t\).

Your users label messages as before, and the learner picks \(t\). In testing it behaves beautifully. With a few dozen labels, the line lands within a whisker of where your users would have drawn it themselves.

Then comes the launch review. Since lesson 2, every launch has needed a guarantee: how many labels, and how sure can you be? Lesson 2 answered it for a finite list of rules. Lesson 1’s four one-feature rules, for instance, needed 44 labels to promise a true risk (the chance of mislabelling the next message) of at most 10%, on all but 5% of unlucky training sequences. So you open that finite-class guarantee,

\[ m \ge \frac{\log(|\mathcal H|/\delta)}{\varepsilon}, \]

and you stop. First, a reminder of what it says, read from the inside out.

- \(\mathcal H\) is the hypothesis class, the list of rules the learner may choose from, and the bars \(|\mathcal H|\) mean “how many rules it has”.
- \(\delta\) (delta) is the failure chance you tolerate, 0.05 above, so \(|\mathcal H|/\delta\) grows when the class gets bigger or when you ask to be surer.
- \(\log\) is the natural logarithm, which turns a huge number into a modest one: \(\log(4/0.05) = \log 80 \approx 4.38\).
- \(\varepsilon\) (epsilon) is the true risk you tolerate, 0.1 above, and dividing by it says that halving the tolerated risk doubles the labels.
- \(m\) is the number of labelled examples, and \(\ge\) says any number at least this big will do.

In plain words: **if a class has \(|\mathcal H|\) rules and the truth is one of them, about \(\log(|\mathcal H|/\delta)/\varepsilon\) labels make the rule with the fewest training mistakes have true risk at most \(\varepsilon\), with probability at least \(1 - \delta\).** With four rules, \(4.38/0.1 = 43.8\), so 44 labels.

Now the question it asks: how many hypotheses are in your class? There is one threshold rule for every real number \(t\), so \(|\mathcal H|\) is infinite, and so is \(\log |\mathcal H|\). The agnostic version from lesson 4, for labels that no rule fits perfectly, needs \(\log(2|\mathcal H|/\delta)/(2\varepsilon^2)\) examples: the same pieces, with an extra factor 2 inside the log and the tolerance squared below, so halving \(\varepsilon\) now quadruples the labels. With four rules it asks for 254. With infinitely many rules it is infinite again. Here are both, as the class grows:

```python
from math import ceil, log

eps, delta = 0.1, 0.05
for size in (4, 2 ** 64, 2 ** 128):
    realizable = ceil(log(size / delta) / eps)
    agnostic = ceil(log(2 * size / delta) / (2 * eps ** 2))
    print(realizable, agnostic)
```

It prints `44 254`, then `474 2403`, then `918 4621`. Each line is one formula: `size` is \(|\mathcal H|\), and `ceil` rounds up to a whole number of labels. Push the size to infinity and both columns follow it. Every guarantee you own says “you need infinitely many labels”, which is a polite way of saying nothing.

You try the trick from lesson 4. A computer stores \(t\) as a 64-bit number, so in practice the class has at most \(2^{64}\) members, and the finite-class guarantee applies after all:

```python
from math import ceil, log

eps, delta = 0.05, 0.05
by_bits = ceil((64 * log(2) + log(1 / delta)) / eps)
by_threshold_argument = ceil(log(2 / delta) / eps)
print(by_bits, by_threshold_argument)
```

It prints `948 74`. The line for `by_bits` is the finite-class formula with \(|\mathcal H| = 2^{64}\), written using \(\log 2^{64} = 64 \log 2\) and \(\log(a/\delta) = \log a + \log(1/\delta)\). Ignore the second number for now; it is where this lesson is heading. The first says 948 labels buy a true risk of at most 5% with probability at least 95%.

The reviewer is not impressed, for two reasons. The first: if the machine moved to 128-bit numbers, your guarantee would double, while the rule, the users and the messages stayed exactly the same. A number that changes with the hardware is measuring the hardware, not the problem. The second reason is sharper. Lesson 5 proved that the class of all functions on an infinite domain cannot be learned at all, by any learner, from any number of examples. That class is infinite too. So infinitely many hypotheses *can* make learning impossible, and you have no tool that tells your infinite class apart from that one. Launch blocked, until you can say why your class is different.

Yet you have watched thresholds learn. So something about a class matters other than the number of hypotheses in it. What is it? This lesson finds it, gives it a name, and proves that it decides, completely, which classes can be learned.

## Thresholds learn anyway

First let’s show that the reviewer is wrong about thresholds in particular, by proving a guarantee directly.

Here is the setting, in the language of lessons 2 and 3. A message is now represented by its spam score, so the domain set, the set of all possible inputs, is \(\mathcal X = [0, 1]\): every number from 0 to 1, both ends included. Pick a cut-off, say 0.6. A message scoring 0.75 is above it, so it is junk; one scoring 0.3 is below, so it is kept; one scoring exactly 0.6 is not above, so it is kept too. Written as a formula, for each real number \(t\), the **threshold** hypothesis is

\[ h_t(x) = \mathbb 1[x > t]. \]

- \(x\) is a message’s score and \(t\) is the cut-off; the subscript in \(h_t\) says which cut-off this rule uses.
- \(x > t\) is a yes/no question: is the score above the cut-off?
- \(\mathbb 1[\cdot]\) is the **indicator**: it turns “yes” into 1 and “no” into 0.

In plain words: **\(h_t\) says junk (1) when the score is above \(t\) and keep (0) otherwise.** With \(t = 0.6\):

```python
t = 0.6
h_t = lambda x: int(x > t)
print([h_t(x) for x in (0.3, 0.6, 0.75)])
```

It prints `[0, 0, 1]`, the three messages above; `int(...)` plays the indicator.

The class of thresholds is \(\mathcal H_{\mathrm{thr}} = \{ h_t : t \in \mathbb R \}\). Read the braces and colon as “the set of all \(h_t\) such that”, and \(t \in \mathbb R\) as “\(t\) is any real number”: one rule per cut-off, infinitely many rules. We assume realizability: the true labelling is \(f = h_{t^\star}\) for some unknown cut-off \(t^\star\) (read “t star”), the one your users carry in their heads.

A reminder of the notation. The true risk \(L_{\mathcal D}(h)\) is measured against \(\mathcal D\), the distribution of arriving mail, and the training sequence \(S\) is the list of \(m\) labelled examples, drawn i.i.d. (independently, each from the same \(\mathcal D\)). ERM, empirical risk minimization, returns a rule with the fewest mistakes on \(S\); we write \(h_S\) for the rule it returns. Recall from lesson 3 that a class is **PAC learnable** when some learner, given at least \(m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples from any distribution, labelled by any \(f\) the class realizes, returns a hypothesis with true risk at most \(\varepsilon\) with probability at least \(1 - \delta\). Here \(m_{\mathcal H}(\varepsilon, \delta)\), the **sample complexity**, is a number of labels that depends only on the two tolerances, never on \(\mathcal D\) or \(f\). For lesson 1’s four rules at \(\varepsilon = 0.1\), \(\delta = 0.05\), it was 44.

We need one mild assumption about the scores: no single score value carries positive probability. A score computed from a whole message is a continuous quantity, so this is a fair assumption, and it lets us cut the score line into pieces of exactly the probability we want.

What should we expect? With tolerances \(\varepsilon = \delta = 0.1\), we will see that 30 labels are enough for thresholds, whatever the mail looks like, and with \(\varepsilon = \delta = 0.05\), 74. Here is the general statement.

**Claim (thresholds are PAC learnable).** Let \(\varepsilon, \delta \in (0, 1)\), that is, both strictly between 0 and 1. Every ERM learner for \(\mathcal H_{\mathrm{thr}}\), given

\[ m \ge \frac{\log(2/\delta)}{\varepsilon} \]

examples, returns a hypothesis with \(L_{\mathcal D}(h_S) \le \varepsilon\) with probability at least \(1 - \delta\). So \(m_{\mathcal H_{\mathrm{thr}}}(\varepsilon, \delta) \le \lceil \log(2/\delta)/\varepsilon \rceil\).

Compare it with the finite-class formula: \(|\mathcal H|\) has been replaced by 2, as if the infinite class of thresholds were a class of two rules. The ceiling brackets \(\lceil \cdot \rceil\) round up to the next whole number. In plain words: **about \(\log(2/\delta)/\varepsilon\) labels give any ERM threshold a true risk of at most \(\varepsilon\), with probability at least \(1 - \delta\).** At \(\varepsilon = \delta = 0.1\): \(\log 20 \approx 3.00\), divided by 0.1 is 29.96, rounded up to 30. The line computing `m` in the simulation after the proof is this formula, and so is `by_threshold_argument` in the 64-bit snippet, which gave 74 at 0.05 against 948 from counting floating-point numbers.

Before the proof, a picture of where the mistakes can live. Draw the score line with \(t^\star\) marked. Just to the left of \(t^\star\) lies a strip of scores that carries probability exactly \(\varepsilon\); just to the right lies another. If the training sequence puts at least one example in each strip, the learner’s threshold is trapped between them, and every mistake it can make lies inside one strip.

[[diagram:strips]]

Two pieces of notation for the proof. A strip of scores is written with brackets: \((a, b]\) means every score above \(a\) and up to \(b\), the round bracket saying the end is left out and the square one that it is kept in, so with \(t^\star = 0.6\) the strip \((0.5, 0.6]\) holds 0.55 and 0.6 but not 0.5. And \(\mathbb P_{x \sim \mathcal D}[x \in (a, b]]\), read “the probability that a score \(x\) drawn from \(\mathcal D\) lands in the strip”, is the share of arriving mail whose score falls there; for uniform scores it is just the strip’s width, 0.1 here.

*Proof.* Choose \(a_0 < t^\star\) so that \(\mathbb P_{x \sim \mathcal D}[x \in (a_0, t^\star]] = \varepsilon\), and \(a_1 > t^\star\) so that \(\mathbb P[x \in (t^\star, a_1]] = \varepsilon\). Such points exist because no single score carries probability, so the probability of \((a, t^\star]\) moves continuously as \(a\) slides. If all the scores below \(t^\star\) together carry probability at most \(\varepsilon\), set \(a_0 = -\infty\) instead, and similarly \(a_1 = +\infty\).

Now look at what ERM returns. Let \(b_0\) be the largest score among the keep examples in \(S\), and \(b_1\) the smallest score among the junk examples (\(b_0 = -\infty\) if there is no keep example, \(b_1 = +\infty\) if there is no junk example). Under realizability, \(h_{t^\star}\) has empirical risk 0 (no mistakes on \(S\)), so ERM returns some **consistent** \(h_t\), one that also makes no mistakes on \(S\), and \(h_t\) is consistent exactly when every keep score is at most \(t\) and every junk score is above \(t\): that is, \(b_0 \le t < b_1\).

Where can \(h_t\) be wrong? It disagrees with the truth exactly on the scores between \(t\) and \(t^\star\): on \((t, t^\star]\) if \(t < t^\star\), and on \((t^\star, t]\) if \(t \ge t^\star\). Suppose \(S\) has an example in each strip. An example in \((a_0, t^\star]\) is a keep example, so \(b_0 > a_0\), so \(t > a_0\), and when \(t < t^\star\) the mistakes lie in \((a_0, t^\star]\), probability at most \(\varepsilon\). An example in \((t^\star, a_1]\) is a junk example, so \(b_1 \le a_1\), so \(t < a_1\), and when \(t \ge t^\star\) the mistakes lie in \((t^\star, a_1]\), probability at most \(\varepsilon\). (If a strip was set to infinity, the same conclusion holds for free, since all the scores on that side carry at most \(\varepsilon\).) So \(L_{\mathcal D}(h_S) > \varepsilon\) can only happen if some strip is empty.

What is the chance that the left strip receives no example? Each of the \(m\) scores misses it with probability \(1 - \varepsilon\), and they are drawn independently, so all of them miss it with probability \((1 - \varepsilon)^m \le e^{-\varepsilon m}\), using \(1 - \varepsilon \le e^{-\varepsilon}\) from lesson 2, with \(e \approx 2.718\) Euler’s number. The power multiplies \(m\) copies of the miss chance, one per independent example; with \(\varepsilon = 0.1\) and \(m = 30\) it is \(0.9^{30} \approx 0.042\), under \(e^{-3} \approx 0.050\). The same holds for the right strip. The **union bound** from lesson 2 says that the chance that at least one of several bad events happens is at most the sum of their chances. Here the bad events are “left strip empty” and “right strip empty”, so

\[ \mathbb P_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(h_S) > \varepsilon \right] \le 2 e^{-\varepsilon m}, \]

and \(2e^{-\varepsilon m} \le \delta\) exactly when \(m \ge \log(2/\delta)/\varepsilon\), by taking logs of both sides. \(\blacksquare\)

Read the last display from the inside out. \(L_{\mathcal D}(h_S) > \varepsilon\) is the bad event: the rule ERM returned is worse than tolerated. The subscript \(S \sim \mathcal D^m\) on \(\mathbb P\) says where the randomness lives: the whole training sequence, \(m\) examples drawn i.i.d. from \(\mathcal D\), which is what \(\mathcal D^m\) means. On the right, the 2 counts the strips and \(e^{-\varepsilon m}\) is the bound for one strip. In plain words: **the chance of training on an unlucky sequence is at most twice the chance that one given strip stays empty.** With \(\varepsilon = 0.1\) and \(m = 30\):

```python
from math import exp

eps, m = 0.1, 30
one_strip = (1 - eps) ** m
print(round(one_strip, 4), round(exp(-eps * m), 4),
      round(2 * exp(-eps * m), 4))
```

It prints `0.0424 0.0498 0.0996`: one strip stays empty with chance about 4.2%, the exponential bound rounds that up to 5%, and the union bound doubles it to just under \(\delta = 10\%\).

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

It prints `30 0.08507 0.0996`. Thirty labels; even the most unlucky ERM learner fails about 8.5% of the time (the exact figure is \(2(0.9)^{30} - (0.8)^{30} \approx 8.35\%\)); and the bound \(2e^{-3} \approx 9.96\%\) sits just above, under \(\delta = 10\%\).

Notice what the proof never did. It never counted hypotheses. It used one fact about thresholds: two well-placed examples, one on each side of \(t^\star\), pin the threshold down. That is a fact about how thresholds can *label* points, and it is the thread we pull next.

## What a class can do to a few messages

Take a small set of scores, say \(C = \{0.3, 0.7\}\), and ask: in how many different ways can thresholds label these two messages? Each hypothesis gives a pair of labels, one per score. Different thresholds may give the same pair, and we only care about the distinct pairs. A cut-off of 0.5 gives (keep, junk), or \((0, 1)\); a cut-off of 0.1 gives \((1, 1)\); a cut-off of 0.9 gives \((0, 0)\); and 0.55 or 0.4 give \((0, 1)\) again, nothing new. The collection of distinct pairs is what we want to name.

**Definition (restriction).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\{0, 1\}\), and let \(C = \{c_1, \dots, c_m\} \subset \mathcal X\), a set of \(m\) points of the domain (\(\subset\) reads “is a subset of”). The **restriction** of \(\mathcal H\) to \(C\) is the set of label vectors that hypotheses in \(\mathcal H\) produce on \(C\):

\[ \mathcal H_C = \{ (h(c_1), \dots, h(c_m)) : h \in \mathcal H \} \subseteq \{0, 1\}^m. \]

From the inside out:

- \((h(c_1), \dots, h(c_m))\) is one hypothesis’s answers on the \(m\) points, written in order as a vector of 0s and 1s.
- The braces with the colon, \(\{ \dots : h \in \mathcal H \}\), collect that vector for every hypothesis \(h\) in the class. A set keeps each distinct vector once, so repeats vanish.
- \(\{0, 1\}^m\) is the set of all vectors of \(m\) zeros and ones, and \(\subseteq\) says the restriction sits inside it.

In plain words: **\(\mathcal H_C\) is the list of different ways the class can label the points of \(C\).** For \(C = \{0.3, 0.7\}\) and thresholds it is \(\{(0, 0), (0, 1), (1, 1)\}\), three vectors out of the four in \(\{0, 1\}^2\).

Each vector in \(\mathcal H_C\) is a **labelling** of \(C\). There are \(2^m\) possible labellings of \(m\) points, so \(|\mathcal H_C| \le 2^m\), however many hypotheses \(\mathcal H\) has. Here it is in Python for thresholds; the set comprehension in `restriction` is the braces of the formula, one tuple per hypothesis, duplicates dropped:

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

Read the equation as a count. \(|C|\) is how many points \(C\) has, \(2^{|C|}\) is how many labellings those points have in all (two choices per point), and \(|\mathcal H_C|\) is how many of them the class actually produces. In plain words: **the class shatters \(C\) when it can label the points of \(C\) in every possible way.** The snippet above already did the count: on \(\{0.3\}\) thresholds produce 2 of \(2^1 = 2\) labellings, so they shatter it; on \(\{0.3, 0.7\}\) they produce 3 of \(2^2 = 4\), so they do not.

Thresholds shatter every single point \(\{c\}\): take \(t = c - 1\) for junk and \(t = c + 1\) for keep. They shatter no set of two points \(c_1 < c_2\), because \((1, 0)\) is always missing.

[[diagram:restriction]]

Why should shattering matter? Think back to the No-Free-Lunch theorem of lesson 5. Its adversary took a set \(C\) of \(2m\) messages and looked at all \(2^{2m}\) ways to label them. For each labelling it built a distribution, uniform on \(C\) and labelled that way. A learner that has seen at most \(m\) of the \(2m\) messages knows nothing about the labels of the others, and the adversary chose the labelling that made the learner pay. The conclusion: some distribution makes the learner’s true risk at least \(1/8\) with probability at least \(1/7\).

In the PAC model, though, the adversary is restricted: it may only use labellings that some \(h \in \mathcal H\) achieves with risk 0. On \(C\), that means labellings in \(\mathcal H_C\). Now suppose \(\mathcal H\) shatters \(C\). Then every labelling of \(C\) is in \(\mathcal H_C\), the restriction costs the adversary nothing, and the whole argument goes through inside the PAC model. In the statement, \(\mathcal X \times \{0, 1\}\) is the set of pairs (score, label), so \(\mathcal D\) chooses both the message and its label, and \(A(S)\) is the rule learner \(A\) returns after training on \(S\).

**Corollary (shattering defeats every learner).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\{0, 1\}\), and \(m\) a training sequence size. If some set \(C \subset \mathcal X\) of size \(2m\) is shattered by \(\mathcal H\), then for every learner \(A\) there exist a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\) and a hypothesis \(h \in \mathcal H\) with \(L_{\mathcal D}(h) = 0\), such that with probability at least \(1/7\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \ge 1/8\).

In plain words: **if the class can label some \(2m\) messages in every way, then for any learner there is a world the class fits perfectly in which, at least one time in seven, \(m\) labels leave the learner wrong on at least one message in eight.** The class of all rules on lesson 1’s sixteen tiny-world messages labels those sixteen in every way, so with \(m = 8\) labels every learner can be pushed into that trap.

*Proof idea.* Run the No-Free-Lunch construction on the shattered set \(C\). Each of its labellings \(f_i: C \to \{0,1\}\) equals the restriction of some \(h_i \in \mathcal H\), because \(\mathcal H\) shatters \(C\), and \(h_i\) has true risk 0 under the distribution built from \(f_i\). So every distribution the adversary uses is realizable by \(\mathcal H\), and the theorem’s conclusion applies to it unchanged. \(\blacksquare\)

There is an old saying behind this: an explanation that can explain anything explains nothing. If the class can label half the set any way at all and the other half any way at all, then the labels you have seen say nothing about the labels you have not.

## The VC dimension

So the question to ask of a class is not “how many hypotheses?” but “how large a set can it shatter?”

**Definition (VC dimension).** The **VC dimension** of a class \(\mathcal H\), written \(\mathrm{VCdim}(\mathcal H)\), is the largest size of a set \(C \subset \mathcal X\) that \(\mathcal H\) shatters. If \(\mathcal H\) shatters sets of every finite size, its VC dimension is infinite, written \(\infty\).

In plain words: **the VC dimension is the most points the class can label in every possible way.** Thresholds shatter one point and no pair, so their VC dimension is 1.

The name honours Vladimir Vapnik and Alexey Chervonenkis, who discovered this characterization in 1970. An immediate consequence of the corollary:

**Theorem (infinite VC dimension means not PAC learnable).** If \(\mathrm{VCdim}(\mathcal H) = \infty\), then \(\mathcal H\) is not PAC learnable.

*Proof.* Suppose some learner \(A\) PAC learns \(\mathcal H\), and pick \(\varepsilon < 1/8\) and \(\delta < 1/7\). Let \(m = m_{\mathcal H}(\varepsilon, \delta)\). Since the VC dimension is infinite, \(\mathcal H\) shatters some set of size \(2m\). By the corollary there is a realizable distribution on which, with probability at least \(1/7 > \delta\), \(A\) returns a hypothesis with true risk at least \(1/8 > \varepsilon\). That contradicts the PAC guarantee. \(\blacksquare\)

The converse also holds: a finite VC dimension guarantees learnability. This lesson states it and puts it to work, and the next lesson proves it. But first let’s measure some classes. To prove \(\mathrm{VCdim}(\mathcal H) = d\), you need two things, and they are not symmetric:

1. **Some** set of size \(d\) is shattered by \(\mathcal H\). One example is enough.
2. **Every** set of size \(d + 1\) fails to be shattered. One example is not enough; you need an argument that works for any set.

For thresholds, step 1 is any single point, and step 2 is the labelling \((1, 0)\), missing for every pair \(c_1 < c_2\). An infinite class with VC dimension 1: that is what you will tell the reviewer.

## Four classes, measured

**Intervals.** Suppose junk is not “above a line” but “inside a band of scores”: say junk between 0.4 and 0.8, so 0.5 is junk while 0.3 and 0.9 are kept. The class of **intervals** is \(\mathcal H_{\mathrm{int}} = \{ h_{a,b} : a \le b \}\), one rule for every pair of ends \(a\) (the bottom of the band) and \(b\) (the top), with

\[ h_{a,b}(x) = \mathbb 1[a \le x \le b]. \]

The chain \(a \le x \le b\) is one yes/no question, “is the score at least \(a\) and at most \(b\)?”, and the indicator turns the answer into 1 or 0. In plain words: **\(h_{a,b}\) says junk exactly when the score lies inside the band from \(a\) to \(b\).** With \(a = 0.4\), \(b = 0.8\), the scores 0.3, 0.5 and 0.9 get 0, 1 and 0. The `interval` function in the snippet below is this formula, the chained comparison included. (Whether the endpoints belong to the interval changes nothing below.) Two points \(\{1, 2\}\) are shattered: an interval can cover neither, either one, or both. Three points \(c_1 < c_2 < c_3\) are never shattered, because the labelling \((1, 0, 1)\) would need an interval containing \(c_1\) and \(c_3\) but not \(c_2\), which sits between them. So \(\mathrm{VCdim}(\mathcal H_{\mathrm{int}}) = 2\). Python agrees, trying intervals whose ends sit at the points themselves (and one past the end for the empty labelling), which again covers every labelling:

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

**Axis-aligned rectangles.** Now give each message two scores: its spam score \(x_1\) and a second score \(x_2\) for its sender’s reputation. A rectangle rule says junk when both scores fall in a range, for instance a spam score of at least 0.5 together with a reputation of at most 0.3:

\[ h_{(a_1, a_2, b_1, b_2)}(x_1, x_2) = \mathbb 1[a_1 \le x_1 \le a_2 \text{ and } b_1 \le x_2 \le b_2], \]

for \(a_1 \le a_2\) and \(b_1 \le b_2\). Call this class \(\mathcal H_{\mathrm{rec}}\).

It is the interval rule twice, one band per score, joined by “and”; the four numbers in the subscript are the two bands’ ends. Drawn with \(x_1\) across and \(x_2\) up, the junk region is a rectangle with sides parallel to the axes, hence the name. In plain words: **junk exactly when the spam score is in one range and the reputation score is in another.** The rule from the opening sentence is \(a_1 = 0.5\), \(a_2 = 1\), \(b_1 = 0\), \(b_2 = 0.3\):

```python
def rect(a1, a2, b1, b2):
    return lambda x1, x2: int(a1 <= x1 <= a2 and b1 <= x2 <= b2)

h = rect(0.5, 1.0, 0.0, 0.3)
print(h(0.8, 0.2), h(0.8, 0.6), h(0.3, 0.2))
```

It prints `1 0 0`: a spammy message from a bad sender is junk, the same message from a well-known sender is kept, and a clean message from a bad sender is kept. Its VC dimension is 4.

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

**Finite classes.** If \(\mathcal H\) is finite, then \(|\mathcal H_C| \le |\mathcal H|\) for every \(C\), since each labelling comes from at least one hypothesis. Shattering \(C\) needs \(2^{|C|}\) labellings, so it is impossible once \(2^{|C|} > |\mathcal H|\). Hence \(\mathrm{VCdim}(\mathcal H) \le \log_2 |\mathcal H|\). Here \(\log_2\) is the logarithm base 2, the power you must raise 2 to: \(\log_2 16 = 4\), because \(2^4 = 16\). In plain words: **a class of \(|\mathcal H|\) rules can shatter at most as many points as there are doublings in \(|\mathcal H|\).** Lesson 1’s four one-feature rules give \(\log_2 4 = 2\): four rules can make at most four labellings, exactly enough for two points and short of the eight that three points need. The last column of the snippet below computes \(\log_2 |\mathcal H|\) with `log2`. This bound can be very loose. Take the scores \(\{1, 2, \dots, 16\}\) and thresholds on them:

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

**VC dimension and the number of parameters.** In these examples the VC dimension equals the number of numbers that define a hypothesis: one for a threshold, two for an interval, four for a rectangle. That is often the case, and a handy first guess. It is not a law. Here is a rule with a single knob \(\theta\) (theta): multiply the input, any real number, by \(\theta\), take the sine, and call it junk when the sine is positive. With \(\theta = 1\), the inputs 1 and 2 have positive sines (0.84, 0.91) and are junk, while 4 and 5 have negative sines and are kept. As a class, that is \(\{ x \mapsto \lceil 0.5 \sin(\theta x) \rceil : \theta \in \mathbb R \}\). The arrow \(x \mapsto \dots\) reads “the rule that sends \(x\) to”; \(\sin(\theta x)\) lies between \(-1\) and 1, so half of it lies between \(-0.5\) and 0.5; and the ceiling \(\lceil \cdot \rceil\), rounding up to a whole number, turns anything in \((0, 0.5]\) into 1 and anything in \([-0.5, 0]\) into 0. In plain words: **junk exactly when \(\sin(\theta x) > 0\).**

```python
from math import ceil, sin

h = lambda theta, x: ceil(0.5 * sin(theta * x))
print([round(sin(x), 2) for x in (1, 2, 4, 5)])
print([h(1.0, x) for x in (1, 2, 4, 5)])
```

It prints `[0.84, 0.91, -0.76, -0.96]` and `[1, 1, 0, 0]`. This class on the real line has one parameter and infinite VC dimension: for every \(d\) there are \(d\) points it shatters, because turning the single knob \(\theta\) makes the sine wave swing faster or slower, enough to place the junk and keep regions wherever you need. (Proving it is a good exercise in binary expansions.)

Now it’s your turn to hunt for labellings no class can make.

[[lab]]

## The fundamental theorem

We have shown one direction: infinite VC dimension rules out PAC learning. The converse is also true, and together they form the central result of this part of the course. It ties together every idea from lessons 2 to 5: learnability, ERM, uniform convergence and now the VC dimension.

First, uniform convergence from lesson 4, idea before symbols. In lesson 1, the rule “unknown sender means junk” got 25% of its labelled examples wrong but 50% of all mail: a gap of 0.25 between how good it looked and how good it was. A training sequence is **\(\varepsilon\)-representative** when no rule in the class has a gap above \(\varepsilon\): every \(h \in \mathcal H\) has \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\). Here \(L_S(h)\) is the empirical risk, the share of the training sequence \(h\) gets wrong, \(L_{\mathcal D}(h)\) is the true risk, and the bars mean absolute value, the size of the gap whichever way it points. A class has the **uniform convergence property** when there is a sample size \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) such that, for every distribution \(\mathcal D\), at least that many i.i.d. examples are \(\varepsilon\)-representative with probability at least \(1 - \delta\). The superscript UC only marks it as the uniform-convergence sample size, as opposed to the learning one. In plain words: **a class has uniform convergence when enough labels make every rule’s training score close to its true score, for every rule at once and whatever the mail looks like.** Thresholds on 200 uniform scores with \(t^\star = 0.6\), checked over cut-offs \(0, 0.01, \dots, 1\), where the true risk of \(h_t\) is just the distance \(|t - 0.6|\):

```python
import random

rng, t_star, m, eps = random.Random(0), 0.6, 200, 0.1
xs = [rng.random() for _ in range(m)]

def L_S(t):
    return sum((x > t) != (x > t_star) for x in xs) / m

ts = [i / 100 for i in range(101)]
gap = max(abs(L_S(t) - abs(t - t_star)) for t in ts)
print(round(gap, 4), gap <= eps)
```

It prints `0.035 True`: the worst gap over 101 thresholds is 3.5 points, so this training sequence is 0.1-representative. The `max` over `ts` is the “every \(h\)” of the definition.

Two more reminders from lesson 3. The **0-1 loss** charges 1 for a wrong label and 0 for a right one, so average loss is the share of mistakes. A class is **agnostic PAC learnable** when, with no assumption that any rule is perfect, some learner gets within \(\varepsilon\) of the best rule in the class, with probability at least \(1 - \delta\).

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

All three read the same way, so take the first from the inside out.

- \(d\) is the VC dimension and \(\log(1/\delta)\) is the price of confidence: \(d + \log(1/\delta)\) adds the two.
- Dividing by \(\varepsilon^2\) means halving the tolerance quadruples the labels; in the third line, where some rule is perfect, the division is by \(\varepsilon\) alone, so halving the tolerance only doubles them.
- \(C_1\) and \(C_2\) are **absolute constants**: fixed numbers, the same for every class, distribution and tolerance. The theorem says they exist but not what they are.
- The two \(\le\) signs trap the true sample size between a floor and a ceiling of the same shape.

In plain words: **the labels a class needs grow in step with its VC dimension plus the log of one over the failure chance, divided by the tolerance squared (or, when some rule is perfect, roughly by the tolerance).** For thresholds, \(d = 1\), at \(\varepsilon = \delta = 0.05\):

```python
from math import log

d, eps, delta = 1, 0.05, 0.05
agnostic = (d + log(1 / delta)) / eps ** 2
pac_low = (d + log(1 / delta)) / eps
pac_high = (d * log(1 / eps) + log(1 / delta)) / eps
print(round(agnostic), round(pac_low), round(pac_high))
```

It prints `1598 80 120`. These are the shapes without the unknown constants: the agnostic answer is some constant times 1,598, and the realizable one lies between a constant times 80 and a constant times 120. The direct argument’s 74 fits that picture.

Compare with lessons 2 and 4: wherever \(\log |\mathcal H|\) stood, \(d\) now stands. That is the answer to the reviewer in one line. This quantitative version is proved in a later lesson. Its proof needs tools we don’t have yet; the proof in the next lesson gives a weaker bound with the same message.

One caution about scope. Both versions are about binary classification with the 0-1 loss. Similar results hold for some other problems, such as regression with the absolute or squared loss, but not for every learning problem: there are problems that can be learned even though uniform convergence fails, and problems where ERM fails while some other learner succeeds. Later lessons meet both.

[[diagram:theorem-map]]

## The easy arrows

Most of the proof is a chain of short steps. Let’s walk around the circle.

**1 implies 2.** This is lesson 4. If \(S\) is \(\varepsilon/2\)-representative, then for the ERM hypothesis \(h_S\) and any \(h \in \mathcal H\), \(L_{\mathcal D}(h_S) \le L_S(h_S) + \varepsilon/2 \le L_S(h) + \varepsilon/2 \le L_{\mathcal D}(h) + \varepsilon\): each outer step moves at most \(\varepsilon/2\) between training and true risk, and the middle step holds because ERM has the fewest training mistakes. So \(h_S\) is within \(\varepsilon\) of the best in the class, and uniform convergence makes \(S\) \(\varepsilon/2\)-representative with probability at least \(1 - \delta\) once \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\).

**2 implies 3.** An ERM rule is a learner. If it succeeds, the class is agnostic PAC learnable.

**3 implies 4, and 2 implies 5.** PAC learning is the special case of agnostic PAC learning in which some \(h \in \mathcal H\) has true risk 0. In that case \(\min_{h \in \mathcal H} L_{\mathcal D}(h) = 0\), where \(\min_{h \in \mathcal H}\) means “the smallest value over all rules in the class”, the true risk of the best rule, and the agnostic promise “true risk at most \(\min_h L_{\mathcal D}(h) + \varepsilon\)” becomes the PAC promise “true risk at most \(\varepsilon\)”. The same learner, with the same sample complexity, keeps it.

**4 implies 6, and 5 implies 6.** Read them backwards. If the VC dimension were infinite, the theorem in “The VC dimension” says no learner PAC learns \(\mathcal H\), so 4 fails. And 5 implies 4, since a successful ERM rule is a PAC learner, so 5 fails too.

**6 implies 1.** This is the hard one, and its proof gets a lesson of its own, the next one. For now we take it on trust, and this lesson puts the theorem to work. Here is the plan, so you know what is coming. It has two parts. First, **Sauer’s lemma**: a class of VC dimension \(d\), however large, produces only polynomially many labellings of \(m\) points, roughly \(m^d\) instead of \(2^m\). Second, a generalization of lesson 4: uniform convergence holds whenever a class has few labellings in this sense, which we will call a small **effective size**.

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

*Proof.* The learner returns \([x_{\min}, x_{\max}]\), the smallest and largest junk scores in \(S\), or the empty interval if there are none. Every junk score lies in \([a^\star, b^\star]\), so the learned interval sits inside the true one, and the learner errs only on true junk it leaves out. Writing \(\mathcal D(I)\) for the share of mail whose score lands in a strip \(I\), that is \(L_{\mathcal D}(h_S) = \mathcal D([a^\star, x_{\min})) + \mathcal D((x_{\max}, b^\star])\), or \(\mathcal D([a^\star, b^\star])\) if there are no junk examples. If \(\mathcal D([a^\star, b^\star]) \le \varepsilon\), every outcome has risk at most \(\varepsilon\). Otherwise, choose a strip \([a^\star, a']\) at the left end of the true interval and a strip \([b', b^\star]\) at its right end, each with probability exactly \(\varepsilon/2\); they do not overlap, since together they hold \(\varepsilon\) and the interval holds more. If some example lands in the left strip, it is junk, so \(x_{\min} \le a'\) and the left error is at most \(\varepsilon/2\); the same holds on the right. So failure needs an empty strip. Each strip is empty with probability \((1 - \varepsilon/2)^m \le e^{-\varepsilon m/2}\), and the union bound gives \(\mathbb P[L_{\mathcal D}(h_S) > \varepsilon] \le 2e^{-\varepsilon m/2} \le \delta\) once \(m \ge 2\log(2/\delta)/\varepsilon\). \(\blacksquare\)

In the risk formula, \([a^\star, x_{\min})\) is the true junk below the lowest junk example and \((x_{\max}, b^\star]\) the true junk above the highest, so their sum is the junk the learner keeps. The sample size \(2\log(2/\delta)/\varepsilon\) is the threshold formula with \(\varepsilon\) halved, because each strip now gets only half the tolerance. In plain words: **the tightest interval errs only at its two ends, and twice the threshold budget makes both ends small.** At \(\varepsilon = \delta = 0.05\): \(2 \times 3.69/0.05 \approx 147.6\), so 148. In the snippet below, `m_itv` is this formula and `tightest_risk` is the risk formula, with uniform scores so that each \(\mathcal D\) of a strip is its width:

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

Notice the division of labour. The VC dimension decides whether any budget can work: finite for thresholds and intervals, infinite for unions of intervals. For these two classes on the line, a direct argument then pins the number, because the general theorem leaves its constants unknown, and the constants its proof can deliver, as the next lesson shows, are far too generous for a budget.

In the lab below you face the same choice. Each round the hidden truth changes. You choose the class and the number of labels, and the lab tells you whether your choice is certified before it trains. Watch what happens when you pick a class that is too rich, or one too poor.

[[lab-act2]]

## What people get wrong

**“An infinite class can’t be learned.”** Thresholds are infinite and learnable from 74 labels at \(\varepsilon = \delta = 0.05\). What decides learnability is the VC dimension, a property of how the class labels points, not how many hypotheses it lists.

**“To show the VC dimension is at least \(d\), I need every set of \(d\) points to be shattered.”** One shattered set of size \(d\) is enough. Thresholds on \([0, 1]\) with \(t \in [0, 1]\) cannot shatter \(\{0\}\), and their VC dimension is still 1. The asymmetry runs the other way for the upper bound: to show it is less than \(d + 1\), *every* set of \(d + 1\) points must fail, and one failing set proves nothing.

**“The VC dimension is the number of parameters.”** Often, not always. One parameter, \(\theta\), gives the class \(\lceil 0.5 \sin(\theta x) \rceil\) an infinite VC dimension.

## Recap and the next bug

1. The finite-class guarantees pay \(\log |\mathcal H|\), which is useless for an infinite class. Yet thresholds on the score line are PAC learnable with \(m \le \lceil \log(2/\delta)/\varepsilon \rceil\), by a two-strip argument that never counts hypotheses.
2. The **restriction** \(\mathcal H_C\) is the set of labellings a class produces on \(C\). A class **shatters** \(C\) when it produces all \(2^{|C|}\) of them, and its **VC dimension** is the size of the largest set it shatters. If a class shatters \(2m\) points, the No-Free-Lunch argument defeats every learner with \(m\) examples, so infinite VC dimension means not PAC learnable.
3. Thresholds have VC dimension 1, intervals 2, axis-aligned rectangles 4, a finite class at most \(\log_2 |\mathcal H|\), unions of intervals infinity.
4. **The fundamental theorem of statistical learning**: for binary classification with the 0-1 loss, uniform convergence, agnostic PAC learnability by ERM, agnostic PAC learnability, PAC learnability, PAC learnability by ERM and a finite VC dimension are all equivalent, and the sample complexity scales with \(d\) where \(\log |\mathcal H|\) used to stand. Every arrow but one is a short step; the hard one, “finite VC dimension gives uniform convergence”, is still owed.
5. On the score line, ERM for thresholds and intervals is a sort and a scan, \(O(m \log m)\). When the junk is an interval, the tightest interval is certified at \(\varepsilon = \delta = 0.05\) by 148 labels, twice the threshold budget because each end of the interval gets half the tolerance.

Your filter ships with a certificate. Look at what it leans on, though. The 148 came from a strip argument, and that argument needs every label to be right: one junk example in each strip pins the tightest interval. And the theorem that told you intervals were learnable at all rests on the one arrow you took on trust.

Then the product team puts a junk button in every inbox. Labels used to cost a reviewer’s afternoon; now five million clicks arrive each week, for free. But clicks are careless, and about one in twenty is wrong. One careless junk click on a newsletter at 0.97 stretches the tightest interval all the way up to 0.97, and with five million clicks, no interval at all agrees with every label. The strip argument has nothing to say. The fundamental theorem says a finite VC dimension is enough even here, for ERM on noisy labels, but only through the arrow you skipped. Does every class of finite VC dimension really have uniform convergence, and how many clicks does that take? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Consider the class with just two hypotheses on the score line: always keep and always junk. What is its VC dimension? Now consider thresholds \(h_t(x) = \mathbb 1[x > t]\) on \(\mathcal X = [0, 1]\), but with \(t\) allowed only in \([0, 1]\). Which single points does this class shatter, and what is its VC dimension?
2. Let \(\mathcal H_2\) be “junk if the score lies in one of at most two intervals”. Find \(\mathrm{VCdim}(\mathcal H_2)\), with both halves of the proof: a set of that size it shatters, and an argument that works for every larger set.

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (boxes in more dimensions).** A message now carries \(n\) scores instead of two, and a box rule says junk when every score lies in its own range: the rectangle rule with one interval per score, joined by “and”. Show that the VC dimension of boxes with \(n\) scores is \(2n\).
- **Problem 2 (signed intervals).** A signed interval has ends \(a \le b\) and a sign \(s \in \{0, 1\}\): it gives the label \(s\) to every score in \([a, b]\) and the other label, \(1 - s\), to every score outside. So it can say junk inside a band, or keep inside a band and junk everywhere else. Find the VC dimension of the class of signed intervals, with both halves of the proof.
