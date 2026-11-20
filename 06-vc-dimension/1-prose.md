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

A reminder of the scores we judge a rule by. The true risk \(L_{\mathcal D}(h)\) is the chance that \(h\) mislabels the next message drawn from the distribution \(\mathcal D\) of arriving mail, and the training sequence \(S\) is the list of \(m\) labelled examples, drawn i.i.d. (independently, each from the same \(\mathcal D\)). ERM, empirical risk minimization, returns a rule with the fewest mistakes on \(S\); we write \(h_S\) for the rule it returns. Recall from lesson 3 that a class is **PAC learnable** when some learner, given at least \(m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples from any distribution, labelled by any \(f\) the class realizes, returns a hypothesis with true risk at most \(\varepsilon\) with probability at least \(1 - \delta\). Here \(m_{\mathcal H}(\varepsilon, \delta)\), the **sample complexity**, is a number of labels that depends only on the two tolerances, never on \(\mathcal D\) or \(f\). For lesson 1’s four rules at \(\varepsilon = 0.1\), \(\delta = 0.05\), it was 44.

We need one mild assumption about the scores: no single score value carries positive probability. A score computed from a whole message is a continuous quantity, so this is a fair assumption, and it lets us cut the score line into pieces of exactly the probability we want.

What should we expect? With tolerances \(\varepsilon = \delta = 0.1\), we will see that 30 labels are enough for thresholds, whatever the mail looks like, and with \(\varepsilon = \delta = 0.05\), 74. Here is the general statement.

**Claim (thresholds are PAC learnable).** Let \(\varepsilon, \delta \in (0, 1)\), that is, both strictly between 0 and 1. Every ERM learner for \(\mathcal H_{\mathrm{thr}}\), given

\[ m \ge \frac{\log(2/\delta)}{\varepsilon} \]

examples, returns a hypothesis with \(L_{\mathcal D}(h_S) \le \varepsilon\) with probability at least \(1 - \delta\). So \(m_{\mathcal H_{\mathrm{thr}}}(\varepsilon, \delta) \le \lceil \log(2/\delta)/\varepsilon \rceil\).

Compare it with the finite-class formula: \(|\mathcal H|\) has been replaced by 2, as if the infinite class of thresholds were a class of two rules. The ceiling brackets \(\lceil \cdot \rceil\) round up to the next whole number. In plain words: **about \(\log(2/\delta)/\varepsilon\) labels give any ERM threshold a true risk of at most \(\varepsilon\), with probability at least \(1 - \delta\).** At \(\varepsilon = \delta = 0.1\): \(\log 20 \approx 3.00\), divided by 0.1 is 29.96, rounded up to 30. The line computing `m` in the simulation after the proof is this formula, and so is `by_threshold_argument` in the first snippet, which gave 74 at 0.05.

Before the proof, a picture of where the mistakes can live. Draw the score line with \(t^\star\) marked. Just to the left of \(t^\star\) lies a strip of scores that carries probability exactly \(\varepsilon\); just to the right lies another. If the training sequence puts at least one example in each strip, the learner’s threshold is trapped between them, and every mistake it can make lies inside one strip.

[[diagram:strips]]

Two pieces of notation for the proof. A strip of scores is written with brackets: \((a, b]\) means every score above \(a\) and up to \(b\), the round bracket saying the end is left out and the square one that it is kept in, so with \(t^\star = 0.6\) the strip \((0.5, 0.6]\) holds 0.55 and 0.6 but not 0.5. And \(\mathbb P_{x \sim \mathcal D}[x \in (a, b]]\), read “the probability that a score \(x\) drawn from \(\mathcal D\) lands in the strip”, is the share of arriving mail whose score falls there; for uniform scores it is just the strip’s width, 0.1 here.

*Proof.* Choose \(a_0 < t^\star\) so that \(\mathbb P_{x \sim \mathcal D}[x \in (a_0, t^\star]] = \varepsilon\), and \(a_1 > t^\star\) so that \(\mathbb P[x \in (t^\star, a_1]] = \varepsilon\). Such points exist because no single score carries probability, so the probability of \((a, t^\star]\) moves continuously as \(a\) slides. If all the scores below \(t^\star\) together carry probability at most \(\varepsilon\), set \(a_0 = -\infty\) instead, and similarly \(a_1 = +\infty\).

Now look at what ERM returns. Let \(b_0\) be the largest score among the keep examples in \(S\), and \(b_1\) the smallest score among the junk examples (\(b_0 = -\infty\) if there is no keep example, \(b_1 = +\infty\) if there is no junk example). Under realizability, \(h_{t^\star}\) has empirical risk 0 (no mistakes on \(S\)), so ERM returns some **consistent** \(h_t\), one that also makes no mistakes on \(S\), and \(h_t\) is consistent exactly when every keep score is at most \(t\) and every junk score is above \(t\): that is, \(b_0 \le t < b_1\).

Where can \(h_t\) be wrong? It disagrees with the truth exactly on the scores between \(t\) and \(t^\star\): on \((t, t^\star]\) if \(t < t^\star\), and on \((t^\star, t]\) if \(t \ge t^\star\). Suppose \(S\) has an example in each strip. An example in \((a_0, t^\star]\) is a keep example, so \(b_0 > a_0\), so \(t > a_0\), and when \(t < t^\star\) the mistakes lie in \((a_0, t^\star]\), probability at most \(\varepsilon\). An example in \((t^\star, a_1]\) is a junk example, so \(b_1 \le a_1\), so \(t < a_1\), and when \(t \ge t^\star\) the mistakes lie in \((t^\star, a_1]\), probability at most \(\varepsilon\). (If a strip was set to infinity, the same conclusion holds for free, since all the scores on that side carry at most \(\varepsilon\).) So \(L_{\mathcal D}(h_S) > \varepsilon\) can only happen if some strip is empty.

What is the chance that the left strip receives no example? Each of the \(m\) scores misses it with probability \(1 - \varepsilon\), and they are drawn independently, so all of them miss it with probability \((1 - \varepsilon)^m \le e^{-\varepsilon m}\), using \(1 - \varepsilon \le e^{-\varepsilon}\) from lesson 2. Here \(e \approx 2.718\) is Euler’s number, and the power \((1 - \varepsilon)^m\) multiplies \(m\) copies of the miss chance together, one per independent example; with \(\varepsilon = 0.1\) and \(m = 30\) it is \(0.9^{30} \approx 0.042\), under \(e^{-3} \approx 0.050\). The same holds for the right strip. The **union bound** from lesson 2 says that the chance that at least one of several bad events happens is at most the sum of their chances. Here the bad events are “left strip empty” and “right strip empty”, so

\[ \mathbb P_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(h_S) > \varepsilon \right] \le 2 e^{-\varepsilon m}, \]

and \(2e^{-\varepsilon m} \le \delta\) exactly when \(m \ge \log(2/\delta)/\varepsilon\). \(\blacksquare\)

Read the last display from the inside out. \(L_{\mathcal D}(h_S) > \varepsilon\) is the bad event: the rule ERM returned is worse than tolerated. The subscript \(S \sim \mathcal D^m\) on \(\mathbb P\) says where the randomness lives: the whole training sequence, \(m\) examples drawn i.i.d. from \(\mathcal D\), which is what \(\mathcal D^m\) means. On the right, the 2 counts the strips and \(e^{-\varepsilon m}\) is the bound for one strip. In plain words: **the chance of training on an unlucky sequence is at most twice the chance that one given strip stays empty.** Taking logs, \(2e^{-\varepsilon m} \le \delta\) is the same as \(\varepsilon m \ge \log(2/\delta)\), which is the claim. With \(\varepsilon = 0.1\) and \(m = 30\):

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

It prints `30 0.08507 0.0996`. Thirty labels; even the most unlucky ERM learner fails about 8.5% of the time (the exact figure is \(2(0.9)^{30} - (0.8)^{30} \approx 8.35\%\)); and the bound \(2e^{-3} \approx 9.96\%\) sits just above, under \(\delta = 10\%\). With \(\varepsilon = \delta = 0.05\), the same claim asks for 74 labels: the second number the first snippet printed, against 948 from counting floating-point numbers.

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

In the PAC model, though, the adversary is restricted: it may only use labellings that some \(h \in \mathcal H\) achieves with risk 0. On \(C\), that means labellings in \(\mathcal H_C\). Now suppose \(\mathcal H\) shatters \(C\). Then every labelling of \(C\) is in \(\mathcal H_C\), the restriction costs the adversary nothing, and the whole argument goes through inside the PAC model.

**Corollary (shattering defeats every learner).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\{0, 1\}\), and \(m\) a training sequence size. If some set \(C \subset \mathcal X\) of size \(2m\) is shattered by \(\mathcal H\), then for every learner \(A\) there exist a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\) and a hypothesis \(h \in \mathcal H\) with \(L_{\mathcal D}(h) = 0\), such that with probability at least \(1/7\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \ge 1/8\).

Here \(\mathcal X \times \{0, 1\}\) is the set of pairs (score, label), so \(\mathcal D\) chooses both the message and its label, and \(A(S)\) is the rule learner \(A\) returns after training on \(S\). In plain words: **if the class can label some \(2m\) messages in every way, then for any learner there is a world the class fits perfectly in which, at least one time in seven, \(m\) labels leave the learner wrong on at least one message in eight.** The class of all rules on lesson 1’s sixteen tiny-world messages labels those sixteen in every way, so with \(m = 8\) labels every learner can be pushed into that trap.

*Proof idea.* Run the No-Free-Lunch construction on the shattered set \(C\). Each of its labellings \(f_i: C \to \{0,1\}\) equals the restriction of some \(h_i \in \mathcal H\), because \(\mathcal H\) shatters \(C\), and \(h_i\) has true risk 0 under the distribution built from \(f_i\). So every distribution the adversary uses is realizable by \(\mathcal H\), and the theorem’s conclusion applies to it unchanged. \(\blacksquare\)

There is an old saying behind this: an explanation that can explain anything explains nothing. If the class can label half the set any way at all and the other half any way at all, then the labels you have seen say nothing about the labels you have not.

## The VC dimension

So the question to ask of a class is not “how many hypotheses?” but “how large a set can it shatter?”

**Definition (VC dimension).** The **VC dimension** of a class \(\mathcal H\), written \(\mathrm{VCdim}(\mathcal H)\), is the largest size of a set \(C \subset \mathcal X\) that \(\mathcal H\) shatters. If \(\mathcal H\) shatters sets of every finite size, its VC dimension is infinite, written \(\infty\).

In plain words: **the VC dimension is the most points the class can label in every possible way.** Thresholds shattered \(\{0.3\}\) but not \(\{0.3, 0.7\}\); and we saw that no pair works, because \((1, 0)\) is always missing, so their VC dimension is 1.

The name honours Vladimir Vapnik and Alexey Chervonenkis, who discovered this characterization in 1970. An immediate consequence of the corollary:

**Theorem (infinite VC dimension means not PAC learnable).** If \(\mathrm{VCdim}(\mathcal H) = \infty\), then \(\mathcal H\) is not PAC learnable.

*Proof.* Suppose some learner \(A\) PAC learns \(\mathcal H\), and pick \(\varepsilon < 1/8\) and \(\delta < 1/7\). Let \(m = m_{\mathcal H}(\varepsilon, \delta)\). Since the VC dimension is infinite, \(\mathcal H\) shatters some set of size \(2m\). By the corollary there is a realizable distribution on which, with probability at least \(1/7 > \delta\), \(A\) returns a hypothesis with true risk at least \(1/8 > \varepsilon\). That contradicts the PAC guarantee. \(\blacksquare\)

The rest of the lesson proves the converse: a finite VC dimension guarantees learnability. But first let’s measure some classes. To prove \(\mathrm{VCdim}(\mathcal H) = d\), you need two things, and they are not symmetric:

1. **Some** set of size \(d\) is shattered by \(\mathcal H\). One example is enough.
2. **Every** set of size \(d + 1\) fails to be shattered. One example is not enough; you need an argument that works for any set.

For thresholds, step 1 is any single point, and step 2 is the missing labelling \((1, 0)\), which we showed is missing for every pair \(c_1 < c_2\). So \(\mathrm{VCdim}(\mathcal H_{\mathrm{thr}}) = 1\). An infinite class with VC dimension 1. Keep that in mind; it is what you will tell the reviewer.

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

Read it from the inside out. \(a_1 \le x_1 \le a_2\) asks whether the spam score lies in the band from \(a_1\) to \(a_2\); \(b_1 \le x_2 \le b_2\) asks the same of the reputation score with its own band; “and” demands both; the indicator turns the answer into 1 or 0. The four numbers in the subscript name the rule, two ends per score. Drawn with \(x_1\) across and \(x_2\) up, the junk region is a rectangle with sides parallel to the axes, hence the name. In plain words: **junk exactly when the spam score is in one range and the reputation score is in another.** The rule from the opening sentence is \(a_1 = 0.5\), \(a_2 = 1\), \(b_1 = 0\), \(b_2 = 0.3\):

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

We have shown one direction: infinite VC dimension rules out PAC learning. The converse is also true, and together they form the central result of this part of the course. It ties together every idea from lessons 2 to 5: learnability, ERM, uniform convergence and now the VC dimension. Recall from lesson 4 that a class has the **uniform convergence property** when there is a function \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) such that, for every distribution \(\mathcal D\), a training sequence of at least that many i.i.d. examples is \(\varepsilon\)-representative (every \(h \in \mathcal H\) has \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\)) with probability at least \(1 - \delta\).

The idea first, with lesson 1’s numbers. There, the rule “unknown sender means junk” got 25% of its labelled examples wrong but 50% of all mail: a gap of 0.25 between how good it looked and how good it was. A training sequence is representative when no rule in the class has a large gap. In the condition, \(L_S(h)\) is the empirical risk, the share of the training sequence \(h\) gets wrong; \(L_{\mathcal D}(h)\) is the true risk; the bars \(|\cdot|\) around a number mean its absolute value, the size of the gap whichever way it points; and “every \(h \in \mathcal H\)” asks it of all rules at once. The superscript in \(m^{\mathrm{UC}}_{\mathcal H}\) just marks this as the uniform-convergence sample size, as opposed to the learning one. In plain words: **a class has uniform convergence when enough labels make every rule’s training score close to its true score, for every rule at once and whatever the mail looks like.** Thresholds on 200 uniform scores with \(t^\star = 0.6\), checked over cut-offs \(0, 0.01, \dots, 1\), where the true risk of \(h_t\) is just the distance \(|t - 0.6|\):

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

Compare with lessons 2 and 4: wherever \(\log |\mathcal H|\) stood, \(d\) now stands. That is the answer to the reviewer in one line. This quantitative version is proved in a later lesson. Its proof needs tools we don’t have yet; the proof below gives a weaker bound with the same message.

One caution about scope. Both versions are about binary classification with the 0-1 loss. Similar results hold for some other problems, such as regression with the absolute or squared loss, but not for every learning problem: there are problems that can be learned even though uniform convergence fails, and problems where ERM fails while some other learner succeeds. Later lessons meet both.

[[diagram:theorem-map]]

## The easy arrows

Most of the proof is a chain of short steps. Let’s walk around the circle.

**1 implies 2.** This is lesson 4. If \(S\) is \(\varepsilon/2\)-representative, then any ERM hypothesis has true risk within \(\varepsilon\) of the best in the class, and uniform convergence makes \(S\) \(\varepsilon/2\)-representative with probability at least \(1 - \delta\) once \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\).

**2 implies 3.** An ERM rule is a learner. If it succeeds, the class is agnostic PAC learnable.

**3 implies 4, and 2 implies 5.** PAC learning is the special case of agnostic PAC learning in which some \(h \in \mathcal H\) has true risk 0. In that case \(\min_{h \in \mathcal H} L_{\mathcal D}(h) = 0\), where \(\min_{h \in \mathcal H}\) means “the smallest value over all rules in the class”, the true risk of the best rule, and the agnostic promise “true risk at most \(\min_h L_{\mathcal D}(h) + \varepsilon\)” becomes the PAC promise “true risk at most \(\varepsilon\)”. The same learner, with the same sample complexity, keeps it.

**4 implies 6, and 5 implies 6.** Read them backwards. If the VC dimension were infinite, the theorem in “The VC dimension” says no learner PAC learns \(\mathcal H\), so 4 fails. And 5 implies 4, since a successful ERM rule is a PAC learner, so 5 fails too.

**6 implies 1.** This is the hard one, and the rest of the lesson’s theory is its proof. The plan has two parts. First, **Sauer’s lemma**: a class of VC dimension \(d\), however large, produces only polynomially many labellings of \(m\) points, roughly \(m^d\) instead of \(2^m\). Second, a generalization of lesson 4: uniform convergence holds whenever a class has few labellings in this sense, which we will call a small **effective size**.

## Counting labellings: the growth function

We need a name for “how many labellings of \(m\) points, at most”. For thresholds and two points, every pair of distinct scores gets the same three labellings we found on \(\{0.3, 0.7\}\), so the answer for \(m = 2\) is 3. For a class where the choice of points matters, we take the best choice, the set on which the class is most flexible.

**Definition (growth function).** The **growth function** of \(\mathcal H\) is \(\tau_{\mathcal H}: \mathbb N \to \mathbb N\),

\[ \tau_{\mathcal H}(m) = \max_{C \subset \mathcal X : |C| = m} |\mathcal H_C|. \]

From the inside out:

- \(|\mathcal H_C|\) is how many distinct labellings the class produces on the set \(C\).
- Under \(\max\), the condition \(C \subset \mathcal X : |C| = m\) reads “over every set \(C\) of domain points such that \(C\) has \(m\) points”, and \(\max\) keeps the largest count.
- \(\tau\) is the Greek letter tau, and \(\tau_{\mathcal H}: \mathbb N \to \mathbb N\) says the growth function takes a whole number of points (\(\mathbb N\), the natural numbers) and returns a whole number of labellings.

In plain words: **\(\tau_{\mathcal H}(m)\) is the largest number of distinct labellings that \(\mathcal H\) produces on any set of \(m\) points.** For thresholds, \(\tau(2) = 3\). In the snippet below, `tau` counts \(|\mathcal H_C|\) on one set; on the line that is already the maximum, as the next paragraph explains. Always \(\tau_{\mathcal H}(m) \le 2^m\), and if \(m \le \mathrm{VCdim}(\mathcal H)\), then some set of size \(m\) is shattered, so \(\tau_{\mathcal H}(m) = 2^m\). The interesting question is what happens past the VC dimension. Let’s just count, for thresholds and intervals. On the line, only the order of the points matters, so the points \(0, 1, \dots, m-1\) are as good as any \(m\) distinct scores.

The snippet also computes one more count, which will turn out to be the answer. The **binomial coefficient** \(\binom{m}{i}\), read “\(m\) choose \(i\)”, is the number of ways to pick \(i\) of \(m\) points: from four messages you can pick two in \(\binom{4}{2} = 6\) ways, one in \(\binom{4}{1} = 4\) ways, none in \(\binom{4}{0} = 1\) way. The sum \(\sum_{i=0}^{d} \binom{m}{i}\) adds those counts for \(i = 0, 1, \dots, d\): the number of ways to pick at most \(d\) of the \(m\) points. With \(m = 4\) and \(d = 1\) it is \(1 + 4 = 5\). In the code, `comb(m, i)` is \(\binom{m}{i}\) and `sauer(m, d)` is the sum:

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

The left side is the growth function, the most labellings the class makes on \(m\) points. The right side counts the ways to pick at most \(d\) of those \(m\) points, \(\binom{m}{i}\) ways for each size \(i\) from 0 to \(d\). In plain words: **a class that shatters no more than \(d\) points can label \(m\) points in no more ways than there are ways to choose at most \(d\) of them.** For intervals, \(d = 2\), and \(m = 8\): \(1 + 8 + 28 = 37\), the last row of the table, against \(2^8 = 256\). The `sauer` function in the table snippet is this right-hand side.

In particular, if \(1 \le d \le m\), then \(\tau_{\mathcal H}(m) \le (em/d)^d\). This simpler bound is a power with fixed exponent \(d\), a polynomial in \(m\), with \(e \approx 2.718\) again; for \(m = 8\) and \(d = 2\) it is \((4e)^2 \approx 118\), looser than 37 but far below 256. The last snippet of this section checks it.

It is also called the Sauer-Shelah-Perles lemma: Sauer, Shelah with Perles, and Vapnik and Chervonenkis each found it independently.

The proof is a beautiful piece of counting, and the trick is to prove something stronger. Instead of counting labellings against a formula, count them against the shattered subsets. Back to thresholds on \(\{0.3, 0.7\}\): they make 3 labellings. Which parts of the set do they shatter? The **empty set**, written \(\emptyset\), the set with no points, counts as shattered, since its single labelling (the empty one) is produced by any hypothesis. Each single point is shattered. The pair is not. That is 3 shattered subsets, the same as the number of labellings. The stronger claim says the labellings can never outnumber the shattered subsets.

**Stronger claim.** For every class \(\mathcal H\) (non-empty) and every finite set \(C = \{c_1, \dots, c_m\}\),

\[ |\mathcal H_C| \le \left| \{ B \subseteq C : \mathcal H \text{ shatters } B \} \right|. \]

On the left, the number of labellings of \(C\). On the right, the braces collect every subset \(B\) of \(C\) (\(\subseteq\) allows \(B\) to be all of \(C\), or empty) such that the class shatters \(B\), and the outer bars count them. In plain words: **a class labels a set in at most as many ways as the set has shattered pieces.** Here is the count for thresholds on \(\{0.3, 0.7\}\), with one cut-off per gap as before:

```python
from itertools import combinations

C = [0.3, 0.7]
H = [lambda x, t=t: int(x > t) for t in (0.0, 0.5, 1.0)]
HC = {tuple(h(c) for c in C) for h in H}

def shatters(B):
    labels = {tuple(h(c) for c in B) for h in H}
    return len(labels) == 2 ** len(B)

subsets = [B for k in range(3) for B in combinations(C, k)]
print(len(HC), [B for B in subsets if shatters(B)])
```

It prints `3 [(), (0.3,), (0.7,)]`: three labellings, three shattered subsets, the empty one first.

*Why the stronger claim gives Sauer’s lemma.* If \(\mathrm{VCdim}(\mathcal H) \le d\), no subset of size above \(d\) is shattered, so the shattered subsets of \(C\) are among the subsets of size \(0, 1, \dots, d\), of which there are \(\sum_{i=0}^{d} \binom{m}{i}\). Apply this to the set \(C\) that achieves the maximum in \(\tau_{\mathcal H}(m)\).

*Proof of the stronger claim,* by induction on \(m\), for all classes at once.

*Base case, \(m = 1\).* Either every hypothesis gives \(c_1\) the same label, so \(|\mathcal H_C| = 1\) and the only shattered subset is \(\emptyset\); or both labels occur, so \(|\mathcal H_C| = 2\) and both \(\emptyset\) and \(\{c_1\}\) are shattered. Equality either way.

*Inductive step.* Assume the claim for every class and every set of fewer than \(m\) points. Fix \(\mathcal H\) and \(C = \{c_1, \dots, c_m\}\), and let \(C' = \{c_2, \dots, c_m\}\), the set without its first point. Split the labellings in \(\mathcal H_C\) by their **tail**, the labels on \(C'\):

\[ Y_0 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ or } (1, y_2, \dots, y_m) \in \mathcal H_C \}, \]

\[ Y_1 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ and } (1, y_2, \dots, y_m) \in \mathcal H_C \}. \]

Both sets are built the same way. Each vector \((y_2, \dots, y_m)\) is a tail, the labels on every point but the first; the condition after the colon asks whether that tail appears in \(\mathcal H_C\) with first label 0, with first label 1, or with both. The only difference is the word: “or” in \(Y_0\), “and” in \(Y_1\). In plain words: **\(Y_0\) holds every tail that occurs, and \(Y_1\) the tails that occur with both labels on \(c_1\).** On \(\{0.3, 0.7\}\), with \(c_1 = 0.3\), the labellings are \((0, 0)\), \((0, 1)\) and \((1, 1)\). The tails are \((0)\) and \((1)\), so \(Y_0\) has both. Only the tail \((1)\) comes with both first labels, from \((0, 1)\) and \((1, 1)\), so \(Y_1 = \{(1)\}\).

```python
HC = {(0, 0), (0, 1), (1, 1)}
Y0 = {v[1:] for v in HC}
Y1 = {v[1:] for v in HC if (1 - v[0],) + v[1:] in HC}
print(sorted(Y0), sorted(Y1), len(Y0) + len(Y1))
```

It prints `[(0,), (1,)] [(1,)] 3`. In the code, `v[1:]` is the tail, and the test for `Y1` asks whether the vector with its first label flipped is also present. The sizes add up to \(|\mathcal H_C| = 3\), which is Step 1.

*Step 1: \(|\mathcal H_C| = |Y_0| + |Y_1|\).* Group the labellings in \(\mathcal H_C\) by tail. A tail in \(Y_0\) but not \(Y_1\) occurs with exactly one label on \(c_1\), so it accounts for one labelling. A tail in \(Y_1\) accounts for two. Since \(Y_1 \subseteq Y_0\), the total is \(|Y_0 \setminus Y_1| + 2|Y_1| = |Y_0| + |Y_1|\). Here \(Y_0 \setminus Y_1\), “\(Y_0\) minus \(Y_1\)”, is the set of tails in \(Y_0\) but not in \(Y_1\); in the example, \(\{(0)\}\), one labelling, plus twice the one tail in \(Y_1\), gives 3.

*Step 2: bound \(Y_0\).* \(Y_0\) is exactly the restriction of \(\mathcal H\) to \(C'\): the tails that occur are the labellings of \(C'\). By the induction hypothesis on \(C'\),

\[ |Y_0| = |\mathcal H_{C'}| \le |\{ B \subseteq C' : \mathcal H \text{ shatters } B \}| = |\{ B \subseteq C : c_1 \notin B,\ \mathcal H \text{ shatters } B \}|. \]

The notation is the stronger claim’s, applied to \(C'\); the last equality just renames subsets of \(C'\) as subsets of \(C\) that leave out \(c_1\), with \(\notin\) read “is not in”. In the example, \(C' = \{0.7\}\) has two shattered subsets, \(\emptyset\) and \(\{0.7\}\), against \(|Y_0| = 2\).

*Step 3: bound \(Y_1\) through a smaller class.* Let \(\mathcal H'\) be the hypotheses that have a **partner**: another hypothesis in \(\mathcal H\) that agrees with it on \(C'\) and disagrees on \(c_1\). In the example, the cut-off 0.5 labels \((0, 1)\) and the cut-off 0.0 labels \((1, 1)\): same tail, different first label, so each is the other’s partner. The cut-off 1.0 labels \((0, 0)\), and nothing labels \((1, 0)\), so it has no partner. In symbols,

\[ \mathcal H' = \{ h \in \mathcal H : \exists h' \in \mathcal H,\ h'(c_1) = 1 - h(c_1),\ h'(c_i) = h(c_i) \text{ for } i = 2, \dots, m \}. \]

From the inside out: \(h'(c_i) = h(c_i)\) for \(i = 2, \dots, m\) says \(h'\) copies \(h\) on every point of \(C'\); \(h'(c_1) = 1 - h(c_1)\) says it flips the label on \(c_1\), since \(1 - 0 = 1\) and \(1 - 1 = 0\); \(\exists h' \in \mathcal H\) reads “there exists some \(h'\) in the class” with both properties; and the braces keep every \(h\) for which such an \(h'\) exists. In plain words: **\(\mathcal H'\) is the set of rules that have a twin differing only at the first point.**

```python
C, ts = [0.3, 0.7], (0.0, 0.5, 1.0)
lab = {t: tuple(int(c > t) for c in C) for t in ts}

def has_partner(t):
    return any(lab[u][0] == 1 - lab[t][0] and
               lab[u][1:] == lab[t][1:] for u in ts)

H1 = [t for t in ts if has_partner(t)]
print(H1, {lab[t][1:] for t in H1})
```

It prints `[0.0, 0.5] {(1,)}`: two cut-offs have partners, and their tails make exactly \(Y_1\). The `any` is the \(\exists\).

A tail is in \(Y_1\) exactly when it is the tail of a hypothesis with a partner, so \(Y_1 = \mathcal H'_{C'}\). By the induction hypothesis, now applied to the class \(\mathcal H'\) on \(C'\),

\[ |Y_1| = |\mathcal H'_{C'}| \le |\{ B \subseteq C' : \mathcal H' \text{ shatters } B \}|. \]

Same notation as Step 2, now for the partner class. In the example, \(\mathcal H'\) labels 0.7 only as junk, so it shatters only \(\emptyset\): one subset, against \(|Y_1| = 1\).

*Step 4: each set shattered by \(\mathcal H'\) grows by \(c_1\).* Suppose \(\mathcal H'\) shatters \(B \subseteq C'\). Take any labelling of \(B \cup \{c_1\}\). Some \(g \in \mathcal H'\) produces its part on \(B\). The partner \(g'\) of \(g\) agrees with \(g\) on all of \(C'\), hence on \(B\), and gives \(c_1\) the other label; and \(g'\) is itself in \(\mathcal H'\), with \(g\) as its partner. One of \(g, g'\) gives \(c_1\) the label we want. So \(\mathcal H'\) shatters \(B \cup \{c_1\}\), and since \(\mathcal H' \subseteq \mathcal H\), so does \(\mathcal H\). The map \(B \mapsto B \cup \{c_1\}\) is one-to-one from subsets of \(C'\) to subsets of \(C\) that contain \(c_1\), so

\[ |Y_1| \le |\{ B \subseteq C : c_1 \in B,\ \mathcal H \text{ shatters } B \}|. \]

Here \(B \cup \{c_1\}\) is \(B\) with the point \(c_1\) added (\(\cup\) reads “together with”), and \(B \mapsto B \cup \{c_1\}\) is the rule that sends each \(B\) to that bigger set. In the example, \(\mathcal H'\) shatters \(\emptyset\), which grows into \(\{0.3\}\), and thresholds do shatter \(\{0.3\}\): one shattered subset containing \(c_1\), against \(|Y_1| = 1\).

*Step 5: add.* The shattered subsets of \(C\) split into those without \(c_1\) (Step 2) and those with it (Step 4), so

\[ |\mathcal H_C| = |Y_0| + |Y_1| \le |\{ B \subseteq C : \mathcal H \text{ shatters } B \}|. \qquad \blacksquare \]

In the example that is \(3 = 2 + 1 \le 2 + 1 = 3\).

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

The `Y0` and `Y1` lines are the same two lines as in the small example above, and `n_sh` counts the right side of the stronger claim. All 2,000 assertions hold, and it prints the last trial, `23 14 9 26`: 23 labellings, which split into 14 tails plus 9 doubled tails, against 26 shattered subsets.

**The polynomial bound.** It remains to show \(\sum_{i=0}^{d} \binom{m}{i} \le (em/d)^d\) when \(1 \le d \le m\). Since \(d/m \le 1\), raising it to a smaller power makes it bigger, so \((d/m)^d \le (d/m)^i\) for every \(i \le d\). Hence

\[ \left(\frac{d}{m}\right)^d \sum_{i=0}^{d} \binom{m}{i} \le \sum_{i=0}^{d} \binom{m}{i} \left(\frac{d}{m}\right)^i \le \sum_{i=0}^{m} \binom{m}{i} \left(\frac{d}{m}\right)^i = \left(1 + \frac{d}{m}\right)^m \le e^d. \]

The second step adds non-negative terms, the third is the binomial theorem, and the last uses \(1 + u \le e^u\). Multiplying both sides by \((m/d)^d\) gives the bound.

Read the chain left to right, writing \(q = d/m\) for the fraction. The first expression is the Sauer sum shrunk by \(q^d\). The second gives each term its own, larger, factor \(q^i\). The third lets \(i\) run past \(d\) all the way to \(m\), so it only adds terms. The **binomial theorem** says that sum is exactly \((1 + q)^m\): multiply out \(m\) copies of \(1 + q\), and choosing \(q\) from \(i\) of the brackets can be done in \(\binom{m}{i}\) ways. Finally \(1 + q \le e^q\), raised to the power \(m\), gives \(e^{qm} = e^d\). In plain words: **shrinking the Sauer sum by \((d/m)^d\) leaves something no bigger than \(e^d\), so the sum is at most \((em/d)^d\).** With the interval numbers \(m = 8\), \(d = 2\), so \(q = 1/4\):

```python
from math import comb, e

m, d = 8, 2
q = d / m
left = q ** d * sum(comb(m, i) for i in range(d + 1))
mid = sum(comb(m, i) * q ** i for i in range(d + 1))
full = sum(comb(m, i) * q ** i for i in range(m + 1))
print(round(left, 4), mid, round(full, 4),
      round((1 + q) ** m, 4), round(e ** d, 4))
```

It prints `2.3125 4.75 5.9605 5.9605 7.3891`: each link of the chain in order, the third and fourth equal as the binomial theorem promises. Multiplying 7.39 back by \(4^2 = 16\) gives the 118 from before, against the true 37. A quick check over every \(1 \le d \le m < 200\), and a look at the sizes involved for \(d = 2\):

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

Now the second part of the plan. Lesson 4 proved uniform convergence for a finite class by a union bound over its \(|\mathcal H|\) hypotheses, paying \(\log |\mathcal H|\). The idea here is to pay only for the labellings, \(\log \tau_{\mathcal H}\). Throughout, \(\mathcal D\) is a distribution over examples \(z = (x, y) \in \mathcal X \times \{0, 1\}\), the loss is \(\ell(h, z) = \mathbb 1[h(x) \ne y]\), and we assume \(\mathcal X\) has at least \(2m\) points (true for the score line). The loss \(\ell\) (a script l) is the 0-1 loss as a formula: the indicator of a wrong label. With \(t^\star = 0.6\), the threshold \(h_{0.5}\) on the example \(z = (0.55, 0)\) says junk while the truth says keep, so \(\ell(h_{0.5}, z) = 1\); averaged over \(S\), these losses give \(L_S(h)\).

Two more symbols, both from lesson 2. \(\mathbb E[\cdot]\) is the **expected value**, the long-run average of a random number; with a subscript such as \(\mathbb E_{S \sim \mathcal D^m}\), it says which draw is being averaged over. And \(\sup_{h \in \mathcal H}\), the **supremum**, is the largest value over all rules in the class, or, when no single rule attains it, the value the rules creep up to; for our purposes, read it as “the worst case over the class”. In the uniform-convergence snippet earlier, the `max` over 101 thresholds was a stand-in for a supremum over all of them.

**Theorem (uniform convergence from a small growth function).** For every class \(\mathcal H\), every distribution \(\mathcal D\) and every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H\) satisfies

\[ |L_{\mathcal D}(h) - L_S(h)| \le \frac{4 + 2\sqrt{\log \tau_{\mathcal H}(2m)}}{\delta \sqrt{2m}}. \]

From the inside out. \(\tau_{\mathcal H}(2m)\) is the growth function at twice the training size, the number of labellings that matters; its log, square-rooted, is the price of the class; adding 4 and doubling come from the proof; and the bottom, \(\delta \sqrt{2m}\), shrinks the whole thing as labels grow and inflates it as you demand more confidence. On the left is the gap for one rule, and “every \(h\)” makes it a bound on the worst gap. In plain words: **with probability at least \(1 - \delta\), no rule’s training score is off by more than a quantity that depends on the class only through how many ways it labels \(2m\) points.** For thresholds, \(\tau(2m) = 2m + 1\); with \(m = 200\) and \(\delta = 0.1\):

```python
from math import log, sqrt

m, delta = 200, 0.1
tau = 2 * m + 1
expected = (4 + 2 * sqrt(log(tau))) / sqrt(2 * m)
print(tau, round(expected, 4), round(expected / delta, 4))
```

It prints `401 0.4448 4.4483`. The middle number is the bound before dividing by \(\delta\), and the last is the theorem’s bound: 4.45, more than 1, so at 200 labels it says nothing, since a gap between two risks never exceeds 1. The bound shrinks like \(\sqrt{\log m / m}\), so it does bite eventually; the end of the next section finds where.

The proof bounds the expected worst gap, \(\mathbb E_{S \sim \mathcal D^m}\left[\sup_{h \in \mathcal H} |L_{\mathcal D}(h) - L_S(h)|\right]\), by \((4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/\sqrt{2m}\), the middle number above, and then converts expectation into probability. It is the deepest proof in the lesson, so let’s take it in seven steps, each saying what changes and why.

*Step 1: from expectation to probability.* Let \(Z = \sup_{h} |L_{\mathcal D}(h) - L_S(h)|\), a non-negative random variable. **Markov’s inequality** says that for a non-negative \(Z\) and any \(a > 0\), \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\). (Proof: \(Z \ge a \cdot \mathbb 1[Z \ge a]\) pointwise; take expectations.) With \(a = \mathbb E[Z]/\delta\), the probability that \(Z \ge \mathbb E[Z]/\delta\) is at most \(\delta\). So a bound on \(\mathbb E[Z]\), divided by \(\delta\), holds with probability at least \(1 - \delta\). (If \(\mathbb E[Z] = 0\), then \(Z = 0\) with probability 1 and there is nothing to prove.) It remains to bound \(\mathbb E[Z]\).

Markov’s inequality, read slowly: \(\mathbb P[Z \ge a]\) is the chance that \(Z\) reaches the level \(a\), and \(\mathbb E[Z]/a\) is its average measured in units of \(a\). In plain words: **a non-negative quantity can rarely be many times its own average.** If a gap averages 0.04, it can reach 0.4 at most a tenth of the time. Here is one rule’s gap, for the threshold \(h_{0.5}\) with \(t^\star = 0.6\), whose true risk is 0.1, over 20,000 training sequences of 30 uniform scores:

```python
import random

rng, m, delta = random.Random(0), 30, 0.1
Z = [abs(sum(0.5 < rng.random() <= 0.6 for _ in range(m)) / m
         - 0.1) for _ in range(20_000)]
EZ = sum(Z) / len(Z)
share = sum(z >= EZ / delta for z in Z) / len(Z)
print(round(EZ, 4), round(EZ / delta, 4), share)
```

It prints `0.0427 0.4273 0.0`. The gap averages 0.043, Markov allows it to reach ten times that, 0.43, at most 10% of the time, and in fact it never did. Markov is safe but generous, and that generosity is the \(1/\delta\) in the theorem.

*Step 2: a ghost sample.* The trouble with \(L_{\mathcal D}(h)\) is that it involves the whole distribution. Replace it with something that looks like \(L_S\). Imagine a second, independent training sequence \(S' = (z'_1, \dots, z'_m) \sim \mathcal D^m\), a **ghost sample** that nobody ever collects. Each \(\ell(h, z'_i)\) has mean \(L_{\mathcal D}(h)\), so by linearity of expectation \(L_{\mathcal D}(h) = \mathbb E_{S'}[L_{S'}(h)]\). Then

\[ |L_{\mathcal D}(h) - L_S(h)| = \left| \mathbb E_{S'}\left[ L_{S'}(h) - L_S(h) \right] \right| \le \mathbb E_{S'}\left| L_{S'}(h) - L_S(h) \right| \le \mathbb E_{S'}\left[ \sup_{h' \in \mathcal H} |L_{S'}(h') - L_S(h')| \right]. \]

The first inequality is “the absolute value of an average is at most the average of absolute values”. The second holds because the quantity inside the expectation can only grow when we replace it by its supremum over the class. The right side no longer depends on \(h\), so it also bounds the supremum over \(h\) of the left side.

Read the chain once more as a whole. The left is the gap we care about. The first expression rewrites the true risk as the average ghost score, \(\mathbb E_{S'}\) being the average over all the ghost samples you might draw. The prime in \(h'\) only gives the worst-case rule its own name. In plain words: **the gap between true and training risk is at most the average, over imaginary second samples, of the worst gap between two training scores.** For the threshold \(h_{0.5}\) (true risk 0.1) and one real sample of 30 scores:

```python
import random

rng, m = random.Random(0), 30
wrong = lambda x: 0.5 < x <= 0.6
L = lambda xs: sum(map(wrong, xs)) / m
S = [rng.random() for _ in range(m)]
ghosts = [L([rng.random() for _ in range(m)])
          for _ in range(20_000)]
mean = sum(ghosts) / len(ghosts)
spread = sum(abs(g - L(S)) for g in ghosts) / len(ghosts)
print(L(S), round(mean, 4), round(abs(mean - L(S)), 4),
      round(spread, 4))
```

It prints `0.1 0.1001 0.0001 0.0427`. The ghost scores average 0.1001, the true risk, as linearity promised; this real sample happens to have no gap at all; and the average absolute gap to a ghost, 0.043, is larger, as the first inequality allows.

Taking \(\mathbb E_S\) of both sides, and writing each score as an average of losses,

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

Nothing new here but the notation \(\sum_{i=1}^{m}\), “add up for \(i = 1, 2, \dots, m\)”: \(L_{S'}(h) - L_S(h)\) is the average, over positions \(i\), of the ghost’s loss minus the real loss at that position, and \(\mathbb E_{S, S'}\) averages over both samples.

*Step 3: random signs.* The \(2m\) examples \(z_1, \dots, z_m, z'_1, \dots, z'_m\) are i.i.d. So if we swap \(z_i\) with \(z'_i\) for some indices \(i\), the joint distribution of the \(2m\) examples does not change, and neither does the expectation of any function of them. Swapping at index \(i\) flips the sign of the \(i\)-th term \(\ell(h, z'_i) - \ell(h, z_i)\). So for every sign vector \(\sigma \in \{\pm 1\}^m\),

\[ \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right] = \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

Here \(\sigma\) (sigma) is a list of \(m\) signs, each \(+1\) or \(-1\), and \(\{\pm 1\}^m\) is the set of all \(2^m\) such lists; \(\sigma_i = -1\) means “swap the real and ghost example at position \(i\)”, which turns \(\ell(h, z'_i) - \ell(h, z_i)\) into its negative. In plain words: **because the real and ghost examples are interchangeable, swapping any of them leaves the average worst gap unchanged.** A world small enough to average exactly: scores 0.55, 0.65 or 0.9, equally likely, \(t^\star = 0.6\), two rules (\(t = 0.5\) and \(t = 0.7\)) and \(m = 2\), so there are four sign vectors:

```python
from itertools import product

pts, t_star, m = (0.55, 0.65, 0.9), 0.6, 2
H = [lambda x, t=t: (x > t) != (x > t_star) for t in (0.5, 0.7)]

def value(sigma):
    total = 0
    for z in product(pts, repeat=2 * m):
        S, G = z[:m], z[m:]
        total += max(abs(sum(s * (h(g) - h(x))
                             for s, x, g in zip(sigma, S, G))) / m
                     for h in H)
    return round(total / len(pts) ** (2 * m), 4)

print([value(s) for s in product((1, -1), repeat=m)])
```

It prints `[0.4938, 0.4938, 0.4938, 0.4938]`: the same value for every sign vector. Each `h` returns the loss itself, `max` is the supremum over the two rules, and the loop over all \(3^4\) equally likely draws is \(\mathbb E_{S, S'}\).

The right side has the same value for all \(2^m\) sign vectors, so it also equals their average. Draw \(\sigma\) uniformly, each \(\sigma_i\) an independent fair \(\pm 1\) coin. Averaging over the finitely many \(\sigma\) and swapping the order of the two averages (linearity of expectation),

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\, \mathbb E_{\sigma}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

The only new piece is \(\mathbb E_\sigma\), the average over the fair coin flips, taken inside the average over the two samples; in the small world above it is the average of four equal numbers, 0.4938.

[[diagram:ghost-sample]]

*Step 4: the infinite class becomes finite.* Now fix \(S\) and \(S'\), and let \(C\) be the set of instances appearing in them, at most \(2m\) points. Each loss \(\ell(h, z_i) = \mathbb 1[h(x_i) \ne y_i]\) depends on \(h\) only through its labels on \(C\). So the expression inside the supremum depends on \(h\) only through its labelling in \(\mathcal H_C\), and the supremum over the infinite class is a maximum over at most \(|\mathcal H_C|\) distinct values. How large can \(|\mathcal H_C|\) be? If \(|C| < 2m\), add points of \(\mathcal X\) to reach a set \(C^+\) of exactly \(2m\) points; each labelling of \(C\) is the shortening of some labelling of \(C^+\), so \(|\mathcal H_C| \le |\mathcal H_{C^+}| \le \tau_{\mathcal H}(2m)\). This is where the growth function enters, and it is the heart of the matter: once \(S\) and \(S'\) are fixed, an infinite class acts like a finite one.

*Step 5: Hoeffding and the union bound.* For each labelling, pick one hypothesis \(h\) that produces it, and write

\[ \theta_h = \frac{1}{m} \sum_{i=1}^{m} \sigma_i c_i, \qquad c_i = \ell(h, z'_i) - \ell(h, z_i) \in \{-1, 0, 1\}. \]

Here \(c_i\) is the loss difference at position \(i\): 1 if \(h\) is wrong on the ghost example and right on the real one, \(-1\) the other way round, 0 if it does the same on both. \(\theta_h\) (theta) multiplies each difference by its random sign and averages. In plain words: **\(\theta_h\) is \(h\)’s ghost-minus-real gap after a fair coin has decided, position by position, which example counts as ghost.** If \(m = 4\), \(c = (1, 0, -1, 1)\) and the signs come up \((+1, -1, -1, +1)\), then \(\theta_h = (1 + 0 + 1 + 1)/4 = 0.75\).

With \(S, S'\) fixed, the \(c_i\) are fixed numbers and only the signs are random. The terms \(\sigma_i c_i\) are independent, each has mean 0, and each lies in \([-1, 1]\). **Hoeffding’s inequality** from lesson 4 says that an average of \(m\) independent variables with values in \([a, b]\) is more than \(\rho\) from its mean with probability at most \(2\exp(-2m\rho^2/(b - a)^2)\), where \(\exp(u)\) is another way to write \(e^u\): the more variables and the narrower their range, the less likely a big miss. (Lesson 4 stated it for identically distributed variables, but its proof multiplies one bound per variable and never uses that they share a distribution, so it holds for independent ones.) Here \(b - a = 2\), so

\[ \mathbb P_\sigma\left[ |\theta_h| > \rho \right] \le 2 \exp\left( -\frac{m \rho^2}{2} \right). \]

A union bound over the at most \(|\mathcal H_C|\) chosen hypotheses gives

\[ \mathbb P_\sigma\left[ \max_{h} |\theta_h| > \rho \right] \le 2 |\mathcal H_C| \exp\left( -\frac{m \rho^2}{2} \right) \quad \text{for every } \rho > 0. \]

Read both displays from the inside out. \(\rho\) (rho) is a level you choose, \(|\theta_h| > \rho\) is the event that the signed gap overshoots it, \(\mathbb P_\sigma\) says only the coins are random. Hoeffding’s \(2\exp(-2m\rho^2/(b - a)^2)\) with \(b - a = 2\) becomes \(2\exp(-m\rho^2/2)\). The union bound then pays once per labelling, the factor \(|\mathcal H_C|\), and \(\max_h\) is the largest over the chosen rules. In plain words: **one rule’s signed gap rarely exceeds \(\rho\), and the chance that any of the \(|\mathcal H_C|\) rules does is at most that many times larger.** For thresholds on 400 scores, \(|\mathcal H_C| = 401\), with \(m = 200\), and the worst-spread case where every \(c_i = 1\):

```python
import math
import random

rng, m, n_labellings = random.Random(0), 200, 401
c = [1] * m
theta = lambda: sum(rng.choice((-1, 1)) * ci for ci in c) / m
draws = [abs(theta()) for _ in range(20_000)]
for rho in (0.2, 0.35):
    one = 2 * math.exp(-m * rho ** 2 / 2)
    seen = sum(x > rho for x in draws) / len(draws)
    print(rho, seen, f"{one:.2g}", f"{n_labellings * one:.2g}")
```

It prints `0.2 0.0032 0.037 15`, then `0.35 0.0 9.6e-06 0.0038`. At \(\rho = 0.2\) the observed overshoot rate, 0.3%, sits under Hoeffding’s 3.7%, but the union bound over 401 labellings gives 15, which says nothing. At \(\rho = 0.35\) the union bound is down to 0.4%: paying for 401 labellings costs only a modestly higher level, because the exponential falls so fast.

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
