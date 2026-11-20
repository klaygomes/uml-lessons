# Compression Bounds

## Four messages out of ten thousand

The auditor is back. Compliance sent her at the end of lesson 17, with the rule that every filtering decision must be explainable, and in lesson 18 she read your decision trees. She has been reading your filters ever since. This time she has the tightest-box filter on her desk, the one from lesson 8. It looks at two scores of a message, \(x_1\), the share of capital letters in the subject, and \(x_2\), the share of the body taken up by links, and it calls a message junk when both scores fall inside a box: one interval for capitals, one for links. The learner draws the tightest such box around the junk examples. This quarter it was trained on \(m = 10{,}000\) messages that your users labelled.

She did something nobody on the team had thought of. She deleted the training messages one at a time and retrained after each deletion, to see which ones mattered. Almost none did. In the end she kept exactly four messages: the junk message with the fewest capitals, the one with the most capitals, the one with the fewest links and the one with the most links. She retrained on those four alone, and out came the same box, to the last decimal.

“So your filter is really four messages,” she says. “The other 9,996 played no part in it. Doesn’t that mean they are an exam it never saw? And if it passes that exam, is that a certificate?”

It is a sharp question, and the team has no tool that answers it. Lesson 6 certifies the box through its VC dimension, which talks about the class of all boxes and says nothing about how the learner used its examples. Lesson 13 certifies stable learners. Nothing you own turns “this hypothesis can be rebuilt from a few of its examples” into a statement about its true risk. This lesson builds that tool. It is called a **compression bound**, and the auditor’s instinct turns out to be very nearly right, with one twist that costs a logarithm.

Here is her experiment in Python. The messages are drawn at random, junk exactly when both scores fall inside the true box \([0.3, 0.8] \times [0.2, 0.6]\), the learner draws the tightest box around the junk, and then we keep only four positions and rebuild:

```python
import random
rng = random.Random(30)
a, b = (0.3, 0.2), (0.8, 0.6)
def inside(x, lo, hi):
    return all(l <= v <= u for v, l, u in zip(x, lo, hi))
xs = [(rng.random(), rng.random()) for _ in range(10_000)]
S = [(x, int(inside(x, a, b))) for x in xs]
P = [i for i, (x, y) in enumerate(S, 1) if y == 1]
I = [f(P, key=lambda i: S[i - 1][0][j]) for j in (0, 1)
     for f in (min, max)]
kept = [S[i - 1][0] for i in I]
lo = [min(x[j] for x in kept) for j in (0, 1)]
hi = [max(x[j] for x in kept) for j in (0, 1)]
errs = sum(int(inside(x, lo, hi)) != y for x, y in S)
print(len(S), len(P), I, errs)
```

It prints `10000 2010 [1003, 4750, 3348, 5037] 0`: ten thousand messages, 2,010 of them junk, the four kept positions, and a rebuilt box that makes no mistake on any of the ten thousand.

## Rebuilding a box, example by example

Let’s slow down and do the same thing by hand on a sequence small enough to hold in your head. These are the eight examples from lesson 8, with label 1 for junk and 0 for keep:

| i | capitals \(x_1\) | links \(x_2\) | label |
|---|---|---|---|
| 1 | 0.35 | 0.25 | junk |
| 2 | 0.60 | 0.50 | junk |
| 3 | 0.50 | 0.30 | junk |
| 4 | 0.75 | 0.40 | junk |
| 5 | 0.10 | 0.30 | keep |
| 6 | 0.90 | 0.50 | keep |
| 7 | 0.50 | 0.80 | keep |
| 8 | 0.40 | 0.10 | keep |

Recall how lesson 8 drew the tightest box: for each score, the interval runs from the smallest to the largest value that score takes among the junk examples. That gave \([0.35, 0.75] \times [0.25, 0.5]\). Now ask a different question: which examples actually *set* those four numbers? Before you read on, find them in the table.

Each end of each interval is set by one junk example, so there are four slots to fill.

| slot | what it asks | junk values | winner |
|---|---|---|---|
| 1 | fewest capitals | 0.35, 0.60, 0.50, 0.75 | example 1 (0.35) |
| 2 | most capitals | 0.35, 0.60, 0.50, 0.75 | example 4 (0.75) |
| 3 | fewest links | 0.25, 0.50, 0.30, 0.40 | example 1 (0.25) |
| 4 | most links | 0.25, 0.50, 0.30, 0.40 | example 2 (0.50) |

So the box is pinned by the list of positions \((1, 4, 1, 2)\). Notice that example 1 fills two slots, because it has both the fewest capitals and the fewest links. The list has four entries but only three different examples. Hold on to that; it matters for the theory.

Now **rebuild**: hand the learner only the examples at those positions, \((z_1, z_4, z_1, z_2)\), and let it draw the tightest box around their junk. The smallest capitals among them is 0.35 and the largest 0.75; the smallest links 0.25 and the largest 0.5. Same box. And since it is the same box, it makes the same zero mistakes on all eight examples. The four keep messages, 5 to 8, were never consulted at all.

```python
S = [((0.35, 0.25), 1), ((0.6, 0.5), 1), ((0.5, 0.3), 1),
     ((0.75, 0.4), 1), ((0.1, 0.3), 0), ((0.9, 0.5), 0),
     ((0.5, 0.8), 0), ((0.4, 0.1), 0)]
P = [i for i, (x, y) in enumerate(S, 1) if y == 1]
I = [f(P, key=lambda i: S[i - 1][0][j]) for j in (0, 1)
     for f in (min, max)]
print(I, len(set(I)))
```

It prints `[1, 4, 1, 2] 3`: four slots, three distinct examples.

There are two halves to what just happened, and they deserve names. The first half looked at the whole training sequence and *chose* a short list of positions. The second half looked *only* at the examples at those positions and built a hypothesis. A pair like that, where the hypothesis built from the short list is consistent with the whole sequence, is a **compression scheme**, and the length of the list is its **size**. For boxes in two scores the size is 4. We will make all of this precise soon; first, play with it.

[[diagram:compress-rebuild]]

In the lab below, the dots are training messages. Drag them around, delete them, add new ones, and watch which ones light up as kept. Before you touch anything, predict: if you delete a message that is not lit, can the box change?

[[lab]]

## An exam the box never saw

The auditor’s intuition goes like this. Suppose you build a hypothesis from a handful of examples, and then grade it on a completely separate batch it never touched. The grade on that separate batch is an honest estimate of the true risk, for exactly the reason lesson 2 gave: the batch is a fresh sample from \(\mathcal D\), and the hypothesis was fixed before the batch was drawn.

Let’s state that protocol carefully, because the whole lesson leans on it. First draw \(k\) examples, call this sequence \(T\), and build a hypothesis \(h_T\) from it, by any method you like. Then draw a fresh sequence of further examples, call it \(V\) (for “validation”), and compute the empirical risk of \(h_T\) on \(V\). Recall that the empirical risk \(L_V(h)\) is the fraction of the examples in \(V\) that \(h\) gets wrong, and the true risk \(L_{\mathcal D}(h)\) is the probability that \(h\) mislabels a fresh message drawn from the distribution \(\mathcal D\). The question is how far apart \(L_V(h_T)\) and \(L_{\mathcal D}(h_T)\) can be.

You already own one answer. **Hoeffding’s inequality** from lesson 4 says that the average of \(n\) independent quantities in \([0, 1]\) lies within \(\sqrt{\log(2/\delta)/(2n)}\) of its mean, with probability at least \(1 - \delta\). (Here \(\log\) is the natural logarithm, \(\delta\) is the confidence parameter, the chance we allow ourselves to be wrong, and \(n\) is the number of examples in \(V\).) With \(n = 1{,}000\) and \(\delta = 0.05\) that slack is about 0.043.

Now look at that number next to a good filter. If the true risk is 0.02, a slack of 0.043 is more than twice the thing you are measuring. Hoeffding is wasteful here, and it is wasteful for a reason you can see. It has to cover the worst case, a quantity that is 0 half the time and 1 half the time, which swings as much as any quantity in \([0, 1]\) can. A mistake indicator with rate 0.02 is almost always 0. Its swings are tiny. The count of mistakes in 1,000 messages has mean 20 and, as you will compute in a moment, a spread of only about 4.4 either way. We want a tool that notices how small the swings are.

## Bernstein’s inequality

The tool is called Bernstein’s inequality, and the idea it carries is this: **a sum of small, quiet quantities stays close to its mean, and how close depends on how much the quantities actually vary, not on the worst they could do.**

To measure “how much a quantity varies” we use its **variance**. For a random number \(Z\) with mean 0, the variance is \(\mathbb E[Z^2]\), the average squared distance from the mean. (\(\mathbb E\) is the expectation: the average value, weighted by probability.) Take a single message and let \(Z\) be “1 if the filter errs on it, else 0”, minus the error rate 0.02, so that its mean is 0. It equals \(0.98\) with probability 0.02 and \(-0.02\) with probability 0.98, so

\[ \mathbb E[Z^2] = 0.02 \cdot 0.98^2 + 0.98 \cdot 0.02^2 = 0.02 \cdot 0.98 = 0.0196. \]

Over 1,000 independent messages the variances add, to \(19.6\), and the typical spread of the mistake count is the square root, \(\sqrt{19.6} \approx 4.4\). Compare the worst-case quantity Hoeffding prepares for: variance \(1/4\) per message, 250 in total, spread about 16. Bernstein lets us use the 19.6.

[[diagram:variance]]

**Bernstein’s inequality.** Let \(Z_1, \dots, Z_m\) be independent and identically distributed random variables with mean zero, and suppose \(|Z_i| < M\) with probability 1 for every \(i\). Then for every \(t > 0\),

\[ \mathbb P\left[ \sum_{i=1}^m Z_i > t \right] \le \exp\left( - \frac{t^2 / 2}{\sum_{j=1}^m \mathbb E[Z_j^2] + M t / 3} \right). \]

Read it from the inside out.

- \(Z_i\) is the \(i\)-th random quantity, in our story “mistake on message \(i\), minus the error rate”. Independent and identically distributed (i.i.d.) means each is drawn the same way and none influences another.
- \(|Z_i| < M\) says no single quantity can be larger than \(M\) in size; \(|\cdot|\) is the absolute value here. For mistake indicators minus their mean, \(M = 1\) works.
- \(\sum_{i=1}^m Z_i\) adds the \(m\) quantities. In the story it is “the number of mistakes, minus the expected number”.
- \(t\) is a threshold, how far above its mean we fear the sum might land.
- \(\mathbb P[\dots]\) is the probability of the event inside the brackets: the sum exceeds its mean by more than \(t\).
- In the denominator, \(\sum_j \mathbb E[Z_j^2]\) is the total variance, the 19.6 above, and \(Mt/3\) is a correction for the rare large jump.
- \(\exp(\cdot)\) is \(e\) raised to the given power, so a large fraction inside means a tiny probability.

In plain words: **the chance that a sum of bounded, mean-zero quantities overshoots by \(t\) shrinks like \(e^{-t^2/(2 \cdot \text{variance})}\) as long as \(t\) is small next to the variance divided by \(M\), and like \(e^{-3t/(2M)}\) beyond that.** Push the story through it. With 1,000 messages at error rate 0.02, ask for the chance of 15 or more extra mistakes, that is more than 35 in all: \(t = 15\), total variance 19.6, \(M = 1\), so the exponent is \(-112.5 / (19.6 + 5) \approx -4.57\) and the bound is about 0.0103. Hoeffding’s version of the same question, \(e^{-2t^2/m}\), gives 0.64, which is almost useless. The exact binomial probability is smaller still.

```python
from math import comb, exp
m, L_D, t = 1000, 0.02, 15
var_sum = m * L_D * (1 - L_D)
bernstein = exp(-(t ** 2 / 2) / (var_sum + t / 3))
hoeffding = exp(-2 * t ** 2 / m)
exact = sum(comb(m, j) * L_D ** j * (1 - L_D) ** (m - j)
            for j in range(m + 1) if j - m * L_D > t)
print(round(var_sum, 1), round(bernstein, 4))
print(round(hoeffding, 4), round(exact, 4))
```

It prints `19.6 0.0103` and then `0.6376 0.0007`. In the code, `var_sum` is \(\sum_j \mathbb E[Z_j^2]\), `bernstein` is the right-hand side of the inequality term by term, and `exact` adds up the binomial probabilities of every mistake count more than 15 above 20. Bernstein is within a factor of fifteen of the truth; Hoeffding is off by a factor of nine hundred.

We will use Bernstein’s inequality as a tool, without proof. Its proof follows the same path as Hoeffding’s (bound the exponential moment \(\mathbb E[e^{\lambda Z}]\), then apply Markov’s inequality), but it keeps track of the variance along the way.

## One fixed hypothesis, both directions

Now we turn Bernstein into a statement about risks. Fix one hypothesis \(h\) before seeing any data. Its empirical risk on a random sample is an average of mistake indicators, so it should be close to its true risk, and when the true risk is small the slack should be small too. There are two directions to say this in: the training error is probably not much *above* the true risk, and the true risk is probably not much above the training error. The second is the one we need, because in practice we can see \(L_S(h)\) and want to bound \(L_{\mathcal D}(h)\).

We state it for any loss with values in \([0, 1]\). Recall that \(\mathcal H\) is the hypothesis class, \(\mathcal Z\) the set of all possible examples, and that a loss function \(\ell(h, z)\) scores hypothesis \(h\) on example \(z\); for the 0-1 loss it is 1 on a mistake and 0 otherwise, and then \(L_S(h) = \frac1m \sum_i \ell(h, z_i)\) and \(L_{\mathcal D}(h) = \mathbb E_{z \sim \mathcal D}[\ell(h, z)]\), where \(z \sim \mathcal D\) reads “\(z\) is drawn from \(\mathcal D\)”.

**Lemma (a fixed hypothesis, measured by Bernstein).** Let \(\ell: \mathcal H \times \mathcal Z \to [0, 1]\) be a loss function, \(\mathcal D\) any distribution over examples, and \(h\) a fixed hypothesis. Then for every \(\delta \in (0, 1)\),

\[ \text{(1)} \quad \mathbb P_{S \sim \mathcal D^m}\left[ L_S(h) \ge L_{\mathcal D}(h) + \sqrt{\frac{2 L_{\mathcal D}(h) \log(1/\delta)}{m}} + \frac{2 \log(1/\delta)}{3m} \right] \le \delta, \]

\[ \text{(2)} \quad \mathbb P_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(h) \ge L_S(h) + \sqrt{\frac{2 L_S(h) \log(1/\delta)}{m}} + \frac{4 \log(1/\delta)}{m} \right] \le \delta. \]

Read them from the inside out.

- \(S \sim \mathcal D^m\) under the \(\mathbb P\) says the probability is over a random training sequence of \(m\) examples drawn i.i.d. from \(\mathcal D\).
- \(\log(1/\delta)\) is the price of confidence: at \(\delta = 0.05\) it is about 3.0.
- The square-root term is the variance-sensitive part. It has the risk itself under the root, so it vanishes when the risk is zero.
- The last term, \(\log(1/\delta)/m\) times a constant, is the price of rare large jumps. It shrinks like \(1/m\), much faster than Hoeffding’s \(1/\sqrt m\).
- Part (1) has the true risk inside the root; part (2) has the empirical risk, which you can actually compute.

In plain words, part (2): **with probability at least \(1 - \delta\), the true risk is at most the training error, plus a square root that scales with the training error, plus \(4\log(1/\delta)/m\).** For a hypothesis with true risk 0.02 and \(m = 1{,}000\), \(\delta = 0.05\): if it shows the training error 0.02, the bound allows a true risk up to \(0.02 + \sqrt{2 \cdot 0.02 \cdot 3.0/1000} + 4 \cdot 3.0/1000 \approx 0.02 + 0.011 + 0.012\), a slack of 0.023 against Hoeffding’s 0.043.

The simulation below draws 20,000 training sequences for a hypothesis with true risk 0.02, and counts how often each bad event of the lemma happens. The lemma promises at most 5% each.

```python
import random
from math import log, sqrt
rng, m, delta, L_D = random.Random(1), 1000, 0.05, 0.02
L = log(1 / delta)
bad1 = bad2 = 0
for _ in range(20_000):
    L_S = sum(rng.random() < L_D for _ in range(m)) / m
    bad1 += L_S >= L_D + sqrt(2 * L_D * L / m) + 2 * L / (3 * m)
    bad2 += L_D >= L_S + sqrt(2 * L_S * L / m) + 4 * L / m
print(round(bad1 / 20_000, 4), round(bad2 / 20_000, 4))
print(round(sqrt(2 * 0.02 * L / m) + 4 * L / m, 4))
```

It prints `0.0037 0.0` and `0.0229`. Each line of the loop is one of the two events, symbol for symbol. Both happen far less often than 5%: the constants are safe, not tight.

*Proof of (1).* Define \(\alpha_i = \ell(h, z_i) - L_{\mathcal D}(h)\) for \(i \in [m]\), where \([m]\) is the set of positions \(\{1, \dots, m\}\). Each \(\alpha_i\) is a function of \(z_i\) alone, and the \(z_i\) are i.i.d., so the \(\alpha_i\) are i.i.d. too. Their mean is \(\mathbb E[\ell(h, z_i)] - L_{\mathcal D}(h) = 0\), by the definition of the true risk.

Next, the variance. Expanding the square and using linearity of expectation,

\[ \mathbb E[\alpha_i^2] = \mathbb E[\ell(h, z_i)^2] - 2 L_{\mathcal D}(h)\, \mathbb E[\ell(h, z_i)] + L_{\mathcal D}(h)^2 = \mathbb E[\ell(h, z_i)^2] - L_{\mathcal D}(h)^2 \le \mathbb E[\ell(h, z_i)^2] \le L_{\mathcal D}(h). \]

The middle equality substitutes \(\mathbb E[\ell(h, z_i)] = L_{\mathcal D}(h)\). The first inequality drops a square, which is never negative. The last uses \(\ell \in [0, 1]\): a number between 0 and 1 is at least its own square, so \(\ell^2 \le \ell\), and the expectation of \(\ell\) is \(L_{\mathcal D}(h)\). For a mistake indicator this is the 0.0196 \(\le\) 0.02 of the previous section.

Also \(|\alpha_i| \le 1\), since both \(\ell\) and \(L_{\mathcal D}(h)\) lie in \([0, 1]\). So \(|\alpha_i| < M\) for every \(M > 1\), and Bernstein applies for each such \(M\). The probability on the left does not depend on \(M\), so it is at most the smallest of these bounds, and letting \(M\) shrink to 1 gives the bound with \(M = 1\). Finally, the total variance is at most \(m L_{\mathcal D}(h)\), and a larger denominator only makes the exponent less negative, so replacing the total variance by \(m L_{\mathcal D}(h)\) only loosens the bound:

\[ \mathbb P\left[ \sum_{i=1}^m \alpha_i > t \right] \le \exp\left( - \frac{t^2/2}{m L_{\mathcal D}(h) + t/3} \right). \]

Now choose \(t\) so that the right side equals \(\delta\). Taking logarithms, that means \(\frac{t^2}{2} = \log(1/\delta)\left(m L_{\mathcal D}(h) + \frac t3\right)\), a quadratic equation in \(t\):

\[ t^2 - \tfrac23 \log(1/\delta)\, t - 2 \log(1/\delta)\, m L_{\mathcal D}(h) = 0. \]

Its positive root is

\[ t = \tfrac13 \log(1/\delta) + \sqrt{\tfrac19 \log^2(1/\delta) + 2 \log(1/\delta)\, m L_{\mathcal D}(h)} \le \tfrac23 \log(1/\delta) + \sqrt{2 \log(1/\delta)\, m L_{\mathcal D}(h)}, \]

where the inequality is \(\sqrt{a + b} \le \sqrt a + \sqrt b\) (square both sides: \(a + b \le a + b + 2\sqrt{ab}\)). Call the right side \(t'\).

Divide by \(m\): since \(\frac1m \sum_i \alpha_i = L_S(h) - L_{\mathcal D}(h)\), the event “\(\sum_i \alpha_i \ge t'\)” is exactly the bad event in (1). If \(L_{\mathcal D}(h) = 0\), the loss is 0 with probability 1, so \(L_S(h) = 0\) and the bad event in (1) cannot happen. Otherwise \(t' > t\), because \(\sqrt{a+b} < \sqrt a + \sqrt b\) when both are positive, so \(\{\sum_i \alpha_i \ge t'\}\) is contained in \(\{\sum_i \alpha_i > t\}\), whose probability is at most \(\delta\). That proves (1).

*Proof of (2).* Apply Bernstein to \(-\alpha_1, \dots, -\alpha_m\) instead. They are still i.i.d., with mean zero, the same variances and \(|-\alpha_i| \le 1\), so with the same \(t\), with probability at least \(1 - \delta\) we have \(\sum_i (-\alpha_i) \le t\). Dividing by \(m\) and using the bound on \(t\),

\[ L_{\mathcal D}(h) - L_S(h) \le a + \sqrt{b\, L_{\mathcal D}(h)}, \qquad a = \frac{2\log(1/\delta)}{3m}, \quad b = \frac{2 \log(1/\delta)}{m}. \]

This bounds the true risk by itself, under a square root. To free it, write \(x = \sqrt{L_{\mathcal D}(h)}\) and \(c = L_S(h) + a\). The inequality reads \(x^2 - \sqrt b\, x - c \le 0\). A quadratic with positive leading coefficient is non-positive only between its roots, so \(x\) is at most the larger root:

\[ x \le \frac{\sqrt b + \sqrt{b + 4c}}{2}. \]

Squaring,

\[ L_{\mathcal D}(h) = x^2 \le \frac{2b + 4c + 2\sqrt{b^2 + 4bc}}{4} \le \frac{2b + 4c + 2(b + 2\sqrt{bc})}{4} = b + c + \sqrt{bc}, \]

where the middle step uses \(\sqrt{b^2 + 4bc} \le b + 2\sqrt{bc}\) (square the right side: \(b^2 + 4b\sqrt{bc} + 4bc\), which is larger). Substitute \(c = L_S(h) + a\) and split the root with \(\sqrt{bL_S + ab} \le \sqrt{bL_S} + \sqrt{ab}\):

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{b\, L_S(h)} + \left(a + b + \sqrt{ab}\right). \]

Finally, the constants. With \(\lambda = \log(1/\delta)/m\) we have \(a = \frac23\lambda\), \(b = 2\lambda\) and \(\sqrt{ab} = \frac{2}{\sqrt3}\lambda\), so \(a + b + \sqrt{ab} = \left(\frac23 + 2 + \frac{2}{\sqrt 3}\right)\lambda \approx 3.82\,\lambda < 4\lambda\). And \(\sqrt{b L_S(h)} = \sqrt{2 L_S(h)\log(1/\delta)/m}\). So with probability at least \(1 - \delta\), the true risk is *strictly* less than the right side of (2), which means the bad event of (2) has probability at most \(\delta\). \(\blacksquare\)

The constant is worth checking by machine, together with the whole chain of algebra. The snippet computes \((a + b + \sqrt{ab})/\lambda\), then throws 100,000 random pairs \((L_S, L_{\mathcal D})\) at the starting inequality and records the worst value of \(\left(L_{\mathcal D} - L_S - \sqrt{bL_S}\right)/\lambda\) among the pairs that satisfy it. The proof says that worst value never reaches 3.82.

```python
import random
from math import log, sqrt
m, L = 1000, log(1 / 0.05)
a, b = 2 * L / (3 * m), 2 * L / m
print(round((a + b + sqrt(a * b)) / (L / m), 3))
rng, worst = random.Random(5), 0.0
for _ in range(100_000):
    L_S, L_D = rng.random(), rng.random()
    if L_D - L_S <= a + sqrt(b * L_D):
        gap = L_D - L_S - sqrt(b * L_S)
        worst = max(worst, gap / (L / m))
print(round(worst, 3))
```

It prints `3.821` and `2.074`.

## The hold-out lemma

Back to the protocol: build \(h_T\) from \(k\) examples, then grade it on a fresh, independent sequence \(V\). Part (2) of the lemma was about a hypothesis fixed in advance. Here \(h_T\) is random, since it depends on \(T\), but it is fixed *before* \(V\) is drawn, and that is all we need.

**Lemma (hold-out).** Let the loss take values in \([0, 1]\). Let \(T\) be a sequence of \(k\) examples, \(h_T\) any hypothesis computed from \(T\), and \(V\) a sequence of examples drawn i.i.d. from \(\mathcal D\), independently of \(T\). Then

\[ \mathbb P\left[ L_{\mathcal D}(h_T) - L_V(h_T) \ge \sqrt{\frac{2 L_V(h_T) \log(1/\delta)}{|V|}} + \frac{4 \log(1/\delta)}{|V|} \right] \le \delta. \]

Here \(|V|\) is the number of examples in \(V\), \(L_V(h_T)\) is the fraction of them that \(h_T\) gets wrong, and the probability is over both \(T\) and \(V\). In plain words: **a hypothesis graded on examples it never saw has a true risk at most its grade plus a Bernstein slack sized by that grade and by the number of examples graded.** With \(|V| = 9{,}996\), a grade of 0 and \(\delta = 0.05\), the slack is \(4 \cdot 3.0 / 9{,}996 \approx 0.0012\): if the auditor had chosen the four messages before seeing any data, the box would be certified to within about a tenth of a percent.

*Proof.* Condition on \(T\). Once \(T\) is known, \(h_T\) is one fixed hypothesis. Because \(V\) is independent of \(T\), knowing \(T\) does not change how \(V\) is distributed: given \(T\), \(V\) is still \(|V|\) examples drawn i.i.d. from \(\mathcal D\). So part (2) of the previous lemma applies, with \(V\) in the role of \(S\) and \(|V|\) in the role of \(m\), and the conditional probability of the bad event, given \(T\), is at most \(\delta\). This holds for every possible value of \(T\). The unconditional probability is the average of the conditional ones over \(T\) (the law of total probability), and an average of numbers that are each at most \(\delta\) is at most \(\delta\). \(\blacksquare\)

Here is the protocol as an experiment. Each round draws \(k = 4\) fresh messages, builds the tightest box around their junk (or the empty box, which says keep everywhere, if none of the four is junk), then grades it on 2,000 fresh messages. Because the box built from junk always sits inside the true box, its true risk is exactly the area of the true box, 0.2, minus the area of the built box. The lemma promises the bad event in at most 5% of rounds.

```python
import random
from math import log, sqrt
rng, k, n, delta = random.Random(15), 4, 2000, 0.05
a, b = (0.3, 0.2), (0.8, 0.6)
def inside(x, lo, hi):
    return all(lo[j] <= x[j] <= hi[j] for j in (0, 1))
draw = lambda: (rng.random(), rng.random())
bad = 0
for _ in range(2000):
    P = [x for x in (draw() for _ in range(k)) if inside(x, a, b)]
    lo = [min((x[j] for x in P), default=1) for j in (0, 1)]
    hi = [max((x[j] for x in P), default=0) for j in (0, 1)]
    L_D = 0.2 - max(0, hi[0] - lo[0]) * max(0, hi[1] - lo[1])
    V = [draw() for _ in range(n)]
    L_V = sum(inside(x, a, b) != inside(x, lo, hi) for x in V) / n
    c = log(1 / delta) / n
    bad += L_D - L_V >= sqrt(2 * L_V * c) + 4 * c
print(bad / 2000)
```

It prints `0.0005`: one bad round in two thousand.

## The catch: the four were chosen by looking

So far so good. But reread the auditor’s claim. Her four messages were not chosen in advance. They were chosen *after* looking at all ten thousand: they are the extremes, and you only know which messages are extreme once you have seen them all. The remaining 9,996 are not “an exam drawn after the hypothesis was fixed”. They are “whatever was left over after a choice that looked at everything”.

Does that matter? It can. Imagine a sneakier learner that, for every possible choice of four positions, builds a box, grades it on the other 9,996, and returns whichever box got the best grade. Each individual grade is honest, but the *best* of many honest grades is biased: it is the one that got lucky. This is the same trap as lesson 2’s overfitting and lesson 11’s reused validation set. Picking the winner after looking at the scores spoils the scores.

There is a way out, and it is the oldest trick in the course. First notice that the hold-out argument never cared *how* \(T\) was drawn, only that \(V\) was independent of it. So consider this version of the protocol. Before seeing any data, agree on a list of \(k\) positions \(I = (i_1, \dots, i_k)\), each between 1 and \(m\), repeats allowed. Then draw \(S = (z_1, \dots, z_m)\), set \(T = (z_{i_1}, \dots, z_{i_k})\), and let \(V\) be the examples at the other positions. Since the \(z\)’s are independent, the examples at the chosen positions are independent of the examples at the rest, so the hold-out lemma holds for this fixed \(I\).

The learner, however, does not use a fixed \(I\). It picks one after looking. So we make the hold-out lemma hold for *every* \(I\) at once, with the **union bound** from lesson 2: the probability that at least one of several events happens is at most the sum of their probabilities. How many lists of \(k\) positions are there? Each of the \(k\) entries is one of \(m\) positions, so \(m^k\). For the auditor’s box that is \(10{,}000^4 = 10^{16}\), an absurd number of events. But in the bound it will only appear through its logarithm, and \(\log 10^{16} = 4 \log 10{,}000 \approx 36.8\). That is the price of choosing by looking: about \(k \log m\) where the fixed protocol paid nothing.

```python
from math import log
m, k = 10_000, 4
print(m ** k, round(k * log(m), 2))
```

It prints `10000000000000000 36.84`.

[[diagram:union-over-indices]]

## The compression bound

We are ready for the main result. It is stated for any learner whose output can be rebuilt from \(k\) of its own examples by a fixed **reconstruction function** \(B\): a function that takes a sequence of \(k\) examples and returns a hypothesis from \(\mathcal H\). The learner \(A\) may choose which \(k\) positions to use after looking at all of \(S\); the theorem only asks that its output is \(B\) applied to the examples at *some* \(k\) positions. For a list of positions \(I = (i_1, \dots, i_k)\) we write \(S_I = (z_{i_1}, \dots, z_{i_k})\) for the examples at those positions, and the set of all such lists is written \([m]^k\), “lists of length \(k\) with entries from \([m]\)”.

**Theorem (the compression bound).** Let \(k\) be a positive integer, \(B: \mathcal Z^k \to \mathcal H\) a function from sequences of \(k\) examples to hypotheses, and \(m \ge 2k\). Let the loss take values in \([0, 1]\). Let \(A\) be a learner that, on every training sequence \(S\) of \(m\) examples, returns \(A(S) = B(S_I)\) for some \(I = (i_1, \dots, i_k) \in [m]^k\), and let \(V\) be the examples of \(S\) whose positions are not in \(I\). Then for every \(\delta \in (0,1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ L_{\mathcal D}(A(S)) \le L_V(A(S)) + \sqrt{L_V(A(S)) \, \frac{4k \log(m/\delta)}{m}} + \frac{8k \log(m/\delta)}{m}. \]

Read it from the inside out.

- \(\mathcal Z^k\) is the set of sequences of \(k\) examples, so \(B: \mathcal Z^k \to \mathcal H\) says \(B\) turns any \(k\) examples into a hypothesis. For boxes, \(B\) draws the tightest box around the junk among its \(k\) examples.
- \(A(S)\) is the hypothesis the learner returns. The theorem does not care how it chooses \(I\); it can look at every example.
- \(L_V(A(S))\) is the fraction of the leftover examples, the ones not at a chosen position, that the output gets wrong. This is the auditor’s exam.
- \(k \log(m/\delta)\) is the price tag. Compare it with the hold-out lemma’s \(\log(1/\delta)\): the extra \(k \log m\) pays for the union over the \(m^k\) lists.
- Dividing by \(m\) and not by \(|V|\) costs a factor of 2 and makes the bound simpler; that is where \(m \ge 2k\) comes in.

In plain words: **if a learner’s output can always be rebuilt from \(k\) of its \(m\) examples, its true risk is at most its error on the examples it did not keep, plus a slack of order \(k \log(m/\delta)/m\).** For the auditor: \(k = 4\), \(m = 10{,}000\), \(\delta = 0.05\), and the leftover error is 0, so the bound is \(8 \cdot 4 \cdot \log(200{,}000)/10{,}000 \approx 32 \cdot 12.2/10{,}000 \approx 0.039\). Had the box made 1% mistakes on the leftovers, the bound would be 0.063.

```python
from math import log, sqrt
def bound(L_V, k, m, delta):
    c = k * log(m / delta) / m
    return L_V + sqrt(L_V * 4 * c) + 8 * c
for L_V in (0.0, 0.01):
    print(L_V, round(bound(L_V, 4, 10_000, 0.05), 4))
```

It prints `0.0 0.0391` and `0.01 0.063`. The function `bound` is the right side term by term, with `c` standing for \(k\log(m/\delta)/m\).

*Proof.* For every list \(I \in [m]^k\), let \(h_I = B(S_I)\), the hypothesis rebuilt from the examples at the positions in \(I\), and let \(V_I\) be the examples at the positions not in \(I\). Let \(n = m - k\).

*Step 1: one fixed list.* Fix \(I\). The hypothesis \(h_I\) depends only on the examples at the positions in \(I\), and \(V_I\) consists of the examples at the other positions. Because the \(m\) examples of \(S\) are drawn independently, these two groups are independent, and \(V_I\) is a sequence of i.i.d. draws from \(\mathcal D\). So the hold-out lemma applies with \(T = S_I\) and \(V = V_I\). Since \(I\) may contain repeats, \(|V_I|\) can be larger than \(n\), but never smaller: \(|V_I| \ge m - k = n\). Both terms of the slack get smaller as \(|V|\) grows, so the slack computed with \(n\) in place of \(|V_I|\) is at least as large. A larger threshold makes the bad event smaller, so the event

\[ E_I = \left\{ L_{\mathcal D}(h_I) - L_{V_I}(h_I) \ge \sqrt{\frac{2 L_{V_I}(h_I) \log(1/\delta)}{n}} + \frac{4 \log(1/\delta)}{n} \right\} \]

is contained in the hold-out lemma’s bad event, and \(\mathbb P[E_I] \le \delta\).

*Step 2: all lists at once.* By the union bound over the \(m^k\) lists,

\[ \mathbb P\left[ \exists I \in [m]^k : E_I \right] \le \sum_{I \in [m]^k} \mathbb P[E_I] \le m^k \delta. \]

Here \(\exists I\) reads “there exists a list \(I\)”, and the sum runs over every list in \([m]^k\).

*Step 3: the learner’s list is one of them.* Whatever \(S\) is, the learner returns \(A(S) = h_{I}\) for the particular list \(I\) it chose, and then \(V = V_I\). If the bound fails for \(A(S)\), then \(E_I\) happens for that \(I\), so “some \(E_I\) happens”. The learner’s choice can depend on \(S\) in any way at all; it cannot escape an event that covers every choice. So with probability at least \(1 - m^k \delta\), for the learner’s own list,

\[ L_{\mathcal D}(A(S)) \le L_V(A(S)) + \sqrt{\frac{2 L_V(A(S)) \log(1/\delta)}{n}} + \frac{4 \log(1/\delta)}{n}. \]

*Step 4: rename the confidence.* Let \(\delta' = m^k \delta\), the failure probability we actually end up with, and assume \(\delta' < 1\). Then \(\log(1/\delta) = \log(m^k / \delta') = k \log m + \log(1/\delta')\). Since \(k \ge 1\) and \(\log(1/\delta') > 0\), we have \(\log(1/\delta') \le k \log(1/\delta')\), so

\[ \log(1/\delta) \le k \log m + k \log(1/\delta') = k \log(m/\delta'). \]

*Step 5: from \(n\) to \(m\).* The assumption \(k \le m/2\) gives \(n = m - k \ge m/2\), so \(1/n \le 2/m\). Substituting both facts,

\[ \sqrt{\frac{2 L_V \log(1/\delta)}{n}} \le \sqrt{L_V \cdot \frac{4 k \log(m/\delta')}{m}}, \qquad \frac{4 \log(1/\delta)}{n} \le \frac{8 k \log(m/\delta')}{m}. \]

So with probability at least \(1 - \delta'\) the bound of the theorem holds with \(\delta'\) in place of \(\delta\). Since \(\delta'\) was an arbitrary number in \((0, 1)\), renaming it \(\delta\) proves the theorem. \(\blacksquare\)

When the rebuilt hypothesis makes no mistake on the leftovers, the first two terms vanish, and we get the clean form that answers the auditor.

**Corollary (realizable form).** Under the conditions of the theorem, if also \(L_V(A(S)) = 0\), then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ L_{\mathcal D}(A(S)) \le \frac{8 k \log(m/\delta)}{m}. \]

In plain words: **a learner whose output is rebuilt from \(k\) of its \(m\) examples and makes no mistake on the rest has true risk at most \(8k\log(m/\delta)/m\).** For the auditor’s box, that is the 0.039 above. For a smaller run of \(m = 2{,}000\) labels it is about 0.17. How close to the truth is it? The next snippet trains the tightest box on 1,000 random training sequences of 2,000 messages each, computes each box’s exact true risk (true box area minus learned box area, as before), and compares the worst one with the bound.

```python
import random
from math import log
rng, m, k, delta = random.Random(7), 2000, 4, 0.05
a, b = (0.3, 0.2), (0.8, 0.6)
area = lambda lo, hi: (hi[0] - lo[0]) * (hi[1] - lo[1])
risks = []
for _ in range(1000):
    xs = [(rng.random(), rng.random()) for _ in range(m)]
    P = [x for x in xs
         if all(a[j] <= x[j] <= b[j] for j in (0, 1))]
    lo = [min(x[j] for x in P) for j in (0, 1)]
    hi = [max(x[j] for x in P) for j in (0, 1)]
    risks.append(area(a, b) - area(lo, hi))
eps = 8 * k * log(m / delta) / m
print(round(eps, 4), round(max(risks), 4))
print(sum(r > eps for r in risks) / len(risks))
```

It prints `0.1695 0.0062` and `0.0`. The bound holds every time, and it is loose: the worst of a thousand boxes had true risk 0.0062, nearly thirty times below the bound. The compression bound knows nothing about boxes except the number 4. It trades sharpness for generality, and it is the generality we came for.

## Compression schemes, defined

The theorem needs a learner of the form \(A(S) = B(S_I)\) whose rebuilt hypothesis is perfect on the leftovers. A compression scheme is the guarantee that such a learner exists for a whole class, whatever the training sequence. It has the two halves we met in the worked example: a **compression function**, written \(\kappa\) (the Greek letter kappa, for “keep”), that reads all \(m\) examples and returns \(k\) positions, and the reconstruction function \(B\), which reads the examples at those positions and returns a hypothesis.

Recall from lesson 3 that a training sequence is **realizable** by \(\mathcal H\) when some \(h \in \mathcal H\) labels every example in it correctly: \(y_i = h(x_i)\) for all \(i\).

**Definition (compression scheme).** Let \(\mathcal H\) be a class of functions from \(\mathcal X\) to \(\mathcal Y\) (the set of messages and the set of labels), and \(k\) a positive integer. \(\mathcal H\) has a compression scheme of size \(k\) if for every \(m\) there are functions \(\kappa: \mathcal Z^m \to [m]^k\) and \(B: \mathcal Z^k \to \mathcal H\) such that, for every \(h \in \mathcal H\) and every training sequence of the form \(S = ((x_1, h(x_1)), \dots, (x_m, h(x_m)))\), if \(I = \kappa(S)\) then the hypothesis \(h' = B(S_I)\) satisfies \(L_S(h') = 0\).

Read it piece by piece.

- \(\kappa: \mathcal Z^m \to [m]^k\) turns a whole training sequence into a list of \(k\) positions. For boxes in two scores, it returns the positions of the four extreme junk examples.
- \(B: \mathcal Z^k \to \mathcal H\) turns the kept examples into a hypothesis; for boxes, the tightest box around their junk.
- “for every \(h \in \mathcal H\) and every training sequence of the form \((x_i, h(x_i))\)” says: whatever the true hypothesis in the class is, and whatever messages arrive, as long as the labels are realizable by the class.
- \(L_S(h') = 0\) says the rebuilt hypothesis is consistent with the *whole* sequence, the kept examples and the leftovers alike.

In plain words: **a class has a compression scheme of size \(k\) when every realizable training sequence can be boiled down to \(k\) of its examples from which a hypothesis consistent with the whole sequence can be rebuilt.** On the eight examples, \(\kappa\) returned \((1, 4, 1, 2)\), \(B\) drew \([0.35, 0.75] \times [0.25, 0.5]\), and it made 0 mistakes on all eight.

Two remarks. First, \(L_S(h') = 0\) means the scheme is an ERM learner in the realizable case (an empirical risk minimizer, lesson 2’s learner that returns a hypothesis with the smallest training error), and it also means \(L_V(h') = 0\) on the leftovers, so the corollary applies straight away: **a class with a compression scheme of size \(k\) is learnable, with true risk at most \(8k\log(m/\delta)/m\).** Second, the definition is a promise about every \(h\) and every sequence, not about one lucky run. The snippet below tests the box scheme in three scores (\(k = 6\)) against 300 random true boxes, each with 500 random messages, and counts the runs where the rebuilt box makes any mistake.

```python
import random
rng, d, failures = random.Random(16), 3, 0
for _ in range(300):
    lo = [rng.uniform(0, 0.5) for _ in range(d)]
    hi = [l + rng.uniform(0, 0.5) for l in lo]
    h = lambda x, lo, hi: int(all(a <= v <= b
                                  for v, a, b in zip(x, lo, hi)))
    xs = [[rng.random() for _ in range(d)] for _ in range(500)]
    S = [(x, h(x, lo, hi)) for x in xs]
    P = [i for i, (x, y) in enumerate(S) if y == 1] or [0]
    I = [f(P, key=lambda i: S[i][0][j]) for j in range(d)
         for f in (min, max)]
    T = [S[i][0] for i in I if S[i][1] == 1]
    lo2 = [min((x[j] for x in T), default=1) for j in range(d)]
    hi2 = [max((x[j] for x in T), default=0) for j in range(d)]
    failures += any(h(x, lo2, hi2) != y for x, y in S)
print(len(I), failures)
```

It prints `6 0`: six positions kept, no failures. (When a sequence has no junk at all, `P` falls back to position 0, the kept example is a keep message, and the rebuilt “box” is empty, so it says keep everywhere, which is consistent.)

## When no hypothesis is perfect

Real mail is not realizable. Users mislabel, spammers imitate keep messages, and no box gets everything right. In that **agnostic** setting (lesson 3’s word for “no promise that some \(h \in \mathcal H\) is perfect”), what should a compression scheme promise? Not zero mistakes; that may be impossible. The natural promise is: **the rebuilt hypothesis does at least as well on \(S\) as any hypothesis in the class.**

**Definition (compression scheme for unrealizable sequences).** \(\mathcal H\) has a compression scheme of size \(k\) for unrealizable sequences if for every \(m\) there are \(\kappa: \mathcal Z^m \to [m]^k\) and \(B: \mathcal Z^k \to \mathcal H\) such that for every training sequence \(S = ((x_1, y_1), \dots, (x_m, y_m))\), with any labels at all, the hypothesis \(h' = B(S_{\kappa(S)})\) satisfies \(L_S(h') \le L_S(h)\) for every \(h \in \mathcal H\).

The only change is the last line: \(L_S(h') = 0\) became \(L_S(h') \le L_S(h)\) for all \(h\), which says that \(h'\) is an ERM hypothesis, one with the smallest empirical risk in the class. In plain words: **every training sequence, realizable or not, can be boiled down to \(k\) examples from which an empirical risk minimizer can be rebuilt.** For example, if the best box in the class makes 13 mistakes on \(S\), the rebuilt box makes at most 13.

Do we need a new scheme for every class? No. For binary labels, a realizable scheme can be upgraded for free.

**Lemma (from realizable to unrealizable).** Let \(\mathcal H\) be a class for binary classification with a compression scheme of size \(k\) in the realizable case. Then it has a compression scheme of size \(k\) for unrealizable sequences.

*Proof.* Build the new compression function in three moves. Given any \(S\), first find an ERM hypothesis \(h \in \mathcal H\), one with \(L_S(h) \le L_S(g)\) for every \(g \in \mathcal H\). Second, discard every example on which \(h\) errs; call what remains \(S'\). Every example in \(S'\) carries the label \(h\) gives it, so \(S'\) is realizable by \(h \in \mathcal H\), and the realizable scheme applies to it. Third, run the realizable compression function on \(S'\) and translate the \(k\) positions it returns into positions in \(S\). The reconstruction function is the realizable one, unchanged.

Now check the promise. The rebuilt hypothesis \(h'\) is consistent with \(S'\), so it can only err on the discarded examples, and there are exactly \(m L_S(h)\) of those. So \(L_S(h') \le L_S(h) \le L_S(g)\) for every \(g \in \mathcal H\). \(\blacksquare\)

The compression bound needs no change for this case: it already allows \(L_V(A(S)) > 0\). That is the agnostic form of the guarantee, and the corollary is its realizable special case.

Here is the upgrade on the simplest class there is, thresholds on one score: \(h_\theta(x) = 1\) when \(x \ge \theta\), “junk when the capitals share is at least \(\theta\)”. Its realizable scheme has size 1: keep the junk example with the smallest score, and rebuild \(\theta\) as that score. The snippet draws 200 messages whose true threshold is 0.6, flips 10% of the labels, finds an ERM threshold by trying every score, discards its mistakes, keeps one example, and rebuilds.

```python
import random
rng, m = random.Random(8), 200
xs = [rng.random() for _ in range(m)]
S = [(x, int(x >= 0.6) ^ (rng.random() < 0.1)) for x in xs]
errors = lambda th: sum(int(x >= th) != y for x, y in S)
erm = min(sorted(xs) + [2.0], key=errors)
clean = [i for i, (x, y) in enumerate(S, 1)
         if int(x >= erm) == y]
junk = [i for i in clean if S[i - 1][1] == 1]
i1 = min(junk, key=lambda i: S[i - 1][0])
th = S[i1 - 1][0]
print(errors(erm), len(clean), i1, errors(th))
```

It prints `13 187 186 13`: the ERM threshold makes 13 mistakes, 187 examples survive the discard, the scheme keeps example 186, and the threshold rebuilt from that single example also makes 13 mistakes.

[[diagram:agnostic-upgrade]]

## Boxes: four numbers, four examples

Now the examples. Because of the upgrade lemma, it is enough to give a realizable scheme for each class, and that is what we do.

**Axis aligned rectangles in \(\mathbb R^d\)** (boxes with one interval per score, \(d\) scores) form an uncountably infinite class: there is a box for every choice of \(2d\) real endpoints. Still, there is a simple scheme of size \(k = 2d\).

- \(\kappa\): for each score \(j = 1, \dots, d\), keep the position of a junk example with the smallest \(j\)-th score and one with the largest. If there is no junk example, keep any \(2d\) positions.
- \(B\): draw the tightest box around the junk examples among the kept ones; if none of them is junk, return the hypothesis that says keep everywhere.

*Why it works.* If \(S\) has junk, the kept junk examples include an extreme in both directions for every score, so the box rebuilt from them has exactly the same \(2d\) endpoints as the tightest box around *all* the junk. Lesson 8 proved that the tightest box is consistent with any realizable sequence. If \(S\) has no junk, every example is keep and the all-keep hypothesis is consistent. Either way \(L_S(B(S_{\kappa(S)})) = 0\). \(\blacksquare\)

That is the auditor’s four: \(d = 2\) scores, \(k = 2d = 4\).

## Halfspaces: the shortest vector in the hull

The second family is the halfspaces of lesson 9. Here we switch to that lesson’s labels, \(y = +1\) for junk and \(-1\) for keep. A message is a vector \(\mathbf x \in \mathbb R^d\), and a **homogeneous halfspace** is the hypothesis \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\) for a weight vector \(\mathbf w \in \mathbb R^d\), where \(\operatorname{sign}\) returns \(+1\) for a positive number and \(-1\) otherwise, and \(\langle \mathbf w, \mathbf x \rangle = \sum_j w_j x_j\) is the inner product and “homogeneous” means the boundary passes through the origin. A training sequence is realizable when some \(\mathbf w^\star\) has \(y_i \langle \mathbf w^\star, \mathbf x_i \rangle > 0\) for every \(i\). We will find a scheme of size \(d\): in two dimensions, two examples are enough to rebuild a consistent halfspace, however many there are.

**One trick first: fold the labels in.** Since \(y_i \langle \mathbf w, \mathbf x_i \rangle = \langle \mathbf w, y_i \mathbf x_i \rangle\), the halfspace gets example \(i\) right exactly when \(\langle \mathbf w, \mathbf u_i \rangle > 0\), where \(\mathbf u_i = y_i \mathbf x_i\) is the message flipped through the origin if it is keep. So we may as well pretend every label is \(+1\) and look for a \(\mathbf w\) that has a positive inner product with every \(\mathbf u_i\).

**The convex hull.** Recall from lesson 26: a **convex combination** of the points \(\mathbf u_1, \dots, \mathbf u_m\) is \(\sum_i \lambda_i \mathbf u_i\) with weights \(\lambda_i \ge 0\) that add up to 1, a weighted average. Their **convex hull**, written \(\operatorname{conv}\{\mathbf u_1, \dots, \mathbf u_m\}\), is the set of all such averages: the smallest convex region containing the points, the shape a rubber band would make around them in the plane.

**The scheme.** \(\kappa\) finds the point \(\mathbf w\) of the convex hull with the smallest norm (the point of the hull closest to the origin; recall that \(\|\mathbf w\| = \sqrt{\langle \mathbf w, \mathbf w \rangle}\) is the length of \(\mathbf w\)), writes it as a convex combination of at most \(d\) of the \(\mathbf u_i\), and keeps their positions. \(B\) takes the kept examples, forms their \(\mathbf u\)’s, and returns the point of smallest norm in *their* convex hull.

Here is a small instance before the proof. Six messages in two scores, three junk and three keep:

| message \(\mathbf x\) | label | \(\mathbf u = y\mathbf x\) |
|---|---|---|
| (1, 2) | +1 | (1, 2) |
| (3, 1) | +1 | (3, 1) |
| (2, 2) | +1 | (2, 2) |
| (−1, −0.5) | −1 | (1, 0.5) |
| (−2, 1) | −1 | (2, −1) |
| (0.5, −2) | −1 | (−0.5, 2) |

Where would you guess the point of the hull nearest the origin lies? The candidates closest to the origin are \((1, 0.5)\), at distance 1.12, and the edge from \((2, -1)\) to \((-0.5, 2)\), which passes even closer. The nearest point turns out to be on that edge: \(\mathbf w = (42/61, 35/61) \approx (0.689, 0.574)\), at distance 0.90. So \(\kappa\) keeps the two keep messages \((-2, 1)\) and \((0.5, -2)\), and \(B\), seeing only those two, finds the same \(\mathbf w\).

To compute a nearest point in Python we use a classic iterative method (Gilbert’s algorithm, a cousin of the Frank and Wolfe method): start at any hull point, find the data point most opposed to the current \(\mathbf w\), that is, with the smallest \(\langle \mathbf w, \mathbf u \rangle\), and move to the nearest point of the segment between them; repeat. Each step stays inside the hull and never increases the norm.

```python
def dot(u, v): return sum(a * b for a, b in zip(u, v))
def min_norm(U, steps=20_000):
    w = list(U[0])
    for _ in range(steps):
        p = min(U, key=lambda u: dot(w, u))
        d = [wi - pi for wi, pi in zip(w, p)]
        if dot(d, d) == 0: break
        a = min(1.0, max(0.0, dot(w, d) / dot(d, d)))
        w = [wi - a * di for wi, di in zip(w, d)]
    return w
S = [((1, 2), 1), ((3, 1), 1), ((2, 2), 1),
     ((-1, -0.5), -1), ((-2, 1), -1), ((0.5, -2), -1)]
U = [tuple(y * xi for xi in x) for x, y in S]
w = min_norm(U)
face = [u for u in U if dot(w, u) - dot(w, w) < 1e-6]
print([round(c, 3) for c in w], face)
w2 = min_norm(face)
print([round(c, 3) for c in w2], min(dot(w2, u) for u in U) > 0)
```

It prints `[0.689, 0.574] [(2, -1), (-0.5, 2)]` and then `[0.689, 0.574] True`: the nearest point, the two points it rests on, the same point rebuilt from those two alone, and a check that it has a positive inner product with all six \(\mathbf u_i\), so it labels all six messages correctly. Now let’s prove that this always works, in five steps.

[[diagram:min-norm]]

*Step 1: the origin is not in the hull.* By realizability, \(\langle \mathbf w^\star, \mathbf u_i \rangle > 0\) for every \(i\). Then for any convex combination, \(\langle \mathbf w^\star, \sum_i \lambda_i \mathbf u_i \rangle = \sum_i \lambda_i \langle \mathbf w^\star, \mathbf u_i \rangle > 0\), because the weights are non-negative and at least one is positive. The origin has inner product 0 with everything, so it is not a convex combination.

*Step 2: the nearest point exists and is unique.* The hull of finitely many points is closed and bounded, and the norm is continuous, so the norm attains a minimum on it; call a minimizer \(\mathbf w\). By step 1, \(\mathbf w \ne \mathbf 0\). Suppose \(\mathbf v \ne \mathbf w\) were another minimizer, with \(\|\mathbf v\| = \|\mathbf w\| = r\). Their midpoint is in the hull (the hull is convex), and the parallelogram identity (expand both squared norms on the right to check it) gives \(\left\|\frac{\mathbf v + \mathbf w}{2}\right\|^2 = \frac{\|\mathbf v\|^2 + \|\mathbf w\|^2}{2} - \frac{\|\mathbf v - \mathbf w\|^2}{4} < r^2\), a point nearer than the nearest. So the minimizer is unique. (In fact \(\mathbf w\) points in the direction of the widest-margin separator of lesson 15, but we will not need that.)

*Step 3: \(\mathbf w\) separates the data.* Suppose, for contradiction, that \(\langle \mathbf w, \mathbf u_i \rangle \le 0\) for some \(i\). Walk from \(\mathbf w\) a little way toward \(\mathbf u_i\): let \(\mathbf w' = (1 - \alpha)\mathbf w + \alpha \mathbf u_i\) with

\[ \alpha = \frac{\|\mathbf w\|^2}{\|\mathbf u_i\|^2 + \|\mathbf w\|^2} \in (0, 1). \]

(\(\mathbf u_i \ne \mathbf 0\), since it is in the hull and the origin is not.) \(\mathbf w'\) is on the segment between two hull points, so it is in the hull. Expand its squared norm:

\[ \|\mathbf w'\|^2 = (1-\alpha)^2 \|\mathbf w\|^2 + \alpha^2 \|\mathbf u_i\|^2 + 2\alpha(1-\alpha)\langle \mathbf w, \mathbf u_i \rangle \le (1-\alpha)^2 \|\mathbf w\|^2 + \alpha^2 \|\mathbf u_i\|^2. \]

The inequality drops the cross term, which is \(\le 0\) because \(\alpha(1-\alpha) > 0\) and \(\langle \mathbf w, \mathbf u_i \rangle \le 0\). Now substitute \(\alpha\) and \(1 - \alpha = \|\mathbf u_i\|^2 / (\|\mathbf u_i\|^2 + \|\mathbf w\|^2)\):

\[ (1-\alpha)^2 \|\mathbf w\|^2 + \alpha^2 \|\mathbf u_i\|^2 = \frac{\|\mathbf u_i\|^4 \|\mathbf w\|^2 + \|\mathbf w\|^4 \|\mathbf u_i\|^2}{(\|\mathbf w\|^2 + \|\mathbf u_i\|^2)^2} = \frac{\|\mathbf u_i\|^2 \|\mathbf w\|^2}{\|\mathbf w\|^2 + \|\mathbf u_i\|^2} = \|\mathbf w\|^2 \cdot \frac{1}{\|\mathbf w\|^2/\|\mathbf u_i\|^2 + 1} < \|\mathbf w\|^2. \]

The second equality factors \(\|\mathbf u_i\|^2\|\mathbf w\|^2\) out of the numerator and cancels one copy of the denominator; the last step holds because \(\|\mathbf w\|^2 / \|\mathbf u_i\|^2 > 0\). So \(\mathbf w'\) is a hull point strictly nearer than \(\mathbf w\), a contradiction. Hence \(\langle \mathbf w, \mathbf u_i \rangle > 0\) for every \(i\): **the nearest point of the hull is a consistent halfspace**, an ERM hypothesis in the realizable case.

*Step 4: \(\mathbf w\) rests on at most \(d\) points.* Step 3 can be sharpened. The same walk toward any hull point \(\mathbf p\) shows that \(\langle \mathbf w, \mathbf p \rangle \ge \|\mathbf w\|^2\) for every \(\mathbf p\) in the hull. Indeed, for \(\alpha \in (0, 1]\) the point \(\mathbf w + \alpha(\mathbf p - \mathbf w)\) is in the hull, so its squared norm is at least \(\|\mathbf w\|^2\):

\[ \|\mathbf w\|^2 + 2\alpha \langle \mathbf w, \mathbf p - \mathbf w \rangle + \alpha^2 \|\mathbf p - \mathbf w\|^2 \ge \|\mathbf w\|^2. \]

Cancel \(\|\mathbf w\|^2\), divide by \(\alpha > 0\), and let \(\alpha\) shrink to 0: \(\langle \mathbf w, \mathbf p - \mathbf w \rangle \ge 0\), that is, \(\langle \mathbf w, \mathbf p \rangle \ge \|\mathbf w\|^2\). Call a **flat** any line, plane, or higher-dimensional analogue, not necessarily through the origin. In plain words: **every point of the hull lies on or beyond the flat through \(\mathbf w\) perpendicular to \(\mathbf w\), on the side away from the origin.** In the example, the inner products minus \(\|\mathbf w\|^2\) are 1.033, 1.836, 1.721, 0.172, 0 and 0: never negative, and zero exactly on the two kept points.

```python
from fractions import Fraction as Fr
U = [(1, 2), (3, 1), (2, 2), (1, 0.5), (2, -1), (-0.5, 2)]
w = (Fr(42, 61), Fr(35, 61))
ww = w[0] ** 2 + w[1] ** 2
gaps = [w[0] * Fr(a) + w[1] * Fr(b) - ww for a, b in U]
print([round(float(g), 3) for g in gaps])
lam = Fr(29, 61)
mix = [lam * Fr(p) + (1 - lam) * Fr(q)
       for p, q in zip(U[4], U[5])]
print(mix == list(w))
```

It prints `[1.033, 1.836, 1.721, 0.172, 0.0, 0.0]` and `True`, the second line confirming in exact fractions that \(\mathbf w = \frac{29}{61}(2, -1) + \frac{32}{61}(-0.5, 2)\).

Now write \(\mathbf w = \sum_i \lambda_i \mathbf u_i\) as a convex combination. Then \(\|\mathbf w\|^2 = \langle \mathbf w, \mathbf w \rangle = \sum_i \lambda_i \langle \mathbf w, \mathbf u_i \rangle\). Each \(\langle \mathbf w, \mathbf u_i \rangle\) is at least \(\|\mathbf w\|^2\), and the weights add up to 1, so the sum is at least \(\|\mathbf w\|^2\), with equality only if \(\langle \mathbf w, \mathbf u_i \rangle = \|\mathbf w\|^2\) for every \(i\) with \(\lambda_i > 0\). Since the sum *is* \(\|\mathbf w\|^2\), equality holds. So \(\mathbf w\) is a convex combination of points that all lie on the flat \(F = \{\mathbf p : \langle \mathbf w, \mathbf p \rangle = \|\mathbf w\|^2\}\). (In set-builder notation, the braces collect the points \(\mathbf p\) such that, read the colon as “such that”, their inner product with \(\mathbf w\) is \(\|\mathbf w\|^2\).) That flat is a line in the plane and, in general, a hyperplane of dimension \(d - 1\). This is the precise meaning of “\(\mathbf w\) lies on a face of the hull”.

To get down to \(d\) points we need one classical fact.

**Carathéodory’s theorem.** If a point is a convex combination of finitely many points of a flat of dimension \(D\), it is a convex combination of at most \(D + 1\) of them.

The idea is easy to see in the plane, \(D = 2\). The centre \((1, 1)\) of the square with corners \((0,0), (2,0), (0,2), (2,2)\) is the average of all four corners, with weights \(\frac14\) each. But it is also the midpoint of the diagonal from \((2,0)\) to \((0,2)\): two points suffice, well within \(D + 1 = 3\). The proof finds such a shortcut mechanically.

*Proof.* Let \(\mathbf w = \sum_{j=1}^q \lambda_j \mathbf p_j\) with all \(\lambda_j > 0\) (drop any zero weights), \(\sum_j \lambda_j = 1\), and \(q \ge D + 2\). The \(q - 1 \ge D + 1\) difference vectors \(\mathbf p_2 - \mathbf p_1, \dots, \mathbf p_q - \mathbf p_1\) live in the \(D\)-dimensional space of directions of the flat, so they are linearly dependent: \(\sum_{j \ge 2} c_j (\mathbf p_j - \mathbf p_1) = \mathbf 0\) with the \(c_j\) not all zero. Set \(\mu_j = c_j\) for \(j \ge 2\) and \(\mu_1 = -\sum_{j \ge 2} c_j\). Then \(\sum_j \mu_j \mathbf p_j = \mathbf 0\) and \(\sum_j \mu_j = 0\), and since the \(\mu_j\) are not all zero but add up to zero, some \(\mu_j > 0\). Let \(\tau = \min\{\lambda_j / \mu_j : \mu_j > 0\}\) and \(\lambda'_j = \lambda_j - \tau \mu_j\). Then every \(\lambda'_j \ge 0\) (by the choice of \(\tau\)), \(\sum_j \lambda'_j = 1 - \tau \cdot 0 = 1\), \(\sum_j \lambda'_j \mathbf p_j = \mathbf w - \tau \cdot \mathbf 0 = \mathbf w\), and at least one \(\lambda'_j\) is 0. Drop it. Repeat until at most \(D + 1\) points remain. \(\blacksquare\)

Here is one step of the proof on the square. The dependence \(\mathbf p_1 - \mathbf p_2 - \mathbf p_3 + \mathbf p_4 = \mathbf 0\) gives \(\mu = (1, -1, -1, 1)\), the step size is \(\tau = \frac14\), and two weights drop to zero at once:

```python
from fractions import Fraction as Fr
P = [(0, 0), (2, 0), (0, 2), (2, 2)]
lam = [Fr(1, 4)] * 4
mu = [1, -1, -1, 1]
print([sum(m * p[i] for m, p in zip(mu, P)) for i in (0, 1)],
      sum(mu))
t = min(l / m for l, m in zip(lam, mu) if m > 0)
lam2 = [l - t * m for l, m in zip(lam, mu)]
w = [sum(l * p[i] for l, p in zip(lam2, P)) for i in (0, 1)]
print([str(l) for l in lam2], [str(c) for c in w])
```

It prints `[0, 0] 0`, confirming \(\sum_j \mu_j \mathbf p_j = \mathbf 0\) and \(\sum_j \mu_j = 0\), and then `['0', '1/2', '1/2', '0'] ['1', '1']`: the same point, now from two corners.

[[diagram:caratheodory]]

Applied to the points on the flat \(F\), which has dimension \(D = d - 1\), Carathéodory says \(\mathbf w\) is a convex combination of at most \(d\) of the \(\mathbf u_i\). Those are the positions \(\kappa\) keeps. (If there are fewer than \(d\), repeat one; repeating a point does not change a hull.) Applied in the whole space, \(D = d\), it would give \(d + 1\); the face is what saves one.

*Step 5: \(B\) rebuilds exactly \(\mathbf w\).* \(B\) sees the kept points and returns the nearest point of their hull. That smaller hull sits inside the big one, so its nearest point is at least as far from the origin as \(\mathbf w\). But \(\mathbf w\) itself is in the smaller hull, by step 4. So \(\mathbf w\) is a nearest point of the smaller hull too, and by the uniqueness of step 2 (applied to the smaller hull) it is *the* one. \(B\) returns \(\mathbf w\), which by step 3 is consistent with all of \(S\). \(\blacksquare\)

So homogeneous halfspaces in \(\mathbb R^d\) have a compression scheme of size \(d\), and the compression bound certifies them with \(k = d\).

## Polynomials: halfspaces in disguise

Suppose junk is a middle band of capitals: shouting a little is suspicious, but a message that is all capitals is usually an automated alert your users want to keep. No threshold can say “junk when \(x\) is neither small nor large”, but a polynomial can: \(p(x) = c_0 + c_1 x + c_2 x^2\) with a negative \(c_2\) and two real roots is positive exactly between them, so \(\operatorname{sign}(p(x))\) says junk on a middle band.

Here is the observation that makes polynomials easy. Write \(\psi(x) = (1, x, x^2)\). Then \(p(x) = \langle \mathbf c, \psi(x) \rangle\) with \(\mathbf c = (c_0, c_1, c_2)\). So a degree-2 polynomial rule on one score is a homogeneous halfspace on the three numbers \(\psi(x)\). Lesson 16 used the same lift. In general, for \(\mathbf x \in \mathbb R^d\) and degree \(r\), let \(\psi(\mathbf x)\) list every **monomial** of degree at most \(r\), a product such as \(x_1 x_3^2\) of at most \(r\) coordinates, together with the constant 1. Every polynomial of degree \(r\) is \(\langle \mathbf c, \psi(\mathbf x) \rangle\) for some vector of coefficients \(\mathbf c\), so the class \(\mathbf x \mapsto \operatorname{sign}(p(\mathbf x))\) is the class of homogeneous halfspaces in the lifted space \(\mathbb R^{d'}\), where \(d'\) is the number of monomials:

\[ d' = \binom{d + r}{r} = O(d^r) \text{ for fixed } r. \]

Here \(\binom{d+r}{r}\) (“\(d + r\) choose \(r\)”) counts the ways to choose the exponents of \(d\) coordinates adding up to at most \(r\), and \(O(d^r)\) says it grows no faster than a constant times \(d^r\) as \(d\) grows. In plain words: **a degree-\(r\) polynomial rule in \(d\) scores is a halfspace in about \(d^r\) lifted scores, so it has a compression scheme of that size.** For one score and degree 2, \(d' = \binom{3}{2} = 3\), the three entries of \((1, x, x^2)\).

```python
from math import comb
for d, r in ((1, 2), (2, 2), (3, 2), (4, 3), (30, 2)):
    print(d, r, comb(d + r, r))
```

It prints `1 2 3`, `2 2 6`, `3 2 10`, `4 3 35` and `30 2 496`. Thirty scores with quadratic rules would cost a compression of size 496.

The scheme itself is the halfspace scheme, run on \(\mathbf u_i = y_i \psi(x_i)\): \(\kappa\) keeps at most \(d'\) examples, and \(B\) lifts them, finds the nearest point of their hull, and returns the polynomial whose coefficients are that point. Eight messages, junk in the middle band of a centred capitals score:

```python
def dot(u, v): return sum(a * b for a, b in zip(u, v))
def min_norm(U):
    w = list(U[0])
    for _ in range(100_000):
        p = min(U, key=lambda u: dot(w, u))
        d = [a - b for a, b in zip(w, p)]
        if dot(d, d) < 1e-24: break
        s = min(1.0, max(0.0, dot(w, d) / dot(d, d)))
        w = [a - s * b for a, b in zip(w, d)]
    return w
S = [(-0.93, -1), (-0.68, -1), (-0.31, 1), (0.02, 1),
     (0.13, 1), (0.22, 1), (0.57, -1), (0.97, -1)]
U = [(y, y * x, y * x * x) for x, y in S]
w = min_norm(U)
I = sorted(range(8), key=lambda i: dot(w, U[i]))[:3]
w2 = min_norm([U[i] for i in I])
print([S[i][0] for i in I], [round(c, 3) for c in w2])
print(min(dot(w2, u) for u in U) > 0)
```

It prints `[0.22, -0.68, 0.57] [0.034, -0.016, -0.148]` and `True`. Three messages are kept, one junk and two keep, and the polynomial rebuilt from them, \(p(x) = 0.034 - 0.016x - 0.148x^2\), labels all eight correctly. Notice the kept points are the three with the smallest \(\langle \mathbf w, \mathbf u_i \rangle\), the face of step 4. (Picking “the \(d'\) smallest” is a shortcut that works here because exactly three points sit on the face. When more than \(d'\) points sit on it, \(\kappa\) must run the Carathéodory reduction to choose among them.)

[[diagram:lift]]

## Margin: what the Perceptron remembers

The last example needs no geometry beyond lesson 9. Suppose every message has length at most 1, \(\|\mathbf x_i\| \le 1\), and the training sequence is separated with **margin** \(\gamma > 0\): some unit vector \(\mathbf w^\star\) has \(y_i \langle \mathbf w^\star, \mathbf x_i \rangle \ge \gamma\) for every \(i\), so every message sits at least \(\gamma\) from the boundary on its correct side. Lesson 9 proved that the Perceptron then stops after at most \((R/\gamma)^2\) updates, where \(R\) is the largest message length; here \(R \le 1\), so at most \(1/\gamma^2\).

Now look at what the Perceptron’s final weight vector *is*. It starts at \(\mathbf 0\), and each update adds \(y_i \mathbf x_i\) for the example it got wrong. So the output is

\[ \mathbf w = \sum_{t=1}^{T} y_{i_t} \mathbf x_{i_t}, \]

where \(T\) is the number of updates and \(i_t\) is the position of the example used in update \(t\). In plain words: **the Perceptron’s answer is the sum of the examples it stumbled on, each pointed along its label.** That is a compression scheme. \(\kappa\) runs the Perceptron and keeps the list of update positions \((i_1, \dots, i_T)\), with repeats, since the Perceptron may stumble on the same example several times. \(B\) adds up \(y \mathbf x\) over the kept examples, which reproduces the Perceptron’s final \(\mathbf w\) exactly, and that vector makes no mistake on \(S\) because the Perceptron only stops when nothing is wrong. The size is \(T \le 1/\gamma^2\). This is where repeats in \([m]^k\) earn their keep: a list of update positions is naturally a list with repeats.

```python
import math, random
rng, gamma, u = random.Random(2), 0.1, (math.cos(1), math.sin(1))
dot = lambda w, x: w[0] * x[0] + w[1] * x[1]
S = []
while len(S) < 200:
    x = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    if math.hypot(*x) <= 1 and gamma <= abs(dot(u, x)) <= 0.2:
        S.append((x, 1 if dot(u, x) > 0 else -1))
w, I = [0.0, 0.0], []
while (i := next((i for i, (x, y) in enumerate(S, 1)
                  if y * dot(w, x) <= 0), None)):
    (x, y) = S[i - 1]
    w, I = [w[0] + y * x[0], w[1] + y * x[1]], I + [i]
v = [sum(S[i - 1][1] * S[i - 1][0][j] for i in I) for j in (0, 1)]
print(len(I), round(1 / gamma ** 2), len(set(I)))
print(sum(y * dot(v, x) <= 0 for x, y in S))
```

It prints `21 100 5` and `0`: 200 messages squeezed into a band just outside margin 0.1, a Perceptron run of 21 updates (the bound allows 100) that used only 5 distinct messages, and a replayed vector \(\mathbf v\) with no mistakes.

One technical wrinkle. The theorem was stated for a scheme of exactly \(k\) positions, but the Perceptron’s list has *at most* \(1/\gamma^2\) entries, and padding it would change the sum. The fix is to let the union bound in the proof run over all lists of every length \(j = 1, \dots, k\), and give \(B\) the length. There are \(\sum_{j=1}^k m^j \le 2m^k\) such lists when \(m \ge 2\) (a geometric sum whose terms at least double each time), and each still leaves at least \(m - k\) examples, so the same proof goes through with \(\delta' = 2m^k\delta\). That changes \(\log(m/\delta)\) into \(\log(2m/\delta)\) in the final bound, and nothing else.

[[diagram:replay]]

## The algorithms

Two schemes are worth writing down as code, because they are what the lab runs and what you will build: the box scheme and the Perceptron scheme. Here \(S\) has \(m\) examples with \(d\) scores each.

```
CompressBox(S):
  P ← positions i with y_i = 1
  if P is empty: return (1, 1, ..., 1)        (2d entries)
  I ← empty list
  for j = 1..d:
    append to I a position in P with the smallest x_{i,j}
    append to I a position in P with the largest x_{i,j}
  return I

RebuildBox(T):                                 (T = k examples)
  Q ← messages x in T with label 1
  if Q is empty: return the all-keep hypothesis
  for j = 1..d:
    a_j ← min of x_j over Q;  b_j ← max of x_j over Q
  return h(x) = 1 if a_j ≤ x_j ≤ b_j for all j, else 0

CompressPerceptron(S):
  w ← 0;  I ← empty list
  while some i has y_i ⟨w, x_i⟩ ≤ 0:
    w ← w + y_i x_i;  append i to I
  return I

Replay(T):
  return the halfspace of w = sum of y x over (x, y) in T
```

**CompressBox** makes one pass per score over the junk examples: \(O(md)\) time, \(O(d)\) space for the \(2d\) positions beyond the input. **RebuildBox** reads \(2d\) examples of \(d\) scores: \(O(d^2)\) time. The compressed certificate, \(2d\) messages, does not grow with \(m\) at all, which is the whole point. **CompressPerceptron** is lesson 9’s Perceptron with a notebook: \(O(m d / \gamma^2)\) time on separable data with margin \(\gamma\) and radius 1, since each of at most \(1/\gamma^2\) updates follows a scan of \(O(md)\), and \(O(d + 1/\gamma^2)\) space. **Replay** costs \(O(d/\gamma^2)\).

The halfspace scheme is costlier. Finding the nearest point of a hull is a convex quadratic program; Gilbert’s method above approaches it step by step, with each step costing \(O(md)\). The Carathéodory reduction then removes one point per round, at most \(m\) rounds, each a Gaussian elimination on \(d + 1\) points of the face, \(O(d^3)\), so \(O(md^3)\) in all. The reconstruction solves the same problem on \(d\) points only.

**Build it yourself.** Using only the standard library:

```python
from math import log

S = [((0.35, 0.25), 1), ((0.6, 0.5), 1), ((0.5, 0.3), 1),
     ((0.75, 0.4), 1), ((0.1, 0.3), 0), ((0.9, 0.5), 0),
     ((0.5, 0.8), 0), ((0.4, 0.1), 0)]
S3 = [((0.4, 0.5, 0.2), 1), ((0.7, 0.3, 0.6), 1),
      ((0.5, 0.45, 0.4), 1), ((0.2, 0.4, 0.3), 0),
      ((0.6, 0.9, 0.5), 0), ((0.55, 0.4, 0.95), 0)]

def compress_box(S, d): ...
def reconstruct_box(T): ...
def predict(box, x): ...
def certified_risk(k, m, delta): ...
def min_labels(k, eps, delta): ...

if __name__ == "__main__":
    I = compress_box(S, 2)
    assert I == [1, 4, 1, 2]
    box = reconstruct_box([S[i - 1] for i in I])
    assert box == [(0.35, 0.75), (0.25, 0.5)]
    assert all(predict(box, x) == y for x, y in S)
    I3 = compress_box(S3, 3)
    box3 = reconstruct_box([S3[i - 1] for i in I3])
    assert len(I3) == 6
    assert all(predict(box3, x) == y for x, y in S3)
    assert compress_box([((0.1, 0.1), 0)], 2) == [1, 1, 1, 1]
    empty = reconstruct_box([((0.1, 0.1), 0)] * 4)
    assert predict(empty, (0.1, 0.1)) == 0
    assert round(certified_risk(4, 10_000, 0.05), 4) == 0.0391
    assert min_labels(4, 0.05, 0.05) == 7640
    assert min_labels(6, 0.05, 0.05) == 11884
```

1. Implement the five functions. Positions are 1-based; `compress_box` returns \(2d\) of them, smallest then largest for each score, ties broken by the first position; `reconstruct_box` returns a list of \(d\) intervals, or `None` for the all-keep hypothesis; `certified_risk` is the realizable bound \(8k\log(m/\delta)/m\); `min_labels` is the smallest \(m \ge 2k\) whose certified risk is at most \(\varepsilon\).
2. Add `compress_perceptron(S)` and `replay(T)` for \(\pm 1\) labels, and check on random data with margin 0.1 that the list has at most 100 entries and the replay makes no mistake.
3. Re-run the auditor’s experiment from the opening with `compress_box`, then delete one *unkept* message and one *kept* message in turn. Which deletion can change the box?

## Fix the bug

The auditor’s report is back, and it has a request in it. Next quarter the filter gets a third score, \(x_3\), the share of misspelled words in the subject, so it will draw boxes in three scores. Compliance wants a written certificate before training: with probability at least 95% (\(\delta = 0.05\)), the filter’s true risk will be at most 5% (\(\varepsilon = 0.05\)). Labels cost money, and the budget is 12,000.

You now have the tool. A certificate needs two things, and the lab checks both. First, a compression scheme for boxes in three scores: a compression function and a reconstruction function whose rebuilt box makes no mistake on the whole training sequence, so that the corollary applies. Second, a number of labels \(m\) for which \(8k\log(m/\delta)/m \le 0.05\), with \(k\) the size of *your* scheme. The smaller your \(k\), the fewer labels you need. Before you open the lab, work out what each \(k\) costs:

```python
from math import log
def min_labels(k, eps, delta):
    m = 2 * k
    while 8 * k * log(m / delta) / m > eps:
        m += 1
    return m
print([min_labels(k, 0.05, 0.05) for k in (4, 6, 8, 12)])
```

It prints `[7640, 11884, 16245, 25212]`. A scheme of size 4 would fit the budget easily, but can four examples always pin down a box in three scores? A cautious scheme that keeps two examples per end of each interval has size 12 and needs 25,212 labels. Somewhere in between is the smallest size your scheme can *guarantee*, and only that one fits.

[[lab-act2]]

## What people get wrong

**“The leftover examples are a fresh test set, so their error estimates the true risk directly.”** They are not fresh: the kept examples were chosen after looking at all of them. The hold-out lemma applies to every *fixed* list of positions, and the union over all \(m^k\) lists is what makes it apply to the learner’s chosen one. That union costs \(k\log m\), which is why the bound has \(k\log(m/\delta)\) where a true hold-out would have \(\log(1/\delta)\).

**“The size of the scheme is the number of different examples it keeps.”** The size is the length \(k\) of the list the compression function returns, fixed for the class in advance. On the eight examples the box scheme kept three different messages in four slots, and the bound still uses \(k = 4\). The theorem’s union runs over all lists of length \(k\), and that count, \(m^k\), is what the size measures.

**“A small certified risk means the bound is close to the truth.”** The bound only uses \(k\). On 2,000 labels it certified 0.17 while the worst of a thousand real boxes had true risk 0.006. The certificate is a guarantee, not a forecast; its strength is that it needs nothing but the size of the scheme.

**“Any class with a compression scheme in the realizable case needs a new scheme for noisy data.”** For binary labels the upgrade is free: find an ERM hypothesis, discard its mistakes, compress what is left. The bound then compares the true risk with the leftover error, which is no longer zero.

## Where this sits

Look at what this lesson added to the list of reasons a learner generalizes. Lesson 4 said: when every hypothesis’s training error tracks its true error at once (uniform convergence). Lesson 13 said: when the learner barely changes if one example changes (stability). Today: when the learner’s output can be described by a few of its own examples. It is the most “engineering” of the three, because it is a fact about the learner’s output that you can check by deleting data and retraining, exactly as the auditor did.

The three are connected. A class of binary hypotheses with a compression scheme of size \(k\) is learnable, as the corollary shows, so by the fundamental theorem of lesson 6 its VC dimension is finite. The converse is a deeper question: does every class of finite VC dimension have a compression scheme of finite size? Manfred Warmuth posed it as an open problem. A later result by Moran and Yehudayoff showed that the answer is yes, with a size that may grow exponentially in the VC dimension; whether a size of the order of the VC dimension always suffices is still being studied.

## Recap and the next bug

1. A compression scheme of size \(k\) is a pair of functions: \(\kappa\) chooses \(k\) positions from the training sequence (repeats allowed), and \(B\) rebuilds a hypothesis from the examples at those positions, consistent with the whole sequence in the realizable case and an ERM in the unrealizable case.
2. Bernstein’s inequality bounds a sum of bounded, mean-zero quantities using their variance. For a fixed hypothesis it gives \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{2L_S(h)\log(1/\delta)/m} + 4\log(1/\delta)/m\) with probability at least \(1 - \delta\): a slack of order \(1/m\) when the error is small.
3. A hypothesis built from some examples and graded on others it never saw obeys that bound on the others. When the kept examples are chosen by looking, a union bound over all \(m^k\) lists pays \(k \log m\), giving the compression bound \(L_{\mathcal D} \le L_V + \sqrt{L_V \cdot 4k\log(m/\delta)/m} + 8k\log(m/\delta)/m\), and \(8k\log(m/\delta)/m\) when the leftovers are all correct.
4. A realizable scheme for binary labels upgrades to an unrealizable one of the same size: ERM, discard its mistakes, compress the rest.
5. Boxes in \(d\) scores compress to \(2d\) examples; homogeneous halfspaces in \(\mathbb R^d\) to \(d\), via the nearest point of the hull and Carathéodory on its face; degree-\(r\) polynomial rules to \(\binom{d+r}{r} = O(d^r)\), by lifting to monomials; and data separated with margin \(\gamma\) to the at most \(1/\gamma^2\) updates of the Perceptron.

The certificate is signed, and the auditor is satisfied. Then your own team comes back with a request of a different kind. Over the years you have all come to trust some rules more than others before seeing any mail at all: narrow boxes around last year’s campaign, halfspaces that lean on the link score. And instead of shipping one rule, they want to ship a weighted mix: pick a rule at random from a distribution you believe in, and let it decide each message. Every bound so far, VC, Rademacher, stability, compression, treats the learner’s output as one rule. What does a guarantee for a whole weighted mix of rules look like, and how does a prior belief over rules, held before any data arrives, turn into one? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. On the eight examples, the box scheme returned the list \((1, 4, 1, 2)\). How many leftover examples are there, \(|V|\)? In the proof of the compression bound, which number plays the role of \(n\), and why is it safe to use it even though \(|V|\) is different? What does the realizable bound give for \(m = 8\), \(k = 4\), \(\delta = 0.05\), and what does that tell you?
2. In the proof of the compression bound, Step 3 says that the learner’s choice of list “cannot escape” the union event. Explain exactly why the hold-out lemma cannot be applied directly to the learner’s own leftovers \(V\), and why paying for \(m^k\) events costs only about \(k\log m\) in the final bound.
3. Boxes in \(d = 5\) scores. Using the realizable bound with the box scheme, what is the smallest \(m\) that certifies true risk at most \(\varepsilon = 0.1\) with confidence \(1 - \delta = 0.99\)? Which size \(k\) do you use, and why can’t it be smaller with this scheme?

## Problems

These are this lesson’s problems. Work them on paper; hints are available on request, never full solutions.

1. **Lines.** Give compression schemes of size 1 for thresholds on one score (\(h_\theta(x) = 1\) when \(x \ge \theta\)) and of size 2 for intervals (\(h_{a,b}(x) = 1\) when \(a \le x \le b\)), and prove they are compression schemes. Then use the realizable bound to find the number of labels each needs for \(\varepsilon = \delta = 0.05\). Lesson 6 showed the tightest-interval learner needs only \(2\log(2/\delta)/\varepsilon\). Compare, and explain where the compression bound loses.
2. **Noisy mail.** In the thresholds experiment, the rebuilt threshold made 13 mistakes on 200 examples with a scheme of size 1. Show that for any scheme of size \(k\), \(L_V(h) \le \frac{m}{m-k} L_S(h) \le 2L_S(h)\), and use it to turn the compression bound into a guarantee in terms of the training error \(L_S\). What does it certify for the noisy thresholds at \(\delta = 0.05\)?
3. **Conjunctions.** Lesson 8’s conjunctions over \(d\) yes/no features (“junk when F and not N”, and so on) are learned in the realizable case by starting with all \(2d\) literals and deleting every literal that some junk example violates. Show that conjunctions have a compression scheme of size \(d + 1\). Which examples should \(\kappa\) keep, and why does the conjunction rebuilt from them equal the one learned from all of \(S\)?
