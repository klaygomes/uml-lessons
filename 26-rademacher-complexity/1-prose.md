# Rademacher Complexity

## The certificate that certifies nothing

The production filter has been running on fifty thousand words for a while now. Every message is a vector of 50,000 yes/no features, one per word in the vocabulary, and the filter is a weight vector \(\mathbf w\) with one weight per word: it calls a message junk when the score \(\langle \mathbf w, \mathbf x \rangle\) is positive. The team keeps the weight vector short, its length is never allowed past a fixed budget, because lessons 13 and 15 showed that short weight vectors are stable and leave wide margins. And it works. On a month of fresh mail that nobody trained on, it misclassifies about 3 messages in 100.

Then operations asks for something new. The lesson 15 certificate covered one learner, soft-SVM with one regularization weight, and only on average over training sequences. But the team no longer ships exactly that. Some weeks they stop the optimizer early, some weeks they hand-tune a few weights, some weeks they try a different solver. Operations wants a single written promise that covers *whatever* weight vector the team ships from the allowed set, holds with probability 95%, and is computed from the 50,000 labels the users can afford:

> For every weight vector of length at most \(B\), true risk is at most training risk plus a number we can write down today.

You already know how to write promises like that. A promise for every hypothesis at once is exactly the uniform convergence property of lesson 4, and you have three ways to pay for it, each measuring how rich the class is.

- **Size** (lesson 4): \(\log |\mathcal H|\). Our class is infinite, so this one is out at once.
- **VC dimension** (lesson 6): halfspaces over 50,000 words have VC dimension 50,000.
- **Description length** (lesson 7): write every weight as a 32-bit number and a weight vector takes 1.6 million bits.

Let’s put numbers in. Lesson 6’s bound says that with probability at least \(1 - \delta\) every hypothesis has \(|L_{\mathcal D}(h) - L_S(h)| \le (4 + 2\sqrt{d \log(2em/d)})/(\delta\sqrt{2m})\), with \(d\) the VC dimension. Lesson 7’s bound adds \(\sqrt{(|h| + \log(2/\delta))/(2m)}\), with \(|h|\) the description length in bits. (Here \(\log\) is the natural logarithm, as everywhere in this course, \(L_{\mathcal D}\) is the true risk and \(L_S\) the empirical risk.)

```python
from math import e, log, sqrt

d, m, delta = 50_000, 50_000, 0.05
vc = (4 + 2 * sqrt(d * log(2 * e * m / d))) / (delta * sqrt(2 * m))
bits = 32 * d
mdl = sqrt((bits + log(2 / delta)) / (2 * m))
print(round(vc, 1), round(mdl, 2))
```

It prints `37.1 4.0`. A risk is a number between 0 and 1, so a promise that the gap is at most 37, or at most 4, promises nothing at all. Both certificates are worthless at this budget, and yet the filter plainly works.

Here is the thing to notice. All three measures were computed before a single message arrived. The VC dimension asks about the *worst* set of 50,000 messages the class could face, and the description length asks how many bits the *worst* weight vector needs. None of them looks at the mail you actually have: short messages, a couple of dozen words each, scored by weight vectors that are short too. So here is the question for this lesson. Can you measure how rich a class is on the very data in hand, and turn that measurement into a certificate?

The answer is yes, and the measurement has a name, the **Rademacher complexity**. By the end of this lesson you will have proved its generalization bound from scratch, learned a small calculus for computing it, used it to show that norm-bounded linear predictors need a number of labels that does not depend on the number of features, and finally proved the margin bound that lesson 15 stated and promised.

## Splitting the labels in two

Start with a question an engineer would ask. You cannot see the true risk, but you can hold some labels back. Suppose you split your labelled messages into two halves at random, and for each hypothesis you compare its empirical risk on the first half with its empirical risk on the second. Both halves come from the same distribution. For any one fixed hypothesis, the two numbers should be close. But if the class is so rich that *some* hypothesis does much better on one half than on the other, then the class can tell two random halves of the same mail apart, and it can just as well tell your training sequence apart from the mail of next week. The largest gap over the class, between two random halves, is a measure of how much a training sequence can fool you.

Let’s try it on six labelled messages and three hypotheses. For each hypothesis we write its **loss vector**: entry \(i\) is 1 if the hypothesis gets example \(i\) wrong, and 0 if it gets it right.

| hypothesis | loss vector on examples 1 to 6 |
|---|---|
| \(h_1\) | 0 0 1 0 0 0 |
| \(h_2\) | 0 1 1 0 1 0 |
| \(h_3\) | 1 0 0 1 0 0 |

Put examples 1, 3 and 4 in the first half and examples 2, 5 and 6 in the second. On the first half \(h_1\) gets one of three wrong, \(1/3\); on the second, none. Its gap is \(1/3 - 0 = 1/3\). For \(h_3\) the gap is \(2/3 - 0 = 2/3\): it looks terrible on one half and perfect on the other.

Now a bookkeeping trick that turns the split into arithmetic. Write the split as a vector of **signs**, \(\sigma_i = +1\) if example \(i\) is in the first half and \(\sigma_i = -1\) if it is in the second. Our split is \(\sigma = (+1, -1, +1, +1, -1, -1)\). When both halves have \(m/2\) examples, the gap of a hypothesis with loss vector \(\mathbf a\) is

\[ \frac{1}{m/2} \sum_{i : \sigma_i = +1} a_i - \frac{1}{m/2} \sum_{i : \sigma_i = -1} a_i = \frac{2}{m} \sum_{i=1}^{m} \sigma_i a_i. \]

Read it from the left. \(\sum_{i : \sigma_i = +1} a_i\) adds up the losses over the examples whose sign is \(+1\) (the colon reads “such that”), so dividing by the half size \(m/2\) gives the first half’s empirical risk; the second term does the same for the signs \(-1\). On the right, \(\sum_{i=1}^m \sigma_i a_i\) adds every loss with its sign attached, so the first half’s losses count positively and the second half’s negatively, and \(2/m\) is the same as \(1/(m/2)\). In plain words: **the gap between two halves is a signed sum of the losses, with the split written as signs.** For \(h_1\) with our \(\sigma\), the signed sum is \(+1 \cdot 1 = 1\) (only example 3 is lost, and it is in the first half), and \(\frac{2}{6} \cdot 1 = 1/3\), the gap we computed by hand.

```python
A = {"h1": (0, 0, 1, 0, 0, 0),
     "h2": (0, 1, 1, 0, 1, 0),
     "h3": (1, 0, 0, 1, 0, 0)}
sigma = (1, -1, 1, 1, -1, -1)
m = len(sigma)
for name, a in A.items():
    first = [ai for ai, s in zip(a, sigma) if s == 1]
    second = [ai for ai, s in zip(a, sigma) if s == -1]
    gap = sum(first) / len(first) - sum(second) / len(second)
    signed = 2 / m * sum(s * ai for s, ai in zip(sigma, a))
    print(name, round(gap, 4), round(signed, 4))
```

It prints `h1 0.3333 0.3333`, `h2 -0.3333 -0.3333` and `h3 0.6667 0.6667`. The two columns agree row by row: the half-split gap and the signed sum are the same number.

[[diagram:split]]

One split is one roll of the dice, though. A class might get a large gap on this split by accident and a small one on most others. So the natural measure averages over random splits: draw every sign independently, as a fair coin, and ask how large the *best* signed sum in the class is on average. That average is the Rademacher complexity.

## Fitting random signs: the Rademacher complexity

There is a second way to read the same signed sum, and it is the one that gives the idea its punch. Forget the halves. Think of \(\sigma\) as a vector of random fake labels, pure noise, and of the loss vector \(\mathbf a\) as a hypothesis’s answers. The signed sum \(\sum_i \sigma_i a_i\) is large when \(\mathbf a\) is large exactly where the noise says \(+1\) and small where it says \(-1\). So the best signed sum over the class asks: **how well can the class fit pure noise on these very messages?** A class that can fit any noise you throw at it is rich enough to fit the accidents of your training sequence too. A class that cannot is safe.

Watch it on the six examples. Here are six sign vectors drawn at random (\(+\) for \(+1\), \(-\) for \(-1\)), the signed sum of each hypothesis divided by \(m = 6\), the best of the three, and, for comparison, the best over the class of *every* loss vector in \(\{0, 1\}^6\), the memorizer’s class from lesson 1. That class can always put a 1 exactly where the sign is \(+\) and a 0 where it is \(-\), so its best value is the fraction of \(+\) signs.

| signs | \(h_1\) | \(h_2\) | \(h_3\) | best of three | best of every vector |
|---|---|---|---|---|---|
| − − − + − + | −1/6 | −1/2 | 0 | 0 | 1/3 |
| + − − − + + | −1/6 | −1/6 | 0 | 0 | 1/2 |
| + + + − − + | 1/6 | 1/6 | 0 | 1/6 | 2/3 |
| + + + + − − | 1/6 | 1/6 | 1/3 | 1/3 | 2/3 |
| − − − − + − | −1/6 | −1/6 | −1/3 | −1/6 | 1/6 |
| − + − + + + | −1/6 | 1/6 | 0 | 1/6 | 2/3 |

Notice two things. The three-hypothesis class sometimes cannot fit the noise at all: in row 5 even its best answer is negative. And the class of every vector always fits it, as well as the signs allow. Averaged over these six rows, the three-hypothesis class scores \(1/12 \approx 0.083\) and the rich class \(1/2\). Six rows is a rough estimate; the exact average uses all \(2^6 = 64\) sign vectors, and we are about to compute it.

**Definition (Rademacher complexity of a set of vectors).** Let \(A \subset \mathbb R^m\) be a set of vectors. Draw \(\sigma = (\sigma_1, \dots, \sigma_m)\) with independent entries, each \(+1\) or \(-1\) with probability \(1/2\). The **Rademacher complexity** of \(A\) is

\[ R(A) = \frac{1}{m}\, \mathbb E_{\sigma}\left[ \sup_{\mathbf a \in A} \sum_{i=1}^{m} \sigma_i a_i \right]. \]

Read it from the inside out.

- \(a_i\) is entry \(i\) of a vector \(\mathbf a\) in \(A\); \(\sigma_i a_i\) attaches the \(i\)-th sign to it.
- \(\sum_{i=1}^m \sigma_i a_i\) adds the \(m\) signed entries. It is the inner product \(\langle \sigma, \mathbf a \rangle\), the correlation between the noise and the vector.
- \(\sup_{\mathbf a \in A}\) is the **supremum** over the set: the smallest number that is at least every value of the expression as \(\mathbf a\) ranges over \(A\). For a finite set it is simply the maximum. It picks the vector that fits this particular noise best.
- \(\mathbb E_\sigma[\cdot]\) is the expectation, the average over the random signs. The variables \(\sigma_i\) are called **Rademacher variables**, which is where the name comes from. For \(m\) signs there are \(2^m\) equally likely sign vectors, so the expectation is a plain average over all of them.
- \(\frac{1}{m}\) turns a sum of \(m\) terms into a per-example number.

In plain words: **the Rademacher complexity of a set of vectors is how well its best member correlates, on average, with random signs.** For the single vector \(h_1\) alone, there is nothing to choose, and the average of \(\sum_i \sigma_i a_i\) is 0 because each \(\sigma_i\) has mean 0: a class of one hypothesis fits no noise at all.

```python
from itertools import product

def rademacher(A, m):
    signs = list(product((-1, 1), repeat=m))
    best = [max(sum(s * a for s, a in zip(sig, v)) for v in A)
            for sig in signs]
    return sum(best) / (len(signs) * m)

A = [(0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (1, 0, 0, 1, 0, 0)]
everything = list(product((0, 1), repeat=6))
print(round(rademacher(A[:1], 6), 4))
print(round(rademacher(A, 6), 4))
print(round(rademacher(everything, 6), 4))
```

It prints `0.0`, `0.1823` and `0.5`. One hypothesis: 0. Three hypotheses: 0.1823. Every loss vector: one half, the most a class of 0/1 loss vectors can ever score, because the best a vector can do is collect a 1 on every \(+\) sign, and on average half the signs are \(+\).

Now connect the vectors to a class. Let \(\mathcal H\) be a hypothesis class and \(\ell\) a loss function, so \(\ell(h, z)\) is the loss of hypothesis \(h\) on example \(z\). It saves a lot of ink to treat each hypothesis through its loss: write \(f_h(z) = \ell(h, z)\) and collect these functions into the **loss class**

\[ \mathcal F = \ell \circ \mathcal H = \{ z \mapsto \ell(h, z) : h \in \mathcal H \}. \]

The braces with a colon read “the set of all functions \(z \mapsto \ell(h, z)\) such that \(h\) is in \(\mathcal H\)”; \(z \mapsto \ell(h, z)\) is the function that sends an example \(z\) to its loss under \(h\); and \(\circ\) is a reminder that each function is “loss after hypothesis”. In plain words: **one loss function per hypothesis.** For a function \(f \in \mathcal F\) we keep the old notation, \(L_{\mathcal D}(f) = \mathbb E_{z \sim \mathcal D}[f(z)]\) for its true risk and \(L_S(f) = \frac1m \sum_{i=1}^m f(z_i)\) for its empirical risk on \(S = (z_1, \dots, z_m)\). (A reminder: \(z \sim \mathcal D\) reads “\(z\) drawn from \(\mathcal D\)”, and \(\mathbb E\) averages over that draw.)

On a training sequence \(S\), every function in \(\mathcal F\) leaves a vector of \(m\) losses, exactly the loss vectors of our table. Collect them:

\[ \mathcal F \circ S = \{ (f(z_1), \dots, f(z_m)) : f \in \mathcal F \}. \]

In plain words: **all the loss vectors the class can produce on these \(m\) examples.** Many hypotheses may produce the same vector; the set keeps each vector once. The **Rademacher complexity of \(\mathcal F\) with respect to \(S\)** is \(R(\mathcal F \circ S)\), the definition above applied to that set:

\[ R(\mathcal F \circ S) = \frac{1}{m}\, \mathbb E_{\sigma \sim \{\pm 1\}^m}\left[ \sup_{f \in \mathcal F} \sum_{i=1}^m \sigma_i f(z_i) \right]. \]

The subscript \(\sigma \sim \{\pm 1\}^m\) says the signs are drawn uniformly from the \(2^m\) sign vectors. In plain words: **how well the class’s losses on your messages can fit random signs.** This is the number the lesson is about, and notice what it depends on: the class, and the \(m\) examples you actually hold. Nothing about the worst possible messages, nothing about how the weight vectors are written down.

[[diagram:fit-noise]]

Try it before we prove anything. In the lab below you pick a class (a handful of fixed hypotheses, short weight vectors, long weight vectors, every labelling), you draw random signs over your labelled messages, and you watch the class’s best fit. Slide the number of messages up and see which classes stop being able to fit noise, and how fast.

[[lab]]

## Representative samples and the ghost sample

Why should fitting noise have anything to do with true risk? Here is the link, and it is the deepest proof in the lesson.

Lesson 4 called a training sequence **\(\varepsilon\)-representative** when every hypothesis’s empirical risk is within \(\varepsilon\) of its true risk, and showed that on an \(\varepsilon/2\)-representative sequence ERM returns a hypothesis within \(\varepsilon\) of the best in the class. So the one number to control is the worst gap. In the notation of loss classes, the **representativeness** of \(S\) is

\[ \mathrm{Rep}_{\mathcal D}(\mathcal F, S) = \sup_{f \in \mathcal F} \left( L_{\mathcal D}(f) - L_S(f) \right). \]

Here \(L_{\mathcal D}(f) - L_S(f)\) is how much \(f\) looks better on the training sequence than it really is, and the supremum takes the worst function in the class. In plain words: **the largest amount by which any hypothesis in the class flatters itself on \(S\).** It is one-sided on purpose: a hypothesis that looks *worse* on \(S\) than it is does no harm. If \(\mathrm{Rep}_{\mathcal D}(\mathcal F, S) = 0.02\), then every hypothesis in the class has true risk at most its empirical risk plus 0.02.

**Lemma (symmetrization).** For every loss class \(\mathcal F\) and distribution \(\mathcal D\),

\[ \mathbb E_{S \sim \mathcal D^m}\left[ \mathrm{Rep}_{\mathcal D}(\mathcal F, S) \right] \le 2\, \mathbb E_{S \sim \mathcal D^m}\left[ R(\mathcal F \circ S) \right]. \]

The subscript \(S \sim \mathcal D^m\) says the training sequence has \(m\) examples drawn i.i.d. from \(\mathcal D\) (independently, each from the same \(\mathcal D\)). In plain words: **on average over training sequences, the worst self-flattery in the class is at most twice its ability to fit random signs.**

You have seen the two ideas of the proof before, in lesson 6: a ghost sample and random signs. Here they are again, now carried all the way to a clean statement.

*Step 1: replace the true risk by a ghost sample.* Let \(S' = (z'_1, \dots, z'_m)\) be a second training sequence, drawn i.i.d. from \(\mathcal D\) independently of \(S\), a **ghost sample** that nobody ever collects. Each \(f(z'_i)\) has mean \(L_{\mathcal D}(f)\), so by linearity of expectation \(L_{\mathcal D}(f) = \mathbb E_{S'}[L_{S'}(f)]\). Therefore, for every \(f\),

\[ L_{\mathcal D}(f) - L_S(f) = \mathbb E_{S'}\left[ L_{S'}(f) - L_S(f) \right] \le \mathbb E_{S'}\left[ \sup_{g \in \mathcal F} \left( L_{S'}(g) - L_S(g) \right) \right]. \]

The equality moves \(L_S(f)\), which does not depend on \(S'\), inside the expectation. The inequality holds because, for every outcome of \(S'\), the term for \(f\) is at most the largest term in the class, and an inequality that holds outcome by outcome survives averaging. The right side no longer depends on \(f\), so it also bounds the supremum over \(f\) of the left side: “the supremum of an average is at most the average of the supremum”. Taking \(\mathbb E_S\) of both sides,

\[ \mathbb E_S\left[ \mathrm{Rep}_{\mathcal D}(\mathcal F, S) \right] \le \mathbb E_{S, S'}\left[ \sup_{f \in \mathcal F} \frac1m \sum_{i=1}^m \left( f(z'_i) - f(z_i) \right) \right]. \]

*Step 2: swapping a pair changes nothing.* Fix one index \(j\). The pair \(z_j, z'_j\) is two independent draws from \(\mathcal D\), so exchanging them gives the same joint distribution of all \(2m\) examples, and the expectation of any function of the \(2m\) examples is unchanged by the exchange. Exchanging \(z_j\) and \(z'_j\) turns the \(j\)-th term \(f(z'_j) - f(z_j)\) into \(f(z_j) - f(z'_j)\), its negative, and leaves the other terms alone. So the expectation above has the same value whether the \(j\)-th term carries a \(+\) sign or a \(-\) sign.

*Step 3: so a random sign changes nothing either.* Let \(\sigma_j\) be a fair sign, independent of everything else. The expectation with \(\sigma_j\) in front of the \(j\)-th term is the average of the “\(+\)” case and the “\(-\)” case, each with weight \(1/2\), and by Step 2 those two cases are equal, so their average is the same value again. Repeat this for \(j = 1, 2, \dots, m\), one index at a time:

\[ \mathbb E_{S, S'}\left[ \sup_{f} \sum_{i} \left( f(z'_i) - f(z_i) \right) \right] = \mathbb E_{S, S', \sigma}\left[ \sup_{f} \sum_{i} \sigma_i \left( f(z'_i) - f(z_i) \right) \right]. \]

[[diagram:ghost]]

*Step 4: split the supremum.* For every \(f\), \(\sum_i \sigma_i f(z'_i) - \sum_i \sigma_i f(z_i)\) is at most \(\sup_g \sum_i \sigma_i g(z'_i) + \sup_g \sum_i (-\sigma_i) g(z_i)\), because each piece is at most its own supremum. So the supremum over \(f\) is at most the sum of the two suprema, and

\[ \mathbb E_{S, S', \sigma}\left[ \sup_{f} \sum_{i} \sigma_i \left( f(z'_i) - f(z_i) \right) \right] \le \mathbb E_{S', \sigma}\left[ \sup_{f} \sum_i \sigma_i f(z'_i) \right] + \mathbb E_{S, \sigma}\left[ \sup_{f} \sum_i (-\sigma_i) f(z_i) \right]. \]

*Step 5: recognise the Rademacher complexity twice.* The vector \(-\sigma\) has exactly the same distribution as \(\sigma\) (flipping every fair coin gives fair coins), so the second expectation equals \(\mathbb E_{S, \sigma}[\sup_f \sum_i \sigma_i f(z_i)] = m\, \mathbb E_S[R(\mathcal F \circ S)]\). The first is the same quantity for \(S'\), and \(S'\) has the same distribution as \(S\), so it is \(m\, \mathbb E_S[R(\mathcal F \circ S)]\) too. Dividing the chain by \(m\),

\[ \mathbb E_S\left[ \mathrm{Rep}_{\mathcal D}(\mathcal F, S) \right] \le 2\, \mathbb E_S\left[ R(\mathcal F \circ S) \right]. \qquad \blacksquare \]

You can watch the lemma hold in a world where the true risk is known exactly. The messages are 20 spam scores, \(0, 1, \dots, 19\), equally likely; a message is junk when its score is at least 12; the class is the 21 thresholds of lesson 6, and the loss is the 0-1 loss. The snippet averages \(\mathrm{Rep}\) over random training sequences of 30 messages, and, separately, twice the best signed fit with one random sign vector per sequence (an unbiased estimate of \(2\,\mathbb E_S[R(\mathcal F \circ S)]\)).

```python
import random

rng = random.Random(0)
X, T, m, runs = range(20), range(21), 30, 4000
f = lambda x: int(x >= 12)
loss = lambda t, x: int(int(x >= t) != f(x))
L_D = {t: sum(loss(t, x) for x in X) / 20 for t in T}
rep = rad = 0.0
for _ in range(runs):
    S = [rng.randrange(20) for _ in range(m)]
    rep += max(L_D[t] - sum(loss(t, x) for x in S) / m
               for t in T)
    sig = [rng.choice((-1, 1)) for _ in S]
    rad += max(sum(s * loss(t, x) for s, x in zip(sig, S))
               for t in T) / m
print(round(rep / runs, 3), round(2 * rad / runs, 3))
```

It prints `0.111 0.242`: the average worst self-flattery, 0.111, sits under twice the average Rademacher complexity, 0.242, as the lemma promises.

## A first guarantee, in expectation

The lemma already gives a learning guarantee. Write \(\mathrm{ERM}_{\mathcal H}(S)\) for the hypothesis ERM returns on \(S\), and \(\ell \circ \mathcal H \circ S\) for the loss vectors of the class on \(S\) (that is \(\mathcal F \circ S\) with \(\mathcal F = \ell \circ \mathcal H\)).

**Theorem (ERM in expectation).** For every \(h^\star \in \mathcal H\),

\[ \mathbb E_{S}\left[ L_{\mathcal D}(\mathrm{ERM}_{\mathcal H}(S)) - L_{\mathcal D}(h^\star) \right] \le 2\, \mathbb E_{S}\left[ R(\ell \circ \mathcal H \circ S) \right], \]

and if \(h^\star\) is the best hypothesis in the class, then for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S\), \(L_{\mathcal D}(\mathrm{ERM}_{\mathcal H}(S)) - L_{\mathcal D}(h^\star) \le 2\,\mathbb E_{S'}[R(\ell \circ \mathcal H \circ S')]/\delta\).

In plain words: **on average, ERM falls short of any fixed competitor by at most twice the expected Rademacher complexity, and dividing by \(\delta\) turns that into a high-probability statement.**

*Proof.* Write \(h_S = \mathrm{ERM}_{\mathcal H}(S)\). First, \(L_{\mathcal D}(h_S) - L_S(h_S) \le \mathrm{Rep}_{\mathcal D}(\mathcal F, S)\) by the definition of the supremum, so by the symmetrization lemma \(\mathbb E_S[L_{\mathcal D}(h_S) - L_S(h_S)] \le 2\,\mathbb E_S[R]\). Second, \(h^\star\) is fixed before \(S\) is drawn, so \(\mathbb E_S[L_S(h^\star)] = L_{\mathcal D}(h^\star)\), and ERM gives \(L_S(h_S) \le L_S(h^\star)\) for every \(S\); averaging, \(\mathbb E_S[L_S(h_S)] \le L_{\mathcal D}(h^\star)\). Add the two: \(\mathbb E_S[L_{\mathcal D}(h_S)] - L_{\mathcal D}(h^\star) \le 2\,\mathbb E_S[R]\). For the last claim, when \(h^\star\) is the best in the class the variable \(L_{\mathcal D}(h_S) - L_{\mathcal D}(h^\star)\) is never negative, so Markov’s inequality (lesson 4: \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\) for \(Z \ge 0\)) with \(a = 2\,\mathbb E[R]/\delta\) bounds the chance of exceeding it by \(\delta\). \(\blacksquare\)

A factor \(1/\delta\) is expensive. For 95% confidence it multiplies the bound by 20, and for 99.9% by 1,000. Lesson 4 met the same problem with Markov’s inequality and fixed it with Hoeffding’s, whose price is only \(\sqrt{\log(1/\delta)}\). We need the same upgrade here, but Hoeffding’s inequality is about averages, and \(\mathrm{Rep}\) is a supremum of averages. We need a tool for functions that are not averages.

## McDiarmid’s inequality

What made an average concentrate? Each example could move it only a little: change one of \(m\) examples with values in \([a, b]\), and the average moves by at most \((b - a)/m\). It turns out that this is the only property that matters. Any function of independent variables that cannot be moved much by changing one variable concentrates just as well.

Here is the property on our quantity. Change one example in \(S\). Every empirical risk \(L_S(f)\) is an average of \(m\) losses, so if every loss lies in \([-c, c]\), every \(L_S(f)\) moves by at most \(2c/m\). Every term \(L_{\mathcal D}(f) - L_S(f)\) moves by at most \(2c/m\), so the supremum of those terms moves by at most \(2c/m\) as well. (If every term goes up by at most \(t\), the largest term goes up by at most \(t\).) So \(\mathrm{Rep}\) cannot be moved by more than \(2c/m\) by any single example.

**Definition (bounded differences).** A function \(f: V^m \to \mathbb R\) of \(m\) variables has **bounded differences** with constant \(c\) if, for every \(i \in [m]\) and all \(x_1, \dots, x_m, x'_i \in V\),

\[ \left| f(x_1, \dots, x_m) - f(x_1, \dots, x_{i-1}, x'_i, x_{i+1}, \dots, x_m) \right| \le c. \]

Here \(V\) is the set each variable takes values in, \(V^m\) is the set of all lists of \(m\) such values, \([m]\) is shorthand for the positions \(\{1, \dots, m\}\), and the second call to \(f\) is the same list with only entry \(i\) replaced by \(x'_i\). In plain words: **changing any one input moves the output by at most \(c\).** The fraction of junk in 50 messages has bounded differences with \(c = 1/50\).

**Theorem (McDiarmid’s inequality).** Let \(f\) have bounded differences with constant \(c\), and let \(X_1, \dots, X_m\) be independent random variables with values in \(V\). Then for every \(t > 0\),

\[ \mathbb P\left[ f(X_1, \dots, X_m) - \mathbb E[f] \ge t \right] \le e^{-2t^2/(m c^2)}, \]

the same holds for \(\mathbb E[f] - f\), and so, for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\),

\[ \left| f(X_1, \dots, X_m) - \mathbb E[f(X_1, \dots, X_m)] \right| \le c \sqrt{\frac{m \log(2/\delta)}{2}}. \]

In the first display, \(\mathbb P[\cdot]\) is the probability of the event in brackets, the event that \(f\) lands at least \(t\) above its average, and \(e^{-2t^2/(mc^2)}\) shrinks fast as \(t\) grows. In the second, the bars are an absolute value, a distance from the average in either direction. In plain words: **a function that no single input can move by more than \(c\) stays within \(c\sqrt{m\log(2/\delta)/2}\) of its average, except with probability \(\delta\).** For an average of \(m\) variables in \([0, 1]\), \(c = 1/m\) and the radius is \(\sqrt{\log(2/\delta)/(2m)}\), exactly Hoeffding’s inequality from lesson 4, so McDiarmid’s inequality contains Hoeffding’s as a special case.

Let’s test it on a supremum, the kind of function we care about. Draw \(m = 100\) uniform scores in \([0, 1]\) and compute the worst gap, over all thresholds \(t\), between the fraction of scores below \(t\) and its true value \(t\). Changing one score moves every fraction by at most \(1/100\), so \(c = 1/100\).

```python
import math
import random

rng, m, delta, runs = random.Random(1), 100, 0.05, 20_000

def worst_gap(u):
    u = sorted(u)
    return max(max((i + 1) / m - x, x - i / m)
               for i, x in enumerate(u))

vals = [worst_gap([rng.random() for _ in range(m)])
        for _ in range(runs)]
mean = sum(vals) / runs
t = (1 / m) * math.sqrt(math.log(2 / delta) * m / 2)
miss = sum(abs(v - mean) > t for v in vals) / runs
print(round(mean, 4), round(t, 4), miss)
```

It prints `0.0855 0.1358 0.00015`: the worst gap averages 0.0855, McDiarmid’s radius at \(\delta = 0.05\) is 0.1358, and only 0.015% of the runs stray further, well under the 5% allowed.

[[diagram:bounded-differences]]

Now the proof. The idea is to reveal the variables one at a time and watch our best guess of \(f\) change. Each reveal moves the guess by a step that averages to zero and lies in a short interval, and Hoeffding’s lemma from lesson 4 handles exactly such steps. A reminder of that lemma: **if \(Y\) takes values in an interval of length \(L\) and \(\mathbb E[Y] = 0\), then \(\mathbb E[e^{\lambda Y}] \le e^{\lambda^2 L^2/8}\) for every \(\lambda > 0\).**

*Step 1: guesses after \(k\) reveals.* For \(k = 0, 1, \dots, m\) define

\[ g_k(x_1, \dots, x_k) = \mathbb E\left[ f(x_1, \dots, x_k, X_{k+1}, \dots, X_m) \right], \]

the average of \(f\) when the first \(k\) inputs are fixed at \(x_1, \dots, x_k\) and the rest are still random. So \(g_0 = \mathbb E[f]\), a number, and \(g_m = f\). Because the variables are independent, fixing the first \(k\) does not change the distribution of the rest, so \(g_k\) is also the conditional expectation of \(f\) given the first \(k\) variables. Write \(D_k = g_k(X_1, \dots, X_k) - g_{k-1}(X_1, \dots, X_{k-1})\) for the step the \(k\)-th reveal causes. The steps telescope:

\[ f(X_1, \dots, X_m) - \mathbb E[f] = g_m - g_0 = \sum_{k=1}^m D_k. \]

*Step 2: each step averages to zero.* Fix the first \(k - 1\) values. Averaging \(g_k(x_1, \dots, x_{k-1}, X_k)\) over \(X_k\) averages \(f\) over \(X_k, \dots, X_m\), which is \(g_{k-1}(x_1, \dots, x_{k-1})\). So, with the first \(k - 1\) values fixed, \(D_k\) has mean 0.

*Step 3: each step lies in a short interval.* With the first \(k - 1\) values fixed, compare two possible values \(x\) and \(x'\) of the \(k\)-th variable:

\[ g_k(\dots, x) - g_k(\dots, x') = \mathbb E\left[ f(\dots, x, X_{k+1}, \dots) - f(\dots, x', X_{k+1}, \dots) \right] \le c, \]

because the two calls differ in input \(k\) only, so every outcome of the difference is at most \(c\). So, as \(x\) varies, \(g_k\) stays in an interval of length at most \(c\), and \(D_k\), which is \(g_k\) minus a fixed number, does too. The snippet checks Steps 2 and 3 exhaustively for a small \(f\): three variables with values in \(\{0, 1, 2\}\) and \(f\) the number of distinct values, which has bounded differences with \(c = 1\).

```python
from itertools import product

V, m = (0, 1, 2), 3
f = lambda x: len(set(x))

def g(prefix):
    rest = list(product(V, repeat=m - len(prefix)))
    return sum(f(prefix + r) for r in rest) / len(rest)

widest = 0.0
for k in range(1, m + 1):
    for p in product(V, repeat=k - 1):
        vals = [g(p + (v,)) for v in V]
        mean_step = sum(vals) / len(V) - g(p)
        widest = max(widest, max(vals) - min(vals))
        assert abs(mean_step) < 1e-12
print(round(g(()), 4), round(widest, 4))
```

It prints `2.1111 1.0`: the starting guess \(\mathbb E[f] = 19/9\), every step has mean exactly 0 (the asserts pass), and the widest interval any step moves in is exactly \(c = 1\).

*Step 4: multiply the steps’ exponential moments.* Fix \(\lambda > 0\). Peel off the last step: with \(X_1, \dots, X_{m-1}\) fixed, the first \(m - 1\) steps are fixed numbers, and Hoeffding’s lemma applied to \(D_m\) (mean 0, interval length at most \(c\)) gives

\[ \mathbb E_{X_m}\left[ e^{\lambda \sum_{k \le m} D_k} \right] = e^{\lambda \sum_{k < m} D_k}\, \mathbb E_{X_m}\left[ e^{\lambda D_m} \right] \le e^{\lambda \sum_{k < m} D_k}\, e^{\lambda^2 c^2/8}. \]

Now average over \(X_1, \dots, X_{m-1}\) and repeat the argument with \(D_{m-1}\), then \(D_{m-2}\), down to \(D_1\). Each round contributes one factor \(e^{\lambda^2 c^2/8}\), so \(\mathbb E[e^{\lambda(f - \mathbb E f)}] \le e^{m \lambda^2 c^2/8}\).

*Step 5: Markov, then the best \(\lambda\).* Since \(e^{\lambda u}\) increases with \(u\), Markov’s inequality gives

\[ \mathbb P[f - \mathbb E f \ge t] = \mathbb P\left[ e^{\lambda(f - \mathbb E f)} \ge e^{\lambda t} \right] \le e^{-\lambda t}\, e^{m\lambda^2 c^2/8}. \]

The exponent \(-\lambda t + m \lambda^2 c^2/8\) is smallest at \(\lambda = 4t/(mc^2)\), where it equals \(-2t^2/(mc^2)\). The same argument applied to \(-f\), which has the same bounded differences, bounds \(\mathbb P[\mathbb E f - f \ge t]\). By the union bound, \(\mathbb P[|f - \mathbb E f| \ge t] \le 2e^{-2t^2/(mc^2)}\). Setting the right side to \(\delta\) and solving, \(t = c\sqrt{m\log(2/\delta)/2}\). \(\blacksquare\)

## The bound you can compute from your data

Now we can state the main theorem. We assume every loss is bounded, \(|\ell(h, z)| \le c\) for every \(h\) and \(z\).

**Theorem (Rademacher generalization bounds).** Let \(\delta \in (0, 1)\).

1. With probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H\) satisfies
\[ L_{\mathcal D}(h) - L_S(h) \le 2\, \mathbb E_{S' \sim \mathcal D^m}\left[ R(\ell \circ \mathcal H \circ S') \right] + c\sqrt{\frac{2\log(2/\delta)}{m}}. \]
2. With probability at least \(1 - \delta\), every \(h \in \mathcal H\) satisfies
\[ L_{\mathcal D}(h) - L_S(h) \le 2\, R(\ell \circ \mathcal H \circ S) + 3c\sqrt{\frac{2\log(4/\delta)}{m}}. \]
3. For every fixed \(h^\star \in \mathcal H\), with probability at least \(1 - \delta\),
\[ L_{\mathcal D}(\mathrm{ERM}_{\mathcal H}(S)) - L_{\mathcal D}(h^\star) \le 2\, R(\ell \circ \mathcal H \circ S) + 4c\sqrt{\frac{2\log(8/\delta)}{m}}. \]

Read the first one term by term. \(\mathbb E_{S'}[R(\ell \circ \mathcal H \circ S')]\) is the class’s ability to fit noise, averaged over training sequences \(S'\) of size \(m\) (the prime only says it is a fresh variable, not your \(S\)). The square root is the price of confidence, with the \(\sqrt{\log(1/\delta)}\) we wanted. The second one replaces the average complexity by \(R(\ell \circ \mathcal H \circ S)\), the complexity measured on *your* training sequence, at the cost of a larger constant. In plain words: **with high probability, every hypothesis’s true risk is at most its empirical risk, plus twice the class’s ability to fit noise on your messages, plus a confidence term.** The second form is a **data-dependent bound**: the same messages are used to choose the hypothesis and to judge how rich the class is on them.

*Proof of 1.* We showed above that \(\mathrm{Rep}_{\mathcal D}(\mathcal F, S)\) has bounded differences with constant \(2c/m\) as a function of the \(m\) independent examples. McDiarmid’s inequality with that constant gives, with probability at least \(1 - \delta\),

\[ \mathrm{Rep}_{\mathcal D}(\mathcal F, S) \le \mathbb E_S[\mathrm{Rep}_{\mathcal D}(\mathcal F, S)] + \frac{2c}{m}\sqrt{\frac{m\log(2/\delta)}{2}} = \mathbb E_S[\mathrm{Rep}] + c\sqrt{\frac{2\log(2/\delta)}{m}}. \]

The symmetrization lemma bounds \(\mathbb E_S[\mathrm{Rep}]\) by \(2\,\mathbb E_{S'}[R]\), and every \(h\) has \(L_{\mathcal D}(h) - L_S(h) \le \mathrm{Rep}\) by the definition of the supremum.

*Proof of 2.* The measured complexity \(R(\ell \circ \mathcal H \circ S)\) also has bounded differences with constant \(2c/m\): changing \(z_i\) changes only the term \(\sigma_i f(z_i)\), by at most \(2c\), for every \(\sigma\) and \(f\); so every supremum moves by at most \(2c\), and so does their average, and dividing by \(m\) gives \(2c/m\). Now spend half the confidence on each of two events. Part 1 with \(\delta/2\) holds with probability at least \(1 - \delta/2\) and gives the confidence term \(c\sqrt{2\log(4/\delta)/m}\). McDiarmid with \(\delta/2\) gives, with probability at least \(1 - \delta/2\), \(\mathbb E_{S'}[R] \le R(\ell \circ \mathcal H \circ S) + c\sqrt{2\log(4/\delta)/m}\). By the union bound both hold together with probability at least \(1 - \delta\), and then the bound is \(2R(\ell \circ \mathcal H \circ S) + 2c\sqrt{\cdot} + c\sqrt{\cdot}\). (You will also see this written with \(4c\), which is weaker and therefore also true.)

*Proof of 3.* Write \(h_S = \mathrm{ERM}_{\mathcal H}(S)\). Add and subtract empirical risks:

\[ L_{\mathcal D}(h_S) - L_{\mathcal D}(h^\star) = \big(L_{\mathcal D}(h_S) - L_S(h_S)\big) + \big(L_S(h_S) - L_S(h^\star)\big) + \big(L_S(h^\star) - L_{\mathcal D}(h^\star)\big). \]

The middle bracket is at most 0, because ERM minimizes the empirical risk. The first bracket is bounded by part 2 with \(\delta/2\): \(2R + 3c\sqrt{2\log(8/\delta)/m}\), with probability at least \(1 - \delta/2\). For the last bracket, \(h^\star\) is fixed before \(S\) is drawn, so \(L_S(h^\star)\) is an average of \(m\) i.i.d. losses in \([-c, c]\), an interval of length \(2c\), with mean \(L_{\mathcal D}(h^\star)\). Hoeffding’s inequality with \(\delta/2\) gives, with probability at least \(1 - \delta/2\), \(L_S(h^\star) - L_{\mathcal D}(h^\star) \le 2c\sqrt{\log(4/\delta)/(2m)} = c\sqrt{2\log(4/\delta)/m}\), which is at most \(c\sqrt{2\log(8/\delta)/m}\). A union bound over the two events finishes it. \(\blacksquare\)

Put numbers in. Suppose the loss is the ramp loss of lesson 15, so \(c = 1\), you have \(m = 50{,}000\) labels, \(\delta = 0.05\), and the class’s complexity, average or measured, is 0.045.

```python
from math import log, sqrt

m, delta, c, R_hat = 50_000, 0.05, 1.0, 0.045
part1 = 2 * R_hat + c * sqrt(2 * log(2 / delta) / m)
part2 = 2 * R_hat + 3 * c * sqrt(2 * log(4 / delta) / m)
print(round(part1, 4), round(part2, 4))
```

It prints `0.1021 0.1297`. Compare with 37.1 from the VC route. Everything now hinges on the complexity term: if a class cannot fit noise on 50,000 messages, the certificate is tight. So the next job is to compute \(R\) for the classes we care about, and that is what the calculus below is for.

## Rademacher calculus

Computing a supremum over an infinite class, averaged over \(2^m\) sign vectors, sounds hopeless. It rarely has to be done from scratch. Four rules let you build the complexity of a complicated set from simple pieces.

[[diagram:calculus]]

### Rule 1: shifting and scaling

**Lemma (shift and scale).** For every \(A \subset \mathbb R^m\), number \(c \in \mathbb R\) and vector \(\mathbf a_0 \in \mathbb R^m\),

\[ R(\{ c\,\mathbf a + \mathbf a_0 : \mathbf a \in A \}) = |c|\, R(A). \]

The set on the left stretches every vector of \(A\) by the factor \(c\) and then moves it by the same fixed vector \(\mathbf a_0\). In plain words: **moving a set does not change its complexity, and stretching it by \(c\) multiplies its complexity by \(|c|\).**

*Proof.* For each \(\sigma\), \(\sum_i \sigma_i (c a_i + a_{0,i}) = c\sum_i \sigma_i a_i + \sum_i \sigma_i a_{0,i}\). The second sum does not depend on \(\mathbf a\), so it leaves the supremum, and its average over \(\sigma\) is 0 because every \(\sigma_i\) has mean 0. If \(c \ge 0\), \(\sup_{\mathbf a} c\sum_i\sigma_i a_i = c\sup_{\mathbf a}\sum_i\sigma_i a_i\). If \(c < 0\), it is \(|c|\sup_{\mathbf a}\sum_i(-\sigma_i)a_i\), and \(-\sigma\) has the same distribution as \(\sigma\), so after averaging it is \(|c|\,mR(A)\) again. \(\blacksquare\)

```python
from itertools import product

def rademacher(A, m):
    signs = list(product((-1, 1), repeat=m))
    best = [max(sum(s * a for s, a in zip(sig, v)) for v in A)
            for sig in signs]
    return sum(best) / (len(signs) * m)

A = [(0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (1, 0, 0, 1, 0, 0)]
c, a0 = -3, (1, 2, 3, 4, 5, 6)
moved = [tuple(c * ai + bi for ai, bi in zip(a, a0)) for a in A]
print(round(rademacher(A, 6), 4), round(rademacher(moved, 6), 4))
```

It prints `0.1823 0.5469`: multiplying our three loss vectors by \(-3\) and shifting them by \((1, \dots, 6)\) multiplies the complexity by exactly 3.

### Rule 2: mixing adds nothing

A **convex combination** of vectors \(\mathbf a^{(1)}, \dots, \mathbf a^{(N)}\) is \(\sum_{j=1}^N \alpha_j \mathbf a^{(j)}\) with weights \(\alpha_j \ge 0\) that add up to 1 (written \(\|\boldsymbol\alpha\|_1 = 1\), the sum of the absolute values of the weights). The **convex hull** of \(A\) is the set of all convex combinations of finitely many vectors of \(A\): every way of mixing them. Averaging hypotheses, as boosting did in lesson 10, lands in the convex hull.

**Lemma (convex hull).** Let \(A'\) be the convex hull of \(A\). Then \(R(A') = R(A)\).

In plain words: **mixing the vectors of a set cannot fit noise any better than the best unmixed vector.**

*Proof.* For a fixed \(\sigma\), a mix scores \(\langle \sigma, \sum_j \alpha_j \mathbf a^{(j)} \rangle = \sum_j \alpha_j \langle \sigma, \mathbf a^{(j)} \rangle\), a weighted average of the members’ scores. A weighted average is never more than its largest entry, and putting all the weight on the best member reaches it, so for a fixed choice of members the best weights give \(\max_j \langle \sigma, \mathbf a^{(j)} \rangle\). Taking the supremum over the choice of members too, \(\sup_{\mathbf a' \in A'} \langle \sigma, \mathbf a' \rangle = \sup_{\mathbf a \in A}\langle \sigma, \mathbf a \rangle\). The two suprema are equal for every \(\sigma\), so their averages are equal. \(\blacksquare\)

```python
import random
from itertools import product

def rademacher(A, m):
    signs = list(product((-1, 1), repeat=m))
    best = [max(sum(s * a for s, a in zip(sig, v)) for v in A)
            for sig in signs]
    return sum(best) / (len(signs) * m)

A = [(0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (1, 0, 0, 1, 0, 0)]
rng, hull = random.Random(2), list(A)
for _ in range(200):
    w = [rng.random() for _ in A]
    alpha = [wi / sum(w) for wi in w]
    hull.append(tuple(sum(al * a[i] for al, a in zip(alpha, A))
                      for i in range(6)))
print(round(rademacher(A, 6), 4), round(rademacher(hull, 6), 4))
```

It prints `0.1823 0.1823`: adding 200 random mixes changes nothing.

### Rule 3: finite sets grow only logarithmically

How rich can a finite set be? Intuition from lesson 4 says the price of \(N\) hypotheses is \(\log N\). Massart’s lemma makes that precise for the Rademacher complexity. We need two small facts first.

The first is an inequality about exponentials: for every real \(a\),

\[ \frac{e^{a} + e^{-a}}{2} \le e^{a^2/2}. \]

The left side is the average of \(e^{\sigma a}\) over a fair sign \(\sigma\), which is exactly the quantity we will meet. In plain words: **the average of \(e^{a}\) and \(e^{-a}\) is at most \(e^{a^2/2}\).** At \(a = 1\): \((2.718 + 0.368)/2 = 1.543 \le e^{0.5} = 1.649\). *Proof.* Expand both sides as power series. The odd powers cancel on the left, leaving \(\sum_{n \ge 0} a^{2n}/(2n)!\); the right side is \(\sum_{n \ge 0} a^{2n}/(2^n n!)\). Term by term the left is smaller, because \((2n)! = \prod_{k=1}^{2n} k \ge \prod_{k=1}^{n} (2k) = 2^n n!\) (keep only the even factors, each at least 1 in the rest). \(\blacksquare\)

The second is **Jensen’s inequality** for the logarithm: since \(\log\) is concave, the log of an average is at least the average of the logs, \(\mathbb E[\log Y] \le \log \mathbb E[Y]\) for a positive random variable \(Y\). (Lesson 14 used Jensen for convex functions; a concave function flips the inequality.)

**Lemma (Massart).** Let \(A = \{\mathbf a_1, \dots, \mathbf a_N\}\) be \(N \ge 2\) vectors in \(\mathbb R^m\), with mean \(\bar{\mathbf a} = \frac1N \sum_{j=1}^N \mathbf a_j\). Then

\[ R(A) \le \max_{\mathbf a \in A} \|\mathbf a - \bar{\mathbf a}\| \, \frac{\sqrt{2\log N}}{m}. \]

Here \(\|\mathbf a - \bar{\mathbf a}\|\) is the Euclidean length of the gap between a vector and the mean of the set, and the maximum takes the vector furthest from the mean. In plain words: **a finite set fits noise no better than its spread times \(\sqrt{2\log N}\), divided by \(m\).** For 0/1 loss vectors of length \(m\), the spread is at most \(\sqrt m\), so the bound is at most \(\sqrt{2\log N/m}\): a thousand hypotheses on 50,000 examples fit noise at most \(\sqrt{2 \cdot 6.91/50{,}000} \approx 0.017\).

*Proof.* By the shift rule we may move the set by \(-\bar{\mathbf a}\) without changing \(R\), so assume \(\bar{\mathbf a} = \mathbf 0\), and write \(M = \max_{\mathbf a}\|\mathbf a\|\) (if \(M = 0\) the set is \(\{\mathbf 0\}\) and \(R = 0\)). Fix \(\lambda > 0\) and let \(A' = \lambda A\).

*Step 1: from a max to a sum of exponentials.* For each \(\sigma\), \(\max_{\mathbf a \in A'} \langle \sigma, \mathbf a\rangle = \log \max_{\mathbf a \in A'} e^{\langle \sigma, \mathbf a\rangle} \le \log \sum_{\mathbf a \in A'} e^{\langle \sigma, \mathbf a\rangle}\), because the largest of positive numbers is at most their sum, and \(\log\) increases.

*Step 2: Jensen.* Averaging over \(\sigma\) and using \(\mathbb E[\log Y] \le \log\mathbb E[Y]\),

\[ m R(A') \le \log \sum_{\mathbf a \in A'} \mathbb E_\sigma\left[ e^{\sum_i \sigma_i a_i} \right] = \log \sum_{\mathbf a \in A'} \prod_{i=1}^m \mathbb E_{\sigma_i}\left[ e^{\sigma_i a_i} \right]. \]

The equality holds because \(e^{\sum_i \sigma_i a_i} = \prod_i e^{\sigma_i a_i}\), and the expectation of a product of independent variables is the product of their expectations.

*Step 3: the exponential inequality.* Each factor is \((e^{a_i} + e^{-a_i})/2 \le e^{a_i^2/2}\), so the product is at most \(e^{\|\mathbf a\|^2/2}\), and

\[ m R(A') \le \log \sum_{\mathbf a \in A'} e^{\|\mathbf a\|^2/2} \le \log\left( N e^{\lambda^2 M^2/2} \right) = \log N + \frac{\lambda^2 M^2}{2}. \]

*Step 4: undo the scaling and choose \(\lambda\).* By the scale rule \(R(A') = \lambda R(A)\), so \(R(A) \le (\log N + \lambda^2 M^2/2)/(\lambda m)\) for every \(\lambda > 0\). The choice \(\lambda = \sqrt{2\log N}/M\) makes the two terms equal, each \(\log N\), and the bound becomes \(2\log N \cdot M/(m\sqrt{2\log N}) = M\sqrt{2\log N}/m\). \(\blacksquare\)

```python
import math
import random
from itertools import product

rng, m, N = random.Random(3), 10, 20
A = [tuple(rng.randint(0, 1) for _ in range(m)) for _ in range(N)]
signs = list(product((-1, 1), repeat=m))
exact = sum(max(sum(s * a for s, a in zip(sig, v)) for v in A)
            for sig in signs) / (len(signs) * m)
mean = [sum(v[i] for v in A) / N for i in range(m)]
spread = max(math.dist(v, mean) for v in A)
bound = spread * math.sqrt(2 * math.log(N)) / m
print(round(exact, 4), round(bound, 4))
```

It prints `0.2852 0.4229`: twenty random 0/1 vectors of length 10 have exact complexity 0.2852, under Massart’s 0.4229.

### Rule 4: Lipschitz functions contract

The last rule is the one that connects losses to scores. Most losses we use are a function of the score: the ramp loss of lesson 15 is \(\min\{1, \max\{0, 1 - y\langle\mathbf w, \mathbf x\rangle\}\}\), a function of \(\langle \mathbf w, \mathbf x\rangle\). Recall from lesson 12 that a function \(\phi: \mathbb R \to \mathbb R\) is **\(\rho\)-Lipschitz** if \(|\phi(\alpha) - \phi(\beta)| \le \rho|\alpha - \beta|\) for all \(\alpha, \beta\): it never stretches distances by more than \(\rho\). The ramp loss is 1-Lipschitz in the score: its slope is 0 or \(\pm 1\).

**Lemma (contraction).** For each \(i \in [m]\) let \(\phi_i: \mathbb R \to \mathbb R\) be \(\rho\)-Lipschitz, and for \(\mathbf a \in \mathbb R^m\) write \(\phi(\mathbf a) = (\phi_1(a_1), \dots, \phi_m(a_m))\). Then

\[ R(\{\phi(\mathbf a) : \mathbf a \in A\}) \le \rho\, R(A). \]

The set on the left applies the \(i\)-th function to the \(i\)-th entry of every vector. In plain words: **passing every entry through a function that stretches by at most \(\rho\) multiplies the complexity by at most \(\rho\).** So the complexity of the ramp losses is at most the complexity of the scores, which is far easier to compute.

*Proof.* We take \(\rho = 1\); for general \(\rho\), apply the result to \(\phi_i/\rho\), which is 1-Lipschitz, and then the scale rule. We also apply the functions one coordinate at a time: let \(A_1 = \{(\phi_1(a_1), a_2, \dots, a_m) : \mathbf a \in A\}\), only the first entry changed. If we show \(R(A_1) \le R(A)\), the same argument (with coordinate 1 renamed) applies to coordinate 2 of \(A_1\), and so on; after \(m\) steps every entry has been passed through its function, and \(R\) never went up. We assume \(A\) is bounded, so every supremum is finite.

*Step 1: average out the first sign.* Write \(G(\mathbf a) = \sum_{i=2}^m \sigma_i a_i\), which does not involve \(\sigma_1\). Averaging over \(\sigma_1 = +1\) and \(\sigma_1 = -1\), each with probability \(1/2\),

\[ m R(A_1) = \frac12\, \mathbb E_{\sigma_2, \dots, \sigma_m}\left[ \sup_{\mathbf a \in A}\big(\phi_1(a_1) + G(\mathbf a)\big) + \sup_{\mathbf a' \in A}\big(-\phi_1(a'_1) + G(\mathbf a')\big) \right]. \]

*Step 2: two suprema become one.* A sum of two suprema over independent choices is the supremum over pairs:

\[ \sup_{\mathbf a}(\cdots) + \sup_{\mathbf a'}(\cdots) = \sup_{\mathbf a, \mathbf a' \in A}\big(\phi_1(a_1) - \phi_1(a'_1) + G(\mathbf a) + G(\mathbf a')\big). \]

*Step 3: Lipschitz.* \(\phi_1(a_1) - \phi_1(a'_1) \le |a_1 - a'_1|\), so the expression inside is at most \(|a_1 - a'_1| + G(\mathbf a) + G(\mathbf a')\).

*Step 4: drop the absolute value.* The expression \(|a_1 - a'_1| + G(\mathbf a) + G(\mathbf a')\) is symmetric in the pair: swapping \(\mathbf a\) and \(\mathbf a'\) leaves it unchanged. For each pair, either \(a_1 \ge a'_1\), and it equals \(a_1 - a'_1 + G(\mathbf a) + G(\mathbf a')\), or \(a_1 < a'_1\), and it equals the same expression for the swapped pair \((\mathbf a', \mathbf a)\). Either way its value is reached by \(a_1 - a'_1 + G(\mathbf a) + G(\mathbf a')\) at some pair, and conversely that expression is never more than the one with the absolute value. So the two suprema over pairs are equal.

*Step 5: run Steps 1 and 2 backwards.* The supremum over pairs of \(a_1 - a'_1 + G(\mathbf a) + G(\mathbf a')\) is \(\sup_{\mathbf a}(a_1 + G(\mathbf a)) + \sup_{\mathbf a'}(-a'_1 + G(\mathbf a'))\), and half its average over \(\sigma_2, \dots, \sigma_m\) is, by Step 1 read for \(A\) instead of \(A_1\), exactly \(m R(A)\). So \(m R(A_1) \le m R(A)\). \(\blacksquare\)

The snippet takes 30 random score vectors on 10 examples, passes them through the ramp loss with random labels, \(\phi_i(a) = \mathrm{ramp}(y_i a)\), and compares exact complexities.

```python
import random
from itertools import product

rng, m = random.Random(4), 10
y = [rng.choice((-1, 1)) for _ in range(m)]
A = [tuple(rng.uniform(-2, 2) for _ in range(m))
     for _ in range(30)]
ramp = lambda s: min(1.0, max(0.0, 1.0 - s))
phiA = [tuple(ramp(yi * ai) for yi, ai in zip(y, a)) for a in A]

def rademacher(A):
    signs = list(product((-1, 1), repeat=m))
    return sum(max(sum(s * a for s, a in zip(sig, v)) for v in A)
               for sig in signs) / (len(signs) * m)

print(round(rademacher(phiA), 4), round(rademacher(A), 4))
```

It prints `0.2802 0.727`: the ramp losses fit noise less than half as well as the raw scores.

## Linear classes: complexity without the dimension

Now the payoff. Take the two simplest norm-bounded classes of linear predictors:

\[ \mathcal H_1 = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x\rangle : \|\mathbf w\|_1 \le 1 \}, \qquad \mathcal H_2 = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x\rangle : \|\mathbf w\|_2 \le 1 \}. \]

Each class is a set of score functions. \(\|\mathbf w\|_2 = \sqrt{\sum_j w_j^2}\) is the usual Euclidean length (written \(\|\mathbf w\|\) in earlier lessons), and \(\|\mathbf w\|_1 = \sum_j |w_j|\) adds up the absolute values of the weights. We will also need \(\|\mathbf x\|_\infty = \max_j |x_j|\), the largest entry in absolute value. In plain words: **\(\mathcal H_2\) is every linear score with Euclidean length at most 1, and \(\mathcal H_1\) every linear score whose weights add up, in absolute value, to at most 1.** For a yes/no word vector with 16 words present, \(\|\mathbf x\|_2 = 4\), \(\|\mathbf x\|_1 = 16\) and \(\|\mathbf x\|_\infty = 1\). On a training sequence of messages \(S = (\mathbf x_1, \dots, \mathbf x_m)\), \(\mathcal H_2 \circ S = \{(\langle \mathbf w, \mathbf x_1\rangle, \dots, \langle \mathbf w, \mathbf x_m\rangle) : \|\mathbf w\|_2 \le 1\}\) is the set of score vectors, and likewise for \(\mathcal H_1\).

[[diagram:balls]]

**Lemma (Euclidean ball).** For any vectors \(\mathbf x_1, \dots, \mathbf x_m\),

\[ R(\mathcal H_2 \circ S) \le \frac{\max_i \|\mathbf x_i\|_2}{\sqrt m}. \]

In plain words: **unit-length weight vectors fit noise no better than the longest message’s length divided by \(\sqrt m\).** No dimension appears anywhere: the vectors may live in \(\mathbb R^{50{,}000}\), or even in an infinite-dimensional Hilbert space (lesson 16), and the bound does not change. For our mail, at most 25 words per message, \(\max_i\|\mathbf x_i\| \le 5\), and at \(m = 50{,}000\) the bound is \(5/223.6 = 0.0224\).

*Proof.* *Step 1: the best weight vector.* For each \(\sigma\), \(\sum_i \sigma_i\langle\mathbf w, \mathbf x_i\rangle = \langle \mathbf w, \mathbf v\rangle\) with \(\mathbf v = \sum_i \sigma_i\mathbf x_i\), by linearity of the inner product. By the Cauchy–Schwarz inequality (lesson 9), \(\langle\mathbf w, \mathbf v\rangle \le \|\mathbf w\|\|\mathbf v\| \le \|\mathbf v\|\), and \(\mathbf w = \mathbf v/\|\mathbf v\|\) reaches it, so the supremum is exactly \(\|\sum_i \sigma_i \mathbf x_i\|\), and \(m R(\mathcal H_2 \circ S) = \mathbb E_\sigma\|\sum_i \sigma_i \mathbf x_i\|\).

*Step 2: Jensen.* For any random variable \(Y\), \((\mathbb E Y)^2 \le \mathbb E[Y^2]\), because the difference is the variance, which is never negative. So \(\mathbb E\|\mathbf v\| \le \sqrt{\mathbb E\|\mathbf v\|^2}\).

*Step 3: the cross terms vanish.* Expand the square: \(\|\sum_i\sigma_i\mathbf x_i\|^2 = \sum_{i,j}\sigma_i\sigma_j\langle\mathbf x_i, \mathbf x_j\rangle\). For \(i \ne j\), \(\mathbb E[\sigma_i\sigma_j] = \mathbb E[\sigma_i]\mathbb E[\sigma_j] = 0\) by independence; for \(i = j\), \(\sigma_i^2 = 1\). So \(\mathbb E\|\mathbf v\|^2 = \sum_i\|\mathbf x_i\|^2 \le m\max_i\|\mathbf x_i\|^2\).

Putting the steps together, \(m R \le \sqrt{m}\max_i\|\mathbf x_i\|\), and dividing by \(m\) gives the lemma. \(\blacksquare\)

Step 3 actually gave something sharper and fully measured on the data: \(R(\mathcal H_2 \circ S) \le \sqrt{\sum_i \|\mathbf x_i\|^2}/m\), which uses the *average* message length, not the longest. The snippet makes 400 messages with between 5 and 25 words each, estimates \(\mathbb E_\sigma\|\sum_i\sigma_i\mathbf x_i\|/m\) from 300 sign draws, and compares it with both bounds, once with a 50-word vocabulary and once with 50,000 words.

```python
import math
import random

rng, m, draws = random.Random(5), 400, 300
for d in (50, 50_000):
    S = [rng.sample(range(d), rng.randint(5, 25))
         for _ in range(m)]
    total = 0.0
    for _ in range(draws):
        v = {}
        for x in S:
            s = rng.choice((-1, 1))
            for j in x:
                v[j] = v.get(j, 0) + s
        total += math.sqrt(sum(c * c for c in v.values())) / m
    jensen = math.sqrt(sum(len(x) for x in S)) / m
    worst = max(math.sqrt(len(x)) for x in S) / math.sqrt(m)
    print(d, round(total / draws, 4), round(jensen, 4), worst)
```

It prints `50 0.19 0.1941 0.25` and `50000 0.1958 0.1958 0.25`. A thousandfold larger vocabulary, and the complexity hardly moves: about 0.19 either way, under the data bound of about 0.195, under the worst-case 0.25.

**Lemma (\(\ell_1\) ball).** For any vectors \(\mathbf x_1, \dots, \mathbf x_m \in \mathbb R^n\),

\[ R(\mathcal H_1 \circ S) \le \max_i\|\mathbf x_i\|_\infty\sqrt{\frac{2\log(2n)}{m}}. \]

In plain words: **weights of total absolute size at most 1 fit noise no better than the largest single feature value times \(\sqrt{2\log(2n)/m}\).** The dimension \(n\) is back, but only inside a logarithm: for yes/no features, \(n = 50{,}000\) and \(m = 50{,}000\), it is \(\sqrt{2 \cdot 11.51/50{,}000} = 0.0215\).

*Proof.* *Step 1: Hölder’s inequality.* For any \(\mathbf w, \mathbf v\), \(\langle\mathbf w, \mathbf v\rangle = \sum_j w_jv_j \le \sum_j|w_j||v_j| \le \max_j|v_j|\sum_j|w_j| = \|\mathbf w\|_1\|\mathbf v\|_\infty\), and putting weight \(\pm 1\) on the largest entry of \(\mathbf v\) reaches it. So \(m R(\mathcal H_1 \circ S) = \mathbb E_\sigma\|\sum_i\sigma_i\mathbf x_i\|_\infty\).

*Step 2: rewrite as a finite set.* For each feature \(j \in [n]\), let \(\mathbf v_j = (x_{1,j}, \dots, x_{m,j}) \in \mathbb R^m\) be that feature’s column across the messages. Entry \(j\) of \(\sum_i\sigma_i\mathbf x_i\) is \(\langle\sigma, \mathbf v_j\rangle\), so \(\|\sum_i\sigma_i\mathbf x_i\|_\infty = \max_j|\langle\sigma, \mathbf v_j\rangle| = \max_{\mathbf v \in V}\langle\sigma, \mathbf v\rangle\) with \(V = \{\mathbf v_1, \dots, \mathbf v_n, -\mathbf v_1, \dots, -\mathbf v_n\}\). That is \(m R(V)\) for a set of \(2n\) vectors.

*Step 3: Massart.* The mean of \(V\) is \(\mathbf 0\), since every vector comes with its negative. Each column has \(\|\mathbf v_j\|_2 \le \sqrt m\max_i\|\mathbf x_i\|_\infty\), because it has \(m\) entries, each at most \(\max_i\|\mathbf x_i\|_\infty\) in absolute value. Massart’s lemma with \(N = 2n\) gives \(R(V) \le \sqrt m\max_i\|\mathbf x_i\|_\infty\sqrt{2\log(2n)}/m\), which is the claim. \(\blacksquare\)

```python
import math
import random

rng, m, draws = random.Random(5), 400, 300
for d in (50, 50_000):
    S = [rng.sample(range(d), rng.randint(5, 25))
         for _ in range(m)]
    total = 0.0
    for _ in range(draws):
        v = {}
        for x in S:
            s = rng.choice((-1, 1))
            for j in x:
                v[j] = v.get(j, 0) + s
        total += max(abs(c) for c in v.values()) / m
    bound = math.sqrt(2 * math.log(2 * d) / m)
    print(d, round(total / draws, 4), round(bound, 4))
```

It prints `50 0.0638 0.1517` and `50000 0.0075 0.2399`. The bound grows a little with the vocabulary, through \(\log(2d)\). The measured value goes *down*, to 0.0075: with 50,000 words each word appears in very few of the 400 messages, so no single column can line up with the noise. That is the Rademacher complexity doing what the VC dimension cannot, looking at the messages you have.

## Generalization bounds for SVM

Now assemble the pieces. In this section \(R\) stands for the radius of the messages, as in lesson 9: every \(\mathbf x\) has \(\|\mathbf x\| \le R\). (Lesson 15 wrote the radius as \(\rho\); here \(\rho\) is needed for the Lipschitz constant, so the radius goes back to \(R\). The Rademacher complexity keeps its brackets, \(R(\cdot)\), so the two never meet.) The class is \(\mathcal H = \{\mathbf w : \|\mathbf w\|_2 \le B\}\), the weight vectors within the **norm budget** \(B\), and the loss is a function of the score and the label: \(\ell(\mathbf w, (\mathbf x, y)) = \phi(\langle\mathbf w, \mathbf x\rangle, y)\), where for every label \(y\) the function \(a \mapsto \phi(a, y)\) is \(\rho\)-Lipschitz. The hinge loss, \(\phi(a, y) = \max\{0, 1 - ya\}\), and the ramp loss are both of this form with \(\rho = 1\) when \(y = \pm 1\).

**Theorem (norm-bounded linear predictors).** Suppose \(\|\mathbf x\|_2 \le R\) with probability 1 under \(\mathcal D\), and \(|\phi(a, y)| \le c\) for all \(a \in [-BR, BR]\). Then for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ \text{for all } \mathbf w \in \mathcal H: \quad L_{\mathcal D}(\mathbf w) \le L_S(\mathbf w) + \frac{2\rho BR}{\sqrt m} + c\sqrt{\frac{2\log(2/\delta)}{m}}. \]

Read it term by term: \(L_S(\mathbf w)\) is what you measure on your labels, \(2\rho BR/\sqrt m\) is the complexity term (Lipschitz constant, norm budget, radius, over \(\sqrt m\)), and the last term is the confidence term. In plain words: **every weight vector within the norm budget has true risk at most its training risk plus \(2\rho BR/\sqrt m\) plus a confidence term, and the number of features appears nowhere.** For the ramp loss (\(\rho = 1\), \(c = 1\)), \(B = 2\), \(R = 5\), \(m = 50{,}000\), \(\delta = 0.05\): \(20/223.6 + 0.0121 = 0.0894 + 0.0121 = 0.1016\) on top of the training risk.

*Proof.* Let \(\mathcal F\) be the loss class. Its vectors on \(S\) are \((\phi(\langle\mathbf w, \mathbf x_1\rangle, y_1), \dots, \phi(\langle\mathbf w, \mathbf x_m\rangle, y_m))\). Each is the score vector of \(\mathbf w\) with \(\phi_i(a) = \phi(a, y_i)\) applied to entry \(i\), and each \(\phi_i\) is \(\rho\)-Lipschitz, so the contraction lemma gives \(R(\mathcal F \circ S) \le \rho R(\mathcal H \circ S)\). The score vectors of the ball of radius \(B\) are \(B\) times those of the unit ball, so by the scale rule and the Euclidean-ball lemma, \(R(\mathcal H\circ S) = BR(\mathcal H_2 \circ S) \le BR/\sqrt m\). This holds for every \(S\) the distribution can produce, so also for the average over \(S'\). Every score satisfies \(|\langle\mathbf w, \mathbf x\rangle| \le BR\) by Cauchy–Schwarz, so every loss is at most \(c\) in absolute value, and part 1 of the Rademacher theorem applies. \(\blacksquare\)

Now the promise from lesson 15. There, hard-SVM was claimed to need a number of labels that grows with \((\rho/\gamma)^2\), the radius over the margin, and not with the dimension. Here is the proof, in this lesson’s letters.

**Theorem (hard-SVM).** Suppose some \(\mathbf w^\star\) has \(y\langle\mathbf w^\star, \mathbf x\rangle \ge 1\) with probability 1 under \(\mathcal D\), and \(\|\mathbf x\| \le R\) with probability 1. Let \(\mathbf w_S\) be the homogeneous hard-SVM solution, the shortest \(\mathbf w\) with \(y_i\langle\mathbf w, \mathbf x_i\rangle \ge 1\) for every \(i\). Then with probability at least \(1 - \delta\),

\[ \mathbb P_{(\mathbf x, y)\sim\mathcal D}\left[ y \ne \mathrm{sign}(\langle\mathbf w_S, \mathbf x\rangle) \right] \le \frac{2R\|\mathbf w^\star\|}{\sqrt m} + \sqrt{\frac{2\log(2/\delta)}{m}}. \]

The left side is the 0-1 true risk of hard-SVM, the chance it mislabels a fresh message. In plain words: **hard-SVM’s true risk is at most twice the radius times the separator’s length over \(\sqrt m\), plus a confidence term.** If the mail is separable with margin \(\gamma\) by a unit vector \(\mathbf u\), then \(\mathbf w^\star = \mathbf u/\gamma\) has \(y\langle\mathbf w^\star, \mathbf x\rangle \ge 1\) and \(\|\mathbf w^\star\| = 1/\gamma\), so the first term is \(2R/(\gamma\sqrt m) = \sqrt{4(R/\gamma)^2/m}\): lesson 15’s statement exactly.

*Proof.* Use the ramp loss: it is 1-Lipschitz in the score, takes values in \([0, 1]\) so \(c = 1\), and is at least the 0-1 loss (it charges 1 whenever \(y\langle\mathbf w, \mathbf x\rangle \le 0\)), so the 0-1 true risk is at most \(L_{\mathcal D}^{\text{ramp}}(\mathbf w_S)\). Take \(B = \|\mathbf w^\star\|\). With probability 1 every training example satisfies \(y_i\langle\mathbf w^\star, \mathbf x_i\rangle \ge 1\), so \(\mathbf w^\star\) is feasible for hard-SVM, and the shortest feasible vector has \(\|\mathbf w_S\| \le B\): \(\mathbf w_S\) is in the class. Every training example has signed score at least 1 under \(\mathbf w_S\), so its ramp loss is 0 and \(L_S^{\text{ramp}}(\mathbf w_S) = 0\). The norm-bounded theorem, which holds for every \(\mathbf w\) in the class at once, applies to \(\mathbf w_S\) in particular. \(\blacksquare\)

```python
import math

def hard_svm_bound(R_times_norm, m, delta):
    complexity = 2 * R_times_norm / math.sqrt(m)
    confidence = math.sqrt(2 * math.log(2 / delta) / m)
    return complexity + confidence

for m in (1_000, 10_000, 100_000):
    print(m, round(hard_svm_bound(10, m, 0.05), 4))
```

It prints `1000 0.7183`, `10000 0.2272` and `100000 0.0718`, the very numbers lesson 15 quoted for \(\rho/\gamma = 10\). The promise is kept.

There is one awkward thing about that theorem: \(\|\mathbf w^\star\|\) is the length of an unknown separator. Can the bound use the length of the vector you actually got? Yes, by spending confidence across norm budgets, the trick lesson 7 used for structural risk minimization.

**Theorem (a bound from the learned norm).** Under the same assumptions, with probability at least \(1 - \delta\), if \(\|\mathbf w_S\| \ge 2\),

\[ \mathbb P\left[ y \ne \mathrm{sign}(\langle\mathbf w_S, \mathbf x\rangle) \right] \le \frac{4R\|\mathbf w_S\|}{\sqrt m} + 2\sqrt{\frac{\log\left(4\log_2\|\mathbf w_S\|/\delta\right)}{m}}. \]

Here \(\log_2\) is the base-2 logarithm, so \(\log_2\|\mathbf w_S\|\) counts how many doublings of 1 reach the learned length. In plain words: **you pay twice the complexity term for not knowing the budget in advance, plus a confidence term that grows only like the log of the log of the length.**

*Proof.* For each integer \(i \ge 1\) let \(B_i = 2^i\), \(\mathcal H_i = \{\mathbf w : \|\mathbf w\| \le B_i\}\) and \(\delta_i = \delta/(2i^2)\). The norm-bounded theorem (ramp loss, \(c = \rho = 1\)) gives, for each \(i\), with probability at least \(1 - \delta_i\), \(L_{\mathcal D}(\mathbf w) \le L_S(\mathbf w) + 2B_iR/\sqrt m + \sqrt{2\log(2/\delta_i)/m}\) for every \(\mathbf w \in \mathcal H_i\). Since \(\sum_{i \ge 1}\delta/(2i^2) = \delta\pi^2/12 < \delta\), the union bound makes all of them hold at once with probability at least \(1 - \delta\). Given \(\mathbf w\) with \(\|\mathbf w\| \ge 2\), take \(i = \lceil\log_2\|\mathbf w\|\rceil\), the smallest \(i\) with \(2^i \ge \|\mathbf w\|\) (\(\lceil\cdot\rceil\) rounds up). Then \(\mathbf w \in \mathcal H_i\), \(B_i < 2\|\mathbf w\|\), and \(i < \log_2\|\mathbf w\| + 1 \le 2\log_2\|\mathbf w\|\) (because \(\log_2\|\mathbf w\| \ge 1\)), so \(2/\delta_i = (2i)^2/\delta \le (4\log_2\|\mathbf w\|)^2/\delta\). Then \(2\log(2/\delta_i) \le 4\log(4\log_2\|\mathbf w\|) + 2\log(1/\delta) \le 4\log(4\log_2\|\mathbf w\|/\delta)\), and its square root over \(m\) is the second term. Apply this to \(\mathbf w_S\), whose ramp training risk is 0. \(\blacksquare\)

```python
import math

def norm_bound(R, w_norm, m, delta):
    complexity = 4 * R * w_norm / math.sqrt(m)
    level = math.log(4 * math.log2(w_norm) / delta)
    return complexity + 2 * math.sqrt(level / m)

for w_norm in (2, 4, 8):
    print(w_norm, round(norm_bound(5, w_norm, 50_000, 0.05), 4))
```

It prints `2 0.1976`, `4 0.3779` and `8 0.7365`: with \(R = 5\) and 50,000 labels, a learned separator of length 2 is certified at about 0.2, while one of length 8 is not useful. Every number is computed from the training sequence alone.

None of these bounds mention the dimension of \(\mathbf w\). That is exactly why lesson 16 could run SVM in a feature space with a billion coordinates, or infinitely many, and still expect it to generalize: only the norm and the radius count.

## Predictors with low \(\ell_1\) norm

The same proof works for the \(\ell_1\) ball. Let \(\mathcal H = \{\mathbf w \in \mathbb R^d : \|\mathbf w\|_1 \le B\}\), with the loss as before.

**Theorem (\(\ell_1\)-bounded linear predictors).** Suppose \(\|\mathbf x\|_\infty \le R\) with probability 1 and \(|\phi(a, y)| \le c\) for \(a \in [-BR, BR]\). Then for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\),

\[ \text{for all } \mathbf w \in \mathcal H: \quad L_{\mathcal D}(\mathbf w) \le L_S(\mathbf w) + 2\rho BR\sqrt{\frac{2\log(2d)}{m}} + c\sqrt{\frac{2\log(2/\delta)}{m}}. \]

In plain words: **the same certificate, with the \(\ell_1\) budget, the largest feature value, and an extra \(\sqrt{2\log(2d)}\).** *Proof:* repeat the previous proof word for word, using the \(\ell_1\)-ball lemma instead of the Euclidean one, and Hölder’s inequality, \(|\langle\mathbf w, \mathbf x\rangle| \le \|\mathbf w\|_1\|\mathbf x\|_\infty \le BR\), for the bound on the loss. \(\blacksquare\)

The two theorems look alike, but \(B\) and \(R\) mean different things. In the Euclidean one, \(B\) bounds \(\|\mathbf w\|_2\) and \(R\) bounds \(\|\mathbf x\|_2\). In the \(\ell_1\) one, \(B\) bounds \(\|\mathbf w\|_1\), a stronger demand on the weights, while \(R\) bounds \(\|\mathbf x\|_\infty\), a much weaker demand on the messages. Which one wins depends on your prior knowledge about the messages and about good predictors. Take a predictor that puts weight 0.5 on each of \(k\) key words, so \(\|\mathbf w\|_2 = 0.5\sqrt k\) and \(\|\mathbf w\|_1 = 0.5k\), and messages with a given number of distinct words, so \(\|\mathbf x\|_2\) is the square root of that number and \(\|\mathbf x\|_\infty = 1\). The snippet prints the two complexity terms (with \(\rho = 1\)) at \(d = m = 50{,}000\).

```python
import math

d, m = 50_000, 50_000
for words, k in ((25, 1), (25, 10), (400, 10), (400, 40)):
    w2, w1 = 0.5 * math.sqrt(k), 0.5 * k
    l2 = 2 * w2 * math.sqrt(words) / math.sqrt(m)
    l1 = 2 * w1 * 1 * math.sqrt(2 * math.log(2 * d) / m)
    print(words, k, round(l2, 4), round(l1, 4))
```

It prints `25 1 0.0224 0.0215`, `25 10 0.0707 0.2146`, `400 10 0.2828 0.2146` and `400 40 0.5657 0.8584`. Short messages and a predictor spread over ten words: the Euclidean budget wins by a factor of three. Long messages of 400 words and ten key words: the \(\ell_1\) budget wins. Long messages but forty key words: Euclidean again. The \(\ell_1\) penalty that lesson 25 used to make the filter sparse pays off when messages are long and the truth uses few words.

## Estimating the Rademacher complexity as code

The data-dependent bound asks for \(R(\mathcal F \circ S)\), an average over \(2^m\) sign vectors. For \(m = 50{,}000\) nobody enumerates \(2^{50{,}000}\) of anything, but averages are what random sampling is good at: draw \(T\) sign vectors, compute the best fit for each, and average. The inner step, “best fit to these signs”, is an optimization over the class, and for the classes of this lesson it has a closed form.

```
EstimateRademacher(S, best_fit, T):
  total ← 0
  for t = 1..T:
    draw σ_1, ..., σ_m independent fair signs
    total ← total + best_fit(S, σ)
  return total / (T · m)

best_fit for a finite set A:  max over a in A of Σ_i σ_i a_i
best_fit for the l2 ball of radius B:  B · ‖Σ_i σ_i x_i‖_2
best_fit for the l1 ball of radius B:  B · ‖Σ_i σ_i x_i‖_∞
```

**Cost.** For a finite set of \(N\) vectors of length \(m\), each draw costs \(O(Nm)\), so \(O(TNm)\) time and \(O(m)\) extra space. For the norm balls over sparse messages with at most \(s\) words each, building \(\sum_i\sigma_i\mathbf x_i\) in a hash map costs \(O(ms)\) per draw, so \(O(Tms)\) time and \(O(\min(d, ms))\) space, independent of the vocabulary size \(d\). Exact enumeration costs \(O(2^m Nm)\), fine for \(m = 10\) and hopeless beyond about 25.

**How many draws?** Each draw’s value \(\frac1m\mathrm{best\_fit}\) lies in an interval of known length (for the \(\ell_2\) ball, \([0, BR]\), by the triangle inequality), and the draws are independent, so Hoeffding’s inequality tells you how large \(T\) must be for a given accuracy; that error has to be added to the certificate. The contraction lemma also means you never need the loss class itself: estimate the linear class and multiply by \(\rho\).

**Build it yourself.** Using only the standard library:

```python
import random

A = [(0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (1, 0, 0, 1, 0, 0)]
S = [[0, 1, 2], [1, 3], [2, 4, 5, 6], [0, 6], [7]]

def rademacher_exact(A): ...
def rademacher_mc(A, T, rng): ...
def l2_ball_mc(S, B, T, rng): ...
def massart_bound(A): ...
def certificate(ramp, B, R, m, delta): ...

if __name__ == "__main__":
    assert abs(rademacher_exact(A) - 0.1823) < 1e-4
    est = rademacher_mc(A, 20_000, random.Random(0))
    assert abs(est - 0.1823) < 0.01
    assert rademacher_exact(A) <= massart_bound(A)
    r = l2_ball_mc(S, 2.0, 20_000, random.Random(1))
    assert r <= 2.0 * 2.0 / len(S) ** 0.5
    c = certificate(0.035, 2, 5, 50_000, 0.05 / 8)
    assert abs(c - 0.1396) < 1e-4
```

1. Implement the five functions. `A` is a list of equal-length tuples; `S` is a list of messages, each the list of word indices present; `certificate` returns training ramp risk plus \(2BR/\sqrt m\) plus \(\sqrt{2\log(2/\delta)/m}\).
2. Reproduce 0.1823 for the three loss vectors, and check the Monte Carlo estimate against it for \(T = 100, 1{,}000, 10{,}000\).
3. Reproduce the Euclidean-ball experiment: 400 messages of 5 to 25 words, vocabularies of 50 and 50,000.
4. Use Hoeffding’s inequality to choose \(T\) so that the Monte Carlo estimate of \(R(\mathcal H_2\circ S)\) is within 0.005 of the truth with probability 0.99, when \(B = 1\) and \(R = 5\).

## Fix the bug

Back to operations and its promise. The setting: 50,000 word features; every message has at most 25 distinct words, so \(\|\mathbf x\|_2 \le 5\) and \(\|\mathbf x\|_\infty \le 1\); a budget of 50,000 labels; and the target is a certified 0-1 true risk of at most 0.15 with confidence 95%, for every weight vector the team might ship from the chosen class. The loss is the ramp loss, so \(c = \rho = 1\) and the 0-1 true risk is at most the ramp true risk.

The team offers eight classes: a Euclidean budget \(B \in \{1, 2, 4, 8\}\) or an \(\ell_1\) budget \(B \in \{1, 2, 4, 8\}\). For each, they trained on the 50,000 labels and measured the best training ramp risk in the class. A bigger budget fits better, so its ramp risk is smaller; its complexity term is larger. And here is the catch you must not miss: you will choose the class *after* seeing those eight numbers. A certificate for a class picked by looking is only honest if all eight certificates hold at once. So split the confidence, \(\delta/8\) each, and use the union bound, exactly as in the learned-norm theorem.

```python
import math

m, delta, d, R2, Rinf = 50_000, 0.05, 50_000, 5, 1
menu = {("l2", 1): 0.25, ("l2", 2): 0.035, ("l2", 4): 0.008,
        ("l2", 8): 0.004, ("l1", 1): 0.40, ("l1", 2): 0.12,
        ("l1", 4): 0.03, ("l1", 8): 0.01}
conf = math.sqrt(2 * math.log(2 * len(menu) / delta) / m)
for (norm, B), ramp in menu.items():
    if norm == "l2":
        cx = 2 * B * R2 / math.sqrt(m)
    else:
        cx = 2 * B * Rinf * math.sqrt(2 * math.log(2 * d) / m)
    print(norm, B, round(ramp + cx + conf, 4))
```

It prints `l2 1 0.3099`, `l2 2 0.1396`, `l2 4 0.2021`, `l2 8 0.377`, `l1 1 0.4581`, `l1 2 0.221`, `l1 4 0.2169` and `l1 8 0.3685`. Work out for yourself why the smallest training risk does not give the best certificate before you open the lab.

[[diagram:certificate]]

In the lab below, you choose the norm (Euclidean or \(\ell_1\)), the budget \(B\), and the number of labels up to 50,000. The certificate is computed from the theorem, split into its three terms, next to the VC route, which certifies nothing at any budget on the slider (37.1 at the full 50,000). You pass when the certificate is at most 0.15 within the budget; no random draw can pass for you.

[[lab-act2]]

## What people get wrong

**“The Rademacher complexity is a property of the class.”** It is a property of the class *on a training sequence*: \(R(\mathcal F\circ S)\) changes with \(S\). That is the whole point. The same class of 50,000-word halfspaces is extremely rich on adversarially chosen messages (VC dimension 50,000) and fits almost no noise on short messages with a short weight vector.

**“A small training risk means a good certificate.”** The certificate is a sum. In act 2 the budget \(B = 4\) has a training ramp risk of 0.008, but its complexity term is 0.179, and the certificate, 0.2021, misses the target that \(B = 2\) meets. Fit and richness pull in opposite directions, as they have since lesson 5.

**“I can pick the class after looking and keep the full \(\delta\).”** The theorem holds for one class fixed in advance. Choosing among eight after seeing their training risks requires all eight bounds at once, which is why the confidence is split into eight.

**“The \(\ell_1\) ball is always the right choice for sparse predictors.”** Its complexity term trades \(\|\mathbf x\|_2\) for \(\|\mathbf x\|_\infty\sqrt{2\log(2d)}\). For short messages, where \(\|\mathbf x\|_2\) is small anyway, the Euclidean budget can win even when the predictor is fairly sparse.

## Recap and the next bug

1. The **Rademacher complexity** \(R(\mathcal F\circ S)\) measures how well a class’s losses on your \(m\) examples can fit random signs. It is computed on the data in hand.
2. **Symmetrization**: the expected worst self-flattery of a class is at most twice its expected Rademacher complexity. The proof replaces the true risk by a ghost sample and the swaps by random signs.
3. **McDiarmid’s inequality**: a function of independent variables with bounded differences concentrates around its mean at the Hoeffding rate. With it, the bound holds with probability \(1 - \delta\) at the cost of \(\sqrt{\log(1/\delta)/m}\), and can even use the complexity measured on \(S\).
4. **Rademacher calculus**: shifting adds nothing and scaling multiplies by \(|c|\); convex hulls add nothing; finite sets cost \(\sqrt{2\log N}\) (Massart); \(\rho\)-Lipschitz losses multiply by at most \(\rho\) (contraction).
5. Euclidean-norm linear classes have complexity at most \(BR/\sqrt m\), and \(\ell_1\)-norm classes at most \(BR\sqrt{2\log(2d)/m}\). The resulting SVM and \(\ell_1\) bounds do not depend on the number of features, and they prove the hard-SVM bound of lesson 15.

The certificate ships. Then the data team comes back with a harder class. They want to certify a filter whose loss vectors are not a ball, not a finite set, and not a Lipschitz image of either: a two-layer filter from lesson 20, say. For a class like that the supremum inside the Rademacher complexity has no closed form, and Massart’s lemma, the one tool that works for any shape, applies only to finite sets. So the team asks: could a rich class be replaced, for every practical purpose, by a finite set of representative hypotheses, close to every member of the class? If every loss vector is within a small distance of one of the representatives, Massart’s lemma would apply to the representatives, and the small distance would cost only a little. How many representatives does it take? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Two extreme classes on a training sequence of \(m\) distinct messages, with the 0-1 loss. (a) A class with a single hypothesis: what is \(R(\ell\circ\mathcal H\circ S)\), and what does part 1 of the Rademacher theorem give? Compare it with Hoeffding’s inequality for one hypothesis from lesson 4. (b) A class that contains every labelling of the \(m\) messages: what is \(R(\ell\circ\mathcal H\circ S)\), and what does part 1 give at \(m = 1{,}000\), \(\delta = 0.05\)? Why is that the answer you should expect?
2. In the proof of the contraction lemma, the absolute value \(|a_1 - a'_1|\) is dropped in Step 4. Explain exactly why this does not change the supremum, and why the argument needs \(\mathbf a\) and \(\mathbf a'\) to range over the *same* set with the *same* signs \(\sigma_2, \dots, \sigma_m\) in both parts.
3. A single Euclidean class fixed in advance, no menu: \(R = 5\), \(B = 2\), training ramp risk 0.035, \(\delta = 0.05\). What is the smallest number of labels for which the norm-bounded theorem certifies a 0-1 true risk of at most 0.10?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (thresholds by Massart).** For the thresholds on the score line with the 0-1 loss, show that on any \(m\) messages the loss class produces at most \(m + 1\) distinct loss vectors, and use Massart’s lemma to show \(R(\ell\circ\mathcal H\circ S) \le \sqrt{2\log(m + 1)/m}\). Evaluate it at \(m = 1{,}000\).
- **Problem 2 (one-sided confidence).** Part 1 of the Rademacher theorem has \(\log(2/\delta)\) because it uses the two-sided form of McDiarmid’s inequality. Show that it holds with \(\log(1/\delta)\) instead.
- **Problem 3 (the Euclidean bound is tight).** Let \(\mathbf x_i = \mathbf e_i\), the \(i\)-th standard basis vector of \(\mathbb R^m\), for \(i = 1, \dots, m\). Compute \(R(\mathcal H_2\circ S)\) exactly and compare it with the bound \(\max_i\|\mathbf x_i\|/\sqrt m\).
