# Proving the Fundamental Theorem

## The auditor’s question

The filter is about to get its first outside audit. Since lesson 6 it has scored every message with the spam score, a number between 0 and 1, and it labels a message junk when the score falls inside an interval it has learned. The class of intervals has VC dimension 2, and the certificate your team filed back then rests on one sentence from lesson 6: the fundamental theorem of statistical learning, quantitative version. It says the number of labels a class needs grows like \(d + \log(1/\delta)\) over \(\varepsilon^2\) when labels are noisy, and roughly like \(d \log(1/\varepsilon) + \log(1/\delta)\) over \(\varepsilon\) when some interval is perfect.

The certificate has two lines. The clean line covers the mailboxes where one interval really does separate junk from keep. The noisy line covers the shared mailboxes, where users disagree and no interval is perfect. Both promise a filter within \(\varepsilon = 0.1\) of the best possible, with confidence \(1 - \delta = 0.95\), and both budgets were computed by plugging the numbers into the theorem’s shapes:

```python
from math import log

d, eps, delta = 2, 0.1, 0.05
agnostic_shape = (d + log(1 / delta)) / eps ** 2
realizable_shape = (d * log(1 / eps) + log(1 / delta)) / eps
print(round(agnostic_shape), round(realizable_shape))
```

It prints `500 76`. So the certificate says: 76 labels for the clean line, 500 for the noisy line.

The auditor reads it twice and puts down her pen. “Lesson 6 says there *are* constants \(C_1\) and \(C_2\) that make this true. You used 1 for both. Where is the proof that 1 works? And the lower constant: how do you know a smaller budget could not have done the job? If you can’t show me a floor and a ceiling, this certificate is a guess with a theorem’s name on it.”

She is right, and it is a little embarrassing. Lesson 6 stated the quantitative theorem and promised the proof for later. Your team has been quoting its rates ever since without anyone checking where they come from. This lesson pays that debt. We will prove three things, each with explicit constants:

1. **The ceiling with noisy labels**: enough labels, of order \((d \log(1/\varepsilon) + \log(1/\delta))/\varepsilon^2\), make ERM succeed for every distribution.
2. **The floor with noisy labels**: fewer than about \((d + \log(1/\delta))/\varepsilon^2\) labels make *every* learner fail on some distribution.
3. **The ceiling with clean labels**: when some hypothesis is perfect, order \((d \log(1/\varepsilon) + \log(1/\delta))/\varepsilon\) labels are enough, the faster \(1/\varepsilon\) rate.

By the end you will rewrite both lines of the certificate with numbers you can defend, and you will see why no budget below the floor could ever be defended at all.

## What exactly must be proved

Let’s set out the vocabulary, since every step below leans on it.

A **hypothesis class** \(\mathcal H\) is a set of hypotheses \(h: \mathcal X \to \{0, 1\}\), where \(\mathcal X\) is the domain (here, spam scores) and the label 1 means junk. Its **VC dimension** \(d\) (lesson 6) is the size of the largest set of messages that \(\mathcal H\) **shatters**, that is, labels in all \(2^{|C|}\) possible ways. For intervals, \(d = 2\): any two scores can be labelled every way, but no three can, since “junk, keep, junk” would need a gap inside an interval.

A training sequence \(S = (z_1, \dots, z_m)\) is \(m\) examples \(z_i = (x_i, y_i)\) drawn i.i.d. from a distribution \(\mathcal D\) over messages and labels; we write \(S \sim \mathcal D^m\). The **true risk** \(L_{\mathcal D}(h)\) is the chance that \(h\) mislabels a fresh example, and the **empirical risk** \(L_S(h)\) is the fraction of \(S\) that \(h\) gets wrong. The **best in the class** is \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\), the smallest true risk any hypothesis in \(\mathcal H\) can reach. When a learner \(A\) returns \(A(S)\), the amount by which it falls short of the best in the class,

\[ L_{\mathcal D}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D}(h), \]

is its **estimation error**, as in lesson 5. Here \(A(S)\) is the hypothesis the learner returns on \(S\), \(L_{\mathcal D}(A(S))\) is its true risk, and the \(\min\) is the best in the class. In plain words: **how much worse the learner’s hypothesis is than the best one it could have picked.** If the best interval errs on 12% of noisy mail and the learner’s errs on 15%, the estimation error is 0.03. A run is a **miss** when the estimation error exceeds \(\varepsilon\).

A class is **agnostic PAC learnable** (lesson 3) when some learner, given enough examples, avoids a miss with probability at least \(1 - \delta\), for every distribution \(\mathcal D\). The **sample complexity** \(m_{\mathcal H}(\varepsilon, \delta)\) is the smallest number of examples for which some learner can promise this. When some hypothesis has true risk 0, the setting is **realizable**, and the promise is simply \(L_{\mathcal D}(A(S)) \le \varepsilon\).

Two kinds of statement bracket the sample complexity, and it helps to name them now.

- A **ceiling** says: this many examples are enough, for one particular learner. It is an upper bound on \(m_{\mathcal H}(\varepsilon, \delta)\).
- A **floor** says: with fewer examples than this, *every* learner misses too often on some distribution. It is a lower bound on \(m_{\mathcal H}(\varepsilon, \delta)\).

A ceiling is proved by building a learner and analysing it. A floor is harder in spirit: you must defeat all learners at once, including ones nobody has invented yet. Lesson 5 did exactly this for the No-Free-Lunch theorem, and we will borrow its trick.

Throughout, \(\log\) is the natural logarithm, and \(e \approx 2.718\) is its base.

[[diagram:ceiling-floor]]

The picture above is the certificate’s noisy line on a number line of budgets. Below the floor, no learner can succeed, whatever it does. Above the ceiling, ERM is proved to succeed. In between lies a band where nobody has a proof either way, and an auditor cannot sign there. The whole lesson is about pinning those two fences down.

## The ceiling with noisy labels

### The plan, in words

Lesson 4 gave the strategy. Call a training sequence **\(\varepsilon\)-representative** when every hypothesis in the class has empirical risk within \(\varepsilon\) of its true risk. If \(S\) is \(\varepsilon/2\)-representative, then ERM, which returns a hypothesis with the smallest empirical risk, has estimation error at most \(\varepsilon\): its empirical risk is the lowest, and neither it nor the best hypothesis can be more than \(\varepsilon/2\) away from its true risk. So we need one thing: a number of examples that makes \(S\) representative with probability at least \(1 - \delta\).

Lesson 26 built the tool for this: Rademacher complexity. Let’s recall the three facts we need before we use them.

### Three facts from lesson 26

Fix the training sequence and line up, for each hypothesis, the vector of its losses on the \(m\) examples. A loss of 1 marks a mistake. Collect these vectors in a set \(A \subseteq \mathbb R^m\), the set of real vectors of length \(m\). The **random signs** \(\sigma_1, \dots, \sigma_m\) are independent, each \(+1\) or \(-1\) with chance one half, the same signs lesson 6 used for its ghost sample. The **Rademacher complexity** of \(A\) is

\[ R(A) = \frac{1}{m} \, \mathbb E_{\sigma} \left[ \sup_{a \in A} \sum_{i=1}^{m} \sigma_i a_i \right]. \]

From the inside out:

- \(a_i\) is coordinate \(i\) of a vector \(a\), the loss of one hypothesis on example \(i\).
- \(\sum_{i=1}^{m} \sigma_i a_i\) adds up the losses, each with a random sign. Think of the signs as splitting the examples into two random teams, and the sum as “mistakes on team plus minus mistakes on team minus”.
- \(\sup_{a \in A}\) (the supremum, which for a finite set is just the maximum) picks the vector that makes that difference as large as possible: the hypothesis that best “fits” the random split.
- \(\mathbb E_{\sigma}\) averages over all \(2^m\) equally likely sign vectors, and \(\frac{1}{m}\) turns the total into a per-example rate.

In plain words: **how well, on average, the class can match a random split of the training sequence into two teams.** A class that can fit any random split is rich and dangerous; a class that cannot is safe. On one example, with \(A = \{(0), (1)\}\), the sign is \(+1\) half the time (the best vector scores 1) and \(-1\) half the time (the best vector, \((0)\), scores 0), so \(R(A) = \tfrac12\).

The second fact turns this into a guarantee. Let \(\ell(h, z)\) be the loss of \(h\) on example \(z\), and suppose \(|\ell(h, z)| \le c\) for every \(h\) and \(z\), where the bars are absolute value. Write \(\ell \circ \mathcal H \circ S'\) for the set of loss vectors on a sequence \(S'\). Then, **with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H\) satisfies**

\[ L_{\mathcal D}(h) - L_S(h) \le 2 \, \mathbb E_{S' \sim \mathcal D^m} \left[ R(\ell \circ \mathcal H \circ S') \right] + c \sqrt{\frac{2 \log(2/\delta)}{m}}. \]

Reading it: the left side is how much hypothesis \(h\) flatters itself on \(S\), true risk minus empirical risk. On the right, \(S'\) is another training sequence drawn from \(\mathcal D^m\), and \(\mathbb E_{S'}\) averages the Rademacher complexity of the loss vectors over all such sequences; the factor 2 comes from lesson 26’s ghost-sample argument. The last term is the price of confidence: it shrinks like \(1/\sqrt m\) and grows only with the log of \(1/\delta\). In plain words: **no hypothesis can look better on the training sequence than it really is by more than twice the class’s average Rademacher complexity plus a small confidence term.** With the 0-1 loss, \(c = 1\); at \(m = 10{,}000\) and \(\delta = 0.05\), the confidence term is \(\sqrt{2 \log 40 / 10{,}000} \approx 0.027\).

The third fact, **Massart’s lemma**, bounds the Rademacher complexity of a finite set. If \(A = \{a_1, \dots, a_N\}\) has \(N\) vectors and \(\bar a = \frac1N \sum_{j=1}^{N} a_j\) is their average, then

\[ R(A) \le \max_{a \in A} \|a - \bar a\| \, \frac{\sqrt{2 \log N}}{m}. \]

Here \(\|a - \bar a\|\) is the Euclidean length of the vector from the average to \(a\), the square root of the sum of its squared coordinates, and \(\max_{a \in A}\) takes the longest one. In plain words: **a finite set of vectors can fit random signs only as well as its spread times the square root of the log of how many vectors there are.** The log is what makes this useful: a million hypotheses cost only \(\sqrt{2 \log 10^6} \approx 5.3\).

Now we can follow the chain.

[[diagram:ceiling-chain]]

### Step 1: as many loss vectors as labellings

Take eight messages with scores \(0, 1, \dots, 7\) (only their order matters) and some noisy labels, say junk at positions 2, 3 and 5. Each interval labels the eight messages somehow, and comparing with the users’ labels turns that labelling into a mistake pattern. How many different mistake patterns can intervals produce? Before you count, notice something: if you know the users’ labels, the labelling and the mistake pattern determine each other. A mistake at position \(i\) means “the opposite of \(y_i\)”, no mistake means “the same as \(y_i\)”. So there are exactly as many mistake patterns as labellings.

Formally, fix \(S = ((x_1, y_1), \dots, (x_m, y_m))\) and let

\[ A = \left\{ \left( \mathbb 1[h(x_1) \neq y_1], \dots, \mathbb 1[h(x_m) \neq y_m] \right) : h \in \mathcal H \right\}. \]

The indicator \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 otherwise, so each coordinate is the 0-1 loss of \(h\) on one example. The braces collect one vector per hypothesis, and the colon reads “such that \(h\) ranges over \(\mathcal H\)”; two hypotheses with the same mistakes give the same vector, counted once. In plain words: **\(A\) is the set of all mistake patterns the class can make on this training sequence.** The map from labelling \((h(x_1), \dots, h(x_m))\) to mistake pattern flips coordinate \(i\) exactly when \(y_i = 1\), and flipping twice undoes it, so the map is a bijection. Hence \(|A|\) equals the number of labellings the class produces on \(x_1, \dots, x_m\), which is at most the growth function \(\tau_{\mathcal H}(m)\) of lesson 6. By Sauer’s lemma, for \(m \ge d\),

\[ |A| \le \tau_{\mathcal H}(m) \le \left( \frac{em}{d} \right)^d. \]

On the left, the number of mistake patterns; in the middle, the most labellings of any \(m\) messages; on the right, Sauer’s polynomial, \(em/d\) raised to the power \(d\). In plain words: **a class of VC dimension \(d\) makes at most about \(m^d\) mistake patterns, not \(2^m\).** For intervals on eight messages, \((4e)^2 \approx 118\), against \(2^8 = 256\):

```python
from math import e

m, d = 8, 2
y = (0, 0, 1, 1, 0, 1, 0, 0)
cuts = [c - 0.5 for c in range(m + 1)]
H_C = {tuple(int(a < x < b) for x in range(m))
       for a in cuts for b in cuts if a <= b}
A = {tuple(int(h_i != y_i) for h_i, y_i in zip(h, y))
     for h in H_C}
print(len(H_C), len(A), round((e * m / d) ** d, 1))
```

It prints `37 37 118.2`: 37 labellings, 37 mistake patterns, exactly as the bijection says, and Sauer’s bound sits comfortably above both.

### Step 2: Massart turns the count into a complexity

Every coordinate of a vector in \(A\) is 0 or 1, and every coordinate of the average \(\bar a\) lies between 0 and 1, so every coordinate of \(a - \bar a\) lies between \(-1\) and \(1\). Its square is at most 1, the sum of \(m\) such squares is at most \(m\), and so \(\|a - \bar a\| \le \sqrt m\). Massart’s lemma then gives

\[ R(A) \le \sqrt m \cdot \frac{\sqrt{2 \log |A|}}{m} = \sqrt{\frac{2 \log |A|}{m}} \le \sqrt{\frac{2 d \log(em/d)}{m}}, \]

where the last step puts in Step 1’s \(|A| \le (em/d)^d\), whose log is \(d \log(em/d)\). In plain words: **the class’s Rademacher complexity on any training sequence is at most the square root of \(2d \log(em/d)\) over \(m\).** It does not depend on which sequence we drew, so it also bounds the average \(\mathbb E_{S'}\) in the second fact.

Let’s check it by brute force on ten messages, where all \(2^{10} = 1{,}024\) sign vectors can be tried:

```python
from itertools import product
from math import e, log, sqrt

m, d = 10, 2
y = (0, 0, 1, 1, 0, 1, 0, 0, 1, 0)
cuts = [c - 0.5 for c in range(m + 1)]
H_C = {tuple(int(a < x < b) for x in range(m))
       for a in cuts for b in cuts if a <= b}
A = {tuple(int(h != t) for h, t in zip(h_c, y)) for h_c in H_C}
signs = list(product([-1, 1], repeat=m))
R = sum(max(sum(s * a for s, a in zip(sig, v)) for v in A)
        for sig in signs) / len(signs) / m
massart = sqrt(2 * log(len(A)) / m)
sauer = sqrt(2 * d * log(e * m / d) / m)
print(len(A), round(R, 3), round(massart, 3), round(sauer, 3))
```

It prints `56 0.308 0.897 1.022`. The exact complexity is 0.308; Massart with the \(\sqrt m\) length gives 0.897, and Sauer’s count gives 1.022. Each step loosens the bound, which is the price of a proof that works for every class and every sequence. Ten messages is far too few for anything useful; the point is that the inequalities hold, and they shrink like \(\sqrt{\log m / m}\).

### Step 3: both sides at once

Put Step 2 into the second fact with \(c = 1\) and confidence \(\delta/2\). Since \(2\sqrt{x} = \sqrt{4x}\), twice the complexity bound is \(\sqrt{8 d \log(em/d)/m}\), and with probability at least \(1 - \delta/2\), every \(h\) has

\[ L_{\mathcal D}(h) - L_S(h) \le \sqrt{\frac{8 d \log(em/d)}{m}} + \sqrt{\frac{2 \log(4/\delta)}{m}}. \]

That is one side: no hypothesis looks much *better* on \(S\) than it is. We also need the other side, that none looks much *worse*. Here is the trick. Use the loss \(-\ell\) instead of \(\ell\). It is still bounded by \(c = 1\) in absolute value, its true and empirical risks are \(-L_{\mathcal D}(h)\) and \(-L_S(h)\), and its loss vectors are \(-A\). Flipping every vector does not change the Rademacher complexity, because \(\sigma\) and \(-\sigma\) are equally likely, so \(R(-A) = R(A)\). The same argument, again at confidence \(\delta/2\), gives the same bound for \(L_S(h) - L_{\mathcal D}(h)\).

Each of the two bounds fails with probability at most \(\delta/2\). By the **union bound** (lesson 2: the chance that at least one of several events happens is at most the sum of their chances), at least one fails with probability at most \(\delta\). So with probability at least \(1 - \delta\), both hold, and every \(h\) has

\[ |L_{\mathcal D}(h) - L_S(h)| \le \sqrt{\frac{8 d \log(em/d)}{m}} + \sqrt{\frac{2 \log(4/\delta)}{m}} \le 2 \sqrt{\frac{8 d \log(em/d) + 2 \log(4/\delta)}{m}}. \]

The last step uses \(\sqrt a + \sqrt b \le 2\sqrt{a + b}\), which holds because each of \(\sqrt a\) and \(\sqrt b\) is at most \(\sqrt{a + b}\). The bars on the left are absolute value: the gap between true and empirical risk, whichever way it points. In plain words: **with probability at least \(1 - \delta\), every interval’s training score is within this many points of its true score, all at once.** Here is the right-hand side for intervals at \(\delta = 0.05\):

```python
from math import e, log, sqrt

d, delta = 2, 0.05
def gap_bound(m):
    inside = 8 * d * log(e * m / d) + 2 * log(4 / delta)
    return 2 * sqrt(inside / m)
for m in (1_000, 10_000, 119_191, 547_743):
    print(m, round(gap_bound(m), 4))
```

It prints

```
1000 0.7048
10000 0.2538
119191 0.0821
547743 0.0405
```

At a thousand labels the bound is useless (0.70, more than any risk can differ), at ten thousand it says 0.25, and only in the hundreds of thousands does it get below 0.05. The last two rows are no accident; the next step explains where they come from.

### Step 4: solving for \(m\)

We want the bound to be at most a target \(\varepsilon\). Squaring and rearranging, that is

\[ \frac{m \varepsilon^2}{4} \ge 8 d \log\left(\frac{em}{d}\right) + 2 \log\left(\frac{4}{\delta}\right). \]

The trouble is that \(m\) appears on both sides, once inside a log. We need a way to pull it out, and a small lemma does it.

**The tangent-line lemma.** For all \(x > 0\) and \(t > 0\), \(\log x \le \log t + \frac{x}{t} - 1\).

The right side is the tangent line to the log curve at the point \(t\): a straight line with slope \(1/t\) that touches the curve at \(x = t\). In plain words: **the log curve never rises above any of its tangent lines.** At \(t = 7\) and \(x = 20\): \(\log 20 \approx 3.00\), while the tangent gives \(\log 7 + 20/7 - 1 \approx 3.80\).

*Proof.* Let \(g(x) = \log t + x/t - 1 - \log x\). Its derivative \(1/t - 1/x\) is negative for \(x < t\) and positive for \(x > t\), so \(g\) is smallest at \(x = t\), where \(g(t) = \log t + 1 - 1 - \log t = 0\). So \(g(x) \ge 0\) everywhere. \(\blacksquare\)

Now choose the tangent point cleverly. Apply the lemma with \(x = em/d\) and \(t = 64e/\varepsilon^2\), and multiply by \(8d\):

\[ 8d \log\left(\frac{em}{d}\right) \le 8d \log\left(\frac{64e}{\varepsilon^2}\right) + 8d \cdot \frac{em}{d} \cdot \frac{\varepsilon^2}{64 e} - 8d = 8d \log\left(\frac{64}{\varepsilon^2}\right) + \frac{m \varepsilon^2}{8}. \]

The middle term simplifies to \(m\varepsilon^2/8\), and \(\log(64e/\varepsilon^2) - 1 = \log(64/\varepsilon^2)\) because \(\log e = 1\). So the requirement is met as soon as

\[ \frac{m\varepsilon^2}{4} \ge 8d \log\left(\frac{64}{\varepsilon^2}\right) + \frac{m\varepsilon^2}{8} + 2 \log\left(\frac{4}{\delta}\right), \]

that is, as soon as \(m\varepsilon^2/8 \ge 8d \log(64/\varepsilon^2) + 2\log(4/\delta)\). Multiplying by \(8/\varepsilon^2\):

**The ceiling for uniform convergence.** Let \(\mathcal H\) have VC dimension \(d \ge 1\). For every distribution \(\mathcal D\), every \(\varepsilon, \delta \in (0, 1)\), and every

\[ m \ge \frac{64 \, d \log(64/\varepsilon^2) + 16 \log(4/\delta)}{\varepsilon^2}, \]

a training sequence \(S \sim \mathcal D^m\) is \(\varepsilon\)-representative with probability at least \(1 - \delta\).

Read it once more. The numerator has the VC dimension times a log of \(1/\varepsilon\), plus the log of \(1/\delta\); the whole is divided by \(\varepsilon^2\). In plain words: **about \(64 d \log(64/\varepsilon^2) + 16\log(4/\delta)\) labels per unit of \(\varepsilon^2\) make every hypothesis’s training score honest to within \(\varepsilon\).** (The value of \(m\) is far above \(d\), so Sauer’s form from Step 1 applies.)

To get the learning guarantee, remember that ERM needs representativeness at level \(\varepsilon/2\). Putting \(\varepsilon/2\) in place of \(\varepsilon\) multiplies \(64/\varepsilon^2\) by 4 inside the log and the whole fraction by 4:

**The ceiling with noisy labels.** ERM agnostically PAC learns \(\mathcal H\) with

\[ m_{\mathcal H}(\varepsilon, \delta) \le \frac{256 \, d \log(256/\varepsilon^2) + 64 \log(4/\delta)}{\varepsilon^2}. \]

For the certificate’s noisy line, \(d = 2\), \(\varepsilon = 0.1\), \(\delta = 0.05\):

```python
from math import log

def tangent(x, t):
    return log(t) + x / t - 1
xs = [k / 10 for k in range(1, 10_001)]
print(all(log(x) <= tangent(x, 7.0) + 1e-12 for x in xs))

def m_uc(eps, delta, d):
    return (64 * d * log(64 / eps**2)
            + 16 * log(4 / delta)) / eps**2
def m_agnostic(eps, delta, d):
    return m_uc(eps / 2, delta, d)
print(round(m_uc(0.1, 0.05, 2)), round(m_agnostic(0.1, 0.05, 2)))
```

It prints `True` (the log stays under its tangent at 10,000 test points) and `119191 547743`. Those are the two numbers from the table in Step 3: 119,191 labels make every interval honest to within 0.1, and 547,743 make it honest to within 0.05, which is what ERM needs for estimation error 0.1. The certificate claimed 500.

### What this ceiling does and does not prove

Look at the shape: \((d \log(1/\varepsilon) + \log(1/\delta))/\varepsilon^2\), up to constants. The theorem of lesson 6 promises \(d + \log(1/\delta)\) in the numerator, with no \(\log(1/\varepsilon)\) next to the \(d\). Our proof is off by that log factor. Removing it needs a sharper analysis of Rademacher complexity called **chaining**, which measures the class at many scales at once instead of counting its labellings at one; that proof is beyond this course. So be honest with the auditor: the ceiling we have proved is the theorem’s shape up to a factor of \(\log(1/\varepsilon)\), with explicit constants, and its constants are large. Large, but proved.

## The floor, part one: one message, two worlds

### A game you cannot win

Now switch sides. You are the adversary, and you want to embarrass every learner at once, using as few tools as possible. Here is the simplest weapon.

Pick one message, call it \(c\), that the class can label either way: some hypothesis \(h^+\) in \(\mathcal H\) calls it junk and some \(h^-\) calls it keep. (Any class with VC dimension at least 1 has such a message.) Now build two worlds. In both, every message that arrives is \(c\). In the **junk world**, users label \(c\) junk 55% of the time and keep 45% of the time. In the **keep world**, it is the other way round. The best hypothesis in the junk world labels \(c\) junk and errs 45% of the time; labelling it keep errs 55% of the time. So whichever world you are in, guessing its direction wrong costs exactly \(0.55 - 0.45 = 0.1\) in true risk.

The learner sees \(m\) labels of the same message and must decide which world it is in. That is all it can do. With 10 labels, how often do you think the best possible learner guesses wrong? Write down a number before you read on.

The natural guess is the majority vote: say junk if more than half the labels say junk, flip a coin on a tie. Let’s simulate it, and compute its failure exactly with the binomial distribution: the number of labels that point the wrong way is a sum of \(m\) independent coins that each come up “wrong” with chance 0.45.

```python
import random
from math import comb

def vote_loss(k, m):
    return 1 if 2 * k > m else 0.5 if 2 * k == m else 0
def exact_fail(m, eps):
    p = (1 - eps) / 2
    return sum(comb(m, k) * p**k * (1 - p)**(m - k)
               * vote_loss(k, m) for k in range(m + 1))
rng, eps, trials = random.Random(1), 0.1, 20_000
for m in (10, 80, 300):
    wrong = [sum(rng.random() < (1 - eps) / 2 for _ in range(m))
             for _ in range(trials)]
    sim = sum(vote_loss(k, m) for k in wrong) / trials
    print(m, round(sim, 3), round(exact_fail(m, eps), 3))
```

It prints

```
10 0.376 0.379
80 0.185 0.186
300 0.04 0.041
```

With 10 labels, the vote is wrong almost 38% of the time. With 80, still 19%. Only around 300 does it drop below 5%. Here `k` counts the labels pointing the wrong way, `vote_loss` is 1 when they win the vote and one half on a tie, and the simulation lands on the exact value each time.

Is the majority vote just a poor learner? No, and this is the heart of the floor: we will prove that *no* learner, however clever, can beat a certain bound in one of the two worlds. The lab lets you play the game first.

[[diagram:two-worlds]]

[[lab]]

### The two worlds, formally

Let \(\varepsilon \in (0, 1)\). For \(b \in \{0, 1\}\), the world \(\mathcal D_b\) puts all its probability on the two examples \((c, b)\) and \((c, 1 - b)\):

\[ \mathcal D_b\big((c, b)\big) = \frac{1 + \varepsilon}{2}, \qquad \mathcal D_b\big((c, 1 - b)\big) = \frac{1 - \varepsilon}{2}. \]

Here \(\mathcal D_b(\cdot)\) is the probability that world \(b\) gives to one example, \((c, b)\) is “message \(c\) with label \(b\)”, and \(1 - b\) is the other label. In plain words: **in world \(b\), message \(c\) carries label \(b\) a little more than half the time.** With \(\varepsilon = 0.1\), that is \(0.55\) and \(0.45\), the story above.

The true risk of any hypothesis \(h\) in world \(b\) depends only on its label at \(c\): if \(h(c) = b\), it errs exactly when the label is \(1 - b\), with chance \((1 - \varepsilon)/2\); otherwise it errs with chance \((1 + \varepsilon)/2\). Since \(h^+\) and \(h^-\) are in \(\mathcal H\), the best in the class is \((1 - \varepsilon)/2\), and so the estimation error of the learner’s output is

\[ L_{\mathcal D_b}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D_b}(h) = \varepsilon \cdot \mathbb 1\big[A(S)(c) \neq b\big]. \]

\(A(S)(c)\) is the label the learner’s hypothesis gives to \(c\), and the indicator is 1 when that label is not the world’s direction. In plain words: **the learner pays \(\varepsilon\) if it guesses the world wrong, and nothing if it guesses right.**

Every training sequence from either world is \(m\) copies of \(c\) with labels \(y = (y_1, \dots, y_m) \in \{0, 1\}^m\), so a learner is, for our purposes, just a guessing rule \(g\) that maps a label vector \(y\) to a guess \(g(y) \in \{0, 1\}\). (A learner that also flips its own coins is a mixture of such rules; every bound below holds for each rule, hence for the mixture.) Write \(P_b(y)\) for the chance of seeing \(y\) in world \(b\): each label equal to \(b\) contributes a factor \((1 + \varepsilon)/2\), each other label a factor \((1 - \varepsilon)/2\), and independence multiplies them.

### Step 1: one world must hurt

We cannot know which world the learner will face, but the adversary can choose the worse one, and the worse of two numbers is at least their average:

\[ \max_{b \in \{0,1\}} P_b[g(y) \neq b] \ge \frac12 \Big( P_1[g(y) = 0] + P_0[g(y) = 1] \Big) = \frac12 \sum_{y} \Big( P_1(y) \, \mathbb 1[g(y) = 0] + P_0(y) \, \mathbb 1[g(y) = 1] \Big) \ge \frac12 \sum_y \min\big(P_1(y), P_0(y)\big). \]

From left to right: \(P_b[g(y) \neq b]\) is the chance that rule \(g\) guesses wrong in world \(b\), and \(\max_b\) takes the worse world. The first inequality replaces the maximum by the average. The equality writes each probability as a sum over all \(2^m\) label vectors \(y\), keeping those on which \(g\) guesses wrong. The last step is the key: for each \(y\), exactly one of the two indicators is 1, so the bracket is either \(P_1(y)\) or \(P_0(y)\), and either way it is at least the smaller of the two, \(\min(P_1(y), P_0(y))\). In plain words: **whatever the rule does with a label vector, it is wrong in one of the worlds, and the overlap of the two worlds’ distributions is a toll no rule can dodge.** And the majority vote pays exactly this toll, because it always picks the world under which \(y\) is more likely, which is the world with the larger of the two probabilities.

### Step 2: the overlap lemma

So we need a lower bound on the overlap \(\sum_y \min(P_1(y), P_0(y))\). The tool is a quantity called the **Bhattacharyya coefficient**, \(B = \sum_y \sqrt{P_1(y) P_0(y)}\): add up the geometric means of the two probabilities. It is 1 when the distributions are identical and 0 when they never overlap.

**The overlap lemma.** For any two probability distributions \(P, Q\) on a finite set, with \(B = \sum_y \sqrt{P(y)Q(y)}\),

\[ \sum_y \min\big(P(y), Q(y)\big) \ge 1 - \sqrt{1 - B^2}. \]

On the left, the shared mass of the two distributions; on the right, a function of \(B\) that is 0 when \(B = 0\) and 1 when \(B = 1\). In plain words: **two distributions that are similar in the square-root sense must share a lot of mass.** With \(B = 0.951\), the right side is \(1 - \sqrt{1 - 0.904} \approx 0.69\).

*Proof.* Three short steps.

First, for two numbers, \(\min(p, q) = \frac12(p + q - |p - q|)\): the average minus half the gap. Summing over \(y\), and using that each distribution sums to 1, \(\sum_y \min(P, Q) = 1 - \frac12 \sum_y |P(y) - Q(y)|\).

Second, factor each gap as a difference of squares: \(|p - q| = |\sqrt p - \sqrt q| \, (\sqrt p + \sqrt q)\). Then apply the **Cauchy–Schwarz inequality**, which says that for any two lists of numbers, \(\sum_y u_y v_y \le \sqrt{\sum_y u_y^2} \, \sqrt{\sum_y v_y^2}\):

\[ \sum_y |P(y) - Q(y)| \le \sqrt{\sum_y \big(\sqrt{P(y)} - \sqrt{Q(y)}\big)^2} \, \sqrt{\sum_y \big(\sqrt{P(y)} + \sqrt{Q(y)}\big)^2}. \]

Third, expand the squares. \((\sqrt p - \sqrt q)^2 = p + q - 2\sqrt{pq}\), which sums to \(2 - 2B\); likewise \((\sqrt p + \sqrt q)^2\) sums to \(2 + 2B\). So \(\sum_y |P - Q| \le \sqrt{(2 - 2B)(2 + 2B)} = 2\sqrt{1 - B^2}\), and putting this into the first step gives \(\sum_y \min(P, Q) \ge 1 - \sqrt{1 - B^2}\). \(\blacksquare\)

For our two worlds, the coefficient is easy to compute, because a sum over all label vectors of a product over labels factors into a product of sums over single labels:

\[ B = \prod_{i=1}^{m} \left( \sqrt{\tfrac{1 + \varepsilon}{2} \cdot \tfrac{1 - \varepsilon}{2}} + \sqrt{\tfrac{1 - \varepsilon}{2} \cdot \tfrac{1 + \varepsilon}{2}} \right) = \left( \sqrt{1 - \varepsilon^2} \right)^m = (1 - \varepsilon^2)^{m/2}. \]

The \(\prod\) is a product over the \(m\) labels; for one label, whichever value it takes, the two worlds give it probabilities \((1+\varepsilon)/2\) and \((1-\varepsilon)/2\) in some order, so each label contributes twice the geometric mean of the two, \(\sqrt{1 - \varepsilon^2}\). In plain words: **each extra label pulls the two worlds apart by a factor of \(\sqrt{1 - \varepsilon^2}\).** With \(\varepsilon = 0.1\) and \(m = 10\), \(B = 0.99^5 \approx 0.951\).

Putting Steps 1 and 2 together: every rule, in one of the two worlds, guesses wrong with probability at least

\[ \frac12 \left( 1 - \sqrt{1 - (1 - \varepsilon^2)^m} \right). \]

Let’s check the lemma against the exact numbers, grouping label vectors by how many 1s they contain (all \(\binom{m}{k}\) vectors with \(k\) ones have the same probability):

```python
from math import comb, sqrt

def overlap(m, eps):
    hi, lo = (1 + eps) / 2, (1 - eps) / 2
    sum_min = bc = 0.0
    for k in range(m + 1):
        p1 = hi**k * lo**(m - k)
        p0 = lo**k * hi**(m - k)
        sum_min += comb(m, k) * min(p1, p0)
        bc += comb(m, k) * sqrt(p1 * p0)
    return sum_min, bc
for m in (10, 80):
    sum_min, bc = overlap(m, 0.1)
    print(m, round(sum_min / 2, 4), round(bc, 4),
          round((1 - 0.1**2) ** (m / 2), 4),
          round((1 - sqrt(1 - bc**2)) / 2, 4))
```

It prints `10 0.3786 0.951 0.951 0.3454` and `80 0.1857 0.669 0.669 0.1284`. The first column after \(m\) is half the overlap: 0.3786 and 0.1857, the majority vote’s exact failure rates from the simulation. So the majority vote pays exactly the toll. The next two columns agree: the coefficient computed by brute force equals \((1 - \varepsilon^2)^{m/2}\). The last column is the lemma’s floor, 0.345 and 0.128, safely below.

### Step 3: from a power to an exponential

The floor \((1 - \varepsilon^2)^m\) is exact but awkward. We trade it for an exponential using one inequality: for \(0 \le x \le \frac12\),

\[ \log(1 - x) \ge -\frac{x}{1 - x} \ge -2x. \]

The first step holds because \(\log \frac{1}{1 - x} = \log\left(1 + \frac{x}{1 - x}\right) \le \frac{x}{1 - x}\), using \(\log(1 + u) \le u\), which is the tangent-line lemma at \(t = 1\). The second holds because \(1 - x \ge \frac12\). So if \(\varepsilon^2 \le \frac12\), then \((1 - \varepsilon^2)^m \ge e^{-2m\varepsilon^2}\). The function \(u \mapsto \frac12(1 - \sqrt{1 - u})\) grows with \(u\), so every rule fails in one world with probability at least

\[ \frac12 \left( 1 - \sqrt{1 - e^{-2m\varepsilon^2}} \right). \]

In plain words: **the best possible failure rate shrinks with \(m\varepsilon^2\), not faster.** That product \(m\varepsilon^2\) is the whole story of the \(1/\varepsilon^2\) rate: to keep the failure fixed while halving \(\varepsilon\), you must quadruple \(m\).

### The floor theorem, part one

**The floor with noisy labels, confidence part.** Let \(\mathcal H\) have VC dimension at least 1, let \(0 < \varepsilon < 1/\sqrt2\) and \(0 < \delta < 1/4\). If

\[ m < \frac{\log\big(1/(4\delta)\big)}{2\varepsilon^2}, \]

then for every learner there is a distribution on which it misses with probability greater than \(\delta\). Hence \(m_{\mathcal H}(\varepsilon, \delta) \ge \log(1/(4\delta))/(2\varepsilon^2)\).

The numerator is the log of \(1/(4\delta)\), positive because \(\delta < 1/4\); the denominator is twice \(\varepsilon^2\). In plain words: **fewer than about half of \(\log(1/(4\delta))/\varepsilon^2\) labels cannot buy accuracy \(\varepsilon\) with confidence \(1 - \delta\), for any learner.** For the certificate, \(\log 5 / 0.02 \approx 80.5\).

*Proof.* One subtlety first. A miss means estimation error *greater than* \(\varepsilon\), but our two worlds charge exactly \(\varepsilon\) for a wrong guess, which is not a miss. So we build the worlds with a slightly larger gap. The right side of the condition shrinks continuously as \(\varepsilon\) grows, and it is strictly greater than \(m\) at \(\varepsilon\) itself, so we can pick \(\varepsilon'\) with \(\varepsilon < \varepsilon' < 1/\sqrt2\) and still \(m < \log(1/(4\delta))/(2\varepsilon'^2)\).

Build the two worlds with \(\varepsilon'\) in place of \(\varepsilon\). A wrong guess now costs \(\varepsilon' > \varepsilon\), a genuine miss. Since \(\varepsilon'^2 < \frac12\), Step 3 applies: in one of the worlds the learner guesses wrong with probability at least \(\frac12(1 - \sqrt{1 - e^{-2m\varepsilon'^2}})\).

The condition on \(m\) says \(2m\varepsilon'^2 < \log(1/(4\delta))\), so \(e^{-2m\varepsilon'^2} > 4\delta\), so \(1 - e^{-2m\varepsilon'^2} < 1 - 4\delta\), and the failure probability is strictly greater than \(\frac12(1 - \sqrt{1 - 4\delta})\).

Finally, \(\frac12(1 - \sqrt{1 - 4\delta}) \ge \delta\). Rearranged, that is \(\sqrt{1 - 4\delta} \le 1 - 2\delta\); both sides are nonnegative since \(\delta < 1/4\), so we may square: \(1 - 4\delta \le 1 - 4\delta + 4\delta^2\), which is true. So the learner misses with probability greater than \(\delta\). \(\blacksquare\)

```python
from math import exp, log, sqrt

xs = [k / 1000 for k in range(0, 501)]
print(all(log(1 - x) >= -2 * x for x in xs))
deltas = [k / 1000 for k in range(1, 250)]
print(all((1 - sqrt(1 - 4 * dl)) / 2 >= dl for dl in deltas))
eps, delta = 0.1, 0.05
print(round(log(1 / (4 * delta)) / (2 * eps**2), 2))
m = 80
floor_bound = (1 - sqrt(1 - exp(-2 * m * eps**2))) / 2
print(round(floor_bound, 4), floor_bound > delta)
```

It prints `True`, `True`, `80.47`, then `0.0533 True`. The two inequalities hold on a fine grid, the floor for the certificate is 80.47 labels, and at 80 labels the proved failure rate, 0.0533, is just above \(\delta = 0.05\). The exact failure of the best learner at 80 labels is 0.186, far above. The proof gives away a lot, but not so much that the floor becomes useless.

## The floor, part two: \(d\) messages, \(d\) coins

### Spreading the labels thin

The first floor used one message, so it could only charge for confidence: it explains the \(\log(1/\delta)\) but says nothing about \(d\). To make the VC dimension appear, the adversary uses all of it.

Take \(d\) messages that the class shatters. For each of them, flip a hidden coin to decide its direction, junk or keep, and let users label it in that direction a little more than half the time, with bias \(\rho\) (the Greek letter rho): in its direction with chance \((1 + \rho)/2\). Messages arrive uniformly at random. The learner’s \(m\) labels are now spread over \(d\) messages, about \(m/d\) each, and it must guess every message’s direction. Each wrong guess costs \(\rho/d\) in true risk.

Picture ten balance beams, one per message, with labels dropping onto them at random, each label tipping its beam a little toward junk or toward keep. With 30 labels, each beam gets about three. How many beams would you expect to tip the wrong way?

[[diagram:d-coins]]

```python
import random
from math import sqrt

d, rho, trials = 10, 0.2, 4_000
rng = random.Random(2)
def wrong_fraction(m):
    b = [rng.randint(0, 1) for _ in range(d)]
    votes = [0] * d
    for _ in range(m):
        i = rng.randrange(d)
        y = b[i] if rng.random() < (1 + rho) / 2 else 1 - b[i]
        votes[i] += 1 if y == 1 else -1
    guess = [int(v > 0) if v else rng.randint(0, 1)
             for v in votes]
    return sum(g != t for g, t in zip(guess, b)) / d
for m in (10, 30, 100):
    sim = sum(wrong_fraction(m) for _ in range(trials)) / trials
    bound = (1 - sqrt(2 * rho**2 * m / d)) / 2
    print(m, round(sim, 3), round(bound, 3))
```

It prints

```
10 0.433 0.359
30 0.368 0.255
100 0.265 0.053
```

The middle column is the fraction of beams the majority vote gets wrong, averaged over 4,000 games; a beam with no labels, or a tie, gets a coin flip. With 30 labels, more than a third of the beams are still wrong. The last column is the floor we are about to prove, \(\frac12(1 - \sqrt{2\rho^2 m/d})\): always below the simulation, and above one quarter as long as \(m\) is below \(d/(8\rho^2) = 31.25\). Hold on to that threshold.

### The \(d\) worlds, formally

Let \(C = \{c_1, \dots, c_d\}\) be shattered by \(\mathcal H\), and let \(0 < \rho \le 1/\sqrt2\). For each direction vector \(b = (b_1, \dots, b_d) \in \{0, 1\}^d\), the world \(\mathcal D_b\) is

\[ \mathcal D_b\big((c_i, b_i)\big) = \frac1d \cdot \frac{1 + \rho}{2}, \qquad \mathcal D_b\big((c_i, 1 - b_i)\big) = \frac1d \cdot \frac{1 - \rho}{2}, \qquad i \in [d]. \]

Here \([d] = \{1, \dots, d\}\), the \(\frac1d\) is the chance that message \(c_i\) is the one that arrives, and the second factor is the chance of each label given that it arrived. In plain words: **pick one of the \(d\) messages uniformly, then label it in its direction with chance \((1+\rho)/2\).** With \(d = 3\) and \(\rho = 0.2\), example \((c_1, b_1)\) has probability \(\frac13 \cdot 0.6 = 0.2\).

**The risk identity.** For every function \(f: \mathcal X \to \{0,1\}\),

\[ L_{\mathcal D_b}(f) - \min_{h \in \mathcal H} L_{\mathcal D_b}(h) = \rho \cdot \frac{\left| \{ i \in [d] : f(c_i) \neq b_i \} \right|}{d}. \]

The set in the numerator collects the messages on which \(f\) goes against the world’s direction, the colon reading “such that”, and the bars count them. In plain words: **the estimation error is \(\rho\) times the fraction of messages \(f\) gets backwards.**

*Proof.* Message \(c_i\) arrives with chance \(\frac1d\). If \(f(c_i) = b_i\), \(f\) errs on it with chance \((1 - \rho)/2\); otherwise with chance \((1 + \rho)/2\). So \(L_{\mathcal D_b}(f)\) is \(\frac1d\) times the sum of these over \(i\). No function can do better than \((1-\rho)/2\) on any message, and because \(C\) is shattered, some \(h \in \mathcal H\) has \(h(c_i) = b_i\) for every \(i\) and achieves \((1 - \rho)/2\) on all of them. So the best in the class is \((1 - \rho)/2\), and each backwards message adds \(\frac1d\left(\frac{1+\rho}{2} - \frac{1-\rho}{2}\right) = \frac{\rho}{d}\). \(\blacksquare\)

```python
d, rho = 3, 0.2
b, f = (1, 0, 1), (1, 1, 0)

def risk(g):
    total = 0.0
    for i in range(d):
        for y in (0, 1):
            p_y = (1 + rho) / 2 if y == b[i] else (1 - rho) / 2
            total += p_y / d * (g[i] != y)
    return total
mismatches = sum(f_i != b_i for f_i, b_i in zip(f, b))
print(round(risk(b), 4), round(risk(f), 4))
print(round(risk(f) - risk(b), 4), round(rho * mismatches / d, 4))
```

It prints `0.4 0.5333` and `0.1333 0.1333`: the best hypothesis errs 40% of the time, which is \((1 - 0.2)/2\); \(f\) gets two of the three messages backwards and errs 53.3%; the difference matches \(0.2 \cdot 2/3\).

### Step 1: average over the worlds

Here is lesson 5’s trick. The adversary may pick the worst world, and the worst is at least the average. Choose \(b\) uniformly at random from all \(2^d\) direction vectors; then

\[ \max_{b} \mathbb E_{S \sim \mathcal D_b^m}\left[ L_{\mathcal D_b}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D_b}(h) \right] \ge \mathbb E_{b} \, \mathbb E_{S \sim \mathcal D_b^m} \left[ \frac{\rho}{d} \sum_{i=1}^{d} \mathbb 1\big[A(S)(c_i) \neq b_i\big] \right] = \frac{\rho}{d} \sum_{i=1}^{d} \mathbb E_{b} \, \mathbb E_{S} \, \mathbb 1\big[A(S)(c_i) \neq b_i\big]. \]

On the left, the expected estimation error in the worst world; \(\mathbb E_{S \sim \mathcal D_b^m}\) averages over the training sequences of that world. In the middle, the risk identity has replaced the estimation error by \(\rho/d\) times the number of backwards messages, and \(\mathbb E_b\) averages over the uniformly random world. On the right, linearity of expectation (the average of a sum is the sum of the averages) splits the count into one term per message. In plain words: **the worst world costs at least the average world, and the average world’s cost is \(\rho/d\) times the expected number of messages guessed backwards.**

To sample \(S\) from \(\mathcal D_b\), first choose which message each example is, \(j = (j_1, \dots, j_m)\) with each \(j_r\) uniform on \([d]\), then draw each label \(y_r\) toward \(b_{j_r}\) with chance \((1 + \rho)/2\). The positions \(j\) do not depend on \(b\).

### Step 2: one message at a time, and the majority vote is best

Fix the positions \(j\) and one message \(i\). Let \(n_i\) be how many of the \(m\) examples landed on \(c_i\), let \(y^I\) be their labels, and let \(y^{\neg I}\) be all the other labels. Now an important observation: the labels \(y^I\) depend only on \(b_i\), and the other labels depend only on the other directions \(b_{\neg i}\). Since \(b\) is uniform, its coordinates are independent fair coins, so the pair \((b_i, y^I)\) is independent of \(y^{\neg I}\). The other messages’ labels tell the learner nothing about \(b_i\).

So, for each fixed value of the other labels, the learner faces exactly the two-world game of part one with \(n_i\) labels and gap \(\rho\): the direction \(b_i\) is a fair coin, the labels \(y^I\) lean toward it, and the learner’s answer at \(c_i\) is some guess. Step 1 of part one says the chance of a wrong guess, averaged over \(b_i\), is at least half the overlap, and the overlap does not depend on the other labels. Averaging over them changes nothing, so

\[ \mathbb E_{b} \, \mathbb E_{y} \, \mathbb 1\big[A(S)(c_i) \neq b_i\big] \ge \frac12 \sum_{y^I} \min\big(P(y^I \mid b_i = 1), P(y^I \mid b_i = 0)\big) \ge \frac12\left(1 - \sqrt{1 - e^{-2\rho^2 n_i}}\right). \]

\(P(y^I \mid b_i = 1)\) is the chance of the labels \(y^I\) when message \(i\) leans junk; the vertical bar reads “given”. The second inequality is the overlap lemma and the exponential step of part one, with \(n_i\) labels and \(\rho\) in place of \(\varepsilon\) (allowed because \(\rho^2 \le \frac12\)). When \(n_i = 0\), both sides are \(\frac12\): with no labels, you can only flip a coin.

The first inequality becomes an equality for one learner: the **majority vote**, which at each message guesses the direction that most of its labels point to, with a coin flip on a tie. As we saw, it picks the larger of the two probabilities for every \(y^I\), so it pays exactly the overlap. This rule is also called the **maximum-likelihood** learner, because it picks the direction under which the observed labels are most likely. In plain words: **on every message, the best any learner can do, on average over the worlds, is the majority vote, and even the majority vote errs at least this often.** You can check the “best” part by brute force: with three labels there are \(2^3 = 8\) label vectors and \(2^8 = 256\) deterministic guessing rules.

```python
from itertools import product

n, rho = 3, 0.2
ys = list(product([0, 1], repeat=n))
def p(y, b):
    hits = sum(t == b for t in y)
    return ((1 + rho) / 2)**hits * ((1 - rho) / 2)**(n - hits)
def error(g):
    return sum(0.5 * p(y, b) * (g[k] != b)
               for k, y in enumerate(ys) for b in (0, 1))
rules = list(product([0, 1], repeat=len(ys)))
best = min(error(g) for g in rules)
majority = tuple(int(2 * sum(y) > n) for y in ys)
print(len(rules), round(best, 4), round(error(majority), 4))
```

It prints `256 0.352 0.352`: of all 256 rules, the best errs 35.2% of the time, and the majority vote is that best.

### Step 3: from \(n_i\) to \(m/d\)

The bound depends on \(n_i\), which is random. Two small inequalities make it depend on \(m\) instead.

First, \(1 - e^{-a} \le a\) for \(a \ge 0\) (the exponential lies above its tangent line \(1 - a\)). With \(a = 2\rho^2 n_i\), \(\sqrt{1 - e^{-2\rho^2 n_i}} \le \sqrt{2\rho^2 n_i}\), so the per-message bound is at least \(\frac12(1 - \sqrt{2\rho^2 n_i})\).

Second, for any nonnegative random quantity \(X\), \(\mathbb E[\sqrt X] \le \sqrt{\mathbb E[X]}\). This is a case of **Jensen’s inequality** for the concave square root, and here is a one-line proof: the variance of \(\sqrt X\) is never negative, so \((\mathbb E \sqrt X)^2 \le \mathbb E[(\sqrt X)^2] = \mathbb E[X]\).

Each of the \(m\) examples lands on \(c_i\) with chance \(\frac1d\), so \(\mathbb E_j[n_i] = m/d\). Averaging the per-message bound over the positions \(j\):

\[ \mathbb E_j \left[ \frac12\left(1 - \sqrt{2\rho^2 n_i}\right) \right] = \frac12 \left( 1 - \sqrt{2\rho^2} \, \mathbb E_j\big[\sqrt{n_i}\big] \right) \ge \frac12\left(1 - \sqrt{\frac{2\rho^2 m}{d}}\right). \]

Now sum over the \(d\) messages in Step 1’s right-hand side: \(\frac{\rho}{d}\) times \(d\) copies of the same bound.

\[ \max_{b} \mathbb E_{S \sim \mathcal D_b^m}\left[ L_{\mathcal D_b}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D_b}(h) \right] \ge \frac{\rho}{2} \left( 1 - \sqrt{\frac{2\rho^2 m}{d}} \right). \]

In plain words: **in the worst of the \(d\)-coin worlds, every learner’s expected estimation error is at least \(\rho/2\) times one minus the square root of \(2\rho^2 m/d\).** The key quantity is \(\rho^2 m / d\): labels per message times the squared bias. If \(m < d/(8\rho^2)\), then \(2\rho^2 m/d < \frac14\), the square root is below \(\frac12\), and the expected estimation error exceeds \(\rho/4\). That is the 31.25 from the simulation, where the last column crossed one quarter.

### Step 4: from an average to a probability

An expected estimation error above \(\rho/4\) does not yet mean frequent misses: maybe the error is huge rarely and tiny usually. But it cannot be huge, because it is at most \(\rho\) (all messages backwards). A quantity trapped in a range with a high average must often be high. Here is the precise version, the same reasoning as lesson 5’s \(1/7\).

**A bounded-average lemma.** If \(Z\) takes values in \([0, 1]\) and \(0 < a < 1\), then \(\mathbb P[Z > a] \ge \dfrac{\mathbb E[Z] - a}{1 - a}\).

*Proof.* Let \(q = \mathbb P[Z > a]\). When \(Z \le a\) it contributes at most \(a\) to the average; when \(Z > a\) it contributes at most 1. So \(\mathbb E[Z] \le a(1 - q) + q\), which rearranges to \(q \ge (\mathbb E[Z] - a)/(1 - a)\). \(\blacksquare\)

Apply it to \(Z = \) estimation error divided by \(\rho\), which lies in \([0, 1]\) by the risk identity, in the worst world, with \(\mathbb E[Z] > \frac14\). Choose \(\rho = 8\varepsilon\) and \(a = \varepsilon/\rho = \frac18\):

\[ \mathbb P\left[ L_{\mathcal D_b}(A(S)) - \min_{h} L_{\mathcal D_b}(h) > \varepsilon \right] = \mathbb P\left[ Z > \tfrac18 \right] > \frac{\frac14 - \frac18}{1 - \frac18} = \frac17. \]

The condition \(\rho \le 1/\sqrt2\) becomes \(\varepsilon \le 1/(8\sqrt2) \approx 0.088\), and \(m < d/(8\rho^2)\) becomes \(m < d/(512\varepsilon^2)\).

**The floor with noisy labels, dimension part.** Let \(\mathcal H\) have VC dimension \(d \ge 1\), let \(0 < \varepsilon \le 1/(8\sqrt2)\) and \(0 < \delta < 1/7\). If \(m < d/(512\varepsilon^2)\), every learner misses with probability greater than \(1/7 > \delta\) on some distribution. Hence \(m_{\mathcal H}(\varepsilon, \delta) \ge d/(512\varepsilon^2)\).

In plain words: **fewer than \(d/(512\varepsilon^2)\) labels leave some world where every learner misses more than one time in seven.** The constant 512 is ugly, and the honest truth is that this proof spends constants freely; the shape \(d/\varepsilon^2\) is what matters.

### Both floors together

For \(\delta \le 1/8\), both parts apply, so \(m_{\mathcal H}(\varepsilon, \delta)\) is at least the larger of the two floors, and the larger of two numbers is at least their average. Also, \(\delta \le 1/8\) gives \(\log(1/\delta) \ge 3\log 2\), so \(\log 4 = 2\log 2 \le \frac23 \log(1/\delta)\) and \(\log(1/(4\delta)) \ge \frac13 \log(1/\delta)\). Then

\[ m_{\mathcal H}(\varepsilon, \delta) \ge \frac12 \left( \frac{\log(1/\delta)}{6\varepsilon^2} + \frac{d}{512\varepsilon^2} \right) \ge \frac{d + \log(1/\delta)}{1024\,\varepsilon^2}. \]

In plain words: **for small enough \(\varepsilon\) and \(\delta\), no learner can do with fewer than \((d + \log(1/\delta))/(1024\varepsilon^2)\) labels.** That is the floor of the fundamental theorem, with \(C_1 = 1/1024\).

```python
from math import log, sqrt

def floor_log(eps, delta):
    return log(1 / (4 * delta)) / (2 * eps**2)
def floor_d(eps, d):
    return d / (512 * eps**2)
markov = (1 / 4 - 1 / 8) / (1 - 1 / 8)
print(round(markov, 4), round(1 / (8 * sqrt(2)), 4))
eps, delta, d = 0.05, 0.05, 100
print(round(floor_log(eps, delta)), round(floor_d(eps, d)),
      round((d + log(1 / delta)) / (1024 * eps**2)))
```

It prints `0.1429 0.0884`, the \(1/7\) and the largest allowed \(\varepsilon\), then `322 78 40`: for a class with \(d = 100\) at \(\varepsilon = \delta = 0.05\), the confidence floor is 322 labels, the dimension floor 78, and the combined formula 40, a little weaker than either, as averaging promised. For the certificate’s \(\varepsilon = 0.1\), the dimension part does not apply (0.1 is above 0.088), but the confidence floor of 80.47 does.

## The ceiling with clean labels: \(\varepsilon\)-nets

### Catching every bad hypothesis

Now the clean line. Some interval \(c \in \mathcal H\) is perfect, and ERM returns a hypothesis with no mistakes on \(S\). When could that hypothesis still be bad?

Identify each hypothesis with the set of messages it calls junk. A hypothesis \(h\) disagrees with the truth \(c\) on the **mistake region** \(h \,\triangle\, c = \{x : h(x) \neq c(x)\}\), the **symmetric difference** of the two sets: messages in exactly one of them. Its true risk is the probability mass of that region, \(L_{\mathcal D}(h) = \mathcal D(h \,\triangle\, c)\), where \(\mathcal D(\cdot)\) now means the chance that a fresh message lands in the set.

[[diagram:symmetric-difference]]

A bad hypothesis, with true risk at least \(\varepsilon\), has a mistake region of mass at least \(\varepsilon\). If even one training message falls in that region, the hypothesis makes a mistake on \(S\), and ERM will not pick it. So ERM is safe as soon as the training messages hit every heavy mistake region. With noisy labels we needed every hypothesis’s score to be accurate; with clean labels, we only need every heavy region to be *hit*. That is a much weaker demand, and it is why the rate improves from \(1/\varepsilon^2\) to \(1/\varepsilon\).

Here is the idea for plain intervals of scores, with scores uniform on \([0, 1]\). Every interval of length at least \(\varepsilon = 0.1\) must contain a training score. That fails exactly when two neighbouring scores (or a score and an end of \([0,1]\)) are 0.1 or more apart. How many random scores do you need before that becomes rare?

```python
import random

def is_eps_net(points, eps):
    xs = [0.0] + sorted(points) + [1.0]
    return max(b - a for a, b in zip(xs, xs[1:])) < eps
rng, eps, trials = random.Random(3), 0.1, 10_000
for m in (20, 50, 100):
    hits = sum(is_eps_net([rng.random() for _ in range(m)], eps)
               for _ in range(trials))
    print(m, round(hits / trials, 3))
```

It prints `20 0.007`, `50 0.755` and `100 0.997`: with 100 scores, all heavy intervals are hit 99.7% of the time.

### The definition

**Definition (\(\varepsilon\)-net).** Let \(\mathcal R\) be a class of subsets of \(\mathcal X\) and \(\mathcal D\) a distribution over \(\mathcal X\). A sequence of points \(S\) is an **\(\varepsilon\)-net** for \(\mathcal R\) with respect to \(\mathcal D\) if

\[ \forall r \in \mathcal R: \quad \mathcal D(r) \ge \varepsilon \;\Rightarrow\; r \cap S \neq \emptyset. \]

\(\forall\) reads “for every”, \(\mathcal D(r)\) is the mass of the set \(r\), the arrow \(\Rightarrow\) reads “implies”, and \(r \cap S \neq \emptyset\) says that at least one point of \(S\) lies in \(r\) (\(\emptyset\) is the empty set). In plain words: **every heavy set in the class contains at least one of the points.** In the snippet, `is_eps_net` tests exactly this for intervals: every gap shorter than 0.1 means no interval of mass 0.1 fits between the points. Shattering and VC dimension make sense for any class of sets, by treating a set as the hypothesis that says 1 on it.

### The \(\varepsilon\)-net theorem

**The \(\varepsilon\)-net theorem.** Let \(\mathcal R\) have VC dimension \(d \ge 1\), let \(\varepsilon, \delta \in (0, 1)\), and let

\[ m \ge \frac{8}{\varepsilon} \left( 2d \log\left(\frac{16e}{\varepsilon}\right) + \log\left(\frac{2}{\delta}\right) \right). \]

Then, with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), \(S\) is an \(\varepsilon\)-net for \(\mathcal R\).

The bracket has the VC dimension times a log of \(1/\varepsilon\), plus the log of \(2/\delta\), and the whole is multiplied by \(8/\varepsilon\), not divided by \(\varepsilon^2\). In plain words: **about \(16 d \log(16e/\varepsilon)/\varepsilon\) random points, plus a confidence term, hit every heavy set of a class of VC dimension \(d\).** For intervals at \(\varepsilon = 0.1\), \(\delta = 0.05\), that is 2,239.2 points.

The proof has the same skeleton as the uniform-convergence proof in lesson 6: a ghost sample, then a swap. Let \(B\) be the bad event “\(S\) is not an \(\varepsilon\)-net”: some \(r \in \mathcal R\) has \(\mathcal D(r) \ge \varepsilon\) and contains no point of \(S\). Draw a **ghost sample** \(T \sim \mathcal D^m\), independent of \(S\), that nobody ever collects, and let \(B'\) be the event “some \(r \in \mathcal R\) has \(\mathcal D(r) \ge \varepsilon\), contains no point of \(S\), and contains more than \(\varepsilon m/2\) points of \(T\)”, counting points of \(T\) with repetition.

**Claim 1.** If \(\varepsilon m \ge 8\), then \(\mathbb P[B] \le 2\,\mathbb P[B']\).

The idea: a heavy set that \(S\) missed is very likely to be hit often by \(T\), because it is heavy.

*Proof.* \(B'\) can only happen when \(B\) does, so, averaging first over \(T\) for a fixed \(S\) (allowed because \(S\) and \(T\) are independent),

\[ \mathbb P[B'] = \mathbb E_{S}\Big[ \mathbb 1[S \in B] \cdot \mathbb P_{T}\big[(S, T) \in B'\big] \Big]. \]

Fix an \(S\) in \(B\), and pick one witness \(r_S\): a set with mass \(p = \mathcal D(r_S) \ge \varepsilon\) that \(S\) misses. If \(T\) puts more than \(\varepsilon m/2\) points in \(r_S\), then \((S, T)\) is in \(B'\). The number \(X\) of points of \(T\) in \(r_S\) is a sum of \(m\) independent coins with success chance \(p\): mean \(pm\), variance \(mp(1-p)\). Since \(pm/2 \ge \varepsilon m/2\), it suffices that \(X > pm/2\). By **Chebyshev’s inequality** (lesson 4: a quantity is at least \(t\) away from its mean with probability at most its variance over \(t^2\)),

\[ \mathbb P[X \le pm/2] \le \mathbb P\big[|X - pm| \ge pm/2\big] \le \frac{mp(1-p)}{(pm/2)^2} = \frac{4(1 - p)}{pm} < \frac{4}{\varepsilon m} \le \frac12. \]

So \(\mathbb P_T[(S, T) \in B'] \ge \frac12\) for every \(S\) in \(B\), and \(\mathbb P[B'] \ge \frac12 \mathbb P[B]\). \(\blacksquare\)

```python
from math import comb

def lower_tail(m, p):
    return sum(comb(m, k) * p**k * (1 - p)**(m - k)
               for k in range(m + 1) if k <= p * m / 2)
for m, p in ((80, 0.1), (80, 0.3), (200, 0.1)):
    print(m, p, round(lower_tail(m, p), 4),
          round(4 * (1 - p) / (p * m), 4))
```

It prints

```
80 0.1 0.088 0.45
80 0.3 0.0015 0.1167
200 0.1 0.0081 0.18
```

The exact chance that a set of mass \(p\) gets at most half its expected share of \(T\), next to Chebyshev’s bound. At \(\varepsilon m = 8\) (first row) the bound is 0.45, just under one half, which is all Claim 1 needs; the truth is 0.088.

**Claim 2 (the swap).** \(\mathbb P[B'] \le \tau_{\mathcal R}(2m) \, e^{-\varepsilon m/4}\), where \(\tau_{\mathcal R}\) is the growth function.

The idea is lesson 6’s: put all \(2m\) points in one row, \(S\) first and \(T\) second. Because they are i.i.d., any other way of splitting the row into two halves of \(m\) is equally likely to have produced it. And for a set that holds many of the \(2m\) points, it is very unlikely that a random half avoids all of them.

[[diagram:ghost-swap]]

*Proof.* Write the row as \(U = (u_1, \dots, u_{2m})\), with \(S\) the first \(m\) entries. Let \(\alpha = \varepsilon m/2\). If \((S, T) \in B'\), some \(r\) misses the first half and holds more than \(\alpha\) of the \(2m\) points (all of them in the second half). Forget the condition \(\mathcal D(r) \ge \varepsilon\); this only makes the event larger. What matters about \(r\) is only which positions of the row it contains, its **pattern**, and the class produces at most \(\tau_{\mathcal R}(2m)\) distinct patterns on \(2m\) points. By the union bound over patterns,

\[ \mathbb P[B'] \le \mathbb E_{U} \left[ \sum_{\text{patterns } P \text{ of } \mathcal R \text{ on } U} \mathbb 1\big[|P| \ge \alpha\big] \cdot \mathbb 1\big[P \text{ misses the first half}\big] \right]. \]

The sum runs over the distinct patterns, \(|P|\) counts the positions in a pattern, and each term is 1 for a heavy pattern that the first half misses. In plain words: **the bad event needs at least one heavy pattern to dodge \(S\) completely, and we add up the chances pattern by pattern.**

Now the swap. The entries of \(U\) are i.i.d., so permuting them gives a row with the same distribution, and the set of patterns and their sizes do not care about order. So the expectation is unchanged if “the first half” is replaced by any fixed set of \(m\) positions \(J\), and hence also by a uniformly random \(J\) among all \(\binom{2m}{m}\) of them. For a fixed row and a pattern with \(r \ge \alpha\) positions, the chance that a random \(J\) avoids all \(r\) of them is the chance of drawing \(m\) balls from a bag of \(2m\) without touching any of \(r\) red ones:

\[ \frac{\binom{2m - r}{m}}{\binom{2m}{m}} = \prod_{k=0}^{m-1} \frac{2m - r - k}{2m - k} \le \left(1 - \frac{r}{2m}\right)^m \le \left(1 - \frac{\varepsilon}{4}\right)^m \le e^{-\varepsilon m/4}. \]

The product draws the balls one at a time: the \(k\)-th draw avoids the red balls with chance \(1 - r/(2m - k) \le 1 - r/(2m)\). Then \(r \ge \alpha = \varepsilon m/2\) gives \(r/(2m) \ge \varepsilon/4\), and \(1 - x \le e^{-x}\) finishes. In plain words: **a random half of the row dodges every one of \(\varepsilon m/2\) marked points with chance at most \(e^{-\varepsilon m/4}\).** Each of the at most \(\tau_{\mathcal R}(2m)\) patterns contributes at most this much, which proves the claim. \(\blacksquare\)

```python
from math import comb, exp

def miss_all(m, r):
    return comb(2 * m - r, m) / comb(2 * m, m)
m, eps = 100, 0.1
r = round(eps * m / 2)
print(r, f"{miss_all(m, r):.4f}", f"{(1 - r / (2*m))**m:.4f}",
      f"{exp(-eps * m / 4):.4f}")
```

It prints `5 0.0297 0.0795 0.0821`: with 200 points and 5 marked, a random half dodges all five with chance 0.0297, under both bounds.

*Completing the proof.* Claims 1 and 2 and Sauer’s lemma (\(\tau_{\mathcal R}(2m) \le (2em/d)^d\)) give

\[ \mathbb P[S \text{ is not an } \varepsilon\text{-net}] \le 2 \left(\frac{2em}{d}\right)^d e^{-\varepsilon m/4}. \]

We need this to be at most \(\delta\), that is, \(\varepsilon m/4 \ge d \log(2em/d) + \log(2/\delta)\). Use the tangent-line lemma with \(x = 2em/d\) and \(t = 16e/\varepsilon\): \(d\log(2em/d) \le d\log(16/\varepsilon) + \varepsilon m/8\). So it is enough that \(\varepsilon m/8 \ge d\log(16/\varepsilon) + \log(2/\delta)\), that is, \(m \ge \frac{8}{\varepsilon}(d \log(16/\varepsilon) + \log(2/\delta))\). The \(m\) in the theorem is larger than this, since \(2d\log(16e/\varepsilon) > d\log(16/\varepsilon)\), so it qualifies. It also gives \(\varepsilon m \ge 16 d\log(16e) > 8\), as Claim 1 required. \(\blacksquare\)

```python
from math import e, exp, log

def failure_bound(m, eps, d):
    return 2 * (2 * e * m / d)**d * exp(-eps * m / 4)
def m_net(eps, delta, d):
    return 8 / eps * (2 * d * log(16 * e / eps) + log(2 / delta))
d, eps, delta = 2, 0.1, 0.05
print(round(m_net(eps, delta, d), 1))
for m in (500, 1_000, 2_240):
    print(m, f"{failure_bound(m, eps, d):.2g}")
```

It prints `2239.2`, then `500 14`, `1000 0.00021` and `2240 3.5e-17`. At 500 points the bound is 14, which says nothing (a probability is at most 1); at 2,240 it is astronomically small. The theorem’s \(m\) is generous, but it is what we proved, and a certificate quotes what was proved.

### From nets to learning

One gap remains. The \(\varepsilon\)-net theorem is about a class of sets, and we need it for the mistake regions \(h \,\triangle\, c\). Let \(\mathcal H_c = \{h \,\triangle\, c : h \in \mathcal H\}\), the class of all mistake regions against the truth \(c\).

Does \(\mathcal H_c\) have the same VC dimension as \(\mathcal H\)? Yes, and the reason is a bijection again. On a finite set \(C\), a pattern \(P\) of \(\mathcal H\) becomes the pattern \(P \,\triangle\, (c \cap C)\) of \(\mathcal H_c\): flip membership exactly at the points where \(c\) says junk. Flipping twice undoes it, so this is a bijection between the patterns of the two classes on \(C\). They produce the same number of patterns on every set, so they shatter the same sets and have the same VC dimension.

```python
m = 6
cuts = [c - 0.5 for c in range(m + 1)]
H_C = {tuple(int(a < x < b) for x in range(m))
       for a in cuts for b in cuts if a <= b}
c = (0, 1, 1, 1, 0, 0)
H_c = {tuple(h_i ^ c_i for h_i, c_i in zip(h, c)) for h in H_C}
print(len(H_C), len(H_c), (1, 0, 0, 0, 0, 0) in H_c)
```

It prints `22 22 True`. The operator `^` is exclusive or, the symmetric difference on bits. Intervals make 22 patterns on six points, and so do their mistake regions against \(c\), even though a mistake region need not be an interval: `(1, 0, 0, 0, 0, 0)` is the region of the interval that covers only the first point, and the regions of wider intervals that swallow \(c\) come in two pieces.

**The ceiling with clean labels.** Let \(\mathcal H\) have VC dimension \(d\), let \(\mathcal D\) be a distribution over \(\mathcal X\), and let the labels come from some \(c \in \mathcal H\). For \(\varepsilon, \delta \in (0,1)\) and \(m\) as in the \(\varepsilon\)-net theorem, with probability at least \(1 - \delta\) every ERM hypothesis has \(L_{\mathcal D}(h) < \varepsilon\). Hence

\[ m_{\mathcal H}(\varepsilon, \delta) \le \frac{8}{\varepsilon}\left( 2d \log\left(\frac{16e}{\varepsilon}\right) + \log\left(\frac{2}{\delta}\right) \right). \]

*Proof.* By the \(\varepsilon\)-net theorem for \(\mathcal H_c\), with probability at least \(1 - \delta\), \(S\) is an \(\varepsilon\)-net for \(\mathcal H_c\). On that event, take any \(h\) with \(L_{\mathcal D}(h) = \mathcal D(h \,\triangle\, c) \ge \varepsilon\): some training message lies in \(h \,\triangle\, c\), so \(h\) makes a mistake on \(S\), and \(L_S(h) > 0\). But \(c\) itself has \(L_S(c) = 0\), so ERM returns a hypothesis with empirical risk 0, and that hypothesis cannot be such an \(h\). Hence its true risk is below \(\varepsilon\). \(\blacksquare\)

In plain words: **with this many clean labels, every hypothesis consistent with them is good, because every bad one was caught red-handed.** The shape is \((d\log(1/\varepsilon) + \log(1/\delta))/\varepsilon\) up to constants (for \(\varepsilon \le 1/2\), where \(\log(1/\varepsilon)\) is at least \(\log 2\) and can absorb the constant \(\log(16e)\)), which is exactly the realizable ceiling of the fundamental theorem.

[[diagram:eps-net]]

## The majority vote, as code

The floor proofs introduced one learner, the majority vote, and showed it is the best possible in the \(d\)-coin worlds. Here it is with the game that evaluates it.

```
MajorityVote(S, d):
  votes ← array of d zeros
  for each example (i, y) in S:
    votes[i] ← votes[i] + (1 if y = 1 else −1)
  for i = 1..d:
    if votes[i] > 0: guess[i] ← 1
    else if votes[i] < 0: guess[i] ← 0
    else: guess[i] ← a fair coin flip
  return guess

EstimationError(guess, b, ρ):
  wrong ← number of i with guess[i] ≠ b[i]
  return ρ · wrong / d
```

Here an example is written \((i, y)\): the index \(i\) of the shattered message it landed on, and its label. **MajorityVote** reads each example once and keeps one counter per message: \(O(m + d)\) time and \(O(d)\) space. **EstimationError** is \(O(d)\). The exact failure rate of the vote on one message with \(n\) labels, which the floor proofs bound from below, is a sum over the \(n + 1\) possible counts of wrong labels: \(O(n)\) terms. For the two-world game it is the `exact_fail` function above.

**Build it yourself.** Using only the Python standard library:

```python
import random
from math import comb, sqrt

S = [(0, 1), (2, 0), (0, 1), (1, 1), (2, 0),
     (0, 0), (1, 0), (2, 1), (1, 1)]
d, rho = 3, 0.2

def majority_vote(S, d, rng): ...
def estimation_error(guess, b, rho): ...
def exact_point_error(n, rho): ...
def coin_floor(n, rho): ...

if __name__ == "__main__":
    assert majority_vote(S, d, random.Random(0)) == [1, 1, 0]
    err = estimation_error([1, 1, 0], (1, 0, 1), rho)
    assert abs(err - rho * 2 / 3) < 1e-12
    assert exact_point_error(0, rho) == 0.5
    assert round(exact_point_error(3, rho), 4) == 0.352
    assert all(exact_point_error(n, r) >= coin_floor(n, r) - 1e-12
               for n in range(60) for r in (0.1, 0.3, 0.7))
```

1. Implement `majority_vote` (the pseudocode, ties broken with `rng`), `estimation_error` (the risk identity), `exact_point_error(n, rho)` (the majority vote’s exact chance of guessing one message wrong from \(n\) labels of bias \(\rho\), counting a tie as one half), and `coin_floor(n, rho)`, the overlap lemma’s \(\frac12(1 - \sqrt{1 - (1 - \rho^2)^n})\).
2. Run the asserts. The last one is the overlap lemma, checked exactly for 180 cases.
3. Reproduce the \(d\)-coin table (10, 30 and 100 labels on ten messages) with your functions.
4. In the two-world game with \(\varepsilon = 0.1\), find the smallest \(m\) for which the majority vote’s failure is at most 0.05. Compare it with the proved floor of 80.47 and explain the gap.

## Fix the bug

Back to the auditor. Here is everything we proved, for her numbers: intervals, \(d = 2\), \(\varepsilon = 0.1\), \(\delta = 0.05\).

```python
from math import ceil, e, log

d, eps, delta = 2, 0.1, 0.05
floor = log(1 / (4 * delta)) / (2 * eps**2)
agnostic = (256 * d * log(256 / eps**2)
            + 64 * log(4 / delta)) / eps**2
realizable = 8 / eps * (2 * d * log(16 * e / eps)
                        + log(2 / delta))
print(ceil(floor), ceil(agnostic), ceil(realizable))
```

It prints `81 547743 2240`. Read it as the auditor will.

- **Noisy line, floor**: no learner can certify \(\varepsilon = 0.1\) at \(\delta = 0.05\) with 80 labels or fewer. The current 500 is above the floor, so it is not ruled out.
- **Noisy line, ceiling**: 547,743 labels are proved to be enough for ERM. The current 500 is in the band where nobody has a proof. Not certifiable.
- **Clean line, ceiling**: 2,240 labels are proved to be enough. The current 76 is not certifiable either.

In the lab below, the certificate has both lines on sliders, and the company will pay for at most 600,000 labels in total. Before you move anything, answer one question: finance proposes cutting the noisy line to 60 labels to save money. What does the floor say about that? Then set a budget for each line that the auditor can sign.

[[lab-act2]]

## What people get wrong

**“The ceiling is the number of labels you need.”** The ceiling is the number of labels this proof can vouch for. The two-world game needs about 269 labels in truth, the proved floor says 81, and the proved ceiling for intervals says 547,743. The truth sits somewhere between the fences, and the fences are far apart because the proofs spend constants freely.

**“The floor only applies to bad learners.”** The floor applies to every learner, because the adversary picks the world after the learner is fixed. The majority vote, the best learner in those worlds, still misses more than \(\delta\) of the time below the floor.

**“Clean labels just make the same bound smaller.”** They change the rate. With noise, the learner must estimate every hypothesis’s risk to within \(\varepsilon\), which costs \(1/\varepsilon^2\). Without noise, it only needs every heavy mistake region to be hit once, which costs \(1/\varepsilon\). At \(\varepsilon = 0.1\) that is the difference between 2,240 and 547,743.

**“The proof shows the theorem exactly as stated.”** The noisy ceiling we proved has an extra \(\log(1/\varepsilon)\) next to \(d\). The theorem’s cleaner shape needs chaining, which we did not prove.

## Recap and the next bug

1. **The noisy ceiling**: mistake patterns are as many as labellings, Sauer bounds them by \((em/d)^d\), Massart turns that into a Rademacher complexity of order \(\sqrt{d\log(m)/m}\), and the generalization bound of lesson 26, applied to both signs of the loss, makes every hypothesis honest to within \(\varepsilon\) once \(m\) is of order \((d\log(1/\varepsilon) + \log(1/\delta))/\varepsilon^2\).
2. **The confidence floor**: two worlds that differ only in the direction of one message, and the overlap lemma, show that every learner needs at least \(\log(1/(4\delta))/(2\varepsilon^2)\) labels.
3. **The dimension floor**: \(d\) shattered messages with hidden directions, averaged over all \(2^d\) worlds, show that every learner needs at least \(d/(512\varepsilon^2)\) labels; the majority vote is the best learner there and still fails.
4. **The clean ceiling**: an \(\varepsilon\)-net hits every heavy mistake region, a ghost sample and a swap show that order \((d\log(1/\varepsilon) + \log(1/\delta))/\varepsilon\) random points form one, and then every consistent hypothesis is good.
5. Together: the fundamental theorem’s floor with \(C_1 = 1/1024\), its realizable ceiling, and its noisy ceiling up to a \(\log(1/\varepsilon)\).

The auditor signs. You have barely put the certificate away when the product team walks in with a new plan. The filter will no longer sort mail into junk and keep: it will sort every message into one of hundreds of folders, invoices, newsletters, travel, receipts, each team’s project. A hypothesis now returns one of hundreds of labels, not two, and the VC dimension, which counts the ways to label points with 0 and 1, no longer applies. What replaces it? And which multiclass problems are learnable at all? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the confidence floor, the theorem requires \(\varepsilon < 1/\sqrt2\) and \(\delta < 1/4\), and the proof builds its two worlds with a gap \(\varepsilon'\) slightly larger than \(\varepsilon\). For each of these three choices, say which step of the proof would break without it.
2. In the dimension floor, the learner sees the labels of all \(d\) messages, yet the proof treats each message as if only its own labels mattered. Why are the labels of the other messages useless for guessing the direction \(b_i\)? Which assumption about how \(b\) is chosen makes this true, and would the argument survive if the adversary always chose \(b = (1, 1, \dots, 1)\)?
3. For a class with \(d = 2\), take \(\varepsilon = 0.05\) and \(\delta = 0.01\). Compute the confidence floor, the dimension floor, the noisy ceiling and the clean ceiling. Which floor is larger, and by how much does the noisy ceiling exceed the clean one?

## Problems

These problems were written for this lesson. Work them with pencil and Python; hints are available on request, never solutions.

1. **A floor for clean labels.** Suppose \(\mathcal H\) contains two hypotheses that agree on a message \(x_0\) and disagree on a message \(x_1\). Put mass \(1 - 2\varepsilon\) on \(x_0\) and \(2\varepsilon\) on \(x_1\), and let either hypothesis be the truth. Show that for \(\varepsilon \le 1/4\) and \(\delta < 1/2\), every learner needs \(m \ge \log(1/(2\delta))/(4\varepsilon)\) clean labels. Why does this floor scale like \(1/\varepsilon\) and not \(1/\varepsilon^2\)?
2. **A cheaper noisy ceiling.** In Step 3 of the noisy ceiling we used \(\sqrt a + \sqrt b \le 2\sqrt{a + b}\). Prove the sharper \(\sqrt a + \sqrt b \le \sqrt{2(a + b)}\), redo Step 4 with it, and recompute the certificate’s noisy line. How many labels does the auditor save?
3. **(Harder.) The floor is nearly tight for its own worlds.** In the \(d\)-coin worlds, bound the majority vote’s expected estimation error from above, using Hoeffding’s inequality on each message and the fact that each \(n_i\) is a sum of \(m\) independent coins with chance \(1/d\). Show that order \(d\log(1/\varepsilon)/\varepsilon^2\) labels make it at most \(\varepsilon\), so the \(d/\varepsilon^2\) floor cannot be improved by more than a log in these worlds.
