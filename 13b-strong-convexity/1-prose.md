# Strong Convexity and the Fitting-Stability Tradeoff

## The certificate that came from a simulator

Lesson 13a ended with a fix you could defend. The ridge filter with \(\lambda = 0.1\) sat still from night to night, and you backed it with two numbers: averaged over 20,000 simulated nights of twenty labels, its training fit was 0.311 and its replace-one change 0.026, so its overfitting gap was 0.026 and its expected production error about 0.337. Both promises kept.

Then the auditor read the certificate and asked one question: where did the 20,000 nights come from? From the simulator, the little program that draws campaign messages by flipping the campaign’s own coins. “So your certificate is a fact about the simulator,” she said. “The real campaign does not come with its coins.”

Could you measure the replace-one change on real mail instead? Each swap costs one fresh labelled message and one full retrain. The label budget has just grown to 2,000 messages a night, and at that size, with \(\lambda = 0.18\), a single swap changes the loss by about 0.00023 on average, while individual swaps scatter around that by about 0.00037. To pin the average down to within a tenth of itself, you need about \((0.00037/0.000023)^2 \approx 260\) swaps: 260 extra labels and 260 retrains of a 2,000-message filter every night, just to check the filter. And the answer would still say nothing about next month’s campaign, whose coins nobody knows either.

So here is the bug. The fix works, but the only evidence for it comes from a world you made up. The manager wants a number she can sign *before* the data arrive: a bound on the expected production error that holds for every distribution, computed only from things you actually know, namely \(\lambda\), the label budget \(m\), one property of the loss, and the training fit you see every night.

That needs a proof that the penalty makes the learner stable, whatever the distribution. This lesson builds it. The proof rests on one property of the penalty, strong convexity, and it ends in a certificate you can compute without a simulator.

## What you need from lesson 13a

Here is the toolkit, one line each, so you can check that nothing has slipped.

- **The campaign world.** Five near-twin words; a message is \(\mathbf x \in \mathbb R^5\) with \(x_j = f + e_j\), where the shared signal \(f\) is \(\pm 0.4\) and points the right way 80% of the time, and each private jitter \(e_j\) is \(\pm 0.04\). Every message has \(\|\mathbf x\| \le 0.98\), and the best weight vector, 0.2994 on every word, has production error 0.320.
- **The one-word world.** The same campaign with only the word “free”: \(x = 0.4\,y\) with probability 0.8 and \(-0.4\,y\) otherwise, with true risk \(L_{\mathcal D}(w) = \tfrac12(0.16 w^2 - 0.48 w + 1)\).
- **The half square loss.** \(\ell(\mathbf w, z) = \tfrac12(\langle \mathbf w, \mathbf x \rangle - y)^2\) for one example \(z = (\mathbf x, y)\): half the squared distance between the score and the label.
- **The two risks.** \(L_S(\mathbf w)\), the empirical risk, is the average loss over the training sequence \(S = (z_1, \dots, z_m)\); \(L_{\mathcal D}(\mathbf w)\), the true risk, is the loss averaged over the distribution \(\mathcal D\). The story calls the true risk of the night’s filter its **production error**.
- **The regularized learner.** \(f_S(\mathbf w) = L_S(\mathbf w) + \lambda\|\mathbf w\|^2\) is the regularized objective, \(\lambda > 0\) the regularization weight, and the RLM rule returns \(A(S) = \operatorname{argmin}_{\mathbf w} f_S(\mathbf w)\), the weight vector where \(f_S\) is smallest. With the half square loss, this is **ridge regression**, and its weights solve \((2\lambda m I + A)\mathbf w = \mathbf b\), with \(A = \sum_i \mathbf x_i \mathbf x_i^\top\) and \(\mathbf b = \sum_i y_i \mathbf x_i\) (the matrix \(A\) and the learner \(A(S)\) share a letter; the learner always carries its argument).
- **The three-example world.** The one-word world with \(S = ((0.4, 1), (-0.4, -1), (0.4, -1))\) and \(\lambda = 0.05\). Ridge returns \(w = 0.5128\).
- **The replace-one sequence.** \(S^{(i)}\) is \(S\) with its \(i\)th example \(z_i\) replaced by a fresh example \(z' \sim \mathcal D\), drawn independently of \(S\). The **replace-one change** is \(\ell(A(S^{(i)}), z_i) - \ell(A(S), z_i)\): how much worse the learner does on \(z_i\) once it no longer trains on it. In the three-example world, replacing \(z_3\) by \((0.4, 1)\) moves ridge to 1.5385, and the change on \(z_3\) is 0.5786.
- **Stability.** A learner is **on-average-replace-one-stable with rate \(\varepsilon(m)\)** when, for every distribution, the expected replace-one change is at most \(\varepsilon(m)\), a number that never grows as \(m\) grows.

And the one identity that holds it all together, the theorem that **stable rules do not overfit**. For every learner \(A\) and every distribution \(\mathcal D\), with \(S \sim \mathcal D^m\), \(z' \sim \mathcal D\) and \(i\) uniform on \([m] = \{1, \dots, m\}\), all independent,

\[ \mathbb E_{S} \left[ L_{\mathcal D}(A(S)) - L_S(A(S)) \right] = \mathbb E_{(S, z') \sim \mathcal D^{m+1},\; i \sim U(m)} \left[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \right]. \]

Read it from the inside out.

- \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) examples drawn i.i.d. from \(\mathcal D\)”, and \((S, z') \sim \mathcal D^{m+1}\) adds the fresh example as one more independent draw.
- \(i \sim U(m)\) picks the replaced position uniformly, each with probability \(1/m\).
- On the left, \(L_{\mathcal D}(A(S)) - L_S(A(S))\) is how much the learner’s output looks better on its training sequence than it really is, and \(\mathbb E_S\) averages that **overfitting gap** over training sequences.
- On the right, the expectation averages the replace-one change over \(S\), \(z'\) and \(i\).

In plain words: **on average, a learner’s true risk exceeds its training fit by exactly as much as its loss on an example grows when that example is swapped out.** Lesson 13a measured both sides for ERM in the one-word world with \(m = 5\):

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    return .4 * y * rng.choice((1, 1, 1, 1, -1)), y

loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
L_D = lambda w: (.16 * w * w - .48 * w + 1) / 2
A = lambda S: sum(x * y for x, y in S) / (.16 * len(S))
rng, m, N, gap, swap = random.Random(1), 5, 200_000, 0, 0
for _ in range(N):
    S, z = [draw(rng) for _ in range(m)], draw(rng)
    i = rng.randrange(m)
    w, w_i = A(S), A(S[:i] + [z] + S[i + 1:])
    gap += L_D(w) - sum(loss(w, s) for s in S) / m
    swap += loss(w_i, S[i]) - loss(w, S[i])
print(round(gap / N, 3), round(swap / N, 3))
```

It prints `0.128 0.128`: the overfitting gap on the left and the replace-one change on the right, computed from completely different quantities, land on the same number. Adding the expected fit to both sides gives the split lesson 13a’s fix was built on: **the expected production error is the expected fit plus the expected replace-one change.**

The identity is exact, but it only tells you *what* to bound. To bound the replace-one change for every distribution, you need to know why the regularized learner changes its mind so little. That is where the penalty’s shape comes in.

## Strong convexity

Why is RLM stable? Because the penalty does more than make the objective convex. It makes it *strongly* convex, with a quadratic floor under every minimizer. Here is the definition.

Start with a picture you already have. Lesson 12a called a function convex when its graph lies below every chord, the straight segment between two points of the graph. A parabola like \(0.05\,w^2\) does better than that. Between \(w = 0\) and \(w = 1\), the chord at the midpoint has height 0.025, while the parabola there is only 0.0125: it sags below its chord by 0.0125. Stretch the chord to twice the width, from \(-1\) to 1, and the sag at the midpoint becomes 0.05, four times as much. A sag that grows with the square of the chord’s width is what “strongly” will mean.

**Definition (strongly convex function).** A function \(f\) is **\(\sigma\)-strongly convex**, for \(\sigma \ge 0\), if for all \(\mathbf w, \mathbf u\) and all \(\alpha \in (0, 1)\),

\[ f(\alpha \mathbf w + (1 - \alpha)\mathbf u) \le \alpha f(\mathbf w) + (1 - \alpha) f(\mathbf u) - \frac{\sigma}{2}\,\alpha(1 - \alpha)\,\|\mathbf w - \mathbf u\|^2. \]

Compare it with convexity from lesson 12a, which asks the graph to lie below every chord. Strong convexity asks for more: below the chord *by a margin* that grows with the square of the distance between the endpoints. Every convex function is \(0\)-strongly convex, so this is a strengthening. (We write \(\sigma\) for the strength to keep \(\lambda\) for the regularization weight.)

Read it from the inside out.

- \(\alpha\mathbf w + (1 - \alpha)\mathbf u\), with \(\alpha \in (0, 1)\) (\(\alpha\) strictly between 0 and 1), is a point on the segment between \(\mathbf u\) and \(\mathbf w\): \(\alpha\) of the way from \(\mathbf u\) toward \(\mathbf w\).
- The left side is the function’s height at that point.
- \(\alpha f(\mathbf w) + (1 - \alpha) f(\mathbf u)\) is the height of the chord, the straight line between the two endpoints on the graph, at the same point.
- \(\frac\sigma2 \alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2\) is the required margin: zero at the endpoints, largest in the middle, and growing with the squared distance \(\|\mathbf w - \mathbf u\|^2\).

In plain words: **a strongly convex function sags below each of its chords by at least a fixed multiple of the chord’s squared length.** For the opening parabola \(0.05\,w^2\) on the chord from 0 to 1, with \(\sigma = 0.1\) and \(\alpha = 0.5\), the margin is \(0.05 \cdot 0.25 \cdot 1 = 0.0125\): exactly the sag we measured, so the inequality holds with equality. The snippet after the lemma checks this, term by term, at random points.

[[diagram:strong-convexity]]

**Lemma (strong convexity).**

1. The function \(f(\mathbf w) = \lambda \|\mathbf w\|^2\) is \(2\lambda\)-strongly convex.
2. If \(f\) is \(\sigma\)-strongly convex and \(g\) is convex, then \(f + g\) is \(\sigma\)-strongly convex.
3. If \(f\) is \(\sigma\)-strongly convex and \(\mathbf u\) is a minimizer of \(f\), then for every \(\mathbf w\),
\[ f(\mathbf w) - f(\mathbf u) \ge \frac{\sigma}{2}\|\mathbf w - \mathbf u\|^2. \]

In plain words, part 3 says: **around its minimizer, a \(\sigma\)-strongly convex function rises at least as fast as the parabola \(\frac\sigma2\|\mathbf w - \mathbf u\|^2\).** For the kinked function \(f(w) = 0.25\,w^2 + |w - 1|\) used below, with minimizer \(u = 1\) and \(\sigma = 0.5\): at \(w = 3\), \(f(3) - f(1) = 4.25 - 0.25 = 4\), and the floor asks only for \(0.25 \cdot (3 - 1)^2 = 1\).

*Proof of 1.* Expand the squared norm of a mixture. With \(\mathbf p = \alpha \mathbf w + (1 - \alpha)\mathbf u\),

\[ \|\mathbf p\|^2 = \alpha^2\|\mathbf w\|^2 + 2\alpha(1 - \alpha)\langle \mathbf w, \mathbf u \rangle + (1 - \alpha)^2\|\mathbf u\|^2. \]

Now compute \(\alpha\|\mathbf w\|^2 + (1 - \alpha)\|\mathbf u\|^2 - \|\mathbf p\|^2\). The \(\|\mathbf w\|^2\) coefficient is \(\alpha - \alpha^2 = \alpha(1 - \alpha)\), the \(\|\mathbf u\|^2\) coefficient is \((1 - \alpha) - (1 - \alpha)^2 = \alpha(1 - \alpha)\), and the cross term is \(-2\alpha(1 - \alpha)\langle \mathbf w, \mathbf u \rangle\). Together,

\[ \alpha\|\mathbf w\|^2 + (1 - \alpha)\|\mathbf u\|^2 - \|\mathbf p\|^2 = \alpha(1 - \alpha)\left(\|\mathbf w\|^2 - 2\langle \mathbf w, \mathbf u \rangle + \|\mathbf u\|^2\right) = \alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2. \]

Multiply by \(\lambda\): the chord of \(\lambda\|\cdot\|^2\) lies above the function by exactly \(\lambda\,\alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2 = \frac{2\lambda}{2}\alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2\). The definition holds with \(\sigma = 2\lambda\), with equality.

*Proof of 2.* Write the strong convexity inequality for \(f\) and the convexity inequality for \(g\) at the same \(\mathbf w, \mathbf u, \alpha\), and add them. The right sides add to \(\alpha(f + g)(\mathbf w) + (1 - \alpha)(f + g)(\mathbf u) - \frac\sigma2\alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2\), since \(g\) contributes no margin term.

*Proof of 3.* Write the definition with the mixture \(\mathbf u + \alpha(\mathbf w - \mathbf u) = \alpha\mathbf w + (1 - \alpha)\mathbf u\), subtract \(f(\mathbf u)\) from both sides, and divide by \(\alpha > 0\):

\[ \frac{f(\mathbf u + \alpha(\mathbf w - \mathbf u)) - f(\mathbf u)}{\alpha} \le f(\mathbf w) - f(\mathbf u) - \frac{\sigma}{2}(1 - \alpha)\|\mathbf w - \mathbf u\|^2. \]

(The \(f(\mathbf u)\) terms: \(\alpha f(\mathbf w) + (1 - \alpha)f(\mathbf u) - f(\mathbf u) = \alpha(f(\mathbf w) - f(\mathbf u))\).) Since \(\mathbf u\) is a minimizer, the numerator on the left is at least 0, so the left side is at least 0. Therefore \(f(\mathbf w) - f(\mathbf u) \ge \frac\sigma2(1 - \alpha)\|\mathbf w - \mathbf u\|^2\) for every \(\alpha \in (0, 1)\). Let \(\alpha \to 0\). \(\blacksquare\)

Notice that part 3 never took a derivative. It holds at kinks too, which matters because the hinge loss and the absolute loss have them.

First, part 1 as an experiment: the chord of \(\lambda\|\cdot\|^2\) sits above the function by exactly the margin, at random points in \(\mathbb R^4\).

```python
import random

rng, lam = random.Random(0), 0.3
f = lambda w: lam * sum(v * v for v in w)
sq = lambda w, u: sum((p - q) ** 2 for p, q in zip(w, u))
for _ in range(3):
    w = [rng.uniform(-5, 5) for _ in range(4)]
    u = [rng.uniform(-5, 5) for _ in range(4)]
    a = rng.random()
    mix = [a * p + (1 - a) * q for p, q in zip(w, u)]
    chord = a * f(w) + (1 - a) * f(u)
    dip = (2 * lam / 2) * a * (1 - a) * sq(w, u)
    print(round(chord - f(mix), 6), round(dip, 6))
```

It prints `2.765292 2.765292`, `0.290599 0.290599` and `3.51921 3.51921`: equality, every time.

Now part 3 at a kink. The function \(f(w) = 0.25\,w^2 + |w - 1|\) is the sum of a \(0.5\)-strongly convex function and a convex one, so by parts 1 and 2 it is \(0.5\)-strongly convex, and part 3 promises \(f(w) - f(u) \ge 0.25\,(w - u)^2\) around its minimizer \(u\), even though \(f\) has no derivative at \(w = 1\).

```python
lam = 0.25
f = lambda w: lam * w * w + abs(w - 1)
grid = [i / 100 for i in range(-1000, 1001)]
u = min(grid, key=f)
ratios = [(f(w) - f(u)) / (w - u) ** 2 for w in grid if w != u]
print(u, round(min(ratios), 4))
```

It prints `1.0 0.2955`. The minimizer sits exactly on the kink, and across the grid the rise is never less than 0.2955 times the squared distance, above the promised 0.25.

What does this give us for RLM? If every loss \(\ell(\cdot, z)\) is convex, then \(L_S\) is convex (an average of convex functions, lesson 12a), and by parts 1 and 2 the regularized objective \(f_S = L_S + \lambda\|\cdot\|^2\) is \(2\lambda\)-strongly convex. Two consequences follow straight away.

- *At most one minimizer.* If \(\mathbf u\) and \(\mathbf u'\) both minimize \(f_S\), part 3 gives \(0 = f_S(\mathbf u') - f_S(\mathbf u) \ge \lambda\|\mathbf u' - \mathbf u\|^2\), so \(\mathbf u' = \mathbf u\). So \(A(S)\) is well defined, as soon as a minimizer exists.
- *A minimizer exists.* Losses are nonnegative, so \(f_S(\mathbf w) \ge \lambda\|\mathbf w\|^2\), while \(f_S(\mathbf 0) = L_S(\mathbf 0)\). Every \(\mathbf w\) with \(\lambda\|\mathbf w\|^2 > L_S(\mathbf 0)\) is worse than \(\mathbf 0\), so we may search the closed ball \(\|\mathbf w\|^2 \le L_S(\mathbf 0)/\lambda\). A convex function on \(\mathbb R^d\) is continuous, and a continuous function on a closed bounded set attains its minimum.

## Why the regularized learner is stable

Now the main argument. It has one idea: compare the objective of \(S\) with the objective of \(S^{(i)}\). They differ in a single example, so they are almost the same function, and the quadratic floor turns “almost the same function” into “almost the same minimizer”.

Fix \(S\), \(z'\) and \(i\), and assume every loss \(\ell(\cdot, z)\) is convex. Let \(A\) be the RLM rule, \(A(S) = \operatorname{argmin}_{\mathbf w} f_S(\mathbf w)\).

*Step 1: the floor.* \(f_S\) is \(2\lambda\)-strongly convex and \(A(S)\) minimizes it, so by part 3 of the lemma with \(\sigma = 2\lambda\), for every \(\mathbf v\),

\[ f_S(\mathbf v) - f_S(A(S)) \ge \lambda \|\mathbf v - A(S)\|^2. \]

*Step 2: two objectives, one example apart.* The empirical risks of \(S\) and \(S^{(i)}\) share \(m - 1\) terms. \(S\) has \(\ell(\cdot, z_i)/m\) where \(S^{(i)}\) has \(\ell(\cdot, z')/m\), so for every \(\mathbf w\),

\[ L_S(\mathbf w) = L_{S^{(i)}}(\mathbf w) + \frac{\ell(\mathbf w, z_i) - \ell(\mathbf w, z')}{m}. \]

Use this for two points \(\mathbf v\) and \(\mathbf u\), add the same penalty to both sides, and subtract:

\[ f_S(\mathbf v) - f_S(\mathbf u) = f_{S^{(i)}}(\mathbf v) - f_{S^{(i)}}(\mathbf u) + \frac{\ell(\mathbf v, z_i) - \ell(\mathbf u, z_i)}{m} + \frac{\ell(\mathbf u, z') - \ell(\mathbf v, z')}{m}. \]

*Step 3: choose the points.* Take \(\mathbf v = A(S^{(i)})\) and \(\mathbf u = A(S)\). Since \(\mathbf v\) minimizes \(f_{S^{(i)}}\), the first difference \(f_{S^{(i)}}(\mathbf v) - f_{S^{(i)}}(\mathbf u)\) is at most 0. Dropping it,

\[ f_S(A(S^{(i)})) - f_S(A(S)) \le \frac{\ell(A(S^{(i)}), z_i) - \ell(A(S), z_i)}{m} + \frac{\ell(A(S), z') - \ell(A(S^{(i)}), z')}{m}. \]

*Step 4: combine with the floor.* Step 1 with \(\mathbf v = A(S^{(i)})\) bounds the left side from below. So

\[ \lambda \|A(S^{(i)}) - A(S)\|^2 \le \frac{\ell(A(S^{(i)}), z_i) - \ell(A(S), z_i)}{m} + \frac{\ell(A(S), z') - \ell(A(S^{(i)}), z')}{m}. \]

Call this the **stability inequality**. Read it from the inside out.

- \(\|A(S^{(i)}) - A(S)\|\) is the distance between the two outputs, how far the weights moved; squared and multiplied by \(\lambda\) on the left.
- On the right, the first fraction is the replace-one change on \(z_i\), divided by \(m\); the second is the reverse change on \(z'\): how much \(A(S)\), which never trained on \(z'\), loses on it compared with \(A(S^{(i)})\), which did.

One example’s worth of disagreement can only move a \(2\lambda\)-strongly convex minimizer a little. In plain words: **\(\lambda\) times the squared movement of the minimizer is at most the two swapped examples’ disagreement, divided by \(m\).** Take the three-example world from lesson 13a, where replacing \(z_3 = (0.4, -1)\) by \(z' = (0.4, 1)\) moved ridge from 0.5128 to 1.5385. The left side is \(0.05 \cdot (1.5385 - 0.5128)^2 \approx 0.0526\). On the right, the change on \(z_3\) is 0.5786, the reverse change on \(z'\) is 0.2419, and their sum divided by \(m = 3\) is \(0.2735\):

```python
S, lam, z_new, i = [(.4, 1), (-.4, -1), (.4, -1)], .05, (.4, 1), 2
loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
def A(S):
    A_, b = sum(x * x for x, _ in S), sum(x * y for x, y in S)
    return b / (A_ + 2 * lam * len(S))
S_i = S[:i] + [z_new] + S[i + 1:]
w, w_i, z_i, m = A(S), A(S_i), S[i], len(S)
d_i = loss(w_i, z_i) - loss(w, z_i)
d_z = loss(w, z_new) - loss(w_i, z_new)
print(round(w, 4), round(w_i, 4), round(d_i, 4))
print(round(lam * (w_i - w) ** 2, 4), round((d_i + d_z) / m, 4))
```

It prints `0.5128 1.5385 0.5786`, then `0.0526 0.2735`. The floor holds with room to spare.

Let’s check it numerically on ridge regression in the one-word world, where the ridge minimizer is \(w = \sum_i x_i y_i / ((0.16 + 2\lambda)m)\). We print the smallest and largest slack, right side minus left side, over 10,000 random swaps; the inequality says it is never negative.

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    return .4 * y * rng.choice((1, 1, 1, 1, -1)), y

loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
lam, m, rng, slack = 0.05, 10, random.Random(2), []
A = lambda S: sum(x * y for x, y in S) / ((.16 + 2 * lam) * m)
for _ in range(10_000):
    S = [draw(rng) for _ in range(m + 1)]
    z, i = S.pop(), rng.randrange(m)
    u, v = A(S), A(S[:i] + [z] + S[i + 1:])
    d_i = loss(v, S[i]) - loss(u, S[i])
    d_z = loss(u, z) - loss(v, z)
    slack.append((d_i + d_z) / m - lam * (v - u) ** 2)
print(round(min(slack), 6), round(max(slack), 4))
```

It prints `0.0 0.0199`. Never negative; the 0 comes from swaps where \(z'\) happens to equal \(z_i\) and nothing moves.

Now watch the floor at work in the campaign world, before we squeeze a rate out of it. The lab trains ridge on \(m\) campaign messages, replaces one of them with a fresh message, retrains, and shows how far the weight vector moved, the distance \(\|A(S^{(i)}) - A(S)\|\), next to both sides of the stability inequality. A prediction first: with \(\lambda = 0.1\) and \(m = 200\), when one label is replaced, about how far do the five weights move, measured as the length of the change? Write it down, then press “Replace one label” a few times. After that, move \(m\) and \(\lambda\) and watch how the movement scales: what happens when you multiply \(m\) by ten?

[[lab]]

The stability inequality bounds a movement by losses at the moved points. To turn it into a stability *rate*, we need to know how much a loss can change when its argument moves. Lesson 12a gave us two ways to say that: Lipschitzness and smoothness. Each gives its own proof.

## Lipschitz losses

Recall from lesson 12a: \(\ell(\cdot, z)\) is **\(\rho\)-Lipschitz** if \(|\ell(\mathbf v, z) - \ell(\mathbf w, z)| \le \rho\|\mathbf v - \mathbf w\|\) for all \(\mathbf v, \mathbf w\). The hinge loss and the absolute loss on messages with \(\|\mathbf x\| \le \rho\) are examples.

Write \(\Delta = \|A(S^{(i)}) - A(S)\|\). By Lipschitzness, each of the two differences on the right of the stability inequality is at most \(\rho\Delta\):

\[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \le \rho\,\Delta, \qquad \ell(A(S), z') - \ell(A(S^{(i)}), z') \le \rho\,\Delta. \]

So the stability inequality gives \(\lambda\Delta^2 \le 2\rho\Delta/m\). If \(\Delta = 0\) there is nothing to prove; otherwise divide by \(\lambda\Delta\):

\[ \|A(S^{(i)}) - A(S)\| \le \frac{2\rho}{\lambda m}. \]

The minimizer moves by at most \(2\rho/(\lambda m)\): less with more examples, less with a heavier penalty. Plug this back into the first Lipschitz bound:

\[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \le \rho\,\Delta \le \frac{2\rho^2}{\lambda m}. \]

This holds for every \(S\), \(z'\) and \(i\), not just on average, so it holds in expectation too.

**Corollary (RLM is stable, Lipschitz case).** If every loss \(\ell(\cdot, z)\) is convex and \(\rho\)-Lipschitz, then the RLM rule with the regularizer \(\lambda\|\mathbf w\|^2\) is on-average-replace-one-stable with rate \(\frac{2\rho^2}{\lambda m}\). By the theorem that stable rules do not overfit,

\[ \mathbb E_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(A(S)) - L_S(A(S)) \right] \le \frac{2\rho^2}{\lambda m}. \]

In plain words: **with a \(\rho\)-Lipschitz loss, the regularized learner overfits, on average, by at most \(2\rho^2/(\lambda m)\).** Each factor reads directly: a steeper loss (\(\rho\)) hurts quadratically, a heavier penalty (\(\lambda\)) and more examples (\(m\)) help in proportion.

Let’s watch it for the absolute loss \(|w x - y|\) in the one-word world. Its Lipschitz constant is \(|x| = 0.4\), so with \(\lambda = 0.05\) and \(m = 20\) the bound is \(2 \cdot 0.16/(0.05 \cdot 20) = 0.32\), the example rate lesson 13a quoted when it defined stability. The absolute loss has a kink, so there is no formula for the minimizer; we find it by ternary search, which works on any function of one variable with a single valley.

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    return .4 * y * rng.choice((1, 1, 1, 1, -1)), y

loss = lambda w, z: abs(w * z[0] - z[1])
rho, lam, m, rng, gaps = .4, .05, 20, random.Random(3), []
def rlm(S, lo=-20.0, hi=20.0):
    F = lambda w: lam * w * w + sum(loss(w, z) for z in S) / m
    for _ in range(100):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, b) if F(a) <= F(b) else (a, hi)
    return lo
for _ in range(2000):
    S = [draw(rng) for _ in range(m + 1)]
    z, i = S.pop(), rng.randrange(m)
    w, w_i = rlm(S), rlm(S[:i] + [z] + S[i + 1:])
    gaps.append(loss(w_i, S[i]) - loss(w, S[i]))
print(round(sum(gaps) / 2000, 4), round(max(gaps), 4),
      round(2 * rho ** 2 / (lam * m), 4))
```

It prints `0.0305 0.16 0.32`. The average replace-one change is 0.0305, and even the worst single swap out of 2,000, 0.16, stays under the bound of 0.32, which the proof says holds swap by swap.

## Smooth losses

The half square loss of our campaign filter is not Lipschitz on all of \(\mathbb R^d\): its slope grows without limit as the score moves away from the label. It is **\(\beta\)-smooth**, though, meaning its gradient is \(\beta\)-Lipschitz, and nonnegative. Lesson 12a proved two facts about such functions that we now need.

- *The quadratic ceiling.* If \(f\) is \(\beta\)-smooth, then \(f(\mathbf v) \le f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle + \frac\beta2\|\mathbf v - \mathbf w\|^2\).
- *Self-boundedness.* If \(f\) is \(\beta\)-smooth and nonnegative, then \(\|\nabla f(\mathbf w)\|^2 \le 2\beta f(\mathbf w)\). (Set \(\mathbf v = \mathbf w - \nabla f(\mathbf w)/\beta\) in the ceiling and use \(f(\mathbf v) \ge 0\).) Where the loss is small, its gradient is small.

For the half square loss, \(\nabla_{\mathbf w}\ell = (\langle \mathbf w, \mathbf x \rangle - y)\,\mathbf x\), so \(\|\nabla\ell(\mathbf v) - \nabla\ell(\mathbf w)\| = |\langle \mathbf v - \mathbf w, \mathbf x \rangle|\,\|\mathbf x\| \le \|\mathbf x\|^2\|\mathbf v - \mathbf w\|\), by Cauchy–Schwarz. So it is \(\beta\)-smooth with \(\beta = \|\mathbf x\|^2\), and in the campaign world \(\beta = 5 \cdot 0.44^2 = 0.968\) covers every message.

For this loss both facts hold with equality. Since \(\|\nabla\ell\|^2 = (\langle \mathbf w, \mathbf x \rangle - y)^2\|\mathbf x\|^2 = 2\|\mathbf x\|^2\,\ell\), **the squared gradient is exactly \(2\beta\) times the loss.** For the opening message, \(\|\mathbf x\|^2 = 0.84\) and the loss at \(\mathbf w = (0.3, \dots, 0.3)\) is 0.0753, so both sides are about 0.1265:

```python
w, x, y = [0.3] * 5, [0.44, 0.36, 0.44, 0.44, 0.36], 1
miss = sum(a * b for a, b in zip(w, x)) - y
beta = sum(v * v for v in x)
grad = [miss * v for v in x]
loss = miss ** 2 / 2
print(round(beta, 4), round(sum(g * g for g in grad), 4),
      round(2 * beta * loss, 4))
```

It prints `0.84 0.1265 0.1265`.

We need one extra assumption: the penalty must be heavy enough compared with the smoothness, \(\beta \le \lambda m / 2\), or equivalently \(\lambda \ge 2\beta/m\). The reason: the quadratic ceiling lets the two swapped losses grow by \(\beta\Delta^2/m\) together, and the penalty’s floor \(\lambda\Delta^2\) has to outgrow that with room to spare, as step 2 will show. Keep the same notation, \(\Delta = \|A(S^{(i)}) - A(S)\|\), and write \(a = \ell(A(S), z_i)\) and \(b = \ell(A(S^{(i)}), z')\) for the losses of each hypothesis on an example it was trained on.

*Step 1: each swapped loss changes by little.* By the ceiling applied to \(\ell(\cdot, z_i)\) at \(\mathbf w = A(S)\), then Cauchy–Schwarz, then self-boundedness,

\[ \ell(A(S^{(i)}), z_i) - a \le \langle \nabla\ell(A(S), z_i), A(S^{(i)}) - A(S) \rangle + \frac\beta2\Delta^2 \le \sqrt{2\beta a}\;\Delta + \frac\beta2\Delta^2. \]

The same argument with the roles of \(S\) and \(S^{(i)}\) exchanged, applied to \(\ell(\cdot, z')\) at \(A(S^{(i)})\), gives

\[ \ell(A(S), z') - b \le \sqrt{2\beta b}\;\Delta + \frac\beta2\Delta^2. \]

*Step 2: the minimizer moves by little.* Put both into the stability inequality:

\[ \lambda\Delta^2 \le \frac{\sqrt{2\beta}\,(\sqrt a + \sqrt b)\,\Delta + \beta\Delta^2}{m}. \]

If \(\Delta = 0\) the replace-one change is 0 and we are done. Otherwise multiply by \(m\), move \(\beta\Delta^2\) to the left, and divide by \(\Delta\): \((\lambda m - \beta)\Delta \le \sqrt{2\beta}(\sqrt a + \sqrt b)\). The assumption \(\beta \le \lambda m/2\) gives \(\lambda m - \beta \ge \lambda m/2 > 0\), so

\[ \Delta \le \frac{2\sqrt{2\beta}}{\lambda m}\left(\sqrt a + \sqrt b\right) = \frac{\sqrt{8\beta}}{\lambda m}\left(\sqrt a + \sqrt b\right). \]

*Step 3: back into step 1.* Substitute this bound for \(\Delta\) in the first inequality of step 1:

\[ \ell(A(S^{(i)}), z_i) - a \le \sqrt{2\beta a}\,\frac{\sqrt{8\beta}}{\lambda m}(\sqrt a + \sqrt b) + \frac\beta2\cdot\frac{8\beta}{(\lambda m)^2}(\sqrt a + \sqrt b)^2 = \frac{4\beta}{\lambda m}\sqrt a(\sqrt a + \sqrt b) + \frac{4\beta^2}{(\lambda m)^2}(\sqrt a + \sqrt b)^2. \]

Now simplify with three small inequalities. First, \(\sqrt a(\sqrt a + \sqrt b) \le (\sqrt a + \sqrt b)^2\), since \(\sqrt b \ge 0\). Second, \(\beta/(\lambda m) \le 1/2\) by assumption, so \(4\beta^2/(\lambda m)^2 \le 2\beta/(\lambda m)\). That makes the whole right side at most \(\frac{6\beta}{\lambda m}(\sqrt a + \sqrt b)^2\). Third, \((p + q)^2 \le 2(p^2 + q^2)\), because \(2(p^2 + q^2) - (p + q)^2 = (p - q)^2 \ge 0\). With \(p = \sqrt a\), \(q = \sqrt b\),

\[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \le \frac{12\beta}{\lambda m}\left(\ell(A(S), z_i) + \ell(A(S^{(i)}), z')\right). \]

*Step 4: take expectations.* Over \(S, z', i\), the first term on the right has mean \(\mathbb E[L_S(A(S))]\), as in lesson 13a’s proof that stable rules do not overfit. So does the second: \(S^{(i)}\) is itself an i.i.d. sequence of \(m\) examples whose \(i\)th entry is \(z'\), so \(\ell(A(S^{(i)}), z')\) is distributed exactly like \(\ell(A(S), z_i)\). Therefore the right side has mean \(\frac{24\beta}{\lambda m}\mathbb E[L_S(A(S))]\).

**Corollary (RLM is stable, smooth case).** If every loss \(\ell(\cdot, z)\) is convex, nonnegative and \(\beta\)-smooth, and \(\lambda \ge 2\beta/m\), then the RLM rule with the regularizer \(\lambda\|\mathbf w\|^2\) satisfies

\[ \mathbb E\left[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \right] \le \frac{24\beta}{\lambda m}\,\mathbb E_S\left[L_S(A(S))\right]. \]

Read it from the inside out.

- \(\mathbb E_S[L_S(A(S))]\) is the expected fit: the training loss of the learner’s own output, averaged over training sequences.
- \(\frac{24\beta}{\lambda m}\) is the fraction of that fit the swap can cost; it shrinks with \(\lambda\) and \(m\) and grows with the smoothness constant \(\beta\).
- The condition \(\lambda \ge 2\beta/m\) is what made step 2 work.

In plain words: **with a smooth nonnegative loss and enough penalty, the replace-one change is at most a fraction \(24\beta/(\lambda m)\) of the expected fit.** For the one-word ridge below, \(\beta = 0.16\), \(\lambda = 0.05\), \(m = 200\): the fraction is \(24 \cdot 0.16/10 = 0.384\), and times an expected fit of 0.3451 that is 0.1325.

If moreover \(\ell(\mathbf 0, z) \le C\) for every \(z\), then for every \(S\), \(L_S(A(S)) \le L_S(A(S)) + \lambda\|A(S)\|^2 \le L_S(\mathbf 0) + \lambda\|\mathbf 0\|^2 \le C\), because \(A(S)\) minimizes the regularized objective; so the rate is at most \(\frac{24\beta C}{\lambda m}\).

Notice the shape: the stability term is a *fraction* of the expected fit. A filter that fits well is also very stable, which the Lipschitz bound could not say.

Let’s see the bound and the truth side by side, with the same numbers; the condition holds easily, since \(2\beta/m = 0.0016 \le \lambda\).

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    return .4 * y * rng.choice((1, 1, 1, 1, -1)), y

loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
beta, lam, m, N = .16, .05, 200, 20_000
A = lambda S: sum(x * y for x, y in S) / ((.16 + 2 * lam) * m)
rng, swap, fit = random.Random(4), 0, 0
for _ in range(N):
    S = [draw(rng) for _ in range(m + 1)]
    z, i = S.pop(), rng.randrange(m)
    w, w_i = A(S), A(S[:i] + [z] + S[i + 1:])
    swap += loss(w_i, S[i]) - loss(w, S[i])
    fit += sum(loss(w, s) for s in S) / m
print(round(swap / N, 4), round(fit / N, 4))
print(round(24 * beta / (lam * m) * fit / N, 4))
```

It prints `0.002 0.3451`, then `0.1325`. The measured replace-one change is 0.002, the expected fit 0.345, and the bound 0.1325. True, and loose by a factor of about 65. That is typical of worst-case bounds: they hold for every distribution, so they cannot be tight for a friendly one. Still, a loose bound you can prove beats a tight guess you cannot.

## Fit against stability

We now have every piece. Split the expected true risk into the two quantities a learner controls:

\[ \mathbb E_S[L_{\mathcal D}(A(S))] = \mathbb E_S[L_S(A(S))] + \mathbb E_S[L_{\mathcal D}(A(S)) - L_S(A(S))]. \]

Read it from the inside out: each \(\mathbb E_S\) averages over training sequences; the first term on the right is the expected fit, the second the expected overfitting gap; the identity holds because adding and subtracting \(L_S(A(S))\) inside an expectation changes nothing. In plain words: **expected production error equals expected fit plus expected overfitting.** In the one-word world with \(m = 200\) and \(\lambda = 0.1\) (the table later in this section), the fit is 0.3744 and the true risk 0.3759, so the gap is 0.0015.

The first term is the **fit term**: how well \(A(S)\) fits the training sequence. The second is the overfitting gap, which equals the stability term by the theorem. We have bounded the stability term, and it shrinks as \(\lambda\) grows. The fit term grows as \(\lambda\) grows. Is that last claim obvious? It is true for every single \(S\), and the proof is short.

**Claim (more penalty, worse fit).** For a fixed \(S\) and \(0 < \lambda_1 < \lambda_2\), let \(\mathbf w_1\) and \(\mathbf w_2\) be the RLM outputs. Then \(\|\mathbf w_1\| \ge \|\mathbf w_2\|\) and \(L_S(\mathbf w_1) \le L_S(\mathbf w_2)\).

*Proof.* Each output beats the other on its own objective: \(L_S(\mathbf w_1) + \lambda_1\|\mathbf w_1\|^2 \le L_S(\mathbf w_2) + \lambda_1\|\mathbf w_2\|^2\) and \(L_S(\mathbf w_2) + \lambda_2\|\mathbf w_2\|^2 \le L_S(\mathbf w_1) + \lambda_2\|\mathbf w_1\|^2\). Add the two: the \(L_S\) terms cancel, leaving \((\lambda_2 - \lambda_1)(\|\mathbf w_2\|^2 - \|\mathbf w_1\|^2) \le 0\), so \(\|\mathbf w_2\| \le \|\mathbf w_1\|\). Put this into the first inequality: \(L_S(\mathbf w_1) - L_S(\mathbf w_2) \le \lambda_1(\|\mathbf w_2\|^2 - \|\mathbf w_1\|^2) \le 0\). \(\blacksquare\)

So \(\lambda\) is a dial with a cost at each end. That is the **fitting-stability tradeoff**, and you have met its cousin before: lesson 5’s bias-complexity tradeoff, and lesson 12b’s choice of the ball radius \(B\).

To get a guarantee we still need an upper bound on the fit term that does not depend on the sample. Here it is. Fix *any* vector \(\mathbf w^*\), for example the best weight vector in some class. Since \(A(S)\) minimizes the regularized objective and the penalty is nonnegative,

\[ L_S(A(S)) \le L_S(A(S)) + \lambda\|A(S)\|^2 \le L_S(\mathbf w^*) + \lambda\|\mathbf w^*\|^2. \]

Take expectations. Since \(\mathbf w^*\) is fixed, \(\mathbb E_S[L_S(\mathbf w^*)] = \frac1m\sum_i \mathbb E[\ell(\mathbf w^*, z_i)] = L_{\mathcal D}(\mathbf w^*)\), because each \(z_i \sim \mathcal D\). So

\[ \mathbb E_S[L_S(A(S))] \le L_{\mathcal D}(\mathbf w^*) + \lambda\|\mathbf w^*\|^2. \]

In plain words: **on average, the regularized learner fits its training sequence at least as well as any fixed vector \(\mathbf w^*\) fits the world, plus that vector’s price.** Here \(\mathbf w^*\) is any vector chosen before seeing data, and \(L_{\mathcal D}(\mathbf w^*)\) its true risk. In the one-word world the best fixed weight is the bottom of the parabola, \(w^* = 0.48/0.32 = 1.5\), with \(L_{\mathcal D}(1.5) = 0.32\); with \(\lambda = 0.1\) its price is \(0.1 \cdot 2.25 = 0.225\), so the bound promises an expected fit of at most 0.545. The exact expected fit is 0.3744: comfortably below.

```python
L_D = lambda w: (.16 * w * w - .48 * w + 1) / 2
lam, w_star = .1, 0.48 / 0.32
print(w_star, round(L_D(w_star), 4), round(lam * w_star ** 2, 4),
      round(L_D(w_star) + lam * w_star ** 2, 4))
```

It prints `1.5 0.32 0.225 0.545`.

Add the Lipschitz stability bound and you get the first guarantee of the lesson.

**Corollary (oracle inequality, Lipschitz case).** If every loss is convex and \(\rho\)-Lipschitz, the RLM rule with regularization \(\lambda\|\mathbf w\|^2\) satisfies, for every \(\mathbf w^*\),

\[ \mathbb E_S[L_{\mathcal D}(A(S))] \le L_{\mathcal D}(\mathbf w^*) + \lambda\|\mathbf w^*\|^2 + \frac{2\rho^2}{\lambda m}. \]

Read it term by term: \(L_{\mathcal D}(\mathbf w^*)\) is how good the comparison vector is; \(\lambda\|\mathbf w^*\|^2\) is what the penalty costs you for wanting to be near it, the fit side of the trade; \(2\rho^2/(\lambda m)\) is the stability side. In plain words: **the regularized learner’s expected true risk is at most any vector’s true risk plus that vector’s price plus the overfitting allowance.** Large \(\lambda\) makes the middle term big, small \(\lambda\) makes the last term big.

This kind of bound is called an **oracle inequality**: think of \(\mathbf w^*\) as a hypothesis with low risk that only an oracle knows. The bound says how many examples \(A(S)\) needs to be almost as good as \(\mathbf w^*\), *had we known* \(\|\mathbf w^*\|\) to set \(\lambda\). In practice we do not know it, and we tune \(\lambda\) on a validation set, as in lesson 11. But if we do know a bound on the norm, the best \(\lambda\) can be computed.

**Corollary (convex-Lipschitz-bounded problems are learnable).** Let \((\mathcal H, Z, \ell)\) be convex-Lipschitz-bounded with parameters \(\rho, B\), and set \(\lambda = \sqrt{2\rho^2/(B^2 m)}\). Then the RLM rule satisfies

\[ \mathbb E_S[L_{\mathcal D}(A(S))] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \rho B\sqrt{\frac8m}. \]

In particular, for every \(\varepsilon > 0\), if \(m \ge 8\rho^2B^2/\varepsilon^2\), then for every distribution, \(\mathbb E_S[L_{\mathcal D}(A(S))] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \varepsilon\).

Read it from the inside out: \(\min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w)\) is the true risk of the best vector in the bounded class \(\mathcal H\) (lesson 12b’s ball of radius \(B\)), and \(\rho B\sqrt{8/m}\) is the excess, which shrinks like one over the square root of \(m\). In plain words: **with \(\lambda\) set from \(\rho\), \(B\) and \(m\), RLM gets within \(\rho B\sqrt{8/m}\) of the best vector in the ball, on average.** With \(\rho = 1\), \(B = 0.7\) and \(m = 1{,}568\), the best \(\lambda\) is \(\sqrt{2/(0.49 \cdot 1568)} \approx 0.051\) and the excess is \(0.7\sqrt{8/1568} \approx 0.05\). The snippet minimizes \(\lambda B^2 + 2\rho^2/(\lambda m)\) over a grid and compares with the formulas:

```python
rho, B, m = 1, .7, 1568
excess = lambda lam: lam * B ** 2 + 2 * rho ** 2 / (lam * m)
grid = [k / 10000 for k in range(1, 5000)]
best = min(grid, key=excess)
print(best, round((2 * rho ** 2 / (B ** 2 * m)) ** .5, 4))
print(round(excess(best), 4), round(rho * B * (8 / m) ** .5, 4))
```

It prints `0.051 0.051`, then `0.05 0.05`: the grid’s best \(\lambda\) and best excess match the closed forms.

*Proof.* Take \(\mathbf w^*\) a minimizer of \(L_{\mathcal D}\) over \(\mathcal H\) (if the minimum is not attained, take a near-minimizer and let the slack go to 0). It has \(\|\mathbf w^*\| \le B\), so the oracle inequality gives an excess of at most \(\lambda B^2 + 2\rho^2/(\lambda m)\). With \(c = 2\rho^2/m\), the function \(\lambda \mapsto \lambda B^2 + c/\lambda\) is smallest at \(\lambda = \sqrt{c/B^2}\), where both terms equal \(\sqrt{cB^2}\), so the sum is \(2B\sqrt c = 2B\sqrt{2\rho^2/m} = \rho B\sqrt{8/m}\). Finally, \(\rho B\sqrt{8/m} \le \varepsilon\) exactly when \(m \ge 8\rho^2B^2/\varepsilon^2\). \(\blacksquare\)

That keeps half of lesson 12b’s promise: every convex-Lipschitz-bounded problem, including halfspaces with the hinge loss, is learnable, by RLM, with a proof that never used uniform convergence. (Some convex-Lipschitz-bounded problems lack the uniform convergence property, so this route is genuinely new, not a detour.)

The smooth case gives a multiplicative version: put the smooth stability bound into the decomposition, then use the fit bound.

**Corollary (oracle inequality, smooth case).** If every loss is convex, \(\beta\)-smooth and nonnegative, and \(\lambda \ge 2\beta/m\), then for every \(\mathbf w^*\),

\[ \mathbb E_S[L_{\mathcal D}(A(S))] \le \left(1 + \frac{24\beta}{\lambda m}\right)\mathbb E_S[L_S(A(S))] \le \left(1 + \frac{24\beta}{\lambda m}\right)\left(L_{\mathcal D}(\mathbf w^*) + \lambda\|\mathbf w^*\|^2\right). \]

Read it from the inside out: the factor \(1 + \frac{24\beta}{\lambda m}\) is 1 (perfect fit-to-truth transfer) plus the smooth stability fraction; it multiplies first the expected fit, then the fit bound. In plain words: **expected production error is at most the expected fit, inflated by \(1 + 24\beta/(\lambda m)\).** With \(\beta = 0.16\), \(m = 200\), \(\lambda = 0.1\): the factor is \(1 + 3.84/20 = 1.192\), and times the expected fit 0.3744 that certifies 0.4463, the row \(\lambda = 0.1\) of the table below.

For example, \(\lambda = 24\beta/m\) gives an expected true risk at most twice the expected fit, and an expected fit at most \(L_{\mathcal D}(\mathbf w^*) + \frac{24\beta}{m}\|\mathbf w^*\|^2\).

The first inequality is the one you will use in the lab, because both of its ingredients can be measured. The fit is something you see every night. The factor is something you choose. Here is the tradeoff in the one-word world with \(m = 200\), computed exactly by summing over the number of agreeing examples, which is binomial. Each line shows \(\lambda\), the expected fit, the true expected risk, and the certified bound \((1 + 24\beta/(\lambda m))\) times the fit.

```python
from math import comb

beta, m = .16, 200
L_D = lambda w: (.16 * w * w - .48 * w + 1) / 2
for lam in (.002, .005, .01, .02, .05, .1, .2):
    fit = risk = 0
    for k in range(m + 1):
        p = comb(m, k) * .8 ** k * .2 ** (m - k)
        w = .4 * (2 * k - m) / ((.16 + 2 * lam) * m)
        sq = k * (.4 * w - 1) ** 2 + (m - k) * (.4 * w + 1) ** 2
        fit += p * sq / (2 * m)
        risk += p * L_D(w)
    bound = (1 + 24 * beta / (lam * m)) * fit
    print(lam, round(fit, 4), round(risk, 4), round(bound, 4))
```

| \(\lambda\) | expected fit | expected true risk | certified bound |
|---|---|---|---|
| 0.002 | 0.3185 | 0.3216 | 3.3762 |
| 0.005 | 0.3190 | 0.3220 | 1.5441 |
| 0.01 | 0.3206 | 0.3235 | 0.9363 |
| 0.02 | 0.3257 | 0.3282 | 0.6383 |
| 0.05 | 0.3453 | 0.3472 | 0.4778 |
| 0.1 | 0.3744 | 0.3759 | 0.4463 |
| 0.2 | 0.4111 | 0.4120 | 0.4505 |

Read the columns. The fit rises with \(\lambda\), as the claim promised. The certified bound falls steeply, bottoms out near \(\lambda = 0.1\) at 0.446, then rises again as the fit term takes over. That U shape is the tradeoff. Notice also that the true risk is best at the *smallest* \(\lambda\) here: the one-word world has no valley, so ERM is already stable, and the bound is too pessimistic to see it. The bound picks the \(\lambda\) it can *prove* things about, not the one that is truly best. In the campaign world, with its valley, the two agree much better.

[[diagram:tradeoff]]

Finally, the smooth case gives its own learnability result, and with it the guarantee for ridge regression.

**Corollary (convex-smooth-bounded problems are learnable).** Let \((\mathcal H, Z, \ell)\) be convex-smooth-bounded with parameters \(\beta, B\), and assume \(\ell(\mathbf 0, z) \le 1\) for all \(z\). For any \(\varepsilon \in (0, 1)\), let \(m \ge 150\beta B^2/\varepsilon^2\) and \(\lambda = \varepsilon/(3B^2)\). Then for every distribution,

\[ \mathbb E_S[L_{\mathcal D}(A(S))] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \varepsilon. \]

Read it the same way as the Lipschitz version: \(\beta\) is the smoothness constant, \(B\) the radius of the class, \(\varepsilon \in (0, 1)\) the accuracy you ask for, and \(\ell(\mathbf 0, z) \le 1\) says the all-zero vector never loses more than 1. In plain words: **with about \(150\beta B^2/\varepsilon^2\) examples and \(\lambda = \varepsilon/(3B^2)\), RLM is within \(\varepsilon\) of the best vector in the ball, on average.** With \(\beta = 1\), \(B = 0.7\), \(\varepsilon = 0.05\): \(150 \cdot 0.49/0.0025 = 29{,}400\) examples and \(\lambda = 0.05/1.47 \approx 0.034\), the middle row of the table at the end of this section.

*Proof.* First check that the smooth stability bound applies: \(\lambda m = \varepsilon m/(3B^2) \ge 150\beta B^2/(3B^2\varepsilon) = 50\beta/\varepsilon \ge 50\beta \ge 2\beta\), using \(\varepsilon < 1\). So \(\lambda \ge 2\beta/m\), and with \(C = 1\),

\[ \mathbb E_S[L_{\mathcal D}(A(S))] = \mathbb E_S[L_S(A(S))] + \text{stability term} \le \left(L_{\mathcal D}(\mathbf w^*) + \lambda\|\mathbf w^*\|^2\right) + \frac{24\beta}{\lambda m}, \]

for every \(\mathbf w^*\). Take \(\mathbf w^*\) a minimizer of \(L_{\mathcal D}\) over \(\mathcal H\) (or a near-minimizer, as before). Then \(\lambda\|\mathbf w^*\|^2 \le \lambda B^2 = \varepsilon/3\), and \(\frac{24\beta}{\lambda m} = \frac{72\beta B^2}{\varepsilon m} \le \frac{72\varepsilon}{150} = 0.48\,\varepsilon\). The excess is at most \(\varepsilon/3 + 0.48\,\varepsilon < 0.82\,\varepsilon \le \varepsilon\). \(\blacksquare\)

**Theorem (ridge regression learns).** Let \(\mathcal D\) be a distribution over \(\mathcal X \times [-1, 1]\), with \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le 1\}\), and let \(\mathcal H = \{\mathbf w \in \mathbb R^d : \|\mathbf w\| \le B\}\). For any \(\varepsilon \in (0, 1)\), let \(m \ge 150B^2/\varepsilon^2\). Then ridge regression with \(\lambda = \varepsilon/(3B^2)\) satisfies

\[ \mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S))] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \varepsilon. \]

*Proof.* The half square loss is convex and nonnegative, and \(\beta\)-smooth with \(\beta = \|\mathbf x\|^2 \le 1\), as computed in the smooth losses section. The ball \(\mathcal H\) is convex with norms at most \(B\). And \(\ell(\mathbf 0, (\mathbf x, y)) = y^2/2 \le 1/2 \le 1\). So the problem is convex-smooth-bounded with \(\beta = 1\), and the previous corollary applies. \(\blacksquare\)

What does this cost in the campaign world? Every message has \(\|\mathbf x\| \le 0.98 \le 1\) and \(y \in \{\pm 1\}\). The best vector has norm \(0.2994\sqrt5 \approx 0.67\), so \(B = 0.7\) contains it. The same arithmetic, next to the Lipschitz corollary for a 1-Lipschitz loss such as the absolute loss on the unit ball:

```python
from math import ceil

B, rho, beta = .7, 1, 1
for eps in (.1, .05, .02):
    m_lip = ceil(8 * rho ** 2 * B ** 2 / eps ** 2)
    lam_lip = (2 * rho ** 2 / (B ** 2 * m_lip)) ** .5
    m_ridge = ceil(150 * beta * B ** 2 / eps ** 2)
    lam_ridge = eps / (3 * B ** 2)
    a = 24 * beta / (lam_ridge * m_ridge)
    extra = lam_ridge * B ** 2 + a
    print(eps, m_lip, round(lam_lip, 4), m_ridge,
          round(lam_ridge, 4), round(extra, 4))
```

| \(\varepsilon\) | Lipschitz: \(m\) | Lipschitz: \(\lambda\) | ridge: \(m\) | ridge: \(\lambda\) | ridge: proved excess |
|---|---|---|---|---|---|
| 0.1 | 392 | 0.102 | 7,350 | 0.068 | 0.0813 |
| 0.05 | 1,568 | 0.051 | 29,400 | 0.034 | 0.0407 |
| 0.02 | 9,800 | 0.0204 | 183,750 | 0.0136 | 0.0163 |

Each ridge row proves an excess of about \(0.81\,\varepsilon\), inside the promised \(\varepsilon\). These are archive-sized numbers, not nightly ones: a guarantee to within 0.02 of the best wants 183,750 labelled messages. Keep that in mind for the end of the lesson.

## Ridge regression as code

Here is ridge regression as an algorithm. It takes a training sequence of \(m\) examples with \(d\) features and the regularization weight \(\lambda\).

```
Ridge(S, λ):
  A ← d×d matrix of zeros;  b ← vector of d zeros
  for i = 1..m:
    for j = 1..d:
      b[j] ← b[j] + y_i · x_{i,j}
      for k = 1..d:
        A[j][k] ← A[j][k] + x_{i,j} · x_{i,k}
  for j = 1..d:
    A[j][j] ← A[j][j] + 2λm
  solve A w = b by Gaussian elimination
  return w
```

**Cost.** Building \(A\) takes \(O(md^2)\) time and building \(\mathbf b\) takes \(O(md)\). Gaussian elimination on a \(d \times d\) system takes \(O(d^3)\) time. So ridge costs \(O(md^2 + d^3)\) time in total, and \(O(d^2)\) space beyond the data, since \(A\) and \(\mathbf b\) can be accumulated one example at a time. Because \(2\lambda m I + A\) has all eigenvalues at least \(2\lambda m\), elimination never meets a zero pivot in exact arithmetic, and the system is well conditioned when \(\lambda m\) is not tiny compared with the largest eigenvalue of \(A\). Notice the factor \(m\) in the running time: every retrain reads every example in the training sequence.

**Build it yourself.** Using only the Python standard library:

```python
S = [((0.4, 0.4, 0.0), 1), ((0.4, 0.4, 0.2), 1),
     ((0.4, 0.4, 0.6), -1), ((-0.4, -0.4, 0.6), -1),
     ((-0.4, -0.4, 0.2), -1), ((0.4, 0.4, 0.4), 1),
     ((-0.4, -0.4, 0.0), 1), ((-0.4, -0.4, 0.4), -1)]

def loss(w, z): ...
def ridge(S, lam): ...
def gradient(S, w, lam): ...

if __name__ == "__main__":
    try:
        ridge(S, 0)
        raise AssertionError("twins: A is singular at lam = 0")
    except ZeroDivisionError:
        pass
    lam, z = 0.05, ((0.4, 0.4, 0.6), 1)
    w = ridge(S, lam)
    assert [round(v, 4) for v in w] == [0.4762, 0.4762, -0.625]
    assert max(abs(g) for g in gradient(S, w, lam)) < 1e-9
    for i in range(len(S)):
        v = ridge(S[:i] + [z] + S[i + 1:], lam)
        lhs = lam * sum((a - b) ** 2 for a, b in zip(v, w))
        rhs = (loss(v, S[i]) - loss(w, S[i])
               + loss(w, z) - loss(v, z)) / len(S)
        assert lhs <= rhs + 1e-12
```

The starter data has three words: “free” and “gift”, which are exact twins in every message, and “meeting”.

1. Implement `loss` (the half square loss), `gradient` (the gradient of the ridge objective \(f_S\)) and `ridge`, which builds \(2\lambda m I + A\) and \(\mathbf b\) and solves by Gaussian elimination with partial pivoting. Raise `ZeroDivisionError` when the largest available pivot has absolute value below \(10^{-12}\). The asserts check that ERM fails on the twins, that ridge succeeds and splits the weight evenly between them, that the gradient vanishes at your solution, and the stability inequality for every \(i\).
2. Rerun lesson 13a’s campaign week with your `ridge` in place of its `fit` and reproduce both of its tables.
3. Shrink \(\lambda\) toward 0 (0.01, 0.001, \(10^{-6}\)) on the starter data. What happens to the twin weights? Compare with lesson 9b’s pseudoinverse solution, which split the weight of two always-together words evenly.
4. Measure the average replace-one change in the campaign world for \(\lambda \in \{0, 0.01, 0.1, 0.5\}\) with \(m = 20\), and check it against the true overfitting gap. Predict first which \(\lambda\) has the smallest gap, and whether that \(\lambda\) has the smallest production error.

## Fix the bug

Back to the campaign filter and the auditor’s question. The penalty steadied the filter, but the manager now asks for more than a steady dashboard and more than a simulator’s say-so. Before the filter goes into the nightly job for good, she wants a number: a certified bound on the expected production error, one that holds whatever the data look like, not a week that happened to go well.

You have the tool. In the campaign world the half square loss is convex, nonnegative and \(0.968\)-smooth, so for any \(\lambda \ge 2\beta/m\) the smooth oracle inequality certifies

\[ \mathbb E[\text{production error}] \le \left(1 + \frac{24 \cdot 0.968}{\lambda m}\right) \times \mathbb E[\text{fit}]. \]

The lab measures the expected fit by averaging the training fit over many nightly retrains, the one number the real mail room gets for free every night, with no extra labels and no swaps, and it computes the factor from your \(\lambda\) and the label budget \(m\). Twenty labels a night was never going to certify anything: at \(m = 20\) and \(\lambda = 0.1\), the factor alone is about 12.6. That is why the manager approved the bigger budget of 2,000 labels a night, and she has set a target. Your job is to choose \(\lambda\) so that the certificate meets the target. Then the budget changes, and so does the target, and you do it again.

Before you start, write down a prediction: when the budget doubles, should the best \(\lambda\) go up, go down, or stay the same? Then use the oracle inequalities to check your answer before you touch the slider.

[[lab-act2]]

The fix holds for the hinge filter of lesson 12a as well. Its loss is not smooth, but it is Lipschitz, so the same penalty makes it stable with rate \(2\rho^2/(\lambda m)\), and the Lipschitz corollary tells you how to set \(\lambda\) from \(\rho\), \(B\) and \(m\). The weights stop jumping when a user corrects one label, and now you can say why.

## What people get wrong

**“Strong convexity is just convexity with a fancier name.”** Convexity lets the graph run along its chords over a whole flat stretch, which is exactly the valley that made ERM slide. Strong convexity forbids that: the graph must sag below every chord by a margin that grows with the chord’s squared length, and that margin is the quadratic floor the stability proof stands on. The penalty \(\lambda\|\mathbf w\|^2\) supplies it even when the loss has none.

**“More regularization is always safer.”** The stability term falls with \(\lambda\), but the fit term rises, for every training sequence. In the one-word table the certificate bottoms out near \(\lambda = 0.1\) and climbs again after.

**“The bound’s best \(\lambda\) is the truly best \(\lambda\).”** The bound is a worst case over all distributions. In the one-word world it prefers \(\lambda \approx 0.1\) while the true risk prefers the smallest \(\lambda\). Use the bound to certify; use a validation set (lesson 11) to tune.

**“These are high-probability guarantees.”** They are bounds on the *expected* true risk over training sequences. Any single night can be worse. The first problem below shows how to convert an expectation bound into a statement with probability at least \(1 - \delta\).

## Recap and the next bug

1. The identity from lesson 13a says *what* to bound: the expected overfitting gap equals the expected replace-one change. This lesson bounds it for every distribution at once.
2. A function is **\(\sigma\)-strongly convex** when it sags below each chord by at least \(\frac\sigma2\alpha(1 - \alpha)\|\mathbf w - \mathbf u\|^2\). The penalty \(\lambda\|\mathbf w\|^2\) is \(2\lambda\)-strongly convex, adding a convex loss keeps that, and around its minimizer a \(\sigma\)-strongly convex function rises at least like \(\frac\sigma2\|\mathbf w - \mathbf u\|^2\), kinks and all.
3. The penalty makes the objective **\(2\lambda\)-strongly convex**, which puts a quadratic floor \(\lambda\|\mathbf v - A(S)\|^2\) under the minimizer. That floor gives the **stability inequality**, and from it a rate of \(2\rho^2/(\lambda m)\) for convex \(\rho\)-Lipschitz losses and \(\frac{24\beta}{\lambda m}\mathbb E[L_S(A(S))]\) for convex, nonnegative, \(\beta\)-smooth losses with \(\lambda \ge 2\beta/m\).
4. \(\lambda\) trades **fit** against **stability**. The **oracle inequalities** turn the trade into guarantees: convex-Lipschitz-bounded problems are learnable with \(m \ge 8\rho^2B^2/\varepsilon^2\), convex-smooth-bounded ones with \(m \ge 150\beta B^2/\varepsilon^2\), and ridge regression learns with \(\lambda = \varepsilon/(3B^2)\).
5. The certificate, \(1 + 24\beta/(\lambda m)\) times the expected fit, needs no simulator: only \(\lambda\), \(m\), the smoothness \(\beta\) and the fit you see every night. When the budget doubles, the best \(\lambda\) goes down.

The regularized filters ship, and the dashboard finally sits still. The manager is pleased for about a day. Then she looks at the retrain logs. The regularized problem is the right problem, but solving it exactly means reading the whole mail archive, every message, and the archive grows by thousands of labels a day. On the full archive, each exact retrain of the regularized hinge filter now takes hours, and the spammers change their campaign faster than that. Every method you know touches all \(m\) examples before it can say anything. Is there a learner whose cost per step does not depend on the size of the archive at all, one that looks at a single message, nudges the weights, and moves on, and still comes with a guarantee? That is the next lesson: stochastic gradient descent.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Take ridge regression on a fixed training sequence in the campaign world and let \(\lambda \to \infty\). What does \(A(S)\) tend to, and what happens to the fit term and the stability term of the smooth oracle inequality, and to the certificate? Why can the certificate never fall below the expected fit, whatever \(\lambda\) you choose?
2. Is a single half square loss \(\ell(\mathbf w, (\mathbf x, y)) = \tfrac12(\langle \mathbf w, \mathbf x \rangle - y)^2\), as a function of \(\mathbf w \in \mathbb R^5\), strongly convex? If yes, with which \(\sigma\); if not, name a direction along which it fails. What changes in the one-word world, where \(w\) is a single number and \(x = \pm 0.4\)? Use your answer to explain why the stability proof takes its floor from the penalty and not from the loss.
3. Messages lie in the unit ball and labels in \([-1, 1]\). For the absolute loss with \(\mathcal H = \{\|\mathbf w\| \le 2\}\) and \(\varepsilon = 0.1\), how many examples does the convex-Lipschitz-bounded corollary ask for, and with which \(\lambda\)? For ridge regression with the same \(B\) and \(\varepsilon\)? Check that the ridge choice satisfies \(\lambda \ge 2\beta/m\).

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (from an expected risk to a high-probability guarantee).** A learner \(A\) guarantees: if \(m \ge m_{\mathcal H}(\varepsilon)\), then for every distribution \(\mathcal D\), \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S))] \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). (a) Show that for every \(\delta \in (0, 1)\), if \(m \ge m_{\mathcal H}(\varepsilon\delta)\), then with probability at least \(1 - \delta\), \(L_{\mathcal D}(A(S)) \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). (b) Assuming the loss is bounded by 1, design a procedure that reaches the same kind of guarantee with about \(m_{\mathcal H}(\varepsilon/2)\lceil\log_2(1/\delta)\rceil + \frac{\log(4/\delta) + \log\lceil\log_2(1/\delta)\rceil}{\varepsilon^2}\) examples, a far milder dependence on \(\delta\) than part (a). (Use lesson 11.)
- **Problem 2 (other norms).** Replace \(\|\cdot\|\) everywhere by an arbitrary norm, and define strong convexity with respect to it by the same inequality. (a) Show that parts 2 and 3 of the strong convexity lemma still hold. (b) Let \(R\) be \(2\lambda\)-strongly convex with respect to this norm, let \(A(S) = \operatorname{argmin}_{\mathbf w}(L_S(\mathbf w) + R(\mathbf w))\), and assume every loss is convex and \(\rho\)-Lipschitz with respect to the same norm: \(\ell(\mathbf w, z) - \ell(\mathbf v, z) \le \rho\|\mathbf w - \mathbf v\|\). Prove that \(A\) is on-average-replace-one-stable with rate \(2\rho^2/(\lambda m)\).
