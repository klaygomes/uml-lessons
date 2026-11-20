# SGD Variants and Faster Rates

## The filter the auditor signed

Lesson 14a ended with a filter that is no longer a day behind. It learns from the morning’s fresh labels, one message per step, and its guarantee is about the true risk itself. Then the auditor from lesson 13b came back with the certificate she had signed, and it was for a different filter.

Her filter is lesson 13’s regularized filter: the weight vector that minimizes the penalized objective on the audited training sequence, the two million labelled messages the team keeps on file, with a penalty weight of \(\lambda = 0.01\). She can check that filter against the audited messages. The streaming filter is a different learner, and its guarantee is about a stream nobody kept. So she has made an offer. She does not need the exact minimizer. She will sign any filter whose regularized objective is, on average, within 0.05 of the exact minimum.

The trouble is the solver. The exact solver for that objective takes six hours on two million messages, which is how this whole story started. Lesson 14a warned against the obvious shortcut, looping SGD over a fixed archive, because it breaks the fresh-message guarantee for the true risk. But here we do not want the true risk. We want the minimum of one fixed function of the audited messages. Can SGD find it, one message per step, and can you certify how close it lands, before the six hours are up?

It can, and it does better than you might expect. The penalty does more than steady the filter, as lesson 13 showed: it curves the objective upward, and that curvature buys SGD a much faster rate than the \(1/\sqrt T\) of lesson 14a. Getting there takes three relatives of SGD: a projection that keeps the iterates inside the class, a step size that shrinks as the walk goes on, and a rate for strongly convex functions. Along the way we also meet smooth losses, where the rate improves when the best filter is good.

## What you need from lesson 14a

Here is the toolkit, one line each, so you can check that nothing has slipped.

- **The six-message world.** Six kinds of messages, equally likely, each with two scores \(\mathbf x = (x_1, x_2)\), how strongly it talks about “free” and about “meeting”, and a label \(y\), \(+1\) for junk and \(-1\) for keep: A (0.8, 0.2) junk, B (0.6, 0.6) junk, C (0.9, 0.4) junk, D (0.1, 0.9) keep, E (0.4, 0.8) keep, F (0.7, 0.3) keep. Every message has length \(\|\mathbf x\| = \sqrt{x_1^2 + x_2^2} \le 1\).
- **Weight vectors and scores.** A filter is a weight vector \(\mathbf w = (w_1, w_2)\); its score for a message is the inner product \(\langle \mathbf w, \mathbf x \rangle = w_1x_1 + w_2x_2\), and the **signed score** \(y\langle \mathbf w, \mathbf x \rangle\) is positive exactly when the filter is right.
- **The hinge loss.** \(\ell(\mathbf w, (\mathbf x, y)) = \max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\), zero once the signed score is at least 1. Its average over the six messages is the hinge risk, smallest at \(\mathbf w^\star = (15/7, -65/28)\) with value \(163/336 \approx 0.4851\).
- **The two risks.** \(L_{\mathcal D}(\mathbf w)\), the true risk, is the expected loss on the next message to arrive; \(L_S(\mathbf w)\), the empirical risk, is the average loss over a training sequence \(S\) of \(m\) messages.
- **Convex and Lipschitz.** \(f\) is convex when its graph sits on or below every chord, and \(\rho\)-Lipschitz when \(|f(\mathbf u) - f(\mathbf w)| \le \rho\|\mathbf u - \mathbf w\|\), so its slope is never steeper than \(\rho\).
- **Subgradients.** A vector \(\mathbf v\) is a subgradient of \(f\) at \(\mathbf w\) when \(f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \mathbf v \rangle\) for every \(\mathbf u\): the slope of a supporting plane. \(\partial f(\mathbf w)\) is the set of them, it is \(\{\nabla f(\mathbf w)\}\) where \(f\) is differentiable, and a convex \(f\) is \(\rho\)-Lipschitz exactly when every subgradient has norm at most \(\rho\). For the hinge loss the rule is: \(\mathbf 0\) if the signed score is at least 1, and \(-y\mathbf x\) otherwise.
- **Iterates and steps.** The walk starts at \(\mathbf w^{(1)} = \mathbf 0\) and updates \(\mathbf w^{(t+1)} = \mathbf w^{(t)} - \eta\mathbf v_t\): \(\mathbf w^{(t)}\) is the **iterate** before step \(t\) (the superscript counts steps), \(\eta > 0\) the **step size**, \(\mathbf v_t\) the step’s direction. The output is the **average** \(\bar{\mathbf w} = \frac1T\sum_{t=1}^T \mathbf w^{(t)}\).
- **Jensen’s inequality.** For convex \(f\), \(f(\bar{\mathbf w}) \le \frac1T\sum_t f(\mathbf w^{(t)})\): the average point sits no higher than the iterates do on average.
- **SGD.** Stochastic gradient descent chooses each \(\mathbf v_t\) at random so that \(\mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}] \in \partial f(\mathbf w^{(t)})\). Here \(\mathbb E\) is the expected value, the bar reads “given”, and \(\mathbf v_{1:t-1}\) is shorthand for the earlier choices \(\mathbf v_1, \dots, \mathbf v_{t-1}\): averaged over this step’s random choice with the past held fixed, the step direction is a subgradient at the current iterate.
- **The SGD guarantee.** If \(f\) is convex, \(\|\mathbf v_t\| \le \rho\) always, and \(\mathbf w^\star\) is a lowest point of \(f\) in the ball \(\{\mathbf w : \|\mathbf w\| \le B\}\) (the weight vectors of length at most \(B\)), then \(\mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le B^2/(2\eta T) + \eta\rho^2/2\), the **certified gap**, which is \(B\rho/\sqrt T\) at \(\eta = B/(\rho\sqrt T)\). The proof’s one random step fixes the past, pulls the fixed vector \(\mathbf w^{(t)} - \mathbf w^\star\) out of the inner product, and applies the subgradient inequality to the conditional mean; the **law of total expectation** then averages over the past.
- **SGD-Risk.** To minimize the true risk, draw a **fresh example** \(z_t \sim \mathcal D\) at each step (drawn from \(\mathcal D\), independent of the past) and step along a subgradient of the loss on it. A **convex-Lipschitz-bounded** problem (lesson 12: weight vectors of norm at most \(B\), each loss convex and \(\rho\)-Lipschitz) is learned within \(\varepsilon\) from \(T \ge B^2\rho^2/\varepsilon^2\) fresh examples.

And the one identity everything in this lesson leans on. The step lemma of lesson 14a was proved by completing a square, which turns each step’s progress into a drop in squared distance to the target:

\[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle = \frac{1}{2\eta} \left( \|\mathbf w^{(t)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 \right) + \frac\eta2 \|\mathbf v_t\|^2. \]

Read it from the inside out.

- \(\|\mathbf w^{(t)} - \mathbf w^\star\|^2\) is the squared distance from the iterate to the target, and the bracket is how much that squared distance drops in step \(t\).
- \(\frac{1}{2\eta}\) scales the drop, and \(\frac\eta2\|\mathbf v_t\|^2\) is the charge for the step’s length.
- On the left, \(\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle\) is the **progress** of step \(t\): large when the step, which goes along \(-\mathbf v_t\), heads towards \(\mathbf w^\star\).

In plain words: **a step’s progress is the drop in squared distance to the target, over \(2\eta\), plus a charge for the step’s length.** Summed over the steps, the drops telescope: each squared distance appears once with a plus and once with a minus, and only the first and last survive. Here is the first step of lesson 14a’s SGD walk, which drew junk message B at \(\mathbf w^{(1)} = \mathbf 0\), so \(\mathbf v_1 = (-0.6, -0.6)\), with \(\eta = 0.5\) and the hinge risk’s lowest point as the target:

```python
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
sub = lambda u, v: (u[0] - v[0], u[1] - v[1])
sq = lambda u: dot(u, u)
eta, w_star = 0.5, (15 / 7, -65 / 28)
w1, v1 = (0.0, 0.0), (-0.6, -0.6)
w2 = sub(w1, (eta * v1[0], eta * v1[1]))
a1, a2 = sq(sub(w1, w_star)), sq(sub(w2, w_star))
progress = dot(sub(w1, w_star), v1)
print(round(a1, 4), round(a2, 4))
print(round(progress, 4),
      round((a1 - a2) / (2 * eta) + eta / 2 * sq(v1), 4))
```

It prints `9.9809 10.268`, then `-0.1071 -0.1071`. This step moved *away* from the target: message B pushed the weight on “meeting” up, the squared distance grew from 9.98 to 10.27, and the progress is negative. Both sides agree anyway, because the identity holds for every step, good or bad. Every variant below changes one ingredient of this identity, and asks what survives.

## Variants

The SGD of lesson 14a has a few useful relatives. Each comes with the same style of guarantee, and one of them buys a much faster rate.

### Adding a projection step

The analysis compared SGD with the best \(\mathbf w^\star\) of norm at most \(B\), the class \(\mathcal H = \{\mathbf w : \|\mathbf w\| \le B\}\). But nothing stops the iterates from wandering outside that ball, and then \(\bar{\mathbf w}\) need not be in \(\mathcal H\) either. If the class is part of your inductive bias, you want the output inside it. The fix is a two-part update:

1. \(\mathbf w^{(t + \frac12)} = \mathbf w^{(t)} - \eta \mathbf v_t\),
2. \(\mathbf w^{(t+1)} = \operatorname{argmin}_{\mathbf w \in \mathcal H} \|\mathbf w - \mathbf w^{(t+\frac12)}\|\).

The half in the superscript marks a halfway point, the ordinary step before any correction. The second part is the **projection** onto \(\mathcal H\): \(\operatorname{argmin}_{\mathbf w \in \mathcal H}\) picks the point of \(\mathcal H\) closest to the halfway point. In plain words: **take the usual step, then pull back to the nearest point of the class.** For a ball it is just rescaling, \(\mathbf w \mapsto \mathbf w \cdot \min(1, B/\|\mathbf w\|)\), where the arrow \(\mapsto\) reads “is sent to”: with \(B = 1\), the point \((3, 4)\) of length 5 is sent to \((0.6, 0.8)\), and \((0.3, 0.4)\) stays put. The `project` line of the check below is this rescaling. Every iterate is now in \(\mathcal H\), and since \(\mathcal H\) is convex, so is their average. Does the guarantee survive? It does, because projecting never moves you further from any point of the set.

**Lemma (projection).** Let \(\mathcal H\) be a closed convex set and let \(\mathbf v = \operatorname{argmin}_{\mathbf x \in \mathcal H} \|\mathbf x - \mathbf w\|^2\) be the projection of \(\mathbf w\) onto \(\mathcal H\). Then for every \(\mathbf u \in \mathcal H\),

\[ \|\mathbf w - \mathbf u\|^2 - \|\mathbf v - \mathbf u\|^2 \ge 0. \]

In plain words: **projecting onto a convex set never moves you further from any point inside it.** With \(\mathbf w = (3, 4)\), its projection \(\mathbf v = (0.6, 0.8)\) onto the unit ball and \(\mathbf u = \mathbf 0\): \(25 - 1 = 24 \ge 0\).

*Proof.* Fix \(\mathbf u \in \mathcal H\). For every \(\alpha \in (0, 1)\), the point \(\mathbf v + \alpha(\mathbf u - \mathbf v)\) lies on the segment between two points of \(\mathcal H\), so it is in \(\mathcal H\) by convexity. Since \(\mathbf v\) is the closest point of \(\mathcal H\) to \(\mathbf w\),

\[ \|\mathbf v - \mathbf w\|^2 \le \|\mathbf v + \alpha(\mathbf u - \mathbf v) - \mathbf w\|^2 = \|\mathbf v - \mathbf w\|^2 + 2\alpha \langle \mathbf v - \mathbf w, \mathbf u - \mathbf v \rangle + \alpha^2 \|\mathbf u - \mathbf v\|^2, \]

where the equality expands the square of a sum. Cancel \(\|\mathbf v - \mathbf w\|^2\) and divide by \(\alpha > 0\): \(2\langle \mathbf v - \mathbf w, \mathbf u - \mathbf v \rangle \ge -\alpha \|\mathbf u - \mathbf v\|^2\). Let \(\alpha \to 0\): \(\langle \mathbf v - \mathbf w, \mathbf u - \mathbf v \rangle \ge 0\). In words, the angle at \(\mathbf v\) between the way back to \(\mathbf w\) and the way to \(\mathbf u\) is at least a right angle. Now expand once more, writing \(\mathbf w - \mathbf u = (\mathbf w - \mathbf v) + (\mathbf v - \mathbf u)\):

\[ \|\mathbf w - \mathbf u\|^2 = \|\mathbf w - \mathbf v\|^2 + \|\mathbf v - \mathbf u\|^2 + 2\langle \mathbf v - \mathbf w, \mathbf u - \mathbf v \rangle \ge \|\mathbf v - \mathbf u\|^2, \]

since the first and last terms are nonnegative. \(\blacksquare\)

[[diagram:projection]]

A check with a hundred thousand random points outside and inside the unit ball:

```python
import math
import random

rng, B = random.Random(0), 1.0
norm = lambda u: math.hypot(*u)
dist2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
project = lambda w: tuple(c * min(1, B / norm(w)) for c in w)
gaps = []
for _ in range(100000):
    w = (rng.uniform(-4, 4), rng.uniform(-4, 4))
    r = B * math.sqrt(rng.random())
    a = rng.uniform(0, 2 * math.pi)
    u = (r * math.cos(a), r * math.sin(a))
    gaps.append(dist2(w, u) - dist2(project(w), u))
print(min(gaps) >= 0)
```

It prints `True`.

Now plug the lemma into the step lemma. The only place the update rule entered its proof was the identity with \(\|\mathbf w^{(t+1)} - \mathbf w^\star\|^2\). With projection, that identity holds for the half-step \(\mathbf w^{(t+\frac12)}\) instead, since \(\mathbf w^{(t+\frac12)} = \mathbf w^{(t)} - \eta\mathbf v_t\). Since \(\mathbf w^\star \in \mathcal H\), the lemma gives \(\|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 \le \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2\), so putting \(\mathbf w^{(t+1)}\) in place of the half-step can only make the right side larger:

\[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \le \frac{1}{2\eta} \left( \|\mathbf w^{(t)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 \right) + \frac\eta2 \|\mathbf v_t\|^2. \]

The per-step equality becomes an inequality in the right direction, the telescoping goes through unchanged, and the step lemma, the SGD guarantee and the learning corollary all hold with projection added.

### Variable step size

Instead of a fixed \(\eta\), you can shrink the step as you go, \(\eta_t = B/(\rho\sqrt t)\). The intuition: early on you are far from the minimum and big steps help; later, when you are close, careful steps avoid overshooting it. This schedule achieves a bound similar to the SGD guarantee, and it has a practical advantage: it does not need to know \(T\) in advance. You will prove it in Problem 2.

### Other averaging

The output need not be the plain average of all iterates. Alternatives include returning \(\mathbf w^{(t)}\) for a random \(t \in [T]\) (one of the steps \(1, \dots, T\)), averaging only the last \(\alpha T\) iterates for some \(\alpha \in (0, 1)\), or a weighted average of the last few. These schemes can converge faster in some situations, notably for the strongly convex functions we turn to now.

### Strongly convex functions

Recall from lesson 13 that a strongly convex function sags below each of its chords by at least a fixed multiple of the chord’s squared length. Lesson 13 wrote the constant as \(\sigma\); here we call it \(\lambda\), because below it will be the regularization weight. So \(f\) is **\(\lambda\)-strongly convex** when, for all \(\mathbf w, \mathbf u\) and \(\alpha \in (0, 1)\),

\[ f(\alpha\mathbf w + (1-\alpha)\mathbf u) \le \alpha f(\mathbf w) + (1-\alpha) f(\mathbf u) - \frac\lambda2 \alpha(1-\alpha)\|\mathbf w - \mathbf u\|^2. \]

The first two terms on the right are the chord, as in plain convexity, and the subtracted term is the guaranteed sag, largest at the chord’s middle, \(\alpha = \frac12\). In plain words: **the graph curves upward at least as much as the bowl \(\frac\lambda2\|\mathbf w\|^2\).** The bowl itself, with \(\lambda = 0.1\) in one dimension, \(w = 1\), \(u = -1\) and \(\alpha = \frac12\): the left side is \(f(0) = 0\), and the right side is \(0.05 - 0.05 \cdot \frac14 \cdot 4 = 0\), so the bowl just meets the requirement. That extra curvature buys a much better rate. We need one inequality, which strengthens the subgradient inequality by a quadratic term; proving it is Problem 1.

**Claim (strong convexity and subgradients).** If \(f\) is \(\lambda\)-strongly convex, then for every \(\mathbf w, \mathbf u\) and \(\mathbf v \in \partial f(\mathbf w)\),

\[ \langle \mathbf w - \mathbf u, \mathbf v \rangle \ge f(\mathbf w) - f(\mathbf u) + \frac\lambda2 \|\mathbf w - \mathbf u\|^2. \]

Rearranged, \(f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \mathbf v \rangle + \frac\lambda2\|\mathbf w - \mathbf u\|^2\). In plain words: **a strongly convex function rises above each supporting plane by at least \(\frac\lambda2\) times the squared distance.** For the same bowl at \(w = 1\), the subgradient is \(v = 0.1\), and both sides equal \(0.2\) at \(u = -1\). A random check of both inequalities on a bowl with a crease, \(0.05w^2 + |w|\):

```python
import random

rng, lam = random.Random(0), 0.1
f = lambda w: lam / 2 * w * w + abs(w)
sub = lambda w: lam * w + (1 if w > 0 else -1 if w < 0 else 0)
chord, plane = [], []
for _ in range(100000):
    w, u, a = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.random()
    sag = lam / 2 * a * (1 - a) * (w - u) ** 2
    mid = f(a * w + (1 - a) * u)
    chord.append(a * f(w) + (1 - a) * f(u) - sag - mid)
    plane.append((w - u) * sub(w) - (f(w) - f(u))
                 - lam / 2 * (w - u) ** 2)
print(min(chord) > -1e-12, min(plane) > -1e-12)
```

It prints `True True`.

The algorithm changes in two places: the step size is \(\eta_t = 1/(\lambda t)\), decreasing, and there is a projection onto a closed convex set \(\mathcal H\).

```
SGD-StronglyConvex(f, λ, T, H):
  w(1) ← 0
  for t = 1..T:
    choose v_t at random so that
      E[v_t | v_1, ..., v_{t-1}] ∈ ∂f(w(t))
    η_t ← 1 / (λ t)
    w(t+½) ← w(t) − η_t v_t
    w(t+1) ← argmin_{w ∈ H} ‖w − w(t+½)‖²
  return w̄ = (1/T) Σ_{t=1..T} w(t)
```

**Theorem (SGD for strongly convex functions).** Let \(f\) be \(\lambda\)-strongly convex, assume \(\mathbb E[\|\mathbf v_t\|^2] \le \rho^2\) for every \(t\), and let \(\mathbf w^\star \in \operatorname{argmin}_{\mathbf w \in \mathcal H} f(\mathbf w)\). Then

\[ \mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le \frac{\rho^2}{2\lambda T}\left(1 + \log T\right). \]

The assumption now bounds the average squared step length \(\mathbb E[\|\mathbf v_t\|^2]\) rather than every step, and \(\log\) is the natural logarithm again. In plain words: **with step sizes \(1/(\lambda t)\), the expected gap shrinks like \(\log T\) over \(T\).** With \(\rho = 1\), \(\lambda = 0.1\) and \(T = 1{,}000\): \((1 + 6.908)/200 \approx 0.040\). The last snippet of “SGD for regularized loss minimization” computes this bound (with \(4\rho^2\) in place of \(\rho^2\)). Compare the rates: \(1/\sqrt T\) before, \(\log(T)/T\) now. To halve the gap you used to need four times the steps; now about twice.

[[diagram:rates]]

*Proof.* Write \(a_t = \|\mathbf w^{(t)} - \mathbf w^\star\|^2\) for the squared distance to the optimum, and \(\nabla^{(t)} = \mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}]\), a subgradient of \(f\) at \(\mathbf w^{(t)}\) by the algorithm’s requirement. All expectations below are over the whole run.

*Lower bound on the progress term.* The claim, at \(\mathbf w^{(t)}\) with \(\mathbf u = \mathbf w^\star\) and the subgradient \(\nabla^{(t)}\), holds for every outcome:

\[ \langle \mathbf w^{(t)} - \mathbf w^\star, \nabla^{(t)} \rangle \ge f(\mathbf w^{(t)}) - f(\mathbf w^\star) + \frac\lambda2 a_t. \]

*Upper bound on the progress term.* Expand the half-step as in the step lemma: \(\|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 = a_t - 2\eta_t\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle + \eta_t^2 \|\mathbf v_t\|^2\). The projection lemma, with \(\mathbf w^\star \in \mathcal H\), gives \(a_{t+1} \le \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2\). Together,

\[ a_t - a_{t+1} \ge a_t - \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 = 2\eta_t \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle - \eta_t^2 \|\mathbf v_t\|^2. \]

Take expectations. By the same total-expectation argument as in the SGD guarantee of lesson 14a (fix the past, pull out the fixed vector \(\mathbf w^{(t)} - \mathbf w^\star\)), \(\mathbb E\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle = \mathbb E\langle \mathbf w^{(t)} - \mathbf w^\star, \nabla^{(t)} \rangle\). Using \(\mathbb E\|\mathbf v_t\|^2 \le \rho^2\) and dividing by \(2\eta_t\),

\[ \mathbb E\langle \mathbf w^{(t)} - \mathbf w^\star, \nabla^{(t)} \rangle \le \frac{\mathbb E[a_t - a_{t+1}]}{2\eta_t} + \frac{\eta_t}{2}\rho^2. \]

*Combine and sum.* Take the expectation of the lower bound and chain it with the upper bound:

\[ \mathbb E[f(\mathbf w^{(t)})] - f(\mathbf w^\star) \le \mathbb E\left[ \frac{a_t - a_{t+1}}{2\eta_t} - \frac\lambda2 a_t \right] + \frac{\eta_t}{2}\rho^2. \]

Now \(1/(2\eta_t) = \lambda t/2\), so the bracket is \(\frac\lambda2\left( t a_t - t a_{t+1} - a_t \right) = \frac\lambda2\left( (t-1)a_t - t\,a_{t+1} \right)\). Summed over \(t = 1, \dots, T\), this telescopes: the term \(-t\,a_{t+1}\) at step \(t\) cancels the term \((t' - 1)a_{t'}\) at step \(t' = t + 1\), the first term is \(0 \cdot a_1 = 0\), and what remains is \(-\frac{\lambda T}{2} a_{T+1} \le 0\). So

\[ \sum_{t=1}^T \left( \mathbb E[f(\mathbf w^{(t)})] - f(\mathbf w^\star) \right) \le \frac{\rho^2}{2} \sum_{t=1}^T \eta_t = \frac{\rho^2}{2\lambda} \sum_{t=1}^T \frac1t \le \frac{\rho^2}{2\lambda}(1 + \log T). \]

The last step bounds the harmonic sum: \(\sum_{t=1}^T 1/t \le 1 + \int_1^T ds/s = 1 + \log T\), because each \(1/t\) for \(t \ge 2\) is at most the integral of \(1/s\) over \([t-1, t]\). Finally divide by \(T\) and apply Jensen, \(f(\bar{\mathbf w}) \le \frac1T\sum_t f(\mathbf w^{(t)})\), inside the expectation. \(\blacksquare\)

(Sharper analyses remove the \(\log T\): for instance by averaging only the last \(T/2\) iterates, and it is also known that the bound holds for the last iterate \(\mathbf w^{(T)}\) itself. We will not prove those.)

Before we put this rate to work, watch it run. Take the six messages again, and add to their hinge risk the bowl with \(\lambda = 0.1\): the function is \(f(\mathbf w) = 0.05\|\mathbf w\|^2\) plus the average hinge loss over the six. The bowl is \(0.1\)-strongly convex (lesson 13 showed it, and the section “SGD for regularized loss minimization” recalls why), and adding the convex hinge risk keeps that, so the theorem applies. At \(\mathbf w = (1, -1)\) this \(f\) is about 0.7833, and its lowest value is about 0.7466. Each step draws one message at random and steps along \(\lambda\mathbf w^{(t)}\) plus that message’s hinge subgradient; the section “SGD for regularized loss minimization” shows that this is an unbiased estimate of a subgradient of \(f\).

In the lab below you pick the schedule, either a constant step size \(\eta\) or the shrinking \(\eta_t = 1/(\lambda t)\), and the number of steps \(T\). You watch the path on the landscape of \(f\), and how far the average lands above the bottom, over 100 runs. Before you start, predict: with \(T = 4{,}000\) for both, which ends closer to the bottom, a constant \(\eta = 0.2\) or \(\eta_t = 1/(\lambda t)\)? Then push \(T\) up and watch which one keeps improving.

[[lab]]

## Smooth losses

Lesson 12 had a second family besides convex-Lipschitz: losses that are **\(\beta\)-smooth**, meaning the gradient is \(\beta\)-Lipschitz, \(\|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \beta\|\mathbf v - \mathbf w\|\): the slope itself changes by at most \(\beta\) (beta) times the distance moved. The square loss and the logistic loss on bounded messages are examples. Lesson 12 also showed the key property we need: a nonnegative \(\beta\)-smooth function is **self-bounded**, \(\|\nabla f(\mathbf w)\|^2 \le 2\beta f(\mathbf w)\), the squared slope at most \(2\beta\) times the height. Its gradient is small wherever its value is small. (For \(f(w) = \frac12 w^2\), which is 1-smooth, both sides equal \(w^2\).) Here is why that helps: in the step lemma, the price of noise was \(\frac\eta2\sum_t\|\mathbf v_t\|^2\), and self-boundedness turns that price into a multiple of the loss itself.

**Theorem (SGD on smooth losses).** Assume that for every \(z\), the loss \(\ell(\cdot, z)\) is convex, \(\beta\)-smooth and nonnegative, and let \(\eta < 1/\beta\). Then SGD-Risk satisfies, for every \(\mathbf w^\star\),

\[ \mathbb E[L_{\mathcal D}(\bar{\mathbf w})] \le \frac{1}{1 - \eta\beta}\left( L_{\mathcal D}(\mathbf w^\star) + \frac{\|\mathbf w^\star\|^2}{2\eta T} \right). \]

Inside the bracket are the comparator’s own risk and the familiar distance term; the factor \(1/(1 - \eta\beta)\), a little above 1 when \(\eta\) is small, inflates both. In plain words: **SGD’s expected risk is at most the comparator’s risk plus \(\|\mathbf w^\star\|^2/(2\eta T)\), inflated by \(1/(1 - \eta\beta)\).** With \(\beta = 1\), \(\eta = 0.25\), \(T = 100\), \(\|\mathbf w^\star\| = 1\) and \(L_{\mathcal D}(\mathbf w^\star) = 0.5\): \((0.5 + 0.02)/0.75 \approx 0.693\). The `excess` line of the snippet after the corollary below is this formula, minus the comparator’s risk.

*Proof.* Let \(z_1, \dots, z_T\) be the fresh examples SGD draws, and \(f_t(\cdot) = \ell(\cdot, z_t)\). Then \(\mathbf v_t = \nabla f_t(\mathbf w^{(t)})\). Each \(f_t\) is convex, so its tangent plane at \(\mathbf w^{(t)}\) gives \(f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \le \langle \mathbf v_t, \mathbf w^{(t)} - \mathbf w^\star \rangle\). Sum over \(t\) and apply the step lemma:

\[ \sum_{t=1}^T \left( f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \right) \le \sum_{t=1}^T \langle \mathbf v_t, \mathbf w^{(t)} - \mathbf w^\star \rangle \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac\eta2 \sum_{t=1}^T \|\mathbf v_t\|^2. \]

Self-boundedness gives \(\|\mathbf v_t\|^2 \le 2\beta f_t(\mathbf w^{(t)})\), so the last sum is at most \(\eta\beta\sum_t f_t(\mathbf w^{(t)})\). Divide by \(T\), move that term to the left, and divide by \(1 - \eta\beta > 0\):

\[ \frac1T \sum_{t=1}^T f_t(\mathbf w^{(t)}) \le \frac{1}{1 - \eta\beta}\left( \frac1T \sum_{t=1}^T f_t(\mathbf w^\star) + \frac{\|\mathbf w^\star\|^2}{2\eta T} \right). \]

This holds for every outcome. Now take expectations over \(z_1, \dots, z_T\). On the right, \(\mathbf w^\star\) is fixed and each \(z_t \sim \mathcal D\), so \(\mathbb E[f_t(\mathbf w^\star)] = L_{\mathcal D}(\mathbf w^\star)\). On the left, \(\mathbf w^{(t)}\) depends only on \(z_1, \dots, z_{t-1}\), and \(z_t\) is independent of them; so, conditioning on the past as before, \(\mathbb E[f_t(\mathbf w^{(t)}) \mid z_{1:t-1}] = L_{\mathcal D}(\mathbf w^{(t)})\), and \(\mathbb E[f_t(\mathbf w^{(t)})] = \mathbb E[L_{\mathcal D}(\mathbf w^{(t)})]\). Finally Jensen, for the convex \(L_{\mathcal D}\), gives \(\frac1T\sum_t L_{\mathcal D}(\mathbf w^{(t)}) \ge L_{\mathcal D}(\bar{\mathbf w})\). \(\blacksquare\)

Notice the shape: if the best hypothesis has small risk, the noise term almost vanishes. Setting the knobs turns this into a sample complexity:

**Corollary (SGD learns convex-smooth-bounded problems).** Consider a convex-smooth-bounded learning problem with parameters \(\beta, B\) (lesson 12: \(\mathcal H\) convex with \(\|\mathbf w\| \le B\), each \(\ell(\cdot, z)\) convex, nonnegative and \(\beta\)-smooth), with \(\mathbf 0 \in \mathcal H\) and \(\ell(\mathbf 0, z) \le 1\) for every \(z\). For every \(\varepsilon \in (0, 1]\), set \(\eta = 1/(\beta(1 + 3/\varepsilon))\). Then SGD-Risk with \(T \ge 12B^2\beta/\varepsilon^2\) gives

\[ \mathbb E[L_{\mathcal D}(\bar{\mathbf w})] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \varepsilon. \]

This is the same promise as for convex-Lipschitz-bounded problems, now bought with \(12B^2\beta/\varepsilon^2\) messages: with \(B = \beta = 1\) and \(\varepsilon = 0.1\), that is 1,200.

*Proof.* Let \(\mathbf w^\star\) minimize \(L_{\mathcal D}\) over \(\mathcal H\), and \(L^\star = L_{\mathcal D}(\mathbf w^\star)\). Since \(\mathbf 0 \in \mathcal H\), \(L^\star \le L_{\mathcal D}(\mathbf 0) \le 1\), and \(\|\mathbf w^\star\| \le B\). With this \(\eta\), \(\eta\beta = \varepsilon/(\varepsilon + 3)\), so \(1/(1 - \eta\beta) = 1 + \varepsilon/3\). And \(\|\mathbf w^\star\|^2/(2\eta T) \le B^2\beta(1 + 3/\varepsilon)/(2T) \le \varepsilon^2(1 + 3/\varepsilon)/24 = \varepsilon(\varepsilon + 3)/24\), using \(T \ge 12B^2\beta/\varepsilon^2\). The theorem gives

\[ \mathbb E[L_{\mathcal D}(\bar{\mathbf w})] - L^\star \le \frac\varepsilon3 L^\star + \left(1 + \frac\varepsilon3\right)\frac{\varepsilon(\varepsilon + 3)}{24} \le \frac\varepsilon3 + \frac43 \cdot \frac{4\varepsilon}{24} = \frac{5\varepsilon}{9} \le \varepsilon, \]

where the middle step used \(L^\star \le 1\) and \(\varepsilon \le 1\) (so \(1 + \varepsilon/3 \le 4/3\) and \(\varepsilon + 3 \le 4\)). \(\blacksquare\)

The arithmetic, for a few values of \(\varepsilon\) and \(L^\star \in \{0, 0.5, 1\}\), with \(\beta = B = 1\):

```python
beta, B = 1.0, 1.0
for eps in (1.0, 0.5, 0.1, 0.01):
    eta = 1 / (beta * (1 + 3 / eps))
    T = 12 * B**2 * beta / eps**2
    excess = [(L + B**2 / (2 * eta * T)) / (1 - eta * beta) - L
              for L in (0.0, 0.5, 1.0)]
    print(eps, [round(e, 4) for e in excess], max(excess) <= eps)
```

It prints `1.0 [0.2222, 0.3889, 0.5556] True`, then `0.5 [0.0851, 0.1684, 0.2517] True`, `0.1 [0.0133, 0.03, 0.0467] True` and `0.01 [0.0013, 0.0029, 0.0046] True`. The worst case, \(L^\star = 1\) at \(\varepsilon = 1\), is exactly \(5/9\). Lesson 13’s regularized learner needed \(150\beta B^2/\varepsilon^2\) examples for this guarantee; SGD needs \(12\beta B^2/\varepsilon^2\).

## SGD for regularized loss minimization

SGD matches regularized loss minimization’s worst-case sample complexity, and beats its constants: \(B^2\rho^2/\varepsilon^2\) against \(8\rho^2B^2/\varepsilon^2\) for convex-Lipschitz-bounded problems (lesson 14a), and \(12\beta B^2/\varepsilon^2\) against \(150\beta B^2/\varepsilon^2\) for convex-smooth-bounded ones (“Smooth losses”). But on some distributions the regularized solution is genuinely better, and sometimes you want exactly the lesson 13 objective:

\[ \min_{\mathbf w} \; f(\mathbf w) = \frac\lambda2 \|\mathbf w\|^2 + L_S(\mathbf w). \]

Here \(\min_{\mathbf w}\) asks for the smallest value over all weight vectors, \(\frac\lambda2\|\mathbf w\|^2\) is the bowl, a charge on squared length scaled by the regularization weight \(\lambda\) (lambda), and \(L_S\) is the training fit. Lesson 13 charged \(\lambda\|\mathbf w\|^2\); the half is for convenience and only rescales \(\lambda\). In plain words: **find the weight vector with the best trade between training fit and half of \(\lambda\) times its squared length.** With \(\lambda = 0.1\), our six messages as \(S\) and the hinge loss, \(\mathbf w = (1, -1)\) pays \(0.1\) for its length and \(0.6833\) for its fit:

```python
S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
L_S = lambda w: sum(max(0, 1 - y * dot(w, x)) for x, y in S) / 6
lam, w = 0.1, (1.0, -1.0)
bowl = lam / 2 * dot(w, w)
print(round(bowl, 4), round(L_S(w), 4), round(bowl + L_S(w), 4))
```

It prints `0.1 0.6833 0.7833`.

Can SGD solve it fast? This \(f\) is \(\lambda\)-strongly convex: lesson 13 showed \(\lambda\|\mathbf w\|^2\) is \(2\lambda\)-strongly convex, so \(\frac\lambda2\|\mathbf w\|^2\) is \(\lambda\)-strongly convex, and adding the convex \(L_S\) keeps that. A strongly convex continuous function has a unique minimizer. So we can use the strongly convex variant with \(\mathcal H = \mathbb R^d\), where the projection does nothing. We need an unbiased subgradient estimate:

Pick \(z\) uniformly at random from \(S\), take \(\mathbf v_t \in \partial \ell(\mathbf w^{(t)}, z)\) by a fixed rule, and step along \(\lambda\mathbf w^{(t)} + \mathbf v_t\).

Why is its conditional mean a subgradient of \(f\)? Given the past, the mean is \(\lambda\mathbf w^{(t)} + \frac1m \sum_{i=1}^m \mathbf v^{(i)}\), where \(\mathbf v^{(i)}\) is the rule’s subgradient for example \(z_i\). For every \(\mathbf u\), each example gives \(\ell(\mathbf u, z_i) \ge \ell(\mathbf w^{(t)}, z_i) + \langle \mathbf u - \mathbf w^{(t)}, \mathbf v^{(i)} \rangle\); averaging over \(i\) gives the same for \(L_S\) with the averaged subgradient. And the bowl is differentiable and convex with gradient \(\lambda\mathbf w\), so \(\frac\lambda2\|\mathbf u\|^2 \ge \frac\lambda2\|\mathbf w^{(t)}\|^2 + \langle \mathbf u - \mathbf w^{(t)}, \lambda\mathbf w^{(t)} \rangle\). Add the two inequalities: the sum of subgradients is a subgradient of the sum.

Now look at what the update does with \(\eta_t = 1/(\lambda t)\):

\[ \begin{aligned} \mathbf w^{(t+1)} &= \mathbf w^{(t)} - \frac{1}{\lambda t}\left( \lambda \mathbf w^{(t)} + \mathbf v_t \right) = \left(1 - \frac1t\right)\mathbf w^{(t)} - \frac{1}{\lambda t}\mathbf v_t = \frac{t-1}{t}\mathbf w^{(t)} - \frac{1}{\lambda t}\mathbf v_t \\ &= \frac{t-1}{t}\left( \frac{t-2}{t-1}\mathbf w^{(t-1)} - \frac{1}{\lambda(t-1)}\mathbf v_{t-1} \right) - \frac{1}{\lambda t}\mathbf v_t = \dots = -\frac{1}{\lambda t}\sum_{i=1}^t \mathbf v_i. \end{aligned} \]

Each unrolling multiplies the earlier coefficient by \(\frac{t-1}{t}\), which turns \(\frac{1}{\lambda(t-1)}\) into \(\frac{1}{\lambda t}\), so every past subgradient ends up with the same weight \(\frac{1}{\lambda t}\); the unrolling stops at \(\frac{0}{1}\mathbf w^{(1)} = \mathbf 0\). In plain words: **the iterate is just minus the running average of the subgradients, divided by \(\lambda\).** With \(\lambda = 0.1\), if the first draw is message B, then \(\mathbf w^{(2)} = -\frac{1}{0.1}(-0.6, -0.6) = (6, 6)\). If the loss is \(\rho\)-Lipschitz, every \(\|\mathbf v_i\| \le \rho\), so \(\|\lambda\mathbf w^{(t)}\| \le \frac1{t-1}\sum_{i<t} \|\mathbf v_i\| \le \rho\), and therefore

\[ \|\lambda\mathbf w^{(t)} + \mathbf v_t\| \le 2\rho. \]

In plain words: **each step vector is at most twice as long as a loss subgradient.** Here is the closed form checked against the step-by-step update, on our six messages with \(\lambda = 0.1\) and the hinge loss (\(\rho = 1\)):

```python
import math
import random

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
lam, rho, T, rng = 0.1, 1.0, 1000, random.Random(0)
w, vs, biggest = (0.0, 0.0), [], 0.0
for t in range(1, T + 1):
    x, y = rng.choice(S)
    v = (-y * x[0], -y * x[1]) if y * dot(w, x) < 1 else (0, 0)
    vs.append(v)
    w = tuple(c - (lam * c + g) / (lam * t) for c, g in zip(w, v))
    biggest = max(biggest, lam * math.hypot(*w))
closed = [-sum(u[j] for u in vs) / (lam * T) for j in (0, 1)]
print([round(c, 6) for c in w], [round(c, 6) for c in closed])
print(round(biggest, 3), "<=", rho)
```

It prints `[1.517, -1.25] [1.517, -1.25]`, then `0.906 <= 1.0`.

So the step vectors satisfy \(\mathbb E\|\lambda\mathbf w^{(t)} + \mathbf v_t\|^2 \le (2\rho)^2\), and the strongly convex theorem, with \(4\rho^2\) in place of \(\rho^2\), gives after \(T\) steps

\[ \mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le \frac{4\rho^2}{2\lambda T}(1 + \log T) = \frac{2\rho^2}{\lambda T}(1 + \log T). \]

In plain words: **after \(T\) one-example steps, the regularized objective is, on average, within \(2\rho^2(1 + \log T)/(\lambda T)\) of its minimum.** The snippet mirrors it term by term, with the harmonic sum beside it:

```python
import math

lam, rho = 0.1, 1.0
for T in (10, 100, 1000, 10000):
    harmonic = sum(1 / t for t in range(1, T + 1))
    bound = (2 * rho) ** 2 / (2 * lam * T) * (1 + math.log(T))
    print(T, round(harmonic, 3), round(1 + math.log(T), 3),
          round(bound, 4))
```

It prints `10 2.929 3.303 6.6052`, `100 5.187 5.605 1.121`, `1000 7.485 7.908 0.1582` and `10000 9.788 10.21 0.0204`. The harmonic sum always sits under \(1 + \log T\), and the bound falls almost like \(1/T\). Each step reads one example from \(S\), so this is a solver for the lesson 13 objective whose cost does not grow with \(m\). That is the solver the auditor asked for, and the one the next lesson builds on.

## The regularized solver as code

Here is the solver the auditor’s filter will run. It uses the closed form, so it never stores more than one running sum of subgradients: each step adds the drawn message’s hinge subgradient to the sum, and the new iterate is minus that sum over \(\lambda t\).

```
SGD-RLM(S, λ, T):
  w ← 0 in R^d;  sum ← 0 in R^d;  total ← 0 in R^d
  for t = 1..T:
    total ← total + w
    (x, y) ← a message drawn uniformly at random from S
    if y ⟨w, x⟩ < 1:
      sum ← sum − y · x
    w ← −sum / (λ t)
  return w̄ = total / T
```

**Cost.** Each step reads one message, does one inner product and at most one vector addition, and rescales \(\mathbf w\): \(O(d)\) time, or \(O(s)\) with sparse messages of \(s\) nonzero counts if \(\mathbf w\) is kept as the sum together with the scalar \(-1/(\lambda t)\). Total \(O(Td)\) time and \(O(d)\) working space, on top of the \(O(md)\) it takes to keep the audited sequence on file. The certificate asks for the smallest \(T\) with \(2\rho^2(1 + \log T)/(\lambda T) \le \varepsilon\), and that count has no \(m\) in it: the exact solver’s cost grows with the archive, this one’s does not.

**Build it yourself.** Using only the standard library:

```python
import math
import random

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]

def objective(w, S, lam): ...
def rlm_certificate(lam, rho, T): ...
def steps_for(lam, rho, eps): ...
def project(w, B): ...
def sgd_rlm(S, lam, T, rng): ...

if __name__ == "__main__":
    assert round(objective((1, -1), S, 0.1), 4) == 0.7833
    assert round(rlm_certificate(0.1, 1, 10000), 4) == 0.0204
    assert steps_for(0.1, 1, 0.05) == 3685
    assert steps_for(0.01, 1, 0.05) == 47035
    assert [round(c, 9) for c in project((3, 4), 1)] == [0.6, 0.8]
    assert project((0.3, 0.4), 1) == (0.3, 0.4)
    w_bar, w_last = sgd_rlm(S, 0.1, 1000, random.Random(0))
    assert [round(c, 3) for c in w_last] == [1.517, -1.25]
    f_star = objective((11 / 7, -9 / 7), S, 0.1)
    rng = random.Random(1)
    gaps = [objective(sgd_rlm(S, 0.1, 1000, rng)[0], S, 0.1)
            - f_star for _ in range(100)]
    assert 0 <= sum(gaps) / 100 <= rlm_certificate(0.1, 1, 1000)
```

1. Implement the functions. `objective` is \(\frac\lambda2\|\mathbf w\|^2 + L_S(\mathbf w)\) with the hinge loss. `rlm_certificate` is \(2\rho^2(1 + \log T)/(\lambda T)\). `steps_for` returns the smallest \(T\) whose certificate is at most \(\varepsilon\) (the certificate falls as \(T\) grows past 1, so a doubling search followed by a binary search works). `project` scales \(\mathbf w\) back onto the ball of radius \(B\) when it is outside and leaves it alone otherwise.
2. `sgd_rlm` follows the pseudocode, calls `rng.choice(S)` exactly once per step, and returns `(w_bar, w_last)`: the average of \(\mathbf w^{(1)}, \dots, \mathbf w^{(T)}\) and the iterate after the last step. With `random.Random(0)` it must reproduce the closed-form snippet of “SGD for regularized loss minimization”. The lowest point of this objective is \((11/7, -9/7)\); check it against a fine grid.
3. Add a constant step size to `sgd_rlm` as an option, with the plain update \(\mathbf w \leftarrow \mathbf w - \eta(\lambda\mathbf w + \mathbf v_t)\). For \(T = 1{,}000\) to \(10{,}000\), plot the average gap over 100 runs for a few values of \(\eta\) next to the \(1/(\lambda t)\) schedule. Which keeps falling?
4. Now replace the random draw by a fixed cycle through \(S\) in order. Does the conditional mean of the step, given the past, still lie in \(\partial f(\mathbf w^{(t)})\)? Does the certificate still apply?

## Fix the bug

Back to the auditor. She will sign the filter that minimizes the regularized objective \(f(\mathbf w) = \frac\lambda2\|\mathbf w\|^2 + L_S(\mathbf w)\) on the audited training sequence, or any filter whose objective is, on average over SGD’s random draws, within a tolerance \(\varepsilon\) of that minimum. You now have the solver and its certificate. Each step draws one message from the audited sequence and steps along \(\lambda\mathbf w^{(t)} + \mathbf v_t\), and with the schedule \(\eta_t = 1/(\lambda t)\) the strongly convex theorem certifies the gap \(2\rho^2(1 + \log T)/(\lambda T)\) of “SGD for regularized loss minimization”.

The retrain job runs on a shared machine, so each night it gets a slot that fits a fixed number of steps. That is your budget. In the lab below, each round gives you the penalty weight \(\lambda\), the Lipschitz constant \(\rho = 1\) (messages are scaled to length at most 1), the tolerance \(\varepsilon\) and the step budget. You choose the schedule, a constant step size or \(\eta_t = 1/(\lambda t)\), and the number of steps \(T\). The filter ships only if this lesson can certify it: the schedule is one the strongly convex theorem covers, \(T\) fits the budget, and the certified gap is at most \(\varepsilon\). The lab runs the six-message world as a stand-in for the audited sequence and shows the measured gap over 100 runs, so you can watch it sit under your certificate. A constant step can land close in that measurement too. Ask yourself what certifies it. The measurement does not ship the filter; the certificate does.

Before you press train, a prediction. With \(\lambda = 0.1\), \(\rho = 1\) and a tolerance of 0.05, at least how many steps does the certificate need? Write a number down, and compare it with the \(B^2\rho^2/\varepsilon^2\) count of lesson 14a.

[[lab-act2]]

The auditor’s own numbers were \(\lambda = 0.01\) and a tolerance of 0.05. The snippet finds the smallest \(T\) the certificate accepts, prints the certified gap one step before and at that \(T\), and counts the work at about 200 nonzero word counts per message:

```python
import math

lam, rho, eps = 0.01, 1.0, 0.05
cert = lambda T: 2 * rho**2 * (1 + math.log(T)) / (lam * T)
T = 1
while cert(T) > eps:
    T += 1
print(T, f"{cert(T - 1):.7f} {cert(T):.7f}")
print(f"{T * 200:.1e}")
```

It prints `47035 0.0500005 0.0499996`, then `9.4e+06`. Forty-seven thousand steps, each reading one audited message: about \(9.4 \times 10^6\) operations, a fraction of a second, where the exact solver took six hours. And the count does not mention the two million at all. The auditor gets her filter, certified, every night.

The streaming filter of lesson 14a gets something too. Its optional projection, the scaling back onto the ball of radius \(B\), now has its proof: with the projection on, the SGD guarantee holds unchanged, and the shipped weights always sit inside the class the guarantee talks about.

## What people get wrong

**“Projecting the iterates breaks the guarantee, because the walk stops following the subgradients.”** Projecting onto a closed convex set never moves a point further from anything inside the set, including the target \(\mathbf w^\star\). So the per-step identity becomes an inequality in the right direction, and every bound survives.

**“Any shrinking step size gets the \(\log(T)/T\) rate.”** The fast rate needs a strongly convex function *and* the schedule \(\eta_t = 1/(\lambda t)\) matched to its curvature. On a merely convex function, the shrinking schedule \(B/(\rho\sqrt t)\) still gives only a rate of order \(1/\sqrt T\) (the second problem below). And if you claim more curvature than the function has, using a \(\lambda\) larger than the true one, the telescoping in the proof leaves positive terms behind and the bound is lost.

**“The fast rate means SGD learns from fewer fresh messages.”** The \(\log(T)/T\) rate is about reaching the minimum of a strongly convex objective, such as the regularized objective on a fixed training sequence. How good that minimum is as a filter is lesson 13’s question, with its own price for the penalty. For the true risk of a convex-Lipschitz-bounded problem, the count is still \(B^2\rho^2/\varepsilon^2\) fresh messages.

**“The measured gap is far below the certificate, so fewer steps would do.”** The measurement is one world and 100 runs. The certificate holds for every training sequence the auditor could hand you. A filter ships on what you can prove, not on how one test went.

## Recap and the next bug

1. **Projection** takes the usual step, then moves to the nearest point of the class \(\mathcal H\); for a ball it is a rescaling. Projecting onto a closed convex set never moves you further from any point of it, so the step lemma, the SGD guarantee and the learning corollary all hold with projection added.
2. The step size can shrink as \(\eta_t = B/(\rho\sqrt t)\), with no need to know \(T\) in advance and a bound of the same order, and the output can be another average of the iterates.
3. For a **\(\lambda\)-strongly convex** function, SGD with \(\eta_t = 1/(\lambda t)\) and projection gets \(\mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le \rho^2(1 + \log T)/(2\lambda T)\), a rate of \(\log(T)/T\) instead of \(1/\sqrt T\).
4. For convex, nonnegative, **\(\beta\)-smooth** losses, SGD-Risk gets a bound that shrinks with the best risk, and learns convex-smooth-bounded problems from \(12B^2\beta/\varepsilon^2\) examples.
5. SGD solves **regularized loss minimization**: step along \(\lambda\mathbf w^{(t)} + \mathbf v_t\) with \(\eta_t = 1/(\lambda t)\), and the iterate is minus the running average of the subgradients over \(\lambda\). Each step vector has length at most \(2\rho\), the gap is at most \(2\rho^2(1 + \log T)/(\lambda T)\), and each step costs one example, whatever the size of the training sequence.

The audited filter retrains in seconds now, and the auditor signs it. Then a colleague looks at the weights it shipped on a quiet week, when the junk and the keep messages could be split perfectly by a line. “Fine, it makes no mistakes on the labelled mail. But so do thousands of other weight vectors. And yours runs so close to some messages that adding a single word to one of them flips its label.” On separable mail, many weight vectors make no training mistakes. Which separator should we pick, and why would that choice generalize better? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The projection lemma asks for a closed convex set \(\mathcal H\). Find a set that is not convex, a point \(\mathbf w\), its projection \(\mathbf v\) and a point \(\mathbf u\) of the set for which projecting moves you *further* from \(\mathbf u\). Which line of the lemma’s proof fails for your set, and what would go wrong in the step lemma if SGD projected onto it?
2. Run SGD on the regularized objective of the six-message world with \(\lambda = 0.1\) and the hinge loss, with \(\eta_t = 1/(\lambda t)\). The first draw is keep message D, the second junk message A. Compute \(\mathbf w^{(2)}\) and \(\mathbf w^{(3)}\), both step by step and from the closed form, and check \(\|\lambda\mathbf w^{(t)}\| \le \rho\) for both. Why does the strongly convex theorem need this bound here, when lesson 14a’s hinge loss alone had \(\|\mathbf v_t\| \le \rho\) for free?
3. With \(\lambda = 0.1\) and \(\rho = 1\), how many steps does the certificate \(2\rho^2(1 + \log T)/(\lambda T)\) need to reach a tolerance of 0.02? By what factor does the certificate fall when \(T\) goes from 10,000 to 40,000, and by what factor would lesson 14a’s \(B\rho/\sqrt T\) fall over the same change?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (strong convexity and subgradients).** Prove the claim used for strongly convex functions: if \(f\) is \(\lambda\)-strongly convex and \(\mathbf v \in \partial f(\mathbf w)\), then \(\langle \mathbf w - \mathbf u, \mathbf v \rangle \ge f(\mathbf w) - f(\mathbf u) + \frac\lambda2\|\mathbf w - \mathbf u\|^2\) for every \(\mathbf u\). Start from the definition of strong convexity at the point \(\mathbf w + \alpha(\mathbf u - \mathbf w)\), and remember how lesson 13 handled a minimizer.
- **Problem 2 (variable step size; harder).** Prove an analog of the SGD guarantee for the step sizes \(\eta_t = B/(\rho\sqrt t)\), with the projection step onto \(\{\mathbf w : \|\mathbf w\| \le B\}\): a bound of order \(B\rho/\sqrt T\) on \(\mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star)\).
