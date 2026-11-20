# Convex Learnability and Surrogates

## The certificate operations would not sign

Lesson 12a ended on a small victory and an awkward email. The job that would have needed \(5 \times 10^{29}\) years finished in 87 steps: a walk down lesson 9’s logistic risk on the ten messages, stopped by a certificate proving that no weight vector in the ball of radius 3 does more than 0.01 better on those ten messages. Operations read the report and sent back two questions, and the certificate answers neither.

The first question: “Certified on which mail?” The proof is about the training sequence, the messages you already have. Operations wants a promise about next month’s mail, bought with the 3,000 labels the budget pays for. A colleague reading over your shoulder says the loss is convex, so of course it learns; just add labels.

The second question: “We count mistakes.” The certificate is about the logistic loss, a smooth score. A filter is judged by how many messages it gets wrong, the 0-1 loss, and lesson 12a’s opening showed that nobody can minimize that directly.

So the filter cannot ship. Operations’ requirement is concrete: a written bound, before training, saying that the expected fraction of mistakes on fresh mail is at most 0.36, using at most 3,000 labels. By the end of this lesson you will meet it with your own hands. On the way you will find out that the colleague is wrong: convexity alone does not learn, not even in one dimension, and the reason tells you exactly what to add.

### What you need from lesson 12a

A short reminder of the pieces this lesson leans on, each built and proved in lesson 12a.

- **The ten messages.** Each message is \(\mathbf x = (1, \text{free}, \text{meeting})\), a constant 1 for the bias term followed by two word counts; labels are \(+1\) for junk and \(-1\) for keep.

| message | \(\mathbf x = (1, \text{free}, \text{meeting})\) | label \(y\) |
|---|---|---|
| A | \((1, 3, 0)\) | \(+1\) |
| B | \((1, 2, 1)\) | \(+1\) |
| C | \((1, 4, 2)\) | \(+1\) |
| D | \((1, 1, 0)\) | \(+1\) |
| E | \((1, 1, 2)\) | \(-1\) |
| F | \((1, 0, 1)\) | \(-1\) |
| G | \((1, 2, 3)\) | \(-1\) |
| H | \((1, 0, 0)\) | \(-1\) |
| B′ | \((1, 2, 1)\) | \(-1\) |
| E′ | \((1, 1, 2)\) | \(+1\) |

- **Signed score and 0-1 loss.** For a weight vector \(\mathbf w\), the inner product \(\langle \mathbf w, \mathbf x \rangle\) multiplies matching entries and adds them up; the signed score \(y\langle \mathbf w, \mathbf x \rangle\) is positive when the halfspace agrees with the label. The 0-1 loss is \(\mathbb 1[y\langle \mathbf w, \mathbf x \rangle \le 0]\), 1 for a mistake (a zero score counts as one) and 0 otherwise. Lesson 9’s \(\mathbf w^\circ = (-1, 2, -2)\) makes the two forced mistakes, B′ and E′, so its 0-1 risk is 0.2, the floor.
- **Length.** \(\|\mathbf x\|\) is the Euclidean length, the square root of the sum of squared entries. The longest message is C, with \(\|\mathbf x\| = \sqrt{21} \approx 4.58\), and we write \(R\) for a bound on every message’s length.
- **Convex.** A set is convex when it contains the segment between any two of its points, such as the ball \(\{\mathbf w : \|\mathbf w\| \le B\}\). A function is convex when its graph lies on or below every chord, \(f(\alpha\mathbf u + (1 - \alpha)\mathbf v) \le \alpha f(\mathbf u) + (1 - \alpha) f(\mathbf v)\) for \(\alpha \in [0, 1]\). For convex functions every local minimum is global, and the tangent inequality \(f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\) puts a floor under the graph, where the gradient \(\nabla f(\mathbf w)\) lists how steeply \(f\) rises in each weight.
- **Building convex losses.** Convexity survives composition with a linear function, maxima, and sums with nonnegative weights. So the square loss, the logistic loss \(\log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\) and the **hinge loss** \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\), which charges how far the signed score falls short of 1, are convex in \(\mathbf w\), and so are their empirical risks.
- **Lipschitz.** \(f\) is \(\rho\)-Lipschitz when its output never moves more than \(\rho\) times as far as its input. Constants multiply under composition, so the hinge loss on a message \(\mathbf x\) is \(\|\mathbf x\|\)-Lipschitz in \(\mathbf w\), and \(R\)-Lipschitz on every message when \(\|\mathbf x\| \le R\).
- **Smooth.** \(f\) is \(\beta\)-smooth when its gradient is \(\beta\)-Lipschitz. Then a step of size \(1/\beta\) against the gradient lowers \(f\) by at least \(\|\nabla f\|^2/(2\beta)\), and a nonnegative \(\beta\)-smooth \(f\) is **self-bounded**, \(\|\nabla f(\mathbf w)\|^2 \le 2\beta f(\mathbf w)\). Through a message \(\mathbf x\), a \(\beta\)-smooth scalar loss becomes \(\beta\|\mathbf x\|^2\)-smooth: the square loss is \(2\|\mathbf x\|^2\)-smooth and the logistic loss \(\|\mathbf x\|^2/4\)-smooth. Softplus, \(\log(1 + \exp(x))\), is \(\frac14\)-smooth.
- **Convex learning problem.** A learning problem \((\mathcal H, Z, \ell)\) has a hypothesis class \(\mathcal H\), here a set of weight vectors, a set of examples \(Z\), and a loss \(\ell(\mathbf w, z) \ge 0\). It is convex when \(\mathcal H\) is a convex set and every \(\ell(\cdot, z)\), the loss as a function of \(\mathbf w\) alone, is convex. Then ERM, with the empirical risk \(L_S(\mathbf w)\) the average loss of \(\mathbf w\) over the training sequence \(S\), returns \(\operatorname{argmin}_{\mathbf w \in \mathcal H} L_S(\mathbf w)\), the weights in the class with the lowest empirical risk, is a convex optimization problem.
- **Certificate.** For convex differentiable \(f\) and every \(\mathbf u\) with \(\|\mathbf u\| \le B\), \(f(\mathbf w) - f(\mathbf u) \le \|\nabla f(\mathbf w)\|(\|\mathbf w\| + B)\): a short gradient proves you are close to the best in the ball.

## Is convexity enough?

Lesson 12a showed that convex problems can be *optimized*. Can they be *learned*? That is a different question: it asks whether the hypothesis ERM returns on \(m\) examples has small true risk, with a number of examples that depends only on \(\varepsilon\) and \(\delta\). As in lesson 3, \(\varepsilon\) is the extra true risk you tolerate above the best in the class, and \(\delta\) the chance you allow of an unlucky training sequence; the true risk \(L_{\mathcal D}(\mathbf w)\) is the average loss on a fresh example drawn from the distribution \(\mathcal D\). A problem is **agnostic PAC learnable**, in lesson 3’s words, when some learner meets that \((\varepsilon, \delta)\) promise for every distribution, with no assumption that a perfect hypothesis exists.

You have good reasons to hope so. Lessons 6 and 9 showed that halfspaces in \(d\) dimensions are learnable (perhaps not efficiently). Lesson 4’s discretization trick suggested that any class described by \(d\) numbers is learnable with a number of examples that grows with \(d\). So surely, for a fixed small \(d\), every convex learning problem over \(\mathbb R^d\) is learnable?

No. Not even for \(d = 1\). This does not contradict lesson 6, which is about binary classification with the 0-1 loss, while here losses are real-valued and can be huge. Nor does it contradict the discretization trick, which assumed a bounded loss and that a finite number of bits per parameter is enough. Here is the counterexample, and it is worth going through slowly, because every step shows a different way that “convex” fails to imply “learnable”.

**Claim (linear regression in one dimension is not learnable).** Let \(\mathcal H = \mathbb R\), examples \(z = (x, y) \in \mathbb R^2\), and the square loss \(\ell(w, (x, y)) = (wx - y)^2\) (homogeneous: no bias term). This problem is convex, and it is not agnostic PAC learnable by any deterministic algorithm.

A **deterministic** algorithm is one whose output is fully determined by its training sequence: the same \(S\) always gives the same \(\hat w\). (A slightly longer argument handles learners that flip coins too.)

*Proof.* Let \(A\) be any deterministic algorithm and suppose, for contradiction, that it is an agnostic PAC learner for this problem. Then there is a function \(m(\cdot, \cdot)\) such that for every distribution \(\mathcal D\) and every \(\varepsilon, \delta\), if \(A\) receives \(m \ge m(\varepsilon, \delta)\) examples drawn i.i.d. from \(\mathcal D\) (independently, each from the same \(\mathcal D\)), then with probability at least \(1 - \delta\) it outputs \(\hat w = A(S)\) with

\[ L_{\mathcal D}(\hat w) - \min_w L_{\mathcal D}(w) \le \varepsilon. \]

Here \(\hat w\) (“w hat”) is the learner’s output, \(L_{\mathcal D}(\hat w)\) its true risk, and \(\min_w L_{\mathcal D}(w)\) the smallest true risk any weight achieves, the best in the class. In plain words: **the learner’s true risk is within \(\varepsilon\) of the best possible.** If the best weight had risk 1 and the learner’s had 1.004, the gap 0.004 would pass the \(\varepsilon = 1/100\) we are about to choose.

Choose \(\varepsilon = 1/100\) and \(\delta = 1/2\), take any \(m \ge m(\varepsilon, \delta)\), and set

\[ \mu = \frac{\log(100/99)}{2m}. \]

The numerator \(\log(100/99) \approx 0.01005\) is a small fixed number; dividing by \(2m\) makes \(\mu\) shrink as the training sequence grows. In plain words: **\(\mu\) is a probability chosen so small that \(m\) draws will almost never see an event of that probability.** With \(m = 100\), \(\mu \approx 5.03 \times 10^{-5}\), the first number the snippet after the proof prints, from the line `mu = math.log(100 / 99) / (2 * m)`.

*Step 1: two worlds.* Define two distributions on examples. The first, \(\mathcal D_1\), puts probability \(\mu\) on the example \(z_1 = (1, 0)\) and probability \(1 - \mu\) on \(z_2 = (\mu, -1)\). The second, \(\mathcal D_2\), puts all its probability on \(z_2\). Think of \(z_2\) as the everyday example both worlds share, and \(z_1\) as a rare example that only the first world contains.

*Step 2: both worlds usually show the same training sequence.* Let \(E\) be the event that all \(m\) training examples are \(z_2\). Under \(\mathcal D_2\), \(E\) has probability 1. Under \(\mathcal D_1\), the examples are independent and each is \(z_2\) with probability \(1 - \mu\), so its probability is \(\mathbb P[E] = (1 - \mu)^m\). We bound this using the inequality \(1 - x \ge e^{-2x}\), valid for \(x \in [0, 1/2]\). (Proof: the function \(q(x) = \log(1 - x) + 2x\) has \(q(0) = 0\) and \(q(1/2) = 1 - \log 2 > 0\), and it is concave, the upside-down of convex, because \(q''(x) = -1/(1 - x)^2 < 0\); a concave function lies above its chord, which here is a segment from 0 up to a positive value, so \(q \ge 0\) on \([0, 1/2]\).) Here \(\mu \le \log(100/99)/2 < 0.006\), so

\[ (1 - \mu)^m \ge e^{-2\mu m} = e^{-\log(100/99)} = \frac{99}{100}. \]

The left side multiplies the chance \(1 - \mu\) of a single everyday draw by itself \(m\) times, once per independent example. The first step applies \(1 - x \ge e^{-2x}\) to each factor; the second substitutes \(2\mu m = \log(100/99)\); the third uses that \(e\) to the power \(-\log t\) is \(1/t\). In plain words: **all \(m\) draws are the everyday example with probability at least 99%.** With \(m = 100\) the snippet’s `(1 - mu) ** m` gives 0.995, above 0.99.

So under either distribution, with probability at least 0.99 the learner sees the sequence \(S_2 = (z_2, \dots, z_2)\). Because \(A\) is deterministic, on \(S_2\) it outputs one fixed number; call it \(\hat w\). The learner cannot tell which world it is in.

*Step 3: the adversary picks the world after seeing \(\hat w\).* There are two cases.

If \(\hat w < -1/(2\mu)\), pick \(\mathcal D_1\). The rare example punishes a large negative weight. Keep only its term of the risk, since the other is a square and cannot be negative: \(L_{\mathcal D_1}(\hat w) \ge \mu (\hat w \cdot 1 - 0)^2 = \mu \hat w^2 > \mu \cdot \frac{1}{4\mu^2} = \frac{1}{4\mu}\). Meanwhile the best in the class is no worse than \(w = 0\): \(\min_w L_{\mathcal D_1}(w) \le L_{\mathcal D_1}(0) = \mu \cdot 0 + (1 - \mu)(0 - (-1))^2 = 1 - \mu\). So

\[ L_{\mathcal D_1}(\hat w) - \min_w L_{\mathcal D_1}(w) \ge \frac{1}{4\mu} - (1 - \mu) > \varepsilon, \]

since \(\mu < 0.006\) makes \(1/(4\mu) > 41\). (The display only chains the two bounds just found: the learner’s risk minus the best is at least \(1/(4\mu)\) minus \(1 - \mu\).)

If \(\hat w \ge -1/(2\mu)\), pick \(\mathcal D_2\). Then \(\mu \hat w \ge -1/2\), so \(L_{\mathcal D_2}(\hat w) = (\mu \hat w + 1)^2 \ge (1/2)^2 = 1/4\), while \(w = -1/\mu\) gets loss 0, so \(\min_w L_{\mathcal D_2}(w) = 0\). The excess is at least \(1/4 > \varepsilon\).

*Step 4: conclude.* In the chosen world, the event \(E\) has probability at least 0.99, and on \(E\) the learner outputs \(\hat w\), whose excess risk exceeds \(\varepsilon\). So the learner fails with probability at least \(0.99 > \delta = 1/2\), contradicting the assumption. Since \(A\) was arbitrary, no deterministic algorithm learns the problem. \(\square\)

[[diagram:two-worlds]]

Let’s put numbers on it with \(m = 100\). The snippet computes \(\mu\), the chance of the all-\(z_2\) sequence under \(\mathcal D_1\), the best weight in the first world, the two excess risks at the cut \(\hat w = -1/(2\mu)\), and then, over a fine grid of every \(\hat w\) the learner might output, the *smaller* of the two penalties the adversary can impose.

```python
import math

m = 100
mu = math.log(100 / 99) / (2 * m)
print(f"{mu:.3g}", round((1 - mu) ** m, 4))
L1 = lambda w: mu * w ** 2 + (1 - mu) * (mu * w + 1) ** 2
L2 = lambda w: (mu * w + 1) ** 2
w1 = -(1 - mu) / (1 + mu - mu * mu)
print(round(w1, 4), round(L1(w1), 4))
cut = -1 / (2 * mu)
print(round(L1(cut) - L1(w1), 1), L2(cut))
grid = [-k / (10000 * mu) for k in range(-5000, 25001)]
print(round(min(max(L1(w) - L1(w1), L2(w)) for w in grid), 2))
```

It prints `5.03e-05 0.995`, then `-0.9999 0.9999`, then `4974.2 0.25`, then `0.99`. In the first world the best weight is about \(-1\), with risk about 1 (the formula `w1` sets the derivative of the quadratic \(L_{\mathcal D_1}\) to zero). At the cut, the first world charges an excess of 4,974 and the second 0.25, as the proof said. And the last line is sobering: whatever \(\hat w\) the learner picks, one of the two worlds charges it an excess of at least 0.99, a hundred times the \(\varepsilon\) we asked for.

What went wrong? The hypothesis class is unbounded, so the learner can be asked to choose between \(w \approx -1\) and \(w \approx -1/\mu\), two answers that are astronomically far apart, on the evidence of a sequence that cannot tell them apart. A natural fix is to **bound** the class: fix a scalar \(B\) in advance and require \(\|\mathbf w\| \le B\) for every hypothesis.

**Boundedness and convexity together are still not enough.** Keep the square loss, but now let \(\mathcal H = \{w : |w| \le 1\} \subset \mathbb R\), which is convex and bounded. Run the same argument with the two examples \(z_1 = (1/\mu, 0)\) and \(z_2 = (1, -1)\), again with \(\mathcal D_1\) putting mass \(\mu\) on \(z_1\) and \(\mathcal D_2\) all its mass on \(z_2\). Step 2 is unchanged. In step 3: if the learner returns \(\hat w < -1/2\), pick \(\mathcal D_1\) and get

\[ L_{\mathcal D_1}(\hat w) - \min_w L_{\mathcal D_1}(w) \ge \mu (\hat w / \mu)^2 - L_{\mathcal D_1}(0) \ge \frac{1}{4\mu} - (1 - \mu) > \varepsilon \]

(the same chain as before: the rare example’s loss is now \((\hat w \cdot \frac1\mu - 0)^2\), so its term is \(\mu (\hat w/\mu)^2 = \hat w^2/\mu > \frac{1}{4\mu}\) because \(\hat w^2 > \frac14\), and \(L_{\mathcal D_1}(0) = 1 - \mu\) as before);

if \(\hat w \ge -1/2\), pick \(\mathcal D_2\) and get \(L_{\mathcal D_2}(\hat w) - \min_w L_{\mathcal D_2}(w) \ge (-1/2 + 1)^2 - 0 > \varepsilon\), since \(w = -1\) is in the class and has loss 0.

The snippet replays this bounded version for three values of \(\mu\): for every \(\hat w\) in \([-1, 1]\), it computes the larger of the two excess risks, and prints the smallest such value the learner can hope for.

```python
for mu in (0.1, 0.01, 0.001):
    L1 = lambda w: mu * (w / mu) ** 2 + (1 - mu) * (w + 1) ** 2
    L2 = lambda w: (w + 1) ** 2
    grid = [-1 + k / 1000 for k in range(2001)]
    best1 = min(L1(w) for w in grid)
    worst = min(max(L1(w) - best1, L2(w)) for w in grid)
    print(mu, round(worst, 3))
```

It prints `0.1 0.497`, `0.01 0.812` and `0.001 0.939`. As \(\mu\) shrinks (which is what more examples force the adversary to do), the guaranteed penalty grows towards 1.

Now play the adversary’s game yourself. In the lab you are the learner: you see \(m\) copies of \(z_2\), you choose \(\hat w\), and the adversary picks the world that hurts you most. A toggle switches between the unbounded class and the bounded one. Before you touch anything, a prediction to write down: at \(m = 100\), the smallest excess the adversary can force is about 0.99. Give the learner ten times more labels, \(m = 1{,}000\). Does that number fall below 0.1, fall to about 0.5, or stay about the same?

[[lab]]

More labels did not help. The smallest excess you can hold the adversary to is 0.956 at \(m = 10\), 0.986 at \(m = 100\) (the snippet’s 0.99, rounded), 0.996 at \(m = 1{,}000\) and 0.999 at \(m = 10{,}000\): it creeps *up*, because a larger \(m\) lets the adversary pick a rarer \(z_1\). In every case, the chance that all \(m\) examples are \(z_2\) stays at 0.995. The bounded toggle gives the same four numbers, because the adversary simply moves the damage into the size of \(z_1\): the readout of the loss’s steepness on the class, \(2/\mu^2\), climbs from \(7.92 \times 10^6\) at \(m = 10\) to \(7.92 \times 10^{10}\) at \(m = 1{,}000\).

Look at where the damage comes from. The class is bounded, but the example \(z_1 = (1/\mu, 0)\) is enormous, and the loss on it, \((w/\mu)^2\), has slope \(2w/\mu^2\), as large as \(2/\mu^2\) on the class. A single rare example can swing the risk by a huge amount over a tiny change in \(w\). The cure is to forbid exactly that: require the loss to be Lipschitz, or smooth, with a constant fixed in advance. (A reminder from lesson 12a: \(\rho\)-Lipschitz means the loss never moves more than \(\rho\) times as far as \(\mathbf w\); \(\beta\)-smooth means its gradient is \(\beta\)-Lipschitz.)

## Convex-Lipschitz-bounded and convex-smooth-bounded problems

Here are the two kinds of convex learning problems this course will learn.

**Definition (convex-Lipschitz-bounded problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex-Lipschitz-bounded** with parameters \(\rho, B\) if

- the hypothesis class \(\mathcal H\) is a convex set and \(\|\mathbf w\| \le B\) for all \(\mathbf w \in \mathcal H\);
- for all \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex and \(\rho\)-Lipschitz function.

Example: let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \rho\}\), \(\mathcal Y = \mathbb R\), \(\mathcal H = \{\mathbf w \in \mathbb R^d : \|\mathbf w\| \le B\}\), and the absolute loss \(\ell(\mathbf w, (\mathbf x, y)) = |\langle \mathbf w, \mathbf x \rangle - y|\). The class is a ball, so it is convex and bounded by \(B\). The loss is \(|\cdot|\), convex and 1-Lipschitz, composed with the linear function \(\mathbf w \mapsto \langle \mathbf w, \mathbf x \rangle - y\), so it is convex and \(\|\mathbf x\|\)-Lipschitz, hence \(\rho\)-Lipschitz. This is regression with the absolute loss, messages in a ball of radius \(\rho\), and homogeneous linear hypotheses of norm at most \(B\).

Our own example matters more. **Halfspaces with the hinge loss over the ball \(\|\mathbf w\| \le B\), on messages with \(\|\mathbf x\| \le R\), form a convex-Lipschitz-bounded problem with parameters \(R, B\):** the ball is convex and bounded, and lesson 12a showed the hinge loss is convex and \(\|\mathbf x\|\)-Lipschitz, hence \(R\)-Lipschitz.

**Definition (convex-smooth-bounded problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex-smooth-bounded** with parameters \(\beta, B\) if

- the hypothesis class \(\mathcal H\) is a convex set and \(\|\mathbf w\| \le B\) for all \(\mathbf w \in \mathcal H\);
- for all \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex, nonnegative, and \(\beta\)-smooth function.

The nonnegativity is there to make the loss self-bounded, \(\|\nabla \ell\|^2 \le 2\beta \ell\), which is what replaces the Lipschitz condition in the proofs to come.

Example: let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \sqrt{\beta/2}\}\), \(\mathcal Y = \mathbb R\), \(\mathcal H = \{\mathbf w : \|\mathbf w\| \le B\}\), and the square loss \(\ell(\mathbf w, (\mathbf x, y)) = (\langle \mathbf w, \mathbf x \rangle - y)^2\). The loss is convex and nonnegative, and \((2\|\mathbf x\|^2)\)-smooth by lesson 12a’s claim on smoothness through a linear function, and \(2\|\mathbf x\|^2 \le 2 \cdot \beta/2 = \beta\). So this regression problem is convex-smooth-bounded with parameters \(\beta, B\).

Go back to the counterexamples. The first had no bound \(B\). The second had \(B = 1\), but on \(z_1 = (1/\mu, 0)\) both the slope and the smoothness constant of the loss reached \(2/\mu^2\), with no limit fixed in advance.

**The promise, proved in the next two lessons.** These two kinds of problems are learnable: convexity, boundedness, and Lipschitzness or smoothness of the loss together are sufficient. Lessons 13 and 14 prove it by giving learners that succeed. To have a number to work with, here is the result lesson 14 proves for the convex-Lipschitz-bounded case, stated now and used in the lab. There is a simple learner (stochastic gradient descent, which looks at each example once) whose output \(\bar{\mathbf w}\) on \(m\) i.i.d. examples satisfies

\[ \mathbb E_S \left[ L_{\mathcal D}(\bar{\mathbf w}) \right] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \frac{\rho B}{\sqrt m}. \]

- \(\bar{\mathbf w}\) (“w bar”) is the learner’s output; \(L_{\mathcal D}(\bar{\mathbf w})\) its true risk.
- \(\mathbb E_S[\cdot]\) is the expectation over the training sequence: the average of the bracket over all the training sequences \(S\) you might have drawn, each weighted by its probability.
- \(\min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w)\) is the best true risk in the class, and \(\rho B / \sqrt m\) is the extra charge, which grows with the steepness \(\rho\) and the radius \(B\) and shrinks like one over the square root of the number of examples.

In plain words: **on average over training sequences, the learner is within \(\rho B/\sqrt m\) of the best in the class.** With the ten messages’ \(\rho = R = \sqrt{21}\), \(B = 1.5\) and \(m = 3{,}000\) labels, the extra charge is \(4.58 \cdot 1.5 / 54.8 \approx 0.1255\). Equivalently, solving \(\rho B / \sqrt m \le \varepsilon\) for \(m\), \(m \ge \rho^2 B^2 / \varepsilon^2\) examples bring the expected excess risk down to \(\varepsilon\); for \(\varepsilon = 0.05\) that is 18,900:

```python
import math

rho, B, m = math.sqrt(21), 1.5, 3000
gap = rho * B / math.sqrt(m)
print(round(gap, 4))
eps = 0.05
print(math.ceil(rho ** 2 * B ** 2 / eps ** 2))
```

It prints `0.1255` and `18900`. Note two things. The bound does not mention \(d\) at all; the number of words can be as large as you like. And the guarantee is about the expectation over training sequences, not “with probability at least \(1 - \delta\)”; lesson 13 has tools for the second form.

## Surrogate loss functions

Now back to the bug. The natural loss for the filter is the 0-1 loss, and it is not convex in \(\mathbf w\), as lesson 12a showed by a chord and by the terraces of its lab. Minimizing it on non-separable data is NP-hard. The popular way around the hardness is to **upper bound** the nonconvex loss by a convex one. A **convex surrogate loss** for a loss \(\ell\) is a loss that

1. is convex, and
2. upper bounds the original loss: it is at least \(\ell\) on every hypothesis and every example.

For halfspaces, the hinge loss is a convex surrogate for the 0-1 loss. Lesson 12a showed it is convex. The upper bound is a two-case check on the signed score \(z = y\langle \mathbf w, \mathbf x \rangle\). If \(z \le 0\), the 0-1 loss is 1 and the hinge loss is \(\max\{0, 1 - z\} \ge 1 - z \ge 1\). If \(z > 0\), the 0-1 loss is 0 and the hinge loss is a maximum with 0, so it is at least 0. Either way, \(\ell^{0-1}(\mathbf w, (\mathbf x, y)) \le \ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))\).

[[diagram:hinge]]

On our ten messages, with lesson 9’s halfspace \(\mathbf w^\circ = (-1, 2, -2)\), here are the signed scores, the 0-1 losses and the hinge losses, plus a grid check of the upper bound.

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w = (-1, 2, -2)
z = [y * dot(w, x) for x, y in zip(X, Y)]
zero_one = [int(t <= 0) for t in z]
hinge = [max(0, 1 - t) for t in z]
print(z)
print(zero_one, sum(zero_one) / 10)
print(hinge, sum(hinge) / 10)
grid = [k / 100 for k in range(-500, 501)]
print(all(max(0, 1 - t) >= (t <= 0) for t in grid))
```

It prints

```
[5, 1, 3, 1, 3, 3, 3, 1, -1, -3]
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1] 0.2
[0, 0, 0, 0, 0, 0, 0, 0, 2, 4] 0.6
True
```

The eight undisputed messages score at least 1 and cost nothing under either loss; B′ and E′ cost 1 each under the 0-1 loss and 2 and 4 under the hinge. So the hinge risk is 0.6 where the 0-1 risk is 0.2: always above, and harsher on confident mistakes.

Does the logistic loss of lesson 9 also qualify? It is convex, but at signed score 0 it equals \(\log 2 \approx 0.693 < 1\), so it is not an upper bound on the 0-1 loss. Rescale it, though, by dividing by \(\log 2\), and it becomes \(\log_2(1 + \exp(-z))\), which equals 1 at \(z = 0\), is decreasing, and is positive everywhere, so it is at least the 0-1 loss for every \(z\). We will call this the **base-2 logistic loss**. It is convex, \((1/\log 2)\)-Lipschitz in the signed score, and \((1/(4\log 2))\)-smooth. The square loss in the form \((1 - z)^2\) is another convex upper bound (for \(z \le 0\) it is at least 1). You will meet all three in the lab.

**What the switch guarantees.** Learn the hinge problem with some learner \(A\). The generalization requirement for a hinge-loss learner has the form

\[ L^{\text{hinge}}_{\mathcal D}(A(S)) \le \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) + \varepsilon, \]

where \(L^{\text{hinge}}_{\mathcal D}(\mathbf w) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}[\ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))]\) is the hinge true risk: \((\mathbf x, y) \sim \mathcal D\) reads “a fresh example drawn from \(\mathcal D\)”, and \(\mathbb E\) averages its hinge loss over those draws. If \(\mathcal D\) simply picked one of the ten messages uniformly at random, this expectation would be the plain average of the ten hinge losses, 0.6 for \(\mathbf w^\circ\), the number the previous snippet printed. The requirement, in plain words: **the learner’s hinge true risk is within \(\varepsilon\) of the best hinge true risk in the class.** Because the hinge loss is at least the 0-1 loss on every example, its expectation is at least the 0-1 expectation for every \(\mathbf w\) (expectation preserves pointwise inequalities). So the left side is at least \(L^{0-1}_{\mathcal D}(A(S))\), and

\[ L^{0-1}_{\mathcal D}(A(S)) \le \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) + \varepsilon. \]

Only the left side changed, from hinge to 0-1 true risk. That is the deal: a guarantee on mistakes, paid for in hinge loss. To see what it costs, add and subtract the best 0-1 risk in the class:

\[ L^{0-1}_{\mathcal D}(A(S)) \le \underbrace{\min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)}_{\text{approximation error}} + \underbrace{\left( \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) - \min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w) \right)}_{\text{optimization error}} + \underbrace{\varepsilon}_{\text{estimation error}}. \]

Each underbrace just labels the term above it, and the first two terms add back up to the best hinge risk, so only the bookkeeping changed. In plain words: **mistakes are at most the best possible mistakes, plus the price of the surrogate, plus the price of finite data.**

- **Approximation error**, \(\min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)\): how well the class can do at all on this distribution. You met it in lesson 5.
- **Estimation error**, \(\varepsilon\): the price of seeing only a training sequence and not \(\mathcal D\). Also lesson 5.
- **Optimization error**, \(\min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) - \min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)\): the gap between the best achievable surrogate risk and the best achievable 0-1 risk. It is the price of our inability to minimize the 0-1 loss directly. Its size depends on the distribution and on which surrogate you use.

[[diagram:decomposition]]

Let’s see the three terms in a concrete world, the one the lab uses. Messages arrive as the eight undisputed types of the ten messages, A to H, each with probability \(1/8\). Each carries its usual label with probability 0.95, and the opposite label with probability 0.05 (your users misclick). Every message has \(\|\mathbf x\| \le R = \sqrt{21}\). No hypothesis can do better than 0.05 here, and \(\mathbf w^\circ\) achieves it, so the approximation error is 0.05. Take the class \(\|\mathbf w\| \le B\) with \(B = 1.5\). A numerical minimization of the hinge risk over that ball (the build-it challenge asks you to reproduce it) lands on \(\mathbf w = (-0.2, 1.1, -1.0)\). Let’s evaluate it, then plug in lesson 14’s rate with \(\rho = R\) and \(m = 3{,}000\) labels.

```python
import math

T = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
noise, R, B, m = 0.05, math.sqrt(21), 1.5, 3000
dot = lambda u, v: sum(a * b for a, b in zip(u, v))

def risk(w, loss):
    z = [y * dot(w, (1,) + x) for x, y in T]
    return sum((1 - noise) * loss(t) + noise * loss(-t)
               for t in z) / len(T)

w = (-0.2, 1.1, -1.0)
L01 = risk(w, lambda t: float(t <= 0))
Lh = risk(w, lambda t: max(0.0, 1 - t))
print(round(math.sqrt(dot(w, w)), 3), round(L01, 3), round(Lh, 4))
est = R * B / math.sqrt(m)
print(round(est, 4), round(Lh + est, 3))
print(math.ceil((R * B / (0.36 - Lh)) ** 2))
```

It prints `1.5 0.05 0.2238`, then `0.1255 0.349`, then `2546`. The vector sits on the edge of the ball, its 0-1 risk is the best possible 0.05, and its hinge risk is 0.2238. So the optimization error is \(0.2238 - 0.05 = 0.1738\). With 3,000 labels the estimation term is \(RB/\sqrt m = 0.1255\), and the certified bound on the expected 0-1 risk of the learned filter is \(0.05 + 0.1738 + 0.1255 = 0.349\), the three underbraced terms in order. To certify 0.36 with \(B = 1.5\) you need at least 2,546 labels.

Is 0.349 a disappointing number, when the best halfspace makes only 5% mistakes? It is a *bound*, and a loose one, just like lesson 9’s mistake bound that promised 142 updates when the Perceptron used 3. The actual filter will usually do far better, and the lab shows you that. What the bound buys is a promise you can write down before you train, with no NP-hard search anywhere in sight.

## Surrogate ERM as code

This lesson introduces one algorithmic idea: replace the loss by a convex surrogate, and run ERM on the surrogate, which is now a convex optimization problem. Lesson 14 gives the general-purpose method for that. Here we build the simplest honest version, for a surrogate that is also smooth, using only what lesson 12a proved: the descent step that smoothness guarantees, and a stopping rule that convexity certifies.

We need two small facts first. Lesson 12a proved both for its fix; here they are again, with their proofs, in the form the algorithm uses.

**Fact 1 (smoothness of an average).** If \(f_i\) is \(\beta_i\)-smooth for \(i = 1, \dots, m\), then \(f = \frac1m \sum_i f_i\) is \(\bar\beta\)-smooth with \(\bar\beta = \frac1m \sum_i \beta_i\). *Proof.* \(\nabla f = \frac1m \sum_i \nabla f_i\), so by the triangle inequality \(\|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \frac1m \sum_i \|\nabla f_i(\mathbf v) - \nabla f_i(\mathbf w)\| \le \frac1m \sum_i \beta_i \|\mathbf v - \mathbf w\|\). \(\square\)

For the base-2 logistic loss, the scalar function \(a \mapsto \log_2(1 + e^{-a})\) has second derivative \(\frac{1}{\log 2} \cdot \frac{1}{(1 + e^{-a})(1 + e^{a})} \le \frac{1}{4 \log 2}\), so by the claim on smoothness through a linear function each example’s loss is \(\|\mathbf x_i\|^2/(4\log 2)\)-smooth, and the empirical risk is \(\beta\)-smooth with \(\beta = \frac{1}{4\log 2} \cdot \frac1m \sum_i \|\mathbf x_i\|^2\). In plain words: **the average of the losses is as smooth as the average of their constants.** On the ten messages the squared lengths average 7.4, so \(\beta = 7.4/(4 \log 2) \approx 2.669\), the `beta` line of the run below.

**Fact 2 (a certificate of near-optimality).** If \(f\) is convex and differentiable, then for every \(\mathbf w\) and every \(\mathbf u\) with \(\|\mathbf u\| \le B\),

\[ f(\mathbf w) - f(\mathbf u) \le \|\nabla f(\mathbf w)\| \, (\|\mathbf w\| + B). \]

The left side is how much worse you are than a competitor \(\mathbf u\) in the ball; the right side multiplies the length of the gradient by the farthest you could be from any such competitor, \(\|\mathbf w\| + B\). In plain words: **a short gradient proves you are close to the best in the ball.** In the run below, after 100 steps the gradient has length 0.0059 and the weights length about 1.84, so with \(B = 3\) the certificate is \(0.0059 \times 4.84 \approx 0.0286\); the line `cert = gn * (math.sqrt(dot(w, w)) + 3)` is this right side.

*Proof.* The tangent inequality at \(\mathbf w\) gives \(f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\), so \(f(\mathbf w) - f(\mathbf u) \le \langle \nabla f(\mathbf w), \mathbf w - \mathbf u \rangle \le \|\nabla f(\mathbf w)\| \, \|\mathbf w - \mathbf u\|\) by Cauchy–Schwarz, and \(\|\mathbf w - \mathbf u\| \le \|\mathbf w\| + \|\mathbf u\| \le \|\mathbf w\| + B\). \(\square\)

Taking the minimum over the ball, \(f(\mathbf w) - \min_{\|\mathbf u\| \le B} f(\mathbf u)\) is at most the same quantity. So a small gradient is not just a hint that you are near the bottom; it is a *proof*, and the proof needs convexity. On the 0-1 terraces the gradient is zero everywhere and proves nothing.

```
SmoothSurrogateERM(S, β, B, tol):
  w ← 0 in R^d
  repeat:
    g ← ∇L_S(w)                        // O(md)
    if ‖g‖ · (‖w‖ + B) ≤ tol: return w  // certified
    w ← w − g / β

ZeroOneERM_BruteForce(S):
  for each labelling a halfspace induces on S:  // Σ_{i≤d} C(m,i)
    count its mistakes                          // O(md)
  return a halfspace with the fewest
```

**Cost.** Each step of SmoothSurrogateERM computes one gradient, a sum over the \(m\) examples of a \(d\)-dimensional vector: \(O(md)\) time. It keeps \(\mathbf w\) and \(\mathbf g\): \(O(d)\) space beyond the training sequence. With \(T\) steps, \(O(Tmd)\) time. Each step lowers \(L_S\) by at least \(\|\mathbf g\|^2/(2\beta)\), by lesson 12a’s smoothness argument, and the stopping rule is correct by Fact 2. How many steps it takes to reach a given tolerance is lesson 14’s business; this lesson only guarantees that every step helps and that the answer, when it comes, is certified. Compare the brute-force 0-1 learner: up to \(\sum_{i \le d} \binom{m}{i}\) labellings at \(O(md)\) each, which is the \(5 \times 10^{29}\) years from lesson 12a’s opening.

Run it on the ten messages with the base-2 logistic loss and \(B = 3\), checking at every step that the guaranteed decrease really happened.

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y, m = [(1,) + x for x, _ in S], [y for _, y in S], len(S)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
f = lambda w: sum(math.log2(1 + math.exp(-y * dot(w, x)))
                  for x, y in zip(X, Y)) / m
grad = lambda w: [sum(-y * x[j] / (1 + math.exp(y * dot(w, x)))
                      for x, y in zip(X, Y)) / (m * math.log(2))
                  for j in range(3)]
beta = sum(dot(x, x) for x in X) / m / (4 * math.log(2))
w, ok = [0.0, 0.0, 0.0], True
for t in range(1, 301):
    g = grad(w)
    v = [a - b / beta for a, b in zip(w, g)]
    ok = ok and f(w) - f(v) >= dot(g, g) / (2 * beta) - 1e-12
    w = v
    if t in (1, 10, 100, 300):
        gn = math.sqrt(dot(grad(w), grad(w)))
        cert = gn * (math.sqrt(dot(w, w)) + 3)
        print(t, round(f(w), 4), round(gn, 4), round(cert, 4))
print(round(beta, 3), ok, [round(c, 2) for c in w])
```

It prints

```
1 0.9352 0.3397 1.0773
10 0.756 0.1381 0.5413
100 0.6966 0.0059 0.0286
300 0.6963 0.0001 0.0004
2.669 True [-0.92, 1.38, -0.95]
```

The columns are the step, the empirical risk, the gradient norm and the certificate. Starting from 1 (the base-2 loss of a zero score), the risk falls to 0.6963, and after 300 steps the certificate proves it is within 0.0004 of the best possible over the ball of radius 3. The smoothness constant is \(\beta = 2.669\), every single step achieved its guaranteed decrease (`True`), and the weights are lesson 9’s logistic weights, \((-0.92, 1.38, -0.95)\). That is no coincidence: dividing a loss by \(\log 2\) does not move its minimizer, and \(0.6963 \times \log 2 \approx 0.483\) is lesson 9’s logistic risk. It is also the walk of lesson 12a’s fix, started at \(\mathbf 0\) instead: the base-2 gradient and \(\beta\) are the natural-log ones divided by \(\log 2\), so every step \(\mathbf g/\beta\) is the same.

**Build it yourself.** Fill in the functions and make every assert pass. The data are the ten messages and the lab world (the eight types with 5% label noise).

```python
import math

S10 = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
       ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
       ((2, 1), -1), ((1, 2), 1)]
TYPES = S10[:8]
NOISE = 0.05

def zero_one(z): ...
def hinge(z): ...
def logistic2(z): ...
def squared(z): ...
def risk(w, S, loss): ...
def world_risk(w, loss): ...
def grad_logistic2(w, S): ...
def smooth_descent(S, steps): ...
def certified_bound(floor, rho, B, m): ...
def labels_needed(floor, rho, B, target): ...

if __name__ == "__main__":
    W0, W15 = (-1, 2, -2), (-0.2, 1.1, -1.0)
    assert risk(W0, S10, zero_one) == 0.2
    assert risk(W0, S10, hinge) == 0.6
    grid = [k / 100 for k in range(-500, 501)]
    for loss in (hinge, logistic2, squared):
        assert all(loss(z) >= zero_one(z) for z in grid)
    assert math.log(1 + math.exp(0)) < zero_one(0)
    w = smooth_descent(S10, 300)
    assert round(risk(w, S10, logistic2), 4) == 0.6963
    assert round(world_risk(W15, zero_one), 3) == 0.05
    floor = world_risk(W15, hinge)
    assert round(floor, 4) == 0.2238
    R = math.sqrt(21)
    assert round(certified_bound(floor, R, 1.5, 3000), 3) == 0.349
    assert labels_needed(floor, R, 1.5, 0.36) == 2546
    rho2 = R / math.log(2)
    assert labels_needed(0.3675, rho2, 4.0, 0.36) is None
```

1. The losses take the signed score \(z\). `risk(w, S, loss)` prepends the constant 1 to each message. `world_risk` averages over the eight types, weighting each type’s usual label by 0.95 and the flipped label by 0.05.
2. `smooth_descent` starts at \(\mathbf 0\) and uses step \(1/\beta\) with \(\beta\) from Fact 1, and returns the final weights. Before returning, print the certificate of Fact 2 for \(B = 3\).
3. `labels_needed` returns the smallest \(m\) with `certified_bound(...) <= target`, or `None` when the floor is already above the target (no label budget can help then).
4. Harder: the hinge loss is not smooth, so `smooth_descent` cannot minimize it. Minimize the hinge risk of the lab world over the ball \(\|\mathbf w\| \le B\) anyway: at each step move against a slope vector (for a term with signed score below 1, use \(-y\mathbf x\); otherwise \(\mathbf 0\)), with step \(B/(R\sqrt t)\) at step \(t\), and pull \(\mathbf w\) back onto the ball by rescaling whenever \(\|\mathbf w\| > B\). Keep the best point seen. You should reproduce 0.2238 at \(B = 1.5\) and about 0.1625 for \(B \ge 2.7\). Lesson 14 explains why this works.

## Fix the bug

Back to the mail room. Operations will not ship a filter without a written guarantee, and the 0-1 search will never finish. So you will ship a hinge-loss filter instead, and certify it with the bound you just derived: the expected 0-1 risk is at most the best surrogate risk over your class plus \(\rho B/\sqrt m\).

The lab’s world is the one from the decomposition: eight message types, 5% of labels flipped, \(R = \sqrt{21}\). Your label budget is 3,000, and operations wants the certified expected 0-1 risk at most 0.36. You choose the surrogate (hinge, base-2 logistic, or the square surrogate \((1 - z)^2\)) and the norm bound \(B\). The lab computes the floor (the best surrogate risk over your ball) exactly, the Lipschitz constant of your surrogate over your ball, and the bound. Before you start, a prediction to write down: with the hinge loss, is a bigger \(B\) always better?

[[lab-act2]]

The trade you just navigated is the bias-complexity tradeoff of lesson 5, reappearing in a new costume. A small ball cannot reach low surrogate risk, so its floor is high. A large ball lowers the floor, but the estimation term \(\rho B/\sqrt m\) grows with \(B\). In this world, with the hinge loss and 3,000 labels, the window, on a grid of steps of 0.1, is \(B\) from 1.4 to 1.9; the best is \(B = 1.5\), with bound 0.349. And no choice of \(B\) rescues the other two surrogates: the base-2 logistic loss cannot get its floor below 0.3675 and the square surrogate below 0.3644, both already over the target before a single label is counted.

## What people get wrong

**“The loss is convex, so the problem is learnable.”** Convexity makes ERM an optimization problem without false bottoms. It says nothing about how many examples you need. One-dimensional linear regression with the square loss is convex and not learnable at all. You also need a bounded class and a Lipschitz or smooth loss.

**“More labels always help.”** Not when the class or the loss is unconstrained. In the two-worlds game the smallest excess you could force the adversary down to rose with \(m\), from 0.956 at 10 labels to 0.999 at 10,000, because more labels let the adversary hide a rarer example.

**“Bounding the weights fixes it.”** Not by itself. The bounded counterexample kept \(|w| \le 1\) and still failed, because one rare, huge example made the loss arbitrarily steep. The constants \(\rho\) or \(\beta\) must be fixed in advance, which in practice means bounding the messages, \(\|\mathbf x\| \le R\), as well.

**“Minimizing the hinge loss minimizes the number of mistakes.”** It minimizes an upper bound on them. The two minimizers can differ, and the optimization error is exactly the price. In the lab world it was 0.17, more than three times the best possible 0-1 risk.

**“Any convex loss is a surrogate for the 0-1 loss.”** It must also sit above it. The natural-log logistic loss is \(\log 2 \approx 0.69\) at a zero score, below the 0-1 loss’s 1; divide by \(\log 2\) to make it a surrogate.

**“A bigger \(B\) is always better, because the class is richer.”** Richer classes lower the floor and raise the estimation term \(\rho B/\sqrt m\). Past the point where the floor stops falling (about \(B = 2.7\) in the lab world), more \(B\) only costs labels.

## Recap and the next bug

1. In a **convex learning problem**, ERM is a convex optimization problem, but convexity is not enough to learn, even in one dimension: whatever a deterministic learner outputs, one of two worlds that usually show it the same training sequence charges an excess of about 0.99, and more labels make it worse. Convexity plus a bounded class is still not enough, because one rare, huge example can make the loss arbitrarily steep.
2. **Convex-Lipschitz-bounded** and **convex-smooth-bounded** problems add a norm bound \(B\) on the class and a Lipschitz or smoothness constant fixed in advance. They are learnable, as lessons 13 and 14 prove; with Lipschitz losses, \(m \ge \rho^2 B^2/\varepsilon^2\) examples suffice in expectation. Halfspaces with the hinge loss on messages of length at most \(R\) are convex-Lipschitz-bounded with parameters \(R, B\).
3. A **convex surrogate** is convex and upper bounds the original loss. The hinge loss is one for the 0-1 loss of halfspaces, and so is the base-2 logistic loss; the natural-log logistic loss is not, since it is below 1 at a zero score. The 0-1 risk of the learned halfspace is at most the approximation error plus the **optimization error** plus the estimation error.
4. For a smooth surrogate, ERM is a walk with step \(1/\beta\) that lowers the empirical risk at every step and stops with a certificate, at \(O(md)\) time per step instead of a search through every labelling.
5. The bound to certify is the best surrogate risk over the ball plus \(\rho B/\sqrt m\). A small ball raises the first term, a large ball the second: in the lab world with the hinge loss and 3,000 labels, \(B\) from 1.4 to 1.9 meets 0.36, and the best is \(B = 1.5\), with bound 0.349.

The hinge filter ships with its certificate. A week later, a user writes in to correct a single label, one message out of 3,000, and someone retrains out of habit. The new weights look nothing like the old ones. Nothing is wrong with the bound, which never promised that the learner’s output would sit still; it promised something about the average over training sequences. But it makes you uneasy. The guarantees in this course have all come from uniform convergence, every hypothesis’s empirical risk staying close to its true risk at once. Is there a different reason a learner could generalize: a rule whose output barely moves when one example changes, and which learns every convex-Lipschitz-bounded problem *because* it is stable? That is the next lesson: regularization and stability.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Consider the **squared hinge loss** \(\ell(\mathbf w, (\mathbf x, y)) = (\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\})^2\), with messages \(\|\mathbf x\| \le R\) and class \(\|\mathbf w\| \le B\). Is it convex in \(\mathbf w\)? Is it a convex surrogate for the 0-1 loss? Is the problem convex-Lipschitz-bounded, and with which \(\rho\)? Is it convex-smooth-bounded, and with which \(\beta\)? Is it Lipschitz over all of \(\mathbb R^d\)?
2. In the proof that one-dimensional linear regression is not learnable, point to the exact sentence where determinism of the learner is used, and explain why the argument needs the probability of the all-\(z_2\) sequence to exceed \(1 - \delta\), not just to be positive. Then take the bounded version (\(|w| \le 1\), \(z_1 = (1/\mu, 0)\)) and say which condition of a convex-Lipschitz-bounded problem it violates, with the offending constant as a function of \(\mu\).
3. In the lab world (\(R = \sqrt{21}\)), the best hinge risk over the ball \(\|\mathbf w\| \le 1.9\) is 0.2003, and the best base-2 logistic risk over \(\|\mathbf w\| \le 4\) is 0.3676. For each, compute the certified bound at \(m = 3{,}000\) and the number of labels needed to certify 0.36. Which choice would you hand to operations?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (logistic regression is both kinds of problem).** Let \(\mathcal H = \mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le B\}\) for some scalar \(B > 0\), \(\mathcal Y = \{\pm 1\}\), and \(\ell(\mathbf w, (\mathbf x, y)) = \log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\). Show that this learning problem is both convex-Lipschitz-bounded and convex-smooth-bounded, and give the Lipschitz and smoothness parameters.
- **Problem 2 (harder: convex-Lipschitz-bounded does not mean efficiently learnable).** Lesson 13 will show that every convex-Lipschitz-bounded problem is learnable from the statistical point of view. Our motivation, though, was computational. Show that convexity is not enough for efficiency, even in one dimension. Let \(\mathcal H = [0, 1]\), let the set of examples \(Z\) be the set of all Turing machines, and define the loss: \(\ell(0, T) = 1\) if \(T\) halts on input 0 and 0 otherwise; \(\ell(1, T) = 0\) if \(T\) halts on input 0 and 1 otherwise; and \(\ell(h, T) = h\,\ell(0, T) + (1 - h)\,\ell(1, T)\) for \(h \in (0, 1)\). (a) Show that the problem is convex-Lipschitz-bounded. (b) Show that no computable algorithm can learn it.
