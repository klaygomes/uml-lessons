# Soft-SVM and Support Vectors

## The job that returned nothing

Lesson 15a ended with a promise kept. Hard-SVM’s line left every labelled message more than one word from the boundary, 1.342 on the eight messages of the quiet week, and the facilities team got its lunch invitations. Then a month of new labels arrived, the nightly job ran hard-SVM on them, and in the morning there was no new filter, only a line in the log: the quadratic program has no feasible point. The old filter kept running on last month’s weights.

Two of the new labels explain it.

| message | gist | “free” \(x_1\) | “meeting” \(x_2\) | label \(y\) |
|---|---|---|---|---|
| I | free consultation: book a meeting, meeting link inside | 1 | 2 | junk, \(+1\) |
| J | free cake, free coffee, free hugs: Ana’s farewell meeting | 3 | 1 | keep, \(-1\) |

Look at I. It has exactly the counts of G, the lunch invitation, but your users called it junk. No hypothesis that sees only these two counts can label both correctly, halfspace or not. The ten messages are not separable, the hard-SVM program has no feasible point at all, and hard-SVM returns nothing.

So here is the bug. Real labels are noisy, people disagree about what is junk, and a learner that returns nothing whenever two users disagree is no learner at all. Worse, operations still wants its written guarantee, and hard-SVM’s dimension-free promise assumed mail that is separable with a margin. This mail is not. We need a version that tolerates some violations, and we need to know whether its guarantee survives. This lesson builds that version, soft-SVM, proves that its guarantee still has no dimension in it, and then looks inside: which messages hold the separator in place, a second optimization problem that finds them, and a learner simple enough to run on millions of messages.

## What you need from lesson 15a

Here is the toolkit, one line each, so you can check that nothing has slipped.

- **The eight messages.** A message is a vector of word counts over two words, \(x_1\) for “free” and \(x_2\) for “meeting”, and the label \(y\) is \(+1\) for junk and \(-1\) for keep. These are the eight messages of the quiet week:

| message | gist | “free” \(x_1\) | “meeting” \(x_2\) | label \(y\) |
|---|---|---|---|---|
| A | FREE trial, FREE setup, FREE support, FREE month, FREE gift: book a meeting, any meeting | 5 | 2 | junk, \(+1\) |
| B | FREE gift, FREE delivery, FREE | 3 | 0 | junk, \(+1\) |
| C | free phone, free case, free charger, free delivery, no meeting needed | 4 | 1 | junk, \(+1\) |
| D | free prize, free cruise, free spa, free! | 4 | 0 | junk, \(+1\) |
| E | see you tomorrow | 0 | 0 | keep, \(-1\) |
| F | meeting moved to Tuesday | 0 | 1 | keep, \(-1\) |
| G | free lunch at the meeting, meeting room 4 | 1 | 2 | keep, \(-1\) |
| H | free pizza after the meeting; meeting notes; next meeting Friday | 1 | 3 | keep, \(-1\) |

- **Halfspaces.** A halfspace \((\mathbf w, b)\) labels a message \(\mathbf x\) with \(\operatorname{sign}(\langle \mathbf w, \mathbf x \rangle + b)\). Here \(\mathbf w\) is the **weight vector**, one weight per word; \(b\) is the **bias term**; \(\langle \mathbf w, \mathbf x \rangle = w_1 x_1 + \dots + w_d x_d\) is the inner product; \(\langle \mathbf w, \mathbf x \rangle + b\) is the **score**; and \(\|\mathbf w\| = \sqrt{w_1^2 + \dots + w_d^2}\) is the length of \(\mathbf w\). The hyperplane is where the score is 0. The Perceptron’s line, \(\mathbf w = (4, -1)\) and \(b = -3\), scores G at \(4 - 2 - 3 = -1\): keep.
- **The signed score** \(y (\langle \mathbf w, \mathbf x \rangle + b)\), the score times the label, is positive exactly when the halfspace gets the message right. A training sequence \(S\) is **separable** when some halfspace gives every message a positive signed score. G’s signed score under the Perceptron’s line is \((-1)(-1) = 1\).
- **Distance and margin.** The distance from \(\mathbf x\) to the hyperplane is \(|\langle \mathbf w, \mathbf x \rangle + b| / \|\mathbf w\|\), and the **margin** of a separating hyperplane is the smallest distance from it to a training message. When \(\|\mathbf w\| = 1\), a **unit** vector, the margin is \(\min_i y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\), the signed score of the worst-placed message. The Perceptron’s line left 0.243 on the eight messages; hard-SVM’s leaves 1.342.
- **Hard-SVM** returns the separating halfspace with the largest margin by solving the **quadratic program** “minimize \(\|\mathbf w\|^2\) over \((\mathbf w, b)\) such that \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1\) for every \(i\)” and rescaling the solution \((\mathbf w_0, b_0)\) to unit length; the largest margin is \(1/\|\mathbf w_0\|\). On the eight messages, \(\mathbf w_0 = (\tfrac23, -\tfrac13)\) and \(b_0 = -1\), the score \((2x_1 - x_2)/3 - 1\); B, E and G sit exactly at signed score 1, and \(1/\|\mathbf w_0\| = 3/\sqrt5 \approx 1.342\).
- **Homogeneous halfspaces** have no bias term, \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\). **Lifting** a message to \((1, x_1, x_2)\) turns the bias term into the first weight. The lifted hard-SVM program, “minimize \(\|\mathbf w\|^2\) such that \(y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1\)”, then penalizes \(b\) too; we say it **regularizes** the bias term. On the eight messages it returns the same separator, \((b, w_1, w_2) = (-1, \tfrac23, -\tfrac13)\), with lifted margin \(3/\sqrt{14} \approx 0.802\), because the lifted length includes the bias term.
- **Separability with a \((\gamma, \rho)\)-margin.** A distribution \(\mathcal D\) over labelled messages is separable with a \((\gamma, \rho)\)-margin when some unit separator keeps every message that can arrive at signed score at least \(\gamma\) (gamma), and every message has length at most \(\rho\) (rho), the **radius** (lesson 9a’s \(R\)). Scaling the data scales \(\gamma\) and \(\rho\) together, so only the ratio \(\rho/\gamma\) means anything. For the eight lifted messages, \(\gamma = 0.8018\) and \(\rho = \sqrt{30} \approx 5.4772\), the length of A’s lifted vector \((1, 5, 2)\).
- **Hard-SVM’s guarantee.** If \(\mathcal D\) is separable with a \((\gamma, \rho)\)-margin by a homogeneous halfspace, then with probability at least \(1 - \delta\) over a training sequence \(S \sim \mathcal D^m\) (\(m\) examples drawn independently from \(\mathcal D\); \(\delta\), delta, is the chance of an unlucky sample), homogeneous hard-SVM’s **0-1 true risk**, the chance that it mislabels a fresh message, is at most \(\sqrt{4(\rho/\gamma)^2/m} + \sqrt{2\log(2/\delta)/m}\), with \(\log\) the natural logarithm. No dimension \(d\) appears. With \(\rho/\gamma = 10\), \(m = 1000\) and \(\delta = 0.05\), that is about 0.718.
- **The VC route.** Homogeneous halfspaces over \(d\) words have VC dimension \(d\), and by the fundamental theorem of lesson 6a, with far fewer than \(d/\varepsilon\) labels no learner can promise true risk at most \(\varepsilon\) (epsilon, the accuracy target) on every distribution.

Here are the lifted eight messages under the lifted hard-SVM solution: the signed scores, the lifted margin and the radius.

```python
import math
from fractions import Fraction as Fr

S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X = [(1,) + x for x, _ in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w0 = (Fr(-1), Fr(2, 3), Fr(-1, 3))
signed = [y * dot(w0, x) for x, (_, y) in zip(X, S)]
print([str(s) for s in signed])
gamma = min(signed) / math.sqrt(dot(w0, w0))
rho = max(math.sqrt(dot(x, x)) for x in X)
ratio2 = (rho / gamma) ** 2
print(round(gamma, 4), round(rho, 4), round(ratio2, 2))
```

It prints `['5/3', '1', '4/3', '5/3', '1', '4/3', '1', '4/3']`, then `0.8018 5.4772 46.67`. Every signed score is at least 1, three of them exactly 1 (B, E and G), and the difficulty ratio \((\rho/\gamma)^2\) is 46.67. Now add I and J, and all of this breaks.

## Mail that is almost separable

The hard program demands \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1\) for every \(i\). A natural relaxation lets each constraint be violated, at a price. Introduce a **slack variable** \(\xi_i \ge 0\) for each example and ask only for \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1 - \xi_i\). The slack \(\xi_i\) measures how much the constraint for example \(i\) is violated. Then minimize the norm of \(\mathbf w\) (which stands for the margin) and the average slack (which stands for the violations) together, with a parameter \(\lambda > 0\) setting the exchange rate between them.

**Definition (soft-SVM).** Given \(S\) and a parameter \(\lambda > 0\), **soft-SVM** solves

\[ \min_{\mathbf w, b, \boldsymbol \xi} \left( \lambda \|\mathbf w\|^2 + \frac{1}{m} \sum_{i=1}^m \xi_i \right) \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1 - \xi_i \text{ and } \xi_i \ge 0 \text{ for all } i, \]

and returns \((\mathbf w, b)\).

From the inside out: \(\xi_i\) (xi) is example \(i\)’s slack, and \(\boldsymbol\xi\) in bold is the list of all \(m\) of them. \(\sum_{i=1}^m \xi_i\) adds them up, and \(\frac1m\) turns the total into an average. \(\lambda\) (lambda) multiplies the squared length of \(\mathbf w\). The subscript under \(\min\) says the solver may choose \(\mathbf w\), \(b\) and every slack, and “subject to” lists the rules it must obey: each signed score is at least 1 minus that example’s slack, and no slack is negative. In plain words: **find the halfspace that best trades a short weight vector against a small average shortfall below signed score 1.** Take the hard-SVM separator of the eight messages, \(\mathbf w = (\tfrac23, -\tfrac13)\), \(b = -1\), on all ten. I gets score \(\tfrac23 - \tfrac23 - 1 = -1\), signed score \(-1\), so its slack must be at least \(1 - (-1) = 2\); J gets signed score \(-\tfrac23\), slack at least \(\tfrac53\); the other eight need none. With \(\lambda = 0.1\) the objective is \(0.1 \cdot \tfrac59 + (2 + \tfrac53)/10 = \tfrac{19}{45}\):

```python
from fractions import Fraction as Fr

S10 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
       ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1),
       ((1, 2), 1), ((3, 1), -1)]
w, b, lam = (Fr(2, 3), Fr(-1, 3)), Fr(-1), Fr(1, 10)
s = [y * (w[0] * x[0] + w[1] * x[1] + b) for x, y in S10]
xi = [max(Fr(0), 1 - si) for si in s]
ok = all(si >= 1 - k and k >= 0 for si, k in zip(s, xi))
obj = lam * (w[0] ** 2 + w[1] ** 2) + sum(xi) / len(S10)
print([str(k) for k in xi], ok)
print(obj, round(float(obj), 4))
```

It prints `['0', '0', '0', '0', '0', '0', '0', '0', '2', '5/3'] True`, then `19/45 0.4222`. Notice the line that picks each slack: the smallest value obeying both rules is \(\max\{0, 1 - \text{signed score}\}\), where \(\max\{\cdot, \cdot\}\) keeps the larger of two numbers. Hold on to that; it is about to get a name.

A large \(\lambda\) makes the norm expensive: the learner prefers a short \(\mathbf w\), a wide margin, and pays for it with more violations. A small \(\lambda\) makes violations expensive: the learner fits the examples harder and accepts a narrower margin.

The slack variables are a modelling device, and they disappear on inspection. Recall the **hinge loss** you met in lesson 12a,

\[ \ell^{\text{hinge}}((\mathbf w, b), (\mathbf x, y)) = \max\{0, \, 1 - y (\langle \mathbf w, \mathbf x \rangle + b)\}, \]

and write \(L_S^{\text{hinge}}((\mathbf w, b))\) for its average over the training sequence. It is zero for an example with signed score at least 1, and grows linearly as the score falls below 1. Inside out: the signed score, its shortfall below 1, and \(\max\) with 0 so that a comfortable message pays nothing. The superscript “hinge” names the loss; the subscript \(S\), as for the empirical risk, says the average runs over the training messages. In plain words: **the hinge loss charges a message for how far its signed score falls short of 1.** I under the hard-SVM separator: \(\max\{0, 1 - (-1)\} = 2\), its slack from the snippet.

**Claim (slack is hinge loss).** Soft-SVM is the same problem as

\[ \min_{\mathbf w, b} \left( \lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}((\mathbf w, b)) \right). \]

*Proof.* Fix \((\mathbf w, b)\) and minimize the soft-SVM objective over the slacks alone. The objective is a sum, and each \(\xi_i\) appears in exactly one term and one pair of constraints, so we can minimize each \(\xi_i\) separately. Write \(s_i = y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). The constraints on \(\xi_i\) are \(\xi_i \ge 0\) and \(\xi_i \ge 1 - s_i\), so \(\xi_i \ge \max\{0, 1 - s_i\}\), and the smallest allowed value is exactly \(\max\{0, 1 - s_i\} = \ell^{\text{hinge}}((\mathbf w, b), (\mathbf x_i, y_i))\). Plugging in the best slacks turns the objective into \(\lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}((\mathbf w, b))\). Minimizing over \((\mathbf w, b, \boldsymbol\xi)\) is the same as first minimizing over \(\boldsymbol\xi\) for each \((\mathbf w, b)\) and then over \((\mathbf w, b)\), so the two problems have the same minimizers in \((\mathbf w, b)\) and the same optimal value. \(\blacksquare\)

The slack snippet above already did this inner minimization: its line `max(Fr(0), 1 - si)` is the hinge loss.

So soft-SVM is **regularized loss minimization** (RLM), the rule of lesson 13a: return \(\operatorname{argmin}_{\mathbf w} \left( L_S(\mathbf w) + R(\mathbf w) \right)\), the empirical risk under some loss plus a regularization function \(R\), here the Tikhonov penalty \(R(\mathbf w) = \lambda \|\mathbf w\|^2\) on the norm. It has a bias, in the sense of lesson 1, toward short weight vectors, which is to say toward wide margins. As before, the homogeneous version drops \(b\) (or lifts the messages) and solves

\[ \min_{\mathbf w} \left( \lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}(\mathbf w) \right), \qquad L_S^{\text{hinge}}(\mathbf w) = \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\}. \]

This is the same reading as before, with \(b\) folded into \(\mathbf w\): **a penalty on the length of \(\mathbf w\) plus the average hinge loss.** The function `soft_svm_objective` below mirrors it term by term. Let’s run it on the ten messages, lifted, with \(\lambda = 0.1\). The minimizer turns out to be \(\mathbf w = (-\tfrac12, \tfrac12, -\tfrac12)\), the score \((x_1 - x_2 - 1)/2\); the solver behind that claim appears in the duality section. Compare its objective with the hard-SVM separator of the eight messages, which ignores I and J.

```python
S10 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
       ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1),
       ((1, 2), 1), ((3, 1), -1)]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
X = [(1,) + x for x, _ in S10]
Y = [y for _, y in S10]

def soft_svm_objective(w, lam):
    hinge = [max(0.0, 1 - y * dot(w, x)) for x, y in zip(X, Y)]
    return lam * dot(w, w) + sum(hinge) / len(X), hinge

for w in [(-0.5, 0.5, -0.5), (-1, 2 / 3, -1 / 3)]:
    value, hinge = soft_svm_objective(w, 0.1)
    print(round(value, 4), [round(h, 2) for h in hinge])
```

It prints `0.475 [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 1.5]` and then `0.5222 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.67]`.

Read the slacks of the soft-SVM solution. A, B, C, D, F, G and H pay nothing. E, lifted \((1, 0, 0)\) with label \(-1\), has signed score \(\tfrac12\) and pays 0.5: it is on the right side but inside the margin. I pays 2 and J pays 1.5: both are on the wrong side. Total hinge \(4/10 = 0.4\), plus \(0.1 \times \|\mathbf w\|^2 = 0.1 \times 0.75 = 0.075\), gives 0.475. The old hard-SVM separator has no slack on the eight original messages, but its longer \(\mathbf w\) (\(\|\mathbf w\|^2 = 14/9\)) costs \(0.156\) and its slacks on I and J cost 0.367, for 0.522. Soft-SVM accepts a little violation on E to buy a shorter vector.

[[diagram:slack]]

Before going further, try it with your hands. In the lab below you have the ten messages and a slider for \(\lambda\). Watch the line, the slack on each message, the two parts of the objective and the lifted margin \(1/\|\mathbf w\|\) move as you slide.

[[lab]]

## What soft-SVM costs in labels

Hard-SVM needed the data to be separable with a margin. Soft-SVM needs no such thing, and it comes with its own guarantee, which we can now prove from tools you already have. Three of them, recalled in a line each:

- A function \(f\) is **\(\rho\)-Lipschitz** if \(|f(\mathbf w) - f(\mathbf u)| \le \rho \|\mathbf w - \mathbf u\|\) for all \(\mathbf w, \mathbf u\) (lesson 12a): moving the input a distance \(t\) moves the output by at most \(\rho t\). For example, \(f(w) = 3w\) on the number line is 3-Lipschitz: a step of 0.1 changes it by 0.3.
- The **Cauchy–Schwarz inequality**, \(|\langle \mathbf a, \mathbf b \rangle| \le \|\mathbf a\| \|\mathbf b\|\) (proved in lesson 9a): an inner product is never bigger than the product of the two lengths. For \((4, -1)\) and G’s \((1, 2)\), \(2 \le \sqrt{17}\sqrt{5} \approx 9.2\).
- The **oracle inequality** for RLM of lesson 13b (the Lipschitz case): if for every example \(z = (\mathbf x, y)\) the loss \(\ell(\cdot, z)\) (the dot is the slot for \(\mathbf w\), with the example held fixed) is convex and \(\rho\)-Lipschitz, then the RLM rule \(A(S) = \operatorname{argmin}_{\mathbf w} \left( L_S(\mathbf w) + \lambda \|\mathbf w\|^2 \right)\) satisfies, for every vector \(\mathbf u\),

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}(A(S)) \right] \le L_{\mathcal D}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}. \]

Inside out: \(A(S)\) is the weight vector the learner returns on the training sequence \(S\), and \(L_{\mathcal D}(A(S))\) is its true risk, its average loss on fresh examples from \(\mathcal D\). The \(\mathbb E\) with \(S \sim \mathcal D^m\) under it averages that true risk over all the training sequences you might have drawn. On the right, \(\mathbf u\) is any competitor at all: its own true risk, plus the penalty it would pay, plus \(2\rho^2/(\lambda m)\), the most the learner can overfit on average. In plain words: **on average, the regularized learner is almost as good as any fixed weight vector, paying that vector’s penalty plus a term that shrinks with more labels.** With \(\rho = 5\), \(\lambda = 0.0354\) and \(m = 10{,}000\), the last term is \(50/354 \approx 0.141\); `soft_svm_bound`, later in this section, computes the whole right side. To use the inequality for soft-SVM we need the hinge loss to be convex and Lipschitz.

Convexity is quick. Recall from lesson 12a that \(f\) is **convex** when **the value at a blend is at most the blend of the values**: for every \(\mathbf w, \mathbf u\) and every mixing weight \(\theta \in [0, 1]\) (lesson 12a called it \(\alpha\); we save \(\alpha\) for later),

\[ f(\theta \mathbf w + (1 - \theta) \mathbf u) \le \theta f(\mathbf w) + (1 - \theta) f(\mathbf u). \]

E’s hinge loss is \(\max\{0, 1 + b\}\), with \(b\) the first, bias, weight. It is 0.5 at the soft-SVM solution, 1 at \(\mathbf 0\), and 0.75 halfway between, exactly the blend \(\tfrac12 \cdot 0.5 + \tfrac12 \cdot 1\): on its sloped part the hinge is a straight line.

Now the argument. For a fixed example \((\mathbf x, y)\), the map \(\mathbf w \mapsto 1 - y \langle \mathbf w, \mathbf x \rangle\) (the arrow \(\mapsto\) reads “sends \(\mathbf w\) to”) is affine, a linear function plus a constant, and the hinge loss is the pointwise maximum of it and the constant 0. A maximum of two convex functions is convex: for \(\theta \in [0, 1]\), each of the two functions at \(\theta \mathbf w + (1-\theta) \mathbf u\) is at most \(\theta\) times its value at \(\mathbf w\) plus \((1-\theta)\) times its value at \(\mathbf u\), and each of those is at most the same combination of the maxima. Here is the inequality tested on random triples, `lhs` and `rhs` being its two sides:

```python
import random

rng = random.Random(0)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
hinge = lambda w, x, y: max(0, 1 - y * dot(w, x))
worst = -1.0
for _ in range(100_000):
    x, w, u = ([rng.uniform(-2, 2) for _ in "abc"] for _ in "xwu")
    y, th = rng.choice((1, -1)), rng.random()
    mix = [th * a + (1 - th) * c for a, c in zip(w, u)]
    lhs = hinge(mix, x, y)
    rhs = th * hinge(w, x, y) + (1 - th) * hinge(u, x, y)
    worst = max(worst, lhs - rhs)
print(worst <= 1e-12)
```

It prints `True`: in a hundred thousand tries, the left side never beats the right.

**Claim (the hinge loss is Lipschitz).** For a fixed example \((\mathbf x, y)\) with \(y \in \{\pm 1\}\), the function \(f(\mathbf w) = \max\{0, 1 - y \langle \mathbf w, \mathbf x \rangle\}\) is \(\|\mathbf x\|\)-Lipschitz.

*Proof.* First a fact about the function \(t \mapsto \max\{0, t\}\): for all real \(a\) and \(c\), \(|\max\{0, a\} - \max\{0, c\}| \le |a - c|\). Check the cases. If both are at most 0, the left side is 0. If both are positive, the two sides are equal. If \(a > 0 \ge c\), the left side is \(a\), and \(a \le a - c\) because \(c \le 0\); the case \(c > 0 \ge a\) is the same with the roles swapped.

Now take \(a = 1 - y \langle \mathbf w, \mathbf x \rangle\) and \(c = 1 - y \langle \mathbf u, \mathbf x \rangle\). Then \(a - c = y \langle \mathbf u - \mathbf w, \mathbf x \rangle\), and

\[ |f(\mathbf w) - f(\mathbf u)| \le |y| \, |\langle \mathbf u - \mathbf w, \mathbf x \rangle| \le \|\mathbf u - \mathbf w\| \, \|\mathbf x\|, \]

using \(|y| = 1\) and Cauchy–Schwarz with \(\mathbf a = \mathbf u - \mathbf w\) and \(\mathbf b = \mathbf x\). \(\blacksquare\)

The ratio \(|f(\mathbf w) - f(\mathbf u)| / (\|\mathbf x\| \|\mathbf w - \mathbf u\|)\) should never exceed 1, and it should get close to 1 when both \(\mathbf w\) and \(\mathbf u\) sit on the sloped part of the hinge and \(\mathbf w - \mathbf u\) points along \(\mathbf x\):

```python
import math, random

rng = random.Random(0)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
hinge = lambda w, x, y: max(0, 1 - y * dot(w, x))
worst = 0
for _ in range(100_000):
    x = [rng.uniform(-2, 2) for _ in range(3)]
    w = [rng.uniform(-2, 2) for _ in range(3)]
    u = [rng.uniform(-2, 2) for _ in range(3)]
    y = rng.choice((1, -1))
    gap = abs(hinge(w, x, y) - hinge(u, x, y))
    ratio = gap / (math.sqrt(dot(x, x)) * math.dist(w, u))
    worst = max(worst, ratio)
print(round(worst, 4))
```

It prints `0.9999`: over a hundred thousand random triples, the ratio comes within a hair of 1 and never passes it.

So when every message lies in the ball of radius \(\rho\), the hinge loss is convex and \(\rho\)-Lipschitz on that domain, and this is the second role of \(\rho\): the radius of the messages is the Lipschitz constant of the loss. Now the guarantee.

**Theorem (soft-SVM’s sample complexity depends on the norm, not the dimension).** Let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \rho\}\), let \(\mathcal D\) be a distribution over \(\mathcal X \times \{\pm 1\}\), and let \(A(S)\) be the homogeneous soft-SVM solution on \(S \sim \mathcal D^m\) with parameter \(\lambda\). Then for every \(\mathbf u\),

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{\text{hinge}}(A(S)) \right] \le L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}, \]

and the same bound holds for the expected 0-1 true risk \(\mathbb E_S [L_{\mathcal D}^{0-1}(A(S))]\). Moreover, for every \(B > 0\), choosing \(\lambda = \sqrt{2\rho^2 / (B^2 m)}\) gives

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{0-1}(A(S)) \right] \le \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{\text{hinge}}(A(S)) \right] \le \min_{\mathbf w : \|\mathbf w\| \le B} L_{\mathcal D}^{\text{hinge}}(\mathbf w) + \sqrt{\frac{8 \rho^2 B^2}{m}}. \]

The first display is the oracle inequality with the hinge loss in it, and \(\mathcal X\) is the ball of messages of length at most \(\rho\). In the second, the superscripts name the loss: \(L_{\mathcal D}^{0-1}\) is the chance of a wrong label on fresh mail, \(L_{\mathcal D}^{\text{hinge}}\) the average hinge loss on it. \(B\) is a budget on length, and \(\min_{\mathbf w : \|\mathbf w\| \le B}\) is the best hinge risk any weight vector of length at most \(B\) can reach. In plain words: **on average, soft-SVM’s error rate on new mail is at most the best hinge risk within the length budget, plus a term that grows with the radius and the budget and shrinks like one over the square root of the number of labels.** With \(\rho = 5\), \(B = 2\), \(m = 10{,}000\) and a best hinge risk of 0.04: \(\sqrt{8 \cdot 25 \cdot 4/10{,}000} \approx 0.283\), so about 0.323, the number the snippet after the proof prints.

*Proof.* *The first bound.* Soft-SVM is regularized loss minimization with the hinge loss, which we just showed is convex and \(\rho\)-Lipschitz for every example in \(\mathcal X \times \{\pm 1\}\). The guarantee of lesson 13b applies word for word with \(L_{\mathcal D} = L_{\mathcal D}^{\text{hinge}}\).

*From hinge to 0-1.* For every example, the 0-1 loss is at most the hinge loss. Write \(s = y \langle \mathbf w, \mathbf x \rangle\). If \(s \le 0\), the halfspace errs or sits on the boundary, the 0-1 loss is at most 1, and the hinge loss is \(1 - s \ge 1\). If \(s > 0\), the 0-1 loss is 0, and the hinge loss is nonnegative. So \(L_{\mathcal D}^{0-1}(\mathbf w) \le L_{\mathcal D}^{\text{hinge}}(\mathbf w)\) for every \(\mathbf w\), taking expectations over \(\mathcal D\) on both sides of a pointwise inequality, and then over \(S\) as well.

*Choosing \(\lambda\).* Fix \(B > 0\) and any \(\mathbf u\) with \(\|\mathbf u\| \le B\). Then \(\lambda \|\mathbf u\|^2 \le \lambda B^2\), so the first bound gives \(L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda B^2 + 2\rho^2/(\lambda m)\). With \(\lambda = \sqrt{2\rho^2/(B^2 m)}\), the two terms that depend on \(\lambda\) become

\[ \lambda B^2 = \sqrt{\frac{2\rho^2 B^4}{B^2 m}} = \sqrt{\frac{2 \rho^2 B^2}{m}}, \qquad \frac{2\rho^2}{\lambda m} = \frac{2\rho^2}{m} \sqrt{\frac{B^2 m}{2 \rho^2}} = \sqrt{\frac{4 \rho^4 B^2 m}{2 \rho^2 m^2}} = \sqrt{\frac{2\rho^2 B^2}{m}}. \]

Both lines only move factors in and out of square roots (\(c = \sqrt{c^2}\) for positive \(c\)). They are equal, which is no accident: this \(\lambda\) is exactly where the increasing term \(\lambda B^2\) and the decreasing term \(2\rho^2/(\lambda m)\) cross, and for a sum \(a\lambda + c/\lambda\) with \(a, c > 0\) the crossing point is the minimum. Their sum is \(2\sqrt{2\rho^2 B^2/m} = \sqrt{8 \rho^2 B^2/m}\). The bound holds for every \(\mathbf u\) in the ball of radius \(B\), so it holds for the best one, which gives the minimum over \(\|\mathbf w\| \le B\). \(\blacksquare\)

Two remarks on what this says. First, it is a bound on the *expected* true risk, averaged over the random training sequence, not a high-probability bound like the hard-SVM theorem; Markov’s inequality turns the one into the other, at some cost in the constants. Second, look at what controls it: the radius \(\rho\), the norm budget \(B\) (which plays the part of \(1/\gamma\), since a separator of norm \(B\) at score level 1 has margin \(1/B\)), and, when the mail is not separable, the best hinge risk available within that norm. No \(d\).

Here are the theorem’s numbers, \(\rho = 5\), \(B = 2\), hinge risk 0.04, at two label counts:

```python
import math

rho, B, hinge_u = 5, 2, 0.04

def soft_svm_bound(lam, m):
    return hinge_u + lam * B ** 2 + 2 * rho ** 2 / (lam * m)

for m in (10_000, 50_000):
    lam = math.sqrt(2 * rho ** 2 / (B ** 2 * m))
    grid = min(soft_svm_bound(k / 1e5, m)
               for k in range(1, 10**4))
    closed = hinge_u + math.sqrt(8 * rho ** 2 * B ** 2 / m)
    print(m, round(lam, 4), round(soft_svm_bound(lam, m), 4),
          round(grid, 4), round(closed, 4))
```

It prints `10000 0.0354 0.3228 0.3228 0.3228` and `50000 0.0158 0.1665 0.1665 0.1665`. The formula’s \(\lambda\), a brute-force search over \(\lambda\), and the closed form all agree: with 50,000 labels the expected 0-1 true risk is certified below 0.1665.

## Margins against dimension

How much does dropping \(d\) actually buy? Compare the two kinds of bound. The VC route of lesson 6a, for homogeneous halfspaces with VC dimension \(d\), makes the estimation error of an ERM hypothesis shrink like \(\sqrt{d/m}\). The margin route makes it shrink like \(\sqrt{\rho^2 B^2/m}\). Which is smaller depends on the problem, so let’s look at a realistic one.

Say you want to tell whether a short document is about sports. Represent it as a **bag of words**: fix a dictionary of \(d\) words, and let \(x_i = 1\) if word \(i\) appears in the document and 0 otherwise, so \(\mathbf x \in \{0, 1\}^d\). Then \(\|\mathbf x\|^2\) is the number of distinct dictionary words in the document, and \(\rho^2\) is the largest such number. A halfspace assigns a weight to every word, and it is natural to assume that positive and negative weights on a few dozen words decide the topic with reasonable accuracy, so \(B^2\) under 100 is plausible. Overall, \(\rho^2 B^2\) below 10,000 is a reasonable assumption. A typical dictionary, on the other hand, is far bigger than 10,000 words; English has more than 100,000.

```python
import math

d, rho2_B2 = 100_000, 10_000
for m in (10**5, 10**6, 10**7):
    vc_term = math.sqrt(d / m)
    margin_term = math.sqrt(rho2_B2 / m)
    print(m, round(vc_term, 3), round(margin_term, 3))
```

It prints `100000 1.0 0.316`, `1000000 0.316 0.1` and `10000000 0.1 0.032`. At every \(m\), the margin term is \(\sqrt{10}\) times smaller, and to reach the same value the VC route needs ten times as many examples: an order of magnitude, for the same halfspaces, just from asking for a wide margin. (The constants in front differ between the two bounds, so read this as a comparison of how they scale, not of their exact values.) And the gap only widens as the dictionary grows, because the margin term does not see the dictionary at all.

[[diagram:bounds]]

It is not a free lunch, though, and you know by now (lesson 5) that there is none. One can build problems where the margin bound is worse than the VC bound: the mail might be separable only by a hyperplane that passes close to many messages. Preferring a large margin is another **inductive bias**. It can sharply decrease the estimation error, and it can also increase the approximation error, when the truth is a halfspace with a narrow margin and the learner, steered toward short \(\mathbf w\), never reaches it.

## The ramp loss

One more subtlety about the soft-SVM guarantee. Its approximation term is the best *hinge* risk, \(\min_{\|\mathbf w\| \le B} L_{\mathcal D}^{\text{hinge}}(\mathbf w)\), while a VC bound measures the approximation error with the 0-1 loss. Since the hinge loss is at least the 0-1 loss, the hinge approximation term can be larger, and the 0-1 approximation error never exceeds it.

Could we get the nice \(\sqrt{\rho^2 B^2/m}\) estimation term with the 0-1 loss itself? No. The 0-1 loss is **scale insensitive**: multiplying \(\mathbf w\) by any positive number changes no label and no 0-1 loss, so the norm of \(\mathbf w\), and the margin, mean nothing to it. A norm-based bound needs a loss that notices scale. The **ramp loss** is a compromise that notices scale and is otherwise much closer to the 0-1 loss:

\[ \ell^{\text{ramp}}(\mathbf w, (\mathbf x, y)) = \min\{1, \ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))\} = \min\{1, \max\{0, 1 - y \langle \mathbf w, \mathbf x \rangle\}\}. \]

Inside out: the hinge loss, then \(\min\{1, \cdot\}\), which keeps the smaller of the two, so the charge is capped at 1. In plain words: **the ramp loss is the hinge loss, never more than 1.** J under the soft-SVM solution pays hinge 1.5 but ramp only 1. Here are the three losses as functions of the signed score \(s = y\langle \mathbf w, \mathbf x \rangle\), each line one formula:

```python
zero_one = lambda s: 1 if s <= 0 else 0
hinge = lambda s: max(0.0, 1.0 - s)
ramp = lambda s: min(1.0, hinge(s))
for s in (-2, -0.5, 0, 0.5, 1, 2):
    print(s, zero_one(s), hinge(s), ramp(s))
```

It prints `-2 1 3.0 1.0`, `-0.5 1 1.5 1.0`, `0 1 1.0 1.0`, `0.5 0 0.5 0.5`, `1 0 0.0 0.0` and `2 0 0.0 0.0`.

On a mistake (\(s \le 0\)) the ramp loss charges exactly 1, like the 0-1 loss, while the hinge loss keeps growing. On a comfortable correct answer (\(s \ge 1\)) both charge 0. The ramp and 0-1 losses differ only on examples that are correct but not by a significant margin, \(0 < s < 1\). Generalization bounds for the ramp loss come with lesson 26.

[[diagram:hinge]]

So why does SVM use the hinge loss and not the ramp? Computation. The hinge loss is convex, so minimizing it is efficient: every local minimum is global, and the methods of lessons 12a to 14b apply. The ramp loss is not convex (it is flat for very negative \(s\), then slopes, then is flat again), and minimizing it is computationally intractable.

## Support vectors

Where does the name come from? Go back to the eight separable messages and the three that sat exactly at score 1: B, E and G. The claim behind the name is that hard-SVM’s weight vector is **supported** by those messages alone: it is a linear combination of them, and the other five could move around (as long as they stay outside the margin) without changing it.

We work with homogeneous hard-SVM, \(\mathbf w_0 = \operatorname{argmin}_{\mathbf w} \|\mathbf w\|^2\) subject to \(y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1\) for all \(i\). Two facts first, so that \(\mathbf w_0\) is well defined when \(S\) is separable. It **exists**: take any feasible \(\mathbf w'\); the feasible vectors with \(\|\mathbf w\| \le \|\mathbf w'\|\) form a closed and bounded set (each constraint is a closed halfspace), it is nonempty, and a continuous function on a nonempty closed bounded set attains its minimum. It is **unique**: if \(\mathbf w\) and \(\mathbf w'\) were two different minimizers, both of norm \(r\), their midpoint would be feasible (average the two constraint inequalities) and, by the identity \(\|\tfrac{\mathbf w + \mathbf w'}{2}\|^2 = \tfrac12 \|\mathbf w\|^2 + \tfrac12 \|\mathbf w'\|^2 - \tfrac14 \|\mathbf w - \mathbf w'\|^2\), it would have squared norm \(r^2 - \tfrac14 \|\mathbf w - \mathbf w'\|^2 < r^2\), contradicting minimality.

**Theorem (support vectors).** Let \(\mathbf w_0\) be the homogeneous hard-SVM solution on a separable \(S\), and let \(I = \{ i : |\langle \mathbf w_0, \mathbf x_i \rangle| = 1\}\). Then there are coefficients \(\alpha_i\), \(i \in I\), with

\[ \mathbf w_0 = \sum_{i \in I} \alpha_i \mathbf x_i. \]

The messages \(\{\mathbf x_i : i \in I\}\) are called the **support vectors**.

Reading it: \(I\) is the set of positions \(i\) such that message \(i\)’s score under \(\mathbf w_0\) is exactly \(\pm 1\), the messages on the edge of the margin. Each \(\alpha_i\) (alpha) is a number, and \(\sum_{i \in I}\) adds the scaled messages over those positions only. In plain words: **the hard-SVM weight vector is built from the messages on the margin and nothing else.** On our lifted messages \(I\) is B, E and G, as the check further down shows.

Since \(\mathbf w_0\) is feasible, \(y_i \langle \mathbf w_0, \mathbf x_i \rangle \ge 1 > 0\) for every \(i\), so \(|\langle \mathbf w_0, \mathbf x_i \rangle| = y_i \langle \mathbf w_0, \mathbf x_i \rangle\). Thus \(I\) is the set of examples whose constraint holds with equality, the ones at distance exactly \(1/\|\mathbf w_0\|\) from the hyperplane, and every other example has \(y_i \langle \mathbf w_0, \mathbf x_i \rangle > 1\).

The standard proof applies a general optimization tool, the **Fritz John conditions**, which say that at a constrained minimum the gradient of the objective (here \(2\mathbf w_0\)) is a combination of the gradients of the tight constraints (here \(-y_i \mathbf x_i\)); solving for \(\mathbf w_0\) gives the theorem. We will not prove that tool. Here is instead a direct proof that uses nothing but the geometry of the inner product.

*Proof.* Let \(V\) be the span of the support vectors, the set of all linear combinations of \(\{\mathbf x_i : i \in I\}\). Every vector splits uniquely into a part in \(V\) and a part perpendicular to every vector of \(V\) (its orthogonal projection onto \(V\) and the remainder), so write \(\mathbf w_0 = \mathbf p + \mathbf r\) with \(\mathbf p \in V\) and \(\langle \mathbf r, \mathbf x_i \rangle = 0\) for every \(i \in I\). The theorem says exactly that \(\mathbf r = \mathbf 0\). Suppose not; we will build a feasible vector shorter than \(\mathbf w_0\), which is impossible.

For a number \(t\) with \(0 < t \le 1\), consider \(\mathbf w_t = \mathbf w_0 - t \mathbf r\), which shaves off part of the component that the support vectors cannot see.

*The support vectors do not notice.* For \(i \in I\), \(y_i \langle \mathbf w_t, \mathbf x_i \rangle = y_i \langle \mathbf w_0, \mathbf x_i \rangle - t y_i \langle \mathbf r, \mathbf x_i \rangle = 1 - 0 = 1\). Their constraints still hold.

*The others have room to spare.* For \(i \notin I\), we know \(y_i \langle \mathbf w_0, \mathbf x_i \rangle > 1\). Then \(y_i \langle \mathbf w_t, \mathbf x_i \rangle \ge y_i \langle \mathbf w_0, \mathbf x_i \rangle - t |\langle \mathbf r, \mathbf x_i \rangle|\), and this is still at least 1 as long as \(t |\langle \mathbf r, \mathbf x_i \rangle| \le y_i \langle \mathbf w_0, \mathbf x_i \rangle - 1\), a positive number. There are finitely many examples, so we can choose one \(t \in (0, 1]\) small enough for all of them at once. Then \(\mathbf w_t\) is feasible.

*But it is shorter.* Expand the square:

\[ \|\mathbf w_t\|^2 = \|\mathbf w_0\|^2 - 2t \langle \mathbf w_0, \mathbf r \rangle + t^2 \|\mathbf r\|^2. \]

This is the square of a sum of two arrows, \(\|\mathbf a + \mathbf c\|^2 = \|\mathbf a\|^2 + \|\mathbf c\|^2 + 2\langle \mathbf a, \mathbf c \rangle\), the vector form of \((a + c)^2\) that lesson 15a used for the distance to a hyperplane, with \(\mathbf a = \mathbf w_0\) and \(\mathbf c = -t\mathbf r\). Since \(\mathbf p\) lies in \(V\) and \(\mathbf r\) is perpendicular to every vector of \(V\), \(\langle \mathbf p, \mathbf r \rangle = 0\), so \(\langle \mathbf w_0, \mathbf r \rangle = \langle \mathbf p + \mathbf r, \mathbf r \rangle = \|\mathbf r\|^2\). Hence \(\|\mathbf w_t\|^2 = \|\mathbf w_0\|^2 - (2t - t^2) \|\mathbf r\|^2\), and for \(0 < t \le 1\) we have \(2t - t^2 = t(2 - t) > 0\), so \(\|\mathbf w_t\|^2 < \|\mathbf w_0\|^2\). A feasible vector shorter than the minimizer is a contradiction, so \(\mathbf r = \mathbf 0\) and \(\mathbf w_0 = \mathbf p \in V\). \(\blacksquare\)

One edge case falls out for free: \(I\) cannot be empty. If it were, \(V = \{\mathbf 0\}\) and the theorem would give \(\mathbf w_0 = \mathbf 0\), which violates every constraint.

[[diagram:support]]

On our eight lifted messages, the support vectors are B, E and G, and the next section’s solver finds the coefficients. Here they are, so you can check by hand: in the form \(\mathbf w_0 = \sum_i \alpha_i y_i \mathbf x_i\) that the next section derives, \(\alpha_B = \tfrac{5}{18}\), \(\alpha_E = \tfrac{10}{9}\), \(\alpha_G = \tfrac16\), and

\[ \tfrac{5}{18}(1, 3, 0) - \tfrac{10}{9}(1, 0, 0) - \tfrac16 (1, 1, 2) = \left(\tfrac{5 - 20 - 3}{18}, \tfrac{15 - 3}{18}, -\tfrac{6}{18}\right) = \left(-1, \tfrac23, -\tfrac13\right). \]

Each vector is a lifted message, \((1, x_1, x_2)\), and the minus signs are the labels of E and G. Here is the same sum in exact fractions, with \(I\) found from the scores rather than by eye:

```python
from fractions import Fraction as Fr

S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X = [(1,) + x for x, _ in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w0 = (Fr(-1), Fr(2, 3), Fr(-1, 3))
I = [i for i, x in enumerate(X) if abs(dot(w0, x)) == 1]
alpha = {1: Fr(5, 18), 4: Fr(10, 9), 6: Fr(1, 6)}
w = [sum(alpha[i] * S[i][1] * X[i][k] for i in I)
     for k in range(3)]
print(["ABCDEFGH"[i] for i in I], [str(c) for c in w])
```

It prints `['B', 'E', 'G'] ['-1', '2/3', '-1/3']`. Five messages, the ones far from the line, contribute nothing.

## Duality

Many properties of SVMs were historically discovered by studying a different optimization problem, the **dual** of hard-SVM. Our development never needed it, but it explains where the coefficients \(\alpha_i\) come from, and it is the door to the next lesson, so let’s derive it for the homogeneous program.

*Step 1: hide the constraints inside a max.* Define

\[ g(\mathbf w) = \max_{\boldsymbol\alpha \in \mathbb R^m : \boldsymbol\alpha \ge \mathbf 0} \; \sum_{i=1}^m \alpha_i (1 - y_i \langle \mathbf w, \mathbf x_i \rangle). \]

Read it from the inside out. \(1 - y_i \langle \mathbf w, \mathbf x_i \rangle\) is example \(i\)’s shortfall below signed score 1, negative when the constraint holds with room to spare. Each \(\alpha_i\) is a nonnegative weight on it, \(\boldsymbol\alpha \ge \mathbf 0\) means every coordinate of the list \(\boldsymbol\alpha = (\alpha_1, \dots, \alpha_m)\) is at least 0, and \(\max\) lets an adversary choose the weights to make the weighted sum as large as possible. In plain words: **\(g\) is the largest penalty an adversary can charge for broken constraints.** At \(\mathbf w_0\) the shortfalls of our eight lifted messages add up to \(-\tfrac73\), so any equal weight \(a\) on all of them gives \(-\tfrac73 a\), at best 0; at \(\mathbf w = \mathbf 0\) every shortfall is 1, and the sum is \(8a\), as large as you like:

```python
S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))

def inner_sum(w, alpha):
    return sum(a * (1 - y * dot(w, x))
               for a, x, y in zip(alpha, X, Y))

for w in [(-1, 2 / 3, -1 / 3), (0, 0, 0)]:
    sums = [inner_sum(w, [a] * 8) for a in (0, 1, 10, 100)]
    print([round(s, 2) for s in sums])
```

It prints `[0.0, -2.33, -23.33, -233.33]` and `[0, 8, 80, 800]`. The general argument: if \(\mathbf w\) satisfies every constraint, each \(1 - y_i \langle \mathbf w, \mathbf x_i \rangle\) is at most 0, so every term with \(\alpha_i \ge 0\) is at most 0, and the maximum is 0, reached at \(\boldsymbol\alpha = \mathbf 0\). If some constraint fails, its \(1 - y_i \langle \mathbf w, \mathbf x_i \rangle\) is positive, and sending that \(\alpha_i\) to infinity (with the others at 0) makes the sum as large as you like, so \(g(\mathbf w) = \infty\). So \(g\) is 0 on the feasible set and \(\infty\) off it, and hard-SVM is the unconstrained problem \(\min_{\mathbf w} \left( \|\mathbf w\|^2 + g(\mathbf w) \right)\).

*Step 2: a min of a max.* Halving the objective does not change the minimizer, and \(\tfrac12 g = g\) because \(g\) only takes the values 0 and \(\infty\). So hard-SVM has the same solution as

\[ \min_{\mathbf w} \; \max_{\boldsymbol\alpha \ge \mathbf 0} \left( \tfrac12 \|\mathbf w\|^2 + \sum_{i=1}^m \alpha_i (1 - y_i \langle \mathbf w, \mathbf x_i \rangle) \right). \]

*Step 3: swap.* Exchanging the min and the max can only lower the value:

\[ \min_{\mathbf w} \max_{\boldsymbol\alpha \ge \mathbf 0} \Big( \cdots \Big) \;\ge\; \max_{\boldsymbol\alpha \ge \mathbf 0} \min_{\mathbf w} \Big( \cdots \Big). \]

The dots stand for the same expression on both sides. On the left the minimizer moves first and the maximizer answers; on the right the maximizer must commit first, and moving second can only help the minimizer. This inequality, true for any function of two variables, is called **weak duality**, and proving it is one of this lesson’s problems. For hard-SVM something stronger holds, **strong duality**: the two sides are equal. We state strong duality without proof; it is a standard result of convex optimization. The right-hand side is the **dual problem**.

*Step 4: solve the inner minimization.* For fixed \(\boldsymbol\alpha\), the inner function of \(\mathbf w\) is \(\tfrac12 \|\mathbf w\|^2\) plus a linear function, unconstrained and differentiable, and convex (a convex quadratic plus a linear term). A convex differentiable function is minimized exactly where its gradient vanishes, and the gradient is \(\mathbf w - \sum_i \alpha_i y_i \mathbf x_i\). So

\[ \mathbf w = \sum_{i=1}^m \alpha_i y_i \mathbf x_i. \]

In plain words: **for fixed weights \(\alpha_i\), the best \(\mathbf w\) is the sum of the messages, each scaled by its weight and its label.** It is the sum the support-vector snippet computed. The optimal weight vector is a combination of the training messages, a fact the next lesson builds on.

*Step 5: substitute.* Put \(\mathbf w = \sum_j \alpha_j y_j \mathbf x_j\) back in. The sum \(\sum_i \alpha_i y_i \langle \mathbf w, \mathbf x_i \rangle = \langle \mathbf w, \sum_i \alpha_i y_i \mathbf x_i \rangle = \|\mathbf w\|^2\), so the objective becomes \(\tfrac12 \|\mathbf w\|^2 + \sum_i \alpha_i - \|\mathbf w\|^2 = \sum_i \alpha_i - \tfrac12 \|\mathbf w\|^2\). Expanding \(\|\mathbf w\|^2 = \langle \sum_i \alpha_i y_i \mathbf x_i, \sum_j \alpha_j y_j \mathbf x_j \rangle\) one pair of terms at a time (the inner product of two sums is the sum of the inner products of their terms, as \((a + c)(a' + c')\) multiplies out) gives the dual problem:

\[ \max_{\boldsymbol\alpha \in \mathbb R^m : \boldsymbol\alpha \ge \mathbf 0} \left( \sum_{i=1}^m \alpha_i - \frac12 \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \langle \mathbf x_j, \mathbf x_i \rangle \right). \]

The double sum runs over every pair of positions \(i\) and \(j\), so it is \(\|\mathbf w\|^2\) written out pair by pair. In plain words: **choose nonnegative weights to make their total minus half the squared length of the weight vector they build as large as possible.** At the coefficients \(\tfrac5{18}, \tfrac{10}9, \tfrac16\) on B, E and G the total is \(\tfrac{14}{9}\), the vector built is \(\mathbf w_0\) with \(\|\mathbf w_0\|^2 = 1 + \tfrac49 + \tfrac19 = \tfrac{14}9\), and the dual value is \(\tfrac{14}9 - \tfrac79 = \tfrac79\), the last number the solver below prints.

Look at what it needs: only the inner products \(\langle \mathbf x_j, \mathbf x_i \rangle\) between messages, never the coordinates of a message on their own. Hold on to that; it is the whole point of the next lesson.

[[diagram:duality]]

The dual is easy to attack one coordinate at a time. Fix every \(\alpha_j\) except \(\alpha_i\), and let \(\mathbf w = \sum_j \alpha_j y_j \mathbf x_j\). Raising \(\alpha_i\) by \(\delta\) moves \(\mathbf w\) by \(\delta y_i \mathbf x_i\), so the dual value \(\sum_j \alpha_j - \tfrac12\|\mathbf w\|^2\) changes by \(\delta (1 - y_i \langle \mathbf w, \mathbf x_i \rangle) - \tfrac12 \delta^2 \|\mathbf x_i\|^2\) (expand the square, using \(y_i^2 = 1\)). That is a downward parabola in \(\delta\), highest at \(\delta = (1 - y_i \langle \mathbf w, \mathbf x_i \rangle)/\|\mathbf x_i\|^2\), and clipping the new \(\alpha_i\) at 0 respects \(\alpha_i \ge 0\): the solver’s line `new = max(0.0, alpha[i] + gap / dot(X[i], X[i]))`. Sweeping over the coordinates again and again climbs to the maximum. (This method, dual coordinate ascent, is a standard solver; it is the one that produced the numbers we have been quoting.) At the end, compare the primal value \(\tfrac12 \|\mathbf w_0\|^2\) with the dual value:

```python
S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
alpha, w = [0.0] * 8, [0.0] * 3
for sweep in range(5000):
    for i in range(8):
        gap = 1 - Y[i] * dot(w, X[i])
        new = max(0.0, alpha[i] + gap / dot(X[i], X[i]))
        w = [c + (new - alpha[i]) * Y[i] * xi
             for c, xi in zip(w, X[i])]
        alpha[i] = new
print([round(a, 4) for a in alpha])
print([round(c, 4) for c in w])
primal, dual = dot(w, w) / 2, sum(alpha) - dot(w, w) / 2
print(round(primal, 4), round(dual, 4))
```

It prints `[0.0, 0.2778, 0.0, 0.0, 1.1111, 0.0, 0.1667, 0.0]`, then `[-1.0, 0.6667, -0.3333]`, then `0.7778 0.7778`. The coefficients are \(\tfrac{5}{18}\), \(\tfrac{10}{9}\) and \(\tfrac16\) on B, E and G and zero elsewhere, exactly the support vectors; the weight vector is the homogeneous hard-SVM solution of lesson 15a; and primal and dual values agree at \(\tfrac79\), as strong duality promises.

The same method solves soft-SVM, with one change. Repeating the derivation for the soft-SVM program with its slack variables gives a dual whose only difference is a ceiling: each coefficient must stay between 0 and \(1/(2\lambda m)\), so the update clips at both ends. That is the solver behind the soft-SVM solution \((-\tfrac12, \tfrac12, -\tfrac12)\) we quoted for the ten messages.

## Soft-SVM by stochastic gradient descent

Quadratic-program solvers and dual coordinate ascent both work, but for the real mail system, with millions of messages and fifty thousand words, there is a much simpler method you already own. Lesson 14b showed how to solve regularized loss minimization with stochastic gradient descent. Let’s apply it to soft-SVM, in the homogeneous form

\[ \min_{\mathbf w} \left( \frac{\lambda}{2} \|\mathbf w\|^2 + \frac1m \sum_{i=1}^m \max\{0, 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\} \right). \]

(Notice the \(\lambda/2\). It makes the algebra below cleaner. Running this method with parameter \(\lambda\) solves soft-SVM as we defined it with parameter \(\lambda/2\); to get soft-SVM with parameter \(\lambda\), run it with \(2\lambda\).)

Recall the method from lesson 14b. The objective \(f(\mathbf w) = \tfrac{\lambda}{2}\|\mathbf w\|^2 + L_S(\mathbf w)\) is **\(\lambda\)-strongly convex** (lesson 13b): it curves upward at least as much as the bowl \(\tfrac{\lambda}{2}\|\mathbf w\|^2\), which is what makes a step size shrinking like \(1/t\) safe. **Stochastic gradient descent** (SGD) walks downhill using a cheap, noisy slope computed from one example at a time. At step \(t\), pick an example uniformly at random from \(S\) and a subgradient \(\mathbf v_t\) of its loss at the current weights \(\mathbf w^{(t)}\) (the superscript in parentheses counts steps; it is not a power); then \(\lambda \mathbf w^{(t)} + \mathbf v_t\) is, in expectation, a subgradient of \(f\), and the update with step size \(1/(\lambda t)\) is

\[ \mathbf w^{(t+1)} = \mathbf w^{(t)} - \frac{1}{\lambda t} \left( \lambda \mathbf w^{(t)} + \mathbf v_t \right) = \left(1 - \frac1t\right) \mathbf w^{(t)} - \frac{1}{\lambda t} \mathbf v_t, \]

starting from \(\mathbf w^{(1)} = \mathbf 0\).

From the inside out: \(\lambda \mathbf w^{(t)}\) is the slope of the bowl, \(\mathbf v_t\) the slope of one example’s loss, their sum the step direction, and \(\frac{1}{\lambda t}\) the step size, shrinking as \(t\) grows. Expanding gives the right-hand form. In plain words: **shrink the current weights by the factor \(1 - 1/t\), then step against one example’s slope.** At \(t = 1\) the factor is 0; at \(t = 2\) the weights are halved before the step. The snippet below runs exactly this line, `w = [wi - gi / (lam * t) ...]`.

*The update in closed form.* We claim \(\mathbf w^{(t+1)} = -\frac{1}{\lambda t} \sum_{j=1}^t \mathbf v_j\) for every \(t \ge 1\). For \(t = 1\), the factor \(1 - 1/1 = 0\) kills \(\mathbf w^{(1)}\) and leaves \(-\mathbf v_1/\lambda\). If it holds for \(t - 1\), that is \(\mathbf w^{(t)} = -\frac{1}{\lambda(t-1)} \sum_{j<t} \mathbf v_j\), then

\[ \mathbf w^{(t+1)} = \frac{t-1}{t} \cdot \frac{-1}{\lambda (t-1)} \sum_{j<t} \mathbf v_j - \frac{1}{\lambda t} \mathbf v_t = -\frac{1}{\lambda t} \sum_{j \le t} \mathbf v_j. \]

The factor \(\frac{t-1}{t}\) turns \(\frac{1}{\lambda(t-1)}\) into \(\frac{1}{\lambda t}\), so the old subgradients and the new one end up with the same weight. So the weights at step \(t\) are just the running sum of the subgradients, scaled. Check it against the step-by-step update with fifty random subgradients:

```python
import random

rng, lam = random.Random(0), 0.2
v = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(50)]
w = [0.0, 0.0, 0.0]
for t in range(1, 51):
    g = [lam * wi + vi for wi, vi in zip(w, v[t - 1])]
    w = [wi - gi / (lam * t) for wi, gi in zip(w, g)]
closed = [-sum(col) / (lam * 50) for col in zip(*v)]
print([round(c, 6) for c in w])
print([round(c, 6) for c in closed])
```

Both lines print `[-0.685902, -0.263015, -0.231963]`.

*The subgradient of the hinge loss.* Lesson 14a computed it: at \(\mathbf w\), for the example \((\mathbf x, y)\), take \(\mathbf v = \mathbf 0\) if \(y \langle \mathbf w, \mathbf x \rangle \ge 1\) (the flat part) and \(\mathbf v = -y\mathbf x\) otherwise (the sloped part). Write \(\boldsymbol\theta^{(t)} = -\sum_{j<t} \mathbf v_j\) for the running sum. Then \(\boldsymbol\theta\) changes only on examples with signed score below 1, by adding \(y_i \mathbf x_i\), and the closed form gives \(\mathbf w^{(t)} = \boldsymbol\theta^{(t)}/(\lambda (t-1))\) for \(t \ge 2\). The standard form of the algorithm, below, divides by \(\lambda t\) instead: it is defined already at \(t = 1\), and the two differ by the factor \((t-1)/t\), which tends to 1. That is the whole algorithm.

```
SGD-SoftSVM(S, λ, T):
  θ ← (0, ..., 0) in R^d
  total ← (0, ..., 0) in R^d
  for t = 1..T:
    w ← θ / (λ t)
    total ← total + w
    choose i uniformly at random from {1, ..., m}
    if y_i ⟨w, x_i⟩ < 1:
      θ ← θ + y_i x_i
  return w̄ = total / T
```

Look at the update line. It is the Perceptron’s update, \(y_i \mathbf x_i\), triggered not only on mistakes but on every example with signed score below 1: soft-SVM by SGD is a Perceptron that insists on a margin and shrinks its weights by \(1/(\lambda t)\) as it goes.

[[diagram:sgd]]

How good is \(\bar{\mathbf w}\)? Lesson 14b’s analysis of exactly this objective with a \(\rho\)-Lipschitz loss gives

\[ \mathbb E[f(\bar{\mathbf w})] - \min_{\mathbf w} f(\mathbf w) \le \frac{2\rho^2}{\lambda T} (1 + \log T). \]

The bar on \(\bar{\mathbf w}\) marks the average of the \(T\) iterates, the `total / T` of the pseudocode. \(\mathbb E\) averages over the random choices of examples, and \(\min_{\mathbf w} f(\mathbf w)\) is the best objective value there is, so the left side is how far, on average, the returned weights are from optimal. On the right, \(1 + \log T\) grows very slowly while \(T\) in the denominator grows fast. In plain words: **after \(T\) steps, the averaged weights are on average within about \(\log T / T\) of the best objective, scaled by \(2\rho^2/\lambda\).**

Let’s run it on the ten almost-separable messages, lifted, with \(\lambda = 0.2\) and \(T = 100{,}000\). By the remark above, that solves soft-SVM with parameter 0.1, whose minimizer we quoted as \((-\tfrac12, \tfrac12, -\tfrac12)\) with objective 0.475.

```python
import random

S10 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
       ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1),
       ((1, 2), 1), ((3, 1), -1)]
X, Y = [(1,) + x for x, _ in S10], [y for _, y in S10]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
rng, lam, T = random.Random(0), 0.2, 100_000
theta, total = [0.0] * 3, [0.0] * 3
for t in range(1, T + 1):
    w = [c / (lam * t) for c in theta]
    total = [s + c for s, c in zip(total, w)]
    i = rng.randrange(len(X))
    if Y[i] * dot(w, X[i]) < 1:
        theta = [c + Y[i] * xi for c, xi in zip(theta, X[i])]
w_bar = [s / T for s in total]
hinge = sum(max(0, 1 - y * dot(w_bar, x)) for x, y in zip(X, Y))
print([round(c, 3) for c in w_bar])
print(round(lam / 2 * dot(w_bar, w_bar) + hinge / len(X), 4))
```

It prints `[-0.518, 0.502, -0.484]` and `0.4752`: within 0.0002 of the optimum 0.475. The guarantee, with \(\rho^2 = 30\) (the longest lifted message is A, \((1, 5, 2)\)), promises an expected gap of at most \(2 \cdot 30 \cdot (1 + \log 100{,}000)/(0.2 \cdot 100{,}000)\):

```python
import math

rho2, lam, T = 30, 0.2, 100_000
print(round(2 * rho2 / (lam * T) * (1 + math.log(T)), 4))
```

It prints `0.0375`. As with the Perceptron’s mistake bound, the guarantee is a worst case, and on friendly data the method does far better.

## SVMs as code

Here are the two learners side by side, homogeneous, with messages already lifted if a bias term is wanted, so they live in \(\mathbb R^d\).

```
HardSVM(S):
  w0 ← argmin ||w||^2 subject to y_i ⟨w, x_i⟩ ≥ 1 for all i
  return h(x) = sign(⟨w0, x⟩)

SGD-SoftSVM(S, λ, T):
  as in the previous section
  return h(x) = sign(⟨w̄, x⟩)
```

**HardSVM** hands the work to a quadratic-program solver, which runs in time polynomial in \(m\) and \(d\); which polynomial depends on the solver. Through the dual, one sweep of dual coordinate ascent costs \(O(md)\) time (each of the \(m\) coordinate steps computes one inner product and updates \(\mathbf w\)), with \(O(m + d)\) memory for \(\boldsymbol\alpha\) and \(\mathbf w\) beyond the data. It needs separable data; on anything else the program is infeasible.

**SGD-SoftSVM** does \(O(d)\) work per step: one inner product, at most one vector addition, and one addition to the running total. That is \(O(Td)\) time and \(O(d)\) memory beyond the data, for any \(m\): the number of steps is set by the accuracy you want, through the bound above, not by the number of examples. It never needs separability. Predicting a new message costs \(O(d)\) for either learner.

**Build it yourself.** Using only the Python standard library:

```python
import math
import random

S8 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
      ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
S10 = S8 + [((1, 2), 1), ((3, 1), -1)]

def lift(x): ...
def hinge_risk(w, S): ...
def soft_svm_objective(w, S, lam): ...
def sgd_soft_svm(S, lam, T, rng): ...
def hard_svm(S, sweeps=5000): ...
def plane_margin(w, S): ...
def at_or_inside_margin(w, S, tol=1e-6): ...

if __name__ == "__main__":
    w, alpha = hard_svm(S8)
    assert all(abs(a - b) < 1e-3
               for a, b in zip(w, (-1, 2 / 3, -1 / 3)))
    assert at_or_inside_margin(w, S8) == [1, 4, 6]
    support = [i for i, a in enumerate(alpha) if a > 1e-6]
    assert support == [1, 4, 6]
    assert abs(plane_margin(w, S8) - 3 / math.sqrt(5)) < 1e-3
    gap = plane_margin((-3, 4, -1), S8) - 1 / math.sqrt(17)
    assert abs(gap) < 1e-9
    hinge = hinge_risk((-1, 2 / 3, -1 / 3), S10)
    assert abs(hinge - 11 / 30) < 1e-9
    w_soft = (-0.5, 0.5, -0.5)
    value = soft_svm_objective(w_soft, S10, 0.1)
    assert abs(value - 0.475) < 1e-9
    inside = at_or_inside_margin(w_soft, S10)
    assert inside == [0, 1, 2, 4, 5, 6, 8, 9]
    w = sgd_soft_svm(S10, 0.1, 200_000, random.Random(0))
    assert soft_svm_objective(w, S10, 0.1) < 0.48
```

1. `lift(x)` puts a constant 1 in front of the counts; every weight vector is \((b, w_1, w_2)\). `hinge_risk` averages the hinge loss over `S` and `soft_svm_objective` returns \(\lambda \|\mathbf w\|^2\) plus it, with the bias term inside the norm.
2. `sgd_soft_svm(S, lam, T, rng)` must minimize `soft_svm_objective` with parameter `lam`, so it runs the SGD procedure with \(2\lambda\). It returns \(\bar{\mathbf w}\) as a list.
3. `hard_svm` runs dual coordinate ascent on the lifted messages and returns `(w, alpha)`. `plane_margin(w, S)` is the smallest signed distance in the count plane, \(\min_i y_i(\langle \mathbf w, (1, \mathbf x_i) \rangle) / \|(w_1, w_2)\|\). `at_or_inside_margin` lists the indices with \(y_i \langle \mathbf w, (1, \mathbf x_i) \rangle \le 1\) (up to `tol`).
4. On `S10`, `at_or_inside_margin` returns message A (index 0), whose signed score is exactly 1, but the dual coefficient of A at the soft-SVM solution is 0. Explain why “on the margin” and “has a nonzero coefficient” are not the same thing.
5. Run `sgd_soft_svm` on `S10` for \(\lambda\) from 0.001 to 1 and plot the lifted margin \(1/\|\mathbf w\|\) (bias term included) and the hinge risk against \(\lambda\). Which way does each move, and why? Then plot the plane margin \(1/\|(w_1, w_2)\|\) as well. It does not move in one direction: explain why, remembering which program regularizes the bias term.

## Fix the bug

Time to ship. The production filter tracks a vocabulary of \(d = 50{,}000\) words, each a yes/no feature as in lesson 1: \(x_j = 1\) when word \(j\) appears in the message. Messages are short, and none contains more than 25 distinct vocabulary words, so every message vector has length at most \(\rho = 5\). The mail is almost separable: a few messages are mislabelled or genuinely ambiguous, like the consultation that looks exactly like a lunch invitation.

Operations has the same demand as in lesson 9a: a written guarantee, computed before training, that fits the label budget of 50,000 labels. The guarantee they want is a certified expected 0-1 true risk of at most 0.18.

First, what does the VC route offer? Homogeneous halfspaces over 50,000 words have VC dimension 50,000, and the fundamental theorem of lesson 6a says that with far fewer than \(d/\varepsilon\) labels, no learner can promise accuracy \(\varepsilon\) on every distribution. Fifty thousand labels is a single multiple of \(d\), before any accuracy factor. That route is closed.

The margin route is open, because the soft-SVM theorem has no \(d\) in it. In the lab below, the team has measured three candidate comparison separators on a large pool of labelled mail: a wide one (norm \(B = \sqrt2\), hinge risk 0.12), a middle one (\(B = 2\), hinge risk 0.04) and a tight one (\(B = 4\), hinge risk 0.01). You choose a comparator, the regularization parameter \(\lambda\) and the number of labels \(m\), and the lab computes the certificate

\[ L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}. \]

It is the right side of the soft-SVM theorem, exactly what `soft_svm_bound` computed. Only a certificate at most 0.18, within the budget, ships the filter. Then the lab trains soft-SVM by SGD with your \(\lambda\) on your \(m\) labels and shows its error on fresh mail, but the training run does not count: what ships is the certificate, because tomorrow’s training sequence will be different. Before you touch the sliders, work out which comparator can possibly make it, and roughly where \(\lambda\) must be.

[[lab-act2]]

## What people get wrong

**“The margin bound beats the VC bound.”** It scales better when \(\rho^2 B^2\) is much smaller than \(d\), as for short documents over a large dictionary. It can be worse on mail that is separable only with a narrow margin, and the hinge loss in its approximation term can exceed the 0-1 error. Preferring a large margin is an inductive bias, with the usual price.

**“Every message on the margin pulls on the separator.”** The weight vector is a combination of messages whose constraints are tight (for soft-SVM, messages on or inside the margin), but being on the margin does not force a nonzero coefficient. In the ten-message soft-SVM example, message A sits exactly at score 1, yet its dual coefficient is 0: removing it would change nothing.

**“Soft-SVM with a tiny \(\lambda\) is hard-SVM.”** On some samples it is, but no single \(\lambda\) works for every separable sample. Whether the two always agree for one well-chosen \(\lambda\) is one of this lesson’s problems, and the answer may surprise you.

## Recap and the next bug

1. **Soft-SVM** allows violations through slack variables, and is exactly regularized loss minimization with the **hinge loss**. Because the hinge loss is convex and \(\rho\)-Lipschitz, its expected 0-1 true risk is at most the best hinge risk of norm \(B\) plus \(\sqrt{8\rho^2 B^2/m}\), again with no \(d\).
2. The hard-SVM weight vector is a combination of the **support vectors**, the messages on the margin. The **dual problem** finds the coefficients and involves the messages only through their inner products.
3. Soft-SVM is solved by SGD in \(O(Td)\) time: a Perceptron-like update \(\boldsymbol\theta \leftarrow \boldsymbol\theta + y_i \mathbf x_i\) whenever the signed score is below 1, with weights \(\boldsymbol\theta/(\lambda t)\), averaged.

The filter ships with its certificate, and the nightly job stops coming back empty-handed. Then the data team brings you something unsettling. They took a month of junk your users labelled and looked at the words, and junk is often marked not by single words but by combinations: “free” alone is innocent, “meeting” alone is innocent, but “free” together with “click” and without “meeting” is almost always junk. No halfspace on word counts can express “this word and that word, but not that one”. The obvious repair is to add a feature for every pair of words. But with 50,000 words that is about 1.25 billion pairs, every message becomes a vector in a space of over a billion dimensions, and just writing one down is out of the question. The margin bound does not care about the dimension, so statistically you could afford that space. But can a linear method work in that space without ever building it? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof of the support-vector theorem, the vector \(\mathbf w_t = \mathbf w_0 - t\mathbf r\) must stay feasible. Why must \(t\) be small, and where exactly does the proof use the fact that there are finitely many examples? Why does the proof need \(t \le 1\) rather than any positive \(t\)? And why can the set of support vectors never be empty?
2. Messages have radius \(\rho = 5\), and some separator \(\mathbf u\) of norm \(B = 2\) has hinge risk 0.04. With \(m = 20{,}000\) labels, which \(\lambda\) does the soft-SVM theorem tell you to use, and what expected 0-1 true risk does it certify? How many labels would you need to certify 0.15?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (hard versus soft).** Prove or refute: there is a \(\lambda > 0\) such that for every sample \(S\) of \(m > 1\) examples that is separable by homogeneous halfspaces, hard-SVM and soft-SVM with parameter \(\lambda\) return exactly the same weight vector.
- **Problem 2 (weak duality).** Prove that for any function \(f\) of two variables \(x \in \mathcal X\) and \(y \in \mathcal Y\),
\[ \min_{x \in \mathcal X} \max_{y \in \mathcal Y} f(x, y) \ge \max_{y \in \mathcal Y} \min_{x \in \mathcal X} f(x, y). \]
