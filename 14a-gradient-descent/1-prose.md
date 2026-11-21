# From Gradient Descent to SGD

## A filter that is always a day behind

The regularized filter from lesson 13a is the best thing the team has shipped. It is a halfspace over word counts, trained with the hinge loss plus a penalty on the size of the weight vector, and on paper it has everything you could want: a convex problem, a stable learner, a guarantee on its true risk. The archive it trains on holds two million labelled messages.

And yet the complaints are back. Every morning the spammers try something new, and every morning the filter is fooled by it until the evening. The retrain logs the manager pulled up at the end of lesson 13b make the reason plain: the exact solver for the regularized problem takes about six hours on the full archive, and the archive grows by thousands of labels a day. By the time a new filter ships, the junk it was trained to catch is already yesterday’s junk. The filter is always a day behind.

What would a faster learner even look like? Here is a thought that sounds almost too lazy to work. Instead of solving the whole optimization problem over two million messages, start with some weight vector, look at *one* message, nudge the weights a little so that message is handled better, and move on to the next message. Each nudge costs about as much as reading one message. Could a few thousand nudges, each based on a single message, produce a filter with a guarantee as good as the exact solver’s?

This lesson says yes, and proves it. The tool is **stochastic gradient descent**. We will build it in three steps: gradient descent, which walks downhill on a function using its slope; subgradients, which let us walk downhill on functions with creases, like the hinge loss; and finally the stochastic version, which replaces the true slope with a cheap, noisy one computed from a single message. The surprise at the end is that the noisy version does not just approximate the exact solver: it minimizes the true risk directly.

## Walking downhill

Let’s shrink the world so we can see every number. Pretend there are only six kinds of messages, each arriving equally often. Each is described by two scores built from word counts: \(x_1\), how strongly it talks about “free”, and \(x_2\), how strongly it talks about “meeting”. The counts are scaled so that every message has length \(\|\mathbf x\| \le 1\), where the double bars, as in lesson 9a, give a vector’s length, \(\|\mathbf x\| = \sqrt{x_1^2 + x_2^2}\) (message A below has length \(\sqrt{0.68} \approx 0.82\)). As in lesson 9a, labels are \(+1\) for junk and \(-1\) for keep.

| message | gist | \(x_1\) | \(x_2\) | label \(y\) |
|---|---|---|---|---|
| A | FREE prize, FREE shipping | 0.8 | 0.2 | junk, \(+1\) |
| B | free trial, book a meeting with sales | 0.6 | 0.6 | junk, \(+1\) |
| C | free free crypto | 0.9 | 0.4 | junk, \(+1\) |
| D | meeting notes | 0.1 | 0.9 | keep, \(-1\) |
| E | free coffee at the meeting | 0.4 | 0.8 | keep, \(-1\) |
| F | the newsletter: free webinar this week | 0.7 | 0.3 | keep, \(-1\) |

We will learn a homogeneous halfspace (one with no bias term, so its boundary passes through the origin) with weight vector \(\mathbf w = (w_1, w_2)\). Recall the pieces from lessons 9a and 12a. The inner product \(\langle \mathbf w, \mathbf x \rangle = w_1 x_1 + w_2 x_2\) is the filter’s score for a message, positive meaning junk, and multiplying it by the label gives the **signed score** \(y\langle \mathbf w, \mathbf x \rangle\), positive exactly when the filter is right. For this first walk we turn the signed score into a penalty with the logistic loss of lesson 9b, \(\ell(\mathbf w, (\mathbf x, y)) = \log(1 + e^{-y \langle \mathbf w, \mathbf x \rangle})\). Here \(e \approx 2.718\) is Euler’s number and \(\log\) is the natural logarithm: the penalty is large for a very negative signed score, exactly \(\log 2 \approx 0.693\) at a signed score of 0, and near 0 for a large positive one. We use it because it is smooth everywhere. With the six messages equally likely, the true risk, the expected loss on the next message to arrive, is simply the average loss over the six:

\[ f(\mathbf w) = L_{\mathcal D}(\mathbf w) = \frac16 \sum_{i=1}^6 \log\left(1 + e^{-y_i \langle \mathbf w, \mathbf x_i \rangle}\right). \]

Read it from the inside out. This is the one place we read every symbol slowly; later formulas reuse them.

- \(\mathbf x_i\) and \(y_i\) are message \(i\)’s scores and label, for \(i = 1\) (A) up to \(i = 6\) (F): the subscript says which message.
- \(\log(1 + e^{-y_i\langle \mathbf w, \mathbf x_i \rangle})\) is the logistic loss of \(\mathbf w\) on message \(i\).
- \(\sum_{i=1}^6\) adds that loss up for \(i = 1, 2, \dots, 6\), and \(\frac16\) turns the total into an average.
- \(f\) is short for \(L_{\mathcal D}\), the true risk; the subscript \(\mathcal D\) names the distribution of messages, here each kind with probability \(\frac16\).

In plain words: **the risk of a weight vector is its average logistic loss over the six kinds of message.** At \(\mathbf w = \mathbf 0\) every signed score is 0 and every loss is \(\log 2\), so \(f(\mathbf 0) = 0.693\). At \(\mathbf w = (1, -1)\), message A gets signed score \(0.8 - 0.2 = 0.6\) and loss \(\log(1 + e^{-0.6}) = 0.437\), and averaging all six gives 0.567:

```python
import math

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
loss = lambda w, x, y: math.log(1 + math.exp(-y * dot(w, x)))
f = lambda w: sum(loss(w, x, y) for x, y in S) / 6
w = (1.0, -1.0)
print([round(loss(w, x, y), 3) for x, y in S])
print(round(f((0.0, 0.0)), 3), round(f(w), 3))
```

It prints `[0.437, 0.693, 0.474, 0.371, 0.513, 0.913]`, then `0.693 0.567`. The newsletter F, a keep message full of “free”, costs the most.

Think of \(f\) as a landscape over the plane of weight vectors: every point \(\mathbf w\) has a height \(f(\mathbf w)\), and we are looking for the lowest point. We cannot see the whole landscape at once. What we can compute cheaply, at the point where we stand, is the slope.

Slope in which direction, though? On a landscape over a plane you can ask two simple questions. If I nudge only \(w_1\) a little, how fast does the height change? And if I nudge only \(w_2\)? At \(\mathbf w = \mathbf 0\), nudging \(w_1\) up by 0.001 lowers \(f\) by about 0.000092, a rate of \(-0.092\) per unit, and nudging \(w_2\) up raises it at a rate of \(0.067\). Those rates, side by side, are the slope we will use. Formally, the **gradient** of a differentiable function \(f: \mathbb R^d \to \mathbb R\) at \(\mathbf w\) is the vector of its partial derivatives,

\[ \nabla f(\mathbf w) = \left( \frac{\partial f(\mathbf w)}{\partial w_1}, \dots, \frac{\partial f(\mathbf w)}{\partial w_d} \right). \]

- \(f: \mathbb R^d \to \mathbb R\) says \(f\) takes a list of \(d\) real numbers (a weight vector; here \(d = 2\)) and returns one real number. Differentiable means it has no creases, so the rates exist.
- \(\frac{\partial f(\mathbf w)}{\partial w_j}\), the partial derivative, is the rate at which \(f\) changes when only the coordinate \(w_j\) moves; the curly \(\partial\) marks “partial”.
- \(\nabla\), read “grad”, stacks the \(d\) rates into one vector, and the dots stand for the coordinates in between.

In plain words: **the gradient lists, for each weight, how fast the risk rises when you nudge that weight alone.** The nudge itself, in code:

```python
import math

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in S) / 6
h, w = 1e-6, (0.0, 0.0)
partials = [(f((w[0] + h, w[1])) - f(w)) / h,
            (f((w[0], w[1] + h)) - f(w)) / h]
print([round(p, 3) for p in partials])
```

It prints `[-0.092, 0.067]`. The gradient points in the direction in which \(f\) increases fastest. So to go down, step the other way, and make the step proportional to the slope: on steep ground a long step, on flat ground a short one. A fixed number \(\eta > 0\) (the Greek letter eta), the **step size**, sets the proportion; you will also hear it called the learning rate. **Gradient descent** starts at \(\mathbf w^{(1)} = \mathbf 0\) and repeats one update:

\[ \mathbf w^{(t+1)} = \mathbf w^{(t)} - \eta \nabla f(\mathbf w^{(t)}). \]

The superscript \((t)\) is a step counter, not a power: \(\mathbf w^{(t)}\) is where you stand before step \(t\), and each \(\mathbf w^{(t)}\) is called an **iterate**. The gradient is taken where you stand, \(\eta\) times it is the step, and the minus sign sends it downhill. In plain words: **the next weight vector is the current one moved against the slope, by step size times slope.** With \(\eta = 0.5\), the first step goes from \((0, 0)\) to \((0, 0) - 0.5 \cdot (-0.092, 0.067) = (0.046, -0.0335)\).

After \(T\) steps the algorithm outputs the **average** of the iterates,

\[ \bar{\mathbf w} = \frac{1}{T} \sum_{t=1}^T \mathbf w^{(t)}. \]

The sum adds the iterates \(\mathbf w^{(1)}\) up to \(\mathbf w^{(T)}\), coordinate by coordinate, \(\frac1T\) divides by how many there are, and the bar over \(\mathbf w\) is the usual mark for an average. In plain words: **the output is the average position of the walk, not where it ended.** Two steps, with the gradients of the table below, and the average of the three iterates:

```python
grads = [(-0.092, 0.067), (-0.089, 0.067)]
eta, ws = 0.5, [(0.0, 0.0)]
for g in grads:
    w = ws[-1]
    ws.append((w[0] - eta * g[0], w[1] - eta * g[1]))
w_bar = tuple(sum(w[j] for w in ws) / len(ws) for j in (0, 1))
print(ws)
print([round(c, 4) for c in w_bar])
```

It prints `[(0.0, 0.0), (0.046, -0.0335), (0.0905, -0.067)]`, then `[0.0455, -0.0335]` (the table rounds the third iterate to 0.090).

Why the average and not the last iterate? Returning \(\mathbf w^{(T)}\), or the best iterate, are both reasonable, but the average is what makes the clean proofs below work, especially once the steps become noisy. Hold on to that.

There is a second way to see the update, and it explains *why* the gradient is the right direction. Near \(\mathbf w^{(t)}\), a differentiable \(f\) is well approximated by its first-order Taylor approximation, \(f(\mathbf w) \approx f(\mathbf w^{(t)}) + \langle \mathbf w - \mathbf w^{(t)}, \nabla f(\mathbf w^{(t)}) \rangle\), a tilted plane: today’s height, plus each rate in the gradient times how far you moved along that coordinate. Minimizing the plane alone is hopeless, since it goes down forever. But the approximation is only trustworthy near \(\mathbf w^{(t)}\), so let’s minimize the plane plus a penalty for wandering off:

\[ \mathbf w^{(t+1)} = \operatorname*{argmin}_{\mathbf w} \; \frac12 \|\mathbf w - \mathbf w^{(t)}\|^2 + \eta \left( f(\mathbf w^{(t)}) + \langle \mathbf w - \mathbf w^{(t)}, \nabla f(\mathbf w^{(t)}) \rangle \right). \]

- \(\operatorname{argmin}_{\mathbf w}\) means “the \(\mathbf w\) at which what follows is smallest”: the place, not the smallest value itself.
- \(\frac12\|\mathbf w - \mathbf w^{(t)}\|^2\) is half the squared distance from where you stand, the penalty for wandering.
- The bracket is the tilted plane, and \(\eta\) weighs it against the penalty.

In plain words: **the next iterate is the point that best trades going down the plane against staying close.** The objective is a quadratic in \(\mathbf w\). Its gradient with respect to \(\mathbf w\) is \((\mathbf w - \mathbf w^{(t)}) + \eta \nabla f(\mathbf w^{(t)})\), and setting that to zero gives exactly \(\mathbf w = \mathbf w^{(t)} - \eta \nabla f(\mathbf w^{(t)})\). So \(\eta\) is a dial between “trust the slope” and “stay where you are”. A brute-force search over a grid of 160,801 points around the origin agrees with the update’s first step:

```python
import itertools

w_t, g, eta = (0.0, 0.0), (-0.092, 0.067), 0.5
def objective(w):
    d = (w[0] - w_t[0], w[1] - w_t[1])
    plane = 0.693 + d[0] * g[0] + d[1] * g[1]
    return 0.5 * (d[0] ** 2 + d[1] ** 2) + eta * plane
grid = [k / 2000 for k in range(-200, 201)]
best = min(itertools.product(grid, grid), key=objective)
print(best, (w_t[0] - eta * g[0], w_t[1] - eta * g[1]))
```

It prints `(0.046, -0.0335) (0.046, -0.0335)`.

Now walk. The gradient of the logistic loss on one message is \(-y \mathbf x / (1 + e^{y \langle \mathbf w, \mathbf x \rangle})\): the vector \(-y\mathbf x\), shrunk by a factor between 0 and 1 that gets smaller as the signed score grows (at \(\mathbf w = \mathbf 0\) the factor is \(\frac12\)). The gradient \(\nabla f\) is the average of the six, and the `grad` line of the snippet below is exactly this formula. Take \(\eta = 0.5\). Before you look at the table, guess: will the first step move \(w_1\) up or down, and \(w_2\)?

| \(t\) | \(\mathbf w^{(t)}\) | \(\nabla f(\mathbf w^{(t)})\) | \(f(\mathbf w^{(t)})\) |
|---|---|---|---|
| 1 | (0, 0) | (−0.092, 0.067) | 0.693 |
| 2 | (0.046, −0.033) | (−0.089, 0.067) | 0.687 |
| 3 | (0.090, −0.067) | (−0.087, 0.066) | 0.681 |
| 4 | (0.134, −0.100) | (−0.084, 0.066) | 0.675 |
| 5 | (0.176, −0.133) | (−0.082, 0.066) | 0.669 |
| 6 | (0.217, −0.166) | (−0.080, 0.066) | 0.664 |
| 7 | (0.257, −0.199) | (−0.078, 0.065) | 0.658 |

The gradient’s first coordinate is negative, so \(w_1\) goes up: “free” becomes evidence for junk. The second is positive, so \(w_2\) goes down: “meeting” becomes evidence for keep. That is the story the six messages tell, and gradient descent found it without being told. Notice also how timid the walk is. The landscape is nearly flat near the origin, so steps of size \(\eta \|\nabla f\| \approx 0.06\) make slow progress.

[[diagram:descent]]

How close does the average get to the bottom? Let’s run it for longer. The star marks the best weight vector, \(\mathbf w^\star\), so the lowest value of \(f\) is \(f(\mathbf w^\star)\); it can be found by running gradient descent for a very long time, and the snippet does just that. For reasons the next section explains, we set \(\eta = 5/\sqrt T\) when we run for \(T\) steps.

```python
import math

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in S) / len(S)
grad = lambda w: [sum(-y * x[j] / (1 + math.exp(y * dot(w, x)))
                      for x, y in S) / len(S) for j in (0, 1)]

def gd(eta, T):
    w, total = [0.0, 0.0], [0.0, 0.0]
    for _ in range(T):
        total = [a + b for a, b in zip(total, w)]
        w = [a - eta * b for a, b in zip(w, grad(w))]
    return [a / T for a in total], w

w_star = gd(1.0, 20000)[1]
f_star = f(w_star)
print([round(c, 3) for c in w_star], round(f_star, 3))
for T in (100, 400, 2500):
    w_bar = gd(5 / math.sqrt(T), T)[0]
    print(T, round(f(w_bar) - f_star, 3), 5 / math.sqrt(T))
```

It prints `[3.102, -3.393] 0.474`, the bottom \(\mathbf w^\star\) and its height \(f(\mathbf w^\star)\), then `100 0.07 0.5`, `400 0.032 0.25` and `2500 0.007 0.1`. The middle column is the gap \(f(\bar{\mathbf w}) - f(\mathbf w^\star)\), and it shrinks as \(T\) grows. The last column is a number we have not explained yet, and the gap always sits below it. That last column is a guarantee, and proving it is the next job.

Before the proof, play with the walk yourself. In the lab below you set \(\eta\) and \(T\) and watch the path of iterates on the landscape of \(f\), with the average marked. Try a tiny \(\eta\), then a huge one. Which goes wrong, and how?

[[lab]]

## How far from the bottom?

You will have seen two failures in the lab. With a tiny step size the walk crawls and never gets anywhere in \(T\) steps. With a huge one it overshoots, zigzags across the valley, and the average lands somewhere off to the side. A good \(\eta\) sits in between, and the right value depends on \(T\). Let’s find it by proving a bound.

We need three facts from lesson 12a. A function \(f\) is **convex** when \(f(\alpha \mathbf u + (1-\alpha)\mathbf w) \le \alpha f(\mathbf u) + (1-\alpha) f(\mathbf w)\) for all \(\mathbf u, \mathbf w\) and \(\alpha \in [0, 1]\): between any two points the graph sits on or below the straight chord joining them, and \(\alpha\), a number from 0 to 1, says where along the chord you are. A differentiable convex function lies above each of its tangent planes, the **tangent inequality**:

\[ f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \nabla f(\mathbf w) \rangle \quad \text{for all } \mathbf u, \mathbf w. \]

The right side is the tilted plane from the last section, built at \(\mathbf w\) and evaluated at \(\mathbf u\). In plain words: **a convex function never dips below any of its tangent planes.** At \(\mathbf w = \mathbf 0\) the plane predicts \(0.693 + \langle (1, -1), (-0.092, 0.067) \rangle = 0.534\) for \(\mathbf u = (1, -1)\), and the true height there is 0.567, above it.

And \(f\) is **\(\rho\)-Lipschitz** when \(|f(\mathbf u) - f(\mathbf w)| \le \rho \|\mathbf u - \mathbf w\|\) for all \(\mathbf u, \mathbf w\). The single bars are absolute value, so the left side is how far apart two heights are; \(\rho\) (rho) caps that at \(\rho\) times the distance between the points. In plain words: **the function’s slope is never steeper than \(\rho\).** Our \(f\) is 1-Lipschitz, as we will see, and indeed its heights 0.693 at \(\mathbf 0\) and 0.567 at \((1, -1)\) differ by 0.126, well under 1 times the distance \(\sqrt 2 \approx 1.414\) between the points.

Fix any vector \(\mathbf w^\star\) (think of it as the minimizer, though the argument works for every \(\mathbf w^\star\)) and let \(B \ge \|\mathbf w^\star\|\). We want to bound \(f(\bar{\mathbf w}) - f(\mathbf w^\star)\).

**Step 1: from the average to the iterates.** Convexity extends from two points to \(T\) points: \(f\left(\frac1T \sum_t \mathbf w^{(t)}\right) \le \frac1T \sum_t f(\mathbf w^{(t)})\). This is **Jensen’s inequality** for finite averages, and it follows from the definition by induction on \(T\): write the average of \(T\) points as \(\frac{T-1}{T}\) times the average of the first \(T - 1\) plus \(\frac1T\) times the last, apply convexity with \(\alpha = \frac{T-1}{T}\), then the induction hypothesis. Subtracting \(f(\mathbf w^\star)\) from both sides,

\[ f(\bar{\mathbf w}) - f(\mathbf w^\star) \le \frac1T \sum_{t=1}^T \left( f(\mathbf w^{(t)}) - f(\mathbf w^\star) \right). \]

Each term of the sum is how far one iterate sits above the bottom, and \(\frac1T\) averages them. In plain words: **the average point is no further above the bottom than the iterates are on average.** For the three iterates computed above, \(f(\bar{\mathbf w}) = 0.68680\) and the average height is \(0.68684\). This snippet checks the tangent inequality and Jensen on those numbers:

```python
import math

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in S) / 6
w, g, u = (0.0, 0.0), (-0.092, 0.067), (1.0, -1.0)
plane = f(w) + dot((u[0] - w[0], u[1] - w[1]), g)
print(round(f(u), 3), ">=", round(plane, 3))
ws = [(0.0, 0.0), (0.046, -0.0335), (0.0905, -0.067)]
w_bar = tuple(sum(v[j] for v in ws) / 3 for j in (0, 1))
print(f"{f(w_bar):.5f} <= {sum(map(f, ws)) / 3:.5f}")
```

It prints `0.567 >= 0.534`, then `0.68680 <= 0.68684`. The landscape is nearly flat there, so Jensen wins by a hair.

**Step 2: from function values to inner products.** The tangent inequality at \(\mathbf w^{(t)}\), with \(\mathbf u = \mathbf w^\star\), rearranges to

\[ f(\mathbf w^{(t)}) - f(\mathbf w^\star) \le \langle \mathbf w^{(t)} - \mathbf w^\star, \nabla f(\mathbf w^{(t)}) \rangle. \]

In plain words: **an iterate’s height above the target is at most its slope measured along the arrow from the target to the iterate.** At \(\mathbf w^{(1)} = \mathbf 0\), with the bottom \(\mathbf w^\star \approx (3.102, -3.393)\) and \(f(\mathbf w^\star) \approx 0.474\) printed by the snippet above, the left side is \(0.693 - 0.474 = 0.219\) and the right side is \(\langle -\mathbf w^\star, (-0.092, 0.067) \rangle = 0.513\).

Combining the two steps,

\[ f(\bar{\mathbf w}) - f(\mathbf w^\star) \le \frac1T \sum_{t=1}^T \langle \mathbf w^{(t)} - \mathbf w^\star, \nabla f(\mathbf w^{(t)}) \rangle. \]

The right side no longer mentions \(f\) except through the vectors we stepped along. Everything now rests on one lemma about *any* sequence of steps, and it is the heart of this lesson.

**Lemma (the step lemma).** Let \(\mathbf v_1, \dots, \mathbf v_T\) be any sequence of vectors. Any algorithm that starts at \(\mathbf w^{(1)} = \mathbf 0\) and updates \(\mathbf w^{(t+1)} = \mathbf w^{(t)} - \eta \mathbf v_t\) satisfies, for every \(\mathbf w^\star\),

\[ \sum_{t=1}^T \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

- \(\mathbf v_t\) is the direction of step \(t\), whatever it is, and \(\|\mathbf v_t\|^2\) its squared length.
- \(\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle\) is large when the step, which goes along \(-\mathbf v_t\), heads towards \(\mathbf w^\star\): call it the progress of step \(t\).
- On the right, \(\|\mathbf w^\star\|^2/(2\eta)\) is the squared distance from the start \(\mathbf 0\) to the target, and \(\frac\eta2\sum_t\|\mathbf v_t\|^2\) charges for the length of every step.

In plain words: **whatever directions you step along, the total progress towards any target is at most its squared distance from the start over \(2\eta\), plus \(\frac\eta2\) times the total squared step length.** One step of our walk, with the same \(\mathbf w^\star\): the left side is the 0.513 of Step 2, and the right side is \(21.14/1 + 0.25 \cdot 0.013 \approx 21.14\). True, with room to spare.

In particular, if \(\|\mathbf v_t\| \le \rho\) for every \(t\), \(\|\mathbf w^\star\| \le B\), and \(\eta = \sqrt{B^2 / (\rho^2 T)} = B/(\rho \sqrt T)\), then

\[ \frac1T \sum_{t=1}^T \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \le \frac{B\rho}{\sqrt T}. \]

Here \(B\) caps the target’s length and \(\rho\) every step’s length. In plain words: **the average progress per step is at most \(B\rho/\sqrt T\).** With \(B = 5\), \(\rho = 1\) and \(T = 100\) that is \(5/10 = 0.5\), the 0.5 in the first snippet’s last column.

*Proof.* The trick is to complete a square, so that each term becomes a difference of squared distances to \(\mathbf w^\star\). For any vectors \(\mathbf a, \mathbf b\), expanding \(\|\mathbf a - \mathbf b\|^2 = \|\mathbf a\|^2 - 2\langle \mathbf a, \mathbf b \rangle + \|\mathbf b\|^2\) gives \(\langle \mathbf a, \mathbf b \rangle = \frac12 \left( \|\mathbf a\|^2 + \|\mathbf b\|^2 - \|\mathbf a - \mathbf b\|^2 \right)\): an inner product written with lengths only (for \(\mathbf a = (1, 0)\) and \(\mathbf b = (1, 1)\), both sides are 1). Apply it with \(\mathbf a = \mathbf w^{(t)} - \mathbf w^\star\) and \(\mathbf b = \eta \mathbf v_t\), after pulling out a factor \(1/\eta\):

\[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle = \frac1\eta \langle \mathbf w^{(t)} - \mathbf w^\star, \eta \mathbf v_t \rangle = \frac{1}{2\eta} \left( \|\mathbf w^{(t)} - \mathbf w^\star\|^2 + \eta^2 \|\mathbf v_t\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star - \eta \mathbf v_t\|^2 \right). \]

By the update rule, \(\mathbf w^{(t)} - \eta \mathbf v_t = \mathbf w^{(t+1)}\), so the last norm is \(\|\mathbf w^{(t+1)} - \mathbf w^\star\|^2\):

\[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle = \frac{1}{2\eta} \left( \|\mathbf w^{(t)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 \right) + \frac\eta2 \|\mathbf v_t\|^2. \]

This is an equality, true for every \(t\): **a step’s progress is the drop in squared distance to the target, over \(2\eta\), plus a charge for the step’s length.** Now sum over \(t = 1, \dots, T\). The first part is a **telescoping sum**: each squared distance appears once with a plus sign and once with a minus sign, except the very first and the very last, so it collapses to \(\|\mathbf w^{(1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(T+1)} - \mathbf w^\star\|^2\). With three steps and squared distances 9, 7, 6 and 4, the drops \((9 - 7) + (7 - 6) + (6 - 4)\) add up to \(9 - 4 = 5\). Hence

\[ \sum_{t=1}^T \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle = \frac{1}{2\eta} \left( \|\mathbf w^{(1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(T+1)} - \mathbf w^\star\|^2 \right) + \frac\eta2 \sum_{t=1}^T \|\mathbf v_t\|^2. \]

Dropping the term \(-\|\mathbf w^{(T+1)} - \mathbf w^\star\|^2\), which is never positive, can only make the right side larger, and \(\mathbf w^{(1)} = \mathbf 0\) turns \(\|\mathbf w^{(1)} - \mathbf w^\star\|^2\) into \(\|\mathbf w^\star\|^2\). That is the first claim.

For the second, bound \(\|\mathbf w^\star\|^2 \le B^2\) and each \(\|\mathbf v_t\|^2 \le \rho^2\), so the right side is at most \(B^2/(2\eta) + \eta T \rho^2 / 2\). Divide by \(T\) to get \(B^2/(2\eta T) + \eta \rho^2/2\), and plug in \(\eta = B/(\rho\sqrt T)\): the first term becomes \(B\rho/(2\sqrt T)\) and so does the second, which add up to \(B\rho/\sqrt T\). \(\blacksquare\)

[[diagram:telescope]]

Notice what the lemma does *not* assume. The vectors \(\mathbf v_t\) can be anything: gradients, garbage, or random draws. That generality is about to pay for everything. Here it is with fifty random vectors in three dimensions: the equality before we drop the last distance, and the bound after.

```python
import random

rng, eta, T, d = random.Random(0), 0.3, 50, 3
sq = lambda u: sum(c * c for c in u)
sub = lambda u, v: [a - b for a, b in zip(u, v)]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w_star = [rng.uniform(-2, 2) for _ in range(d)]
vs = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(T)]
w, lhs = [0.0] * d, 0.0
for v in vs:
    lhs += dot(sub(w, w_star), v)
    w = [a - eta * b for a, b in zip(w, v)]
exact = ((sq(w_star) - sq(sub(w, w_star))) / (2 * eta)
         + eta / 2 * sum(sq(v) for v in vs))
bound = sq(w_star) / (2 * eta) + eta / 2 * sum(sq(v) for v in vs)
print(round(lhs, 6), round(exact, 6), round(bound, 3))
```

It prints `26.350282 26.350282 30.413`: the telescoped identity holds to the last digit, and the bound sits above it.

To apply the lemma to gradient descent, take \(\mathbf v_t = \nabla f(\mathbf w^{(t)})\). We need \(\|\nabla f(\mathbf w^{(t)})\| \le \rho\), and a little later we will prove that a convex \(\rho\)-Lipschitz function has exactly that property. Granting it for a moment:

**Corollary (gradient descent on convex-Lipschitz functions).** Let \(f\) be convex and \(\rho\)-Lipschitz, and let \(\mathbf w^\star \in \operatorname{argmin}_{\{\mathbf w : \|\mathbf w\| \le B\}} f(\mathbf w)\). If gradient descent runs on \(f\) for \(T\) steps with \(\eta = B/(\rho\sqrt T)\), then

\[ f(\bar{\mathbf w}) - f(\mathbf w^\star) \le \frac{B\rho}{\sqrt T}. \]

Consequently, for every \(\varepsilon > 0\), to reach \(f(\bar{\mathbf w}) - f(\mathbf w^\star) \le \varepsilon\) it suffices to run \(T \ge B^2\rho^2/\varepsilon^2\) steps.

The braces \(\{\mathbf w : \|\mathbf w\| \le B\}\) read “the weight vectors \(\mathbf w\) such that \(\|\mathbf w\| \le B\)”, the ball of radius \(B\); the argmin under them is the set of lowest points of \(f\) in that ball, and \(\in\) says \(\mathbf w^\star\) is one of them. The accuracy \(\varepsilon\) (epsilon) is how close to the bottom we ask to be. In plain words: **after \(T\) steps the average sits at most \(B\rho/\sqrt T\) above the best point of the ball, so \(B^2\rho^2/\varepsilon^2\) steps buy accuracy \(\varepsilon\).** With \(B = 5\), \(\rho = 1\) and \(\varepsilon = 0.1\), that is \(25/0.01 = 2{,}500\) steps, the last row of the first snippet.

*Proof.* Steps 1 and 2 bound the gap by the average of \(\langle \mathbf w^{(t)} - \mathbf w^\star, \nabla f(\mathbf w^{(t)}) \rangle\), and the step lemma bounds that by \(B\rho/\sqrt T\). For the second sentence, \(B\rho/\sqrt T \le \varepsilon\) is the same as \(T \ge B^2\rho^2/\varepsilon^2\). \(\blacksquare\)

Now the mystery column in the first snippet is explained. Every message in our tiny world has \(\|\mathbf x\| \le 1\), and the logistic loss on one message has gradient of norm \(\|\mathbf x\| / (1 + e^{y\langle \mathbf w, \mathbf x \rangle}) < \|\mathbf x\| \le 1\), so \(f\) is 1-Lipschitz. The minimizer has norm about 4.6, so \(B = 5\) covers it. The corollary promises a gap of at most \(5/\sqrt T\): 0.5, 0.25 and 0.1 for our three runs, against the measured 0.07, 0.032 and 0.007. It is a worst-case promise, and on this gentle landscape it has plenty of room.

## When the floor has a crease

The filter in production does not use the logistic loss. It uses the **hinge loss** of lesson 12a, \(\ell(\mathbf w, (\mathbf x, y)) = \max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\), where \(\max\{a, b\}\) is the larger of the two numbers (message A at \(\mathbf w = \mathbf 0\) has signed score 0 and loss \(\max\{0, 1\} = 1\)). It is zero once a message is on the right side with a signed score of at least 1, and growing linearly otherwise. Its graph has a crease where \(y \langle \mathbf w, \mathbf x \rangle = 1\), and at the crease there is no gradient. Gradient descent, as written, does not know what to do there.

What we used about the gradient in the analysis was not that it is a derivative. We used one inequality: the tangent plane lies below the function. Is there something else, at a crease, that lies below the function and touches it?

Look at the simplest crease, \(f(x) = |x|\) at \(x = 0\). The line \(y = 0.5x\) passes through the bottom of the V and stays below it everywhere. So does \(y = -0.3x\), and so does \(y = x\). Any line through the origin with slope between \(-1\) and \(1\) works. A slope of \(1.1\) does not: for small positive \(x\), the line climbs above the V. So at the crease there is not one supporting line but a whole fan of them.

[[diagram:subgradient]]

This is in fact a characterization of convexity, a standard result of convex analysis that we state without proof. It needs one recap: a convex set (lesson 12a) contains the segment between any two of its points, and an open one leaves every point a little room around it inside the set, as the whole plane does.

**Lemma (supporting planes).** Let \(A\) be an open convex set. A function \(f: A \to \mathbb R\) is convex if and only if for every \(\mathbf w \in A\) there is a vector \(\mathbf v\) with

\[ f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \mathbf v \rangle \quad \text{for all } \mathbf u \in A. \]

This is the tangent inequality with \(\mathbf v\) in place of the gradient, and \(\mathbf u \in A\) reads “\(\mathbf u\) in \(A\)”. In plain words: **a function is convex exactly when, at every point, some plane through its graph stays below it everywhere.** For \(|x|\) at \(w = 0\) with \(v = 0.5\) and \(u = 2\): \(|2| = 2 \ge 0 + 2 \cdot 0.5 = 1\). The fan snippet below checks this inequality for six slopes at once.

**Definition (subgradient).** A vector \(\mathbf v\) satisfying that inequality is a **subgradient** of \(f\) at \(\mathbf w\). The set of all subgradients of \(f\) at \(\mathbf w\) is written \(\partial f(\mathbf w)\) (it is also called the differential set, or the subdifferential). The curly \(\partial\) is back, now with no fraction: read \(\partial f(\mathbf w)\) as “the slopes of all supporting planes at \(\mathbf w\)”. For \(|x|\) at 0 it is every slope from \(-1\) to \(1\).

For a function of one variable, a subgradient at \(w\) is the slope of a line that touches \(f\) at \(w\) and is nowhere above \(f\). Where a function is smooth, nothing new happens:

**Claim (smooth points).** If \(f\) is convex and differentiable at \(\mathbf w\), then \(\partial f(\mathbf w) = \{\nabla f(\mathbf w)\}\), the set whose only member is the gradient.

*Proof.* The gradient is a subgradient: that is the tangent inequality. For the converse, let \(\mathbf v \in \partial f(\mathbf w)\), pick any direction \(\mathbf e\), and use the subgradient inequality at \(\mathbf u = \mathbf w + s\mathbf e\) for small \(s > 0\): \(f(\mathbf w + s\mathbf e) - f(\mathbf w) \ge s \langle \mathbf e, \mathbf v \rangle\). Divide by \(s\) and let \(s \to 0\). The left side tends to the directional derivative \(\langle \mathbf e, \nabla f(\mathbf w) \rangle\), so \(\langle \mathbf e, \nabla f(\mathbf w) - \mathbf v \rangle \ge 0\). This holds for every \(\mathbf e\), including \(-\mathbf e\), so \(\langle \mathbf e, \nabla f(\mathbf w) - \mathbf v \rangle = 0\) for every \(\mathbf e\). Taking \(\mathbf e = \nabla f(\mathbf w) - \mathbf v\) gives \(\|\nabla f(\mathbf w) - \mathbf v\|^2 = 0\). \(\blacksquare\)

So for \(f(x) = |x|\), the claim settles every point except the crease, and the fan of lines settles the crease:

\[ \partial f(x) = \begin{cases} \{1\} & x > 0, \\ \{-1\} & x < 0, \\ [-1, 1] & x = 0. \end{cases} \]

The big brace lists cases: each line reads “this set, if this condition holds”, and \([-1, 1]\) is the interval of all numbers from \(-1\) to \(1\). In plain words: **the slope of \(|x|\) is 1 on the right, \(-1\) on the left, and anything from \(-1\) to 1 at the bottom of the V.** Here is the fan checked by brute force, on a grid of 601 points:

```python
grid = [k / 100 for k in range(-300, 301)]
for v in (-1.0, -0.5, 0.0, 0.7, 1.0, 1.1):
    ok = all(abs(u) >= 0 + v * (u - 0) for u in grid)
    print(v, ok)
```

It prints `True` for \(-1, -0.5, 0, 0.7\) and \(1\), and `False` for \(1.1\).

In practice we never need the whole set \(\partial f(\mathbf w)\); one member is enough to take a step. For a maximum of smooth convex pieces, there is a simple recipe: take the gradient of whichever piece is on top.

**Claim (subgradient of a maximum).** Let \(g(\mathbf w) = \max_{i \in [r]} g_i(\mathbf w)\) for convex differentiable functions \(g_1, \dots, g_r\). Given \(\mathbf w\), let \(j \in \operatorname{argmax}_i g_i(\mathbf w)\). Then \(\nabla g_j(\mathbf w) \in \partial g(\mathbf w)\).

Here \([r]\) is the list \(\{1, \dots, r\}\), \(\max_{i \in [r]}\) takes the largest of the \(r\) values, and \(\operatorname{argmax}_i\) is the index of a piece that reaches it. In plain words: **the gradient of whichever piece is on top is a subgradient of the maximum.** For message A at \(\mathbf w = \mathbf 0\), the hinge’s two pieces are 0 and 1, so the second is on top.

*Proof.* Since \(g_j\) is convex and differentiable, its tangent plane lies below it: \(g_j(\mathbf u) \ge g_j(\mathbf w) + \langle \mathbf u - \mathbf w, \nabla g_j(\mathbf w) \rangle\) for all \(\mathbf u\). Now replace both ends by \(g\). On the left, \(g(\mathbf u) \ge g_j(\mathbf u)\) because \(g\) is the maximum. On the right, \(g_j(\mathbf w) = g(\mathbf w)\) because \(j\) is the piece on top at \(\mathbf w\). So \(g(\mathbf u) \ge g(\mathbf w) + \langle \mathbf u - \mathbf w, \nabla g_j(\mathbf w) \rangle\) for all \(\mathbf u\). \(\blacksquare\)

The hinge loss on one message is the maximum of two affine pieces in \(\mathbf w\): the constant \(0\), with gradient \(\mathbf 0\), and \(1 - y\langle \mathbf w, \mathbf x \rangle\), with gradient \(-y\mathbf x\). Pick the piece on top (and at the crease, where they tie, either one will do):

\[ \mathbf v = \begin{cases} \mathbf 0 & \text{if } 1 - y\langle \mathbf w, \mathbf x \rangle \le 0, \\ -y\mathbf x & \text{if } 1 - y\langle \mathbf w, \mathbf x \rangle > 0. \end{cases} \]

In plain words: **if the message is already handled with a signed score of at least 1, the subgradient is zero; otherwise it is \(-y\mathbf x\), so a step against it pushes the weights towards \(y\mathbf x\).** That should look familiar. It is the Perceptron’s update from lesson 9a, with a stricter idea of “handled”. At \(\mathbf w = \mathbf 0\) every signed score is 0, so junk message B gives \(\mathbf v = -(0.6, 0.6)\); at \(\mathbf w = (5, 0)\), message A has signed score 4 and gives \(\mathbf 0\). The snippet also averages the six subgradients at \(\mathbf 0\) and compares that average with the slope of the hinge risk measured by nudging each weight; keep the second line in mind for “One message per step”:

```python
S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
L_D = lambda w: sum(max(0, 1 - y * dot(w, x)) for x, y in S) / 6

def v(w, x, y):
    if 1 - y * dot(w, x) <= 0:
        return (0.0, 0.0)
    return (-y * x[0], -y * x[1])

w, h = (0.0, 0.0), 1e-6
print(v((5.0, 0.0), *S[0]), [v(w, x, y) for x, y in S[:2]])
mean = [round(sum(v(w, x, y)[j] for x, y in S) / 6, 4)
        for j in (0, 1)]
slope = [round((L_D((h, 0)) - L_D(w)) / h, 4),
         round((L_D((0, h)) - L_D(w)) / h, 4)]
print(mean, slope)
```

It prints `(0.0, 0.0) [(-0.8, -0.2), (-0.6, -0.6)]`, then `[-0.1833, 0.1333] [-0.1833, 0.1333]`.

## Lipschitz means bounded subgradients

The gradient descent corollary needed \(\|\mathbf v_t\| \le \rho\). Here is the promised fact, now for subgradients, and it goes both ways. Think of \(|x|\) again: it is 1-Lipschitz, and its subgradients, from \(-1\) to 1, are never longer than 1. A convex function with a subgradient of length 3 somewhere would have to climb at rate 3 along it.

**Lemma (Lipschitz and subgradients).** Let \(A\) be an open convex set and \(f: A \to \mathbb R\) convex. Then \(f\) is \(\rho\)-Lipschitz over \(A\) if and only if \(\|\mathbf v\| \le \rho\) for every \(\mathbf w \in A\) and every \(\mathbf v \in \partial f(\mathbf w)\).

*Proof.* First, suppose every subgradient has norm at most \(\rho\). Take \(\mathbf w, \mathbf u \in A\) and \(\mathbf v \in \partial f(\mathbf w)\). The subgradient inequality, rearranged, says \(f(\mathbf w) - f(\mathbf u) \le \langle \mathbf v, \mathbf w - \mathbf u \rangle\). The Cauchy-Schwarz inequality of lesson 9a, \(\langle \mathbf a, \mathbf b \rangle \le \|\mathbf a\| \|\mathbf b\|\), bounds the inner product by \(\|\mathbf v\| \|\mathbf w - \mathbf u\| \le \rho \|\mathbf w - \mathbf u\|\). The same argument with the roles of \(\mathbf w\) and \(\mathbf u\) swapped (using a subgradient at \(\mathbf u\)) gives \(f(\mathbf u) - f(\mathbf w) \le \rho \|\mathbf w - \mathbf u\|\). Together, \(|f(\mathbf w) - f(\mathbf u)| \le \rho\|\mathbf w - \mathbf u\|\).

Now suppose \(f\) is \(\rho\)-Lipschitz, and take \(\mathbf w \in A\) and \(\mathbf v \in \partial f(\mathbf w)\) with \(\mathbf v \ne \mathbf 0\) (the zero vector has norm \(0 \le \rho\) anyway). Because \(A\) is open, there is a small radius \(r > 0\) such that \(\mathbf u = \mathbf w + r \mathbf v / \|\mathbf v\|\) is still in \(A\). This \(\mathbf u\) sits at distance \(\|\mathbf u - \mathbf w\| = r\) from \(\mathbf w\), in the direction of \(\mathbf v\), so \(\langle \mathbf u - \mathbf w, \mathbf v \rangle = r \|\mathbf v\|\). The subgradient inequality gives \(f(\mathbf u) - f(\mathbf w) \ge r \|\mathbf v\|\), and Lipschitzness gives \(f(\mathbf u) - f(\mathbf w) \le \rho r\). So \(r \|\mathbf v\| \le \rho r\), and dividing by \(r > 0\), \(\|\mathbf v\| \le \rho\). \(\blacksquare\)

For the hinge loss, the subgradient is \(\mathbf 0\) or \(-y\mathbf x\), whose norm is \(\|\mathbf x\|\). So the hinge loss on a message with \(\|\mathbf x\| \le \rho\) is \(\rho\)-Lipschitz in \(\mathbf w\), and its gradient-or-subgradient never has norm above \(\rho\). Both facts, checked at a hundred thousand random pairs of points for a keep message with \(\|\mathbf x\| = 1\):

```python
import math
import random

rng, x, y = random.Random(0), (0.6, 0.8), -1
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
loss = lambda w: max(0.0, 1 - y * dot(w, x))

def subgradient(w):
    if 1 - y * dot(w, x) <= 0:
        return (0.0, 0.0)
    return (-y * x[0], -y * x[1])

gaps, ratios = [], []
for _ in range(100000):
    w = (rng.uniform(-3, 3), rng.uniform(-3, 3))
    u = (rng.uniform(-3, 3), rng.uniform(-3, 3))
    d = (u[0] - w[0], u[1] - w[1])
    gaps.append(loss(u) - loss(w) - dot(d, subgradient(w)))
    ratios.append(abs(loss(u) - loss(w)) / math.hypot(*d))
print(min(gaps) > -1e-12, round(max(ratios), 4))
```

It prints `True 1.0`: the subgradient’s supporting plane never pokes above the loss, and the steepest slope found is exactly \(\|\mathbf x\| = 1\).

That settles **subgradient descent**: run gradient descent, but at each step use any subgradient \(\mathbf v_t \in \partial f(\mathbf w^{(t)})\) instead of the gradient. The analysis does not change by a single symbol. Step 2 used only the inequality \(f(\mathbf w^{(t)}) - f(\mathbf w^\star) \le \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle\), which is the subgradient definition with \(\mathbf u = \mathbf w^\star\). The step lemma holds for any vectors. And the Lipschitz lemma supplies \(\|\mathbf v_t\| \le \rho\). So the corollary, \(f(\bar{\mathbf w}) - f(\mathbf w^\star) \le B\rho/\sqrt T\), holds for every convex \(\rho\)-Lipschitz function, creases and all.

## One message per step

We can now minimize the hinge risk by subgradient descent. But look at what one step costs on the real archive. Recall the empirical risk \(L_S(\mathbf w) = \frac1m \sum_i \ell(\mathbf w, z_i)\): the average loss over the \(m\) labelled messages \(z_i = (\mathbf x_i, y_i)\) of the training sequence \(S\). Its subgradient is the average of the \(m\) per-message subgradients, so every step reads all two million messages. That is the six-hour solver all over again, one full pass per step.

Here is the lazy idea from the opening, made precise. What if, instead of the exact subgradient, we stepped along a *random* vector that is only right **on average**?

**Stochastic gradient descent (SGD)** for minimizing \(f(\mathbf w)\):

```
SGD(f, η, T):
  w(1) ← 0
  for t = 1, 2, ..., T:
    choose v_t at random so that
      E[v_t | v_1, ..., v_{t-1}] ∈ ∂f(w(t))
    w(t+1) ← w(t) − η v_t
  return w̄ = (1/T) Σ_{t=1..T} w(t)
```

The only requirement is the line with \(\mathbb E\), the conditional expectation. Read it piece by piece:

- \(\mathbb E[\cdot]\) is the expected value: the average of a random quantity over its possible outcomes, each weighted by its probability.
- The bar \(\mid\) reads “given”: average only over the random choice made at step \(t\), with the earlier choices \(\mathbf v_1, \dots, \mathbf v_{t-1}\) held fixed. Fixing them fixes \(\mathbf w^{(t)}\).
- \(\in \partial f(\mathbf w^{(t)})\) says that average must be one of the subgradients at the current iterate.

In plain words: **averaged over this step’s random choice, with the past held fixed, the step direction must be a subgradient at the current point.** Each individual \(\mathbf v_t\) may point quite the wrong way.

Where would such a vector come from? In our tiny world the answer is almost embarrassing. The true risk is the average hinge loss over six equally likely messages. Pick one of the six at random, uniformly, and take the subgradient of the hinge loss on that message alone. On average over the pick, that is the average of the six subgradients, which is a subgradient of the true risk. You have already seen it happen: at \(\mathbf w = \mathbf 0\), the second line of the hinge snippet printed the average of the six as \((-0.1833, 0.1333)\), the same as the measured slope of the risk. (We prove this properly in a moment.)

Let’s walk it, one message per step, with \(\eta = 0.5\). Each row shows the iterate before the step, the message drawn, its signed score \(y\langle \mathbf w, \mathbf x \rangle\), and the subgradient. Before reading the table: at step 1, from \(\mathbf w = \mathbf 0\), every signed score is 0. Will the first step move the weights, whatever message is drawn?

| \(t\) | \(\mathbf w^{(t)}\) | drawn | \(y\langle \mathbf w^{(t)}, \mathbf x \rangle\) | \(\mathbf v_t\) |
|---|---|---|---|---|
| 1 | (0, 0) | B, junk | 0 | (−0.6, −0.6) |
| 2 | (0.3, 0.3) | E, keep | −0.36 | (0.4, 0.8) |
| 3 | (0.1, −0.1) | A, junk | 0.06 | (−0.8, −0.2) |
| 4 | (0.5, 0) | C, junk | 0.45 | (−0.9, −0.4) |
| 5 | (0.95, 0.2) | A, junk | 0.8 | (−0.8, −0.2) |
| 6 | (1.35, 0.3) | D, keep | −0.405 | (0.1, 0.9) |
| 7 | (1.3, −0.15) | D, keep | 0.005 | (0.1, 0.9) |
| 8 | (1.25, −0.6) | D, keep | 0.415 | (0.1, 0.9) |

and \(\mathbf w^{(9)} = (1.2, -1.05)\). Yes, the first step always moves: a signed score of 0 is below 1, so the hinge loss is active. Look at step 2. The junk message B pushed both weights up, including the weight on “meeting”, which is the wrong lesson; the keep message E promptly pushed “meeting” back down. Every step is a noisy opinion from one message, and the noise partly cancels. Step 5 is interesting too: A was already on the right side, with signed score 0.8, but not by a margin of 1, so it still pulled. Steps 6 to 8 drew D three times in a row, which is luck of the draw, and each time D was handled a little better.

[[diagram:sgd]]

So the path zigzags. Does the average still land near the bottom? The smallest hinge risk in this world is \(163/336 \approx 0.4851\), at \(\mathbf w^\star = (15/7, -65/28)\), found by checking every point where two of the six creases cross (the hinge risk is piecewise linear, so its minimum sits at such a corner). Run SGD a thousand times for each \(T\), with \(\eta = 5/\sqrt T\), and average the gap:

```python
import math
import random

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
L_D = lambda w: sum(max(0, 1 - y * dot(w, x)) for x, y in S) / 6

def sgd(T, rng, eta):
    w, ws = (0.0, 0.0), []
    for _ in range(T):
        ws.append(w)
        x, y = rng.choice(S)
        if y * dot(w, x) < 1:
            w = (w[0] + eta * y * x[0], w[1] + eta * y * x[1])
    return (sum(v[0] for v in ws) / T, sum(v[1] for v in ws) / T)

L_star, rng = L_D((15 / 7, -65 / 28)), random.Random(0)
for T in (100, 400, 2500):
    eta = 5 / math.sqrt(T)
    runs = [L_D(sgd(T, rng, eta)) for _ in range(1000)]
    print(T, round(sum(runs) / 1000 - L_star, 3), eta)
```

It prints `100 0.056 0.5`, then `400 0.024 0.25`, then `2500 0.009 0.1`. The middle column is the average gap \(\mathbb E[L_{\mathcal D}(\bar{\mathbf w})] - L_{\mathcal D}(\mathbf w^\star)\), where the \(\mathbb E\) averages over the random draws of a whole run and the snippet estimates it by averaging a thousand runs. The last column is \(\eta = 5/\sqrt T\), which here, with \(B = 5\) and \(\rho = 1\), is also the gradient-descent bound \(B\rho/\sqrt T\). The noisy walk keeps the same promise as the exact one, on average. Note what each step did: it read one message. Now let’s prove that this is no accident.

## The guarantee survives the noise

**Theorem (the SGD guarantee).** Let \(B, \rho > 0\), let \(f\) be convex, and let \(\mathbf w^\star \in \operatorname{argmin}_{\mathbf w : \|\mathbf w\| \le B} f(\mathbf w)\). Run SGD for \(T\) steps with step size \(\eta\), and assume \(\|\mathbf v_t\| \le \rho\) with probability 1 for every \(t\). Then

\[ \mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le \frac{B^2}{2\eta T} + \frac{\eta \rho^2}{2}. \]

Here \(\mathbb E[f(\bar{\mathbf w})]\) is the risk of the output averaged over all the random draws of a run, and “with probability 1” means the bound on \(\|\mathbf v_t\|\) never fails. In plain words: **averaged over runs, the output sits at most \(B^2/(2\eta T) + \eta\rho^2/2\) above the best point of the ball.** With \(B = 5\), \(\rho = 1\), \(T = 100\) and \(\eta = 0.5\), that is \(0.25 + 0.25 = 0.5\), against the 0.056 measured above; the `certificate` line of the snippet at the end of this section is this formula.

With \(\eta = B/(\rho\sqrt T)\) the right side is \(B\rho/\sqrt T\), so, as for gradient descent, \(T \ge B^2\rho^2/\varepsilon^2\) steps buy expected accuracy \(\varepsilon\).

The expectation is over the random choices \(\mathbf v_1, \dots, \mathbf v_T\), and we write \(\mathbf v_{1:t}\) for the first \(t\) of them. The point of the proof is to repeat the deterministic argument and see where randomness enters. It enters in exactly one place.

*Proof.* **Step 1, Jensen, now inside an expectation.** For every outcome of the random choices, Jensen’s inequality gives \(f(\bar{\mathbf w}) - f(\mathbf w^\star) \le \frac1T \sum_t (f(\mathbf w^{(t)}) - f(\mathbf w^\star))\). An inequality that holds for every outcome survives taking expectations, so

\[ \mathbb E_{\mathbf v_{1:T}}\left[ f(\bar{\mathbf w}) - f(\mathbf w^\star) \right] \le \mathbb E_{\mathbf v_{1:T}}\left[ \frac1T \sum_{t=1}^T \left( f(\mathbf w^{(t)}) - f(\mathbf w^\star) \right) \right]. \]

The subscript on \(\mathbb E\) names what is random, here all \(T\) choices; otherwise this is Step 1 of gradient descent with an average over runs wrapped around both sides.

**Step 2, the step lemma, for every outcome.** The step lemma holds for *any* sequence of vectors, so it holds for whatever sequence SGD happened to draw. With \(\|\mathbf w^\star\| \le B\) and \(\|\mathbf v_t\| \le \rho\), dividing by \(T\):

\[ \frac1T \sum_{t=1}^T \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \le \frac{B^2}{2\eta T} + \frac{\eta\rho^2}{2} \quad \text{for every outcome, hence also in expectation.} \]

**Step 3, the missing link.** In the deterministic proof we connected the two sides with the subgradient inequality, term by term. Now \(\mathbf v_t\) is random and need not be a subgradient at all, so that link is broken for individual outcomes. We restore it in expectation: it remains to show that for every \(t\),

\[ \mathbb E_{\mathbf v_{1:T}}\left[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \right] \ge \mathbb E_{\mathbf v_{1:T}}\left[ f(\mathbf w^{(t)}) - f(\mathbf w^\star) \right]. \]

Summing over \(t\), dividing by \(T\), and using linearity of expectation then chains Steps 1 to 3 into the theorem.

The term at time \(t\) depends only on \(\mathbf v_{1:t}\) (because \(\mathbf w^{(t)}\) is built from \(\mathbf v_{1:t-1}\)), so its expectation over \(\mathbf v_{1:T}\) equals its expectation over \(\mathbf v_{1:t}\). Now use the **law of total expectation**: for random variables \(\alpha, \beta\) and a function \(g\), \(\mathbb E_\alpha[g(\alpha)] = \mathbb E_\beta\left[ \mathbb E_\alpha[g(\alpha) \mid \beta] \right]\). Average first over \(\mathbf v_t\) with the past held fixed, then over the past. Take \(\alpha = \mathbf v_{1:t}\) and \(\beta = \mathbf v_{1:t-1}\):

\[ \mathbb E_{\mathbf v_{1:t}}\left[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \right] = \mathbb E_{\mathbf v_{1:t-1}}\left[ \mathbb E\left[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \mid \mathbf v_{1:t-1} \right] \right]. \]

In plain words: **averaging the progress over all of the first \(t\) choices is the same as averaging over step \(t\)’s choice with the past fixed, then averaging that over the past.** Take \(t = 2\) in the six-message world with \(\eta = 0.5\) and the hinge loss. The first draw gives one of six iterates \(\mathbf w^{(2)}\), the second draw one of six messages, so there are 36 equally likely pairs. Averaging the progress over all 36 at once, or over the second draw first and then over the first, gives the same number, and that number is at least the expected gap \(\mathbb E[L_{\mathcal D}(\mathbf w^{(2)})] - L_{\mathcal D}(\mathbf w^\star)\), which is what Step 3 claims:

```python
S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
L_D = lambda w: sum(max(0, 1 - y * dot(w, x)) for x, y in S) / 6
eta, w_star = 0.5, (15 / 7, -65 / 28)

def v(w, x, y):
    return (-y * x[0], -y * x[1]) if y * dot(w, x) < 1 else (0, 0)

term = lambda w, g: dot((w[0] - w_star[0], w[1] - w_star[1]), g)
W2 = [(eta * y * x[0], eta * y * x[1]) for x, y in S]
flat = sum(term(w, v(w, *z)) for w in W2 for z in S) / 36
nested = sum(sum(term(w, v(w, *z)) for z in S) / 6
             for w in W2) / 6
gap = sum(L_D(w) - L_D(w_star) for w in W2) / 6
print(round(flat, 4), round(nested, 4), round(gap, 4))
```

It prints `0.6767 0.6767 0.4892`. (The list `W2` uses that from \(\mathbf 0\) every message is active, so the first step is \(\eta y\mathbf x\).) The next two displays explain why the inequality holds.

Once the past \(\mathbf v_{1:t-1}\) is fixed, \(\mathbf w^{(t)}\) is fixed too, not random any more. The inner product is linear in \(\mathbf v_t\), and expectation is linear, so the fixed vector \(\mathbf w^{(t)} - \mathbf w^\star\) comes out of the inner expectation:

\[ \mathbb E\left[ \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \mid \mathbf v_{1:t-1} \right] = \left\langle \mathbf w^{(t)} - \mathbf w^\star, \, \mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}] \right\rangle. \]

SGD’s requirement says \(\mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}] \in \partial f(\mathbf w^{(t)})\). So the subgradient inequality at \(\mathbf w^{(t)}\), with \(\mathbf u = \mathbf w^\star\), applies to that conditional mean:

\[ \left\langle \mathbf w^{(t)} - \mathbf w^\star, \, \mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}] \right\rangle \ge f(\mathbf w^{(t)}) - f(\mathbf w^\star). \]

Take the outer expectation over \(\mathbf v_{1:t-1}\) of both sides. The left side becomes \(\mathbb E_{\mathbf v_{1:t}}[\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle]\) by the total-expectation step, and the right side \(\mathbb E_{\mathbf v_{1:t-1}}[f(\mathbf w^{(t)}) - f(\mathbf w^\star)]\), which equals the expectation over \(\mathbf v_{1:T}\) since it depends only on the past. That is Step 3.

Chaining: the expected gap is at most the expected average of \(f(\mathbf w^{(t)}) - f(\mathbf w^\star)\) (Step 1), which is at most the expected average of the inner products (Step 3), which is at most \(B^2/(2\eta T) + \eta\rho^2/2\) (Step 2). The choice \(\eta = B/(\rho\sqrt T)\) balances the two terms, exactly as in the step lemma. \(\blacksquare\)

So the randomness cost us nothing but the word “expected”: the guarantee is about the average over runs, which is what our thousand-run simulation measured.

The first form of the theorem is worth staring at, because it is what you tune in practice. Call \(B^2/(2\eta T) + \eta\rho^2/2\) the **certified gap**. The first term punishes a timid step size: with small \(\eta\), you cannot travel the distance \(B\) to the minimizer in \(T\) steps. The second punishes a bold one: with large \(\eta\), each step’s noise of size up to \(\rho\) throws you around. For \(B = 5\), \(\rho = 1\) and \(T = 10{,}000\):

```python
import math

B, rho, T = 5, 1, 10000
certificate = lambda eta: B**2 / (2 * eta * T) + eta * rho**2 / 2
for eta in (0.01, 0.025, 0.05, 0.1, 0.2):
    print(eta, round(certificate(eta), 4))
eta_best = B / (rho * math.sqrt(T))
print(eta_best, certificate(eta_best), B * rho / math.sqrt(T))
```

It prints `0.01 0.13`, `0.025 0.0625`, `0.05 0.05`, `0.1 0.0625` and `0.2 0.1063`, then `0.05 0.05 0.05`. Halving or doubling the best step size costs the same (both give 0.0625, a quarter more), because the two terms trade places. A factor of four away costs more than doubling the gap.

[[diagram:tradeoff]]

## Minimizing the true risk directly

Now the payoff. Every learner so far in this course has worked the same way: collect a training sequence \(S\), compute something from the empirical risk \(L_S\), and then prove separately that \(L_S\) tracks \(L_{\mathcal D}\). SGD lets us skip the middle. We want to minimize

\[ L_{\mathcal D}(\mathbf w) = \mathbb E_{z \sim \mathcal D}[\ell(\mathbf w, z)]. \]

Here \(z \sim \mathcal D\) reads “\(z\) drawn from \(\mathcal D\)”: a fresh labelled message arrives at random, \(\ell(\mathbf w, z)\) is the loss on it, and \(\mathbb E_{z \sim \mathcal D}\) averages over that draw. In plain words: **the true risk is the expected loss on the next message to arrive.** In the six-message world it is the plain average over six, so the hinge risk of \(\mathbf w = \mathbf 0\) is 1; the `L_D` line of every snippet here is this formula.

We cannot compute \(\nabla L_{\mathcal D}\), because we do not know \(\mathcal D\). But SGD does not need the gradient. It needs a random vector whose conditional expectation is a (sub)gradient, an **unbiased estimate**. And nature hands us one with every fresh message.

**Differentiable losses.** At step \(t\), draw a fresh example \(z \sim \mathcal D\), independent of everything so far, and set \(\mathbf v_t = \nabla \ell(\mathbf w^{(t)}, z)\), the gradient of the loss on that one example with respect to \(\mathbf w\). Given the past, \(\mathbf w^{(t)}\) is fixed and only \(z\) is random, so

\[ \mathbb E[\mathbf v_t \mid \mathbf w^{(t)}] = \mathbb E_{z \sim \mathcal D}\left[ \nabla \ell(\mathbf w^{(t)}, z) \right] = \nabla \mathbb E_{z \sim \mathcal D}\left[ \ell(\mathbf w^{(t)}, z) \right] = \nabla L_{\mathcal D}(\mathbf w^{(t)}). \]

Read the chain left to right: the step’s average given the current iterate, the average of the one-message gradients, the gradient of the average loss, and the gradient of the true risk. In plain words: **on average, the gradient on one fresh message is the gradient of the true risk.** The middle step swaps the gradient and the expectation; that is linearity of the gradient, exact when \(\mathcal D\) has finitely many outcomes, as in our tiny world (for general \(\mathcal D\) it needs mild regularity conditions that hold for the losses in this course). Here it is in our six-message world at \(\mathbf w = (1, -0.5)\): the exact gradient of \(L_{\mathcal D}\), a finite-difference estimate of it, and the average of 200,000 one-message gradients.

```python
import math
import random

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
loss = lambda w, x, y: math.log(1 + math.exp(-y * dot(w, x)))
L_D = lambda w: sum(loss(w, x, y) for x, y in S) / 6
v = lambda w, x, y: [-y * c / (1 + math.exp(y * dot(w, x)))
                     for c in x]

w, h, rng = (1.0, -0.5), 1e-6, random.Random(0)
exact = [sum(v(w, x, y)[j] for x, y in S) / 6 for j in (0, 1)]
numeric = [(L_D((w[0] + h, w[1])) - L_D(w)) / h,
           (L_D((w[0], w[1] + h)) - L_D(w)) / h]
draws = [v(w, *rng.choice(S)) for _ in range(200000)]
mean = [sum(d[j] for d in draws) / len(draws) for j in (0, 1)]
for name, g in (("exact", exact), ("numeric", numeric),
                ("sampled", mean)):
    print(name, [round(c, 2) for c in g])
```

All three lines print `[-0.02, 0.08]`.

**Non-differentiable losses, like the hinge.** Let \(\mathbf v_t\) be a subgradient of \(\ell(\cdot, z)\) at \(\mathbf w^{(t)}\), chosen by a fixed rule (such as the hinge rule above). For every \(\mathbf u\), the subgradient inequality says

\[ \ell(\mathbf u, z) - \ell(\mathbf w^{(t)}, z) \ge \langle \mathbf u - \mathbf w^{(t)}, \mathbf v_t \rangle. \]

Take the expectation of both sides over \(z \sim \mathcal D\), conditioned on \(\mathbf w^{(t)}\). Expectation preserves inequalities, and on the left it turns each loss into a risk:

\[ L_{\mathcal D}(\mathbf u) - L_{\mathcal D}(\mathbf w^{(t)}) = \mathbb E\left[ \ell(\mathbf u, z) - \ell(\mathbf w^{(t)}, z) \mid \mathbf w^{(t)} \right] \ge \mathbb E\left[ \langle \mathbf u - \mathbf w^{(t)}, \mathbf v_t \rangle \mid \mathbf w^{(t)} \right] = \langle \mathbf u - \mathbf w^{(t)}, \mathbb E[\mathbf v_t \mid \mathbf w^{(t)}] \rangle. \]

The last equality pulls the fixed vector \(\mathbf u - \mathbf w^{(t)}\) out of the expectation, as in the proof. In plain words: **the average of the one-message subgradients supports the true risk, just as each one supports its own loss**, as the hinge snippet’s \((-0.1833, 0.1333)\) showed at \(\mathbf 0\). Since this holds for every \(\mathbf u\), the conditional mean \(\mathbb E[\mathbf v_t \mid \mathbf w^{(t)}]\) is a subgradient of \(L_{\mathcal D}\) at \(\mathbf w^{(t)}\). Because the fresh \(z\) is independent of the whole past, conditioning on \(\mathbf w^{(t)}\) or on the whole past \(\mathbf v_{1:t-1}\) gives the same answer, so SGD’s requirement holds.

So SGD for the true risk is just this:

```
SGD-Risk(η, T):
  w(1) ← 0
  for t = 1, 2, ..., T:
    receive a fresh example z_t ~ D
    pick v_t ∈ ∂ℓ(w(t), z_t)
    w(t+1) ← w(t) − η v_t
  return w̄ = (1/T) Σ_{t=1..T} w(t)
```

Every step uses a new example, so the number of steps *is* the number of examples. A bound on steps is a sample complexity. Recall from lesson 12b that a **convex-Lipschitz-bounded** learning problem with parameters \(\rho, B\) has a convex hypothesis class \(\mathcal H\) with \(\|\mathbf w\| \le B\) for all \(\mathbf w \in \mathcal H\), and a loss \(\ell(\cdot, z)\) that is convex and \(\rho\)-Lipschitz for every \(z\). Plugging \(f = L_{\mathcal D}\) into the SGD guarantee:

**Corollary (SGD learns convex-Lipschitz-bounded problems).** Consider a convex-Lipschitz-bounded learning problem with parameters \(\rho, B\). For every \(\varepsilon > 0\), if SGD-Risk runs for

\[ T \ge \frac{B^2\rho^2}{\varepsilon^2} \]

steps (that is, on \(T\) fresh examples) with \(\eta = B/(\rho\sqrt T)\), its output satisfies

\[ \mathbb E[L_{\mathcal D}(\bar{\mathbf w})] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \varepsilon. \]

Here \(\min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w)\) is the smallest true risk of any weight vector in the class. In plain words: **on average, \(B^2\rho^2/\varepsilon^2\) fresh messages bring SGD within \(\varepsilon\) of the best weight vector in the class.** In the six-message world with \(B = 5\), \(\rho = 1\) and \(\varepsilon = 0.05\), that is 10,000 messages for an expected hinge risk of at most \(163/336 + 0.05 \approx 0.535\).

*Proof.* \(L_{\mathcal D}\) is an average of convex functions, so it is convex. Each \(\mathbf v_t\) is a subgradient of a \(\rho\)-Lipschitz convex function, so \(\|\mathbf v_t\| \le \rho\) by the Lipschitz lemma. We just showed \(\mathbb E[\mathbf v_t \mid \text{past}] \in \partial L_{\mathcal D}(\mathbf w^{(t)})\). Take \(\mathbf w^\star\) a minimizer of \(L_{\mathcal D}\) over \(\mathcal H\); it has \(\|\mathbf w^\star\| \le B\), and the minimum of \(L_{\mathcal D}\) over the ball \(\{\|\mathbf w\| \le B\}\) can only be smaller, so the SGD guarantee, applied with that ball, bounds the gap to \(\min_{\mathcal H} L_{\mathcal D}\) too. \(\blacksquare\)

For the mail room, the hinge loss on messages with \(\|\mathbf x\| \le 1\) is 1-Lipschitz, so \(\rho = 1\). Say the team keeps the weight vector within \(B = 5\) and wants the filter’s expected hinge risk within \(\varepsilon = 0.05\) of the best in the class. (Since the 0-1 loss is at most the hinge loss, as lesson 12b showed, the filter’s expected error rate is at most its expected hinge risk.) Count the work, with two million messages in the archive and about 200 nonzero word counts per message:

```python
import math

B, rho, eps = 5, 1, 0.05
m, nonzero = 2_000_000, 200
T = math.ceil(B**2 * rho**2 / eps**2)
gd_work = T * m * nonzero
sgd_work = T * nonzero
print(T, f"{gd_work:.1e}", f"{sgd_work:.1e}", gd_work // sgd_work)
print(math.ceil(8 * rho**2 * B**2 / eps**2))
```

It prints `10000 4.0e+12 2.0e+06 2000000`, then `80000`. Ten thousand steps. Gradient descent on the archive would spend \(4 \times 10^{12}\) operations reading every message at every step (and would still only be minimizing \(L_S\), with a generalization argument owed on top). SGD spends \(2 \times 10^6\), two million times less, and its guarantee is already about the true risk. The second line is lesson 13b’s count for regularized loss minimization, \(8\rho^2B^2/\varepsilon^2\): the same order, eight times more examples, and an optimization problem still to solve on top.

[[diagram:stream]]

There is a catch hiding in the word “fresh”, and it matters for the fix. The proof used that \(z_t\) is independent of \(\mathbf w^{(t)}\). If you loop over the same archive twice, the second time round each message has already shaped the weights, and the estimate is no longer unbiased for \(\nabla L_{\mathcal D}\). The guarantee above is for one pass over fresh examples.

## Stochastic gradient descent as code

Here is the learner the mail room will run: SGD on the hinge loss, with an optional projection onto the ball \(\{\|\mathbf w\| \le B\}\): whenever a step carries the weights outside the ball, scale them back onto its edge. With \(B = 1\), the weights \((3, 4)\), of length 5, are scaled by \(\frac15\) to \((0.6, 0.8)\), and \((0.3, 0.4)\) stays put. This keeps every iterate, and so the average, inside the class the guarantee compares with; lesson 14b proves that it costs the guarantee nothing.

```
SGD-Hinge(stream, T, η, B):
  w ← 0 in R^d;  total ← 0 in R^d
  for t = 1..T:
    total ← total + w
    (x, y) ← next fresh message from stream
    if y ⟨w, x⟩ < 1:
      w ← w + η · y · x
    if B is given and ‖w‖ > B:
      w ← (B / ‖w‖) · w
  return w̄ = total / T
```

**Cost.** Each step reads one message and does one inner product and at most one vector addition: \(O(d)\) time, or \(O(s)\) with sparse messages of \(s\) nonzero counts (keeping \(\|\mathbf w\|\) and the running total up to date sparsely needs a little bookkeeping). Total \(O(Td)\) time, and \(O(d)\) space for \(\mathbf w\) and the running total, independent of the archive size. The exact gradient descent step on \(L_S\) costs \(O(md)\), so \(T\) steps cost \(O(Tmd)\) and \(O(md)\) space to hold the archive. The SGD guarantee says \(T = \lceil B^2\rho^2/\varepsilon^2 \rceil\) steps suffice, with no \(m\) and no \(d\) in the count.

**Build it yourself.** Using only the standard library:

```python
import math
import random

S = [((0.8, 0.2), 1), ((0.6, 0.6), 1), ((0.9, 0.4), 1),
     ((0.1, 0.9), -1), ((0.4, 0.8), -1), ((0.7, 0.3), -1)]

def hinge(w, x, y): ...
def hinge_subgradient(w, x, y): ...
def risk(w, S): ...
def steps_needed(B, rho, eps): ...
def certificate(B, rho, T, eta): ...
def project(w, B): ...
def sgd(sample, T, eta, B=None): ...

if __name__ == "__main__":
    v = hinge_subgradient((0, 0), (0.6, 0.6), 1)
    assert v == (-0.6, -0.6)
    assert hinge_subgradient((5, 0), (0.8, 0.2), 1) == (0, 0)
    assert steps_needed(5, 1, 0.05) == 10000
    assert steps_needed(4, 2, 0.1) == 6400
    assert abs(certificate(5, 1, 10000, 0.05) - 0.05) < 1e-12
    assert [round(c, 9) for c in project((3, 4), 1)] == [0.6, 0.8]
    assert project((0.3, 0.4), 1) == (0.3, 0.4)
    rng = random.Random(1)
    w_bar, w_last = sgd(lambda: rng.choice(S), 8, 0.5)
    assert [round(c, 9) for c in w_last] == [1.2, -1.05]
    L_star, rng = risk((15 / 7, -65 / 28), S), random.Random(0)
    runs = [risk(sgd(lambda: rng.choice(S), 400, 0.25)[0], S)
            for _ in range(300)]
    excess = sum(runs) / 300 - L_star
    assert excess <= certificate(5, 1, 400, 0.25)
    w_bar, w_last = sgd(lambda: rng.choice(S), 400, 2.0, B=1)
    assert math.hypot(*w_last) <= 1 + 1e-12
```

1. Implement the functions. `hinge_subgradient` returns `(0, 0)` when the signed score is at least 1 and \(-y\mathbf x\) otherwise. `steps_needed` is \(\lceil B^2\rho^2/\varepsilon^2 \rceil\) (round away floating-point dust before the ceiling). `certificate` is the certified gap \(B^2/(2\eta T) + \eta\rho^2/2\). `project` scales \(\mathbf w\) back onto the ball when it is outside and leaves it alone otherwise.
2. `sgd` takes a function `sample()` that returns one fresh `(x, y)`, calls it exactly once per step, and returns `(w_bar, w_last)`: the average of \(\mathbf w^{(1)}, \dots, \mathbf w^{(T)}\) (add the current iterate to the total *before* the step) and the iterate after the last step. With `random.Random(1)` and \(\eta = 0.5\), eight steps must reproduce the table in “One message per step”.
3. Run `sgd` with \(T = 2{,}500\) and \(\eta\) from 0.01 to 1. Plot the average excess hinge risk over 300 runs against \(\eta\), next to the certified gap. Where is each smallest?
4. Now cheat: draw 100 messages once, then run 2,500 steps cycling through those 100. Compare \(L_S\) and \(L_{\mathcal D}\) of the output with an honest one-pass run. Which risk did the cheating run minimize?

## Fix the bug

Back to the mail room. The new plan is to retrain continuously with SGD on the hinge loss, one fresh message per step. The constraint is fresh messages. Only messages your users label after the spammers’ latest change describe today’s junk, and they arrive at a fixed rate: so many per hour. The new filter must ship within the retrain window, so the number of steps \(T\) cannot exceed the rate times the window. That is your budget.

In the lab below, each round gives you the norm bound \(B\), the Lipschitz constant \(\rho = 1\) (messages are scaled to length at most 1), the target \(\varepsilon\), the labelling rate and the window. You choose the step size \(\eta\) and the number of steps \(T\). The filter ships only if the SGD guarantee certifies it: \(T\) within the budget, and the certified gap \(B^2/(2\eta T) + \eta\rho^2/2\) at most \(\varepsilon\). The lab then runs SGD on a stream of fresh messages, so you can watch the measured excess sit under your certificate. But the measurement does not ship the filter; the certificate does. Work out the smallest \(T\) and the best \(\eta\) for it before you press train.

[[lab-act2]]

## What people get wrong

**“SGD is an approximation of gradient descent, so its guarantee must be worse.”** On a learning problem it is the other way round. Gradient descent on \(L_S\) only optimizes the empirical risk and still owes a generalization argument. SGD on fresh examples optimizes \(L_{\mathcal D}\) directly, with \(B^2\rho^2/\varepsilon^2\) examples, eight times fewer than lesson 13b’s bound for regularized loss minimization.

**“The guarantee holds for every run.”** It holds in expectation over the random draws. A single run can do worse. If you need a statement with probability, the expectation bound still helps: the excess risk is a nonnegative random variable, so Markov’s inequality (lesson 4: a nonnegative variable reaches \(a\) with probability at most its mean over \(a\)) says it exceeds \(\varepsilon/\delta\) with probability at most \(\delta\) (a crude conversion; sharper ones exist).

**“More passes over the archive make SGD better.”** For the true-risk guarantee, every step needs a fresh example, independent of the current weights. Looping over the same messages again turns SGD into an optimizer of \(L_S\), and the guarantee for \(L_{\mathcal D}\) no longer follows from this proof.

**“At a crease there is no slope, so the method breaks.”** Any subgradient works. At the hinge’s crease you can take \(\mathbf 0\), or \(-y\mathbf x\), or anything on the segment between them; the analysis only uses the supporting-plane inequality and the norm bound \(\rho\).

**“A bigger step size gets there faster.”** The certified gap \(B^2/(2\eta T) + \eta\rho^2/2\) grows without limit as \(\eta\) grows. Doubling the best step size costs as much as halving it.

## Recap and the next bug

1. Gradient descent steps against the gradient, \(\mathbf w^{(t+1)} = \mathbf w^{(t)} - \eta\nabla f(\mathbf w^{(t)})\), and outputs the average iterate. For convex \(\rho\)-Lipschitz \(f\) and a comparator of norm at most \(B\), the gap is at most \(B\rho/\sqrt T\) with \(\eta = B/(\rho\sqrt T)\).
2. The engine is the step lemma: for *any* step vectors, the sum of \(\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle\) is at most \(\|\mathbf w^\star\|^2/(2\eta) + \frac\eta2\sum_t\|\mathbf v_t\|^2\), by completing a square and telescoping.
3. A subgradient is the slope of a supporting plane. It exists at every point of a convex function, equals the gradient where \(f\) is differentiable, and has norm at most \(\rho\) everywhere exactly when \(f\) is \(\rho\)-Lipschitz. For the hinge loss it is \(\mathbf 0\) or \(-y\mathbf x\).
4. SGD steps along a random vector whose conditional mean is a subgradient. The guarantee \(\mathbb E[f(\bar{\mathbf w})] - f(\mathbf w^\star) \le B^2/(2\eta T) + \eta\rho^2/2\) survives, by the law of total expectation.
5. On a learning problem, the (sub)gradient of the loss on one fresh example is an unbiased estimate for the true risk. SGD therefore learns convex-Lipschitz-bounded problems from \(B^2\rho^2/\varepsilon^2\) examples, in \(O(d)\) time per example and \(O(d)\) space.

The filter retrains in minutes now, on the morning’s fresh labels, and it is no longer a day behind. Then the auditor from lesson 13b comes back with the certificate she signed. “This is for the regularized filter,” she says: the weight vector that minimizes lesson 13a’s penalized objective on the audited training sequence, the two million labelled messages the team keeps on file. “I can check that filter against the audited messages. Your streaming filter is a different learner, and its guarantee is about a stream nobody kept.” She will sign a filter only if it is provably close to the minimum of that objective, and the exact solver for it still takes six hours. Can SGD solve a fixed objective on a fixed training sequence, one message per step, and certify how close to the exact minimum it lands? And how fast? That is lesson 14b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A message sits exactly on the hinge’s crease: \(y\langle \mathbf w, \mathbf x \rangle = 1\). Describe the whole set \(\partial \ell(\mathbf w, (\mathbf x, y))\) at that point. Does it matter, for the SGD guarantee, which member the algorithm picks? What property of the picked vector does the proof actually use?
2. In the proof of the SGD guarantee, point to the single step that uses \(\mathbb E[\mathbf v_t \mid \mathbf v_{1:t-1}] \in \partial f(\mathbf w^{(t)})\). Then: the team runs SGD-Risk for a second pass over the same 10,000 messages. Which equality in “Minimizing the true risk directly” fails on the second pass, and why?
3. Messages are no longer scaled, and their norms go up to \(\rho = 2\); the class has \(B = 4\), and the target is \(\varepsilon = 0.1\). How many fresh messages does the SGD guarantee need, and with which \(\eta\)? A colleague doubles your \(\eta\) and keeps \(T\): what gap is certified now? And if only 5,000 fresh messages arrive in the window, what is the best \(\varepsilon\) you can certify?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (the Perceptron as subgradient descent).** Let \(S = ((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m))\) with \(y_i \in \{\pm 1\}\), assume some \(\mathbf w\) has \(y_i\langle \mathbf w, \mathbf x_i \rangle \ge 1\) for every \(i\), let \(\mathbf w^\star\) have the smallest norm among such vectors, and let \(R = \max_i \|\mathbf x_i\|\). Define \(f(\mathbf w) = \max_{i \in [m]} \left(1 - y_i\langle \mathbf w, \mathbf x_i \rangle\right)\). (a) Show that \(\min_{\|\mathbf w\| \le \|\mathbf w^\star\|} f(\mathbf w) = 0\), and that every \(\mathbf w\) with \(f(\mathbf w) < 1\) separates \(S\). (b) Show how to compute a subgradient of \(f\). (c) Describe and analyse subgradient descent on \(f\), and compare the algorithm and its guarantee with the Perceptron of lesson 9a.
