# Regularization and Stability

## The filter that would not sit still

Lesson 12 ended with an uneasy feeling. A user corrected one label, someone retrained the hinge filter out of habit, and the new weights looked nothing like the old ones. The bound was fine, because it never promised that the output would sit still. But the feeling turned into a bug report within the week.

The team now retrains every convex filter every night. The newest one scores messages from a marketing campaign, and it is the junk score of lesson 9: a weight vector \(\mathbf w\), a score \(\langle \mathbf w, \mathbf x \rangle\), and the square loss, which lesson 12 showed is convex and smooth. Each night it is trained on the twenty most recent labelled campaign messages, so from one night to the next only a handful of labels change: three old ones drop out, three new ones come in.

Here is what the dashboard showed for one week. The weight on the word “free” went from 2.2 to −8.2, back to −2.2, then to −5.9. The weight on “offer” jumped from −0.3 to 11.8. And the production error, the filter’s average loss on fresh campaign messages, swung from 0.384 to 0.632 and back to 0.387 in three nights. Three labels out of twenty changed each night. Nobody would call that a big change in the data. Yet the learner’s output jumped around like a nervous bird, and its true quality jumped with it.

What is going on? The learner did exactly what it was told: it minimized the empirical risk, perfectly, every night. Nothing is broken in the optimization, and lesson 12 promised us the problem is convex. So why does a tiny change in the input make such a big change in the output, and why does that hurt the true risk? And the deeper question, the one lesson 12 left hanging: could a learner that *refuses* to jump around generalize for that very reason?

This lesson answers both. The fix is one extra term in the objective. The theory behind it is a new way to prove that a learner does not overfit, one that never mentions uniform convergence.

## A week of retrains

Let’s build a world small enough to compute everything in it, and big enough to show the bug. Call it the **campaign world**.

The campaign uses five words that always travel together: “free”, “gift”, “prize”, “bonus” and “offer”. A message is a vector \(\mathbf x \in \mathbb R^5\), one coordinate per word, scaled so that each coordinate is about \(\pm 0.4\). Labels are \(y = +1\) for junk and \(y = -1\) for keep, as in lesson 9. Here is how a message is drawn:

1. Flip a fair coin for the label \(y \in \{+1, -1\}\).
2. Draw the campaign signal \(f = 0.4\,y\) with probability 0.8 and \(f = -0.4\,y\) with probability 0.2. So 80% of the time the campaign words point the right way.
3. Each word’s coordinate is the signal plus its own tiny jitter: \(x_j = f + e_j\), where each \(e_j\) is \(+0.04\) or \(-0.04\) with equal probability, independently of everything else.

The five words are near-twins. They carry the same information, \(f\), plus a little noise that has nothing to do with the label. Every message has \(\|\mathbf x\| \le \sqrt{5 \cdot 0.44^2} \approx 0.98\), so it lies in the unit ball.

The loss we use in this lesson is the **half square loss**,

\[ \ell(\mathbf w, (\mathbf x, y)) = \tfrac12 \left( \langle \mathbf w, \mathbf x \rangle - y \right)^2, \]

the square loss of lesson 9 with a factor \(\tfrac12\) in front. The half changes nothing about which \(\mathbf w\) is best, and it makes the constants below come out cleaner.

Read it from the inside out.

- \(\mathbf w = (w_1, \dots, w_5)\) is the weight vector, one weight per word, and \(\mathbf x\) is the message’s vector of word coordinates. The pair \((\mathbf x, y)\) is one example \(z\), as in every lesson since lesson 1.
- \(\langle \mathbf w, \mathbf x \rangle = \sum_j w_j x_j\) is the inner product from lesson 9: multiply each weight by its word’s coordinate and add. It is the filter’s **score** for the message.
- \(\langle \mathbf w, \mathbf x \rangle - y\) is the miss: how far the score lands from the label \(+1\) or \(-1\).
- Squaring makes every miss count positively, and bigger misses count much more; the \(\tfrac12\) halves the result.
- \(\ell(\mathbf w, z)\) is lesson 3’s notation for “the loss of hypothesis \(\mathbf w\) on example \(z\)”, one example at a time.

In plain words: **the loss is half the squared distance between the filter’s score and the message’s label.** A junk message (\(y = 1\)) whose five coordinates are 0.44, 0.36, 0.44, 0.44, 0.36, scored by \(\mathbf w = (0.3, \dots, 0.3)\), gets score \(0.3 \times 2.04 = 0.612\), misses by \(-0.388\), and pays \(\tfrac12 \cdot 0.388^2 \approx 0.0753\).

In this world the true risk has a formula. Expand the square and take expectations term by term, where \(\mathbb E[\cdot]\) means “the average value of” over random messages. For two different words, \(\mathbb E[x_j x_k] = \mathbb E[f^2] = 0.16\), since the jitters are independent of each other and of \(f\) and have mean 0. For the same word, \(\mathbb E[x_j^2] = 0.16 + 0.04^2 = 0.1616\). And \(\mathbb E[y x_j] = \mathbb E[y f] = 0.4 \cdot (0.8 - 0.2) = 0.24\). Writing \(s = \sum_j w_j\) for the total weight and \(q = \|\mathbf w\|^2\),

\[ L_{\mathcal D}(\mathbf w) = \tfrac12 \left( 0.16\, s^2 + 0.0016\, q - 0.48\, s + 1 \right). \]

Read it from the inside out.

- \(s = \sum_j w_j\) is the total weight, the sum of the five weights; \(q = \|\mathbf w\|^2 = \sum_j w_j^2\) is the squared length of the weight vector (lesson 9’s Euclidean norm, squared).
- \(L_{\mathcal D}(\mathbf w)\) is the true risk from lesson 1: the loss averaged over every message that could arrive, each weighted by how often \(\mathcal D\) sends it. Here it is \(\mathbb E_{z \sim \mathcal D}[\ell(\mathbf w, z)]\), where \(z \sim \mathcal D\) reads “\(z\) drawn from \(\mathcal D\)”.
- \(0.16\,s^2\) is the average squared score coming from the shared signal \(f\); \(0.0016\,q\) is the extra squared score coming from the five private jitters; \(-0.48\,s\) is twice the average agreement between score and label, subtracted; and the 1 is the average of \(y^2\).
- The outer \(\tfrac12\) is the half of the half square loss.

In plain words: **the production error is a parabola in the total weight, plus a small charge for the length of the weight vector.** For \(\mathbf w = (0.3, \dots, 0.3)\): \(s = 1.5\) and \(q = 0.45\), so the four terms are 0.36, 0.00072, \(-0.72\) and 1, which add to 0.64072, and half of that is 0.32036. The snippet computes the loss of the one message above, then the four terms, one by one:

```python
w, x, y = [0.3] * 5, [0.44, 0.36, 0.44, 0.44, 0.36], 1
score = sum(a * b for a, b in zip(w, x))
print(round(score, 3), round((score - y) ** 2 / 2, 4))
s, q = sum(w), sum(v * v for v in w)
terms = [.16 * s * s, .0016 * q, -.48 * s, 1]
print([round(t, 5) for t in terms], round(sum(terms) / 2, 5))
```

It prints `0.612 0.0753`, then `[0.36, 0.00072, -0.72, 1] 0.32036`.

Look at what this says. The true risk depends almost entirely on the *total* weight \(s\). How you split that total among the five words matters only through the tiny term \(0.0016\,q\). Keep that in mind.

Here is the world as code, with a check of the formula against brute force: draw 200,000 messages and average the loss of \(\mathbf w = (0.3, \dots, 0.3)\).

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    f = .4 * y * rng.choice((1, 1, 1, 1, -1))
    return [f + rng.choice((.04, -.04)) for _ in range(5)], y

def L_D(w):
    s, q = sum(w), sum(v * v for v in w)
    return (.16 * s * s + .0016 * q - .48 * s + 1) / 2

rng, w, N, total = random.Random(0), [0.3] * 5, 200_000, 0
for _ in range(N):
    x, y = draw(rng)
    total += (sum(a * b for a, b in zip(w, x)) - y) ** 2 / 2
print(round(total / N, 3), round(L_D(w), 3))
print(round(L_D([0.2994] * 5), 3), round(L_D([0] * 5), 3))
```

It prints `0.321 0.32`, then `0.32 0.5`. The simulation agrees with the formula to within its noise. The best weight vector puts 0.2994 on every word (with \(c\) on each word, \(s = 5c\) and \(q = 5c^2\), so the bracket is \(4.008c^2 - 2.4c + 1\), lowest at \(c = 2.4/8.016 \approx 0.2994\)), with true risk 0.320, and the all-zero vector, which always scores 0, has true risk 0.5.

Now the nightly retrain. Start with twenty labelled messages, fit by ERM (which is exactly least squares from lesson 9), and each night replace three of the twenty with fresh ones. Before you look at the table, a prediction: how much do you expect the weights to move when 3 of 20 examples change? (The code that produced this table is in the ridge regression section, where it runs two learners side by side on the same messages.)

| night | ERM weights (free, gift, prize, bonus, offer) | production error |
|---|---|---|
| 1 | 2.2, 5.7, −6.4, 0.4, −0.3 | 0.384 |
| 2 | −8.2, −3.3, −8.6, 9.9, 11.8 | 0.632 |
| 3 | −2.2, 2.7, −6.1, 1.5, 5.7 | 0.387 |
| 4 | −5.9, −5.4, −2.6, 5.5, 10.1 | 0.485 |
| 5 | −5.0, −4.8, −2.5, 5.7, 8.3 | 0.448 |
| 6 | −6.1, −4.7, −2.9, 4.9, 10.3 | 0.478 |

That is the dashboard from the opening. Notice two things. First, the individual weights are wild: many are ten to forty times the best value of 0.3, and their signs flip from night to night. Second, look at the totals. Night 1 sums to 1.62, night 2 to 1.61, night 3 to 1.58, and all six nights stay between 1.52 and 1.70. The total \(s\) barely moves. The learner is not confused about how junky the campaign is. It is confused about how to *split* the blame among five words that always appear together.

Why would the data leave the split undecided? Think of the empirical risk as a landscape over the weights. Along some directions it rises steeply; along others it is nearly flat. For the half square loss, the curvature of \(L_S\) along a unit direction \(\mathbf u\) (the second derivative of \(t \mapsto L_S(\mathbf w + t\mathbf u)\)) is

\[ \frac{1}{m} \sum_{i=1}^m \langle \mathbf u, \mathbf x_i \rangle^2, \]

because each term \(\tfrac12(\langle \mathbf w + t\mathbf u, \mathbf x_i \rangle - y_i)^2\) is a parabola in \(t\) with second derivative \(\langle \mathbf u, \mathbf x_i \rangle^2\). Read it from the inside out.

- \(\mathbf u\) is a direction in weight space, with length 1; \(\langle \mathbf u, \mathbf x_i \rangle\) is how much message \(i\)’s score changes when you move the weights one unit along \(\mathbf u\).
- Squaring and averaging over the \(m\) training messages, \(\frac1m \sum_{i=1}^m\), gives the average squared change: the sum runs over \(i = 1\) to \(m\), one term per example.

In plain words: **the curvature along a direction is the average squared effect that direction has on the training scores.** A direction that barely changes any score, such as moving weight from “free” to “gift” when both words always appear together, has curvature near 0. For the five-word direction \((1, 1, 1, 1, 1)/\sqrt5\), each message’s score moves by about \(5 \times 0.4/\sqrt5 \approx 0.89\), whose square is about 0.8: close to the 0.814 the code finds below.

Let’s measure it on night 1’s twenty messages, along the direction “raise every word together” and along two directions that only shuffle weight between words.

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    f = .4 * y * rng.choice((1, 1, 1, 1, -1))
    return [f + rng.choice((.04, -.04)) for _ in range(5)], y

rng = random.Random(16)
S = [draw(rng) for _ in range(20)]
def curvature(u):
    n2 = sum(a * a for a in u)
    dots = [sum(a * b for a, b in zip(u, x)) for x, _ in S]
    return sum(t * t for t in dots) / (len(S) * n2)
print(round(curvature([1, 1, 1, 1, 1]), 3))
print(round(curvature([1, -1, 0, 0, 0]), 4))
print(round(curvature([0, 0, 1, 1, -2]), 4))
```

It prints `0.814`, then `0.0019`, then `0.0019`. Along the total, the landscape curves 400 times more steeply than along the shuffles. In the shuffle directions the empirical risk is a long, almost flat **valley**: moving from “all on free” to “all on bonus” barely changes the training fit. The bottom of a nearly flat valley is decided by whatever tiny tilt the jitter happens to give it, and three new examples change that tilt completely. So the minimizer slides from one end of the valley to the other.

Then why does production suffer? Because the valley is only *nearly* flat, and fresh messages have their own jitter. A weight vector with huge entries of opposite signs turns each fresh message’s jitter into a large, random swing of the score. That is the \(0.0016\,q\) term in the formula: on night 2, \(q\) is about 400, which alone adds about 0.3 to the true risk. The learner fitted the noise in the training jitter, and it paid in production. That is overfitting, seen from the inside.

[[diagram:valley]]

Over many independent training sequences of twenty, ERM’s production error averages 0.427, far above the best possible 0.320. Play with the retrain yourself before we fix anything. The lab runs the campaign world with twenty labels a night. You can step through nights, replace a single label, and watch the five weights and the production error. The slider \(\lambda\) does nothing until the next section explains it; leave it at 0 for now.

[[lab]]

## Pay for the size: regularized loss minimization

How do you tell the learner to stop sliding along the valley? You could restrict the hypothesis class, as lesson 1 taught, and allow only weight vectors with small norm. Lesson 12 did exactly that with its bounded classes \(\|\mathbf w\| \le B\). Here is a softer version of the same idea: let the learner go anywhere, but charge it for the size of the weights.

Put a number on it. On night 2 the ERM weights had squared length \(\sum_j w_j^2 \approx 389\), while the best weights, 0.3 on every word, have squared length \(5 \times 0.09 = 0.45\). Suppose every unit of squared length cost the learner 0.1 on top of its training loss. Night 2’s weights would then pay about 38.9, a hundred times more than any training loss in this world, so the learner would never choose them; the best weights would pay only 0.045. The learner keeps minimizing a training score, but the score now includes a price for size.

**Definition (regularized loss minimization).** A **regularization function** is a map \(R : \mathbb R^d \to \mathbb R\). The **regularized loss minimization** rule, RLM for short, returns a hypothesis in

\[ \operatorname*{argmin}_{\mathbf w} \left( L_S(\mathbf w) + R(\mathbf w) \right). \]

The regularization function measures, intuitively, how “complex” a hypothesis is, and RLM balances a low empirical risk against a “simpler” hypothesis. Many choices of \(R\) are possible, each reflecting some prior belief about the problem, much as the description language did for minimum description length in lesson 7. This lesson uses one of the simplest:

\[ R(\mathbf w) = \lambda \|\mathbf w\|^2, \qquad \lambda > 0, \qquad \|\mathbf w\| = \sqrt{\textstyle\sum_{i=1}^d w_i^2}, \]

which gives the learning rule

\[ A(S) = \operatorname*{argmin}_{\mathbf w} \left( L_S(\mathbf w) + \lambda \|\mathbf w\|^2 \right). \]

Read it from the inside out.

- \(\|\mathbf w\|^2 = \sum_{j=1}^d w_j^2\) is the squared length of the weight vector, and \(\lambda \|\mathbf w\|^2\) is its price: \(\lambda\) units of objective per unit of squared length.
- \(L_S(\mathbf w)\) is the empirical risk from lesson 1, now with a general loss: \(\frac1m \sum_{i=1}^m \ell(\mathbf w, z_i)\), the average loss on the training sequence \(S\).
- The sum \(L_S(\mathbf w) + \lambda\|\mathbf w\|^2\) is the regularized objective: fit plus price.
- \(\operatorname{argmin}_{\mathbf w}\) means “the \(\mathbf w\) at which the expression is smallest”, not the smallest value itself. The subscript says what we are allowed to vary.
- \(A(S)\) is the learner’s output on \(S\), as in lesson 1.

In plain words: **RLM returns the weight vector with the best trade between training fit and squared length.** Take the one-word world (a single “free” coordinate) with three examples, \((0.4, +1)\), \((-0.4, -1)\) and \((0.4, -1)\), the last a junk-looking message that was labelled keep, and \(\lambda = 0.05\). At \(w = 1\) the three losses are 0.18, 0.18 and 0.98, so \(L_S(1) = 1.34/3 \approx 0.4467\), the price is \(0.05 \cdot 1^2 = 0.05\), and the objective is 0.4967. A grid search finds the minimizer:

```python
S, lam = [(0.4, 1), (-0.4, -1), (0.4, -1)], 0.05
loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
L_S = lambda w: sum(loss(w, z) for z in S) / len(S)
f_S = lambda w: L_S(w) + lam * w * w
grid = [i / 1000 for i in range(-3000, 3001)]
for w in (0.0, 1.0, min(grid, key=f_S)):
    print(w, round(L_S(w), 4), round(lam * w * w, 4),
          round(f_S(w), 4))
```

It prints `0.0 0.5 0.0 0.5`, `1.0 0.4467 0.05 0.4967` and `0.513 0.4527 0.0132 0.4658`. At \(w = 0\) the price is nothing but the fit is poor; at \(w = 1\) the fit is better but the price is 0.05; the argmin, 0.513, pays a little of each and has the smallest total.

This choice is called **Tikhonov regularization**, and the number \(\lambda\) is the **regularization weight** (you will also see it called the regularization parameter). We write \(f_S(\mathbf w) = L_S(\mathbf w) + \lambda \|\mathbf w\|^2\) for the **regularized objective**.

Look at what the penalty does to the valley. Along any unit direction \(\mathbf u\), the function \(t \mapsto \lambda \|\mathbf w + t\mathbf u\|^2\) is a parabola with second derivative \(2\lambda\). So the penalty adds \(2\lambda\) to the curvature in *every* direction, steep or flat. With \(\lambda = 0.1\), the shuffle directions go from 0.0019 to 0.2019, and the total direction from 0.814 to 1.014. The steep direction hardly notices. The flat valley gets a floor with a clear lowest point, and that point no longer depends on the jitter’s whims.

[[diagram:penalty]]

There are two ways to read this rule, and both will pay off.

**As structural risk minimization.** Lesson 7 chose among nested classes by trading empirical risk against a complexity charge. Define \(\mathcal H_1 \subset \mathcal H_2 \subset \mathcal H_3 \subset \dots\) with \(\mathcal H_i = \{\mathbf w : \|\mathbf w\|^2 \le i\}\). If the number of examples each \(\mathcal H_i\) needs grows with \(i\), then charging \(\lambda \|\mathbf w\|^2\) is a smooth version of “pay more for a bigger class”, and RLM behaves like SRM over this sequence.

**As a stabilizer.** The penalty makes the minimizer sit still when the data move a little. That is the reading this lesson develops, and it leads to a proof that the learner does not overfit. But first, let’s see the rule work on the campaign.

## Ridge regression

Apply RLM with Tikhonov regularization to linear regression with the half square loss:

\[ A(S) = \operatorname*{argmin}_{\mathbf w \in \mathbb R^d} \left( \lambda \|\mathbf w\|^2 + \frac1m \sum_{i=1}^m \tfrac12 \left( \langle \mathbf w, \mathbf x_i \rangle - y_i \right)^2 \right). \]

Linear regression solved this way is called **ridge regression**.

To solve it, do what lesson 9 did for least squares: set the gradient to zero. The gradient of \(\lambda \|\mathbf w\|^2\) is \(2\lambda \mathbf w\). The gradient of one loss term is \((\langle \mathbf w, \mathbf x_i \rangle - y_i)\,\mathbf x_i\), by the chain rule; the half cancels the 2 from the square. So

\[ \nabla f_S(\mathbf w) = 2\lambda \mathbf w + \frac1m \sum_{i=1}^m (\langle \mathbf w, \mathbf x_i \rangle - y_i)\,\mathbf x_i. \]

Multiply by \(m\) and use the same matrices as lesson 9, \(A = \sum_{i=1}^m \mathbf x_i \mathbf x_i^\top\) and \(\mathbf b = \sum_{i=1}^m y_i \mathbf x_i\). Since \(\sum_i \mathbf x_i \langle \mathbf x_i, \mathbf w \rangle = A\mathbf w\), the condition \(\nabla f_S(\mathbf w) = \mathbf 0\) becomes the linear system

\[ (2\lambda m I + A)\,\mathbf w = \mathbf b, \]

where \(I\) is the \(d \times d\) identity matrix. (The matrix \(A\) and the learner \(A(S)\) share a letter; the learner always carries its argument.)

Read the system from the inside out.

- \(\mathbf x_i \mathbf x_i^\top\) is the \(d \times d\) matrix whose entry \((j, k)\) is \(x_{i,j} x_{i,k}\) (lesson 9); \(A\) adds them up over all \(m\) examples, so entry \((j, k)\) of \(A\) is \(\sum_i x_{i,j}x_{i,k}\), how often words \(j\) and \(k\) light up together.
- \(\mathbf b = \sum_i y_i \mathbf x_i\) adds up the messages, each signed by its label: entry \(j\) is how much word \(j\) leans toward junk.
- \(I\) is the identity matrix, 1 on the diagonal and 0 elsewhere, so \(2\lambda m I\) adds \(2\lambda m\) to every diagonal entry of \(A\) and nothing else.
- \(\nabla f_S(\mathbf w)\) is the gradient: the vector of partial derivatives, one per weight, pointing uphill.

In plain words: **the ridge weights solve least squares’ linear system with \(2\lambda m\) added to the diagonal.** In the three-example world, \(A = 0.16 + 0.16 + 0.16 = 0.48\), \(\mathbf b = 0.4 + 0.4 - 0.4 = 0.4\), and \(2\lambda m = 2 \cdot 0.05 \cdot 3 = 0.3\), so \(w = 0.4/0.78 \approx 0.5128\), the same point the grid search found, and the gradient there is zero:

```python
S, lam = [(0.4, 1), (-0.4, -1), (0.4, -1)], 0.05
m = len(S)
A = sum(x * x for x, _ in S)
b = sum(y * x for x, y in S)
w = b / (2 * lam * m + A)
grad = 2 * lam * w + sum((w * x - y) * x for x, y in S) / m
print(round(A, 2), round(b, 2), round(2 * lam * m, 2))
print(round(w, 4), round(grad, 12) + 0)
```

It prints `0.48 0.4 0.3`, then `0.5128 0.0`.

**Claim (ridge always has a unique solution).** For every \(\lambda > 0\), the matrix \(2\lambda m I + A\) is invertible, all its eigenvalues are at least \(2\lambda m\), and

\[ \mathbf w = (2\lambda m I + A)^{-1} \mathbf b \]

is the unique minimizer of the ridge objective.

*Proof.* First, \(A\) is **positive semidefinite**: for every vector \(\mathbf v\), \(\mathbf v^\top A \mathbf v = \sum_i \mathbf v^\top \mathbf x_i \mathbf x_i^\top \mathbf v = \sum_i \langle \mathbf v, \mathbf x_i \rangle^2 \ge 0\). So for every \(\mathbf v\),

\[ \mathbf v^\top (2\lambda m I + A)\,\mathbf v = 2\lambda m \|\mathbf v\|^2 + \mathbf v^\top A \mathbf v \ge 2\lambda m \|\mathbf v\|^2. \]

If \((2\lambda m I + A)\mathbf v = \mathbf 0\), the left side is 0, so \(2\lambda m\|\mathbf v\|^2 \le 0\) and \(\mathbf v = \mathbf 0\). A square matrix whose only null vector is \(\mathbf 0\) is invertible. If \(\mu\) is an eigenvalue with unit eigenvector \(\mathbf v\), then \(\mu = \mathbf v^\top (2\lambda m I + A)\mathbf v \ge 2\lambda m\).

It remains to see that the solution is a minimizer, not just a flat point. Expand the objective around any \(\mathbf w\) with a step \(\boldsymbol\Delta\), exactly as lesson 9 did:

\[ f_S(\mathbf w + \boldsymbol\Delta) = f_S(\mathbf w) + \langle \nabla f_S(\mathbf w), \boldsymbol\Delta \rangle + \frac{1}{2m}\sum_{i=1}^m \langle \boldsymbol\Delta, \mathbf x_i \rangle^2 + \lambda \|\boldsymbol\Delta\|^2. \]

This is exact: \(\lambda\|\mathbf w + \boldsymbol\Delta\|^2 = \lambda\|\mathbf w\|^2 + 2\lambda\langle \mathbf w, \boldsymbol\Delta \rangle + \lambda\|\boldsymbol\Delta\|^2\), and each loss term is a parabola whose expansion has no higher terms. At the solution the gradient vanishes, so \(f_S(\mathbf w + \boldsymbol\Delta) - f_S(\mathbf w) \ge \lambda\|\boldsymbol\Delta\|^2\), which is strictly positive for every \(\boldsymbol\Delta \ne \mathbf 0\). So the solution is the unique minimizer. \(\blacksquare\)

Two remarks. Lesson 9 had to reach for the pseudoinverse when \(A\) was singular, for instance when two words always appeared together. Ridge never needs it: \(2\lambda m\) lifts every eigenvalue off zero. And the last line of the proof, “the objective rises at least \(\lambda\|\boldsymbol\Delta\|^2\) as you leave the minimizer”, is the property the whole stability argument will rest on. Hold on to it.

Now rerun the same week, same messages, same replacements, with \(\lambda = 0.1\). The function `fit` below builds the augmented matrix \([\,2\lambda m I + A \mid \mathbf b\,]\) by appending each label to its message, then solves the system by Gauss–Jordan elimination (divide a row by its diagonal entry, subtract multiples of it from every other row to clear that column, repeat for each column); with `lam = 0` it is plain least squares. Each night prints two lines, ERM first and ridge second.

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    f = .4 * y * rng.choice((1, 1, 1, 1, -1))
    return [f + rng.choice((.04, -.04)) for _ in range(5)], y

def fit(S, lam):
    X, k = [x + [y] for x, y in S], 2 * lam * len(S)
    M = [[sum(r[i] * r[j] for r in X) + k * (i == j)
          for j in range(6)] for i in range(5)]
    for c in range(5):
        M[c] = [v / M[c][c] for v in M[c]]
        for r in set(range(5)) - {c}:
            M[r] = [v - M[r][c] * p for v, p in zip(M[r], M[c])]
    return [row[5] for row in M]

L_D = lambda w: (.16 * sum(w) ** 2 + .0016 * sum(v * v for v in w)
                 - .48 * sum(w) + 1) / 2
rng = random.Random(16)
S = [draw(rng) for _ in range(20)]
for night in range(1, 7):
    for lam, n in ((0, 1), (0.1, 2)):
        w = fit(S, lam)
        print(night, [round(v, n) for v in w], round(L_D(w), 3))
    for i in rng.sample(range(20), 3):
        S[i] = draw(rng)
```

| night | ERM production error | ridge weights, \(\lambda = 0.1\) | ridge production error |
|---|---|---|---|
| 1 | 0.384 | 0.29, 0.31, 0.22, 0.29, 0.27 | 0.321 |
| 2 | 0.632 | 0.22, 0.28, 0.17, 0.27, 0.26 | 0.328 |
| 3 | 0.387 | 0.26, 0.31, 0.22, 0.29, 0.31 | 0.321 |
| 4 | 0.485 | 0.24, 0.27, 0.25, 0.30, 0.34 | 0.321 |
| 5 | 0.448 | 0.24, 0.27, 0.26, 0.30, 0.34 | 0.321 |
| 6 | 0.478 | 0.24, 0.26, 0.26, 0.29, 0.35 | 0.321 |

The ridge weights sit near the best value of 0.3 and move by a few hundredths per night. The production error hugs the optimum of 0.320. Over many independent training sequences of twenty, ridge with \(\lambda = 0.1\) averages 0.338 against ERM’s 0.427.

That is the fix, demonstrated. Now we need to understand it, because “it worked this week” is not a guarantee, and because \(\lambda = 0.1\) came out of nowhere. Why should a penalty that pins the weights also protect the true risk? The answer goes through a single idea: stability.

## Stable rules do not overfit

Let’s name what went wrong precisely. On night 2, ERM’s weights had an average loss of 0.202 on their own twenty training messages, and 0.632 on fresh messages: they looked three times better on the data they were fitted to than they really were. That difference is what overfitting means. A learner \(A\) **overfits** when the true risk of its output, \(L_{\mathcal D}(A(S))\), is much larger than its empirical risk, \(L_S(A(S))\). Throughout this lesson we look at the average over training sequences, the **overfitting gap**

\[ \mathbb E_{S \sim \mathcal D^m} \left[ L_{\mathcal D}(A(S)) - L_S(A(S)) \right], \]

where \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) examples drawn i.i.d. from \(\mathcal D\)” (lesson 2), and \(\mathbb E_{S \sim \mathcal D^m}\) averages over those draws. Our guarantees will be about expected risks, not “with probability at least \(1 - \delta\)”. The first problem at the end of this lesson shows how to turn one into the other.

Now, what does “a small change of the input” mean? Here is the change the nightly retrain makes, shrunk to its smallest: replace one example. Given a training sequence \(S = (z_1, \dots, z_m)\) and one more example \(z'\), let

\[ S^{(i)} = (z_1, \dots, z_{i-1}, z', z_{i+1}, \dots, z_m) \]

be \(S\) with its \(i\)th example replaced by \(z'\). We call \(S^{(i)}\) the **replace-one sequence**.

And how do we measure the change in the output? Compare the loss of the two hypotheses on the example that was swapped out, \(z_i\):

\[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i). \]

The first hypothesis never saw \(z_i\); the second was trained on it. For a sensible learner this difference is at least 0, since knowing an example should help on that example. But if it is *large*, the learner changes its mind drastically about \(z_i\) depending on whether it has seen it, and that smells like overfitting: a learner that fits each example individually instead of learning the pattern.

Read the pieces from the inside out.

- \(S^{(i)}\): the superscript \((i)\) names which position was replaced; everything else in the sequence is untouched. \(z'\) is one more example drawn from the same \(\mathcal D\).
- \(A(S^{(i)})\) and \(A(S)\) are the two weight vectors the learner returns on the two sequences.
- \(\ell(A(S^{(i)}), z_i)\) is the loss of the first on the swapped-out example, and \(\ell(A(S), z_i)\) the loss of the second on the same example; we subtract them.

In plain words: **the replace-one change is how much worse the learner does on an example once it no longer gets to train on it.** In the three-example world with \(\lambda = 0.05\), replace the keep-labelled \(z_3 = (0.4, -1)\) by \(z' = (0.4, +1)\). Ridge moves from \(w = 0.5128\) to \(w = 1.5385\); on \(z_3\) the loss goes from 0.7262 to 1.3047, a change of 0.5786:

```python
S, lam, z_new, i = [(.4, 1), (-.4, -1), (.4, -1)], .05, (.4, 1), 2
loss = lambda w, z: (w * z[0] - z[1]) ** 2 / 2
def A(S):
    A_, b = sum(x * x for x, _ in S), sum(x * y for x, y in S)
    return b / (A_ + 2 * lam * len(S))
S_i = S[:i] + [z_new] + S[i + 1:]
w, w_i, z_i, m = A(S), A(S_i), S[i], len(S)
print(round(w, 4), round(w_i, 4))
change = loss(w_i, z_i) - loss(w, z_i)
print(round(loss(w_i, z_i), 4), round(loss(w, z_i), 4),
      round(change, 4))
lhs = lam * (w_i - w) ** 2
rhs = (change + loss(w, z_new) - loss(w_i, z_new)) / m
print(round(lhs, 4), round(rhs, 4))
```

It prints `0.5128 1.5385`, then `1.3047 0.7262 0.5786`; we will use its last line, `0.0526 0.2735`, a little later.

[[diagram:replace-one]]

The surprise is that this smell is not just a hint. It is an exact identity.

**Theorem (stable rules do not overfit).** Let \(\mathcal D\) be a distribution, let \(S = (z_1, \dots, z_m) \sim \mathcal D^m\), let \(z' \sim \mathcal D\) be independent of \(S\), and let \(i\) be drawn uniformly from \([m] = \{1, \dots, m\}\), independently of both. Then for every learner \(A\),

\[ \mathbb E_{S} \left[ L_{\mathcal D}(A(S)) - L_S(A(S)) \right] = \mathbb E_{(S, z') \sim \mathcal D^{m+1},\; i \sim U(m)} \left[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \right]. \]

We assume the learner is deterministic (a randomized learner can be handled by conditioning on its random bits) and that the expectations exist, which holds whenever the loss is bounded.

Read it from the inside out.

- \((S, z') \sim \mathcal D^{m+1}\): the \(m\) training examples and the extra one are \(m + 1\) independent draws from \(\mathcal D\).
- \(i \sim U(m)\): \(U(m)\) is the uniform distribution on \([m]\), so each position is replaced with probability \(1/m\).
- The expectation on the right averages the replace-one change over all three sources of randomness; the one on the left averages the true risk minus the empirical risk over \(S\) only.

In plain words: **on average, a learner’s true risk exceeds its training fit by exactly as much as its loss on an example grows when that example is swapped out.**

*Proof.* We compute the two expectations on the left separately, then subtract.

*The true risk.* Fix a training sequence \(S\). Then \(A(S)\) is a fixed hypothesis, and because \(z'\) is independent of \(S\), conditioning on \(S\) leaves \(z' \sim \mathcal D\). By the definition of the true risk, \(\mathbb E_{z'}[\ell(A(S), z')] = L_{\mathcal D}(A(S))\). Averaging over \(S\),

\[ \mathbb E_S[L_{\mathcal D}(A(S))] = \mathbb E_{S, z'}[\ell(A(S), z')]. \]

*The swap.* The \(m + 1\) examples \((z_1, \dots, z_m, z')\) are i.i.d. draws from \(\mathcal D\). The joint distribution of an i.i.d. sequence does not change when you exchange two of its positions, because it is a product of identical factors. Exchange position \(i\) with the last position. The training sequence \(S\) becomes \(S^{(i)}\), and the extra example \(z'\) becomes \(z_i\). So the pair \((S^{(i)}, z_i)\) has exactly the same distribution as the pair \((S, z')\), and therefore, for every fixed \(i\),

\[ \mathbb E_{S, z'}[\ell(A(S), z')] = \mathbb E_{S, z'}[\ell(A(S^{(i)}), z_i)]. \]

Since this holds for every \(i\), it also holds on average over \(i \sim U(m)\): \(\mathbb E_S[L_{\mathcal D}(A(S))] = \mathbb E_{S, z', i}[\ell(A(S^{(i)}), z_i)]\).

*The empirical risk.* By definition \(L_S(A(S)) = \frac1m \sum_{i=1}^m \ell(A(S), z_i)\), and an average over \(i \in [m]\) with weights \(1/m\) is an expectation over \(i \sim U(m)\). So \(\mathbb E_S[L_S(A(S))] = \mathbb E_{S, i}[\ell(A(S), z_i)]\), and adding \(z'\) to the expectation changes nothing since the expression does not involve it.

*Subtract.* By linearity of expectation, the difference of the two left-hand expectations is the expectation of the difference, which is the right-hand side. \(\blacksquare\)

Read it again: the expected overfitting gap *is* the expected replace-one change. Not bounded by it. Equal to it. A learner overfits, on average, exactly as much as it changes its mind about an example it has seen.

Here it is on the smallest learner we can think of. Take the **one-word world**: the campaign world with only the word “free”, so \(x = 0.4\,y\) with probability 0.8 and \(x = -0.4\,y\) otherwise, and the half square loss. Its true risk is \(L_{\mathcal D}(w) = \tfrac12(0.16 w^2 - 0.48 w + 1)\), by the same expansion as before. Every message has \(x^2 = 0.16\), so ERM is \(w = \sum_i x_i y_i / (0.16\,m)\). With only \(m = 5\) examples, it overfits. Measure both sides of the theorem:

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

It prints `0.128 0.128`. An exact computation over the number of agreeing examples gives \(\mathbb E[L_{\mathcal D}] = 0.384\) and \(\mathbb E[L_S] = 0.256\), a gap of exactly 0.128. The two estimates, computed from completely different quantities, land on the same number, as the theorem says they must.

This identity deserves a name for the case where the right side is small.

**Definition (on-average-replace-one stability).** Let \(\varepsilon : \mathbb N \to \mathbb R\) be a monotonically decreasing function. A learner \(A\) is **on-average-replace-one-stable with rate \(\varepsilon(m)\)** if for every distribution \(\mathcal D\),

\[ \mathbb E_{(S, z') \sim \mathcal D^{m+1},\; i \sim U(m)} \left[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \right] \le \varepsilon(m). \]

The rate plays the role of an accuracy, which is why it borrows the letter \(\varepsilon\). We will often just say **stable**. By the theorem, a learner stable with rate \(\varepsilon(m)\) has expected overfitting gap at most \(\varepsilon(m)\): stability and not overfitting are the same thing, on average.

In plain words: **a learner is stable with rate \(\varepsilon(m)\) when, for every distribution, swapping out one of \(m\) examples costs it at most \(\varepsilon(m)\) of loss on that example, on average.** “Monotonically decreasing” means the allowance never grows as \(m\) grows. For example, the next sections prove rates like \(2\rho^2/(\lambda m)\); with \(\rho = 0.4\), \(\lambda = 0.05\) and \(m = 20\) that is \(2 \cdot 0.16/1 = 0.32\), and doubling \(m\) halves it.

But not overfitting is not the same as learning. Take the learner that ignores its input and always returns \(\mathbf w = \mathbf 0\). It is perfectly stable: \(A(S^{(i)}) = A(S)\), so the replace-one change is exactly 0, and so is the overfitting gap. In the campaign world it has true risk 0.5 every night, with no swings at all. A useful learner must do two things at once: **fit** the training sequence, meaning low empirical risk, and be **stable**. The regularization weight \(\lambda\) is the dial between the two, and the rest of the lesson is about that dial.

## Strong convexity

Why is RLM stable? Because the penalty does more than make the objective convex. It makes it *strongly* convex, with a quadratic floor under every minimizer. Here is the definition.

Start with a picture you already have. Lesson 12 called a function convex when its graph lies below every chord, the straight segment between two points of the graph. A parabola like \(0.05\,w^2\) does better than that. Between \(w = 0\) and \(w = 1\), the chord at the midpoint has height 0.025, while the parabola there is only 0.0125: it sags below its chord by 0.0125. Stretch the chord to twice the width, from \(-1\) to 1, and the sag at the midpoint becomes 0.05, four times as much. A sag that grows with the square of the chord’s width is what “strongly” will mean.

**Definition (strongly convex function).** A function \(f\) is **\(\sigma\)-strongly convex**, for \(\sigma \ge 0\), if for all \(\mathbf w, \mathbf u\) and all \(\alpha \in (0, 1)\),

\[ f(\alpha \mathbf w + (1 - \alpha)\mathbf u) \le \alpha f(\mathbf w) + (1 - \alpha) f(\mathbf u) - \frac{\sigma}{2}\,\alpha(1 - \alpha)\,\|\mathbf w - \mathbf u\|^2. \]

Compare it with convexity from lesson 12, which asks the graph to lie below every chord. Strong convexity asks for more: below the chord *by a margin* that grows with the square of the distance between the endpoints. Every convex function is \(0\)-strongly convex, so this is a strengthening. (We write \(\sigma\) for the strength to keep \(\lambda\) for the regularization weight.)

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

What does this give us for RLM? If every loss \(\ell(\cdot, z)\) is convex, then \(L_S\) is convex (an average of convex functions, lesson 12), and by parts 1 and 2 the regularized objective \(f_S = L_S + \lambda\|\cdot\|^2\) is \(2\lambda\)-strongly convex. Two consequences follow straight away.

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

One example’s worth of disagreement can only move a \(2\lambda\)-strongly convex minimizer a little. In plain words: **\(\lambda\) times the squared movement of the minimizer is at most the two swapped examples’ disagreement, divided by \(m\).** In the three-example world from the replace-one section, the left side is \(0.05 \cdot (1.5385 - 0.5128)^2 \approx 0.0526\) and the right side is \(0.2735\): that was the last line, `0.0526 0.2735`, of that snippet.

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

The stability inequality bounds a movement by losses at the moved points. To turn it into a stability *rate*, we need to know how much a loss can change when its argument moves. Lesson 12 gave us two ways to say that: Lipschitzness and smoothness. Each gives its own proof.

## Lipschitz losses

Recall from lesson 12: \(\ell(\cdot, z)\) is **\(\rho\)-Lipschitz** if \(|\ell(\mathbf v, z) - \ell(\mathbf w, z)| \le \rho\|\mathbf v - \mathbf w\|\) for all \(\mathbf v, \mathbf w\). The hinge loss and the absolute loss on messages with \(\|\mathbf x\| \le \rho\) are examples.

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

Let’s watch it for the absolute loss \(|w x - y|\) in the one-word world. Its Lipschitz constant is \(|x| = 0.4\), so with \(\lambda = 0.05\) and \(m = 20\) the bound is the 0.32 we computed when we defined stability. The absolute loss has a kink, so there is no formula for the minimizer; we find it by ternary search, which works on any function of one variable with a single valley.

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

The half square loss of our campaign filter is not Lipschitz on all of \(\mathbb R^d\): its slope grows without limit as the score moves away from the label. It is **\(\beta\)-smooth**, though, meaning its gradient is \(\beta\)-Lipschitz, and nonnegative. Lesson 12 proved two facts about such functions that we now need.

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

*Step 4: take expectations.* Over \(S, z', i\), the first term on the right has mean \(\mathbb E[L_S(A(S))]\), as in the proof that stable rules do not overfit. So does the second: \(S^{(i)}\) is itself an i.i.d. sequence of \(m\) examples whose \(i\)th entry is \(z'\), so \(\ell(A(S^{(i)}), z')\) is distributed exactly like \(\ell(A(S), z_i)\). Therefore the right side has mean \(\frac{24\beta}{\lambda m}\mathbb E[L_S(A(S))]\).

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

So \(\lambda\) is a dial with a cost at each end. That is the **fitting-stability tradeoff**, and you have met its cousin before: lesson 5’s bias-complexity tradeoff, and lesson 12’s choice of the ball radius \(B\).

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

Read it from the inside out: \(\min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w)\) is the true risk of the best vector in the bounded class \(\mathcal H\) (lesson 12’s ball of radius \(B\)), and \(\rho B\sqrt{8/m}\) is the excess, which shrinks like one over the square root of \(m\). In plain words: **with \(\lambda\) set from \(\rho\), \(B\) and \(m\), RLM gets within \(\rho B\sqrt{8/m}\) of the best vector in the ball, on average.** With \(\rho = 1\), \(B = 0.7\) and \(m = 1{,}568\), the best \(\lambda\) is \(\sqrt{2/(0.49 \cdot 1568)} \approx 0.051\) and the excess is \(0.7\sqrt{8/1568} \approx 0.05\). The snippet minimizes \(\lambda B^2 + 2\rho^2/(\lambda m)\) over a grid and compares with the formulas:

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

That keeps half of lesson 12’s promise: every convex-Lipschitz-bounded problem, including halfspaces with the hinge loss, is learnable, by RLM, with a proof that never used uniform convergence. (Some convex-Lipschitz-bounded problems lack the uniform convergence property, so this route is genuinely new, not a detour.)

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
2. Rerun the campaign week with your `ridge` in place of `fit` and reproduce both tables.
3. Shrink \(\lambda\) toward 0 (0.01, 0.001, \(10^{-6}\)) on the starter data. What happens to the twin weights? Compare with lesson 9’s pseudoinverse solution, which split the weight of two always-together words evenly.
4. Measure the average replace-one change in the campaign world for \(\lambda \in \{0, 0.01, 0.1, 0.5\}\) with \(m = 20\), and check it against the true overfitting gap. Predict first which \(\lambda\) has the smallest gap, and whether that \(\lambda\) has the smallest production error.

## Fix the bug

Back to the campaign filter. The penalty steadied it, but the manager now asks for more than a steady dashboard. Before the filter goes into the nightly job for good, she wants a number: a certified bound on the expected production error, one that holds whatever the data look like, not a week that happened to go well.

You have the tool. In the campaign world the half square loss is convex, nonnegative and \(0.968\)-smooth, so for any \(\lambda \ge 2\beta/m\) the smooth oracle inequality certifies

\[ \mathbb E[\text{production error}] \le \left(1 + \frac{24 \cdot 0.968}{\lambda m}\right) \times \mathbb E[\text{fit}]. \]

The lab measures the expected fit by averaging the training fit over many nightly retrains, and it computes the factor from your \(\lambda\) and the label budget \(m\). Twenty labels a night was never going to certify anything: at \(m = 20\) and \(\lambda = 0.1\), the factor alone is about 12.6. So the manager has approved a bigger label budget, and set a target. Your job is to choose \(\lambda\) so that the certificate meets the target. Then the budget changes, and so does the target, and you do it again.

Before you start, write down a prediction: when the budget doubles, should the best \(\lambda\) go up, go down, or stay the same? Then use the oracle inequalities to check your answer before you touch the slider.

[[lab-act2]]

The fix holds for the hinge filter of lesson 12 as well. Its loss is not smooth, but it is Lipschitz, so the same penalty makes it stable with rate \(2\rho^2/(\lambda m)\), and the Lipschitz corollary tells you how to set \(\lambda\) from \(\rho\), \(B\) and \(m\). The weights stop jumping when a user corrects one label, and now you can say why.

## What people get wrong

**“Regularization is about keeping weights small because small weights are good.”** Small is not the goal. The penalty adds curvature \(2\lambda\) in every direction, which matters in the flat directions the data cannot pin down and barely matters elsewhere. What you buy is stability, and stability is exactly a bound on overfitting.

**“A stable learner is a good learner.”** The constant learner that always returns \(\mathbf 0\) is perfectly stable and has production error 0.5 in the campaign world. Stability bounds the gap between true and empirical risk; you still need a small empirical risk. A good learner fits *and* is stable.

**“More regularization is always safer.”** The stability term falls with \(\lambda\), but the fit term rises, for every training sequence. In the one-word table the certificate bottoms out near \(\lambda = 0.1\) and climbs again after.

**“The bound’s best \(\lambda\) is the truly best \(\lambda\).”** The bound is a worst case over all distributions. In the one-word world it prefers \(\lambda \approx 0.1\) while the true risk prefers the smallest \(\lambda\). Use the bound to certify; use a validation set (lesson 11) to tune.

**“These are high-probability guarantees.”** They are bounds on the *expected* true risk over training sequences. Any single night can be worse. The first problem below shows how to convert an expectation bound into a statement with probability at least \(1 - \delta\).

## Recap and the next bug

1. ERM can be **unstable**: when the empirical risk has a nearly flat valley, as with near-twin words, a few changed labels move the minimizer far, and the true risk pays for the large weights.
2. **Regularized loss minimization** minimizes \(L_S(\mathbf w) + R(\mathbf w)\). With **Tikhonov regularization** \(R(\mathbf w) = \lambda\|\mathbf w\|^2\) and the half square loss it is **ridge regression**, solved by \((2\lambda m I + A)\mathbf w = \mathbf b\) in \(O(md^2 + d^3)\) time; the system is always invertible.
3. **Stable rules do not overfit**: the expected overfitting gap equals the expected change in loss on \(z_i\) when \(z_i\) is replaced by a fresh example. A learner is **on-average-replace-one-stable** with rate \(\varepsilon(m)\) when that change is at most \(\varepsilon(m)\).
4. The penalty makes the objective **\(2\lambda\)-strongly convex**, which puts a quadratic floor \(\lambda\|\mathbf v - A(S)\|^2\) under the minimizer. That floor gives the **stability inequality**, and from it a rate of \(2\rho^2/(\lambda m)\) for convex \(\rho\)-Lipschitz losses and \(\frac{24\beta}{\lambda m}\mathbb E[L_S(A(S))]\) for convex, nonnegative, \(\beta\)-smooth losses with \(\lambda \ge 2\beta/m\).
5. \(\lambda\) trades **fit** against **stability**. The **oracle inequalities** turn the trade into guarantees: convex-Lipschitz-bounded problems are learnable with \(m \ge 8\rho^2B^2/\varepsilon^2\), convex-smooth-bounded ones with \(m \ge 150\beta B^2/\varepsilon^2\), and ridge regression learns with \(\lambda = \varepsilon/(3B^2)\).

The regularized filters ship, and the dashboard finally sits still. The manager is pleased for about a day. Then she looks at the retrain logs. The regularized problem is the right problem, but solving it exactly means reading the whole mail archive, every message, and the archive grows by thousands of labels a day. On the full archive, each exact retrain of the regularized hinge filter now takes hours, and the spammers change their campaign faster than that. Every method you know touches all \(m\) examples before it can say anything. Is there a learner whose cost per step does not depend on the size of the archive at all, one that looks at a single message, nudges the weights, and moves on, and still comes with a guarantee? That is the next lesson: stochastic gradient descent.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The constant learner always returns \(\mathbf w = \mathbf 0\). What is its on-average-replace-one stability rate, and what is its expected overfitting gap? In the campaign world, what is its expected true risk? Now take ridge regression on a fixed training sequence and let \(\lambda \to \infty\): what does \(A(S)\) tend to, and what happens to the fit term and the stability term of the smooth oracle inequality?
2. The proof that stable rules do not overfit rests on the step \(\mathbb E[\ell(A(S), z')] = \mathbb E[\ell(A(S^{(i)}), z_i)]\). Explain exactly which property of the \(m + 1\) examples makes it true. Then suppose the replacement \(z'\) were drawn from a different distribution \(\mathcal D'\), say junk messages only, while \(S \sim \mathcal D^m\). Which step of the proof breaks, and what does the right-hand side of the identity measure instead?
3. Messages lie in the unit ball and labels in \([-1, 1]\). For the absolute loss with \(\mathcal H = \{\|\mathbf w\| \le 2\}\) and \(\varepsilon = 0.1\), how many examples does the convex-Lipschitz-bounded corollary ask for, and with which \(\lambda\)? For ridge regression with the same \(B\) and \(\varepsilon\)? Check that the ridge choice satisfies \(\lambda \ge 2\beta/m\).

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (from an expected risk to a high-probability guarantee).** A learner \(A\) guarantees: if \(m \ge m_{\mathcal H}(\varepsilon)\), then for every distribution \(\mathcal D\), \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S))] \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). (a) Show that for every \(\delta \in (0, 1)\), if \(m \ge m_{\mathcal H}(\varepsilon\delta)\), then with probability at least \(1 - \delta\), \(L_{\mathcal D}(A(S)) \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). (b) Assuming the loss is bounded by 1, design a procedure that reaches the same kind of guarantee with about \(m_{\mathcal H}(\varepsilon/2)\lceil\log_2(1/\delta)\rceil + \frac{\log(4/\delta) + \log\lceil\log_2(1/\delta)\rceil}{\varepsilon^2}\) examples, a far milder dependence on \(\delta\) than part (a). (Use lesson 11.)
- **Problem 2 (stability plus nearly minimizing the empirical risk is enough).** Call a learner an **asymptotic empirical risk minimizer** with rate \(\varepsilon_2(m)\) if for every \(\mathcal D\), \(\mathbb E_{S \sim \mathcal D^m}[L_S(A(S)) - \min_{h \in \mathcal H} L_S(h)] \le \varepsilon_2(m)\), and say it **learns** \(\mathcal H\) with rate \(\varepsilon(m)\) if \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D}(h)] \le \varepsilon(m)\) for every \(\mathcal D\). Prove: if \(A\) is on-average-replace-one-stable with rate \(\varepsilon_1(m)\) and an asymptotic empirical risk minimizer with rate \(\varepsilon_2(m)\), then it learns \(\mathcal H\) with rate \(\varepsilon_1(m) + \varepsilon_2(m)\).
- **Problem 3 (other norms).** Replace \(\|\cdot\|\) everywhere by an arbitrary norm, and define strong convexity with respect to it by the same inequality. (a) Show that parts 2 and 3 of the strong convexity lemma still hold. (b) Let \(R\) be \(2\lambda\)-strongly convex with respect to this norm, let \(A(S) = \operatorname{argmin}_{\mathbf w}(L_S(\mathbf w) + R(\mathbf w))\), and assume every loss is convex and \(\rho\)-Lipschitz with respect to the same norm: \(\ell(\mathbf w, z) - \ell(\mathbf v, z) \le \rho\|\mathbf w - \mathbf v\|\). Prove that \(A\) is on-average-replace-one-stable with rate \(2\rho^2/(\lambda m)\).
