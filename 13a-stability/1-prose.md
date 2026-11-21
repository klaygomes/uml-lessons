# Regularization and Stability

## The filter that would not sit still

Lesson 12b ended with an uneasy feeling. A user corrected one label, someone retrained the hinge filter out of habit, and the new weights looked nothing like the old ones. The bound was fine, because it never promised that the output would sit still. But the feeling turned into a bug report within the week.

The team now retrains every convex filter every night. The newest one scores messages from a marketing campaign, and it is the junk score of lesson 9b: a weight vector \(\mathbf w\), a score \(\langle \mathbf w, \mathbf x \rangle\), and the square loss, which lesson 12a showed is convex and smooth. Each night it is trained on the twenty most recent labelled campaign messages, so from one night to the next only a handful of labels change: three old ones drop out, three new ones come in.

Here is what the dashboard showed for one week. The weight on the word “free” went from 2.2 to −8.2, back to −2.2, then to −5.9. The weight on “offer” jumped from −0.3 to 11.8. And the production error, the filter’s average loss on fresh campaign messages, swung from 0.384 to 0.632 and back to 0.387 in three nights. Three labels out of twenty changed each night. Nobody would call that a big change in the data. Yet the learner’s output jumped around like a nervous bird, and its true quality jumped with it.

What is going on? The learner did exactly what it was told: it minimized the empirical risk, perfectly, every night. Nothing is broken in the optimization, and lesson 12a promised us the problem is convex. So why does a tiny change in the input make such a big change in the output, and why does that hurt the true risk? And the deeper question, the one lesson 12b left hanging: could a learner that *refuses* to jump around generalize for that very reason?

This lesson answers both. The fix is one extra term in the objective. The theory behind it is a new way to see that a learner does not overfit, one that never mentions uniform convergence, and lesson 13b turns it into a proof.

## A week of retrains

Let’s build a world small enough to compute everything in it, and big enough to show the bug. Call it the **campaign world**.

The campaign uses five words that always travel together: “free”, “gift”, “prize”, “bonus” and “offer”. A message is a vector \(\mathbf x \in \mathbb R^5\), one coordinate per word, scaled so that each coordinate is about \(\pm 0.4\). Labels are \(y = +1\) for junk and \(y = -1\) for keep, as in lesson 9a. Here is how a message is drawn:

1. Flip a fair coin for the label \(y \in \{+1, -1\}\).
2. Draw the campaign signal \(f = 0.4\,y\) with probability 0.8 and \(f = -0.4\,y\) with probability 0.2. So 80% of the time the campaign words point the right way.
3. Each word’s coordinate is the signal plus its own tiny jitter: \(x_j = f + e_j\), where each \(e_j\) is \(+0.04\) or \(-0.04\) with equal probability, independently of everything else.

The five words are near-twins. They carry the same information, \(f\), plus a little noise that has nothing to do with the label. Every message has \(\|\mathbf x\| \le \sqrt{5 \cdot 0.44^2} \approx 0.98\), so it lies in the unit ball.

The loss we use in this lesson is the **half square loss**,

\[ \ell(\mathbf w, (\mathbf x, y)) = \tfrac12 \left( \langle \mathbf w, \mathbf x \rangle - y \right)^2, \]

the square loss of lesson 9b with a factor \(\tfrac12\) in front. The half changes nothing about which \(\mathbf w\) is best, and it makes the constants below come out cleaner.

Read it from the inside out.

- \(\mathbf w = (w_1, \dots, w_5)\) is the weight vector, one weight per word, and \(\mathbf x\) is the message’s vector of word coordinates. The pair \((\mathbf x, y)\) is one example \(z\), as in every lesson since lesson 1.
- \(\langle \mathbf w, \mathbf x \rangle = \sum_j w_j x_j\) is the inner product from lesson 9a: multiply each weight by its word’s coordinate and add. It is the filter’s **score** for the message.
- \(\langle \mathbf w, \mathbf x \rangle - y\) is the miss: how far the score lands from the label \(+1\) or \(-1\).
- Squaring makes every miss count positively, and bigger misses count much more; the \(\tfrac12\) halves the result.
- \(\ell(\mathbf w, z)\) is lesson 3’s notation for “the loss of hypothesis \(\mathbf w\) on example \(z\)”, one example at a time.

In plain words: **the loss is half the squared distance between the filter’s score and the message’s label.** A junk message (\(y = 1\)) whose five coordinates are 0.44, 0.36, 0.44, 0.44, 0.36, scored by \(\mathbf w = (0.3, \dots, 0.3)\), gets score \(0.3 \times 2.04 = 0.612\), misses by \(-0.388\), and pays \(\tfrac12 \cdot 0.388^2 \approx 0.0753\).

In this world the true risk has a formula. Expand the square and take expectations term by term, where \(\mathbb E[\cdot]\) means “the average value of” over random messages. For two different words, \(\mathbb E[x_j x_k] = \mathbb E[f^2] = 0.16\), since the jitters are independent of each other and of \(f\) and have mean 0. For the same word, \(\mathbb E[x_j^2] = 0.16 + 0.04^2 = 0.1616\). And \(\mathbb E[y x_j] = \mathbb E[y f] = 0.4 \cdot (0.8 - 0.2) = 0.24\). Writing \(s = \sum_j w_j\) for the total weight and \(q = \|\mathbf w\|^2\),

\[ L_{\mathcal D}(\mathbf w) = \tfrac12 \left( 0.16\, s^2 + 0.0016\, q - 0.48\, s + 1 \right). \]

Read it from the inside out.

- \(s = \sum_j w_j\) is the total weight, the sum of the five weights; \(q = \|\mathbf w\|^2 = \sum_j w_j^2\) is the squared length of the weight vector (lesson 9a’s Euclidean norm, squared).
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

Now the nightly retrain. Start with twenty labelled messages, fit by ERM (which is exactly least squares from lesson 9b), and each night replace three of the twenty with fresh ones. Before you look at the table, a prediction: how much do you expect the weights to move when 3 of 20 examples change? (The code that produced this table is in the ridge regression section, where it runs two learners side by side on the same messages.)

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

How do you tell the learner to stop sliding along the valley? You could restrict the hypothesis class, as lesson 1 taught, and allow only weight vectors with small norm. Lesson 12b did exactly that with its bounded classes \(\|\mathbf w\| \le B\). Here is a softer version of the same idea: let the learner go anywhere, but charge it for the size of the weights.

Put a number on it. On night 2 the ERM weights had squared length \(\sum_j w_j^2 \approx 389\), while the best weights, 0.3 on every word, have squared length \(5 \times 0.09 = 0.45\). Suppose every unit of squared length cost the learner 0.1 on top of its training loss. Night 2’s weights would then pay about 38.9, a hundred times more than any training loss in this world, so the learner would never choose them; the best weights would pay only 0.045. The learner keeps minimizing a training score, but the score now includes a price for size.

**Definition (regularized loss minimization).** A **regularization function** is a map \(R : \mathbb R^d \to \mathbb R\). The **regularized loss minimization** rule, RLM for short, returns a hypothesis in

\[ \operatorname*{argmin}_{\mathbf w} \left( L_S(\mathbf w) + R(\mathbf w) \right). \]

The regularization function measures, intuitively, how “complex” a hypothesis is, and RLM balances a low empirical risk against a “simpler” hypothesis. Many choices of \(R\) are possible, each reflecting some prior belief about the problem, much as the description language did for minimum description length in lesson 7b. This lesson uses one of the simplest:

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

**As structural risk minimization.** Lesson 7a chose among nested classes by trading empirical risk against a complexity charge. Define \(\mathcal H_1 \subset \mathcal H_2 \subset \mathcal H_3 \subset \dots\) with \(\mathcal H_i = \{\mathbf w : \|\mathbf w\|^2 \le i\}\). If the number of examples each \(\mathcal H_i\) needs grows with \(i\), then charging \(\lambda \|\mathbf w\|^2\) is a smooth version of “pay more for a bigger class”, and RLM behaves like SRM over this sequence.

**As a stabilizer.** The penalty makes the minimizer sit still when the data move a little. That is the reading this lesson and lesson 13b develop, and it leads to a proof that the learner does not overfit. But first, let’s see the rule work on the campaign.

## Ridge regression

Apply RLM with Tikhonov regularization to linear regression with the half square loss:

\[ A(S) = \operatorname*{argmin}_{\mathbf w \in \mathbb R^d} \left( \lambda \|\mathbf w\|^2 + \frac1m \sum_{i=1}^m \tfrac12 \left( \langle \mathbf w, \mathbf x_i \rangle - y_i \right)^2 \right). \]

Linear regression solved this way is called **ridge regression**.

To solve it, do what lesson 9b did for least squares: set the gradient to zero. The gradient of \(\lambda \|\mathbf w\|^2\) is \(2\lambda \mathbf w\). The gradient of one loss term is \((\langle \mathbf w, \mathbf x_i \rangle - y_i)\,\mathbf x_i\), by the chain rule; the half cancels the 2 from the square. So

\[ \nabla f_S(\mathbf w) = 2\lambda \mathbf w + \frac1m \sum_{i=1}^m (\langle \mathbf w, \mathbf x_i \rangle - y_i)\,\mathbf x_i. \]

Multiply by \(m\) and use the same matrices as lesson 9b, \(A = \sum_{i=1}^m \mathbf x_i \mathbf x_i^\top\) and \(\mathbf b = \sum_{i=1}^m y_i \mathbf x_i\). Since \(\sum_i \mathbf x_i \langle \mathbf x_i, \mathbf w \rangle = A\mathbf w\), the condition \(\nabla f_S(\mathbf w) = \mathbf 0\) becomes the linear system

\[ (2\lambda m I + A)\,\mathbf w = \mathbf b, \]

where \(I\) is the \(d \times d\) identity matrix. (The matrix \(A\) and the learner \(A(S)\) share a letter; the learner always carries its argument.)

Read the system from the inside out.

- \(\mathbf x_i \mathbf x_i^\top\) is the \(d \times d\) matrix whose entry \((j, k)\) is \(x_{i,j} x_{i,k}\) (lesson 9b); \(A\) adds them up over all \(m\) examples, so entry \((j, k)\) of \(A\) is \(\sum_i x_{i,j}x_{i,k}\), how often words \(j\) and \(k\) light up together.
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

It remains to see that the solution is a minimizer, not just a flat point. Expand the objective around any \(\mathbf w\) with a step \(\boldsymbol\Delta\), exactly as lesson 9b did:

\[ f_S(\mathbf w + \boldsymbol\Delta) = f_S(\mathbf w) + \langle \nabla f_S(\mathbf w), \boldsymbol\Delta \rangle + \frac{1}{2m}\sum_{i=1}^m \langle \boldsymbol\Delta, \mathbf x_i \rangle^2 + \lambda \|\boldsymbol\Delta\|^2. \]

This is exact: \(\lambda\|\mathbf w + \boldsymbol\Delta\|^2 = \lambda\|\mathbf w\|^2 + 2\lambda\langle \mathbf w, \boldsymbol\Delta \rangle + \lambda\|\boldsymbol\Delta\|^2\), and each loss term is a parabola whose expansion has no higher terms. At the solution the gradient vanishes, so \(f_S(\mathbf w + \boldsymbol\Delta) - f_S(\mathbf w) \ge \lambda\|\boldsymbol\Delta\|^2\), which is strictly positive for every \(\boldsymbol\Delta \ne \mathbf 0\). So the solution is the unique minimizer. \(\blacksquare\)

Two remarks. Lesson 9b had to reach for the pseudoinverse when \(A\) was singular, for instance when two words always appeared together. Ridge never needs it: \(2\lambda m\) lifts every eigenvalue off zero. And the last line of the proof, “the objective rises at least \(\lambda\|\boldsymbol\Delta\|^2\) as you leave the minimizer”, is the property the whole stability argument will rest on. Hold on to it.

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

where \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) examples drawn i.i.d. from \(\mathcal D\)” (lesson 2), and \(\mathbb E_{S \sim \mathcal D^m}\) averages over those draws. Our guarantees will be about expected risks, not “with probability at least \(1 - \delta\)”. A problem at the end of lesson 13b shows how to turn one into the other.

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
w, w_i, z_i = A(S), A(S_i), S[i]
print(round(w, 4), round(w_i, 4))
change = loss(w_i, z_i) - loss(w, z_i)
print(round(loss(w_i, z_i), 4), round(loss(w, z_i), 4),
      round(change, 4))
```

It prints `0.5128 1.5385`, then `1.3047 0.7262 0.5786`.

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

In plain words: **a learner is stable with rate \(\varepsilon(m)\) when, for every distribution, swapping out one of \(m\) examples costs it at most \(\varepsilon(m)\) of loss on that example, on average.** “Monotonically decreasing” means the allowance never grows as \(m\) grows. For example, lesson 13b proves rates like \(2\rho^2/(\lambda m)\); with \(\rho = 0.4\), \(\lambda = 0.05\) and \(m = 20\) that is \(2 \cdot 0.16/1 = 0.32\), and doubling \(m\) halves it.

But not overfitting is not the same as learning. Take the learner that ignores its input and always returns \(\mathbf w = \mathbf 0\). It is perfectly stable: \(A(S^{(i)}) = A(S)\), so the replace-one change is exactly 0, and so is the overfitting gap. In the campaign world it has true risk 0.5 every night, with no swings at all. A useful learner must do two things at once: **fit** the training sequence, meaning low empirical risk, and be **stable**. The regularization weight \(\lambda\) is the dial between the two, and the fix at the end of this lesson is about setting that dial.

## Fix the bug

Back to the campaign filter. Ridge with \(\lambda = 0.1\) had a good week, but, as the ridge regression section admitted, “it worked this week” is not a guarantee. Before the filter goes into the nightly job for good, the manager wants two promises, and she wants them about the average night, not about any one night:

1. The morning dashboard shows the training fit \(L_S(A(S))\). It should flatter the filter by at most 0.03 on average: an expected overfitting gap of at most 0.03.
2. The expected production error should be at most 0.34.

Neither number can be read off the dashboard, because the production error needs the whole distribution. But the theorem hands you both. The gap *is* the expected replace-one change, and so the expected production error splits into two pieces you can measure on training data and swapped-in examples alone:

\[ \mathbb E_S[L_{\mathcal D}(A(S))] = \mathbb E_S[L_S(A(S))] + \mathbb E_{S, z', i}\left[ \ell(A(S^{(i)}), z_i) - \ell(A(S), z_i) \right]. \]

Read it from the inside out.

- On the left, \(L_{\mathcal D}(A(S))\) is the production error of the filter trained on \(S\), and \(\mathbb E_S\) averages it over training sequences: the expected production error.
- The first term on the right, \(\mathbb E_S[L_S(A(S))]\), is the **expected fit**: the training loss of the learner’s own output, averaged over training sequences.
- The second term is the expected replace-one change, averaged over \(S\), the fresh example \(z'\) and the position \(i\), exactly as in the theorem.
- Why it holds: write \(L_{\mathcal D}(A(S)) = L_S(A(S)) + (L_{\mathcal D}(A(S)) - L_S(A(S)))\), take expectations, and let the theorem replace the expected gap by the expected replace-one change.

In plain words: **the expected production error is the expected fit plus the expected replace-one change.** For ERM in the campaign world, the fit averages about 0.240 and the replace-one change about 0.184; together 0.424, while the formula for the true risk, averaged over the same training sequences, gives 0.425. The snippet measures all three over 20,000 independent training sequences of twenty, with `train` the same solver as `fit` in the ridge regression section:

```python
import random

def draw(rng):
    y = rng.choice((1, -1))
    f = .4 * y * rng.choice((1, 1, 1, 1, -1))
    return [f + rng.choice((.04, -.04)) for _ in range(5)], y

def train(S, lam):
    X, k = [x + [y] for x, y in S], 2 * lam * len(S)
    M = [[sum(r[i] * r[j] for r in X) + k * (i == j)
          for j in range(6)] for i in range(5)]
    for c in range(5):
        M[c] = [v / M[c][c] for v in M[c]]
        for r in set(range(5)) - {c}:
            M[r] = [v - M[r][c] * p for v, p in zip(M[r], M[c])]
    return [row[5] for row in M]

loss = lambda w, z: (sum(a * b for a, b in zip(w, z[0]))
                     - z[1]) ** 2 / 2
L_D = lambda w: (.16 * sum(w) ** 2 + .0016 * sum(v * v for v in w)
                 - .48 * sum(w) + 1) / 2
rng, m, N, lam = random.Random(1), 20, 20_000, 0
fit = swap = prod = 0
for _ in range(N):
    S, z = [draw(rng) for _ in range(m)], draw(rng)
    i = rng.randrange(m)
    w, w_i = train(S, lam), train(S[:i] + [z] + S[i + 1:], lam)
    fit += sum(loss(w, s) for s in S) / m
    swap += loss(w_i, S[i]) - loss(w, S[i])
    prod += L_D(w)
print(round(fit / N, 3), round(swap / N, 3))
print(round((fit + swap) / N, 3), round(prod / N, 3))
```

It prints `0.24 0.184`, then `0.424 0.425`. ERM fails both promises: its gap is six times the allowance, and its production error is far above 0.34.

The lab runs the same measurement for the \(\lambda\) you choose: it averages the fit and the replace-one change over 20,000 simulated nights of twenty labels, each with one fresh message swapped in, and shows the two promises next to their targets, with the formula’s production error beside them as a check. Success needs both averages inside their targets. Tonight’s production error is shown too, but it never counts: one lucky night proves nothing.

Before you touch the slider, a prediction. If you push \(\lambda\) all the way up to 1, what happens to the replace-one change, and what happens to the expected production error? Write both down, then find a \(\lambda\) that keeps both promises.

[[lab-act2]]

Look at what your fix was built from: an average fit, an average replace-one change, and one identity. Nothing about the size of the hypothesis class, nothing about uniform convergence. The same identity holds for the hinge filter of lesson 12b, or for any learner at all: measure how much it changes its mind about an example it has seen, and you have measured how much it overfits.

## What people get wrong

**“Regularization is about keeping weights small because small weights are good.”** Small is not the goal. The penalty adds curvature \(2\lambda\) in every direction, which matters in the flat directions the data cannot pin down and barely matters elsewhere. What you buy is stability, and stability is exactly a bound on overfitting.

**“A stable learner is a good learner.”** The constant learner that always returns \(\mathbf 0\) is perfectly stable and has production error 0.5 in the campaign world. Stability bounds the gap between true and empirical risk; you still need a small empirical risk. A good learner fits *and* is stable.

**“Stability is only a hint that a learner generalizes.”** On average it is not a hint at all; it is an identity. In the one-word world the expected overfitting gap and the expected replace-one change both came out at 0.128, and in the campaign world the fit plus the replace-one change matched the production error to within the simulation’s noise. A learner overfits, on average, exactly as much as it is unstable.

## Recap and the next bug

1. ERM can be **unstable**: when the empirical risk has a nearly flat valley, as with near-twin words, a few changed labels move the minimizer far, and the true risk pays for the large weights.
2. **Regularized loss minimization** minimizes \(L_S(\mathbf w) + R(\mathbf w)\). With **Tikhonov regularization** \(R(\mathbf w) = \lambda\|\mathbf w\|^2\) and the half square loss it is **ridge regression**, solved by \((2\lambda m I + A)\mathbf w = \mathbf b\); the system is always invertible, and the objective rises at least \(\lambda\|\boldsymbol\Delta\|^2\) as you leave its minimizer.
3. **Stable rules do not overfit**: the expected overfitting gap equals the expected change in loss on \(z_i\) when \(z_i\) is replaced by a fresh example. A learner is **on-average-replace-one-stable** with rate \(\varepsilon(m)\) when that change is at most \(\varepsilon(m)\).
4. So the expected production error is the expected fit plus the expected replace-one change. A useful learner needs both small, and \(\lambda\) is the dial between them: too small and the learner is unstable, too large and it stops fitting.

The ridge filter ships with its two promises kept, and the dashboard finally sits still. Then the auditor reads your certificate and asks where the 20,000 nights came from. From the simulator: a program that draws campaign messages by flipping the campaign’s own coins. Your promises are facts about the simulator. The real mail room gets one night at a time, the real campaign does not come with its coins, and next month’s campaign will be different again. Is there a way to know that the penalty makes the learner stable *before* seeing the distribution, a proof that holds for every distribution at once, with \(\lambda\) chosen from numbers you actually know? That is lesson 13b: strong convexity.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The constant learner always returns \(\mathbf w = \mathbf 0\). What is its on-average-replace-one stability rate, and what is its expected overfitting gap? In the campaign world, what is its expected true risk? Now take ridge regression on a fixed training sequence and let \(\lambda \to \infty\): what does \(A(S)\) tend to, what happens to the fit and to the replace-one change, and which of the manager’s two promises fails?
2. The proof that stable rules do not overfit rests on the step \(\mathbb E[\ell(A(S), z')] = \mathbb E[\ell(A(S^{(i)}), z_i)]\). Explain exactly which property of the \(m + 1\) examples makes it true. Then suppose the replacement \(z'\) were drawn from a different distribution \(\mathcal D'\), say junk messages only, while \(S \sim \mathcal D^m\). Which step of the proof breaks, and what does the right-hand side of the identity measure instead?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (stability plus nearly minimizing the empirical risk is enough).** Call a learner an **asymptotic empirical risk minimizer** with rate \(\varepsilon_2(m)\) if for every \(\mathcal D\), \(\mathbb E_{S \sim \mathcal D^m}[L_S(A(S)) - \min_{h \in \mathcal H} L_S(h)] \le \varepsilon_2(m)\), and say it **learns** \(\mathcal H\) with rate \(\varepsilon(m)\) if \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S)) - \min_{h \in \mathcal H} L_{\mathcal D}(h)] \le \varepsilon(m)\) for every \(\mathcal D\). Prove: if \(A\) is on-average-replace-one-stable with rate \(\varepsilon_1(m)\) and an asymptotic empirical risk minimizer with rate \(\varepsilon_2(m)\), then it learns \(\mathcal H\) with rate \(\varepsilon_1(m) + \varepsilon_2(m)\).
