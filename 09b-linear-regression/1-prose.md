# Linear and Logistic Regression

## The score nobody can vouch for

Lesson 9a ended with the halfspace filter in production and the facilities team getting its lunch invitations again. It also ended with a request. The new inbox sorts messages by how suspicious they are, and for that it needs a **junk score**: the percentage of recipients who click “report junk” on a message. The inbox had a launch date, so a teammate shipped the score with a rule tuned by hand: “each free is worth 20 points”.

Monday brings the screenshots. A message with three “free”s shows a junk score of 60, and 70% of its recipients reported it. A message with one “free” shows 20, and only 10% did. Nothing is wildly off, but nothing is right either, and the product manager asks two questions you cannot answer yet. Is there a better line? And if you hand one over, can you prove that no line does better?

The teammate’s defence is honest: “I tried a hundred settings, and this one had the smallest average miss.” But a hundred settings out of infinitely many proves nothing about the hundred-and-first. You met this wall at the start of lesson 9a, where brute force over real-valued weights was hopeless. The way past it there was a certificate: a separator whose margin bounded the Perceptron’s updates, checked on the labelled messages before training started. This lesson finds the same kind of certificate for a score, a property you can check on your weights, using only the messages you have, that proves no other line has a smaller risk. Then, because the designers want it too, the same linear score will give a probability of junk.

## What you need from lesson 9a

This lesson leans on a handful of ideas from lesson 9a. Here they are in one place, each with a number you will see again.

- **Domain and labels.** From lesson 1, \(\mathcal X\) is the domain set, every message that could arrive, and \(\mathcal Y\) is the label set, every answer the learner may give. For halfspaces, \(\mathcal X = \mathbb R^d\) and \(\mathcal Y = \{-1, +1\}\).
- **Word counts.** A message is a vector \(\mathbf x \in \mathbb R^d\), a list of \(d\) real numbers: entry \(x_j\) counts word \(j\) of a vocabulary of \(d\) words, so “free trial, book a meeting” on the vocabulary (“free”, “meeting”) is \((2, 1)\). With several messages, \(\mathbf x_i\) is message \(i\), \(x_{i,j}\) is its entry \(j\), and \(y_i\) is its label.
- **Inner product.** \(\langle \mathbf w, \mathbf x \rangle = \sum_{i=1}^d w_i x_i\) multiplies matching entries and adds them. For \(\mathbf w = (2, -2)\) and \(\mathbf x = (1, 2)\) it is \(2 - 4 = -2\).
- **Affine functions and the bias term.** \(L_d = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle + b \}\) is the set of every score you can build from \(d\) weights and one constant \(b\), the **bias term** (nothing to do with the inductive bias of lesson 1); the arrow \(\mapsto\) reads “is sent to”. With \(\mathbf w = (2, -2)\) and \(b = -1\), the message \((1, 2)\) scores \(-3\).
- **The constant 1.** Gluing \(b\) onto the front of the weights and a 1 onto the front of every message leaves every score unchanged, so we usually write just \(\langle \mathbf w, \mathbf x \rangle\), with the 1 already inside \(\mathbf x\). That is the **homogeneous** form: \((-1, 2, -2)\) with \((1, 1, 2)\) gives \(-1 + 2 - 4 = -3\) again.
- **Halfspaces and the signed score.** \(HS_d = \operatorname{sign} \circ L_d\): score the message, then take the sign, where \(\operatorname{sign}(a)\) is \(+1\) when \(a > 0\) and \(-1\) otherwise, and the circle \(\circ\) means “do one function, then feed its answer to the next”. Labels are \(+1\) for junk and \(-1\) for keep, so the **signed score** \(y \langle \mathbf w, \mathbf x \rangle\), the score times the label, is positive exactly when the halfspace is right.
- **Length.** \(\|\mathbf x\| = \sqrt{x_1^2 + \dots + x_d^2}\), the straight-line distance from the origin, so \(\|\mathbf x\|^2 = \langle \mathbf x, \mathbf x \rangle\); the vector \((1, 1, 2)\) has squared length 6.
- **Matrices and linear programs.** A matrix is a table of numbers, \(m \times d\) for \(m\) rows and \(d\) columns, and \(A \mathbf w\) lists the inner products of its rows with \(\mathbf w\). A **linear program** makes a weighted sum of the unknowns as large as possible subject to “at least” conditions \(A \mathbf w \ge \mathbf v\), and standard solvers handle it efficiently.
- **ERM, separable data and the Perceptron.** ERM (lesson 2) returns a hypothesis with the smallest empirical risk; under the 0-1 loss a mistake costs 1 and a right answer 0. A training sequence is **separable** when some halfspace gets every example right. The **Perceptron** starts from zero weights and, on each mistake, adds \(y_i \mathbf x_i\) to the weights.
- **The eight messages.** Lesson 9a’s running example, with \(x_1\) counting “free” and \(x_2\) counting “meeting”. The weights \((-1, 2, -2)\), bias term first, separate them, with signed scores 5, 1, 3, 1, 3, 3, 3 and 1.

| message | gist | “free” \(x_1\) | “meeting” \(x_2\) | label \(y\) |
|---|---|---|---|---|
| A | FREE gift, FREE delivery, FREE | 3 | 0 | junk, \(+1\) |
| B | free trial, book a meeting with sales | 2 | 1 | junk, \(+1\) |
| C | free free free … your free meeting with an advisor | 4 | 2 | junk, \(+1\) |
| D | claim your free prize | 1 | 0 | junk, \(+1\) |
| E | free lunch at the meeting, meeting room 4 | 1 | 2 | keep, \(-1\) |
| F | meeting moved to Tuesday | 0 | 1 | keep, \(-1\) |
| G | free coffee at the meeting; agenda for the meeting; meeting notes | 2 | 3 | keep, \(-1\) |
| H | see you tomorrow | 0 | 0 | keep, \(-1\) |

## A junk score: linear regression

Let’s say the product manager’s request in the course’s language. Each message comes with a number, its junk score, and the learner has to predict that number.

That is a **regression** problem: the label set is \(\mathcal Y = \mathbb R\), not \(\{\pm 1\}\). Linear regression is a common tool for modelling how a real-valued outcome depends on some explanatory variables. Its hypothesis class is just the affine functions, with no sign on top:

\[ \mathcal H_{\text{reg}} = L_d = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle + b \; : \; \mathbf w \in \mathbb R^d, \, b \in \mathbb R \}. \]

This is the same set \(L_d\) we read slowly in lesson 9a, now given the name \(\mathcal H_{\text{reg}}\), “the hypothesis class for regression”: **every weighted sum plus a constant, used directly as the prediction.**

Here are six messages with their junk scores, using one word, the count of “free”. Notice the shape before any fitting: the score climbs slowly, then steeply, then flattens out near 100, because past a point every extra “free” convinces almost nobody new.

| “free” count \(x\) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| junk score \(y\) | 2 | 10 | 35 | 70 | 88 | 95 |

The 0-1 loss is useless here. If a message’s true score is 30, then predicting 30.001 and predicting 90 are both “wrong”, but one is clearly better. So we need a loss that grows with the size of the miss. Take the teammate’s guess, “each free is worth 20 points”, \(h(x) = 20x\). On the message with three “free”s it predicts 60 against a true 70, a miss of \(-10\); squaring the miss gives a cost of 100, while a miss of 2 would cost only 4. That is the **square loss** you met in lesson 3,

\[ \ell(h, (\mathbf x, y)) = (h(\mathbf x) - y)^2. \]

The letter \(\ell\) names the loss; its inputs are the hypothesis \(h\) and one labelled example \((\mathbf x, y)\). Inside out: \(h(\mathbf x)\) is the prediction, subtracting \(y\) gives the miss, and the exponent 2 squares it, so misses in either direction cost the same and big misses cost much more. In plain words: **the cost of a prediction is the square of how far it missed.** With it, recall from lesson 1 that \(L_S(h)\) is the empirical risk of \(h\) on the training sequence \(S\), the average loss over the examples; here it is the average squared miss,

\[ L_S(h) = \frac{1}{m} \sum_{i=1}^m (h(\mathbf x_i) - y_i)^2, \]

often called the mean squared error. Inside out: one squared miss per example \(i\), the sum \(\sum_{i=1}^m\) adds all \(m\) of them, and \(\frac1m\) divides by how many there are. In plain words: **the empirical risk is the average squared miss over the labelled messages.** For \(h(x) = 20x\) the misses are \(-2, 10, 5, -10, -8, 5\), the squares add to 318, and \(318 / 6 = 53\):

```python
xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
h = lambda x: 20 * x
losses = [(h(x) - y) ** 2 for x, y in zip(xs, ys)]
print(losses, sum(losses) / len(xs))
```

It prints `[4, 100, 25, 100, 64, 25] 53.0`. Other losses are possible. The **absolute loss** \(|h(\mathbf x) - y|\), where the bars now mean “size without the sign” (a miss of \(-10\) costs 10), is one, and ERM with it can again be written as a linear program; that is Problem 1 at the end of this lesson.

What does lesson 6 say about how many examples we need? Nothing, directly: the VC dimension is defined for classes with two labels. One honest route is the discretization trick from lesson 4. If every weight and the bias term are stored as 64-bit floats, the class is finite, with at most \(2^{64(d+1)}\) members, and the finite-class bounds apply, provided the loss is bounded (scores between 0 and 100, with predictions clipped to that range, would do). Later lessons give sharper tools for regression.

## Least squares

**Least squares** is the algorithm that solves ERM for linear regression with the square loss. In homogeneous form (with the constant 1 inside \(\mathbf x\)), it looks for

\[ \operatorname*{argmin}_{\mathbf w} L_S(h_{\mathbf w}) = \operatorname*{argmin}_{\mathbf w} \frac{1}{m} \sum_{i=1}^m (\langle \mathbf w, \mathbf x_i \rangle - y_i)^2. \]

The new word is \(\operatorname{argmin}\), “the argument that minimizes”. Where \(\min\) returns the smallest value, \(\operatorname{argmin}_{\mathbf w}\) returns the \(\mathbf w\) that achieves it; the subscript says what we are free to choose. The right side just writes \(L_S\) out, with the score \(\langle \mathbf w, \mathbf x_i \rangle\) as the prediction. In plain words: **find the weights whose line has the smallest average squared miss.** With \(\mathbf x = (1, x)\), the guess \(20x\) is \(\mathbf w = (0, 20)\) with risk 53; \((5, 18)\) has risk 76; and \((-2.429, 20.971)\), which we will derive below, beats both:

```python
xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
L = lambda w: sum((w[0] + w[1] * x - y) ** 2
                  for x, y in zip(xs, ys)) / 6
cands = [(0, 20), (5, 18), (-2.429, 20.971)]
print([round(L(w), 2) for w in cands])
print(min(cands, key=L))
```

It prints `[53.0, 76.0, 50.25]`, then `(-2.429, 20.971)`: among these three, that is the argmin, and `min(..., key=L)` is Python’s argmin. Trying candidates cannot prove a winner, though; we need a method.

The method: compute the gradient, the vector of partial derivatives, and set it to zero. A **partial derivative** \(\partial L / \partial w_j\) is the rate at which \(L\) changes when you nudge only \(w_j\) and hold the other weights still: at \(\mathbf w = (0, 20)\), nudging the slope up by a tiny amount lowers the risk at about 5.667 times that amount, so \(\partial L_S / \partial w_1 \approx -5.667\). The **gradient** \(\nabla L_S(\mathbf w)\) (the symbol \(\nabla\) is read “nabla” or “grad”) lists all those rates, one per weight; at a lowest point every rate is zero, because no nudge helps. The partial derivative of one term with respect to \(w_j\) is, by the chain rule (the slope of \(u^2\) is \(2u\) times the slope of \(u\)), \(2(\langle \mathbf w, \mathbf x_i \rangle - y_i) \, x_{i,j}\), because \(\partial \langle \mathbf w, \mathbf x_i \rangle / \partial w_j = x_{i,j}\). Collecting all \(j\) and averaging over \(i\),

\[ \nabla L_S(\mathbf w) = \frac{2}{m} \sum_{i=1}^m (\langle \mathbf w, \mathbf x_i \rangle - y_i) \, \mathbf x_i, \]

and we need to solve \(\nabla L_S(\mathbf w) = \mathbf 0\). Inside out: \(\langle \mathbf w, \mathbf x_i \rangle - y_i\) is the miss on message \(i\); multiplying the vector \(\mathbf x_i\) by it gives a vector pointing along the message, long when the miss is big; the sum adds one such vector per message; and \(\frac2m\) averages and carries the 2 from the square. In plain words: **the gradient is twice the average of each message’s vector weighted by its miss.** At \(\mathbf w = (0, 20)\) the misses are \(-2, 10, 5, -10, -8, 5\); they add to 0, giving the first entry, and weighted by \(x = 0, \dots, 5\) they add to \(-17\), so the second entry is \(\tfrac26 \cdot (-17) \approx -5.667\). Nudging each weight by a millionth and watching the risk confirms the formula:

```python
xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
X = [(1, x) for x in xs]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
L = lambda w: sum((dot(w, x) - y) ** 2
                  for x, y in zip(X, ys)) / 6
w, h = (0, 20), 1e-6
grad = [2 / 6 * sum((dot(w, x) - y) * x[j]
                    for x, y in zip(X, ys)) for j in (0, 1)]
nudge = lambda j: [c + h * (k == j) for k, c in enumerate(w)]
numeric = [(L(nudge(j)) - L(w)) / h for j in (0, 1)]
print([round(g, 3) for g in grad])
print([round(g, 3) for g in numeric])
```

It prints `[0.0, -5.667]` twice: the formula and the nudges agree. Now solve \(\nabla L_S(\mathbf w) = \mathbf 0\). Multiply by \(m/2\) and split the sum: \(\sum_i \mathbf x_i \langle \mathbf x_i, \mathbf w \rangle = \sum_i y_i \mathbf x_i\). One more symbol: the small \(\top\) is the **transpose**, which turns a column into a row, and \(\mathbf x \mathbf x^\top\), a column times a row, is the square table whose entry \((j, k)\) is \(x_j x_k\). For \(\mathbf x = (1, 3)\) it is \(\begin{pmatrix} 1 & 3 \\ 3 & 9 \end{pmatrix}\). The vector \(\mathbf x_i \langle \mathbf x_i, \mathbf w \rangle\) is the matrix \(\mathbf x_i \mathbf x_i^\top\) times \(\mathbf w\), because entry \(j\) of both is \(x_{i,j} \sum_k x_{i,k} w_k\). So the condition is the linear system

\[ A \mathbf w = \mathbf b, \qquad A = \sum_{i=1}^m \mathbf x_i \mathbf x_i^\top, \qquad \mathbf b = \sum_{i=1}^m y_i \mathbf x_i. \]

(This \(\mathbf b\) is a vector of the system, not the bias term; the bias term is inside \(\mathbf w\) now.) If \(A\) is invertible, the ERM solution is \(\mathbf w = A^{-1} \mathbf b\), where the **inverse** \(A^{-1}\) is the matrix that undoes \(A\), the way dividing undoes multiplying.

In plain words: **the best weights solve one small linear system, built by adding a table and a vector per message.** For our six messages, the tables for \(x = 0, \dots, 5\) add to \(A = \begin{pmatrix} 6 & 15 \\ 15 & 55 \end{pmatrix}\) (six messages, total count 15, total squared count 55), and \(\mathbf b = (300, 1117)\) (total score, and total count times score). The solving snippet below builds both, entry by entry, as these same sums.

But wait: setting a gradient to zero finds a flat point, and flat points can be maxima or saddles. Why is this one a minimum? Because the square loss has a special shape, and we can see it exactly. For any \(\mathbf w\) and any step \(\boldsymbol\Delta\), expand each term:

\[ (\langle \mathbf w + \boldsymbol\Delta, \mathbf x_i \rangle - y_i)^2 = (\langle \mathbf w, \mathbf x_i \rangle - y_i)^2 + 2(\langle \mathbf w, \mathbf x_i \rangle - y_i) \langle \boldsymbol\Delta, \mathbf x_i \rangle + \langle \boldsymbol\Delta, \mathbf x_i \rangle^2. \]

Average over \(i\). The middle terms average to \(\langle \nabla L_S(\mathbf w), \boldsymbol\Delta \rangle\), by the gradient formula above, so

\[ L_S(\mathbf w + \boldsymbol\Delta) = L_S(\mathbf w) + \langle \nabla L_S(\mathbf w), \boldsymbol\Delta \rangle + \frac{1}{m} \sum_{i=1}^m \langle \boldsymbol\Delta, \mathbf x_i \rangle^2. \]

Here \(\boldsymbol\Delta\) (capital delta) is any change to the weights. The first display is just \((a + c)^2 = a^2 + 2ac + c^2\), with \(a\) the old miss and \(c = \langle \boldsymbol\Delta, \mathbf x_i \rangle\) the change in the prediction. In plain words: **the new risk is the old risk, plus the gradient’s inner product with the step, plus the average squared change in prediction.** From \(\mathbf w = (0, 20)\), the step \(\boldsymbol\Delta = (0, 1)\) gives \(53 + (-5.667) + \tfrac{55}{6} \approx 56.5\), and the risk of \((0, 21)\) is indeed 56.5. The snippet below checks the identity on random weights and steps.

The last term is never negative. So if \(\nabla L_S(\mathbf w) = \mathbf 0\), then \(L_S(\mathbf w + \boldsymbol\Delta) \ge L_S(\mathbf w)\) for every \(\boldsymbol\Delta\): every solution of \(A \mathbf w = \mathbf b\) is a global minimizer. Conversely, a minimizer of a differentiable function has zero gradient. So ERM for least squares is *exactly* solving \(A \mathbf w = \mathbf b\). (This exact expansion is a first taste of convexity, the subject of lesson 12.)

```python
import random

xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
X = [(1, x) for x in xs]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
L = lambda w: sum((dot(w, x) - y) ** 2
                  for x, y in zip(X, ys)) / 6
grad = lambda w: [sum(2 * (dot(w, x) - y) * x[i]
                      for x, y in zip(X, ys)) / 6 for i in (0, 1)]
rng = random.Random(1)
for _ in range(3):
    w = [rng.uniform(-9, 9) for _ in range(2)]
    D = [rng.uniform(-9, 9) for _ in range(2)]
    quad = sum(dot(D, x) ** 2 for x in X) / 6
    rhs = L(w) + dot(grad(w), D) + quad
    lhs = L([a + d for a, d in zip(w, D)])
    print(round(lhs, 6), round(rhs, 6))
```

Each line prints the same number twice (the first is `3347.029002 3347.029002`): the expansion is exact, not an approximation.

Now solve the six messages. With the constant 1 appended, \(\mathbf x_i = (1, x_i)\), so \(A\) is \(2 \times 2\), and a \(2 \times 2\) system has a closed form, **Cramer’s rule**. For \(A = \begin{pmatrix} p & q \\ r & s \end{pmatrix}\), the **determinant** is \(\det = ps - qr\); when it is not 0, \(A\) is invertible and \(\mathbf w = \big( (s b_1 - q b_2)/\det, \; (p b_2 - r b_1)/\det \big)\). In plain words: **cross-multiply, subtract, divide by the determinant.** Here \(\det = 6 \cdot 55 - 15 \cdot 15 = 105\), and the snippet mirrors the formula line by line:

```python
xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
X, R = [(1, x) for x in xs], (0, 1)
A = [[sum(x[i] * x[j] for x in X) for j in R] for i in R]
b = [sum(y * x[i] for x, y in zip(X, ys)) for i in R]
det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
w = [(A[1][1] * b[0] - A[0][1] * b[1]) / det,
     (A[0][0] * b[1] - A[1][0] * b[0]) / det]
res = [w[0] + w[1] * x - y for x, y in zip(xs, ys)]
grad = [2 / 6 * sum(r * x[i] for r, x in zip(res, X)) for i in R]
mse = sum(r * r for r in res) / 6
print(A, b, det)
print([round(c, 3) for c in w], round(mse, 2))
print([round(g, 9) + 0 for g in grad])
```

It prints `[[6, 15], [15, 55]] [300, 1117] 105`, then `[-2.429, 20.971] 50.25`, then `[0.0, 0.0]`. The fitted line is \(\hat y = -2.429 + 20.971 \, x\) (the hat on \(\hat y\) marks a prediction, as opposed to a true score \(y\)): each “free” adds about 21 points of junk score. Its empirical risk is 50.25, and the gradient there is zero, as promised. Here is the fit message by message:

| \(x\) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| \(y\) | 2 | 10 | 35 | 70 | 88 | 95 |
| line \(\hat y\) | −2.4 | 18.5 | 39.5 | 60.5 | 81.5 | 102.4 |
| miss \(\hat y - y\) | −4.4 | 8.5 | 4.5 | −9.5 | −6.5 | 7.4 |

The misses have a pattern: too high, too low, too high, as the straight line cuts across a curve. And the line predicts a negative percentage at 0 and more than 100% at 5. Hold that thought for the next section.

[[diagram:least-squares]]

Before the singular case, play with it. In the lab below you can drag the line through the six messages, watch each miss drawn as a square, and watch the two entries of the gradient shrink as you close in on the best line.

[[lab]]

### When \(A\) cannot be inverted

Suppose two words always appear together the same number of times: say marketing templates always pair “free” with “gift”. Then every message has \(x_1 = x_2\), and \(A\) is **singular**: it has no inverse. Is ERM stuck?

No. Take three such messages, whose “free” and “gift” counts are both 1, 2 and 3, with junk scores 10, 35 and 70 (no constant 1 this time, to keep the numbers small). Then \(A = \begin{pmatrix} 14 & 14 \\ 14 & 14 \end{pmatrix}\) and \(\mathbf b = (290, 290)\). The determinant is \(14 \cdot 14 - 14 \cdot 14 = 0\), so there is no inverse. But look at what \(A\) does to two special directions. It sends \((1, 1)\) to \((28, 28)\), the same direction stretched by 28; and it sends \((1, -1)\) to \((0, 0)\), squashed to nothing. A direction that \(A\) only stretches is an **eigenvector**, and the stretch factor is its **eigenvalue**: 28 for \((1, 1)\), 0 for \((1, -1)\). The eigenvalue 0 is exactly why \(A\) cannot be undone: it flattens a whole direction, and nothing can unflatten it. The way out is to undo \(A\) along the directions it stretches and leave the flattened one alone.

Here is the general argument. It needs one fact from linear algebra, the **spectral theorem**: a real symmetric matrix (one equal to its own transpose) can be written \(A = V D V^\top\), where \(D\) is **diagonal**, zero everywhere off the main diagonal, holding the eigenvalues, and \(V\) is **orthonormal**, meaning its columns \(\mathbf v_1, \dots, \mathbf v_d\) are eigenvectors of length 1 and mutually perpendicular, so \(V^\top V\) is the identity \(I\), the matrix that changes nothing. Read \(V D V^\top\) right to left: \(V^\top\) measures how much of a vector lies along each eigenvector, \(D\) stretches each of those amounts by its eigenvalue, and \(V\) puts the pieces back together. For the twins, \(\mathbf v_1 = (1, 1)/\sqrt 2\) and \(\mathbf v_2 = (1, -1)/\sqrt 2\), and \(D\) has 28 and 0 on its diagonal. Our \(A\) is symmetric because each \(\mathbf x_i \mathbf x_i^\top\) is.

Define \(D^+\) as the diagonal matrix with \(D^+_{i,i} = 1/D_{i,i}\) when \(D_{i,i} \ne 0\) and \(D^+_{i,i} = 0\) otherwise, and set

\[ A^+ = V D^+ V^\top, \qquad \hat{\mathbf w} = A^+ \mathbf b. \]

Read it like \(V D V^\top\), with one change in the middle: \(D^+\) divides by each nonzero eigenvalue instead of multiplying, and puts 0 where the eigenvalue is 0. The plus sign marks this “pseudo” inverse, and the hat on \(\hat{\mathbf w}\) marks the weights it produces. In plain words: **undo \(A\) along every direction it stretches, and ignore the direction it flattens.** For the twins, \(D^+\) has \(1/28\) and 0 on its diagonal, so \(A^+ = \tfrac{1}{28} \mathbf v_1 \mathbf v_1^\top = \tfrac{1}{56} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\) and \(\hat{\mathbf w} = (580/56, 580/56) \approx (10.357, 10.357)\):

```python
r = 2 ** -0.5
V = [[r, r], [r, -r]]
D_plus = [1 / 28, 0]
A_plus = [[sum(V[i][k] * D_plus[k] * V[j][k] for k in (0, 1))
           for j in (0, 1)] for i in (0, 1)]
b = [290, 290]
w_hat = [sum(a * c for a, c in zip(row, b)) for row in A_plus]
print([[round(56 * a, 6) for a in row] for row in A_plus])
print([round(c, 3) for c in w_hat])
```

It prints `[[1.0, 1.0], [1.0, 1.0]]`, which is \(56 A^+\), then `[10.357, 10.357]`.

**Claim.** \(A \hat{\mathbf w} = \mathbf b\), always. So \(\hat{\mathbf w}\) is an ERM solution even when \(A\) is singular.

*Proof.* First compute, using \(V^\top V = I\):

\[ A \hat{\mathbf w} = V D V^\top V D^+ V^\top \mathbf b = V D D^+ V^\top \mathbf b = \sum_{i \,:\, D_{i,i} \ne 0} \mathbf v_i \mathbf v_i^\top \mathbf b. \]

The last step holds because \(D D^+\) is diagonal with a 1 where \(D_{i,i} \ne 0\) and a 0 elsewhere; the sum runs only over those \(i\) (read the colon under the \(\sum\) as “such that”). In plain words: **applying \(A\) after \(A^+\) keeps the part of \(\mathbf b\) along the stretched directions and drops the rest.** For the twins only \(\mathbf v_1\) survives, and \(\mathbf v_1 \mathbf v_1^\top \mathbf b = \tfrac12 (290 + 290, 290 + 290) = (290, 290)\), all of \(\mathbf b\): nothing was dropped. The **span** of some vectors is every combination of them, with any real weights; the span of \(\mathbf v_1\) is the diagonal line through \((1, 1)\). The right-hand side is the orthogonal projection of \(\mathbf b\) onto \(U = \operatorname{span}\{\mathbf v_i : D_{i,i} \ne 0\}\), the closest point to \(\mathbf b\) in \(U\), and a projection leaves a vector of \(U\) unchanged: if \(\mathbf b = \sum_{k} c_k \mathbf v_k\) over those \(k\), then \(\mathbf v_i^\top \mathbf b = c_i\) by orthonormality. So it is enough to show \(\mathbf b \in U\). We do it in three small steps.

*(a) \(U\) is the range of \(A\)*, the set \(\{A \mathbf v\}\). Indeed \(A \mathbf v = \sum_i D_{i,i} \mathbf v_i \langle \mathbf v_i, \mathbf v \rangle\) only involves the \(\mathbf v_i\) with \(D_{i,i} \ne 0\), and each such \(\mathbf v_i = A (\mathbf v_i / D_{i,i})\) is in the range.

*(b) The range of \(A\) is the span of the messages \(\mathbf x_1, \dots, \mathbf x_m\).* One direction: \(A \mathbf v = \sum_i \mathbf x_i \langle \mathbf x_i, \mathbf v \rangle\) is a combination of messages. For the other, look at the **null space**, the vectors with \(A \mathbf v = \mathbf 0\). For such \(\mathbf v\), \(0 = \langle \mathbf v, A \mathbf v \rangle = \sum_i \langle \mathbf x_i, \mathbf v \rangle^2\), so \(\mathbf v\) is perpendicular to every message. And since \(A\) is symmetric, \(\langle \mathbf v, A \mathbf u \rangle = \langle A \mathbf v, \mathbf u \rangle\) for all \(\mathbf u\), and the null space is exactly the set of vectors perpendicular to the range: if \(A \mathbf v = \mathbf 0\), every \(\langle \mathbf v, A \mathbf u \rangle\) is 0; if \(\mathbf v\) is perpendicular to the range, then \(\|A \mathbf v\|^2 = \langle \mathbf v, A (A \mathbf v) \rangle = 0\). So every message is perpendicular to every vector that is perpendicular to the range, which in finite dimensions means every message lies in the range.

*(c) \(\mathbf b = \sum_i y_i \mathbf x_i\) is a combination of messages*, so by (b) and (a) it lies in \(U\). \(\blacksquare\)

Now check the claim on the twins, and compare \(\hat{\mathbf w}\) with another candidate that puts all the weight on “free”:

```python
cs, ys = [1, 2, 3], [10, 35, 70]
X, R = [(c, c) for c in cs], (0, 1)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
A = [[sum(x[i] * x[j] for x in X) for j in R] for i in R]
b = [sum(y * x[i] for x, y in zip(X, ys)) for i in R]
s = A[0][0]
A_plus = [[1 / (4 * s)] * 2, [1 / (4 * s)] * 2]
w_hat, w_other = [dot(row, b) for row in A_plus], [b[0] / s, 0]
L = lambda w: sum((dot(w, x) - y) ** 2 for x, y in zip(X, ys)) / 3
print(A, b, A[0][0] * A[1][1] - A[0][1] * A[1][0])
for w in (w_hat, w_other):
    Aw = [round(dot(row, w), 6) for row in A]
    print([round(c, 3) for c in w], Aw, round(L(w), 3))
```

It prints `[[14, 14], [14, 14]] [290, 290] 0`, then `[10.357, 10.357] [290.0, 290.0] 72.619`, then `[20.714, 0] [290.0, 290.0] 72.619`. The determinant is 0, yet \(\hat{\mathbf w}\) solves the system, splitting the weight evenly between the twin words. Another solution, all the weight on “free”, solves it too and has the same empirical risk. When \(A\) is singular the minimizer is not unique, and \(A^+\) picks one.

## Curves from lines: polynomial regression

The straight line missed the S-shape. Some tasks call for nonlinear predictors, and the cleanest are polynomials: besides a constant and a multiple of the count, allow a multiple of the count squared, the count cubed, and so on. Our line is the simplest case: at three “free”s it predicts \(-2.429 + 20.971 \cdot 3 \approx 60.5\). A one-dimensional polynomial of degree \(n\) is

\[ p(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_n x^n, \]

named by its coefficient vector \((a_0, \dots, a_n)\). The small raised numbers are powers (\(x^2 = x \cdot x\)), the subscripts on \(a\) number the coefficients to match, the dots mean “keep going in the same pattern”, and the degree \(n\) is the highest power used. In plain words: **a polynomial adds a constant, a multiple of \(x\), a multiple of \(x^2\), and so on up to \(x^n\).** The class of **polynomial regression** predictors of degree \(n\) is \(\mathcal H^n_{\text{poly}} = \{x \mapsto p(x)\}\), every such polynomial, with \(\mathcal X = \mathcal Y = \mathbb R\).

It looks like a new problem, but it is the old one in disguise. Define the map \(\psi: \mathbb R \to \mathbb R^{n+1}\) by \(\psi(x) = (1, x, x^2, \dots, x^n)\). Then

\[ p(x) = a_0 + a_1 x + \dots + a_n x^n = \langle \mathbf a, \psi(x) \rangle. \]

Here \(\psi\) (psi) turns one count into a list of \(n + 1\) numbers, its powers from 0 to \(n\), and \(\mathbf a\) is the coefficient vector. Multiplying matching entries and adding gives \(a_0 \cdot 1 + a_1 x + \dots\), which is the polynomial. In plain words: **a polynomial of \(x\) is an inner product of its coefficients with the list of powers of \(x\).** For three “free”s, \(\psi(3) = (1, 3)\) at degree 1 and \((1, 3, 9, 27)\) at degree 3, and the line’s coefficients with \(\psi(3)\) give 60.484:

```python
a = (-2.429, 20.971)
psi = lambda x, n: [x ** k for k in range(n + 1)]
x = 3
terms = [ak * pk for ak, pk in zip(a, psi(x, 1))]
print(psi(x, 1), psi(x, 3), round(sum(terms), 3))
```

It prints `[1, 3] [1, 3, 9, 27] 60.484`.

So turn each example \((x_i, y_i)\) into \((\psi(x_i), y_i)\) and run least squares on the new vectors; the weights it returns are the coefficients. Learning reduces to a problem we have already solved.

```python
xs, ys = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
dot = lambda u, v: sum(p * q for p, q in zip(u, v))
def solve(M):
    n = len(M)
    for c in range(n):
        for r in set(range(n)) - {c}:
            f = M[r][c] / M[c][c]
            M[r] = [a - f * q for a, q in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]
for n in (1, 3):
    P = [[x ** k for k in range(n + 1)] for x in xs]
    col = lambda i: [p[i] for p in P]
    M = [[dot(col(i), col(j)) for j in range(n + 1)]
         + [dot(col(i), ys)] for i in range(n + 1)]
    a = solve(M)
    e = [dot(a, p) - y for p, y in zip(P, ys)]
    print(n, [round(c, 2) for c in a], round(dot(e, e) / 6, 2))
```

It prints `1 [-2.43, 20.97] 50.25` and `3 [1.75, -4.12, 14.76, -2.05] 2.86`. Degree 1 is the line from before. The cubic \(1.75 - 4.12x + 14.76x^2 - 2.05x^3\) follows the S-shape and cuts the empirical risk from 50.25 to 2.86. (The `solve` function is Gauss–Jordan elimination on the augmented matrix \([A \mid \mathbf b]\); it skips row swaps, which is safe here because these \(A\) have positive pivots.) Its predictions, message by message, are 1.7, 10.3, 36.2, 67.0, 90.4 and 94.3.

[[diagram:polynomial]]

Should you now reach for degree 5? With six points it would hit every one exactly, empirical risk 0, and the build-it challenge asks you to see what it predicts for a message with six “free”s. Lessons 1 and 2 already told you what a perfect fit with too rich a class is worth.

## A probability of junk: logistic regression

One more request, this time from the inbox designers. They want to grey out messages that are *probably* junk and show the filter’s confidence on hover. A halfspace says \(+1\) or \(-1\) and nothing else, even for a message whose score is a hair above zero. Can a linear learner say “junk, with probability 0.73”?

In **logistic regression** the hypotheses map \(\mathbb R^d\) to the interval \([0, 1]\), the numbers from 0 to 1, and \(h(\mathbf x)\) is read as the probability that the label of \(\mathbf x\) is \(+1\). What we want is a squashing function: feed it the score, get back a number between 0 and 1 that is near 1 for big positive scores (A scores 5: surely junk), near 0 for big negative ones, and exactly one half at score 0, where the halfspace is undecided. The class composes such a **sigmoid** (“S-shaped”) function with the linear functions; specifically the **logistic function**

\[ \varphi_{\text{sig}}(z) = \frac{1}{1 + \exp(-z)}, \]

where \(z\) stands for the score, and \(\exp(-z)\) is \(e^{-z}\), the number \(e \approx 2.718\) raised to the power \(-z\); it is huge when \(z\) is very negative and tiny when \(z\) is large. Inside out: flip the sign of the score, raise \(e\) to it, add 1, and take one over the result. When \(z\) is large the bottom is close to 1, so the answer is close to 1; when \(z\) is very negative the bottom is huge, so the answer is close to 0. In plain words: **the logistic function squashes any score into a probability between 0 and 1, with score 0 landing on one half.** At \(z = 1\), message B’s score: \(e^{-1} \approx 0.368\), the bottom is 1.368, and the probability is 0.731.

```python
import math

for z in (-3, 0, 1, 5):
    e = math.exp(-z)
    print(z, round(e, 3), round(1 + e, 3), round(1 / (1 + e), 3))
```

It prints `-3 20.086 21.086 0.047`, `0 1.0 2.0 0.5`, `1 0.368 1.368 0.731` and `5 0.007 1.007 0.993`: the score, \(\exp(-z)\), the bottom, and the probability. So, using homogeneous linear functions,

\[ H_{\text{sig}} = \varphi_{\text{sig}} \circ L_d = \{ \mathbf x \mapsto \varphi_{\text{sig}}(\langle \mathbf w, \mathbf x \rangle) \; : \; \mathbf w \in \mathbb R^d \}. \]

This reads exactly like \(HS_d\), with the logistic function in place of the sign: **score the message, then squash the score into a probability.**

The logistic hypothesis and the halfspace with the same \(\mathbf w\) agree whenever \(|\langle \mathbf w, \mathbf x \rangle|\) is large. They differ near the boundary: there the halfspace still answers a flat \(\pm 1\), while the logistic hypothesis admits it is unsure and gives a probability just above or just below one half.

[[diagram:logistic]]

Now the loss. If the label is \(y = +1\), we want \(h_{\mathbf w}(\mathbf x)\) to be large; if \(y = -1\), we want \(1 - h_{\mathbf w}(\mathbf x)\), the probability given to \(-1\), to be large. Compute the second one:

\[ 1 - h_{\mathbf w}(\mathbf x) = 1 - \frac{1}{1 + \exp(-\langle \mathbf w, \mathbf x \rangle)} = \frac{\exp(-\langle \mathbf w, \mathbf x \rangle)}{1 + \exp(-\langle \mathbf w, \mathbf x \rangle)} = \frac{1}{1 + \exp(\langle \mathbf w, \mathbf x \rangle)}, \]

where the last step multiplies top and bottom by \(\exp(\langle \mathbf w, \mathbf x \rangle)\). The first step writes \(1\) as \((1 + \exp(-\langle \mathbf w, \mathbf x \rangle))\) over itself and subtracts the tops. In plain words: **the probability of keep is the logistic function of the negated score.** For message B, score 1, the probability of junk is 0.731 and of keep \(1/(1 + e^1) = 1/3.718 \approx 0.269\), and the two add to 1; the `assert` in the next snippet checks this identity at six scores. Look at the two cases side by side: the probability the hypothesis gives to the *true* label is \(1 / (1 + \exp(-y \langle \mathbf w, \mathbf x \rangle))\) for both \(y = +1\) and \(y = -1\). One formula, thanks to \(\pm 1\) labels once more. A sensible loss should grow as that probability falls, that is, as \(1 + \exp(-y \langle \mathbf w, \mathbf x \rangle)\) grows. The **logistic loss** takes its logarithm (an increasing function, so the order is kept):

\[ \ell(h_{\mathbf w}, (\mathbf x, y)) = \log \left( 1 + \exp(-y \langle \mathbf w, \mathbf x \rangle) \right). \]

Inside out: \(y \langle \mathbf w, \mathbf x \rangle\) is the signed score from lesson 9a; the minus sign and \(\exp\) turn it into a number that is tiny when the signed score is large and huge when it is very negative; adding 1 gives the bottom of the probability of the true label; and \(\log\), the natural logarithm, the inverse of \(\exp\) (so \(\log e^3 = 3\)), turns that into a cost. Since the probability is one over that bottom, the loss is also \(-\log\) of the probability given to the true label. In plain words: **the loss is minus the log of the probability the hypothesis gave to the right answer.** Message B, signed score 1, gets probability 0.731 and pays \(\log 1.368 \approx 0.313\). The snippet mirrors this with `sigma` for \(\varphi_{\text{sig}}\) and `loss` for \(\ell\), as functions of the signed score \(z\):

```python
import math

sigma = lambda z: 1 / (1 + math.exp(-z))
loss = lambda z: math.log(1 + math.exp(-z))
for z in (-3, -1, 0, 1, 3, 5):
    assert abs((1 - sigma(z)) - sigma(-z)) < 1e-12
    print(z, round(sigma(z), 3), round(loss(z), 3))
```

It prints `-3 0.047 3.049`, `-1 0.269 1.313`, `0 0.5 0.693`, `1 0.731 0.313`, `3 0.953 0.049` and `5 0.993 0.007`. With our halfspace \(\mathbf w = (-1, 2, -2)\), message A has signed score 5, so the filter gives it probability 0.993 of junk and pays 0.007. A message on the boundary pays \(\log 2 \approx 0.693\). And a confident mistake, signed score \(-3\), costs 3.049: the loss punishes confidence in the wrong direction much more than hesitation.

Given \(S\), the ERM problem of logistic regression is

\[ \operatorname*{argmin}_{\mathbf w \in \mathbb R^d} \frac{1}{m} \sum_{i=1}^m \log \left( 1 + \exp(-y_i \langle \mathbf w, \mathbf x_i \rangle) \right). \]

Every piece is already familiar: the logistic loss of example \(i\), averaged over the \(m\) examples, and \(\operatorname{argmin}\) returns the weights that make that average smallest. In plain words: **find the weights with the smallest average logistic loss.** At \(\mathbf w = \mathbf 0\) every signed score is 0 and every loss is \(\log 2 \approx 0.693\), so the average is 0.693; the gradient descent snippet below starts there and drives it down.

Its great advantage is that this objective is a **convex** function of \(\mathbf w\), which makes it efficiently solvable by standard methods, separable data or not. Here is why it is convex. Picture the logistic loss as a curve over the signed score, and pick two points on it: \(-3\) with loss 3.049 and 1 with loss 0.313. Halfway between them, at \(-1\), the curve sits at 1.313, below the halfway point of the straight chord joining them, \((3.049 + 0.313)/2 = 1.681\). A curve that always sags below its chords like that is convex. In symbols, a function \(f\) is convex when \(f(\theta \mathbf u + (1 - \theta) \mathbf v) \le \theta f(\mathbf u) + (1 - \theta) f(\mathbf v)\) for all \(\mathbf u, \mathbf v\) and \(\theta \in [0, 1]\). Here \(\theta\) (theta) is a mixing weight between 0 and 1. The left side evaluates \(f\) at a point mixed from \(\mathbf u\) and \(\mathbf v\); the right side mixes the two values of \(f\) with the same weights, which is the height of the chord. In plain words: **the chord lies above the graph.**

```python
import math

g = lambda z: math.log(1 + math.exp(-z))
u, v, theta = -3, 1, 0.5
mid = g(theta * u + (1 - theta) * v)
chord = theta * g(u) + (1 - theta) * g(v)
print(round(mid, 3), round(chord, 3), mid <= chord)
```

It prints `1.313 1.681 True`. One pair of points is an example, not a proof; here is the proof. For one variable, write \(g(z) = \log(1 + e^{-z})\). Write \(g'\) for its slope and \(g''\) for how fast the slope changes. The slope of \(\log u\) is \(1/u\) times the slope of \(u\), so \(g'(z) = -e^{-z}/(1 + e^{-z})\), and multiplying top and bottom by \(e^z\) turns that into \(-1/(e^z + 1) = -\varphi_{\text{sig}}(-z)\). The same two rules give the logistic function’s own slope, \(\varphi_{\text{sig}}'(z) = e^{-z}/(1 + e^{-z})^2 = \varphi_{\text{sig}}(z) \, \varphi_{\text{sig}}(-z)\), so

\[ g''(z) = \varphi_{\text{sig}}(z) \, \varphi_{\text{sig}}(-z) > 0. \]

In plain words: **the slope of the loss always increases, by the product of the two class probabilities.** At \(z = 1\) that is \(0.731 \cdot 0.269 \approx 0.197\), and measuring the change of slope over steps of a thousandth agrees:

```python
import math

g = lambda z: math.log(1 + math.exp(-z))
sigma = lambda z: 1 / (1 + math.exp(-z))
z, h = 1, 1e-3
numeric = (g(z + h) - 2 * g(z) + g(z - h)) / h ** 2
print(round(numeric, 3), round(sigma(z) * sigma(-z), 3))
```

It prints `0.197 0.197`. A function of one variable whose second derivative is never negative is convex. Each term of the risk is \(g\) applied to \(z_i(\mathbf w) = y_i \langle \mathbf w, \mathbf x_i \rangle\), which is linear in \(\mathbf w\), so \(z_i(\theta \mathbf u + (1 - \theta) \mathbf v) = \theta z_i(\mathbf u) + (1 - \theta) z_i(\mathbf v)\), and convexity of \(g\) carries straight over to the term. An average of convex functions is convex. Lesson 12 develops this properly, and lesson 14 gives a simple algorithm for minimizing such functions.

There is also a statistical reading. Suppose the labels really were drawn by the logistic model: a message \(\mathbf x\) is junk with probability \(\mathbb P[y = +1 \mid \mathbf x] = \varphi_{\text{sig}}(\langle \mathbf w, \mathbf x \rangle)\), where the bar \(\mid\) reads “given”, given that the message is \(\mathbf x\). By the identity above, the probability of the observed label is \(\varphi_{\text{sig}}(y \langle \mathbf w, \mathbf x \rangle)\) in both cases. The examples are independent, and the chance of several independent events all happening is the product of their chances, so the probability of the whole training sequence is \(\prod_i \varphi_{\text{sig}}(y_i \langle \mathbf w, \mathbf x_i \rangle)\), where the capital pi \(\prod_i\) is like \(\sum_i\) but multiplies, one factor per example. Its negative logarithm, since \(\log(ab) = \log a + \log b\) turns the product into a sum, is \(\sum_i \log(1 + \exp(-y_i \langle \mathbf w, \mathbf x_i \rangle))\), which is \(m\) times the logistic risk. So minimizing the logistic risk is the same as choosing the \(\mathbf w\) under which the data is most likely, the **maximum likelihood** approach that lesson 24 studies. In plain words: **the most likely weights and the lowest logistic risk are the same weights.** With \(\mathbf w = (-1, 2, -2)\) on the eight messages, the probabilities of the true labels multiply to about 0.3195, minus its log is 1.1408, and eight times the average logistic loss is 1.1408 too:

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
w = (-1, 2, -2)
sigma = lambda z: 1 / (1 + math.exp(-z))
z = [y * (w[0] + w[1] * x[0] + w[2] * x[1]) for x, y in S]
like = math.prod(sigma(zi) for zi in z)
risk = sum(math.log(1 + math.exp(-zi)) for zi in z) / len(S)
print([round(sigma(zi), 3) for zi in z])
print(round(like, 4), round(-math.log(like), 4))
print(round(len(S) * risk, 4))
```

It prints `[0.993, 0.731, 0.953, 0.731, 0.953, 0.953, 0.953, 0.731]`, then `0.3195 1.1408`, then `1.1408`.

Let’s fit it. Your users have now labelled two more messages, and they disagree with themselves: a second “free trial, book a meeting” (\(x = (2, 1)\)) was marked keep, and a second “free lunch at the meeting” (\(x = (1, 2)\)) was marked junk. No halfspace can be consistent with that. As a preview of lesson 14, minimize the logistic risk by plain **gradient descent**: start at \(\mathbf w = \mathbf 0\) and repeatedly step against the gradient, which by \(g'\) above and the chain rule (the slope of \(z_i\) in \(w_j\) is \(y_i x_{i,j}\)) is \(\frac1m \sum_i -y_i \mathbf x_i \, \varphi_{\text{sig}}(-y_i \langle \mathbf w, \mathbf x_i \rangle)\). Read it inside out: \(\varphi_{\text{sig}}(-y_i \langle \mathbf w, \mathbf x_i \rangle)\) is the probability the current weights give to the *wrong* label, so it is large on badly handled messages; it scales the vector \(-y_i \mathbf x_i\), the message pointed against its label; and the sum and \(\frac1m\) average over messages. In plain words: **each message pulls the weights towards its own label, harder the more the weights currently doubt it.** At \(\mathbf w = \mathbf 0\) every wrong-label probability is one half, so junk message A contributes \(-\tfrac12 (1, 3, 0)\) before averaging, and stepping against the gradient moves the weights towards A by \(\tfrac{1}{2m}\) of its vector: a softened Perceptron update. In the code, `g` is this gradient and the step size is 1.

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
sigma = lambda z: 1 / (1 + math.exp(-z))
risk = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                     for x, y in zip(X, Y)) / len(X)
w = [0.0, 0.0, 0.0]
for step in range(1000):
    g = [sum(-y * x[j] * sigma(-y * dot(w, x))
             for x, y in zip(X, Y)) / len(X) for j in range(3)]
    w = [wj - gj for wj, gj in zip(w, g)]
print(round(risk([0, 0, 0]), 3), round(risk(w), 3))
print([round(c, 2) for c in w])
print([round(sigma(dot(w, x)), 2) for x in X])
```

It prints `0.693 0.483`, then `[-0.92, 1.38, -0.95]`, then `[0.96, 0.71, 0.94, 0.61, 0.19, 0.13, 0.27, 0.28, 0.71, 0.19]`. The risk falls from \(\log 2\) to 0.483. The weights keep the same story as the halfspace (a negative bias term, “free” pushes up, “meeting” pushes down), but now every message gets a probability: A is junk with probability 0.96, the empty message H with 0.28, and the disputed “free trial” at \((2, 1)\) with 0.71, exactly the kind of hedge the designers asked for.

## Regression and logistic regression as code

```
LeastSquares(S):
  A ← d × d zero matrix;  b ← zero vector in R^d
  for i = 1..m:
    A ← A + x_i x_iᵀ
    b ← b + y_i · x_i
  if A is invertible: return w = solve(A, b)
  else:               return w = A⁺ b

PolynomialRegression(S, n):
  ψ(x) = (1, x, ..., x^n)
  S' ← ((ψ(x_1), y_1), ..., (ψ(x_m), y_m))
  a ← LeastSquares(S')
  return p(x) = ⟨a, ψ(x)⟩

LogisticRegression(S, steps, η):
  w ← 0 in R^d
  repeat steps times:
    g ← (1/m) Σ_i  −y_i · x_i · φ_sig(−y_i ⟨w, x_i⟩)
    w ← w − η · g
  return h(x) = φ_sig(⟨w, x⟩)
```

**LeastSquares** spends \(O(md^2)\) time building \(A\) (each table \(\mathbf x_i \mathbf x_i^\top\) has \(d^2\) entries) and \(O(md)\) building \(\mathbf b\), then \(O(d^3)\) solving the \(d \times d\) system by Gaussian elimination, or computing \(A^+\) from an eigendecomposition. Total \(O(md^2 + d^3)\) time and \(O(d^2)\) space, independent of \(m\) once \(A\) is built: you can stream the examples.

**PolynomialRegression** adds \(O(mn)\) to compute the features, then runs least squares in dimension \(n + 1\): \(O(mn^2 + n^3)\).

**LogisticRegression** costs \(O(md)\) per gradient step and \(O(d)\) space. How many steps it needs, and how to choose the step size \(\eta\), is the business of lessons 12 and 14; the loop above is a preview, not part of this lesson’s theory.

**Build it yourself.**

```python
import math

XS, YS = [0, 1, 2, 3, 4, 5], [2, 10, 35, 70, 88, 95]
S10 = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
       ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
       ((2, 1), -1), ((1, 2), 1)]

def solve(A, b): ...
def least_squares(X, y): ...
def poly_features(x, n): ...
def mse(w, X, y): ...
def logistic_risk(w, S): ...
def logistic_grad(w, S): ...
def fit_logistic(S, steps=1000, eta=1.0): ...

if __name__ == "__main__":
    L1 = [(1, x) for x in XS]
    w1 = least_squares(L1, YS)
    assert [round(c, 3) for c in w1] == [-2.429, 20.971]
    P3 = [poly_features(x, 3) for x in XS]
    assert round(mse(least_squares(P3, YS), P3, YS), 2) == 2.86
    P5 = [poly_features(x, 5) for x in XS]
    assert mse(least_squares(P5, YS), P5, YS) < 1e-6
    twins = [(c, c) for c in (1, 2, 3)]
    assert least_squares(twins, [10, 35, 70]) is None
    zero = (0, 0, 0)
    assert abs(logistic_risk(zero, S10) - math.log(2)) < 1e-12
    w = fit_logistic(S10)
    assert round(logistic_risk(w, S10), 3) == 0.483
    assert max(abs(g) for g in logistic_grad(w, S10)) < 1e-6
```

1. `solve` is Gaussian elimination with row swaps (pick the largest pivot); it returns `None` when a pivot is below `1e-9`. `least_squares` builds \(A\) and \(\mathbf b\) and calls it.
2. `poly_features(x, n)` returns \(\psi(x)\). The degree-5 fit has empirical risk essentially 0. Predict a message with six “free”s using the line, the cubic and the degree-5 polynomial. Which do you believe, and what does lesson 2 call the degree-5 behaviour?
3. Extend `least_squares` to return \(A^+ \mathbf b\) for the `twins` data instead of `None`. For a \(2 \times 2\) symmetric matrix you can find the eigenvalues from the trace and determinant.
4. `logistic_risk` and `logistic_grad` take `w` with the bias term first and messages without the 1. Run `fit_logistic` on the eight separable messages for 100, 1,000 and 10,000 steps and print \(\|\mathbf w\|\) and the risk each time. What is happening, and why can it not happen on `S10`?

## Fix the bug

Back to the product manager’s two questions. Is there a better line than “20 points per free”, and can you prove that yours is the best? You now have both answers. The best line solves \(A \mathbf w = \mathbf b\), and the expansion from “Least squares” is the proof: at weights where the gradient is zero, any step \(\boldsymbol\Delta\) changes the risk by the average squared change in prediction, which is never negative. So a zero gradient is a certificate, the same kind of thing a separator’s margin was in lesson 9a. You check it on the labelled messages you have, before the inbox goes live, and it speaks for every other line at once, including the ones nobody tried.

The teammate’s line fails that check on the six messages. Its gradient is \((0, -5.667)\), and the second entry says that raising the slope lowers the risk, so a better line exists. The fitted line \((-2.429, 20.971)\) passes, with risk 50.25 against 53.

In the lab below, each round brings a new batch of labelled messages from another team’s inbox. You can drag the line, or work out \(A\) and \(\mathbf b\) from the batch and type the weights. The lab only accepts weights it can certify: a risk within 0.01 of the smallest any line can reach, which by the expansion means a gradient of zero up to the rounding of the numbers you type. A line that merely looks good will not pass, and neither will the teammate’s. One round brings twin words that always appear together: its determinant is 0, Cramer’s rule divides by zero, and you will need another way to solve \(A \mathbf w = \mathbf b\). Three rounds in a row and the junk score ships.

[[lab-act2]]

## What people get wrong

**“A flat point might only be a local dip.”** For many losses, yes. For the square loss, no: the expansion in “Least squares” shows that at a zero gradient every step adds an average of squares, so every flat point is a global minimum. That is exactly why a zero gradient can serve as a certificate.

**“The perfect fit is the best fit.”** The degree-5 polynomial passes through all six messages, empirical risk 0. A class rich enough to pass through any six points can do that whatever the scores are, which is the memorizer’s trick from lesson 1: a perfect fit with too rich a class says nothing about the next message. Step 2 of the build-it challenge shows how far its prediction strays.

**“A logistic probability of 0.71 means 71% of such messages are junk.”** Only if the data really follow the logistic model. The number is what the fitted hypothesis says; whether it is calibrated is a separate question.

## Recap and the next bug

1. Linear regression predicts a number with the affine functions \(L_d\), and the square loss makes its empirical risk the average squared miss.
2. Least squares solves ERM for linear regression with the square loss: solve \(A \mathbf w = \mathbf b\), with \(A^+\) when \(A\) is singular.
3. For the square loss a zero gradient is a certificate: the exact expansion shows every solution of \(A \mathbf w = \mathbf b\) is a global minimizer, so no untried line can beat it.
4. Polynomial regression is least squares on \(\psi(x) = (1, x, \dots, x^n)\).
5. Logistic regression passes the linear score through the logistic function to get a probability, and its loss \(\log(1 + \exp(-y \langle \mathbf w, \mathbf x \rangle))\) makes ERM a convex problem, even when the data are not separable.

The filter weighs its evidence now, and the inbox has scores and probabilities. Then a colleague from the search team stops by your desk with a different idea. “Your halfspace is fine, but the single-feature rules weren’t useless. They’re absurdly cheap: one word, one threshold, you can find the best one in a single sorted pass. Each one on its own is barely better than a coin, sure. But what if you trained a few hundred of them, each one paying attention to the messages the others got wrong, and let them vote?” Can a crowd of weak rules, each only slightly better than guessing, add up to a strong one? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. On the six messages the fitted line has gradient zero and risk 50.25, and \(A = \begin{pmatrix} 6 & 15 \\ 15 & 55 \end{pmatrix}\). Without refitting and without looking at the scores again, use the expansion from “Least squares” to find the risk of the line whose bias term is 1 larger, and of the line whose slope is 1 larger. Why is \(A\) all you need?
2. The logistic loss of a signed score \(z\) is \(\log(1 + e^{-z})\), and the 0-1 loss is 1 when \(z \le 0\) and 0 otherwise. Show that \(\log(1 + e^{-z}) / \log 2\) is at least the 0-1 loss for every \(z\). What does that tell you about weights whose average logistic loss on the training sequence is 0.1?
3. For the twin words, every \(\mathbf w = (w_1, w_2)\) with \(w_1 + w_2 \approx 20.714\) solves \(A \mathbf w = \mathbf b\). Which one does \(A^+ \mathbf b\) pick, and what makes it special among them? Use step (b) of the proof: what is the null space of this \(A\), and why is every message perpendicular to it?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (absolute loss as a linear program).** Show how to cast ERM for linear regression with the absolute loss, \(\min_{\mathbf w} \sum_{i=1}^m |\langle \mathbf w, \mathbf x_i \rangle - y_i|\), as a linear program. Start by proving that for any real \(c\), \(|c| = \min_{a \ge 0} a\) subject to \(c \le a\) and \(c \ge -a\).
