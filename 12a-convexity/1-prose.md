# Convexity

## The search that never comes back

Lesson 11 ended with a learning curve that would not come down. Your colleague’s halfspace filter, the one from lesson 9 that weighs word counts, kept a stubborn empirical risk even though the validation numbers said the class was good enough. The diagnosis was uncomfortable: the learner was not really doing ERM, the rule from lesson 2 that returns the hypothesis with the fewest training mistakes. The mail is not separable, so no halfspace gets every example right, and finding the one with the *fewest* mistakes is computationally hard. The code had been running a heuristic and hoping.

So this morning the team tried to do it properly. The plan was honest brute force: the real filter uses 30 words plus the bias term, the users have labelled 300 messages, and lesson 6 tells you that halfspaces cannot produce every possible labelling of those 300 messages, only a limited number of them. Enumerate the labellings a halfspace can produce, count the mistakes of each, keep the best. The job started at nine. At lunch it was still running. Someone did the arithmetic and quietly cancelled it.

Here is that arithmetic. Each message is a list of 31 numbers (30 word counts plus a constant 1 for the bias term), a point of \(\mathbb R^{31}\), the space of all lists of 31 real numbers. Homogeneous halfspaces there (no separate offset, since the constant coordinate plays that role) have VC dimension 31: as lesson 6 showed, 31 is the largest number of messages they can label in every possible way. Sauer’s lemma, also from lesson 6, then caps how many labellings they can produce of \(m\) messages when the VC dimension is \(d\): at most \(\sum_{i=0}^{d} \binom{m}{i}\). Read it from the inside out. \(\binom{m}{i}\), “\(m\) choose \(i\)”, is how many ways there are to pick \(i\) messages out of \(m\); the big sigma \(\sum_{i=0}^{d}\) adds those counts for \(i = 0, 1, \dots, d\). So: **the number of labellings is at most the number of ways to pick at most \(d\) of the messages.** With \(m = 300\) and \(d = 31\), the first three terms are already \(1 + 300 + 44{,}850\), and the terms keep growing up to \(i = 31\). Say each labelling costs \(m \cdot d\) operations to score, and the machine does a billion operations a second. The snippet mirrors the sum with `comb(m, i)` inside `sum(...)`.

```python
from math import comb

m, d = 300, 31
labelings = sum(comb(m, i) for i in range(d + 1))
print(len(str(labelings)), f"{labelings:.2e}")
seconds = labelings * m * d / 1e9
print(f"{seconds / 3.15e7:.1e}")
```

It prints `43 1.70e+42` and then `5.0e+29`. A 43-digit number of labellings, and about \(5 \times 10^{29}\) years of work. And that is generous, because it pretends we already know how to list the labellings one by one. As lesson 8 warned, the problem is not just slow to brute-force: finding the halfspace with the fewest mistakes on non-separable data is NP-hard, so no clever trick is expected to make it fast.

You remember what lesson 9 did with its ten small example messages, though. It did not count mistakes at all. It minimized the logistic loss, a smooth penalty that grows as a message lands further on the wrong side, and the minimization went smoothly, separable data or not, with a guarantee that the objective was “convex”. That word was used once and promised a whole lesson. This is the lesson.

The idea is simple to say. Counting mistakes is hard because the 0-1 loss, which charges 1 per mistake and 0 otherwise, is a landscape of flat terraces and cliffs. Replace it with a loss shaped like a bowl, one that is never smaller than the 0-1 loss, and minimize that instead. Bowls are easy: walk downhill and you reach the bottom. And because the new loss sits above the old one, a guarantee on the new loss is a guarantee on the mistakes too. By the end of this lesson you will know exactly what “shaped like a bowl” means, and you will run a walk down a bowl that proves, when it stops, that it has reached the bottom. Lesson 12b then asks which bowls can be learned from a finite label budget (not all of them, which is the surprise), and how much the switch costs you.

## Worked example 1: what the 0-1 loss looks like from inside

Let’s use the ten messages from the end of lesson 9, because you already know them. Each message has two word counts, “free” and “meeting”, and we put a constant 1 in front so the bias term is just the first weight. Labels are \(+1\) for junk and \(-1\) for keep. The last two messages are the ones your users disagreed with themselves about: a second “free trial, book a meeting” at \((2, 1)\) marked keep, and a second “free lunch at the meeting” at \((1, 2)\) marked junk.

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

A reminder from lesson 9: for a weight vector \(\mathbf w\) (bold letters are vectors), the inner product \(\langle \mathbf w, \mathbf x \rangle = w_1 x_1 + w_2 x_2 + w_3 x_3\) multiplies matching entries and adds them up, and the halfspace says junk when it is positive. Take lesson 9’s weights \(\mathbf w^\circ = (-1, 2, -2)\) and message B′ at \((1, 2, 1)\): the inner product is \(-1 + 4 - 2 = 1\), so the halfspace says junk, but the label is \(-1\), keep. Multiply by the label and you get \(-1\), a negative number that flags the disagreement. In general, call \(y \langle \mathbf w, \mathbf x \rangle\) the **signed score** of the example: positive when the halfspace agrees with the label, negative when it disagrees. The **0-1 loss** of a halfspace on an example charges 1 for a mistake and 0 otherwise:

\[ \ell^{0-1}(\mathbf w, (\mathbf x, y)) = \mathbb 1[\, y \langle \mathbf w, \mathbf x \rangle \le 0 \,]. \]

- \(\ell\) is the usual letter for a loss; the superscript \(0\text{-}1\) names which loss, and its two arguments are the hypothesis \(\mathbf w\) and the example \((\mathbf x, y)\).
- Inside, \(y \langle \mathbf w, \mathbf x \rangle\) is the signed score, and \(\le 0\) asks the yes/no question “is it zero or negative?”.
- \(\mathbb 1[\cdot]\) is the indicator: it turns a yes into 1 and a no into 0.

In plain words: **the loss is 1 when the signed score is not positive, and 0 otherwise**, so a score of exactly zero counts as a mistake. On B′ the score is \(-1\), so the loss is 1:

```python
w, x, y = (-1, 2, -2), (1, 2, 1), -1
score = sum(a * b for a, b in zip(w, x))
z = y * score
loss = 1 if z <= 0 else 0
print(score, z, loss)
```

It prints `1 -1 1`: the inner product, the signed score, the loss. The empirical risk \(L^{0-1}_S(\mathbf w)\) is, as in every lesson since lesson 1, the average of this loss over the training sequence \(S\), the fraction of mistakes; the function `risk01` in the next snippet is exactly that average. No weight vector can do better than 0.2 here: B and B′ are the same message with opposite labels, so every halfspace gets one of them wrong, and the same goes for E and E′. Lesson 9’s halfspace \(\mathbf w^\circ = (-1, 2, -2)\) reaches that floor, with exactly those two mistakes.

Now picture the empirical risk as a landscape over the weights. Take \(\mathbf w^\circ\) and its mirror image \(-\mathbf w^\circ\), and walk in a straight line from one to the other. Before you read the numbers: \(L^{0-1}_S(-\mathbf w^\circ) = 0.8\), since flipping every weight flips every prediction. Halfway along the line you are at \(\mathbf w = \mathbf 0\). What is the risk there, and how does it compare with the average of 0.2 and 0.8?

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))

def risk01(w):
    return sum(y * dot(w, x) <= 0 for x, y in zip(X, Y)) / 10

u, v = (-1, 2, -2), (1, -2, 2)
for a in (0, 0.25, 0.5, 0.75, 1):
    w = tuple(a * p + (1 - a) * q for p, q in zip(u, v))
    chord = a * risk01(u) + (1 - a) * risk01(v)
    print(a, risk01(w), round(chord, 2))
```

It prints

```
0 0.8 0.8
0.25 0.8 0.65
0.5 1.0 0.5
0.75 0.2 0.35
1 0.2 0.2
```

The middle column is the risk along the walk; the last column is the straight line between the two end values. At the midpoint the risk jumps to 1.0, twice the straight-line value: at \(\mathbf w = \mathbf 0\) every signed score is 0, and every example counts as a mistake. And for the first half of the walk the risk sits at 0.8 without moving, then drops off a cliff to 0.2.

[[diagram:staircase]]

That flatness is the real problem for any search that only looks nearby. Stand at \(-\mathbf w^\circ\). Its signed scores are \(-5, -1, -3, -1, -3, -3, -3, -1, 1, 3\), all at least 1 in absolute value. The double bars \(\|\mathbf x\|\) are the length (Euclidean norm) of a vector, the square root of the sum of its squared entries: for C, \(\|(1, 4, 2)\| = \sqrt{1 + 16 + 4} = \sqrt{21} \approx 4.58\), and no message is longer. Move the weights by any \(\boldsymbol\Delta\) with \(\|\boldsymbol\Delta\| < 1/\sqrt{21} \approx 0.218\). The Cauchy–Schwarz inequality says an inner product is never bigger in size than the product of the two lengths, \(|\langle \mathbf a, \mathbf b \rangle| \le \|\mathbf a\| \|\mathbf b\|\). So each score changes by \(|\langle \boldsymbol\Delta, \mathbf x \rangle| \le \|\boldsymbol\Delta\| \|\mathbf x\| < 1\), so no score changes sign, and the risk stays exactly 0.8. Every nearby point is equally bad. A search that steps to a better neighbour has nowhere to go, even though 0.2 is available elsewhere. The 0-1 loss gives a searcher no sense of direction.

The midpoint test above is the right instinct for what a good landscape looks like. Let’s make it a definition.

## Convex sets and convex functions

Start with the space the weights live in. A set is convex when you can walk between any two of its points in a straight line without leaving it.

**Definition (convex set).** A set \(C\) in a vector space is **convex** if for any two vectors \(\mathbf u, \mathbf v \in C\), the line segment between them is contained in \(C\): for every \(\alpha \in [0, 1]\),

\[ \alpha \mathbf u + (1 - \alpha) \mathbf v \in C. \]

- \(\mathbf u, \mathbf v \in C\) reads “\(\mathbf u\) and \(\mathbf v\) are elements of \(C\)”; \(\alpha \in [0, 1]\) says \(\alpha\) is a number between 0 and 1, ends included.
- \(\alpha \mathbf u\) scales every entry of \(\mathbf u\) by \(\alpha\); \((1 - \alpha)\mathbf v\) scales \(\mathbf v\) by the leftover weight; adding them mixes the two vectors, with weights that sum to 1.

The point \(\alpha \mathbf u + (1 - \alpha) \mathbf v\) is called a **convex combination** of \(\mathbf u\) and \(\mathbf v\). As \(\alpha\) runs from 1 to 0 it slides from \(\mathbf u\) to \(\mathbf v\). In plain words: **every point on the segment between two members is also a member.** The walk of worked example 1 was exactly this, with \(\mathbf u = \mathbf w^\circ\) and \(\mathbf v = -\mathbf w^\circ\): at \(\alpha = 0.25\) it stood at \(0.25(-1, 2, -2) + 0.75(1, -2, 2) = (0.5, -1, 1)\), and the line `w = tuple(a * p + (1 - a) * q ...)` of that snippet computes it.

A disc is convex, and so is a ball \(\{\mathbf w : \|\mathbf w\| \le B\}\) in any dimension: if \(\|\mathbf u\|, \|\mathbf v\| \le B\), the triangle inequality (the length of a sum is at most the sum of the lengths) gives \(\|\alpha \mathbf u + (1 - \alpha)\mathbf v\| \le \alpha B + (1 - \alpha) B = B\). The whole space \(\mathbb R^d\) is convex. A crescent moon or a ring is not: pick two points on opposite sides of the hole, and the segment between them leaves the set.

[[diagram:convex-set]]

**Definition (convex function).** Let \(C\) be a convex set. A function \(f : C \to \mathbb R\) is **convex** if for every \(\mathbf u, \mathbf v \in C\) and \(\alpha \in [0, 1]\),

\[ f(\alpha \mathbf u + (1 - \alpha)\mathbf v) \le \alpha f(\mathbf u) + (1 - \alpha) f(\mathbf v). \]

Read it from the inside out. \(f : C \to \mathbb R\) says \(f\) takes a point of \(C\) and returns one real number. On the left, \(f\) is evaluated at the mixed point, the height of the graph partway along. On the right, the two end heights \(f(\mathbf u)\) and \(f(\mathbf v)\) are mixed with the same weights, the height of the straight segment (the **chord**) joining them at the same spot. In plain words: **between any two points, the graph of \(f\) lies on or below its chord.** That is exactly the test the 0-1 risk failed: at \(\alpha = 1/2\) the left side was 1.0 and the right side 0.5, the middle and last columns of the walk snippet, whose `chord` line is the right side term for term. One counterexample is enough, so \(L^{0-1}_S\) is not convex.

There is a second way to see the same property, through a set. Take \(f(x) = x^2\): the point \((1, 2)\) sits above the graph, since \(1^2 = 1 \le 2\), while \((2, 3)\) sits below it, since \(2^2 = 4 > 3\). The **epigraph** of \(f\) collects everything on or above its graph:

\[ \operatorname{epigraph}(f) = \{ (\mathbf x, \beta) : f(\mathbf x) \le \beta \}. \]

The braces build a set, and the colon reads “such that”: all pairs of an input \(\mathbf x\) and a height \(\beta\) such that the graph at \(\mathbf x\) is no higher than \(\beta\). In plain words: **the epigraph is the region on and above the graph.** For \(x^2\), mixing two of its points halfway, \((1, 2)\) and \((-2, 5)\), lands at \((-0.5, 3.5)\), still inside:

```python
f = lambda x: x * x
in_epi = lambda p: f(p[0]) <= p[1]
p, q, a = (1, 2), (-2, 5), 0.5
mix = (a * p[0] + (1 - a) * q[0], a * p[1] + (1 - a) * q[1])
print(in_epi(p), in_epi(q), in_epi((2, 3)))
print(mix, in_epi(mix))
```

It prints `True True False` and `(-0.5, 3.5) True`.

**Claim.** \(f\) is convex if and only if its epigraph is a convex set.

*Proof.* Suppose \(f\) is convex, and take two points of the epigraph, \((\mathbf u, a)\) and \((\mathbf v, b)\), so \(f(\mathbf u) \le a\) and \(f(\mathbf v) \le b\). Their convex combination is \((\alpha \mathbf u + (1 - \alpha)\mathbf v, \; \alpha a + (1 - \alpha) b)\), and convexity of \(f\) gives \(f(\alpha \mathbf u + (1 - \alpha)\mathbf v) \le \alpha f(\mathbf u) + (1 - \alpha) f(\mathbf v) \le \alpha a + (1 - \alpha) b\), where the second step uses \(\alpha, 1 - \alpha \ge 0\). So the combination is in the epigraph. Conversely, suppose the epigraph is convex. The points \((\mathbf u, f(\mathbf u))\) and \((\mathbf v, f(\mathbf v))\) are in it, so their combination \((\alpha \mathbf u + (1 - \alpha)\mathbf v, \; \alpha f(\mathbf u) + (1 - \alpha) f(\mathbf v))\) is too, which says precisely that \(f(\alpha \mathbf u + (1 - \alpha)\mathbf v) \le \alpha f(\mathbf u) + (1 - \alpha) f(\mathbf v)\). \(\square\)

So a convex function is one whose region above the graph has no dents, which is the bowl from the opening, made precise.

[[diagram:chord]]

You can test the chord inequality by brute force. The snippet below tries 100,000 random triples \((u, v, \alpha)\) on \([-3, 3]\) for four functions of one variable and prints the worst violation, the largest amount by which the left side ever exceeded the right. The fourth is the 0-1 loss as a function of the signed score, \(\mathbb 1[z \le 0]\).

```python
import math, random

tests = {"x^2": lambda x: x * x,
         "softplus": lambda x: math.log(1 + math.exp(x)),
         "abs": abs,
         "step": lambda x: 1.0 if x <= 0 else 0.0}
rng = random.Random(0)
for name, f in tests.items():
    worst = 0.0
    for _ in range(100000):
        u, v = rng.uniform(-3, 3), rng.uniform(-3, 3)
        a = rng.random()
        gap = f(a * u + (1 - a) * v) - (a * f(u) + (1 - a) * f(v))
        worst = max(worst, gap)
    print(name, round(worst, 2))
```

It prints `x^2 0.0`, `softplus 0.0`, `abs 0.0` and `step 0.99`. The first three never violate the chord; the step violates it by almost its full height, when \(u\) is below 0, \(v\) just above it, and the mixed point lands at or below 0 while carrying most of its weight on \(v\): the left side is 1 and the chord barely above 0. (A random test can only find violations; proving convexity takes an argument, which is what the next few sections supply.)

## Every local minimum is global

Here is the property that makes convexity worth all this. Remember the plateau around \(-\mathbf w^\circ\): nothing nearby was lower, and yet 0.2 existed far away. Let \(B(\mathbf u, r) = \{\mathbf v : \|\mathbf v - \mathbf u\| \le r\}\) be the ball of radius \(r\) around \(\mathbf u\): the set of points \(\mathbf v\) such that the distance \(\|\mathbf v - \mathbf u\|\) from \(\mathbf u\) is at most \(r\). We say \(f(\mathbf u)\) is a **local minimum** of \(f\) at \(\mathbf u\) if there is some \(r > 0\) such that \(f(\mathbf v) \ge f(\mathbf u)\) for every \(\mathbf v \in B(\mathbf u, r)\): nothing nearby is lower. A **global minimum** is lower than or equal to every point, near or far. So \(-\mathbf w^\circ\) is a local minimum of the 0-1 risk, with value 0.8, and not a global one. Sample 10,000 points of the ball of radius 0.2 around it and collect the risks:

```python
import math, random

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
risk01 = lambda w: sum(y * dot(w, x) <= 0
                       for x, y in zip(X, Y)) / 10
u, r = (1, -2, 2), 0.2
rng, seen = random.Random(3), set()
for _ in range(10000):
    d = [rng.gauss(0, 1) for _ in u]
    s = r * rng.random() / math.sqrt(dot(d, d))
    seen.add(risk01([a + s * b for a, b in zip(u, d)]))
print(seen)
```

It prints `{0.8}`: each point is \(\mathbf u\) plus a random direction scaled to a length below \(r\), so it lies in \(B(\mathbf u, r)\), and none of them does better.

**Claim.** If \(f\) is convex, every local minimum of \(f\) is a global minimum.

*Proof.* Let \(f(\mathbf u)\) be a local minimum with radius \(r\), and let \(\mathbf v\) be any point, possibly far away. Walk a small fraction \(\alpha\) of the way from \(\mathbf u\) towards \(\mathbf v\): the point \(\mathbf u + \alpha(\mathbf v - \mathbf u)\) is at distance \(\alpha \|\mathbf v - \mathbf u\|\) from \(\mathbf u\), so for \(\alpha > 0\) small enough (any \(\alpha \le r / \|\mathbf v - \mathbf u\|\), and at most 1) it is inside \(B(\mathbf u, r)\). Local minimality then gives

\[ f(\mathbf u) \le f(\mathbf u + \alpha(\mathbf v - \mathbf u)). \]

But \(\mathbf u + \alpha(\mathbf v - \mathbf u) = \alpha \mathbf v + (1 - \alpha)\mathbf u\) is a convex combination, so convexity gives

\[ f(\mathbf u + \alpha(\mathbf v - \mathbf u)) = f(\alpha \mathbf v + (1 - \alpha)\mathbf u) \le (1 - \alpha) f(\mathbf u) + \alpha f(\mathbf v). \]

This is the chord inequality again, with the roles of \(\mathbf u\) and \(\mathbf v\) swapped, so nothing new to read. Chain the two: \(f(\mathbf u) \le (1 - \alpha) f(\mathbf u) + \alpha f(\mathbf v)\). Subtract \((1 - \alpha) f(\mathbf u)\) from both sides to get \(\alpha f(\mathbf u) \le \alpha f(\mathbf v)\), and divide by \(\alpha > 0\): \(f(\mathbf u) \le f(\mathbf v)\). Since \(\mathbf v\) was arbitrary, \(f(\mathbf u)\) is a global minimum. \(\square\)

Read that as a promise to a search algorithm: on a convex landscape, the moment you cannot improve locally, you are done globally.

## The tangent below the graph

A second property gives a searcher a sense of direction, and a way to know how far from the bottom it is. Take lesson 9’s logistic risk on the ten messages, the average of \(\log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\) (here \(\exp(t) = e^t\) and \(\log\) is the natural logarithm, its inverse), and stand at \(\mathbf w = \mathbf 0\), where it equals \(\log 2 \approx 0.6931\). Nudge one weight at a time by a tiny amount and watch how fast the risk changes: nudging the bias weight changes nothing, nudging the “free” weight up lowers the risk at rate 0.3, nudging the “meeting” weight up raises it at rate 0.1. Those three rates, collected into a vector, are the gradient. When \(f\) is differentiable, its **gradient** at \(\mathbf w\) is the vector of partial derivatives,

\[ \nabla f(\mathbf w) = \left( \frac{\partial f(\mathbf w)}{\partial w_1}, \dots, \frac{\partial f(\mathbf w)}{\partial w_d} \right). \]

- \(\nabla\), “nabla” or “grad”, names the gradient; \(\nabla f(\mathbf w)\) is a vector with one entry per weight.
- \(\frac{\partial f(\mathbf w)}{\partial w_j}\), the partial derivative with respect to \(w_j\), is the rate at which \(f\) changes when only the \(j\)-th weight moves and the others stay put.
- The subscripts run from 1 to \(d\), the number of weights.

In plain words: **the gradient lists, weight by weight, how steeply \(f\) rises.** The snippet nudges each weight by \(10^{-6}\) and divides the change by the nudge:

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in zip(X, Y)) / 10
w, h = [0.0, 0.0, 0.0], 1e-6
grad = []
for j in range(3):
    up = [c + h * (k == j) for k, c in enumerate(w)]
    grad.append(round((f(up) - f(w)) / h, 4))
print(round(f(w), 4), grad)
u = (-1, 2, -2)
diff = [a - b for a, b in zip(u, w)]
tangent = f(w) + dot(grad, diff)
print(round(f(u), 4), round(tangent, 4), f(u) >= tangent)
```

Its first line is `0.6931 [0.0, -0.3, 0.1]`; the second line uses the tangent plane, which comes next. The tangent plane at \(\mathbf w\) is the linear function \(\mathbf u \mapsto f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\) (the arrow \(\mapsto\) reads “sends \(\mathbf u\) to”): start at the height \(f(\mathbf w)\) and add, for each weight, its rate times how far \(\mathbf u\) moved in it. For a convex function, that tangent lies below the graph everywhere, not just near \(\mathbf w\).

**Claim (tangent inequality).** If \(f : \mathbb R^d \to \mathbb R\) is convex and differentiable, then for all \(\mathbf w, \mathbf u\),

\[ f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle. \]

The left side is the true height at \(\mathbf u\); the right side is the tangent plane at \(\mathbf w\) evaluated at \(\mathbf u\). In plain words: **the tangent plane at any point is a floor under the whole graph.** From \(\mathbf w = \mathbf 0\), predict the logistic risk at \(\mathbf u = \mathbf w^\circ = (-1, 2, -2)\): the tangent says \(0.6931 + (0 \cdot (-1) - 0.3 \cdot 2 + 0.1 \cdot (-2)) = -0.1069\), and the real risk is above it (the logistic risk is convex, as a later section proves). The last three lines of the gradient snippet compute both, and print `0.5503 -0.1069 True`.

*Proof.* For \(\alpha \in (0, 1]\), convexity with the points \(\mathbf u\) and \(\mathbf w\) gives \(f(\mathbf w + \alpha(\mathbf u - \mathbf w)) \le (1 - \alpha) f(\mathbf w) + \alpha f(\mathbf u)\). Subtract \(f(\mathbf w)\) and divide by \(\alpha\):

\[ \frac{f(\mathbf w + \alpha(\mathbf u - \mathbf w)) - f(\mathbf w)}{\alpha} \le f(\mathbf u) - f(\mathbf w). \]

The left side is a rise over a run: how much \(f\) changes when you step a fraction \(\alpha\) of the way towards \(\mathbf u\), divided by \(\alpha\). As \(\alpha \to 0\) (read “as \(\alpha\) shrinks towards 0”), the left side tends to the slope of \(\alpha \mapsto f(\mathbf w + \alpha(\mathbf u - \mathbf w))\) at 0, and the chain rule says that slope is each weight’s rate times how fast that weight moves, \(\langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\). The right side does not depend on \(\alpha\), so the inequality survives the limit. Rearranging gives the claim. \(\square\)

[[diagram:tangent]]

Two consequences are worth reading off right away. If \(\nabla f(\mathbf w) = \mathbf 0\), the tangent inequality says \(f(\mathbf u) \ge f(\mathbf w)\) for every \(\mathbf u\): a flat spot of a convex function is a global minimum. And more usefully, if the gradient is merely *small*, you are close to the bottom. We will turn that into a stopping rule with a certificate at the end of this lesson. Lesson 14 extends the tangent inequality to functions with kinks, such as \(|x|\), that have no gradient at some points.

For functions of one variable there is an easy test.

**Lemma (convexity in one variable).** Let \(f : \mathbb R \to \mathbb R\) be twice differentiable, with first and second derivatives \(f'\) and \(f''\). The following are equivalent:

1. \(f\) is convex;
2. \(f'\) is monotonically nondecreasing;
3. \(f''\) is nonnegative.

*Proof.* (1 ⇒ 2.) Take \(a < c\). The tangent inequality at \(a\), with \(u = c\), gives \(f(c) \ge f(a) + f'(a)(c - a)\); at \(c\), with \(u = a\), it gives \(f(a) \ge f(c) + f'(c)(a - c)\). Add the two and cancel \(f(a) + f(c)\): \(0 \ge f'(a)(c - a) - f'(c)(c - a)\), that is, \((f'(c) - f'(a))(c - a) \ge 0\). Since \(c - a > 0\), \(f'(c) \ge f'(a)\).

(2 ⇒ 1.) Take \(u < v\), \(\alpha \in (0, 1)\) and \(x = \alpha u + (1 - \alpha) v\), so \(u < x < v\). The mean value theorem says that between two points a differentiable function somewhere has slope equal to its average slope, so there are points \(\xi_1 \in (u, x)\) and \(\xi_2 \in (x, v)\) with \(f(x) - f(u) = f'(\xi_1)(x - u)\) and \(f(v) - f(x) = f'(\xi_2)(v - x)\). Since \(\xi_1 < \xi_2\) and \(f'\) is nondecreasing, \(f'(\xi_1) \le f'(\xi_2)\). Now \(x - u = (1 - \alpha)(v - u)\) and \(v - x = \alpha(v - u)\), so

\[ \alpha\,(f(x) - f(u)) = \alpha (1 - \alpha)(v - u) f'(\xi_1) \le \alpha(1 - \alpha)(v - u) f'(\xi_2) = (1 - \alpha)(f(v) - f(x)). \]

The outer two expressions are the two mean-value equations multiplied by \(\alpha\) and \(1 - \alpha\), and the middle step only swaps the smaller slope for the larger. Expand: \(\alpha f(x) - \alpha f(u) \le (1 - \alpha) f(v) - (1 - \alpha) f(x)\), so \(f(x) \le \alpha f(u) + (1 - \alpha) f(v)\). The cases \(u > v\) and \(\alpha \in \{0, 1\}\) follow by symmetry or are trivial.

(2 ⇔ 3.) A differentiable function, here \(f'\), is nondecreasing exactly when its derivative is nonnegative: if \(f'' \ge 0\), the mean value theorem gives \(f'(c) - f'(a) = f''(\xi)(c - a) \ge 0\) for \(a < c\); and if \(f'\) is nondecreasing, every difference quotient \((f'(a + t) - f'(a))/t\) with \(t > 0\) is nonnegative, so their limit \(f''(a)\) is too. \(\square\)

Two examples. The function \(f(x) = x^2\) has \(f''(x) = 2 > 0\), so it is convex. The **softplus** function \(f(x) = \log(1 + \exp(x))\) has

\[ f'(x) = \frac{\exp(x)}{1 + \exp(x)} = \frac{1}{\exp(-x) + 1}, \]

which is increasing because \(\exp(-x)\) decreases; so softplus is convex. The first fraction is the chain rule (the derivative of \(\log(1 + t)\) is \(1/(1 + t)\), times the derivative \(\exp(x)\) of the inside); the second divides top and bottom by \(\exp(x)\). In plain words: **the slope of softplus climbs from 0 towards 1 as \(x\) grows.** At \(x = 0\) it is \(1/(1 + 1) = 1/2\), and the line `df = lambda x: 1 / (1 + math.exp(-x))` in the next snippet is the second form, term for term. Differentiating once more gives \(\frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{1}{e^{x} + 1}\), so the second derivative is \(f'(x) f'(-x)\), the logistic function of lesson 9 times its mirror image. Let’s check all three statements on a grid, and the tangent inequality too.

```python
import math

f = lambda x: math.log(1 + math.exp(x))
df = lambda x: 1 / (1 + math.exp(-x))
d2f = lambda x: df(x) * df(-x)
grid = [k / 10 for k in range(-60, 61)]
print(all(df(a) <= df(b) for a, b in zip(grid, grid[1:])))
print(round(min(d2f(x) for x in grid), 4),
      round(max(d2f(x) for x in grid), 4))
gaps = [f(u) - (f(w) + df(w) * (u - w))
        for w in grid for u in grid]
print(round(min(gaps), 6))
```

It prints `True`, then `0.0025 0.25`, then `0.0`. The derivative never decreases along the grid, the second derivative stays between 0.0025 and 0.25 (it is largest at 0, where it equals \(1/4\); remember that number), and over all 14,641 pairs on the grid the graph is never below a tangent.

## Building convex losses from simple pieces

Checking convexity from scratch for every loss would be tedious. Instead we build losses from convex pieces with operations that keep convexity. The first operation is the one every linear learner uses: feed a linear function of the weights into a scalar function.

**Claim (composition with a linear function).** Suppose \(f : \mathbb R^d \to \mathbb R\) can be written as \(f(\mathbf w) = g(\langle \mathbf w, \mathbf x \rangle + y)\) for some \(\mathbf x \in \mathbb R^d\), \(y \in \mathbb R\), and \(g : \mathbb R \to \mathbb R\). If \(g\) is convex, then \(f\) is convex.

*Proof.* Let \(\mathbf w_1, \mathbf w_2 \in \mathbb R^d\) and \(\alpha \in [0, 1]\). The inner product is linear in its first argument, and \(y = \alpha y + (1 - \alpha) y\), so

\[
\begin{aligned}
f(\alpha \mathbf w_1 + (1 - \alpha)\mathbf w_2) &= g(\langle \alpha \mathbf w_1 + (1 - \alpha)\mathbf w_2, \mathbf x \rangle + y) \\
&= g(\alpha \langle \mathbf w_1, \mathbf x \rangle + (1 - \alpha)\langle \mathbf w_2, \mathbf x \rangle + y) \\
&= g(\alpha(\langle \mathbf w_1, \mathbf x \rangle + y) + (1 - \alpha)(\langle \mathbf w_2, \mathbf x \rangle + y)) \\
&\le \alpha g(\langle \mathbf w_1, \mathbf x \rangle + y) + (1 - \alpha) g(\langle \mathbf w_2, \mathbf x \rangle + y),
\end{aligned}
\]

where the first three lines only rewrite the argument of \(g\) (the inner product splits across the mixed weights), and the last step is convexity of \(g\) at the two numbers \(\langle \mathbf w_i, \mathbf x \rangle + y\). The last line is \(\alpha f(\mathbf w_1) + (1 - \alpha) f(\mathbf w_2)\). \(\square\)

Two losses you have met are convex because of this claim. The square loss \(f(\mathbf w) = (\langle \mathbf w, \mathbf x \rangle - y)^2\) is \(g(a) = a^2\) composed with a linear function. The logistic loss \(f(\mathbf w) = \log(1 + \exp(-y \langle \mathbf w, \mathbf x \rangle))\), for \(y \in \{\pm 1\}\), is softplus composed with \(\langle \mathbf w, -y\mathbf x \rangle\). (Lesson 9 made this argument inline; now it has a name.)

The second pair of operations lets us combine several convex functions.

**Claim (maxima and nonnegative sums).** For \(i = 1, \dots, r\), let \(f_i : \mathbb R^d \to \mathbb R\) be convex. Then these are convex too:

- \(g(\mathbf x) = \max_{i \in [r]} f_i(\mathbf x)\);
- \(g(\mathbf x) = \sum_{i=1}^r w_i f_i(\mathbf x)\), where \(w_i \ge 0\) for all \(i\).

Here \([r]\) is shorthand for \(\{1, 2, \dots, r\}\), so \(\max_{i \in [r]} f_i(\mathbf x)\) evaluates all \(r\) functions at \(\mathbf x\) and keeps the largest value; \(\sum_{i=1}^r w_i f_i(\mathbf x)\) multiplies each value by its weight \(w_i\) and adds them up. In plain words: **the upper envelope of convex functions is convex, and so is any blend of them with nonnegative weights.** Take \(f_1(x) = x\), \(f_2(x) = -x\), \(f_3(x) = x^2/4\), and weights \(1/2\) each. At \(u = -3\), \(v = 1\), \(\alpha = 1/2\), the maximum has \(g(-1) = 1\) against a chord value of \((3 + 1)/2 = 2\), and the sum has \(0.125\) against \(0.625\):

```python
fs = [lambda x: x, lambda x: -x, lambda x: x * x / 4]
g_max = lambda x: max(f(x) for f in fs)
g_sum = lambda x: sum(0.5 * f(x) for f in fs)
u, v, a = -3, 1, 0.5
for g in (g_max, g_sum):
    left = g(a * u + (1 - a) * v)
    print(left, a * g(u) + (1 - a) * g(v))
```

It prints `1.0 2.0` and `0.125 0.625`: both below the chord.

*Proof.* For the maximum,

\[
\begin{aligned}
g(\alpha \mathbf u + (1 - \alpha)\mathbf v) &= \max_i f_i(\alpha \mathbf u + (1 - \alpha)\mathbf v) \\
&\le \max_i \left[ \alpha f_i(\mathbf u) + (1 - \alpha) f_i(\mathbf v) \right] \\
&\le \alpha \max_i f_i(\mathbf u) + (1 - \alpha) \max_i f_i(\mathbf v) \\
&= \alpha g(\mathbf u) + (1 - \alpha) g(\mathbf v).
\end{aligned}
\]

The first inequality applies convexity of each \(f_i\) inside the maximum. The second holds because, for each \(i\), \(\alpha f_i(\mathbf u) \le \alpha \max_j f_j(\mathbf u)\) and \((1 - \alpha) f_i(\mathbf v) \le (1 - \alpha)\max_j f_j(\mathbf v)\), so the bracket is at most the right side for every \(i\), and hence so is its maximum. For the weighted sum,

\[
\begin{aligned}
g(\alpha \mathbf u + (1 - \alpha)\mathbf v) &= \sum_i w_i f_i(\alpha \mathbf u + (1 - \alpha)\mathbf v) \\
&\le \sum_i w_i \left[ \alpha f_i(\mathbf u) + (1 - \alpha) f_i(\mathbf v) \right] \\
&= \alpha \sum_i w_i f_i(\mathbf u) + (1 - \alpha) \sum_i w_i f_i(\mathbf v) = \alpha g(\mathbf u) + (1 - \alpha) g(\mathbf v),
\end{aligned}
\]

where the inequality needs \(w_i \ge 0\): multiplying an inequality by a negative weight would flip it. \(\square\)

For example, \(|x| = \max\{x, -x\}\), and both \(x\) and \(-x\) are convex (a linear function meets its own chord with equality), so \(|x|\) is convex. And an empirical risk \(L_S(\mathbf w) = \frac1m \sum_i \ell(\mathbf w, z_i)\), the average loss over the \(m\) examples \(z_1, \dots, z_m\) of \(S\), is a sum with weights \(1/m \ge 0\), so it is convex whenever each \(\ell(\cdot, z_i)\) is.

Now the loss this lesson is really about. The 0-1 loss charges B′ (signed score \(-1\)) and E′ (signed score \(-3\)) the same 1, though E′ is the more confident mistake. Suppose instead we charge “how far the score falls short of 1”, and nothing when it doesn’t: B′ pays \(1 - (-1) = 2\), E′ pays 4, and message A, with score 5, pays 0. Take the signed score \(z = y \langle \mathbf w, \mathbf x \rangle\) and define the **hinge loss**

\[ \ell^{\text{hinge}}(\mathbf w, (\mathbf x, y)) = \max\{0, \; 1 - y \langle \mathbf w, \mathbf x \rangle\}. \]

Inside, \(1 - y\langle \mathbf w, \mathbf x \rangle\) is the shortfall of the signed score below 1; \(\max\{0, \cdot\}\) keeps the larger of 0 and that shortfall, so a negative shortfall becomes 0. In plain words: **the hinge loss charges how far the signed score falls short of 1.** It is zero when the signed score is at least 1 (right, with room to spare), and it grows linearly as the score falls below 1, so a confident mistake costs more than a narrow one. On the signed scores of \(\mathbf w^\circ\) for the ten messages, plus a narrow correct score of 0.5:

```python
scores = [5, 1, 3, 1, 3, 3, 3, 1, -1, -3, 0.5]
print([max(0, 1 - z) for z in scores])
```

It prints `[0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0.5]`. Even the narrow correct score pays 0.5: the hinge loss asks for a margin, not just the right side. It is the maximum of two functions of \(\mathbf w\), the constant 0 and \(1 - y\langle \mathbf w, \mathbf x \rangle\), and both are linear plus a constant, hence convex. By the claim on maxima, the hinge loss is convex in \(\mathbf w\), and by the claim on sums, so is its empirical risk. Time to look at the two landscapes yourself.

[[lab]]

In the lab you saw the 0-1 risk as terraces and the hinge risk as a bowl, on a slice of weight space where the bias term is fixed at \(-1\). A walker that follows the slope stalls at once on the 0-1 terraces, because the slope is zero everywhere it is defined. On the hinge bowl the same walker slides to the bottom, at hinge risk 0.50, and the halfspace it stops at makes only the two forced mistakes. That is the whole strategy in one picture. But notice what you had to do to know the walker had arrived: you looked at the picture. A program cannot look. The rest of this lesson builds the tools that let a walk *prove* it has reached the bottom: how steep a loss can be, how fast its slope can change, and a certificate that turns a small gradient into a guarantee.

## Lipschitzness: losses that cannot change too fast

Convexity says the landscape has one valley. It says nothing about how steep the walls are, and steepness turns out to matter for learning. The absolute value \(|x|\) never climbs faster than slope 1: from 3 to \(-1\) the input moves 4 and the output only 2. That cap on the slope is the property we want, measured with the Euclidean norm.

**Definition (Lipschitzness).** Let \(C \subseteq \mathbb R^d\). A function \(f : \mathbb R^d \to \mathbb R^k\) is **\(\rho\)-Lipschitz** over \(C\) if for every \(\mathbf w_1, \mathbf w_2 \in C\),

\[ \| f(\mathbf w_1) - f(\mathbf w_2) \| \le \rho \, \| \mathbf w_1 - \mathbf w_2 \|. \]

\(C \subseteq \mathbb R^d\) says \(C\) is a subset of \(\mathbb R^d\), and \(f : \mathbb R^d \to \mathbb R^k\) allows the output to be a vector of \(k\) numbers, which is why the left side uses the length \(\|\cdot\|\) too (for one number it is just the absolute value). The left side is how far apart the outputs are; the right side is \(\rho\) times how far apart the inputs are. In plain words: **the output never moves more than \(\rho\) times as far as the input.** For \(|x|\) with \(\rho = 1\), the pair \(3, -1\) above gives \(2 \le 1 \cdot 4\). For a differentiable \(f : \mathbb R \to \mathbb R\) there is an easy sufficient condition. The mean value theorem gives \(f(w_1) - f(w_2) = f'(u)(w_1 - w_2)\) for some \(u\) between \(w_1\) and \(w_2\), so if \(|f'| \le \rho\) everywhere, \(f\) is \(\rho\)-Lipschitz.

Four examples, each a small proof.

- **\(|x|\) is 1-Lipschitz over \(\mathbb R\).** By the triangle inequality, \(|x_1| - |x_2| = |x_1 - x_2 + x_2| - |x_2| \le |x_1 - x_2| + |x_2| - |x_2| = |x_1 - x_2|\). Swapping the roles of \(x_1\) and \(x_2\) gives \(|x_2| - |x_1| \le |x_1 - x_2|\), and together \(\big||x_1| - |x_2|\big| \le |x_1 - x_2|\).
- **Softplus \(\log(1 + \exp(x))\) is 1-Lipschitz over \(\mathbb R\).** Its derivative is \(1/(\exp(-x) + 1)\), which lies strictly between 0 and 1.
- **\(x^2\) is not \(\rho\)-Lipschitz over \(\mathbb R\) for any \(\rho\).** Take \(x_1 = 0\) and \(x_2 = 1 + \rho\). Then \(f(x_2) - f(x_1) = (1 + \rho)^2 > \rho(1 + \rho) = \rho|x_2 - x_1|\). It is \(\rho\)-Lipschitz on the bounded set \(C = \{x : |x| \le \rho/2\}\), though: there \(|x_1^2 - x_2^2| = |x_1 + x_2|\,|x_1 - x_2| \le 2(\rho/2)|x_1 - x_2| = \rho|x_1 - x_2|\).
- **A linear function \(f(\mathbf w) = \langle \mathbf v, \mathbf w \rangle + b\) is \(\|\mathbf v\|\)-Lipschitz.** By the Cauchy–Schwarz inequality, \(|f(\mathbf w_1) - f(\mathbf w_2)| = |\langle \mathbf v, \mathbf w_1 - \mathbf w_2 \rangle| \le \|\mathbf v\| \, \|\mathbf w_1 - \mathbf w_2\|\).

[[diagram:lipschitz]]

The snippet measures the steepest chord slope \(|f(a) - f(b)|/|b - a|\) over a fine grid, which is the smallest \(\rho\) that works on that grid: the definition divided through by the input distance, mirrored by `abs(f(a) - f(b)) / (b - a)`. With \(\rho = 3\), it also checks the \(x^2\) counterexample.

```python
import math

def max_slope(f, lo, hi, n=400):
    pts = [lo + (hi - lo) * k / n for k in range(n + 1)]
    return max(abs(f(a) - f(b)) / (b - a)
               for i, a in enumerate(pts) for b in pts[i + 1:])

rho = 3
print(round(max_slope(abs, -5, 5), 3))
print(round(max_slope(lambda x: math.log(1 + math.exp(x)),
                      -5, 5), 3))
print(round(max_slope(lambda x: x * x, -rho / 2, rho / 2), 3))
print((1 + rho) ** 2, rho * (1 + rho))
```

It prints `1.0`, `0.993`, `2.993` and `16 12`. The absolute value reaches slope exactly 1; softplus approaches 1 but never reaches it on a bounded interval; \(x^2\) on \([-1.5, 1.5]\) stays under \(\rho = 3\); and over the whole line, \(16 > 12\) shows \(x^2\) outrunning slope 3 between 0 and 4.

Lipschitz constants multiply under composition, which is how we get them for losses.

**Claim (composition).** Let \(f(\mathbf x) = g_1(g_2(\mathbf x))\), where \(g_1\) is \(\rho_1\)-Lipschitz and \(g_2\) is \(\rho_2\)-Lipschitz. Then \(f\) is \((\rho_1 \rho_2)\)-Lipschitz. In particular, if \(g_2(\mathbf x) = \langle \mathbf v, \mathbf x \rangle + b\) for some \(\mathbf v \in \mathbb R^d\) and \(b \in \mathbb R\), then \(f\) is \((\rho_1 \|\mathbf v\|)\)-Lipschitz.

*Proof.* Apply the two Lipschitz conditions one after the other:

\[ |f(\mathbf w_1) - f(\mathbf w_2)| = |g_1(g_2(\mathbf w_1)) - g_1(g_2(\mathbf w_2))| \le \rho_1 \|g_2(\mathbf w_1) - g_2(\mathbf w_2)\| \le \rho_1 \rho_2 \|\mathbf w_1 - \mathbf w_2\|. \]

Each \(\le\) is one Lipschitz condition: first for \(g_1\), whose inputs are the two values of \(g_2\), then for \(g_2\). The special case uses \(\rho_2 = \|\mathbf v\|\) from the linear example. \(\square\)

Apply it to the hinge loss. As a function of the signed score, \(\phi(a) = \max\{0, 1 - a\}\) is 1-Lipschitz: both pieces, \(0\) and \(1 - a\), are 1-Lipschitz, and a maximum of two \(\rho\)-Lipschitz functions is \(\rho\)-Lipschitz, because changing each argument of a maximum by at most \(t\) changes the maximum by at most \(t\). The signed score \(\mathbf w \mapsto y \langle \mathbf w, \mathbf x \rangle = \langle y \mathbf x, \mathbf w \rangle\) is linear with \(\|y \mathbf x\| = \|\mathbf x\|\), since \(|y| = 1\). By the composition claim:

**The hinge loss on an example \((\mathbf x, y)\) is \(\|\mathbf x\|\)-Lipschitz in \(\mathbf w\).** So if every message has \(\|\mathbf x\| \le R\), the hinge loss is \(R\)-Lipschitz on every example.

Let’s test it on two of our messages, the longest, C at \(\sqrt{21}\), and the shortest, H at 1, with 50,000 random pairs of weight vectors each.

```python
import math, random

X = [(1, 3, 0), (1, 2, 1), (1, 4, 2), (1, 1, 0), (1, 1, 2),
     (1, 0, 1), (1, 2, 3), (1, 0, 0)]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
norm = lambda u: math.sqrt(dot(u, u))
hinge = lambda w, x, y: max(0, 1 - y * dot(w, x))
rng = random.Random(2)
for x in (X[2], X[7]):
    ratio = 0.0
    for _ in range(50000):
        w1, w2 = ([rng.uniform(-2, 2) for _ in x] for _ in "12")
        d = [a - b for a, b in zip(w1, w2)]
        ratio = max(ratio, abs(hinge(w1, x, 1) - hinge(w2, x, 1))
                    / norm(d))
    print(x, round(ratio, 2), round(norm(x), 2))
```

It prints `(1, 4, 2) 4.58 4.58` and `(1, 0, 0) 1.0 1.0`: the steepest observed ratio climbs right up to \(\|\mathbf x\|\) and never past it. For the ten messages, \(R = \sqrt{21} \approx 4.58\).

## Smoothness: gradients that cannot change too fast

Lipschitzness bounds how fast the loss changes. Smoothness bounds how fast its *slope* changes. Softplus has slope \(1/2\) at 0 and \(0.7311\) at 1: over a distance of 1 its slope changed by \(0.2311\), and as we will see it never changes faster than \(1/4\) per unit of distance.

**Definition (smoothness).** A differentiable function \(f : \mathbb R^d \to \mathbb R\) is **\(\beta\)-smooth** if its gradient is \(\beta\)-Lipschitz: for all \(\mathbf v, \mathbf w\),

\[ \|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \beta \|\mathbf v - \mathbf w\|. \]

It is the Lipschitz definition with the gradient \(\nabla f\) in place of \(f\) and \(\beta\) in place of \(\rho\). In plain words: **the slope never changes more than \(\beta\) times as fast as the point moves.** Softplus’s \(0.2311 \le \frac14 \cdot 1\) is one instance; the first line of the softplus snippet below takes the largest such ratio over a grid.

Why care? Because smoothness puts a ceiling over the function, just as convexity put a floor under it.

**Claim (quadratic upper bound).** If \(f\) is \(\beta\)-smooth, then for all \(\mathbf v, \mathbf w\),

\[ f(\mathbf v) \le f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle + \frac{\beta}{2} \|\mathbf v - \mathbf w\|^2. \]

The first two terms on the right are the tangent plane at \(\mathbf w\); the third adds a bowl, \(\beta/2\) times the squared distance from \(\mathbf w\). In plain words: **the tangent plus a bowl of curvature \(\beta\) is a ceiling over the graph.** For softplus at \(w = 0\), \(v = 2\): the tangent gives \(0.6931 + 0.5 \cdot 2 = 1.6931\), the bowl adds \(\frac18 \cdot 4 = 0.5\), and the ceiling 2.1931 is above the true \(f(2) = 2.1269\), which is above the tangent floor. The `upper` line of the softplus snippet below is this inequality, term by term.

*Proof.* Walk from \(\mathbf w\) to \(\mathbf v\) along the segment and set \(h(t) = f(\mathbf w + t(\mathbf v - \mathbf w))\) for \(t \in [0, 1]\). By the chain rule, \(h'(t) = \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)), \mathbf v - \mathbf w \rangle\), which is continuous in \(t\) because the gradient is Lipschitz. The fundamental theorem of calculus gives

\[ f(\mathbf v) - f(\mathbf w) = h(1) - h(0) = \int_0^1 \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)), \mathbf v - \mathbf w \rangle \, dt. \]

The integral sign \(\int_0^1 \dots \, dt\) adds up the quantity inside over every \(t\) from 0 to 1, in infinitely thin slices of width \(dt\). In plain words: **the total change along the segment is the sum of all the small changes along the way.** For \(f(x) = x^2\) from \(w = 1\) to \(v = 3\), the slope along the way is \(h'(t) = 2(1 + 2t) \cdot 2 = 4 + 8t\), and adding it up from 0 to 1 gives \(4 + 4 = 8 = 9 - 1\). Add and subtract \(\nabla f(\mathbf w)\) inside the inner product:

\[ f(\mathbf v) - f(\mathbf w) = \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle + \int_0^1 \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)) - \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle \, dt. \]

By Cauchy–Schwarz and then smoothness, the integrand is at most \(\|\nabla f(\mathbf w + t(\mathbf v - \mathbf w)) - \nabla f(\mathbf w)\| \, \|\mathbf v - \mathbf w\| \le \beta t \|\mathbf v - \mathbf w\|^2\), since the two points where the gradient is taken are \(t\|\mathbf v - \mathbf w\|\) apart. Adding up \(\beta t\) from 0 to 1 gives \(\beta/2\) (the area of a triangle of width 1 and height \(\beta\)), so the integral is at most \(\frac\beta2 \|\mathbf v - \mathbf w\|^2\), which is the claim. \(\square\)

Put this next to the tangent inequality. When \(f\) is convex and \(\beta\)-smooth, the difference between \(f\) and its first-order approximation is caught between two bounds:

\[ 0 \le f(\mathbf v) - f(\mathbf w) - \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle \le \frac{\beta}{2}\|\mathbf v - \mathbf w\|^2. \]

The middle is the height of the graph above its tangent: for softplus at \(w = 0\), \(v = 2\), it is \(2.1269 - 1.6931 = 0.4338\), between 0 and the bowl’s 0.5.

[[diagram:sandwich]]

The ceiling says something practical about stepping downhill. Choose \(\mathbf v = \mathbf w - \frac{1}{\beta}\nabla f(\mathbf w)\), a step against the gradient of length \(\|\nabla f(\mathbf w)\|/\beta\). Then \(\mathbf v - \mathbf w = -\frac1\beta \nabla f(\mathbf w)\), and the quadratic upper bound becomes

\[ f(\mathbf v) \le f(\mathbf w) - \frac1\beta \|\nabla f(\mathbf w)\|^2 + \frac{\beta}{2} \cdot \frac{1}{\beta^2} \|\nabla f(\mathbf w)\|^2 = f(\mathbf w) - \frac{1}{2\beta}\|\nabla f(\mathbf w)\|^2. \]

Each term is the quadratic upper bound with \(\mathbf v - \mathbf w\) replaced by the step: the inner product of the gradient with \(-\frac1\beta \nabla f(\mathbf w)\) is \(-\frac1\beta\) times the squared length \(\|\nabla f(\mathbf w)\|^2\), and the bowl term shrinks to half that size. Rearranged: \(\frac{1}{2\beta}\|\nabla f(\mathbf w)\|^2 \le f(\mathbf w) - f(\mathbf v)\). In plain words: **one step of size \(1/\beta\) against the gradient is *guaranteed* to lower \(f\) by at least \(\|\nabla f(\mathbf w)\|^2/(2\beta)\).** For softplus at \(w = 2\), the slope is 0.8808, so the guarantee is \(0.8808^2 / (2 \cdot \frac14) = 1.5516\); the last line of the snippet below takes the step and finds an actual drop of 1.9297. If, in addition, \(f(\mathbf v) \ge 0\) for all \(\mathbf v\), the right side is at most \(f(\mathbf w)\), and we get

\[ \|\nabla f(\mathbf w)\|^2 \le 2\beta f(\mathbf w). \]

A function with this property is called **self-bounded**. The left side is the squared steepness, the right side \(2\beta\) times the height. In plain words: **where its value is small, its gradient must be small too.** For \(x^2\) (2-smooth) at \(w = 3\): \(6^2 = 36\) and \(2 \cdot 2 \cdot 9 = 36\), equality, as the next paragraph explains; the third line of the snippet below checks \(2\beta f - (f')^2 \ge 0\) for softplus. Lessons 13 and 14 lean on it.

Examples. \(f(x) = x^2\) has \(f'(x) = 2x\), which is 2-Lipschitz, so \(x^2\) is 2-smooth, and for it both bounds hold with equality: \(v^2 = w^2 + 2w(v - w) + (v - w)^2\) and \((2w)^2 = 2 \cdot 2 \cdot w^2\). Softplus has second derivative \(f'(x) f'(-x) = 1/((1 + e^{-x})(1 + e^{x}))\), which the convexity grid already showed peaks at \(1/4\). To prove it, expand the bottom: \(2 + e^x + e^{-x} \ge 4\), because \(e^x + e^{-x} - 2 = (e^{x/2} - e^{-x/2})^2 \ge 0\). So \(f'\) is \(\frac14\)-Lipschitz (by the mean value theorem, as for Lipschitz functions), softplus is \(\frac14\)-smooth, and since it is nonnegative it is self-bounded too.

```python
import math

f = lambda x: math.log(1 + math.exp(x))
df = lambda x: 1 / (1 + math.exp(-x))
beta = 1 / 4
grid = [k / 20 for k in range(-100, 101)]
print(round(max(abs(df(a) - df(b)) / (b - a)
                for a, b in zip(grid, grid[1:])), 4))
upper = min(f(w) + df(w) * (v - w) + beta / 2 * (v - w) ** 2
            - f(v) for w in grid for v in grid)
print(round(upper, 6))
print(round(min(2 * beta * f(w) - df(w) ** 2 for w in grid), 4))
w = 2.0
v = w - df(w) / beta
print(round(f(w) - f(v), 4), round(df(w) ** 2 / (2 * beta), 4))
```

It prints `0.2499`, `0.0`, `0.0033` and `1.9297 1.5516`: the slope never changes faster than \(\frac14\); the ceiling is never below the function (the gap is 0 only at \(v = w\)); softplus is self-bounded on the grid; and the step from \(w = 2\) beats its guaranteed drop.

As with convexity and Lipschitzness, smoothness passes through a linear function, with a factor.

**Claim (smoothness through a linear function).** Let \(f(\mathbf w) = g(\langle \mathbf w, \mathbf x \rangle + b)\), where \(g : \mathbb R \to \mathbb R\) is \(\beta\)-smooth, \(\mathbf x \in \mathbb R^d\) and \(b \in \mathbb R\). Then \(f\) is \((\beta \|\mathbf x\|^2)\)-smooth.

*Proof.* By the chain rule, \(\nabla f(\mathbf w) = g'(\langle \mathbf w, \mathbf x \rangle + b)\,\mathbf x\). So for any \(\mathbf v, \mathbf w\),

\[ \|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| = |g'(\langle \mathbf v, \mathbf x \rangle + b) - g'(\langle \mathbf w, \mathbf x \rangle + b)| \, \|\mathbf x\| \le \beta \, |\langle \mathbf v - \mathbf w, \mathbf x \rangle| \, \|\mathbf x\| \le \beta \|\mathbf x\|^2 \|\mathbf v - \mathbf w\|, \]

using the chain-rule formula for both gradients (they share the factor \(\mathbf x\), whose length comes out), the smoothness of \(g\) (its derivative is \(\beta\)-Lipschitz) and then Cauchy–Schwarz. \(\square\)

In plain words: **through a message \(\mathbf x\), a \(\beta\)-smooth scalar loss becomes \(\beta\|\mathbf x\|^2\)-smooth.** The logistic loss on message C, with \(\|\mathbf x\|^2 = 21\), should then be \(21/4 = 5.25\)-smooth. The snippet draws 50,000 pairs of nearby weight vectors, where the signed score is near 0 and softplus bends hardest, and takes the largest ratio of gradient change to weight change, the definition’s two sides:

```python
import math, random

x, y = (1, 4, 2), 1
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
norm = lambda u: math.sqrt(dot(u, u))
dg = lambda a: -1 / (1 + math.exp(a))
grad = lambda w: [dg(y * dot(w, x)) * y * c for c in x]
rng, ratio = random.Random(4), 0.0
for _ in range(50000):
    v, w = ([rng.uniform(-0.2, 0.2) for _ in x] for _ in "vw")
    gap = [a - b for a, b in zip(grad(v), grad(w))]
    d = [a - b for a, b in zip(v, w)]
    ratio = max(ratio, norm(gap) / norm(d))
print(round(ratio, 2), dot(x, x) / 4)
```

It prints `5.21 5.25`: close to the constant, never past it. The line `grad` is the chain-rule formula \(g'(\cdot)\,y\mathbf x\), with `dg` the slope of \(a \mapsto \log(1 + e^{-a})\).

So, for \(\mathbf x \in \mathbb R^d\) and \(y \in \mathbb R\), the square loss \(f(\mathbf w) = (\langle \mathbf w, \mathbf x \rangle - y)^2\) is \((2\|\mathbf x\|^2)\)-smooth (here \(g(a) = a^2\) and \(b = -y\)), and for \(y \in \{\pm 1\}\) the logistic loss \(f(\mathbf w) = \log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\) is \((\|\mathbf x\|^2/4)\)-smooth (softplus composed with \(\langle \mathbf w, -y\mathbf x \rangle\), and \(\|-y\mathbf x\| = \|\mathbf x\|\)), as the snippet just measured. The hinge loss is not smooth at all: at signed score 1 it has a kink, where its gradient jumps from \(-y\mathbf x\) to \(\mathbf 0\), and no gradient exists there.

## Convex learning problems

Now we can say what kind of learning problem all this is about. Lesson 3 defined a learning problem by three things: a hypothesis class \(\mathcal H\), a set of examples \(Z\), and a loss function \(\ell : \mathcal H \times Z \to \mathbb R_+\), which takes a hypothesis and an example (the \(\times\) pairs them) and returns a nonnegative number (\(\mathbb R_+\)). Until now \(\mathcal H\) was a set of functions from messages to labels. But \(\mathcal H\) can be any set, and for linear learners it is natural to identify each hypothesis with its weight vector. So from here on, \(\mathcal H \subseteq \mathbb R^d\), a hypothesis is written \(\mathbf w\), and the loss is \(\ell(\mathbf w, z)\). For each example \(z\), write \(\ell(\cdot, z)\) for the function \(\mathbf w \mapsto \ell(\mathbf w, z)\).

**Definition (convex learning problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex** if the hypothesis class \(\mathcal H\) is a convex set and, for every \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex function.

Linear regression with the square loss is one. Its hypotheses \(\mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle\) are parametrized by \(\mathbf w \in \mathbb R^d\), so take \(\mathcal H = \mathbb R^d\), a convex set; the examples are \(Z = \mathbb R^d \times \mathbb R = \mathbb R^{d+1}\); and \(\ell(\mathbf w, (\mathbf x, y)) = (\langle \mathbf w, \mathbf x \rangle - y)^2\) is convex in \(\mathbf w\), as we just showed. Logistic regression is another, and so is “halfspaces with the hinge loss”.

**Lemma (ERM is a convex optimization problem).** If \(\ell\) is a convex loss and \(\mathcal H\) is a convex set, then the ERM problem over \(\mathcal H\), minimizing the empirical risk over \(\mathcal H\), is a convex optimization problem: minimizing a convex function over a convex set.

*Proof.* Recall from lesson 2 that ERM returns a hypothesis with the smallest empirical risk. In symbols, \(\mathrm{ERM}_{\mathcal H}(S) = \operatorname{argmin}_{\mathbf w \in \mathcal H} L_S(\mathbf w)\): \(\min\) would give the smallest value of \(L_S\) over \(\mathcal H\), while \(\operatorname{argmin}\) gives the argument, the weight vector \(\mathbf w\) that achieves it. In plain words: **ERM picks the weights in the class with the lowest training score.** Among four candidate weight vectors with the 0-1 loss on the ten messages, it picks \(\mathbf w^\circ\):

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
L_S = lambda w: sum(y * dot(w, x) <= 0
                    for x, y in zip(X, Y)) / len(S)
H = [(1, -2, 2), (0, 0, 0), (-1, 1, 0), (-1, 2, -2)]
print([L_S(w) for w in H], min(H, key=L_S))
```

It prints `[0.8, 1.0, 0.5, 0.2] (-1, 2, -2)`: `min(H, key=L_S)` is the argmin. For a training sequence \(S = (z_1, \dots, z_m)\), \(L_S(\mathbf w) = \frac1m \sum_{i=1}^m \ell(\mathbf w, z_i)\) is a sum of convex functions with nonnegative weights \(1/m\), so it is convex by the claim on sums. The constraint \(\mathbf w \in \mathcal H\) is membership in a convex set. \(\square\)

Under mild conditions, problems of this shape can be solved efficiently by generic optimization algorithms, and lesson 14 presents a very simple one. Combined with the local-minimum claim, this is why a convex ERM problem is not a search through \(10^{42}\) labellings: any point you cannot improve locally is already the answer.


## A certificate for the bottom

The lab left one question open. Your walker on the hinge bowl reached 0.50, and you could see from the heatmap that this was the bottom. A program cannot see a heatmap. When should it stop, and how can it prove to someone who was not watching that the point it stopped at is as good as any other? The 0-1 walker is the warning: its gradient was zero at the very start, and it was nowhere near the best. A zero gradient on a terrace proves nothing.

On a convex function it proves a lot, and the tangent inequality is the reason. Let’s see the idea in numbers before any formula. Take lesson 9’s logistic risk on the ten messages, the average of \(\log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\) from the tangent section, and stand at act 1’s starting point, \(\mathbf w = (-1, -2, 2)\). The risk there is 2.272, the gradient has length 0.804, and the weights have length 3. Now compare yourself with every weight vector \(\mathbf u\) in the ball of radius 3. None of them is farther from you than \(3 + 3 = 6\). The tangent plane at \(\mathbf w\) is a floor under the whole graph, and that floor falls by at most 0.804 per unit of distance, in whatever direction you walk. So no \(\mathbf u\) in the ball can have a risk more than \(0.804 \times 6 \approx 4.82\) below yours. That is a weak promise at the start. Walk downhill and the gradient shrinks, and the same promise becomes strong.

**Claim (a certificate of near-optimality).** If \(f\) is convex and differentiable, then for every \(\mathbf w\) and every \(\mathbf u\) with \(\|\mathbf u\| \le B\),

\[ f(\mathbf w) - f(\mathbf u) \le \|\nabla f(\mathbf w)\| \, (\|\mathbf w\| + B). \]

- \(\|\mathbf u\| \le B\) says the competitor \(\mathbf u\) lies in the ball of radius \(B\) around \(\mathbf 0\), the convex set \(\{\mathbf u : \|\mathbf u\| \le B\}\) from the section on convex sets.
- On the left, \(f(\mathbf w) - f(\mathbf u)\) is how much worse you are than that competitor (negative if you are better).
- On the right, \(\|\nabla f(\mathbf w)\|\) is the length of the gradient, the steepest rate of the tangent floor, and \(\|\mathbf w\| + B\) is the farthest any competitor in the ball can be from you: \(\|\mathbf w\|\) to get back to \(\mathbf 0\), then at most \(B\) more.

In plain words: **a short gradient proves you are close to the best in the ball.** At the starting point, \(0.804 \times (3 + 3) = 4.8241\), the number from above. The line `cert = norm(g) * (norm(w) + B)` of the snippet is the right side, term for term:

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
norm = lambda u: math.sqrt(dot(u, u))
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in zip(X, Y)) / 10
grad = lambda w: [sum(-y * x[j] / (1 + math.exp(y * dot(w, x)))
                      for x, y in zip(X, Y)) / 10
                  for j in range(3)]
B, w = 3, [-1.0, -2.0, 2.0]
g = grad(w)
cert = norm(g) * (norm(w) + B)
print(round(f(w), 4), round(norm(g), 4), norm(w), round(cert, 4))
```

It prints `2.272 0.804 3.0 4.8241`: the risk, the gradient length, the weight length and the certificate. The line `grad` is the chain rule applied to each example’s loss: the slope of \(a \mapsto \log(1 + e^{-a})\) at the signed score, which is \(-1/(1 + e^{a})\), times \(y\mathbf x\).

*Proof.* The tangent inequality at \(\mathbf w\) gives \(f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\). Move the terms around: \(f(\mathbf w) - f(\mathbf u) \le \langle \nabla f(\mathbf w), \mathbf w - \mathbf u \rangle\). By Cauchy–Schwarz the inner product is at most \(\|\nabla f(\mathbf w)\| \, \|\mathbf w - \mathbf u\|\), and by the triangle inequality \(\|\mathbf w - \mathbf u\| \le \|\mathbf w\| + \|\mathbf u\| \le \|\mathbf w\| + B\). \(\square\)

The claim holds for every \(\mathbf u\) in the ball, so it holds for the best one: \(f(\mathbf w) - \min_{\|\mathbf u\| \le B} f(\mathbf u)\) is at most the same quantity. So a small gradient is not just a hint that you are near the bottom; it is a *proof*. Look at which step of the proof carries the weight: the very first, the tangent inequality, and that needs convexity. Without convexity the formula still produces a number. It just no longer means anything.

Now the walk itself. The smoothness section proved that a step of size \(1/\beta\) against the gradient lowers a \(\beta\)-smooth function by at least \(\|\nabla f\|^2/(2\beta)\). So we need \(\beta\) for the logistic risk. Each example’s loss is \(\|\mathbf x_i\|^2/4\)-smooth, by the claim on smoothness through a linear function, where \(\mathbf x_i\) is the \(i\)-th message. The risk averages ten such losses. How smooth is an average? Message H has \(\|\mathbf x\|^2 = 1\), so its loss is \(1/4\)-smooth, and message C has 21, so its loss is \(21/4\)-smooth. You would guess that the average of the two losses is \((1/4 + 21/4)/2 = 2.75\)-smooth: the average of the constants. The guess is right.

**Claim (smoothness of an average).** If \(f_i\) is \(\beta_i\)-smooth for \(i = 1, \dots, m\), then \(f = \frac1m \sum_i f_i\) is \(\bar\beta\)-smooth with \(\bar\beta = \frac1m \sum_i \beta_i\).

Here \(\bar\beta\) (“beta bar”) is the plain average of the \(m\) constants \(\beta_1, \dots, \beta_m\), just as \(f\) is the plain average of the \(m\) functions. In plain words: **the average of the losses is as smooth as the average of their constants.**

*Proof.* The gradient of an average is the average of the gradients, \(\nabla f = \frac1m \sum_i \nabla f_i\). So by the triangle inequality, \(\|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \frac1m \sum_i \|\nabla f_i(\mathbf v) - \nabla f_i(\mathbf w)\| \le \frac1m \sum_i \beta_i \|\mathbf v - \mathbf w\|\), using each \(f_i\)’s smoothness in the last step. \(\square\)

For our ten messages the squared lengths are listed by the snippet, and their average divided by 4 is \(\beta\):

```python
X = [(1, 3, 0), (1, 2, 1), (1, 4, 2), (1, 1, 0), (1, 1, 2),
     (1, 0, 1), (1, 2, 3), (1, 0, 0), (1, 2, 1), (1, 1, 2)]
sq = [sum(c * c for c in x) for x in X]
beta = sum(sq) / len(X) / 4
print(sq)
print(sum(sq) / len(X), beta)
```

It prints `[10, 6, 21, 2, 6, 2, 14, 1, 6, 6]` and then `7.4 1.85`. So the logistic risk on the ten messages is 1.85-smooth, and the step is \(1/1.85\).

Put the two facts together and you have a walk that cannot fool itself. Start at act 1’s point. Step against the gradient with step \(1/\beta\), check at every step that the risk fell by at least the guaranteed \(\|\nabla f\|^2/(2\beta)\), and stop the moment the certificate drops to 0.01 or below.

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
norm = lambda u: math.sqrt(dot(u, u))
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in zip(X, Y)) / 10
grad = lambda w: [sum(-y * x[j] / (1 + math.exp(y * dot(w, x)))
                      for x, y in zip(X, Y)) / 10
                  for j in range(3)]
risk01 = lambda w: sum(y * dot(w, x) <= 0
                       for x, y in zip(X, Y)) / 10
beta, B, w, ok = 1.85, 3, [-1.0, -2.0, 2.0], True
for t in range(1, 301):
    g = grad(w)
    v = [a - b / beta for a, b in zip(w, g)]
    ok = ok and f(w) - f(v) >= dot(g, g) / (2 * beta) - 1e-12
    w = v
    cert = norm(grad(w)) * (norm(w) + B)
    if t in (1, 10, 50) or cert <= 0.01:
        print(t, round(f(w), 4), round(cert, 4), risk01(w))
    if cert <= 0.01:
        break
print(ok, [round(c, 2) for c in w])
```

It prints

```
1 1.9381 4.1654 0.7
10 0.7174 1.1867 0.5
50 0.4836 0.0651 0.2
87 0.4827 0.01 0.2
True [-0.9, 1.36, -0.93]
```

The columns are the step, the logistic risk, the certificate and the 0-1 risk. The certificate shrinks from 4.17 to 0.01 in 87 steps, and every step delivered its guaranteed decrease (`True`). The walk stops at weights close to lesson 9’s logistic weights \((-0.92, 1.38, -0.95)\), with a proof that no weight vector in the ball of radius 3 has a logistic risk more than 0.01 lower. As a bonus, the 0-1 risk is 0.2, the floor that the two contradictory pairs force.

**Cost.** Each step computes one gradient, a sum over the \(m\) examples of a \(d\)-dimensional vector: \(O(md)\) time, and \(O(d)\) space for \(\mathbf w\) and the gradient beyond the training sequence. For the real filter, \(m = 300\) and \(d = 31\), that is about 9,300 multiply-adds per step. Compare a search through \(1.70 \times 10^{42}\) labellings. How many steps a given tolerance needs in general is lesson 14’s business; this lesson guarantees that every step helps and that the answer, when it comes, is certified.

## Fix the bug

Back to the mail room. The brute-force job is cancelled for good. In its place, operations will accept a training job that finishes within 300 steps and returns a weight vector with a certificate of 0.01 or less: a proof, by the claim above with \(B = 3\), that no weight vector in the ball does more than 0.01 better on the job’s own training objective.

The lab gives you the ten messages and the walker from act 1. You choose the starting point (click on the slice), the loss the walker minimizes, and its step. The loss menu has four entries: the 0-1 loss; the hinge loss; lesson 9’s logistic loss; and the **sigmoid loss** \(1/(1 + e^{z})\) of the signed score \(z\), a smooth S-shaped curve that falls from 1 to 0 and looks like a softened version of the 0-1 step. The step menu has two: \(1/\beta\), and \(3/\beta\), which looks three times as fast. The lab shows the risk, the gradient length and the certificate at every step, and it awards the fix only when the certificate is a proof, not merely a small number. Before you start, a prediction to write down: you pick the 0-1 loss, and at the start the walker’s gradient is zero, so the certificate formula reads 0. What has it proved?

[[lab-act2]]

Only one setting earns the fix: the logistic loss with step \(1/\beta\). From act 1’s point it needs 87 steps, and from starting points spread over the whole slice at most 173, inside the budget of 300. Every other setting fails for a reason this lesson named. The 0-1 loss reads 0 at once and proves nothing, because its risk is not convex and the certificate’s first step, the tangent inequality, does not hold. The hinge loss is convex but has kinks, so there is no \(\beta\), and near its bottom the slope never gets short: in 300 steps of the act 1 walker the certificate never drops below 0.4271. The step \(3/\beta\) on the logistic loss reached a certificate of 0.01 in only 27 steps, but nothing guaranteed it: the quadratic upper bound promises a decrease for a step of \(1/\beta\), and for \(3/\beta\) it allows an increase. And the sigmoid loss is the nasty one. With step \(3/\beta\) from act 1’s point, it walks off to weights of length 5.38 and reports a certificate of 0.01 after 181 steps, at a point where half the messages are wrong. The number is small, and it is a lie; the concept check asks you to catch it.

## What people get wrong

**“A zero gradient means you have reached the bottom.”** Only on a convex function. On the 0-1 terraces the gradient is zero everywhere it exists, at 0.8 and at 0.2 alike.

**“Smooth means convex.”** Smoothness bounds how fast the slope changes; it says nothing about which way the curve bends. The sigmoid loss \(1/(1 + e^{z})\) is smooth and fails the chord test: at \(z = -4\) and \(z = 0\) its values are 0.982 and 0.5, so the chord’s midpoint is 0.741, but the curve at \(z = -2\) is 0.8808, above it.

**“A bigger step gets there faster, so it is better.”** On the logistic risk, \(3/\beta\) happened to work. The guarantee is for \(1/\beta\), and the quadratic upper bound allows a step of \(3/\beta\) to go up instead of down. Problem 1 lets you watch it happen on the simplest smooth function there is.

**“A random chord test that finds no violation proves convexity.”** A test can only find violations. Convexity is proved by an argument, usually by building the function from convex pieces with the composition, maximum and sum rules.

**“The hinge loss is convex, so the gradient certificate works for it.”** The certificate needs a gradient, and the hinge loss has none at its kink, where the slope jumps. Near its bottom the walker’s slope keeps jumping between pieces instead of shrinking. Lesson 14 handles functions with kinks.

## Recap and the next bug

1. A set is **convex** when it contains every segment between its points; a function is **convex** when its graph lies below every chord, equivalently when its epigraph is convex. The 0-1 risk of halfspaces is not convex; its landscape is terraces.
2. For convex functions, every **local minimum is global**, and the **tangent inequality** \(f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\) puts a floor under the whole graph.
3. Convexity survives composition with a linear function, maxima, and nonnegative sums. So the square, logistic and hinge losses are convex in \(\mathbf w\), and so are their empirical risks.
4. A **\(\rho\)-Lipschitz** function changes by at most \(\rho\) times the change in its input; a **\(\beta\)-smooth** function has a \(\beta\)-Lipschitz gradient, which gives a quadratic ceiling, a guaranteed decrease of \(\|\nabla f\|^2/(2\beta)\) per step of size \(1/\beta\), and, for nonnegative functions, **self-boundedness**. Through a linear function with message \(\mathbf x\), the constants become \(\rho\|\mathbf x\|\) and \(\beta\|\mathbf x\|^2\); an average of smooth functions is as smooth as the average of their constants.
5. In a **convex learning problem**, ERM is a convex optimization problem.
6. For a convex function, the **certificate** \(\|\nabla f(\mathbf w)\|(\|\mathbf w\| + B)\) bounds how far \(f(\mathbf w)\) can be above the best value in the ball of radius \(B\). With step \(1/\beta\), every step helps and the stopping point is proved near-optimal; without convexity, the same number proves nothing.

The job that would have needed \(5 \times 10^{29}\) years now finishes in 87 steps, with a certificate. You send the report to operations, proud of it. The reply comes back within the hour, and it has two questions. “Certified on which mail? Your proof is about the ten messages you trained on. We need a promise about next month’s mail, from the 3,000 labels we can pay for.” And: “We count mistakes, not logistic loss.” A colleague reading over your shoulder shrugs: the loss is convex, so of course it learns; just add labels. Is that true? Does minimizing a convex loss on a finite training sequence actually learn, and how does a bound on a convex loss become a promise about mistakes? That is lesson 12b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof that every local minimum of a convex function is global, point to the exact inequality where convexity is used. Then take the 0-1 risk of the ten messages at \(\mathbf u = -\mathbf w^\circ = (1, -2, 2)\), with value 0.8, and \(\mathbf v = \mathbf w^\circ\), with value 0.2. For a small \(\alpha > 0\), what would convexity force the risk at \(\mathbf u + \alpha(\mathbf v - \mathbf u)\) to be at most, and what is it really?
2. In the lab, the sigmoid loss with step \(3/\beta\) reported a certificate of 0.01 at sigmoid risk 0.5011, with weights of length 5.38. Say which step of the certificate’s proof fails, and why. Then find a weight vector in the ball of radius 3 whose sigmoid risk on the ten messages beats the reported point by more than 0.01, and compute that risk. Does it matter that the reported weights lie outside the ball?
3. On the logistic risk of the ten messages (\(\beta = 1.85\)), the walker stands at a point with gradient length 0.02 and weight length 1.8. With \(B = 3\), at most how far above the best risk in the ball can it be? And at least how much will its next step of size \(1/\beta\) lower the risk?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (how big a step is safe?).** This problem was written for this lesson. (a) Let \(f\) be \(\beta\)-smooth and \(0 < \eta \le 1/\beta\). Use the quadratic upper bound to show that the step \(\mathbf v = \mathbf w - \eta \nabla f(\mathbf w)\) satisfies \(f(\mathbf v) \le f(\mathbf w) - \frac\eta2 \|\nabla f(\mathbf w)\|^2\). (b) Take \(f(x) = x^2\), which is 2-smooth, and run the step \(x \leftarrow x - \eta f'(x)\) from \(x = 1\) with \(\eta = 1/\beta\), \(\eta = 2/\beta\) and \(\eta = 3/\beta\). Describe what happens in each case, and explain why the lab of this lesson awards the fix to step \(1/\beta\) and not to \(3/\beta\), even on a run where \(3/\beta\) reached a small certificate first.
