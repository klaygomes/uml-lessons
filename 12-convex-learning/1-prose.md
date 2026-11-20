# Convex Learning Problems

## The search that never comes back

Lesson 11 ended with a learning curve that would not come down. Your colleague’s halfspace filter, the one from lesson 9 that weighs word counts, kept a stubborn empirical risk even though the validation numbers said the class was good enough. The diagnosis was uncomfortable: the learner was not really doing ERM. The mail is not separable, so no halfspace gets every example right, and finding the halfspace with the *fewest* training mistakes is computationally hard. The code had been running a heuristic and hoping.

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

You remember what lesson 9 did with the same ten messages, though. It did not count mistakes at all. It minimized the logistic loss, and the minimization went smoothly, separable data or not, with a guarantee that the objective was “convex”. That word was used once and promised a whole lesson. This is the lesson.

The idea is simple to say. Counting mistakes is hard because the 0-1 loss is a landscape of flat terraces and cliffs. Replace it with a loss shaped like a bowl, one that is never smaller than the 0-1 loss, and minimize that instead. Bowls are easy: walk downhill and you reach the bottom. And because the new loss sits above the old one, a guarantee on the new loss is a guarantee on the mistakes too. By the end of the lesson you will know exactly what “shaped like a bowl” means, which bowls can be learned from a finite label budget (not all of them, which is the surprise), and how much the switch costs you.

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

It prints `1 -1 1`: the inner product, the signed score, the loss. The empirical risk \(L^{0-1}_S(\mathbf w)\) is, as in every lesson since lesson 2, the average of this loss over the training sequence \(S\), the fraction of mistakes; the function `risk01` in the next snippet is exactly that average. No weight vector can do better than 0.2 here: B and B′ are the same message with opposite labels, so every halfspace gets one of them wrong, and the same goes for E and E′. Lesson 9’s halfspace \(\mathbf w^\circ = (-1, 2, -2)\) reaches that floor, with exactly those two mistakes.

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

So what would a good landscape look like? The midpoint test above is the right instinct. Let’s make it a definition.

## Convex sets and convex functions

Start with the space the weights live in. A set is convex when you can walk between any two of its points in a straight line without leaving it.

**Definition (convex set).** A set \(C\) in a vector space is **convex** if for any two vectors \(\mathbf u, \mathbf v \in C\), the line segment between them is contained in \(C\): for every \(\alpha \in [0, 1]\),

\[ \alpha \mathbf u + (1 - \alpha) \mathbf v \in C. \]

- \(\mathbf u, \mathbf v \in C\) reads “\(\mathbf u\) and \(\mathbf v\) are elements of \(C\)”; \(\alpha \in [0, 1]\) says \(\alpha\) is a number between 0 and 1, ends included.
- \(\alpha \mathbf u\) scales every entry of \(\mathbf u\) by \(\alpha\); \((1 - \alpha)\mathbf v\) scales \(\mathbf v\) by the leftover weight; adding them mixes the two vectors, with weights that sum to 1.

The point \(\alpha \mathbf u + (1 - \alpha) \mathbf v\) is called a **convex combination** of \(\mathbf u\) and \(\mathbf v\). As \(\alpha\) runs from 1 to 0 it slides from \(\mathbf u\) to \(\mathbf v\). In plain words: **every point on the segment between two members is also a member.** The walk of worked example 1 was exactly this, with \(\mathbf u = \mathbf w^\circ\) and \(\mathbf v = -\mathbf w^\circ\): at \(\alpha = 0.25\) it stood at \(0.25(-1, 2, -2) + 0.75(1, -2, 2) = (0.5, -1, 1)\), and the line `w = tuple(a * p + (1 - a) * q ...)` of that snippet computes it.

A disc is convex, and so is a ball \(\{\mathbf w : \|\mathbf w\| \le B\}\) in any dimension: if \(\|\mathbf u\|, \|\mathbf v\| \le B\), the triangle inequality gives \(\|\alpha \mathbf u + (1 - \alpha)\mathbf v\| \le \alpha B + (1 - \alpha) B = B\). The whole space \(\mathbb R^d\) is convex. A crescent moon or a ring is not: pick two points on opposite sides of the hole, and the segment between them leaves the set.

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

It prints `x^2 0.0`, `softplus 0.0`, `abs 0.0` and `step 0.99`. The first three never violate the chord; the step violates it by almost the full height of the step, whenever \(u\) and \(v\) are on the same side of 0 at a distance from it and the combination lands on the other side. (A random test can only find violations; proving convexity takes an argument, which is what the next few sections supply.)

## Every local minimum is global

Here is the property that makes convexity worth all this. Remember the plateau around \(-\mathbf w^\circ\): every point within distance 0.218 had risk 0.8, nothing nearby was lower, and yet 0.2 existed far away. Let \(B(\mathbf u, r) = \{\mathbf v : \|\mathbf v - \mathbf u\| \le r\}\) be the ball of radius \(r\) around \(\mathbf u\): the set of points \(\mathbf v\) such that the distance \(\|\mathbf v - \mathbf u\|\) from \(\mathbf u\) is at most \(r\). We say \(f(\mathbf u)\) is a **local minimum** of \(f\) at \(\mathbf u\) if there is some \(r > 0\) such that \(f(\mathbf v) \ge f(\mathbf u)\) for every \(\mathbf v \in B(\mathbf u, r)\): nothing nearby is lower. A **global minimum** is lower than or equal to every point, near or far. The point \(-\mathbf w^\circ\) of the worked example is a local minimum of the 0-1 risk, with value 0.8, and it is not global. Sample 10,000 points of the ball of radius 0.2 around it and collect the risks you see:

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

Read that as a promise to a search algorithm. On a convex landscape, the moment you cannot improve locally, you are done globally. There are no false bottoms to get trapped in.

## The tangent below the graph

A second property gives a searcher a sense of direction, and a way to know how far from the bottom it is. Take lesson 9’s logistic risk on the ten messages, the average of \(\log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\), and stand at \(\mathbf w = \mathbf 0\), where it equals \(\log 2 \approx 0.6931\). Nudge one weight at a time by a tiny amount and watch how fast the risk changes: nudging the bias weight changes nothing, nudging the “free” weight up lowers the risk at rate 0.3, nudging the “meeting” weight up raises it at rate 0.1. Those three rates, collected into a vector, are the gradient. When \(f\) is differentiable, its **gradient** at \(\mathbf w\) is the vector of partial derivatives,

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
```

It prints `0.6931 [0.0, -0.3, 0.1]`. The tangent plane at \(\mathbf w\) is the linear function \(\mathbf u \mapsto f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\) (the arrow \(\mapsto\) reads “sends \(\mathbf u\) to”): start at the height \(f(\mathbf w)\) and add, for each weight, its rate times how far \(\mathbf u\) moved in it. For a convex function, that tangent lies below the graph everywhere, not just near \(\mathbf w\).

**Claim (tangent inequality).** If \(f : \mathbb R^d \to \mathbb R\) is convex and differentiable, then for all \(\mathbf w, \mathbf u\),

\[ f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle. \]

The left side is the true height at \(\mathbf u\); the right side is the tangent plane at \(\mathbf w\) evaluated at \(\mathbf u\), built from the pieces just walked through. In plain words: **the tangent plane at any point is a floor under the whole graph.** From \(\mathbf w = \mathbf 0\), predict the logistic risk at \(\mathbf u = \mathbf w^\circ = (-1, 2, -2)\): the tangent says \(0.6931 + (0 \cdot (-1) - 0.3 \cdot 2 + 0.1 \cdot (-2)) = -0.1069\), and the real risk is above it (the logistic risk is convex, as a later section proves):

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
f = lambda w: sum(math.log(1 + math.exp(-y * dot(w, x)))
                  for x, y in zip(X, Y)) / 10
w, u, g = (0, 0, 0), (-1, 2, -2), (0, -0.3, 0.1)
diff = [a - b for a, b in zip(u, w)]
tangent = f(w) + dot(g, diff)
print(round(f(u), 4), round(tangent, 4), f(u) >= tangent)
```

It prints `0.5503 -0.1069 True`.

*Proof.* For \(\alpha \in (0, 1]\), convexity with the points \(\mathbf u\) and \(\mathbf w\) gives \(f(\mathbf w + \alpha(\mathbf u - \mathbf w)) \le (1 - \alpha) f(\mathbf w) + \alpha f(\mathbf u)\). Subtract \(f(\mathbf w)\) and divide by \(\alpha\):

\[ \frac{f(\mathbf w + \alpha(\mathbf u - \mathbf w)) - f(\mathbf w)}{\alpha} \le f(\mathbf u) - f(\mathbf w). \]

The left side is a rise over a run: how much \(f\) changes when you step a fraction \(\alpha\) of the way towards \(\mathbf u\), divided by \(\alpha\). As \(\alpha \to 0\) (read “as \(\alpha\) shrinks towards 0”), the left side tends to the derivative of \(f\) at \(\mathbf w\) in the direction \(\mathbf u - \mathbf w\), which for a differentiable function is \(\langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\). The right side does not depend on \(\alpha\), so the inequality survives the limit. Rearranging gives the claim. \(\square\)

[[diagram:tangent]]

Two consequences are worth reading off right away. If \(\nabla f(\mathbf w) = \mathbf 0\), the tangent inequality says \(f(\mathbf u) \ge f(\mathbf w)\) for every \(\mathbf u\): a flat spot of a convex function is a global minimum. And more usefully, if the gradient is merely *small*, you are close to the bottom. We will turn that into a stopping rule with a certificate when we write the algorithm. Lesson 14 extends the tangent inequality to functions with kinks, such as \(|x|\), that have no gradient at some points.

For functions of one variable there is an easy test.

**Lemma (convexity in one variable).** Let \(f : \mathbb R \to \mathbb R\) be twice differentiable, with first and second derivatives \(f'\) and \(f''\). The following are equivalent:

1. \(f\) is convex;
2. \(f'\) is monotonically nondecreasing;
3. \(f''\) is nonnegative.

*Proof.* (1 ⇒ 2.) Take \(a < c\). The tangent inequality at \(a\), with \(u = c\), gives \(f(c) \ge f(a) + f'(a)(c - a)\); at \(c\), with \(u = a\), it gives \(f(a) \ge f(c) + f'(c)(a - c)\). Add the two and cancel \(f(a) + f(c)\): \(0 \ge f'(a)(c - a) - f'(c)(c - a)\), that is, \((f'(c) - f'(a))(c - a) \ge 0\). Since \(c - a > 0\), \(f'(c) \ge f'(a)\).

(2 ⇒ 1.) Take \(u < v\), \(\alpha \in (0, 1)\) and \(x = \alpha u + (1 - \alpha) v\), so \(u < x < v\). By the mean value theorem there are points \(\xi_1 \in (u, x)\) and \(\xi_2 \in (x, v)\) with \(f(x) - f(u) = f'(\xi_1)(x - u)\) and \(f(v) - f(x) = f'(\xi_2)(v - x)\). Since \(\xi_1 < \xi_2\) and \(f'\) is nondecreasing, \(f'(\xi_1) \le f'(\xi_2)\). Now \(x - u = (1 - \alpha)(v - u)\) and \(v - x = \alpha(v - u)\), so

\[ \alpha\,(f(x) - f(u)) = \alpha (1 - \alpha)(v - u) f'(\xi_1) \le \alpha(1 - \alpha)(v - u) f'(\xi_2) = (1 - \alpha)(f(v) - f(x)). \]

The outer two expressions are the two mean-value equations multiplied by \(\alpha\) and \(1 - \alpha\), and the middle step only swaps the smaller slope for the larger. Expand: \(\alpha f(x) - \alpha f(u) \le (1 - \alpha) f(v) - (1 - \alpha) f(x)\), so \(f(x) \le \alpha f(u) + (1 - \alpha) f(v)\). The cases \(u > v\) and \(\alpha \in \{0, 1\}\) follow by symmetry or are trivial.

(2 ⇔ 3.) A differentiable function, here \(f'\), is nondecreasing exactly when its derivative is nonnegative: if \(f'' \ge 0\), the mean value theorem gives \(f'(c) - f'(a) = f''(\xi)(c - a) \ge 0\) for \(a < c\); and if \(f'\) is nondecreasing, every difference quotient \((f'(a + t) - f'(a))/t\) with \(t > 0\) is nonnegative, so their limit \(f''(a)\) is too. \(\square\)

Two examples. The function \(f(x) = x^2\) has \(f''(x) = 2 > 0\), so it is convex. The **softplus** function \(f(x) = \log(1 + \exp(x))\) has

\[ f'(x) = \frac{\exp(x)}{1 + \exp(x)} = \frac{1}{\exp(-x) + 1}, \]

which is increasing because \(\exp(-x)\) decreases; so softplus is convex. Here \(\exp(x) = e^x\) and \(\log\) is the natural logarithm, its inverse. The first fraction is the chain rule (the derivative of \(\log(1 + t)\) is \(1/(1 + t)\), times the derivative \(\exp(x)\) of the inside); the second divides top and bottom by \(\exp(x)\). In plain words: **the slope of softplus climbs from 0 towards 1 as \(x\) grows.** At \(x = 0\) it is \(1/(1 + 1) = 1/2\), and the line `df = lambda x: 1 / (1 + math.exp(-x))` in the next snippet is the second form, term for term. Its second derivative is \(f'(x) f'(-x)\), the logistic function of lesson 9 times its mirror image. Let’s check all three statements on a grid, and the tangent inequality too.

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

It prints `[0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0.5]`. It is the maximum of two functions of \(\mathbf w\), the constant 0 and \(1 - y\langle \mathbf w, \mathbf x \rangle\), and both are linear plus a constant, hence convex. By the claim on maxima, the hinge loss is convex in \(\mathbf w\), and by the claim on sums, so is its empirical risk. Compare it with the 0-1 risk on the ten messages: draw 20,000 random pairs of weight vectors and a random \(\alpha\), and count how often each risk breaks the chord inequality.

```python
import random

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1),
     ((2, 1), -1), ((1, 2), 1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
hinge = lambda w: sum(max(0, 1 - y * dot(w, x))
                      for x, y in zip(X, Y)) / 10
zero_one = lambda w: sum(y * dot(w, x) <= 0
                         for x, y in zip(X, Y)) / 10
rng = random.Random(1)
bad = {hinge: 0, zero_one: 0}
for _ in range(20000):
    u, v = ([rng.uniform(-3, 3) for _ in X[0]] for _ in "uv")
    a = rng.random()
    w = [a * p + (1 - a) * q for p, q in zip(u, v)]
    for f in bad:
        bad[f] += f(w) > a * f(u) + (1 - a) * f(v) + 1e-12
print(bad[hinge], bad[zero_one])
```

It prints `0 7965`. The hinge risk never breaks the chord; the 0-1 risk breaks it in about 40% of random trials. Time to look at the two landscapes yourself.

[[lab]]

In the lab you saw the 0-1 risk as terraces and the hinge risk as a bowl, on a slice of weight space where the bias term is fixed at \(-1\). A walker that follows the slope stalls at once on the 0-1 terraces, because the slope is zero everywhere it is defined. On the hinge bowl the same walker slides to the bottom, at hinge risk 0.50, and the halfspace it stops at makes only the two forced mistakes. That is the whole strategy in one picture. The rest of the lesson asks whether it is *sound*: does minimizing a convex loss on a finite training sequence actually learn?

## Lipschitzness: losses that cannot change too fast

Convexity says the landscape has one valley. It says nothing about how steep the walls are, and steepness turns out to matter for learning. We measure it with the Euclidean norm.

**Definition (Lipschitzness).** Let \(C \subseteq \mathbb R^d\). A function \(f : \mathbb R^d \to \mathbb R^k\) is **\(\rho\)-Lipschitz** over \(C\) if for every \(\mathbf w_1, \mathbf w_2 \in C\),

\[ \| f(\mathbf w_1) - f(\mathbf w_2) \| \le \rho \, \| \mathbf w_1 - \mathbf w_2 \|. \]

\(C \subseteq \mathbb R^d\) says \(C\) is a subset of \(\mathbb R^d\), and \(f : \mathbb R^d \to \mathbb R^k\) allows the output to be a vector of \(k\) numbers, which is why the left side uses the length \(\|\cdot\|\) too (for one number it is just the absolute value). The left side is how far apart the outputs are; the right side is \(\rho\) times how far apart the inputs are. In plain words: **the output never moves more than \(\rho\) times as far as the input.** For \(f(x) = |x|\) with \(\rho = 1\), take \(x_1 = 3\) and \(x_2 = -1\): the outputs differ by \(|3 - 1| = 2\), the inputs by 4, and \(2 \le 1 \cdot 4\). Intuitively, a Lipschitz function cannot change too fast: moving the input by a distance \(t\) moves the output by at most \(\rho t\). For a differentiable \(f : \mathbb R \to \mathbb R\) there is an easy sufficient condition. The mean value theorem gives \(f(w_1) - f(w_2) = f'(u)(w_1 - w_2)\) for some \(u\) between \(w_1\) and \(w_2\), so if \(|f'| \le \rho\) everywhere, \(f\) is \(\rho\)-Lipschitz.

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

Lipschitzness bounds how fast the loss changes. Smoothness bounds how fast its *slope* changes.

**Definition (smoothness).** A differentiable function \(f : \mathbb R^d \to \mathbb R\) is **\(\beta\)-smooth** if its gradient is \(\beta\)-Lipschitz: for all \(\mathbf v, \mathbf w\),

\[ \|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \beta \|\mathbf v - \mathbf w\|. \]

It is the Lipschitz definition with the gradient \(\nabla f\) in place of \(f\) and \(\beta\) in place of \(\rho\). In plain words: **the slope never changes more than \(\beta\) times as fast as the point moves.** Softplus has slope \(1/2\) at 0 and \(0.7311\) at 1, a change of \(0.2311\) over a distance of 1, within \(\beta = 1/4\); the first line of the softplus snippet below takes the largest such ratio over a grid.

Why care? Because smoothness puts a ceiling over the function, just as convexity put a floor under it.

**Claim (quadratic upper bound).** If \(f\) is \(\beta\)-smooth, then for all \(\mathbf v, \mathbf w\),

\[ f(\mathbf v) \le f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle + \frac{\beta}{2} \|\mathbf v - \mathbf w\|^2. \]

The first two terms on the right are the tangent plane at \(\mathbf w\); the third adds a bowl, \(\beta/2\) times the squared distance from \(\mathbf w\). In plain words: **the tangent plus a bowl of curvature \(\beta\) is a ceiling over the graph.** For softplus at \(w = 0\), \(v = 2\): the tangent gives \(0.6931 + 0.5 \cdot 2 = 1.6931\), the bowl adds \(\frac18 \cdot 4 = 0.5\), and the ceiling 2.1931 is above the true \(f(2) = 2.1269\), which is above the tangent floor. The `upper` line of the softplus snippet below is this inequality, term by term.

*Proof.* Walk from \(\mathbf w\) to \(\mathbf v\) along the segment and set \(h(t) = f(\mathbf w + t(\mathbf v - \mathbf w))\) for \(t \in [0, 1]\). By the chain rule, \(h'(t) = \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)), \mathbf v - \mathbf w \rangle\), which is continuous in \(t\) because the gradient is Lipschitz. The fundamental theorem of calculus gives

\[ f(\mathbf v) - f(\mathbf w) = h(1) - h(0) = \int_0^1 \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)), \mathbf v - \mathbf w \rangle \, dt. \]

The integral sign \(\int_0^1 \dots \, dt\) adds up the quantity inside over every \(t\) from 0 to 1, in infinitely thin slices of width \(dt\). In plain words: **the total change along the segment is the sum of all the small changes along the way.** For \(f(x) = x^2\) from \(w = 1\) to \(v = 3\), the slope along the way is \(h'(t) = 2(1 + 2t) \cdot 2\), and adding it up over 100,000 thin slices gives back \(9 - 1 = 8\):

```python
f = lambda x: x * x
df = lambda x: 2 * x
w, v, n = 1, 3, 100000
h1 = lambda t: df(w + t * (v - w)) * (v - w)
area = sum(h1((k + 0.5) / n) for k in range(n)) / n
print(f(v) - f(w), round(area, 6))
```

It prints `8 8.0`. Add and subtract \(\nabla f(\mathbf w)\) inside the inner product:

\[ f(\mathbf v) - f(\mathbf w) = \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle + \int_0^1 \langle \nabla f(\mathbf w + t(\mathbf v - \mathbf w)) - \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle \, dt. \]

By Cauchy–Schwarz and then smoothness, the integrand is at most \(\|\nabla f(\mathbf w + t(\mathbf v - \mathbf w)) - \nabla f(\mathbf w)\| \, \|\mathbf v - \mathbf w\| \le \beta t \|\mathbf v - \mathbf w\| \cdot \|\mathbf v - \mathbf w\|\), and \(\int_0^1 \beta t \, dt = \beta/2\). \(\square\)

Put this next to the tangent inequality. When \(f\) is convex and \(\beta\)-smooth, the difference between \(f\) and its first-order approximation is caught between two bounds:

\[ 0 \le f(\mathbf v) - f(\mathbf w) - \langle \nabla f(\mathbf w), \mathbf v - \mathbf w \rangle \le \frac{\beta}{2}\|\mathbf v - \mathbf w\|^2. \]

[[diagram:sandwich]]

The ceiling says something practical about stepping downhill. Choose \(\mathbf v = \mathbf w - \frac{1}{\beta}\nabla f(\mathbf w)\), a step against the gradient of length \(\|\nabla f(\mathbf w)\|/\beta\). Then \(\mathbf v - \mathbf w = -\frac1\beta \nabla f(\mathbf w)\), and the quadratic upper bound becomes

\[ f(\mathbf v) \le f(\mathbf w) - \frac1\beta \|\nabla f(\mathbf w)\|^2 + \frac{\beta}{2} \cdot \frac{1}{\beta^2} \|\nabla f(\mathbf w)\|^2 = f(\mathbf w) - \frac{1}{2\beta}\|\nabla f(\mathbf w)\|^2. \]

Rearranged: \(\frac{1}{2\beta}\|\nabla f(\mathbf w)\|^2 \le f(\mathbf w) - f(\mathbf v)\). One step of size \(1/\beta\) is *guaranteed* to lower \(f\) by at least \(\|\nabla f(\mathbf w)\|^2/(2\beta)\). If, in addition, \(f(\mathbf v) \ge 0\) for all \(\mathbf v\), the right side is at most \(f(\mathbf w)\), and we get

\[ \|\nabla f(\mathbf w)\|^2 \le 2\beta f(\mathbf w). \]

A function with this property is called **self-bounded**: where its value is small, its gradient must be small too. Lessons 13 and 14 lean on it.

Examples. \(f(x) = x^2\) has \(f'(x) = 2x\), which is 2-Lipschitz, so \(x^2\) is 2-smooth. For this particular function the quadratic upper bound and the self-bounded inequality both hold with equality: \(v^2 = w^2 + 2w(v - w) + (v - w)^2\) and \((2w)^2 = 2 \cdot 2 \cdot w^2\). Softplus has \(f'(x) = 1/(1 + \exp(-x))\) and

\[ |f''(x)| = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \frac{1}{(1 + \exp(-x))(1 + \exp(x))} \le \frac14, \]

because the product \((1 + e^{-x})(1 + e^{x}) = 2 + e^x + e^{-x} \ge 4\) (the sum \(e^x + e^{-x}\) is at least 2, by the inequality between arithmetic and geometric means). So \(f'\) is \(\frac14\)-Lipschitz, softplus is \(\frac14\)-smooth, and since it is nonnegative it is self-bounded too.

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

It prints `0.2499`, `0.0`, `0.0033` and `1.9297 1.5516`. Line by line: the derivative’s steepest slope on the grid is just under \(\frac14\); the quadratic ceiling is never below the function (the smallest gap is 0, reached when \(v = w\)); \(2\beta f - (f')^2\) stays positive, so softplus is self-bounded on the grid; and one step of size \(1/\beta = 4\) from \(w = 2\) lowers \(f\) by 1.9297, more than the guaranteed 1.5516.

As with convexity and Lipschitzness, smoothness passes through a linear function, with a factor.

**Claim (smoothness through a linear function).** Let \(f(\mathbf w) = g(\langle \mathbf w, \mathbf x \rangle + b)\), where \(g : \mathbb R \to \mathbb R\) is \(\beta\)-smooth, \(\mathbf x \in \mathbb R^d\) and \(b \in \mathbb R\). Then \(f\) is \((\beta \|\mathbf x\|^2)\)-smooth.

*Proof.* By the chain rule, \(\nabla f(\mathbf w) = g'(\langle \mathbf w, \mathbf x \rangle + b)\,\mathbf x\). So for any \(\mathbf v, \mathbf w\),

\[ \|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| = |g'(\langle \mathbf v, \mathbf x \rangle + b) - g'(\langle \mathbf w, \mathbf x \rangle + b)| \, \|\mathbf x\| \le \beta \, |\langle \mathbf v - \mathbf w, \mathbf x \rangle| \, \|\mathbf x\| \le \beta \|\mathbf x\|^2 \|\mathbf v - \mathbf w\|, \]

using the smoothness of \(g\) (its derivative is \(\beta\)-Lipschitz) and then Cauchy–Schwarz. The same two ingredients also give the quadratic upper bound for \(f\) directly: apply \(g\)’s scalar bound with the step \(\langle \mathbf v - \mathbf w, \mathbf x \rangle\), then bound \(\langle \mathbf v - \mathbf w, \mathbf x \rangle^2 \le \|\mathbf v - \mathbf w\|^2 \|\mathbf x\|^2\). \(\square\)

So, for \(\mathbf x \in \mathbb R^d\) and \(y \in \mathbb R\), the square loss \(f(\mathbf w) = (\langle \mathbf w, \mathbf x \rangle - y)^2\) is \((2\|\mathbf x\|^2)\)-smooth (here \(g(a) = a^2\) and \(b = -y\)), and for \(y \in \{\pm 1\}\) the logistic loss \(f(\mathbf w) = \log(1 + \exp(-y\langle \mathbf w, \mathbf x \rangle))\) is \((\|\mathbf x\|^2/4)\)-smooth (softplus composed with \(\langle \mathbf w, -y\mathbf x \rangle\), and \(\|-y\mathbf x\| = \|\mathbf x\|\)). The hinge loss is not smooth at all: at signed score 1 it has a kink, where its gradient jumps from \(-y\mathbf x\) to \(\mathbf 0\), and no gradient exists there.

## Convex learning problems

Now we can say what kind of learning problem all this is about. Lesson 3 defined a learning problem by three things: a hypothesis class \(\mathcal H\), a set of examples \(Z\), and a loss function \(\ell : \mathcal H \times Z \to \mathbb R_+\). Until now \(\mathcal H\) was a set of functions from messages to labels. But \(\mathcal H\) can be any set, and for linear learners it is natural to identify each hypothesis with its weight vector. So from here on, \(\mathcal H \subseteq \mathbb R^d\), a hypothesis is written \(\mathbf w\), and the loss is \(\ell(\mathbf w, z)\). For each example \(z\), write \(\ell(\cdot, z)\) for the function \(\mathbf w \mapsto \ell(\mathbf w, z)\).

**Definition (convex learning problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex** if the hypothesis class \(\mathcal H\) is a convex set and, for every \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex function.

Linear regression with the square loss is one. Its hypotheses \(\mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle\) are parametrized by \(\mathbf w \in \mathbb R^d\), so take \(\mathcal H = \mathbb R^d\), a convex set; the examples are \(Z = \mathbb R^d \times \mathbb R = \mathbb R^{d+1}\); and \(\ell(\mathbf w, (\mathbf x, y)) = (\langle \mathbf w, \mathbf x \rangle - y)^2\) is convex in \(\mathbf w\), as we just showed. Logistic regression is another, and so is “halfspaces with the hinge loss”.

**Lemma (ERM is a convex optimization problem).** If \(\ell\) is a convex loss and \(\mathcal H\) is a convex set, then the ERM problem over \(\mathcal H\), minimizing the empirical risk over \(\mathcal H\), is a convex optimization problem: minimizing a convex function over a convex set.

*Proof.* The ERM rule is \(\mathrm{ERM}_{\mathcal H}(S) = \operatorname{argmin}_{\mathbf w \in \mathcal H} L_S(\mathbf w)\). For a training sequence \(S = (z_1, \dots, z_m)\), \(L_S(\mathbf w) = \frac1m \sum_{i=1}^m \ell(\mathbf w, z_i)\) is a sum of convex functions with nonnegative weights \(1/m\), so it is convex by the claim on sums. The constraint \(\mathbf w \in \mathcal H\) is membership in a convex set. \(\square\)

Under mild conditions, problems of this shape can be solved efficiently by generic optimization algorithms, and lesson 14 presents a very simple one. Combined with the local-minimum claim, this is why a convex ERM problem is not a search through \(10^{42}\) labellings: any point you cannot improve locally is already the answer.

## Is convexity enough?

So convex problems can be *optimized*. Can they be *learned*? That is a different question: it asks whether the hypothesis ERM returns on \(m\) examples has small true risk, with a number of examples that depends only on \(\varepsilon\) and \(\delta\).

You have good reasons to hope so. Lessons 6 and 9 showed that halfspaces in \(d\) dimensions are learnable (perhaps not efficiently). Lesson 4’s discretization trick suggested that any class described by \(d\) numbers is learnable with a number of examples that grows with \(d\). So surely, for a fixed small \(d\), every convex learning problem over \(\mathbb R^d\) is learnable?

No. Not even for \(d = 1\). This does not contradict lesson 6, which is about binary classification with the 0-1 loss, while here losses are real-valued and can be huge. Nor does it contradict the discretization trick, which assumed a bounded loss and that a finite number of bits per parameter is enough. Here is the counterexample, and it is worth going through slowly, because every step shows a different way that “convex” fails to imply “learnable”.

**Claim (linear regression in one dimension is not learnable).** Let \(\mathcal H = \mathbb R\), examples \(z = (x, y) \in \mathbb R^2\), and the square loss \(\ell(w, (x, y)) = (wx - y)^2\) (homogeneous: no bias term). This problem is convex, and it is not agnostic PAC learnable by any deterministic algorithm.

A **deterministic** algorithm is one whose output is fully determined by its training sequence: the same \(S\) always gives the same \(\hat w\). (A slightly longer argument handles learners that flip coins too.)

*Proof.* Let \(A\) be any deterministic algorithm and suppose, for contradiction, that it is an agnostic PAC learner for this problem. Then there is a function \(m(\cdot, \cdot)\) such that for every distribution \(\mathcal D\) and every \(\varepsilon, \delta\), if \(A\) receives \(m \ge m(\varepsilon, \delta)\) examples drawn i.i.d. from \(\mathcal D\), then with probability at least \(1 - \delta\) it outputs \(\hat w = A(S)\) with

\[ L_{\mathcal D}(\hat w) - \min_w L_{\mathcal D}(w) \le \varepsilon. \]

Choose \(\varepsilon = 1/100\) and \(\delta = 1/2\), take any \(m \ge m(\varepsilon, \delta)\), and set

\[ \mu = \frac{\log(100/99)}{2m}. \]

*Step 1: two worlds.* Define two distributions on examples. The first, \(\mathcal D_1\), puts probability \(\mu\) on the example \(z_1 = (1, 0)\) and probability \(1 - \mu\) on \(z_2 = (\mu, -1)\). The second, \(\mathcal D_2\), puts all its probability on \(z_2\). Think of \(z_2\) as the everyday example both worlds share, and \(z_1\) as a rare example that only the first world contains.

*Step 2: both worlds usually show the same training sequence.* Let \(E\) be the event that all \(m\) training examples are \(z_2\). Under \(\mathcal D_2\), \(E\) has probability 1. Under \(\mathcal D_1\), the examples are independent and each is \(z_2\) with probability \(1 - \mu\), so \(\mathbb P[E] = (1 - \mu)^m\). We bound this using the inequality \(1 - x \ge e^{-2x}\), valid for \(x \in [0, 1/2]\). (Proof: the function \(q(x) = \log(1 - x) + 2x\) has \(q(0) = 0\) and \(q(1/2) = 1 - \log 2 > 0\), and it is concave because \(q''(x) = -1/(1 - x)^2 < 0\); a concave function lies above its chord, which here is a segment from 0 up to a positive value, so \(q \ge 0\) on \([0, 1/2]\).) Here \(\mu \le \log(100/99)/2 < 0.006\), so

\[ (1 - \mu)^m \ge e^{-2\mu m} = e^{-\log(100/99)} = \frac{99}{100}. \]

So under either distribution, with probability at least 0.99 the learner sees the sequence \(S_2 = (z_2, \dots, z_2)\). Because \(A\) is deterministic, on \(S_2\) it outputs one fixed number; call it \(\hat w\). The learner cannot tell which world it is in.

*Step 3: the adversary picks the world after seeing \(\hat w\).* There are two cases.

If \(\hat w < -1/(2\mu)\), pick \(\mathcal D_1\). The rare example punishes a large negative weight: \(L_{\mathcal D_1}(\hat w) \ge \mu (\hat w \cdot 1 - 0)^2 = \mu \hat w^2 > \mu \cdot \frac{1}{4\mu^2} = \frac{1}{4\mu}\). Meanwhile the best in the class is no worse than \(w = 0\): \(\min_w L_{\mathcal D_1}(w) \le L_{\mathcal D_1}(0) = \mu \cdot 0 + (1 - \mu)(0 - (-1))^2 = 1 - \mu\). So

\[ L_{\mathcal D_1}(\hat w) - \min_w L_{\mathcal D_1}(w) \ge \frac{1}{4\mu} - (1 - \mu) > \varepsilon, \]

since \(\mu < 0.006\) makes \(1/(4\mu) > 41\).

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

It prints `5.03e-05 0.995`, then `-0.9999 0.9999`, then `4974.2 0.25`, then `0.99`. With 100 examples, the rare example appears with probability \(5 \times 10^{-5}\), and 99.5% of training sequences never show it. In the first world the best weight is about \(-1\), with risk about 1. At the cut, the first world charges an excess of 4,974 and the second charges 0.25, just as the proof said. And the last line is sobering: whatever \(\hat w\) the learner picks, one of the two worlds charges it an excess of at least 0.99, a hundred times the \(\varepsilon\) we asked for. (The formula for the first world’s best weight comes from setting the derivative of the quadratic \(L_{\mathcal D_1}\) to zero.)

What went wrong? The hypothesis class is unbounded, so the learner can be asked to choose between \(w \approx -1\) and \(w \approx -1/\mu\), two answers that are astronomically far apart, on the evidence of a sequence that cannot tell them apart. A natural fix is to **bound** the class: fix a scalar \(B\) in advance and require \(\|\mathbf w\| \le B\) for every hypothesis.

**Boundedness and convexity together are still not enough.** Keep the square loss, but now let \(\mathcal H = \{w : |w| \le 1\} \subset \mathbb R\), which is convex and bounded. Run the same argument with the two examples \(z_1 = (1/\mu, 0)\) and \(z_2 = (1, -1)\), again with \(\mathcal D_1\) putting mass \(\mu\) on \(z_1\) and \(\mathcal D_2\) all its mass on \(z_2\). Step 2 is unchanged. In step 3: if the learner returns \(\hat w < -1/2\), pick \(\mathcal D_1\) and get

\[ L_{\mathcal D_1}(\hat w) - \min_w L_{\mathcal D_1}(w) \ge \mu (\hat w / \mu)^2 - L_{\mathcal D_1}(0) \ge \frac{1}{4\mu} - (1 - \mu) > \varepsilon; \]

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

Look at where the damage comes from now. The class is bounded, but the example \(z_1 = (1/\mu, 0)\) is enormous, and the loss on it, \((w/\mu)^2\), has slope \(2w/\mu^2\), as large as \(2/\mu^2\) on the class. A single rare example can swing the risk by a huge amount over a tiny change in \(w\). The cure is to forbid exactly that: require the loss to be Lipschitz, or smooth, with a constant fixed in advance.

## Convex-Lipschitz-bounded and convex-smooth-bounded problems

Here are the two kinds of convex learning problems this course will learn.

**Definition (convex-Lipschitz-bounded problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex-Lipschitz-bounded** with parameters \(\rho, B\) if

- the hypothesis class \(\mathcal H\) is a convex set and \(\|\mathbf w\| \le B\) for all \(\mathbf w \in \mathcal H\);
- for all \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex and \(\rho\)-Lipschitz function.

Example: let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \rho\}\), \(\mathcal Y = \mathbb R\), \(\mathcal H = \{\mathbf w \in \mathbb R^d : \|\mathbf w\| \le B\}\), and the absolute loss \(\ell(\mathbf w, (\mathbf x, y)) = |\langle \mathbf w, \mathbf x \rangle - y|\). The class is a ball, so it is convex and bounded by \(B\). The loss is \(|\cdot|\), convex and 1-Lipschitz, composed with the linear function \(\mathbf w \mapsto \langle \mathbf w, \mathbf x \rangle - y\), so it is convex and \(\|\mathbf x\|\)-Lipschitz, hence \(\rho\)-Lipschitz. This is regression with the absolute loss, messages in a ball of radius \(\rho\), and homogeneous linear hypotheses of norm at most \(B\).

Our own example matters more. **Halfspaces with the hinge loss over the ball \(\|\mathbf w\| \le B\), on messages with \(\|\mathbf x\| \le R\), form a convex-Lipschitz-bounded problem with parameters \(R, B\):** the ball is convex and bounded, and we showed the hinge loss is convex and \(\|\mathbf x\|\)-Lipschitz, hence \(R\)-Lipschitz.

**Definition (convex-smooth-bounded problem).** A learning problem \((\mathcal H, Z, \ell)\) is **convex-smooth-bounded** with parameters \(\beta, B\) if

- the hypothesis class \(\mathcal H\) is a convex set and \(\|\mathbf w\| \le B\) for all \(\mathbf w \in \mathcal H\);
- for all \(z \in Z\), the loss \(\ell(\cdot, z)\) is a convex, nonnegative, and \(\beta\)-smooth function.

The nonnegativity is there to make the loss self-bounded, \(\|\nabla \ell\|^2 \le 2\beta \ell\), which is what replaces the Lipschitz condition in the proofs to come.

Example: let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \sqrt{\beta/2}\}\), \(\mathcal Y = \mathbb R\), \(\mathcal H = \{\mathbf w : \|\mathbf w\| \le B\}\), and the square loss \(\ell(\mathbf w, (\mathbf x, y)) = (\langle \mathbf w, \mathbf x \rangle - y)^2\). The loss is convex and nonnegative, and \((2\|\mathbf x\|^2)\)-smooth by the claim on smoothness through a linear function, and \(2\|\mathbf x\|^2 \le 2 \cdot \beta/2 = \beta\). So this regression problem is convex-smooth-bounded with parameters \(\beta, B\).

Go back to the counterexamples and see which condition each one breaks. The first had no bound \(B\). The second had \(B = 1\), but the loss on \(z_1 = (1/\mu, 0)\) was neither \(\rho\)-Lipschitz nor \(\beta\)-smooth for any constant fixed in advance, since its slope reached \(2/\mu^2\) and its smoothness constant was \(2/\mu^2\), both growing without limit as \(\mu\) shrinks.

**The promise, proved in the next two lessons.** These two kinds of problems are learnable: convexity, boundedness, and Lipschitzness or smoothness of the loss together are sufficient. Lessons 13 and 14 prove it by giving learners that succeed. To have a number to work with, here is the result lesson 14 proves for the convex-Lipschitz-bounded case, stated now and used in the lab. There is a simple learner (stochastic gradient descent, which looks at each example once) whose output \(\bar{\mathbf w}\) on \(m\) i.i.d. examples satisfies

\[ \mathbb E_S \left[ L_{\mathcal D}(\bar{\mathbf w}) \right] \le \min_{\mathbf w \in \mathcal H} L_{\mathcal D}(\mathbf w) + \frac{\rho B}{\sqrt m}. \]

Equivalently, \(m \ge \rho^2 B^2 / \varepsilon^2\) examples bring the expected excess risk down to \(\varepsilon\). Note two things. The bound does not mention \(d\) at all; the number of words can be as large as you like. And the guarantee is about the expectation over training sequences, not “with probability at least \(1 - \delta\)”; lesson 13 has tools for the second form.

## Surrogate loss functions

Now back to the bug. The natural loss for the filter is the 0-1 loss, and it is not convex in \(\mathbf w\), as worked example 1 showed by a chord and as the lab showed by a terrace. Minimizing it on non-separable data is NP-hard. The popular way around the hardness is to **upper bound** the nonconvex loss by a convex one. A **convex surrogate loss** for a loss \(\ell\) is a loss that

1. is convex, and
2. upper bounds the original loss: it is at least \(\ell\) on every hypothesis and every example.

For halfspaces, the hinge loss is a convex surrogate for the 0-1 loss. We already know it is convex. The upper bound is a two-case check on the signed score \(z = y\langle \mathbf w, \mathbf x \rangle\). If \(z \le 0\), the 0-1 loss is 1 and the hinge loss is \(\max\{0, 1 - z\} \ge 1 - z \ge 1\). If \(z > 0\), the 0-1 loss is 0 and the hinge loss is a maximum with 0, so it is at least 0. Either way, \(\ell^{0-1}(\mathbf w, (\mathbf x, y)) \le \ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))\).

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

Read it message by message. The eight undisputed messages all have signed score at least 1, so the hinge loss charges them nothing, just like the 0-1 loss. B′ has score \(-1\): one mistake for the 0-1 loss, a charge of 2 for the hinge loss. E′ has score \(-3\), a confident mistake, and the hinge loss charges 4. So the hinge risk is 0.6 where the 0-1 risk is 0.2: always above, and more severe on confident mistakes. Notice too that the hinge loss can be positive on a *correct* prediction, when the signed score is between 0 and 1. It asks for a margin, not just the right side.

Does the logistic loss of lesson 9 also qualify? It is convex, but at signed score 0 it equals \(\log 2 \approx 0.693 < 1\), so it is not an upper bound on the 0-1 loss. Rescale it, though, by dividing by \(\log 2\), and it becomes \(\log_2(1 + \exp(-z))\), which equals 1 at \(z = 0\), is decreasing, and is positive everywhere, so it is at least the 0-1 loss for every \(z\). We will call this the **base-2 logistic loss**. It is convex, \((1/\log 2)\)-Lipschitz in the signed score, and \((1/(4\log 2))\)-smooth. The square loss in the form \((1 - z)^2\) is another convex upper bound (for \(z \le 0\) it is at least 1). You will meet all three in the lab.

**What the switch guarantees.** Learn the hinge problem with some learner \(A\). The generalization requirement for a hinge-loss learner has the form

\[ L^{\text{hinge}}_{\mathcal D}(A(S)) \le \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) + \varepsilon, \]

where \(L^{\text{hinge}}_{\mathcal D}(\mathbf w) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}[\ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))]\). Because the hinge loss is at least the 0-1 loss on every example, its expectation is at least the 0-1 expectation for every \(\mathbf w\) (expectation preserves pointwise inequalities). So the left side is at least \(L^{0-1}_{\mathcal D}(A(S))\), and

\[ L^{0-1}_{\mathcal D}(A(S)) \le \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) + \varepsilon. \]

That is the deal: a guarantee on mistakes, paid for in hinge loss. To see what it costs, add and subtract the best 0-1 risk in the class:

\[ L^{0-1}_{\mathcal D}(A(S)) \le \underbrace{\min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)}_{\text{approximation error}} + \underbrace{\left( \min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) - \min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w) \right)}_{\text{optimization error}} + \underbrace{\varepsilon}_{\text{estimation error}}. \]

So the 0-1 risk of the learned halfspace is bounded by three terms.

- **Approximation error**, \(\min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)\): how well the class can do at all on this distribution. You met it in lesson 5.
- **Estimation error**, \(\varepsilon\): the price of seeing only a training sequence and not \(\mathcal D\). Also lesson 5.
- **Optimization error**, \(\min_{\mathbf w \in \mathcal H} L^{\text{hinge}}_{\mathcal D}(\mathbf w) - \min_{\mathbf w \in \mathcal H} L^{0-1}_{\mathcal D}(\mathbf w)\): the gap between the best achievable surrogate risk and the best achievable 0-1 risk. It is the price of our inability to minimize the 0-1 loss directly. Its size depends on the distribution and on which surrogate you use.

[[diagram:decomposition]]

Let’s see the three terms in a concrete world, the one the lab uses. Messages arrive as the eight undisputed types of the worked example, A to H, each with probability \(1/8\). Each carries its usual label with probability 0.95, and the opposite label with probability 0.05 (your users misclick). Every message has \(\|\mathbf x\| \le R = \sqrt{21}\). No hypothesis can do better than 0.05 here, and \(\mathbf w^\circ\) achieves it, so the approximation error is 0.05. Take the class \(\|\mathbf w\| \le B\) with \(B = 1.5\). A numerical minimization of the hinge risk over that ball (the build-it challenge asks you to reproduce it) lands on \(\mathbf w = (-0.2, 1.1, -1.0)\). Let’s evaluate it, then plug in lesson 14’s rate with \(\rho = R\) and \(m = 3{,}000\) labels.

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

It prints `1.5 0.05 0.2238`, then `0.1255 0.349`, then `2546`. The vector sits on the edge of the ball, its 0-1 risk is the best possible 0.05, and its hinge risk is 0.2238. So the optimization error is \(0.2238 - 0.05 = 0.1738\). With 3,000 labels the estimation term is \(RB/\sqrt m = 0.1255\), and the certified bound on the expected 0-1 risk of the learned filter is 0.349. To certify 0.36 with \(B = 1.5\) you need at least 2,546 labels.

Is 0.349 a disappointing number, when the best halfspace makes only 5% mistakes? It is a *bound*, and a loose one, just like lesson 9’s mistake bound that promised 142 updates when the Perceptron used 3. The actual filter will usually do far better, and the lab shows you that. What the bound buys is a promise you can write down before you train, with no NP-hard search anywhere in sight.

## Surrogate ERM as code

This lesson introduces one algorithmic idea: replace the loss by a convex surrogate, and run ERM on the surrogate, which is now a convex optimization problem. Lesson 14 gives the general-purpose method for that. Here we build the simplest honest version, for a surrogate that is also smooth, using only what this lesson proved: the descent step that smoothness guarantees, and a stopping rule that convexity certifies.

We need two small facts first.

**Fact 1 (smoothness of an average).** If \(f_i\) is \(\beta_i\)-smooth for \(i = 1, \dots, m\), then \(f = \frac1m \sum_i f_i\) is \(\bar\beta\)-smooth with \(\bar\beta = \frac1m \sum_i \beta_i\). *Proof.* \(\nabla f = \frac1m \sum_i \nabla f_i\), so by the triangle inequality \(\|\nabla f(\mathbf v) - \nabla f(\mathbf w)\| \le \frac1m \sum_i \|\nabla f_i(\mathbf v) - \nabla f_i(\mathbf w)\| \le \frac1m \sum_i \beta_i \|\mathbf v - \mathbf w\|\). \(\square\)

For the base-2 logistic loss, the scalar function \(a \mapsto \log_2(1 + e^{-a})\) has second derivative \(\frac{1}{\log 2} \cdot \frac{1}{(1 + e^{-a})(1 + e^{a})} \le \frac{1}{4 \log 2}\), so by the claim on smoothness through a linear function each example’s loss is \(\|\mathbf x_i\|^2/(4\log 2)\)-smooth, and the empirical risk is \(\beta\)-smooth with \(\beta = \frac{1}{4\log 2} \cdot \frac1m \sum_i \|\mathbf x_i\|^2\).

**Fact 2 (a certificate of near-optimality).** If \(f\) is convex and differentiable, then for every \(\mathbf w\) and every \(\mathbf u\) with \(\|\mathbf u\| \le B\),

\[ f(\mathbf w) - f(\mathbf u) \le \|\nabla f(\mathbf w)\| \, (\|\mathbf w\| + B). \]

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

**Cost.** Each step of SmoothSurrogateERM computes one gradient, a sum over the \(m\) examples of a \(d\)-dimensional vector: \(O(md)\) time. It keeps \(\mathbf w\) and \(\mathbf g\): \(O(d)\) space beyond the training sequence. With \(T\) steps, \(O(Tmd)\) time. Each step lowers \(L_S\) by at least \(\|\mathbf g\|^2/(2\beta)\), by the smoothness argument above, and the stopping rule is correct by Fact 2. How many steps it takes to reach a given tolerance is lesson 14’s business; this lesson only guarantees that every step helps and that the answer, when it comes, is certified. Compare the brute-force 0-1 learner: up to \(\sum_{i \le d} \binom{m}{i}\) labellings at \(O(md)\) each, which is the \(5 \times 10^{29}\) years from the opening.

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

The columns are the step, the empirical risk, the gradient norm and the certificate. Starting from 1 (the base-2 loss of a zero score), the risk falls to 0.6963, and after 300 steps the certificate proves it is within 0.0004 of the best possible over the ball of radius 3. The smoothness constant is \(\beta = 2.669\), every single step achieved its guaranteed decrease (`True`), and the weights are lesson 9’s logistic weights, \((-0.92, 1.38, -0.95)\). That is no coincidence: dividing a loss by \(\log 2\) does not move its minimizer, and \(0.6963 \times \log 2 \approx 0.483\) is lesson 9’s logistic risk.

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
2. `smooth_descent` starts at \(\mathbf 0\) and uses step \(1/\beta\) with \(\beta\) from Fact 1. Make it also return the certificate of Fact 2 for \(B = 3\).
3. `labels_needed` returns the smallest \(m\) with `certified_bound(...) <= target`, or `None` when the floor is already above the target (no label budget can help then).
4. Harder: the hinge loss is not smooth, so `smooth_descent` cannot minimize it. Minimize the hinge risk of the lab world over the ball \(\|\mathbf w\| \le B\) anyway: at each step move against a slope vector (for a term with signed score below 1, use \(-y\mathbf x\); otherwise \(\mathbf 0\)), with step \(B/(R\sqrt t)\) at step \(t\), and pull \(\mathbf w\) back onto the ball by rescaling whenever \(\|\mathbf w\| > B\). Keep the best point seen. You should reproduce 0.2238 at \(B = 1.5\) and about 0.1625 for \(B \ge 2.7\). Lesson 14 explains why this works.

## Fix the bug

Back to the mail room. Operations will not ship a filter without a written guarantee, and the 0-1 search will never finish. So you will ship a hinge-loss filter instead, and certify it with the bound you just derived: the expected 0-1 risk is at most the best surrogate risk over your class plus \(\rho B/\sqrt m\).

The lab’s world is the one from the decomposition: eight message types, 5% of labels flipped, \(R = \sqrt{21}\). Your label budget is 3,000, and operations wants the certified expected 0-1 risk at most 0.36. You choose the surrogate (hinge, base-2 logistic, or the square surrogate \((1 - z)^2\)) and the norm bound \(B\). The lab computes the floor (the best surrogate risk over your ball) exactly, the Lipschitz constant of your surrogate over your ball, and the bound. Before you start, a prediction to write down: with the hinge loss, is a bigger \(B\) always better?

[[lab-act2]]

The trade you just navigated is the bias-complexity tradeoff of lesson 5, reappearing in a new costume. A small ball cannot reach low surrogate risk, so its floor is high. A large ball lowers the floor, but the estimation term \(\rho B/\sqrt m\) grows with \(B\). In this world, with the hinge loss and 3,000 labels, the window, on a grid of steps of 0.1, is \(B\) from 1.4 to 1.9; the best is \(B = 1.5\), with bound 0.349. And no choice of \(B\) rescues the other two surrogates: the base-2 logistic loss cannot get its floor below 0.3675 and the square surrogate below 0.3644, both already over the target before a single label is counted.

## What people get wrong

**“The loss is convex, so the problem is learnable.”** Convexity makes ERM an optimization problem without false bottoms. It says nothing about how many examples you need. One-dimensional linear regression with the square loss is convex and not learnable at all. You also need a bounded class and a Lipschitz or smooth loss.

**“Bounding the weights fixes it.”** Not by itself. The bounded counterexample kept \(|w| \le 1\) and still failed, because one rare, huge example made the loss arbitrarily steep. The constants \(\rho\) or \(\beta\) must be fixed in advance, which in practice means bounding the messages, \(\|\mathbf x\| \le R\), as well.

**“Minimizing the hinge loss minimizes the number of mistakes.”** It minimizes an upper bound on them. The two minimizers can differ, and the optimization error is exactly the price. In the lab world it was 0.17, more than three times the best possible 0-1 risk.

**“Any convex loss is a surrogate for the 0-1 loss.”** It must also sit above it. The natural-log logistic loss is \(\log 2 \approx 0.69\) at a zero score, below the 0-1 loss’s 1; divide by \(\log 2\) to make it a surrogate.

**“A bigger \(B\) is always better, because the class is richer.”** Richer classes lower the floor and raise the estimation term \(\rho B/\sqrt m\). Past the point where the floor stops falling (about \(B = 2.7\) in the lab world), more \(B\) only costs labels.

## Recap and the next bug

1. A set is **convex** when it contains every segment between its points; a function is **convex** when its graph lies below every chord, equivalently when its epigraph is convex. The 0-1 risk of halfspaces is not convex; its landscape is terraces.
2. For convex functions, every **local minimum is global**, and the **tangent inequality** \(f(\mathbf u) \ge f(\mathbf w) + \langle \nabla f(\mathbf w), \mathbf u - \mathbf w \rangle\) turns a small gradient into a certificate of near-optimality.
3. Convexity survives composition with a linear function, maxima, and nonnegative sums. So the square, logistic and hinge losses are convex in \(\mathbf w\), and so are their empirical risks.
4. A **\(\rho\)-Lipschitz** function changes by at most \(\rho\) times the change in its input; a **\(\beta\)-smooth** function has a \(\beta\)-Lipschitz gradient, which gives a quadratic ceiling, a guaranteed decrease of \(\|\nabla f\|^2/(2\beta)\) per step of size \(1/\beta\), and, for nonnegative functions, **self-boundedness**. Through a linear function with message \(\mathbf x\), the constants become \(\rho\|\mathbf x\|\) and \(\beta\|\mathbf x\|^2\).
5. In a **convex learning problem**, ERM is a convex optimization problem. But convexity is not enough to learn, even in one dimension, and convexity plus a bounded class is still not enough. **Convex-Lipschitz-bounded** and **convex-smooth-bounded** problems are learnable, as lessons 13 and 14 prove; with Lipschitz losses, \(m \ge \rho^2 B^2/\varepsilon^2\) examples suffice in expectation.
6. A **convex surrogate** is convex and upper bounds the original loss. The hinge loss is one for the 0-1 loss of halfspaces, and it is \(R\)-Lipschitz on messages of norm at most \(R\). The 0-1 risk of the learned halfspace is at most the approximation error plus the **optimization error** plus the estimation error.

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
