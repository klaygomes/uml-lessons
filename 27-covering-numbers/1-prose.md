# Covering Numbers

## The bound nobody can compute

Last lesson ended on a good note. You learned to measure a class’s richness on the very messages you have, with the Rademacher complexity, and you used it to certify a filter that every count made before seeing data had given up on. The review board liked it so much that it now asks for a Rademacher certificate on every filter that ships.

This week’s filter is new. It gives each message a **score** between \(-1\) and \(1\), built from two real-valued features that last month’s feature work produced (call them \(\psi_1\) and \(\psi_2\)): the score of message \(x\) is \(w_1 \psi_1(x) + w_2 \psi_2(x)\), for any pair of weights that keeps every score inside \([-1, 1]\). Junk gets a high score and keep a low one. Your users have labelled \(m\) messages, and the board wants a number: how well can this class fit random signs on those \(m\) messages?

You sit down to compute it, and you hit a wall within the hour. The Rademacher complexity is an average over all \(2^m\) ways to put random signs on your messages, and inside that average sits a best-possible fit taken over infinitely many score functions, one for every allowed pair of weights. With \(m = 10{,}000\) labels the average alone has \(2^{10{,}000}\) terms. You can’t compute it exactly, and a random estimate is a guess, not a certificate.

So you reach for the one tool from last lesson that turns a set into a number without any of that: Massart’s lemma. It is beautiful, and it says that a finite set of \(N\) score vectors has Rademacher complexity that grows only like \(\sqrt{\log N}\). But your set is not finite. Plug in \(N = \infty\) and Massart’s lemma tells you nothing at all.

Here is the question this lesson answers. Infinitely many score functions, yes, but surely most of them are nearly the same on your \(m\) messages. If you could pick a finite handful of representatives, so that every member of the class sits close to one of them, could you bound the whole class by the handful plus a small correction for “close”? And how many representatives does it take?

## Score vectors, and what “close” means

First, a reminder of how lesson 26 turned a class into a set of vectors, because everything in this lesson lives in that set.

Take the \(m\) labelled messages \(x_1, \dots, x_m\). Any one score function \(s\) gives each of them a number, so it gives a list of \(m\) numbers, \((s(x_1), \dots, s(x_m))\). Call that list the **score vector** of \(s\). It is a point of \(\mathbb R^m\), the space of all lists of \(m\) real numbers. The **score set** \(A\) is the set of score vectors of every score function in the class. Two score functions that agree on all \(m\) messages give the same score vector, so the score set is exactly what the labelled messages can see of the class.

To say “close”, we need a distance between score vectors. We use the ordinary straight-line distance. For two vectors \(a\) and \(a'\) in \(\mathbb R^m\),

\[ \|a - a'\| = \sqrt{\sum_{i=1}^{m} (a_i - a'_i)^2}. \]

Read it from the inside out. \(a_i\) is the \(i\)-th entry of \(a\), the score that \(a\) gives message \(i\), and \(a_i - a'_i\) is how much the two vectors disagree on that message. Squaring makes every disagreement positive. The big sigma \(\sum_{i=1}^m\) adds those squares over all positions \(i = 1, 2, \dots, m\), and the square root brings the units back to scores. The double bars \(\|\cdot\|\) are the name of this quantity, the **Euclidean norm** of the difference. In plain words: **the distance between two score vectors is Pythagoras’ theorem in \(m\) dimensions.** With two messages, the vectors \((0.9, 0.8)\) and \((0.8, 0.85)\) differ by \(0.1\) and \(-0.05\), so their distance is \(\sqrt{0.01 + 0.0025} \approx 0.11\). In Python that is `math.dist((0.9, 0.8), (0.8, 0.85))`, which prints `0.11180339887498948`.

## A handful of representatives

Let’s shrink the mail room again until we can draw it. Suppose there are only \(m = 2\) labelled messages, so every score vector is a point in the plane, and suppose the class has just six score functions, with these score vectors:

\[ (0.9, 0.8),\ (0.7, 0.9),\ (-0.8, -0.6),\ (-0.9, -0.9),\ (0.1, -0.2),\ (-0.2, 0.1). \]

Look at them for a moment before reading on. Do you see that they come in three little clumps? Two vectors near the top right, where both messages score high; two near the bottom left; and two near the middle.

Now pick three **representatives**, one in the middle of each clump: \((0.8, 0.85)\), \((-0.85, -0.75)\) and \((0, 0)\). For each of the six score vectors, find the representative nearest to it and measure the distance.

| score vector | nearest representative | distance |
|---|---|---|
| \((0.9, 0.8)\) | \((0.8, 0.85)\) | 0.11 |
| \((0.7, 0.9)\) | \((0.8, 0.85)\) | 0.11 |
| \((-0.8, -0.6)\) | \((-0.85, -0.75)\) | 0.16 |
| \((-0.9, -0.9)\) | \((-0.85, -0.75)\) | 0.16 |
| \((0.1, -0.2)\) | \((0, 0)\) | 0.22 |
| \((-0.2, 0.1)\) | \((0, 0)\) | 0.22 |

Every score vector is within \(0.5\) of some representative; in fact within \(0.22\). That is the whole idea of this lesson in one table. If you know how the three representatives behave, you know how all six behave, up to an error of at most \(0.5\) in distance.

Notice one more thing: the representative \((0, 0)\) is not itself one of the six score vectors. It doesn’t need to be. A representative is any point of \(\mathbb R^m\) we like; its only job is to be close.

```python
import math
A = [(0.9, 0.8), (0.7, 0.9), (-0.8, -0.6),
     (-0.9, -0.9), (0.1, -0.2), (-0.2, 0.1)]
B = [(0.8, 0.85), (-0.85, -0.75), (0.0, 0.0)]
r = 0.5
nearest = [min(math.dist(a, b) for b in B) for a in A]
print([round(t, 2) for t in nearest])
print(max(nearest) <= r)
```

It prints `[0.11, 0.11, 0.16, 0.16, 0.22, 0.22]` and `True`: the table’s last column, and the verdict that every distance is at most \(r = 0.5\).

Could two representatives do the job at distance \(0.5\)? No, and the reason is worth seeing. The key is the **triangle inequality**, the rule that a detour is never shorter than the direct road: for any three vectors, \(\|u - v\| \le \|u - b\| + \|b - v\|\). So if one representative \(b\) is within \(0.5\) of both \(u\) and \(v\), then \(u\) and \(v\) are at most \(1\) apart. Turn that around: two score vectors more than \(1\) apart can never share a representative at radius \(0.5\).

Now measure. Every top-right vector is more than \(1.2\) from every other vector (the closest such pair, \((0.7, 0.9)\) and \((-0.2, 0.1)\), is \(\sqrt{0.81 + 0.64} \approx 1.20\) apart), so the top-right clump needs a representative of its own that serves nobody else. With only two representatives, the second one would have to serve all four remaining vectors, including \((-0.9, -0.9)\) and \((-0.2, 0.1)\); but those two are \(\sqrt{0.49 + 1} \approx 1.22\) apart, more than \(1\). So two representatives cannot do it, and three is the least that works at distance \(0.5\).

[[diagram:cover]]

## The formal definition

Now the names. Fix a set \(A\) of vectors in \(\mathbb R^m\) and a distance \(r > 0\), the **radius**.

**Definition (cover).** A set \(A'\) of vectors in \(\mathbb R^m\) is an **\(r\)-cover** of \(A\), and we say \(A\) is \(r\)-covered by \(A'\), if

\[ \text{for every } a \in A \text{ there is an } a' \in A' \text{ with } \|a - a'\| \le r. \]

The symbol \(\in\) reads “is a member of”, so \(a \in A\) is “\(a\) is one of the score vectors”. The members of \(A'\) are the **representatives**. In plain words: **every score vector has a representative within distance \(r\).** In the clump example, \(A'\) is the three representatives, \(r = 0.5\), and the snippet above checked the condition vector by vector.

**Definition (covering number).** The **covering number** \(N(r, A)\) is the number of elements of the smallest \(r\)-cover of \(A\):

\[ N(r, A) = \min \left\{ |A'| : A' \text{ is an } r\text{-cover of } A \right\}. \]

Read it from the inside out. The braces collect a number for every possible cover, and the colon reads “such that”: the set of sizes \(|A'|\) such that \(A'\) is an \(r\)-cover. The bars \(|A'|\) count the elements of \(A'\). Then \(\min\) takes the smallest of those sizes. In plain words: **the covering number is how many representatives you need, at the least, to put every score vector within \(r\) of one.** In the clump example we just showed \(N(0.5, A) = 3\). (If no finite cover exists, the covering number is \(+\infty\); for every set in this lesson, finite covers exist.)

Two quick facts you can see without a proof. Shrinking the radius can only make the job harder, so \(N(r, A)\) never goes down as \(r\) goes down: a cover at radius \(r\) is also a cover at any larger radius. And if every score vector is within \(c\) of one point \(\bar a\), then \(\{\bar a\}\) alone is a \(c\)-cover, so \(N(c, A) = 1\). Hold on to that second fact; it is where the chain of this lesson will start.

## Covering a flat set: the grid

The six-vector class was a warm-up. Your real class is infinite, so we need a cover we can build for an infinite set. Here is where the structure of the filter pays off.

Every score function is \(w_1 \psi_1 + w_2 \psi_2\). On the \(m\) messages, its score vector is \(w_1\) times the vector of \(\psi_1\)-values plus \(w_2\) times the vector of \(\psi_2\)-values. So every score vector is a combination of the same two fixed vectors: the whole score set lies in a flat two-dimensional slice of \(\mathbb R^m\), however large \(m\) is. Mathematicians call such a slice a **subspace**, and its **dimension** \(d\) is the number of directions it spreads in, here \(d = 2\).

Also, every score is in \([-1, 1]\), so every score vector has length at most \(\sqrt{1 + 1 + \dots + 1} = \sqrt m\). Write \(c\) for a bound on the lengths, \(\|a\| \le c\) for every \(a \in A\).

A flat \(d\)-dimensional slice can be described with \(d\) coordinates. Choose \(d\) directions \(v_1, \dots, v_d\) in the subspace that have length 1 and are at right angles to each other (an **orthonormal basis**). Every vector of the slice is \(a = \alpha_1 v_1 + \dots + \alpha_d v_d\) for some numbers \(\alpha_1, \dots, \alpha_d\), its coordinates, and because the directions are at right angles and of length 1, lengths and distances in the slice are just the ordinary distances between coordinate lists: \(\|a\| = \sqrt{\alpha_1^2 + \dots + \alpha_d^2}\). So the score set is, for every purpose of distance, a set inside a disc of radius \(c\) in \(d\) dimensions, and \(m\) has disappeared from the picture.

How would you cover a disc? Lay a grid over it. Here is the plan, with \(d = 2\), \(c = \sqrt 2\) (two messages, scores in \([-1, 1]\)) and \(r = 0.6\).

1. Each coordinate of a score vector lies in \([-c, c]\), because one coordinate can never exceed the whole length.
2. Cut \([-c, c]\) into \(n\) equal cells of width \(2c/n\), and put a grid value at the middle of each cell. Any number in \([-c, c]\) is then within half a cell, \(c/n\), of a grid value.
3. Take as representatives every combination of grid values, one per coordinate: \(n^d\) points.
4. Round each coordinate of a score vector to the nearest grid value. Each coordinate moves by at most \(c/n\), so the vector moves by at most \(\sqrt{d \,(c/n)^2} = \sqrt d \, c / n\).
5. Choose \(n\) just large enough that this is at most \(r\): \(n = \lceil c \sqrt d / r \rceil\), where \(\lceil t \rceil\) (the **ceiling** of \(t\)) is the smallest whole number at least \(t\).

With our numbers, \(c \sqrt d / r = \sqrt 2 \cdot \sqrt 2 / 0.6 \approx 3.33\), so \(n = 4\) values per coordinate and \(4^2 = 16\) representatives.

```python
import math, random
def grid_cover(c, d, r):
    n = math.ceil(c * math.sqrt(d) / r)
    width = 2 * c / n
    axis = [-c + width / 2 + j * width for j in range(n)]
    pts = [()]
    for _ in range(d):
        pts = [p + (t,) for p in pts for t in axis]
    return pts
c, d, r = math.sqrt(2), 2, 0.6
B = grid_cover(c, d, r)
rng = random.Random(0)
box = [(rng.uniform(-c, c), rng.uniform(-c, c))
       for _ in range(9000)]
A = [a for a in box if math.hypot(*a) <= c]
worst = max(min(math.dist(a, b) for b in B) for a in A)
bound = (2 * c * math.sqrt(d) / r) ** d
print(len(B), round(bound), round(worst, 3))
```

It prints `16 44 0.498`. The grid has 16 points; the farthest of about 7,000 random points of the disc is \(0.498\) from its nearest representative, safely within \(r = 0.6\); and 44 is the rounder bound we prove next, which is looser but easier to carry around.

[[diagram:grid]]

**Proposition (covering a flat set).** Let \(A \subset \mathbb R^m\) lie in a subspace of dimension \(d\), with \(\|a\| \le c\) for every \(a \in A\). Then for every \(r > 0\),

\[ N(r, A) \le \left\lceil \frac{c \sqrt d}{r} \right\rceil^{d}, \qquad \text{and, when } r \le c\sqrt d, \qquad N(r, A) \le \left( \frac{2 c \sqrt d}{r} \right)^{d}. \]

Here \(\subset\) reads “is contained in”. The fraction \(c \sqrt d / r\) compares the size of the set, \(c\), with the radius, \(r\), and the factor \(\sqrt d\) pays for the rounding errors of \(d\) coordinates adding up. Raising to the power \(d\) counts one choice per coordinate. In plain words: **a flat set of dimension \(d\) needs about \((c/r)^d\) representatives, whatever the number of messages \(m\).** With \(c = \sqrt 2\), \(d = 2\), \(r = 0.6\): the first bound is \(16\), the second is \((4/0.6)^2 \approx 44\), the two numbers the snippet printed.

*Proof.* Let \(v_1, \dots, v_d\) be an orthonormal basis of the subspace, and write each \(a \in A\) as \(a = \sum_{i=1}^d \alpha_i v_i\). Because the \(v_i\) are orthonormal, \(\|a\|^2 = \sum_i \alpha_i^2\), so each \(|\alpha_i| \le \|a\| \le c\).

Let \(n = \lceil c\sqrt d / r \rceil\) and let \(G\) be the \(n\) midpoints of the cells of width \(2c/n\) in \([-c, c]\). Define \(A' = \{ \sum_i \alpha'_i v_i : \alpha'_i \in G \text{ for every } i \}\), which has \(n^d\) elements.

Given \(a \in A\), choose for each \(i\) the midpoint \(\alpha'_i \in G\) of the cell containing \(\alpha_i\), so \(|\alpha_i - \alpha'_i| \le c/n\). Then, again by orthonormality,

\[ \|a - a'\|^2 = \Big\| \sum_{i=1}^d (\alpha_i - \alpha'_i) v_i \Big\|^2 = \sum_{i=1}^d (\alpha_i - \alpha'_i)^2 \le d \, \frac{c^2}{n^2} \le r^2, \]

where the last step uses \(n \ge c\sqrt d / r\). So \(A'\) is an \(r\)-cover with \(n^d\) elements, which proves the first bound.

For the second, when \(t = c\sqrt d / r \ge 1\) we have \(\lceil t \rceil < t + 1 \le 2t\), so \(n^d \le (2c\sqrt d / r)^d\). \(\blacksquare\)

Why the condition \(r \le c\sqrt d\)? For a huge radius the second formula drops below 1, but no set can be covered by fewer than one representative. The first bound has no such problem. In the rest of the lesson the radius is always at most \(c/2\), so both bounds hold.

## Two properties that move covers around

Covers behave well when you transform the score set, and two transformations come up again and again. Both proofs are the same trick: transform the representatives along with the score vectors.

**Shifting and stretching.** Suppose you add the same vector \(a_0\) to every score vector (shift the whole set), or multiply every score vector by the same number \(s > 0\) (stretch it). Shifting moves every score vector and its representative together, so distances don’t change. Stretching by \(s\) multiplies every distance by \(s\), so a cover at radius \(r\) becomes a cover at radius \(s r\).

**Lemma (shift and stretch).** For any \(A \subset \mathbb R^m\), any \(s > 0\) and any \(a_0 \in \mathbb R^m\),

\[ N\big(s r, \{ s a + a_0 : a \in A \}\big) \le N(r, A) \quad \text{for every } r > 0. \]

The set \(\{ s a + a_0 : a \in A \}\) is “every vector you get by taking a score vector \(a\), stretching it by \(s\) and shifting it by \(a_0\)”. In plain words: **stretch a set by \(s\) and the same number of representatives covers it at \(s\) times the radius; shift it and nothing changes.** In the clump example, stretching by \(s = 0.5\) and shifting by \((0.3, -0.1)\) turns the worst distance \(0.224\) into \(0.112\), still with three representatives.

*Proof.* Let \(A'\) be a smallest \(r\)-cover of \(A\) and put \(B' = \{ s a' + a_0 : a' \in A' \}\), which has at most \(|A'|\) elements. Any member of the new set is \(s a + a_0\) for some \(a \in A\); pick \(a' \in A'\) with \(\|a - a'\| \le r\). Then \(\|(s a + a_0) - (s a' + a_0)\| = s\|a - a'\| \le s r\). So \(B'\) is an \(s r\)-cover of the new set. \(\blacksquare\)

Two consequences we will use: with \(s = 1\), shifting never changes a covering number (apply the lemma with \(a_0\) and then with \(-a_0\)); and, writing \(sA\) for the stretched set \(\{s a : a \in A\}\) and applying the lemma at radius \(r/s\), \(N(r, sA) \le N(r/s, A)\).

```python
import math
A = [(0.9, 0.8), (0.7, 0.9), (-0.8, -0.6),
     (-0.9, -0.9), (0.1, -0.2), (-0.2, 0.1)]
B = [(0.8, 0.85), (-0.85, -0.75), (0.0, 0.0)]
s, a0 = 0.5, (0.3, -0.1)
move = lambda v: tuple(s * vi + oi for vi, oi in zip(v, a0))
radius = lambda A, B: max(min(math.dist(a, b) for b in B)
                          for a in A)
before = radius(A, B)
after = radius([move(a) for a in A], [move(b) for b in B])
print(round(before, 3), round(after, 3), len(B))
```

It prints `0.224 0.112 3`: the worst distance halves along with the stretch, and the number of representatives stays three.

**Passing through a loss.** The review board doesn’t care about scores; it cares about losses. Lesson 26 turned scores into losses one message at a time, with a function per message. For the filter, the **ramp loss** of a score \(t\) on a message with label \(y \in \{-1, +1\}\) is \(\min(1, \max(0, 1 - y t))\): zero when the score is on the right side by a margin of 1, one when it is on the wrong side, and a straight ramp in between. A junk message (\(y = +1\)) scored \(t = 0.4\) has \(1 - yt = 0.6\), inside \([0, 1]\), so its loss is \(0.6\); scored \(t = -0.3\), it gets \(1.3\), capped to a loss of \(1\). What matters here is that it is **\(\rho\)-Lipschitz** with \(\rho = 1\): changing the score by some amount changes the loss by at most \(\rho\) times that amount. In symbols, a function \(\varphi: \mathbb R \to \mathbb R\) is \(\rho\)-Lipschitz when \(|\varphi(t) - \varphi(t')| \le \rho\, |t - t'|\) for all \(t, t'\).

Apply such a function to each entry of a score vector, possibly a different function \(\varphi_i\) for each message \(i\) (the loss depends on the message’s label). Write \(\varphi(a) = (\varphi_1(a_1), \dots, \varphi_m(a_m))\) and \(\varphi \circ A = \{\varphi(a) : a \in A\}\) for the set of all loss vectors. Each entry moves at most \(\rho\) times as far as before, so every distance between loss vectors is at most \(\rho\) times the distance between their score vectors.

**Lemma (Lipschitz losses keep covers).** If every \(\varphi_i\) is \(\rho\)-Lipschitz, then

\[ N(\rho r, \varphi \circ A) \le N(r, A) \quad \text{for every } r > 0. \]

In plain words: **after a \(\rho\)-Lipschitz loss, the same number of representatives covers the loss vectors at \(\rho\) times the radius.** For the ramp loss, \(\rho = 1\), so losses are no harder to cover than scores.

*Proof.* Let \(A'\) be a smallest \(r\)-cover of \(A\) and put \(B' = \varphi \circ A'\). For \(a \in A\) pick \(a' \in A'\) with \(\|a - a'\| \le r\). Then

\[ \|\varphi(a) - \varphi(a')\|^2 = \sum_{i=1}^m \big(\varphi_i(a_i) - \varphi_i(a'_i)\big)^2 \le \rho^2 \sum_{i=1}^m (a_i - a'_i)^2 \le (\rho r)^2, \]

where the middle step applies the Lipschitz property entry by entry and squares it. So \(B'\) is a \(\rho r\)-cover of \(\varphi \circ A\) with at most \(|A'|\) elements. \(\blacksquare\)

Here it is on the clump example, with labels \(+1\) for message 1 and \(-1\) for message 2. For each score vector, the first number is its distance to its representative, the second the distance between their loss vectors.

```python
import math
A = [(0.9, 0.8), (0.7, 0.9), (-0.8, -0.6),
     (-0.9, -0.9), (0.1, -0.2), (-0.2, 0.1)]
B = [(0.8, 0.85), (-0.85, -0.75), (0.0, 0.0)]
y = (1, -1)
ramp = lambda t: min(1.0, max(0.0, 1 - t))
phi = lambda v: tuple(ramp(yi * vi) for yi, vi in zip(y, v))
for a in A:
    b = min(B, key=lambda b: math.dist(a, b))
    print(round(math.dist(a, b), 3),
          round(math.dist(phi(a), phi(b)), 3))
```

It prints the pairs `0.112 0.1`, `0.112 0.1`, `0.158 0.15`, `0.158 0.15`, `0.224 0.224` and `0.224 0.0`: the loss distance is never larger than the score distance, and the ramp’s flat parts sometimes squash it to nothing.

## Massart’s lemma, the finite tool

Before we cash covers in for Rademacher complexity, let’s put last lesson’s two key formulas back on the table.

The **Rademacher complexity** of a set \(A \subset \mathbb R^m\) is

\[ R(A) = \frac{1}{m} \, \mathbb E_{\sigma} \Big[ \sup_{a \in A} \sum_{i=1}^m \sigma_i a_i \Big]. \]

Read it from the inside out. \(\sigma = (\sigma_1, \dots, \sigma_m)\) is a vector of **random signs**: each \(\sigma_i\) is \(+1\) or \(-1\) with probability \(1/2\), independently of the others. The sum \(\sum_i \sigma_i a_i\) measures how well the score vector \(a\) lines up with those signs: it is large when \(a\) is high where \(\sigma\) says \(+1\) and low where it says \(-1\). We also write it as \(\langle \sigma, a \rangle\), the **inner product** of \(\sigma\) and \(a\). The \(\sup\) (supremum) takes the best value over all \(a\) in the set, the largest it gets or the limit it approaches. \(\mathbb E_\sigma\) averages over the random signs, and \(1/m\) makes it a per-message number. In plain words: **the Rademacher complexity is how well the class can fit pure noise on your messages, on average, per message.** For the six clump vectors, the four sign patterns give best fits \(1.7\), \(1.8\), \(0.3\) and \(0.3\), whose average is \(1.025\), and dividing by \(m = 2\) gives \(R(A) \approx 0.513\).

**Massart’s lemma**, in the form we need: if \(A\) is finite with \(N\) elements, then

\[ R(A) \le \max_{a \in A} \|a\| \; \frac{\sqrt{2 \log N}}{m}. \]

Here \(\max_{a \in A} \|a\|\) is the length of the longest vector in \(A\), and \(\log\) is the natural logarithm. In plain words: **a finite set can fit noise only as well as its longest vector times \(\sqrt{2\log N}\), per message.** For the clumps, the longest vector is \((-0.9,-0.9)\), of length \(1.27\), and \(1.27 \cdot \sqrt{2 \log 6} / 2 \approx 1.205\), safely above the true \(0.513\).

```python
import itertools, math
A = [(0.9, 0.8), (0.7, 0.9), (-0.8, -0.6),
     (-0.9, -0.9), (0.1, -0.2), (-0.2, 0.1)]
m, N = 2, len(A)
signs = list(itertools.product([-1, 1], repeat=m))
dot = lambda s, a: sum(si * ai for si, ai in zip(s, a))
R = sum(max(dot(s, a) for a in A) for s in signs) / len(signs) / m
longest = max(math.hypot(*a) for a in A)
massart = longest * math.sqrt(2 * math.log(N)) / m
print(round(R, 3), round(massart, 3))
```

It prints `0.513 1.205`.

Lesson 26 stated Massart’s lemma with the longest distance from the average vector \(\bar a\). Its proof first shifted the set so that \(\bar a\) sat at the origin, and from then on used only the length of the longest vector; skip the shift and you get the form above, which is the one we will apply. So that nothing rests on memory, here is the proof in four lines. If \(N = 1\) or \(L = 0\), the set is a single vector, every sign vector’s best fit averages to \(0\), and \(R(A) = 0\). Otherwise, for any \(\lambda > 0\), write \(L = \max_a \|a\|\). Then

\[ m R(A) = \frac{1}{\lambda} \mathbb E \Big[\log \max_{a} e^{\lambda \langle \sigma, a \rangle}\Big] \le \frac{1}{\lambda} \log \sum_{a \in A} \mathbb E\, e^{\lambda \langle \sigma, a \rangle} = \frac{1}{\lambda} \log \sum_{a \in A} \prod_{i=1}^m \frac{e^{\lambda a_i} + e^{-\lambda a_i}}{2}. \]

The first equality holds because \(\log\) and \(e^{(\cdot)}\) undo each other and \(e^{\lambda t}\) is increasing. The inequality replaces the maximum by the sum (all terms are positive), then moves the expectation inside the logarithm, which can only increase the value because \(\log\) is concave (Jensen’s inequality). The last equality uses the independence of the signs: the expectation of a product of independent factors \(e^{\lambda \sigma_i a_i}\) is the product of their expectations, each the average of its two values. Now \((e^t + e^{-t})/2 \le e^{t^2/2}\) (compare the two power series term by term), so the product is at most \(e^{\lambda^2 \|a\|^2 / 2} \le e^{\lambda^2 L^2/2}\), and

\[ m R(A) \le \frac{\log N}{\lambda} + \frac{\lambda L^2}{2}. \]

Choosing \(\lambda = \sqrt{2 \log N} / L\) makes both terms \(L\sqrt{\log N / 2}\), so \(mR(A) \le L\sqrt{2\log N}\). \(\blacksquare\)

## One cover, one bound

Now we can answer the opening question in its simplest form. Take the infinite score set and an \(r\)-cover of it with \(N(r, A)\) representatives. For any sign vector \(\sigma\), the best-fitting score vector \(a\) has a representative \(a'\) within \(r\). Split its fit into two pieces:

\[ \langle \sigma, a \rangle = \langle \sigma, a' \rangle + \langle \sigma, a - a' \rangle. \]

The first piece is the fit of a representative, a member of a finite set, so Massart’s lemma handles it. The second piece is the fit of a short vector, of length at most \(r\), and a short vector can’t line up with anything very well: by the Cauchy–Schwarz inequality, an inner product is at most the product of the two lengths, \(\langle \sigma, u \rangle \le \|\sigma\|\, \|u\|\), and every sign vector has length \(\|\sigma\| = \sqrt{1 + \dots + 1} = \sqrt m\). So the second piece is at most \(\sqrt m \, r\).

**Proposition (one-scale bound).** Let every \(a \in A\) satisfy \(\|a\| \le c\). Then for every \(r > 0\),

\[ R(A) \le \frac{r}{\sqrt m} + (c + r) \, \frac{\sqrt{2 \log N(r, A)}}{m}. \]

The first term is the price of rounding to the representatives: at most \(r\) per sign vector after Cauchy–Schwarz, divided by \(m\). The second is Massart’s lemma on the representatives, whose lengths are at most \(c + r\) (a representative is within \(r\) of a vector of length at most \(c\), and we may throw away representatives that are within \(r\) of nothing). In plain words: **a class fits noise no better than its representatives do, plus a rounding error that shrinks with the radius.** Take \(m = 2\), \(c = \sqrt 2\) and the 16-point grid at \(r = 0.6\): the bound is \(0.6/\sqrt 2 + (1.414 + 0.6)\sqrt{2\log 16}/2 \approx 0.42 + 2.37\). Tiny worlds make every bound look silly; the point is that it is finite.

*Proof.* Let \(A'\) be a smallest \(r\)-cover and drop any representative that is more than \(r\) from every \(a \in A\); what remains is still an \(r\)-cover, with at most \(N(r,A)\) elements, each of length at most \(c + r\) by the triangle inequality. For each \(a \in A\) let \(\pi(a)\) be a representative within \(r\). For a fixed \(\sigma\) and every \(a \in A\),

\[ \langle \sigma, a \rangle = \langle \sigma, \pi(a) \rangle + \langle \sigma, a - \pi(a) \rangle \le \max_{a' \in A'} \langle \sigma, a' \rangle + \sqrt m \, r. \]

The right side doesn’t depend on \(a\), so it also bounds the supremum over \(a \in A\). Average over \(\sigma\), divide by \(m\), and apply Massart’s lemma to \(A'\). \(\blacksquare\)

How good is it? Here is a world we can compute exactly: \(m = 10\) messages, scores in \([-1,1]\), so \(c = \sqrt{10}\), and a two-feature filter whose features, after rescaling, are “on” for the first five messages and for the last five. On that flat set, the best fit to \(\sigma\) is \(c\) times the length of \(\sigma\)’s shadow on the slice, which we can average over all \(2^{10} = 1{,}024\) sign vectors.

```python
import itertools, math
m, d = 10, 2
c = math.sqrt(m)
proj = lambda s: math.hypot(sum(s[:5]), sum(s[5:])) / math.sqrt(5)
signs = itertools.product([-1, 1], repeat=m)
R = c * sum(proj(s) for s in signs) / 2 ** m / m
N = lambda r: math.ceil(c * math.sqrt(d) / r) ** d
one = lambda r: (r / math.sqrt(m)
                 + (c + r) * math.sqrt(2 * math.log(N(r))) / m)
best = min((one(i / 100), i / 100) for i in range(1, 400))
print(round(R, 3), round(best[0], 3), best[1])
```

It prints `0.404 1.233 0.32`: the exact Rademacher complexity is \(0.404\), and the best one-scale bound, at radius \(0.32\), is \(1.233\). Loose, but honest, and computed without touching the infinite set.

There is a tension hiding in that bound, and it is the reason for the rest of the lesson. A small radius makes the rounding term small but needs many representatives; a large radius needs few representatives but rounds badly. As \(m\) grows, the set grows too (\(c = \sqrt m\)), so a fixed radius needs more and more grid points, and the logarithm of their number creeps up like \(\log m\). One scale can’t be right for both pieces. What if we didn’t have to choose?

## Chaining: many scales at once

Here is the idea, as a picture before any formula. Imagine finding your way to a house in a city you don’t know. One instruction, “it’s 3 km north-east”, is short but useless for finding the door. A street address is precise but needs a huge lookup table. What actually works is a chain: country, then city, then street, then house number. Each step is a choice among a modest number of options, and each step is shorter than the last.

Chaining does this for the best-fitting score vector \(a^*\). Build covers of \(A\) at radii \(c, c/2, c/4, \dots, c 2^{-M}\), halving each time. At level \(k\), let \(b^{(k)}\) be the representative nearest to \(a^*\) in the level-\(k\) cover. Then walk:

\[ a^* = \big(a^* - b^{(M)}\big) + \big(b^{(M)} - b^{(M-1)}\big) + \dots + \big(b^{(1)} - b^{(0)}\big) + b^{(0)}. \]

All the middle terms cancel in pairs, so this is a true equation, a **telescoping sum**. Each bracket \(b^{(k)} - b^{(k-1)}\) is a **link** of the chain. Why is this better than one cover? Because a link at level \(k\) is short, at most about \(3c 2^{-k}\) long, and it is one of a limited number of possible links, at most \(N(c2^{-k}, A)^2\) of them. Short links among few choices are exactly what Massart’s lemma likes. Coarse levels have long links but very few choices; fine levels have many choices but very short links. Every level is charged the right price.

Let’s watch one chain in the ten-message world. The sign vector is fixed, \(a^*\) is the best-fitting score vector for it, the cover at level \(k\) is the grid at radius \(c 2^{-k}\), and level 0 is the single point \(0\), since every score vector has length at most \(c\).

```python
import math
m, c = 10, math.sqrt(10)
sigma = (1, 1, -1, 1, 1, -1, -1, 1, -1, -1)
u = (sum(sigma[:5]) / 5 ** 0.5, sum(sigma[5:]) / 5 ** 0.5)
a_star = tuple(c * t / math.hypot(*u) for t in u)
def b(k):
    n = math.ceil(c * math.sqrt(2) / (c * 2 ** -k))
    width = 2 * c / n
    j = lambda t: min(n - 1, int((t + c) // width))
    return tuple(-c + width / 2 + width * j(t)
                 for t in a_star)
chain = [(0.0, 0.0)] + [b(k) for k in range(1, 6)]
steps = [math.dist(chain[k], chain[k - 1]) for k in range(1, 6)]
print([round(s, 2) for s in steps])
print([round(3 * c * 2 ** -k, 2) for k in range(1, 6)])
print(round(math.dist(a_star, chain[-1]), 3), round(c / 32, 3))
```

It prints the link lengths `[2.98, 0.75, 0.37, 0.24, 0.1]`, the promised limits `[4.74, 2.37, 1.19, 0.59, 0.3]`, and the last gap `0.046` against its limit \(c/32 \approx 0.099\). (The vectors are written in the slice’s two coordinates, which preserve every distance.) Every link is shorter than its limit, and the links shrink roughly by half each level.

[[diagram:chain]]

## The chaining bound, proved

**Proposition (chaining bound).** Let \(A \subset \mathbb R^m\), and let \(c > 0\) and \(\bar a \in \mathbb R^m\) be such that \(\|a - \bar a\| \le c\) for every \(a \in A\). (The tightest choice is the smallest such \(c\), \(c = \min_{\bar a} \max_{a \in A} \|a - \bar a\|\).) Then for every whole number \(M \ge 1\),

\[ R(A) \le \frac{c \, 2^{-M}}{\sqrt m} + \frac{6c}{m} \sum_{k=1}^{M} 2^{-k} \sqrt{\log N\big(c 2^{-k}, A\big)}. \]

Read it from the inside out. \(c 2^{-k}\) is the radius at level \(k\): \(c\) halved \(k\) times. \(N(c 2^{-k}, A)\) is how many representatives that level needs, and \(\sqrt{\log N}\) is what Massart’s lemma charges for choosing among them. The factor \(2^{-k}\) is the link length shrinking, and the sum \(\sum_{k=1}^M\) adds the charges of levels 1 to \(M\). The \(6c/m\) collects constants: \(3c2^{-k}\) is the longest a link can be, and a factor 2 comes from links being pairs of representatives. The first term, \(c 2^{-M}/\sqrt m\), is the last rounding, from \(a^*\) to its finest representative. In plain words: **the class fits noise no better than the sum, over halving scales, of each scale’s link length times the square root of the log of its covering number, plus the finest rounding error.** In the ten-thousand-message filter (\(c = \sqrt{10{,}000} = 100\), grids of 9, 36, 144, \(\dots\) points), one level gives \(0.5 + 0.06 \cdot 0.5 \sqrt{\log 9} \approx 0.54\); the snippet after the proof goes further.

*Proof.* **Step 1: center the set.** Shifting every vector of \(A\) by \(-\bar a\) changes no covering number (the shift and stretch lemma) and no Rademacher complexity, because \(\mathbb E \langle \sigma, a - \bar a\rangle = \mathbb E\langle \sigma, a\rangle - \langle \mathbb E\sigma, \bar a\rangle\) and \(\mathbb E \sigma = 0\). So assume \(\bar a = 0\): every \(a \in A\) has \(\|a\| \le c\).

**Step 2: the covers.** Let \(B_0 = \{0\}\); it is a \(c\)-cover of \(A\). For \(k = 1, \dots, M\), let \(B_k\) be a smallest \(c 2^{-k}\)-cover of \(A\), so \(|B_k| = N(c2^{-k}, A)\). (If one of these is infinite, the bound is \(+\infty\) and there is nothing to prove.)

**Step 3: the best fit.** Fix \(\eta > 0\). For each sign vector \(\sigma\), choose \(a^* \in A\) with \(\langle \sigma, a^* \rangle \ge \sup_{a\in A} \langle \sigma, a\rangle - \eta\). (When the supremum is attained, as it is for the closed disc, take \(\eta = 0\).) So \(a^*\) depends on \(\sigma\). For each \(k\), let \(b^{(k)}\) be a member of \(B_k\) nearest to \(a^*\); it depends on \(\sigma\) too, and \(\|a^* - b^{(k)}\| \le c 2^{-k}\) because \(B_k\) is a cover at that radius. In particular \(b^{(0)} = 0\).

**Step 4: the links are short.** By the triangle inequality,

\[ \|b^{(k)} - b^{(k-1)}\| \le \|b^{(k)} - a^*\| + \|a^* - b^{(k-1)}\| \le c2^{-k} + c 2^{-(k-1)} = 3c\,2^{-k}. \]

**Step 5: the links come from small sets.** Define the set of possible links at level \(k\),

\[ \hat B_k = \big\{ b - b' : b \in B_k,\ b' \in B_{k-1},\ \|b - b'\| \le 3c 2^{-k} \big\}. \]

By step 4, \(b^{(k)} - b^{(k-1)} \in \hat B_k\) whatever \(\sigma\) is. And \(\hat B_k\) has at most \(|B_k| \cdot |B_{k-1}|\) elements, one per pair. Covering numbers never go down as the radius shrinks, so \(|B_{k-1}| \le |B_k|\) and \(|\hat B_k| \le N(c2^{-k}, A)^2\).

**Step 6: telescope and split.** Because \(b^{(0)} = 0\), the telescoping sum gives \(a^* = (a^* - b^{(M)}) + \sum_{k=1}^M (b^{(k)} - b^{(k-1)})\). Take the inner product with \(\sigma\) and the expectation over \(\sigma\):

\[ m R(A) \le \mathbb E\langle \sigma, a^* \rangle + \eta = \mathbb E \langle \sigma, a^* - b^{(M)} \rangle + \sum_{k=1}^M \mathbb E \langle \sigma, b^{(k)} - b^{(k-1)} \rangle + \eta. \]

The inequality is the choice of \(a^*\) in step 3, the equality is linearity of the inner product and of the expectation.

**Step 7: the last rounding.** By Cauchy–Schwarz, \(\langle \sigma, a^* - b^{(M)}\rangle \le \|\sigma\|\,\|a^* - b^{(M)}\| \le \sqrt m \, c 2^{-M}\).

**Step 8: each link, by Massart.** Since \(b^{(k)} - b^{(k-1)}\) is one member of \(\hat B_k\), its fit is at most the best fit in \(\hat B_k\): \(\langle \sigma, b^{(k)} - b^{(k-1)}\rangle \le \max_{u \in \hat B_k} \langle \sigma, u \rangle\). Average over \(\sigma\): the right side averages to \(m R(\hat B_k)\). Massart’s lemma, with every \(u \in \hat B_k\) of length at most \(3c2^{-k}\) and at most \(N(c2^{-k},A)^2\) elements, gives

\[ \mathbb E \langle \sigma, b^{(k)} - b^{(k-1)}\rangle \le 3c 2^{-k} \sqrt{2 \log\big(N(c2^{-k}, A)^2\big)} = 6c 2^{-k} \sqrt{\log N(c2^{-k}, A)}, \]

using \(\log(N^2) = 2\log N\) and \(\sqrt{2 \cdot 2} = 2\).

**Step 9: add up.** Put steps 7 and 8 into step 6 and divide by \(m\):

\[ R(A) \le \frac{c\,2^{-M}}{\sqrt m} + \frac{6c}{m}\sum_{k=1}^M 2^{-k}\sqrt{\log N(c2^{-k}, A)} + \frac{\eta}{m}. \]

This holds for every \(\eta > 0\), so it holds with \(\eta = 0\). \(\blacksquare\)

One remark that matters for the lab: the proof only used the sizes of the covers \(B_k\) through Massart’s lemma, and that bound grows with the size. So you may plug in the size of any cover you have actually built at radius \(c2^{-k}\), such as the grid, instead of the true (smallest) covering number. The bound gets a little looser and stays valid.

Here is the bound for the ten-thousand-message filter, with grid covers, for a few depths \(M\):

```python
import math
m, d = 10_000, 2
c = math.sqrt(m)
N = lambda r: math.ceil(c * math.sqrt(d) / r) ** d
def chain(M):
    terms = [2 ** -k * math.sqrt(math.log(N(c * 2 ** -k)))
             for k in range(1, M + 1)]
    return c * 2 ** -M / math.sqrt(m) + 6 * c / m * sum(terms)
for M in (1, 2, 4, 8, 16):
    print(M, round(chain(M), 4))
```

It prints `1 0.5445`, `2 0.3229`, `4 0.1615`, `8 0.1132` and `16 0.1102`. Going deeper helps a lot at first, then barely at all: the terms of the sum shrink like \(2^{-k}\) while their logarithms grow only slowly.

## How deep should the chain go?

The table suggests a question: is there a right depth \(M\), or should you always go deeper? Let’s compute the change from depth \(M-1\) to depth \(M\). Write \(f(M)\) for the right side of the chaining bound and \(N_M = N(c2^{-M}, A)\). Going one level deeper halves the rounding term, which saves \(c2^{-(M-1)}/\sqrt m - c2^{-M}/\sqrt m = c 2^{-M}/\sqrt m\), and adds one term to the sum. So

\[ f(M) - f(M-1) = \frac{c\,2^{-M}}{\sqrt m}\Big( \frac{6\sqrt{\log N_M}}{\sqrt m} - 1 \Big). \]

The factor in front is positive, so level \(M\) helps exactly when \(6 \sqrt{\log N_M} \le \sqrt m\). In plain words: **keep refining while the log of the next cover’s size is below \(m/36\).** Covering numbers grow as the radius shrinks, so once a level fails the test every deeper level fails it too; the best depth is the number of levels that pass. With the grid covers in two dimensions, level sizes are \(9, 36, 144, \dots\), independent of \(m\):

```python
import math
d = 2
N_k = lambda k: math.ceil(2 ** k * math.sqrt(d)) ** d
def best_M(m):
    k = 0
    while 6 * math.sqrt(math.log(N_k(k + 1))) <= math.sqrt(m):
        k += 1
    return k
for m in (100, 400, 1_000, 10_000):
    print(m, best_M(m))
```

It prints `100 1`, `400 7`, `1000 19` and `10000 199`. With only 100 labels, one level is all the theory can pay for; with ten thousand, it pays for far more levels than make any visible difference. This rule is not a heuristic or a matter of taste: it is the derivative of the bound you are certifying, so it is the scale choice the theory itself justifies.

## Letting the chain run forever

When the covering numbers grow in a controlled way, we can let \(M\) go to infinity and get a clean formula.

**Corollary (infinite chain).** Suppose there are numbers \(\alpha, \beta > 0\) with \(\sqrt{\log N(c2^{-k}, A)} \le \alpha + \beta k\) for every \(k \ge 1\). Then

\[ R(A) \le \frac{6c}{m}\,(\alpha + 2\beta). \]

The assumption says the square root of the log covering number grows at most linearly in the number of halvings \(k\). In plain words: **if each halving of the radius costs at most a fixed amount more in \(\sqrt{\log N}\), the whole infinite chain costs \(6c/m\) times the starting cost plus twice the step cost.**

*Proof.* In the chaining bound, replace each \(\sqrt{\log N}\) by \(\alpha + \beta k\) to get \(R(A) \le c 2^{-M}/\sqrt m + (6c/m)\sum_{k=1}^M 2^{-k}(\alpha + \beta k)\) for every \(M\). As \(M \to \infty\), the first term goes to 0 and the sum increases to \(\alpha \sum_{k\ge 1} 2^{-k} + \beta \sum_{k \ge 1} k 2^{-k}\). The first series is the geometric series, \(1/2 + 1/4 + \dots = 1\). For the second, call it \(T\); then \(T - T/2 = \sum_k k2^{-k} - \sum_k k 2^{-(k+1)} = \sum_{k\ge1} 2^{-k} = 1\), so \(T = 2\). \(\blacksquare\)

```python
geometric = sum(2 ** -k for k in range(1, 60))
weighted = sum(k * 2 ** -k for k in range(1, 60))
print(round(geometric, 6), round(weighted, 6))
```

It prints `1.0 2.0`.

**Example (the flat filter).** Let \(A\) lie in a \(d\)-dimensional subspace with every \(\|a\| \le c\) (so the center \(0\) works). At level \(k\) the radius \(c 2^{-k}\) is at most \(c/2 \le c\sqrt d\), so the flat-set proposition gives \(N(c2^{-k}, A) \le (2c\sqrt d / (c 2^{-k}))^d = (2^{k+1}\sqrt d)^d\). Taking logs, \(\log N \le d \log(2\sqrt d) + d k \log 2\). Now use \(\sqrt{x + y} \le \sqrt x + \sqrt y\) (square both sides to check), then \(\log 2 < 1\) and \(\sqrt k \le k\):

\[ \sqrt{\log N(c2^{-k}, A)} \le \sqrt{d \log(2\sqrt d)} + \sqrt{d k \log 2} \le \sqrt{d\log(2\sqrt d)} + \sqrt d \, k. \]

So the corollary applies with \(\alpha = \sqrt{d \log(2 \sqrt d)}\) and \(\beta = \sqrt d\):

\[ R(A) \le \frac{6c}{m}\Big(\sqrt{d \log (2\sqrt d)} + 2 \sqrt d\Big). \]

In plain words: **a flat class of dimension \(d\) and size \(c\) fits noise no better than about \(c\sqrt{d \log d}/m\), with no dependence on anything else.** For your filter, \(d = 2\) and \(c = \sqrt m\), so \(R(A) \le 25.62/\sqrt m\); at \(m = 10{,}000\) that is \(0.2562\).

```python
import math
d = 2
alpha = math.sqrt(d * math.log(2 * math.sqrt(d)))
beta = math.sqrt(d)
N_k = lambda k: (2 ** (k + 1) * math.sqrt(d)) ** d
ok = all(math.sqrt(math.log(N_k(k))) <= alpha + beta * k
         for k in range(1, 60))
const = 6 * (alpha + 2 * beta)
print(round(alpha, 3), round(beta, 3), ok, round(const, 2))
print(round(const / math.sqrt(10_000), 4))
```

It prints `1.442 1.414 True 25.62` and `0.2562`. Notice that the closed form, \(0.2562\), is more than twice the \(0.1102\) we got by plugging the actual grid sizes into the chaining bound. Every “\(\le\)” in the example rounded up, and the rounding adds up. When you need a number, use the sizes of the covers you built; when you need a formula, use the corollary.

## Why bother with many scales?

Here is an honest comparison, because you deserve one. Put the one-scale bound (best radius) and the infinite chain side by side, both multiplied by \(\sqrt m\) so that a constant means “goes down like \(1/\sqrt m\)”.

```python
import math
d = 2
def one_scale(m):
    c = math.sqrt(m)
    N = lambda r: math.ceil(c * math.sqrt(d) / r) ** d
    f = lambda r: r + (1 + r / c) * math.sqrt(2 * math.log(N(r)))
    return min(f(i / 100) for i in range(1, 400))
def chained():
    N_k = lambda k: math.ceil(2 ** k * math.sqrt(d)) ** d
    return 6 * sum(2 ** -k * math.sqrt(math.log(N_k(k)))
                   for k in range(1, 200))
for m in (10 ** 2, 10 ** 4, 10 ** 6, 10 ** 12):
    print(m, round(one_scale(m), 2), round(chained(), 2))
```

It prints `100 4.33 11.02`, `10000 5.26 11.02`, `1000000 6.12 11.02` and `1000000000000 8.14 11.02`.

Two things are true at once. At every label budget you will ever see, the one-scale number is smaller: chaining pays a factor 6 in constants. And the one-scale number keeps growing with \(m\), like \(\sqrt{\log m}\), while the chained number does not move. So chaining buys a different kind of statement: one constant \(C\) with \(R(A) \le C/\sqrt m\) for every \(m\), which the one-scale argument can never give. That is what “sharper” means here: sharper in how the bound depends on \(m\), not smaller at your budget. For this flat class it is a log factor; for classes whose covering numbers grow faster as the radius shrinks, the gap between the two arguments becomes a gap in the rate itself.

The review board’s rule, it turns out, is written for exactly that kind of statement: the company keeps buying labels, and a certificate must not get worse as the archive grows. So the board asks for the chaining bound.

## Lab act 1: covers you can see

Before the formal fix, play with covers in the plane. The lab shows the score set of a two-feature filter as a disc in the slice’s two coordinates.

Predict first: if you halve the radius of a grid cover of a disc, by about what factor does the number of representatives grow? (Write your guess down; the lab asks for it before it unlocks.)

[[lab]]

## The algorithms

Three small procedures carry this lesson: building a grid cover, checking that a set really is a cover, and computing the chaining bound at the depth the theory justifies.

```
GridCover(c, d, r):
  n ← ceil(c · sqrt(d) / r)
  w ← 2c / n
  axis ← [−c + w/2 + j·w for j = 0..n−1]
  return every d-tuple of values from axis

CoveringRadius(A, B):
  worst ← 0
  for a in A:
    worst ← max(worst, min over b in B of ‖a − b‖)
  return worst

ChainingBound(m, c, sizes):
  M ← len(sizes)
  s ← sum over k = 1..M of 2^(−k) · sqrt(log sizes[k])
  return c · 2^(−M) / sqrt(m) + (6c / m) · s

BestDepth(m, sizes):
  k ← 0
  while k < len(sizes) and 6 · sqrt(log sizes[k+1]) ≤ sqrt(m):
    k ← k + 1
  return k
```

**GridCover** produces \(n^d\) points of \(d\) coordinates each, so it takes \(O(d\, n^d)\) time and space with \(n = \lceil c\sqrt d / r\rceil\). For the chain you rarely need the points themselves, only their number \(n^d\) (\(O(1)\) arithmetic) and, for any one vector, its nearest grid point, which is a rounding of each coordinate: \(O(d)\) time, no storage. That is why a grid at level 199 costs nothing to use even though it has astronomically many points.

**CoveringRadius** compares every sample vector with every representative: \(O(|A|\,|B|\,d)\) time and \(O(1)\) extra space. It can only check a finite sample of an infinite set, so it is a test, not a proof; the proof for the grid is the flat-set proposition. With a grid, the nearest representative is a rounding, so the check drops to \(O(|A|\, d)\).

**ChainingBound** and **BestDepth** take \(O(M)\) arithmetic operations given the sizes, and \(O(M)\) space to hold them.

**Build it yourself.** Using only the Python standard library:

```python
import math

A = [(0.9, 0.8), (0.7, 0.9), (-0.8, -0.6),
     (-0.9, -0.9), (0.1, -0.2), (-0.2, 0.1)]
B = [(0.8, 0.85), (-0.85, -0.75), (0.0, 0.0)]

def grid_cover(c, d, r): ...
def covering_radius(A, B): ...
def chaining_bound(m, c, sizes): ...
def best_M(m, sizes): ...

if __name__ == "__main__":
    assert len(grid_cover(math.sqrt(2), 2, 0.6)) == 16
    assert round(covering_radius(A, B), 3) == 0.224
    m = 10_000
    c = math.sqrt(m)
    sizes = [len(grid_cover(c, 2, c * 2 ** -k))
             for k in range(1, 9)]
    assert sizes[:3] == [9, 36, 144]
    assert round(chaining_bound(m, c, sizes), 4) == 0.1132
    assert best_M(1_000, sizes) == 8
    assert best_M(100, sizes) == 1
```

1. Implement the four functions. `sizes[0]` is the size of the level-1 cover; `best_M` must never go deeper than the sizes you give it (that is why it returns 8, not 19, for \(m = 1{,}000\)).
2. Reproduce the depth table from “The chaining bound, proved”.
3. Replace `len(grid_cover(...))` by the formula \(\lceil 2^k\sqrt d\rceil^d\) and push the depth to 200 levels. Confirm the bound settles at about \(11.02/\sqrt m\).
4. Repeat for \(d = 3\). Predict the settled constant from the corollary first, then compare it with what the actual grid sizes give.

## Lab act 2: certify the filter

Now the real job. The board’s rule, from lesson 26: with a loss between 0 and 1, with probability at least \(1 - \delta\) over the labelled messages, every hypothesis in the class has true risk at most its empirical risk plus

\[ 2\,R(\text{loss vectors}) + \sqrt{\frac{2\log(2/\delta)}{m}}. \]

The ramp loss is 1-Lipschitz, so by lesson 26’s contraction lemma the loss vectors fit noise no better than the score vectors, and any bound on \(R(A)\) for the score set serves. To ship, the gap must be at most \(\varepsilon = 0.25\) with \(\delta = 0.05\), using the chaining bound, and you can buy at most 12,000 labels.

In the lab you choose the number of labels \(m\), and you build the chain: for each level, a grid cover at the radius you pick. The lab checks every claimed cover against its radius and shows two numbers side by side. One is a **measured** Rademacher complexity, estimated from random sign vectors. It looks wonderful, and it is not a certificate: it is a random estimate of the thing you were asked to bound. The other is the chaining bound computed from your covers, and only that one counts.

Before you start, work it out on paper. The closed formula from the flat-filter example says \(R(A) \le 25.62/\sqrt m\). Is 12,000 labels enough with that formula? And what does the chaining bound with the actual grid sizes, at the depth the theory justifies, need?

```python
import math
d, delta, target = 2, 0.05, 0.25
N_k = lambda k: math.ceil(2 ** k * math.sqrt(d)) ** d
link = lambda k: 2 ** -k * math.sqrt(math.log(N_k(k)))
def chain(m):
    total, k = 1.0, 0
    while 6 * math.sqrt(math.log(N_k(k + 1))) <= math.sqrt(m):
        k += 1
        total += -2 ** -k + 6 / math.sqrt(m) * link(k)
    return total
conf = lambda m: math.sqrt(2 * math.log(2 / delta) / m)
gap = lambda m: 2 * chain(m) + conf(m)
m = next(m for m in range(100, 10 ** 6, 100) if gap(m) <= target)
print(m, round(chain(m), 4), round(gap(m), 4))
```

It prints `9900 0.1107 0.2488`: with the actual grid sizes at the best depth, 9,900 labels certify a gap of \(0.2488\). The closed formula, by contrast, would need 46,600 labels, far over budget. Shallow chains fail too: at 12,000 labels, a chain of 6 levels still leaves the gap at \(0.2514\).

[[lab-act2]]

## What people get wrong

**“A cover has to be made of members of the class.”** It doesn’t. Representatives can be any vectors; the middle representative of the clump example was not a score vector, and the grid points of the disc mostly aren’t. (If you insist on representatives from the set itself, you may need more of them, but the definition does not ask for it.)

**“Covering numbers depend on the number of messages.”** For a flat set, they don’t: a two-dimensional slice of \(\mathbb R^{10{,}000}\) of size \(c\) needs the same number of representatives at radius \(r\) as a disc of radius \(c\) in the plane. What grows with \(m\) is \(c\), which is why the chain measures radii in units of \(c\).

**“Chaining always gives a smaller number than one cover.”** Not at the budgets in this lesson; the one-scale bound was smaller every time. Chaining wins on shape: its bound does not grow with \(m\).

**“The measured Rademacher complexity is good enough.”** A random estimate of a quantity is not a bound on it. The certificate is a proof that holds for every sign vector, averaged exactly; that is what the chain gives you without ever enumerating the \(2^m\) sign vectors.

## Recap and the next bug

1. An \(r\)-cover of a set of score vectors is a set of representatives with every vector within \(r\) of one; the covering number \(N(r, A)\) is the size of the smallest.
2. A flat set of dimension \(d\) and size \(c\) has \(N(r, A) \le \lceil c\sqrt d / r\rceil^d\), built by a grid, independent of \(m\).
3. Stretching a set by \(s\) multiplies the radius by \(s\) at the same count; shifting changes nothing; a \(\rho\)-Lipschitz loss multiplies the radius by at most \(\rho\).
4. One cover plus Massart’s lemma bounds the Rademacher complexity of an infinite set.
5. Chaining walks from a single point down through halving scales; each link is short and comes from a small set, which gives \(R(A) \le c2^{-M}/\sqrt m + (6c/m)\sum_k 2^{-k}\sqrt{\log N(c2^{-k}, A)}\). Refine while \(6\sqrt{\log N} \le \sqrt m\). For a flat filter it gives a bound of order \(c\sqrt{d\log d}/m\) that never grows with \(m\).

The certificate is signed. Then, at the review, someone from the research team leans back and asks a question that has been waiting since lesson 6. That lesson stated the fundamental theorem of learning: a class is learnable exactly when its VC dimension is finite, with sample sizes of a precise order, and it promised a full proof later. You now hold every tool that proof needs: uniform convergence, Rademacher complexity, Massart’s lemma, and as of today, covers. So can you prove it, both directions, with the exact rates? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Let \(A\) be a finite set of \(N\) vectors, all within \(c\) of the origin. What is \(N(r, A)\) for very small \(r\)? Use it to say what the chaining bound gives as \(M \to \infty\), and compare it with Massart’s lemma applied to \(A\) directly. Which is smaller, and why does that not contradict anything?
2. In the chaining proof, the set of links \(\hat B_k\) collects differences \(b - b'\) with \(b \in B_k\) and \(b' \in B_{k-1}\). Why does the proof need \(b^{(k)} - b^{(k-1)}\) to lie in a set fixed in advance, rather than just knowing that the link is short? And where exactly does the factor 6 come from?
3. A three-feature filter (\(d = 3\)) with scores in \([-1, 1]\) has \(c = \sqrt m\). Using the flat-filter example’s closed formula, what constant \(C\) gives \(R(A) \le C/\sqrt m\), and how many labels make that bound at most \(0.1\)?

## Problems

These three problems belong to this lesson; hints come one step at a time when you ask.

1. **Covers can’t be much smaller than the grid.** Let \(A\) be the full disc of radius \(c\) in a \(d\)-dimensional subspace. Prove that \(N(r, A) \ge (c/r)^d\). How far apart are this lower bound and the grid’s upper bound \(\lceil c\sqrt d/r\rceil^d\) when \(d = 2\)?
2. **Covering the losses directly.** Use the Lipschitz-loss lemma and the flat-set proposition to bound the covering numbers of the ramp-loss vectors of the two-feature filter, then apply the chaining bound to the loss vectors themselves (centered at the vector of all \(1/2\)). Compare the resulting gap at \(m = 12{,}000\) with the one from the lab.
3. **A one-feature filter.** For \(d = 1\), compute \(\alpha\), \(\beta\) and the constant \(6(\alpha + 2\beta)\) of the flat-filter example, and the number of labels needed to certify a gap of \(0.25\) at \(\delta = 0.05\) with the closed formula. Then find the true covering numbers of a segment of length \(2c\) and redo the computation with them.
