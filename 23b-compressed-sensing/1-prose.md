# Compressed Sensing

## Smeared words

Lesson 23a ended with a fast filter and a quiet week. Then the compliance desk wrote in. A customer is disputing a quarantined message, and the desk has to show exactly which watchlist words it contained, and how often. The desk has nothing to go on but what the archive keeps: the squeezed numbers.

Let’s rebuild their problem on a small scale. The watchlist has six words, “free”, “prize”, “winner”, “urgent”, “claim” and “wire”, and every quarantined message uses one of them some number of times. The desk squeezes each message to three numbers, \(y = Wx\), where \(W\) is a **compression matrix**: a table with three rows and six columns, so that \(y\) holds three weighted sums of the six counts. (A message with six counts is a vector \(x \in \mathbb R^6\), and \(\mathbb R^{3,6}\) is the set of tables with three rows and six columns.) The columns of \(W\) are six unit-length directions from the centre of an icosahedron, the twenty-sided solid, to six of its corners, one from each opposite pair. They are as spread apart as six directions in three dimensions can be, and the next sections look at this matrix properly.

The desk engineer’s decoder is the one lesson 23a would suggest: among all messages that squeeze to the stored \(y\), return the shortest. For this \(W\) the shortest is \(\tilde x = \tfrac12 W^\top y\). The \(W^\top\) is the transpose of \(W\), the same table turned on its side so that rows become columns, and the \(\tfrac12\) is there because each of the three rows of \(W\) has squared length 2. The recovered message is \(\tilde x\), as before. Take a message that says “urgent” two and a half times and nothing else.

```python
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
words = ["free", "prize", "winner", "urgent", "claim", "wire"]
x = [0, 0, 0, 2.5, 0, 0]
y = [sum(cols[j][r] * x[j] for j in range(6)) for r in range(3)]
x_back = [sum(cols[j][r] * y[r] for r in range(3)) / 2
          for j in range(6)]
back = [sum(cols[j][r] * x_back[j] for j in range(6))
        for r in range(3)]
print([round(t, 3) for t in y])
for w, t in zip(words, x_back):
    print(w, round(t, 3))
print(max(abs(a - b) for a, b in zip(back, y)) < 1e-9)
```

It prints

```
[-1.314, 2.127, 0.0]
free 0.559
prize -0.559
winner 0.559
urgent 1.25
claim -0.559
wire -0.559
True
```

The stored numbers are three, as promised. The recovery claims that the message used all six watchlist words, with “urgent” at half its true count and the other five at plus or minus 0.559, some of them negative. And the last line says the decoder is not even careless: squeezing the recovered message gives back exactly the stored numbers, so the decoder agrees with the data it was given. It is still wrong, and the audit trail would say that a customer sent six words, three of them with a negative count.

[[diagram:smear]]

You might hope for a better recovery matrix. There is none, and the reason is the null space of lesson 23a. Suppose some matrix \(U\) recovered every one-word message exactly, so that \(UWx = x\) whenever \(x\) has a single nonzero count. The one-word messages include the six messages \(e_1, \dots, e_6\), each with a single count of 1, and every message is a sum of multiples of those. A matrix respects sums and scalings, so \(UWx = x\) would then hold for every message \(x\) at all. But with \(n < d\) some nonzero \(v\) has \(Wv = 0\), and then \(UWv = U0 = 0 \neq v\). Here is such a \(v\) for the desk’s matrix, where the printed numbers are the largest entry of \(Wv\) and the entries of \(UWv\):

```python
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
v = [1, -phi, -phi, 0, 1, 0]
Wv = [sum(cols[j][r] * v[j] for j in range(6)) for r in range(3)]
UWv = [sum(cols[j][r] * Wv[r] for r in range(3)) / 2
       for j in range(6)]
print(round(max(abs(t) for t in Wv), 9))
print([round(abs(t), 9) for t in UWv])
```

It prints `0.0` and `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`: the vector \(v\) is invisible after squeezing, and no matrix can bring it back. In one sentence: **a recovery matrix that is right on every one-word message would be right on every message, and the null space forbids that, so the decoder cannot be a matrix.**

Yet the desk’s messages are not arbitrary vectors. They use one word out of six, and in the real mail a message uses about 60 of 50,000. A decoder that uses that fact can be something other than a matrix, and that is the idea of this lesson. Three questions come one after another. Is a message that uses few words determined by its squeezed numbers at all? Can it be found without trying every possible set of words? And which squeezes make both true, and how can you be sure, rather than hope? The answers are called **compressed sensing**.

## What you need from lesson 23a

Six things carry over, each with a number.

- **Squeezing.** A message is a vector \(x \in \mathbb R^d\) of word counts, and a compression matrix \(W \in \mathbb R^{n,d}\) squeezes it into \(y = Wx\), the \(n\) weighted sums of its counts. With the desk’s six words and three numbers, \(d = 6\) and \(n = 3\).
- **Length.** \(\|v\|\) is the Euclidean length \(\sqrt{v_1^2 + \dots + v_d^2}\), and \(\|v\|^2\) is the sum of the squared entries. The vector \((3, 4)\) has length 5.
- **The null space.** With \(n < d\), some nonzero \(v\) has \(Wv = 0\), and then \(x\) and \(x + v\) squeeze to the same \(y\). In lesson 23a, \((3, 2, 0)\) and \((2, 3, 0)\) both became \((5, 0)\), because \(v = (1, -1, 0)\) is invisible. So exact recovery of every message is impossible.
- **Orthonormal.** Vectors are orthonormal when each has length 1 and every two are perpendicular. If the columns of a matrix \(U\) are orthonormal, then \(U\alpha\) has the same length as \(\alpha\): the matrix with columns \((1, 0)\) and \((0, 1)\) sends \((3, 4)\) to itself, and a turn of the axes sends it to another vector of length 5.
- **The union bound** (lesson 2). The chance that at least one of several bad events happens is at most the sum of their chances: two events of chance 0.01 each leave at most 0.02.
- **Random projections and the Johnson-Lindenstrauss lemma.** A random projection is the squeeze \(x \mapsto Wx\) where every entry of \(W\) is drawn independently from \(N(0, 1/n)\), a bell curve with mean 0 and variance \(1/n\). The lemma says: let \(Q\) be a finite set of nonzero vectors fixed before \(W\) is drawn, let \(\delta \in (0, 1)\), and let \(n\) be such that \(\varepsilon = \sqrt{6 \ln(2|Q|/\delta)/n} \le \tfrac12\). Then with probability at least \(1 - \delta\),

\[ \sup_{x \in Q} \left| \frac{\|Wx\|^2}{\|x\|^2} - 1 \right| \le \varepsilon . \]

Read it from the inside out. The fraction \(\|Wx\|^2/\|x\|^2\) is how much the squeeze stretches the squared length of \(x\); subtracting 1 and taking the absolute value \(|\cdot|\) measures how far that is from a perfect 1; \(\sup_{x \in Q}\) takes the largest of those over all the vectors in \(Q\); \(|Q|\) is how many vectors there are, \(\ln\) is the natural logarithm, and \(\delta\) is the chance you accept for the whole statement to fail. In one sentence: **with probability at least \(1 - \delta\), a random projection to \(n\) numbers keeps every squared length in a fixed finite set within a fraction \(\varepsilon\), whatever \(d\) is.** Lesson 23a’s number: \(|Q| = 19{,}900\), \(\delta = 0.05\), \(n = 906\).

```python
import math
Q, delta, n = 19900, 0.05, 906
eps = math.sqrt(6 * math.log(2 * Q / delta) / n)
print(round(eps, 3))
```

It prints `0.3`, the accuracy that 906 numbers buy for 19,900 distances.

## Sparse messages

The desk’s smear came from the null space: with \(n < d\), no recovery matrix can undo a squeeze. Yet look at an actual message. Out of 50,000 word counts, a typical message has maybe 60 that are not zero. Writing down 50,000 numbers to record 60 of them is absurd.

The count of nonzero entries has a name. For \(x \in \mathbb R^d\),

\[ \|x\|_0 = \big|\{\, i : x_i \neq 0 \,\}\big| . \]

- \(\{\, i : x_i \ne 0\,\}\) is the set of positions \(i\) such that entry \(i\) is not zero: the words the message actually uses.
- \(|\cdot|\) counts how many positions that set contains.
- The subscript 0 is a convention: \(\|x\|_0\) is not a true length (doubling \(x\) does not double it), but it sits naturally next to \(\|x\|_1\) and \(\|x\|_2\), which we meet in a moment.

In one sentence: **\(\|x\|_0\) is the number of distinct words in the message.** A message is **s-sparse** when \(\|x\|_0 \le s\). The message “free free free prize winner winner” has counts 3, 1 and 2 on three words and 0 on the other 49,997:

```python
x = {"free": 3, "prize": 1, "winner": 2}
d = 50000
print(len(x), d - len(x))
```

It prints `3 49997`: \(\|x\|_0 = 3\), so this message is 3-sparse, and so is any message using at most three words.

An s-sparse message can obviously be stored without loss as \(s\) pairs (word index, count). So why not do that? Because that representation is not a fixed linear squeeze: you have to see the whole message first to know which positions to keep. **Compressed sensing** asks for something stranger. Fix one matrix \(W\) with \(n\) rows, the same for every message, chosen before any message exists. Store \(y = Wx\). Then recover every s-sparse \(x\) exactly. A linear squeeze can be computed on the fly: as the parser reads each word, it adds that word’s column of \(W\) to a running total of \(n\) numbers, and the 50,000-count vector is never built. This is also why the idea matters outside the mail room: a camera or a medical scanner can measure the \(n\) weighted sums directly, instead of measuring everything and throwing most of it away. The same works for a message that is sparse after a fixed change of basis, \(x = U\alpha\) with \(U\) orthonormal and \(\alpha\) sparse, since \(Wx = (WU)\alpha\).

How can that work, when we proved that every squeeze has a null space? Because exact recovery of *every* message was impossible, but we only ask for sparse ones. The null space contains no sparse vectors, if \(W\) is chosen well, and the decoder that finds the sparse message is not a matrix, as the desk learned, but it can exist. Three results make this precise:

1. If \(W\) has a property called the restricted isometry property, every sparse message can be recovered exactly from \(Wx\).
2. The recovery can be computed in polynomial time, by a linear program.
3. A random \(W\) has the property, with high probability, once \(n\) is of order \(s \ln d\).

[[diagram:sparse]]

## The restricted isometry property

The idea first. A random projection keeps the lengths of a finite set of vectors. Compressed sensing needs more: it needs \(W\) to keep the length of *every* sparse vector, infinitely many of them. If it does, then no sparse vector is squeezed to zero, and that is exactly what recovery needs.

**Definition (restricted isometry property).** A matrix \(W \in \mathbb R^{n,d}\) is \((\varepsilon, s)\)-RIP if for every \(x \neq 0\) with \(\|x\|_0 \le s\),

\[ \left| \frac{\|Wx\|^2}{\|x\|^2} - 1 \right| \le \varepsilon . \]

- “\(x \ne 0\) with \(\|x\|_0 \le s\)” is every nonzero s-sparse vector.
- The fraction is the squared distortion of \(x\), exactly as in the Johnson-Lindenstrauss lemma.
- An **isometry** keeps all lengths; “restricted” means we only ask for it on sparse vectors, and only up to a fraction \(\varepsilon\).

In one sentence: **\(W\) is \((\varepsilon, s)\)-RIP when it gets the squared length of every s-sparse vector right within a fraction \(\varepsilon\).** The smallest such \(\varepsilon\) for a given \(W\) and \(s\) is its **RIP constant**.

A tiny example makes it concrete, and it is the desk’s matrix from the opening: a vocabulary of six words and \(n = 3\). The columns of \(W\) are the six unit vectors that point at opposite vertex pairs of a regular icosahedron, \((0, \pm 1, \varphi)\), \((\pm 1, \varphi, 0)\), \((\varphi, 0, \pm 1)\), divided by their length \(\sqrt{1 + \varphi^2}\), where \(\varphi = (1 + \sqrt 5)/2\) is the golden ratio. Any two of these columns make the same angle, with \(|\cos| = 1/\sqrt 5 \approx 0.447\). For a 2-sparse \(x\) using words \(i\) and \(j\), \(\|Wx\|^2 = x^\top G x\) where \(G\) is the \(2 \times 2\) table of dot products of columns \(i\) and \(j\), so the distortion is bounded by the two eigenvalues of \(G\), here \(1 \pm 0.447\). Checking all 15 pairs:

```python
from itertools import combinations
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
worst = 0.0
for i, j in combinations(range(6), 2):
    a, b, c = dot(cols[i], cols[i]), dot(cols[i], cols[j]), \
        dot(cols[j], cols[j])
    mid, gap = (a + c) / 2, (((a - c) / 2) ** 2 + b * b) ** 0.5
    worst = max(worst, abs(mid + gap - 1), abs(mid - gap - 1))
print(round(worst, 3), round(1 / (1 + 2 ** 0.5), 3))
```

It prints `0.447 0.414`. So this \(3 \times 6\) matrix is \((0.447, 2)\)-RIP. (The second number, \(1/(1 + \sqrt 2)\), will matter in two sections.) Keep this example in mind: six words, three stored numbers, and every message that uses one word.

**Theorem (sparse messages are determined by their squeeze).** Let \(\varepsilon < 1\) and let \(W\) be \((\varepsilon, 2s)\)-RIP. Let \(x\) be s-sparse, let \(y = Wx\), and let

\[ \tilde x \in \operatorname*{argmin}_{v \,:\, Wv = y} \|v\|_0 \]

be a reconstruction. Then \(\tilde x = x\).

- \(v : Wv = y\) ranges over every vector that squeezes to the stored \(y\).
- \(\operatorname{argmin} \|v\|_0\) picks one of them with the fewest nonzero entries; the \(\in\) says there may be several tied, and \(\tilde x\) is any one of them.

In one sentence: **if \(W\) is RIP for twice the sparsity, the sparsest message that matches the stored numbers is the message itself.** Note the “twice”: the proof compares two s-sparse vectors, and their difference can use \(2s\) words.

*Proof.* Suppose, for contradiction, that \(\tilde x \ne x\). The true \(x\) itself satisfies \(Wx = y\), so it is one of the candidates, and the sparsest candidate can be no less sparse: \(\|\tilde x\|_0 \le \|x\|_0 \le s\). The difference \(x - \tilde x\) is nonzero only where \(x\) or \(\tilde x\) is, so \(\|x - \tilde x\|_0 \le 2s\), and the RIP inequality applies to it. But \(W(x - \tilde x) = y - y = 0\), so its squared distortion ratio is 0, and the RIP inequality reads \(|0 - 1| \le \varepsilon\), that is, \(1 \le \varepsilon\). This contradicts \(\varepsilon < 1\). \(\blacksquare\)

In the icosahedron example, \(s = 1\) and \(W\) is \((0.447, 2)\)-RIP, so every 1-sparse message is determined by its three stored numbers. Here is the sparsest-candidate search for the message “2.5 times word 3”: for a 1-sparse candidate, \(y\) must be a multiple of a single column, so the code tries each column.

```python
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
x = [0, 0, 0, 2.5, 0, 0]
y = [sum(cols[j][r] * x[j] for j in range(6)) for r in range(3)]
for j, w in enumerate(cols):
    c = dot(y, w)
    if max(abs(a - c * b) for a, b in zip(y, w)) < 1e-9:
        print("word", j, "count", round(c, 6))
print([round(t, 3) for t in y])
```

It prints `word 3 count 2.5` and the stored numbers `[-1.314, 2.127, 0.0]`. Exactly one column fits, so exactly one 1-sparse message squeezes to these three numbers, as the theorem says.

The catch is the search. For six words and \(s = 1\) there are six candidate supports. For 50,000 words and messages of 60 words, the number of possible sets of 60 words is \(\binom{50000}{60}\) (“50,000 choose 60”, the number of ways to pick 60 positions out of 50,000), about \(1.0 \times 10^{200}\). Minimizing \(\|v\|_0\) is a combinatorial search, and no computer will finish it.

```python
import math
print(math.comb(6, 1), f"{math.comb(50000, 60):.1e}")
```

It prints `6 1.0e+200`: six supports to try in the tiny example, a one followed by two hundred zeros in the mail room.

## Recovery by a linear program

Here is the surprising part. Replace the count of nonzero entries by the sum of absolute values,

\[ \|v\|_1 = \sum_{i=1}^d |v_i| , \]

the **\(\ell_1\) norm** of \(v\). (For the message above with counts 3, 1 and 2, \(\|x\|_1 = 6\); for comparison \(\|x\|_0 = 3\) and the ordinary length is \(\|x\|_2 = \|x\| = \sqrt{14}\).) In one sentence: **the \(\ell_1\) norm is the total of the counts, ignoring signs.** Minimizing it over all \(v\) with \(Wv = y\) is a convex problem, and in fact a **linear program**: introduce one extra variable \(t_i\) per word and solve

\[ \min_{v,\, t \in \mathbb R^d} \; \sum_{i=1}^d t_i \quad\text{such that}\quad Wv = y,\; -t_i \le v_i \le t_i \text{ for every } i . \]

At the optimum each \(t_i\) is pushed down to \(|v_i|\), so the objective is \(\|v\|_1\): for \(v = (3, -1, 2)\) the smallest allowed \(t\) is \((3, 1, 2)\), total 6. Linear programs can be solved in polynomial time (lesson 9a used one to find a separating halfspace). The question is whether the \(\ell_1\) minimizer is still the true message. It is, under a slightly stronger RIP, and we can even say what happens for messages that are only nearly sparse.

For an arbitrary \(x\), let \(x_s\) be the vector that keeps the \(s\) entries of \(x\) largest in absolute value and sets the rest to 0: the best s-sparse approximation of \(x\). Then \(\|x - x_s\|_1\) is the total of the small entries, how far \(x\) is from being s-sparse. For the message with counts \((3, 1, 2)\) and \(s = 2\), \(x_s\) keeps 3 and 2 and \(\|x - x_s\|_1 = 1\). Here are the three ways of measuring that message, and its small-entry total, term by term:

```python
import math
x = [3, 1, 2]
s = 2
l0 = sum(t != 0 for t in x)
l1 = sum(abs(t) for t in x)
l2 = math.sqrt(sum(t * t for t in x))
kept = sorted(x, key=abs, reverse=True)[:s]
tail = l1 - sum(abs(t) for t in kept)
print(l0, l1, round(l2, 3), tail)
```

It prints `3 6 3.742 1`: three words, a total of six counts, a length of \(\sqrt{14} \approx 3.742\), and a total of 1 outside the two biggest entries.

Before the theorem, play with the idea. In the lab you can pick a message on the desk’s six words and watch decoders at work: the shortest matching message that smeared “urgent”, and the one with the smallest \(\ell_1\) total. A second panel looks for two different messages that squeeze to the same numbers. A third draws, for three words and two stored numbers, all the messages that match one stored \(y\) as a line, and grows a ball around the origin until it touches that line. The ball that measures length with squares touches at a smeared point. The ball that measures the \(\ell_1\) total touches at a corner, where the message is sparse.

[[lab]]

Notice where the smallest-\(\ell_1\) message lands, and ask yourself why it keeps landing on an axis.


**Theorem (\(\ell_1\) recovery).** Let \(\varepsilon < \frac{1}{1 + \sqrt 2}\) and let \(W\) be \((\varepsilon, 2s)\)-RIP. Let \(x\) be an arbitrary vector, \(y = Wx\), and

\[ x^\star \in \operatorname*{argmin}_{v \,:\, Wv = y} \|v\|_1 . \]

Then

\[ \|x^\star - x\|_2 \le 2\, \frac{1 + \rho}{1 - \rho}\, s^{-1/2}\, \|x - x_s\|_1, \qquad \rho = \frac{\sqrt 2\, \varepsilon}{1 - \varepsilon}. \]

- \(x^\star\) is the output of the linear program: the matching vector with the smallest total.
- \(\|x^\star - x\|_2\) is the ordinary length of the recovery error.
- \(\rho\) is a number built from \(\varepsilon\); the condition \(\varepsilon < 1/(1+\sqrt 2)\) is exactly what makes \(\rho < 1\), since \(\sqrt 2 \varepsilon < 1 - \varepsilon\) rearranges to \(\varepsilon(1 + \sqrt 2) < 1\).
- \(s^{-1/2} = 1/\sqrt s\).

In one sentence: **the linear program recovers any message up to an error proportional to the part of it that is not s-sparse; in particular, it recovers every s-sparse message exactly**, because then \(x = x_s\) and the right-hand side is 0. With \(\varepsilon = 0.3\):

```python
eps = 0.3
rho = 2 ** 0.5 * eps / (1 - eps)
C = 2 * (1 + rho) / (1 - rho)
print(round(rho, 3), round(C, 2))
```

It prints `0.606 8.15`: a \((0.3, 2s)\)-RIP matrix recovers an almost-sparse message with error at most \(8.15/\sqrt s\) times its small-entry total. For \(s = 2\) and the \((3, 1, 2)\) message, that is at most \(8.15 \cdot 1/\sqrt 2 \approx 5.8\), and exactly 0 if the message had used only two words.

The proof is the longest in this lesson, so here is its plan. Write \(h = x^\star - x\) for the error. Split the \(d\) positions into blocks of \(s\). Show that the part of \(h\) outside the two most important blocks is small, using only the fact that \(x^\star\) has the smallest \(\ell_1\) norm (Claim 1). Show that the part inside those two blocks is small, using the RIP (Claim 2). Add.

**Notation.** For a vector \(v\) and a set of positions \(I\), \(v_I\) is the vector equal to \(v\) on \(I\) and 0 elsewhere. \([d] = \{1, \dots, d\}\) is the set of all positions, and \(I^c = [d] \setminus I\) is the set of positions not in \(I\). For \(v = (5, -1, 2)\) and \(I = \{1, 3\}\), \(v_I = (5, 0, 2)\) and \(v_{I^c} = (0, -1, 0)\). We assume \(d/s\) is a whole number to keep the blocks equal; nothing changes otherwise except that the last block is smaller.

**The blocks.** Let \(T_0\) be the \(s\) positions where \(|x_i|\) is largest (ties broken arbitrarily), so \(x_{T_0} = x_s\) and \(x_{T_0^c} = x - x_s\). Let \(T_1\) be the \(s\) positions outside \(T_0\) where \(|h_i|\) is largest, \(T_2\) the next \(s\) positions by \(|h_i|\) outside \(T_0 \cup T_1\), and so on until every position is used. Write \(T_{0,1} = T_0 \cup T_1\).

**Lemma (RIP makes disjoint sparse vectors nearly perpendicular).** Let \(W\) be \((\varepsilon, 2s)\)-RIP. For any two disjoint sets \(I, J\) of at most \(s\) positions each, and any vector \(u\),

\[ |\langle W u_I, W u_J \rangle| \le \varepsilon\, \|u_I\|\, \|u_J\| . \]

Here \(\langle a, b \rangle = a \cdot b\) is the dot product. Before the squeeze, \(u_I\) and \(u_J\) are exactly perpendicular, because they have no position in common. In one sentence: **after an RIP squeeze, vectors on disjoint sets of words stay nearly perpendicular, with a cosine at most \(\varepsilon\).**

*Proof.* Both sides scale the same way, so assume \(\|u_I\| = \|u_J\| = 1\). For any vectors, \(\langle a, b \rangle = \frac14 \big(\|a + b\|^2 - \|a - b\|^2\big)\) (expand both squares; the \(\|a\|^2\) and \(\|b\|^2\) terms cancel). With \(a = Wu_I\) and \(b = Wu_J\), and since \(u_I \pm u_J\) uses at most \(2s\) positions and has squared length \(\|u_I\|^2 + \|u_J\|^2 = 2\) (disjoint supports, so Pythagoras), the RIP gives \(2(1 - \varepsilon) \le \|W(u_I \pm u_J)\|^2 \le 2(1 + \varepsilon)\). So

\[ \langle Wu_I, Wu_J\rangle \le \tfrac14\big(2(1 + \varepsilon) - 2(1 - \varepsilon)\big) = \varepsilon, \qquad \langle Wu_I, Wu_J\rangle \ge \tfrac14\big(2(1 - \varepsilon) - 2(1 + \varepsilon)\big) = -\varepsilon. \]

\(\blacksquare\)

On the icosahedron matrix (\(s = 1\), \(\varepsilon = 0.447\)), every pair of distinct single-word vectors should have \(|\langle Wu_I, Wu_J\rangle| / (\|u_I\|\|u_J\|) \le 0.447\):

```python
import random
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
rng = random.Random(5)
ratio = 0.0
for _ in range(10000):
    i, j = rng.sample(range(6), 2)
    a, b = rng.gauss(0, 1), rng.gauss(0, 1)
    wa, wb = [a * t for t in cols[i]], [b * t for t in cols[j]]
    ratio = max(ratio, abs(dot(wa, wb)) / abs(a * b))
print(round(ratio, 3))
```

It prints `0.447`: the bound is met with equality, because these columns all meet at the same angle.

**The split.** By the triangle inequality, \(\|h\| = \|h_{T_{0,1}} + h_{T_{0,1}^c}\| \le \|h_{T_{0,1}}\| + \|h_{T_{0,1}^c}\|\). We will show

- Claim 1: \(\|h_{T_{0,1}^c}\| \le \|h_{T_0}\| + 2 s^{-1/2}\|x - x_s\|_1\);
- Claim 2: \(\|h_{T_{0,1}}\| \le \frac{2\rho}{1 - \rho}\, s^{-1/2} \|x - x_s\|_1\).

Granting them, and using \(\|h_{T_0}\| \le \|h_{T_{0,1}}\|\) (a vector restricted to fewer positions is no longer),

\[ \|h\| \le 2\|h_{T_{0,1}}\| + 2 s^{-1/2}\|x - x_s\|_1 \le \Big(\frac{4\rho}{1 - \rho} + 2\Big) s^{-1/2}\|x - x_s\|_1 = 2\,\frac{1 + \rho}{1 - \rho}\, s^{-1/2}\|x - x_s\|_1, \]

which is the theorem. (The first step uses Claim 1 and \(\|h_{T_0}\| \le \|h_{T_{0,1}}\|\), the second Claim 2, and the last puts \(4\rho + 2(1 - \rho) = 2(1 + \rho)\) over \(1 - \rho\).)

**Claim 1, the tail is small.** This uses no RIP at all, only that \(x^\star\) has the smallest \(\ell_1\) norm.

*Step 1a: each block is bounded by the previous one.* Take \(j \ge 2\). Every entry of \(h\) on \(T_j\) is at most, in absolute value, every entry on \(T_{j-1}\), because the blocks were filled in decreasing order. So the largest entry on \(T_j\) is at most the average absolute entry on \(T_{j-1}\): \(\|h_{T_j}\|_\infty \le \|h_{T_{j-1}}\|_1 / s\), where \(\|v\|_\infty = \max_i |v_i|\) is the largest absolute entry. A vector with at most \(s\) entries, each at most \(M\) in absolute value, has length at most \(\sqrt s\, M\). Together:

\[ \|h_{T_j}\| \le s^{1/2}\, \|h_{T_j}\|_\infty \le s^{-1/2}\, \|h_{T_{j-1}}\|_1 . \]

*Step 1b: add the blocks.* By the triangle inequality and Step 1a,

\[ \|h_{T_{0,1}^c}\| \le \sum_{j \ge 2} \|h_{T_j}\| \le s^{-1/2} \sum_{j \ge 1} \|h_{T_j}\|_1 = s^{-1/2}\, \|h_{T_0^c}\|_1 . \]

The last equality holds because \(T_1, T_2, \dots\) split \(T_0^c\). In one sentence: **the length of everything past the first two blocks is at most the \(\ell_1\) total outside \(T_0\), divided by \(\sqrt s\).** Here is this inequality tested on 10,000 random vectors with \(d = 12\), \(s = 3\) (the code takes the first three positions as \(T_0\) and sorts the rest into blocks); the printed number is the largest ratio of the left side to the right side and must be at most 1:

```python
import random
rng = random.Random(6)
d, s = 12, 3
norm2 = lambda v: sum(t * t for t in v) ** 0.5
worst = 0.0
for _ in range(10000):
    h = [rng.gauss(0, 1) for _ in range(d)]
    rest = sorted(h[s:], key=abs, reverse=True)
    blocks = [rest[k:k + s] for k in range(0, len(rest), s)]
    tail = norm2([t for b in blocks[1:] for t in b])
    l1 = sum(abs(t) for t in rest)
    worst = max(worst, tail / (l1 / s ** 0.5))
print(round(worst, 3))
```

It prints `0.453`.

*Step 1c: \(\ell_1\) minimality controls \(h\) outside \(T_0\).* Since \(x\) itself satisfies \(Wx = y\), the minimizer has \(\|x\|_1 \ge \|x^\star\|_1 = \|x + h\|_1\). Split the sum by \(T_0\) and use \(|a + b| \ge |a| - |b|\) on \(T_0\) and \(|a + b| \ge |b| - |a|\) on \(T_0^c\):

\[ \|x\|_1 \ge \sum_{i \in T_0} |x_i + h_i| + \sum_{i \in T_0^c} |x_i + h_i| \ge \|x_{T_0}\|_1 - \|h_{T_0}\|_1 + \|h_{T_0^c}\|_1 - \|x_{T_0^c}\|_1 . \]

The left side is \(\|x_{T_0}\|_1 + \|x_{T_0^c}\|_1\). Cancel \(\|x_{T_0}\|_1\) and rearrange:

\[ \|h_{T_0^c}\|_1 \le \|h_{T_0}\|_1 + 2\,\|x_{T_0^c}\|_1 = \|h_{T_0}\|_1 + 2\,\|x - x_s\|_1 . \]

In one sentence: **the error outside the big entries of \(x\) is at most the error on them, plus twice the small entries of \(x\).**

*Step 1d: combine.* Put Step 1c into Step 1b, and use \(\|h_{T_0}\|_1 \le \sqrt s\,\|h_{T_0}\|\) (Cauchy-Schwarz for a vector with \(s\) entries: \(\sum |v_i| \cdot 1 \le \sqrt{\sum v_i^2}\,\sqrt{s}\)):

\[ \|h_{T_{0,1}^c}\| \le s^{-1/2}\big(\|h_{T_0}\|_1 + 2\|x - x_s\|_1\big) \le \|h_{T_0}\| + 2 s^{-1/2}\|x - x_s\|_1 . \]

That is Claim 1.

**Claim 2, the head is small.** Now the RIP enters.

*Step 2a: the head’s squeeze equals minus the tail’s.* \(Wh = Wx^\star - Wx = y - y = 0\), and \(h = h_{T_{0,1}} + \sum_{j \ge 2} h_{T_j}\), so \(Wh_{T_{0,1}} = -\sum_{j \ge 2} W h_{T_j}\). Therefore

\[ \|Wh_{T_{0,1}}\|^2 = \Big\langle W h_{T_{0,1}},\; -\sum_{j \ge 2} W h_{T_j} \Big\rangle = -\sum_{j \ge 2} \big\langle W h_{T_0} + W h_{T_1},\; W h_{T_j} \big\rangle . \]

*Step 2b: near-perpendicularity bounds each term.* For \(i \in \{0, 1\}\) and \(j \ge 2\), the sets \(T_i\) and \(T_j\) are disjoint with at most \(s\) positions each, so the lemma gives \(|\langle W h_{T_i}, W h_{T_j}\rangle| \le \varepsilon \|h_{T_i}\|\|h_{T_j}\|\). Also \(\|h_{T_0}\| + \|h_{T_1}\| \le \sqrt 2\, \|h_{T_{0,1}}\|\), because \((a + b)^2 \le 2(a^2 + b^2)\) and \(\|h_{T_0}\|^2 + \|h_{T_1}\|^2 = \|h_{T_{0,1}}\|^2\). So

\[ \|Wh_{T_{0,1}}\|^2 \le \varepsilon\,\big(\|h_{T_0}\| + \|h_{T_1}\|\big) \sum_{j\ge 2}\|h_{T_j}\| \le \sqrt 2\,\varepsilon\, \|h_{T_{0,1}}\| \sum_{j \ge 2} \|h_{T_j}\| \le \sqrt 2\,\varepsilon\,\|h_{T_{0,1}}\|\, s^{-1/2}\|h_{T_0^c}\|_1, \]

the last step by Step 1b.

*Step 2c: the RIP from below.* \(h_{T_{0,1}}\) uses at most \(2s\) positions, so \((1 - \varepsilon)\|h_{T_{0,1}}\|^2 \le \|Wh_{T_{0,1}}\|^2\). Combine with Step 2b and divide by \((1-\varepsilon)\|h_{T_{0,1}}\|\) (if it is 0, Claim 2 holds trivially):

\[ \|h_{T_{0,1}}\| \le \frac{\sqrt 2\, \varepsilon}{1 - \varepsilon}\, s^{-1/2}\, \|h_{T_0^c}\|_1 = \rho\, s^{-1/2}\, \|h_{T_0^c}\|_1 . \]

*Step 2d: close the loop.* By Step 1c and Cauchy-Schwarz again,

\[ \|h_{T_{0,1}}\| \le \rho\, s^{-1/2}\big(\|h_{T_0}\|_1 + 2\|x - x_s\|_1\big) \le \rho\,\|h_{T_0}\| + 2\rho\, s^{-1/2}\|x - x_s\|_1 \le \rho\,\|h_{T_{0,1}}\| + 2\rho\, s^{-1/2}\|x - x_s\|_1 . \]

The head appears on both sides. Since \(\rho < 1\), move it to the left: \((1 - \rho)\|h_{T_{0,1}}\| \le 2\rho\, s^{-1/2}\|x - x_s\|_1\). That is Claim 2, and the theorem is proved. \(\blacksquare\)

Step 2d is where \(\varepsilon < 1/(1 + \sqrt 2)\) is needed: with \(\rho \ge 1\) the inequality says nothing. Our icosahedron matrix has \(\varepsilon = 0.447\), just above the threshold 0.414, so this theorem does not cover it, even though the earlier theorem does. You will find in the build-it challenge that the linear program recovers its 1-sparse messages anyway: the RIP condition is sufficient, not necessary.

[[diagram:rip]]

## Random matrices have the property

The last piece: where do RIP matrices come from? Checking the RIP of a given matrix means checking every set of \(2s\) columns, which is the same impossible search as before. So we do not check. We draw \(W\) at random and prove that it is RIP with high probability.

The idea in words. The Johnson-Lindenstrauss lemma handles a finite set of vectors. The sparse vectors on one fixed set \(I\) of \(s\) words form a whole subspace, infinitely many vectors. But it is enough to control a finite **net** of them, a set of unit vectors so dense that every unit vector of the subspace is close to one of them, and then bootstrap from the net to everything. Finally, take a union bound over all \(\binom{d}{s}\) choices of \(I\).

**Lemma (a finite net of the sphere).** For every \(\varepsilon \in (0, 1]\) and every \(s\), there is a set \(N\) of unit vectors in \(\mathbb R^s\) with \(|N| \le (1 + 2/\varepsilon)^s\) such that every unit vector of \(\mathbb R^s\) is within distance \(\varepsilon\) of some point of \(N\).

In one sentence: **about \((3/\varepsilon)^s\) points are enough to come within \(\varepsilon\) of every direction in \(s\) dimensions.** In two dimensions with \(\varepsilon = 0.25\), the bound allows \((1 + 8)^2 = 81\) points.

*Proof.* Build \(N\) greedily: keep adding a unit vector at distance more than \(\varepsilon\) from all points chosen so far, until none is left. If some unit vector were farther than \(\varepsilon\) from every point of \(N\), we could have added it, so when the process stops, \(N\) has the covering property. It does stop, and the count is bounded, by a volume argument: points of \(N\) are more than \(\varepsilon\) apart, so the balls of radius \(\varepsilon/2\) around them do not overlap; and since the centres have length 1, all these balls lie inside the ball of radius \(1 + \varepsilon/2\) around the origin. A ball of radius \(r\) in \(\mathbb R^s\) has volume \(c_s r^s\) for a constant \(c_s\). Comparing volumes, \(|N|\, c_s (\varepsilon/2)^s \le c_s (1 + \varepsilon/2)^s\), so \(|N| \le (1 + 2/\varepsilon)^s\). \(\blacksquare\)

The greedy construction on the circle:

```python
import math
eps = 0.25
net = []
for k in range(100000):
    t = 2 * math.pi * k / 100000
    p = (math.cos(t), math.sin(t))
    if all(math.dist(p, q) > eps for q in net):
        net.append(p)
far = max(min(math.dist((math.cos(t), math.sin(t)), q)
              for q in net)
          for t in [2 * math.pi * k / 997 for k in range(997)])
print(len(net), round((1 + 2 / eps) ** 2), far <= eps)
```

It prints `25 81 True`: 25 points cover the circle to within 0.25, well under the bound of 81.

[[diagram:net]]

**Lemma (one sparse subspace).** Let \(U\) be an orthonormal \(d \times d\) matrix, let \(I\) be a set of \(s\) positions, and let \(S\) be the span of the columns \(U_i\), \(i \in I\). Let \(\delta, \varepsilon \in (0, 1)\) and let \(n\) satisfy

\[ n \ge 24\, \frac{\ln(2/\delta) + s \ln(12/\varepsilon)}{\varepsilon^2} . \]

Then, with probability at least \(1 - \delta\) over a random projection \(W \in \mathbb R^{n,d}\), every nonzero \(x \in S\) has \(\big|\, \|Wx\|/\|x\| - 1 \big| \le \varepsilon\).

- The span \(S\) is every vector \(U_I \alpha\) with \(\alpha \in \mathbb R^s\), where \(U_I\) is the \(d \times s\) matrix of the chosen columns. With \(U = I\) (the identity), \(S\) is exactly the set of messages that use only the words in \(I\).
- The bound on \(n\) is the Johnson-Lindenstrauss dimension for a net of \((12/\varepsilon)^s\) points at accuracy \(\varepsilon/2\).

In one sentence: **a random projection with a few times \(s\ln(1/\varepsilon)/\varepsilon^2\) rows keeps the length of every vector in one fixed s-dimensional subspace within a fraction \(\varepsilon\).** For the messages that use two given words (\(s = 2\)), with \(\varepsilon = 0.5\) and \(\delta = 0.01\):

```python
import math
s, eps, delta = 2, 0.5, 0.01
net = math.log(2 / delta) + s * math.log(12 / eps)
print(math.ceil(24 * net / eps ** 2))
```

It prints `1119`: about eleven hundred rows to pin down a whole plane of messages.

*Proof.* It is enough to treat unit vectors, \(x = U_I\alpha\) with \(\|\alpha\| = 1\); note \(\|U_I \alpha\| = \|\alpha\|\) because the columns of \(U_I\) are orthonormal.

*The net.* Take a net \(N\) of the unit sphere of \(\mathbb R^s\) at distance \(\varepsilon/4\). By the previous lemma, \(|N| \le (1 + 8/\varepsilon)^s \le (9/\varepsilon)^s \le (12/\varepsilon)^s\), using \(\varepsilon \le 1\). Because \(U_I\) keeps lengths, every unit \(x \in S\) is within \(\varepsilon/4\) of some point \(U_I v\), \(v \in N\).

*The net survives.* Apply the Johnson-Lindenstrauss lemma to the finite set \(\{U_I v : v \in N\}\) with confidence \(\delta\). The hypothesis on \(n\) makes its accuracy \(\sqrt{6\ln(2|N|/\delta)/n}\) at most \(\varepsilon/2 \le 1/2\). So, with probability at least \(1 - \delta\), every net point has \(\big|\|WU_Iv\|^2 - 1\big| \le \varepsilon/2\). For a number \(t \in [0, 1]\), \(\sqrt{1 + t} \le 1 + t\) and \(\sqrt{1 - t} \ge 1 - t\), so also \(1 - \varepsilon/2 \le \|WU_Iv\| \le 1 + \varepsilon/2\).

*The bootstrap.* Let \(a\) be the smallest number such that \(\|Wx\| \le (1 + a)\|x\|\) for every \(x \in S\); it is finite because \(W\) restricted to \(S\) is a linear map on a finite-dimensional space. Take a unit \(x \in S\) and a net point with \(\|x - U_Iv\| \le \varepsilon/4\). The difference \(x - U_I v\) is also in \(S\), so

\[ \|Wx\| \le \|WU_Iv\| + \|W(x - U_Iv)\| \le 1 + \frac\varepsilon2 + (1 + a)\frac\varepsilon4 . \]

This holds for every unit \(x \in S\), and \(a\) was the smallest number that works, so \(a \le \frac\varepsilon2 + (1 + a)\frac\varepsilon4\). Solve for \(a\): \(a\,(1 - \varepsilon/4) \le 3\varepsilon/4\), so \(a \le \frac{3\varepsilon/4}{1 - \varepsilon/4} \le \varepsilon\), where the last step is \(3/4 \le 1 - \varepsilon/4\), true for \(\varepsilon \le 1\). In one sentence: **the worst stretch on the subspace is at most the stretch on the net plus a quarter of itself, which pins it below \(\varepsilon\).** The other side follows from the upper bound just proved:

\[ \|Wx\| \ge \|WU_Iv\| - \|W(x - U_Iv)\| \ge 1 - \frac\varepsilon2 - (1 + \varepsilon)\frac\varepsilon4 \ge 1 - \varepsilon, \]

the last step because \(\frac\varepsilon2 + \frac\varepsilon4 + \frac{\varepsilon^2}4 \le \varepsilon\) is \(\varepsilon^2 \le \varepsilon\). \(\blacksquare\)

The bootstrap’s last inequality, checked on a fine grid of \(\varepsilon\) in \((0, 1]\):

```python
grid = [k / 1000 for k in range(1, 1001)]
ok = all((3 * e / 4) / (1 - e / 4) <= e for e in grid)
print(ok)
```

It prints `True`.

**Theorem (random matrices are RIP).** Let \(U\) be a fixed orthonormal \(d \times d\) matrix, let \(\varepsilon, \delta \in (0, 1)\), let \(s \in [d]\), and let

\[ n \ge 216\, \frac{\ln(2/\delta) + s\ln(36\, d/\varepsilon)}{\varepsilon^2} . \]

Then, with probability at least \(1 - \delta\) over a random projection \(W \in \mathbb R^{n,d}\), the matrix \(WU\) is \((\varepsilon, s)\)-RIP.

In one sentence: **a random projection with a constant times \(s \ln(d/\varepsilon)/\varepsilon^2\) rows is RIP for sparsity \(s\), in the standard basis and in any other fixed orthonormal basis.** The dependence on \(d\) is only logarithmic, which is the whole point: \(n\) of order \(s \ln d\), not \(d\).

*Proof.* Apply the subspace lemma to every set \(I\) of exactly \(s\) positions, with accuracy \(\varepsilon/3\) and confidence \(\delta/\binom ds\). There are \(\binom ds \le d^s\) such sets, so the lemma needs \(n \ge 24 \cdot 9\, \big(\ln(2\binom ds/\delta) + s\ln(36/\varepsilon)\big)/\varepsilon^2\), and since \(\ln \binom ds \le s\ln d\), the hypothesis is enough. By the union bound, with probability at least \(1 - \delta\) all the subspaces succeed at once. Every s-sparse \(\alpha \ne 0\) has its nonzero entries inside some \(I\) of size \(s\), so \(WU\alpha\) lies in that subspace and \(1 - \varepsilon/3 \le \|WU\alpha\|/\|\alpha\| \le 1 + \varepsilon/3\). Squaring, the squared ratio lies between \(1 - 2\varepsilon/3 + \varepsilon^2/9 \ge 1 - \varepsilon\) and \(1 + 2\varepsilon/3 + \varepsilon^2/9 \le 1 + \varepsilon\) (the last because \(\varepsilon^2/9 \le \varepsilon/3\)). That is the RIP. \(\blacksquare\)

To recover messages of 60 words we need \((\varepsilon, 2s)\)-RIP with \(\varepsilon < 0.414\), so \(s = 120\) in this theorem. Let’s see what it asks for, with \(\varepsilon = 0.4\) and \(\delta = 0.01\):

```python
import math
d, s, eps, delta = 50000, 60, 0.4, 0.01
k = 2 * s
n = 216 * (math.log(2 / delta) + k * math.log(36 * d / eps))
n /= eps ** 2
print(f"{n:.2e}", round(n / d))
```

It prints `2.49e+06 50`: two and a half million rows, fifty times more than the 50,000 counts of the full vocabulary. Be honest about what this means. The theorem’s shape, \(s \ln d\), is the real phenomenon; its constant 216 is the price of a proof built from a crude net and a union bound over \(10^{200}\) subsets, and it is far too large to certify anything at the mail room’s size. Sharper analyses exist with much smaller constants, but they are beyond this lesson. So at the mail room’s size this lesson certifies the shape of the answer, not a count, and the fix of this lesson works on a desk small enough to check exactly.

## Compressed sensing as code

```
Encode(W, x):
  W is n × d, drawn once
  output: y ← W x

DecodeSparsest(W, y, s):
  exact when W is (ε, 2s)-RIP with ε < 1
  for k = 0, 1, …, s:
    for every set I of k positions:
      if some v supported on I has W v = y: output v

DecodeL1(W, y):
  exact when W is (ε, 2s)-RIP with ε < 1/(1+√2)
  solve the linear program
    minimize Σ_i t_i  over v, t ∈ ℝ^d
    subject to W v = y,  −t_i ≤ v_i ≤ t_i  for every i
  output: v
```

**Cost.** Encoding costs \(O(nd)\), or \(O(ns)\) for an s-sparse message. `DecodeSparsest` tries up to \(\sum_{k \le s}\binom dk\) supports, each a small least-squares solve of cost \(O(nk^2)\): exponential in \(s\). `DecodeL1` is a linear program with \(2d\) variables and \(n + 2d\) constraints, solvable in time polynomial in \(n\) and \(d\). So squeezing is as cheap as a random projection and all the price is in the decoder, a trade that pays when you squeeze far more often than you decode.

**Build it yourself.** The icosahedron matrix, six words, three stored numbers. For `decode_l1`, use a fact from the theory of linear programs: when the minimum exists, it is reached at a solution with at most \(n\) nonzero entries whose columns are linearly independent. So for this tiny problem you may enumerate every set of \(n = 3\) columns, solve the \(3 \times 3\) system, and keep the solution with the smallest \(\ell_1\) norm.

```python
from itertools import combinations

phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
cols = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]

def measure(cols, x):
    return [sum(cols[j][r] * x[j] for j in range(len(cols)))
            for r in range(len(cols[0]))]

def l0_norm(x): ...
def rip_constant_pairs(cols): ...
def solve(M, y): ...
def decode_l0(cols, y, s): ...
def decode_l1(cols, y): ...

def close(a, b):
    return max(abs(p - q) for p, q in zip(a, b)) < 1e-9

if __name__ == "__main__":
    assert l0_norm([0, 2.5, 0, -1, 0, 0]) == 2
    assert round(rip_constant_pairs(cols), 3) == 0.447
    for j in range(6):
        for c in (2.5, -1, 7):
            x = [0.0] * 6
            x[j] = c
            y = measure(cols, x)
            assert close(decode_l0(cols, y, 1), x)
            assert close(decode_l1(cols, y), x)
    print("all asserts pass")
```

1. Implement `l0_norm` and `rip_constant_pairs`, the RIP constant for \(2s = 2\), from the eigenvalues of each \(2 \times 2\) table of column dot products.
2. Implement `solve(M, y)`, Gaussian elimination with partial pivoting that returns `None` for a singular system.
3. Implement `decode_l0` (try supports of size 1 up to \(s\), solve least squares on each, accept an exact match) and `decode_l1` (the enumeration above). The asserts check all 18 one-word messages.
4. Now take a two-word message, \(x = (1, 0, 0, 2, 0, 0)\). Is the matrix \((\varepsilon, 4)\)-RIP for any \(\varepsilon < 1\)? (Think about four vectors in \(\mathbb R^3\).) Find two different two-word messages with the same three stored numbers, and then see what `decode_l1` returns for \(x\). What does that tell you about recovery that works on one message but has no guarantee?

## PCA or compressed sensing?

You now have two ways to squeeze with a promise of recovery, PCA from lesson 23a and compressed sensing from this lesson, and they rest on different assumptions about the messages.

**PCA assumes the messages lie near a low-dimensional subspace.** It recovers perfectly whenever the archive lies exactly in an \(n\)-dimensional subspace, and it is the best linear squeeze with linear recovery for the archive you fit it on.

**Compressed sensing assumes the messages are sparse, in some known basis.** It recovers perfectly every s-sparse message, from about \(s \ln d\) numbers, with a nonlinear decoder.

Each wins where the other’s assumption fails. The single-word messages \(e_1, \dots, e_d\) are 1-sparse, so compressed sensing recovers them all from order \(\ln d\) numbers, where PCA loses most of each one. In the other direction, take \(m\) messages that lie exactly in an \(n\)-dimensional subspace. PCA recovers them perfectly with \(n\) numbers. They are also \(n\)-sparse in any orthonormal basis whose first \(n\) vectors span the subspace, so compressed sensing works too, but it needs on the order of \(n \ln d\) numbers, and with exactly \(n\) it can fail.
For the mail room, word counts are sparse and do not lie near a small subspace, so compressed sensing’s assumption is the right one for recovering messages, and a random projection is the right tool for keeping distances. If all you need is the best summary of a fixed archive, for instance to plot it or to find its main themes, PCA is the right one.

## Fix the bug

Back to the compliance desk, where the manager again wants a promise, not a hope. Every quarantined message uses at most one watchlist word, and the desk must get that word and its count back exactly, with a guarantee that comes from a result of this lesson and not from a pile of test messages. Two house rules come with it. The watchlist is about to grow to thousands of words and its messages to dozens of flagged words, so the decoder must be the linear program: nobody will search every set of 60 words out of 50,000. And storage is tight: at most four numbers per message.

The lab offers three squeezes of the six words and three decoders. The squeezes are the icosahedron matrix from the opening, with three numbers; a matrix with four rows, whose six columns are \((0, 1, 1, 1)/\sqrt 3\), \((0, 1, 1, -1)/\sqrt 3\), \((0, 1, -1, 1)/\sqrt 3\), \((0, 1, -1, -1)/\sqrt 3\), \((1, 1, 0, 0)/\sqrt 2\) and \((1, -1, 0, 0)/\sqrt 2\); and a matrix with five rows, whose six columns point at the corners of a regular simplex, every two of them with dot product \(-1/5\). The decoders are the shortest matching message (the desk’s current one, the smear), the search for the sparsest matching message, and the linear program.

A message that uses one word has \(\|x\|_0 = 1\), so \(s = 1\), and both results need a matrix that is \((\varepsilon, 2)\)-RIP. For unit columns that constant is easy to read off. Words \(i\) and \(j\) have columns with dot product \(c = w_i \cdot w_j\), and the table of dot products of the two columns is

\[ \begin{pmatrix} 1 & c \\ c & 1 \end{pmatrix}, \]

whose eigenvalues are \(1 + |c|\) and \(1 - |c|\) (the eigenvectors are \((1, 1)\) and \((1, -1)\)). So any message on those two words has \(\|Wx\|^2/\|x\|^2\) between \(1 - |c|\) and \(1 + |c|\), and the RIP constant for \(2s = 2\) is the largest \(|c|\) over all pairs of columns. In one sentence: **for unit columns, the constant for one-word messages is the largest dot product between two different columns.** For the icosahedron every pair of columns has \(|c| = 0.447\), so every message on two words is stretched by a factor between 0.553 and 1.447. The snippet reads it off the three-row and four-row matrices, and the five-row one has \(1/5\) by its definition, next to the program’s threshold \(1/(1 + \sqrt 2)\):

```python
from itertools import combinations
phi = (1 + 5 ** 0.5) / 2
raw = [(0, 1, phi), (0, -1, phi), (1, phi, 0),
       (-1, phi, 0), (phi, 0, 1), (phi, 0, -1)]
ico = [[t / (1 + phi ** 2) ** 0.5 for t in c] for c in raw]
a, b = 3 ** -0.5, 2 ** -0.5
four = [(0, a, s * a, t * a) for s in (1, -1) for t in (1, -1)]
four += [(b, b, 0, 0), (b, -b, 0, 0)]
dot = lambda u, v: sum(p * q for p, q in zip(u, v))

def constant(cols):
    return max(abs(dot(u, v)) for u, v in combinations(cols, 2))

limit = 1 / (1 + 2 ** 0.5)
for name, cols in [("three", ico), ("four", four)]:
    print(name, round(constant(cols), 3), constant(cols) < limit)
print("five", 1 / 5, 1 / 5 < limit)
print(round(limit, 3))
```

It prints `three 0.447 False`, `four 0.408 True`, `five 0.2 True` and `0.414`. The icosahedron is above the program’s threshold, the other two are below it. Before you open the lab, decide which pairs of squeeze and decoder a result of this lesson certifies, and which of them the house rules allow. Then think about this: the lab also counts how many of its 30 test messages (each of the six words, with counts 1 to 5) come back exactly, and with the icosahedron and the program it is all 30. Is that good enough?

[[lab-act2]]

Measuring is not certifying. The icosahedron with the program recovers all 30 test messages, but the program’s theorem needs a constant below 0.414, and the icosahedron’s is 0.447. Six lines in three dimensions cannot be spread further apart than the icosahedron spreads them (a known fact, which we do not prove), so no three-number squeeze of six words is certified for the program, however you choose its columns. The sparsest search is certified for it (0.447 is below 1), but the house rules forbid it: it tries every set of words, six here and \(\binom{50000}{60} \approx 1.0 \times 10^{200}\) in the real mail. The four-row matrix has constant 0.408, below the threshold, so the program recovers every one-word message exactly, from four numbers, with a result to point at. The five-row matrix is certified with room to spare, but it is over the storage budget.

Be honest about how far this reaches. The certificate covers messages of one word, and the four-row matrix has no promise for two: its first four columns lie in a three-dimensional space, so they are dependent, and the concept check asks what that costs. The guarantee for nearly sparse messages is also weak this close to the threshold, as the concept check shows with numbers. And at the real mail’s size, with 60 words per message, the constants of the last theorem say nothing a mail room could certify. What this lesson gives you is the shape of the answer, \(s \ln d\) rather than \(d\), and a way to check a small desk exactly.

## What people get wrong

**“A better recovery matrix would fix the smear.”** No matrix can. One that was right on every one-word message would be right on every message, because messages are sums of one-word messages and a matrix respects sums, and the null space forbids that. The decoder has to be nonlinear, and exploiting sparsity is what makes it so.

**“Exact recovery of a squeezed message is impossible.”** It is impossible for every message, because of the null space. It is possible for every sparse message, when the null space contains no sparse vectors, which is what the RIP for \(2s\) guarantees.

**“If the linear program recovered my test messages, the matrix is fine.”** Success on the messages you tried is not a guarantee for the next one. In the build-it challenge the linear program recovers a two-word message from three numbers, yet some two-word messages are indistinguishable from each other at that size, so no decoder can recover them all.

**“The constants in the RIP theorem tell you how many measurements you need.”** They give a number that certainly suffices, from a crude proof; the \(s \ln d\) shape is the lesson, the 216 is not.

## Recap and the next bug

1. **No recovery matrix can undo a squeeze**, not even for messages that use a single word: a matrix right on every one-word message would be right on every message, and the null space forbids that. The decoder has to be nonlinear.
2. A message is **s-sparse** when \(\|x\|_0 \le s\). **Compressed sensing** fixes one matrix \(W\) in advance and recovers every s-sparse message from \(y = Wx\). If \(W\) is \((\varepsilon, 2s)\)-RIP with \(\varepsilon < 1\), the sparsest message matching \(y\) is the message itself.
3. The sparsest search is a search over \(\binom ds\) supports. With \(\varepsilon < 1/(1+\sqrt 2)\), the **linear program** that minimizes the \(\ell_1\) norm recovers every s-sparse message exactly, and any message up to an error proportional to its part outside the \(s\) largest entries.
4. A random matrix with order \(s\ln(d/\varepsilon)/\varepsilon^2\) rows is RIP with high probability, by a net, the Johnson-Lindenstrauss lemma and a union bound. The constants are far too large to certify a mail room.
5. PCA (lesson 23a) bets on a subspace, compressed sensing on sparsity. The mail room’s word counts are sparse, so recovering messages is compressed sensing’s job, and keeping distances is the random projection’s.
6. On the six-word desk, the four-row matrix has RIP constant 0.408, below the program’s threshold 0.414, so the program recovers every one-word message exactly and certifiably. The icosahedron’s 0.447 recovers every test message but is not certified, and the sparsest search is certified but searches every set of words.

The desk can read its words back, and the filter is fast. At the Friday review, though, a colleague from the statistics team looks at everything you have built this term and frowns. Every learner so far, she says, draws a boundary between junk and keep: a halfspace, a tree, a neighbourhood. None of them says anything about how a junk message comes to be written. Why not model that instead? Describe how a junk message is produced, word by word, and how a keep message is produced, learn both descriptions from the archive, and when a new message arrives, ask which of the two most likely produced it. Is that a better idea than drawing a boundary, or a worse one? That is the next lesson: generative models.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The four-row matrix of the fix is \((0.408, 2)\)-RIP. For which sparsity \(s\) does the sparsest-message result promise exact recovery from it? What would you need for \(s = 2\), and does the matrix have it? In the order free, prize, winner, urgent, claim, wire, find two different two-word messages, each word used once, that give the same four stored numbers, and say what that does to any decoder.
2. A message uses “free” once and “urgent” five times. Take \(s = 1\): what are \(x_s\) and \(\|x - x_s\|_1\)? What does the \(\ell_1\) recovery result promise about \(\|x^\star - x\|_2\) for a matrix with constant 0.408, and for one with constant 0.2? Is the first promise useful, comparing it with the length of \(x\)?
3. A teammate runs the icosahedron matrix with the linear program on 2,000 random one-word messages and sees no errors. Another teammate says the result of this lesson does not cover that matrix. What is true in each claim, and what would you have to change to ship with a certificate?

## Problems

Once the concept check is done, try this. Hints come one at a time in the chat, and no full solution.

- **Problem 1 (how many numbers does sparse recovery need?).** Let \(W \in \mathbb R^{n,d}\) with \(n < 2s \le d\). Show that there are two different s-sparse vectors with the same stored numbers \(Wx\), so that no decoder can recover every s-sparse message from \(W\). Check it against the icosahedron matrix, where \(n = 3\) and \(s = 2\).
