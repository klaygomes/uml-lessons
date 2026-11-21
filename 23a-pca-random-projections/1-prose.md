# PCA and Random Projections

## Fifty thousand counts per message

The data team’s new parser went live on Monday, and by Wednesday everything was slow.

The new parser is thorough. Instead of the two scores the desks used to cluster on, it describes each message by how many times every word of the company’s vocabulary appears in it. The vocabulary has 50,000 words, so every message arrives as a list of 50,000 numbers: a 3 under “free”, a 1 under “prize”, a 2 under “winner”, and zeros almost everywhere else. Each of those numbers is a **word count**, and the list as a whole is a vector \(x \in \mathbb R^d\) with \(d = 50{,}000\).

You saw k-means crawl at the end of lesson 22, and the cost is everywhere. Remember lesson 19: the nearest-neighbour filter compares a new message with every stored one, and each comparison of two vectors with \(d\) entries costs \(d\) subtractions and \(d\) multiplications. With \(d = 50{,}000\), comparing one new message against an archive of 200,000 stored ones is ten billion multiplications. The archive itself, stored as plain 4-byte numbers, needs 200,000 times 50,000 times 4 bytes, which is 40 gigabytes. And lesson 19 also warned you that more dimensions can make a filter worse, not better: the curse of dimensionality says the labels you need can grow exponentially with the number of features. The desks felt it as noisy clusters, with every distance looking like every other.

Your colleague’s proposal is blunt: “Squeeze every message into a few hundred numbers. Nobody needs fifty thousand.” That sounds right. But it raises three questions. This lesson answers the first two, one after another, and the third has lesson 23b to itself.

1. If you squeeze, can you get the message back? And which squeeze loses the least?
2. If what you really need is not the message itself but the distances between messages (that is all nearest neighbour and k-means use), how few numbers keep every distance nearly unchanged, and can you prove it?
3. Most of the 50,000 counts in any one message are zero. Can you exploit that, and recover each message exactly from far fewer numbers?

Each question gets its own method. The first is **principal component analysis**, the second **random projections**, and the third **compressed sensing**, which comes in lesson 23b. All three squeeze the same way, with a matrix, so let’s start there.

## Squeezing a message with a matrix

Before any formula, here is the idea with three words. Suppose the vocabulary were just “free”, “prize” and “meeting”, so a message is three counts. A message with 3 “free”, 2 “prize” and no “meeting” is the vector \((3, 2, 0)\). You want to store it with two numbers instead of three. One obvious way: store the total of “free” and “prize” as the first number, and the count of “meeting” as the second. The message \((3, 2, 0)\) becomes \((5, 0)\). Each stored number is a weighted sum of the counts, with weights you chose in advance.

That is what a matrix does. A **compression matrix** is a matrix \(W\) with \(n\) rows and \(d\) columns, written \(W \in \mathbb R^{n,d}\), where \(n < d\). It squeezes a message \(x\) into \(y = Wx\): the \(i\)-th stored number is the \(i\)-th row of \(W\) multiplied entry by entry with \(x\) and summed. In our example,

\[ W = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \qquad W \begin{pmatrix} 3 \\ 2 \\ 0 \end{pmatrix} = \begin{pmatrix} 1\cdot 3 + 1 \cdot 2 + 0 \cdot 0 \\ 0 \cdot 3 + 0 \cdot 2 + 1 \cdot 0 \end{pmatrix} = \begin{pmatrix} 5 \\ 0 \end{pmatrix}. \]

Reading it piece by piece:

- \(\mathbb R^{n,d}\) is the set of all tables of real numbers with \(n\) rows and \(d\) columns; here \(n = 2\) and \(d = 3\).
- \(Wx\) is the matrix-vector product: row \(i\) of \(W\) dotted with \(x\) gives entry \(i\) of the result. Row 1 is \((1, 1, 0)\), so entry 1 is “free plus prize”. Row 2 is \((0, 0, 1)\), so entry 2 is “meeting”.
- The result \(y = Wx\) lives in \(\mathbb R^n\): it has \(n = 2\) entries instead of \(d = 3\).

In one sentence: **a compression matrix turns each message into \(n\) weighted sums of its counts.**

Getting a message back works the same way in the other direction. A **recovery matrix** \(U \in \mathbb R^{d,n}\), with \(d\) rows and \(n\) columns, turns the stored \(y\) back into a vector \(\tilde x = Uy\) with \(d\) entries, the **recovered message**. Our first number is “free plus prize” and we no longer know the split, so a fair guess is half each:

\[ U = \begin{pmatrix} 0.5 & 0 \\ 0.5 & 0 \\ 0 & 1 \end{pmatrix}, \qquad \tilde x = U W x = U \begin{pmatrix} 5 \\ 0 \end{pmatrix} = \begin{pmatrix} 2.5 \\ 2.5 \\ 0 \end{pmatrix}. \]

Here \(UWx\) means: first squeeze with \(W\), then recover with \(U\). The tilde on \(\tilde x\) marks “an approximation of \(x\)”. The damage is measured by the **squared error** \(\|x - \tilde x\|^2\), where \(\|v\|\) is the Euclidean length of a vector, \(\|v\| = \sqrt{v_1^2 + \dots + v_d^2}\), and \(\|v\|^2\) is the sum of the squared entries. For our message the error is \((3 - 2.5)^2 + (2 - 2.5)^2 + 0^2 = 0.5\). In one sentence: **recovery squeezes and then unsqueezes, and the squared error says how far the round trip lands from where it started.**

Now try the message \((2, 3, 0)\): two “free”, three “prize”. Before you run the code, what do you expect it to be stored as?

```python
W = [[1, 1, 0],
     [0, 0, 1]]
U = [[0.5, 0], [0.5, 0], [0, 1]]

def mul(M, v):
    return [sum(a * b for a, b in zip(row, v)) for row in M]

for x in [(3, 2, 0), (2, 3, 0)]:
    y = mul(W, x)
    x_rec = mul(U, y)
    err = sum((a - b) ** 2 for a, b in zip(x, x_rec))
    print(x, y, x_rec, err)
```

It prints

```
(3, 2, 0) [5, 0] [2.5, 2.5, 0] 0.5
(2, 3, 0) [5, 0] [2.5, 2.5, 0] 0.5
```

Two different messages, the same two stored numbers. No recovery matrix, and in fact no recovery procedure of any kind, can tell them apart afterwards. This is not bad luck with this \(W\). Whenever \(n < d\), the equations \(Wv = 0\) are \(n\) equations in \(d\) unknowns, and such a system always has a solution \(v \neq 0\) (here \(v = (1, -1, 0)\)). Then \(x\) and \(x + v\) are squeezed to the same \(y\), because \(W(x + v) = Wx + Wv = Wx\). So **exact recovery of every message is impossible**; the set of vectors \(v\) with \(Wv = 0\) is called the **null space** of \(W\), and anything in it is invisible after squeezing.

[[diagram:squeeze]]

So every squeeze loses something. The game is to choose what to lose. PCA loses as little squared error as possible on the messages you have. Random projections lose almost nothing of the distances. Compressed sensing, in lesson 23b, loses nothing at all, provided the messages are sparse. This lesson takes the first two, in that order.

## Six messages, three words

Here is a small archive, still with the three words “free”, “prize” and “meeting”. Three messages look like junk, three look like meeting notes.

| message | free | prize | meeting |
|---|---|---|---|
| \(x_1\) | 3 | 2 | 0 |
| \(x_2\) | 2 | 2 | 1 |
| \(x_3\) | 4 | 3 | 0 |
| \(x_4\) | 0 | 1 | 3 |
| \(x_5\) | 1 | 0 | 4 |
| \(x_6\) | 0 | 0 | 2 |

Suppose you may store just one number per message (\(n = 1\)). A single stored number is the dot product of the message with one fixed vector \(u\), and the natural recovery is to walk back along the same \(u\): \(\tilde x = u\,(u \cdot x)\), where \(u \cdot x = u_1 x_1 + u_2 x_2 + u_3 x_3\). If \(u\) has length 1, \(\tilde x\) is the point on the line through \(u\) closest to \(x\), its **projection** onto that line. (We will prove shortly that this is the best recovery for a given line.)

Which line should you pick? Try a few by hand.

- The “free” axis, \(u = (1, 0, 0)\): you store only the “free” count. Message \(x_1 = (3, 2, 0)\) comes back as \((3, 0, 0)\), error \(2^2 = 4\). Summed over all six messages, the error is \(48\): every “prize” and “meeting” count is lost.
- The “meeting” axis, \(u = (0, 0, 1)\): total error also \(48\), by coincidence.
- The junk direction, \(u = (1, 1, 0)/\sqrt 2\): you store \((\text{free} + \text{prize})/\sqrt 2\). Total error \(32\).

What is the smallest total error any single line can achieve? Before reading on, guess: closer to 32, to 20, or to 5?

The **total squared error** of a squeeze \((U, W)\) on the archive is the sum of the round-trip errors of the six messages. Here it is in code, for the “free” axis, for the junk direction, and for the two-number squeeze of the previous section:

```python
X = [(3, 2, 0), (2, 2, 1), (4, 3, 0),
     (0, 1, 3), (1, 0, 4), (0, 0, 2)]

def mul(M, v):
    return [sum(a * b for a, b in zip(row, v)) for row in M]

def objective(U, W):
    return sum(sum((a - b) ** 2 for a, b in
                   zip(x, mul(U, mul(W, x)))) for x in X)

r = 0.5 ** 0.5
print(objective([[1], [0], [0]], [[1, 0, 0]]))
print(round(objective([[r], [r], [0]], [[r, r, 0]]), 2))
print(objective([[0.5, 0], [0.5, 0], [0, 1]],
                [[1, 1, 0], [0, 0, 1]]))
```

It prints `48`, `32.0` and `2.0`. The hand-made two-number squeeze is already good, losing only 2. But is it the best two-number squeeze? And what is the best one-number squeeze? We need the method that answers both, and it turns out the answers are 28.13 for one number and 1.19 for two, both read off one small \(3 \times 3\) matrix built from the archive.

[[lab]]

In the lab you can spin the line yourself on a cloud of messages and watch the total error change, then switch to two numbers and spin a plane. Notice where the minimum sits: along the direction in which the cloud is longest.

## The best linear compression

Here is the problem in words first. You have \(m\) messages. You want to choose a compression matrix and a recovery matrix, both fixed in advance and the same for every message, so that the total squared error over all \(m\) messages is as small as possible. In our archive, \(m = 6\), \(d = 3\), and for \(n = 1\) the best value is what we are after. In symbols, this is the **PCA problem**:

\[ \operatorname*{argmin}_{W \in \mathbb R^{n,d},\; U \in \mathbb R^{d,n}} \; \sum_{i=1}^{m} \| x_i - U W x_i \|^2 . \]

Read it from the inside out.

- \(x_i\) is message number \(i\), one of \(x_1, \dots, x_m\).
- \(UWx_i\) is its round trip: squeeze with \(W\), recover with \(U\).
- \(\|x_i - UWx_i\|^2\) is the squared error of that one message, the sum of the squared differences entry by entry.
- \(\sum_{i=1}^m\) adds those errors over all \(m\) messages, giving the total squared error.
- \(\operatorname{argmin}\) over \(W \in \mathbb R^{n,d}, U \in \mathbb R^{d,n}\) means: the pair \((W, U)\) that makes the total smallest, chosen among all matrices of those shapes. (“min” would be the smallest value; “argmin” is the choice that achieves it.)

In one sentence: **find the squeeze and the unsqueeze that, together, lose the least total squared error on the archive.** With the junk-direction choice from the table (\(n = 1\)), the total was 32; the code above is this formula term by term, with `objective(U, W)` as the sum.

Solving it takes three steps. First, show that the best recovery matrix has orthonormal columns and that the best compression matrix is its transpose. Second, turn the error into a quantity built from one matrix of the archive. Third, find the directions that matrix stretches most.

### Step 1: the unsqueeze should be the transpose of the squeeze

Two definitions first. Vectors are **orthonormal** when each has length 1 and every two of them are perpendicular (their dot product is 0). For a matrix \(U\) with columns \(u_1, \dots, u_n\), the **transpose** \(U^\top\) is the matrix whose rows are those columns; the product \(U^\top U\) collects all the dot products \(u_j \cdot u_k\), so “the columns of \(U\) are orthonormal” is the same as \(U^\top U = I\), the identity matrix (ones on the diagonal, zeros elsewhere).

Here is a number before the lemma. Take a deliberately clumsy one-number squeeze on our archive: store \(0.3 \cdot \text{free} + 0.2 \cdot \text{prize} + 0.1 \cdot \text{meeting}\), and recover by multiplying that number by \((2, 1, 0)\). Every recovered message lies on the line through \((2, 1, 0)\). Now keep the line but change the squeeze: use the unit vector \(v = (2, 1, 0)/\sqrt 5\) for both, storing \(v \cdot x\) and recovering \(v\,(v \cdot x)\). Which has the smaller total error?

```python
X = [(3, 2, 0), (2, 2, 1), (4, 3, 0),
     (0, 1, 3), (1, 0, 4), (0, 0, 2)]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))

def error(u, w):
    return sum(sum((xi - ui * dot(w, x)) ** 2
                   for xi, ui in zip(x, u)) for x in X)

u, w = (2, 1, 0), (0.3, 0.2, 0.1)
v = [a / dot(u, u) ** 0.5 for a in u]
print(round(error(u, w), 2), round(error(v, v), 2))
```

It prints `35.2 32.8`. Same line of recovered points, smaller error, because each message now lands on the point of the line closest to it. That is the whole lemma.

**Lemma (the best pair is a projection).** If \((U, W)\) solves the PCA problem, then the columns of \(U\) are orthonormal, \(U^\top U = I\), and \(W = U^\top\).

*Proof.* Fix any \(U\) and \(W\), and look at all the vectors the round trip can produce, \(R = \{UWx : x \in \mathbb R^d\}\). (Read the braces with the colon as “the set of all \(UWx\) such that \(x\) ranges over \(\mathbb R^d\)”.) \(R\) is a **linear subspace**: a set of vectors closed under adding and scaling, like a line or a plane through the origin. Its dimension is at most \(n\), because every \(UWx\) is \(U\) times a vector with \(n\) entries. To keep notation simple, assume the dimension is exactly \(n\) (if it is smaller, add extra orthonormal directions; this can only shrink the errors below).

Choose a matrix \(V \in \mathbb R^{d,n}\) whose columns are an orthonormal basis of \(R\): \(V^\top V = I\), and every vector of \(R\) is \(Vy\) for some \(y \in \mathbb R^n\).

Now fix one message \(x\) and ask which point \(Vy\) of \(R\) is closest to it. Expand the squared distance:

\[ \|x - Vy\|^2 = \|x\|^2 + y^\top V^\top V y - 2\, y^\top V^\top x = \|x\|^2 + \|y\|^2 - 2\, y \cdot (V^\top x). \]

A vector counts as a one-column matrix, so \(a^\top b = a \cdot b\) and \(y^\top M y\) is \(y \cdot (My)\). The first equality is \(\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\, a \cdot b\) with \(a = x\), \(b = Vy\), and \(\|Vy\|^2 = y^\top V^\top V y\). The second uses \(V^\top V = I\), so \(y^\top V^\top V y = y^\top y = \|y\|^2\). The right-hand side is a bowl in \(y\): its gradient with respect to \(y\) is \(2y - 2V^\top x\), which is zero exactly at \(y = V^\top x\), and since the bowl curves upwards (the \(\|y\|^2\) term) that point is its minimum. So the closest point of \(R\) to \(x\) is \(VV^\top x\).

The old round trip \(UWx\) is also a point of \(R\), so it is at least as far from \(x\) as the closest point:

\[ \|x - UWx\|^2 \ge \|x - VV^\top x\|^2 \quad\text{for every } x. \]

Add this up over \(x_1, \dots, x_m\): the pair \((V, V^\top)\) has a total error no larger than \((U, W)\). Since \((U, W)\) was arbitrary, an optimal pair can always be taken of the form \((V, V^\top)\) with orthonormal columns. \(\blacksquare\)

In the snippet, \(R\) was the line through \((2, 1, 0)\) and \(V\) the unit vector \(v\). So the PCA problem becomes a search over \(U\) alone:

\[ \operatorname*{argmin}_{U \in \mathbb R^{d,n} \,:\, U^\top U = I} \; \sum_{i=1}^m \| x_i - U U^\top x_i \|^2 . \]

The colon under argmin reads “such that”: search only among matrices \(U\) whose columns are orthonormal. In one sentence: **find the \(n\) orthonormal directions such that projecting every message onto them loses the least total squared error.**

### Step 2: the error is what the projection misses

Here is the idea in words. Projecting onto a subspace splits a message into two perpendicular parts: the part inside the subspace, which you keep, and the part outside, which you lose. By Pythagoras, the squared length of the message is the kept part’s squared length plus the lost part’s. So minimizing what you lose is the same as maximizing what you keep.

With numbers: for \(x_1 = (3, 2, 0)\) and the junk direction \(u = (1, 1, 0)/\sqrt 2\), the kept length is \(u \cdot x_1 = 5/\sqrt 2\), whose square is 12.5; the total squared length is \(9 + 4 = 13\); so the loss is \(13 - 12.5 = 0.5\), exactly the error we found for this message at the start.

In symbols, for any \(x\) and any \(U\) with \(U^\top U = I\):

\[ \|x - UU^\top x\|^2 = \|x\|^2 - 2\,x^\top UU^\top x + x^\top U U^\top U U^\top x = \|x\|^2 - x^\top U U^\top x = \|x\|^2 - \operatorname{trace}(U^\top x x^\top U). \]

Reading it:

- The first equality expands \(\|a - b\|^2\) with \(a = x\) and \(b = UU^\top x\).
- The middle term simplifies because \(U^\top U = I\): \(x^\top U (U^\top U) U^\top x = x^\top UU^\top x\), so \(-2\) of it plus \(1\) of it leaves \(-1\).
- \(x^\top U U^\top x = \|U^\top x\|^2\) is the kept part: the squared length of the \(n\) stored numbers.
- The **trace** of a square matrix, \(\operatorname{trace}(M)\), is the sum of its diagonal entries. The matrix \(U^\top x x^\top U\) is \((U^\top x)(U^\top x)^\top\), whose diagonal holds the squares of the \(n\) stored numbers, so its trace is again \(\|U^\top x\|^2\). Writing it as a trace looks like a detour; it is what lets us add the messages together in the next line.
- \(xx^\top\) is the **outer product**, the \(d \times d\) matrix whose entry \((j, k)\) is \(x_j x_k\).

In one sentence: **a message’s error is its squared length minus the squared length of what you stored.**

Now add over the messages. The trace is linear (the trace of a sum is the sum of the traces), so

\[ \sum_{i=1}^m \|x_i - UU^\top x_i\|^2 = \sum_{i=1}^m \|x_i\|^2 - \operatorname{trace}\big(U^\top A\, U\big), \qquad A = \sum_{i=1}^m x_i x_i^\top . \]

The matrix \(A\) is the **scatter matrix** of the archive: entry \((j, k)\) is \(\sum_i x_{i,j} x_{i,k}\), how much word \(j\) and word \(k\) appear together, summed over messages. The first sum does not depend on \(U\). So the PCA problem is to make \(\operatorname{trace}(U^\top A U)\) as large as possible over \(U\) with \(U^\top U = I\). In one sentence: **the best squeeze keeps the most scatter.** Here is the check for the junk direction, computing the error directly and through \(A\):

```python
X = [(3, 2, 0), (2, 2, 1), (4, 3, 0),
     (0, 1, 3), (1, 0, 4), (0, 0, 2)]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
A = [[sum(x[i] * x[j] for x in X) for j in range(3)]
     for i in range(3)]
r = 0.5 ** 0.5
u = (r, r, 0)
direct = sum(dot(x, x) - dot(u, x) ** 2 for x in X)
Au = [dot(row, u) for row in A]
via_A = sum(dot(x, x) for x in X) - dot(u, Au)
print(A)
print(round(direct, 2), round(via_A, 2))
```

It prints `[[30, 22, 6], [22, 18, 5], [6, 5, 30]]` and `32.0 32.0`. The 30 in the corner is \(3^2 + 2^2 + 4^2 + 0 + 1^2 + 0\), the total of the squared “free” counts; the 22 is \(\sum_i \text{free}_i \cdot \text{prize}_i\). The total squared length of the archive is \(30 + 18 + 30 = 78\), and \(u^\top A u = 46\), leaving 32.

### Step 3: eigenvectors of the scatter matrix

An **eigenvector** of a square matrix \(A\) is a nonzero vector \(u\) that \(A\) only stretches: \(Au = \lambda u\) for some number \(\lambda\), its **eigenvalue**. The matrix with rows \((2, 0)\) and \((0, 1)\) sends \((1, 0)\) to \((2, 0)\), so \((1, 0)\) is an eigenvector with eigenvalue 2, while \((1, 1)\) goes to \((2, 1)\), which is not a multiple of it. Two facts about scatter matrices do the rest, and both are standard linear algebra.

- \(A\) is **symmetric** (\(A^\top = A\)), since \(x x^\top\) is. Every symmetric matrix has a **spectral decomposition** \(A = VDV^\top\): \(V\) is a \(d \times d\) matrix with orthonormal columns (\(V^\top V = VV^\top = I\)), those columns are eigenvectors of \(A\), and \(D\) is diagonal with the matching eigenvalues \(D_{1,1}, \dots, D_{d,d}\). We order them so that \(D_{1,1} \ge D_{2,2} \ge \dots \ge D_{d,d}\).
- \(A\) is **positive semidefinite**: \(v^\top A v = \sum_i (v \cdot x_i)^2 \ge 0\) for every \(v\). Taking \(v\) to be an eigenvector shows every eigenvalue is at least 0.

In words, the decomposition says that the scatter of the archive has \(d\) perpendicular natural axes, and along axis \(j\) the archive’s total squared spread is \(D_{j,j}\). The claim of the theorem is that the best \(n\) directions are the \(n\) axes with the most spread.

**Theorem (PCA).** Let \(x_1, \dots, x_m\) be arbitrary vectors in \(\mathbb R^d\), let \(A = \sum_{i=1}^m x_i x_i^\top\), and let \(u_1, \dots, u_n\) be orthonormal eigenvectors of \(A\) for its \(n\) largest eigenvalues. Then setting \(U\) to be the matrix with columns \(u_1, \dots, u_n\) and \(W = U^\top\) solves the PCA problem.

The \(u_j\) are called the **principal components** (or principal vectors) of the archive. In one sentence: **store each message’s coordinates along the archive’s \(n\) longest axes.**

*Proof.* By Steps 1 and 2 it is enough to show that \(\operatorname{trace}(U^\top AU) \le \sum_{j=1}^n D_{j,j}\) for every \(U \in \mathbb R^{d,n}\) with orthonormal columns, and that the eigenvector choice reaches this value.

Fix such a \(U\) and let \(B = V^\top U\), a \(d \times n\) matrix: column \(i\) of \(B\) holds the coordinates of \(u_i\) in the eigenvector basis. Since \(VV^\top = I\), we have \(VB = U\), and so

\[ U^\top A U = B^\top V^\top (VDV^\top) V B = B^\top D B, \qquad \operatorname{trace}(U^\top AU) = \sum_{j=1}^d D_{j,j} \sum_{i=1}^n B_{j,i}^2 . \]

The second formula comes from writing out the diagonal of \(B^\top DB\): its \((i, i)\) entry is \(\sum_j D_{j,j} B_{j,i}^2\), and the trace adds these over \(i\). Call \(\beta_j = \sum_{i=1}^n B_{j,i}^2\), the share of axis \(j\) that the chosen directions capture. Two facts about the \(\beta_j\):

- **Each \(\beta_j\) is at most 1.** The columns of \(B\) are orthonormal, since \(B^\top B = U^\top VV^\top U = U^\top U = I\). Extend them to a full orthonormal basis of \(\mathbb R^d\), a \(d \times d\) matrix \(\tilde B\) whose first \(n\) columns are those of \(B\) and with \(\tilde B^\top \tilde B = I\). A square matrix with \(\tilde B^\top \tilde B = I\) also has \(\tilde B \tilde B^\top = I\) (\(\tilde B^\top\) is then its inverse, and a square matrix’s inverse works on both sides), so every row of \(\tilde B\) has squared length 1: \(\sum_{i=1}^d \tilde B_{j,i}^2 = 1\). Dropping the last \(d - n\) columns can only lower this, so \(\beta_j \le 1\).
- **The \(\beta_j\) add up to \(n\).** \(\sum_j \beta_j = \sum_{i=1}^n \sum_j B_{j,i}^2\) is the sum of the squared lengths of \(n\) unit columns, which is \(n\).

So \(\beta = (\beta_1, \dots, \beta_d)\) lies in \([0, 1]^d\) (every entry between 0 and 1) and has \(\|\beta\|_1 = \sum_j |\beta_j| = n\), and

\[ \operatorname{trace}(U^\top AU) = \sum_{j=1}^d D_{j,j}\, \beta_j \;\le\; \max_{\beta \in [0,1]^d,\; \|\beta\|_1 \le n} \; \sum_{j=1}^d D_{j,j}\,\beta_j . \]

In words: you have \(n\) units of “capture” to spread over the axes, at most one unit per axis, and axis \(j\) pays \(D_{j,j}\) per unit. Obviously you put a full unit on each of the \(n\) best-paying axes. Let’s make “obviously” a proof. Take any feasible \(\beta\) and suppose some \(j \le n\) has \(\beta_j < 1\). If the total is below \(n\), raise \(\beta_j\) towards 1, as far as the total allows; the value cannot drop, since \(D_{j,j} \ge 0\). Otherwise the total is \(n\) while the first \(n\) entries sum to less than \(n\), so some \(k > n\) has \(\beta_k > 0\); move an amount \(t = \min(1 - \beta_j, \beta_k)\) from \(k\) to \(j\). The value changes by \(t\,(D_{j,j} - D_{k,k}) \ge 0\), since \(j < k\) means \(D_{j,j} \ge D_{k,k}\). Each move either fills an axis \(j \le n\) or empties an axis \(k > n\), so after at most \(d\) moves we reach \(\beta = (1, \dots, 1, 0, \dots, 0)\) with \(n\) ones, without ever lowering the value. So the maximum is \(\sum_{j=1}^n D_{j,j}\).

Here is that claim tested against 100,000 random feasible \(\beta\), using our archive’s eigenvalues (computed in the next snippet) and \(n = 2\):

```python
import random
alpha = [49.87, 26.94, 1.19]
n = 2
rng = random.Random(0)
best = 0.0
for _ in range(100000):
    beta = [rng.random() for _ in alpha]
    if sum(beta) <= n:
        best = max(best, sum(a * b for a, b in zip(alpha, beta)))
print(round(best, 2), round(sum(alpha[:n]), 2))
```

It prints `75.92 76.81`: random spreads come close but never pass the top-two sum. With \(n = 1\) the junk direction keeps \(u^\top A u = 46\), below the top eigenvalue 49.87, as the bound says.

Finally, choose \(U = (u_1, \dots, u_n)\), the top eigenvectors. Then \(B = V^\top U\) is the first \(n\) columns of the identity, \(\beta = (1, \dots, 1, 0, \dots, 0)\), and \(\operatorname{trace}(U^\top AU) = \sum_{j=1}^n D_{j,j}\): the upper bound is reached. \(\blacksquare\)

**The leftover error.** The proof also tells us the error of the best squeeze. The trace of \(A\) is \(\sum_i \|x_i\|^2\) (each diagonal entry of \(x_ix_i^\top\) is a squared count), and the trace also equals the sum of all eigenvalues: diagonal entry \(i\) of \(VDV^\top\) is \(\sum_j V_{i,j}^2 D_{j,j}\), and summing over \(i\) gives \(\sum_j D_{j,j}\), because every column of \(V\) has squared length 1. So the smallest total error is

\[ \sum_{i=1}^m \|x_i\|^2 - \sum_{j=1}^n D_{j,j} = \sum_{j=n+1}^{d} D_{j,j}. \]

In one sentence: **PCA loses exactly the spread along the axes it throws away.** Let’s compute it for our archive. The snippet finds the eigenvalues by **power iteration**: multiply a starting vector by \(A\) over and over and rescale it to length 1; the stretch along the longest axis wins, so the vector turns towards \(u_1\). Then it removes that axis from \(A\) (subtract \(\lambda_1 u_1 u_1^\top\)), a step called **deflation**, and repeats.

```python
X = [(3, 2, 0), (2, 2, 1), (4, 3, 0),
     (0, 1, 3), (1, 0, 4), (0, 0, 2)]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
A = [[sum(x[i] * x[j] for x in X) for j in range(3)]
     for i in range(3)]
eigenvalues = []
for _ in range(3):
    u = [1.0, 0.5, 0.25]
    for _ in range(500):
        u = [dot(row, u) for row in A]
        u = [a / dot(u, u) ** 0.5 for a in u]
    lam = dot(u, [dot(row, u) for row in A])
    eigenvalues.append(lam)
    A = [[A[i][j] - lam * u[i] * u[j] for j in range(3)]
         for i in range(3)]
print([round(l, 2) for l in eigenvalues])
print(round(sum(eigenvalues[1:]), 2), round(eigenvalues[2], 2))
```

It prints `[49.87, 26.94, 1.19]` and `28.13 1.19`. The eigenvalues add up to 78, the archive’s total squared length. With one stored number, PCA loses \(26.94 + 1.19 = 28.13\), better than the junk direction’s 32. With two numbers it loses only 1.19, better than the hand-made squeeze’s 2.0: the six messages lie very close to a plane, and PCA finds that plane.

[[diagram:pca]]

**Centering.** In practice one usually subtracts the average message first: compute \(\mu = \frac1m \sum_{i=1}^m x_i\), the **mean**, and run PCA on \(x_1 - \mu, \dots, x_m - \mu\). This measures spread around the centre of the cloud rather than around the origin, which is what “the direction in which the cloud is longest” means for a cloud that sits away from the origin. The problems at the end connect this to the variance of the stored number.

## More words than messages

Now back to the real mail. Say you want PCA on an archive of \(m = 2{,}000\) messages with \(d = 50{,}000\) words. The scatter matrix \(A\) is \(50{,}000 \times 50{,}000\): two and a half billion entries, and computing its eigenvectors takes time of order \(d^3\), about \(10^{14}\) operations. Building \(A\) costs another \(md^2\). But there are only 2,000 messages. Surely the interesting directions can be found from them?

They can. Stack the messages as the rows of a matrix \(X \in \mathbb R^{m,d}\), so row \(i\) is \(x_i^\top\). Then the scatter matrix is \(A = X^\top X\) (entry \((j, k)\) of \(X^\top X\) is \(\sum_i x_{i,j} x_{i,k}\), the same sum as before). Now flip the product around:

\[ B = X X^\top \in \mathbb R^{m,m}, \qquad B_{i,j} = x_i \cdot x_j . \]

- \(B\) has one row and one column per message, so it is \(m \times m\), here \(2{,}000 \times 2{,}000\).
- Entry \((i, j)\) is the dot product of message \(i\) with message \(j\): how much the two messages overlap, word by word.

In one sentence: **\(B\) is the table of overlaps between every pair of messages**, often called the Gram matrix. For two tiny messages, \(x_1 = (1, 0, 2, 0, 0, 1)\) and \(x_2 = (0, 3, 0, 0, 1, 1)\) over six words, \(B_{1,1} = 1 + 4 + 1 = 6\), \(B_{2,2} = 9 + 1 + 1 = 11\), and \(B_{1,2} = 1\), because the only word they share is the last.

The trick: eigenvectors of the small \(B\) give eigenvectors of the big \(A\). Suppose \(Bv = \lambda v\). Multiply both sides on the left by \(X^\top\):

\[ X^\top X X^\top v = \lambda\, X^\top v, \quad\text{that is}\quad A\,(X^\top v) = \lambda\,(X^\top v). \]

The left side regroups \(X^\top (XX^\top) v\) as \((X^\top X)(X^\top v)\), and \(X^\top X = A\). So \(X^\top v\), rescaled to length 1 as \(u = X^\top v / \|X^\top v\|\), is an eigenvector of \(A\) with the same eigenvalue \(\lambda\) (provided \(X^\top v \ne 0\), which holds whenever \(\lambda > 0\), since \(\|X^\top v\|^2 = v^\top B v = \lambda \|v\|^2\)). In one sentence: **find the eigenvectors of the overlap table, then turn each into a mix of messages.** Notice what \(X^\top v\) is: \(\sum_i v_i x_i\), a weighted sum of the messages. Every principal component is a mix of the archive’s own messages.

Here it is on the two tiny messages, checking \(Au = \lambda u\) without ever building the \(6 \times 6\) matrix \(A\) (we compute \(Au\) as \(\sum_i x_i (x_i \cdot u)\)):

```python
X = [(1, 0, 2, 0, 0, 1),
     (0, 3, 0, 0, 1, 1)]
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
B = [[dot(xi, xj) for xj in X] for xi in X]
a, b, d = B[0][0], B[0][1], B[1][1]
lam = (a + d) / 2 + (((a - d) / 2) ** 2 + b * b) ** 0.5
v = (b, lam - a)
Xt_v = [v[0] * p + v[1] * q for p, q in zip(*X)]
u = [t / dot(Xt_v, Xt_v) ** 0.5 for t in Xt_v]
Au = [sum(x[k] * dot(x, u) for x in X) for k in range(6)]
gap = max(abs(p - lam * q) for p, q in zip(Au, u))
print(B, round(lam, 3), [round(t, 3) for t in u])
print(gap < 1e-9)
```

It prints `[[6, 1], [1, 11]] 11.193 [0.057, 0.881, 0.113, 0.0, 0.294, 0.35]` and `True`. (For a \(2 \times 2\) symmetric matrix the top eigenvalue has the closed form used in line 6, and \((b, \lambda - a)\) is its eigenvector.) The principal component leans heavily on the second message, the one with more squared length.

The cost changes from \(O(d^3)\) for the eigenvectors plus \(O(md^2)\) to build \(A\), to \(O(m^3)\) for the eigenvectors of \(B\) plus \(O(m^2 d)\) to build \(B\). For our archive that is roughly \(8 \times 10^9\) plus \(2 \times 10^{11}\), instead of more than \(10^{14}\). And because \(B\) needs only dot products between messages, the same computation works whenever you can compute dot products, even in a feature space too large to write down. You met that idea in lesson 16 as the kernel trick; here it gives **kernel PCA**, and one of this lesson’s problems asks you to work it out.

## PCA as code

Putting the pieces together, the algorithm picks whichever matrix is smaller.

```
PCA(X, n):
  input: X, an m × d matrix whose rows are the messages
         n, the number of components
  if m > d:
    A ← XᵀX
    u_1, …, u_n ← eigenvectors of A with the n largest eigenvalues
  else:
    B ← X Xᵀ
    v_1, …, v_n ← eigenvectors of B with the n largest eigenvalues
    for i = 1..n:  u_i ← Xᵀ v_i / ‖Xᵀ v_i‖
  output: u_1, …, u_n
```

Compressing a message is \(y = (u_1 \cdot x, \dots, u_n \cdot x)\), recovering it is \(\tilde x = \sum_j y_j u_j\).

**Cost.** The two routes take the time found in the previous section, plus \(O(d^2)\) or \(O(m^2)\) space for the matrix, and \(O(nmd)\) to turn \(n\) eigenvectors of \(B\) into components. Storing the components is \(O(nd)\). After that, compressing or recovering one message costs \(O(nd)\) time, or \(O(ns)\) to compress a message with only \(s\) nonzero counts. If you need only the top \(n\) components, power iteration with deflation, as in the snippet above, costs \(O(k^2)\) per multiplication for a \(k \times k\) matrix, times the number of iterations, times \(n\); how many iterations it needs depends on the gap between consecutive eigenvalues.

**A demonstration.** Let’s watch PCA find a line we planted. Draw points \((x, x + y)\) with \(x\) uniform on \([-1, 1]\) and a small Gaussian noise \(y\) with mean 0 and standard deviation 0.1. The points lie near the diagonal line, so the top principal component should be close to \((1/\sqrt 2, 1/\sqrt 2) \approx (0.707, 0.707)\). The snippet centres the points, builds the \(2 \times 2\) scatter matrix, and takes its top eigenvector in closed form:

```python
import random
rng = random.Random(1)
pts = []
for _ in range(1000):
    x = rng.uniform(-1, 1)
    pts.append((x, x + rng.gauss(0, 0.1)))
mu = [sum(p[k] for p in pts) / len(pts) for k in (0, 1)]
c = [(p[0] - mu[0], p[1] - mu[1]) for p in pts]
a = sum(p[0] * p[0] for p in c)
b = sum(p[0] * p[1] for p in c)
d = sum(p[1] * p[1] for p in c)
lam = (a + d) / 2 + (((a - d) / 2) ** 2 + b * b) ** 0.5
u = (b, lam - a)
norm = (u[0] ** 2 + u[1] ** 2) ** 0.5
print([round(t / norm, 3) for t in u])
```

It prints `[0.703, 0.711]`. Projecting a point \((x, x + y)\) onto \((1/\sqrt2, 1/\sqrt 2)\) stores the single number \((2x + y)/\sqrt 2\), and recovering it gives \((x + y/2,\; x + y/2)\): the noise is split evenly between the two coordinates and the point lands on the diagonal.

**Build it yourself.** Using only the Python standard library:

```python
import random

X = [(3, 2, 0), (2, 2, 1), (4, 3, 0),
     (0, 1, 3), (1, 0, 4), (0, 0, 2)]

def draw(m, d, words, seed):
    rng = random.Random(seed)
    msgs = []
    for _ in range(m):
        x = [0] * d
        for _ in range(words):
            x[rng.randrange(d)] += 1
        msgs.append(x)
    return msgs

def top_eigen(M, n, iters=2000): ...
def pca(X, n, use_gram=None): ...
def compress(U, x): ...
def recover(U, y): ...
def objective(U, X): ...

if __name__ == "__main__":
    assert round(objective(pca(X, 1), X), 2) == 28.13
    assert round(objective(pca(X, 2), X), 2) == 1.19
    U = pca(X, 2, use_gram=True)
    assert round(objective(U, X), 2) == 1.19
    wide = draw(8, 3000, 40, 7)
    U = pca(wide, 3)
    assert round(objective(U, wide), 2) == 196.43
    for i in range(3):
        for j in range(3):
            assert abs(sum(a * b for a, b in zip(U[i], U[j]))
                       - (i == j)) < 1e-9
    print("all asserts pass")
```

1. Implement `top_eigen(M, n)`: power iteration with deflation on a symmetric matrix given as a list of rows, returning the \(n\) pairs \((\lambda, u)\) with the largest eigenvalues first. Start each iteration from a vector that is not perpendicular to the answer (for instance \(u_i = 1/(i + 1 + t)\) for the \(t\)-th eigenvector).
2. Implement `pca(X, n, use_gram)`: build \(A = X^\top X\) when `use_gram` is false, and \(B = XX^\top\) with the \(X^\top v / \|X^\top v\|\) step when it is true; when it is `None`, choose as the pseudocode does. Return the components as a list of \(n\) vectors.
3. Implement `compress`, `recover` and `objective` (the total squared error). The first two asserts are the worked example; the third checks that both routes agree.
4. The `wide` archive has 8 messages over 3,000 words. Your `pca` must take the Gram route here: building a \(3{,}000 \times 3{,}000\) scatter matrix in pure Python would take minutes. Before running it, predict the error at \(n = 3\) from the formula for the leftover error: the archive’s total squared length is 326 and the top three eigenvalues of \(B\) are 44.6, 43.18 and 41.79.

## Where PCA breaks

PCA is the best linear squeeze on the archive you trained it on. Two things about the mail room make that less comforting than it sounds.

The first is the shape of word-count data. PCA wins when the messages sit near a low-dimensional subspace, like our six messages near a plane. Word counts do not look like that. Take the extreme case: a vocabulary of \(d\) words, and an archive of \(d\) messages, where message \(i\) is the single word \(i\), once. That message is the vector \(e_i\), all zeros except a 1 in position \(i\) (the **standard basis** vectors). Before reading on, how much can PCA keep with \(n\) numbers per message?

Very little. The scatter matrix is \(A = \sum_i e_i e_i^\top = I\), every eigenvalue is 1, every direction is as good as every other, and the leftover error is the sum of the \(d - n\) discarded eigenvalues, \(d - n\). Per message, that is an average error of \((d - n)/d\), out of a total squared length of 1. With \(d = 10\) and \(n = 3\):

```python
d, n = 10, 3
E = [[int(i == j) for j in range(d)] for i in range(d)]
U = E[:n]
err = 0
for x in E:
    proj = [sum(u[k] * x[k] for k in range(d)) for u in U]
    rec = [sum(p * u[k] for p, u in zip(proj, U))
           for k in range(d)]
    err += sum((a - b) ** 2 for a, b in zip(x, rec))
print(err / d, (d - n) / d)
```

It prints `0.7 0.7`: seven of the ten messages vanish entirely. Whenever \(n \le d/2\), the average error is more than half of each message. Yet each of these messages is described by one number and one word index; they are about as simple as messages get. PCA cannot see that kind of simplicity, because it looks only for a subspace.

The second problem is time. PCA fits the archive it was given. Next week’s messages use other words, and their mass outside the span of last week’s archive is simply lost: every principal component is a mix of last week’s messages, so it is zero on every word that last week never used. And the archive changes every day.

What we want for the nearest-neighbour filter is different: a squeeze chosen once, before any message arrives, that keeps every distance between messages nearly the same, whatever messages arrive. That sounds too good to exist. It exists, and it is random.

## Random projections

Here is the plan, in words. Fill a compression matrix \(W\) with random numbers, chosen before you see any message. Squeeze every message with it. Then prove that, with high probability, every distance between messages comes out nearly unchanged. No archive, no eigenvectors, no fitting.

Why would distances survive a random squeeze? Think of what one row of \(W\) does: it takes a random weighted sum of the counts. Its square is a noisy reading of the message’s squared length, too small sometimes, too large other times, but right on average. With \(n\) independent rows, the average of \(n\) noisy readings is close to the truth, for the same reason that the empirical risk of lesson 4 is close to the true risk. The surprise is how few rows suffice, and that the number does not depend on \(d\) at all.

**Distortion.** Squeezing two messages \(x_1\) and \(x_2\) with \(W\) **does not distort their distance much** when the ratio

\[ \frac{\|Wx_1 - Wx_2\|}{\|x_1 - x_2\|} \]

is close to 1. The numerator is the distance after squeezing, the denominator the distance before. In one sentence: **the squeeze keeps a distance when the distance after is nearly the distance before.** Because \(W\) is linear, \(Wx_1 - Wx_2 = W(x_1 - x_2)\), so the ratio is \(\|Wx\|/\|x\|\) with \(x = x_1 - x_2\), the difference of the two messages. From now on we study one vector \(x\) and the ratio \(\|Wx\|/\|x\|\); keeping distances is keeping the lengths of differences.

**Gaussian entries.** The random numbers we use are **normal** (Gaussian). A standard normal random variable, written \(N(0, 1)\), has the familiar bell-shaped distribution with mean 0 and variance 1; \(N(0, \sigma^2)\) is the same bell stretched to variance \(\sigma^2\). We need one fact about it, which we use without proof: **if \(w_1, \dots, w_d\) are independent \(N(0, 1)\) and \(x \in \mathbb R^d\) is fixed, then \(\sum_j w_j x_j\) is again normal, with mean 0 and variance \(\sum_j x_j^2 = \|x\|^2\).** Here is a message with seven counts, \(x = (3, 0, 1, 2, 0, 4, 1)\), \(\|x\|^2 = 31\), and 100,000 random rows:

```python
import random
rng = random.Random(2)
x = [3, 0, 1, 2, 0, 4, 1]
norm2_x = sum(t * t for t in x)
draws = []
for _ in range(100000):
    w = [rng.gauss(0, 1) for _ in x]
    draws.append(sum(a * b for a, b in zip(w, x)))
mean = sum(draws) / len(draws)
var = sum((t - mean) ** 2 for t in draws) / len(draws)
print(norm2_x, round(mean, 2), round(var, 1))
```

It prints `31 0.01 30.8`: mean near 0, variance near 31. So one row’s reading, squared, is \(\|x\|^2\) times the square of a standard normal, and on average it equals \(\|x\|^2\).

Now a real message. It has five nonzero counts among 50,000 words, and we squeeze it with \(n = 400\) Gaussian rows. Only five columns of \(W\) ever touch it, so the code generates just those, each from its own seed (column \(j\) is always the same 400 numbers). The quantity printed is \(\|Wx\|^2 / (n\|x\|^2)\): the squared length after squeezing, divided by \(n\) because each of the \(n\) rows contributes \(\|x\|^2\) on average.

```python
import random
n = 400
x = {7: 3, 1205: 1, 4410: 2, 30001: 1, 49999: 4}

def column(j):
    rng = random.Random(j)
    return [rng.gauss(0, 1) for _ in range(n)]

Wx = [0.0] * n
for j, count in x.items():
    Wx = [a + count * b for a, b in zip(Wx, column(j))]
norm2_x = sum(c * c for c in x.values())
ratio = sum(a * a for a in Wx) / n / norm2_x
print(norm2_x, round(ratio, 3))
```

It prints `31 1.127`: the squared length came out 12.7% too large. Is that typical, or bad luck? The next result answers exactly that.

### One vector at a time

To state it we need one more name. If \(X_1, \dots, X_k\) are independent standard normals, the sum of their squares \(Z = X_1^2 + \dots + X_k^2\) has the **chi-squared distribution with \(k\) degrees of freedom**, written \(Z \sim \chi^2_k\) (“\(Z\) is drawn from chi-squared \(k\)”). Each \(X_i^2\) has mean 1, so \(\mathbb E[Z] = k\), where \(\mathbb E\) is the expected value, the long-run average.

**Lemma (chi-squared concentration).** Let \(Z \sim \chi^2_k\). Then for every \(\varepsilon \in (0, 1)\), and for every \(\varepsilon \in (0, 1/2]\) respectively,

\[ \mathbb P\big[Z \le (1 - \varepsilon)k\big] \le e^{-\varepsilon^2 k/6}, \qquad \mathbb P\big[Z \ge (1 + \varepsilon)k\big] \le e^{-\varepsilon^2 k / 6}. \]

- \(\mathbb P[\cdot]\) is the probability of the event in brackets.
- \((1 - \varepsilon)k\) and \((1 + \varepsilon)k\) are the edges of a band of relative width \(\varepsilon\) around the mean \(k\).
- \(\varepsilon \in (0, 1/2]\) means \(\varepsilon\) is above 0 and at most one half.

In one sentence: **a sum of \(k\) squared standard normals strays from \(k\) by a fraction \(\varepsilon\) with probability that shrinks like \(e^{-\varepsilon^2 k/6}\).** With \(k = 200\) and \(\varepsilon = 0.3\), each tail has probability at most \(e^{-3} \approx 0.0498\), so straying in either direction has probability at most about 0.0996.

*Proof.* Both tails use the same move, called **Chernoff’s method**: turn the event into one about an exponential, then apply Markov’s inequality. Markov’s inequality (used in lesson 4) says that a random variable \(Y \ge 0\) satisfies \(\mathbb P[Y \ge a] \le \mathbb E[Y]/a\) for every \(a > 0\).

*Lower tail.* Fix \(\lambda > 0\). Since \(t \mapsto e^{-\lambda t}\) is decreasing, \(Z \le (1 - \varepsilon) k\) happens exactly when \(e^{-\lambda Z} \ge e^{-\lambda(1-\varepsilon)k}\). Markov’s inequality gives

\[ \mathbb P\big[Z \le (1-\varepsilon)k\big] \le e^{\lambda(1-\varepsilon)k}\, \mathbb E\big[e^{-\lambda Z}\big] = e^{\lambda(1-\varepsilon)k}\, \Big(\mathbb E\big[e^{-\lambda X_1^2}\big]\Big)^k . \]

The equality uses independence: \(e^{-\lambda Z} = \prod_i e^{-\lambda X_i^2}\), the expectation of a product of independent factors is the product of their expectations, and all \(k\) factors have the same distribution. Next, \(e^{-a} \le 1 - a + a^2/2\) for every \(a \ge 0\). (The difference \(g(a) = 1 - a + a^2/2 - e^{-a}\) is 0 at \(a = 0\), and its derivative \(-1 + a + e^{-a}\) is at least 0 because \(e^{-a} \ge 1 - a\).) With \(a = \lambda X_1^2\), and using the two moments \(\mathbb E[X_1^2] = 1\) and \(\mathbb E[X_1^4] = 3\) of a standard normal,

\[ \mathbb E\big[e^{-\lambda X_1^2}\big] \le 1 - \lambda + \tfrac32 \lambda^2 \le e^{-\lambda + \frac32\lambda^2}, \]

where the last step is \(1 + t \le e^t\). Put this back:

\[ \mathbb P\big[Z \le (1-\varepsilon)k\big] \le e^{\lambda(1-\varepsilon)k - \lambda k + \frac32 \lambda^2 k} = e^{-\varepsilon k \lambda + \frac32 k\lambda^2}. \]

This holds for every \(\lambda > 0\), so pick the best: \(\lambda = \varepsilon/3\) gives the exponent \(-\varepsilon^2 k/3 + \varepsilon^2 k/6 = -\varepsilon^2 k/6\).

*Upper tail.* Here we need the exact exponential moment of one squared normal. For \(\lambda < 1/2\),

\[ \mathbb E\big[e^{\lambda X_1^2}\big] = \int_{-\infty}^{\infty} e^{\lambda t^2} \frac{e^{-t^2/2}}{\sqrt{2\pi}}\,dt = \int_{-\infty}^{\infty} \frac{e^{-(1 - 2\lambda)t^2/2}}{\sqrt{2\pi}}\,dt = (1 - 2\lambda)^{-1/2}. \]

The fraction \(e^{-t^2/2}/\sqrt{2\pi}\) is the bell curve of a standard normal, so the first integral is the definition of the expectation. The two exponentials merge into one. The last step substitutes \(s = t\sqrt{1 - 2\lambda}\), which turns the integrand into the bell curve again (whose integral is 1) times \(dt/ds = (1-2\lambda)^{-1/2}\). In one sentence: **squaring a standard normal and exponentiating with rate \(\lambda\) averages to \((1 - 2\lambda)^{-1/2}\).** Both moments are easy to check by simulation:

```python
import math
import random
rng = random.Random(4)
lam = 0.2
xs = [rng.gauss(0, 1) for _ in range(400000)]
exp_moment = sum(math.exp(lam * x * x) for x in xs) / len(xs)
fourth = sum(x ** 4 for x in xs) / len(xs)
print(round(exp_moment, 3), round((1 - 2 * lam) ** -0.5, 3))
print(round(fourth, 2))
```

It prints `1.289 1.291` and `2.97`, against the exact values 1.291 and 3.

Now the same Chernoff move, with \(e^{+\lambda Z}\) and \(0 < \lambda < 1/2\):

\[ \mathbb P\big[Z \ge (1+\varepsilon)k\big] \le e^{-\lambda(1+\varepsilon)k}\, \mathbb E\big[e^{\lambda Z}\big] = e^{-\lambda(1+\varepsilon)k}\,(1 - 2\lambda)^{-k/2}. \]

Choose \(\lambda = \frac{\varepsilon}{2(1+\varepsilon)}\), which is below 1/2. Then \(1 - 2\lambda = \frac{1}{1+\varepsilon}\) and \(\lambda(1+\varepsilon) = \varepsilon/2\), so the bound becomes

\[ e^{-\varepsilon k/2}\,(1+\varepsilon)^{k/2} = \exp\Big(-\tfrac{k}{2}\big(\varepsilon - \ln(1+\varepsilon)\big)\Big). \]

It remains to show \(\varepsilon - \ln(1 + \varepsilon) \ge \varepsilon^2/3\) when \(\varepsilon \le 1/2\). First, \(\ln(1+\varepsilon) \le \varepsilon - \varepsilon^2/2 + \varepsilon^3/3\) for \(\varepsilon \ge 0\): the difference \(f(\varepsilon) = \varepsilon - \varepsilon^2/2 + \varepsilon^3/3 - \ln(1+\varepsilon)\) is 0 at 0 and has derivative \(1 - \varepsilon + \varepsilon^2 - \frac{1}{1+\varepsilon} = \frac{\varepsilon^3}{1 + \varepsilon} \ge 0\). So \(\varepsilon - \ln(1+\varepsilon) \ge \varepsilon^2/2 - \varepsilon^3/3\), and when \(\varepsilon \le 1/2\) we have \(\varepsilon^3/3 \le \varepsilon^2/6\), leaving at least \(\varepsilon^2/3\). The exponent is then at most \(-\frac k2 \cdot \frac{\varepsilon^2}{3} = -\varepsilon^2 k/6\). \(\blacksquare\)

The restriction \(\varepsilon \le 1/2\) is not fussiness: at \(\varepsilon = 1\) the inequality fails, and so does the clean bound itself. The first snippet checks the inequality on a grid up to 1/2 and at \(\varepsilon = 1\). The second computes the exact chance that a \(\chi^2_{400}\) variable reaches 800, using the exact upper tail of \(\chi^2_k\) at \(t\) for an even \(k\), \(\sum_{j < k/2} e^{-t/2}(t/2)^j / j!\).

```python
import math
grid = [k / 1000 for k in range(1, 501)]
ok = all(e - math.log1p(e) >= e * e / 3 for e in grid)
print(ok, round(1 - math.log(2), 4), round(1 / 3, 4))
```

```python
import math
eps, k = 1.0, 400
t = (1 + eps) * k / 2
tail, term = 0.0, math.exp(-t)
for j in range(k // 2):
    tail += term
    term *= t / (j + 1)
print(f"{tail:.1e} {math.exp(-eps ** 2 * k / 6):.1e}")
```

They print `True 0.3069 0.3333` (at \(\varepsilon = 1\) the left side is 0.3069, below 1/3) and `6.2e-29 1.1e-29` (the true tail is more than five times the would-be bound \(e^{-400/6}\)). Everything in this lesson uses \(\varepsilon \le 1/2\).

**Lemma (one vector survives a random squeeze).** Fix \(x \in \mathbb R^d\), \(x \ne 0\). Let \(W \in \mathbb R^{n,d}\) have independent \(N(0,1)\) entries. Then for every \(\varepsilon \in (0, 1/2]\),

\[ \mathbb P\left[\, \left| \frac{\|(1/\sqrt n)\,Wx\|^2}{\|x\|^2} - 1 \right| > \varepsilon \right] \le 2\, e^{-\varepsilon^2 n/6}. \]

- \((1/\sqrt n)W\) is the matrix \(W\) with every entry divided by \(\sqrt n\); dividing the squared length by \(n\) undoes the “each row contributes \(\|x\|^2\) on average” effect we saw.
- The fraction is the squared distortion ratio; subtracting 1 and taking the absolute value \(|\cdot|\) measures how far it is from perfect.
- The probability is over the random draw of \(W\); \(x\) is fixed first.

In one sentence: **for any one fixed message, a random squeeze to \(n\) numbers gets its squared length wrong by more than a fraction \(\varepsilon\) with probability at most \(2e^{-\varepsilon^2 n/6}\).** For the message above, \(n = 400\) and \(\varepsilon = 0.3\) give at most \(2e^{-6} \approx 0.005\): our 12.7% error was well within the band.

*Proof.* Divide \(x\) by its length; both sides are unchanged, so assume \(\|x\| = 1\). Let \(w_i\) be row \(i\) of \(W\). By the Gaussian fact, \(w_i \cdot x\) is \(N(0, \|x\|^2) = N(0, 1)\). The rows are independent, so the \(n\) values \(w_1 \cdot x, \dots, w_n \cdot x\) are independent standard normals, and \(\|Wx\|^2 = \sum_{i=1}^n (w_i \cdot x)^2\) is \(\chi^2_n\). The event in the lemma is that \(\|Wx\|^2\) leaves the band \([(1-\varepsilon)n, (1+\varepsilon)n]\); it is the union of the two tail events, and by the union bound (lesson 2: the probability that one of several events happens is at most the sum of their probabilities) its probability is at most \(e^{-\varepsilon^2 n/6} + e^{-\varepsilon^2 n/6}\). \(\blacksquare\)

Here is the lemma as a simulation: draw a \(\chi^2_{200}\) variable 20,000 times and count how often \(Z/n\) leaves the band of width 0.3.

```python
import math
import random
rng = random.Random(3)
n, eps, trials = 200, 0.3, 20000
bad = 0
for _ in range(trials):
    Z = sum(rng.gauss(0, 1) ** 2 for _ in range(n))
    bad += abs(Z / n - 1) > eps
print(bad / trials, round(2 * math.exp(-eps ** 2 * n / 6), 4))
```

It prints `0.00375 0.0996`. The bound holds with plenty of room: it is a guarantee, not an estimate.

[[diagram:distortion]]

## The Johnson-Lindenstrauss lemma

One message is not enough. The filter needs every pairwise distance in the archive at once. The fix is the union bound again: if each of \(|Q|\) vectors fails with probability at most \(2e^{-\varepsilon^2 n/6}\), then some vector fails with probability at most \(2|Q|e^{-\varepsilon^2 n/6}\). Set that equal to the failure probability \(\delta\) you can live with, and solve for \(n\).

With numbers: an archive of 200 messages has \(200 \cdot 199/2 = 19{,}900\) pairs. You want every squared distance within 30% (\(\varepsilon = 0.3\)) with probability at least 95% (\(\delta = 0.05\)). Before reading on, guess the order of magnitude of \(n\): tens, hundreds, or thousands?

To keep the next formula readable, we put the \(1/\sqrt n\) inside the matrix: from now on, a **random projection** is \(x \mapsto Wx\) where every entry of \(W \in \mathbb R^{n,d}\) is independent \(N(0, 1/n)\) (a standard normal divided by \(\sqrt n\)).

**Lemma (Johnson-Lindenstrauss).** Let \(Q\) be a finite set of nonzero vectors in \(\mathbb R^d\), let \(\delta \in (0, 1)\), and let \(n\) be an integer such that

\[ \varepsilon = \sqrt{\frac{6 \ln(2|Q|/\delta)}{n}} \le \frac12 . \]

Then, with probability at least \(1 - \delta\) over the choice of a random projection \(W \in \mathbb R^{n,d}\),

\[ \sup_{x \in Q} \left| \frac{\|Wx\|^2}{\|x\|^2} - 1 \right| \le \varepsilon . \]

- \(|Q|\) is how many vectors \(Q\) contains; for distances in an archive of \(m\) messages, \(Q\) is the set of differences \(x_i - x_j\), \(i < j\), and \(|Q| = m(m-1)/2\).
- \(\ln\) is the natural logarithm.
- \(\sup_{x \in Q}\) is the largest value over all \(x\) in \(Q\) (for a finite set it is just the maximum).
- The definition of \(\varepsilon\) can be turned around: the guarantee holds at accuracy \(\varepsilon\) as soon as \(n \ge 6\ln(2|Q|/\delta)/\varepsilon^2\).

In one sentence: **with probability at least \(1 - \delta\), a random projection to \(n = 6\ln(2|Q|/\delta)/\varepsilon^2\) numbers keeps every squared length in \(Q\) within a fraction \(\varepsilon\), and \(n\) does not depend on \(d\).**

*Proof.* \(W\) is \((1/\sqrt n)\) times a matrix with standard normal entries, so for each fixed \(x \in Q\) the one-vector lemma says \(\big|\|Wx\|^2/\|x\|^2 - 1\big| > \varepsilon\) with probability at most \(2e^{-\varepsilon^2 n/6}\) (this needs \(\varepsilon \le 1/2\), which the hypothesis provides). By the union bound over the \(|Q|\) vectors, the probability that the supremum exceeds \(\varepsilon\) is at most \(2|Q|e^{-\varepsilon^2 n/6}\). With \(\varepsilon\) as defined, \(\varepsilon^2 n/6 = \ln(2|Q|/\delta)\), so \(2|Q|e^{-\varepsilon^2 n/6} = 2|Q| \cdot \delta/(2|Q|) = \delta\). \(\blacksquare\)

Our archive, term by term:

```python
import math
m, eps, delta = 200, 0.3, 0.05
Q = m * (m - 1) // 2
n = math.ceil(6 * math.log(2 * Q / delta) / eps ** 2)
print(Q, n)
```

It prints `19900 906`. Nine hundred and six numbers per message instead of 50,000, and the guarantee covers all 19,900 distances at once. Notice what is not in the formula: \(d\). The same 906 would do for a vocabulary of five million words. The dimension you need grows only with the logarithm of the number of distances you want to keep.

Here is the lemma at work on 30 random messages of about 20 words each, out of 50,000. The formula with \(|Q| = 435\) pairs gives \(n = 651\); the code squeezes all 30 messages and reports the worst squared-distance distortion over the 435 pairs.

```python
import math
import random
d, m, n = 50000, 30, 651
rng = random.Random(8)
msgs = [{rng.randrange(d): rng.randint(1, 5) for _ in range(20)}
        for _ in range(m)]
cols = {}
def col(j):
    if j not in cols:
        r = random.Random(j)
        cols[j] = [r.gauss(0, 1 / math.sqrt(n)) for _ in range(n)]
    return cols[j]
proj = [[sum(c * col(j)[i] for j, c in x.items())
         for i in range(n)] for x in msgs]
def sq(a, b):
    return sum((a.get(k, 0) - b.get(k, 0)) ** 2 for k in a | b)
worst = max(abs(math.dist(proj[i], proj[j]) ** 2
                / sq(msgs[i], msgs[j]) - 1)
            for i in range(m) for j in range(i))
print(round(worst, 3))
```

It prints `0.212`, inside the promised 0.3.

Two warnings before we use this. First, the guarantee is about squared lengths; for lengths themselves, \(1 - \varepsilon \le \|Wx\|^2/\|x\|^2 \le 1 + \varepsilon\) gives \(\sqrt{1-\varepsilon} \le \|Wx\|/\|x\| \le \sqrt{1+\varepsilon}\), a slightly tighter band. Second, and more important, **the set \(Q\) must be fixed before \(W\) is drawn**. The probability is over \(W\) for a given \(Q\). If someone picks the messages after seeing \(W\), they can pick two messages whose difference lies in the null space of \(W\) (it always has one, since \(n < d\)), and that distance is squeezed to 0. In the mail room this means: draw \(W\) once, keep it secret from senders, and the guarantee covers any batch of messages that does not depend on it.

[[diagram:jl]]

## Random projection as code

```
RandomProjection(n, d, seed):
  W ← n × d matrix, each entry drawn independently from N(0, 1/n)
  output: x ↦ W x

JLDimension(|Q|, ε, δ):
  requires: 0 < ε ≤ 1/2
  output: ⌈ 6 ln(2|Q|/δ) / ε² ⌉
```

**Cost.** Drawing \(W\) takes \(O(nd)\) time and \(O(nd)\) space, or no stored space at all if each column is regenerated from its own seed, as in the snippets. Squeezing one message costs \(O(nd)\), or \(O(ns)\) for a message with \(s\) nonzero counts (only \(s\) columns are touched). Comparing two squeezed messages then costs \(O(n)\) instead of \(O(d)\). For the 200-message archive at \(n = 906\) with about 60 words per message, squeezing everything is about \(200 \cdot 60 \cdot 906 \approx 10.9\) million multiplications, done once, with no fitting at all.

**Build it yourself.**

```python
import math
import random

def draw(m, d, words, seed):
    rng = random.Random(seed)
    return [{rng.randrange(d): rng.randint(1, 5)
             for _ in range(words)} for _ in range(m)]

def column(j, n):
    rng = random.Random(j)
    return [rng.gauss(0, 1 / math.sqrt(n)) for _ in range(n)]

def jl_dimension(size_Q, eps, delta): ...
def project(msgs, n): ...
def sq_dist(a, b): ...
def max_distortion(msgs, proj): ...

if __name__ == "__main__":
    assert jl_dimension(19900, 0.3, 0.05) == 906
    assert jl_dimension(435, 0.3, 0.05) == 651
    msgs = draw(30, 50000, 20, 8)
    proj = project(msgs, 651)
    assert round(max_distortion(msgs, proj), 3) == 0.212
    proj = project(msgs, 100)
    assert round(max_distortion(msgs, proj), 3) == 0.52
    print("all asserts pass")
```

1. Implement `jl_dimension` from the lemma.
2. Implement `project(msgs, n)`: messages are dictionaries from word index to count; build each squeezed message from the columns `column(j, n)` of the words it uses, generating each column once.
3. Implement `sq_dist` for two dictionaries and `max_distortion`, the largest \(\big|\|Wx_i - Wx_j\|^2/\|x_i - x_j\|^2 - 1\big|\) over pairs. The asserts check you against the snippet above, and against \(n = 100\), where the worst pair is distorted by more than half.
4. Redraw the matrix twenty times (give `column` a second seed, for instance `random.Random(j + 100000 * t)` for draw \(t\)) and run `max_distortion` at \(n = 651\) on the same thirty messages each time. How many of the twenty land under 0.3? How many does the lemma promise?

## Fix the bug

Back to the slow mail room. The nearest-neighbour filter needs the distances between this week’s messages, and your manager wants a promise, not a hope: every one of the pairwise squared distances among this week’s 200 messages must stay within 30% of its true value (\(\varepsilon = 0.3\)), with probability at least 95% (\(\delta = 0.05\)). The budget is at most 1,000 stored numbers per message. And there is a rule of the house: the squeeze is chosen on Monday morning, before this week’s messages arrive, and each message is squeezed as soon as it comes in.

You have two candidates. PCA, fitted on last week’s 200 messages, with up to 199 components. Or a random projection, with \(n\) of your choice. Before you open the lab, decide which one the theory can certify and which \(n\) it needs. Then think about this: the lab also shows the measured worst distortion on this week’s actual messages, and at \(n = 500\) it happens to come out at 0.243, under 0.3. Is \(n = 500\) good enough?

[[lab-act2]]

Measuring is not certifying. At \(n = 500\) the Johnson-Lindenstrauss bound on the failure probability is \(2 \cdot 19{,}900\, e^{-0.09 \cdot 500/6} \approx 22\), which promises nothing (a probability bound above 1 is no bound). The draw you see happened to be fine; next week’s draw has no such promise. Only \(n \ge 906\) carries the 95% guarantee, and it fits the budget. PCA cannot be certified at all: its components are mixes of last week’s messages, which used about 10,700 of the 50,000 words, and on average almost four fifths of each new message’s squared length falls on words outside them and is lost.

## What people get wrong

**“PCA is the best way to reduce dimension.”** It is the best linear squeeze with linear recovery, measured by total squared error, on the archive it was fitted to. With a nonlinear decoder, sparse messages can be recovered from far fewer numbers than PCA needs (lesson 23b), and for messages outside the archive PCA promises nothing.

**“A random matrix loses information at random, so it must be worse than a fitted one.”** For keeping distances, the random projection is the one that comes with a guarantee for messages it has never seen, and its dimension depends on how many distances you keep, not on \(d\). Its weakness is different: the messages must not be chosen after seeing \(W\).

**“The Johnson-Lindenstrauss dimension grows with the number of words.”** There is no \(d\) in \(n = 6\ln(2|Q|/\delta)/\varepsilon^2\). Doubling the vocabulary changes nothing; doubling the archive adds about \(6 \ln 4/\varepsilon^2\) rows.

## Recap and the next bug

1. A **compression matrix** \(W \in \mathbb R^{n,d}\) squeezes a message into \(n\) weighted sums; with \(n < d\) its null space makes exact recovery of every message impossible.
2. **PCA** minimizes the total squared round-trip error over linear squeezes and recoveries. The best pair is \(W = U^\top\) with orthonormal columns, the columns are the eigenvectors of the scatter matrix \(A = \sum_i x_i x_i^\top\) for its \(n\) largest eigenvalues, and the error left over is the sum of the discarded eigenvalues. When \(d \gg m\), eigenvectors of the \(m \times m\) Gram matrix \(XX^\top\) give the same components at cost \(O(m^3 + m^2 d)\).
3. A **random projection** with \(N(0, 1/n)\) entries keeps the squared length of any one fixed vector within \(\varepsilon\) except with probability \(2e^{-\varepsilon^2 n/6}\) (for \(\varepsilon \le 1/2\)), by chi-squared concentration and Chernoff’s method. The **Johnson-Lindenstrauss lemma** adds a union bound: \(n = 6\ln(2|Q|/\delta)/\varepsilon^2\) numbers keep a whole set \(Q\), independently of \(d\).
4. PCA bets on a subspace and fits the archive, so it promises nothing for messages it has not seen. A random projection bets on nothing about the messages, certifies the distances of any batch chosen before it was drawn, and fixes the mail room’s speed. Measuring one draw is not certifying: at \(n = 500\) the measured worst distortion was 0.243 and the lemma’s bound was 22.

The filter is fast again, and for a week nobody complains. Then the compliance desk writes in. A customer is disputing a quarantined message, and the desk has to show exactly which watchlist words it contained and how often. You open the archive: 906 numbers per message, each a weighted sum of the counts. You try to reverse them, and you remember why you cannot. Every squeeze with \(n < d\) has a null space, and the random projection was built to keep distances, not words. Yet look at the message in dispute. It has about 60 nonzero counts out of 50,000. Does the null space care about a message that is almost all zeros? Could a cleverer decoder, using nothing but the fact that most counts are zero, read the words back exactly, from far fewer than 50,000 numbers? That is the next lesson: compressed sensing.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. All six messages of a new archive lie on the plane where the “meeting” count is exactly 1 (so the plane does not pass through the origin), and the messages are not all on one line. Is uncentered PCA with \(n = 2\) exact on this archive, meaning total error 0? Is PCA with \(n = 2\) exact after centering? Use the formula for the leftover error, and say what you must store besides the two numbers per message in the centered case.
2. In the proof of the Johnson-Lindenstrauss lemma, the union bound runs over the vectors of \(Q\). Where does the proof use that \(Q\) is fixed before \(W\) is drawn? Show that the conclusion can fail completely if a sender may choose two messages after seeing \(W\), whatever \(n < d\) is.
3. Your archive grows to 1,000 messages and you want all pairwise squared distances within 25% with probability at least 99%. How many numbers per message does the Johnson-Lindenstrauss lemma ask for? What changes if the vocabulary grows from 50,000 to five million words? And a teammate ships the 1,000-message archive at \(n = 1{,}000\) numbers per message because the worst distortion on her one draw came out under 0.25: what does the lemma’s failure bound say at that \(n\)?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (PCA as maximizing variance).** Let \(x_1, \dots, x_m \in \mathbb R^d\) and let \(x\) be a random vector equal to each \(x_i\) with probability \(1/m\); assume \(\mathbb E[x] = 0\). (a) Show that the unit vector \(w\) maximizing the variance of the stored number, \(\operatorname{Var}[\langle w, x\rangle] = \frac1m \sum_i \langle w, x_i\rangle^2\), is the first principal component. (b) Let \(w_1\) be the first principal component. Show that among unit vectors \(w\) with \(\mathbb E[\langle w_1, x\rangle\langle w, x\rangle] = 0\) (the two stored numbers are uncorrelated), the variance of \(\langle w, x\rangle\) is maximized by the second principal component.
- **Problem 2 (random projections keep dot products).** Let \(Q\) be a finite set of vectors in \(\mathbb R^d\) with \(\|x\| \le 1\) for every \(x \in Q\). Let \(\delta \in (0, 1)\) and let \(n\) be such that \(\varepsilon = \sqrt{6\ln(2|Q|^2/\delta)/n} \le 1/2\). Prove that with probability at least \(1 - \delta\) over a random projection \(W \in \mathbb R^{n,d}\), \(|\langle Wu, Wv\rangle - \langle u, v\rangle| \le \varepsilon\) for every \(u, v \in Q\).
- **Problem 3 (kernel PCA).** Let \(x_1, \dots, x_m\) be points of any domain \(\mathcal X\), let \(\psi\) map \(\mathcal X\) into a possibly infinite-dimensional space with a dot product, and let \(K(x, x') = \langle \psi(x), \psi(x')\rangle\) be a kernel (lesson 16) that can be evaluated in constant time. Kernel PCA applies PCA to \(\psi(x_1), \dots, \psi(x_m)\) and outputs each point’s \(n\) stored numbers. Show how to compute these outputs in time polynomial in \(m\) and \(n\), using only evaluations of \(K\). Check that every matrix product and every eigendecomposition you use involves only finite matrices you can actually compute.
