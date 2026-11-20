# Multiclass, Ranking and Complex Prediction

## Four filters that argue

For sixteen lessons the filter has answered one question: junk or keep. Your users have stopped being grateful for that. They want their mail sorted into four **folders**, work, personal, promotions and junk, and they want the inbox itself in order, with the messages that matter at the top of the morning’s list.

The team did what any sensible team does first. You already own a good learner for yes/no questions, the halfspace of lesson 9, trained with the hinge loss of lesson 15. So they trained four of them. One answers “is this work?”, one “is this personal?”, one “is this a promotion?”, one “is this junk?”. Each was trained on the same labelled messages, relabelled as “this folder” against “everything else”. To file a message, ask all four and pick the folder whose filter says yes.

Within a day the bug reports describe two kinds of nonsense. Some messages get a yes from two filters at once, a message that is both work and a promotion, and the folder it lands in depends on which filter the code happened to ask first. Other messages get a no from all four filters, so they belong nowhere, and they all end up in the first folder in the list. When you dig in, there is a whole kind of message that is never filed correctly, however much data you feed the filters: the routine work mail, the meeting notes and status updates that read a bit like personal mail and a bit like promotions.

The ranking is worse. To order the inbox the team trained one more yes/no filter, “is this message important?”, and put the yes answers on top. The report on it looks wonderful: 98% of its answers are right. Then you notice that only 2% of messages are important, and the filter has learned to say no to every single one. Its error is 2%, it is useless, and the inbox is sorted by nothing at all.

So there are two puzzles. Each yes/no filter is a perfectly good learner for its own question; why does bolting them together give contradictions and a whole folder of mistakes? And why is a ranker with 98% accuracy worthless? This lesson answers both, and then builds learners that ask the right question from the start: a linear learner that chooses among many labels at once, a loss that charges each mistake what it really costs, and ranking losses that score the top of the list.

## Bolting yes/no filters together

Let’s name what the team built, because it is the most common first attempt and it deserves a fair trial. The setting is **multiclass** prediction: we want a hypothesis \(h: \mathcal X \to \mathcal Y\) where the label set \(\mathcal Y\) is a finite set of \(k\) labels, written \(\mathcal Y = \{1, \dots, k\}\). For us, \(\mathcal X\) is the set of messages, as word-count vectors in \(\mathbb R^n\) since lesson 9, and the four folders are labels 1 (work), 2 (personal), 3 (promotions) and 4 (junk).

The idea of a **reduction** is to answer a multiclass question with a learner built for binary questions. Two reductions are standard.

**One-versus-all.** Given a training sequence \(S = ((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m))\), build \(k\) binary training sequences. In \(S_i\), every example keeps its message and gets the label \(+1\) if its folder is \(i\) and \(-1\) otherwise. If four messages are filed in folders 1, 3, 1 and 4, the work filter (\(i = 1\)) sees the labels \(+1, -1, +1, -1\). Written with lesson 1’s indicator \(\mathbb 1[\text{statement}]\), which is 1 when the statement is true and 0 when it is false:

\[ S_i = \left( (\mathbf x_1, (-1)^{\mathbb 1[y_1 \neq i]}), \dots, (\mathbf x_m, (-1)^{\mathbb 1[y_m \neq i]}) \right). \]

Read it from the inside out.

- \(\mathbb 1[\text{statement}]\), the indicator from lesson 1, is 1 when the statement is true and 0 when it is false. Here the statement is \(y_t \neq i\): “example \(t\) is *not* in folder \(i\)”.
- \((-1)\) raised to that indicator is \((-1)^0 = +1\) when example \(t\) is in folder \(i\), and \((-1)^1 = -1\) when it is not. It is a compact way to write a yes/no label as \(\pm 1\).
- Each pair \((\mathbf x_t, \pm 1)\) keeps the message exactly as it was; only the label changes.
- The outer parentheses list all \(m\) pairs in the original order, so \(S_i\) is again a training sequence of length \(m\).

In plain words: **\(S_i\) is the same training sequence with every folder replaced by \(+1\) if it is folder \(i\) and \(-1\) if it is any other folder.** Take four messages filed in folders 1, 3, 1, 4. For \(i = 1\) the labels become \(+1, -1, +1, -1\); for \(i = 3\) they become \(-1, +1, -1, -1\).

```python
folders = [1, 3, 1, 4]
for i in (1, 3):
    S_i = [(f"m{t}", (-1) ** int(y != i))
           for t, y in enumerate(folders, 1)]
    print(i, S_i)
```

It prints `1 [('m1', 1), ('m2', -1), ('m3', 1), ('m4', -1)]`, then `3 [('m1', -1), ('m2', 1), ('m3', -1), ('m4', -1)]`.

Run a binary learner \(A\) on each \(S_i\) to get \(h_i = A(S_i)\), and predict with

\[ h(\mathbf x) \in \operatorname*{argmax}_{i \in [k]} h_i(\mathbf x). \]

The “\(\in\)” admits what the team discovered: when several \(h_i\) say \(+1\), or none does, the argmax is a tie and something has to break it, for instance taking the smallest index. When each \(h_i\) is a halfspace \(\operatorname{sign}(\langle \mathbf w_i, \mathbf x \rangle)\), there is a better option. (Recall from lesson 9 that \(\langle \mathbf u, \mathbf v \rangle\) is the inner product, “multiply matching entries and add”, so \(\langle (2, -1), (1, 1) \rangle = 2 - 1 = 1\).) The number \(\langle \mathbf w_i, \mathbf x \rangle\) before the sign can be read as the filter’s confidence, and we can take the argmax over those real numbers instead, which rarely ties.

Read the prediction rule from the inside out.

- \(h_i(\mathbf x)\) is filter \(i\)’s answer for message \(\mathbf x\): \(+1\) for yes, \(-1\) for no.
- \([k]\) is lesson 1’s shorthand for \(\{1, \dots, k\}\), so \(i \in [k]\) runs over all \(k\) folders.
- \(\operatorname{argmax}_{i \in [k]}\) is not the largest value but *where* it occurs: the set of indices \(i\) whose answer is largest. It is a set because several indices can share the largest value, which is why the rule says \(h(\mathbf x) \in\) that set rather than \(=\).

In plain words: **file the message in a folder whose filter gives the largest answer, and break any tie somehow.** If the four filters answer \(-1, +1, +1, -1\), the argmax is the set \(\{2, 3\}\), and the smallest-index rule picks folder 2.

```python
h = {1: -1, 2: 1, 3: 1, 4: -1}
top = max(h.values())
argmax = [i for i in h if h[i] == top]
print(argmax, min(argmax))
```

It prints `[2, 3] 2`.

**All-pairs.** For every pair of folders \(i < j\), collect only the examples whose folder is \(i\) or \(j\), label them \(+1\) for \(i\) and \(-1\) for \(j\), and train \(h_{i,j}\) on that. That is \(\binom{k}{2}\) binary learners, six for our four folders. To predict, every pair votes, and the folder with the most wins is the answer. With three folders, if the pair filters say that 2 beats 1, 1 beats 3 and 2 beats 3, folder 2 has two wins and is the answer. In symbols:

\[ h(\mathbf x) \in \operatorname*{argmax}_{i \in \mathcal Y} \sum_{j \in \mathcal Y} \operatorname{sign}(j - i)\, h_{i,j}(\mathbf x), \]

where for \(i > j\) we read \(h_{i,j}\) as \(h_{j,i}\), and the factor \(\operatorname{sign}(j - i)\) flips its vote so that a \(+1\) always counts for the lower-numbered folder of the pair, and the term with \(j = i\) is zero.

Read the vote from the inside out.

- \(h_{i,j}(\mathbf x)\) is the pair filter’s answer: \(+1\) means it prefers the lower-numbered folder of the pair, \(-1\) the higher one.
- \(\operatorname{sign}(j - i)\) is \(+1\) when \(i\) is the lower-numbered folder of the pair, \(-1\) when it is the higher one, and 0 when \(j = i\).
- Their product is \(+1\) exactly when the pair filter voted for \(i\), and \(-1\) when it voted against \(i\).
- \(\sum_{j \in \mathcal Y}\) adds that \(\pm 1\) over every rival \(j\), so the sum is folder \(i\)’s wins minus its losses.
- The argmax over \(i\) picks the folder with the best tally.

In plain words: **file the message in the folder with the most pairwise wins minus losses.** With three folders, say the pair filters answer \(h_{1,2} = -1\) (folder 2 beats 1), \(h_{1,3} = +1\) (1 beats 3) and \(h_{2,3} = +1\) (2 beats 3). Folder 1 has one win and one loss, tally 0; folder 2 has two wins, tally 2; folder 3 has two losses, tally \(-2\). Folder 2 wins.

```python
h = {(1, 2): -1, (1, 3): 1, (2, 3): 1}
sign = lambda a: (a > 0) - (a < 0)
def tally(i):
    return sum(sign(j - i) * h[min(i, j), max(i, j)]
               for j in (1, 2, 3) if j != i)
print([tally(i) for i in (1, 2, 3)])
```

It prints `[0, 2, -2]`.

Both reductions are simple and reuse code you already trust. Their weakness is in one sentence: the binary learner has no idea its answers will be combined. Each \(h_i\) is trained to be as accurate as possible on its own question, and nothing in that goal asks whether the \(k\) answers will fit together. Let’s watch that go wrong with real numbers.

## Worked example: the folder in the middle

Shrink the world so you can draw it. Describe each message by two numbers, so it is a point in the plane. Messages from three folders sit in three discs of radius 1, side by side at the same height: personal mail around \((-5, 5)\), work around \((0, 5)\), promotions around \((5, 5)\). Within its disc, a message is spread uniformly. Of all messages, 40% are personal, 20% are work and 40% are promotions. (The two coordinates are invented scores; think of the first as “how commercial the wording is” and the second as a length the three folders share. The geometry is what matters.)

[[diagram:ova-trap]]

Before reading on, consider the filter “work against the rest”. The work disc is squeezed between the other two. Which halfspace would you choose for it?

Try a few. A line that cuts the work disc off from the left also takes all of personal. A line that cuts it off from the right takes all of promotions. A horizontal line cutting the top cap of the work disc cuts the same caps off its neighbours, and they carry twice the mass each. Every halfspace that captures some work mail pays more in the other folders than it gains. The best halfspace is the one that says no to everything, with error 20%, exactly the work mail. We will prove that in a moment.

Now assemble the team’s filter. “Personal against the rest” and “promotions against the rest” are easy: the vertical lines \(x_1 = -2.5\) and \(x_1 = 2.5\) separate them perfectly, so the best halfspaces have error 0. Take a work message. All three filters say no. The argmax is a three-way tie, and breaking it by the smallest index sends it to personal. Every work message is misfiled. The filter that was built from three optimal pieces errs on 20% of all mail.

Is the hypothesis class to blame? No. Here are three weight vectors, one per folder:

\[ \mathbf w_{\text{personal}} = \left(-\tfrac{1}{\sqrt 2}, \tfrac{1}{\sqrt 2}\right), \quad \mathbf w_{\text{work}} = (0, 1), \quad \mathbf w_{\text{promotions}} = \left(\tfrac{1}{\sqrt 2}, \tfrac{1}{\sqrt 2}\right), \]

and the rule “file \(\mathbf x\) in the folder whose weight vector gives the largest \(\langle \mathbf w_y, \mathbf x \rangle\)”.

Two reminders first. \(\langle \mathbf u, \mathbf v \rangle\) is the inner product of lesson 9: multiply matching entries and add, \(u_1 v_1 + u_2 v_2\) in the plane. \(\|\mathbf u\|\) is the length, \(\sqrt{\langle \mathbf u, \mathbf u \rangle}\). Now read the three vectors.

- The subscript names the folder the vector belongs to; nothing more.
- \(\tfrac{1}{\sqrt 2} \approx 0.707\), so \(\mathbf w_{\text{personal}}\) points up and to the left at 45 degrees, \(\mathbf w_{\text{work}}\) straight up, and \(\mathbf w_{\text{promotions}}\) up and to the right.
- Each has length 1: \(0.707^2 + 0.707^2 = 1\).
- The score \(\langle \mathbf w_y, \mathbf x \rangle\) measures how far the message reaches in folder \(y\)’s direction.

In plain words: **three arrows of length 1, pointing up-left, straight up and up-right, and each message goes to the folder whose arrow it reaches furthest along.** The centre of the work disc, \((0, 5)\), scores \(3.54\) for personal, \(5\) for work and \(3.54\) for promotions, so it goes to work.

```python
import math

s = 1 / math.sqrt(2)
W = {"personal": (-s, s), "work": (0, 1), "promotions": (s, s)}
x = (0, 5)
for y, w in W.items():
    print(y, round(math.hypot(*w), 3),
          round(w[0] * x[0] + w[1] * x[1], 2))
```

It prints `personal 1.0 3.54`, then `work 1.0 5`, then `promotions 1.0 3.54`.

Check a work message \(\mathbf x = (x_1, x_2)\) by hand. It scores \(x_2\) for work, \((x_2 - x_1)/\sqrt 2\) for personal and \((x_1 + x_2)/\sqrt 2\) for promotions. Work wins both comparisons exactly when \((\sqrt 2 - 1)\, x_2 > |x_1|\). In the work disc \(x_2 \ge 4\) and \(|x_1| \le 1\), and \(0.414 \times 4 = 1.66 > 1\), so work always wins. A personal message has \(x_1 \le -4\) and \(x_2 \le 6\), so personal beats work because \(-x_1 \ge 4 > 0.414 \times 6\), and beats promotions because \(x_1 < 0\). Promotions is the mirror image. Zero mistakes.

Python draws 10,000 messages and checks all three filters: one-versus-all with ties to the smallest index, all-pairs with the three vertical lines as its pair filters, and the three weight vectors. In the code the folders are 0, 1 and 2, left to right.

```python
import math
import random

rng, centers = random.Random(0), [(-5, 5), (0, 5), (5, 5)]
def draw():
    y = rng.choices([0, 1, 2], weights=[4, 2, 4])[0]
    a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
    return (centers[y][0] + r * math.cos(a),
            centers[y][1] + r * math.sin(a)), y
S = [draw() for _ in range(10_000)]
h = [lambda x: x[0] < -2.5, lambda x: False, lambda x: x[0] > 2.5]
ova = lambda x: max(range(3), key=lambda i: (h[i](x), -i))
pair = {(0, 1): -2.5, (0, 2): 0.0, (1, 2): 2.5}
wins = lambda x, i: sum((x[0] < c) == (i == a) for (a, b), c
                        in pair.items() if i in (a, b))
allpairs = lambda x: max(range(3), key=lambda i: wins(x, i))
s = 1 / math.sqrt(2)
W = [(-s, s), (0, 1), (s, s)]
mv = lambda x: max(range(3), key=lambda i: W[i][0] * x[0]
                   + W[i][1] * x[1])
for p in (ova, allpairs, mv):
    print(f"{sum(p(x) != y for x, y in S) / len(S):.3f}")
```

It prints `0.200`, `0.000` and `0.000`. One-versus-all misfiles exactly the work mail, while the other two make no mistakes. All-pairs survives this particular world because every pair of discs can be split by a line, but it has the same blindness in general: each of its learners sees only two folders and is never told how its vote will be counted.

Hold on to that pair of facts. The class of rules “take the folder with the largest \(\langle \mathbf w_y, \mathbf x \rangle\)” contains a perfect hypothesis, so its approximation error, in the sense of lesson 5 (the true risk of the best rule in the class), is zero. And one-versus-all, run with a perfect binary learner, still cannot find it. The fault is in how the pieces are trained, not in what they can express.

## Why the middle folder is lost

Let’s make the claim about the work filter exact, because “try a few lines” is not a proof.

**Proposition (the middle folder has no good halfspace).** In the three-disc world, every halfspace has error at least 20% on the binary problem “work against the rest”. The error is exactly 20% only if the halfspace contains a zero-probability part of the work disc.

*Proof.* Write a halfspace as \(H = \{\mathbf x : \langle \mathbf u, \mathbf x \rangle + b > 0\}\) with \(\|\mathbf u\| = 1\); it predicts \(+1\) (work) inside \(H\) and \(-1\) outside. The two constant classifiers, all \(+1\) and all \(-1\), have errors 80% and 20%, so the claim holds for them, and from now on \(H\) is a genuine halfspace.

A message of folder \(i\) is \(\mathbf c_i + \mathbf z\), where \(\mathbf c_i\) is the centre of its disc and \(\mathbf z\) is uniform on the unit disc. It lands in \(H\) when \(\langle \mathbf u, \mathbf z \rangle + t_i > 0\), where \(t_i = \langle \mathbf u, \mathbf c_i \rangle + b\) is the signed distance from the centre to the boundary line. The uniform distribution on the unit disc looks the same in every direction, so the probability of that event depends only on \(t_i\), not on \(\mathbf u\). Call it \(g(t_i)\). As \(t\) grows the event only gets larger, so \(g\) is nondecreasing, and of course \(0 \le g \le 1\).

The error of \(H\) adds three parts: personal mail inside \(H\), promotions inside \(H\), and work mail outside \(H\):

\[ \operatorname{err}(H) = 0.4\, g(t_1) + 0.4\, g(t_3) + 0.2\,\bigl(1 - g(t_2)\bigr), \]

with \(t_1, t_2, t_3\) for personal, work and promotions.

Read this formula from the inside out before using it.

- \(t_i\) is the signed distance from folder \(i\)’s centre to the boundary line: positive when the centre is inside \(H\), negative when outside.
- \(g(t_i)\) is the probability that a message of folder \(i\) lands inside \(H\), so \(1 - g(t_2)\) is the probability that a work message lands outside.
- \(0.4\, g(t_1)\) is personal mail that \(H\) wrongly calls work: 40% of all mail times the share of it inside \(H\). The same for \(0.4\, g(t_3)\) and promotions.
- \(0.2\,(1 - g(t_2))\) is work mail that \(H\) misses.

In plain words: **the error of \(H\) is the personal and promotions mail it wrongly lets in plus the work mail it leaves out.** Try the horizontal line \(x_2 = 5.5\), with “inside” above it. All three centres are 0.5 below it, so \(t_1 = t_2 = t_3 = -0.5\), and \(g(-0.5)\) is the area of a unit disc’s cap of depth 0.5 divided by \(\pi\), which is 0.196. The error is \(0.4 \times 0.196 \times 2 + 0.2 \times 0.804 = 0.317\), worse than saying no to everything.

```python
import math

cap = lambda t: math.acos(t) - t * math.sqrt(1 - t * t)
g = lambda t: 1 - cap(t) / math.pi
t1 = t2 = t3 = -0.5
err = 0.4 * g(t1) + 0.4 * g(t3) + 0.2 * (1 - g(t2))
print(round(g(-0.5), 3), round(err, 3))
```

It prints `0.196 0.317`.

Now the geometry enters. The work centre is the midpoint of the other two, \(\mathbf c_2 = (\mathbf c_1 + \mathbf c_3)/2\), and \(t\) is an affine function of the centre, so \(t_2 = (t_1 + t_3)/2\). An average cannot exceed both of the numbers it averages, so \(\max(t_1, t_3) \ge t_2\), and because \(g\) is nondecreasing, \(g(t_1) + g(t_3) \ge g(\max(t_1, t_3)) \ge g(t_2)\). Substitute:

\[ \operatorname{err}(H) \ge 0.4\, g(t_2) + 0.2 - 0.2\, g(t_2) = 0.2 + 0.2\, g(t_2) \ge 0.2. \]

Read the chain left to right. The first “\(\ge\)” replaces \(0.4\,(g(t_1) + g(t_3))\) by the smaller \(0.4\, g(t_2)\), which is exactly what the previous sentence proved. The “\(=\)” collects the two \(g(t_2)\) terms: \(0.4 - 0.2 = 0.2\). The last “\(\ge\)” drops \(0.2\, g(t_2)\), which is never negative. In plain words: **whatever share of work mail a halfspace catches, it lets in at least as much other mail, so its error never drops below the 20% of saying no.** With the line \(x_2 = 5.5\): the error 0.317 is at least \(0.2 + 0.2 \times 0.196 = 0.239\), which is at least 0.2.

```python
import math

cap = lambda t: math.acos(t) - t * math.sqrt(1 - t * t)
g = lambda t: 1 - cap(t) / math.pi
t1, t2, t3 = -0.5, -0.5, -0.5
err = 0.4 * g(t1) + 0.4 * g(t3) + 0.2 * (1 - g(t2))
middle = 0.4 * g(t2) + 0.2 - 0.2 * g(t2)
low = 0.2 + 0.2 * g(t2)
print(round(err, 3), round(middle, 3), round(low, 3))
```

It prints `0.317 0.239 0.239`.

Equality forces \(g(t_2) = 0\): the halfspace catches work mail with probability 0. \(\blacksquare\)

So the best halfspace for the work question is, up to a set of probability zero, one that never says yes on work mail, and the same goes for the perfect personal and promotions filters, which say no on all work mail. Every work message is a three-way tie of noes, and a tie broken by the smallest index files it as personal. One-versus-all errs on at least 20% of the mail, even though a perfect rule of the same shape exists. The proof also tells you exactly what went wrong: the work filter was graded on a question, “work or not?”, whose best answer is “never”, and nobody graded the combination.

The same experiment with data: sweep 360 directions and every threshold, and find the halfspace with the fewest mistakes on 2,000 random messages.

```python
import math
import random

rng, centers = random.Random(1), [(-5, 5), (0, 5), (5, 5)]
def draw():
    y = rng.choices([0, 1, 2], weights=[4, 2, 4])[0]
    a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
    x = (centers[y][0] + r * math.cos(a),
         centers[y][1] + r * math.sin(a))
    return x, (1 if y == 1 else -1)
S = [draw() for _ in range(2_000)]
best = len(S)
for t in (math.radians(deg) for deg in range(360)):
    err = sum(y == -1 for _, y in S)
    c, s = math.cos(t), math.sin(t)
    for _, y in sorted((c * x[0] + s * x[1], y) for x, y in S):
        err += 1 if y == 1 else -1
        best = min(best, err)
work = sum(y == 1 for _, y in S)
print(f"{best / len(S):.3f} {work / len(S):.3f}")
```

It prints `0.210 0.210`: the best halfspace on the sample errs on exactly the fraction of the sample that is work mail, which is what saying no to everything costs.

Now play with it. In the lab you can move the discs and their shares of the mail, switch between one-versus-all and a learner that picks one weight vector per folder, and drag those weight vectors yourself. The second panel is the inbox: slide a threshold over the importance scores and watch accuracy, and the numbers we will define for ranking, move in different directions.

[[lab]]

## One score per folder

The weight vectors that fixed the three discs suggest the cure: train all the folders together, so that the only thing that matters is which folder scores highest. Let’s build that idea properly, starting from something you already know.

A halfspace predicts \(\operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\). Here is the same prediction written as a choice between two labels:

\[ h(\mathbf x) = \operatorname*{argmax}_{y \in \{\pm 1\}} \langle \mathbf w, y\mathbf x \rangle, \]

where \(y\mathbf x\) is \(\mathbf x\) with every entry multiplied by \(y\). If \(\langle \mathbf w, \mathbf x \rangle > 0\), then \(y = +1\) gives the positive number and \(y = -1\) its negative, so the argmax is \(+1\); if \(\langle \mathbf w, \mathbf x \rangle < 0\) it is \(-1\). (When the score is exactly 0 both labels tie, the same boundary case as \(\operatorname{sign}(0)\) in lesson 9.) A halfspace scores each label by an inner product with a vector that depends on both the message and the label, and returns the label with the highest score. That shape is what generalizes.

Read it from the inside out.

- \(y\) ranges over the two labels \(\{\pm 1\}\), that is \(+1\) and \(-1\).
- \(y\mathbf x\) is the message with every entry multiplied by \(y\): \(\mathbf x\) itself for \(y = +1\), its mirror image for \(y = -1\).
- \(\langle \mathbf w, y\mathbf x \rangle = y\langle \mathbf w, \mathbf x \rangle\) is the score of label \(y\).
- The argmax returns the label with the larger score.

In plain words: **a halfspace picks whichever of the two labels gives the larger score \(\langle \mathbf w, y\mathbf x \rangle\).** With \(\mathbf w = (2, -1)\) and \(\mathbf x = (1, 1)\), \(\langle \mathbf w, \mathbf x \rangle = 1\), so label \(+1\) scores 1 and label \(-1\) scores \(-1\): the answer is \(+1\), which is \(\operatorname{sign}(1)\).

```python
w, x = (2, -1), (1, 1)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
score = {y: dot(w, [y * xi for xi in x]) for y in (1, -1)}
print(score, max(score, key=score.get))
```

It prints `{1: 1, -1: -1} 1`.

**Definition (class-sensitive feature mapping).** A **class-sensitive feature mapping** is a function \(\Psi: \mathcal X \times \mathcal Y \to \mathbb R^d\). It takes a message together with a candidate label and returns a vector. You can read the entries of \(\Psi(\mathbf x, y)\) as scores that say how well the label \(y\) fits the message \(\mathbf x\).

**Definition (linear multiclass predictor).** Given \(\Psi\) and a weight vector \(\mathbf w \in \mathbb R^d\), the **linear multiclass predictor** is

\[ h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y) \rangle. \]

Read it from the inside out.

- \(\Psi(\mathbf x, y)\) is a vector in \(\mathbb R^d\) built from the message *and* one candidate label.
- \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle\) collapses that vector to one number, using the same \(\mathbf w\) for every label.
- \(\operatorname{argmax}_{y \in \mathcal Y}\) tries every label in \(\mathcal Y\) and returns the one with the largest number.
- The subscript \(\mathbf w\) in \(h_{\mathbf w}\) says that the predictor is determined by the weight vector.

In plain words: **compute one score per label and return the label with the highest score.** Say a message has \(\Psi(\mathbf x, \text{work}) = (2, 0)\), \(\Psi(\mathbf x, \text{personal}) = (0, 1)\), \(\Psi(\mathbf x, \text{promotions}) = (1, 1)\), and \(\mathbf w = (1, 3)\). The scores are 2, 3 and 4, so \(h_{\mathbf w}(\mathbf x)\) is promotions.

```python
psi = {"work": (2, 0), "personal": (0, 1), "promotions": (1, 1)}
w = (1, 3)
score = {y: w[0] * v[0] + w[1] * v[1] for y, v in psi.items()}
print(score, max(score, key=score.get))
```

It prints `{'work': 2, 'personal': 3, 'promotions': 4} promotions`.

The number \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle\) is the **score** of label \(y\) for message \(\mathbf x\). For a set \(W \subseteq \mathbb R^d\) of allowed weight vectors, for example the ball \(W = \{\mathbf w : \|\mathbf w\| \le B\}\), the pair \((\Psi, W)\) defines a hypothesis class

\[ \mathcal H_{\Psi, W} = \left\{ \mathbf x \mapsto \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y) \rangle \;:\; \mathbf w \in W \right\}. \]

Read it from the inside out.

- \(\mathbf x \mapsto \dots\) reads “the function that sends \(\mathbf x\) to \(\dots\)”: here, the predictor \(h_{\mathbf w}\).
- The braces with a colon, \(\{\, \dots : \mathbf w \in W \}\), read “the set of all \(\dots\) such that \(\mathbf w\) is in \(W\)”: one predictor for every allowed weight vector.
- \(W = \{\mathbf w : \|\mathbf w\| \le B\}\) is the set of weight vectors no longer than \(B\), a ball around the origin.

In plain words: **the hypothesis class is every predictor you get by plugging in an allowed weight vector.** With the three \(\Psi\) vectors above and \(B = 1\), the weights \((1, 0)\) and \((0, 1)\) are allowed and predict work (scores 2, 0, 1) and personal (scores 0, 1, 1, a tie that the argmax breaks by order), while \((1, 3)\) has length 3.16 and is not in the class at all.

```python
import math

psi = {"work": (2, 0), "personal": (0, 1), "promotions": (1, 1)}
for w in ((1, 0), (0, 1), (1, 3)):
    s = {y: w[0] * v[0] + w[1] * v[1] for y, v in psi.items()}
    allowed = math.hypot(*w) <= 1
    print(w, allowed, max(s, key=s.get) if allowed else "-")
```

It prints `(1, 0) True work`, then `(0, 1) True personal`, then `(1, 3) False -`.

With \(\mathcal Y = \{\pm 1\}\), \(\Psi(\mathbf x, y) = y\mathbf x\) and \(W = \mathbb R^d\), this is exactly the class of homogeneous halfspaces. The whole question is now how to choose \(\Psi\), and that is a question of prior knowledge, the inductive bias of lesson 1 in a new costume. Two constructions show the range.

**The multivector construction.** Let \(\mathcal X = \mathbb R^n\) and \(\mathcal Y = \{1, \dots, k\}\), and set \(d = nk\). Define

\[ \Psi(\mathbf x, y) = [\, \underbrace{0, \dots, 0}_{\in \mathbb R^{(y-1)n}},\ \underbrace{x_1, \dots, x_n}_{\in \mathbb R^n},\ \underbrace{0, \dots, 0}_{\in \mathbb R^{(k-y)n}} \,]. \]

Read it piece by piece.

- \(\underbrace{0, \dots, 0}_{\in \mathbb R^{(y-1)n}}\) is a run of zeros; the label under the brace says how many: \((y - 1)n\), one block of \(n\) zeros for every label before \(y\).
- \(x_1, \dots, x_n\) is the message itself, \(n\) entries.
- The last run has \((k - y)n\) zeros, one block for every label after \(y\).
- Altogether \((y - 1)n + n + (k - y)n = kn = d\) entries.

In plain words: **\(\Psi(\mathbf x, y)\) is \(k\) slots of length \(n\), with the message copied into slot \(y\) and zeros everywhere else.** With \(k = 4\), \(n = 3\), \(\mathbf x = (1, 2, 0)\) and \(y = 2\), it is \((0,0,0,\ 1,2,0,\ 0,0,0,\ 0,0,0)\).

```python
k, n, x, y = 4, 3, (1, 2, 0), 2
psi = [0] * ((y - 1) * n) + list(x) + [0] * ((k - y) * n)
print(psi, len(psi))
```

It prints `[0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0] 12`.

So \(\Psi(\mathbf x, y)\) is \(k\) blocks of length \(n\), all zero except block \(y\), which holds \(\mathbf x\). Cut \(\mathbf w \in \mathbb R^{nk}\) into the same blocks, \(\mathbf w = [\mathbf w_1; \dots; \mathbf w_k]\). In the inner product only block \(y\) of \(\mathbf w\) meets a nonzero block of \(\Psi\), so

\[ \langle \mathbf w, \Psi(\mathbf x, y) \rangle = \langle \mathbf w_y, \mathbf x \rangle, \qquad h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w_y, \mathbf x \rangle. \]

Read it from the inside out.

- \(\mathbf w_y\) is block \(y\) of \(\mathbf w\), the \(n\) weights that belong to label \(y\).
- On the left, the inner product runs over all \(nk\) entries, but every entry of \(\Psi\) outside block \(y\) is 0, so only block \(y\) contributes.
- On the right, the same sum written with just that block.
- The second formula is the predictor with the left side replaced by the right.

In plain words: **with the multivector map, the score of label \(y\) is just the inner product of the message with label \(y\)’s own weight vector.** With \(\mathbf w = (2,0,-1 \mid 0,3,0 \mid -1,0,2 \mid -2,-1,3)\) and \(\mathbf x = (1, 2, 0)\), block 2 is \((0, 3, 0)\) and the score of label 2 is \(0 + 6 + 0 = 6\). The snippet a few lines below mirrors both sides for every label.

One weight vector per folder, largest score wins: the rule that fixed the three discs. The name comes from this picture of \(\mathbf w\) as several vectors stacked into one.

[[diagram:multivector]]

```python
k, n = 4, 3
def psi(x, y):
    return [0] * (y * n) + list(x) + [0] * ((k - 1 - y) * n)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w = [2, 0, -1,  0, 3, 0,  -1, 0, 2,  -2, -1, 3]
x = (1, 2, 0)
scores = [dot(w, psi(x, y)) for y in range(k)]
blocks = [dot(w[y * n:(y + 1) * n], x) for y in range(k)]
print(scores, blocks, scores.index(max(scores)))
```

It prints `[2, 6, -1, -4] [2, 6, -1, -4] 1`: the long inner product with \(\Psi\) and the short inner product with block \(y\) agree for every label, and label 1 (personal, in the code’s 0-based numbering) wins.

**A construction that knows about words.** The multivector map assumes nothing about the problem; every folder gets its own copy of every weight. Here is a \(\Psi\) that builds in prior knowledge about text. Let \(\mathcal X\) be the set of messages, \(\mathcal Y\) a set of topics, and fix a dictionary of \(d\) words. For word \(j\), let \(\mathrm{TF}(j, \mathbf x)\), the **term frequency**, be the number of times word \(j\) appears in message \(\mathbf x\). Let \(\mathrm{DF}(j, y)\), the **document frequency**, be the number of times word \(j\) appears in the training messages that are *not* about topic \(y\): it measures how common the word is in other topics. With \(m\) training messages, set

\[ \Psi_j(\mathbf x, y) = \mathrm{TF}(j, \mathbf x) \log\left( \frac{m}{\mathrm{DF}(j, y)} \right). \]

Read it from the inside out.

- \(\mathrm{DF}(j, y)\) counts word \(j\) in the training messages of every topic except \(y\).
- \(m / \mathrm{DF}(j, y)\) is large when the word is rare outside \(y\). The \(\log\) is the natural logarithm; it shrinks large ratios so one rare word cannot dominate.
- \(\mathrm{TF}(j, \mathbf x)\) multiplies by how often the word occurs in this message.
- The subscript \(j\) says this is entry \(j\) of the vector \(\Psi(\mathbf x, y)\); there is one entry per dictionary word.

In plain words: **entry \(j\) is “how often the word is in this message” times “how rare the word is outside topic \(y\)”.** In the corpus below, “meeting” appears twice in \(\mathbf x\) and once in the six training messages outside work, so its work entry is \(2 \log(6/1) = 3.58\). The snippet after the next paragraph mirrors the formula term by term.

This is called **term-frequency inverse-document-frequency**, or **TF-IDF**. The entry is large when word \(j\) appears often in \(\mathbf x\) and rarely outside topic \(y\), which is evidence that \(\mathbf x\) is about \(y\). Unlike the multivector map, its dimension is the dictionary size, whatever the number of topics. (The formula needs \(\mathrm{DF}(j, y) > 0\); a word that never appears outside topic \(y\) would get an infinite entry, so in practice such words are handled specially. In our example every word appears in at least two topics.)

```python
import math

S = [("meeting agenda meeting", "work"), ("agenda sale", "work"),
     ("dinner with mom", "personal"),
     ("dinner meeting", "personal"),
     ("sale sale agenda", "promo"), ("dinner sale", "promo")]
words, m = ["meeting", "agenda", "dinner", "sale"], len(S)
def DF(word, y):
    return sum(t.split().count(word) for t, c in S if c != y)
def psi(x, y):
    tf = [x.split().count(word) for word in words]
    return [round(f * math.log(m / DF(word, y)), 2)
            for f, word in zip(tf, words)]
x = "agenda for the meeting about the meeting room"
for y in ("work", "personal", "promo"):
    print(y, psi(x, y))
```

It prints `work [3.58, 1.79, 0.0, 0.0]`, `personal [2.2, 0.69, 0.0, 0.0]` and `promo [1.39, 1.1, 0.0, 0.0]`. “Meeting” appears once outside work but three times outside promotions, so the same two occurrences in \(\mathbf x\) are worth \(2 \log 6 = 3.58\) as evidence for work and only \(2 \log 2 = 1.39\) for promotions. With any positive weights, work wins.

## What a mistake costs

So far a mistake is a mistake. Your users disagree. A work message filed as personal is a nuisance; a work message filed as junk is a missed deadline. We need a loss that charges each mistake what it costs.

**Definition (cost-sensitive loss).** A **cost** is a function \(\Delta: \mathcal Y \times \mathcal Y \to \mathbb R_+\), where \(\Delta(y', y)\) is the loss of predicting \(y'\) when the correct label is \(y\). We always assume \(\Delta(y, y) = 0\). The zero-one loss is the special case \(\Delta(y', y) = \mathbb 1[y' \neq y]\).

Here is the cost table your users agreed on. Rows are the correct folder, columns the predicted one.

| correct \ predicted | work | personal | promotions | junk |
|---|---|---|---|---|
| work | 0 | 0.5 | 0.5 | 1 |
| personal | 0.5 | 0 | 0.5 | 1 |
| promotions | 0.2 | 0.2 | 0 | 0.2 |
| junk | 0.5 | 0.5 | 0.2 | 0 |

[[diagram:cost]]

Losing work or personal mail to junk costs the most. A promotion in the wrong place costs little, wherever it goes. Junk in a real folder is an annoyance. The empirical risk of a hypothesis is the average cost, and the true risk is its expectation, the average cost over fresh mail with each message weighted by how often it arrives. Three examples costing 0, 1 and 0.2 have average cost 0.4. In symbols,

\[ L_S(h) = \frac{1}{m} \sum_{i=1}^m \Delta(h(\mathbf x_i), y_i), \qquad L^\Delta_{\mathcal D}(h) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}\left[ \Delta(h(\mathbf x), y) \right]. \]

Read each half from the inside out.

- \(\Delta(h(\mathbf x_i), y_i)\) is the cost of what \(h\) predicted for example \(i\), measured against its correct folder.
- \(\sum_{i=1}^m\) adds those costs over the \(m\) examples, and \(\frac{1}{m}\) turns the total into an average. So \(L_S\) is lesson 1’s empirical risk, with costs in place of 0/1 mistakes.
- In the second half, \((\mathbf x, y) \sim \mathcal D\) reads “a fresh example drawn from \(\mathcal D\)”, and \(\mathbb E\), the expectation of lesson 2, is the average over that draw, each possible example weighted by its probability.
- The superscript \(\Delta\) in \(L^\Delta_{\mathcal D}\) records which cost is being averaged.

In plain words: **the empirical risk is the average cost on your labelled examples, and the true risk is the average cost on the next message to arrive.** Three examples: a work message filed as work (cost 0), a work message filed as junk (cost 1), a promotion filed as personal (cost 0.2): \(L_S = (0 + 1 + 0.2)/3 = 0.4\). And if fresh mail were 50% messages \(h\) files correctly, 30% promotions it files as personal and 20% work it files as junk, then \(L^\Delta_{\mathcal D} = 0.5 \times 0 + 0.3 \times 0.2 + 0.2 \times 1 = 0.26\).

```python
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
delta = lambda yp, y: C[y][yp]
pairs = [(0, 0), (3, 0), (1, 2)]
L_S = sum(delta(yp, y) for yp, y in pairs) / len(pairs)
D = [(0.5, (0, 0)), (0.3, (1, 2)), (0.2, (3, 0))]
L_D = sum(p * delta(yp, y) for p, (yp, y) in D)
print(round(L_S, 3), round(L_D, 3))
```

It prints `0.4 0.26`.

## ERM, and why it is hard

With the class \(\mathcal H_{\Psi, W}\) and the cost \(\Delta\) fixed, the ERM rule of lesson 2 says: find \(\mathbf w \in W\) whose predictor has the smallest \(L_S\).

In the realizable case with \(W = \mathbb R^d\), this is a feasibility problem. We need a \(\mathbf w\) with \(y_i = \operatorname{argmax}_y \langle \mathbf w, \Psi(\mathbf x_i, y) \rangle\) for every \(i\), as the unique winner, which is the same as the set of linear inequalities

\[ \forall i \in [m],\ \forall y \in \mathcal Y \setminus \{y_i\}: \quad \langle \mathbf w, \Psi(\mathbf x_i, y_i) \rangle > \langle \mathbf w, \Psi(\mathbf x_i, y) \rangle. \]

Read it from the inside out.

- \(\forall\) reads “for every”. \(i \in [m]\) runs over the \(m\) examples.
- \(\mathcal Y \setminus \{y_i\}\) is the label set with the correct label removed: “every other folder”.
- The inequality says the correct folder’s score is strictly larger than that other folder’s score.

In plain words: **for every example, its correct folder must outscore every other folder.** With the twelve messages of the next section and four folders, that is \(12 \times 3 = 36\) inequalities. The weights that make each folder listen to its own word satisfy all three for the first work message (scores 2, 0, 0, 0) but not for the second personal message, where work and personal tie at 1.

```python
k, n = 4, 5
w = [int(i == y) for y in range(k) for i in range(n)]
sc = lambda x, y: sum(w[y * n + j] * x[j] for j in range(n))
for x, yi in [((2, 0, 0, 0, 1), 0), ((1, 1, 0, 0, 1), 1)]:
    print([sc(x, yi) > sc(x, y) for y in range(k) if y != yi])
print(12 * (k - 1))
```

It prints `[True, True, True]`, then `[False, True, True]`, then `36`.

That is \(m(k - 1)\) strict inequalities in \(\mathbf w\). The strictness is harmless, by the same rescaling as for halfspaces in lesson 9: if some \(\mathbf w\) satisfies them, let \(\gamma > 0\) be the smallest gap among the finitely many, and \(\mathbf w/\gamma\) satisfies every difference \(\ge 1\). So finding a consistent \(\mathbf w\) is a linear program. There is also a multiclass Perceptron, with a mistake bound just like lesson 9’s; you will prove it yourself in the problems.

Without realizability, ERM over this class is computationally hard in general, just as it is for halfspaces. Lesson 12’s answer applies here too: replace the loss with a convex **surrogate** that upper bounds it, and minimize that.

## The generalized hinge loss

For halfspaces the surrogate was the hinge loss \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\): zero when the signed score \(y\langle \mathbf w, \mathbf x \rangle\) is at least 1, and growing by 1 for every unit below that, so a signed score of \(-0.5\) costs 1.5. We need its multiclass version, and the cleanest way is to derive it, one inequality at a time, from the thing we want to bound: \(\Delta(h_{\mathbf w}(\mathbf x), y)\), where \(h_{\mathbf w}(\mathbf x) = \operatorname{argmax}_{y'} \langle \mathbf w, \Psi(\mathbf x, y') \rangle\).

*Step 1.* The predicted label has the largest score, in particular at least the score of the correct label: \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle \le \langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x)) \rangle\).

*Step 2.* So adding the difference of those scores to the cost can only increase it:

\[ \Delta(h_{\mathbf w}(\mathbf x), y) \le \Delta(h_{\mathbf w}(\mathbf x), y) + \langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x)) - \Psi(\mathbf x, y) \rangle. \]

Read the added term from the inside out: \(\Psi(\mathbf x, h_{\mathbf w}(\mathbf x)) - \Psi(\mathbf x, y)\) is the difference between the feature vectors of the predicted and the correct label, and its inner product with \(\mathbf w\) is “score of the prediction minus score of the truth”, which Step 1 showed is never negative. In plain words: **the cost of the prediction is at most that cost plus how far the prediction outscored the truth.** For a work message filed as junk with scores 2 for junk and 1 for work, the left side is 1 and the right side is \(1 + (2 - 1) = 2\).

```python
cost, score_pred, score_true = 1, 2, 1
print(cost, cost + (score_pred - score_true))
```

It prints `1 2`.

*Step 3.* The right side is the expression \(\Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle\) at one particular label, \(y' = h_{\mathbf w}(\mathbf x)\). The maximum over all labels is at least as large.

**Definition (generalized hinge loss).**

\[ \ell(\mathbf w, (\mathbf x, y)) = \max_{y' \in \mathcal Y} \left( \Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle \right). \]

Read it from the inside out.

- \(y'\) is a rival label; \(y\) is the correct one.
- \(\langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle\) is how much the rival outscores the truth (negative if it trails).
- \(\Delta(y', y)\) adds the cost of choosing that rival.
- \(\max_{y' \in \mathcal Y}\) keeps the worst rival, and \(y' = y\) itself contributes \(0 + 0 = 0\).

In plain words: **the loss is the worst rival’s cost plus how far that rival outscores the correct label.** For the work message that mentions “prize” twice, the four terms are 0 (work itself), \(0.5 + 0 - 1 = -0.5\) (personal), \(0.5 + 0 - 1 = -0.5\) (promotions) and \(1 + 2 - 1 = 2\) (junk), so \(\ell = 2\).

```python
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
scores = [1, 0, 0, 2]
y = 0
terms = [C[y][yp] + scores[yp] - scores[y] for yp in range(4)]
print(terms, max(terms))
```

It prints `[0, -0.5, -0.5, 2] 2`.

Steps 1 to 3 prove \(\ell(\mathbf w, (\mathbf x, y)) \ge \Delta(h_{\mathbf w}(\mathbf x), y)\). Notice that the proof never asked which label wins a tie: whatever label the argmax returns, it scores at least as high as \(y\), and that is all Step 1 used.

The term \(y' = y\) of the max is \(\Delta(y, y) + 0 = 0\), so \(\ell \ge 0\). When is \(\ell\) zero? Exactly when every other term is at most 0, that is, when the correct label beats every other label by at least the cost of confusing them:

\[ \forall y' \neq y: \quad \langle \mathbf w, \Psi(\mathbf x, y) \rangle \ge \langle \mathbf w, \Psi(\mathbf x, y') \rangle + \Delta(y', y). \]

Read it as a margin requirement: for every rival \(y'\), the correct label’s score must be at least the rival’s score plus the cost of confusing them. In plain words: **the correct folder beats each rival by at least what mistaking it for that rival would cost.** The first work message scores 2 for work and 0 for every rival; the costs are 0.5, 0.5 and 1, and \(2 \ge 0 + 1\), so all three requirements hold and the loss is 0.

```python
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
scores, y = [2, 0, 0, 0], 0
ok = [scores[y] >= scores[yp] + C[y][yp]
      for yp in range(4) if yp != y]
print(ok)
```

It prints `[True, True, True]`.

The difference \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle - \langle \mathbf w, \Psi(\mathbf x, y') \rangle\) is the **margin** of \(y\) over \(y'\), the word lesson 15 used for halfspaces. So the loss demands a margin that grows with the cost: work must beat junk by 1, but promotions only has to beat the other folders by 0.2. In that case the bound is tight too: with every margin positive, \(h_{\mathbf w}(\mathbf x) = y\), so \(\Delta(h_{\mathbf w}(\mathbf x), y) = 0 = \ell\).

[[diagram:hinge-margin]]

Here are four messages scored with the multivector map, counts of “meeting”, “mom”, “sale”, “prize” and a constant 1, and a weight vector in which each folder simply listens to its own word. In the code, 0 is work, 1 personal, 2 promotions, 3 junk.

```python
C = [[0, .5, .5, 1], [.5, 0, .5, 1],
     [.2, .2, 0, .2], [.5, .5, .2, 0]]
delta = lambda yp, y: C[y][yp]
k, n = 4, 5
def psi(x, y):
    return [0] * (y * n) + list(x) + [0] * ((k - 1 - y) * n)
w = sum(([int(i == y) for i in range(n)] for y in range(k)), [])
score = lambda x, y: sum(a * b for a, b in zip(w, psi(x, y)))
h = lambda x: max(range(k), key=lambda y: score(x, y))
def hinge(x, y):
    return max(delta(yp, y) + score(x, yp) - score(x, y)
               for yp in range(k))
S = [((2, 0, 0, 0, 1), 0), ((1, 1, 0, 0, 1), 1),
     ((0, 0, 1, 1, 1), 3), ((1, 0, 0, 2, 1), 0)]
for x, y in S:
    print(h(x), delta(h(x), y), round(hinge(x, y), 2))
```

Each line prints the prediction, its cost and the generalized hinge loss: `0 0 0`, `0 0.5 0.5`, `2 0.2 0.2`, `3 1 2`. The first message is work with margin 2 over every rival, more than any cost, so the loss is 0. The second is a personal message that ties with work, the tie goes to work, and loss and cost are both 0.5. The last is a work message that mentions “prize” twice: it is filed as junk at cost 1, and the hinge charges 2, because it also counts how far junk outscored work.

Two reminders from lesson 12 before the next result. A function is **convex** when the straight segment between any two points of its graph lies on or above the graph, like \(w \mapsto |w|\). It is **\(\rho\)-Lipschitz** when moving its input a distance \(\delta\) changes its output by at most \(\rho\delta\); \(w \mapsto 3w\) is 3-Lipschitz.

**Proposition (convex and Lipschitz).** For fixed \((\mathbf x, y)\), the function \(\mathbf w \mapsto \ell(\mathbf w, (\mathbf x, y))\) is convex and \(\rho\)-Lipschitz with \(\rho = \max_{y' \in \mathcal Y} \|\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\|\).

*Proof.* For each label \(y'\), the function \(f_{y'}(\mathbf w) = \Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle\) is affine in \(\mathbf w\), hence convex, and \(\ell\) is their maximum. Lesson 12 showed that a maximum of convex functions is convex, so \(\ell\) is convex.

For the Lipschitz bound, first each \(f_{y'}\): by Cauchy–Schwarz, \(|f_{y'}(\mathbf u) - f_{y'}(\mathbf v)| = |\langle \mathbf u - \mathbf v, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle| \le \rho\, \|\mathbf u - \mathbf v\|\). Then the maximum. Let \(y^*\) be a label achieving \(\ell(\mathbf u) = f_{y^*}(\mathbf u)\). Since \(\ell(\mathbf v) \ge f_{y^*}(\mathbf v)\),

\[ \ell(\mathbf u) - \ell(\mathbf v) \le f_{y^*}(\mathbf u) - f_{y^*}(\mathbf v) \le \rho\, \|\mathbf u - \mathbf v\|. \]

Read it left to right. The first “\(\le\)” replaces \(\ell(\mathbf u)\) by \(f_{y^*}(\mathbf u)\), which is equal, and \(\ell(\mathbf v)\) by the smaller \(f_{y^*}(\mathbf v)\), so the difference can only grow. The second is the Lipschitz bound for the single affine function \(f_{y^*}\); \(\|\mathbf u - \mathbf v\|\) is the distance between the two weight vectors. In plain words: **the loss at \(\mathbf u\) exceeds the loss at \(\mathbf v\) by no more than the winning rival’s line does, and that line rises at most \(\rho\) per unit of distance.** Two random weight vectors for the “prize” message give the three numbers in order: 0.65, then at most 3.954, then at most 19.433.

```python
import math
import random

rng, k, n = random.Random(1), 4, 5
x, y = (1, 0, 0, 2, 1), 0
sc = lambda w, yy: sum(w[yy * n + j] * x[j] for j in range(n))
f = lambda w, yp: int(yp != y) + sc(w, yp) - sc(w, y)
ell = lambda w: max(f(w, yp) for yp in range(k))
u, v = ([rng.gauss(0, 1) for _ in range(k * n)] for _ in "uv")
ys = max(range(k), key=lambda yp: f(u, yp))
rho = math.sqrt(2) * math.hypot(*x)
print(round(ell(u) - ell(v), 3), round(f(u, ys) - f(v, ys), 3),
      round(rho * math.dist(u, v), 3))
```

It prints `0.65 3.954 19.433`.

Swapping the roles of \(\mathbf u\) and \(\mathbf v\) bounds \(\ell(\mathbf v) - \ell(\mathbf u)\) the same way, so \(|\ell(\mathbf u) - \ell(\mathbf v)| \le \rho\, \|\mathbf u - \mathbf v\|\). \(\blacksquare\)

For the multivector map, \(\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\) has \(\mathbf x\) in one block and \(-\mathbf x\) in another, so \(\rho = \sqrt 2\, \|\mathbf x\|\). Python can hunt for a pair of weight vectors that breaks the bound:

```python
import math
import random

rng, k, n = random.Random(0), 4, 5
def psi(x, y):
    return [0] * (y * n) + list(x) + [0] * ((k - 1 - y) * n)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
def ell(w, x, y):
    s = lambda yp: dot(w, psi(x, yp))
    return max(int(yp != y) + s(yp) - s(y) for yp in range(k))
x, y = (1, 0, 0, 2, 1), 0
rho = math.sqrt(2) * math.dist(x, [0] * n)
worst = 0.0
for _ in range(10_000):
    w, u = ([rng.gauss(0, 1) for _ in range(k * n)] for _ in "wu")
    ratio = abs(ell(w, x, y) - ell(u, x, y)) / math.dist(w, u)
    worst = max(worst, ratio)
print(round(rho, 3), round(worst, 3))
```

It prints `3.464 2.103`: across 10,000 random pairs, the loss never changes faster than \(\rho = \sqrt{12} = 3.464\) times the distance.

**The binary case.** Why call it a hinge loss? Take \(\mathcal Y = \{\pm 1\}\), the zero-one cost and \(\Psi(\mathbf x, y) = y\mathbf x / 2\). The term \(y' = y\) is 0. The term \(y' = -y\) is \(1 + \langle \mathbf w, (-y\mathbf x - y\mathbf x)/2 \rangle = 1 - y\langle \mathbf w, \mathbf x \rangle\). The maximum of the two is \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\), lesson 15’s hinge loss exactly. (The factor \(1/2\) is there so the gap between the two labels’ scores is \(y\langle \mathbf w, \mathbf x \rangle\) rather than twice that.)

```python
import random

rng = random.Random(0)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
psi = lambda x, y: [y * xi / 2 for xi in x]
def g_hinge(w, x, y):
    s = lambda yp: dot(w, psi(x, yp))
    return max(int(yp != y) + s(yp) - s(y) for yp in (-1, 1))
gap = 0.0
for _ in range(10_000):
    w = [rng.gauss(0, 1) for _ in range(3)]
    x = [rng.gauss(0, 1) for _ in range(3)]
    y = rng.choice((-1, 1))
    hinge = max(0, 1 - y * dot(w, x))
    gap = max(gap, abs(g_hinge(w, x, y) - hinge))
print(f"{gap:.1e}")
```

It prints `8.9e-16`, floating-point dust: the two losses agree.

## Multiclass SVM and SGD

A convex, Lipschitz surrogate is exactly what lessons 13 and 14 know how to learn with. Let’s use both.

**Multiclass SVM.** Lesson 13’s regularized loss minimization, which picks the \(\mathbf w\) minimizing the empirical loss plus \(\lambda\|\mathbf w\|^2\) so that long weight vectors are penalized (with \(\lambda = 0.01\), a vector of length 5 pays 0.25), applied to the generalized hinge loss, gives this rule.

```
MulticlassSVM(S, λ, Δ, Ψ):
  input: (x_1, y_1), ..., (x_m, y_m)
  parameters: λ > 0, cost Δ, class-sensitive map Ψ
  solve:
    w = argmin over w in R^d of
        λ ||w||^2 + (1/m) Σ_i max over y' of
            ( Δ(y', y_i) + <w, Ψ(x_i, y') − Ψ(x_i, y_i)> )
  output: h_w(x) = argmax over y of <w, Ψ(x, y)>
```

The objective is convex, so any convex optimizer solves it. How good is the result? Recall lesson 13’s oracle inequality: for a convex, \(\rho\)-Lipschitz loss, the rule \(A\) that minimizes \(L_S(\mathbf w) + \lambda\|\mathbf w\|^2\) satisfies, for every fixed \(\mathbf w^*\),

\[ \mathbb E_{S \sim \mathcal D^m}\left[ L_{\mathcal D}(A(S)) \right] \le L_{\mathcal D}(\mathbf w^*) + \lambda \|\mathbf w^*\|^2 + \frac{2\rho^2}{\lambda m}. \]

Read it from the inside out.

- \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) examples drawn independently from \(\mathcal D\)”, as in lesson 2. \(A(S)\) is the weight vector the rule returns on it.
- \(L_{\mathcal D}(A(S))\) is its true risk, and \(\mathbb E_{S}\) averages that over all the training sequences you might have drawn.
- \(\mathbf w^*\) is any fixed weight vector you like, chosen before seeing data. \(\lambda\|\mathbf w^*\|^2\) is the price of its length, and \(2\rho^2/(\lambda m)\) is the price of having only \(m\) examples, where \(\rho\) is the Lipschitz constant of the loss.

In plain words: **on average, the regularized learner is as good as any fixed weight vector, up to a length penalty and a sample-size penalty.** With \(L_{\mathcal D}(\mathbf w^*) = 0.05\), \(\|\mathbf w^*\| = 5\), \(\rho = 2\), \(m = 10{,}000\) and \(\lambda = 0.00566\), the right side is \(0.05 + 0.141 + 0.141 = 0.333\).

```python
import math

L_star, norm, rho, m = 0.05, 5, 2, 10_000
lam = math.sqrt(2 * rho**2 / (norm**2 * m))
terms = [L_star, lam * norm**2, 2 * rho**2 / (lam * m)]
print([round(t, 3) for t in terms], round(sum(terms), 3))
```

It prints `[0.05, 0.141, 0.141] 0.333`.

**Theorem (multiclass SVM).** Let \(\Psi\) satisfy \(\|\Psi(\mathbf x, y)\| \le \rho/2\) for all \(\mathbf x, y\), let \(B > 0\), and run multiclass SVM with \(\lambda = \sqrt{2\rho^2/(B^2 m)}\) on \(S \sim \mathcal D^m\), getting \(\mathbf w = A(S)\). Write \(L^{\text{g-hinge}}_{\mathcal D}(\mathbf w) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}[\ell(\mathbf w, (\mathbf x, y))]\). Then

\[ \mathbb E_{S}\left[ L^\Delta_{\mathcal D}(h_{\mathbf w}) \right] \le \mathbb E_{S}\left[ L^{\text{g-hinge}}_{\mathcal D}(\mathbf w) \right] \le \min_{\mathbf u: \|\mathbf u\| \le B} L^{\text{g-hinge}}_{\mathcal D}(\mathbf u) + \sqrt{\frac{8\rho^2 B^2}{m}}. \]

Read it from the inside out.

- \(L^\Delta_{\mathcal D}(h_{\mathbf w})\) is what users feel: the expected cost of the learned predictor on fresh mail.
- \(L^{\text{g-hinge}}_{\mathcal D}(\mathbf w)\) is the expected generalized hinge loss of the same weights, the quantity the learner actually minimizes.
- \(\min_{\mathbf u: \|\mathbf u\| \le B}\) is the best hinge risk of any weight vector of length at most \(B\): the best you could hope for in the class.
- \(\sqrt{8\rho^2B^2/m}\) is the price of learning from \(m\) examples.

In plain words: **the expected cost of the learned filter is at most its expected hinge loss, which is at most the best hinge loss in the ball plus \(\sqrt{8\rho^2B^2/m}\).** With \(\rho = 2\), \(B = 5\), \(m = 10{,}000\) and a best hinge risk of 0.05, the guarantee is \(0.05 + 0.283 = 0.333\), the same total as the oracle inequality above, now in closed form.

```python
import math

best_hinge, rho, B, m = 0.05, 2, 5, 10_000
gap = math.sqrt(8 * rho**2 * B**2 / m)
print(round(gap, 3), round(best_hinge + gap, 3))
```

It prints `0.283 0.333`.

*Proof.* First, the loss is \(\rho\)-Lipschitz. By the triangle inequality, \(\|\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\| \le \|\Psi(\mathbf x, y')\| + \|\Psi(\mathbf x, y)\| \le \rho/2 + \rho/2 = \rho\) for every pair of labels, so the proposition above gives Lipschitz constant at most \(\rho\), and the loss is convex.

The left inequality. For every example, \(\Delta(h_{\mathbf w}(\mathbf x), y) \le \ell(\mathbf w, (\mathbf x, y))\). Taking the expectation over \((\mathbf x, y) \sim \mathcal D\) keeps the inequality, so \(L^\Delta_{\mathcal D}(h_{\mathbf w}) \le L^{\text{g-hinge}}_{\mathcal D}(\mathbf w)\) for every \(\mathbf w\), in particular for the random \(\mathbf w = A(S)\); taking the expectation over \(S\) keeps it again.

The right inequality. Apply the oracle inequality with any \(\mathbf u\) with \(\|\mathbf u\| \le B\) as \(\mathbf w^*\). Then \(\lambda\|\mathbf u\|^2 \le \lambda B^2\), and with the chosen \(\lambda\) the two error terms are equal:

\[ \lambda B^2 = \sqrt{\frac{2\rho^2 B^2}{m}}, \qquad \frac{2\rho^2}{\lambda m} = \frac{2\rho^2}{m} \sqrt{\frac{B^2 m}{2\rho^2}} = \sqrt{\frac{2\rho^2 B^2}{m}}. \]

Read each half by substituting \(\lambda = \sqrt{2\rho^2/(B^2 m)}\). In the first, \(B^2\) moves under the square root as \(B^4\), and one \(B^2\) cancels. In the second, dividing by \(\lambda\) flips the fraction under the root, and \(\frac{2\rho^2}{m}\sqrt{\frac{B^2m}{2\rho^2}} = \sqrt{\frac{4\rho^4 B^2 m}{2\rho^2 m^2}} = \sqrt{\frac{2\rho^2B^2}{m}}\). In plain words: **this choice of \(\lambda\) makes the length penalty and the sample-size penalty equal.** With \(\rho = 2\), \(B = 5\), \(m = 10{,}000\), both are 0.141; the snippet after this proof prints their sum, 0.283, computed both ways.

Their sum is \(2\sqrt{2\rho^2 B^2/m} = \sqrt{8\rho^2 B^2/m}\). The bound holds for every such \(\mathbf u\), so it holds for the minimizing one. \(\blacksquare\)

```python
import math

rho, B, m, eps = 2, 5, 10_000, 0.1
lam = math.sqrt(2 * rho**2 / (B**2 * m))
oracle = lam * B**2 + 2 * rho**2 / (lam * m)
print(round(lam, 5), round(oracle, 3),
      round(math.sqrt(8 * rho**2 * B**2 / m), 3))
print(math.ceil(B**2 * rho**2 / eps**2),
      round(math.sqrt(B**2 / (rho**2 * 10_000)), 3))
```

With messages scaled so \(\|\mathbf x\| \le 1\) (so the multivector map has \(\|\Psi\| \le 1\) and \(\rho = 2\)), \(B = 5\) and \(m = 10{,}000\), it prints `0.00566 0.283 0.283`: \(\lambda\), the oracle sum, and the closed form agree. The second line belongs to the next theorem.

Look at what is missing from the bound: the number of labels. The constant depends on \(\rho\) and \(B\), and nothing else. The number of labels can still sneak in, through the size of \(\Psi\) or because with many labels no short \(\mathbf u\) has small hinge loss, but there is no explicit factor of \(k\). Remember this; it is what makes the section on structured output possible.

**SGD for multiclass learning.** Lesson 14’s stochastic gradient descent needs one subgradient of the loss per example. A subgradient is the slope of a line (or plane) through the current point that stays below the function everywhere; where the function is smooth it is the gradient, and at a corner, like 0 for \(|w|\), any slope between \(-1\) and \(1\) qualifies. The loss is a maximum of affine functions, and lesson 14 showed that the gradient of any function achieving the maximum is a subgradient of the maximum. The affine function for label \(y'\) has gradient \(\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\). So: find the label that achieves the max, and step against its gradient.

```
MulticlassSGD(η, T, Δ, Ψ):
  w(1) ← 0 in R^d
  for t = 1, ..., T:
    sample (x, y) ~ D
    ŷ ← argmax over y' of Δ(y', y) + <w(t), Ψ(x, y') − Ψ(x, y)>
    v_t ← Ψ(x, ŷ) − Ψ(x, y)
    w(t+1) ← w(t) − η v_t
  output w̄ = (1/T) Σ_{t=1..T} w(t)
```

Read the update for the multivector map. If \(\hat y = y\), then \(\mathbf v_t = \mathbf 0\) and nothing happens: the message already wins with the required margin. Otherwise the block of the correct folder moves towards \(\mathbf x\) and the block of the offending folder \(\hat y\) moves away from it. It is the Perceptron’s instinct, with the cost deciding which rival counts as offending.

**Theorem (multiclass SGD).** With \(\|\Psi(\mathbf x, y)\| \le \rho/2\) and \(B > 0\), for every \(\varepsilon > 0\), running multiclass SGD for

\[ T \ge \frac{B^2 \rho^2}{\varepsilon^2} \quad \text{steps, with} \quad \eta = \sqrt{\frac{B^2}{\rho^2 T}}, \]

Read it from the inside out.

- \(\varepsilon\) is the accuracy you ask for, as in every lesson: how far above the best hinge risk you allow the result to be.
- \(T\) is the number of SGD steps, which is also the number of examples SGD reads.
- \(B^2\rho^2/\varepsilon^2\) grows with the size of the ball and the Lipschitz constant, and with one over the square of the accuracy.
- \(\eta\) is the step size; it shrinks as \(1/\sqrt T\).

In plain words: **to get within \(\varepsilon\) of the best weights in the ball, run at least \(B^2\rho^2/\varepsilon^2\) steps with step size \(\sqrt{B^2/(\rho^2T)}\).** With \(B = 5\) and \(\rho = 2\): \(\varepsilon = 0.1\) needs 10,000 steps at \(\eta = 0.025\), and halving \(\varepsilon\) to 0.05 needs four times as many, 40,000, at \(\eta = 0.0125\).

```python
import math

B, rho = 5, 2
for eps in (0.1, 0.05):
    T = math.ceil(B**2 * rho**2 / eps**2)
    print(eps, T, round(math.sqrt(B**2 / (rho**2 * T)), 4))
```

It prints `0.1 10000 0.025`, then `0.05 40000 0.0125`.

gives \(\mathbb E\left[ L^\Delta_{\mathcal D}(h_{\bar{\mathbf w}}) \right] \le \mathbb E\left[ L^{\text{g-hinge}}_{\mathcal D}(\bar{\mathbf w}) \right] \le \min_{\|\mathbf u\| \le B} L^{\text{g-hinge}}_{\mathcal D}(\mathbf u) + \varepsilon\).

*Proof.* The loss is convex and \(\rho\)-Lipschitz, as in the previous proof, and \(\mathbf v_t\) is a subgradient of \(\ell(\cdot, (\mathbf x, y))\) at \(\mathbf w^{(t)}\) by the rule for maxima. These are exactly the conditions of lesson 14’s guarantee for SGD on convex, Lipschitz, bounded learning problems, which gives the right inequality. The left one is the pointwise bound \(\Delta \le \ell\), taken in expectation, as before. \(\blacksquare\)

With the numbers above and \(\varepsilon = 0.1\), the snippet’s second line printed `10000 0.025`: ten thousand examples and a step of 0.025. SGD sees each example once and stores one vector, which is why it is the workhorse.

Here it is on twelve labelled messages, three per folder, with the users’ cost table. Each message has counts of “meeting”, “mom”, “sale” and “prize”, plus a constant 1.

```python
import random

X = [(2, 0, 0, 0), (3, 0, 1, 0), (1, 0, 0, 0), (0, 2, 0, 0),
     (1, 1, 0, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 0, 3, 1),
     (1, 0, 2, 0), (0, 0, 0, 2), (0, 0, 1, 3), (0, 1, 0, 2)]
S = [(x + (1,), i // 3) for i, x in enumerate(X)]
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
k, n, eta, T = 4, 5, 0.05, 5_000
sc = lambda w, x, y: sum(w[y * n + j] * x[j] for j in range(n))
loss = lambda w, x, y, yp: C[y][yp] + sc(w, x, yp) - sc(w, x, y)
w, wbar, rng = [0.0] * (k * n), [0.0] * (k * n), random.Random(0)
for t in range(T):
    wbar = [a + b / T for a, b in zip(wbar, w)]
    x, y = rng.choice(S)
    yh = max(range(k), key=lambda yp: loss(w, x, y, yp))
    for j in range(n):
        w[yh * n + j] -= eta * x[j]
        w[y * n + j] += eta * x[j]
h = lambda x: max(range(k), key=lambda y: sc(wbar, x, y))
hinge = sum(max(loss(wbar, x, y, yp) for yp in range(k))
            for x, y in S) / len(S)
cost = sum(C[y][h(x)] for x, y in S) / len(S)
mistakes = sum(h(x) != y for x, y in S)
print(mistakes, round(cost, 3), round(hinge, 3))
```

It prints `0 0.0 0.0`: no mistakes, no cost, and every message wins by its full required margin. (After only 50 steps the mistakes are already 0 while the average hinge loss is 0.089; the hinge keeps pushing until the margins match the costs.) Sampling uniformly from \(S\) here plays the role of sampling from \(\mathcal D\); on the training sequence it is SGD minimizing \(L_S\).

**Cost.** Each step computes \(|\mathcal Y|\) scores to find \(\hat y\), so with a general \(\Psi\) that costs \(|\mathcal Y|\) evaluations of \(\Psi\) and inner products in \(\mathbb R^d\), \(O(|\mathcal Y|\, d)\) time, plus \(O(d)\) for the update. For the multivector map the scores cost \(O(kn)\) in total and the update touches two blocks, \(O(n)\). Space is \(O(d)\) for \(\mathbf w\) and \(\bar{\mathbf w}\). Prediction is one argmax, \(O(|\mathcal Y|\, d)\). One-versus-all with SGD inside, for comparison, runs \(k\) binary learners over the whole sequence, and all-pairs runs \(\binom{k}{2}\) learners whose sequences add up to \((k - 1)m\) examples; prediction costs \(k\) and \(\binom{k}{2}\) binary evaluations.

## When the labels are sequences

Your users have one more request. Long messages should be shown with the greeting and the signature collapsed, so the filter has to tag every line of a message as greeting, body or signature. A message with \(r\) lines needs a label that is a whole sequence of \(r\) tags. With \(q\) possible tags there are \(q^r\) labels: 243 for a five-line message with three tags, and more than three billion for a twenty-line one. The classic example of the same problem is reading a handwritten word, one letter per image, where the label set is every string of letters.

**Structured output prediction** is multiclass prediction where \(\mathcal Y\) is huge but has a structure we can exploit. Three questions come at once. How do we predict, when the argmax ranges over \(q^r\) labels? How do we train, when each SGD step needs another argmax over \(\mathcal Y\)? And how do we avoid overfitting with so many labels?

The third question is already answered: the guarantees for multiclass SVM and SGD have no explicit dependence on \(|\mathcal Y|\), only on \(\rho\) and \(B\). We just need \(\Psi\) to stay bounded. The first two are answered by choosing \(\Psi\) and \(\Delta\) so that the argmax can be computed by dynamic programming.

Set it up. Each line \(t\) is described by \(n\) features (does it start with “Hi”, does it end with a comma, and so on), so a message is a matrix \(\mathbf x \in \mathbb R^{n \times r}\) whose column \(t\) describes line \(t\). The label is \(\mathbf y = (y_1, \dots, y_r)\) with \(y_t \in [q]\). For the cost, take the fraction of lines tagged wrongly, \(\Delta(\mathbf y', \mathbf y) = \frac{1}{r} \sum_{t=1}^r \mathbb 1[y_t \neq y'_t]\). The map \(\Psi\) has \(d = nq + q^2\) entries of two types:

\[ \Psi_{i,j,1}(\mathbf x, \mathbf y) = \frac{1}{r} \sum_{t=1}^r x_{i,t}\, \mathbb 1[y_t = j], \qquad \Psi_{i,j,2}(\mathbf x, \mathbf y) = \frac{1}{r} \sum_{t=2}^r \mathbb 1[y_t = i]\, \mathbb 1[y_{t-1} = j]. \]

Read each type from the inside out.

- \(x_{i,t}\) is feature \(i\) of line \(t\). \(\mathbb 1[y_t = j]\) is 1 if line \(t\) carries tag \(j\).
- In type 1, \(\sum_{t=1}^r x_{i,t}\mathbb 1[y_t = j]\) adds feature \(i\) over the lines tagged \(j\) only.
- In type 2, \(\mathbb 1[y_t = i]\,\mathbb 1[y_{t-1} = j]\) is 1 when line \(t\) has tag \(i\) and the line before it has tag \(j\); the sum counts those transitions, starting at \(t = 2\) because line 1 has no line before it.
- The factor \(\frac1r\) divides by the number of lines.

In plain words: **a type 1 entry is “how much of feature \(i\) sits on lines tagged \(j\)”, and a type 2 entry is “how often tag \(i\) follows tag \(j\)”, both per line.** For the five-line message of the build-it challenge, tagged greeting, body, body, signature, signature, the feature “ends with a comma” is on lines 1 and 4, so its greeting entry is \(1/5 = 0.2\) and its signature entry is 0.2; “body follows body” happens once, so that entry is 0.2, and “signature follows body” is 0.2 too.

```python
comma = [1, 0, 0, 1, 0]
tags = [0, 1, 1, 2, 2]
r = len(tags)
type1 = [sum(comma[t] * (tags[t] == j) for t in range(r)) / r
         for j in range(3)]
pairs = [(tags[t], tags[t - 1]) for t in range(1, r)]
print(type1, pairs.count((1, 1)) / r, pairs.count((2, 1)) / r)
```

It prints `[0.2, 0.0, 0.2] 0.2 0.2`.

A type 1 entry adds up feature \(i\) over the lines tagged \(j\): it can learn that “starts with Hi” suggests a greeting. A type 2 entry counts how often tag \(i\) follows tag \(j\): it can learn that a signature is followed by more signature, never by a greeting. The weights decide which of these matter.

The trick is that \(\Psi\) is a sum over positions, each term depending only on the current tag and the previous one. Set \(y_0 = 0\), a start symbol that is not a tag, and define

\[ \phi_{i,j,1}(\mathbf x, y_t, y_{t-1}) = x_{i,t}\, \mathbb 1[y_t = j], \qquad \phi_{i,j,2}(\mathbf x, y_t, y_{t-1}) = \mathbb 1[y_t = i]\, \mathbb 1[y_{t-1} = j]. \]

Read them as the summands of \(\Psi\) with the sum and the \(\frac1r\) taken off: \(\phi\) looks at one position \(t\) and only at the tags \(y_t\) and \(y_{t-1}\). In plain words: **\(\phi\) is the contribution of a single line, and \(\Psi\) is the average of those contributions over the lines.** For the five-line message, the comma feature’s greeting entry of \(\phi\) is 1 at line 1 and 0 elsewhere; adding over the lines and dividing by 5 gives back the 0.2 above.

```python
comma, tags, r = [1, 0, 0, 1, 0], [0, 1, 1, 2, 2], 5
phi = [comma[t] * (tags[t] == 0) for t in range(r)]
print(phi, sum(phi) / r)
```

It prints `[1, 0, 0, 0, 0] 0.2`.

(The \(\phi\) also reads which column \(t\) it is at; we keep that implicit.) Then \(\Psi(\mathbf x, \mathbf y) = \frac{1}{r}\sum_{t=1}^r \phi(\mathbf x, y_t, y_{t-1})\): at \(t = 1\) the type 2 term vanishes because \(y_0 = 0\) matches no tag \(j\), which is why the type 2 sum starts at \(t = 2\). The positive factor \(1/r\) does not change which \(\mathbf y\) wins, so

\[ h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{\mathbf y \in \mathcal Y} \sum_{t=1}^r \langle \mathbf w, \phi(\mathbf x, y_t, y_{t-1}) \rangle. \]

Read it from the inside out: \(\langle \mathbf w, \phi(\mathbf x, y_t, y_{t-1}) \rangle\) is line \(t\)’s score given its tag and the previous tag; the sum over \(t\) is the score of the whole tag sequence \(\mathbf y\); the argmax runs over all \(q^r\) sequences. In plain words: **the prediction is the tag sequence whose line-by-line scores add up to the most.** With the build-it weights, the sequence greeting, body, body, signature, signature scores 17, and the all-body sequence scores 3.

```python
LINES = ["Hi Ana,", "The budget is attached.",
         "Numbers are final.", "Thanks,", "Ben"]
F = [lambda s: s.startswith(("Hi", "Dear")),
     lambda s: s.endswith(","), lambda s: len(s.split()) >= 3,
     lambda s: s.istitle() and len(s.split()) == 1]
x = [[int(f(s)) for s in LINES] for f in F]
w1 = [[2, -1, -1], [1, -1, 1], [-1, 2, -1], [-1, -1, 2]]
w2 = [[-2, -1, -3], [2, 1, -1], [-3, 1, 1]]
def line(t, s, p):
    emit = sum(w1[i][s] * x[i][t] for i in range(4))
    return emit + (w2[s][p] if t > 0 else 0)
total = lambda y: sum(line(t, y[t], y[t - 1]) for t in range(5))
print(total((0, 1, 1, 2, 2)), total((1, 1, 1, 1, 1)))
```

It prints `17 3`.

**Dynamic programming.** For a tag \(s\) and a position \(\tau\), let \(M_{s,\tau}\) be the best score of any first \(\tau\) tags that end in \(s\):

\[ M_{s,\tau} = \max_{(y_1, \dots, y_\tau):\, y_\tau = s} \sum_{t=1}^\tau \langle \mathbf w, \phi(\mathbf x, y_t, y_{t-1}) \rangle. \]

Read it from the inside out: the sum is the score of the first \(\tau\) lines under a candidate prefix \((y_1, \dots, y_\tau)\); the condition after the colon, \(y_\tau = s\), keeps only prefixes whose last tag is \(s\); the max takes the best of them. In plain words: **\(M_{s,\tau}\) is the best score you can get on the first \(\tau\) lines if line \(\tau\) is forced to have tag \(s\).** For the build-it message, \(M_{\text{body},2}\) tries the three tags of line 1 with line 2 as body; the best is greeting then body, with score 7.

```python
x = [[1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1]]
w1 = [[2, -1, -1], [1, -1, 1], [-1, 2, -1], [-1, -1, 2]]
w2 = [[-2, -1, -3], [2, 1, -1], [-3, 1, 1]]
emit = lambda t, s: sum(w1[i][s] * x[i][t] for i in range(4))
prefix = {y1: emit(0, y1) + emit(1, 1) + w2[1][y1]
          for y1 in range(3)}
print(prefix, max(prefix.values()))
```

It prints `{0: 7, 1: 1, 2: 1} 7`.

The best full score is \(\max_s M_{s,r}\), and the table obeys the recursion

\[ M_{s,\tau} = \max_{s'} \left( M_{s',\tau-1} + \langle \mathbf w, \phi(\mathbf x, s, s') \rangle \right). \]

Read it from the inside out: \(M_{s',\tau-1}\) is the best prefix ending in \(s'\) one line earlier; \(\langle \mathbf w, \phi(\mathbf x, s, s') \rangle\) is the score of line \(\tau\) with tag \(s\) after tag \(s'\); the max tries every \(s'\). In plain words: **the best way to end line \(\tau\) with tag \(s\) is the best way to end line \(\tau - 1\), plus one more step, over every possible previous tag.** For \(M_{\text{body},2}\): the first column is \(M_{\cdot,1} = (3, -2, 0)\) for greeting, body and signature, and the step into body on line 2 scores 4, 3 and 1 from those three tags, so the max is \(3 + 4 = 7\): the same 7 as before, now from three additions instead of three full prefixes.

```python
x = [[1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1]]
w1 = [[2, -1, -1], [1, -1, 1], [-1, 2, -1], [-1, -1, 2]]
w2 = [[-2, -1, -3], [2, 1, -1], [-3, 1, 1]]
emit = lambda t, s: sum(w1[i][s] * x[i][t] for i in range(4))
M1 = [emit(0, s) for s in range(3)]
step = [emit(1, 1) + w2[1][p] for p in range(3)]
print(M1, step, max(M1[p] + step[p] for p in range(3)))
```

It prints `[3, -2, 0] [4, 3, 1] 7`.

*Why the recursion holds.* Split any sequence \((y_1, \dots, y_\tau)\) with \(y_\tau = s\) into its prefix of length \(\tau - 1\), which ends in some tag \(s' = y_{\tau-1}\), and the last term \(\langle \mathbf w, \phi(\mathbf x, s, s') \rangle\). The last term depends on the prefix only through \(s'\). So for a fixed \(s'\), the best total is the best prefix ending in \(s'\), which is \(M_{s',\tau-1}\) by definition, plus the last term. Maximizing over \(s'\) covers every sequence ending in \(s\). (With \(\tau = 1\) there is no prefix: \(M_{s,1} = \langle \mathbf w, \phi(\mathbf x, s, 0) \rangle\).) Recording which \(s'\) won lets you walk the winning sequence back from the end.

[[diagram:trellis]]

```
TagLines(x, w):
  for each tag s:  M[s, 1] ← <w, φ(x, s, 0)>
  for τ = 2, ..., r:
    for each tag s:
      I[s, τ] ← argmax over s' of M[s', τ−1] + <w, φ(x, s, s')>
      M[s, τ] ← M[I[s, τ], τ−1] + <w, φ(x, s, I[s, τ])>
  y_r ← argmax over s of M[s, r]
  for τ = r, r−1, ..., 2:  y_{τ−1} ← I[y_τ, τ]
  return (y_1, ..., y_r)
```

The table has \(qr\) cells and each takes a max over \(q\) predecessors, so \(O(r q^2)\) evaluations of \(\langle \mathbf w, \phi \rangle\). Each evaluation is the emission part \(\sum_i w_{i,s,1} x_{i,t}\), which costs \(O(n)\) and can be computed once per \((s, t)\), plus one transition weight. Total time \(O(rqn + rq^2)\), against \(O(q^r\, r(n + q))\) for trying every label; space \(O(qr)\) for the two tables.

```python
import itertools
import random

rng, q, r, n = random.Random(3), 3, 6, 4
x = [[rng.randint(0, 3) for _ in range(r)] for _ in range(n)]
w1 = [[rng.gauss(0, 1) for _ in range(q)] for _ in range(n)]
w2 = [[rng.gauss(0, 1) for _ in range(q)] for _ in range(q)]
def phi(t, s, sp):
    emit = sum(w1[i][s] * x[i][t] for i in range(n))
    return emit + (w2[s][sp] if t > 0 else 0)
score = lambda y: sum(phi(t, y[t], y[t - 1]) for t in range(r))
brute = max(itertools.product(range(q), repeat=r), key=score)
M, I = [[phi(0, s, None) for s in range(q)]], []
for t in range(1, r):
    via = lambda s, p: M[-1][p] + phi(t, s, p)
    arg = lambda s: max(range(q), key=lambda p: via(s, p))
    I.append([arg(s) for s in range(q)])
    M.append([via(s, I[-1][s]) for s in range(q)])
y = [max(range(q), key=lambda s: M[-1][s])]
for t in range(r - 2, -1, -1):
    y.insert(0, I[t][y[0]])
print(tuple(y) == brute, brute, round(score(brute), 3))
```

With random weights it prints `True (2, 1, 2, 1, 2, 1) 24.458`: the table, after \(6 \times 3 \times 3\) steps, finds the same winner as the search through all 729 tag sequences. (Code positions start at 0, so position 0 has no transition term.)

Training needs the other argmax, the one inside SGD: \(\hat{\mathbf y} = \operatorname{argmax}_{\mathbf y'} \Delta(\mathbf y', \mathbf y) + \langle \mathbf w, \Psi(\mathbf x, \mathbf y') \rangle\). Our cost is also a sum over positions, so the same table works with one extra term per cell. Working that out is a follow-up in the second build-it challenge below.

## Ranking the inbox

Now the second puzzle. The inbox is not a set of labels; it is an order.

**Definition (ranking hypothesis).** Let \(\mathcal X^*\) be the set of finite sequences of messages, of any length. A **ranking hypothesis** takes a sequence \(\bar{\mathbf x} = (\mathbf x_1, \dots, \mathbf x_r)\) and returns a vector \(\mathbf y \in \mathbb R^r\) of scores; sorting the scores gives the order. Write \(\pi(\mathbf y)\) for the **permutation induced by** \(\mathbf y\): \(\pi(\mathbf y)_i\) is the position of \(y_i\) when the entries of \(\mathbf y\) are sorted in increasing order. The top of the list is the message whose score is largest, the one with \(\pi(\mathbf y)_i = r\).

Take \(\mathbf y = (2, 1, 6, -1, 0.5)\). Sorted increasingly it is \((-1, 0.5, 1, 2, 6)\). The score 2 sits in position 4, the score 1 in position 3, 6 in position 5, \(-1\) in position 1 and 0.5 in position 2, so \(\pi(\mathbf y) = (4, 3, 5, 1, 2)\), and message 3 is the top of the list.

An example is now a pair \((\bar{\mathbf x}, \mathbf y)\): one morning’s inbox and the true relevance scores your users gave its messages. A loss compares the predicted scores \(\mathbf y'\) with the true \(\mathbf y\), \(\ell(h, (\bar{\mathbf x}, \mathbf y)) = \Delta(h(\bar{\mathbf x}), \mathbf y)\). There are several reasonable choices.

**0-1 ranking loss.** \(\Delta(\mathbf y', \mathbf y) = \mathbb 1[\pi(\mathbf y') \neq \pi(\mathbf y)]\): 1 unless the order is exactly right. Almost useless in practice, since swapping the two least important messages costs as much as reversing the inbox.

**Kendall tau loss.** Count the pairs that are in a different order, and divide by the number of pairs:

\[ \Delta(\mathbf y', \mathbf y) = \frac{2}{r(r-1)} \sum_{i=1}^{r-1} \sum_{j=i+1}^r \mathbb 1\left[ \operatorname{sign}(y'_i - y'_j) \neq \operatorname{sign}(y_i - y_j) \right]. \]

Read it from the inside out.

- \(y'_i - y'_j\) compares the predicted scores of messages \(i\) and \(j\); \(\operatorname{sign}\) keeps only which is larger (\(+1\), \(-1\), or 0 for a tie).
- The indicator is 1 when the predicted order of the pair differs from the true order.
- The double sum \(\sum_{i=1}^{r-1}\sum_{j=i+1}^r\) visits every pair once, with \(i < j\).
- There are \(r(r-1)/2\) pairs, so multiplying by \(\frac{2}{r(r-1)}\) turns the count into a fraction.

In plain words: **the Kendall tau loss is the fraction of pairs of messages that the ranker puts in the wrong order.** Three messages with true relevances \((3, 1, 2)\) and scores \((0.5, 0.9, 0.1)\): the pairs (1, 2) and (2, 3) are flipped and (1, 3) is right, so the loss is \(2/3\).

```python
sgn = lambda a: (a > 0) - (a < 0)
y, yp = (3, 1, 2), (0.5, 0.9, 0.1)
r = len(y)
flips = [(i + 1, j + 1) for i in range(r - 1)
         for j in range(i + 1, r)
         if sgn(yp[i] - yp[j]) != sgn(y[i] - y[j])]
print(flips, round(2 * len(flips) / (r * (r - 1)), 3))
```

It prints `[(1, 2), (2, 3)] 0.667`.

Here \(\operatorname{sign}(0) = 0\), so a pair the users rated equal counts as a disagreement whenever the scores separate it. This loss sees partial credit: one swap costs one pair.

**Normalized discounted cumulative gain (NDCG).** Users read the top of the list, so mistakes there should cost more. Choose a nondecreasing **discount** \(D: \mathbb N \to \mathbb R_+\) that weights positions, and define the **discounted cumulative gain**

\[ G(\mathbf y', \mathbf y) = \sum_{i=1}^r D(\pi(\mathbf y')_i)\, y_i. \]

Read it from the inside out: \(\pi(\mathbf y')_i\) is the position the ranker gives message \(i\) (\(r\) is the top); \(D(\cdot)\) turns that position into a weight; \(y_i\) is the message’s true relevance; the sum adds weight times relevance over all messages. In plain words: **the gain adds up the true relevance of every message, weighted by how good a position the ranker put it in.** In the six-message example below (\(k = 3\)), the ranker puts message 1 (relevance 5) on top with weight 1, message 2 (relevance 1) second with weight 0.63 and message 5 (relevance 3) third with weight 0.5, so \(G = 5 + 0.63 + 1.5 = 7.13\).

```python
import math

r, k = 6, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
y, pos = (5, 1, 4, 0, 3, 2), (6, 5, 2, 1, 4, 3)
terms = [round(D(p) * yi, 2) for p, yi in zip(pos, y)]
print(terms, round(sum(D(p) * yi for p, yi in zip(pos, y)), 2))
```

It prints `[5.0, 0.63, 0, 0, 1.5, 0] 7.13`.

Read \(y_i\) as message \(i\)’s true relevance; the gain adds the relevances, each weighted by the discount of the position \(\mathbf y'\) put it in. If all relevances are nonnegative, then \(0 \le G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\): the first inequality because every term is nonnegative, the second because, for a nondecreasing \(D\), putting the largest relevances in the positions with the largest discounts is the best you can do (a sorting fact you will prove in the problems). So the **NDCG loss**

\[ \Delta(\mathbf y', \mathbf y) = 1 - \frac{G(\mathbf y', \mathbf y)}{G(\mathbf y, \mathbf y)} =

Read it from the inside out: \(G(\mathbf y, \mathbf y)\) is the gain of the perfect order, the largest possible; the ratio is the share of that ideal the ranker achieved; one minus it is the share lost. The second form writes the same loss as a sum, message by message, of “weight it deserved minus weight it got”, times its relevance. In plain words: **the NDCG loss is the fraction of the ideal gain that the ranker’s order throws away.** Here \(1 - 7.13/9.02 = 0.21\), and the message-by-message sum gives the same 0.21.

```python
import math

r, k = 6, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
y = (5, 1, 4, 0, 3, 2)
ideal, got = (6, 2, 5, 1, 4, 3), (6, 5, 2, 1, 4, 3)
G = lambda pos: sum(D(p) * yi for p, yi in zip(pos, y))
diff = sum((D(a) - D(b)) * yi for a, b, yi in zip(ideal, got, y))
print(round(1 - G(got) / G(ideal), 2), round(diff / G(ideal), 2))
```

It prints `0.21 0.21`.

\frac{1}{G(\mathbf y, \mathbf y)} \sum_{i=1}^r \left( D(\pi(\mathbf y)_i) - D(\pi(\mathbf y')_i) \right) y_i \]

The NDCG loss lies in \([0, 1]\), and it is 0 when \(\pi(\mathbf y') = \pi(\mathbf y)\). A typical discount gives the top position weight 1, each of the next few positions a little less, and every position below the top \(k\) nothing; for six messages and \(k = 3\) the weights from the bottom are 0, 0, 0, 0.5, 0.63, 1. With a parameter \(k < r\), it is

\[ D(i) = \begin{cases} \dfrac{1}{\log_2(r - i + 2)} & \text{if } i \in \{r - k + 1, \dots, r\}, \\ 0 & \text{otherwise.} \end{cases} \]

The top position, \(i = r\), gets \(1/\log_2 2 = 1\), the next \(1/\log_2 3 = 0.63\), and everything below the top \(k\) gets nothing at all. That is how search engines are judged, and it is how your users read their mail.

Read it from the inside out: \(i\) is a position, with \(r\) the top; \(\{r - k + 1, \dots, r\}\) are the top \(k\) positions; inside them the weight is one over \(\log_2(r - i + 2)\), where \(\log_2\) is the logarithm in base 2, and below them it is 0. In plain words: **the top position counts fully, each lower position in the top \(k\) a bit less, and everything below the top \(k\) not at all.** For \(r = 6\) and \(k = 3\), positions 1 to 6 get weights 0, 0, 0, 0.5, 0.63, 1.

```python
import math

r, k = 6, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
print([round(D(i), 2) for i in range(1, r + 1)])
```

It prints `[0, 0, 0, 0.5, 0.63, 1.0]`.

[[diagram:ranking-losses]]

Six messages, with relevances \(\mathbf y = (5, 1, 4, 0, 3, 2)\) from the users and scores \(\mathbf y' = (0.9, 0.8, 0.1, -0.5, 0.7, 0.2)\) from a ranker:

```python
import math

def pi(y):
    order = sorted(range(len(y)), key=lambda i: y[i])
    return tuple(order.index(i) + 1 for i in range(len(y)))
sgn = lambda a: (a > 0) - (a < 0)
def kendall(yp, y):
    r = len(y)
    bad = sum(sgn(yp[i] - yp[j]) != sgn(y[i] - y[j])
              for i in range(r) for j in range(i + 1, r))
    return 2 * bad / (r * (r - 1))
def ndcg_loss(yp, y, k):
    r = len(y)
    D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
    G = lambda a: sum(D(p) * yi for p, yi in zip(pi(a), y))
    return 1 - G(yp) / G(y)
print(pi((2, 1, 6, -1, 0.5)))
y, yp = (5, 1, 4, 0, 3, 2), (0.9, 0.8, 0.1, -0.5, 0.7, 0.2)
print(pi(yp), round(kendall(yp, y), 3))
print(round(ndcg_loss(yp, y, 3), 3))
```

It prints `(4, 3, 5, 1, 2)`, then `(6, 5, 2, 1, 4, 3) 0.333`, then `0.21`. The ranker gets the top message right but puts message 2, relevance 1, in second place, where message 3, relevance 4, belonged. Five of the fifteen pairs are out of order. With \(k = 3\) the ideal gain is \(5 + 4/\log_2 3 + 3/2 = 9.02\), the ranker’s is \(5 + 1/\log_2 3 + 3/2 = 7.13\), and the loss is \(1 - 7.13/9.02 = 0.21\).

## Linear rankers and their surrogates

The natural linear ranker scores each message with the same weight vector:

\[ h_{\mathbf w}((\mathbf x_1, \dots, \mathbf x_r)) = (\langle \mathbf w, \mathbf x_1 \rangle, \dots, \langle \mathbf w, \mathbf x_r \rangle), \]

and the class is \(\mathcal H_W = \{h_{\mathbf w} : \mathbf w \in W\}\). (A feature map in the style of lesson 16 can be applied first; we keep the plain form.) ERM over a training sequence of inboxes \((\bar{\mathbf x}_1, \mathbf y_1), \dots, (\bar{\mathbf x}_m, \mathbf y_m)\), each with its own length \(r_i\), minimizes \(\sum_i \Delta(h_{\mathbf w}(\bar{\mathbf x}_i), \mathbf y_i)\), and for most ranking losses that is computationally hard. So, once more, convex surrogates.

Read it from the inside out: each message \(\mathbf x_i\) gets its own score \(\langle \mathbf w, \mathbf x_i \rangle\), always with the same \(\mathbf w\), and the ranker returns the list of scores. In plain words: **a linear ranker scores every message with one shared weight vector and sorts by the score.** With \(\mathbf w = (1, -1)\) and three messages \((2, 0)\), \((1, 3)\), \((0, 1)\), the scores are \(2, -2, -1\), so the order from the top is message 1, message 3, message 2.

```python
w = (1, -1)
xs = [(2, 0), (1, 3), (0, 1)]
scores = [w[0] * a + w[1] * b for a, b in xs]
order = sorted(range(1, 4), key=lambda i: -scores[i - 1])
print(scores, order)
```

It prints `[2, -2, -1] [1, 3, 2]`.

**A hinge loss for Kendall tau.** Look at one pair \(i < j\) and write \(s = \operatorname{sign}(y_i - y_j)\). When \(s \neq 0\), the scores disagree with the truth exactly when \(s\,(y'_i - y'_j) \le 0\). When \(s = 0\), the indicator \(\mathbb 1[s(y'_i - y'_j) \le 0]\) is 1, which is at least the true term. So in every case

\[ \mathbb 1\left[ \operatorname{sign}(y'_i - y'_j) \neq \operatorname{sign}(y_i - y_j) \right] \le \mathbb 1\left[ s\,(y'_i - y'_j) \le 0 \right]. \]

Read it from the inside out: on the left, “the pair is in the wrong order”; on the right, “the predicted difference does not have the true sign”, with \(s = \operatorname{sign}(y_i - y_j)\). In plain words: **a pair counted wrong by Kendall tau is always counted wrong by the sign test, and the sign test is a single number, \(s(y'_i - y'_j)\), that a hinge can bound.** The snippet checks all nine combinations of true and predicted order (ahead, tied, behind): the left side never exceeds the right, and they differ only when both are tied.

```python
sgn = lambda a: (a > 0) - (a < 0)
for dy in (1, 0, -1):
    for dyp in (1.0, 0.0, -1.0):
        left = int(sgn(dyp) != sgn(dy))
        right = int(sgn(dy) * dyp <= 0)
        print(dy, dyp, left, right, left <= right)
```

It prints nine lines, one per combination of true and predicted order, with the two indicators and a final `True` on every line: the left side never exceeds the right, and the two differ only on the line `0 0.0 0 1 True`, where both are tied.

For a linear ranker, \(y'_i - y'_j = \langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\). And for any number \(a\), \(\mathbb 1[a \le 0] \le \max\{0, 1 - a\}\): if \(a \le 0\) the right side is at least 1, otherwise the left side is 0. With \(a = s\langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\), averaging over pairs gives

\[ \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) \le \frac{2}{r(r-1)} \sum_{i=1}^{r-1} \sum_{j=i+1}^r \max\left\{0,\ 1 - \operatorname{sign}(y_i - y_j) \langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\right\}. \]

Read it from the inside out: \(\langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\) is the predicted score gap of the pair; multiplying by the true sign makes it positive when the pair is ordered right; \(\max\{0, 1 - \cdot\}\) is the hinge, zero only when the gap is right *and* at least 1; the average runs over pairs as in the Kendall loss. In plain words: **the Kendall loss is at most the average pairwise hinge, which also charges pairs that are right but by less than 1.** Three messages with relevances \((2, 3, 1)\) and \(\mathbf w = (1, -1)\) on \((2, 0), (1, 3), (0, 1)\), so scores \(2, -2, -1\): the pairs (1, 2) and (2, 3) are out of order, the Kendall loss is \(2/3 = 0.667\), and the surrogate averages the hinges 5, 0 and 2 to \(2.333\).

```python
sgn = lambda a: (a > 0) - (a < 0)
w, xs, y = (1, -1), [(2, 0), (1, 3), (0, 1)], (2, 3, 1)
mu = [w[0] * a + w[1] * b for a, b in xs]
P = [(i, j) for i in range(3) for j in range(i + 1, 3)]
kendall = sum(sgn(mu[i] - mu[j]) != sgn(y[i] - y[j])
              for i, j in P) / len(P)
hinge = sum(max(0, 1 - sgn(y[i] - y[j]) * (mu[i] - mu[j]))
            for i, j in P) / len(P)
print(round(kendall, 3), round(hinge, 3))
```

It prints `0.667 2.333`.

Each term is a hinge loss in \(\mathbf w\), hence convex, and it is \(\|\mathbf x_i - \mathbf x_j\|\)-Lipschitz because \(|s| \le 1\). An average with nonnegative weights summing to 1 of convex, \(\rho\)-Lipschitz functions is convex and \(\rho\)-Lipschitz, so the surrogate is convex and Lipschitz with \(\rho \le \max_{i,j} \|\mathbf x_i - \mathbf x_j\|\). SGD applies directly.

**A hinge loss for NDCG.** NDCG depends on the scores only through the permutation they induce, which looks hopeless for a gradient method. Here is the observation that unlocks it. Let \(V\) be the set of all permutations of \([r]\), each written as a vector \(\mathbf v \in [r]^r\) with distinct entries. If the entries of \(\mathbf y'\) are distinct, then

\[ \pi(\mathbf y') = \operatorname*{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i\, y'_i. \]

In words: to make \(\sum_i v_i y'_i\) as large as possible, give the largest position number to the largest score, the next largest to the next, and so on, which is exactly what sorting does. This is the sorting fact again; proving it is Problem 3. Python checks it on a thousand random score vectors against all 720 permutations of six positions:

Read it from the inside out: \(\mathbf v\) is a candidate assignment of positions \(1, \dots, r\) to the messages, one each; \(\sum_i v_i y'_i\) multiplies each message’s position by its score and adds; the argmax over all \(r!\) permutations in \(V\) picks the best assignment. In plain words: **the order the scores induce is the assignment of positions that makes “position times score” add up to the most.** With scores \((0.3, -1, 2)\), sorting gives positions \((2, 1, 3)\) and the sum \(0.6 - 1 + 6 = 5.6\); every other permutation gives less.

```python
import itertools

yp = (0.3, -1, 2)
V = itertools.permutations((1, 2, 3))
total = {v: round(sum(a * b for a, b in zip(v, yp)), 2)
         for v in V}
print(max(total, key=total.get), total)
```

It prints `(2, 1, 3)` and the sum for each of the six permutations; the largest, `5.6`, belongs to `(2, 1, 3)`.

```python
import itertools
import random

rng, r = random.Random(0), 6
pi = lambda y: tuple(sorted(y).index(v) + 1 for v in y)
agree = 0
for _ in range(1_000):
    yp = [rng.gauss(0, 1) for _ in range(r)]
    V = itertools.permutations(range(1, r + 1))
    best = max(V, key=lambda v: sum(a * b for a, b in zip(v, yp)))
    agree += best == pi(yp)
print(agree)
```

It prints `1000`.

Now the surrogate falls out of the multiclass machinery, with permutations playing the role of labels. Define \(\Psi(\bar{\mathbf x}, \mathbf v) = \sum_{i=1}^r v_i \mathbf x_i\). Then

\[ \pi(h_{\mathbf w}(\bar{\mathbf x})) = \operatorname*{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i \langle \mathbf w, \mathbf x_i \rangle = \operatorname*{argmax}_{\mathbf v \in V} \left\langle \mathbf w, \sum_{i=1}^r v_i \mathbf x_i \right\rangle = \operatorname*{argmax}_{\mathbf v \in V} \langle \mathbf w, \Psi(\bar{\mathbf x}, \mathbf v) \rangle, \]

Read the chain left to right. The first equality is the sorting identity with \(y'_i = \langle \mathbf w, \mathbf x_i \rangle\). The second moves the sum inside the inner product, which is allowed because the inner product is linear. The third names the vector inside, \(\Psi(\bar{\mathbf x}, \mathbf v) = \sum_i v_i \mathbf x_i\). In plain words: **ranking by a linear score is the same as choosing the permutation \(\mathbf v\) whose feature vector \(\sum_i v_i \mathbf x_i\) scores highest.** With the three messages and \(\mathbf w = (1, -1)\), both sides give positions \((3, 1, 2)\).

```python
import itertools

w, xs = (1, -1), [(2, 0), (1, 3), (0, 1)]
mu = [w[0] * a + w[1] * b for a, b in xs]
pi = tuple(sorted(mu).index(m) + 1 for m in mu)
def score(v):
    psi = [sum(vi * x[c] for vi, x in zip(v, xs)) for c in (0, 1)]
    return w[0] * psi[0] + w[1] * psi[1]
best = max(itertools.permutations((1, 2, 3)), key=score)
print(pi, best)
```

It prints `(3, 1, 2) (3, 1, 2)`.

so the linear ranker is a linear multiclass predictor whose labels are permutations. The generalized hinge derivation goes through word for word (write \(\Delta(\mathbf v, \mathbf y)\) for the NDCG loss of the order \(\mathbf v\), since \(\pi(\mathbf v) = \mathbf v\)):

\[ \begin{aligned} \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) &\le \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) + \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(h_{\mathbf w}(\bar{\mathbf x}))) \rangle - \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(\mathbf y)) \rangle \\ &\le \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \langle \mathbf w, \Psi(\bar{\mathbf x}, \mathbf v) \rangle - \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(\mathbf y)) \rangle \right] \\ &= \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \sum_{i=1}^r (v_i - \pi(\mathbf y)_i) \langle \mathbf w, \mathbf x_i \rangle \right]. \end{aligned} \]

The first line holds because the predicted permutation maximizes \(\langle \mathbf w, \Psi(\bar{\mathbf x}, \cdot) \rangle\), so the added difference is nonnegative. The second takes the max over all permutations. The last line expands \(\Psi\). The right side is a maximum of affine functions of \(\mathbf w\), so it is convex.

In plain words: **the NDCG loss of the ranker’s order is at most the worst “loss plus score advantage” over all orders, a convex function of \(\mathbf w\).** For the three messages scored \(2, -2, -1\) by \(\mathbf w = (1, -1)\), now with relevances \((2, 3, 1)\) and \(k = 2\), the snippet evaluates the three lines: the NDCG loss is 0.383, adding the score advantage 5.0 gives 5.383, and the max over all six orders is also 5.383, attained here by the predicted order itself.

```python
import itertools
import math

r, k, y = 3, 2, (2, 3, 1)
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
mu = [2, -2, -1]
pos = lambda s: tuple(sorted(s).index(a) + 1 for a in s)
G = lambda v: sum(D(vi) * yi for vi, yi in zip(v, y))
loss = lambda v: 1 - G(v) / G(pos(y))
def adv(v):
    return sum((vi - p) * m for vi, p, m in zip(v, pos(y), mu))
pred = pos(mu)
line3 = max(loss(v) + adv(v)
            for v in itertools.permutations((1, 2, 3)))
print(round(loss(pred), 3), round(loss(pred) + adv(pred), 3),
      round(line3, 3))
```

It prints `0.383 5.383 5.383`.

**The bottleneck is an assignment problem.** An SGD step needs the \(\mathbf v\) that achieves that max. Write \(\mu_i = \langle \mathbf w, \mathbf x_i \rangle\). The term \(-\sum_i \pi(\mathbf y)_i \mu_i\) does not depend on \(\mathbf v\), so drop it. Expand the NDCG loss of \(\mathbf v\): \(\Delta(\mathbf v, \mathbf y) = 1 - \sum_i D(v_i) y_i / G(\mathbf y, \mathbf y)\), and the 1 does not depend on \(\mathbf v\) either. What is left is to maximize \(\sum_i \left( \mu_i v_i - D(v_i)\, y_i / G(\mathbf y, \mathbf y) \right)\), which is the same as

\[ \operatorname*{argmin}_{\mathbf v \in V} \sum_{i=1}^r \left( \alpha_i v_i + \beta_i D(v_i) \right), \qquad \alpha_i = -\mu_i, \quad \beta_i = \frac{y_i}{G(\mathbf y, \mathbf y)}. \]

Read it from the inside out: \(\alpha_i v_i\) rewards putting a high-scoring message (large \(\mu_i\), so very negative \(\alpha_i\)) in a high position; \(\beta_i D(v_i)\) charges for giving a relevant message (large \(y_i\)) a big discount weight, which looks odd until you remember we are now *minimizing* the negative of the loss term; the argmin over \(V\) picks the cheapest assignment. In plain words: **the most violating order is the cheapest assignment of positions, where each message’s cost mixes its predicted score and its true relevance.** For three messages with \(\mu = (2, -2, -1)\) and relevances \((3, 1, 2)\), \(k = 2\): \(\alpha = -\mu = (-2, 2, 1)\), the ideal gain is \(G(\mathbf y, \mathbf y) = 3 + 2/\log_2 3 = 4.26\), and \(\beta = \mathbf y / 4.26 = (0.70, 0.23, 0.47)\). The second snippet below checks on random data that this argmin and the surrogate’s argmax always agree.

```python
import math

mu, y = (2, -2, -1), (3, 1, 2)
G_ideal = 3 * 1 + 2 / math.log2(3)
alpha = [-m for m in mu]
beta = [round(yi / G_ideal, 2) for yi in y]
print(alpha, round(G_ideal, 2), beta)
```

It prints `[-2, 2, 1] 4.26 [0.7, 0.23, 0.47]`.

```python
import itertools
import math
import random

rng, r, k = random.Random(0), 5, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
y = [rng.randint(0, 3) for _ in range(r)]
mu = [rng.gauss(0, 1) for _ in range(r)]
pi_y = tuple(sorted(range(r), key=lambda i: y[i]).index(i) + 1
             for i in range(r))
G = lambda v: sum(D(vi) * yi for vi, yi in zip(v, y))
V = list(itertools.permutations(range(1, r + 1)))
hinge = lambda v: 1 - G(v) / G(pi_y) + sum(
    (vi - pi) * m for vi, pi, m in zip(v, pi_y, mu))
cost = lambda v: sum(-m * vi + yi / G(pi_y) * D(vi)
                     for m, vi, yi in zip(mu, v, y))
print(y, max(V, key=hinge) == min(V, key=cost))
```

It prints `[3, 3, 0, 2, 3] True`: the permutation that maximizes the surrogate is the one that minimizes the \(\alpha, \beta\) cost.

Think of message \(i\) as a task and position \(j\) as a worker, and let \(A_{i,j} = j\alpha_i + D(j)\beta_i\) be the cost of giving task \(i\) to worker \(j\). Choosing \(\mathbf v\) is choosing which position each message gets, one message per position: the **assignment problem**. It has efficient algorithms, the Hungarian method among them, and it is also, perhaps surprisingly, a linear program. Here is why.

Encode an assignment as a matrix \(B \in \{0,1\}^{r \times r}\) with \(B_{i,j} = 1\) when message \(i\) gets position \(j\). The constraints “each message one position, each position one message” say every row and every column sums to 1; such a 0/1 matrix is a **permutation matrix**. With \(\langle A, B \rangle = \sum_{i,j} A_{i,j} B_{i,j}\), the problem is

\[ \min_{B} \langle A, B \rangle \quad \text{subject to } B \text{ a permutation matrix.} \]

Read it from the inside out: \(A_{i,j}\) is the cost of message \(i\) taking position \(j\); \(B_{i,j}\) is 1 if it does and 0 if not; \(\langle A, B \rangle = \sum_{i,j} A_{i,j}B_{i,j}\), the matrix version of the inner product, adds the costs of the choices actually made; the minimum runs over permutation matrices only. In plain words: **pick one cell in every row and every column so that the picked costs add up to as little as possible.** With \(A = \begin{pmatrix} 1 & 4 \\ 3 & 2 \end{pmatrix}\), the two permutation matrices cost \(1 + 2 = 3\) and \(4 + 3 = 7\), so the answer is 3.

```python
A = [[1, 4], [3, 2]]
inner = lambda B: sum(A[i][j] * B[i][j]
                      for i in range(2) for j in range(2))
I, X = [[1, 0], [0, 1]], [[0, 1], [1, 0]]
print(inner(I), inner(X))
```

It prints `3 7`.

The 0/1 constraint is what keeps this from being a linear program. Drop it and keep the rest: a matrix with nonnegative entries whose rows and columns each sum to 1 is called **doubly stochastic**, and the relaxed problem is

\[ \min_{B} \langle A, B \rangle \quad \text{subject to } B \text{ doubly stochastic}, \]

a linear program. The two problems have the same answer, thanks to a classical theorem we will use without proof.

Read it the same way, with \(B\) now allowed to spread each row’s 1 over several cells, as long as entries are nonnegative and every row and column still sums to 1. In plain words: **the same assignment, but fractional assignments are allowed, which makes it a linear program.** For the \(2 \times 2\) example, every doubly stochastic matrix is \(\begin{pmatrix} p & 1-p \\ 1-p & p \end{pmatrix}\), its cost is \(3p + 7(1 - p)\), and the minimum is at \(p = 1\): the permutation matrix, cost 3.

```python
A = [[1, 4], [3, 2]]
def inner(p):
    B = [[p, 1 - p], [1 - p, p]]
    return sum(A[i][j] * B[i][j]
               for i in range(2) for j in range(2))
print([(p, round(inner(p), 2)) for p in (0, 0.25, 0.5, 1)])
```

It prints `[(0, 7), (0.25, 6.0), (0.5, 5.0), (1, 3)]`.

**Theorem (Birkhoff and von Neumann).** The set of doubly stochastic \(r \times r\) matrices is the convex hull of the permutation matrices: every doubly stochastic matrix is a convex combination of permutation matrices.

**Lemma (the relaxation is exact).** Some optimal solution of the relaxed problem is a permutation matrix, and it is optimal for the assignment problem too.

*Proof.* Let \(B\) be an optimal solution of the relaxed problem. By the theorem, \(B = \sum_i \gamma_i C_i\) with each \(C_i\) a permutation matrix, each \(\gamma_i > 0\), and \(\sum_i \gamma_i = 1\). Each \(C_i\) is itself doubly stochastic, so it is a feasible point of the relaxed problem, and since \(B\) is optimal, \(\langle A, B \rangle \le \langle A, C_i \rangle\) for every \(i\).

Suppose the inequality were strict for every \(i\). Then, using linearity of \(\langle A, \cdot \rangle\) and \(\sum_i \gamma_i = 1\),

\[ \langle A, B \rangle = \Big\langle A, \sum_i \gamma_i C_i \Big\rangle = \sum_i \gamma_i \langle A, C_i \rangle > \sum_i \gamma_i \langle A, B \rangle = \langle A, B \rangle, \]

Read it left to right: substitute the blend for \(B\); pull the weights out of the inner product (linearity); use the supposed strict inequality \(\langle A, C_i \rangle > \langle A, B \rangle\) on every term, with positive weights; then \(\sum_i \gamma_i = 1\) collapses the right end back to \(\langle A, B \rangle\). In plain words: **a weighted average cannot be strictly larger than every one of its terms, so some permutation in the blend is at least as cheap as the blend.** In the illustration below, \(0.5 \times 17 + 0.3 \times 23 + 0.2 \times 21 = 19.6\), and the cheapest term, 17, is below it.

```python
gamma, costs = [0.5, 0.3, 0.2], [17, 23, 21]
blend = sum(g * c for g, c in zip(gamma, costs))
print(round(blend, 2), min(costs), min(costs) <= blend)
```

It prints `19.6 17 True`.

a contradiction (the strict inequality survives because every \(\gamma_i > 0\)). So some \(C_i\) has \(\langle A, C_i \rangle = \langle A, B \rangle\). That \(C_i\) is feasible for the relaxed problem with the optimal value, so it is optimal there. And every permutation matrix \(C\) is feasible for the relaxed problem, so \(\langle A, C_i \rangle = \langle A, B \rangle \le \langle A, C \rangle\): \(C_i\) is optimal for the assignment problem too. \(\blacksquare\)

[[diagram:assignment]]

The heart of the proof is that an average is never below its smallest term:

```python
import itertools
import random

rng, r = random.Random(0), 4
A = [[rng.randint(0, 9) for _ in range(r)] for _ in range(r)]
perms = list(itertools.permutations(range(r)))
cost = lambda v: sum(A[i][v[i]] for i in range(r))
picks = rng.sample(perms, 3)
gamma = [0.5, 0.3, 0.2]
B = [[sum(g * (v[i] == j) for g, v in zip(gamma, picks))
      for j in range(r)] for i in range(r)]
AB = sum(A[i][j] * B[i][j] for i in range(r) for j in range(r))
print([sum(row) for row in B], [sum(col) for col in zip(*B)])
print(round(AB, 2), [cost(v) for v in picks])
print(min(map(cost, perms)))
```

It prints `[1.0, 1.0, 1.0, 1.0] [1.0, 1.0, 1.0, 1.0]`, then `19.6 [17, 23, 21]`, then `13`. The blend \(B\) is doubly stochastic, its cost 19.6 is the weighted average of its three permutations’ costs, so it is at least the cheapest of them, 17; and the cheapest permutation of all, 13, beats every blend.

## Important or not: bipartite ranking

Back to the inbox your users actually label. They do not give relevance scores; they mark a message important or not. When \(\mathbf y \in \{\pm 1\}^r\), every message is either relevant or not, and ranking means putting the relevant ones first. This is **bipartite ranking**.

Here is the trap the team fell into. It looks as if a binary filter solves it: label each message, put the yeses on top. But a binary learner minimizes the zero-one loss, and the zero-one loss is the wrong measure when the relevant messages are rare. Suppose 20 of 1,000 messages are important, and the filter says no to all of them. It is right on 980 of them, a zero-one loss of 2%, and it has found none of the important ones. To say precisely what it misses, we need a few counts.

A ranker outputs scores \(\mathbf y' \in \mathbb R^r\), and a **threshold** \(\theta \in \mathbb R\) turns them into yes/no answers, \(\operatorname{sign}(y'_i - \theta)\). (Ties at the threshold are measure-zero for real-valued scores; read the sign of 0 as \(-1\) here.) Four counts describe the result: the flagged messages that are important, the flagged ones that are not, the important ones that are missed, and the unimportant ones correctly left alone. On the morning of ten messages tabulated further below, at \(\theta = 0\), they are 2, 2, 1 and 5. In symbols:

\[ \begin{aligned} a &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{true positives} \\ b &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{false positives} \\ c &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{false negatives} \\ d &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{true negatives} \end{aligned} \]

Read one line from the inside out: \(y_i = +1\) is “message \(i\) is important”; \(\operatorname{sign}(y'_i - \theta) = +1\) is “its score is above the threshold, so it is flagged”; \(\wedge\) reads “and”; the braces with a colon collect the messages \(i\) such that both hold; the bars \(|\cdot|\) count them. In plain words: **\(a\) counts important messages that are flagged, \(b\) unimportant ones that are flagged, \(c\) important ones that are missed and \(d\) unimportant ones that are left alone.** In the ten-message morning below, at \(\theta = 0\), messages 1 to 4 are flagged: two of them important (\(a = 2\)) and two not (\(b = 2\)); one important message is missed (\(c = 1\)); five unimportant ones are left alone (\(d = 5\)). Every message falls in exactly one of the four counts, so they add up to 10.

```python
yp = [2.1, 1.4, 0.9, 0.3, -0.2, -0.4, -0.8, -1.0, -1.5, -2.2]
y = [1, -1, 1, -1, -1, 1, -1, -1, -1, -1]
flag = [1 if s > 0 else -1 for s in yp]
def count(t, f):
    return sum(a == t and b == f for a, b in zip(y, flag))
a, b, c = count(1, 1), count(-1, 1), count(1, -1)
d = count(-1, -1)
print(a, b, c, d, a + b + c + d)
```

It prints `2 2 1 5 10`.

The **recall** (also called sensitivity) is \(a/(a + c)\), the fraction of important messages the ranker catches. The **precision** is \(a/(a + b)\), the fraction of flagged messages that are important. The **specificity** is \(d/(d + b)\), the fraction of unimportant messages correctly left alone. The zero-one loss is \((b + c)/(a + b + c + d)\).

Lower the threshold and you flag more: recall rises, reaching 1 at \(\theta = -\infty\), while precision and specificity usually fall. These measures combine precision and recall in different ways.

- **Averaging sensitivity and specificity:** \(\frac12\left( \frac{a}{a+c} + \frac{d}{d+b} \right)\), the accuracy on important messages averaged with the accuracy on the others, at \(\theta = 0\); the loss is 1 minus it.
- **F₁ score:** the harmonic mean of precision and recall, \(\frac{2}{1/\text{precision} + 1/\text{recall}} = \frac{2a}{2a + b + c}\), at \(\theta = 0\); loss \(1 - F_1\). It is 1 when both are 1, and 0 as soon as either is 0.
- **F\(_\beta\) score:** like F₁ but with \(\beta^2\) times more weight on recall, \(\frac{1 + \beta^2}{1/\text{precision} + \beta^2/\text{recall}} = \frac{(1 + \beta^2)a}{(1 + \beta^2)a + b + \beta^2 c}\), at \(\theta = 0\).
- **Recall at \(k\):** the recall when at most \(k\) messages may be flagged: choose \(\theta\) so that \(a + b \le k\). Right when a person can only look at \(k\) messages.
- **Precision at \(k\):** the precision when at least \(k\) messages are flagged: choose \(\theta\) so that \(a + b \ge k\). The natural choice puts \(\theta\) just below the \(k\)-th largest score, so exactly the top \(k\) are flagged; that is what your users mean by “are the top five any good?”.

Read the F₁ formula from the inside out. \(1/\text{precision}\) and \(1/\text{recall}\) are large when either score is small; dividing 2 by their sum gives the **harmonic mean**, which is pulled towards the smaller of the two. Substituting \(a/(a+b)\) and \(a/(a+c)\) and clearing fractions gives \(2a/(2a + b + c)\). In F\(_\beta\), the weight \(\beta^2\) multiplies \(1/\text{recall}\), so missed important messages count \(\beta^2\) times as much. In plain words: **F₁ is high only when the flagged messages are mostly important and most important messages are flagged.** On the morning tabulated below, at \(\theta = 0\), precision 0.5 and recall 0.667 give F₁ \(= 2/(2 + 1.5) = 0.571\), the same as \(2 \cdot 2/(4 + 2 + 1)\); with \(\beta = 2\), F₂ is 0.625.

```python
a, b, c = 2, 2, 1
prec, rec = a / (a + b), a / (a + c)
F1 = 2 / (1 / prec + 1 / rec)
F2 = (1 + 4) / (1 / prec + 4 / rec)
print(round(F1, 3), round(2 * a / (2 * a + b + c), 3),
      round(F2, 3), round(5 * a / (5 * a + b + 4 * c), 3))
```

It prints `0.571 0.571 0.625 0.625`.

Measures like these, which depend on the predictions for the whole list at once and not on each message separately, are called **multivariate performance measures**. One more is widely used, though it is not built from a threshold. The **AUC** (area under the ROC curve) is the fraction of (important, unimportant) pairs in which the important message gets the higher score. It is one minus the Kendall tau loss restricted to the pairs with different labels, and the Kendall surrogate above handles it: for a pair with \(y_i = y_j\) the hinge term is \(\max\{0, 1 - 0\} = 1\) whatever \(\mathbf w\) is, so in the bipartite case the surrogate is a constant plus the hinge over the mixed pairs alone.

[[diagram:threshold]]

Back to the “nothing is important” filter. Its counts are \(a = 0\), \(b = 0\), \(c = 20\), \(d = 980\):

```python
r, P = 1_000, 20
a, b, c, d = 0, 0, P, r - P
zero_one = (b + c) / r
recall = a / (a + c)
F1 = 2 * a / (2 * a + b + c)
print(zero_one, recall, F1, 1 - F1)
```

It prints `0.02 0.0 0.0 1.0`: zero-one loss 2%, recall 0, so F₁ is 0 and its loss is 1.

Now a real ranker on one morning of ten messages, three of them important, with scores sorted from the top:

| message | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | 2.1 | 1.4 | 0.9 | 0.3 | −0.2 | −0.4 | −0.8 | −1.0 | −1.5 | −2.2 |
| important | yes | no | yes | no | no | yes | no | no | no | no |

Before running the code, count by hand at \(\theta = 0\). Which four are flagged, and how many of them are important?

```python
yp = [2.1, 1.4, 0.9, 0.3, -0.2, -0.4, -0.8, -1.0, -1.5, -2.2]
y = [1, -1, 1, -1, -1, 1, -1, -1, -1, -1]
def counts(theta):
    flag = [s > theta for s in yp]
    a = sum(f and t == 1 for f, t in zip(flag, y))
    b = sum(f and t == -1 for f, t in zip(flag, y))
    return a, b, y.count(1) - a, y.count(-1) - b
a, b, c, d = counts(0)
print(a, b, c, d, round(a / (a + c), 3), round(a / (a + b), 3),
      round(d / (d + b), 3), round(2 * a / (2 * a + b + c), 3))
a3, b3, _, _ = counts(sorted(yp)[-4])
pos = [s for s, t in zip(yp, y) if t == 1]
neg = [s for s, t in zip(yp, y) if t == -1]
auc = sum(p > q for p in pos for q in neg) / (len(pos) * len(neg))
print(round(a3 / (a3 + b3), 3), round(auc, 3), (b + c) / len(y))
```

It prints `2 2 1 5 0.667 0.5 0.714 0.571`, then `0.667 0.81 0.3`. At \(\theta = 0\), messages 1 to 4 are flagged and two are important: \(a = 2\), \(b = 2\), \(c = 1\), \(d = 5\), recall \(2/3\), precision \(1/2\), specificity \(5/7\), F₁ \(4/7 = 0.571\). The top three hold two important messages, so precision at 3 is 0.667. Of the 21 mixed pairs, 17 are in the right order, so the AUC is 0.81. And the zero-one loss is 0.3, exactly what a filter that flags nothing would score on this morning, which tells you again that the zero-one loss is measuring something your users do not care about.

## Learning to rank for the measure you want

How do we train a linear ranker for, say, precision at 5? The measure depends on the scores \(\mathbf y' = h_{\mathbf w}(\bar{\mathbf x})\) only through the yes/no vector they induce,

\[ b(\mathbf y') = (\operatorname{sign}(y'_1 - \theta), \dots, \operatorname{sign}(y'_r - \theta)) \in \{\pm 1\}^r, \]

and the plan is the same as for NDCG: write \(b\) as an argmax, then let the generalized hinge loss do the rest. For every measure above there is a set \(V \subseteq \{\pm 1\}^r\) such that

Read it from the inside out: \(y'_i - \theta\) is how far message \(i\)’s score sits above the threshold; its sign is \(+1\) (flagged) or \(-1\) (not); the vector lists those answers, an element of \(\{\pm 1\}^r\), the set of all length-\(r\) lists of \(\pm 1\). In plain words: **\(b(\mathbf y')\) is the list of flagged / not flagged decisions that the threshold makes from the scores.** Scores \((0.9, -0.2, 0.3)\) at \(\theta = 0\) give \((+1, -1, +1)\).

```python
yp, theta = (0.9, -0.2, 0.3), 0
print(tuple(1 if s - theta > 0 else -1 for s in yp))
```

It prints `(1, -1, 1)`.

\[ b(\mathbf y') = \operatorname*{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i\, y'_i. \]

For \(\theta = 0\), take \(V = \{\pm 1\}^r\): each \(v_i\) is chosen on its own, and \(v_i y'_i\) is largest when \(v_i\) has the sign of \(y'_i\). For precision at \(k\), take \(V_{\ge k}\), the vectors with at least \(k\) entries equal to \(+1\); for recall at \(k\), \(V_{\le k}\), the vectors with at most \(k\). (Checking that these give the right thresholds is a good exercise.)

Read it from the inside out: \(\mathbf v\) is a candidate flag vector from the allowed set \(V\); \(\sum_i v_i y'_i\) adds the scores of flagged messages and subtracts the scores of the others; the argmax picks the best allowed flag vector. In plain words: **thresholding is the same as choosing the allowed flag vector that agrees most with the scores.** For scores \((0.9, -0.2, 0.3)\) and \(V = \{\pm 1\}^3\), the eight candidates give sums from \(-1.4\) to \(1.4\), and the best, \((+1, -1, +1)\), is exactly \(b(\mathbf y')\) at \(\theta = 0\). With \(V_{\ge 3}\), every message must be flagged, and the answer is \((+1, +1, +1)\).

```python
import itertools

yp = (0.9, -0.2, 0.3)
V = list(itertools.product((-1, 1), repeat=3))
total = lambda v: sum(a * b for a, b in zip(v, yp))
V3 = [v for v in V if v.count(1) >= 3]
print(max(V, key=total), round(total(max(V, key=total)), 2),
      max(V3, key=total))
```

It prints `(1, -1, 1) 1.4 (1, 1, 1)`.

Then, assuming the true labels \(\mathbf y\) lie in \(V\),

\[ \begin{aligned} \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) &= \Delta(b(h_{\mathbf w}(\bar{\mathbf x})), \mathbf y) \\ &\le \Delta(b(h_{\mathbf w}(\bar{\mathbf x})), \mathbf y) + \sum_{i=1}^r \left( b_i(h_{\mathbf w}(\bar{\mathbf x})) - y_i \right) \langle \mathbf w, \mathbf x_i \rangle \\ &\le \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \sum_{i=1}^r (v_i - y_i) \langle \mathbf w, \mathbf x_i \rangle \right]. \end{aligned} \]

The first line holds because the measure only sees \(b\). The second adds a nonnegative number: \(b\) maximizes \(\sum_i v_i \langle \mathbf w, \mathbf x_i \rangle\) over \(V\), and \(\mathbf y\) is in \(V\), so \(\sum_i b_i \langle \mathbf w, \mathbf x_i \rangle \ge \sum_i y_i \langle \mathbf w, \mathbf x_i \rangle\). The third takes the max over \(V\). The right side is a maximum of affine functions of \(\mathbf w\), so it is a convex surrogate, and SGD needs its maximizer at every step.

In plain words: **the measure’s loss is at most the worst “loss plus score advantage” over all allowed flag vectors, a convex function of \(\mathbf w\).** For three messages with scores \(\mu = (0.9, -0.2, 0.3)\), true labels \((+1, +1, -1)\) and the F₁ loss at \(\theta = 0\): the flags are \((+1, -1, +1)\), \(a = 1\), \(b = 1\), \(c = 1\), so the loss is \(1 - 2/4 = 0.5\); the second line adds \((1-1)0.9 + (-1-1)(-0.2) + (1+1)0.3 = 1.0\), giving 1.5; and the max over all eight vectors is 1.5, attained by the flags themselves.

```python
import itertools

mu, y = (0.9, -0.2, 0.3), (1, 1, -1)
def f1_loss(v):
    a = sum(s == 1 and t == 1 for s, t in zip(v, y))
    b = sum(s == 1 and t == -1 for s, t in zip(v, y))
    return 1 - 2 * a / (a + b + y.count(1))
adv = lambda v: sum((s - t) * m for s, t, m in zip(v, y, mu))
bv = tuple(1 if m > 0 else -1 for m in mu)
V = itertools.product((-1, 1), repeat=3)
print(round(f1_loss(bv), 2), round(f1_loss(bv) + adv(bv), 2),
      round(max(f1_loss(v) + adv(v) for v in V), 2))
```

It prints `0.5 1.5 1.5`.

**Finding the maximizer.** Suppose the measure is a function of \(a, b, c, d\), and \(V\) is all vectors whose \(a\) and \(b\) satisfy some condition, like \(a + b \le k\). Group the candidates: for each pair \((a, b)\), let \(\bar Y_{a,b}\) be the vectors with exactly \(a\) of the important messages and \(b\) of the others set to \(+1\). Every \(\mathbf v\) lies in exactly one group. And because \(V\) is defined by a condition on \((a, b)\), a group is either entirely inside \(V\) or entirely outside. So we can search group by group.

Inside one group, \(c = P - a\) and \(d = N - b\) are fixed too (with \(P\) and \(N\) the numbers of important and unimportant messages), so \(\Delta\) is constant, and \(\sum_i y_i \mu_i\) never depends on \(\mathbf v\). Only \(\sum_i v_i \mu_i\) is left to maximize, with \(\mu_i = \langle \mathbf w, \mathbf x_i \rangle\). Now

\[ \sum_{i} v_i \mu_i = \sum_{i:\, v_i = 1} \mu_i - \sum_{i:\, v_i = -1} \mu_i = 2\sum_{i:\, v_i = 1} \mu_i - \sum_i \mu_i, \]

so we want the \(+1\) entries on the largest scores: the \(a\) important messages with the highest \(\mu\), and the \(b\) unimportant messages with the highest \(\mu\). The two choices do not interact, so each is a simple top-\(a\) and top-\(b\) pick from a sorted list.

Read it from the inside out: \(\sum_{i:\, v_i = 1}\) adds over the flagged messages only, and \(\sum_{i:\, v_i = -1}\) over the others; the second form adds the flagged scores twice and subtracts the total of all scores, which does not depend on \(\mathbf v\). In plain words: **maximizing the sum is the same as maximizing the total score of the flagged messages.** With \(\mu = (2, -1, 0.5)\) and \(\mathbf v = (+1, -1, +1)\): \(2 + 1 + 0.5 = 3.5\), and \(2 \times 2.5 - 1.5 = 3.5\).

```python
mu, v = (2, -1, 0.5), (1, -1, 1)
lhs = sum(a * b for a, b in zip(v, mu))
flagged = sum(m for s, m in zip(v, mu) if s == 1)
print(lhs, 2 * flagged - sum(mu))
```

It prints `3.5 3.5`.

```
BipartiteArgmax(μ_1..μ_r, y_1..y_r, Δ, allowed):
  P ← number of i with y_i = +1;  N ← number with y_i = −1
  sort the important messages by μ, highest first: i_1, ..., i_P
  sort the others the same way: j_1, ..., j_N
  best ← −∞
  for a = 0, 1, ..., P:
    c ← P − a
    for b = 0, 1, ..., N with allowed(a, b):
      d ← N − b
      v ← +1 on i_1..i_a and j_1..j_b, −1 elsewhere
      α ← Δ(a, b, c, d) + Σ_i v_i μ_i
      if α ≥ best:  best ← α;  v* ← v
  return v*
```

Sorting costs \(O(r \log r)\). There are \((P + 1)(N + 1)\) groups. Built naively, each \(\mathbf v\) and its sum cost \(O(r)\), so \(O(PNr)\) overall; with running prefix sums of the two sorted lists, each sum is \(O(1)\) and the total is \(O(r \log r + PN)\). Space is \(O(r)\). Here it is against a search through all 256 sign vectors, for the F₁ loss with three important messages among eight:

```python
import itertools
import random

rng, r = random.Random(0), 8
mu = [rng.gauss(0, 1) for _ in range(r)]
y = [1, 1, -1, -1, -1, 1, -1, -1]
def delta(v):
    a = sum(s == 1 and t == 1 for s, t in zip(v, y))
    b = sum(s == 1 and t == -1 for s, t in zip(v, y))
    return 1 - 2 * a / (a + b + y.count(1))
def obj(v):
    lin = sum((s - t) * m for s, t, m in zip(v, y, mu))
    return delta(v) + lin
brute = max(map(obj, itertools.product((-1, 1), repeat=r)))
top = lambda lab: sorted((i for i in range(r) if y[i] == lab),
                         key=lambda i: -mu[i])
pos, neg = top(1), top(-1)
v = lambda a, b: [1 if i in pos[:a] + neg[:b] else -1
                  for i in range(r)]
best = max(obj(v(a, b)) for a in range(len(pos) + 1)
           for b in range(len(neg) + 1))
print(round(brute, 4), round(best, 4))
```

It prints `4.7035 4.7035`: 20 groups searched, the same optimum as 256 vectors. (The F₁ loss is \(1 - 2a/(2a + b + c)\), and \(2a + b + c = a + b + P\).)

## Build it yourself

Three challenges, one per family of algorithms. Each starter runs its asserts against your code; all three were run against a reference implementation.

**1. Folders.** Implement the reductions and the multiclass learner.

```python
import math
import random

def draw(rng):
    centers = [(-5, 5), (0, 5), (5, 5)]
    y = rng.choices([0, 1, 2], weights=[4, 2, 4])[0]
    a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
    return (centers[y][0] + r * math.cos(a),
            centers[y][1] + r * math.sin(a)), y

X = [(2, 0, 0, 0), (3, 0, 1, 0), (1, 0, 0, 0), (0, 2, 0, 0),
     (1, 1, 0, 0), (0, 3, 1, 0), (0, 0, 2, 0), (0, 0, 3, 1),
     (1, 0, 2, 0), (0, 0, 0, 2), (0, 0, 1, 3), (0, 1, 0, 2)]
MAIL = [(x + (1,), i // 3) for i, x in enumerate(X)]
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
delta = lambda yp, y: C[y][yp]

def best_halfspace(S): ...
def one_versus_all(S, k, A): ...
def all_pairs(S, k, A): ...
def multivector_psi(x, y, k): ...
def predict(w, x, psi, k): ...
def generalized_hinge(w, x, y, psi, delta, k): ...
def multiclass_sgd(S, psi, delta, k, d, eta, T, rng): ...

if __name__ == "__main__":
    rng = random.Random(0)
    S = [draw(rng) for _ in range(600)]
    fresh = [draw(rng) for _ in range(3000)]
    err = lambda h: sum(h(x) != y for x, y in fresh) / len(fresh)
    assert 0.15 < err(one_versus_all(S, 3, best_halfspace)) < 0.25
    assert err(all_pairs(S, 3, best_halfspace)) < 0.01
    assert multivector_psi((1, 2), 1, 3) == [0, 0, 1, 2, 0, 0]
    psi = lambda x, y: multivector_psi(x, y, 4)
    w = [int(i == j) for j in range(4) for i in range(5)]
    four = [((2, 0, 0, 0, 1), 0), ((1, 1, 0, 0, 1), 1),
            ((0, 0, 1, 1, 1), 3), ((1, 0, 0, 2, 1), 0)]
    hinges = [generalized_hinge(w, x, y, psi, delta, 4)
              for x, y in four]
    assert [round(h, 9) for h in hinges] == [0, 0.5, 0.2, 2]
    wbar = multiclass_sgd(MAIL, psi, delta, 4, 20, 0.05, 5000,
                          random.Random(0))
    assert all(predict(wbar, x, psi, 4) == y for x, y in MAIL)
    avg = sum(generalized_hinge(wbar, x, y, psi, delta, 4)
              for x, y in MAIL) / len(MAIL)
    assert avg < 0.01
    print("ok")
```

- `best_halfspace(S)` takes 2-D examples labelled \(\pm 1\), sweeps 360 directions and every threshold as in “Why the middle folder is lost”, and returns the hypothesis with the fewest mistakes.
- `one_versus_all` and `all_pairs` follow the pseudocode, breaking ties by the smallest folder.
- `multiclass_sgd` follows MulticlassSGD with \(\mathbf w^{(1)} = \mathbf 0\), sampling uniformly from `S` with `rng.choice`, and returns \(\bar{\mathbf w}\).

Then: run SGD on `MAIL` with the zero-one cost instead of `C`, and compare the margins it leaves between work and junk.

**2. Lines.** Implement TagLines and check it against brute force.

```python
import random

LINES = ["Hi Ana,", "The budget is attached.",
         "Numbers are final.", "Thanks,", "Ben"]
FEATURES = [lambda s: s.startswith(("Hi", "Dear")),
            lambda s: s.endswith(","),
            lambda s: len(s.split()) >= 3,
            lambda s: s.istitle() and len(s.split()) == 1]
x = [[int(f(s)) for s in LINES] for f in FEATURES]
w1 = [[2, -1, -1], [1, -1, 1], [-1, 2, -1], [-1, -1, 2]]
w2 = [[-2, -1, -3], [2, 1, -1], [-3, 1, 1]]

def score(y, x, w1, w2): ...
def brute_force(x, w1, w2, q): ...
def viterbi(x, w1, w2, q): ...

if __name__ == "__main__":
    assert viterbi(x, w1, w2, 3) == brute_force(x, w1, w2, 3)
    print(viterbi(x, w1, w2, 3))
    rng = random.Random(0)
    for _ in range(200):
        q, r, n = rng.randint(2, 4), rng.randint(1, 6), 3
        g = lambda rows, cols: [[rng.gauss(0, 1)
                                 for _ in range(cols)]
                                for _ in range(rows)]
        xs = [[rng.randint(0, 3) for _ in range(r)]
              for _ in range(n)]
        a, b = g(n, q), g(q, q)
        assert viterbi(xs, a, b, q) == brute_force(xs, a, b, q)
    print("ok")
```

Tags are 0 greeting, 1 body, 2 signature. `x[i][t]` is feature \(i\) of line \(t\), `w1[i][s]` the weight of feature \(i\) for tag \(s\), and `w2[s][p]` the weight of tag \(s\) following tag \(p\). `score` is \(\sum_t \langle \mathbf w, \phi \rangle\) without the \(1/r\). A correct `viterbi` prints `(0, 1, 1, 2, 2)` for the message above: greeting, two body lines, two signature lines. Then: add the cost term of the SGD argmax, \(\frac1r \mathbb 1[y'_t \neq y_t]\), to each cell, and check the loss-augmented version against brute force too.

**3. Inbox.** Implement the ranking measures and BipartiteArgmax.

```python
import itertools
import random

SCORES = [2.1, 1.4, 0.9, 0.3, -0.2, -0.4, -0.8, -1.0, -1.5, -2.2]
LABELS = [1, -1, 1, -1, -1, 1, -1, -1, -1, -1]

def pi(y): ...
def kendall_loss(yp, y): ...
def ndcg_loss(yp, y, k): ...
def counts(v, y): ...
def precision_at_k(yp, y, k): ...
def auc(yp, y): ...
def bipartite_argmax(mu, y, delta, allowed): ...

def objective(v, mu, y, delta):
    a, b, c, d = counts(v, y)
    lin = sum((s - t) * m for s, t, m in zip(v, y, mu))
    return delta(a, b, c, d) + lin

if __name__ == "__main__":
    assert pi((2, 1, 6, -1, 0.5)) == (4, 3, 5, 1, 2)
    y, yp = (5, 1, 4, 0, 3, 2), (0.9, 0.8, 0.1, -0.5, 0.7, 0.2)
    assert round(kendall_loss(yp, y), 3) == 0.333
    assert round(ndcg_loss(yp, y, 3), 3) == 0.21
    sign = [1 if s > 0 else -1 for s in SCORES]
    assert counts(sign, LABELS) == (2, 2, 1, 5)
    assert round(precision_at_k(SCORES, LABELS, 3), 3) == 0.667
    assert round(auc(SCORES, LABELS), 3) == 0.81
    f1_loss = lambda a, b, c, d: 1 - 2 * a / (2 * a + b + c)
    rng = random.Random(0)
    for _ in range(100):
        mu = [rng.gauss(0, 1) for _ in range(8)]
        lab = rng.sample([1, 1, 1, -1, -1, -1, -1, -1], 8)
        v = bipartite_argmax(mu, lab, f1_loss, lambda a, b: True)
        V = itertools.product((-1, 1), repeat=8)
        best = max(objective(u, mu, lab, f1_loss) for u in V)
        assert abs(objective(v, mu, lab, f1_loss) - best) < 1e-9
    rec = lambda a, b, c, d: 1 - a / (a + c)
    at_most_3 = lambda a, b: a + b <= 3
    for _ in range(100):
        mu = [rng.gauss(0, 1) for _ in range(8)]
        lab = rng.sample([1, 1, 1, -1, -1, -1, -1, -1], 8)
        v = bipartite_argmax(mu, lab, rec, at_most_3)
        V = [u for u in itertools.product((-1, 1), repeat=8)
             if u.count(1) <= 3]
        best = max(objective(u, mu, lab, rec) for u in V)
        assert abs(objective(v, mu, lab, rec) - best) < 1e-9
    print("ok")
```

`counts(v, y)` returns \((a, b, c, d)\) for a sign vector `v`; `bipartite_argmax` returns the maximizing sign vector as a list. Then: use it inside SGD to train a linear ranker for precision at 5 on inboxes you generate, and compare its precision at 5 with a ranker trained by the per-message hinge loss.

## Fix the bug

Time to ship. In the lab’s second act the mail room is realistic: messages are word counts, four folders overlap at their edges, and about one message in ten is important. You have a budget of fresh labelled mail that the learner never trains on: a stream of single messages with their folders, and a stream of whole mornings, each one a day’s inbox with the important messages marked.

You make two choices. First, how the folders are learned: four yes/no filters with one-versus-all, multiclass SVM with the zero-one cost, or multiclass SVM with your users’ cost table. Second, what the inbox ranker is trained and judged by: the accuracy of an “important?” filter, F₁ at threshold 0, the AUC, or precision at 5, the share of important messages among the top five each morning, which is what your users read.

Then you certify both on fresh mail. Lesson 11’s hold-out guarantee applies to any loss with values in \([0, 1]\): if \(\hat h\) was fixed before the \(m_t\) fresh examples were drawn, then with probability at least \(1 - \delta\) its true risk is within \(\sqrt{\log(2/\delta)/(2m_t)}\) of its average loss on them. The users’ cost lies in \([0, 1]\), and so does one morning’s precision at 5, with independent mornings as the examples. You need both certificates at once, so give each \(\delta/2\) and the union bound covers the pair with probability at least \(1 - \delta\).

Read the margin from the inside out: \(m_t\) is the number of fresh examples, \(\delta\) the chance that the guarantee fails, \(\log\) the natural logarithm, and the square root of \(\log(2/\delta)\) over \(2m_t\) is the half-width of the band around the measured average. In plain words: **the measured average is within the margin of the true value, except with probability \(\delta\), and the margin shrinks like one over the square root of the amount of fresh mail.** With \(\delta/2 = 0.025\) and 2,000 fresh messages it is \(\sqrt{\log 80 / 4000} = 0.033\); the snippet mirrors it for three sizes.

```python
import math

delta = 0.05 / 2
margin = lambda m: math.sqrt(math.log(2 / delta) / (2 * m))
for m in (500, 2_000, 10_000):
    print(m, round(margin(m), 4))
```

It prints `500 0.0662`, `2000 0.0331` and `10000 0.0148`. Your users’ conditions, with 95% confidence for the pair: the certified folder cost, measured cost plus margin, at most 0.06, and the certified precision at 5, measured minus margin, at least 0.6. Work out which choices can meet both before you spend the fresh mail.

[[lab-act2]]

## What people get wrong

**“If every yes/no filter is optimal, their combination is optimal.”** In the three-disc world each of the three filters is the best halfspace for its own question, and together they misfile every work message, while a perfect multiclass rule of the same shape exists. Optimal parts, graded separately, do not make an optimal whole.

**“Ties are a corner case.”** With yes/no filters they are the main case: the work messages were all three-way ties. Taking the argmax over real-valued scores makes ties rare, which is why the multiclass predictor is defined with scores, not signs.

**“The generalized hinge loss is just the hinge loss once per class.”** It is one maximum over all labels at once, with a margin requirement that scales with the cost of each confusion. That coupling is what one-versus-all lacks.

**“Many labels means many examples.”** The guarantees for multiclass SVM and SGD depend on \(\rho\) and \(B\), not explicitly on \(|\mathcal Y|\). That is why a label set of \(q^r\) tag sequences is learnable at all. The price can still appear through \(\|\Psi\|\) or through how large \(B\) must be.

**“98% accuracy means the important messages are found.”** With 2% of messages important, saying no to everything scores 98%. Measure what users see: precision at \(k\), recall at \(k\), F₁ or AUC.

**“Precision at 5 can be trained by thresholding a classifier.”** A classifier is trained for the zero-one loss and chooses its own threshold; precision at 5 fixes the number flagged and cares only about the order of the top. The bipartite surrogate trains for the measure itself.

## Recap and the next bug

1. Reductions answer a multiclass question with binary learners: one-versus-all trains \(k\), all-pairs \(\binom{k}{2}\). Because each learner is graded alone, the combination can fail even when the hypothesis class contains a perfect rule; in the three-disc world one-versus-all misfiles all 20% of the work mail.
2. A linear multiclass predictor scores each label with \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle\) and returns the best. The multivector map gives one weight vector per label; TF-IDF builds prior knowledge about words into \(\Psi\).
3. A cost \(\Delta(y', y)\) charges each confusion what it costs. Realizable ERM is a linear program; in general ERM is hard, so we use the generalized hinge loss \(\max_{y'} (\Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle)\), which bounds \(\Delta\), is convex and \(\rho\)-Lipschitz, and becomes the hinge loss for two labels.
4. Multiclass SVM and multiclass SGD learn it with guarantees \(\sqrt{8\rho^2 B^2/m}\) and \(T \ge B^2\rho^2/\varepsilon^2\), with no explicit dependence on the number of labels. SGD needs one argmax per step.
5. Structured output: when labels are sequences and \(\Psi\) is a sum of local terms, dynamic programming computes the argmax in \(O(rqn + rq^2)\) instead of \(q^r\) tries.
6. Ranking: a linear ranker scores every message with one \(\mathbf w\). Kendall tau and NDCG measure orders; their surrogates are a pairwise hinge and a generalized hinge over permutations, whose argmax is an assignment problem, a linear program by the Birkhoff–von Neumann theorem.
7. Bipartite ranking: judge by precision, recall, F₁, precision or recall at \(k\), or AUC, never by zero-one accuracy on rare positives. The surrogate’s argmax is found by trying every \((a, b)\) with the top-\(a\) and top-\(b\) scores.

The folders are right and the inbox has its important messages on top. Then a letter arrives from compliance. From next quarter, every filtering decision must be explainable to an auditor: a person has to be able to read why a given message went to junk, and check that the reason is acceptable. You open the model you are so proud of and find a weight vector with four blocks of several thousand word weights each. The honest answer to “why junk?” is “because this sum of nine thousand products beat that one”, and no auditor will sign that. Is there a hypothesis class that a person can read, a short sequence of yes/no questions about a message, that is still learnable? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Take \(\mathbf w = \mathbf 0\) with any \(\Psi\) and any cost \(\Delta\). What does \(h_{\mathbf w}\) predict, and what is the generalized hinge loss \(\ell(\mathbf 0, (\mathbf x, y))\)? Explain why \(\ell(\mathbf 0, (\mathbf x, y)) \ge \Delta(h_{\mathbf 0}(\mathbf x), y)\) holds no matter how the argmax breaks its ties, pointing to the step of the derivation that makes this so. For your users’ cost table, what is \(\ell(\mathbf 0, \cdot)\) on a work message, and on a promotion?
2. In the proof that the middle folder has no good halfspace, the work disc was the midpoint of the other two and carried 20% of the mail against 40% for each neighbour. (a) Move the work disc to \((-2, 5)\), still between its neighbours but not halfway. Which line of the proof changes, and does the conclusion survive? (b) Now keep the positions but give work 50% of the mail and the others 25% each. Find a halfspace for “work against the rest” with error below 50%. What condition on the three shares does the proof really need?
3. In the ten-message inbox, a better ranker moves message 6 (important) from score \(-0.4\) to \(1.0\) and leaves every other score alone. Recompute \(a, b, c, d\) at \(\theta = 0\), F₁, precision at 3, recall at 3, the AUC and the zero-one loss. Which of these measures moved, and which did not? Explain why precision at 3 did not change even though an important message moved up.

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (well-separated folders are learnable with zero loss).** Let \(S \subset \mathbb R^n \times [k]\) be a set of examples for which there are vectors \(\boldsymbol\mu_1, \dots, \boldsymbol\mu_k\) such that every \((\mathbf x, y) \in S\) lies in the ball of radius \(r \ge 1\) around \(\boldsymbol\mu_y\), and \(\|\boldsymbol\mu_i - \boldsymbol\mu_j\| \ge 4r\) for all \(i \neq j\). Append a constant 1 to every message and apply the multivector construction, so \(\Psi(\mathbf x, y) \in \mathbb R^{k(n+1)}\) holds \((x_1, \dots, x_n, 1)\) in block \(y\). With the zero-one cost, show that there is a \(\mathbf w \in \mathbb R^{k(n+1)}\) with \(\ell(\mathbf w, (\mathbf x, y)) = 0\) for every \((\mathbf x, y) \in S\).
- **Problem 2 (the multiclass Perceptron).** Consider this algorithm.

  ```
  MulticlassPerceptron((x_1, y_1), ..., (x_m, y_m), Ψ):
    w(1) ← 0 in R^d
    for t = 1, 2, ...:
      if some i and y ≠ y_i have
         <w(t), Ψ(x_i, y_i)> ≤ <w(t), Ψ(x_i, y)>:
        w(t+1) ← w(t) + Ψ(x_i, y_i) − Ψ(x_i, y)
      else:
        output w(t)
  ```

  Assume there is a \(\mathbf w^\star\) with \(\langle \mathbf w^\star, \Psi(\mathbf x_i, y_i) \rangle \ge \langle \mathbf w^\star, \Psi(\mathbf x_i, y) \rangle + 1\) for all \(i\) and all \(y \neq y_i\), and let \(R = \max_{i, y} \|\Psi(\mathbf x_i, y_i) - \Psi(\mathbf x_i, y)\|\). Prove that the algorithm stops after at most \((R\|\mathbf w^\star\|)^2\) updates, and that when it stops, \(y_i = \operatorname{argmax}_y \langle \mathbf w^{(t)}, \Psi(\mathbf x_i, y) \rangle\) for every \(i\).
- **Problem 3 (sorting maximizes the matching sum).** Let \(V\) be the set of permutations of \([r]\), written as vectors with distinct entries in \([r]\), and let \(\mathbf y' \in \mathbb R^r\) have distinct entries. Prove that \(\pi(\mathbf y') = \operatorname{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i y'_i\), and that this maximizer is unique. Then use the same idea to prove that \(G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\) for any nondecreasing discount \(D\) and nonnegative relevances \(\mathbf y\), which is what keeps the NDCG loss in \([0, 1]\).
