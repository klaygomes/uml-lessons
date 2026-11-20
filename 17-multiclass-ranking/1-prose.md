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

**One-versus-all.** Given a training sequence \(S = ((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m))\), build \(k\) binary training sequences. In \(S_i\), every example keeps its message and gets the label \(+1\) if its folder is \(i\) and \(-1\) otherwise:

\[ S_i = \left( (\mathbf x_1, (-1)^{\mathbb 1[y_1 \neq i]}), \dots, (\mathbf x_m, (-1)^{\mathbb 1[y_m \neq i]}) \right). \]

The exponent trick just says “\(+1\) when \(y_t = i\), \(-1\) when not”. Run a binary learner \(A\) on each \(S_i\) to get \(h_i = A(S_i)\), and predict with

\[ h(\mathbf x) \in \operatorname*{argmax}_{i \in [k]} h_i(\mathbf x). \]

The “\(\in\)” admits what the team discovered: when several \(h_i\) say \(+1\), or none does, the argmax is a tie and something has to break it, for instance taking the smallest index. When each \(h_i\) is a halfspace \(\operatorname{sign}(\langle \mathbf w_i, \mathbf x \rangle)\), there is a better option. The number \(\langle \mathbf w_i, \mathbf x \rangle\) before the sign can be read as the filter’s confidence, and we can take the argmax over those real numbers instead, which rarely ties.

**All-pairs.** For every pair of folders \(i < j\), collect only the examples whose folder is \(i\) or \(j\), label them \(+1\) for \(i\) and \(-1\) for \(j\), and train \(h_{i,j}\) on that. That is \(\binom{k}{2}\) binary learners, six for our four folders. To predict, every pair votes, and the folder with the most wins is the answer:

\[ h(\mathbf x) \in \operatorname*{argmax}_{i \in \mathcal Y} \sum_{j \in \mathcal Y} \operatorname{sign}(j - i)\, h_{i,j}(\mathbf x), \]

where for \(i > j\) we read \(h_{i,j}\) as \(h_{j,i}\), and the factor \(\operatorname{sign}(j - i)\) flips its vote so that a \(+1\) always counts for the lower-numbered folder of the pair, and the term with \(j = i\) is zero.

Both reductions are simple and reuse code you already trust. Their weakness is in one sentence: the binary learner has no idea its answers will be combined. Each \(h_i\) is trained to be as accurate as possible on its own question, and nothing in that goal asks whether the \(k\) answers will fit together. Let’s watch that go wrong with real numbers.

## Worked example: the folder in the middle

Shrink the world so you can draw it. Describe each message by two numbers, so it is a point in the plane. Messages from three folders sit in three discs of radius 1, side by side at the same height: personal mail around \((-5, 5)\), work around \((0, 5)\), promotions around \((5, 5)\). Within its disc, a message is spread uniformly. Of all messages, 40% are personal, 20% are work and 40% are promotions. (The two coordinates are invented scores; think of the first as “how commercial the wording is” and the second as a length the three folders share. The geometry is what matters.)

[[diagram:ova-trap]]

Before reading on, consider the filter “work against the rest”. The work disc is squeezed between the other two. Which halfspace would you choose for it?

Try a few. A line that cuts the work disc off from the left also takes all of personal. A line that cuts it off from the right takes all of promotions. A horizontal line cutting the top cap of the work disc cuts the same caps off its neighbours, and they carry twice the mass each. Every halfspace that captures some work mail pays more in the other folders than it gains. The best halfspace is the one that says no to everything, with error 20%, exactly the work mail. We will prove that in a moment.

Now assemble the team’s filter. “Personal against the rest” and “promotions against the rest” are easy: the vertical lines \(x_1 = -2.5\) and \(x_1 = 2.5\) separate them perfectly, so the best halfspaces have error 0. Take a work message. All three filters say no. The argmax is a three-way tie, and breaking it by the smallest index sends it to personal. Every work message is misfiled. The filter that was built from three optimal pieces errs on 20% of all mail.

Is the hypothesis class to blame? No. Here are three weight vectors, one per folder:

\[ \mathbf w_{\text{personal}} = \left(-\tfrac{1}{\sqrt 2}, \tfrac{1}{\sqrt 2}\right), \quad \mathbf w_{\text{work}} = (0, 1), \quad \mathbf w_{\text{promotions}} = \left(\tfrac{1}{\sqrt 2}, \tfrac{1}{\sqrt 2}\right), \]

and the rule “file \(\mathbf x\) in the folder whose weight vector gives the largest \(\langle \mathbf w_y, \mathbf x \rangle\)”. Check a work message \(\mathbf x = (x_1, x_2)\) by hand. It scores \(x_2\) for work, \((x_2 - x_1)/\sqrt 2\) for personal and \((x_1 + x_2)/\sqrt 2\) for promotions. Work wins both comparisons exactly when \((\sqrt 2 - 1)\, x_2 > |x_1|\). In the work disc \(x_2 \ge 4\) and \(|x_1| \le 1\), and \(0.414 \times 4 = 1.66 > 1\), so work always wins. A personal message has \(x_1 \le -4\) and \(x_2 \le 6\), so personal beats work because \(-x_1 \ge 4 > 0.414 \times 6\), and beats promotions because \(x_1 < 0\). Promotions is the mirror image. Zero mistakes.

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

Hold on to that pair of facts. The class of rules “take the folder with the largest \(\langle \mathbf w_y, \mathbf x \rangle\)” contains a perfect hypothesis, so its approximation error, in the sense of lesson 5, is zero. And one-versus-all, run with a perfect binary learner, still cannot find it. The fault is in how the pieces are trained, not in what they can express.

## Why the middle folder is lost

Let’s make the claim about the work filter exact, because “try a few lines” is not a proof.

**Proposition (the middle folder has no good halfspace).** In the three-disc world, every halfspace has error at least 20% on the binary problem “work against the rest”. The error is exactly 20% only if the halfspace contains a zero-probability part of the work disc.

*Proof.* Write a halfspace as \(H = \{\mathbf x : \langle \mathbf u, \mathbf x \rangle + b > 0\}\) with \(\|\mathbf u\| = 1\); it predicts \(+1\) (work) inside \(H\) and \(-1\) outside. The two constant classifiers, all \(+1\) and all \(-1\), have errors 80% and 20%, so the claim holds for them, and from now on \(H\) is a genuine halfspace.

A message of folder \(i\) is \(\mathbf c_i + \mathbf z\), where \(\mathbf c_i\) is the centre of its disc and \(\mathbf z\) is uniform on the unit disc. It lands in \(H\) when \(\langle \mathbf u, \mathbf z \rangle + t_i > 0\), where \(t_i = \langle \mathbf u, \mathbf c_i \rangle + b\) is the signed distance from the centre to the boundary line. The uniform distribution on the unit disc looks the same in every direction, so the probability of that event depends only on \(t_i\), not on \(\mathbf u\). Call it \(g(t_i)\). As \(t\) grows the event only gets larger, so \(g\) is nondecreasing, and of course \(0 \le g \le 1\).

The error of \(H\) adds three parts: personal mail inside \(H\), promotions inside \(H\), and work mail outside \(H\):

\[ \operatorname{err}(H) = 0.4\, g(t_1) + 0.4\, g(t_3) + 0.2\,\bigl(1 - g(t_2)\bigr), \]

with \(t_1, t_2, t_3\) for personal, work and promotions. Now the geometry enters. The work centre is the midpoint of the other two, \(\mathbf c_2 = (\mathbf c_1 + \mathbf c_3)/2\), and \(t\) is an affine function of the centre, so \(t_2 = (t_1 + t_3)/2\). An average cannot exceed both of the numbers it averages, so \(\max(t_1, t_3) \ge t_2\), and because \(g\) is nondecreasing, \(g(t_1) + g(t_3) \ge g(\max(t_1, t_3)) \ge g(t_2)\). Substitute:

\[ \operatorname{err}(H) \ge 0.4\, g(t_2) + 0.2 - 0.2\, g(t_2) = 0.2 + 0.2\, g(t_2) \ge 0.2. \]

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

**Definition (class-sensitive feature mapping).** A **class-sensitive feature mapping** is a function \(\Psi: \mathcal X \times \mathcal Y \to \mathbb R^d\). It takes a message together with a candidate label and returns a vector. You can read the entries of \(\Psi(\mathbf x, y)\) as scores that say how well the label \(y\) fits the message \(\mathbf x\).

**Definition (linear multiclass predictor).** Given \(\Psi\) and a weight vector \(\mathbf w \in \mathbb R^d\), the **linear multiclass predictor** is

\[ h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y) \rangle. \]

The number \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle\) is the **score** of label \(y\) for message \(\mathbf x\). For a set \(W \subseteq \mathbb R^d\) of allowed weight vectors, for example the ball \(W = \{\mathbf w : \|\mathbf w\| \le B\}\), the pair \((\Psi, W)\) defines a hypothesis class

\[ \mathcal H_{\Psi, W} = \left\{ \mathbf x \mapsto \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y) \rangle \;:\; \mathbf w \in W \right\}. \]

With \(\mathcal Y = \{\pm 1\}\), \(\Psi(\mathbf x, y) = y\mathbf x\) and \(W = \mathbb R^d\), this is exactly the class of homogeneous halfspaces. The whole question is now how to choose \(\Psi\), and that is a question of prior knowledge, the inductive bias of lesson 1 in a new costume. Two constructions show the range.

**The multivector construction.** Let \(\mathcal X = \mathbb R^n\) and \(\mathcal Y = \{1, \dots, k\}\), and set \(d = nk\). Define

\[ \Psi(\mathbf x, y) = [\, \underbrace{0, \dots, 0}_{\in \mathbb R^{(y-1)n}},\ \underbrace{x_1, \dots, x_n}_{\in \mathbb R^n},\ \underbrace{0, \dots, 0}_{\in \mathbb R^{(k-y)n}} \,]. \]

So \(\Psi(\mathbf x, y)\) is \(k\) blocks of length \(n\), all zero except block \(y\), which holds \(\mathbf x\). Cut \(\mathbf w \in \mathbb R^{nk}\) into the same blocks, \(\mathbf w = [\mathbf w_1; \dots; \mathbf w_k]\). In the inner product only block \(y\) of \(\mathbf w\) meets a nonzero block of \(\Psi\), so

\[ \langle \mathbf w, \Psi(\mathbf x, y) \rangle = \langle \mathbf w_y, \mathbf x \rangle, \qquad h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w_y, \mathbf x \rangle. \]

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

Losing work or personal mail to junk costs the most. A promotion in the wrong place costs little, wherever it goes. Junk in a real folder is an annoyance. The empirical risk of a hypothesis is the average cost, and the true risk is its expectation,

\[ L_S(h) = \frac{1}{m} \sum_{i=1}^m \Delta(h(\mathbf x_i), y_i), \qquad L^\Delta_{\mathcal D}(h) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}\left[ \Delta(h(\mathbf x), y) \right]. \]

## ERM, and why it is hard

With the class \(\mathcal H_{\Psi, W}\) and the cost \(\Delta\) fixed, the ERM rule of lesson 2 says: find \(\mathbf w \in W\) whose predictor has the smallest \(L_S\).

In the realizable case with \(W = \mathbb R^d\), this is a feasibility problem. We need a \(\mathbf w\) with \(y_i = \operatorname{argmax}_y \langle \mathbf w, \Psi(\mathbf x_i, y) \rangle\) for every \(i\), as the unique winner, which is the same as the set of linear inequalities

\[ \forall i \in [m],\ \forall y \in \mathcal Y \setminus \{y_i\}: \quad \langle \mathbf w, \Psi(\mathbf x_i, y_i) \rangle > \langle \mathbf w, \Psi(\mathbf x_i, y) \rangle. \]

That is \(m(k - 1)\) strict inequalities in \(\mathbf w\). The strictness is harmless, by the same rescaling as for halfspaces in lesson 9: if some \(\mathbf w\) satisfies them, let \(\gamma > 0\) be the smallest gap among the finitely many, and \(\mathbf w/\gamma\) satisfies every difference \(\ge 1\). So finding a consistent \(\mathbf w\) is a linear program. There is also a multiclass Perceptron, with a mistake bound just like lesson 9’s; you will prove it yourself in the problems.

Without realizability, ERM over this class is computationally hard in general, just as it is for halfspaces. Lesson 12’s answer applies here too: replace the loss with a convex **surrogate** that upper bounds it, and minimize that.

## The generalized hinge loss

For halfspaces the surrogate was the hinge loss \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\). We need its multiclass version, and the cleanest way is to derive it, one inequality at a time, from the thing we want to bound: \(\Delta(h_{\mathbf w}(\mathbf x), y)\), where \(h_{\mathbf w}(\mathbf x) = \operatorname{argmax}_{y'} \langle \mathbf w, \Psi(\mathbf x, y') \rangle\).

*Step 1.* The predicted label has the largest score, in particular at least the score of the correct label: \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle \le \langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x)) \rangle\).

*Step 2.* So adding the difference of those scores to the cost can only increase it:

\[ \Delta(h_{\mathbf w}(\mathbf x), y) \le \Delta(h_{\mathbf w}(\mathbf x), y) + \langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x)) - \Psi(\mathbf x, y) \rangle. \]

*Step 3.* The right side is the expression \(\Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle\) at one particular label, \(y' = h_{\mathbf w}(\mathbf x)\). The maximum over all labels is at least as large.

**Definition (generalized hinge loss).**

\[ \ell(\mathbf w, (\mathbf x, y)) = \max_{y' \in \mathcal Y} \left( \Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle \right). \]

Steps 1 to 3 prove \(\ell(\mathbf w, (\mathbf x, y)) \ge \Delta(h_{\mathbf w}(\mathbf x), y)\). Notice that the proof never asked which label wins a tie: whatever label the argmax returns, it scores at least as high as \(y\), and that is all Step 1 used.

The term \(y' = y\) of the max is \(\Delta(y, y) + 0 = 0\), so \(\ell \ge 0\). When is \(\ell\) zero? Exactly when every other term is at most 0, that is, when the correct label beats every other label by at least the cost of confusing them:

\[ \forall y' \neq y: \quad \langle \mathbf w, \Psi(\mathbf x, y) \rangle \ge \langle \mathbf w, \Psi(\mathbf x, y') \rangle + \Delta(y', y). \]

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

**Proposition (convex and Lipschitz).** For fixed \((\mathbf x, y)\), the function \(\mathbf w \mapsto \ell(\mathbf w, (\mathbf x, y))\) is convex and \(\rho\)-Lipschitz with \(\rho = \max_{y' \in \mathcal Y} \|\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\|\).

*Proof.* For each label \(y'\), the function \(f_{y'}(\mathbf w) = \Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle\) is affine in \(\mathbf w\), hence convex, and \(\ell\) is their maximum. Lesson 12 showed that a maximum of convex functions is convex, so \(\ell\) is convex.

For the Lipschitz bound, first each \(f_{y'}\): by Cauchy–Schwarz, \(|f_{y'}(\mathbf u) - f_{y'}(\mathbf v)| = |\langle \mathbf u - \mathbf v, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle| \le \rho\, \|\mathbf u - \mathbf v\|\). Then the maximum. Let \(y^*\) be a label achieving \(\ell(\mathbf u) = f_{y^*}(\mathbf u)\). Since \(\ell(\mathbf v) \ge f_{y^*}(\mathbf v)\),

\[ \ell(\mathbf u) - \ell(\mathbf v) \le f_{y^*}(\mathbf u) - f_{y^*}(\mathbf v) \le \rho\, \|\mathbf u - \mathbf v\|. \]

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

**Multiclass SVM.** Lesson 13’s regularized loss minimization, applied to the generalized hinge loss, gives this rule.

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

**Theorem (multiclass SVM).** Let \(\Psi\) satisfy \(\|\Psi(\mathbf x, y)\| \le \rho/2\) for all \(\mathbf x, y\), let \(B > 0\), and run multiclass SVM with \(\lambda = \sqrt{2\rho^2/(B^2 m)}\) on \(S \sim \mathcal D^m\), getting \(\mathbf w = A(S)\). Write \(L^{\text{g-hinge}}_{\mathcal D}(\mathbf w) = \mathbb E_{(\mathbf x, y) \sim \mathcal D}[\ell(\mathbf w, (\mathbf x, y))]\). Then

\[ \mathbb E_{S}\left[ L^\Delta_{\mathcal D}(h_{\mathbf w}) \right] \le \mathbb E_{S}\left[ L^{\text{g-hinge}}_{\mathcal D}(\mathbf w) \right] \le \min_{\mathbf u: \|\mathbf u\| \le B} L^{\text{g-hinge}}_{\mathcal D}(\mathbf u) + \sqrt{\frac{8\rho^2 B^2}{m}}. \]

*Proof.* First, the loss is \(\rho\)-Lipschitz. By the triangle inequality, \(\|\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\| \le \|\Psi(\mathbf x, y')\| + \|\Psi(\mathbf x, y)\| \le \rho/2 + \rho/2 = \rho\) for every pair of labels, so the proposition above gives Lipschitz constant at most \(\rho\), and the loss is convex.

The left inequality. For every example, \(\Delta(h_{\mathbf w}(\mathbf x), y) \le \ell(\mathbf w, (\mathbf x, y))\). Taking the expectation over \((\mathbf x, y) \sim \mathcal D\) keeps the inequality, so \(L^\Delta_{\mathcal D}(h_{\mathbf w}) \le L^{\text{g-hinge}}_{\mathcal D}(\mathbf w)\) for every \(\mathbf w\), in particular for the random \(\mathbf w = A(S)\); taking the expectation over \(S\) keeps it again.

The right inequality. Apply the oracle inequality with any \(\mathbf u\) with \(\|\mathbf u\| \le B\) as \(\mathbf w^*\). Then \(\lambda\|\mathbf u\|^2 \le \lambda B^2\), and with the chosen \(\lambda\) the two error terms are equal:

\[ \lambda B^2 = \sqrt{\frac{2\rho^2 B^2}{m}}, \qquad \frac{2\rho^2}{\lambda m} = \frac{2\rho^2}{m} \sqrt{\frac{B^2 m}{2\rho^2}} = \sqrt{\frac{2\rho^2 B^2}{m}}. \]

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

**SGD for multiclass learning.** Lesson 14’s stochastic gradient descent needs one subgradient of the loss per example. The loss is a maximum of affine functions, and lesson 14 showed that the gradient of any function achieving the maximum is a subgradient of the maximum. The affine function for label \(y'\) has gradient \(\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\). So: find the label that achieves the max, and step against its gradient.

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

A type 1 entry adds up feature \(i\) over the lines tagged \(j\): it can learn that “starts with Hi” suggests a greeting. A type 2 entry counts how often tag \(i\) follows tag \(j\): it can learn that a signature is followed by more signature, never by a greeting. The weights decide which of these matter.

The trick is that \(\Psi\) is a sum over positions, each term depending only on the current tag and the previous one. Set \(y_0 = 0\), a start symbol that is not a tag, and define

\[ \phi_{i,j,1}(\mathbf x, y_t, y_{t-1}) = x_{i,t}\, \mathbb 1[y_t = j], \qquad \phi_{i,j,2}(\mathbf x, y_t, y_{t-1}) = \mathbb 1[y_t = i]\, \mathbb 1[y_{t-1} = j]. \]

(The \(\phi\) also reads which column \(t\) it is at; we keep that implicit.) Then \(\Psi(\mathbf x, \mathbf y) = \frac{1}{r}\sum_{t=1}^r \phi(\mathbf x, y_t, y_{t-1})\): at \(t = 1\) the type 2 term vanishes because \(y_0 = 0\) matches no tag \(j\), which is why the type 2 sum starts at \(t = 2\). The positive factor \(1/r\) does not change which \(\mathbf y\) wins, so

\[ h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{\mathbf y \in \mathcal Y} \sum_{t=1}^r \langle \mathbf w, \phi(\mathbf x, y_t, y_{t-1}) \rangle. \]

**Dynamic programming.** For a tag \(s\) and a position \(\tau\), let \(M_{s,\tau}\) be the best score of any first \(\tau\) tags that end in \(s\):

\[ M_{s,\tau} = \max_{(y_1, \dots, y_\tau):\, y_\tau = s} \sum_{t=1}^\tau \langle \mathbf w, \phi(\mathbf x, y_t, y_{t-1}) \rangle. \]

The best full score is \(\max_s M_{s,r}\), and the table obeys the recursion

\[ M_{s,\tau} = \max_{s'} \left( M_{s',\tau-1} + \langle \mathbf w, \phi(\mathbf x, s, s') \rangle \right). \]

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

Here \(\operatorname{sign}(0) = 0\), so a pair the users rated equal counts as a disagreement whenever the scores separate it. This loss sees partial credit: one swap costs one pair.

**Normalized discounted cumulative gain (NDCG).** Users read the top of the list, so mistakes there should cost more. Choose a nondecreasing **discount** \(D: \mathbb N \to \mathbb R_+\) that weights positions, and define the **discounted cumulative gain**

\[ G(\mathbf y', \mathbf y) = \sum_{i=1}^r D(\pi(\mathbf y')_i)\, y_i. \]

Read \(y_i\) as message \(i\)’s true relevance; the gain adds the relevances, each weighted by the discount of the position \(\mathbf y'\) put it in. If all relevances are nonnegative, then \(0 \le G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\): the first inequality because every term is nonnegative, the second because, for a nondecreasing \(D\), putting the largest relevances in the positions with the largest discounts is the best you can do (a sorting fact you will prove in the problems). So the **NDCG loss**

\[ \Delta(\mathbf y', \mathbf y) = 1 - \frac{G(\mathbf y', \mathbf y)}{G(\mathbf y, \mathbf y)} = \frac{1}{G(\mathbf y, \mathbf y)} \sum_{i=1}^r \left( D(\pi(\mathbf y)_i) - D(\pi(\mathbf y')_i) \right) y_i \]

lies in \([0, 1]\), and it is 0 when \(\pi(\mathbf y') = \pi(\mathbf y)\). A typical discount, with a parameter \(k < r\), is

\[ D(i) = \begin{cases} \dfrac{1}{\log_2(r - i + 2)} & \text{if } i \in \{r - k + 1, \dots, r\}, \\ 0 & \text{otherwise.} \end{cases} \]

The top position, \(i = r\), gets \(1/\log_2 2 = 1\), the next \(1/\log_2 3 = 0.63\), and everything below the top \(k\) gets nothing at all. That is how search engines are judged, and it is how your users read their mail.

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

**A hinge loss for Kendall tau.** Look at one pair \(i < j\) and write \(s = \operatorname{sign}(y_i - y_j)\). When \(s \neq 0\), the scores disagree with the truth exactly when \(s\,(y'_i - y'_j) \le 0\). When \(s = 0\), the indicator \(\mathbb 1[s(y'_i - y'_j) \le 0]\) is 1, which is at least the true term. So in every case

\[ \mathbb 1\left[ \operatorname{sign}(y'_i - y'_j) \neq \operatorname{sign}(y_i - y_j) \right] \le \mathbb 1\left[ s\,(y'_i - y'_j) \le 0 \right]. \]

For a linear ranker, \(y'_i - y'_j = \langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\). And for any number \(a\), \(\mathbb 1[a \le 0] \le \max\{0, 1 - a\}\): if \(a \le 0\) the right side is at least 1, otherwise the left side is 0. With \(a = s\langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\), averaging over pairs gives

\[ \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) \le \frac{2}{r(r-1)} \sum_{i=1}^{r-1} \sum_{j=i+1}^r \max\left\{0,\ 1 - \operatorname{sign}(y_i - y_j) \langle \mathbf w, \mathbf x_i - \mathbf x_j \rangle\right\}. \]

Each term is a hinge loss in \(\mathbf w\), hence convex, and it is \(\|\mathbf x_i - \mathbf x_j\|\)-Lipschitz because \(|s| \le 1\). An average with nonnegative weights summing to 1 of convex, \(\rho\)-Lipschitz functions is convex and \(\rho\)-Lipschitz, so the surrogate is convex and Lipschitz with \(\rho \le \max_{i,j} \|\mathbf x_i - \mathbf x_j\|\). SGD applies directly.

**A hinge loss for NDCG.** NDCG depends on the scores only through the permutation they induce, which looks hopeless for a gradient method. Here is the observation that unlocks it. Let \(V\) be the set of all permutations of \([r]\), each written as a vector \(\mathbf v \in [r]^r\) with distinct entries. If the entries of \(\mathbf y'\) are distinct, then

\[ \pi(\mathbf y') = \operatorname*{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i\, y'_i. \]

In words: to make \(\sum_i v_i y'_i\) as large as possible, give the largest position number to the largest score, the next largest to the next, and so on, which is exactly what sorting does. This is the sorting fact again; proving it is Problem 3. Python checks it on a thousand random score vectors against all 720 permutations of six positions:

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

so the linear ranker is a linear multiclass predictor whose labels are permutations. The generalized hinge derivation goes through word for word (write \(\Delta(\mathbf v, \mathbf y)\) for the NDCG loss of the order \(\mathbf v\), since \(\pi(\mathbf v) = \mathbf v\)):

\[ \begin{aligned} \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) &\le \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) + \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(h_{\mathbf w}(\bar{\mathbf x}))) \rangle - \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(\mathbf y)) \rangle \\ &\le \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \langle \mathbf w, \Psi(\bar{\mathbf x}, \mathbf v) \rangle - \langle \mathbf w, \Psi(\bar{\mathbf x}, \pi(\mathbf y)) \rangle \right] \\ &= \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \sum_{i=1}^r (v_i - \pi(\mathbf y)_i) \langle \mathbf w, \mathbf x_i \rangle \right]. \end{aligned} \]

The first line holds because the predicted permutation maximizes \(\langle \mathbf w, \Psi(\bar{\mathbf x}, \cdot) \rangle\), so the added difference is nonnegative. The second takes the max over all permutations. The last line expands \(\Psi\). The right side is a maximum of affine functions of \(\mathbf w\), so it is convex.

**The bottleneck is an assignment problem.** An SGD step needs the \(\mathbf v\) that achieves that max. Write \(\mu_i = \langle \mathbf w, \mathbf x_i \rangle\). The term \(-\sum_i \pi(\mathbf y)_i \mu_i\) does not depend on \(\mathbf v\), so drop it. Expand the NDCG loss of \(\mathbf v\): \(\Delta(\mathbf v, \mathbf y) = 1 - \sum_i D(v_i) y_i / G(\mathbf y, \mathbf y)\), and the 1 does not depend on \(\mathbf v\) either. What is left is to maximize \(\sum_i \left( \mu_i v_i - D(v_i)\, y_i / G(\mathbf y, \mathbf y) \right)\), which is the same as

\[ \operatorname*{argmin}_{\mathbf v \in V} \sum_{i=1}^r \left( \alpha_i v_i + \beta_i D(v_i) \right), \qquad \alpha_i = -\mu_i, \quad \beta_i = \frac{y_i}{G(\mathbf y, \mathbf y)}. \]

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

The 0/1 constraint is what keeps this from being a linear program. Drop it and keep the rest: a matrix with nonnegative entries whose rows and columns each sum to 1 is called **doubly stochastic**, and the relaxed problem is

\[ \min_{B} \langle A, B \rangle \quad \text{subject to } B \text{ doubly stochastic}, \]

a linear program. The two problems have the same answer, thanks to a classical theorem we will use without proof.

**Theorem (Birkhoff and von Neumann).** The set of doubly stochastic \(r \times r\) matrices is the convex hull of the permutation matrices: every doubly stochastic matrix is a convex combination of permutation matrices.

**Lemma (the relaxation is exact).** Some optimal solution of the relaxed problem is a permutation matrix, and it is optimal for the assignment problem too.

*Proof.* Let \(B\) be an optimal solution of the relaxed problem. By the theorem, \(B = \sum_i \gamma_i C_i\) with each \(C_i\) a permutation matrix, each \(\gamma_i > 0\), and \(\sum_i \gamma_i = 1\). Each \(C_i\) is itself doubly stochastic, so it is a feasible point of the relaxed problem, and since \(B\) is optimal, \(\langle A, B \rangle \le \langle A, C_i \rangle\) for every \(i\).

Suppose the inequality were strict for every \(i\). Then, using linearity of \(\langle A, \cdot \rangle\) and \(\sum_i \gamma_i = 1\),

\[ \langle A, B \rangle = \Big\langle A, \sum_i \gamma_i C_i \Big\rangle = \sum_i \gamma_i \langle A, C_i \rangle > \sum_i \gamma_i \langle A, B \rangle = \langle A, B \rangle, \]

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

Here is the trap the team fell into. It looks as if a binary filter solves it: label each message, put the yeses on top. But a binary learner minimizes the zero-one loss, and the zero-one loss is the wrong measure when the relevant messages are rare. Suppose 20 of 1,000 messages are important, and the filter says no to all of them.

```python
r, P = 1_000, 20
a, b, c, d = 0, 0, P, r - P
zero_one = (b + c) / r
recall = a / (a + c)
F1 = 2 * a / (2 * a + b + c)
print(zero_one, recall, F1, 1 - F1)
```

It prints `0.02 0.0 0.0 1.0`. The numbers \(a, b, c, d\) and the measures need defining, so let’s do that. A ranker outputs scores \(\mathbf y' \in \mathbb R^r\), and a **threshold** \(\theta \in \mathbb R\) turns them into yes/no answers, \(\operatorname{sign}(y'_i - \theta)\). (Ties at the threshold are measure-zero for real-valued scores; read the sign of 0 as \(-1\) here.) Four counts describe the result:

\[ \begin{aligned} a &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{true positives} \\ b &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{false positives} \\ c &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{false negatives} \\ d &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{true negatives} \end{aligned} \]

The **recall** (also called sensitivity) is \(a/(a + c)\), the fraction of important messages the ranker catches. The **precision** is \(a/(a + b)\), the fraction of flagged messages that are important. The **specificity** is \(d/(d + b)\), the fraction of unimportant messages correctly left alone. The zero-one loss is \((b + c)/(a + b + c + d)\).

Lower the threshold and you flag more: recall rises, reaching 1 at \(\theta = -\infty\), while precision and specificity usually fall. These measures combine precision and recall in different ways.

- **Averaging sensitivity and specificity:** \(\frac12\left( \frac{a}{a+c} + \frac{d}{d+b} \right)\), the accuracy on important messages averaged with the accuracy on the others, at \(\theta = 0\); the loss is 1 minus it.
- **F₁ score:** the harmonic mean of precision and recall, \(\frac{2}{1/\text{precision} + 1/\text{recall}} = \frac{2a}{2a + b + c}\), at \(\theta = 0\); loss \(1 - F_1\). It is 1 when both are 1, and 0 as soon as either is 0.
- **F\(_\beta\) score:** like F₁ but with \(\beta^2\) times more weight on recall, \(\frac{1 + \beta^2}{1/\text{precision} + \beta^2/\text{recall}} = \frac{(1 + \beta^2)a}{(1 + \beta^2)a + b + \beta^2 c}\), at \(\theta = 0\).
- **Recall at \(k\):** the recall when at most \(k\) messages may be flagged: choose \(\theta\) so that \(a + b \le k\). Right when a person can only look at \(k\) messages.
- **Precision at \(k\):** the precision when at least \(k\) messages are flagged: choose \(\theta\) so that \(a + b \ge k\). The natural choice puts \(\theta\) just below the \(k\)-th largest score, so exactly the top \(k\) are flagged; that is what your users mean by “are the top five any good?”.

Measures like these, which depend on the predictions for the whole list at once and not on each message separately, are called **multivariate performance measures**. One more is widely used, though it is not built from a threshold. The **AUC** (area under the ROC curve) is the fraction of (important, unimportant) pairs in which the important message gets the higher score. It is one minus the Kendall tau loss restricted to the pairs with different labels, and the Kendall surrogate above handles it: for a pair with \(y_i = y_j\) the hinge term is \(\max\{0, 1 - 0\} = 1\) whatever \(\mathbf w\) is, so in the bipartite case the surrogate is a constant plus the hinge over the mixed pairs alone.

[[diagram:threshold]]

Back to the “nothing is important” filter: zero-one loss 2%, recall 0, so F₁ is 0 and its loss is 1. Now a real ranker on one morning of ten messages, three of them important, with scores sorted from the top:

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

\[ b(\mathbf y') = \operatorname*{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i\, y'_i. \]

For \(\theta = 0\), take \(V = \{\pm 1\}^r\): each \(v_i\) is chosen on its own, and \(v_i y'_i\) is largest when \(v_i\) has the sign of \(y'_i\). For precision at \(k\), take \(V_{\ge k}\), the vectors with at least \(k\) entries equal to \(+1\); for recall at \(k\), \(V_{\le k}\), the vectors with at most \(k\). (Checking that these give the right thresholds is a good exercise.)

Then, assuming the true labels \(\mathbf y\) lie in \(V\),

\[ \begin{aligned} \Delta(h_{\mathbf w}(\bar{\mathbf x}), \mathbf y) &= \Delta(b(h_{\mathbf w}(\bar{\mathbf x})), \mathbf y) \\ &\le \Delta(b(h_{\mathbf w}(\bar{\mathbf x})), \mathbf y) + \sum_{i=1}^r \left( b_i(h_{\mathbf w}(\bar{\mathbf x})) - y_i \right) \langle \mathbf w, \mathbf x_i \rangle \\ &\le \max_{\mathbf v \in V} \left[ \Delta(\mathbf v, \mathbf y) + \sum_{i=1}^r (v_i - y_i) \langle \mathbf w, \mathbf x_i \rangle \right]. \end{aligned} \]

The first line holds because the measure only sees \(b\). The second adds a nonnegative number: \(b\) maximizes \(\sum_i v_i \langle \mathbf w, \mathbf x_i \rangle\) over \(V\), and \(\mathbf y\) is in \(V\), so \(\sum_i b_i \langle \mathbf w, \mathbf x_i \rangle \ge \sum_i y_i \langle \mathbf w, \mathbf x_i \rangle\). The third takes the max over \(V\). The right side is a maximum of affine functions of \(\mathbf w\), so it is a convex surrogate, and SGD needs its maximizer at every step.

**Finding the maximizer.** Suppose the measure is a function of \(a, b, c, d\), and \(V\) is all vectors whose \(a\) and \(b\) satisfy some condition, like \(a + b \le k\). Group the candidates: for each pair \((a, b)\), let \(\bar Y_{a,b}\) be the vectors with exactly \(a\) of the important messages and \(b\) of the others set to \(+1\). Every \(\mathbf v\) lies in exactly one group. And because \(V\) is defined by a condition on \((a, b)\), a group is either entirely inside \(V\) or entirely outside. So we can search group by group.

Inside one group, \(c = P - a\) and \(d = N - b\) are fixed too (with \(P\) and \(N\) the numbers of important and unimportant messages), so \(\Delta\) is constant, and \(\sum_i y_i \mu_i\) never depends on \(\mathbf v\). Only \(\sum_i v_i \mu_i\) is left to maximize, with \(\mu_i = \langle \mathbf w, \mathbf x_i \rangle\). Now

\[ \sum_{i} v_i \mu_i = \sum_{i:\, v_i = 1} \mu_i - \sum_{i:\, v_i = -1} \mu_i = 2\sum_{i:\, v_i = 1} \mu_i - \sum_i \mu_i, \]

so we want the \(+1\) entries on the largest scores: the \(a\) important messages with the highest \(\mu\), and the \(b\) unimportant messages with the highest \(\mu\). The two choices do not interact, so each is a simple top-\(a\) and top-\(b\) pick from a sorted list.

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
      if some i and y ≠ y_i have <w(t), Ψ(x_i, y_i)> ≤ <w(t), Ψ(x_i, y)>:
        w(t+1) ← w(t) + Ψ(x_i, y_i) − Ψ(x_i, y)
      else:
        output w(t)
  ```

  Assume there is a \(\mathbf w^\star\) with \(\langle \mathbf w^\star, \Psi(\mathbf x_i, y_i) \rangle \ge \langle \mathbf w^\star, \Psi(\mathbf x_i, y) \rangle + 1\) for all \(i\) and all \(y \neq y_i\), and let \(R = \max_{i, y} \|\Psi(\mathbf x_i, y_i) - \Psi(\mathbf x_i, y)\|\). Prove that the algorithm stops after at most \((R\|\mathbf w^\star\|)^2\) updates, and that when it stops, \(y_i = \operatorname{argmax}_y \langle \mathbf w^{(t)}, \Psi(\mathbf x_i, y) \rangle\) for every \(i\).
- **Problem 3 (sorting maximizes the matching sum).** Let \(V\) be the set of permutations of \([r]\), written as vectors with distinct entries in \([r]\), and let \(\mathbf y' \in \mathbb R^r\) have distinct entries. Prove that \(\pi(\mathbf y') = \operatorname{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i y'_i\), and that this maximizer is unique. Then use the same idea to prove that \(G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\) for any nondecreasing discount \(D\) and nonnegative relevances \(\mathbf y\), which is what keeps the NDCG loss in \([0, 1]\).
