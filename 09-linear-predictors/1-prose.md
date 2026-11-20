# Linear Predictors

## The filter that listens to one word

Monday morning, and the complaints are polite but pointed. The facilities team sent out “Free lunch at the meeting, meeting room 4” and half the office never saw it: the filter put it in junk. Meanwhile a message that said “Claim your free prize” sailed straight through for some people and got caught for others, depending on how the spammer spelled things that week.

You open the filter and find the culprit quickly. Over the last few lessons the team has grown fond of rules that look at a single feature, because they are cheap to learn and easy to explain. The one in production says: junk exactly when the word “free” appears at least once. A rule like that, which looks at one feature and compares it with a threshold, is called a **single-feature rule**. It is fast, it is readable, and it is deaf to everything except one word.

The team has already done the first half of the fix. Instead of the yes/no features of lesson 1, every message is now turned into a vector of **word counts**: pick a vocabulary of \(d\) words, and let \(x_j\) be the number of times word \(j\) appears in the message. With a vocabulary of two words, “free” and “meeting”, the message “free trial, book a meeting” becomes the list \((2, 1)\): two of the first word, one of the second. A message becomes a point \(\mathbf x \in \mathbb R^d\). Bold letters like \(\mathbf x\) are whole lists of numbers, called vectors, and plain \(x_j\) is entry \(j\) of that list; \(\mathbb R\) is the set of real numbers, \(\mathbb R^d\) is the set of all lists of \(d\) real numbers, and \(\in\) reads “is in”. So \(\mathbf x \in \mathbb R^d\) just says **the message is a list of \(d\) numbers**, and \((2, 1) \in \mathbb R^2\). What the team wants now is a filter that *weighs the evidence*: a lot of “free” pushes towards junk, a lot of “meeting” pushes towards keep, and the verdict comes from the balance, not from whichever word happens to be loudest.

So you write down the obvious thing. Give every word a weight, add up the weighted counts, add a constant, and call the message junk when the total is positive. Then you ask the question this course has trained you to ask: how does the learner find the best weights? And here the brute-force instinct from lesson 2, “try every hypothesis in the class and keep the one with the smallest empirical risk”, runs into a wall. The weights are real numbers, so the class is infinite. Even if you settle for 64-bit floating-point weights, a vocabulary of \(d\) words plus the constant gives \(2^{64(d+1)}\) hypotheses. Read that from the inside out. There are \(d + 1\) numbers to choose, one weight per word and the constant; each is stored in 64 bits, so together they use \(64(d+1)\) bits; and each bit can be 0 or 1, so the number of different settings is 2 multiplied by itself once per bit, which is what the exponent in \(2^{64(d+1)}\) means. In one sentence: **the number of filters you could store is 2 raised to the total number of bits in the weights.** With thirty words that is \(64 \cdot 31 = 1984\) bits, so \(2^{1984}\) filters, a number with 598 digits:

```python
d = 30
bits = 64 * (d + 1)
count = 2 ** bits
print(bits, len(str(count)))
```

It prints `1984 598`. Lesson 8 taught you to count the runtime of a learner, not just its examples; this one would not finish before the sun burns out.

This lesson is about the class you just wrote down, the **halfspaces**, and about why learning it is not hopeless at all. In the right conditions, finding a consistent halfspace is a linear program (a standard optimization problem that off-the-shelf solvers handle, defined below), or a five-line loop with a guarantee on how long it runs. Then the team will ask for two more things, a junk *score* and a *probability* of junk, and the same linear idea will answer both.

## Messages become vectors

Let’s make it concrete with a vocabulary of just two words, so you can draw every message on paper: \(x_1\) counts “free” and \(x_2\) counts “meeting”. Here are eight messages your users labelled. For halfspaces it is convenient to write the labels as \(+1\) for junk and \(-1\) for keep, instead of 1 and 0; you will see why in a moment.

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

[[diagram:counts]]

Before reading on, try to find a single-feature rule that gets all eight right. You may pick either word, any threshold, and either direction (“junk above the threshold” or “junk below it”).

You can’t. The junk message D and the keep message E both contain “free” exactly once, so any rule that looks only at “free” gives them the same label, and one of them is wrong. Likewise the junk message A and the keep message H both contain “meeting” zero times, so no rule on “meeting” alone can tell them apart, whatever its threshold or direction. The best single-feature rule is the one in production, junk when “free” is at least 1, and it gets two of the eight wrong: E and G, the free lunch and the free coffee.

Now weigh the evidence. Take the score \(2x_1 - 2x_2 - 1\): each “free” adds 2, each “meeting” subtracts 2, and the constant \(-1\) means a message with no evidence at all is kept. Score A: \(6 - 0 - 1 = 5\), positive, junk. Score E: \(2 - 4 - 1 = -3\), negative, keep. Do the other six and every sign matches its label. Python agrees, and also checks every single-feature rule for you:

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
sign = lambda a: 1 if a > 0 else -1

def stump_mistakes(j, t, s):
    return sum(s * sign(x[j] - t) != y for x, y in S)

stumps = [(j, t + 0.5, s) for j in (0, 1)
          for t in range(-1, 5) for s in (1, -1)]
print(min(stump_mistakes(*st) for st in stumps))

w, b = (2, -2), -1
score = [w[0] * x[0] + w[1] * x[1] + b for x, _ in S]
print(score)
print(sum(sign(s) != y for s, (_, y) in zip(score, S)))
```

It prints `2`, then `[5, 1, 3, 1, -3, -3, -3, -1]`, then `0`. The best single-feature rule makes two mistakes; the weighted score makes none. Notice something else about those scores: every positive one is at least 1, and every negative one is at most \(-1\). Hold on to that; it is about to matter.

## Weighing the evidence

Let’s name what we just built. Scoring message E took three moves: multiply each count by its word’s weight (\(2 \cdot 1 = 2\) for “free”, \(-2 \cdot 2 = -4\) for “meeting”), add those products (\(2 - 4 = -2\)), then add the constant (\(-2 - 1 = -3\)). The first two moves, “multiply matching entries and add”, have a name: the **inner product** of the weights and the counts, written \(\langle \mathbf w, \mathbf x \rangle\). Here it is \(-2\). A **linear function** in the strict sense is \(\mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle\), where the arrow \(\mapsto\) reads “the message \(\mathbf x\) is sent to”, and \(\langle \mathbf w, \mathbf x \rangle = \sum_{i=1}^d w_i x_i\). Adding the constant gives an **affine function**:

\[ h_{\mathbf w, b}(\mathbf x) = \langle \mathbf w, \mathbf x \rangle + b = \left( \sum_{i=1}^d w_i x_i \right) + b. \]

Here \(\mathbf w \in \mathbb R^d\) is the **weight vector**, one weight per word, and \(b \in \mathbb R\) is the **bias term**. (The name is standard, and unfortunate: the bias term has nothing to do with the inductive bias of lesson 1. We will always say “bias term” for \(b\).) Read the formula from the inside out.

- \(w_i x_i\) is one product: the weight of word \(i\) times how many times word \(i\) appears.
- \(\sum_{i=1}^d\) is a sum: let \(i\) run from 1 up to \(d\), one word at a time, and add up what follows. So the bracket adds one product per word.
- The angle brackets \(\langle \mathbf w, \mathbf x \rangle\) are shorthand for exactly that bracket, the inner product.
- \(+\, b\) adds the bias term, the score of a message with no words at all.
- The subscripts in \(h_{\mathbf w, b}\) say which weights and which bias term this particular function uses; \(h_{\mathbf w, b}(\mathbf x)\) is its score on message \(\mathbf x\).

In plain words: **the score of a message is each word count times its weight, all added up, plus a constant.** Push message E through it with \(\mathbf w = (2, -2)\), \(b = -1\) and \(\mathbf x = (1, 2)\):

```python
w, b = (2, -2), -1
x = (1, 2)
terms = [w_i * x_i for w_i, x_i in zip(w, x)]
inner = sum(terms)
print(terms, inner, inner + b)
```

It prints `[2, -4] -2 -3`: the two products, their sum \(\langle \mathbf w, \mathbf x \rangle = -2\), and the score \(-3\), the same number as in the table check above.

Every choice of weights and bias term gives one such scoring function. The score we found uses \(((2, -2), -1)\); the production filter’s old instinct, “just count the frees”, is \(((1, 0), 0)\); a silly filter that ignores the message and always says 5 is \(((0, 0), 5)\). The collection of all of them is

\[ L_d = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle + b \; : \; \mathbf w \in \mathbb R^d, \, b \in \mathbb R \}. \]

Read it from the inside out. The part before the colon, \(\mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle + b\), describes one member: the function that sends a message to its score. The colon reads “such that”, or here “where”: the part after it says which choices are allowed, any list of \(d\) real weights and any real bias term. The braces \(\{ \cdots \}\) collect every member you get that way into one set, and \(L_d\) is the name of that set, with \(d\) recording how many words there are. In plain words: **\(L_d\) is the set of all scoring functions you get by choosing \(d\) weights and a bias term.** Here are the three members above, each applied to message B, \((2, 1)\):

```python
def h(w, b):
    return lambda x: sum(p * q for p, q in zip(w, x)) + b

members = [h((2, -2), -1), h((1, 0), 0), h((0, 0), 5)]
print([f((2, 1)) for f in members])
```

It prints `[1, 2, 5]`: one message, three members of \(L_2\), three scores.

There is a cheap trick that makes the bias term disappear, and we will use it constantly. Glue \(b\) onto the front of the weights and a constant 1 onto the front of every message: \(\mathbf w' = (b, w_1, \dots, w_d) \in \mathbb R^{d+1}\) and \(\mathbf x' = (1, x_1, \dots, x_d) \in \mathbb R^{d+1}\). Then

\[ \langle \mathbf w', \mathbf x' \rangle = b \cdot 1 + \sum_{i=1}^d w_i x_i = \langle \mathbf w, \mathbf x \rangle + b. \]

The prime in \(\mathbf w'\) and \(\mathbf x'\) only marks the longer, glued lists. Read left to right: the inner product of the glued lists multiplies matching entries, and the first entries are \(b\) and 1, so the first product is \(b \cdot 1\); the remaining entries are the old weights and counts, giving the old sum; and that total is the old score. In plain words: **gluing the bias term onto the weights and a 1 onto the message leaves every score unchanged.** For message B, \(\mathbf w' = (-1, 2, -2)\) and \(\mathbf x' = (1, 2, 1)\), so \(\langle \mathbf w', \mathbf x' \rangle = -1 + 4 - 2 = 1\), the same as \(2 \cdot 2 - 2 \cdot 1 - 1\), and the snippet below agrees.

So every affine function on \(\mathbb R^d\) is a **homogeneous** linear function, one with no bias term, on \(\mathbb R^{d+1}\), applied after appending the constant 1. Whenever it simplifies things, we drop \(b\) and work with \(h_{\mathbf w}(\mathbf x) = \langle \mathbf w, \mathbf x \rangle\), with the understanding that the 1 is already inside \(\mathbf x\). People say “linear” for both kinds.

```python
w, b, x = (2, -2), -1, (2, 1)
w_prime, x_prime = (b,) + w, (1,) + x
affine = sum(wi * xi for wi, xi in zip(w, x)) + b
linear = sum(wi * xi for wi, xi in zip(w_prime, x_prime))
print(affine, linear)
```

It prints `1 1`: message B scored once with a bias term and once with the 1 glued on.

A linear function returns a number, but the filter has to return a label. Remember from lesson 1 that \(\mathcal X\) is the **domain set**, every message that could arrive, and \(\mathcal Y\) is the **label set**, every answer the filter may give. To get a label, feed the score into a second function \(\varphi: \mathbb R \to \mathcal Y\); the notation \(\varphi: \mathbb R \to \mathcal Y\) says “\(\varphi\) takes a real number and returns a label”. For regression, where \(\mathcal Y = \mathbb R\), \(\varphi\) is the identity, which returns its input unchanged, and we will get there. For classification, \(\varphi\) is the **sign** function, \(\operatorname{sign}(a) = +1\) if \(a > 0\) and \(-1\) otherwise: \(\operatorname{sign}(5) = +1\), \(\operatorname{sign}(-3) = -1\), and \(\operatorname{sign}(0) = -1\). This is why the labels became \(\pm 1\): the sign of the score *is* the label.

**Definition (halfspaces).** Let \(\mathcal X = \mathbb R^d\) and \(\mathcal Y = \{-1, +1\}\). The class of **halfspaces** is

\[ HS_d = \operatorname{sign} \circ L_d = \{ \mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle + b) \; : \; \mathbf w \in \mathbb R^d, \, b \in \mathbb R \}. \]

The small circle \(\circ\) means **composition**, “do one function, then feed its answer to the next”: \(\operatorname{sign} \circ L_d\) means “take any member of \(L_d\), compute its score, then take the sign”. The set on the right spells that out with the same braces and “such that” colon as \(L_d\): one member for each choice of \(\mathbf w\) and \(b\), and the member sends \(\mathbf x\) to the sign of its score. The name \(HS_d\) is short for “halfspaces in \(d\) dimensions”. In plain words: **a halfspace scores the message with weights and a bias term, then answers junk if the score is positive and keep otherwise.** With our weights, A scores 5 and gets \(+1\), E scores \(-3\) and gets \(-1\), and the empty message H scores \(-1\) and gets \(-1\):

```python
sign = lambda a: 1 if a > 0 else -1
affine = lambda x: 2 * x[0] - 2 * x[1] - 1
halfspace = lambda x: sign(affine(x))
msgs = [(3, 0), (1, 2), (0, 0)]
print([affine(x) for x in msgs], [halfspace(x) for x in msgs])
```

It prints `[5, -3, -1] [1, -1, -1]`: first the scores, then the labels the composition gives them.

Why the name? Look at the picture in two dimensions. The points where the score is exactly zero, \(\{\mathbf x : \langle \mathbf w, \mathbf x \rangle + b = 0\}\), form a line (in \(d\) dimensions, a **hyperplane**), and that line is perpendicular to \(\mathbf w\). In the homogeneous case it passes through the origin, and the sign of \(\langle \mathbf w, \mathbf x \rangle\) tells you which side \(\mathbf x\) is on: positive when \(\mathbf x\) makes an acute angle with \(\mathbf w\), negative when the angle is obtuse. The hyperplane cuts the space into two halves, and the hypothesis labels one half junk and the other keep. For our score \(2x_1 - 2x_2 - 1\), the boundary is the line \(x_2 = x_1 - \tfrac12\), and the junk side is below it, where “free” outweighs “meeting”.

That set deserves a slow reading too. In \(\{\mathbf x : \langle \mathbf w, \mathbf x \rangle + b = 0\}\) the braces collect messages, and the colon reads “such that”: all messages \(\mathbf x\) such that the score is exactly zero. In plain words: **the boundary is every message the filter scores as perfectly balanced.** Half a “free” and no “meeting”, \((0.5, 0)\), scores \(1 - 0 - 1 = 0\), and so do \((1.5, 1)\) and \((3.5, 3)\). Walking along the line from one of them to the next moves by \((1, 1)\), and \(\langle \mathbf w, (1, 1) \rangle = 2 - 2 = 0\), which is what “the line is perpendicular to \(\mathbf w\)” means: a direction with zero inner product with \(\mathbf w\).

```python
w, b = (2, -2), -1
score = lambda x: w[0] * x[0] + w[1] * x[1] + b
on_line = [(0.5, 0), (1.5, 1), (3.5, 3)]
print([score(x) for x in on_line])
along = (1, 1)
print(w[0] * along[0] + w[1] * along[1])
```

It prints `[0.0, 0.0, 0.0]` and then `0`.

[[diagram:halfspace]]

One convention needs stating, because it bites in proofs: a message exactly on the boundary has score 0, and our sign function calls it keep. In training we will be stricter than that. Multiply the score by the label: junk message A has label \(+1\) and score 5, product 5; keep message E has label \(-1\) and score \(-3\), product 3. Both right, both positive. A junk message scored \(-1\) would give \(-1\), and a keep message scored 0 would give 0. So we ask for \(y \langle \mathbf w, \mathbf x \rangle > 0\) (with the 1 glued onto \(\mathbf x\), so the score already includes the bias term). Inside out: \(\langle \mathbf w, \mathbf x \rangle\) is the score, \(y\) in front multiplies it by the label, and \(> 0\) demands a strictly positive product. We call that product the **signed score**. In plain words: **the score has the label’s sign, and is not zero.** It is a compact way of writing “correct”, and it is the second reason for the \(\pm 1\) labels.

```python
for y, score in [(1, 5), (-1, -3), (-1, 0), (1, -1)]:
    print(y, score, y * score, y * score > 0)
```

It prints `1 5 5 True`, `-1 -3 3 True`, `-1 0 0 False` and `1 -1 -1 False`: the two correct messages pass, the boundary case and the mistake fail.

## Why brute force is hopeless, and what saves us

So the hypothesis class is \(HS_d\), and we would like to run ERM on it. As a reminder from lesson 2, **ERM**, empirical risk minimization, means: return a hypothesis with the smallest **empirical risk**, the fraction of labelled examples it gets wrong. Counting plain mistakes like that is the **0-1 loss**: a mistake costs 1, a correct answer costs 0. The production rule, wrong on E and G, has empirical risk \(2/8 = 0.25\); our weighted score has 0. Two separate worries stand in the way.

The first is statistical. \(HS_d\) is infinite, so lesson 2’s finite-class bound says nothing. But lesson 6 gave us the tool for infinite classes, the **VC dimension**, the size of the largest set of points a class can label in every possible way, and later in this lesson we will prove that the VC dimension of \(HS_d\) is \(d + 1\). Recall the two dials of PAC learning: \(\varepsilon\) is the accuracy you ask for (true risk at most \(\varepsilon\)), and \(\delta\) is the chance of failure you tolerate (the guarantee holds with probability at least \(1 - \delta\)). By the fundamental theorem of lesson 6, ERM over halfspaces is then a PAC learner, and the number of examples it needs grows with the number of words, not with the \(2^{1984}\) settings of the weights: for thirty words, \(\delta = 0.05\) and \(\varepsilon = 0.1\), it is a few hundred times a constant, as we will compute at the end of the VC section. So statistically, halfspaces are fine.

The second worry is computational, and it is the one from the opening: how do you *find* the empirical risk minimizer among infinitely many halfspaces? In general, when no halfspace gets every example right (the **agnostic** case of lesson 3), this is known to be computationally hard. The usual way around it is to minimize a different, friendlier loss instead of the 0-1 loss; logistic regression at the end of this lesson is one example, and lesson 12 studies the idea in depth.

But there is an important case where ERM is easy. Suppose some halfspace gets every training example right. That is the realizability assumption of lesson 2, and for halfspaces it has its own name: the training sequence is **separable**, because a hyperplane separates all the junk from all the keep. Our eight messages are separable; the score \(2x_1 - 2x_2 - 1\) proves it.

In the separable case the minimum empirical risk is 0, so an ERM hypothesis is exactly a halfspace with no mistakes on \(S\). In homogeneous form, we are looking for a vector \(\mathbf w \in \mathbb R^d\) with

\[ y_i \langle \mathbf w, \mathbf x_i \rangle > 0 \quad \text{for all } i = 1, \dots, m. \]

Here \(m\) is the number of labelled examples, eight for us, and the subscript \(i\) picks one of them: \(\mathbf x_i\) is message \(i\) (with the 1 glued on) and \(y_i\) its label. Each condition is the signed score test from above, and “for all \(i = 1, \dots, m\)” demands it for every message at once. In plain words: **find weights whose signed score is positive on every training message.** With \(\mathbf w = (-1, 2, -2)\), bias term first, the eight signed scores are all positive:

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
w = (-1, 2, -2)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
signed = [y * dot(w, (1,) + x) for x, y in S]
print(signed, all(s > 0 for s in signed))
```

It prints `[5, 1, 3, 1, 3, 3, 3, 1] True`. We will see two ways to find such weights when nobody hands them to you: a linear program, and the Perceptron.

## Separable mail is a linear program

Start with a small puzzle in the story. Forget the bias term for a moment and look for two weights \(w_1, w_2\) such that message A, \((3, 0)\), scores at least 1, and message E, \((1, 2)\), scores at most \(-1\). Written as “at least” conditions, that is \(3 w_1 + 0 w_2 \ge 1\) and \(-1 w_1 - 2 w_2 \ge 1\). Each condition is a weighted sum of the unknowns compared with a number, and so is anything we might want to make large, say \(w_1\) itself. A problem of exactly that shape is a **linear program** (LP): a linear objective, maximized under linear inequality constraints. In general,

\[ \max_{\mathbf w \in \mathbb R^d} \; \langle \mathbf u, \mathbf w \rangle \quad \text{subject to} \quad A \mathbf w \ge \mathbf v, \]

where \(\mathbf w\) is the vector of unknowns, \(A\) is an \(m \times d\) matrix, \(\mathbf v \in \mathbb R^m\) and \(\mathbf u \in \mathbb R^d\) are fixed vectors, and \(A \mathbf w \ge \mathbf v\) means every coordinate of \(A \mathbf w\) is at least the matching coordinate of \(\mathbf v\). Read the pieces one at a time.

- A **matrix** \(A\) is a table of numbers; \(m \times d\) means \(m\) rows and \(d\) columns. In the puzzle, the rows are \((3, 0)\) and \((-1, -2)\), one per condition.
- \(A \mathbf w\) is the list of inner products of each row with \(\mathbf w\), one number per row: the left-hand sides of the conditions.
- \(\ge\) between two lists compares them entry by entry, so \(A \mathbf w \ge \mathbf v\) is all the conditions at once, with \(\mathbf v = (1, 1)\) holding the right-hand sides.
- “subject to” means “searching only among \(\mathbf w\) that satisfy”; such \(\mathbf w\) are called **feasible**.
- \(\max_{\mathbf w \in \mathbb R^d}\) says: over all feasible weight vectors, make the value that follows as large as possible. The subscript names what you are allowed to vary.
- \(\langle \mathbf u, \mathbf w \rangle\) is the objective; \(\mathbf u = (1, 0)\) picks out \(w_1\).

In plain words: **among all weight vectors that pass every “at least” test, find one where a chosen weighted sum is largest.** The candidate \(\mathbf w = (1, -1)\) gives \(A \mathbf w = (3, 1)\), which is at least \((1, 1)\) entry by entry, so it is feasible, with objective value 1:

```python
A = [[3, 0], [-1, -2]]
v = [1, 1]
u = [1, 0]
w = [1, -1]
Aw = [sum(a * c for a, c in zip(row, w)) for row in A]
ok = all(p >= q for p, q in zip(Aw, v))
print(Aw, ok, sum(p * q for p, q in zip(u, w)))
```

It prints `[3, 1] True 1`. Linear programs can be solved efficiently, in time polynomial in \(m\), \(d\) and the number of bits used to write the numbers, and good solvers are freely available.

Our constraints \(y_i \langle \mathbf w, \mathbf x_i \rangle > 0\) are almost of that form, except that they are strict, and an LP wants \(\ge\). Here is the step that fixes it, and it is the most useful idea in this section: **a separator can always be rescaled until every score is at least 1.**

**Claim.** If \(S\) is separable, there is a vector \(\bar{\mathbf w}\) with \(y_i \langle \bar{\mathbf w}, \mathbf x_i \rangle \ge 1\) for all \(i\). Conversely, any such vector is an ERM hypothesis.

*Proof.* The star in \(\mathbf w^\star\) marks the separator we start from, and the bar in \(\bar{\mathbf w}\) the rescaled one we build. Let \(\mathbf w^\star\) satisfy \(y_i \langle \mathbf w^\star, \mathbf x_i \rangle > 0\) for all \(i\); it exists because \(S\) is separable. Let \(\gamma\) (gamma) be its smallest signed score, \(\gamma = \min_i y_i \langle \mathbf w^\star, \mathbf x_i \rangle\), where \(\min_i\) means “the smallest value over all \(i\)”. This is a minimum of finitely many positive numbers, so \(\gamma > 0\). Set \(\bar{\mathbf w} = \mathbf w^\star / \gamma\). The inner product is linear in its first argument, so for every \(i\),

\[ y_i \langle \bar{\mathbf w}, \mathbf x_i \rangle = \frac{1}{\gamma} \, y_i \langle \mathbf w^\star, \mathbf x_i \rangle \ge \frac{\gamma}{\gamma} = 1, \]

where the inequality is just \(y_i \langle \mathbf w^\star, \mathbf x_i \rangle \ge \gamma\), the definition of \(\gamma\) as the minimum. Read the display inside out: dividing the weights by \(\gamma\) divides every score by \(\gamma\); each original signed score was at least \(\gamma\); so each new one is at least \(\gamma / \gamma = 1\). In plain words: **divide the separator by its smallest signed score, and the smallest becomes exactly 1.** If the closest call scored 0.5, dividing by 0.5 doubles everything and the closest call now scores 1; the snippet at the end of this section does exactly that on our messages. For the converse, \(y_i \langle \bar{\mathbf w}, \mathbf x_i \rangle \ge 1 > 0\) says every example is labelled correctly, so the empirical risk is 0, the smallest possible. \(\blacksquare\)

Multiplying \(\mathbf w\) by a positive number never changes the sign of any score, so it never changes the hypothesis. That freedom is what we spent.

Now write it as an LP. Let \(A\) be the \(m \times d\) matrix whose rows are the messages multiplied by their labels, \(A_{i,j} = y_i x_{i,j}\), where \(x_{i,j}\) is coordinate \(j\) of \(\mathbf x_i\): the first subscript picks the message, the second the entry, so \(A_{i,j}\) is the entry in row \(i\), column \(j\). In plain words: **each row is a message with the 1 glued on, flipped in sign when the message is keep.** Message A, \((1, 3, 0)\) with label \(+1\), gives the row \((1, 3, 0)\); message E, \((1, 1, 2)\) with label \(-1\), gives \((-1, -1, -2)\). Then row \(i\) of \(A \mathbf w\) is \(y_i \langle \mathbf w, \mathbf x_i \rangle\), and the constraints become \(A \mathbf w \ge \mathbf v\) with \(\mathbf v = (1, \dots, 1) \in \mathbb R^m\). What do we maximize? Nothing in particular: every feasible \(\mathbf w\) is an equally good ERM hypothesis. So we set a dummy objective, \(\mathbf u = (0, \dots, 0)\), and the solver just has to find a feasible point.

Here is the rescaling on our eight messages, after appending the constant 1, starting from a separator whose smallest score is only 0.5. The code mirrors the claim term by term: `A` holds the rows \(y_i \mathbf x_i\), `Aw` computes \(A \mathbf w\), `gamma` is the minimum, and `w_bar` is \(\mathbf w^\star / \gamma\):

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
A = [[y * c for c in (1,) + x] for x, y in S]

def Aw(w):
    return [sum(a * wj for a, wj in zip(row, w)) for row in A]

w_star = (-0.5, 1.5, -1)
gamma = min(Aw(w_star))
w_bar = [wj / gamma for wj in w_star]
print(Aw(w_star))
print(gamma, w_bar)
print(Aw(w_bar), all(v >= 1 for v in Aw(w_bar)))
```

It prints `[4.0, 1.5, 3.5, 1.0, 1.0, 1.5, 0.5, 0.5]`, then `0.5 [-1.0, 3.0, -2.0]`, then `[8.0, 3.0, 7.0, 2.0, 2.0, 3.0, 1.0, 1.0] True`. Messages G and H were the closest calls; after dividing by 0.5, even they clear 1.

## The Perceptron

The LP is a perfectly good answer, but it hands the work to a solver. There is a much older algorithm, Rosenblatt’s **Perceptron** from 1958, that you can write in five lines, and that comes with a beautiful guarantee.

The idea, in words first: start with all weights at zero; find a message the current weights get wrong; if it is junk, add its vector to the weights, and if it is keep, subtract it; repeat until nothing is wrong. Starting from zero weights, every score is 0, so junk message A is wrong, and adding its vector \((1, 3, 0)\) gives the weights \((1, 3, 0)\).

Now the notation. The weights change from step to step, so we label them by step: \(\mathbf w^{(t)}\) means the value of \(\mathbf w\) at step \(t\), with the \(t\) in brackets so it is not mistaken for a power. Start with \(\mathbf w^{(1)} = \mathbf 0\), the all-zeros vector, which labels nothing correctly (every score is 0). At each step \(t\), look for an example that \(\mathbf w^{(t)}\) gets wrong, that is, with \(y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle \le 0\), a signed score that is not positive. If there is one, **update** by adding the message, scaled by its label:

\[ \mathbf w^{(t+1)} = \mathbf w^{(t)} + y_i \mathbf x_i. \]

If there is none, stop and output \(\mathbf w^{(t)}\). Read the update inside out: \(\mathbf x_i\) is the message’s vector, \(y_i \mathbf x_i\) multiplies every entry by the label (so it is the vector itself for junk, its negative for keep), and adding that to \(\mathbf w^{(t)}\) entry by entry gives the next weights \(\mathbf w^{(t+1)}\). In plain words: **on a mistake, add the message to the weights if it is junk and subtract it if it is keep.** Here is that first step:

```python
w = [0, 0, 0]
x, y = (1, 3, 0), 1
print(y * sum(a * b for a, b in zip(w, x)))
w = [wj + y * xj for wj, xj in zip(w, x)]
print(w)
```

It prints `0`, the signed score that makes A a mistake, then `[1, 3, 0]`.

Why is that a sensible thing to do? Ask what the update does to the example it was made for. One more piece of notation first. The **length** (or **norm**) of a vector is \(\|\mathbf x\| = \sqrt{x_1^2 + \dots + x_d^2}\), the ordinary straight-line distance from the origin; the double bars mean “length of”. Its square is the inner product of the vector with itself, \(\|\mathbf x\|^2 = \langle \mathbf x, \mathbf x \rangle\). Message E with the 1 glued on, \((1, 1, 2)\), has \(\|\mathbf x\|^2 = 1 + 1 + 4 = 6\) and length \(\sqrt 6 \approx 2.45\). Now expand the inner product:

\[ y_i \langle \mathbf w^{(t+1)}, \mathbf x_i \rangle = y_i \langle \mathbf w^{(t)} + y_i \mathbf x_i, \mathbf x_i \rangle = y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle + y_i^2 \|\mathbf x_i\|^2 = y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle + \|\mathbf x_i\|^2. \]

The first equality is the update, the second splits the inner product (it is linear in each argument), and the third uses \(y_i^2 = 1\), which is the third reason for \(\pm 1\) labels. The update may make things worse on other examples, which is why we need a proof that the process ends at all. In plain words: **an update raises the signed score of the message it was made for by that message’s squared length.** Take the second step of our run: the weights are \((1, 3, 0)\), and keep message E, \((1, 1, 2)\), scores 4, so its signed score is \(-4\), a mistake. Subtracting E gives \((0, 2, -2)\), which scores E at \(-2\), signed score 2. The gain is \(2 - (-4) = 6\), exactly \(\|\mathbf x\|^2\):

```python
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w, x, y = [1, 3, 0], (1, 1, 2), -1
before = y * dot(w, x)
w_next = [wj + y * xj for wj, xj in zip(w, x)]
after = y * dot(w_next, x)
print(before, after, after - before, dot(x, x))
```

It prints `-4 2 6 6`. The next snippet repeats the check with random weights, messages and labels, to show it is not a coincidence of our numbers:

```python
import random

rng = random.Random(0)
for _ in range(5):
    w = [rng.uniform(-3, 3) for _ in range(3)]
    x = [rng.uniform(-3, 3) for _ in range(3)]
    y = rng.choice((1, -1))
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    w_next = [wj + y * xj for wj, xj in zip(w, x)]
    gain = y * dot(w_next, x) - y * dot(w, x)
    print(round(gain, 6), round(dot(x, x), 6))
```

Each line prints the same number twice (the first is `2.422288 2.422288`): the gain in signed score equals \(\|\mathbf x\|^2\), for random weights, random messages and random labels.

Now run the whole thing on our eight messages, with the constant 1 appended, always picking the first mistake in the table’s order:

```python
S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
X = [(1,) + x for x, _ in S]
Y = [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
mistakes = lambda w: [i for i in range(len(X))
                      if Y[i] * dot(w, X[i]) <= 0]
w, T, updated = [0, 0, 0], 0, []
while mistakes(w):
    i = mistakes(w)[0]
    w = [wj + Y[i] * xj for wj, xj in zip(w, X[i])]
    T, updated = T + 1, updated + [i + 1]
print(T, updated, w)
```

It prints `3 [1, 5, 8] [-1, 2, -2]`. Three updates, on messages A, E and H, and the result is \(\mathbf w = (-1, 2, -2)\): bias term \(-1\), weight 2 on “free”, weight \(-2\) on “meeting”. That is exactly the score \(2x_1 - 2x_2 - 1\) we found by hand. The first two updates are the ones traced above; after them, H, the empty message \((1, 0, 0)\), scores 0, which counts as a mistake, and subtracting it gives \((-1, 2, -2)\).

[[diagram:perceptron]]

Before we prove anything, play with it. In the lab below you can drag a line through the eight messages and watch its mistakes, then step the Perceptron one update at a time and watch its weight vector turn.

[[lab]]

## How many updates? The mistake bound

Three updates on eight messages is quick. Is it always quick? Could the Perceptron chase its tail forever, fixing one example and breaking another? On separable data, no, and the proof is one of the loveliest in the subject. It needs two quantities.

The first is how far the messages reach. Among our eight, with the 1 glued on, the longest is C, \((1, 4, 2)\), with squared length \(1 + 16 + 4 = 21\) and length \(\sqrt{21} \approx 4.58\). That is the **radius** \(R = \max_i \|\mathbf x_i\|\): \(\|\mathbf x_i\|\) is the length of message \(i\), and \(\max_i\) takes the largest over all \(i\). In plain words: **\(R\) is the length of the longest message vector.**

The second is how short a separator can be while still scoring every example at least 1. Our \((-1, 2, -2)\) does that (its smallest signed score is exactly 1), and its length is \(\sqrt{1 + 4 + 4} = 3\). Could a shorter vector pass the same test? The best such length is

\[ B = \min \{ \|\mathbf w\| \; : \; y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1 \text{ for all } i \in [m] \}. \]

Inside out: \([m]\) is shorthand, from lesson 1, for the positions \(\{1, 2, \dots, m\}\); the condition after the colon says \(\mathbf w\) gives every example a signed score of at least 1, the rescaled kind of separator from the LP section; the braces collect the lengths \(\|\mathbf w\|\) of all such vectors; and \(\min\) takes the smallest of those lengths. In plain words: **\(B\) is the length of the shortest vector that scores every example at least 1.** Since \((-1, 2, -2)\) qualifies, \(B \le 3\) for our messages; we will find the exact value later. A small \(B\) means a short vector already clears every example with room to spare.

```python
X = [(1, 3, 0), (1, 2, 1), (1, 4, 2), (1, 1, 0),
     (1, 1, 2), (1, 0, 1), (1, 2, 3), (1, 0, 0)]
Y = [1, 1, 1, 1, -1, -1, -1, -1]
norm = lambda v: sum(c * c for c in v) ** 0.5
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
R = max(norm(x) for x in X)
w = (-1, 2, -2)
print(round(R, 3), round(R ** 2, 3))
print(min(y * dot(w, x) for x, y in zip(X, Y)), norm(w))
```

It prints `4.583 21.0`, the radius and its square, then `1 3.0`: the candidate’s smallest signed score is 1 and its length is 3.

We also need one inequality. Take the candidate \(\mathbf u = (-1, 2, -2)\) and message C, \(\mathbf v = (1, 4, 2)\). Their inner product is \(-1 + 8 - 4 = 3\), while their lengths multiply to \(3 \cdot 4.58 \approx 13.75\). The inner product came out smaller, and it always does: the **Cauchy–Schwarz inequality** says \(\langle \mathbf u, \mathbf v \rangle \le \|\mathbf u\| \|\mathbf v\|\) for all vectors. In plain words: **multiplying matching entries and adding never beats multiplying the two lengths.** Equality needs the vectors to point the same way.

```python
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
norm = lambda v: dot(v, v) ** 0.5
u, v = (-1, 2, -2), (1, 4, 2)
lhs, rhs = dot(u, v), norm(u) * norm(v)
print(lhs, round(rhs, 3), round(lhs / rhs, 3))
```

It prints `3 13.748 0.218`; the last number is the ratio, which the proof below shows is at most 1. Proof: if \(\mathbf v = \mathbf 0\) both sides are 0. Otherwise, for every real \(t\), \(0 \le \|\mathbf u - t \mathbf v\|^2 = \|\mathbf u\|^2 - 2t \langle \mathbf u, \mathbf v \rangle + t^2 \|\mathbf v\|^2\); choosing \(t = \langle \mathbf u, \mathbf v \rangle / \|\mathbf v\|^2\) gives \(0 \le \|\mathbf u\|^2 - \langle \mathbf u, \mathbf v \rangle^2 / \|\mathbf v\|^2\), so \(\langle \mathbf u, \mathbf v \rangle^2 \le \|\mathbf u\|^2 \|\mathbf v\|^2\), and taking square roots finishes it. Dividing through, \(\langle \mathbf u, \mathbf v \rangle / (\|\mathbf u\| \|\mathbf v\|)\) is at most 1: it is the cosine of the angle between the two vectors.

**Theorem (the Perceptron’s mistake bound).** Assume \((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m)\) is separable, and let \(B\) and \(R\) be as above. Then the Perceptron stops after at most \((RB)^2\) updates, and when it stops, \(y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle > 0\) for every \(i \in [m]\).

*Proof.* The second claim is free: the Perceptron only stops when no example has \(y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle \le 0\). The work is in showing that it does stop, and we do that by showing that it cannot make more than \((RB)^2\) updates.

Fix any vector \(\mathbf w^\star\) with \(y_i \langle \mathbf w^\star, \mathbf x_i \rangle \ge 1\) for all \(i\). The LP section showed one exists. Suppose the Perceptron has made \(T \ge 1\) updates, producing \(\mathbf w^{(T+1)}\). The plan is to watch the angle between \(\mathbf w^\star\) and \(\mathbf w^{(T+1)}\), and show its cosine is at least \(\sqrt T / (R \|\mathbf w^\star\|)\). Since a cosine is at most 1, that caps \(T\).

*Step 1: the inner product with \(\mathbf w^\star\) grows by at least 1 per update.* At the start, \(\langle \mathbf w^\star, \mathbf w^{(1)} \rangle = \langle \mathbf w^\star, \mathbf 0 \rangle = 0\). If update \(t\) uses example \(i\), then

\[ \langle \mathbf w^\star, \mathbf w^{(t+1)} \rangle - \langle \mathbf w^\star, \mathbf w^{(t)} \rangle = \langle \mathbf w^\star, \mathbf w^{(t+1)} - \mathbf w^{(t)} \rangle = \langle \mathbf w^\star, y_i \mathbf x_i \rangle = y_i \langle \mathbf w^\star, \mathbf x_i \rangle \ge 1. \]

The first equality is linearity, the second is the update rule, and the inequality is the defining property of \(\mathbf w^\star\). In plain words: **each update adds the signed score of the corrected message under \(\mathbf w^\star\), which is at least 1, to the inner product with \(\mathbf w^\star\).** In our run with \(\mathbf w^\star = (-1, 2, -2)\), the first update adds message A, whose signed score under \(\mathbf w^\star\) is 5, so the inner product jumps from 0 to 5.

Summing these differences over \(t = 1, \dots, T\), the sum telescopes:

\[ \langle \mathbf w^\star, \mathbf w^{(T+1)} \rangle = \sum_{t=1}^{T} \left( \langle \mathbf w^\star, \mathbf w^{(t+1)} \rangle - \langle \mathbf w^\star, \mathbf w^{(t)} \rangle \right) \ge T. \]

“Telescopes” means the sum collapses like a folding telescope: each difference starts where the previous one ended, so writing them out, every middle value appears once with a plus and once with a minus and cancels, leaving the last value minus the first, and the first is 0. Each of the \(T\) differences is at least 1, so the total is at least \(T\). In plain words: **after \(T\) updates the inner product with \(\mathbf w^\star\) is at least \(T\).** Our run’s inner products are 0, 5, 8, 9, with differences 5, 3, 1; they add to 9, the final value, which is at least \(T = 3\). The replay snippet after the proof prints those values in its second column.

*Step 2: the squared length grows by at most \(R^2\) per update.* If update \(t\) uses example \(i\),

\[ \|\mathbf w^{(t+1)}\|^2 = \|\mathbf w^{(t)} + y_i \mathbf x_i\|^2 = \|\mathbf w^{(t)}\|^2 + 2 y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle + y_i^2 \|\mathbf x_i\|^2 \le \|\mathbf w^{(t)}\|^2 + R^2. \]

The middle step expands the square. The inequality uses two facts: example \(i\) was chosen *because* \(y_i \langle \mathbf w^{(t)}, \mathbf x_i \rangle \le 0\), so the cross term is at most 0; and \(y_i^2 \|\mathbf x_i\|^2 = \|\mathbf x_i\|^2 \le R^2\) by the definition of \(R\). This is the one place the proof uses the fact that the Perceptron only updates on mistakes. Starting from \(\|\mathbf w^{(1)}\|^2 = 0\) and applying the inequality \(T\) times,

\[ \|\mathbf w^{(T+1)}\|^2 \le T R^2, \quad \text{so} \quad \|\mathbf w^{(T+1)}\| \le \sqrt T \, R. \]

In plain words: **each update adds at most \(R^2\) to the squared length, so after \(T\) updates the length is at most \(\sqrt T\) times \(R\).** In our run the squared lengths are 0, 10, 8, 9: the steps are \(+10\), \(-2\), \(+1\), each at most \(R^2 = 21\), and the final 9 is far below \(3 \cdot 21 = 63\). The replay snippet’s third and fourth columns show the same numbers.

*Step 3: combine.* The inner product grows linearly in \(T\), the length only like \(\sqrt T\), so the angle has to close. Precisely, by Step 1, Step 2 and Cauchy–Schwarz,

\[ 1 \ge \frac{\langle \mathbf w^\star, \mathbf w^{(T+1)} \rangle}{\|\mathbf w^\star\| \, \|\mathbf w^{(T+1)}\|} \ge \frac{T}{\|\mathbf w^\star\| \sqrt T R} = \frac{\sqrt T}{R \|\mathbf w^\star\|}. \]

Read it left to right. The fraction in the middle is the cosine of the angle between \(\mathbf w^\star\) and the current weights, at most 1 by Cauchy–Schwarz. Its top is at least \(T\) (Step 1) and its bottom is at most \(\|\mathbf w^\star\| \sqrt T R\) (Step 2), and a bigger top over a smaller bottom can only make the fraction smaller, giving the second inequality. Cancelling \(\sqrt T\) gives the last expression. In plain words: **the cosine grows at least like \(\sqrt T\), but can never pass 1.** After our three updates the cosine is \(9 / (3 \cdot 3) = 1\), and the guaranteed lower value is \(\sqrt 3 / (\sqrt{21} \cdot 3) \approx 0.126\); the replay snippet’s last column shows the cosine climbing 0.527, 0.943, 1.0.

So \(\sqrt T \le R \|\mathbf w^\star\|\), that is, \(T \le (R \|\mathbf w^\star\|)^2\). (If \(T = 0\) there is nothing to prove.)

*Step 4: the best \(\mathbf w^\star\).* The bound holds for every \(\mathbf w^\star\) that scores all examples at least 1, so it holds for the shortest one: \(T \le (RB)^2\). In other words, the Perceptron cannot make more than \((RB)^2\) updates. Every pass through its loop either updates or stops, so it stops after at most \((RB)^2\) updates. \(\blacksquare\)

Watch the two invariants on our run, with \(\mathbf w^\star = (-1, 2, -2)\) (smallest signed score 1) and \(R^2 = 21\). Replay the three updates:

```python
X = [(1, 3, 0), (1, 1, 2), (1, 0, 0)]
Y = [1, -1, -1]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
w_bar, R2 = (-1, 2, -2), 21
w = [0, 0, 0]
for t, (x, y) in enumerate(zip(X, Y), start=1):
    w = [wj + y * xj for wj, xj in zip(w, x)]
    cos = dot(w_bar, w) / (dot(w_bar, w_bar) * dot(w, w)) ** 0.5
    print(t, dot(w_bar, w), dot(w, w), t * R2, round(cos, 3))
print(21 * dot(w_bar, w_bar))
```

It prints `1 5 10 21 0.527`, `2 8 8 42 0.943`, `3 9 9 63 1.0`, then `189`. Column two never drops below \(t\) (Step 1), column three never exceeds \(t R^2\) in column four (Step 2), and the cosine climbs to 1. With this \(\mathbf w^\star\) the theorem promises at most \(21 \cdot 9 = 189\) updates. The Perceptron used 3.

### Radius and margin

That gap between 189 and 3 is worth understanding, and the cleanest way is to rewrite \(B\) as a picture.

The idea is room: how far the closest message sits from the separating hyperplane. Scores measure that distance once the weights have length 1. For a **unit vector** \(\mathbf u\), one with \(\|\mathbf u\| = 1\), the number \(|\langle \mathbf u, \mathbf x \rangle|\) (the bars mean size without the sign) is the distance from \(\mathbf x\) to the hyperplane \(\{\mathbf v : \langle \mathbf u, \mathbf v \rangle = 0\}\). The point \(\mathbf x - \langle \mathbf u, \mathbf x \rangle \mathbf u\) lies on the hyperplane, because its score is \(\langle \mathbf u, \mathbf x \rangle - \langle \mathbf u, \mathbf x \rangle \|\mathbf u\|^2 = 0\); the gap from \(\mathbf x\) to it is \(\langle \mathbf u, \mathbf x \rangle \mathbf u\), of length \(|\langle \mathbf u, \mathbf x \rangle|\), and it points along \(\mathbf u\), perpendicular to the hyperplane, so by Pythagoras every other point of the hyperplane is farther. So define the **margin** of a unit vector \(\mathbf u\) on \(S\) as

\[ \gamma_{\mathbf u} = \min_i \, y_i \langle \mathbf u, \mathbf x_i \rangle. \]

Inside out: \(\langle \mathbf u, \mathbf x_i \rangle\) is the distance of message \(i\) to the boundary, with a sign; multiplying by \(y_i\) makes it positive when the message is on its correct side; and \(\min_i\) keeps the smallest. The subscript in \(\gamma_{\mathbf u}\) records which separator is being measured. In plain words: **the margin is how far the closest message sits from the boundary, counted negative if it is on the wrong side.** Shrink our \((-1, 2, -2)\), of length 3, to the unit vector \(\mathbf u = (-1, 2, -2)/3\): every signed score is divided by 3, and the smallest, 1, becomes \(1/3\), so \(\gamma_{\mathbf u} \approx 0.333\), shared by B, D and H.

```python
X = [(1, 3, 0), (1, 2, 1), (1, 4, 2), (1, 1, 0),
     (1, 1, 2), (1, 0, 1), (1, 2, 3), (1, 0, 0)]
Y = [1, 1, 1, 1, -1, -1, -1, -1]
u = [c / 3 for c in (-1, 2, -2)]
dot = lambda p, q: sum(a * b for a, b in zip(p, q))
gaps = [y * dot(u, x) for x, y in zip(X, Y)]
print(round(dot(u, u), 3), [round(g, 3) for g in gaps])
print(round(min(gaps), 3))
```

It prints `1.0`, confirming \(\|\mathbf u\|^2 = 1\), then the eight distances, then their minimum `0.333`.

Let \(\gamma^\star = \sup_{\|\mathbf u\| = 1} \gamma_{\mathbf u}\) be the largest margin any separator achieves. \(\sup\), the supremum, reads “the largest value, or the value approached from below if no single \(\mathbf u\) reaches it”, and the subscript \(\|\mathbf u\| = 1\) says we range over all unit vectors. In plain words: **\(\gamma^\star\) is the widest room any separator can leave.** Our unit vector gives 0.333, so \(\gamma^\star \ge 0.333\); the grid search below finds about 0.384.

**Claim.** \(B = 1/\gamma^\star\), so the mistake bound reads \(T \le (R / \gamma^\star)^2\). Moreover, for *any* unit \(\mathbf u\) with \(\gamma_{\mathbf u} > 0\), \(T \le (R / \gamma_{\mathbf u})^2\).

*Proof.* Take a unit \(\mathbf u\) with \(\gamma_{\mathbf u} > 0\). Then \(\mathbf u / \gamma_{\mathbf u}\) scores every example at least \(\gamma_{\mathbf u} / \gamma_{\mathbf u} = 1\), and its length is \(1 / \gamma_{\mathbf u}\). So \(B \le 1/\gamma_{\mathbf u}\), which already gives the last sentence, through the theorem. Taking the best \(\mathbf u\), \(B \le 1/\gamma^\star\). Conversely, take any \(\mathbf w\) that scores every example at least 1 and let \(\mathbf u = \mathbf w / \|\mathbf w\|\). Then \(\gamma_{\mathbf u} \ge 1 / \|\mathbf w\|\), so \(\gamma^\star \ge 1 / \|\mathbf w\|\), so \(\|\mathbf w\| \ge 1/\gamma^\star\); taking the shortest such \(\mathbf w\), \(B \ge 1/\gamma^\star\). \(\blacksquare\)

In plain words, the claim says: **the shortest vector that scores everything at least 1 is one over the widest margin.** With our unit vector, \(\mathbf u / \gamma_{\mathbf u} = (-1, 2, -2)/3 \div \tfrac13 = (-1, 2, -2)\), length 3, which is \(1 / 0.333\). The grid search below does the same for the widest separator.

[[diagram:margin]]

So the bound is the ratio of two lengths, squared: how far the messages reach from the origin, over how much room the best separator leaves. Data that is spread out but cleanly separated is easy; data where junk and keep nearly touch is hard. And notice what does *not* appear: neither the number of examples \(m\) nor the dimension \(d\). (All of this is in the space with the constant 1 appended, so \(R\) and the margin are measured there.)

What is the best margin on our eight messages? Only the direction of a separator matters, and any separator must give the empty message H a negative score, so its bias term is negative; we may scale it to \(-1\) and search the other two weights on a fine grid:

```python
import math

S = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
     ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
X = [(1,) + x for x, _ in S]
def margin(u):
    n = math.sqrt(sum(c * c for c in u))
    return min(y * sum(a * b for a, b in zip(u, x)) / n
               for x, (_, y) in zip(X, S))
grid = [(-1, a / 30, b / 30) for a in range(-180, 181)
        for b in range(-180, 181)]
best = max(grid, key=margin)
gamma = margin(best)
print([round(c, 3) for c in best], round(gamma, 3))
print(round(1 / gamma, 3), round(21 / gamma ** 2, 1))
```

It prints `[-1, 2.0, -1.333] 0.384`, then `2.603 142.3`. The widest separator is \(( -1, 2, -\tfrac43)\), with margin about 0.384, so \(B \approx 2.603\) and the theorem’s promise is \((RB)^2 \approx 142.3\) updates. (An exact method, solving the small quadratic problem behind \(B\), confirms the same vector.) Our certificate \(\bar{\mathbf w}\) gave 189, a looser but still valid promise. The Perceptron used 3.

So the bound is a worst case over every order of examples and every data set with that radius and margin, and on friendly data the Perceptron does far better. The reverse can happen too: \(B\) can be exponentially large in \(d\) for some data sets, and then the Perceptron may be very slow. In that case solving the linear program is the better route. For many natural data sets, though, \(B\) is moderate and the Perceptron converges quickly.

## How many examples? The VC dimension of halfspaces

Both algorithms find a consistent halfspace. Lesson 6 tells us how many examples make a consistent hypothesis trustworthy, as long as we know the VC dimension. Recall that a class **shatters** a finite set of points \(C\) when it can realize every one of the \(2^{|C|}\) possible labelings of that set (the bars \(|C|\) count how many points \(C\) has, and each point gets one of two labels, so three points have \(2^3 = 8\) labelings), and the **VC dimension** is the size of the largest set it shatters. We start with homogeneous halfspaces, \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\).

**Theorem (VC dimension of homogeneous halfspaces).** The VC dimension of the class of homogeneous halfspaces in \(\mathbb R^d\) is \(d\).

*Proof.* We need two things: some set of \(d\) points is shattered, and no set of \(d + 1\) points is.

*A shattered set of size \(d\).* Let \(\mathbf e_1, \dots, \mathbf e_d\) be the standard basis: \(\mathbf e_i\) is all zeros except a 1 in coordinate \(i\). Given any labeling \(y_1, \dots, y_d \in \{\pm 1\}\), set \(\mathbf w = (y_1, \dots, y_d)\). Then \(\langle \mathbf w, \mathbf e_i \rangle = y_i\), whose sign is \(y_i\). Every labeling is realized.

*No shattered set of size \(d + 1\).* Let \(\mathbf x_1, \dots, \mathbf x_{d+1}\) be any \(d + 1\) vectors in \(\mathbb R^d\). A basic fact of linear algebra is that more than \(d\) vectors in \(\mathbb R^d\) are linearly dependent: there are real numbers \(a_1, \dots, a_{d+1}\), not all zero, with \(\sum_i a_i \mathbf x_i = \mathbf 0\): some weighted mix of the points cancels out to the zero vector. For four points in \(\mathbb R^3\), for instance, \(\mathbf e_1 + \mathbf e_2 + \mathbf e_3 - (1, 1, 1) = \mathbf 0\), with weights \(a = (1, 1, 1, -1)\). Replacing every \(a_i\) by \(-a_i\) if necessary, we may assume some \(a_i\) is positive. Let \(I = \{i : a_i > 0\}\), which is nonempty, and \(J = \{j : a_j < 0\}\), which may be empty. Moving the negative terms across,

\[ \sum_{i \in I} a_i \mathbf x_i = \sum_{j \in J} |a_j| \mathbf x_j. \]

Read the pieces: \(\sum_{i \in I}\) adds over the positions in \(I\) only, \(|a_j|\) is the size of \(a_j\) without its sign, and each side is a weighted mix of points with nonnegative weights. In plain words: **some points with positive weights add up to the same vector as the remaining points with positive weights.** In \(\mathbb R^3\), take the four points \(\mathbf e_1, \mathbf e_2, \mathbf e_3\) and \((1, 1, 1)\), with \(a = (1, 1, 1, -1)\): then \(I = \{1, 2, 3\}\), \(J = \{4\}\), and the display says \(\mathbf e_1 + \mathbf e_2 + \mathbf e_3 = 1 \cdot (1, 1, 1)\), which is true. The snippet after the proof checks this example.

Now consider the labeling that says \(+1\) on \(I\) and \(-1\) everywhere else, and suppose some \(\mathbf w\) realizes it. Then \(\langle \mathbf w, \mathbf x_i \rangle > 0\) for every \(i \in I\) (a sign of \(+1\) needs a positive score) and \(\langle \mathbf w, \mathbf x_j \rangle \le 0\) for every \(j \in J\). Take the inner product of both sides of the last display with \(\mathbf w\):

\[ 0 < \sum_{i \in I} a_i \langle \mathbf x_i, \mathbf w \rangle = \left\langle \sum_{i \in I} a_i \mathbf x_i, \mathbf w \right\rangle = \left\langle \sum_{j \in J} |a_j| \mathbf x_j, \mathbf w \right\rangle = \sum_{j \in J} |a_j| \langle \mathbf x_j, \mathbf w \rangle \le 0. \]

The left inequality holds because \(I\) is nonempty and each term is a positive number times a positive score. The right one holds because each term is a nonnegative number times a score that is at most 0 (and an empty sum is 0). That says \(0 < 0\), a contradiction. In plain words: **the weighted scores of the \(I\) side must be positive, the weighted scores of the \(J\) side cannot be, yet the two sides are the same vector.** With \(\mathbf w = (1, 1, 1)\) the four scores are 1, 1, 1 and 3: the left side is \(1 + 1 + 1 = 3\), the right side is \(1 \cdot 3 = 3\), equal as they must be, and the fourth point is forced positive. So this labeling is not realized, and the \(d + 1\) points are not shattered. \(\blacksquare\)

Python can check both halves in \(\mathbb R^3\). The second half uses four points with the dependency \(\mathbf e_1 + \mathbf e_2 + \mathbf e_3 - (1, 1, 1) = \mathbf 0\), so \(I = \{1, 2, 3\}\), \(J = \{4\}\), and the forbidden labeling is \((+, +, +, -)\):

```python
import itertools, random

dot = lambda u, v: sum(a * b for a, b in zip(u, v))
E = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
for y in itertools.product((1, -1), repeat=3):
    assert [dot(y, e) for e in E] == list(y)
print("e1, e2, e3 shattered: all 8 labelings")

X, a = E + [(1, 1, 1)], (1, 1, 1, -1)
rng = random.Random(0)
for _ in range(100_000):
    w = [rng.gauss(0, 1) for _ in range(3)]
    s = [dot(w, x) for x in X]
    assert abs(dot(a, s)) < 1e-9
    assert not (min(s[:3]) > 0 and s[3] <= 0)
print("(+, +, +, -) never appears")
```

Both lines print. The random weights are not a proof (the proof is above), but they show the mechanism: the weighted sum of the four scores is always 0, so three positive scores force the fourth to be positive too.

[[diagram:vc]]

**Theorem (VC dimension of halfspaces).** The VC dimension of \(HS_d\), halfspaces with a bias term in \(\mathbb R^d\), is \(d + 1\).

*Proof.* *A shattered set of size \(d + 1\):* the points \(\mathbf 0, \mathbf e_1, \dots, \mathbf e_d\). Given labels \(y_0\) for \(\mathbf 0\) and \(y_1, \dots, y_d\), set \(b = y_0 / 2\) and \(w_i = y_i - y_0/2\). At \(\mathbf 0\) the score is \(b = y_0/2\), with sign \(y_0\). At \(\mathbf e_i\) it is \(w_i + b = y_i\), with sign \(y_i\).

*No shattered set of size \(d + 2\):* suppose \(\mathbf x_1, \dots, \mathbf x_{d+2} \in \mathbb R^d\) were shattered by \(HS_d\). Append the constant 1 to each, as in the bias-term trick. For every labeling there is a \((\mathbf w, b)\) realizing it, and then \(\mathbf w' = (b, \mathbf w)\) realizes the same labeling on the vectors \((1, \mathbf x_i)\) with a homogeneous halfspace, because \(\langle \mathbf w', (1, \mathbf x_i) \rangle = \langle \mathbf w, \mathbf x_i \rangle + b\). So \(d + 2\) vectors in \(\mathbb R^{d+1}\) would be shattered by homogeneous halfspaces, contradicting the previous theorem. \(\blacksquare\)

```python
import itertools

d = 3
X = [(0,) * d] + [tuple(int(i == j) for j in range(d))
                  for i in range(d)]
sign = lambda a: 1 if a > 0 else -1
for y in itertools.product((1, -1), repeat=d + 1):
    b = y[0] / 2
    w = [yi - b for yi in y[1:]]
    h = [sign(sum(p * q for p, q in zip(w, x)) + b) for x in X]
    assert h == list(y)
print("0, e1, e2, e3 shattered:", 2 ** (d + 1), "labelings")
```

It prints `0, e1, e2, e3 shattered: 16 labelings`.

So a filter over a vocabulary of \(d\) words, with a bias term, has VC dimension \(d + 1\): 31 for thirty words. The fundamental theorem of lesson 6 now applies in the realizable case: there are constants \(C_1, C_2\) such that the sample complexity of \(HS_d\) satisfies

\[ C_1 \frac{(d + 1) + \log(1/\delta)}{\varepsilon} \le m_{HS_d}(\varepsilon, \delta) \le C_2 \frac{(d + 1) \log(1/\varepsilon) + \log(1/\delta)}{\varepsilon}, \]

and any ERM learner, the LP or the Perceptron, achieves it. Read it piece by piece. \(m_{HS_d}(\varepsilon, \delta)\) is the **sample complexity** from lesson 3: the number of labelled examples needed so that, with probability at least \(1 - \delta\), the learned halfspace has true risk at most \(\varepsilon\). The VC dimension \(d + 1\) appears in both fractions; \(\log\) is the natural logarithm; \(C_1\) and \(C_2\) are fixed numbers that do not depend on \(d\), \(\varepsilon\) or \(\delta\). The lower bound says no learner can do with fewer than the left side; the upper bound says ERM never needs more than the right side. The two differ only by the factor \(\log(1/\varepsilon)\) on the VC dimension. In plain words: **the labels needed are about the VC dimension over the accuracy, give or take a log and a constant.** With thirty words, \(\varepsilon = 0.1\) and \(\delta = 0.05\), the fractions are about 340 and 744:

```python
import math

d, eps, delta = 30, 0.1, 0.05
lower = ((d + 1) + math.log(1 / delta)) / eps
upper = ((d + 1) * math.log(1 / eps)
         + math.log(1 / delta)) / eps
print(round(lower, 1), round(upper, 1))
```

It prints `340.0 743.8`, before the constants. The number of words you track is, up to a log, the price in labels.

## Halfspaces as code

Here are the two learners for separable data. Each takes \(m\) examples whose vectors already include the constant 1, so they live in \(\mathbb R^n\) with \(n = d + 1\).

```
HalfspaceLP(S):
  for i = 1..m:
    for j = 1..n:
      A[i][j] ← y_i · x_{i,j}
  v ← (1, ..., 1) in R^m
  u ← (0, ..., 0) in R^n
  w ← LPSolve(maximize ⟨u, w⟩ subject to A w ≥ v)
  return h(x) = sign(⟨w, x⟩)

Perceptron(S):
  w ← (0, ..., 0) in R^n
  loop:
    if some i has y_i · ⟨w, x_i⟩ ≤ 0:
      w ← w + y_i · x_i
    else:
      return h(x) = sign(⟨w, x⟩)
```

**HalfspaceLP** spends \(O(mn)\) time and space building \(A\) (as in lesson 8, \(O(mn)\) reads “at most a constant times \(m\) times \(n\)”; eight messages with three entries each is 24 multiplications), and then calls a solver whose running time is polynomial in \(m\), \(n\) and the number of bits in the input. Which polynomial depends on the solver.

**Perceptron** keeps one vector, so it needs \(O(n)\) space beyond the data. Each pass through the loop scans for a mistake in \(O(mn)\) time, and by the mistake bound there are at most \((RB)^2\) updates, so the total is \(O(mn (RB)^2)\) time on separable data. On data that is not separable it never stops, so a real implementation also takes a cap on the number of updates. Predicting a new message costs \(O(n)\) for either learner.

**Build it yourself.** Using only the Python standard library:

```python
import itertools

S8 = [((3, 0), 1), ((2, 1), 1), ((4, 2), 1), ((1, 0), 1),
      ((1, 2), -1), ((0, 1), -1), ((2, 3), -1), ((0, 0), -1)]
S10 = S8 + [((2, 1), -1), ((1, 2), 1)]

def lift(x): ...
def lp_matrix(S): ...
def lp_vertex_solve(A, v): ...
def perceptron(S, max_updates=10_000): ...
def certified_bound(S, w): ...

if __name__ == "__main__":
    A, v, u = lp_matrix(S8)
    assert A[0] == [1, 3, 0] and A[4] == [-1, -1, -2]
    assert v == [1] * 8 and u == [0, 0, 0]
    w = lp_vertex_solve(A, v)
    assert all(sum(a * c for a, c in zip(r, w)) >= 1 - 1e-9
               for r in A)
    assert perceptron(S8) == ([-1, 2, -2], 3)
    assert abs(certified_bound(S8, (-1, 2, -2)) - 189) < 1e-9
    assert abs(certified_bound(S8, (-0.5, 1.5, -1)) - 294) < 1e-9
    assert perceptron(S10) is None
    assert lp_vertex_solve(*lp_matrix(S10)[:2]) is None
```

1. `lift(x)` appends the constant 1 in front. `lp_matrix(S)` returns \(A\), \(\mathbf v\) and \(\mathbf u\) as lists.
2. `perceptron` picks the first mistake in list order, returns `(w, T)` with `w` a list, and returns `None` once it has made `max_updates` updates without separating. `S10` adds two messages your users disagree on, so it is not separable.
3. `certified_bound(S, w)` returns \((R \|\mathbf w\| / \min_i y_i \langle \mathbf w, \mathbf x_i \rangle)^2\), the theorem’s promise using \(\mathbf w / \gamma\) as the certificate. Prove to yourself that it does not change when you multiply \(\mathbf w\) by a positive number.
4. `lp_vertex_solve` is a tiny LP solver for small \(n\): when the feasible region \(\{\mathbf w : A \mathbf w \ge \mathbf v\}\) is nonempty and the rows of \(A\) span \(\mathbb R^n\), it has a corner where \(n\) linearly independent constraints hold with equality. Try every set of \(n\) rows, solve those \(n\) equations by Gaussian elimination, and return the first solution that satisfies all \(m\) constraints, or `None`. How many sets does it try in the worst case, and why does that make it useless for a vocabulary of thirty words?
5. Shuffle `S8` a hundred times and record the Perceptron’s update counts. Does any order come near 142?

## Fix the bug

Time to ship. The team assembles a fresh batch of thirty labelled messages, still on the two words so you can see them, and they are separable. The filter will be retrained by the Perceptron every night on whatever new labels arrive, in whatever order they arrive. Operations has one requirement before it goes into the nightly job: a written guarantee, before training starts, of the most updates it can possibly take, for *every* order of the examples, and that number has to fit their budget.

That is a job for the mistake bound, not for luck. Running the Perceptron once and reporting how many updates it took proves nothing about tomorrow’s order. What you can certify is \((R / \gamma_{\mathbf u})^2\), using the radius of the messages and the margin of a separator \(\mathbf u\) you find yourself. The wider your separator’s margin, the smaller the certified number, and the theorem guarantees that every order of the examples finishes within it.

In the lab below, each round brings a new batch. You read off the radius, drag a separator until its margin is as wide as you can make it, and submit your certified bound. Then the lab runs the Perceptron in twenty random orders and checks every run against your number. The budget is set so that a separator that merely separates will not do: you need one close to the widest. Three rounds in a row and the filter ships.

[[lab-act2]]

## A junk score: linear regression

The halfspace filter is in, and the facilities team gets its lunch invitations again. Then the product manager drops by with a different question. “Junk or keep” is too blunt for the new inbox design, which sorts messages by how suspicious they are. Could the learner predict a **junk score**: the percentage of recipients who click “report junk” on a message?

That is a **regression** problem: the label set is \(\mathcal Y = \mathbb R\), not \(\{\pm 1\}\). Linear regression is a common tool for modelling how a real-valued outcome depends on some explanatory variables. Its hypothesis class is just the affine functions, with no sign on top:

\[ \mathcal H_{\text{reg}} = L_d = \{ \mathbf x \mapsto \langle \mathbf w, \mathbf x \rangle + b \; : \; \mathbf w \in \mathbb R^d, \, b \in \mathbb R \}. \]

This is the same set \(L_d\) we read slowly at the start of the lesson, now given the name \(\mathcal H_{\text{reg}}\), “the hypothesis class for regression”: **every weighted sum plus a constant, used directly as the prediction.**

Here are six messages with their junk scores, using one word, the count of “free”. Notice the shape before any fitting: the score climbs slowly, then steeply, then flattens out near 100, because past a point every extra “free” convinces almost nobody new.

| “free” count \(x\) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| junk score \(y\) | 2 | 10 | 35 | 70 | 88 | 95 |

The 0-1 loss is useless here. If a message’s true score is 30, then predicting 30.001 and predicting 90 are both “wrong”, but one is clearly better. So we need a loss that grows with the size of the miss. Try a rough guess, “each free is worth 20 points”, \(h(x) = 20x\). On the message with three “free”s it predicts 60 against a true 70, a miss of \(-10\); squaring the miss gives a cost of 100, while a miss of 2 would cost only 4. That is the **square loss** you met in lesson 3,

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

Read the system once more as a whole. \(A\) adds up the tables \(\mathbf x_i \mathbf x_i^\top\), one per message; \(\mathbf b\) adds up the message vectors, each scaled by its label; and \(A \mathbf w = \mathbf b\) asks for the weights that match them. In plain words: **the best weights solve one small linear system, built by adding a table and a vector per message.** For our six messages, the tables for \(x = 0, \dots, 5\) add to \(A = \begin{pmatrix} 6 & 15 \\ 15 & 55 \end{pmatrix}\) (six messages, total count 15, total squared count 55), and \(\mathbf b = (300, 1117)\) (total score, and total count times score). The solving snippet below builds both, entry by entry, as these same sums.

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

A polynomial in \(x\) is a *linear* function of \(\psi(x)\). So turn each example \((x_i, y_i)\) into \((\psi(x_i), y_i)\) and run least squares on the new vectors; the weights it returns are the coefficients. Learning reduces to a problem we have already solved.

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

So the logistic hypothesis and the halfspace with the same \(\mathbf w\) agree whenever \(|\langle \mathbf w, \mathbf x \rangle|\) is large. They differ near the boundary: there the halfspace still answers a flat \(\pm 1\), while the logistic hypothesis admits it is unsure and gives a probability just above or just below one half.

[[diagram:logistic]]

Now the loss. If the label is \(y = +1\), we want \(h_{\mathbf w}(\mathbf x)\) to be large; if \(y = -1\), we want \(1 - h_{\mathbf w}(\mathbf x)\), the probability given to \(-1\), to be large. Compute the second one:

\[ 1 - h_{\mathbf w}(\mathbf x) = 1 - \frac{1}{1 + \exp(-\langle \mathbf w, \mathbf x \rangle)} = \frac{\exp(-\langle \mathbf w, \mathbf x \rangle)}{1 + \exp(-\langle \mathbf w, \mathbf x \rangle)} = \frac{1}{1 + \exp(\langle \mathbf w, \mathbf x \rangle)}, \]

where the last step multiplies top and bottom by \(\exp(\langle \mathbf w, \mathbf x \rangle)\). The first step writes \(1\) as \((1 + \exp(-\langle \mathbf w, \mathbf x \rangle))\) over itself and subtracts the tops. In plain words: **the probability of keep is the logistic function of the negated score.** For message B, score 1, the probability of junk is 0.731 and of keep \(1/(1 + e^1) = 1/3.718 \approx 0.269\), and the two add to 1; the `assert` in the next snippet checks this identity at six scores. Look at the two cases side by side: the probability the hypothesis gives to the *true* label is \(1 / (1 + \exp(-y \langle \mathbf w, \mathbf x \rangle))\) for both \(y = +1\) and \(y = -1\). One formula, thanks to \(\pm 1\) labels once more. A sensible loss should grow as that probability falls, that is, as \(1 + \exp(-y \langle \mathbf w, \mathbf x \rangle)\) grows. The **logistic loss** takes its logarithm (natural log, an increasing function, so the order is kept):

\[ \ell(h_{\mathbf w}, (\mathbf x, y)) = \log \left( 1 + \exp(-y \langle \mathbf w, \mathbf x \rangle) \right). \]

Inside out: \(y \langle \mathbf w, \mathbf x \rangle\) is the signed score from the halfspace sections; the minus sign and \(\exp\) turn it into a number that is tiny when the signed score is large and huge when it is very negative; adding 1 gives the bottom of the probability of the true label; and \(\log\), the natural logarithm, the inverse of \(\exp\) (so \(\log e^3 = 3\)), turns that into a cost. Since the probability is one over that bottom, the loss is also \(-\log\) of the probability given to the true label. In plain words: **the loss is minus the log of the probability the hypothesis gave to the right answer.** Message B, signed score 1, gets probability 0.731 and pays \(\log 1.368 \approx 0.313\). The snippet mirrors this with `sigma` for \(\varphi_{\text{sig}}\) and `loss` for \(\ell\), as functions of the signed score \(z\):

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

In plain words: **the slope of the loss always increases, by the product of the two class probabilities.** At \(z = 1\) that is \(0.731 \cdot 0.269 \approx 0.197\), and nudging the slope by a thousandth agrees:

```python
import math

g = lambda z: math.log(1 + math.exp(-z))
sigma = lambda z: 1 / (1 + math.exp(-z))
z, h = 1, 1e-3
numeric = (g(z + h) - 2 * g(z) + g(z - h)) / h ** 2
print(round(numeric, 3), round(sigma(z) * sigma(-z), 3))
```

It prints `0.197 0.197`. A function of one variable whose second derivative is never negative is convex. Each term of the risk is \(g\) applied to \(z_i(\mathbf w) = y_i \langle \mathbf w, \mathbf x_i \rangle\), which is linear in \(\mathbf w\), so \(z_i(\theta \mathbf u + (1 - \theta) \mathbf v) = \theta z_i(\mathbf u) + (1 - \theta) z_i(\mathbf v)\), and convexity of \(g\) carries straight over to the term. An average of convex functions is convex. Lesson 12 develops this properly, and lesson 14 gives a simple algorithm for minimizing such functions.

There is also a statistical reading. Suppose the labels really were drawn with \(\mathbb P[y = +1 \mid \mathbf x] = \varphi_{\text{sig}}(\langle \mathbf w, \mathbf x \rangle)\). By the identity above, the probability of the observed label is \(\varphi_{\text{sig}}(y \langle \mathbf w, \mathbf x \rangle)\) in both cases, so the probability of the whole training sequence, with its examples independent, is \(\prod_i \varphi_{\text{sig}}(y_i \langle \mathbf w, \mathbf x_i \rangle)\). Its negative logarithm is \(\sum_i \log(1 + \exp(-y_i \langle \mathbf w, \mathbf x_i \rangle))\), which is \(m\) times the logistic risk. So minimizing the logistic risk is the same as choosing the \(\mathbf w\) under which the data is most likely, the **maximum likelihood** approach that lesson 24 studies.

The notation there needs reading. \(\mathbb P[y = +1 \mid \mathbf x]\) is the probability that the label is junk, where the bar \(\mid\) reads “given”: given that the message is \(\mathbf x\). The capital pi \(\prod_i\) is like \(\sum_i\) but multiplies instead of adds, one factor per example, because the chance of several independent events all happening is the product of their chances. And the negative logarithm turns the product into a sum, since \(\log(ab) = \log a + \log b\). In plain words: **the most likely weights and the lowest logistic risk are the same weights.** With \(\mathbf w = (-1, 2, -2)\) on the eight messages, the probabilities of the true labels multiply to about 0.3195, minus its log is 1.1408, and eight times the average logistic loss is 1.1408 too:

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

Let’s fit it. Your users have now labelled two more messages, and they disagree with themselves: a second “free trial, book a meeting” (\(x = (2, 1)\)) was marked keep, and a second “free lunch at the meeting” (\(x = (1, 2)\)) was marked junk. No halfspace can be consistent with that. As a preview of lesson 14, minimize the logistic risk by plain **gradient descent**: start at \(\mathbf w = \mathbf 0\) and repeatedly step against the gradient, which by the chain rule is \(\frac1m \sum_i -y_i \mathbf x_i \, \varphi_{\text{sig}}(-y_i \langle \mathbf w, \mathbf x_i \rangle)\). Read it inside out: \(\varphi_{\text{sig}}(-y_i \langle \mathbf w, \mathbf x_i \rangle)\) is the probability the current weights give to the *wrong* label, so it is large on badly handled messages; it scales the vector \(-y_i \mathbf x_i\), the message pointed against its label; and the sum and \(\frac1m\) average over messages. In plain words: **each message pulls the weights towards its own label, harder the more the weights currently doubt it.** At \(\mathbf w = \mathbf 0\) every wrong-label probability is one half, so junk message A contributes \(-\tfrac12 (1, 3, 0)\) before averaging, and stepping against the gradient moves the weights towards A by \(\tfrac{1}{2m}\) of its vector: a softened Perceptron update. In the code, `g` is this gradient and the step size is 1.

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

**LeastSquares** spends \(O(md^2)\) time building \(A\) (each outer product has \(d^2\) entries) and \(O(md)\) building \(\mathbf b\), then \(O(d^3)\) solving the \(d \times d\) system by Gaussian elimination, or computing \(A^+\) from an eigendecomposition. Total \(O(md^2 + d^3)\) time and \(O(d^2)\) space, independent of \(m\) once \(A\) is built: you can stream the examples.

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

## What people get wrong

**“The Perceptron needs about \((RB)^2\) updates.”** It needs *at most* that many. The bound is a worst case over every order of the examples, and on our eight messages it promised 142 while the Perceptron used 3. What the bound buys is a guarantee you can write down before training; it is not a forecast.

**“Any separator gives the bound \((RB)^2\).”** A separator \(\mathbf u\) that you happen to know certifies \((R / \gamma_{\mathbf u})^2\), which is at least \((RB)^2\) and can be much larger when \(\mathbf u\) barely separates. Only the widest separator gives \((RB)^2\) itself. That is the whole game of the lab’s second act.

**“Halfspaces are easy to learn.”** In the separable case, yes: a linear program, or the Perceptron. Without separability, finding a halfspace with the smallest empirical risk under the 0-1 loss is computationally hard. That is why the practical answer for messy data is to change the loss, as logistic regression does.

**“The bias term is the inductive bias.”** Same word, different ideas. The bias term \(b\) is one more weight, the score of a message with no evidence. The inductive bias is the decision to search halfspaces at all, made before seeing any data.

**“A logistic probability of 0.71 means 71% of such messages are junk.”** Only if the data really follow the logistic model. The number is what the fitted hypothesis says; whether it is calibrated is a separate question.

## Recap and the next bug

1. A halfspace labels a message by the sign of a weighted sum of its features plus a bias term. Appending a constant 1 to every message turns the bias term into one more weight.
2. In the separable case, ERM over halfspaces is a linear program: rescale any separator so every signed score is at least 1, and ask a solver for a feasible point.
3. The Perceptron adds \(y_i \mathbf x_i\) to the weights on each mistake. On separable data it stops after at most \((RB)^2 = (R/\gamma^\star)^2\) updates, a bound with no \(m\) and no \(d\) in it, proved by watching the angle to a good separator close.
4. Homogeneous halfspaces in \(\mathbb R^d\) have VC dimension \(d\), and halfspaces with a bias term have \(d + 1\), so the number of examples grows with the number of words.
5. Least squares solves ERM for linear regression with the square loss: solve \(A \mathbf w = \mathbf b\), with \(A^+\) when \(A\) is singular. Polynomial regression is least squares on \(\psi(x) = (1, x, \dots, x^n)\).
6. Logistic regression passes the linear score through the logistic function to get a probability, and its loss \(\log(1 + \exp(-y \langle \mathbf w, \mathbf x \rangle))\) makes ERM a convex problem, even when the data are not separable.

The filter weighs its evidence now, and the inbox has scores and probabilities. Then a colleague from the search team stops by your desk with a different idea. “Your halfspace is fine, but the single-feature rules weren’t useless. They’re absurdly cheap: one word, one threshold, you can find the best one in a single sorted pass. Each one on its own is barely better than a coin, sure. But what if you trained a few hundred of them, each one paying attention to the messages the others got wrong, and let them vote?” Can a crowd of weak rules, each only slightly better than guessing, add up to a strong one? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A colleague drops the constant 1 coordinate “to save memory”, so the Perceptron runs on the raw two-word counts with a homogeneous halfspace. The empty message H, \(\mathbf x = (0, 0)\), labelled keep, is still in the training sequence. What does the Perceptron do? Is the training sequence separable by homogeneous halfspaces, and what does the mistake bound say about it?
2. In Step 2 of the mistake-bound proof, exactly where is the fact that the Perceptron only updates on mistakes used? Now consider a variant that also updates on examples it gets right but with a small score, whenever \(y_i \langle \mathbf w, \mathbf x_i \rangle < 1\). Does Step 1 still hold? Redo Step 2 and Step 3 for this variant and state the bound you get.
3. Someone proposes \(\mathbf w = (-3, 5, -4)\) (bias term first) as a certificate for the eight messages. Check that it separates them, compute the bound \((R / \gamma_{\mathbf u})^2\) it certifies, and compare it with the 189 from \((-1, 2, -2)\) and the best possible 142.3. Which certificate would you hand to operations, and why?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (absolute loss as a linear program).** Show how to cast ERM for linear regression with the absolute loss, \(\min_{\mathbf w} \sum_{i=1}^m |\langle \mathbf w, \mathbf x_i \rangle - y_i|\), as a linear program. Start by proving that for any real \(c\), \(|c| = \min_{a \ge 0} a\) subject to \(c \le a\) and \(c \ge -a\).
- **Problem 2 (the mistake bound is tight).** Show that for every positive integer \(m\) there are a dimension \(d\), a vector \(\mathbf w^\star \in \mathbb R^d\) and examples \((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m)\) with \(R = \max_i \|\mathbf x_i\| \le 1\), \(\|\mathbf w^\star\|^2 = m\) and \(y_i \langle \mathbf x_i, \mathbf w^\star \rangle \ge 1\) for all \(i\) (so \((RB)^2 \le m\)), on which the Perceptron makes \(m\) updates before it stops.
- **Problem 3 (balls; the second part is harder).** A closed ball in \(\mathbb R^d\) labels \(\mathbf x\) with 1 when \(\|\mathbf x - \mathbf v\| \le r\) and 0 otherwise. (a) Using the map \(\phi(\mathbf x) = (\mathbf x, \|\mathbf x\|^2) \in \mathbb R^{d+1}\), show that if balls shatter \(\mathbf x_1, \dots, \mathbf x_m\), then halfspaces in \(\mathbb R^{d+1}\) shatter \(\phi(\mathbf x_1), \dots, \phi(\mathbf x_m)\), taking \(\operatorname{sign}(0) = +1\) for this problem. What does that say about the VC dimension of balls? (b) Find \(d + 1\) points in \(\mathbb R^d\) that balls shatter, and conclude that \(d + 1 \le \mathrm{VCdim} \le d + 2\).
