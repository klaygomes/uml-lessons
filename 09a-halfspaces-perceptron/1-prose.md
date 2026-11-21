# Halfspaces and the Perceptron

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

This lesson is about the class you just wrote down, the **halfspaces**, and about why learning it is not hopeless at all. In the right conditions, finding a consistent halfspace is a linear program (a standard optimization problem that off-the-shelf solvers handle, defined below), or a five-line loop with a guarantee on how long it runs. At the end the team will ask for two more things, a junk *score* and a *probability* of junk, and lesson 9b will answer both with the same linear idea.

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

A linear function returns a number, but the filter has to return a label. Remember from lesson 1 that \(\mathcal X\) is the **domain set**, every message that could arrive, and \(\mathcal Y\) is the **label set**, every answer the filter may give. To get a label, feed the score into a second function \(\varphi: \mathbb R \to \mathcal Y\); the notation \(\varphi: \mathbb R \to \mathcal Y\) says “\(\varphi\) takes a real number and returns a label”. For regression, where \(\mathcal Y = \mathbb R\), \(\varphi\) is the identity, which returns its input unchanged, and lesson 9b gets there. For classification, \(\varphi\) is the **sign** function, \(\operatorname{sign}(a) = +1\) if \(a > 0\) and \(-1\) otherwise: \(\operatorname{sign}(5) = +1\), \(\operatorname{sign}(-3) = -1\), and \(\operatorname{sign}(0) = -1\). This is why the labels became \(\pm 1\): the sign of the score *is* the label.

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

The first is statistical. \(HS_d\) is infinite, so lesson 2’s finite-class bound says nothing. But lesson 6a gave us the tool for infinite classes, the **VC dimension**, the size of the largest set of points a class can label in every possible way, and later in this lesson we will prove that the VC dimension of \(HS_d\) is \(d + 1\). Recall the two dials of PAC learning: \(\varepsilon\) is the accuracy you ask for (true risk at most \(\varepsilon\)), and \(\delta\) is the chance of failure you tolerate (the guarantee holds with probability at least \(1 - \delta\)). By the fundamental theorem of lesson 6a, ERM over halfspaces is then a PAC learner, and the number of examples it needs grows with the number of words, not with the \(2^{1984}\) settings of the weights: for thirty words, \(\delta = 0.05\) and \(\varepsilon = 0.1\), it is a few hundred times a constant, as we will compute at the end of the VC section. So statistically, halfspaces are fine.

The second worry is computational, and it is the one from the opening: how do you *find* the empirical risk minimizer among infinitely many halfspaces? In general, when no halfspace gets every example right (the **agnostic** case of lesson 3), this is known to be computationally hard. The usual way around it is to minimize a different, friendlier loss instead of the 0-1 loss; logistic regression, in lesson 9b, is one example, and lesson 12b studies the idea in depth.

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

It prints `-4 2 6 6`.

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

Both algorithms find a consistent halfspace. Lesson 6a tells us how many examples make a consistent hypothesis trustworthy, as long as we know the VC dimension. Recall that a class **shatters** a finite set of points \(C\) when it can realize every one of the \(2^{|C|}\) possible labelings of that set (the bars \(|C|\) count how many points \(C\) has, and each point gets one of two labels, so three points have \(2^3 = 8\) labelings), and the **VC dimension** is the size of the largest set it shatters. We start with homogeneous halfspaces, \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\).

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

So a filter over a vocabulary of \(d\) words, with a bias term, has VC dimension \(d + 1\): 31 for thirty words. The fundamental theorem of lesson 6a now applies in the realizable case: there are constants \(C_1, C_2\) such that the sample complexity of \(HS_d\) satisfies

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

## What people get wrong

**“The Perceptron needs about \((RB)^2\) updates.”** It needs *at most* that many. The bound is a worst case over every order of the examples, and on our eight messages it promised 142 while the Perceptron used 3. What the bound buys is a guarantee you can write down before training; it is not a forecast.

**“Any separator gives the bound \((RB)^2\).”** A separator \(\mathbf u\) that you happen to know certifies \((R / \gamma_{\mathbf u})^2\), which is at least \((RB)^2\) and can be much larger when \(\mathbf u\) barely separates. Only the widest separator gives \((RB)^2\) itself. That is the whole game of the lab’s second act.

**“Halfspaces are easy to learn.”** In the separable case, yes: a linear program, or the Perceptron. Without separability, finding a halfspace with the smallest empirical risk under the 0-1 loss is computationally hard. That is why the practical answer for messy data is to change the loss, as the logistic regression of lesson 9b does.

**“The bias term is the inductive bias.”** Same word, different ideas. The bias term \(b\) is one more weight, the score of a message with no evidence. The inductive bias is the decision to search halfspaces at all, made before seeing any data.

## Recap and the next bug

1. A halfspace labels a message by the sign of a weighted sum of its features plus a bias term. Appending a constant 1 to every message turns the bias term into one more weight.
2. In the separable case, ERM over halfspaces is a linear program: rescale any separator so every signed score is at least 1, and ask a solver for a feasible point.
3. The Perceptron adds \(y_i \mathbf x_i\) to the weights on each mistake. On separable data it stops after at most \((RB)^2 = (R/\gamma^\star)^2\) updates, a bound with no \(m\) and no \(d\) in it, proved by watching the angle to a good separator close.
4. Homogeneous halfspaces in \(\mathbb R^d\) have VC dimension \(d\), and halfspaces with a bias term have \(d + 1\), so the number of examples grows with the number of words.

The filter weighs its evidence now, and the facilities team gets its lunch invitations again. Then the product manager drops by with a different question. “Junk or keep” is too blunt for the new inbox design, which sorts messages by how suspicious they are. Could the learner predict a junk score, the percentage of recipients who click “report junk” on a message? The inbox designers, right behind, would also like the filter’s confidence on hover. A halfspace answers with a sign and nothing else. Can the same weighted sum give back a number instead, and when it does, can you prove that no other line gives a better one? That is lesson 9b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A colleague drops the constant 1 coordinate “to save memory”, so the Perceptron runs on the raw two-word counts with a homogeneous halfspace. The empty message H, \(\mathbf x = (0, 0)\), labelled keep, is still in the training sequence. What does the Perceptron do? Is the training sequence separable by homogeneous halfspaces, and what does the mistake bound say about it?
2. In Step 2 of the mistake-bound proof, exactly where is the fact that the Perceptron only updates on mistakes used? Now consider a variant that also updates on examples it gets right but with a small score, whenever \(y_i \langle \mathbf w, \mathbf x_i \rangle < 1\). Does Step 1 still hold? Redo Step 2 and Step 3 for this variant and state the bound you get.
3. Someone proposes \(\mathbf w = (-3, 5, -4)\) (bias term first) as a certificate for the eight messages. Check that it separates them, compute the bound \((R / \gamma_{\mathbf u})^2\) it certifies, and compare it with the 189 from \((-1, 2, -2)\) and the best possible 142.3. Which certificate would you hand to operations, and why?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (the mistake bound is tight).** Show that for every positive integer \(m\) there are a dimension \(d\), a vector \(\mathbf w^\star \in \mathbb R^d\) and examples \((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m)\) with \(R = \max_i \|\mathbf x_i\| \le 1\), \(\|\mathbf w^\star\|^2 = m\) and \(y_i \langle \mathbf x_i, \mathbf w^\star \rangle \ge 1\) for all \(i\) (so \((RB)^2 \le m\)), on which the Perceptron makes \(m\) updates before it stops.
- **Problem 2 (balls; the second part is harder).** A closed ball in \(\mathbb R^d\) labels \(\mathbf x\) with 1 when \(\|\mathbf x - \mathbf v\| \le r\) and 0 otherwise. (a) Using the map \(\phi(\mathbf x) = (\mathbf x, \|\mathbf x\|^2) \in \mathbb R^{d+1}\), show that if balls shatter \(\mathbf x_1, \dots, \mathbf x_m\), then halfspaces in \(\mathbb R^{d+1}\) shatter \(\phi(\mathbf x_1), \dots, \phi(\mathbf x_m)\), taking \(\operatorname{sign}(0) = +1\) for this problem. What does that say about the VC dimension of balls? (b) Find \(d + 1\) points in \(\mathbb R^d\) that balls shatter, and conclude that \(d + 1 \le \mathrm{VCdim} \le d + 2\).
