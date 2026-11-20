# Nonuniform Learnability

## The list that could grow forever

Your teammate has a pull request open, and she is excited about it. The filter, she argues, has been too timid. Every lesson so far has fixed a hypothesis class in advance and stayed inside it, and every real mistake she sees in the inbox would have been caught by one more test. So her change lets the filter learn a **decision list**: a chain of tests, read from the top, where each test looks at one feature of the message and, if that feature is present, hands out a label on the spot. If no test fires, a default label applies. Here is one:

```
if the sender is unknown:      junk
else if the subject says FREE: keep
else if it has a link:         junk
otherwise:                     keep
```

Her point is that nobody knows how long the right list is. Three tests? Five? Twelve? So her learner may use lists of any length, over any feature in the catalogue, and the catalogue keeps growing: the team adds a new feature most weeks. Whatever the junk looks like, some list will describe it.

The reviewer blocks the change, and for a reason you already know. Lists of any length can fit any labels at all on as many messages as you like. Recall from lesson 6 that a class **shatters** a set of messages when, for every way of labelling them, some hypothesis in the class produces exactly those labels, and that its **VC dimension** is the largest number of messages it can shatter. A class that can fit any labels on as many messages as you like has infinite VC dimension. Recall too that a class is **PAC learnable** (lesson 3) when one number of labels, fixed before you see any data, is enough to make the learner nearly as good as the best hypothesis in the class with high probability, whatever the mail stream looks like. The fundamental theorem of lesson 6 says flatly that a class with infinite VC dimension is not PAC learnable. No number of labels gives a guarantee that holds for every hypothesis in the class and every distribution. Rejected.

Your teammate is not wrong either, though, and that is the bug. Common sense says something the theorem does not seem to allow. A three-test list that fits a thousand labelled messages well deserves your trust. A forty-test list that fits thirty labelled messages perfectly deserves none. Short lists should be easy to trust, long ones should need more evidence, and nothing should be ruled out forever. The fundamental theorem gives one sample size for the whole class. What we want is a guarantee whose price depends on the hypothesis.

That is exactly what this lesson builds. By the end you will know how to split your confidence across list lengths, so that a short list is certified with few labels and a long one with many, and you will see the same idea reappear as a precise version of Occam’s razor.

## A mail room with no last feature

Let’s set up the world carefully, because the details matter here.

Features are numbered in the order the team added them. You know the first four from lesson 1: feature 1 is F (the subject says “FREE”), feature 2 is L (it has a link), feature 3 is U (the sender is unknown) and feature 4 is N (it was sent at night). Feature 5 is A (it has an attachment) and feature 6 is R (it is a reply in a thread you started). There is no last feature. A message is the infinite bit sequence \(x = (x_1, x_2, x_3, \dots)\) with \(x_j = 1\) when feature \(j\) is present, and every message has only finitely many features present. So the domain set \(\mathcal X\) is infinite, but countable: you could list every message one after another.

A decision list is written as its tests in order, each test a pair (feature number, label), then a semicolon and the default label. The list in the opening tests feature 3 and says junk, then feature 1 and says keep, then feature 2 and says junk, and otherwise says keep, so it is written \(((3, 1), (1, 0), (2, 1); 0)\). In general a list with \(k\) tests is written \(h = ((j_1, b_1), \dots, (j_k, b_k); b_0)\), where each \(j_i\) is a feature number and each \(b_i \in \{0, 1\}\) is a label (1 for junk, 0 for keep). It labels a message like this:

\[ h(x) = \begin{cases} b_i & \text{if } i \text{ is the first test with } x_{j_i} = 1, \\ b_0 & \text{if no test fires.} \end{cases} \]

Read it from the inside out. The subscript \(i\) numbers the tests from the top, so \(j_i\) is the feature that test \(i\) looks at and \(b_i\) is the label it hands out; \(b_0\) is the default, numbered 0 because it comes after every test. A subscript on a subscript, \(x_{j_i}\), means “bit number \(j_i\) of the message \(x\)”, so \(x_{j_i} = 1\) says “test \(i\) fires on \(x\)”. The big left brace with two rows is a **case split**: the value is whatever the first row says when its condition holds, and the second row’s otherwise. In plain words: **go down the tests, and the first one whose feature is present decides the label; if none fires, the default decides.** Take a message with “FREE” and a link but a known sender, so \(x_1 = 1\), \(x_2 = 1\), \(x_3 = 0\). Test 1 looks at \(x_3 = 0\) and does not fire. Test 2 looks at \(x_1 = 1\) and fires, so \(h(x) = b_2 = 0\): keep.

```python
def predict(tests, default, x):
    for j, b in tests:
        if x[j - 1]:
            return b
    return default

h = [(3, 1), (1, 0), (2, 1)]
for x in [(1, 1, 0, 0), (1, 1, 1, 0), (0, 0, 0, 1)]:
    print(x, predict(h, 0, x))
```

It prints `(1, 1, 0, 0) 0`, `(1, 1, 1, 0) 1` and `(0, 0, 0, 1) 0`. The loop is the “first test that fires” and the final `return default` is the second row. Python counts positions from 0, which is why feature \(j\) sits at `x[j - 1]`. The last message has only N present, which no test looks at, so the default answers.

Now we sort lists by how big they are. Let \(\mathcal H_n\) be the decision lists with at most \(n\) tests, each on a different feature among features \(1, \dots, n\). Call this the class of lists of **size** at most \(n\). The **size** of a list is the smallest \(n\) for which it belongs to \(\mathcal H_n\): its number of tests or its highest feature number, whichever is larger (and at least 1). The opening list has three tests and its highest feature is 3, so it has size 3. The one-test list “feature 7 present, junk; otherwise keep”, \(((7, 1); 0)\), has one test but uses feature 7, so it has size 7. The empty list “everything is junk”, \((();1)\), has size 1. Every list of size 3 also counts as a list of size at most 4, at most 5, and so on, so each class contains the one before: \(\mathcal H_1 \subseteq \mathcal H_2 \subseteq \cdots\), where \(\subseteq\) reads “is contained in”. Your teammate’s class is all of them put together, their union:

\[ \mathcal H = \bigcup_{n \in \mathbb N} \mathcal H_n. \]

The big cup \(\bigcup\) means “put together everything in any of these sets”, and the subscript \(n \in \mathbb N\) says which sets: one for each \(n\) in \(\mathbb N = \{1, 2, 3, \dots\}\), the positive whole numbers. In plain words: **a list belongs to your teammate’s class when it belongs to some \(\mathcal H_n\), that is, when it has some finite size.** The opening list is in it because it is in \(\mathcal H_3\).

```python
def size(tests):
    features = [j for j, b in tests]
    return max([1, len(tests)] + features)

print(size([(3, 1), (1, 0), (2, 1)]))
print(size([(7, 1)]), size([]))
```

It prints `3`, then `7 1`: the size is the larger of the test count and the highest feature number, never below 1, and that is the first \(\mathcal H_n\) the list lands in.

Why tie the features a list may use to its size? Because without that, even one-test lists would be hopeless: a single test “feature \(j\) is present” can pick out any subset of \(k\) messages if the messages are allowed to carry many features, so one-test lists over an unbounded catalogue would already have infinite VC dimension. With the tie, every \(\mathcal H_n\) is a finite class, and you can count it. Try \(n = 1\) by hand. A list in \(\mathcal H_1\) has no tests, and then only a default, 0 or 1: two lists. Or it has one test, on feature 1, with a label and a default: \(2 \times 2 = 4\) lists. That makes 6. In general a list in \(\mathcal H_n\) has some number \(k \le n\) of tests, an ordered choice of \(k\) distinct features out of \(n\), a label for each test and a default label, so

\[ |\mathcal H_n| = 2 \sum_{k=0}^{n} 2^k \frac{n!}{(n-k)!}. \]

Read it from the inside out. The vertical bars \(|\cdot|\) around a set mean “how many elements it has”, so \(|\mathcal H_n|\) is the number of lists. The exclamation mark is the **factorial**, \(n! = 1 \times 2 \times \cdots \times n\), and \(n!/(n-k)! = n(n-1)\cdots(n-k+1)\) counts the ordered ways to pick \(k\) different features out of \(n\): \(n\) choices for the first test, \(n - 1\) for the second, and so on. The factor \(2^k\) is a junk-or-keep label for each of the \(k\) tests. The **sum** \(\sum_{k=0}^{n}\) adds the term up for every test count \(k\) from 0 to \(n\), and the 2 in front is the default label. In plain words: **for each possible number of tests, count the ordered features times the labels, add over all test counts, and double for the default.** For \(n = 2\) the terms are 1 (no tests), \(2 \times 2 = 4\) (one test, on feature 1 or 2) and \(4 \times 2 = 8\) (two tests, in either order), which sum to 13 and double to 26.

```python
import itertools
import math

def size_H(n):
    terms = [2**k * math.perm(n, k) for k in range(n + 1)]
    return terms, 2 * sum(terms)

def brute(n):
    return sum(2 ** (k + 1) for k in range(n + 1)
               for _ in itertools.permutations(range(n), k))

for n in (1, 2, 3):
    terms, total = size_H(n)
    print(n, terms, total, brute(n), round(math.log2(total), 2))
```

It prints `1 [1, 2] 6 6 2.58`, `2 [1, 4, 8] 26 26 4.7` and `3 [1, 6, 24, 48] 158 158 7.3`. `math.perm(n, k)` is \(n!/(n-k)!\), and `brute` lists every ordered choice of features one by one and gets the same totals. So there are 6 lists of size at most 1, 26 of size at most 2, 158 of size at most 3, and then it explodes: 151,946 of size at most 6.

Each \(\mathcal H_n\) is finite, so its VC dimension is finite. Shattering \(d\) messages means producing all \(2^d\) labellings of them, which takes at least \(2^d\) different hypotheses, so \(2^d \le |\mathcal H_n|\). Taking the base-2 logarithm, \(\log_2\), which asks “2 to what power gives this number?”, turns that into \(d \le \log_2 |\mathcal H_n|\). The last column above is that logarithm: the 6 lists of \(\mathcal H_1\) can shatter at most 2 messages, since \(2^3 = 8\) labellings would need 8 lists. The union is another story. Here is the reviewer’s argument, made exact.

**The union has infinite VC dimension.** Fix any \(k\), and let \(e_i\) be the message with feature \(i\) present and nothing else, for \(i = 1, \dots, k\). Take any labelling \(y_1, \dots, y_k\) of these \(k\) messages. The list \(((1, y_1), (2, y_2), \dots, (k, y_k); 0)\) is in \(\mathcal H_k\), and on \(e_i\) its first \(i - 1\) tests do not fire (those features are absent) while test \(i\) does, so it returns \(y_i\). Every labelling is realized, so \(\mathcal H\) shatters \(\{e_1, \dots, e_k\}\). Since \(k\) was arbitrary, the VC dimension of \(\mathcal H\) is infinite. \(\blacksquare\)

[[diagram:decision-list]]

You can watch the shattering happen for \(k = 5\): all 32 labellings, each fitted by its own list. The `predict` function is the one from the case split above; `E` holds the five messages \(e_1, \dots, e_5\).

```python
import itertools

def predict(tests, default, x):
    for j, b in tests:
        if x[j - 1]:
            return b
    return default

k = 5
E = [tuple(int(i == j) for i in range(k)) for j in range(k)]
hits = 0
for y in itertools.product((0, 1), repeat=k):
    tests = list(zip(range(1, k + 1), y))
    hits += all(predict(tests, 0, x) == t for x, t in zip(E, y))
print(hits, 2**k)
```

It prints `32 32`. So the reviewer is right, and lesson 6 settles it: \(\mathcal H\) is not PAC learnable. Recall the two dials of a PAC guarantee: the **accuracy** \(\varepsilon\), how close to the best hypothesis you want to get (say within 0.1), and the **confidence** \(\delta\), the chance you allow for the guarantee to fail (say 0.05). A PAC learnable class comes with a **sample size** \(m_{\mathcal H}(\varepsilon, \delta)\), a number of labels that depends only on those two dials. For decision lists no such number works for every list and every distribution. Hold that thought, and let’s look at what actually happens on data.

## Thirty labels, six sizes

To have something concrete, here is the mail stream we’ll use for the whole lesson. Each of features 1 to 6 is present with probability 1/2, independently; features beyond 6 happen never to appear in this stream (the learner does not know that). The labels follow the opening list, extended by two more tests:

\[ f = ((3, 1), (1, 0), (2, 1), (5, 0), (4, 1); 0), \]

Read in words, \(f\) says: unknown sender, junk; “FREE”, keep; link, junk; attachment, keep; sent at night, junk; otherwise keep. Then, as in lesson 3’s noisy mail room, 10% of the labels are flipped because users disagree. Recall that the **true risk** \(L_{\mathcal D}(h)\) of a list is the chance that it mislabels the next message to arrive, where \(\mathcal D\) is the distribution of the mail stream. So the best any hypothesis can do is a true risk of 0.10, and \(f\) achieves it.

What about a list that disagrees with \(f\) on a fraction \(p\) of messages? Where it agrees with \(f\), it is wrong only when the label was flipped, 10% of the time. Where it disagrees, it is wrong unless the label was flipped, 90% of the time. So its true risk is \((1 - p) \cdot 0.1 + p \cdot 0.9\), which simplifies to \(0.1 + 0.8p\): **the noise floor of 0.1, plus 0.8 for every unit of disagreement with \(f\).** “Everything is junk” disagrees with \(f\) on the 22 of the 64 combinations of features 1 to 6 that \(f\) calls keep, so \(p = 22/64 = 0.34375\) and its true risk is \(0.1 + 0.8 \times 0.34375 = 0.375\).

```python
import itertools

def predict(tests, default, x):
    for j, b in tests:
        if x[j - 1]:
            return b
    return default

f = [(3, 1), (1, 0), (2, 1), (5, 0), (4, 1)]
X = list(itertools.product((0, 1), repeat=6))
p = sum(predict(f, 0, x) != 1 for x in X) / len(X)
print(p, (1 - p) * 0.1 + p * 0.9, round(0.1 + 0.8 * p, 4))
```

It prints `0.34375 0.375 0.375`: the two-case sum and the simplified form agree.

Your users label \(m = 30\) messages, drawn **i.i.d.** from this stream: independently, each from the same distribution \(\mathcal D\). Recall that the **empirical risk** \(L_S(h)\) is the fraction of the labelled messages, the training sequence \(S\), that \(h\) gets wrong, and that **ERM** (empirical risk minimization, lesson 2) returns a hypothesis with the smallest empirical risk in its class. For each size cap \(n\), run ERM over \(\mathcal H_n\): find the lists in \(\mathcal H_n\) with the smallest empirical risk. Before you look at the table, guess: as the cap grows, what happens to the best empirical risk, and what happens to the true risk of the lists that achieve it?

| cap \(n\) | \(\lvert\mathcal H_n\rvert\) | smallest \(L_S\) | lists tied at it | their true risks |
|---|---|---|---|---|
| 1 | 6 | 0.267 | 2 | 0.375 |
| 2 | 26 | 0.267 | 5 | 0.375 |
| 3 | 158 | 0.167 | 3 | 0.125 to 0.175 |
| 4 | 1,266 | 0.133 | 6 | 0.225 to 0.275 |
| 5 | 12,662 | 0.133 | 127 | 0.100 to 0.425 |
| 6 | 151,946 | 0.133 | 720 | 0.100 to 0.425 |

Read it top to bottom. The empirical risk can only go down, because each class contains the one before. The true risk does not follow. At cap 3, ERM lands on lists with true risk 0.125 or 0.175, which is decent. At cap 4 the empirical risk improves to 0.133 and the true risk gets worse. At cap 6, 720 lists tie for the best empirical risk, and their true risks run from the best possible, 0.100, to 0.425, worse than labelling everything junk. Thirty examples simply cannot tell those 720 lists apart. That is lesson 1’s 256 consistent hypotheses all over again, on a bigger stage.

How much can thirty, or three hundred, examples certify for each class? Lesson 4 answered that for a finite class, with two tools. **Hoeffding’s inequality** says that the average of \(m\) independent mistakes-or-not rarely strays far from its expected value, so for one fixed list the empirical risk is close to the true risk. The **union bound** says that the chance that at least one of several bad events happens is at most the sum of their chances, so the guarantee can be stretched to every list in the class at once, at the price of a factor \(|\mathcal H_n|\). Together they say that with probability at least \(1 - \delta\), every \(h \in \mathcal H_n\) has

\[ |L_{\mathcal D}(h) - L_S(h)| \le \sqrt{\frac{\log(2|\mathcal H_n|/\delta)}{2m}}. \]

Throughout, \(\log\) is the natural logarithm, the power of \(e \approx 2.718\) that gives the number. Read it from the inside out. On the left, the bars mean something different from the bars around a set: around a number they are the **absolute value**, the distance from zero, so \(|L_{\mathcal D}(h) - L_S(h)|\) is how far the score on the labels is from the true risk, in either direction. On the right, \(2|\mathcal H_n|/\delta\) grows with the number of lists and with how sure you want to be; the logarithm tames it; dividing by \(2m\) shrinks it as labels come in; and the square root \(\sqrt{\cdot}\) turns the result into a distance on the same scale as a risk. We call that right-hand side the **gap**. In plain words: **with probability at least \(1 - \delta\), no list in \(\mathcal H_n\) has an empirical risk that is off from its true risk by more than the gap.** For \(n = 3\), \(m = 300\) and \(\delta = 0.05\): \(2 \times 158 / 0.05 = 6{,}320\), its logarithm is 8.751, divided by 600 that is 0.01459, and the square root is 0.121. Here is that gap at \(m = 300\) and \(\delta = 0.05\), for each cap, with the function `eps` doing those four steps in one line:

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def eps(n, m, delta):
    return math.sqrt(math.log(2 * size_H(n) / delta) / (2 * m))

for n in range(1, 7):
    print(n, size_H(n), round(eps(n, 300, 0.05), 3))
```

It prints `1 6 0.096`, `2 26 0.108`, `3 158 0.121`, `4 1266 0.134`, `5 12662 0.148` and `6 151946 0.161`. The gap grows slowly, only with the logarithm of the class size, but it grows without bound as \(n\) does. A guarantee for the whole union would need the gap for \(n = \infty\), and there is no such number. That is the fundamental theorem’s verdict seen from the inside.

Now try it yourself. In the lab, pick a label count and a size cap, and watch the empirical risk, the spread of true risks among the tied lists, and the gap.

[[lab]]

## A confidence budget, shared out

Here is the way out. Look again at the gap formula. The \(\delta\) in it is a **confidence budget**: the probability you allow for the guarantee to fail. Lesson 4 spent the whole budget on one class. Nothing forces you to do that. You can split it.

Say your budget is \(\delta = 0.05\). Give size 1 a share of 0.03, size 2 a share of 0.0076, size 3 a share of 0.0034, and so on, shares that never add up to more than 0.05. In symbols, size \(n\) gets a share \(\delta_n\), with \(\delta_1 + \delta_2 + \cdots \le \delta\); the dots mean the sum goes on forever, one share per size. For each \(n\) separately, the finite-class bound with confidence \(\delta_n\) fails with probability at most \(\delta_n\). By the union bound, the probability that any of them fails is at most \(\sum_n \delta_n \le \delta\), where \(\sum_n\) is the sum over every size \(n\): the shares together, never more than the budget. So with probability at least \(1 - \delta\), all of them hold at once: every list of size \(n\), for every \(n\), is within its own gap of its true risk.

It is convenient to write each share as a fraction of the budget: size 3’s share of 0.0034 is about 6.8% of 0.05. So write \(\delta_n = w(n) \cdot \delta\), where \(w(n)\) is the fraction of the budget that size \(n\) gets. The **weight function** \(w: \mathbb N \to [0, 1]\) takes a size \(n\) (the arrow reads “maps each element of \(\mathbb N\) to a number in”) and returns a number between 0 and 1, and the fractions must satisfy \(\sum_n w(n) \le 1\). A weight says how much you believe in a size before seeing any data. If there were only finitely many sizes, say \(N\) of them, you could give each the same weight \(1/N\): no preference at all. With infinitely many sizes that is impossible (equal positive weights would sum to infinity), so you must lean towards some sizes. Two standard choices are

\[ w(n) = \frac{6}{\pi^2 n^2} \qquad \text{and} \qquad w(n) = 2^{-n}. \]

Read the first from the inside out: \(n^2\) is the size squared, so the weight falls off with the square of the size; \(\pi \approx 3.1416\) is the circle constant, and \(6/\pi^2 \approx 0.608\) is just the scaling that makes the weights add up to exactly 1, because \(\sum_n 1/n^2 = 1 + 1/4 + 1/9 + \cdots = \pi^2/6\). In the second, a negative exponent means “one over”, \(2^{-n} = 1/2^n\), so each size gets half the weight of the one before, and \(1/2 + 1/4 + 1/8 + \cdots = 1\). In plain words: **the first gives size \(n\) a share proportional to \(1/n^2\), the second a share proportional to \(1/2^n\), and both spend exactly the whole budget.** For size 3, the first gives \(0.608 / 9 = 0.0675\), so a share \(0.0675 \times 0.05 = 0.0034\) of the budget; the second gives \(1/8 = 0.125\).

```python
import math

def w_pi(n):
    return 6 / (math.pi**2 * n**2)

def w_two(n):
    return 2.0 ** -n

for n in (1, 2, 3, 6):
    share = w_pi(n) * 0.05
    print(n, round(w_pi(n), 4), w_two(n), round(share, 5))
print(round(sum(w_pi(n) for n in range(1, 10**6)), 6))
print(sum(w_two(n) for n in range(1, 60)))
```

It prints `1 0.6079 0.5 0.0304`, `2 0.152 0.25 0.0076`, `3 0.0675 0.125 0.00338` and `6 0.0169 0.015625 0.00084`, then `0.999999` and `1.0`: a million terms of the first already sum to 1 within a millionth, and the second reaches 1 to machine precision. We’ll use the first from now on. It gives size 1 about 61% of the budget, size 3 about 6.8%, size 6 about 1.7%, and every size something; the second starves the long sizes faster, giving size 6 only 1.6% and size 20 less than a millionth of the budget.

[[diagram:budget]]

The price is visible. A list of size \(n\) now gets the gap computed with \(w(n)\delta\) instead of \(\delta\). Dividing by a smaller number makes \(2|\mathcal H_n|/(w(n)\delta)\) bigger by a factor \(1/w(n)\), and a logarithm turns a factor into an addition, so this adds \(\log(1/w(n))\) inside the square root. For size 3 that is \(\log(1/0.0675) = 2.70\), on top of the 8.75 we had, and the gap at 300 labels grows from 0.121 to 0.138. For long lists, whose share is tiny, that is extra evidence they must pay for. And now the learner has an obvious thing to do: pick the list whose **bound**, empirical risk plus its own gap, is smallest. That learner is called **structural risk minimization** (SRM).

Let’s run it at \(m = 300\) labels. The empirical risks below are the best in each class on one draw of 300 messages from our stream.

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def w(n):
    return 6 / (math.pi**2 * n**2)

def eps(n, m, delta):
    return math.sqrt(math.log(2 * size_H(n) / delta) / (2 * m))

LS = [0.37, 0.37, 0.1333, 0.1333, 0.1133, 0.1133]
for n, r in enumerate(LS, start=1):
    e = eps(n, 300, w(n) * 0.05)
    print(n, round(w(n), 4), r, round(e, 4), round(r + e, 4))
```

| size \(n\) | \(w(n)\) | best \(L_S\) | gap | bound |
|---|---|---|---|---|
| 1 | 0.6079 | 0.37 | 0.0998 | 0.4698 |
| 2 | 0.152 | 0.37 | 0.1213 | 0.4913 |
| 3 | 0.0675 | 0.1333 | 0.1381 | **0.2714** |
| 4 | 0.038 | 0.1333 | 0.1533 | 0.2866 |
| 5 | 0.0243 | 0.1133 | 0.1676 | 0.2809 |
| 6 | 0.0169 | 0.1133 | 0.1812 | 0.2945 |

(The snippet prints these rows: `eps` is the gap formula from before, called with the share `w(n) * 0.05` in place of the whole budget, and the last column is the best empirical risk plus that gap. The exact empirical risks are 111/300, 40/300 and 34/300, which is why the build-it challenge below reports 0.2715 for size 3.) Size 1 and 2 lists cannot fit the data. Size 5 fits it best, but its gap eats the advantage. Size 3 has the smallest bound, so SRM ships the three-test list from the opening: “unknown sender, junk; FREE, keep; link, junk; otherwise keep”. Its true risk is 0.125. With 1,000 labels, the same procedure ships the five-test list \(f\), true risk 0.100, because now the evidence pays for the extra length. With 30 labels it ships the size-1 list “everything is junk”, with a bound of 0.58: it is telling you honestly that thirty labels certify almost nothing.

That is your teammate’s wish, granted with a guarantee attached. Long lists are allowed. They just have to earn it.

[[diagram:srm-curve]]

## The formal statements

Let’s make all of this exact. From here on, \(\mathcal D\) is any distribution over \(\mathcal X \times \{0, 1\}\), pairs of a message and a label, as in lesson 3’s **agnostic** setting, where labels may be noisy and no hypothesis need be perfect. The loss is the **0-1 loss**: 1 for a wrong label, 0 for a right one, so a risk is a fraction of mistakes.

**Recap from earlier lessons.** First, in words. A class is agnostic PAC learnable when you can name, before seeing any data, how many labels make the learner’s true risk at most \(\varepsilon\) worse than the best list in the class, with probability at least \(1 - \delta\), whatever the stream. In symbols, a class \(\mathcal H\) is *agnostic PAC learnable* (lesson 3) if there are a learner \(A\) and a function \(m_{\mathcal H}: (0,1)^2 \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\) and every \(\mathcal D\), if \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ L_{\mathcal D}(A(S)) \le \min_{h' \in \mathcal H} L_{\mathcal D}(h') + \varepsilon. \]

Read the pieces. \(A(S)\) is the hypothesis the learner returns on the training sequence \(S\). \(S \sim \mathcal D^m\) reads “\(S\) is drawn from \(\mathcal D^m\)”: \(m\) examples, each drawn independently from \(\mathcal D\). \((0,1)\) is the set of numbers strictly between 0 and 1, and \((0,1)^2\) is a pair of them, so \(m_{\mathcal H}: (0,1)^2 \to \mathbb N\) is a rule that takes the two dials \(\varepsilon, \delta\) and returns a whole number of labels. On the right of the display, \(\min_{h' \in \mathcal H} L_{\mathcal D}(h')\) is the smallest true risk of any \(h'\) in the class, the best you could hope to find. In plain words: **with enough labels, fixed in advance, the learner lands within \(\varepsilon\) of the best hypothesis in the class, except with probability \(\delta\).** In \(\mathcal H_3\) on our stream the best list has true risk 0.125, so with \(\varepsilon = 0.1\) the returned list must have true risk at most 0.225.

```python
best_in_class, eps = 0.125, 0.1
for risk in (0.125, 0.175, 0.375):
    print(risk, risk <= best_in_class + eps)
```

It prints `0.125 True`, `0.175 True` and `0.375 False`: the lists ERM found at cap 3 on thirty labels would meet this target, and “everything is junk” would not.

A class has the *uniform convergence property* (lesson 4) when enough labels make every hypothesis’s empirical risk close to its true risk at the same time. In symbols, with function \(m^{\mathrm{UC}}_{\mathcal H}\): for every \(\varepsilon, \delta\) and \(\mathcal D\), a training sequence of \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples is \(\varepsilon\)-**representative** with probability at least \(1 - \delta\): every \(h \in \mathcal H\) has \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\). The superscript UC is a name tag, “uniform convergence”, not a power. A finite class has it with \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le \lceil \log(2|\mathcal H|/\delta)/(2\varepsilon^2) \rceil\). That is the gap formula from “Thirty labels, six sizes” turned around: set the gap equal to \(\varepsilon\) and solve for \(m\). The brackets \(\lceil \cdot \rceil\) round up to the next whole number, since you cannot label part of a message. In plain words: **a finite class needs about \(\log(2|\mathcal H|/\delta)\) divided by \(2\varepsilon^2\) labels before every member is within \(\varepsilon\) of its true risk.** For \(\mathcal H_3\), \(\varepsilon = 0.05\) and \(\delta = 0.05\): \(\log 6{,}320 = 8.751\), divided by \(2 \times 0.0025 = 0.005\), is 1,750.2, so 1,751 labels.

```python
import math

size, eps, delta = 158, 0.05, 0.05
inside = math.log(2 * size / delta)
print(round(inside, 3), math.ceil(inside / (2 * eps**2)))
```

It prints `8.751 1751`. And the *fundamental theorem* (lesson 6) says that for binary classification with the 0-1 loss, uniform convergence, agnostic PAC learnability, PAC learnability and finite VC dimension are all equivalent.

**Competing with one hypothesis.** Say that a hypothesis \(h\) is \((\varepsilon, \delta)\)-**competitive** with another hypothesis \(h'\) if, with probability higher than \(1 - \delta\), \(L_{\mathcal D}(h) \le L_{\mathcal D}(h') + \varepsilon\). With \(\varepsilon = 0.1\), a list is competitive with \(f\) when, except with probability \(\delta\), its true risk is at most \(0.1 + 0.1 = 0.2\). Agnostic PAC learning asks the returned hypothesis to be competitive with every \(h' \in \mathcal H\) at once, at one sample size. The relaxation is to let the sample size depend on which \(h'\) you compete with: to be competitive with “everything is junk” you might need 500 labels, and to be competitive with \(f\) you might need 3,400.

**Definition (nonuniform learnability).** A hypothesis class \(\mathcal H\) is **nonuniformly learnable** if there exist a learner \(A\) and a function \(m^{\mathrm{NUL}}_{\mathcal H}: (0,1)^2 \times \mathcal H \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\) and every \(h \in \mathcal H\), if \(m \ge m^{\mathrm{NUL}}_{\mathcal H}(\varepsilon, \delta, h)\), then for every distribution \(\mathcal D\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon. \]

The new piece is the signature \(m^{\mathrm{NUL}}_{\mathcal H}: (0,1)^2 \times \mathcal H \to \mathbb N\). The cross \(\times\) means “together with”: the function takes the two dials and also a hypothesis, and returns a number of labels. NUL is a name tag for “nonuniform learnability”. The display itself is the competitive inequality with \(A(S)\) on the left. In plain words: **for each hypothesis \(h\) there is a number of labels, possibly different for each \(h\), beyond which the learner is within \(\varepsilon\) of \(h\) on every stream, except with probability \(\delta\).** With the made-up numbers above and 2,500 labels in hand, the learner is guaranteed to compete with the short lists but not yet with \(f\).

```python
m_nul = {"everything junk": 500, "three tests": 2300,
         "five tests": 3400}
m = 2500
for h, need in m_nul.items():
    print(h, need, m >= need)
```

It prints `everything junk 500 True`, `three tests 2300 True` and `five tests 3400 False`. The dictionary is the function \(m^{\mathrm{NUL}}\) at one fixed \(\varepsilon, \delta\); the SRM theorem below replaces these made-up entries with real ones.

Compare the order of the words. In agnostic PAC learning, one \(m_{\mathcal H}(\varepsilon, \delta)\) serves every hypothesis. Here, \(m^{\mathrm{NUL}}\) may grow with \(h\): to compete with a long list, you may need many more examples than to compete with a short one. It still may not depend on \(\mathcal D\).

Agnostic PAC learnability implies nonuniform learnability: the best in the class is never worse than any particular \(h\), so the agnostic guarantee gives \(L_{\mathcal D}(A(S)) \le \min_{h'} L_{\mathcal D}(h') + \varepsilon \le L_{\mathcal D}(h) + \varepsilon\) for every \(h\), so \(m^{\mathrm{NUL}}_{\mathcal H}(\varepsilon, \delta, h) = m_{\mathcal H}(\varepsilon, \delta)\) works. The converse fails, as the decision lists will show.

**The gap function.** The uniform convergence function answers “how many labels for this accuracy?”. SRM needs the question turned around: “with the labels I have, how small an accuracy can I certify?”. With 1,751 labels, \(\mathcal H_3\) at \(\delta = 0.05\) certifies 0.05, as we just computed; with 300 it certifies only 0.121. Let \(\mathcal H = \bigcup_{n \in \mathbb N} \mathcal H_n\), where each \(\mathcal H_n\) has the uniform convergence property with \(m^{\mathrm{UC}}_{\mathcal H_n}\). Define

\[ \varepsilon_n(m, \delta) = \min\{\varepsilon \in (0, 1) : m^{\mathrm{UC}}_{\mathcal H_n}(\varepsilon, \delta) \le m\}. \]

Read it from the inside out. \(m^{\mathrm{UC}}_{\mathcal H_n}(\varepsilon, \delta) \le m\) asks one yes/no question about an accuracy \(\varepsilon\): do the \(m\) labels you have suffice for it? The braces \(\{\varepsilon \in (0, 1) : \dots\}\) collect every accuracy between 0 and 1 for which the answer is yes; read the colon as “such that”. Then \(\min\) takes the smallest of them. (Strictly, the smallest might only be approached, never reached; then read \(\min\) as the largest number below every member of the set, which changes nothing here.) In plain words: **\(\varepsilon_n(m, \delta)\) is the smallest gap that \(m\) examples can certify for \(\mathcal H_n\) at confidence \(\delta\).** The snippet tries every accuracy on a grid of steps of 0.0001 for \(\mathcal H_3\) and keeps the smallest one that \(m\) labels pay for, next to the square-root formula.

```python
import math

def m_uc(size, eps, delta):
    return math.ceil(math.log(2 * size / delta) / (2 * eps**2))

grid = [i / 10000 for i in range(1, 10000)]
for m in (300, 1751):
    ok = [e for e in grid if m_uc(158, e, 0.05) <= m]
    root = math.sqrt(math.log(2 * 158 / 0.05) / (2 * m))
    print(m, min(ok), round(root, 4))
```

It prints `300 0.1208 0.1208` and `1751 0.05 0.05`: the search and the formula agree. Two facts follow from the definition, and they are all the proofs below use:

1. For every \(m\), \(\delta\) and \(\mathcal D\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H_n\) has \(|L_{\mathcal D}(h) - L_S(h)| \le \varepsilon_n(m, \delta)\).
2. If \(m \ge m^{\mathrm{UC}}_{\mathcal H_n}(\varepsilon, \delta)\), then \(\varepsilon_n(m, \delta) \le \varepsilon\).

For a finite \(\mathcal H_n\), the formula \(\sqrt{\log(2|\mathcal H_n|/\delta)/(2m)}\) from the previous section has both properties, with \(m^{\mathrm{UC}}\) the finite-class bound, so we may use it in place of \(\varepsilon_n\). That is what the snippets do. If a weight is 0, read \(\varepsilon_n(m, 0)\) as \(+\infty\): a class with no budget gets no guarantee of its own.

**Theorem (the SRM bound).** Let \(w: \mathbb N \to [0, 1]\) with \(\sum_{n=1}^{\infty} w(n) \le 1\), and let \(\mathcal H = \bigcup_n \mathcal H_n\) with each \(\mathcal H_n\) having the uniform convergence property. Then for every \(\delta \in (0, 1)\) and every distribution \(\mathcal D\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), the following holds simultaneously for every \(n \in \mathbb N\) and every \(h \in \mathcal H_n\):

\[ |L_{\mathcal D}(h) - L_S(h)| \le \varepsilon_n(m, w(n) \cdot \delta). \]

Consequently, with probability at least \(1 - \delta\),

\[ \forall h \in \mathcal H, \quad L_{\mathcal D}(h) \le L_S(h) + \min_{n : h \in \mathcal H_n} \varepsilon_n(m, w(n) \cdot \delta). \]

This is the confidence budget from “A confidence budget, shared out”, now in symbols. In the first display, \(\varepsilon_n(m, w(n) \cdot \delta)\) is the gap of class \(\mathcal H_n\) computed with its share \(w(n)\delta\) instead of the whole budget. In the second, the upside-down A, \(\forall\), reads “for every”, and \(\min_{n : h \in \mathcal H_n}\) reads “the smallest, over every \(n\) such that \(h\) is in \(\mathcal H_n\)”: a list of size 3 is also in \(\mathcal H_4\), \(\mathcal H_5\) and so on, each class offers it a gap, and it may use the best one. In plain words: **except with probability \(\delta\), every list’s true risk is at most its empirical risk plus the smallest gap among the classes that contain it.** For the three-test list at 300 labels, with empirical risk 40/300, the classes \(\mathcal H_3, \dots, \mathcal H_6\) offer gaps 0.1381, 0.1533, 0.1676 and 0.1812, so the bound is \(0.1333 + 0.1381 = 0.2715\).

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def gap(n, m, delta):
    budget = 6 / (math.pi**2 * n**2) * delta
    return math.sqrt(math.log(2 * size_H(n) / budget) / (2 * m))

LS = 40 / 300
gaps = {n: gap(n, 300, 0.05) for n in range(3, 7)}
print({n: round(g, 4) for n, g in gaps.items()})
print(round(LS + min(gaps.values()), 4))
```

It prints `{3: 0.1381, 4: 0.1533, 5: 0.1676, 6: 0.1812}` and `0.2715`. The dictionary runs over \(n \ge 3\) because those are the classes containing the list, and `min` is the \(\min_{n : h \in \mathcal H_n}\).

*Proof.* Name the bad events. For each \(n\), let \(B_n\) be the set of training sequences \(S\) on which some \(h \in \mathcal H_n\) has \(|L_{\mathcal D}(h) - L_S(h)| > \varepsilon_n(m, w(n)\delta)\).

First, each bad event is unlikely. Write \(\mathcal D^m(B)\) for the probability that a training sequence drawn from \(\mathcal D^m\) lands in the set \(B\). Apply fact 1 to the class \(\mathcal H_n\) with confidence \(\delta_n = w(n)\delta\): it says exactly that \(\mathcal D^m(B_n) \le w(n)\delta\). (When \(w(n) = 0\), \(B_n\) is empty by our convention, and the inequality holds trivially.)

Second, the union of the bad events is unlikely. For any finite \(N\), the union bound from lesson 2 gives

\[ \mathcal D^m\Big(\bigcup_{n \le N} B_n\Big) \le \sum_{n \le N} \mathcal D^m(B_n) \le \delta \sum_{n \le N} w(n) \le \delta. \]

Read it left to right: the chance that at least one of the first \(N\) bad events happens, then the sum of their chances (the union bound), then each chance replaced by its share, then the shares, which never exceed the budget. In plain words: **the first \(N\) classes together fail with probability at most the shares they were given.** With \(\delta = 0.05\), the first class alone may use 0.0304 of it, the first six 0.0453, and the first thousand 0.05 to four decimals:

```python
import math

delta = 0.05
for N in (1, 6, 1000):
    w = sum(6 / (math.pi**2 * n**2) for n in range(1, N + 1))
    print(N, round(delta * w, 4))
```

It prints `1 0.0304`, `6 0.0453` and `1000 0.05`. The sets \(\bigcup_{n \le N} B_n\) grow with \(N\) and their union over all \(N\) is \(\bigcup_n B_n\). A probability measure is continuous along increasing sequences of events, so \(\mathcal D^m(\bigcup_n B_n)\) is the limit of these probabilities as \(N \to \infty\), and a limit of numbers that are all at most \(\delta\) is at most \(\delta\).

Third, outside the union, every inequality holds. If \(S\) is in none of the \(B_n\), then for every \(n\) and every \(h \in \mathcal H_n\) the gap is at most \(\varepsilon_n(m, w(n)\delta)\). That event has probability at least \(1 - \delta\), which is the first claim. For the second, a hypothesis \(h\) may sit in several classes, and each of them gives a valid upper bound on \(L_{\mathcal D}(h)\); the smallest of them is still valid. \(\blacksquare\)

Write \(n(h) = \min\{n : h \in \mathcal H_n\}\). For our lists, \(n(h)\) is the size. Since the minimum over \(n\) is at most the term at \(n(h)\), the theorem gives, with probability at least \(1 - \delta\), for all \(h \in \mathcal H\),

\[ L_{\mathcal D}(h) \le L_S(h) + \varepsilon_{n(h)}\big(m, w(n(h)) \cdot \delta\big). \]

This reuses the notation just walked through, with the class picked by the list’s own size: **true risk at most empirical risk plus the gap of the list’s own class.** For our lists the gaps grow with \(n\), so the smallest gap is the one at \(n(h)\) and nothing is lost: the three-test list still gets \(0.1333 + 0.1381 = 0.2715\), the first entry of the dictionary above.

**The SRM rule** returns a hypothesis that minimizes this bound:

\[ \mathrm{SRM}(S) \in \operatorname*{argmin}_{h \in \mathcal H} \Big[ L_S(h) + \varepsilon_{n(h)}\big(m, w(n(h)) \cdot \delta\big) \Big]. \]

The square brackets hold the bound of one list. \(\operatorname{argmin}_{h \in \mathcal H}\) reads “the hypotheses \(h\) in \(\mathcal H\) at which the bracket is smallest”: not the smallest value itself, but the lists that achieve it. Several lists can tie, so the argmin is a set, and \(\in\) says SRM returns one of its members. In plain words: **SRM returns a list whose empirical risk plus its own gap is as small as possible.** At 300 labels that is the three-test list, with bound 0.2714 in the SRM table of “A confidence budget, shared out”, whose snippet computes exactly this bracket for the best list of each size.

Unlike ERM, it does not just chase low empirical risk. It trades some of it for a smaller gap, a bias towards classes it can certify. Recall the split from lesson 5: the true risk of the returned list is the **approximation error**, the best true risk in the class it came from, plus the **estimation error**, how far it falls short of that best. SRM gives up a little approximation error for the sake of a smaller estimation error.

**Theorem (SRM is a nonuniform learner).** Let \(\mathcal H = \bigcup_n \mathcal H_n\) with each \(\mathcal H_n\) having the uniform convergence property, and let \(w(n) = 6/(\pi^2 n^2)\). Then \(\mathcal H\) is nonuniformly learnable by the SRM rule, with

\[ m^{\mathrm{NUL}}_{\mathcal H}(\varepsilon, \delta, h) \le m^{\mathrm{UC}}_{\mathcal H_{n(h)}}\Big(\frac{\varepsilon}{2}, \frac{6\delta}{(\pi n(h))^2}\Big). \]

Read it from the inside out. \(6\delta/(\pi n(h))^2\) is \(w(n(h)) \cdot \delta\), the share of the budget given to the list’s size, written out. \(\varepsilon/2\) is half the accuracy, because the proof will pay the gap twice. And \(m^{\mathrm{UC}}_{\mathcal H_{n(h)}}\) is the uniform convergence sample size of the list’s own class, run at that half accuracy and that share. In plain words: **to compete with a list of size \(n\), SRM needs only as many labels as uniform convergence on \(\mathcal H_n\) needs at half the accuracy and the size’s share of the confidence.** For the three-test list, \(\varepsilon = 0.1\) and \(\delta = 0.05\): the share is \(0.05 \times 0.0675 = 0.00338\), \(\log(2 \times 158 / 0.00338) = 11.45\), and divided by \(2 \times 0.05^2 = 0.005\) that is 2,290 labels.

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def m_nul(eps, delta, n):
    budget = 6 * delta / (math.pi * n) ** 2
    top = math.log(2 * size_H(n) / budget)
    return math.ceil(top / (2 * (eps / 2) ** 2))

print([m_nul(0.1, 0.05, n) for n in (1, 3, 5)])
print([m_nul(0.1, 0.1, n) for n in range(1, 7)])
```

The first line prints `[1196, 2290, 3371]`: these are the real entries for the made-up dictionary after the definition, for sizes 1, 3 and 5 (“everything is junk”, the three-test list, \(f\)). The second line, at \(\varepsilon = \delta = 0.1\), prints `[1058, 1628, 2151, 2682, 3232, 3802]`; we will need it in the characterization proof.

*Proof.* Fix \(\varepsilon\), \(\delta\), a hypothesis \(h \in \mathcal H\) to compete with, and a distribution \(\mathcal D\). Let \(m \ge m^{\mathrm{UC}}_{\mathcal H_{n(h)}}(\varepsilon/2, w(n(h))\delta)\), and let \(A\) be the SRM rule. The weights sum to exactly 1, so the SRM bound theorem applies: let \(G\) be its good event, on which every \(h' \in \mathcal H\) has \(|L_{\mathcal D}(h') - L_S(h')| \le \varepsilon_{n(h')}(m, w(n(h'))\delta)\). It has probability at least \(1 - \delta\). We now show that on \(G\), \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon\), in four steps.

Step 1, the returned hypothesis. On \(G\), the inequality holds for every \(h'\) at once, so it holds in particular for \(h' = A(S)\), even though \(A(S)\) was chosen after looking at \(S\). This is the whole reason we need a bound that is simultaneous over the class: a bound for one fixed hypothesis would say nothing about a hypothesis picked using the data. So \(L_{\mathcal D}(A(S)) \le L_S(A(S)) + \varepsilon_{n(A(S))}(m, w(n(A(S)))\delta)\).

Step 2, the choice. SRM minimizes exactly that right-hand side over \(\mathcal H\), so its value at \(A(S)\) is at most its value at \(h\): \(L_{\mathcal D}(A(S)) \le L_S(h) + \varepsilon_{n(h)}(m, w(n(h))\delta)\).

Step 3, back to the true risk of \(h\). On \(G\), the inequality also holds for \(h\) itself, in the other direction: \(L_S(h) \le L_{\mathcal D}(h) + \varepsilon_{n(h)}(m, w(n(h))\delta)\). So \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + 2\,\varepsilon_{n(h)}(m, w(n(h))\delta)\).

Step 4, the sample size. Because \(m \ge m^{\mathrm{UC}}_{\mathcal H_{n(h)}}(\varepsilon/2, w(n(h))\delta)\), fact 2 gives \(\varepsilon_{n(h)}(m, w(n(h))\delta) \le \varepsilon/2\). Hence \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon\) on \(G\), with probability at least \(1 - \delta\). The sample size we required depends on \(h\) only through \(n(h)\), and not on \(\mathcal D\), which is the definition. \(\blacksquare\)

In particular: **every countable union of classes with the uniform convergence property is nonuniformly learnable.** It is the nonuniform version of lesson 4’s “uniform convergence is enough for learning”.

**Theorem (characterization).** A class \(\mathcal H\) of binary classifiers is nonuniformly learnable if and only if it is a countable union of agnostic PAC learnable classes.

*Proof, if.* Suppose \(\mathcal H = \bigcup_n \mathcal H_n\) with each \(\mathcal H_n\) agnostic PAC learnable. By the fundamental theorem, each \(\mathcal H_n\) has the uniform convergence property. By the previous theorem, SRM learns \(\mathcal H\) nonuniformly.

*Proof, only if.* Suppose \(A\) learns \(\mathcal H\) nonuniformly with \(m^{\mathrm{NUL}}_{\mathcal H}\). For each \(n\), let

\[ \mathcal H_n = \{h \in \mathcal H : m^{\mathrm{NUL}}_{\mathcal H}(1/10, 1/10, h) \le n\}. \]

In words: **\(\mathcal H_n\) collects the hypotheses that the learner can compete with, to within 1/10 and except with probability 1/10, using at most \(n\) examples.** These are new classes, sorted by how many labels they need rather than by list size. If the learner were SRM on our lists, the second line printed above says sizes 1 and 2 need 1,058 and 1,628 labels, so \(\mathcal H_{2000}\) would hold every list of size at most 2, and \(\mathcal H_{4000}\) every list of size at most 6.

Every \(h\) has some finite value of \(m^{\mathrm{NUL}}\), so it lands in some \(\mathcal H_n\), and \(\mathcal H = \bigcup_n \mathcal H_n\). It remains to show each \(\mathcal H_n\) is agnostic PAC learnable, and by the fundamental theorem it is enough to show its VC dimension is finite. We show it is less than \(2n\).

Suppose not: then \(\mathcal H_n\) shatters some set \(C\) of \(2n\) messages (a subset of a shattered set is shattered). Recall the core of lesson 5’s No-Free-Lunch theorem: for any learner that sees \(m\) examples and any set \(C\) of size \(2m\), there is a labelling \(f_i: C \to \{0, 1\}\) such that, for the distribution \(\mathcal D_i\) that picks a message of \(C\) uniformly and labels it by \(f_i\),

\[ \mathbb E_{S \sim \mathcal D_i^m}\big[L_{\mathcal D_i}(A(S))\big] \ge 1/4. \]

Here \(\mathbb E\) is the **expected value**, the average over all the random training sequences, and its subscript \(S \sim \mathcal D_i^m\) says what is random: \(m\) examples drawn from \(\mathcal D_i\). The subscript \(i\) only numbers the possible labellings of \(C\). In plain words: **for any learner there is a labelling of \(2m\) messages on which, averaged over its \(m\) random examples, it is wrong at least a quarter of the time.** Take the memorizer, guessing keep on anything unseen, with \(m = 5\) examples from 10 messages. It misses on average \(0.9^5 \approx 59\%\) of the messages, and on a random labelling half of those are junk, so its average risk is about 0.295. Averaged over random labellings it is at least 1/4, so some labelling is at least that bad.

```python
import random

rng = random.Random(0)
n, trials, total = 5, 20000, 0.0
C = range(2 * n)
for _ in range(trials):
    f = [rng.randint(0, 1) for _ in C]
    seen = {rng.choice(C) for _ in range(n)}
    wrong = sum(f[c] != 0 for c in C if c not in seen)
    total += wrong / len(C)
print(round(total / trials, 4), 1 / 10 * 1 + 1 * 1 / 10)
```

It prints `0.2967 0.2`. The first number is the average risk, above 1/4 as promised; the second is the upper bound the next step derives, and the two cannot both hold.

Apply it to \(A\) with \(m = n\). Because \(\mathcal H_n\) shatters \(C\), some \(h \in \mathcal H_n\) agrees with \(f_i\) on all of \(C\), so \(L_{\mathcal D_i}(h) = 0\). Because \(h \in \mathcal H_n\), \(m = n \ge m^{\mathrm{NUL}}(1/10, 1/10, h)\), so with probability at least \(9/10\), \(L_{\mathcal D_i}(A(S)) \le 1/10\). Split the expectation over that event and its complement, and use that a risk is never above 1:

\[ \mathbb E\big[L_{\mathcal D_i}(A(S))\big] \le \tfrac{1}{10} \cdot 1 + 1 \cdot \tfrac{1}{10} = \tfrac{1}{5} < \tfrac{1}{4}. \]

Each product is a probability times the worst risk on that part: on the good event, probability at most 1, risk at most 1/10; on the bad event, probability at most 1/10, risk at most 1. In plain words: **a learner that is almost always nearly perfect cannot average a quarter of mistakes.** That is the `0.2` the snippet printed. That contradicts No-Free-Lunch. So the VC dimension of \(\mathcal H_n\), written \(\mathrm{VCdim}(\mathcal H_n)\), is less than \(2n\), and \(\mathcal H_n\) is agnostic PAC learnable. \(\blacksquare\)

**Decision lists, settled.** Our \(\mathcal H\) is a countable union of finite classes, each with the uniform convergence property, so it is nonuniformly learnable by SRM. Its VC dimension is infinite, so it is not PAC learnable. Nonuniform learnability is a strict relaxation of agnostic PAC learnability, and your teammate’s class sits exactly in the gap. (The same happens on the real line with polynomial threshold classifiers, the sign of a polynomial: degree \(n\) gives VC dimension \(n + 1\), and all degrees together give infinite VC dimension.)

**The price of not knowing the size.** If someone had told you in advance that the right list has size 3, you would run ERM on \(\mathcal H_3\) and need \(m^{\mathrm{UC}}_{\mathcal H_3}(\varepsilon/2, \delta)\) examples to get within \(\varepsilon\) of the best list there. SRM, not knowing, needs \(m^{\mathrm{UC}}_{\mathcal H_3}(\varepsilon/2, w(3)\delta)\). For finite classes the difference is

\[ \frac{\log(2|\mathcal H_n|/(w(n)\delta))}{2(\varepsilon/2)^2} - \frac{\log(2|\mathcal H_n|/\delta)}{2(\varepsilon/2)^2} = \frac{2 \log(1/w(n))}{\varepsilon^2} = \frac{2\log(\pi^2 n^2/6)}{\varepsilon^2}, \]

which grows only like the logarithm of the size. Read it left to right. Each fraction is the finite-class sample size \(\log(2|\mathcal H_n|/\delta)/(2\varepsilon^2)\) at accuracy \(\varepsilon/2\): the first with the share \(w(n)\delta\), the second with the whole budget. Their numerators differ only inside the logarithm, by the factor \(1/w(n)\), and the log of a product is a sum of logs, so the difference is \(\log(1/w(n))\) over \(2(\varepsilon/2)^2 = \varepsilon^2/2\). The last step writes \(1/w(n) = \pi^2 n^2/6\). In plain words: **not knowing the size costs \(2\log(1/w(n))/\varepsilon^2\) extra labels, the price of the size’s share of the budget.** For \(n = 3\) and \(\varepsilon = 0.1\): \(\log(9\pi^2/6) = 2.695\), times 2, divided by 0.01, is 539 labels. The snippet below computes both sample sizes separately, with the same `size_H` as before; their difference at size 3, \(2{,}290 - 1{,}751\), is that 539. With \(\varepsilon = 0.1\) and \(\delta = 0.05\):

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def m_uc(n, eps, delta):
    return math.log(2 * size_H(n) / delta) / (2 * eps**2)

eps, delta = 0.1, 0.05
for n in (3, 5, 20):
    alone = m_uc(n, eps / 2, delta)
    w = 6 / (math.pi**2 * n**2)
    inside = m_uc(n, eps / 2, w * delta)
    print(n, math.ceil(alone), math.ceil(inside))
```

It prints `3 1751 2290`, `5 2628 3371` and `20 12217 13514`. Knowing the size in advance would save 539 labels at size 3. That is the value of prior knowledge about the order of the classes, made into a number. For classes of VC dimension \(n\) the same calculation, with lesson 6’s quantitative bound, gives a difference of order \(\log(2n)/\varepsilon^2\).

**A limit that remains.** You might now hope that every class is nonuniformly learnable. It is not. Over any infinite domain, the class of all functions to \(\{0, 1\}\) is not a countable union of classes of finite VC dimension, so by the characterization it is not nonuniformly learnable. That is No-Free-Lunch again, for the relaxed notion. (For each single function there is a trivial learner, ERM over the class containing only that function; there is no learner for all of them.) The proof is a diagonal construction we won’t do here.

## Lists by the bit: minimum description length

SRM spreads the budget over classes. There is a sharper version: spread it over individual hypotheses. Our \(\mathcal H\) is countable (each list is a finite object), so we can list its members as \(h_1, h_2, \dots\) and write it as a union of one-member classes, \(\mathcal H = \bigcup_n \{h_n\}\). Then each hypothesis gets its own weight \(w(h)\), say 1/128 for the list “unknown sender, junk; otherwise keep”.

A single fixed hypothesis has uniform convergence with no union bound at all. Its mistakes are \(\mathbb 1[h(x_i) \ne y_i]\), where the **indicator** \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false. These are i.i.d. numbers in \([0, 1]\) (the numbers from 0 to 1, ends included) with mean \(L_{\mathcal D}(h)\), so by Hoeffding’s inequality (lesson 4), \(\mathbb P[|L_S(h) - L_{\mathcal D}(h)| > \varepsilon] \le 2e^{-2m\varepsilon^2}\). Here \(\mathbb P\) is the probability over the training sequence, and \(e^{-2m\varepsilon^2}\) is \(e \approx 2.718\) raised to the power \(-2m\varepsilon^2\), which shrinks very fast as \(m\) grows. In plain words: **the chance that one fixed list’s score on \(m\) labels is off by more than \(\varepsilon\) falls exponentially in \(m\varepsilon^2\).** At \(m = 300\) and \(\varepsilon = 0.1\) it is at most \(2e^{-6} = 0.005\). Setting this to \(\delta\) and solving for \(\varepsilon\) gives the gap \(\sqrt{\log(2/\delta)/(2m)}\), which at \(\delta = 0.05\) is \(\sqrt{3.689/600} = 0.078\):

```python
import math

m, eps, delta = 300, 0.1, 0.05
print(round(2 * math.exp(-2 * m * eps**2), 5))
print(round(math.sqrt(math.log(2 / delta) / (2 * m)), 4))
```

It prints `0.00496` and `0.0784`. Giving the list \(h\) the share \(w(h)\delta\) instead of \(\delta\) turns \(\log(2/\delta)\) into \(\log(2/\delta) - \log w(h)\), and the SRM rule becomes

\[ \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{-\log w(h) + \log(2/\delta)}{2m}} \bigg]. \]

Read it from the inside out. A weight is below 1, so its logarithm is negative and \(-\log w(h) = \log(1/w(h))\) is a positive cost: the smaller the weight, the bigger the cost. Add the confidence cost \(\log(2/\delta)\), divide by \(2m\), take the square root, and add the empirical risk; the argmin picks the lists where that total is smallest. In plain words: **return a list with the smallest empirical risk plus a gap that grows as its weight shrinks.** With weight 1/128 the cost is \(\log 128 = 4.852\); at 100 labels that list has empirical risk 0.17, so its bracket is \(0.17 + \sqrt{(4.852 + 3.689)/200} = 0.3767\).

```python
import math

LS, m, delta = 0.17, 100, 0.05
w = 2.0 ** -7
inside = -math.log(w) + math.log(2 / delta)
print(round(-math.log(w), 3), round(math.log(2 / delta), 3))
print(round(LS + math.sqrt(inside / (2 * m)), 4))
```

It prints `4.852 3.689` and `0.3767`.

Now the prior knowledge is entirely in the weights. Where do good weights come from? From writing hypotheses down.

**Description languages.** Fix the alphabet \(\Sigma = \{0, 1\}\) (the capital Greek sigma names the set of symbols we may write). A **string** is a finite sequence of symbols, such as \(\sigma = (0, 1, 1, 1, 0)\), of length \(|\sigma| = 5\): around a string, the bars mean its length, a third meaning after “how many” and “absolute value”. The set of all strings, of every finite length, is \(\Sigma^*\), the star meaning “any number of symbols”. A **description language** for \(\mathcal H\) is a function \(d: \mathcal H \to \Sigma^*\) that maps each hypothesis \(h\) to a string \(d(h)\), its description. Its length is written \(|h|\), the **description length** of \(h\). The language is **prefix-free** if for every two distinct \(h, h'\), \(d(h)\) is not a prefix of \(d(h')\): no description is exactly the first \(|h|\) symbols of a longer one.

Here is a prefix-free language for decision lists. First, a code for feature numbers: write \(j\) in binary, with \(b\) digits, and put \(b - 1\) zeros in front. So 1 is `1`, 2 is `010`, 3 is `011`, 4 is `00100`, 5 is `00101`, and feature \(j\) costs \(2\lfloor \log_2 j \rfloor + 1\) bits. Read that inside out: \(\log_2 j\) is the power of 2 that gives \(j\), the brackets \(\lfloor \cdot \rfloor\) round it down to a whole number, and that whole number plus 1 is how many binary digits \(j\) has; doubling it and adding 1 counts the digits plus the leading zeros. In plain words: **a feature number costs twice its binary digit count, minus one.** For \(j = 5\), \(\log_2 5 = 2.32\), rounded down to 2, so \(2 \times 2 + 1 = 5\) bits, and indeed `00101` has five. A reader can always tell where a feature number ends: count the leading zeros, then read one more digit than that. Then a list is written test by test: a `1` meaning “a test follows”, the feature’s code, and the test’s label bit; and at the end a `0` meaning “no more tests”, followed by the default label bit.

The list “unknown sender, junk; otherwise keep”, \(((3, 1); 0)\), is written `1 011 1 0 0`, which is 7 bits. “Everything is junk”, \((();1)\), is `01`, which is 2 bits. The three-test list from the opening takes 15 bits and the five-test list \(f\) takes 29.

```python
import math

def gamma(j):
    b = bin(j)[2:]
    return "0" * (len(b) - 1) + b

def code(tests, default):
    body = "".join("1" + gamma(j) + str(b) for j, b in tests)
    return body + "0" + str(default)

for j in range(1, 6):
    print(j, gamma(j), 2 * math.floor(math.log2(j)) + 1)
f = [(3, 1), (1, 0), (2, 1), (5, 0), (4, 1)]
print(code([(3, 1)], 0), code([], 1))
print(len(code(f[:3], 0)), len(code(f, 0)))
```

It prints the five feature codes with lengths 1, 3, 3, 5, 5, matching the formula, then `1011100 01` and `15 29`. `bin(j)[2:]` is \(j\) in binary, and `gamma` puts the zeros in front.

Why is this prefix-free? Read any codeword from the left, the way a decoder would. Each flag bit says whether a test follows, each feature code announces its own length, and after the end flag exactly one bit remains. So the decoder knows where the codeword ends from the symbols alone. If one codeword were a proper prefix of another, decoding the longer one would reach that point and stop, so the longer one could have nothing after it, which is a contradiction.

Now give each codeword the weight 1 over 2 to the power of its length: “everything is junk”, 2 bits, gets 1/4; the 7-bit list gets 1/128. Can a prefix-free code overspend, so that these weights add to more than 1? Try the small prefix-free set `0`, `10`, `110`, `111`: \(1/2 + 1/4 + 1/8 + 1/8 = 1\), exactly the whole budget. Shorten any word and you create a prefix. That is the general rule. (In this lemma \(S\) names a set of strings, not a training sequence.)

**Lemma (Kraft inequality).** If \(S \subseteq \{0, 1\}^*\) is a prefix-free set of strings, then

\[ \sum_{\sigma \in S} \frac{1}{2^{|\sigma|}} \le 1. \]

Read it from the inside out: \(|\sigma|\) is the length of the string \(\sigma\), \(1/2^{|\sigma|}\) halves once per symbol, and \(\sum_{\sigma \in S}\) adds that weight over every string in the set. In plain words: **in a prefix-free code, the weights 1/2 per symbol, multiplied along each codeword, add up to at most 1.** Drop prefix-freeness and it fails: `0`, `01`, `1` has `0` as a prefix of `01`, and its sum is \(1/2 + 1/4 + 1/2 = 1.25\).

```python
S = ["0", "10", "110", "111"]
clash = any(a != b and b.startswith(a) for a in S for b in S)
print(clash, sum(1 / 2 ** len(s) for s in S))
T = ["0", "01", "1"]
print(sum(1 / 2 ** len(s) for s in T))
```

It prints `False 1.0` and `1.25`.

*Proof.* Start with a finite \(S\), and let \(L\) be the length of its longest string. Toss a fair coin \(L\) times, with faces 0 and 1. For each \(\sigma \in S\), let \(E_\sigma\) be the event “the first \(|\sigma|\) tosses spell out \(\sigma\)”. The tosses are independent and each matches its symbol with probability 1/2, so \(\mathbb P[E_\sigma] = 2^{-|\sigma|}\).

Now the key step. For two distinct \(\sigma, \sigma' \in S\), the events \(E_\sigma\) and \(E_{\sigma'}\) cannot both happen: if they did, both strings would be beginnings of the same toss sequence, so the shorter one would be a prefix of the longer one (or they would be equal), which prefix-freeness forbids. So the events are pairwise disjoint, and the probability of their union is the sum of their probabilities. A probability is at most 1:

\[ \sum_{\sigma \in S} 2^{-|\sigma|} = \mathbb P\Big[\bigcup_{\sigma \in S} E_\sigma\Big] \le 1. \]

This is the lemma’s sum, written with a negative exponent, equal to the chance that some codeword is spelled out by the tosses. For `0`, `10`, `110`, `111`, the toss sequences starting 0, 10, 110 and 111 cover every outcome exactly once, so the chance is 1.

For an infinite \(S\), every finite subset is prefix-free too, so every partial sum of the series is at most 1, and so is the series. \(\blacksquare\)

You can think of the coin as a process: toss until the outcome so far is in \(S\), then stop. Prefix-freeness means the process stops at \(\sigma\) exactly when the tosses follow \(\sigma\), so \(2^{-|\sigma|}\) is the probability of stopping at \(\sigma\), and stopping probabilities add up to at most 1.

[[diagram:kraft-tree]]

Let’s check it on the list code: every list with at most two tests over features 1 to 8. The `gamma` and `code` functions are the ones from the code-length snippet above.

```python
import itertools

def gamma(j):
    b = bin(j)[2:]
    return "0" * (len(b) - 1) + b

def code(tests, default):
    body = "".join("1" + gamma(j) + str(b) for j, b in tests)
    return body + "0" + str(default)

T = [(j, b) for j in range(1, 9) for b in (0, 1)]
C = [code(t, d) for k in range(3)
     for t in itertools.product(T, repeat=k) for d in (0, 1)]
clash = any(a != b and b.startswith(a) for a in C for b in C)
print(len(C), clash, sum(2 ** -len(c) for c in C))
```

It prints `546 False 0.8181228637695312`: 546 codewords, no codeword is a prefix of another, and their Kraft sum is about 0.818. Include more features and more tests and the sum creeps up towards 1, and over every list it is exactly 1. Here is why, in three sums. First, the feature code: \(\sum_j 2^{-(2\lfloor \log_2 j \rfloor + 1)} = \sum_{k \ge 0} 2^k \cdot 2^{-(2k+1)} = 1\). The left sum runs over every feature \(j\); the middle one groups the features by \(k + 1\), their number of binary digits. There are \(2^k\) numbers with \(k + 1\) digits (feature 1 alone, then 2 and 3, then 4 to 7), each costing \(2k + 1\) bits, so each group adds \(2^k / 2^{2k+1} = 1/2^{k+1}\): \(1/2 + 1/4 + 1/8 + \cdots = 1\). Second, one test, with its flag bit, its feature code and its label bit, contributes a factor \(2 \cdot \frac14 \cdot 1 = \frac12\): two choices of label, \(\frac14\) for the two single bits, and 1 for the feature code. The end, a flag and a default bit, contributes \(2 \cdot \frac14 = \frac12\) too. Third, a list with \(k\) tests gets \((1/2)^k \cdot \frac12\), and \(\sum_{k \ge 0} (1/2)^k \cdot \frac12 = 1\). In plain words: **the code spends exactly the whole budget: no prefix-free code could make lists any cheaper overall.**

```python
import math

feature = sum(2.0 ** -(2 * math.floor(math.log2(j)) + 1)
              for j in range(1, 2**20))
one_test = 2 * feature * 0.25
total = sum(one_test**k * 0.5 for k in range(200))
print(round(feature, 6), round(one_test, 6), round(total, 6))
```

It prints `0.999999 0.5 0.999999`: features up to about a million already reach 1 within a millionth. This code wastes no budget at all.

So any prefix-free language hands you a weight function for free: \(w(h) = 2^{-|h|}\). Kraft says it sums to at most 1.

**Theorem (the MDL bound).** Let \(\mathcal H\) be a hypothesis class and \(d: \mathcal H \to \{0, 1\}^*\) a prefix-free description language for it. Then for every sample size \(m\), every \(\delta \in (0, 1)\) and every distribution \(\mathcal D\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ \forall h \in \mathcal H, \quad L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}}. \]

It is the per-hypothesis SRM bound with the weight cost \(-\log w(h)\) replaced by \(|h|\), the description length in bits. In plain words: **except with probability \(\delta\), every list’s true risk is at most its empirical risk plus a gap that grows with the square root of its length in bits.** The 7-bit list at 100 labels, empirical risk 0.17: \(0.17 + \sqrt{(7 + 3.689)/200} = 0.17 + 0.231 = 0.401\). The snippet after the MDL rule below computes this bound as `mdl_bound`.

*Proof.* Distinct hypotheses get distinct descriptions (a string is a prefix of itself, so equal descriptions would break prefix-freeness), so \(\{d(h) : h \in \mathcal H\}\) is a prefix-free set, and the Kraft inequality gives \(\sum_h 2^{-|h|} \le 1\). That also makes \(\mathcal H\) countable, since there are only countably many strings. List its members as \(h_1, h_2, \dots\), take \(\mathcal H_n = \{h_n\}\) and \(w(n) = 2^{-|h_n|}\). Each singleton has the uniform convergence property with gap \(\varepsilon_n(m, \delta) = \sqrt{\log(2/\delta)/(2m)}\), by Hoeffding as above. The SRM bound theorem then says that with probability at least \(1 - \delta\), every \(h\) has

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{\log(2^{|h|}) + \log(2/\delta)}{2m}}. \]

Here \(\log(2^{|h|})\) is \(\log(1/w(n))\), the weight cost from before. Finally, \(\log(2^{|h|}) = |h| \log 2 < |h|\), because \(\log 2 \approx 0.693\), and making the numerator larger only loosens the bound. For the 7-bit list, \(7 \log 2 = 4.852\), the cost the earlier snippet printed, and the theorem rounds it up to 7. \(\blacksquare\)

The bound suggests its learner: trade empirical risk against description length. That is the **minimum description length** (MDL) rule:

\[ \mathrm{MDL}(S) \in \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}} \bigg]. \]

The bracket is the MDL bound of one list, and the argmin, as for SRM, returns one of the lists where it is smallest. In plain words: **MDL ships the list whose fit plus a cost for its length in bits is smallest.** At 100 labels on our stream that is the 7-bit list, with value 0.4012, beating “everything is junk”, whose 2 bits do not make up for its worse fit.

Here is the bound for a list of a given description length; `mdl_bound` is the bracket, term by term. Two lists with the same empirical risk 0.12 at \(m = 300\), one of 15 bits and one of 60; and how many examples a 29-bit list needs before its complexity term drops to 0.05.

```python
import math

def mdl_bound(LS, bits, m, delta=0.05):
    return LS + math.sqrt((bits + math.log(2 / delta)) / (2 * m))

print(round(mdl_bound(0.12, 15, 300), 4))
print(round(mdl_bound(0.12, 60, 300), 4))
need = (29 + math.log(2 / 0.05)) / (2 * 0.05**2)
print(math.ceil(need))
```

It prints `0.2965`, `0.4458` and `6538`. The last line sets the complexity term \(\sqrt{(29 + \log 40)/(2m)}\) equal to 0.05 and solves for \(m\). Same fit, very different certificates. On our stream, MDL ships 2 bits (“everything is junk”) at 30 labels, the 7-bit list “unknown sender, junk; otherwise keep” at 100 labels, the 15-bit three-test list at 300 and 1,000 labels, and the 29-bit list \(f\) at 3,000. The same staircase as SRM, with steps in different places, because it is a different prior.

The language need not be ours. Take the class of every filter you could write in some programming language, and describe each program by the bits of its compressed source file. That is a prefix-free description language too, and the MDL bound applies to it: it covers every program you could ever write.

## Occam’s razor, and its catch

Read the MDL bound as a philosophical statement. Two hypotheses with the same empirical risk get different guarantees, and the shorter one gets the better one. In words:

> A short explanation (that is, a hypothesis with a short description) tends to be more valid than a long explanation.

That is **Occam’s razor**, after William of Ockham, a fourteenth-century English logician who is believed to have been the first to state it explicitly. The MDL bound is one justification of it, made precise: the longer a hypothesis’s description, the more examples it must fit before its small empirical risk certifies a small true risk.

Now a puzzle, and it’s a good one. Description length depends on the language. Suppose \(h'\) is much shorter than \(h\) in our list code, and both have the same empirical risk; MDL says to prefer \(h'\). But we could have chosen a different prefix-free language, one that writes \(h\) in 3 bits and \(h'\) in 100,000. Under that language, MDL prefers \(h\). They are the same two lists. Which is it?

The answer is that no hypothesis is more trustworthy than another in itself. What matters is the order of events: the language must be chosen before the training sequence is seen. Recall the simplest case. If you commit to one hypothesis before seeing any data, Hoeffding gives you \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{\log(2/\delta)/(2m)}\), whatever that hypothesis is, however strange. Choosing a description language, or any weighting of hypotheses, is a weaker form of committing in advance. Instead of putting the whole commitment on one hypothesis, you spread it over many. As long as you spread it independently of the training sequence, the guarantee holds. And just as the single hypothesis could be anything, so can the language.

What breaks the guarantee is choosing the language after looking at the labels, giving the list that happens to fit a 3-bit code. Then the weights depend on \(S\), the bad events in the union bound are no longer the ones you budgeted for, and the proof says nothing. Your teammate cannot pick her weights after the run.

[[diagram:commit-order]]

## SRM and MDL as code

Here are the two learners in pseudocode, as the theory states them.

```
SRM(S, δ):
  prior knowledge: H = H_1 ∪ H_2 ∪ ..., each with gap ε_n
                   weights w(1), w(2), ... with Σ w(n) ≤ 1
  return h in argmin over h ∈ H of
         L_S(h) + ε_{n(h)}(m, w(n(h)) · δ)

MDL(S, δ):
  prior knowledge: H countable, described by a prefix-free
                   language, |h| the length of d(h)
  return h in argmin over h ∈ H of
         L_S(h) + sqrt((|h| + log(2/δ)) / (2m))
```

Both take an argmin over an infinite class. To run them at all, we need a way to stop.

**A stopping rule for SRM.** Suppose the classes are nested and the gap \(g_n = \varepsilon_n(m, w(n)\delta)\) does not decrease with \(n\) (true for our lists: \(|\mathcal H_n|\) grows and \(w(n)\) shrinks). Then the SRM minimum equals \(\min_n \big[\min_{h \in \mathcal H_n} L_S(h) + g_n\big]\): for each \(n\), the best empirical risk in \(\mathcal H_n\) plus that class’s gap, then the smallest over \(n\). That is exactly the SRM table at 300 labels, row by row, with its smallest entry 0.2714 at \(n = 3\). Indeed, for \(h \in \mathcal H_n\) we have \(n(h) \le n\), so \(L_S(h) + g_{n(h)} \le L_S(h) + g_n\): the right-hand side is never smaller than the SRM minimum; and choosing \(n = n(h)\) for the SRM minimizer shows it is never larger. So SRM is “ERM in \(\mathcal H_1\), then in \(\mathcal H_2\), and so on, keeping the best bound”. And since \(L_S \ge 0\), once \(g_n\) reaches the best bound found so far, no larger \(n\) can win, and you can stop.

```
SRM-nested(S, δ):
  best ← +∞
  for n = 1, 2, 3, ...:
    g ← ε_n(m, w(n) · δ)
    if g ≥ best: stop
    h_n ← ERM over H_n on S
    if L_S(h_n) + g < best: best ← L_S(h_n) + g; ship ← h_n
  return ship
```

MDL stops the same way: enumerate hypotheses in order of description length, and stop at the first length \(\ell\) with \(\sqrt{(\ell + \log(2/\delta))/(2m)} \ge\) the best bound so far.

**Cost.** Recall that \(O(\cdot)\) means “grows at most in proportion to”: \(O(m)\) time doubles at most when \(m\) doubles, ignoring constant factors. ERM over \(\mathcal H_n\) by brute force evaluates every list on every example; a list of size at most \(n\) is evaluated in \(O(n)\) time per message. So SRM-nested up to the stopping size \(N\) takes \(O(m N \sum_{n \le N} |\mathcal H_n|) = O(m N |\mathcal H_N|)\) time, since the last class dominates the sum, and \(O(m + N)\) space if lists are generated one at a time. MDL up to length \(\ell^*\) examines at most \(2^{\ell^* + 1}\) codewords (there are at most \(2^\ell\) strings of length \(\ell\)), so \(O(m \ell^* 2^{\ell^*})\) time.

In our simulated stream only features 1 to 6 ever appear, so a list in \(\mathcal H_n\) for \(n > 6\) behaves on the data like a list in \(\mathcal H_6\) (its tests on later features never fire), and stopping at size 6 is exact. The learner on the real stream cannot know that. How far does the stopping rule have to go when it doesn’t?

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def eps(n, m, delta):
    w = 6 / (math.pi**2 * n**2)
    budget = w * delta
    return math.sqrt(math.log(2 * size_H(n) / budget) / (2 * m))

for m, best in ((300, 0.2715), (1000, 0.1938), (3000, 0.151)):
    n = 1
    while eps(n, m, 0.05) < best:
        n += 1
    print(m, n, f"{size_H(n):.2e}")
```

The `while` loop is the stopping rule: grow \(n\) until the gap alone reaches the best bound. It prints `300 14 4.71e+15`, `1000 22 1.55e+28` and `3000 37 6.24e+54`. With 1,000 labels and the best bound at 0.1938, the rule cannot stop before size 22, and \(\mathcal H_{22}\) holds about \(1.55 \times 10^{28}\) lists. MDL is no better: at 1,000 labels its best bound is 0.2227, so it cannot stop before a length \(\ell\) with \(\ell \ge 2m \cdot 0.2227^2 - \log 40 \approx 95.5\): it must examine codes of up to 96 bits. Keep those numbers in mind.

**Build it yourself.** Using only the standard library. The starter code simulates the stream (the hidden list `TRUTH` is there only to generate labels; your learners must not read it).

```python
import itertools
import math
import random
from collections import Counter

TRUTH = (((3, 1), (1, 0), (2, 1), (5, 0), (4, 1)), 0)

def predict(h, x):
    tests, default = h
    for j, b in tests:
        if x[j - 1]:
            return b
    return default

def draw(m, seed, d=6, noise=0.1):
    rng = random.Random(seed)
    S = []
    for _ in range(m):
        x = tuple(int(rng.random() < 0.5) for _ in range(d))
        y = predict(TRUTH, x)
        if rng.random() < noise:
            y = 1 - y
        S.append((x, y))
    return S

def lists(n): ...
def size_H(n): ...
def eps(n, m, delta): ...
def erm(n, S): ...
def srm(S, delta=0.05, n_max=6): ...
def code_len(h): ...
def mdl(S, delta=0.05, n_max=6): ...
def true_risk(h, d=6, noise=0.1): ...

if __name__ == "__main__":
    assert size_H(3) == 158 and size_H(6) == 151946
    assert sum(1 for _ in lists(3)) == size_H(3)
    assert code_len(TRUTH) == 29
    assert sum(2.0 ** -code_len(h) for h in lists(4)) <= 1
    bound, n, h = srm(draw(300, 7))
    assert n == 3 and round(bound, 4) == 0.2715
    assert round(true_risk(h), 3) == 0.125
    bound, n, h = srm(draw(1000, 7))
    assert n == 5 and true_risk(h) == 0.1
    value, h = mdl(draw(100, 7))
    assert code_len(h) == 7 and round(value, 4) == 0.4012
    print("all asserts pass")
```

1. `lists(n)` generates every list in \(\mathcal H_n\) as `(tests, default)`, with distinct features among `1..n`; `size_H(n)` is the count formula; `eps` is the finite-class gap.
2. `erm(n, S)` returns `(L_S, h)` for a best list in \(\mathcal H_n\). Group identical examples with `Counter` first: there are only 64 distinct messages, so each list is evaluated 64 times instead of \(m\).
3. `srm` implements SRM-nested with \(w(n) = 6/(\pi^2 n^2)\), capped at `n_max`, and returns `(bound, n, h)`.
4. `code_len` is the description length of the list code above; `mdl` searches all lists over features 1 to `n_max` (exact on this stream, by the argument above) and returns `(value, h)`.
5. `true_risk` is exact: average the disagreement with `TRUTH` over all 64 messages, then \(0.1 + 0.8p\).
6. Reproduce the 30-label table from “Thirty labels, six sizes”, including the 720 tied lists at cap 6, and the SRM table at 300 labels.

## Fix the bug

Back to the pull request. The reviewer agrees to a deal: lists of any length may ship, provided the shipped list comes with a certificate, an SRM or MDL bound at \(\delta = 0.05\), of at most 0.20, the weights are written into the change before any labels are drawn, and the labelling budget is 1,000 messages.

In the lab below you set the weights across sizes 1 to 6 (the rest of the budget goes to larger sizes, which never win on this stream) or switch to MDL, which weighs each list by its bits. Then you choose how many labels to buy, the weights lock, the labels are drawn, and the lab shows the bound for each size and which list SRM or MDL would ship. After it ships, you get to see next week’s inbox: the list’s true risk.

Before your first try, a prediction. You put all of the weight on size 1, and buy all 1,000 labels. What ships, and does the reviewer sign?

[[lab-act2]]

## What people get wrong

**“Nonuniform learnability means the class is learnable after all, just more slowly.”** It is a weaker promise, not a slower version of the same one. A PAC guarantee tells you before you collect any data how many examples you need. A nonuniform guarantee tells you how many you need to compete with a particular \(h\), and you do not know which \(h\) is best until you have learned. The fundamental theorem still stands: decision lists of any length are not PAC learnable.

**“SRM picks the list with the smallest true risk.”** It picks the list with the smallest certified bound. At 30 labels on our stream, SRM ships “everything is junk”, with true risk 0.375, while ERM on \(\mathcal H_3\) would have found lists with true risk 0.125 to 0.175. SRM is not being stupid: with 30 labels it cannot tell those lists from their 720 neighbours at cap 6, and it refuses to certify what it cannot check. A bound is a promise about the worst case, and it is only as tight as the evidence.

**“Choose the weights to match the data.”** Weights encode prior knowledge, and they must be fixed before the training sequence is seen. Weights tuned after looking at the labels void the union bound, exactly like Occam’s language puzzle. Putting all the weight on size 3 is legal if you believed it in advance (and it ships a list with bound 0.1921 at 1,000 labels), but then a size-5 list can never ship, however much data arrives.

**“Shorter descriptions are truer.”** Description length is relative to a language. The razor is about committing before you look, not about any intrinsic simplicity.

## Weaker still: consistency

There is an even weaker way to define learning: let the sample size depend on the distribution as well.

A word of warning first, because of a name clash. In lesson 1, a *hypothesis* was consistent when it agreed with every example. Here, **consistency** is a property of a *learning rule*, borrowed from statistics, and it has nothing to do with fitting the examples. Same word, different idea.

In words first: a consistent learning rule, given enough labels, gets within \(\varepsilon\) of any hypothesis you name, but “enough” may depend on the mail stream itself, so on a stream it has never met you cannot say in advance how many. The memorizer on a stream dominated by a few repeated messages needs a hundred labels; on a stream of a million equally likely messages it needs millions.

**Definition (consistency).** Let \(Z\) be a domain set, \(\mathcal P\) a set of probability distributions over \(Z\), and \(\mathcal H\) a hypothesis class. A learning rule \(A\) is **consistent** with respect to \(\mathcal H\) and \(\mathcal P\) if there is a function \(m^{\mathrm{CON}}_{\mathcal H}: (0,1)^2 \times \mathcal H \times \mathcal P \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\), every \(h \in \mathcal H\) and every \(\mathcal D \in \mathcal P\), if \(m \ge m^{\mathrm{CON}}_{\mathcal H}(\varepsilon, \delta, h, \mathcal D)\) then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon\). If \(\mathcal P\) is the set of all distributions, \(A\) is **universally consistent** with respect to \(\mathcal H\).

Here \(Z\) plays the role of message-label pairs, \(\mathcal X \times \{0,1\}\), and \(\mathcal P\) is the family of streams you are willing to face. The only change from nonuniform learnability is the signature: \(m^{\mathrm{CON}}_{\mathcal H}: (0,1)^2 \times \mathcal H \times \mathcal P \to \mathbb N\) takes the two dials, a hypothesis and also a distribution, and returns a number of labels. In plain words: **for each stream and each hypothesis there is some number of labels beyond which the rule competes with that hypothesis on that stream.** The two streams just mentioned get two very different entries of \(m^{\mathrm{CON}}\), as the snippet below shows.

A nonuniform learner for \(\mathcal H\) is universally consistent for \(\mathcal H\) (take \(m^{\mathrm{CON}}\) to ignore \(\mathcal D\)). The converse fails, and the counterexample is an old friend: lesson 1’s memorizer. Let Memorize return, on a message \(x\), the majority label among the copies of \(x\) in the training sequence, and a fixed default label if \(x\) never appeared. Over any countable domain with a finite label set, Memorize is universally consistent for the class of all classifiers. Yet over an infinite domain the class of all classifiers is not nonuniformly learnable, as we saw. So consistency is strictly weaker. (It is also unsettling: lesson 1’s cautionary tale meets the definition.)

Why does Memorize succeed eventually? Take the noise-free case. It errs only on messages it has never seen, so its true risk is at most the probability mass of the unseen messages. By linearity of expectation and the independence of the \(m\) draws, the expected unseen mass is

\[ \mathbb E_{S}\Big[\sum_{x} \mathcal D(x)\, \mathbb 1[x \notin S]\Big] = \sum_x \mathcal D(x)\, \mathbb P[x \notin S] = \sum_x \mathcal D(x) \big(1 - \mathcal D(x)\big)^m. \]

Read it from the inside out. \(\mathcal D(x)\) is the probability that message \(x\) arrives. \(\mathbb 1[x \notin S]\) is 1 when \(x\) is not among the training messages (\(\notin\) reads “is not in”) and 0 otherwise, so the sum \(\sum_x\) over every message adds up the probability of the unseen ones: the **missing mass**. \(\mathbb E_S\) averages that over random training sequences. Linearity moves the average inside the sum, the average of an indicator is the probability of its statement, and \(x\) is missed by all \(m\) independent draws with probability \((1 - \mathcal D(x))^m\), one factor per draw. In plain words: **on average, the unseen mail weighs the sum, over messages, of each message’s probability times its chance of being missed \(m\) times in a row.** If message 1 arrives half the time, at \(m = 10\) it contributes \(0.5 \times 0.5^{10} = 0.0005\); message 4, at \(1/16\), contributes \(0.0625 \times (15/16)^{10} = 0.033\); message 8, at \(1/256\), only 0.004 because it is so rare.

```python
def contribution(q, m):
    return q * (1 - q) ** m

for x in (1, 4, 8):
    print(x, round(contribution(2.0 ** -x, 10), 5))
```

It prints `1 0.00049`, `4 0.03278` and `8 0.00376`. The snippet below adds these terms up, as `missing_mass`.

For any fixed \(\mathcal D\) on a countable domain this goes to 0 as \(m\) grows. But how fast depends entirely on \(\mathcal D\). Compare a stream where message \(x\) arrives with probability \(2^{-x}\) with one spread evenly over a million messages:

```python
def missing_mass(p, m):
    return sum(q * (1 - q) ** m for q in p)

geometric = [2.0 ** -x for x in range(1, 200)]
for m in (10, 100, 1000):
    g = round(missing_mass(geometric, m), 4)
    print(m, g, round((1 - 1e-6) ** m, 4))
```

It prints `10 0.1312 1.0`, `100 0.0143 0.9999` and `1000 0.0014 0.999`. Same learner, same \(m\): on the first stream it has seen nearly everything that matters, and on the second it has seen almost nothing. Consistency promises that each stream is eventually conquered, but it never says when. You will prove the consistency of Memorize yourself in the end-of-lesson problems.

**No-Free-Lunch, revisited.** Lesson 5 proved that no learner learns the class of all classifiers over an infinite domain. Memorize is consistent for exactly that class. Why is that not a contradiction? Look at the order of the quantifiers. No-Free-Lunch says: for every learner and every training sequence size \(m\), there exist a distribution and a labelling function on which the learner, given \(m\) examples, is likely to fail. Consistency of Memorize says: for every distribution and labelling function, there exists a size \(m\), depending on both, beyond which Memorize is likely to succeed. No-Free-Lunch fixes \(m\) first and then picks a bad world; consistency fixes the world first and then picks \(m\).

[[diagram:quantifiers]]

## Which notion of learnability do you want?

We now have three definitions: PAC (one sample size for every hypothesis and every distribution), nonuniform (the size may depend on the hypothesis), and consistency (it may depend on the distribution too). Which is right depends on what you want a definition for. Let’s go through the questions an engineer would ask.

**What is the risk of the hypothesis I learned?** PAC and nonuniform learning both bound the true risk of the returned hypothesis by its empirical risk plus a computable term. Consistency gives no such bound. (You can always estimate the risk on held-out labelled messages, a validation set, which lesson 11 covers.)

**How many examples do I need to be as good as the best hypothesis in the class?** PAC gives a crisp answer before you start. Nonuniform learning does not: the number depends on the best hypothesis, which you don’t know. Consistency does not either, and the number depends on \(\mathcal D\) as well. In this sense PAC is the only useful definition. Keep in mind, though, that even PAC only promises to approach the best in the class: if the class has a large approximation error, the risk may still be large. And when a learner fails, PAC lets you bound the estimation error and so blame the approximation error and change the class; a nonuniform learner lets you change the weights; a consistent learner tells you nothing about which part failed, or how many more examples would help.

**How do I learn, and how do I say what I know?** PAC learning, through No-Free-Lunch, shows prior knowledge is necessary, gives a crisp way to state it (choose a class) and a generic learner for it (ERM). Nonuniform learning also gives a crisp way to state prior knowledge (weights over classes or hypotheses) and a generic learner (SRM). SRM is especially good at **model selection**, when you are unsure which class to use. Fit lists by ERM with cap 6 and you overfit, as the 30-label table showed; cap them at 1 and you underfit; SRM over all sizes picks the size from the data, paying only a small extra in estimation error and the loss of knowing in advance how many examples you need. Consistency gives no learning paradigm and no way to state prior knowledge at all. Memorize needs none.

**Should I prefer a consistent learner?** It sounds desirable to be consistent for every function from \(\mathcal X\) to \(\mathcal Y\): with enough data you would eventually match the Bayes optimal predictor (lesson 3: the best possible predictor of all, with true risk 0.1 on our stream). The argument has two problems. First, on the distributions you actually meet, the consistent learner may need more examples than you will ever have. Second, consistency is cheap to add. Over a countable domain with a finite label set, take any nonuniform learner: run it, compute its bound, and if the bound is small enough, return its hypothesis; otherwise fall back to Memorize. The result is consistent for every classifier. A property that easy to bolt on is a poor reason to choose one learner over another.

## Recap and the next bug

1. Nonuniformly learnable means: for each \(h\), enough examples, \(m^{\mathrm{NUL}}(\varepsilon, \delta, h)\), make the learner competitive with \(h\), for every distribution. Agnostic PAC learnable classes are nonuniformly learnable; the converse fails.
2. A class is nonuniformly learnable exactly when it is a countable union of agnostic PAC learnable classes, equivalently of classes with finite VC dimension. Decision lists of any length are such a union, with infinite VC dimension.
3. The SRM bound splits the confidence budget \(\delta\) across classes with weights summing to at most 1, so every hypothesis gets its own gap, \(\varepsilon_{n(h)}(m, w(n(h))\delta)\). SRM returns the hypothesis with the smallest bound, and it is a nonuniform learner with \(m^{\mathrm{NUL}} \le m^{\mathrm{UC}}_{\mathcal H_{n(h)}}(\varepsilon/2, w(n(h))\delta)\).
4. With a prefix-free description language, the Kraft inequality turns description lengths into weights, and the MDL bound is \(L_S(h) + \sqrt{(|h| + \log(2/\delta))/(2m)}\): Occam’s razor, valid because the language is fixed before the data.
5. Consistency lets the sample size depend on the distribution too. Memorize is universally consistent over countable domains, which shows how weak that promise is.

The reviewer signs. Your filter now ships the five-test list with a certificate, and your teammate is happy. Then the operations team runs it on the real mail stream, where hundreds of features are in use, and the training job never finishes. Of course it doesn’t: with 1,000 labels, SRM’s stopping rule has to examine lists up to size 22, about \(1.55 \times 10^{28}\) of them, and MDL has to reach codes of 96 bits. The statistics are fine; a thousand labels are enough. The computation is hopeless. So far we have only asked how many examples a learner needs. When is learning hard because of the computing time, not the data? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Your teammate proposes the weights \(w(n) = 1/n\): “longer lists are less likely, but nothing is ruled out.” Which step of the SRM bound’s proof breaks, and what does the theorem then promise? Separately, she proposes \(w(7) = 0\) and \(w(n) = 6/(\pi^2 n^2)\) for every other \(n\). Is that a legal weight function, and what does it do to a list of size 7?
2. In the proof of the Kraft inequality, point to the exact sentence that uses prefix-freeness. Then check the claim fails without it: compute the Kraft sum of the code \(\{0, 01, 1\}\), and name two events in the proof that overlap. Finally, suppose the list code dropped its end flag, so a list is written as its tests followed by the default bit. Find two lists whose codes break prefix-freeness.
3. On 300 labels, one list has 15 bits and empirical risk 0.12; another has 29 bits and empirical risk 0.10. Which does MDL prefer at \(\delta = 0.05\), and by how much? Assuming the two empirical risks stay where they are, what is the smallest \(m\) at which MDL prefers the 29-bit list?
