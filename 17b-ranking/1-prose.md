# Ranking and Bipartite Ranking

## The top five are a lottery

Lesson 17a ended with the folders fixed. A multiclass learner, trained with your users’ cost table, files fresh mail at a certified cost below 0.06, and the work mail in the middle is no longer lost. But your users asked for two things on the first day, and the second is the one they notice every morning: the inbox itself in order, with the messages that matter at the top of the list.

To order the inbox the team trained one more yes/no filter, “is this message important?”, and put the yes answers on top. The report on it looks wonderful: 98% of its answers are right. Then you notice that only 2% of messages are important, and the filter has learned to say no to every single one. Its error is 2%, it is useless, and the inbox is sorted by nothing at all.

The team’s patch fails more quietly. They retrain the filter on a busier mailbox, a hundred messages a morning with about one in twenty important, and now it does say yes, about three times a morning. Those messages go on top, and the rest follow in the order they arrived. The report says 97% of its answers are right. Your users read the top five each morning, and only about half of those five are important, although sorting the same messages by the filter’s own scores, before it turns them into yes and no, would put about two thirds there. (These numbers come from the lab world you will meet at the end of the lesson.)

So the bug has two parts. Why is a filter with 98% accuracy worthless as a ranker? And what should an inbox order be trained for, and certified by, so that your users can trust the top of their list? This lesson builds the answer: losses that judge a whole order, losses that look hardest at the top, and a way to train a linear ranker for exactly the number your users care about, the share of important messages among the top five.

## What you need from lesson 17a

Ranking will reuse the multiclass machinery of lesson 17a almost word for word, so here is the toolkit, one line each, with the two formulas this lesson leans on read once more.

- **Messages and labels.** A message is a vector \(\mathbf x \in \mathbb R^n\) of word counts. A multiclass hypothesis \(h: \mathcal X \to \mathcal Y\) picks one label from a finite label set \(\mathcal Y\); the folders were the labels 1 (work), 2 (personal), 3 (promotions) and 4 (junk).
- **Inner product and length** (lesson 9). \(\langle \mathbf u, \mathbf v \rangle\) multiplies matching entries and adds, so \(\langle (2, -1), (1, 1) \rangle = 2 - 1 = 1\); \(\|\mathbf u\| = \sqrt{\langle \mathbf u, \mathbf u \rangle}\) is the length.
- **The indicator** (lesson 1). \(\mathbb 1[\text{statement}]\) is 1 when the statement is true and 0 when it is false.
- **Argmax.** \(\operatorname{argmax}_y f(y)\) is not the largest value but where it occurs: the set of \(y\) that achieve it, a set because several can tie.
- **Class-sensitive feature mapping.** \(\Psi(\mathbf x, y) \in \mathbb R^d\) is a vector built from the message together with one candidate label, and the **score** of label \(y\) is \(\langle \mathbf w, \Psi(\mathbf x, y) \rangle\).
- **Cost.** \(\Delta(y', y) \ge 0\) is the loss of predicting \(y'\) when the correct label is \(y\), with \(\Delta(y, y) = 0\). Your users’ table charged 1 for work filed as junk and 0.2 for a misplaced promotion.
- **Hinge loss** (lesson 15). For a signed score \(a\), \(\max\{0, 1 - a\}\) is 0 when \(a \ge 1\) and grows by 1 for every unit below, so \(a = -0.5\) costs 1.5.
- **Convex and Lipschitz** (lesson 12). A function is convex when the straight segment between any two points of its graph lies on or above the graph, like \(w \mapsto |w|\), and a maximum of convex functions is convex. It is \(\rho\)-Lipschitz when moving its input a distance \(\delta\) changes its output by at most \(\rho\delta\); \(w \mapsto 3w\) is 3-Lipschitz.
- **Subgradients and SGD** (lesson 14). A subgradient is the slope of a line (or plane) through the current point that stays below the function everywhere; for a maximum of affine functions, the gradient of a piece that achieves the maximum is one. SGD steps against one subgradient per example and returns the average of its weight vectors.
- **The hold-out guarantee** (lesson 11), which certified the folders: for a hypothesis fixed before \(m_t\) fresh examples are drawn, and a loss with values in \([0, 1]\), the true risk is within a margin \(\sqrt{\log(2/\delta)/(2m_t)}\) of the average on the fresh examples, except with probability \(\delta\). “Fix the bug” reads this margin term by term.

The **linear multiclass predictor** returns the label with the highest score:

\[ h_{\mathbf w}(\mathbf x) = \operatorname*{argmax}_{y \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y) \rangle. \]

Read it from the inside out: \(\Psi(\mathbf x, y)\) pairs the message with one candidate label; the inner product with the single shared weight vector \(\mathbf w\) collapses it to one number, the score; the argmax tries every label in \(\mathcal Y\) and returns the one with the largest score; the subscript \(\mathbf w\) says the predictor is fixed by the weights. In plain words: **compute one score per label and return the label with the highest score.** With \(\Psi(\mathbf x, \text{work}) = (2, 0)\), \(\Psi(\mathbf x, \text{personal}) = (0, 1)\), \(\Psi(\mathbf x, \text{promotions}) = (1, 1)\) and \(\mathbf w = (1, 3)\), the scores are 2, 3 and 4, and the answer is promotions.

```python
psi = {"work": (2, 0), "personal": (0, 1), "promotions": (1, 1)}
w = (1, 3)
score = {y: w[0] * v[0] + w[1] * v[1] for y, v in psi.items()}
print(score, max(score, key=score.get))
```

It prints `{'work': 2, 'personal': 3, 'promotions': 4} promotions`. This lesson reuses that shape with a surprising label set: every possible order of the inbox.

The **generalized hinge loss** is the convex surrogate that trains it:

\[ \ell(\mathbf w, (\mathbf x, y)) = \max_{y' \in \mathcal Y} \left( \Delta(y', y) + \langle \mathbf w, \Psi(\mathbf x, y') - \Psi(\mathbf x, y) \rangle \right). \]

Read it from the inside out: \(y'\) is a rival label and \(y\) the correct one; the inner product of \(\mathbf w\) with the difference \(\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\) is how much the rival outscores the truth, negative if it trails; \(\Delta(y', y)\) adds the cost of choosing that rival; \(\max_{y' \in \mathcal Y}\) keeps the worst rival, and the rival \(y' = y\) contributes \(0 + 0 = 0\), so the loss is never negative. In plain words: **the loss is the worst rival’s cost plus how far that rival outscores the correct label.** A work message scored 1 for work, 0 for personal and promotions and 2 for junk has the four terms 0, \(0.5 + 0 - 1 = -0.5\), \(-0.5\) and \(1 + 2 - 1 = 2\), so \(\ell = 2\), while the cost of the prediction it actually makes (junk) is 1.

```python
C = [[0, .5, .5, 1], [.5, 0, .5, 1], [.2, .2, 0, .2],
     [.5, .5, .2, 0]]
scores = [1, 0, 0, 2]
y = 0
terms = [C[y][yp] + scores[yp] - scores[y] for yp in range(4)]
print(terms, max(terms))
```

It prints `[0, -0.5, -0.5, 2] 2`.

Lesson 17a proved four facts about this loss, and this lesson uses every one. First, it is at least the cost of the prediction, \(\ell(\mathbf w, (\mathbf x, y)) \ge \Delta(h_{\mathbf w}(\mathbf x), y)\), in three steps: the predicted label scores at least as high as the correct one; so adding “score of the prediction minus score of the truth” to the cost can only increase it; and the maximum over all rivals is at least that one term. Second, it is convex in \(\mathbf w\), as a maximum of affine functions. Third, it is \(\rho\)-Lipschitz with \(\rho = \max_{y'} \|\Psi(\mathbf x, y') - \Psi(\mathbf x, y)\|\). Fourth, SGD on it needs one argmax per step: the rival \(\hat y\) that achieves the maximum, and the step \(\Psi(\mathbf x, \hat y) - \Psi(\mathbf x, y)\). The guarantees for multiclass SVM and SGD then depend on \(\rho\) and on the length bound \(B\) of the weights, not explicitly on the number of labels, which is why a label set with millions of members is not hopeless.

## Ranking the inbox

Start with the general problem, and come back to important-or-not once the tools are ready. The inbox is not a set of labels; it is an order. A ranker gives every message of the morning a score, and the order is the scores sorted, highest on top: scores 2, 1 and 6 put message 3 first, then 1, then 2.

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

**Normalized discounted cumulative gain (NDCG).** Users read the top of the list, so mistakes there should cost more. Give every position a weight, a **discount** \(D: \mathbb N \to \mathbb R_+\), nondecreasing because position \(r\) is the top. A typical discount, with a parameter \(k < r\), gives the top position weight 1, each of the next \(k - 1\) positions a little less, and every position below the top \(k\) nothing:

\[ D(i) = \begin{cases} \dfrac{1}{\log_2(r - i + 2)} & \text{if } i \in \{r - k + 1, \dots, r\}, \\ 0 & \text{otherwise.} \end{cases} \]

Read it from the inside out: \(i\) is a position, with \(r\) the top; \(\{r - k + 1, \dots, r\}\) are the top \(k\) positions; inside them the weight is one over \(\log_2(r - i + 2)\), where \(\log_2\) is the logarithm in base 2, so the top gets \(1/\log_2 2 = 1\) and the next \(1/\log_2 3 = 0.63\); below them it is 0. In plain words: **the top position counts fully, each lower position in the top \(k\) a bit less, and everything below the top \(k\) not at all.** For \(r = 6\) and \(k = 3\), positions 1 to 6 get weights 0, 0, 0, 0.5, 0.63, 1. That is how search engines are judged, and it is how your users read their mail.

```python
import math

r, k = 6, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
print([round(D(i), 2) for i in range(1, r + 1)])
```

It prints `[0, 0, 0, 0.5, 0.63, 1.0]`.

Take six messages with relevances \(\mathbf y = (5, 1, 4, 0, 3, 2)\) from the users and scores \(\mathbf y' = (0.9, 0.8, 0.1, -0.5, 0.7, 0.2)\) from a ranker, which put them in positions \(\pi(\mathbf y') = (6, 5, 2, 1, 4, 3)\). The **discounted cumulative gain** adds each message’s true relevance, weighted by the discount of the position the ranker gave it:

\[ G(\mathbf y', \mathbf y) = \sum_{i=1}^r D(\pi(\mathbf y')_i)\, y_i. \]

Read it from the inside out: \(\pi(\mathbf y')_i\) is the position the ranker gives message \(i\); \(D(\cdot)\) turns that position into a weight; \(y_i\) is the message’s true relevance; the sum adds weight times relevance over all messages. In plain words: **the gain adds up the true relevance of every message, weighted by how good a position the ranker put it in.** With \(k = 3\), the ranker puts message 1 (relevance 5) on top with weight 1, message 2 (relevance 1) second with weight 0.63 and message 5 (relevance 3) third with weight 0.5, so \(G = 5 + 0.63 + 1.5 = 7.13\).

```python
import math

r, k = 6, 3
D = lambda i: 1 / math.log2(r - i + 2) if i > r - k else 0
y, pos = (5, 1, 4, 0, 3, 2), (6, 5, 2, 1, 4, 3)
terms = [round(D(p) * yi, 2) for p, yi in zip(pos, y)]
print(terms, round(sum(D(p) * yi for p, yi in zip(pos, y)), 2))
```

It prints `[5.0, 0.63, 0, 0, 1.5, 0] 7.13`.

If all relevances are nonnegative, then \(0 \le G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\): the first inequality because every term is nonnegative, the second because, for a nondecreasing \(D\), putting the largest relevances in the positions with the largest discounts is the best you can do (a sorting fact you will prove in the problem at the end). The perfect order here would put messages 1, 3 and 5 on top, for \(5 + 4 \times 0.63 + 3 \times 0.5 = 9.02\). So the **NDCG loss** is the share of that ideal gain the ranker throws away:

\[ \Delta(\mathbf y', \mathbf y) = 1 - \frac{G(\mathbf y', \mathbf y)}{G(\mathbf y, \mathbf y)} = \frac{1}{G(\mathbf y, \mathbf y)} \sum_{i=1}^r \left( D(\pi(\mathbf y)_i) - D(\pi(\mathbf y')_i) \right) y_i. \]

Read it from the inside out: \(G(\mathbf y, \mathbf y)\) is the gain of the perfect order, the largest possible; the ratio is the share of that ideal the ranker achieved; one minus it is the share lost. The second form writes the same loss as a sum, message by message, of “weight it deserved minus weight it got”, times its relevance. In plain words: **the NDCG loss is the fraction of the ideal gain that the ranker’s order throws away.** Here \(1 - 7.13/9.02 = 0.21\), and the message-by-message sum gives the same 0.21. It lies in \([0, 1]\), and it is 0 when \(\pi(\mathbf y') = \pi(\mathbf y)\).

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

[[diagram:ranking-losses]]

All three pieces in one place, on the same six messages:

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

It prints `(4, 3, 5, 1, 2)`, then `(6, 5, 2, 1, 4, 3) 0.333`, then `0.21`. The ranker gets the top message right but puts message 2, relevance 1, in second place, where message 3, relevance 4, belonged. Five of the fifteen pairs are out of order, a Kendall tau loss of 1/3, while NDCG, which only looks at the top three, charges 0.21.

## Linear rankers and their surrogates

The natural linear ranker scores each message with the same weight vector:

\[ h_{\mathbf w}((\mathbf x_1, \dots, \mathbf x_r)) = (\langle \mathbf w, \mathbf x_1 \rangle, \dots, \langle \mathbf w, \mathbf x_r \rangle), \]

and the class is \(\mathcal H_W = \{h_{\mathbf w} : \mathbf w \in W\}\). (A feature map in the style of lesson 16 can be applied first; we keep the plain form.)

Read it from the inside out: each message \(\mathbf x_i\) gets its own score \(\langle \mathbf w, \mathbf x_i \rangle\), always with the same \(\mathbf w\), and the ranker returns the list of scores. In plain words: **a linear ranker scores every message with one shared weight vector and sorts by the score.** With \(\mathbf w = (1, -1)\) and three messages \((2, 0)\), \((1, 3)\), \((0, 1)\), the scores are \(2, -2, -1\), so the order from the top is message 1, message 3, message 2.

```python
w = (1, -1)
xs = [(2, 0), (1, 3), (0, 1)]
scores = [w[0] * a + w[1] * b for a, b in xs]
order = sorted(range(1, 4), key=lambda i: -scores[i - 1])
print(scores, order)
```

It prints `[2, -2, -1] [1, 3, 2]`.

ERM over a training sequence of inboxes \((\bar{\mathbf x}_1, \mathbf y_1), \dots, (\bar{\mathbf x}_m, \mathbf y_m)\), each with its own length \(r_i\), minimizes \(\sum_i \Delta(h_{\mathbf w}(\bar{\mathbf x}_i), \mathbf y_i)\), and for most ranking losses that is computationally hard. So, once more, convex surrogates.

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

Read it from the inside out: \(\mathbf v\) is a candidate assignment of positions \(1, \dots, r\) to the messages, one each; \(\sum_i v_i y'_i\) multiplies each message’s position by its score and adds; the argmax over all \(r!\) permutations in \(V\) picks the best assignment. In plain words: **the order the scores induce is the assignment of positions that makes “position times score” add up to the most.** With scores \((0.3, -1, 2)\), sorting gives positions \((2, 1, 3)\) and the sum \(0.6 - 1 + 6 = 5.6\); every other permutation gives less. The reason is that to make the sum as large as possible you give the largest position number to the largest score, the next largest to the next, and so on, which is exactly what sorting does: the sorting fact again, and proving it is Problem 1.

```python
import itertools

yp = (0.3, -1, 2)
V = itertools.permutations((1, 2, 3))
total = {v: round(sum(a * b for a, b in zip(v, yp)), 2)
         for v in V}
print(max(total, key=total.get), total)
```

It prints `(2, 1, 3)` and the sum for each of the six permutations; the largest, `5.6`, belongs to `(2, 1, 3)`. The next snippet checks the identity on a thousand random score vectors against all 720 permutations of six positions.

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

so the linear ranker is a linear multiclass predictor whose labels are permutations.

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

The generalized hinge derivation goes through word for word (write \(\Delta(\mathbf v, \mathbf y)\) for the NDCG loss of the order \(\mathbf v\), since \(\pi(\mathbf v) = \mathbf v\)):

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

Read it from the inside out: \(\alpha_i v_i\) rewards putting a high-scoring message (large \(\mu_i\), so very negative \(\alpha_i\)) in a high position; \(\beta_i D(v_i)\) charges for giving a relevant message (large \(y_i\)) a big discount weight, because the max hunts for orders with a *large* loss, and a relevant message near the top lowers the loss; the argmin over \(V\) picks the cheapest assignment. In plain words: **the most violating order is the cheapest assignment of positions, where each message’s cost mixes its predicted score and its true relevance.** For three messages with \(\mu = (2, -2, -1)\) and relevances \((3, 1, 2)\), \(k = 2\): \(\alpha = -\mu = (-2, 2, 1)\), the ideal gain is \(G(\mathbf y, \mathbf y) = 3 + 2/\log_2 3 = 4.26\), and \(\beta = \mathbf y / 4.26 = (0.70, 0.23, 0.47)\). The second snippet below checks on random data that this argmin and the surrogate’s argmax always agree.

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

a contradiction (the strict inequality survives because every \(\gamma_i > 0\)).

Read it left to right: substitute the blend for \(B\); pull the weights out of the inner product (linearity); use the supposed strict inequality \(\langle A, C_i \rangle > \langle A, B \rangle\) on every term, with positive weights; then \(\sum_i \gamma_i = 1\) collapses the right end back to \(\langle A, B \rangle\). In plain words: **a weighted average cannot be strictly larger than every one of its terms, so some permutation in the blend is at least as cheap as the blend.** The snippet after the proof blends three permutations with weights 0.5, 0.3 and 0.2: \(0.5 \times 17 + 0.3 \times 23 + 0.2 \times 21 = 19.6\), and the cheapest term, 17, is below it.

So some \(C_i\) has \(\langle A, C_i \rangle = \langle A, B \rangle\). That \(C_i\) is feasible for the relaxed problem with the optimal value, so it is optimal there. And every permutation matrix \(C\) is feasible for the relaxed problem, so \(\langle A, C_i \rangle = \langle A, B \rangle \le \langle A, C \rangle\): \(C_i\) is optimal for the assignment problem too. \(\blacksquare\)

[[diagram:assignment]]

Here is the blend from the proof, on a random \(4 \times 4\) cost matrix:

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

A ranker outputs scores \(\mathbf y' \in \mathbb R^r\), and a **threshold** \(\theta \in \mathbb R\) turns them into yes/no answers, \(\operatorname{sign}(y'_i - \theta)\). (Ties at the threshold are measure-zero for real-valued scores; read the sign of 0 as \(-1\) here.) Here is one morning of ten messages, three of them important, scored by a ranker and sorted from the top:

| message | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | 2.1 | 1.4 | 0.9 | 0.3 | −0.2 | −0.4 | −0.8 | −1.0 | −1.5 | −2.2 |
| important | yes | no | yes | no | no | yes | no | no | no | no |

Before reading on, count by hand at \(\theta = 0\): which messages are flagged, and how many of them are important? Four counts describe the result: the flagged messages that are important, the flagged ones that are not, the important ones that are missed, and the unimportant ones correctly left alone. On this morning they are 2, 2, 1 and 5. In symbols:

\[ \begin{aligned} a &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{true positives} \\ b &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = +1\}| && \text{false positives} \\ c &= |\{i : y_i = +1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{false negatives} \\ d &= |\{i : y_i = -1 \wedge \operatorname{sign}(y'_i - \theta) = -1\}| && \text{true negatives} \end{aligned} \]

Read one line from the inside out: \(y_i = +1\) is “message \(i\) is important”; \(\operatorname{sign}(y'_i - \theta) = +1\) is “its score is above the threshold, so it is flagged”; \(\wedge\) reads “and”; the braces with a colon collect the messages \(i\) such that both hold; the bars \(|\cdot|\) count them. In plain words: **\(a\) counts important messages that are flagged, \(b\) unimportant ones that are flagged, \(c\) important ones that are missed and \(d\) unimportant ones that are left alone.** On this morning, at \(\theta = 0\), messages 1 to 4 are flagged: two of them important (\(a = 2\)) and two not (\(b = 2\)); one important message is missed (\(c = 1\)); five unimportant ones are left alone (\(d = 5\)). Every message falls in exactly one of the four counts, so they add up to 10.

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

Read the F₁ formula from the inside out. \(1/\text{precision}\) and \(1/\text{recall}\) are large when either score is small; dividing 2 by their sum gives the **harmonic mean**, which is pulled towards the smaller of the two. Substituting \(a/(a+b)\) and \(a/(a+c)\) and clearing fractions gives \(2a/(2a + b + c)\). In F\(_\beta\), the weight \(\beta^2\) multiplies \(1/\text{recall}\), so missed important messages count \(\beta^2\) times as much. In plain words: **F₁ is high only when the flagged messages are mostly important and most important messages are flagged.** On the ten-message morning, at \(\theta = 0\), precision 0.5 and recall 0.667 give F₁ \(= 2/(2 + 1.5) = 0.571\), the same as \(2 \cdot 2/(4 + 2 + 1)\); with \(\beta = 2\), F₂ is 0.625.

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

Here is the whole morning in code, at \(\theta = 0\) and for the top three:

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

Now play with it. In the lab the ten-message morning is a list sorted by score. Slide the threshold \(\theta\), and the \(k\) of precision and recall at \(k\); drag a message’s score to reorder the list; and watch the zero-one loss, recall, precision, specificity, F₁, precision at \(k\) and the AUC move in different directions. One button replays the opening bug: 1,000 messages, 2% of them important, and a filter that says no to everything.

[[lab]]

## Learning to rank for the measure you want

How do we train a linear ranker for, say, precision at 5? The measure depends on the scores \(\mathbf y' = h_{\mathbf w}(\bar{\mathbf x})\) only through the yes/no vector they induce,

\[ b(\mathbf y') = (\operatorname{sign}(y'_1 - \theta), \dots, \operatorname{sign}(y'_r - \theta)) \in \{\pm 1\}^r. \]

Read \(b\) from the inside out: \(y'_i - \theta\) is how far message \(i\)’s score sits above the threshold; its sign is \(+1\) (flagged) or \(-1\) (not); the vector lists those answers, an element of \(\{\pm 1\}^r\), the set of all length-\(r\) lists of \(\pm 1\). In plain words: **\(b(\mathbf y')\) is the list of flagged / not flagged decisions that the threshold makes from the scores.** Scores \((0.9, -0.2, 0.3)\) at \(\theta = 0\) give \((+1, -1, +1)\).

```python
yp, theta = (0.9, -0.2, 0.3), 0
print(tuple(1 if s - theta > 0 else -1 for s in yp))
```

It prints `(1, -1, 1)`.

The plan is the same as for NDCG: write \(b\) as an argmax, then let the generalized hinge loss do the rest. For every measure above there is a set \(V \subseteq \{\pm 1\}^r\) such that

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

It prints `4.7035 4.7035`: 24 groups searched, the same optimum as 256 vectors. (The F₁ loss is \(1 - 2a/(2a + b + c)\), and \(2a + b + c = a + b + P\).)

## Build it yourself

One challenge for this lesson’s algorithms. The starter runs its asserts against your code; it was run against a reference implementation.

**Inbox.** Implement the ranking measures and BipartiteArgmax.

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

Time to ship the inbox. In the lab’s second act the mornings are realistic: each one is a day’s inbox of 100 messages, about one in twenty of them important, and each message is described by two scores. You have a budget of fresh labelled mornings that the learner never trains on, each a whole inbox with its important messages marked.

You make one choice: what the inbox ranker is trained and judged by. The options are the accuracy of an “important?” filter, with the flagged messages on top and the rest in arrival order; F₁ at threshold 0; the AUC; or precision at 5, the share of important messages among the top five each morning, which is what your users read.

Then you certify on fresh mail, the way lesson 17a certified the folders. Lesson 11’s hold-out guarantee applies to any loss with values in \([0, 1]\): if \(\hat h\) was fixed before the \(n\) fresh examples were drawn, then with probability at least \(1 - \delta\) its true risk is within \(\sqrt{\log(2/\delta)/(2n)}\) of its average loss on them. One morning’s precision at 5 lies in \([0, 1]\), so one minus it is such a loss, with independent mornings as the examples. The guarantee then says that the true precision at 5, averaged over all the mornings to come, is at least the measured average minus the margin.

Read the margin from the inside out: \(\hat h\) is the ranker you ship, frozen before the fresh mornings arrive; \(n\) is the number of fresh mornings, \(\delta\) the chance that the guarantee fails, \(\log\) the natural logarithm, and the square root of \(\log(2/\delta)\) over \(2n\) is the half-width of the band around the measured average. In plain words: **the measured precision at 5 is within the margin of the true one, except with probability \(\delta\), and the margin shrinks like one over the square root of the number of fresh mornings.** With \(\delta = 0.05\) and 300 fresh mornings it is \(\sqrt{\log 40 / 600} = 0.078\); the snippet mirrors it for three sizes.

```python
import math

delta = 0.05
margin = lambda n: math.sqrt(math.log(2 / delta) / (2 * n))
for n in (50, 300, 500):
    print(n, round(margin(n), 4))
```

It prints `50 0.1921`, `300 0.0784` and `500 0.0607`. Your users’ condition, with 95% confidence: the certified precision at 5, measured minus margin, at least 0.6. Work out which choices can meet it before you spend the fresh mornings.

[[lab-act2]]

## What people get wrong

**“98% accuracy means the important messages are found.”** With 2% of messages important, saying no to everything scores 98%. Measure what users see: precision at \(k\), recall at \(k\), F₁ or AUC.

**“Precision at 5 can be trained by thresholding a classifier.”** A classifier is trained for the zero-one loss and chooses its own threshold; precision at 5 fixes the number flagged and cares only about the order of the top. The bipartite surrogate trains for the measure itself.

**“A lower Kendall tau loss means a better top of the list.”** Kendall tau counts every pair the same, wherever it sits. On the six messages of “Ranking the inbox”, a ranker that only swaps the top two messages has a Kendall tau loss of \(1/15 = 0.067\) and an NDCG loss of 0.041, while a ranker that gets the top three exactly right and reverses the bottom three has a Kendall tau loss of \(3/15 = 0.2\) and an NDCG loss of 0. If your users read the top, judge the top.

**“Dropping the 0/1 constraint gives a fractional order you cannot use.”** The relaxed problem is a linear program over doubly stochastic matrices, and the lemma shows that some optimal solution of it is a permutation matrix: a blend costs the weighted average of the permutations in it, so one of them is at least as cheap. The relaxation loses nothing.

## Recap and the next bug

1. Ranking: a linear ranker scores every message with one \(\mathbf w\). Kendall tau and NDCG measure orders; their surrogates are a pairwise hinge and a generalized hinge over permutations, whose argmax is an assignment problem, a linear program by the Birkhoff–von Neumann theorem.
2. Bipartite ranking: judge by precision, recall, F₁, precision or recall at \(k\), or AUC, never by zero-one accuracy on rare positives. The surrogate’s argmax is found by trying every \((a, b)\) with the top-\(a\) and top-\(b\) scores.
3. Certify the measure your users read. The hold-out guarantee works for any loss with values in \([0, 1]\), so each fresh morning’s precision at 5 is one example, and \(n\) fresh mornings pin the true value down to within \(\sqrt{\log(2/\delta)/(2n)}\). A certificate on accuracy, however tight, says nothing about the top five.

The folders are right and the inbox has its important messages on top. Then a letter arrives from compliance. From next quarter, every filtering decision must be explainable to an auditor: a person has to be able to read why a given message went to junk, and check that the reason is acceptable. You open the model you are so proud of and find a weight vector with four blocks of several thousand word weights each. The honest answer to “why junk?” is “because this sum of nine thousand products beat that one”, and no auditor will sign that. Is there a hypothesis class that a person can read, a short sequence of yes/no questions about a message, that is still learnable? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Take the six messages of “Ranking the inbox”, with relevances \((5, 1, 4, 0, 3, 2)\). The ideal order lists messages 1, 3, 5, 6, 2, 4 from the top. Ranker P keeps that order but swaps messages 5 and 6, so message 6 is third and message 5 fourth. Ranker Q keeps the top three right and reverses the other three, so its list reads 1, 3, 5, 4, 2, 6. Compute the Kendall tau loss and the NDCG loss with \(k = 3\) for each ranker. Which ranker does each loss prefer, and why? Then find a swap of two neighbouring messages in the ideal order that changes the Kendall tau loss but leaves the NDCG loss at 0.
2. In the ten-message inbox, a better ranker moves message 6 (important) from score \(-0.4\) to \(1.0\) and leaves every other score alone. Recompute \(a, b, c, d\) at \(\theta = 0\), F₁, precision at 3, recall at 3, the AUC and the zero-one loss. Which of these measures moved, and which did not? Explain why precision at 3 did not change even though an important message moved up.
3. On 200 fresh mornings your ranker’s measured precision at 5 is 0.68. With \(\delta = 0.05\), can you certify that its true precision at 5 is at least 0.6? How many fresh mornings would a measured 0.68 need? And why can no number of mornings certify 0.6 for a ranker whose measured precision at 5 stays near 0.55, however many mornings you draw?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (sorting maximizes the matching sum).** Let \(V\) be the set of permutations of \([r]\), written as vectors with distinct entries in \([r]\), and let \(\mathbf y' \in \mathbb R^r\) have distinct entries. Prove that \(\pi(\mathbf y') = \operatorname{argmax}_{\mathbf v \in V} \sum_{i=1}^r v_i y'_i\), and that this maximizer is unique. Then use the same idea to prove that \(G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)\) for any nondecreasing discount \(D\) and nonnegative relevances \(\mathbf y\), which is what keeps the NDCG loss in \([0, 1]\).
