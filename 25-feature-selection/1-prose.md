# Feature Selection and Generation

## The filter that reads every word

Last lesson you built a filter from a story about how mail is made. **Naive Bayes**, you will remember, pretends that once you know whether a message is junk or keep, the words in it appear independently of each other. Under that assumption the learner needs only a handful of numbers: the fraction of junk messages that contain each word, the same fraction for keep messages, and the overall share of junk. It estimates each fraction by counting with add-one pseudo-counts, \((k + 1)/(m + 2)\) for a word seen in \(k\) of \(m\) messages, so that no estimate is ever exactly 0. Then, for a new message, it multiplies the matching probabilities together for each label, one factor per word of the dictionary (present or absent), and answers with the label whose product is larger.

You shipped it with the company’s whole dictionary, 20,000 words. That is \(2 \cdot 20{,}000 + 1 = 40{,}001\) numbers to estimate, and twenty thousand factors to multiply for every message that arrives. Three complaints landed on your desk in the same week.

The first is about speed. The filter touches every word of the dictionary for every message, even though a typical message uses a hundred of them. The operations team would like it to be a lot cheaper.

The second is stranger. Every so often a perfectly ordinary message gets flagged as junk with near certainty. You trace one case and find the culprits: a few rare words, product codes that happened to appear once in your training sequence, each time in a junk message. Smoothing keeps their keep estimate off zero, but a word seen once among 800 junk messages and never among 1,200 keep messages gets the estimates \(2/802\) and \(1/1202\), so it triples the odds of junk on its own, and three such words multiply them by 27. How common is this? Here is a small simulation with a realistic shape: word frequencies fall off like \(1/r\) for the \(r\)-th most common word, each message draws about a hundred words, and the 2,000 training messages are 800 junk and 1,200 keep, all drawn from the same mix, so that no word carries any real signal and every message is junk with probability 0.4 whatever it says. The snippet trains the filter, then computes its log-odds of junk for 2,000 fresh messages, as lesson 24’s code did, by adding logs instead of multiplying: `off[j]` is word \(j\)’s term when it is absent, `on[j]` what changes when it is present. Then it counts how sure the filter is.

```python
import random
from math import exp, log

rng, d = random.Random(0), 20_000
weights = [1 / r for r in range(1, d + 1)]
draw = lambda n: [set(rng.choices(range(d), weights, k=100))
                  for _ in range(n)]
junk, keep, fresh = draw(800), draw(1200), draw(2000)
n1, n0 = [0] * d, [0] * d
for msgs, n in ((junk, n1), (keep, n0)):
    for x in msgs:
        for j in x: n[j] += 1
p1 = [(c + 1) / 802 for c in n1]
p0 = [(c + 1) / 1202 for c in n0]
off = [log((1 - a) / (1 - b)) for a, b in zip(p1, p0)]
on = [log(a / b) - o for a, b, o in zip(p1, p0, off)]
base = log(800 / 1200) + sum(off)
s = [base + sum(on[j] for j in x) for x in fresh]
prob = [1 / (1 + exp(-t)) for t in s]
only = sum(1 for a, b in zip(n1, n0) if a and not b)
print(only, sum(p > 0.5 for p in prob),
      sum(p > 0.99 for p in prob), sum(p < 0.01 for p in prob))
```

It prints `2637 237 22 976`. Purely by chance, 2,637 words showed up in junk and never in keep. An honest filter would give every fresh message a junk probability of 0.4; this one calls 237 of the 2,000 junk, 22 of them with more than 99% certainty, and is more than 99% sure that 976 others are keep. The filter has memorized its rare words, the way the memorizer of lesson 1 memorized whole messages, and smoothing only softened the memory.

The third complaint comes from a colleague who tried to fix things by switching to a linear junk score trained by **ridge regression** (lesson 13: least squares plus a price \(\lambda\|\mathbf w\|^2\) on the squared length of the weight vector), and by adding two numeric features that seemed obviously useful: the message length in bytes, and the sender’s monthly volume, which for bulk senders runs into the millions. The new filter got worse. Those two numbers are thousands or millions of times larger than a yes/no word feature, and they swamp everything else.

So the filter uses too many features, most of them noise, and a few of its features are on the wrong scale. Which features should it keep, and how should it present them to the learner? That is this lesson.

## Features are prior knowledge too

Let’s start by saying exactly what a feature is. A **feature** is any measurement of a message: whether it contains the word “winner”, how long it is, how many links it has. Formally a feature is a function from real messages to numbers, sometimes called a **feature function**, and when we stack \(d\) of them we get a **feature vector** in \(\mathbb R^d\). Everything this course has done so far started after that step: lesson 1’s four bits F, L, U, N were already a choice of features.

In lesson 1 you learned that the hypothesis class is prior knowledge. The representation is prior knowledge too, and it comes earlier. Before you choose halfspaces, or trees, or nearest neighbours, you have already decided what a message *is* to the learner. Lesson 16 made that explicit with an embedding \(\psi\), a map from messages to feature vectors, followed by halfspaces: the choice of \(\psi\) was a bet about the problem. Lesson 20 let a network’s hidden layer learn its own \(\psi\) from labels.

And here is a warning that will shape the whole lesson: how good a feature is depends on the learner that will use it. Picture two features of some numeric quantity \(y\) that a filter wants to predict. The first, \(x_1\), is drawn uniformly from \([-1, 1]\), and \(y = x_1^2\) exactly. The second is \(x_2 = y + z\), where \(z\) is a small noise drawn uniformly from \([-0.01, 0.01]\). If you could keep only one feature, which would you keep?

The first one predicts \(y\) perfectly, so it seems the obvious winner. And it is, if the learner can fit parabolas. But if the learner fits lines, a line in \(x_1\) is useless (the parabola is symmetric, so the best line is flat), while a line in \(x_2\) is almost perfect. Let’s measure the average square loss of the best line through each feature, on 100,000 draws:

```python
import random
from statistics import linear_regression, fmean

rng = random.Random(0)
x1 = [rng.uniform(-1, 1) for _ in range(100_000)]
y = [a * a for a in x1]
x2 = [b + rng.uniform(-0.01, 0.01) for b in y]

def best_line_risk(v, y):
    a, b = linear_regression(v, y)
    return fmean((a * vi + b - yi) ** 2 for vi, yi in zip(v, y))

print(round(best_line_risk(x1, y), 4))
print(round(best_line_risk(x2, y), 6))
```

It prints `0.0888` and `3.3e-05`. (The exact values are the variance of \(x_1^2\), which is \(1/5 - 1/9 = 4/45 \approx 0.0889\), and very nearly the variance of the noise, \(0.02^2/12 \approx 3.3 \times 10^{-5}\).) The feature that “contains the answer” is the bad one for a linear learner. Lesson 5’s no-free-lunch theorem has a version here: there is no feature selector or feature builder that is best for every problem and every learner. Each method below works because of some assumption, and we will try to say which.

The lesson has three parts, and the map below shows how they fit together.

[[diagram:selection-map]]

1. **Feature selection**: from a large pool of \(d\) features, keep a small number \(k\). We will see three families: filter methods, greedy selection, and sparsity-inducing norms.
2. **Feature manipulation and normalization**: change each feature’s scale or shape, by centering, rescaling, clipping or taking logarithms.
3. **Feature learning**: let the data build new features, through dictionary learning and auto-encoders.

## Why keep only a few features?

There are four good reasons to want a filter that looks at only \(k\) of its \(d\) features, with \(k\) much smaller than \(d\). A filter that uses few features needs less memory and runs faster. Some features cost something to measure: looking up the sender’s reputation means a network call, in the same way a medical test costs money, so a predictor that needs fewer of them can be worth a small loss of accuracy. And restricting the class to predictors that use few features shrinks it, which, as lessons 2 to 6 taught you, reduces the estimation error (the part of the risk that comes from learning on a finite sample) and fights overfitting. The rare-word disaster above is overfitting in its purest form.

For the linear predictors of lesson 9, a halfspace or a linear score \(\langle \mathbf w, \mathbf x \rangle\), “uses only \(k\) features” has a clean meaning: the weight vector \(\mathbf w\) has at most \(k\) nonzero entries, because a feature with weight zero never affects the output. (A reminder: \(\langle \mathbf w, \mathbf x \rangle = \sum_j w_j x_j\) is the inner product, the weighted sum of the features.) Take a weight vector over five features, \(\mathbf w = (0, 0.7, 0, -1.2, 0)\). Only features 2 and 4 matter; you never even need to measure the other three. A vector with few nonzero entries is called **sparse**, and the number of nonzero entries has a name and a symbol. With it, feature selection becomes an optimization problem:

\[ \|\mathbf w\|_0 = \left| \{ i \in [d] : w_i \neq 0 \} \right|, \qquad \min_{\mathbf w \in \mathbb R^d} L_S(\mathbf w) \ \ \text{subject to} \ \ \|\mathbf w\|_0 \le k. \]

Read the left half first, from the inside out.

- \([d]\) is the list of feature positions \(\{1, 2, \dots, d\}\).
- \(w_i \neq 0\) asks, for feature \(i\), whether its weight is nonzero, that is, whether the predictor uses it.
- The braces \(\{ i \in [d] : \dots \}\) collect the positions where the answer is yes; read the colon as “such that”.
- The bars \(|\cdot|\) count how many positions that set has.
- \(\|\mathbf w\|_0\) is the name for that count. It is often called the “\(\ell_0\) norm”, although it is not really a norm, as we will see in a moment.

Now the right half.

- \(L_S(\mathbf w)\) is the empirical risk of the predictor with weights \(\mathbf w\) on the training sequence \(S\) (lesson 1: the average loss over your labelled examples).
- \(\min_{\mathbf w \in \mathbb R^d}\) asks for the smallest value of what follows, over every possible real weight vector.
- “subject to \(\|\mathbf w\|_0 \le k\)” throws out every \(\mathbf w\) that uses more than \(k\) features before the minimum is taken.

In plain words: **among all weight vectors that use at most \(k\) features, find the one with the smallest training loss.** For our \(\mathbf w = (0, 0.7, 0, -1.2, 0)\), the set of nonzero positions is \(\{2, 4\}\), so \(\|\mathbf w\|_0 = 2\), and this \(\mathbf w\) is allowed whenever \(k \ge 2\).

Why “not really a norm”? A norm must scale: multiplying a vector by 3 must multiply its norm by 3. But tripling \(\mathbf w\) leaves the count of nonzero entries at 2, not 6. The \(\ell_1\) norm, the sum of absolute values \(\|\mathbf w\|_1 = \sum_i |w_i|\), does scale, and it will come back later as the hero of the lesson.

```python
def l0(w):
    return len([i for i in range(len(w)) if w[i] != 0])

def l1(w):
    return sum(abs(wi) for wi in w)

w = [0, 0.7, 0, -1.2, 0]
print(l0(w), l0([3 * wi for wi in w]), 3 * l0(w))
print(round(l1(w), 2), round(l1([3 * wi for wi in w]), 2))
```

It prints `2 2 6` and `1.9 5.7`: the count ignores the scaling, the \(\ell_1\) norm triples.

How would you solve the problem exactly? Try every set of \(k\) features, fit the best predictor on each, keep the best. The number of sets is the binomial coefficient \(\binom{d}{k}\), the number of ways to choose \(k\) items out of \(d\):

```python
from math import comb

d = 20_000
for k in (1, 3, 10):
    print(k, f"{comb(d, k):.3g}")
```

It prints `1 2e+04`, `3 1.33e+12` and `10 2.82e+36`. Three words out of twenty thousand is already over a trillion least-squares fits; ten is hopeless. And this is not a failure of imagination: the sparsity-constrained problem is known to be computationally hard in general (lesson 8 is where hardness of this kind lives). So we settle for methods that are cheap and often work well, without a promise of finding the best set. We take them in order of how much they know about each other: a filter method scores each feature alone; greedy selection adds features while watching the ones it already has; the \(\ell_1\) norm lets an optimizer decide everything at once.

## Filter methods: score each feature on its own

The simplest idea is a **filter method**: give every feature a score that measures how useful it is on its own, then keep the \(k\) features with the best scores (or keep every feature whose score passes a threshold). The mail filter is the product; a filter method is one way of choosing its features. The scores are computed one feature at a time, ignoring the others.

Which score? The most natural one asks: if the learner could use only this feature, how well would it do? Let’s make that concrete with linear regression and the square loss, the setting of lesson 9. Here is a training sequence of eight messages with labels \(+1\) for junk and \(-1\) for keep, and three candidate features: the number of exclamation marks (“bangs”), the number of links, and whether it was sent at night.

| message | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| label \(y\) | +1 | +1 | +1 | +1 | −1 | −1 | −1 | −1 |
| bangs | 5 | 3 | 0 | 4 | 1 | 0 | 0 | 2 |
| links | 2 | 1 | 3 | 0 | 1 | 2 | 0 | 1 |
| night | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |

Before any formula: which single feature would you bet on? Bangs are high in three of the four junk messages, links look like noise, and night is on for three junk messages and one keep message.

To score a feature, fit the best line through it. Write the feature’s eight values as a vector \(\mathbf v = (v_1, \dots, v_m)\) and the labels as \(\mathbf y = (y_1, \dots, y_m)\), with \(m = 8\). A line predicts \(a v_i + b\) for message \(i\), with a slope \(a\) and an intercept \(b\), and the score is the smallest average square loss any line can reach:

\[ \min_{a, b \in \mathbb R} \ \frac{1}{m} \left\| a\mathbf v + b - \mathbf y \right\|^2. \]

- \(a\mathbf v + b\) is the vector of predictions: each \(v_i\) multiplied by \(a\), and the number \(b\) added to every entry. (Adding a number to a vector always means adding it to each coordinate.)
- Subtracting \(\mathbf y\) gives the vector of misses, one per message.
- \(\|\cdot\|^2\) is the squared length of a vector, the sum of the squares of its entries, so \(\|a\mathbf v + b - \mathbf y\|^2 = \sum_{i=1}^m (a v_i + b - y_i)^2\), the total square loss.
- \(\frac1m\) turns the total into an average: this is the empirical risk \(L_S\) of the line.
- \(\min_{a,b \in \mathbb R}\) asks for the smallest such average over every slope and intercept.

In plain words: **the training error of the best line that uses only this feature.** For bangs, the flat line \(a = 0, b = 0\) predicts 0 everywhere, misses every label by exactly 1, and scores 1.0; the best line, which the snippet below finds with Python’s `statistics.linear_regression` (lesson 9’s least squares for one feature), scores about 0.623. With twenty thousand features we want a formula for that number. Two steps get us there.

**Step 1: centering changes nothing.** Write \(\bar v = \frac1m \sum_{i=1}^m v_i\) for the average of the feature (for bangs, \(15/8 = 1.875\)) and \(\bar y\) for the average label (here 0, four junk and four keep). Subtract the averages from both vectors before fitting, and the best achievable loss is the same:

\[ \min_{a, b \in \mathbb R} \frac{1}{m} \left\| a\mathbf v + b - \mathbf y \right\|^2 = \min_{a, b \in \mathbb R} \frac{1}{m} \left\| a(\mathbf v - \bar v) + b - (\mathbf y - \bar y) \right\|^2. \]

- \(\mathbf v - \bar v\) is the **centered** feature: each value minus the average, so its entries add up to 0. Same for \(\mathbf y - \bar y\).
- The left side is the best line on the raw data; the right side is the best line on the centered data.

In plain words: **shifting the feature and the labels to average zero does not change how well a line can fit them.** That is believable: a line can absorb any shift through its intercept. Proving it carefully is the first problem at the end of this lesson. Here it is checked on bangs, by fitting both sides exactly:

```python
from statistics import fmean, linear_regression

y = [1, 1, 1, 1, -1, -1, -1, -1]
v = [5, 3, 0, 4, 1, 0, 0, 2]
vc = [vi - fmean(v) for vi in v]
yc = [yi - fmean(y) for yi in y]

def best_loss(v, y):
    a, b = linear_regression(v, y)
    return fmean((a * p + b - q) ** 2 for p, q in zip(v, y))

print(round(best_loss(v, y), 4), round(best_loss(vc, yc), 4))
print(linear_regression(vc, yc))
```

It prints `0.6233 0.6233`, and the best line on the centered data has intercept exactly 0 and slope about 0.335. Both facts are about to be proved.

**Step 2: solve the centered problem.** On centered data, write \(\mathbf v' = \mathbf v - \bar v\) and \(\mathbf y' = \mathbf y - \bar y\); both have entries summing to zero. The objective is \(g(a, b) = \frac1m \sum_i (a v'_i + b - y'_i)^2\). It is a convex quadratic in \((a, b)\) (lesson 12), so its minimum is where both partial derivatives vanish.

The derivative in \(b\) is \(\frac2m \sum_i (a v'_i + b - y'_i) = \frac2m \left(a \sum_i v'_i + m b - \sum_i y'_i\right) = 2b\), because the two centered sums are zero. So \(b = 0\): on centered data, the best line passes through the origin.

With \(b = 0\), the objective is \(\frac1m \|a\mathbf v' - \mathbf y'\|^2 = \frac1m\left(a^2 \|\mathbf v'\|^2 - 2a \langle \mathbf v', \mathbf y' \rangle + \|\mathbf y'\|^2\right)\), by expanding the square. Setting its derivative in \(a\), \(\frac1m(2a\|\mathbf v'\|^2 - 2\langle \mathbf v', \mathbf y' \rangle)\), to zero gives \(a = \langle \mathbf v', \mathbf y' \rangle / \|\mathbf v'\|^2\) (we assume the feature is not constant, so \(\|\mathbf v'\| > 0\)). Plug it back in: the middle term becomes \(-2\langle \mathbf v', \mathbf y' \rangle^2/\|\mathbf v'\|^2\) and the first becomes \(+\langle \mathbf v', \mathbf y' \rangle^2/\|\mathbf v'\|^2\), so they combine into one negative term. The best loss is

\[ \frac{1}{m}\left( \|\mathbf y - \bar y\|^2 - \frac{\langle \mathbf v - \bar v, \ \mathbf y - \bar y \rangle^2}{\|\mathbf v - \bar v\|^2} \right). \]

- \(\|\mathbf y - \bar y\|^2\) is the total square loss of the flat line at height \(\bar y\): how much the labels vary.
- \(\langle \mathbf v - \bar v, \mathbf y - \bar y \rangle\) is the inner product of the centered feature with the centered labels. It is large and positive when the feature is above its average exactly where the label is above its average.
- Dividing its square by \(\|\mathbf v - \bar v\|^2\) measures how much of the labels’ variation a line in this feature can explain.

In plain words: **the best line’s error is the labels’ total variation, minus the part the feature explains.** For bangs: the labels vary by \(\|\mathbf y - \bar y\|^2 = 8\), the inner product is 9, the feature’s centered squared length is 26.875, so the loss is \((8 - 81/26.875)/8 \approx 0.6233\), the exact fit’s answer, and the slope is \(9/26.875 \approx 0.335\).

```python
from statistics import fmean

y = [1, 1, 1, 1, -1, -1, -1, -1]
v = [5, 3, 0, 4, 1, 0, 0, 2]
m = len(y)
vc = [vi - fmean(v) for vi in v]
yc = [yi - fmean(y) for yi in y]
inner = sum(p * q for p, q in zip(vc, yc))
norm2_v = sum(p * p for p in vc)
norm2_y = sum(q * q for q in yc)
a = inner / norm2_v
print(round(inner, 4), round(norm2_v, 4), norm2_y, round(a, 4))
print(round((norm2_y - inner ** 2 / norm2_v) / m, 4))
```

It prints `9.0 26.875 8.0 0.3349` and `0.6233`, line for line the formula.

Now look at which part depends on the feature. The first term, \(\|\mathbf y - \bar y\|^2\), is the same for every feature. So a feature is better exactly when \(\langle \mathbf v - \bar v, \mathbf y - \bar y \rangle^2/\|\mathbf v - \bar v\|^2\) is larger. Divide that by the constant \(\|\mathbf y - \bar y\|^2\) and take a square root, and you get a number with a famous name.

**Proposition (the filter score is a correlation).** Ranking features by the training error of their best single-feature line, smallest first, is the same as ranking them by the absolute value of

\[ \rho(\mathbf v, \mathbf y) = \frac{\langle \mathbf v - \bar v, \ \mathbf y - \bar y \rangle}{\|\mathbf v - \bar v\| \, \|\mathbf y - \bar y\|} = \frac{\frac1m \langle \mathbf v - \bar v, \ \mathbf y - \bar y \rangle}{\sqrt{\frac1m \|\mathbf v - \bar v\|^2} \ \sqrt{\frac1m \|\mathbf y - \bar y\|^2}}, \]

largest first. Moreover \(-1 \le \rho \le 1\), and \(|\rho| = 1\) exactly when some line in \(\mathbf v\) fits \(\mathbf y\) with zero error.

- \(\rho\) (Greek “rho”) is **Pearson’s correlation coefficient**.
- In the second form, the numerator is the empirical covariance of the feature and the label: the average of the product of their deviations from their averages.
- Each square root in the denominator is an empirical standard deviation: the square root of an average squared deviation.

In plain words: **a filter that keeps the features whose best line fits best is keeping the features most correlated with the label.**

*Proof.* By the two steps, the best loss is \(\frac1m \|\mathbf y - \bar y\|^2 (1 - \rho^2)\): factor \(\|\mathbf y - \bar y\|^2\) out of the bracket and recognize \(\rho^2\) in what remains. Since \(\frac1m\|\mathbf y - \bar y\|^2\) is positive and the same for every feature (we assume the labels are not all equal), smaller loss means larger \(\rho^2\), which means larger \(|\rho|\). The Cauchy–Schwarz inequality, \(|\langle \mathbf p, \mathbf q \rangle| \le \|\mathbf p\| \|\mathbf q\|\), gives \(|\rho| \le 1\), and the loss formula shows \(|\rho| = 1\) exactly when the best loss is zero. \(\blacksquare\)

Here are all three features of the table, scored by Python’s `statistics.correlation`, which computes exactly \(\rho\), next to the best loss \(\frac1m\|\mathbf y - \bar y\|^2(1 - \rho^2) = 1 - \rho^2\) (here \(\frac1m\|\mathbf y - \bar y\|^2 = 8/8 = 1\)):

```python
from statistics import correlation

y = [1, 1, 1, 1, -1, -1, -1, -1]
features = {"bangs": [5, 3, 0, 4, 1, 0, 0, 2],
            "links": [2, 1, 3, 0, 1, 2, 0, 1],
            "night": [1, 1, 0, 1, 0, 1, 0, 0]}
for name, v in features.items():
    rho = correlation(v, y)
    print(name, round(rho, 4), round(1 - rho ** 2, 4))
```

It prints `bangs 0.6138 0.6233`, `links 0.2582 0.9333`, `night 0.5 0.75`: bangs gets back the loss of 0.6233 found above. A filter method with \(k = 1\) keeps bangs, and with \(k = 2\) keeps bangs and night. Scoring each feature costs one pass over its \(m\) values, so the whole filter method costs \(O(dm)\) time plus \(O(d \log d)\) to sort the scores, and \(O(d)\) extra space. Cheap enough for twenty thousand words.

Pearson’s coefficient is not the only score in use. Two other popular ones are the **mutual information** between a feature and the label (lesson 24’s relative entropy from the joint distribution of the pair to the product of their separate distributions: how much knowing the feature reduces the surprise about the label) and the **area under the ROC curve** of the feature (the chance that a random junk message has a higher value of it than a random keep message). They all share the weakness we are about to find.

## When a lonely score lies

A correlation of zero says that the best line in this feature alone is flat: the feature, alone, cannot help a linear predictor. It does not say the feature is useless. Here is a case where the filter method throws away the one feature you cannot do without.

Your mail system has an old hand-tuned score from last year, “old”, which nobody documented. Unknown to you, last year’s engineer built it as half the junk signal minus half the link indicator, because back then links mostly meant newsletters. Write junk as \(y = \pm 1\), and “has a link” as \(\pm 1\), independent of junk. Then old \(= (y - \text{link})/2\), and therefore \(y = \text{link} + 2 \cdot \text{old}\) exactly. Here are eight messages covering every combination twice, with a third feature, bangs, that agrees with the label on six of the eight.

| message | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| label \(y\) | +1 | +1 | +1 | +1 | −1 | −1 | −1 | −1 |
| link | +1 | −1 | +1 | −1 | +1 | −1 | +1 | −1 |
| old | 0 | 1 | 0 | 1 | −1 | 0 | −1 | 0 |
| bangs | +1 | +1 | +1 | −1 | −1 | −1 | −1 | +1 |

Predict before you compute: what correlation does link get, and which two features will a filter keep?

```python
from statistics import correlation

y = [1, 1, 1, 1, -1, -1, -1, -1]
link = [1, -1, 1, -1, 1, -1, 1, -1]
old = [0, 1, 0, 1, -1, 0, -1, 0]
bangs = [1, 1, 1, -1, -1, -1, -1, 1]
for name, v in (("link", link), ("old", old), ("bangs", bangs)):
    print(name, round(correlation(v, y), 4))
fit = [1 * a + 2 * b for a, b in zip(link, old)]
print(fit == y)
```

It prints `link 0.0`, `old 0.7071`, `bangs 0.5`, and `True`. Link has correlation exactly zero, so the filter method keeps old and bangs, and the best least-squares fit on those two, \(w_{\text{old}}\cdot\text{old} + w_{\text{bangs}}\cdot\text{bangs}\), still has average square loss \(3/7 \approx 0.43\) (you will reproduce it in the build-it challenge). Meanwhile link together with old fits every label perfectly. No predictor can do well here without link, and the filter ranks it last.

[[diagram:filter-blind]]

Why zero? Let’s compute it on the distribution itself rather than on eight messages. Let link be a fair \(\pm1\) coin \(x_1\), let \(z\) be an independent fair \(\pm1\) coin (the junk signal), and set \(x_2 = -\tfrac12 x_1 + \tfrac12 z\) (this is old) and \(y = x_1 + 2x_2\). All three have mean zero, so the correlation of \(x_1\) with \(y\) is zero exactly when \(\mathbb E[y x_1] = 0\). Substitute \(y = x_1 + 2x_2\), then \(x_2\), and use linearity of expectation:

\[ \mathbb E[y x_1] = \mathbb E[x_1^2] + 2\,\mathbb E[x_2 x_1] = \mathbb E[x_1^2] - \mathbb E[x_1^2] + \mathbb E[z x_1] = 0. \]

- \(\mathbb E[\cdot]\) is the expectation: the average of the quantity inside over the random draw, weighted by probability.
- \(\mathbb E[x_1^2] = 1\), because \(x_1 = \pm 1\) always squares to 1.
- \(2\,\mathbb E[x_2 x_1] = 2\,\mathbb E[(-\tfrac12 x_1 + \tfrac12 z) x_1] = -\mathbb E[x_1^2] + \mathbb E[z x_1]\), by multiplying out.
- \(\mathbb E[z x_1] = \mathbb E[z]\,\mathbb E[x_1] = 0\), because independent variables have expected product equal to the product of their expectations, and each has mean zero.

In plain words: **link’s direct contribution to the label is cancelled exactly by its hidden contribution through old, so on its own it looks unrelated.** With the four equally likely values of \((x_1, z)\), Python computes each piece:

```python
from itertools import product
from statistics import fmean

cases = list(product([-1, 1], [-1, 1]))
x1 = [a for a, z in cases]
x2 = [-a / 2 + z / 2 for a, z in cases]
y = [p + 2 * q for p, q in zip(x1, x2)]
E = lambda u: fmean(u)
print(E(x1), E(x2), E(y))
print(E([p * p for p in x1]),
      2 * E([q * p for p, q in zip(x1, x2)]),
      E([yi * p for yi, p in zip(y, x1)]))
print(E([yi * q for yi, q in zip(y, x2)]))
```

It prints `0.0 0.0 0.0`, then `1.0 -1.0 0.0` (the two pieces and their sum), then `0.5` for \(\mathbb E[y x_2]\). With a large enough training sequence, link’s Pearson score will be close to zero and it will almost certainly not be selected. The mutual-information and ROC scores suffer from the same blindness: any score that looks at one feature at a time cannot see a feature whose value lies only in combination.

The lab below puts you in this world, with link, old, bangs and a few dozen noise words, and lets you run the filter method with any \(k\) and compare it with a set of features you pick by hand.

[[lab]]

## Greedy selection: add the feature that helps most now

The filter’s mistake was to judge each feature in isolation. The fix is to judge each candidate *next to the features already chosen*, by actually running the learner. That is **forward greedy selection**. Start with no features. In each round, try adding each feature not yet chosen, fit the learner on the chosen set plus that one candidate, and keep the candidate whose fitted predictor has the smallest risk, measured on the training sequence or, better, on a validation set as in lesson 11. Stop when you have \(k\) features, where \(k\) is your budget, or when the predictor is accurate enough.

Let’s watch it on the cancelling world. Round 1 fits each feature alone, which is exactly the filter method’s question: old wins with error \(4/8 = 0.5\). Round 2 tries old plus link, and old plus bangs, and old plus link fits perfectly. The greedy method finds the pair the filter method could not, because in round 2 it asks the right question: given old, which feature helps?

Now let’s make the rounds precise for linear regression with the square loss. Put the \(m\) training messages as the rows of a matrix \(X\), so \(X\) has \(m\) rows and \(d\) columns, and write \(X_j\) for its \(j\)-th column, the values of feature \(j\) on all \(m\) messages. For a set of feature positions \(I\), let \(X_I\) be the matrix made of just the columns in \(I\). We drop the intercept here, as lesson 9 did with its homogeneous form: you can always include a constant feature among the columns.

Here is a smaller example we will follow through the next three subsections: six messages, labels \(\mathbf y = (1, 1, 1, -1, -1, -1)\), and three features,

\[ X_1 = (1, 1, 0, 0, -1, 0), \quad X_2 = (2, 0, 1, -1, 0, -2), \quad X_3 = (0, 1, 1, 0, 0, -1). \]

Each is a column of six numbers, one per message. Round \(t\) of forward greedy selection picks

\[ j_t \in \operatorname*{argmin}_{j \notin I_{t-1}} \ \min_{\mathbf w \in \mathbb R^{t}} \left\| X_{I_{t-1} \cup \{j\}} \mathbf w - \mathbf y \right\|^2, \qquad I_t = I_{t-1} \cup \{j_t\}. \]

Read it from the inside out.

- \(I_{t-1}\) is the set of features chosen in the first \(t - 1\) rounds; \(I_0\) is empty.
- \(I_{t-1} \cup \{j\}\) is that set with the candidate \(j\) added, \(t\) features in all.
- \(X_{I_{t-1} \cup \{j\}} \mathbf w\) is the prediction of a linear predictor that uses only those \(t\) columns, with weights \(\mathbf w \in \mathbb R^t\).
- \(\|\cdot - \mathbf y\|^2\) is its total square loss on the training sequence.
- \(\min_{\mathbf w}\) fits the weights: this is a least-squares fit, lesson 9’s algorithm, on \(t\) columns.
- \(\operatorname{argmin}_{j \notin I_{t-1}}\) returns the candidate \(j\) (among those not already chosen) whose fitted loss is smallest. The symbol \(\in\) is there because several candidates can tie; any one of them will do.

In plain words: **in each round, add the feature that, fitted together with the ones you already have, gives the smallest training loss.** In round 1 of the six-message example, each single column is fitted by the weight \(w = \langle X_j, \mathbf y \rangle / \|X_j\|^2\) (the same calculation as the slope in the filter-method section, without centering), and the losses are:

```python
def dot(a, b): return sum(p * q for p, q in zip(a, b))

y = [1, 1, 1, -1, -1, -1]
X = {1: [1, 1, 0, 0, -1, 0],
     2: [2, 0, 1, -1, 0, -2],
     3: [0, 1, 1, 0, 0, -1]}
for j, Xj in X.items():
    w = dot(Xj, y) / dot(Xj, Xj)
    r = [w * p - q for p, q in zip(Xj, y)]
    print(j, round(w, 4), round(dot(r, r), 4))
```

It prints `1 1.0 3.0`, `2 0.6 2.4`, `3 1.0 3.0`. Starting from a loss of \(\|\mathbf y\|^2 = 6\) for the empty predictor, feature 2 brings it down to 2.4, the most, so \(I_1 = \{2\}\).

The cost is the catch. Round \(t\) solves up to \(d\) least-squares problems with \(t\) columns each, about \(O(mt^2 + t^3)\) apiece, so \(k\) rounds cost about \(O(d k^3 m)\) in total when \(k \le m\). With twenty thousand candidates that is a lot of refitting. Can we decide which candidate wins *without* refitting all of them? Yes, for least squares, and the trick is geometry.

## Orthogonal matching pursuit

The trick needs two ideas from linear algebra, which we build first.

Two vectors are **orthogonal** when their inner product is 0, like the axes of a graph. A list of vectors is **orthonormal** when each has length 1 and every pair is orthogonal. If the columns of a matrix \(V\) are orthonormal, then \(V^\top V = I\), the identity matrix, because entry \((a, b)\) of \(V^\top V\) is the inner product of columns \(a\) and \(b\). An orthonormal list that spans the same space as the chosen columns of \(X\) is called an **orthonormal basis** of that space: every combination of chosen columns is also a combination of basis vectors, and vice versa.

The **projection** of a vector \(\mathbf p\) onto the space spanned by the orthonormal columns of \(V\) is \(VV^\top \mathbf p = \sum_a \langle \mathbf v_a, \mathbf p \rangle \mathbf v_a\): for each basis vector, measure how much of \(\mathbf p\) points along it, and add up those pieces. What is left over, \(\mathbf u = \mathbf p - VV^\top \mathbf p\), is orthogonal to every column of \(V\): indeed \(V^\top \mathbf u = V^\top \mathbf p - (V^\top V) V^\top \mathbf p = V^\top \mathbf p - V^\top \mathbf p = \mathbf 0\). That one line is where \(V^\top V = I\) gets used, and the whole method rests on it.

In our example, after round 1 the chosen column is \(X_2\), with length \(\sqrt{10}\). One basis vector suffices: \(\mathbf v_1 = X_2/\sqrt{10}\). For a candidate such as \(X_1\), the part of \(X_1\) that \(X_2\) can already produce is its projection onto \(\mathbf v_1\), and the leftover \(\mathbf u_1 = X_1 - \langle \mathbf v_1, X_1 \rangle \mathbf v_1\) is what \(X_1\) adds that is genuinely new.

[[diagram:omp-projection]]

**Orthogonal matching pursuit (OMP)** keeps such an orthonormal basis of the chosen columns and, in each round, scores every candidate by its new part. Before round \(t\), let the columns of \(V_{t-1}\) be an orthonormal basis of the span of the chosen columns, and let \(\boldsymbol\theta_{t-1}\) be the best coefficients on that basis, so that \(V_{t-1}\boldsymbol\theta_{t-1}\) is the best prediction the chosen features can make. For each candidate \(j\), split \(X_j = V_{t-1}V_{t-1}^\top X_j + \mathbf u_j\) into its projection and its new part \(\mathbf u_j\).

**Proposition (OMP picks what forward greedy picks).** For every candidate \(j\) with \(\mathbf u_j \neq \mathbf 0\),

\[ \min_{\mathbf w \in \mathbb R^{t}} \left\| X_{I_{t-1} \cup \{j\}} \mathbf w - \mathbf y \right\|^2 = \left\| V_{t-1}\boldsymbol\theta_{t-1} - \mathbf y \right\|^2 - \frac{\langle \mathbf u_j, \mathbf y \rangle^2}{\|\mathbf u_j\|^2}, \]

so forward greedy selection’s choice is \(j_t \in \operatorname{argmax}_j \langle \mathbf u_j, \mathbf y \rangle^2/\|\mathbf u_j\|^2\) over the candidates with \(\mathbf u_j \neq \mathbf 0\). (A candidate with \(\mathbf u_j = \mathbf 0\) lies in the span already, and adding it leaves the loss unchanged.)

- The left side is forward greedy’s round-\(t\) score for candidate \(j\): the loss after refitting.
- \(\|V_{t-1}\boldsymbol\theta_{t-1} - \mathbf y\|^2\) is the loss of the current predictor, before adding anything. It is the same for every candidate.
- \(\langle \mathbf u_j, \mathbf y \rangle^2 / \|\mathbf u_j\|^2\) is how much the new part of feature \(j\) can explain of the labels, the same shape as the filter method’s score but computed on the *new part*; \(\operatorname{argmax}_j\) returns the candidate where it is largest.

In plain words: **adding a feature lowers the loss by exactly the squared alignment of its new part with the labels, so the best feature to add is the one whose new part lines up best.**

*Proof.* Fix a candidate \(j\) with \(\mathbf u_j \ne \mathbf 0\), and write \(V = V_{t-1}\), \(\mathbf u = \mathbf u_j\).

*Same space.* The columns of \(X_{I_{t-1} \cup \{j\}}\) span the same space as the columns of \(V\) together with \(\mathbf u\). Each chosen column is a combination of the columns of \(V\) by the definition of a basis; \(X_j = VV^\top X_j + \mathbf u\) is a combination of them and \(\mathbf u\); and conversely \(\mathbf u = X_j - VV^\top X_j\) is a combination of \(X_j\) and chosen columns. Minimizing the distance from \(\mathbf y\) over the same set of vectors gives the same minimum, so the left side equals \(\min_{\boldsymbol\theta, \alpha} \|V\boldsymbol\theta + \alpha\mathbf u - \mathbf y\|^2\).

*Expand.* For any \(\boldsymbol\theta\) and number \(\alpha\), expanding the square of a sum gives
\(\|V\boldsymbol\theta - \mathbf y + \alpha\mathbf u\|^2 = \|V\boldsymbol\theta - \mathbf y\|^2 + \alpha^2\|\mathbf u\|^2 + 2\alpha\langle \mathbf u, V\boldsymbol\theta - \mathbf y \rangle.\)

*The cross term loses \(\boldsymbol\theta\).* Since \(\mathbf u\) is orthogonal to every column of \(V\), \(\langle \mathbf u, V\boldsymbol\theta \rangle = \langle V^\top \mathbf u, \boldsymbol\theta \rangle = 0\). So the cross term is \(-2\alpha\langle \mathbf u, \mathbf y \rangle\), which does not involve \(\boldsymbol\theta\).

*Minimize the two parts separately.* The expression is now \(\|V\boldsymbol\theta - \mathbf y\|^2 + (\alpha^2\|\mathbf u\|^2 - 2\alpha\langle \mathbf u, \mathbf y \rangle)\): one part depends only on \(\boldsymbol\theta\), the other only on \(\alpha\), so the joint minimum is the sum of the two separate minima. The first is reached at \(\boldsymbol\theta_{t-1}\) by definition. The second is the parabola of the filter method’s step 2, with \(\mathbf u\) in place of \(\mathbf v'\): it is smallest at \(\alpha = \langle \mathbf u, \mathbf y \rangle/\|\mathbf u\|^2\), with value \(-\langle \mathbf u, \mathbf y \rangle^2/\|\mathbf u\|^2\). Adding the two minima gives the formula.

*Conclude.* The first term of the formula is the same for every candidate, so the smallest refitted loss belongs to the largest \(\langle \mathbf u_j, \mathbf y \rangle^2/\|\mathbf u_j\|^2\). \(\blacksquare\)

The update after choosing \(j_t\) follows from the proof: append the unit vector \(\mathbf u_{j_t}/\|\mathbf u_{j_t}\|\) to the basis, and append its best coefficient, \(\alpha\|\mathbf u_{j_t}\| = \langle \mathbf u_{j_t}, \mathbf y \rangle/\|\mathbf u_{j_t}\|\), to \(\boldsymbol\theta\). One more remark: since \(\langle \mathbf u, V\boldsymbol\theta_{t-1} \rangle = 0\), the inner product \(\langle \mathbf u_j, \mathbf y \rangle\) equals \(\langle \mathbf u_j, \mathbf y - V\boldsymbol\theta_{t-1} \rangle\), the alignment with the current **residual**, the part of the labels not yet explained. That is where the name comes from: OMP *matches* the residual against each feature’s new part.

Let’s check the proposition on round 2 of the example, refitting both candidates by brute force (solving the two-by-two least-squares system) and comparing with the right side:

```python
def dot(a, b): return sum(p * q for p, q in zip(a, b))
def comb(s, a, b): return [s * p + q for p, q in zip(a, b)]

y = [1, 1, 1, -1, -1, -1]
X2 = [2, 0, 1, -1, 0, -2]
X = {1: [1, 1, 0, 0, -1, 0], 3: [0, 1, 1, 0, 0, -1]}
V = [p / dot(X2, X2) ** 0.5 for p in X2]
r = comb(-dot(V, y), V, y)
for j, Xj in X.items():
    u = comb(-dot(V, Xj), V, Xj)
    a, b, c = dot(X2, X2), dot(X2, Xj), dot(Xj, Xj)
    p, q, det = dot(X2, y), dot(Xj, y), a * c - b * b
    w2, wj = (c * p - b * q) / det, (a * q - b * p) / det
    e = comb(-1, y, comb(wj, Xj, [w2 * t for t in X2]))
    print(j, round(dot(e, e), 4),
          round(dot(r, r) - dot(u, y) ** 2 / dot(u, u), 4))
```

It prints `1 1.1538 1.1538` and `3 1.7143 1.7143`: the refitted loss and the formula agree for both candidates, and feature 1 wins round 2, bringing the loss from 2.4 down to about 1.15.

Here is OMP as pseudocode. The step-by-step subtraction can lose accuracy in floating point over many rounds, so implementations often rebuild the basis with a more stable method; the choices are the same.

```
OMP(X, y, T):
  I ← empty; V ← empty list of basis vectors
  for t = 1..T:
    for each j not in I:
      u_j ← X_j − Σ_{v in V} ⟨v, X_j⟩ v
    j_t ← argmax over j ∉ I, u_j ≠ 0, of ⟨u_j, y⟩² / ‖u_j‖²
    append u_{j_t} / ‖u_{j_t}‖ to V; add j_t to I
  return I
```

As written, round \(t\) costs \(O(dmt)\), since each of up to \(d\) candidates is projected onto \(t - 1\) basis vectors of length \(m\). If you keep each \(\mathbf u_j\) from the previous round and remove only its component along the newest basis vector, a round costs \(O(dm)\) and \(T\) rounds cost \(O(Tdm)\) time, with \(O(dm)\) space for the stored new parts. Compare forward greedy’s \(O(dT^3m)\): same choices, far cheaper.

## Cheaper greedy criteria

OMP is special to the square loss. For a general empirical risk \(R(\mathbf w)\), such as the logistic loss of lesson 9, two cheaper versions of the greedy round are common.

The first keeps the old weights frozen and optimizes only the newcomer: pick the \(j\) for which \(\min_{\eta \in \mathbb R} R(\mathbf w_{t-1} + \eta\mathbf e_j)\) is smallest, where \(\mathbf e_j\) is the vector with 1 in position \(j\) and 0 elsewhere. Each candidate costs a one-dimensional minimization instead of a \(t\)-dimensional one.

The second is cheaper still: pick the coordinate where the gradient is steepest. Why should that work? Recall smoothness from lesson 12: a function is \(\beta\)-smooth when its gradient cannot change faster than \(\beta\) times the distance moved, and such a function sits below a parabola that touches it. Moving along coordinate \(j\) by a step \(\eta\), that parabola reads

\[ R(\mathbf w + \eta \mathbf e_j) \le R(\mathbf w) + \eta \, \frac{\partial R(\mathbf w)}{\partial w_j} + \frac{\beta\eta^2}{2}, \qquad\text{and at the best } \eta: \qquad R(\mathbf w + \eta \mathbf e_j) \le R(\mathbf w) - \frac{1}{2\beta}\left(\frac{\partial R(\mathbf w)}{\partial w_j}\right)^2. \]

- \(\frac{\partial R(\mathbf w)}{\partial w_j}\) is the partial derivative: how fast the risk changes as you nudge weight \(j\) alone.
- \(\eta \, \partial R/\partial w_j\) is the first-order change of a step of size \(\eta\); \(\beta\eta^2/2\) is the most that curvature can add on top.
- The right-hand side of the first inequality is a parabola in \(\eta\). Its derivative \(\partial R/\partial w_j + \beta\eta\) vanishes at \(\eta = -\frac1\beta \partial R/\partial w_j\); substituting gives \(-\frac1\beta(\partial R/\partial w_j)^2 + \frac1{2\beta}(\partial R/\partial w_j)^2 = -\frac1{2\beta}(\partial R/\partial w_j)^2\), the second inequality.

In plain words: **a step along coordinate \(j\) is guaranteed to lower the risk by at least the square of that coordinate’s slope, divided by \(2\beta\); so the coordinate with the largest slope in absolute value has the best guarantee.** For the square risk \(R(\mathbf w) = \frac1m\|X\mathbf w - \mathbf y\|^2\) at \(\mathbf w = \mathbf 0\), the slope along \(j\) is \(-\frac2m\langle X_j, \mathbf y \rangle\), and along coordinate \(j\) the curvature is \(\frac2m\|X_j\|^2\), so \(\beta = \max_j \frac2m \|X_j\|^2\) works for every coordinate. On the six-message example:

```python
def dot(a, b): return sum(p * q for p, q in zip(a, b))

y = [1, 1, 1, -1, -1, -1]
X = {1: [1, 1, 0, 0, -1, 0], 2: [2, 0, 1, -1, 0, -2],
     3: [0, 1, 1, 0, 0, -1]}
m = len(y)
R0 = dot(y, y) / m
beta = max(2 * dot(c, c) / m for c in X.values())
for j, Xj in X.items():
    g = -2 * dot(Xj, y) / m
    sure = g ** 2 / (2 * beta)
    best = dot(Xj, y) ** 2 / dot(Xj, Xj) / m
    print(j, round(g, 4), round(sure, 4), round(best, 4))
print(round(R0, 4), round(beta, 4))
```

It prints `1 -1.0 0.15 0.5`, `2 -2.0 0.6 0.6`, `3 -1.0 0.15 0.5`, then `1.0 3.3333`. Each row is the slope, the guaranteed decrease and the actual best decrease along that coordinate (in average loss, so divide the earlier totals by 6). The steepest coordinate is feature 2, the same choice forward greedy made, and for it the guarantee is tight. For features 1 and 3 the guarantee is loose, because \(\beta\) was set by feature 2’s larger curvature. The gradient rule costs one gradient evaluation, \(O(dm)\), per round.

If this “pick the best coordinate, then step” loop feels familiar, it should. AdaBoost from lesson 10 is exactly this greedy rule, run over a list of weak hypotheses \(h_1, \dots, h_d\) treated as features, on the risk \(R(\mathbf w) = \log \sum_{i=1}^m \exp\left(-y_i \sum_j w_j h_j(x_i)\right)\). Inside, \(\sum_j w_j h_j(x_i)\) is the weighted vote on example \(i\); multiplied by the label \(y_i = \pm1\), it is positive when the vote is right, so each \(\exp(\cdot)\) is below 1 for an example voted right and above 1 for one voted wrong, and the log of the sum is a smooth count of the mistakes. With all weights 0, every term is \(e^0 = 1\) and \(R = \log m\). The second problem at the end of this lesson walks you through why.

## Backward elimination

Greedy selection can also run in reverse. **Backward elimination** starts with all \(d\) features and removes one per round: given the current set \(I\), try removing each \(i \in I\), fit the learner on \(I \setminus \{i\}\) (the set without \(i\)), and remove the feature whose absence hurts least, again judged on the training sequence or on a validation set. Stop at \(k\) features.

```
BackwardElimination(X, y, k):
  I ← {1, ..., d}
  while |I| > k:
    i* ← argmin over i in I of LeastSquaresLoss(X, y, I \ {i})
    I ← I \ {i*}
  return I
```

In the cancelling world with a fourth, pure-noise feature added (as in the build-it starter below), backward elimination with \(k = 2\) drops bangs and the noise feature and ends at link and old: at every round, removing either of those two would destroy the perfect fit, while the others cost nothing. It sees interactions from the start, since every feature is in the fit. The price is steep: the first round fits \(d\) problems with \(d - 1\) columns, about \(O(d \cdot (md^2 + d^3))\), which is \(O(d^4)\) when \(m\) is comparable to \(d\), and there are up to \(d\) rounds, so the full run is roughly \(O(d^5)\). Fine for fifty features, out of the question for twenty thousand. Many variants exist, including methods that alternate forward and backward steps.

## Sparsity-inducing norms

Greedy methods build the sparse predictor one decision at a time. The third approach changes the problem instead, so that an ordinary convex optimizer returns sparse solutions on its own.

The trouble with the sparsity constraint \(\|\mathbf w\|_0 \le k\) is that it is not convex: the set of vectors with at most one nonzero entry, in the plane, is the union of the two axes, and the midpoint of \((1, 0)\) and \((0, 1)\) is not on it. Lesson 12 taught you that convexity is what makes minimization reliable. So replace the count by the closest convex stand-in, the \(\ell_1\) norm \(\|\mathbf w\|_1 = \sum_i |w_i|\), which you met above (1.9 for our five-feature \(\mathbf w\)). There are two ways to use it, a hard budget and a price:

\[ \min_{\mathbf w} \ L_S(\mathbf w) \ \ \text{subject to} \ \ \|\mathbf w\|_1 \le k_1 \qquad\qquad \min_{\mathbf w} \ \left( L_S(\mathbf w) + \lambda\|\mathbf w\|_1 \right). \]

- \(k_1\) is a budget on the total absolute weight; \(\lambda > 0\) is a **regularization parameter**, a price per unit of absolute weight, as in lesson 13’s regularized loss minimization, which used \(\lambda\|\mathbf w\|^2\) instead.
- If \(L_S\) is convex, both problems are convex, because \(\|\cdot\|_1\) is a convex function, and can be solved efficiently.

In plain words: **either fit as well as you can while spending at most \(k_1\) of total absolute weight, or fit while paying \(\lambda\) for every unit you spend.** The two are closely related, and one direction is easy to prove.

**Claim (a price gives a budget).** If \(\mathbf w_\lambda\) minimizes the penalized problem, then \(\mathbf w_\lambda\) also solves the budgeted problem with \(k_1 = \|\mathbf w_\lambda\|_1\).

*Proof.* Take any \(\mathbf w\) with \(\|\mathbf w\|_1 \le k_1\). Because \(\mathbf w_\lambda\) minimizes the penalized objective, \(L_S(\mathbf w) + \lambda\|\mathbf w\|_1 \ge L_S(\mathbf w_\lambda) + \lambda k_1\). Rearranged, \(L_S(\mathbf w) \ge L_S(\mathbf w_\lambda) + \lambda(k_1 - \|\mathbf w\|_1) \ge L_S(\mathbf w_\lambda)\), since \(\lambda > 0\) and \(k_1 - \|\mathbf w\|_1 \ge 0\). And \(\mathbf w_\lambda\) itself meets the budget. \(\blacksquare\)

The other direction, that every budget \(k_1\) is matched by some price \(\lambda\), also holds for convex losses, but its proof needs Lagrange duality, which this course does not develop. So you may think in budgets and compute with prices.

### Why the \(\ell_1\) price produces zeros

Why should this give *sparse* answers, with weights exactly zero? Let’s look at the smallest possible case, one weight and a simple quadratic. Suppose a feature’s evidence for the label is summarized by one number \(x\), and you choose a weight \(w\) by minimizing a parabola centred at \(x\) plus the \(\ell_1\) price. Before the formula, picture it: the parabola pulls \(w\) towards \(x\) with a force that weakens as \(w\) approaches \(x\), while the price \(\lambda|w|\) pushes towards zero with a constant force \(\lambda\), whatever the size of \(w\). Near zero, the parabola’s pull is about \(|x|\). If \(|x|\) is smaller than \(\lambda\), the constant push wins all the way and \(w\) stops exactly at zero.

**Proposition (soft thresholding).** For \(x \in \mathbb R\) and \(\lambda > 0\),

\[ \operatorname*{argmin}_{w \in \mathbb R} \left( \tfrac12 w^2 - xw + \lambda|w| \right) = \operatorname{sign}(x)\,\big[\,|x| - \lambda\,\big]_+, \qquad [a]_+ = \max\{a, 0\}. \]

- \(\tfrac12 w^2 - xw\) equals \(\tfrac12 (w - x)^2\) minus a constant, so it is the parabola centred at \(x\).
- \(\operatorname{argmin}_w\) returns the value of \(w\) at which the objective is smallest (here there is exactly one, as the proof shows).
- \(\operatorname{sign}(x)\) is \(+1\) for positive \(x\), \(-1\) for negative \(x\), and 0 for \(x = 0\).
- \([a]_+\) keeps \(a\) if it is positive and replaces it by 0 otherwise.

In plain words: **shrink \(x\) towards zero by \(\lambda\), and if that would cross zero, stop at zero.** With \(\lambda = 0.25\): \(x = 0.6\) gives \(0.35\), \(x = -0.9\) gives \(-0.65\), and \(x = -0.2\), \(0.1\) or exactly \(0.25\) all give 0. The operator is called **soft thresholding**.

*Proof.* Write \(f(w) = \tfrac12 w^2 - xw + \lambda|w|\); note \(f(0) = 0\).

*Case \(|x| \le \lambda\).* For \(w > 0\), \(f(w) = \tfrac12 w^2 + (\lambda - x)w\); both terms are \(\ge 0\) and the first is positive, so \(f(w) > 0\). For \(w < 0\), \(|w| = -w\) and \(f(w) = \tfrac12 w^2 + (\lambda + x)|w|\), again positive since \(\lambda + x \ge 0\). So \(w = 0\) is the unique minimizer, matching \([|x| - \lambda]_+ = 0\).

*Case \(x > \lambda\).* For \(w < 0\), the same computation gives \(f(w) > 0 = f(0)\), so the minimum is not on the negative side. For \(w \ge 0\), \(f(w) = \tfrac12 w^2 - (x - \lambda)w = \tfrac12\big(w - (x - \lambda)\big)^2 - \tfrac12(x - \lambda)^2\), which is smallest exactly at \(w = x - \lambda > 0\), with a negative value. So the unique minimizer is \(x - \lambda = \operatorname{sign}(x)(|x| - \lambda)\).

*Case \(x < -\lambda\).* Replace \(w\) by \(-w\) and \(x\) by \(-x\): \(f\) is unchanged, so the previous case gives the minimizer \(-( -x - \lambda) = x + \lambda = \operatorname{sign}(x)(|x| - \lambda)\). \(\blacksquare\)

```python
def objective(w, x, lam):
    return 0.5 * w * w - x * w + lam * abs(w)

def soft(x, lam):
    sign = (x > 0) - (x < 0)
    return sign * max(abs(x) - lam, 0.0) + 0.0

lam = 0.25
grid = [i / 1000 for i in range(-2000, 2001)]
for x in (-0.9, -0.2, 0.1, 0.25, 0.6):
    w_grid = min(grid, key=lambda w: objective(w, x, lam))
    print(x, round(w_grid, 3), round(soft(x, lam), 3))
```

It prints `-0.9 -0.65 -0.65`, `-0.2 0.0 0.0`, `0.1 0.0 0.0`, `0.25 0.0 0.0`, `0.6 0.35 0.35`: brute force and formula agree on every row.

[[diagram:soft-threshold]]

Now a real regression with one feature. Take \(m\) messages, one feature with values \(x_1, \dots, x_m\), labels \(y_1, \dots, y_m\), the square loss halved for convenience, and the \(\ell_1\) price. Assume the feature has been scaled so that \(\frac1m\sum_i x_i^2 = 1\) (the normalization section shows how), and write \(\langle \mathbf x, \mathbf y \rangle = \sum_i x_i y_i\). Expanding each square, \(\frac1{2m}\sum_i (x_i w - y_i)^2 = \tfrac12\left(\tfrac1m\sum_i x_i^2\right)w^2 - \tfrac1m\langle \mathbf x, \mathbf y \rangle w + \tfrac1{2m}\sum_i y_i^2\), and the last term does not depend on \(w\). With the scaling, the objective is the soft-thresholding objective with \(x\) replaced by \(\langle \mathbf x, \mathbf y \rangle/m\), plus a constant. Compare the same problem priced by the squared weight \(\lambda w^2\) instead, lesson 13’s choice: setting the derivative \(\frac1m\sum_i x_i(x_i w - y_i) + 2\lambda w\) to zero gives the second formula below.

\[ w_{\ell_1} = \operatorname{sign}(\langle \mathbf x, \mathbf y \rangle)\left[\frac{|\langle \mathbf x, \mathbf y \rangle|}{m} - \lambda\right]_+, \qquad\qquad w_{\ell_2} = \frac{\langle \mathbf x, \mathbf y \rangle/m}{\|\mathbf x\|^2/m + 2\lambda}. \]

- \(\langle \mathbf x, \mathbf y \rangle/m\) is the average of feature times label: with the scaling and centred data, it is the feature’s correlation with the label.
- \(w_{\ell_1}\) is soft thresholding of that correlation at \(\lambda\).
- \(w_{\ell_2}\) divides the same correlation by \(\|\mathbf x\|^2/m + 2\lambda\), here \(1 + 2\lambda\): a shrink by a constant factor, never to zero.

In plain words: **with the \(\ell_1\) price, a feature whose correlation with the label is below \(\lambda\) gets weight exactly zero; with the squared price, every feature with any correlation at all keeps some weight.** Take 20 messages where the feature agrees with the label on 11 and disagrees on 9, so \(\langle \mathbf x, \mathbf y \rangle/m = 0.1\), and \(\lambda = 0.15\):

```python
x = [1, -1] * 10
y = [-t for t in x[:9]] + x[9:]
m, lam = len(x), 0.15
xy = sum(p * q for p, q in zip(x, y))
xx = sum(p * p for p in x)
print(xx / m, xy / m)
sign = 1 if xy > 0 else -1
w_l1 = sign * max(abs(xy) / m - lam, 0.0)
w_l2 = (xy / m) / (xx / m + 2 * lam)
print(w_l1, round(w_l2, 4))
```

It prints `1.0 0.1` and `0.0 0.0769`. The \(\ell_1\) learner drops this weak feature entirely; the \(\ell_2\) learner keeps it at 0.077. Multiply that by twenty thousand rare words, each with a small chance correlation like the 2,637 junk-only words of the opening, and you see the difference: the \(\ell_2\) penalty keeps them all, a little; the \(\ell_1\) penalty keeps only the words whose evidence clears the price.

### The LASSO, and solving it one coordinate at a time

Linear regression with the square loss and the \(\ell_1\) price has a name, the **LASSO**:

\[ \operatorname*{argmin}_{\mathbf w \in \mathbb R^d} \left( \frac{1}{2m}\|X\mathbf w - \mathbf y\|^2 + \lambda\|\mathbf w\|_1 \right). \]

- \(X\mathbf w\) is the vector of predictions for all \(m\) messages; \(\|X\mathbf w - \mathbf y\|^2\) their total square loss; \(\frac1{2m}\) makes it half the average.
- \(\lambda\|\mathbf w\|_1\) is the price on total absolute weight.

In plain words: **least squares, plus a price for every unit of weight, so that features not worth their price get weight zero.** Under suitable assumptions on the distribution and on \(\lambda\), the LASSO is known to find the right sparse set of features; we will not prove that here. A low \(\ell_1\) norm has a second payoff, which lesson 26 makes precise: it controls the estimation error directly.

How do you compute it? The one-dimensional result suggests an algorithm, which is not the only one but is the simplest: **coordinate descent**. Hold every weight fixed except \(w_j\), and solve for \(w_j\) exactly. Write \(\mathbf r_j = \mathbf y - \sum_{i \ne j} w_i X_i\) for the labels minus what the other features already predict. As a function of \(w_j\) alone, the objective is \(\frac1{2m}\|w_j X_j - \mathbf r_j\|^2 + \lambda|w_j| + \text{const}\). Expanding the square as before, this is \(\tfrac12 z_j w_j^2 - c_j w_j + \lambda|w_j| + \text{const}\), with \(z_j = \|X_j\|^2/m\) and \(c_j = \langle X_j, \mathbf r_j \rangle/m\). Factor out \(z_j\): it becomes \(z_j\big(\tfrac12 w_j^2 - (c_j/z_j)w_j + (\lambda/z_j)|w_j|\big)\), the soft-thresholding objective with \(x = c_j/z_j\) and price \(\lambda/z_j\), whose minimizer, writing \(\operatorname{soft}(x, \lambda) = \operatorname{sign}(x)[|x| - \lambda]_+\) for the soft-thresholding answer, is \(\operatorname{soft}(c_j/z_j, \lambda/z_j) = \operatorname{soft}(c_j, \lambda)/z_j\) (shrinking both numbers by the same positive factor shrinks the answer by it too). That is the update

\[ w_j \leftarrow \frac{\operatorname{soft}\!\left(\langle X_j, \mathbf r_j \rangle/m, \ \lambda\right)}{\|X_j\|^2/m}, \]

In plain words: **set each weight to its feature’s correlation with what is still unexplained, shrunk by the price, then rescaled; sweep over the features until nothing changes.** Each update lowers the convex objective, and for this problem the sweeps converge to a minimizer; we take that convergence on trust here (it is outside this lesson). On the six-message example:

```python
def dot(a, b): return sum(p * q for p, q in zip(a, b))
def soft(x, lam):
    return ((x > 0) - (x < 0)) * max(abs(x) - lam, 0.0)

y = [1, 1, 1, -1, -1, -1]
X = [[1, 1, 0, 0, -1, 0], [2, 0, 1, -1, 0, -2],
     [0, 1, 1, 0, 0, -1]]
m = len(y)
for lam in (0.01, 0.3, 0.6):
    w = [0.0] * len(X)
    for sweep in range(200):
        for j, Xj in enumerate(X):
            r = [yi - sum(w[k] * X[k][i] for k in range(len(X))
                          if k != j) for i, yi in enumerate(y)]
            w[j] = soft(dot(Xj, r) / m, lam) / (dot(Xj, Xj) / m)
    print(lam, [round(wj, 3) + 0.0 for wj in w])
```

It prints `0.01 [0.607, 0.342, 0.436]`, `0.3 [0.138, 0.392, 0.0]`, `0.6 [0.0, 0.24, 0.0]`. As the price rises, the filter goes from three features to two to one, and the survivor is feature 2, the one the greedy methods chose first.

```
LassoCD(X, y, λ, sweeps):
  w ← 0 in R^d; r ← y           (r = y − Xw throughout)
  for s = 1..sweeps:
    for j = 1..d:
      r ← r + w_j X_j                (take out feature j)
      w_j ← soft(⟨X_j, r⟩/m, λ) / (‖X_j‖²/m)
      r ← r − w_j X_j
  return w
```

Keeping the residual \(\mathbf r\) up to date makes each coordinate update cost \(O(m)\), so a sweep costs \(O(dm)\) time; space is \(O(dm)\) for the data and \(O(d + m)\) besides. (The snippet above recomputes the residual from scratch, which is slower but easier to read.)

## The selectors as code

Time to build all of them. Using only the standard library, implement the five functions below, where `X` is a list of rows (one per message), `y` a list of labels, and `ls_loss(X, y, I)` the total square loss of the least-squares fit on the columns in `I`, without intercept. The starter data is the cancelling world.

```python
y     = [1, 1, 1, 1, -1, -1, -1, -1]
link  = [1, -1, 1, -1, 1, -1, 1, -1]
old   = [0, 1, 0, 1, -1, 0, -1, 0]
bangs = [1, 1, 1, -1, -1, -1, -1, 1]
noise = [1, 1, -1, -1, -1, 1, 1, -1]
X = [list(row) for row in zip(link, old, bangs, noise)]

def ls_loss(X, y, I): ...
def pearson_filter(X, y, k): ...
def forward_greedy(X, y, k): ...
def omp(X, y, k): ...
def backward_elimination(X, y, k): ...
def lasso_cd(X, y, lam, sweeps=500): ...

if __name__ == "__main__":
    assert pearson_filter(X, y, 2) == [1, 2]
    assert forward_greedy(X, y, 2) == [1, 0]
    for k in (1, 2, 3):
        assert omp(X, y, k) == forward_greedy(X, y, k)
    assert sorted(backward_elimination(X, y, 2)) == [0, 1]
    assert ls_loss(X, y, [1, 2]) > 3.4
    assert ls_loss(X, y, [0, 1]) < 1e-9
    assert lasso_cd(X, y, 0.5) == [0.0, 0.0, 0.0, 0.0]
    w = lasso_cd(X, y, 0.2)
    assert w[3] == 0.0 and w[1] > w[0] > 0
    print("all asserts pass")
```

1. Implement `ls_loss` by solving the normal equations of lesson 9 with Gaussian elimination. Break ties in every selector by the smaller feature index, and return the filter’s choice sorted.
2. Make all the asserts pass. Then explain the `lasso_cd(X, y, 0.5)` line: why is 0.5 exactly the price at which every weight dies? (Look at \(\langle X_j, \mathbf y \rangle/m\) for each feature.)
3. Time `forward_greedy` and `omp` on random data with \(m = 200\) and \(d = 50, 100, 200\), and check that the ratio grows the way the complexity analysis predicts.

## When scale decides: normalization

Now the third complaint, the length in bytes and the sender volume in millions. Selection asks *which* features; normalization asks *in what units*. A **feature transformation** is a function applied to one feature’s values, such as subtracting the average or taking a logarithm. **Normalization** is a transformation that puts features on comparable scales. As with selection, there are no transformations that are good or bad in general: each one is good or bad for a particular learner and a particular belief about the problem.

Here is the cleanest example of scale changing what a learner can do. Ridge regression prefers small weights. But “small” depends on units. A feature measured in tiny units needs a huge weight to matter, and ridge will refuse to pay for it.

Build a world with two features. Draw the label \(y\) as a fair \(\pm 1\) coin and an independent fair \(\pm1\) coin \(\alpha\). Feature 1 is \(x_1 = y + 0.5\alpha\): it agrees with the label in sign, plus noise of size 0.5. Feature 2 is \(x_2 = 0.0001\,y\): the label itself, measured in very small units. The perfect predictor is \(\mathbf w^\star = (0, 10{,}000)\), since \(10{,}000 \cdot 0.0001\,y = y\), with true risk 0. But its squared length is \(10^8\).

Recall ridge regression exactly as lesson 13 defined it, with the half square loss:

\[ \operatorname*{argmin}_{\mathbf w} \left( \lambda\|\mathbf w\|^2 + \frac1m \sum_{i=1}^m \tfrac12\left(\langle \mathbf w, \mathbf x_i \rangle - y_i\right)^2 \right), \qquad \text{solved by} \quad (2\lambda m I + A)\,\mathbf w = \mathbf b. \]

- The sum is the average half square loss on the training sequence, \(L_S(\mathbf w)\); \(\lambda\|\mathbf w\|^2 = \lambda\sum_j w_j^2\) is the price on squared length.
- \(A = \sum_i \mathbf x_i\mathbf x_i^\top\) is the \(d \times d\) table of how features light up together, and \(\mathbf b = \sum_i y_i\mathbf x_i\) adds up the messages signed by their labels (lesson 13).
- \(2\lambda m I\) adds \(2\lambda m\) to every diagonal entry of \(A\), which always makes the system solvable.

In plain words: **least squares with \(2\lambda m\) added to the diagonal, which pulls the weights towards zero.** Now compare the objective at two points. At \(\mathbf w^\star = (0, 10^4)\), the loss is 0 and the price is \(\lambda \cdot 10^8\). At \(\mathbf w = (1, 0)\), the prediction is \(x_1 = y + 0.5\alpha\), the miss is always \(0.5\alpha\), the half square loss is \(\tfrac12 \cdot 0.25 = 0.125\), and the price is \(\lambda\). So the useless-looking \((1, 0)\) wins whenever \(\lambda \cdot 10^8 > 0.125 + \lambda\), that is, whenever \(\lambda > 0.125/(10^8 - 1) \approx 1.25 \times 10^{-9}\). Lesson 13’s analysis wants \(\lambda\) at least \(2\beta/m\), of order \(1/m\), so this happens unless you have hundreds of millions of examples. With \(\lambda = 10^{-3}\), averaging over the four equally likely values of \((y, \alpha)\):

```python
from itertools import product

pts = [(y + 0.5 * a, 1e-4 * y, y)
       for y, a in product([-1, 1], [-1, 1])]

def ridge_objective(w1, w2, lam):
    fit = sum((w1 * u + w2 * v - y) ** 2 / 2 for u, v, y in pts)
    return fit / len(pts) + lam * (w1 ** 2 + w2 ** 2)

lam = 1e-3
print(ridge_objective(0, 1e4, lam), ridge_objective(1, 0, lam))
print(f"{0.125 / (1e8 - 1):.3g}")
```

It prints `100000.0 0.126` and `1.25e-09`. The perfect predictor costs a hundred thousand; the sloppy one costs an eighth.

Now normalize: divide each feature by the largest absolute value it takes, so both live in \([-1, 1]\). Feature 1 becomes \((y + 0.5\alpha)/1.5\) and feature 2 becomes \(y\) itself. The perfect predictor is now \((0, 1)\), with squared length 1. Here is ridge’s actual answer before and after, computed on the four-point distribution (that is, with \(A/m\) and \(\mathbf b/m\) replaced by their expectations, the limit of a long training sequence, so the system reads \((A/m + 2\lambda I)\mathbf w = \mathbf b/m\)), together with its true risk under the half square loss:

```python
from itertools import product

def ridge_and_risk(scale1, scale2, lam):
    pts = [((y + 0.5 * a) / scale1, 1e-4 * y / scale2, y)
           for y, a in product([-1, 1], [-1, 1])]
    E = lambda f: sum(f(*p) for p in pts) / len(pts)
    c11, c12 = E(lambda u, v, y: u * u), E(lambda u, v, y: u * v)
    c22 = E(lambda u, v, y: v * v)
    b1, b2 = E(lambda u, v, y: u * y), E(lambda u, v, y: v * y)
    a, d = c11 + 2 * lam, c22 + 2 * lam
    w1 = (d * b1 - c12 * b2) / (a * d - c12 * c12)
    w2 = (a * b2 - c12 * b1) / (a * d - c12 * c12)
    risk = E(lambda u, v, y: (w1 * u + w2 * v - y) ** 2 / 2)
    return round(w1, 4), round(w2, 4), f"{risk:.2g}"

print(ridge_and_risk(1, 1, 1e-3))
print(ridge_and_risk(1.5, 1e-4, 1e-3))
```

It prints `(0.7987, 0.0101, '0.1')` and `(0.0117, 0.9902, '9.5e-06')`. Before normalization, ridge leans on the noisy feature and has true risk 0.1. After, it finds the perfect feature, with true risk about ten in a million. Same data, same learner, same \(\lambda\): only the units changed.

[[diagram:scales]]

There is a second gain, on the guarantee side. Lesson 13’s bounds for regularized loss minimization, and lesson 14’s iteration count for SGD, grow with \(\|\mathbf w^\star\|^2\) times the largest \(\|\mathbf x\|^2\) (the half square loss is \(\|\mathbf x\|^2\)-smooth, which is why lesson 13 took \(\beta = 0.16\) for a word worth 0.4). Here normalization takes \(\|\mathbf w^\star\|^2\) from \(10^8\) to 1, while the largest \(\|\mathbf x\|\) stays about the same (about 1.5 before, about 1.4 after). So the same normalization shrinks the estimation error bound and the runtime by a factor of the order of a hundred million.

### Clipping: when big values lie

Normalization rescales. Some transformations change the shape of a feature, and that can change what the class can express at all. Here the estimation error is not the issue; the approximation error is.

Suppose most messages have a feature value of about \(\pm 1\), in agreement with the label, but rarely, with probability \(1/a\) for some large \(a\), the value is \(a\) times bigger: a message length that is normal except when an attachment is included. Precisely: \(y\) is a fair \(\pm1\) coin; with probability \(1 - 1/a\), \(x = y\), and with probability \(1/a\), \(x = ay\). A single weight \(w\), square loss. The true risk of \(w\) is

\[ L_{\mathcal D}(w) = \mathbb E\left[(wx - y)^2\right] = \left(1 - \tfrac1a\right)(w - 1)^2 + \tfrac1a (aw - 1)^2. \]

- \(L_{\mathcal D}(w)\) is the true risk (lesson 1): the expected loss on a fresh message.
- \(\mathbb E[\cdot]\) averages over the two cases, weighted by their probabilities.
- In the common case, \((wy - y)^2 = (w - 1)^2 y^2 = (w-1)^2\), since \(y^2 = 1\); in the rare case, \((awy - y)^2 = (aw - 1)^2\).

In plain words: **the risk is a mix of the common case, which wants \(w = 1\), and the rare case, which wants \(w = 1/a\).** Setting the derivative \(2(1 - \frac1a)(w - 1) + 2(aw - 1)\) to zero and multiplying by \(a/2\) gives \(w(a^2 + a - 1) = 2a - 1\), so the best weight is

\[ w^\star = \frac{2a - 1}{a^2 + a - 1}, \]

which goes to 0 as \(a\) grows: the rare huge values drag the best weight down, and the common case is left almost unexplained. The risk at \(w^\star\) tends to 1, the risk of predicting 0 always. Now **clip** the feature, \(x \mapsto \operatorname{sign}(x)\min\{1, |x|\}\): any value beyond \(\pm1\) is cut back to \(\pm1\). After clipping, \(x = y\) in both cases, \(w^\star = 1\), and the risk is 0.

```python
def risk(w, a):
    return (1 - 1 / a) * (w - 1) ** 2 + (1 / a) * (a * w - 1) ** 2

for a in (2, 10, 100):
    w_star = (2 * a - 1) / (a * a + a - 1)
    grid = min((i / 10000 for i in range(-20000, 20001)),
               key=lambda w: risk(w, a))
    print(a, round(w_star, 4), round(grid, 4),
          round(risk(w_star, a), 4))
clip = lambda x: (1 if x > 0 else -1) * min(1, abs(x))
print(clip(100), clip(-100), clip(1), clip(-1))
```

It prints `2 0.6 0.6 0.1`, `10 0.1743 0.1743 0.6688`, `100 0.0197 0.0197 0.9608`, then `1 -1 1 -1`. At \(a = 100\), the best linear predictor on the raw feature has risk 0.96, close to useless; on the clipped feature it is perfect. One line of preprocessing removed almost all of the approximation error.

Of course clipping can also hurt: if the huge values carried real information, cutting them off destroys it. Clipping is a bet, the prior belief that values above some threshold say nothing more than the threshold itself. Every transformation is a bet of this kind.

### A toolbox of transformations

Here are the common transformations. For one feature, write its values on the \(m\) training messages as \(f_1, \dots, f_m\), its average as \(\bar f = \frac1m\sum_i f_i\), and its largest and smallest values as \(f_{\max}\) and \(f_{\min}\). We use message lengths in bytes as the running example: 120, 450, 800, 2,300, 5,100 and one message of 900,000 bytes with an attachment.

| transformation | new value \(f_i \leftarrow\) | what it does |
|---|---|---|
| centering | \(f_i - \bar f\) | average becomes 0 |
| unit range | \(\dfrac{f_i - f_{\min}}{f_{\max} - f_{\min}}\) | values span exactly \([0, 1]\) (use \(2 \cdot (\cdot) - 1\) for \([-1, 1]\)) |
| standardization | \(\dfrac{f_i - \bar f}{\sqrt{\nu}}\), with \(\nu = \frac1m \sum_i (f_i - \bar f)^2\) | average 0 and variance 1 |
| clipping | \(\operatorname{sign}(f_i)\min\{b, \lvert f_i \rvert\}\) | values beyond \(\pm b\) cut back to \(\pm b\) |
| sigmoidal | \(\dfrac{1}{1 + \exp(b f_i)}\) | a soft clipping: almost linear near 0, flat far out |
| logarithmic | \(\log(b + f_i)\) | equal ratios become equal steps |

Read the rows one at a time.

- In unit range, the numerator is how far \(f_i\) sits above the smallest value, and the denominator is the full spread, so the ratio is the fraction of the way from the bottom to the top.
- In standardization, \(\nu\) (Greek “nu”) is the empirical variance, the average squared distance from the average; its square root is the standard deviation, so the new value counts how many standard deviations \(f_i\) is from the average.
- In clipping, \(b > 0\) is a threshold you choose; \(\min\{b, |f_i|\}\) caps the size and \(\operatorname{sign}(f_i)\) restores the sign.
- In the sigmoid, \(b\) sets the steepness; with \(b < 0\) the output grows with \(f_i\), from 0 towards 1.
- In the logarithm, \(b \ge 0\) (often 1) keeps the argument positive when \(f_i = 0\). It is the natural choice for **counting features**, such as how many times a word appears: the difference between 0 and 1 occurrences matters far more than the difference between 1,000 and 1,001. The base of the logarithm only rescales the result.

In plain words: **each row maps one feature’s values to new values, using only that feature’s own training values and a parameter \(b\) you choose.** Let’s push the six lengths through each row (centering shown in thousands of bytes, clipping at \(b = 5{,}000\), the sigmoid with \(b = -1/1000\), the logarithm in base 10 with \(b = 1\)):

```python
from math import exp, log10

f = [120, 450, 800, 2300, 5100, 900_000]
m = len(f)
mean = sum(f) / m
nu = sum((fi - mean) ** 2 for fi in f) / m
lo, hi = min(f), max(f)
show = lambda name, g: print(name, [round(g(fi), 2) for fi in f])
show("center", lambda fi: (fi - mean) / 1000)
show("unit", lambda fi: (fi - lo) / (hi - lo))
show("standard", lambda fi: (fi - mean) / nu ** 0.5)
sign = lambda t: 1 if t > 0 else -1
show("clip", lambda fi: sign(fi) * min(5000, abs(fi)))
show("sigmoid", lambda fi: 1 / (1 + exp(-fi / 1000)))
show("log", lambda fi: log10(1 + fi))
```

It prints:

```
center [-151.34, -151.01, -150.66, -149.16, -146.36, 748.54]
unit [0.0, 0.0, 0.0, 0.0, 0.01, 1.0]
standard [-0.45, -0.45, -0.45, -0.45, -0.44, 2.24]
clip [120, 450, 800, 2300, 5000, 5000]
sigmoid [0.53, 0.61, 0.69, 0.91, 0.99, 1.0]
log [2.08, 2.65, 2.9, 3.36, 3.71, 5.95]
```

Look at what one outlier does. Unit range and standardization both let the 900,000-byte message set the scale, so the five ordinary lengths, which differ by a factor of forty, collapse onto nearly the same value. Clipping, the sigmoid and the logarithm keep them apart. That is exactly the difference you will need in the lab.

[[diagram:transforms]]

All of these transform each feature using only its own values. Sometimes the parameters should depend on all features together. A common example scales every feature vector so that the average of some norm of the vectors, say \(\frac1m\sum_i \|\mathbf x_i\|\), becomes 1.

One caution matters for guarantees. A transformation whose constants (\(\bar f\), \(f_{\max}\), \(\nu\)) are computed from the training sequence is part of the learner, since it looked at the data. In the fix below we therefore fix every constant in advance, so that the transformation is part of the hypothesis class we count.

## Feature learning: let the data build the features

Selection keeps some of the given features; transformation reshapes them one at a time. **Feature learning** goes further: it learns a whole map \(\psi\) from messages (or images, or sounds) to feature vectors in \(\mathbb R^k\), so that a simple class, usually linear predictors, works well on top of \(\psi\). You have seen hand-built maps of this kind: lesson 9’s polynomial features and lesson 16’s embeddings. Lesson 20’s networks learned one: their hidden neurons, each a halfspace on the raw signals, are features fitted by backpropagation together with the predictor on top, from labels. The methods below build \(\psi\) without labels. And no-free-lunch applies again: a feature learner can only succeed by assuming something about the distribution.

The motivating example is the **bag of words**, which this filter already uses. Given a dictionary of \(k\) words, a message becomes a vector in \(\{0, 1\}^k\) with a 1 in position \(i\) exactly when word \(i\) appears. Linear predictors on this representation work remarkably well for text: each word is a small measurement of the message, and the learner weighs them. But what is a “word” for an image of a face, where the raw pixels make a poor input for a linear predictor? We would like “visual words”, such as “there is an eye in the image”, and we would like the data to tell us what they are. That is **dictionary learning**.

A first naive idea uses clustering (lesson 22). Learn a clustering \(c\) that sends each instance to one of \(k\) clusters, and treat clusters as words: \(\psi(x)\) is the vector in \(\{0,1\}^k\) with a single 1, in position \(c(x)\). What can a linear predictor do on top of this? For any weights \(\mathbf w\), \(\langle \mathbf w, \psi(x) \rangle = \sum_{i=1}^k w_i \mathbb 1[c(x) = i] = w_{c(x)}\), where \(\mathbb 1[c(x) = i]\) is 1 when \(x\) is in cluster \(i\) and 0 otherwise: exactly one term survives. So **a linear predictor on cluster indicators gives the same value to every instance in a cluster**, and if the clusters come from lesson 22’s k-means algorithm, which assigns each instance to its nearest centre, it is a piecewise constant predictor over the regions nearest each centre.

Clustering (lesson 22) and principal component analysis (lesson 23) are both special cases of one idea. An **auto-encoder** is a pair of functions: an **encoder** \(\psi : \mathbb R^d \to \mathbb R^k\) that turns an instance into a code, and a **decoder** \(\phi : \mathbb R^k \to \mathbb R^d\) that turns the code back into an instance. The pair is learned so that decoding the code gives back something close to what you started with:

\[ \sum_{i=1}^m \left\| \mathbf x_i - \phi(\psi(\mathbf x_i)) \right\|^2 \ \ \text{is small.} \]

- \(\psi(\mathbf x_i)\) is the code of instance \(i\); \(\phi(\psi(\mathbf x_i))\) is its reconstruction.
- \(\|\mathbf x_i - \phi(\psi(\mathbf x_i))\|^2\) is the squared distance between an instance and its reconstruction.
- \(\sum_{i=1}^m\) adds this **reconstruction error** over the \(m\) training instances.

In plain words: **an auto-encoder learns a code from which the instances can be rebuilt.** Without restrictions this is trivial: take \(k = d\) and let both maps be the identity, and the error is 0 while nothing has been learned. So the pair must be restricted. PCA, as lesson 23 set it up, keeps \(k < d\) (lesson 23 called it \(n\)) and makes both maps matrices: a compression matrix \(W\) with \(k\) rows as the encoder and a recovery matrix \(U\) as the decoder, and the best pair is \(U\) with the top \(k\) principal components as orthonormal columns and \(W = U^\top\). The k-means cost of lesson 22 allows any \(k\), even \(k > d\), but uses \(k\) centres \(\boldsymbol\mu_1, \dots, \boldsymbol\mu_k\): the encoder returns the indicator vector of the nearest centre, and the decoder \(\phi(\mathbf v) = \sum_i v_i\boldsymbol\mu_i\) returns that centre. Here are six messages described by two numbers (the base-10 logarithm of the length and the number of bangs) and two centres:

```python
xs = [(2.1, 4), (2.4, 5), (2.0, 6), (3.6, 0), (3.9, 1), (3.5, 0)]
mus = [(2.2, 5.0), (3.7, 0.3)]
dist2 = lambda a, b: sum((p - q) ** 2 for p, q in zip(a, b))

def psi(x):
    c = min(range(len(mus)), key=lambda i: dist2(x, mus[i]))
    return [1 if i == c else 0 for i in range(len(mus))]

def phi(v):
    return tuple(sum(v[i] * mus[i][k] for i in range(len(mus)))
                 for k in range(2))

print([psi(x) for x in xs])
print(round(sum(dist2(x, phi(psi(x))) for x in xs), 2))
w = [0.9, -0.8]
print([sum(wi * vi for wi, vi in zip(w, psi(x))) for x in xs])
```

It prints the six codes `[[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]`, the reconstruction error `2.85`, and `[0.9, 0.9, 0.9, -0.8, -0.8, -0.8]`: a linear predictor on the codes is constant on each cluster, as proved above.

[[diagram:autoencoder]]

What lets k-means use more centres than dimensions without becoming trivial is that its codes are **sparse**: exactly one nonzero entry. The natural extension allows a few. Fix the centres, now called the **dictionary**, \(\boldsymbol\mu_1, \dots, \boldsymbol\mu_k\), let the decoder be \(\phi(\mathbf v) = \sum_i v_i\boldsymbol\mu_i\), and let the encoder find the best code with at most \(s\) nonzero entries:

\[ \psi(\mathbf x) = \operatorname*{argmin}_{\mathbf v \in \mathbb R^k} \left\| \mathbf x - \phi(\mathbf v) \right\|^2 \ \ \text{subject to} \ \ \|\mathbf v\|_0 \le s. \]

- \(\mathbf v\) is a candidate code: how much of each dictionary vector to use.
- \(\phi(\mathbf v)\) is the instance the code describes; \(\|\mathbf x - \phi(\mathbf v)\|^2\) is how far it is from \(\mathbf x\).
- \(\|\mathbf v\|_0 \le s\) allows at most \(s\) dictionary vectors in the description.

In plain words: **describe each instance as well as you can using at most \(s\) dictionary vectors.** Take a dictionary of three vectors in the plane, \(\boldsymbol\mu_1 = (1, 0)\), \(\boldsymbol\mu_2 = (0, 1)\), \(\boldsymbol\mu_3 = (0.7, 0.7)\), and an instance \(\mathbf x = (0.9, 0.5)\). With \(s = 1\), the best code uses one vector with its best coefficient \(\langle \mathbf x, \boldsymbol\mu_i \rangle/\|\boldsymbol\mu_i\|^2\); with \(s = 2\), the first two vectors rebuild \(\mathbf x\) exactly:

```python
mus = [(1.0, 0.0), (0.0, 1.0), (0.7, 0.7)]
x = (0.9, 0.5)
dot = lambda a, b: sum(p * q for p, q in zip(a, b))

def err(pair, coeffs):
    rec = [sum(c * mus[i][k] for c, i in zip(coeffs, pair))
           for k in range(2)]
    return dot([p - q for p, q in zip(x, rec)],
               [p - q for p, q in zip(x, rec)])

for i in range(3):
    c = dot(x, mus[i]) / dot(mus[i], mus[i])
    print("s=1", i + 1, round(c, 4), round(err((i,), (c,)), 4))
print("s=2", 1, 2, err((0, 1), (0.9, 0.5)))
```

It prints `s=1 1 0.9 0.25`, `s=1 2 0.5 0.81`, `s=1 3 1.0 0.08` and `s=2 1 2 0.0`: one vector leaves an error of at least 0.08 (using \(\boldsymbol\mu_3\)), two vectors leave none.

With \(s = 1\) and codes restricted to indicator vectors (a single entry equal to 1), this is the k-means encoder: the best single centre is the nearest one. For larger \(s\) the encoder faces the same hard problem as feature selection, now once per instance (indeed it *is* feature selection, with the dictionary vectors as the columns). So in practice the constraint is replaced by the \(\ell_1\) price, \(\psi(\mathbf x) = \operatorname{argmin}_{\mathbf v}\left(\|\mathbf x - \phi(\mathbf v)\|^2 + \lambda\|\mathbf v\|_1\right)\), which is a LASSO per instance and gives sparse codes for the reason the soft-thresholding proposition explained. Learning the dictionary itself, choosing \(\boldsymbol\mu_1, \dots, \boldsymbol\mu_k\) to make the total reconstruction error small, remains computationally hard, just as minimizing the k-means cost is NP-hard (lesson 22), and in practice it is attacked with heuristics that alternate, like the k-means algorithm: fix the dictionary and find the codes, then fix the codes and refit the dictionary. An auto-encoder whose two maps are lesson 20’s neural networks, trained by SGD on the reconstruction error, is where much of the modern work on learned features lives.

## The guarantee that selects few features

Back to the mail. You can now select features and fix their scales. But the opening bug was overfitting, and the lab will not accept a filter that merely looks good on your 2,000 labels. You need a certificate: a bound on the true risk that holds with high probability, computed before you ship.

Lesson 4 gave you one for any finite class: for every \(h\) in \(\mathcal H\) at once, the training error and the true risk are close. Its size enters only through a logarithm, and that is what makes few features affordable. Let’s count a class of small filters. A filter in \(\mathcal H_k\) does four things, each chosen from a fixed menu:

- it picks \(k\) of the \(d = 20{,}002\) features (20,000 words, the length, the sender volume);
- for each picked feature it picks one of four transformations with constants fixed in advance: raw, unit range on the fixed span \([0, 10^7]\), logarithm \(\log_{10}(1 + \min\{f, 10^7\})/7\), or clipping \(\min\{1, f/1000\}\) (on a 0/1 word all four give back the word, which only makes the count generous);
- it gives each picked feature a sign, \(+1\) or \(-1\);
- it picks a threshold \(\theta\) from the grid \(-k, -k + 0.1, \dots, k\), which has \(20k + 1\) values;

and it says junk exactly when the signed sum of the transformed features exceeds \(\theta\). The number of filters is at most \(\binom{d}{k} \cdot 4^k \cdot 2^k \cdot (20k + 1)\).

Now the bound. You are allowed to try several \(k\) and keep the best certificate, so we split the failure probability \(\delta\) over the values of \(k\), giving \(\mathcal H_k\) the share \(\delta_k = \delta/2^k\); these shares add up to at most \(\delta\), the same device as lesson 7’s structural risk minimization. Lesson 4’s uniform convergence, Hoeffding’s inequality plus the union bound, then says: with probability at least \(1 - \delta\) over the training sequence \(S \sim \mathcal D^m\), simultaneously for every \(k\) and every filter \(h \in \mathcal H_k\),

\[ L_{\mathcal D}(h) \ \le \ L_S(h) + \sqrt{\frac{\log\left(2|\mathcal H_k|/\delta_k\right)}{2m}}. \]

- \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) messages drawn independently from \(\mathcal D\)”, the i.i.d. assumption of lesson 1.
- \(L_{\mathcal D}(h)\) is the true risk with the 0-1 loss, the chance of mislabelling the next message; \(L_S(h)\) the fraction of training messages it gets wrong.
- \(|\mathcal H_k|\) is the number of filters in the class, counted above; \(\log\) is the natural logarithm.
- The square root is the **gap**: the most the training error can understate the true risk, for any filter in the class, unless you were unlucky (probability at most \(\delta\)).

In plain words: **a filter’s true risk is at most its training error plus a gap that grows only with the logarithm of the number of filters you could have chosen.** And here is the point of the whole lesson in one number: the logarithm of \(\binom{d}{k}\) is about \(k\log d\), so the gap grows with the number of features you *use*, not the number you *have*. Let’s compute it for \(m = 2{,}000\) labels and \(\delta = 0.05\), and also for a filter allowed to use all 20,002 features:

```python
from math import comb, log, sqrt

d, m, delta = 20_002, 2_000, 0.05

def log_H(k):
    subsets, transforms = comb(d, k), 4 ** k
    signs, thresholds = 2 ** k, 20 * k + 1
    return log(subsets * transforms * signs * thresholds)

def gap(k):
    log_two_over_delta_k = log(2 / delta) + k * log(2)
    return sqrt((log_H(k) + log_two_over_delta_k) / (2 * m))

for k in (1, 2, 3, 4, 5, d):
    print(k, round(log_H(k), 1), round(gap(k), 4))
```

It prints `1 15.0 0.0697`, `2 27.0 0.0895`, `3 38.3 0.1049`, `4 49.1 0.1179`, `5 59.7 0.1293` and `20002 41605.9 3.724`. With four features, the training error can mislead you by at most about 0.118. With all of them, the “gap” is 3.7, more than the largest possible risk: no certificate at all. That is the opening bug, measured. A filter that reads every word may look perfect on its training sequence and you can promise nothing about it.

Notice what the bound does *not* require: that you pick the features in any particular way. It holds for every filter in \(\mathcal H_k\) at once, so you may choose the \(k\) features by a filter-method score, by forward greedy selection, or by staring at the data, and the certificate still holds. What it does require is that the menu of transformations was fixed before the labels were seen.

## Fix the bug

The lab below gives you the real mail stream: 20,000 words, the length in bytes and the sender volume, and a fixed budget of 2,000 labelled messages, already drawn. You choose up to five features, a transformation for each, and the lab fits the signs and the threshold that minimize the training error and prints \(L_S\), the gap and their sum. Your goal: a filter whose **certified** true risk, \(L_S(h)\) plus the gap, is at most 0.19. A filter whose fresh-message error happens to be low but whose certificate is above 0.19 does not count.

Before you start, think about three things. Which of the transformations in the toolbox survive a 900,000-byte outlier? What does adding a fifth feature cost you in gap, and what must it buy in training error to pay for itself? And what will the filter-method score of the sender volume look like when its values run into the millions?

[[lab-act2]]

## What people get wrong

**“A feature with zero correlation is useless.”** Zero correlation says only that the best line in that feature *alone* is flat. In the cancelling world, link has correlation exactly 0 and is indispensable: together with old it predicts every label. Filter methods cannot see features whose value lies in combination; greedy selection, which scores a candidate next to the features already chosen, can.

**“The \(\ell_2\) penalty also selects features, just more gently.”** It shrinks every weight by a factor but never to zero. With a correlation of 0.1 and \(\lambda = 0.15\), the \(\ell_2\) weight is 0.077 and the \(\ell_1\) weight is exactly 0. Only the \(\ell_1\) price has the constant pull towards zero that stops weak features dead.

**“Normalization is cosmetic: the information is the same.”** For a learner with a fixed preference, such as ridge regression’s preference for short weight vectors or a filter class with fixed weights and thresholds, the units decide what the learner can afford. Rescaling one feature took ridge’s true risk from 0.1 to about \(10^{-5}\), and clipping took a linear predictor’s risk from 0.96 to 0.

**“Unit range is the safe default.”** One outlier sets the range, and every ordinary value is squashed next to zero: five lengths from 120 to 5,100 bytes all became 0.0 or 0.01. If the feature has heavy tails, clip it or take its logarithm first.

**“My filter has 99% training accuracy with all 20,000 words, so it is good.”** With all the features in play, the gap in the finite-class bound is 3.7, so the training accuracy certifies nothing. A filter with four well-chosen features and a worse training error can carry a real guarantee.

## Recap and the next bug

1. A feature is a measurement of a message, and the representation is prior knowledge, chosen before the hypothesis class. How good a feature is depends on the learner: \(x_1\) with \(y = x_1^2\) is perfect for a parabola and useless for a line.
2. Feature selection asks for the best predictor with \(\|\mathbf w\|_0 \le k\), which is hard to solve exactly, so we use cheaper methods.
3. **Filter methods** score each feature alone. For linear regression with the square loss, ranking by the best single-feature line is ranking by \(|\rho|\), Pearson’s correlation, and it costs \(O(dm)\). Filter methods miss features that help only in combination.
4. **Forward greedy selection** adds the feature that helps most next to those chosen. **OMP** does the same for least squares, scoring each candidate by \(\langle \mathbf u_j, \mathbf y \rangle^2/\|\mathbf u_j\|^2\), the alignment of its new part with the labels, in \(O(Tdm)\). The steepest-gradient rule and AdaBoost are greedy selections too. **Backward elimination** removes the least useful feature each round, at a much higher cost.
5. The **\(\ell_1\) norm** is the convex stand-in for the count. Soft thresholding shows why it gives exact zeros: a weight whose evidence is below \(\lambda\) is set to zero. The **LASSO** is least squares with this price, solvable by coordinate descent.
6. **Normalization and transformations** change what a learner can afford and even what it can express: ridge and units, clipping and approximation error, and a toolbox (centering, unit range, standardization, clipping, sigmoid, logarithm), each a bet about the problem.
7. **Feature learning** builds \(\psi\) from data. Auto-encoders minimize reconstruction error under restrictions: PCA (a compression and a recovery matrix, \(k < d\)), k-means (the one-hot code of the nearest centre), sparse coding (at most \(s\) dictionary vectors, or an \(\ell_1\) price).
8. With a finite class of \(k\)-feature filters, the gap in the uniform-convergence bound grows like \(\sqrt{k \log d / m}\): few features can be certified with 2,000 labels, all 20,002 cannot.

Your filter now reads three words and a log-scaled length, carries a certificate below 0.19, and touches four features per message instead of twenty thousand. Then a colleague from the research team looks at the certificate and frowns. “You counted every filter you *could* have built. Most of them behave identically on the mail you actually have. Your bound pays for richness your data never sees.” She is right, and the problem is older than this lesson: every guarantee so far measured a class’s richness before seeing any data (its size in lesson 2, its VC dimension in lesson 6, its description length in lesson 7), as a worst case over every possible set of messages. Can a class’s richness be measured on the very data in hand? That is the next lesson: Rademacher complexity.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the build-it challenge, `lasso_cd(X, y, 0.5)` returns all zeros, and the price 0.5 equals the largest value of \(|\langle X_j, \mathbf y \rangle|/m\) over the features. Prove that, for the LASSO objective \(F(\mathbf w) = g(\mathbf w) + \lambda\|\mathbf w\|_1\) with \(g(\mathbf w) = \frac1{2m}\|X\mathbf w - \mathbf y\|^2\), the vector \(\mathbf w = \mathbf 0\) is a minimizer whenever \(\lambda \ge \max_j |\langle X_j, \mathbf y \rangle|/m\). (Hint: use the tangent-line inequality for the convex function \(g\) from lesson 12, and compute \(\nabla g(\mathbf 0)\).) What happens at a price just below that value?
2. In the proof that OMP picks what forward greedy picks, find the exact step that uses \(V^\top V = I\). Suppose a buggy implementation keeps the chosen columns themselves as \(V\), without orthonormalizing, and computes \(\mathbf u_j = X_j - VV^\top X_j\). Which step of the proof breaks, and why can the cross term no longer be dropped?
3. Take the clipping example with \(a = 4\). Compute \(w^\star\) and \(L_{\mathcal D}(w^\star)\) on the raw feature, exactly as fractions, and the risk after clipping. Then describe a distribution of the same shape (a common case and a rare large value) where clipping at 1 makes the best linear predictor *worse*, and say what prior belief the clipping transformation was betting on.

## Problems

These are this lesson’s own problems, to work on after the concept check. Hints are available on request; try each one for a while first.

1. **Centering is free.** Prove that for any vectors \(\mathbf v, \mathbf y \in \mathbb R^m\),
\(\min_{a, b \in \mathbb R} \frac1m\|a\mathbf v + b - \mathbf y\|^2 = \min_{a, b \in \mathbb R} \frac1m\|a(\mathbf v - \bar v) + b - (\mathbf y - \bar y)\|^2.\)
Prove it by showing each side is at most the other: from a minimizer of one side, build a pair \((a, b)\) for the other side with the same objective value.
2. **AdaBoost is a greedy selection.** Let \(h_1, \dots, h_d\) be hypotheses with values in \(\{\pm 1\}\), and for weights \(\mathbf w \in \mathbb R^d\) let \(f_{\mathbf w} = \sum_j w_j h_j\) and \(R(\mathbf w) = \log \sum_{i=1}^m \exp(-y_i f_{\mathbf w}(x_i))\).
   - First show why a finite list is enough: if \(\mathcal H\) has finite VC dimension, then for fixed \(x_1, \dots, x_m\) there are \(d\) hypotheses \(h_1, \dots, h_d \in \mathcal H\) such that every \(h \in \mathcal H\) agrees on all \(m\) points with one of them.
   - With \(D_i = \exp(-y_i f_{\mathbf w}(x_i))/Z\), where \(Z\) makes \(D\) sum to 1, show \(\partial R/\partial w_j = -\sum_i D_i y_i h_j(x_i)\), and that this equals \(2\varepsilon_j - 1\) with \(\varepsilon_j = \sum_i D_i \mathbb 1[h_j(x_i) \ne y_i]\), the weighted error of \(h_j\). Conclude that if \(\varepsilon_j \le \tfrac12 - \gamma\), the slope along \(j\) has absolute value at least \(2\gamma\), so the steepest-coordinate rule picks a hypothesis with small weighted error, which is what AdaBoost asks of its weak learner.
   - Show that one round of AdaBoost, as in lesson 10, lowers \(R\) by at least \(-\log\sqrt{1 - 4\gamma^2}\).
