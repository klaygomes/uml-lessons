# Model Selection and Validation

## The number that went wrong

Lesson 10 gave the mail room a boosted filter. Each round of boosting adds one decision stump, a tiny hypothesis that looks at a single feature, and the filter after \(T\) rounds is a weighted vote of \(T\) stumps. You saw there that \(T\) is a dial on the bias-complexity tradeoff: too few rounds and the vote is too crude to describe junk, too many and it starts to describe the accidents of the training sequence. The obvious question is the one lesson 10 left hanging. How do you set \(T\)?

Your team answered it the way most teams would. Users had labelled 300 messages. The team trained on 100 of them and put the other 200 aside as a **test set**, examples the learner never sees during training. They ran boosting for 60 rounds, which gives sixty filters at once, one after each round, and measured every one of them on the 200 test messages. The filter after 11 rounds made the fewest mistakes: 33 out of 200, a test error of 16.5%. The report said “16.5% error on held-out data”, everyone nodded, and the filter shipped.

A month later the complaints came in, and they added up to a filter that was wrong on about 21% of incoming mail. Not a disaster. But not 16.5% either, and the difference is far too large to wave away.

Here is the strange part. Nobody cheated. No test message was ever used to train a stump. The 200 messages were set aside honestly, at random, before training began. So where did four and a half points of error hide?

This lesson is about that question, and its name is **model selection**: choosing among learners, or among settings of one learner (such as the number of rounds), using the data itself. By the end you will know exactly what the team did wrong, how much their number could be trusted, how to split a labelling budget so that the reported number is honest, and what to do when the honest number turns out to be bad.

## Worked example 1: sixty candidates, one test set

Our mail room for this lesson is a simulation we built, which lets us do something no real team can: compute the true risk of every filter exactly. Messages have 20 yes/no features. Seven of them are warning signs, and a message is junk when at least four of the seven are present, except that one label in ten is flipped, because users disagree and click carelessly (the noise of lesson 3). The best any hypothesis can do here, the Bayes risk, is therefore 0.10.

Here are nine of the team’s sixty filters. A quick reminder of the names from lesson 1, since they fill every table of this lesson. Write \(h_T\) for the filter you get after \(T\) rounds, so \(h_{11}\) is the vote of eleven stumps. Its **empirical risk** \(L_S(h_T)\) is the fraction of the 100 training examples it gets wrong; the subscript \(S\) names the training sequence it is scored on. Its **true risk** \(L_{\mathcal D}(h_T)\) is the chance that it mislabels the next message to arrive, where \(\mathcal D\) is the distribution incoming mail is drawn from. For each filter you see both, plus its error on the 200 test messages, which is the same kind of fraction as \(L_S\), only counted on the test messages. Before you read the table, guess: is the filter the test set picked also the filter with the smallest true risk?

| rounds \(T\) | empirical risk \(L_S\) | test error | true risk \(L_{\mathcal D}\) |
|---|---|---|---|
| 1 | 0.27 | 0.355 | 0.375 |
| 4 | 0.23 | 0.290 | 0.300 |
| 8 | 0.14 | 0.225 | 0.238 |
| 9 | 0.12 | 0.180 | 0.200 |
| 11 | 0.15 | **0.165** | 0.208 |
| 16 | 0.13 | 0.205 | 0.231 |
| 23 | 0.13 | **0.165** | 0.227 |
| 32 | 0.13 | 0.220 | 0.239 |
| 60 | 0.15 | 0.220 | 0.247 |

Look at three things. First, the truly best filter is \(T = 9\), with true risk 0.200, and the test set did not pick it. Second, the two winners on the test set, \(T = 11\) and \(T = 23\), have true risks 0.208 and 0.227, both higher than what the test set says about them. The team broke the tie toward fewer rounds and reported 0.165 for a filter whose true risk is 0.208. Third, and this is the heart of it: from \(T = 9\) onwards the true risks sit on a gentle plateau between 0.200 and 0.249, while the test errors of those same filters bounce between 0.165 and 0.235. Each test error is the true risk plus noise. When you pick the smallest of many noisy numbers, you are also picking the one whose noise happened to be most negative.

Was this test set unlucky? We can check, because the simulation can draw as many test sets as we like. Keep the same training sequence and the same sixty filters, draw 2,000 fresh test sets of 200 messages, and let the team’s procedure pick a winner on each. On average the reported test error is 0.188 while the true risk of the winner is 0.210, and the true risk is above the reported number on 80% of the draws. The gap is not bad luck. It is built into the procedure.

You may recognise the pattern from lesson 4, where the best-looking rule on a dashboard was the most flattered one: the winner’s curse. To see how strong it can be, strip the problem to its bones. Suppose sixty candidates all have true risk exactly 0.2, and their mistakes on the 200 test messages are independent. Each test error is then a binomial count divided by 200, and the chance that the smallest of sixty counts is at least \(k\) is the chance that one count is at least \(k\), raised to the sixtieth power. Python can add that up exactly:

```python
from math import comb

def tail(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p)**(n - i)
               for i in range(k, n + 1))

n, p, r = 200, 0.2, 60
expected_min = sum(tail(k, n, p) ** r for k in range(1, n + 1))
one_lucky = 1 - tail(34, n, p)
print(round(expected_min / n, 3))
print(round(one_lucky, 3), round(1 - (1 - one_lucky) ** r, 4))
```

It prints `0.137`, then `0.124 0.9996`. Every one of the sixty is a 20% filter, and yet the winner reports 13.7% on average. One candidate alone gets a test error of 16.5% or better only 12.4% of the time; the best of sixty does so 99.96% of the time. (The first line uses a small identity. Say a filter with risk 0.2 sees two messages, and \(N\) counts its mistakes. Its average count is \(2 \times 0.2 = 0.4\). The chance of at least one mistake is \(1 - 0.8^2 = 0.36\), the chance of two is \(0.2^2 = 0.04\), and \(0.36 + 0.04 = 0.4\) as well. That is no accident: for a count \(N\) that takes values \(0, 1, \dots, n\), \(\mathbb E[N] = \sum_{k \ge 1} \mathbb P[N \ge k]\). Here \(\mathbb E[N]\) is the expectation, the long-run average of \(N\); \(\mathbb P[N \ge k]\) is the probability that the count reaches \(k\); and \(\sum_{k \ge 1}\) adds those probabilities for \(k = 1, 2, \dots\) up. In words: **the average count is the sum, over every level \(k\), of the chance of reaching it**, because a count of 2 is reached at level 1 and at level 2, and so is counted twice. In the code, `tail(k, n, p)` is \(\mathbb P[N \ge k]\) for one candidate, raising it to the power `r` gives the chance that all sixty reach \(k\), since the minimum of independent counts is at least \(k\) exactly when every one of them is, and `sum(... for k in range(1, n + 1))` is the \(\sum_{k \ge 1}\).) The team’s real candidates are far from independent, since the filter after 11 rounds and the filter after 12 share eleven stumps, which is why their gap was milder than this one. But the mechanism is the same.

[[diagram:winner]]

Try it yourself before we name the cure. In the lab you hold the training sequence fixed, choose how many rounds to consider and how many test messages to use, and watch the reported number and the true risk of the winner, draw after draw.

[[lab]]

So the team’s number was optimistic because the test set did two jobs at once: it chose the filter, and it graded the filter it chose. There are two ways forward. One is to stop measuring and bound the true risk with a theorem, choosing \(T\) to make the bound small. The other is to keep measuring, but on examples that played no part in any choice. We’ll take them in that order.

## Choosing by a bound: structural risk minimization

Lesson 7 met a learner that faces a whole sequence of hypothesis classes, \(\mathcal H_1, \mathcal H_2, \mathcal H_3, \dots\), and does not want to commit to one of them in advance. For the boosted filter, take \(\mathcal H_T\) to be the class of weighted votes of \(T\) stumps, the class lesson 10 wrote \(L(B, T)\). For a polynomial fitted to points, \(\mathcal H_d\) would be the polynomials of degree at most \(d\). In both cases a larger index means a richer class: a smaller gap to the truth, and a larger gap between empirical and true risk.

To compare classes we need to know how much each one can be trusted on \(m\) examples, where \(m\) is the length of the training sequence. Recall from lesson 4 the idea of **uniform convergence**: once the training sequence is long enough, every hypothesis in the class has an empirical risk close to its true risk, all of them at the same time, except on a rare unlucky draw. Two numbers say what “close” and “rare” mean. The accuracy \(\varepsilon\) (the Greek letter epsilon) is how far apart the two risks may be, say 0.1; the confidence parameter \(\delta\) (delta) is the chance of an unlucky draw we tolerate, say 0.05. The **sample complexity** \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) is how many examples that takes for the class \(\mathcal H\); the superscript UC just stands for uniform convergence. Formally, for every distribution \(\mathcal D\), a training sequence of at least that many examples satisfies, with probability at least \(1 - \delta\), \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\) for every \(h \in \mathcal H\) at the same time. The bars \(|\cdot|\) around a difference are the absolute value, the distance between the two numbers whichever is larger, and \(h \in \mathcal H\) reads “\(h\) belongs to \(\mathcal H\)”.

A richer class needs more examples, and we will describe how rich class \(d\) is by a single number \(g(d)\), where \(g: \mathbb N \to \mathbb R\) is some increasing function from the counting numbers \(\mathbb N = \{1, 2, 3, \dots\}\) to the real numbers \(\mathbb R\). For binary classification you can take \(g(d)\) to be the VC dimension of \(\mathcal H_d\) times the universal constant from lesson 6’s fundamental theorem; for the boosting classes, \(g\) simply grows with \(T\). Halving the accuracy should cost four times the examples, and asking for more confidence should cost only a little. So assume that for every \(d\),

\[ m^{\mathrm{UC}}_{\mathcal H_d}(\varepsilon, \delta) \le \frac{g(d) \log(1/\delta)}{\varepsilon^2}. \]

Read it from the inside out.

- \(\log(1/\delta)\) is the natural logarithm of \(1/\delta\). At \(\delta = 0.05\), \(1/\delta = 20\) and \(\log 20 \approx 3.0\); making \(\delta\) ten times smaller only adds about 2.3.
- \(\varepsilon^2\) in the denominator is the accuracy squared, so \(\varepsilon = 0.1\) divides by 0.01, a factor of 100.
- \(g(d)\) multiplies everything by the richness of class \(d\).
- The subscript \(\mathcal H_d\) on \(m^{\mathrm{UC}}\) says whose sample complexity this is, and \(\le\) says the formula is a ceiling: the class may need fewer.

In plain words: **class \(d\) is trustworthy to within \(\varepsilon\), except with probability \(\delta\), once it has \(g(d)\log(1/\delta)/\varepsilon^2\) examples.** With \(g(d) = 5\), \(\varepsilon = 0.1\) and \(\delta = 0.05\), that is \(5 \times 3.0 / 0.01 \approx 1{,}498\) examples. Turn it around and ask what accuracy 1,498 examples buy: solving for \(\varepsilon\) gives back 0.1, and the proof below uses exactly this turning around.

```python
from math import log, sqrt

g_d, eps, delta = 5, 0.1, 0.05
m_needed = g_d * log(1 / delta) / eps**2
eps_back = sqrt(g_d * log(1 / delta) / m_needed)
print(round(m_needed), round(eps_back, 3))
```

It prints `1498 0.1`.

Now the idea of the proposition, in words first. Each class \(d\) makes its own promise, “empirical risk plus my margin is an upper bound on true risk”, and a richer class has a wider margin. We want all those promises to hold at once, for infinitely many classes, while still failing with total probability at most \(\delta\). The trick is to give class 1 most of the failure budget, class 2 a quarter as much, class 3 a ninth, and so on, so the shares add up to \(\delta\). The price of a smaller share is a slightly wider margin, which is where the \(2\log d\) below comes from.

**Proposition (the structural risk minimization bound).** Under that assumption, for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), for every \(d \in \mathbb N\) and every \(h \in \mathcal H_d\),

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{g(d)\left(\log(1/\delta) + 2\log d + \log(\pi^2/6)\right)}{m}}. \]

Before the formula, the setup: \(\delta \in (0, 1)\) says \(\delta\) is a number strictly between 0 and 1, and \(S \sim \mathcal D^m\) reads “\(S\) is drawn from \(\mathcal D^m\)”: \(m\) examples, each drawn independently from \(\mathcal D\). Now the bound, from the inside out. Inside the parentheses are three logarithms: \(\log(1/\delta)\) is the confidence cost from the assumption, \(2\log d\) is the extra cost of class \(d\) getting a smaller share of the failure budget, and \(\log(\pi^2/6) \approx 0.50\) is a small constant from the shares, with \(\pi \approx 3.14\). Multiplying by \(g(d)\), dividing by \(m\) and taking the square root \(\sqrt{\cdot}\) turns that into the margin of class \(d\), the **complexity term**. Adding it to \(L_S(h)\) gives the ceiling on \(L_{\mathcal D}(h)\). In plain words: **with probability at least \(1 - \delta\), every hypothesis in every class has a true risk no larger than its empirical risk plus a margin that grows with its class.** With \(g(d) = d = 1\), \(\delta = 0.05\) and \(m = 100\), the parentheses hold \(3.0 + 0 + 0.5 = 3.5\), divided by 100 gives 0.035, and its square root is 0.19: a stump with empirical risk 0.27 has true risk at most 0.46. The snippet a little further on, `srm_term`, computes this complexity term line for line.

*Proof.* We go in three steps: turn the sample complexity into an accuracy, give each class its own share of the failure probability, and add the shares up.

First, the accuracy one class can promise. Fix \(d\) and a confidence \(\delta_d\), and set \(\varepsilon_d = \sqrt{g(d)\log(1/\delta_d)/m}\). This is the turning around from the snippet above, where 1,498 examples bought back \(\varepsilon = 0.1\). Squaring and rearranging gives \(g(d)\log(1/\delta_d)/\varepsilon_d^2 = m\), so by the assumption \(m^{\mathrm{UC}}_{\mathcal H_d}(\varepsilon_d, \delta_d) \le m\). A training sequence of \(m\) examples is therefore long enough: with probability at least \(1 - \delta_d\), every \(h \in \mathcal H_d\) has \(|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon_d\). Call the complementary event, where some \(h \in \mathcal H_d\) is further off, \(F_d\). So \(\mathbb P[F_d] \le \delta_d\).

Second, the shares. We cannot give every class the full \(\delta\), because there are infinitely many classes and the failures could pile up. So we give class \(d\) the share \(\delta_d = w(d)\,\delta\) with weights \(w(d) = \frac{6}{\pi^2 d^2}\). These weights add up to exactly 1, because \(\sum_{d \ge 1} 1/d^2 = \pi^2/6\): the sum \(1 + 1/4 + 1/9 + \dots\) of the reciprocal squares, over every \(d\) from 1 on, is a famous constant, about 1.645. In plain words: **class \(d\) gets a share of the failure budget proportional to \(1/d^2\), and the shares use up exactly \(\delta\).** At \(\delta = 0.05\), class 1 gets 0.0304, class 2 gets 0.0076, class 3 gets 0.0034:

```python
from math import pi

delta = 0.05
w = lambda d: 6 / (pi**2 * d**2)
print(round(sum(w(d) for d in range(1, 10**6)), 5))
print([round(w(d) * delta, 4) for d in (1, 2, 3)])
```

It prints `1.0` and `[0.0304, 0.0076, 0.0034]`.

Third, the union bound from lesson 2: the chance that at least one of several events happens is at most the sum of their chances. The probability that some class fails is

\[ \mathbb P\Big[\bigcup_{d} F_d\Big] \le \sum_{d} \mathbb P[F_d] \le \sum_d w(d)\,\delta = \delta. \]

The big \(\bigcup_d\) is a union, “\(F_1\) or \(F_2\) or \(F_3\) or …”, the event that at least one class fails. The first \(\le\) is the union bound, the second uses \(\mathbb P[F_d] \le \delta_d = w(d)\,\delta\), and the last step is the weights adding to 1. In plain words: **the chance that any class breaks its promise is at most the sum of their shares, which is \(\delta\).** In the snippet, that is \(0.0304 + 0.0076 + 0.0034 + \dots = 0.05\).

Outside that event, every class \(d\) and every \(h \in \mathcal H_d\) satisfy \(L_{\mathcal D}(h) \le L_S(h) + \varepsilon_d\). It remains to write \(\varepsilon_d\) out. Since \(1/\delta_d = \pi^2 d^2/(6\delta)\), its logarithm is \(\log(1/\delta) + 2\log d + \log(\pi^2/6)\), which is the expression under the square root. \(\blacksquare\)

The rule that follows is called **structural risk minimization** (SRM): pick the pair \(d\) and \(h \in \mathcal H_d\) that makes the right-hand side smallest. It is bound minimization. The empirical risk falls as \(d\) grows, the complexity term rises, and SRM looks for the bottom of their sum. That is exactly what model selection asks for: a filter with 60 rounds may have a smaller empirical risk than a filter with 9, but if its complexity term is much larger, SRM prefers the 9.

[[diagram:srm]]

So why didn’t the team just do this? Put numbers in. Take the kindest case, \(g(d) = d\) with the constant set to 1, and \(\delta = 0.05\):

```python
from math import log, pi, sqrt

def srm_term(d, m, delta):
    g = d
    return sqrt(g * (log(1 / delta) + 2 * log(d)
                     + log(pi**2 / 6)) / m)

for m in (100, 1000):
    print(m, [round(srm_term(d, m, 0.05), 2)
              for d in (1, 5, 10, 20)])
```

It prints `100 [0.19, 0.58, 0.9, 1.38]` and `1000 [0.06, 0.18, 0.28, 0.44]`. With the team’s 100 training examples, the complexity term for \(d = 5\) is already 0.58, more than a coin’s error; the bound is true but says nothing. Even at 1,000 examples it adds 0.28 at \(d = 10\). The bound has to hold for every hypothesis in the class and every distribution at once, so it is pessimistic about the particular filter and the particular mail room in front of you. SRM is useful in some situations, and it was the right tool for proving that learning is possible in lesson 7. For picking \(T\) on a real budget, we want something sharper.

## Measuring instead: the hold-out set

Here is the sharper tool, and it is almost embarrassingly simple. To learn the true risk of one hypothesis \(h\), show it messages it has never had any influence on, and count. Had the team fixed \(h_{11}\) before looking at the test set, its 33 mistakes on 200 messages, 0.165, would have been exactly this kind of count.

**Definition (validation set).** A **validation set** is a sequence \(V = ((x_1, y_1), \dots, (x_{m_v}, y_{m_v}))\) of \(m_v\) fresh examples drawn i.i.d. from \(\mathcal D\), independently of the training sequence \(S\). (I.i.d., from lesson 1, means independently and identically distributed: every example comes from the same \(\mathcal D\) and no draw influences another. Like the training sequence, \(V\) is really a sequence, but “validation set” is the name everyone uses.) The **validation risk** of a hypothesis is its empirical risk on \(V\),

\[ L_V(h) = \frac{1}{m_v} \sum_{i=1}^{m_v} \ell\big(h, (x_i, y_i)\big). \]

Read it from the inside out. \((x_i, y_i)\) is the \(i\)-th validation example, a message and its label. \(\ell\big(h, (x_i, y_i)\big)\) is the loss from lesson 3, the price \(h\) pays on that example; for the 0-1 loss it is 1 for a mistake and 0 otherwise. \(\sum_{i=1}^{m_v}\) adds the losses for \(i = 1\) up to \(m_v\), and \(\frac{1}{m_v}\) turns the total into an average. The subscript \(v\) in \(m_v\) just marks it as the validation size. In plain words: **the average loss of \(h\) on the held-out examples.** For \(h_{11}\), 33 losses of 1 and 167 of 0 average to 0.165:

```python
losses = [1] * 33 + [0] * 167
m_v = len(losses)
L_V = sum(losses) / m_v
print(m_v, L_V)
```

It prints `200 0.165`.

Using \(L_V\) of the learner’s output as the estimate of its true risk is called **validation**. In practice you do not draw \(V\) separately: you take the examples you have, shuffle them, and hold some out before training. Drawing \(S\) and then an independent \(V\) is the same thing as randomly partitioning one i.i.d. pool into two parts, which is why \(V\) is also called a **hold-out set**.

The tool for the proof is **Hoeffding’s inequality**, which lesson 4 introduced. The idea: an average of many independent, bounded numbers rarely strays far from its expected value, and the chance of straying shrinks exponentially as you add more numbers. Take a filter of true risk 0.2 on 200 test messages. Its test error could in principle be anything from 0 to 1, but the chance that it lands more than 0.1 away from 0.2 is at most about 0.037. Formally, let \(\theta_1, \dots, \theta_n\) be i.i.d. random variables with values in \([a, b]\) and mean \(\mu\). Then for every \(\varepsilon > 0\),

\[ \mathbb P\left[\left|\frac{1}{n}\sum_{i=1}^n \theta_i - \mu\right| > \varepsilon\right] \le 2\exp\left(-\frac{2n\varepsilon^2}{(b - a)^2}\right). \]

Read it from the inside out.

- \(\theta_1, \dots, \theta_n\) (theta) are the \(n\) numbers being averaged, here one 0-or-1 loss per test message, so \(n = 200\).
- \([a, b]\) is the interval they live in; for losses, \(a = 0\) and \(b = 1\), so \((b - a)^2 = 1\).
- \(\mu\) (mu) is the mean, the value each \(\theta_i\) has on average; here the true risk, 0.2.
- \(\frac{1}{n}\sum_{i=1}^n \theta_i\) is their average, the test error, and \(|\cdot - \mu| > \varepsilon\) is the event that it misses \(\mu\) by more than \(\varepsilon\).
- On the right, \(\exp(x)\) means \(e^x\), with \(e \approx 2.718\). The exponent \(-2n\varepsilon^2/(b-a)^2\) gets more negative as \(n\) grows, so the bound shrinks fast.

In plain words: **the chance that an average of \(n\) bounded independent numbers misses its mean by more than \(\varepsilon\) is at most \(2e^{-2n\varepsilon^2/(b-a)^2}\).** With \(n = 200\) and \(\varepsilon = 0.1\), the exponent is \(-2 \times 200 \times 0.01 = -4\), and \(2e^{-4} \approx 0.037\). The exact binomial chance is far smaller still:

```python
from math import comb, exp

n, mu, eps, a, b = 200, 0.2, 0.1, 0, 1
bound = 2 * exp(-2 * n * eps**2 / (b - a)**2)
exact = sum(comb(n, k) * mu**k * (1 - mu)**(n - k)
            for k in range(n + 1) if abs(k / n - mu) > eps)
print(round(bound, 4), f"{exact:.1e}")
```

It prints `0.0366 3.2e-04`.

Applied to one fixed hypothesis, that says the validation risk is close to the true risk. With 200 validation messages and \(\delta = 0.05\) the margin comes out at 0.096: with probability 95%, the count lands within about ten points of the truth.

**Theorem (the hold-out guarantee).** Let \(h\) be a hypothesis and let the loss take values in \([0, 1]\). Let \(V\) be a validation set of size \(m_v\), drawn independently of \(h\). Then for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over the choice of \(V\),

\[ |L_V(h) - L_{\mathcal D}(h)| \le \sqrt{\frac{\log(2/\delta)}{2 m_v}}. \]

From the inside out: \(\log(2/\delta)\) is the confidence cost (the 2 comes from the 2 in front of Hoeffding’s bound), dividing by \(2m_v\) spreads it over the validation examples, and the square root turns it into a margin. On the left, \(|L_V(h) - L_{\mathcal D}(h)|\) is the distance between the measured and the true risk. In plain words: **with probability at least \(1 - \delta\), the validation risk of a hypothesis fixed in advance is within \(\sqrt{\log(2/\delta)/(2m_v)}\) of its true risk.** At \(m_v = 200\) and \(\delta = 0.05\), \(\log 40 \approx 3.69\), divided by 400 gives 0.0092, and the square root is 0.096. The snippet after the proof checks the formula against the exact binomial at \(m_v = 500\), and the `margin` function in the next section computes it with `r = 1`.

*Proof.* The hypothesis usually comes out of a learner, \(h = A(S)\), so it is itself random. The whole proof is about handling that carefully.

*Step 1: freeze the hypothesis.* Condition on the training sequence, \(S = s\) (and on any coin flips the learner makes). Given that, \(h = A(s)\) is one fixed function.

*Step 2: the validation set does not notice.* \(V\) was drawn independently of \(S\). Conditioning on an independent variable does not change a distribution, so given \(S = s\), the examples of \(V\) are still i.i.d. from \(\mathcal D\). This is the only place independence is used, and it is the step that matters most.

*Step 3: losses are i.i.d. with the right mean.* Set \(\theta_i = \ell(h, (x_i, y_i))\). With \(h\) fixed, each \(\theta_i\) is a function of the \(i\)-th example only, so the \(\theta_i\) are i.i.d. They lie in \([0, 1]\), and their mean is \(\mathbb E[\theta_i] = \mathbb E_{z \sim \mathcal D}[\ell(h, z)] = L_{\mathcal D}(h)\), the definition of the true risk: the average loss of \(h\) on an example \(z\) drawn from \(\mathcal D\). Their average is \(L_V(h)\).

*Step 4: Hoeffding, conditionally.* With \(a = 0\), \(b = 1\) and \(n = m_v\),

\[ \mathbb P\left[|L_V(h) - L_{\mathcal D}(h)| > \varepsilon \,\middle|\, S = s\right] \le 2 e^{-2 m_v \varepsilon^2}. \]

This is Hoeffding with \((b - a)^2 = 1\); the only new mark is the vertical bar inside \(\mathbb P[\cdot \mid S = s]\), read “given that \(S = s\)”: the probability computed in the world where the training sequence is already known.

*Step 5: unfreeze.* The bound on the right does not depend on \(s\). By the law of total probability, the unconditional probability is the average of the conditional ones over the distribution of \(S\), and an average of numbers that are each at most \(2e^{-2m_v\varepsilon^2}\) is at most \(2e^{-2m_v\varepsilon^2}\).

*Step 6: solve for \(\varepsilon\).* Setting \(2e^{-2m_v\varepsilon^2} = \delta\) gives \(\varepsilon = \sqrt{\log(2/\delta)/(2m_v)}\). \(\blacksquare\)

Notice what is missing from the bound: the learner, the hypothesis class, the size of the training sequence. It holds whether \(h\) is a 60-round vote or a lookup table. Here it is against the exact binomial probability, for a hypothesis of true risk 0.2 and a validation set of 500:

```python
from math import comb, log, sqrt

L_D, m_v, delta = 0.2, 500, 0.05
eps = sqrt(log(2 / delta) / (2 * m_v))
fail = sum(comb(m_v, k) * L_D**k * (1 - L_D)**(m_v - k)
           for k in range(m_v + 1)
           if abs(k / m_v - L_D) > eps)
print(round(eps, 4), round(fail, 4))
```

It prints `0.0607 0.0007`. The theorem promises that \(L_V\) lands within 0.0607 of 0.2 except with probability at most 0.05; the real failure probability is 0.0007. Hoeffding is conservative, but it never lies.

How does this compare with the bounds of earlier lessons? Suppose \(h\) came from ERM over a class of VC dimension \(d\). Lesson 6’s fundamental theorem gives \(L_{\mathcal D}(h) \le L_S(h) + C\sqrt{(d + \log(1/\delta))/m}\) for a universal constant \(C\), while the hold-out guarantee gives \(L_{\mathcal D}(h) \le L_V(h) + \sqrt{\log(2/\delta)/(2m_v)}\). With \(m = m_v = 1000\), \(d = 21\) (halfspaces over our 20 features, as lesson 9 counted) and, very generously, \(C = 1\):

```python
from math import log, sqrt

m, d, delta, C = 1000, 21, 0.05, 1
vc_term = C * sqrt((d + log(1 / delta)) / m)
holdout_term = sqrt(log(2 / delta) / (2 * m))
print(round(vc_term, 3), round(holdout_term, 3))
```

It prints `0.155 0.043`. With validation sets of the same order as the training sequence, the estimate is sharper by a factor that grows like the square root of the VC dimension. The price is real, though: those validation examples are labels your users clicked that the learner does not get to train on.

[[diagram:holdout]]

Now we can say exactly what went wrong in the mail room. The team reported \(L_V(h_{11})\), using the test set as \(V\). But \(h_{11}\) was not some hypothesis fixed before \(V\) was drawn: it was chosen *because* of \(V\). Step 2 fails. Given that the procedure picked \(h_{11}\), the test messages are no longer a fresh i.i.d. draw from \(\mathcal D\); they are a draw on which \(h_{11}\) is known to have done at least as well as fifty-nine rivals. The theorem’s hypothesis was violated, so its conclusion was never on offer.

## Validation for model selection

Choosing was not the mistake. Validation is a natural tool for choosing. Train several learners, or one learner with several settings, on \(S\), and collect their outputs in a **candidate set** \(H = \{h_1, \dots, h_r\}\). For the boosted filter, \(h_T\) is the vote after \(T\) rounds. Then draw a fresh validation set and pick the candidate with the smallest validation risk. In other words, run ERM over \(H\), on \(V\).

This is learning a finite class, the problem lesson 2 solved, with one twist: \(H\) is not fixed in advance, because it depends on \(S\). But \(V\) is independent of \(S\), and so independent of \(H\), and that is enough for the finite-class argument to go through. The only mistake the team made was charging for one hypothesis when they had compared sixty. Charging properly means splitting the failure budget \(\delta\) among the sixty, each getting \(\delta/60\), and on 200 messages that widens the margin from 0.096 to 0.139.

**Theorem (validation over a finite candidate set).** Let \(H = \{h_1, \dots, h_r\}\) be any set of hypotheses and let the loss take values in \([0, 1]\). Let \(V\) be a validation set of size \(m_v\), drawn independently of \(H\). Then for every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over the choice of \(V\),

\[ \forall h \in H: \quad |L_{\mathcal D}(h) - L_V(h)| \le \sqrt{\frac{\log(2|H|/\delta)}{2 m_v}}. \]

Only two marks are new compared with the hold-out guarantee. \(\forall h \in H\) reads “for every \(h\) in \(H\)”, and \(|H|\), bars around a set, is how many candidates it has, so \(|H| = r\). The confidence cost \(\log(2|H|/\delta)\) is the single-hypothesis cost with \(\delta\) replaced by each candidate’s share \(\delta/|H|\). In plain words: **with probability at least \(1 - \delta\), every one of the \(r\) candidates has a validation risk within \(\sqrt{\log(2r/\delta)/(2m_v)}\) of its true risk, all at once.** For the team, \(r = 60\) and \(m_v = 200\): \(\log(2 \times 60/0.05) = \log 2400 \approx 7.78\), divided by 400 is 0.0195, and the square root is 0.139. The `margin` function below computes exactly this.

*Proof.* Condition on \(H\), exactly as in steps 1 and 2 of the hold-out proof: given \(H\), its \(r\) members are fixed functions, and \(V\) is still i.i.d. from \(\mathcal D\). Set \(\varepsilon = \sqrt{\log(2r/\delta)/(2m_v)}\) and let \(B_j\) be the event that \(|L_V(h_j) - L_{\mathcal D}(h_j)| > \varepsilon\). For each fixed \(j\), the hold-out guarantee applied with confidence \(\delta/r\) gives

\[ \mathbb P[B_j \mid H] \le 2e^{-2 m_v \varepsilon^2} = 2e^{-\log(2r/\delta)} = \frac{\delta}{r}. \]

The events \(B_1, \dots, B_r\) are far from independent (all of them are about the same \(V\)), but the union bound does not care:

\[ \mathbb P\Big[\bigcup_{j=1}^r B_j \,\Big|\, H\Big] \le \sum_{j=1}^r \frac{\delta}{r} = \delta. \]

This is the same union bound as in the structural risk minimization proof, now over \(r\) equal shares of \(\delta/r\) instead of shares shrinking like \(1/d^2\).

Outside the union, every candidate is within \(\varepsilon\). The bound does not depend on \(H\), so averaging over \(H\), as in step 5, keeps it. \(\blacksquare\)

What does this buy for the hypothesis we actually pick? Write \(\hat h\) for the candidate with the smallest validation risk and \(h^\circ\) for the candidate with the smallest true risk. On the good event of the theorem, three inequalities chain together:

\[ L_{\mathcal D}(\hat h) \le L_V(\hat h) + \varepsilon \le L_V(h^\circ) + \varepsilon \le L_{\mathcal D}(h^\circ) + 2\varepsilon. \]

The first and the last are the theorem, applied to \(\hat h\) and to \(h^\circ\). The middle one holds because \(\hat h\) has the smallest validation risk of all candidates, \(h^\circ\) included. So **the chosen candidate is within \(2\varepsilon\) of the best candidate**, and its reported validation risk is within \(\varepsilon\) of its own true risk, both with probability at least \(1 - \delta\).

Push the team’s numbers through, reading the test set as a validation set over sixty candidates, so \(\varepsilon = 0.139\). The chosen \(\hat h\) is \(h_{11}\) and the best candidate \(h^\circ\) is \(h_9\). Left to right: 0.208, then \(0.165 + 0.139 = 0.304\), then \(0.180 + 0.139 = 0.319\), then \(0.200 + 2 \times 0.139 = 0.478\). Each number is at least the one before, as promised:

```python
eps = 0.139
L_V = {11: 0.165, 9: 0.180}
L_D = {11: 0.208, 9: 0.200}
print(L_D[11], round(L_V[11] + eps, 3),
      round(L_V[9] + eps, 3), round(L_D[9] + 2 * eps, 3))
```

It prints `0.208 0.304 0.319 0.478`.

Here is how the margin \(\varepsilon\) grows with the number of candidates, at \(\delta = 0.05\), for validation sets of 200 and 2,000:

```python
from math import log, sqrt

def margin(r, m_v, delta=0.05):
    return sqrt(log(2 * r / delta) / (2 * m_v))

for r in (1, 7, 60, 1000):
    print(r, round(margin(r, 200), 3), round(margin(r, 2000), 3))
```

It prints:

```
1 0.096 0.03
7 0.119 0.038
60 0.139 0.044
1000 0.163 0.051
```

Read the third line. The team compared 60 candidates on 200 messages, so the honest statement was: “with probability at least 95%, the true risk of the filter we chose is at most \(0.165 + 0.139 = 0.304\).” The true risk was 0.208. Their number was never a lie about the filter; it was a lie about how much it could be trusted.

Notice also how gently the margin grows. The number of candidates enters only through a logarithm, so going from one candidate to a thousand costs less than doubling it. That is a gift, and it is also a trap: if you try very many methods, so that \(|H|\) is large compared with the validation set, you are back in the business of overfitting, this time to \(V\).

[[diagram:union]]

## The model-selection curve

Before trusting any single winner, it pays to look at the whole picture. The **model-selection curve** plots, against the complexity of the candidates (here the number of rounds \(T\)), two numbers: the empirical risk on the training sequence and the validation risk. Here it is for the team’s 100 training examples, with a fresh validation set of 1,000 messages, at a few values of \(T\):

| rounds \(T\) | 1 | 2 | 4 | 8 | 9 | 10 | 11 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| \(L_S\) | 0.27 | 0.27 | 0.23 | 0.14 | 0.12 | 0.14 | 0.15 | 0.13 | 0.13 | 0.15 |
| \(L_V\) | 0.351 | 0.351 | 0.284 | 0.237 | 0.191 | 0.205 | 0.194 | 0.206 | 0.221 | 0.231 |
| \(L_{\mathcal D}\) | 0.375 | 0.375 | 0.300 | 0.238 | 0.200 | 0.209 | 0.208 | 0.231 | 0.239 | 0.249 |

The empirical risk drops quickly and then flattens around 0.13, because the vote fits the training sequence better as it grows. (It wobbles a little rather than falling every round, because boosting minimizes a smooth stand-in for the 0-1 loss, not the 0-1 loss itself.) The validation risk falls too, bottoms out near \(T = 9\), and then climbs: past that point each new stump fits accidents of the 100 examples, and the true risk climbs with it. A falling empirical risk with a rising validation risk is overfitting, caught in the act.

The curve also tells you whether you searched in the right place. Real learners often have several parameters, some of them real numbers (lesson 13’s regularization strength is one), so you cannot try every value. The working method is to start with a rough grid, plot the curve, and then zoom in with a finer grid where the action is. It matters to check that the grid actually covers the interesting regime. On this validation set:

- the grid \(\{1, 30, 60\}\) picks \(T = 30\), true risk 0.242;
- the grid \(\{1, 2, 4, 8, 16, 32, 64\}\) picks \(T = 16\), true risk 0.231;
- zooming in on \(\{6, 7, \dots, 12\}\), around the bottom of the second grid’s curve, picks \(T = 9\), true risk 0.200, the best of all 64.

Someone who stopped at the first grid would ship a poor filter and never know there was a better one.

One piece of bookkeeping before we move on. Zooming in means the second grid was chosen after looking at \(V\), so the candidate set depends on \(V\), and the theorem as stated does not cover it. Here there is an easy repair: every filter you could possibly have compared is one of \(h_1, \dots, h_{64}\), all fixed by \(S\) alone, so applying the theorem to all 64 covers any way of choosing among them, adaptive or not. The rule of thumb: count every candidate you might have looked at, not just the ones in the final grid.

[[diagram:selection-curve]]

## k-fold cross validation

Validation assumes you can afford to set examples aside. When labels are scarce, holding out a fifth of them for validation means training on a fifth less. **k-fold cross validation** is designed to give a good estimate of the true risk without wasting so much data.

Partition the training sequence \(S\) of \(m\) examples into \(k\) parts, called **folds**, \(S_1, \dots, S_k\), each of size \(m/k\) (assume \(k\) divides \(m\)). For each fold \(i\), train on the other \(k - 1\) folds, \(S \setminus S_i\), and measure the result on fold \(i\). The average of these \(k\) errors is the estimate of the true risk. The special case \(k = m\), where each fold is a single example, is called **leave-one-out**.

For model selection, do this for every parameter value \(\theta\) in a set \(\Theta\), pick the value with the smallest average, and then retrain with that value on all of \(S\). Write \(A(S'; \theta)\) for the learner run on \(S'\) with parameter \(\theta\):

```
KFoldSelect(S, Θ, A, k):
  partition S into S_1, S_2, ..., S_k
  for each θ in Θ:
    for i = 1..k:
      h_{i,θ} ← A(S \ S_i; θ)
    error(θ) ← (1/k) · Σ_{i=1..k} L_{S_i}(h_{i,θ})
  θ* ← argmin_{θ in Θ} error(θ)
  return θ*, h_{θ*} = A(S; θ*)
```

Reading the notation: \(S \setminus S_i\) is “\(S\) without fold \(i\)”, \(L_{S_i}(h_{i,\theta})\) is the empirical risk of the fold-\(i\) hypothesis on the fold it never saw, so `error(θ)` is the average of the \(k\) held-out errors, and \(\operatorname{argmin}_{\theta \in \Theta}\) returns the parameter value, not the error, at which `error(θ)` is smallest. With \(k = 5\) and 100 examples, each fold has 20 messages and each training run uses 80. The snippet below mirrors it: `rest` is \(S \setminus S_i\), `risk(A(rest), f)` is \(L_{S_i}(h_{i,\theta})\), and `sum(errs) / k` is `error(θ)`.

Let’s watch it choose between two learners on 100 examples from a plain mail room, where label = feature 1 with 10% of labels flipped: ERM over the twenty one-feature hypotheses of lesson 1, and lesson 1’s memorizer.

```python
import random

rng = random.Random(2)
S = []
for _ in range(100):
    x = tuple(rng.randint(0, 1) for _ in range(20))
    S.append((x, x[0] ^ (rng.random() < 0.1)))
def risk(h, T):
    return sum(h(x) != y for x, y in T) / len(T)
def one_feature(T):
    j = min(range(20), key=lambda j: risk(lambda x: x[j], T))
    return lambda x: x[j]
def memorize(T):
    return lambda x, table=dict(T): table.get(x, 0)

k = 5
folds = [S[i * 20:(i + 1) * 20] for i in range(k)]
for A in (one_feature, memorize):
    errs = []
    for f in folds:
        rest = [z for g in folds if g is not f for z in g]
        errs.append(risk(A(rest), f))
    print(A.__name__, [round(e, 2) for e in errs],
          round(sum(errs) / k, 2))
```

It prints `one_feature [0.05, 0.05, 0.15, 0.15, 0.0] 0.08` and `memorize [0.55, 0.4, 0.45, 0.5, 0.65] 0.51`. The one-feature learner wins, as it should; its true risk is 0.10, and the estimate 0.08 is close. Every one of the 100 examples did double duty: it was trained on four times and tested on once.

Is there a guarantee like the hold-out theorem? Each single term \(L_{S_i}(h_{i,\theta})\) is an honest hold-out estimate, because \(h_{i,\theta}\) never saw fold \(i\). But it estimates the risk of \(h_{i,\theta}\), a hypothesis trained on less data, and the hypothesis you ship is a different one, \(A(S; \theta^*)\). Cross validation often works very well in practice, yet in general it can fail, and the first problem at the end of this lesson builds a learner for which the leave-one-out estimate is wrong by exactly \(1/2\) every time. Exactly when it works is still not fully understood. It is known to give good estimates for local rules such as nearest neighbours (lesson 19), a result of Rogers and Wagner, and for stable learners, whose output barely moves when one example changes (lesson 13).

[[diagram:kfold]]

## Train, validation, test

Now the fix assembles itself. In most practical work you split the labelled examples into three parts:

1. a **training sequence** \(S\), used to train every candidate;
2. a **validation set** \(V\), used to choose among the candidates;
3. a **test set**, used once, at the end, to estimate the true risk of the chosen hypothesis. Write its size \(m_t\) and the chosen hypothesis’s error on it \(L_{\text{test}}(\hat h)\).

Each part has its own guarantee. The validation set gives the selection guarantee: \(\hat h\) is within \(2\varepsilon_v\) of the best candidate, with \(\varepsilon_v = \sqrt{\log(2r/\delta)/(2m_v)}\). The test set is independent of both \(S\) and \(V\), so it is independent of \(\hat h\), and the hold-out guarantee applies to it with \(\varepsilon_t = \sqrt{\log(2/\delta)/(2m_t)}\), no union bound needed. If each guarantee fails with probability at most \(\delta\), the union bound says both hold together with probability at least \(1 - 2\delta\).

How many labels does that take? Turn the margins around: a margin \(\varepsilon\) over \(r\) hypotheses needs \(m \ge \log(2r/\delta)/(2\varepsilon^2)\). That is the margin formula squared and solved for the number of examples: **the set must be at least the confidence cost divided by twice the squared margin.** For \(r = 1\), \(\varepsilon = 0.05\) and \(\delta = 0.05\), that is \(\log 40 / 0.005 \approx 3.69/0.005\), which rounds up to 738; `size_for` below computes it. Suppose the manager wants the reported number within 0.05 of the truth, and the chosen filter within 0.1 of the best candidate (so \(\varepsilon_v = 0.05\)), each with probability at least 95%, from a budget of 2,500 labels:

```python
from math import ceil, log

def size_for(margin, r, delta=0.05):
    return ceil(log(2 * r / delta) / (2 * margin**2))

budget = 2500
m_t = size_for(0.05, 1)
for r in (7, 64):
    m_v = size_for(0.05, r)
    print(r, m_v, m_t, budget - m_v - m_t)
```

It prints `7 1127 738 635` and `64 1570 738 192`. With seven candidates (the doubling grid), the validation set needs 1,127 labels, the test set 738, and 635 are left for training. With all 64 candidates, validation needs 1,570 and only 192 remain to train on. The test set costs the same either way, because it checks one hypothesis.

[[diagram:split]]

There is a subtlety worth facing squarely. The finite-candidate theorem holds for *every* candidate at once, so it also covers the one you chose. That means you could, in principle, choose on a set and report on the same set, as long as you report with the margin for \(r\) hypotheses, not for one. The team’s real crime was the margin, not the reuse. So why keep a separate test set at all? Two reasons. Its margin pays for one hypothesis instead of \(r\), which is cheap. And it protects you from the candidates you forgot to count: the other learners someone tried last quarter, the feature that was added after a bad validation score, the grid you zoomed. All of those were choices made by looking at the data, and only a set that nobody looked at is safe from all of them. The rule people follow is the simple one: touch the test set once.

## Fix the bug

Back to the mail room, with a fresh budget of 2,500 labels. Your manager’s conditions are the ones above: the number you report must be within 0.05 of the filter’s true risk, the filter you ship must be within 0.1 of the best candidate you considered, each with probability at least 95%, and the reported number plus its margin, the risk you can actually certify, must be at most 0.20.

In the lab you choose the grid of rounds to consider and how to split the 2,500 labels between training, validation and test. The lab computes the margins your split earns, honestly, from the theorems of this lesson; if you give the validation set nothing, the test set does the choosing and is charged for every candidate. Work out the margins before you train.

[[lab-act2]]

## What to do if learning fails

Now suppose you did everything right: a hypothesis class, a learner, parameters tuned on a validation set, a number from an untouched test set. And the number is bad. What went wrong, and what should you do next?

There are many things you could change:

- get a larger training sequence;
- change the hypothesis class: enlarge it, reduce it, replace it entirely, or change the parameters you consider;
- change the features that describe each message;
- change the optimization algorithm that carries out your learning rule.

Which one helps depends on why the hypothesis is bad, and lesson 5 gave us the language for “why”. The error of what you ship has two sources. Part of it is the class’s fault: even the best hypothesis in the class makes mistakes. The rest is the data’s fault: the learner, working from finitely many examples, picked something worse than that best one. On the plain mail room with the twenty one-feature hypotheses, the best is feature 1, with true risk 0.10, and every other feature is a coin flip with true risk 0.5. If the learner picked feature 3, the class is to blame for 0.10 and the data for the remaining 0.4.

Now the symbols. Let \(h^*\) be a hypothesis with the smallest true risk in \(\mathcal H\), \(h^* \in \operatorname{argmin}_{h \in \mathcal H} L_{\mathcal D}(h)\): \(\operatorname{argmin}\) is the set of hypotheses at which \(L_{\mathcal D}\) is smallest, and \(\in\) says \(h^*\) is one of them (there may be ties). Let \(h_S\) be what the learner returned on \(S\). The **approximation error** is \(L_{\mathcal D}(h^*)\), and the **estimation error** is \(L_{\mathcal D}(h_S) - L_{\mathcal D}(h^*)\). In plain words: **the approximation error is the true risk of the best hypothesis in the class, and the estimation error is how much worse the learner’s pick is than that best one.**

```python
true_risk = {j: 0.5 for j in range(1, 21)}
true_risk[1] = 0.10
h_star = min(true_risk, key=true_risk.get)
h_S = 3
approx = true_risk[h_star]
estim = true_risk[h_S] - true_risk[h_star]
print(h_star, approx, round(estim, 2))
```

It prints `1 0.1 0.4`: `min(..., key=...)` is the argmin, returning the feature, not its risk.

The approximation error depends only on \(\mathcal D\) and \(\mathcal H\): not on the number of examples, not on the learner. So if it is large, more examples are useless and a smaller class only makes it worse. What can help is a larger class, a different class that encodes different prior knowledge, or a different description of the messages (lesson 25 is about building features). The estimation error does depend on the number of examples. If it is large, more examples help, and so does a smaller class; a larger class makes it worse.

The trouble is that the approximation error involves \(h^*\) and \(\mathcal D\), which we never see. So let’s split the true risk into pieces we *can* measure, using the training sequence and a validation set:

\[ L_{\mathcal D}(h_S) = \big(L_{\mathcal D}(h_S) - L_V(h_S)\big) + \big(L_V(h_S) - L_S(h_S)\big) + L_S(h_S). \]

Nothing here is new notation, only a trick: add and subtract \(L_V(h_S)\) and \(L_S(h_S)\), so the three brackets telescope back to \(L_{\mathcal D}(h_S)\). In plain words: **true risk equals (true minus validation) plus (validation minus empirical) plus empirical.** The first term is small and known to be small: the hold-out guarantee bounds it. When the second term, the gap between validation and empirical risk, is large, we say the learner **overfits**. When the last term, the empirical risk, is large, we say it **underfits**. For the team’s boosted filters, with the 1,000-message validation set, the three pieces add back up to the true risk:

```python
rows = {1: (0.27, 0.351, 0.375), 11: (0.15, 0.194, 0.208),
        64: (0.15, 0.231, 0.249)}
for T, (L_S, L_V, L_D) in rows.items():
    parts = (L_D - L_V, L_V - L_S, L_S)
    print(T, [round(p, 3) for p in parts],
          round(sum(parts), 3))
```

It prints `1 [0.024, 0.081, 0.27] 0.375`, `11 [0.014, 0.044, 0.15] 0.208` and `64 [0.018, 0.081, 0.15] 0.249`, the rows of this table read right to left:

| \(T\) | \(L_S\) | \(L_V - L_S\) | \(L_{\mathcal D} - L_V\) | \(L_{\mathcal D}\) |
|---|---|---|---|---|
| 1 | 0.27 | 0.081 | 0.024 | 0.375 |
| 11 | 0.15 | 0.044 | 0.014 | 0.208 |
| 64 | 0.15 | 0.081 | 0.018 | 0.249 |

One stump underfits: its empirical risk alone is 0.27. Sixty-four rounds overfit: the empirical risk is no better than at 11 rounds, but the gap to the validation risk has doubled.

[[diagram:decomposition]]

Be careful, though: these two terms are not the estimation and approximation errors, and they can be badly different. Take a class of VC dimension \(d\) whose approximation error is \(1/4\), and a training sequence of fewer than \(d\) examples that the class can label in every possible way. Then ERM reaches \(L_S(h_S) = 0\), while the approximation error is \(1/4\). Still, the measurable terms carry real information, in both directions.

**When the empirical risk is large.** Split it once more, around \(h^*\):

\[ L_S(h_S) = \big(L_S(h_S) - L_S(h^*)\big) + \big(L_S(h^*) - L_{\mathcal D}(h^*)\big) + L_{\mathcal D}(h^*). \]

The same telescoping trick, now around the best hypothesis \(h^*\): **the empirical risk equals (how much better the learner’s pick looks than \(h^*\) on \(S\)) plus (how flattering \(S\) is to \(h^*\)) plus the approximation error.** The proposition below bounds the first two pieces.

**Proposition (a large empirical risk means a large approximation error).** If \(h_S\) is an ERM hypothesis over \(\mathcal H\) and the loss lies in \([0, 1]\), then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ L_{\mathcal D}(h^*) \ge L_S(h_S) - \sqrt{\frac{\log(2/\delta)}{2m}}. \]

The square root is the hold-out margin again, with the training size \(m\) in place of \(m_v\). In plain words: **the approximation error is at least the empirical risk minus the single-hypothesis margin.** With \(m = 400\) and \(\delta = 0.05\) the margin is 0.068, so an empirical risk of 0.485 forces an approximation error of at least 0.417; the snippet after the proof computes exactly that.

*Proof.* The first term of the split is at most 0: ERM returns a hypothesis with the smallest empirical risk in \(\mathcal H\), and \(h^*\) is in \(\mathcal H\), so \(L_S(h_S) \le L_S(h^*)\). For the second term, \(h^*\) is defined by \(\mathcal D\) and \(\mathcal H\) alone, so it is one fixed hypothesis that does not depend on \(S\). The training sequence therefore plays the role of a validation set for \(h^*\), and the hold-out guarantee gives, with probability at least \(1 - \delta\), \(L_S(h^*) - L_{\mathcal D}(h^*) \le \sqrt{\log(2/\delta)/(2m)}\). Putting the three terms back together, \(L_S(h_S) \le 0 + \sqrt{\log(2/\delta)/(2m)} + L_{\mathcal D}(h^*)\), which rearranges to the claim. \(\blacksquare\)

A learner that can only answer with a constant, on a mail room where junk and keep are equally common, shows it at work:

```python
import random
from math import log, sqrt

rng = random.Random(1)
m, delta = 400, 0.05
ys = [rng.randint(0, 1) ^ (rng.random() < 0.1) for _ in range(m)]
train_risk = min(sum(ys), m - sum(ys)) / m
eps = sqrt(log(2 / delta) / (2 * m))
print(round(train_risk, 3), round(eps, 3),
      round(train_risk - eps, 3))
```

It prints `0.485 0.068 0.417`: an empirical risk of 0.485 on 400 examples proves, with 95% confidence, that no constant has true risk below 0.417. (The truth is 0.5.) No amount of data will rescue this class. Enlarge it, change it, or change the features.

There is one escape hatch in that argument, and it is the word ERM. The approximation error can be small while \(L_S(h_S)\) is large if the learner is *not* returning an ERM hypothesis. Maybe there is a bug in your implementation. Maybe finding an ERM hypothesis is computationally hard, so the learner runs a heuristic that only tries. Sometimes you can tell: some problems come with optimality conditions you can check, which certify that the optimizer really reached the minimum, and the next lesson is about exactly such problems. When the answer depends on a random starting point, restarting from several points and seeing whether better solutions appear is a cheap test.

**When the empirical risk is small.** Now the ambiguity bites. Picture two situations, both with ERM over a class of VC dimension \(d\), and both with \(L_S(h_S) = 0\). In the first, there are fewer than \(d\) examples and the approximation error is high. In the second, there are more than \(2d\) examples and the approximation error is zero. The empirical risk cannot tell them apart. What can?

**Learning curves.** Train the learner on growing prefixes of the training sequence (the first 10%, the first 20%, and so on), and for each prefix plot two numbers: the empirical risk on that prefix and the validation risk on a fixed validation set. In the first situation the validation risk sits near \(1/2\) for every prefix, because nothing is being learned. In the second, it starts high but then falls, and it must fall once the prefix is larger than the VC dimension.

Here are two learners on the plain mail room (label = feature 1, 10% flipped, 20 features), with a validation set of 2,000: the memorizer, whose class has VC dimension \(2^{20}\), far more than any prefix, and ERM over one-feature hypotheses, a class of VC dimension at most 4 that contains a hypothesis with true risk 0.10. Each pair is (empirical risk, validation risk).

```python
import random

rng = random.Random(0)
def draw(m):
    xs = [[rng.randint(0, 1) for _ in range(20)]
          for _ in range(m)]
    return [(tuple(x), x[0] ^ (rng.random() < 0.1)) for x in xs]
def risk(h, S):
    return sum(h(x) != y for x, y in S) / len(S)
def memorize(S):
    return lambda x, table=dict(S): table.get(x, 0)
def one_feature(S):
    j = min(range(20), key=lambda j: risk(lambda x: x[j], S))
    return lambda x: x[j]

V = draw(2000)
for m in (25, 100, 400, 1600):
    S = draw(m)
    hs = [memorize(S), one_feature(S)]
    print(m, [(round(risk(h, S), 2), round(risk(h, V), 2))
              for h in hs])
```

It prints:

```
25 [(0.0, 0.48), (0.16, 0.1)]
100 [(0.0, 0.48), (0.1, 0.1)]
400 [(0.0, 0.48), (0.1, 0.1)]
1600 [(0.0, 0.48), (0.1, 0.1)]
```

The memorizer is the first situation: empirical risk 0, validation risk stuck near \(1/2\) at every size. The one-feature learner has a class small enough that 25 examples already put it close to the end of its curve, and both of its numbers sit at 0.10, its approximation error.

[[diagram:learning-curves]]

In general, as long as the approximation error is above zero, expect the empirical risk to *rise* with the number of examples, since more examples are harder to explain all at once, and the validation risk to *fall*. When the VC dimension is finite and the number of examples grows without end, both converge to the approximation error. So by extrapolating the two curves you can guess the approximation error, or at least an interval that contains it.

That gives the remedy for a small empirical risk with a large validation risk. Either way, the training sequence is too small for this class. If the validation curve has started to fall, more examples are the best cure, if you can afford them; a simpler class is the other reasonable move. If the validation curve sits near \(1/2\), you have no evidence that the approximation error is small, and more examples may not help at all. More data can still settle it, since eventually you would see the validation risk fall or the empirical risk rise. But if labels are expensive, try reducing the complexity of the class first.

Put together, this is the procedure:

1. If learning involves parameters, plot the model-selection curve and make sure you tuned them in the right regime.
2. If the empirical risk is too large, enlarge the class, replace it, or change the features.
3. If the empirical risk is small, plot learning curves and decide whether the problem is estimation error or approximation error.
4. If the approximation error looks small enough, get more data; if you can’t, reduce the complexity of the class.
5. If the approximation error looks large too, change the class or the features entirely.

## The procedures as code

Three procedures came up in this lesson. Write \(\mathrm{Train}(n)\) for the time the learner \(A\) needs on \(n\) examples and \(c\) for the time to evaluate one hypothesis on one example. As in earlier lessons, \(O(\cdot)\) reads “grows at most like, up to a constant factor”, and \(|\Theta|\) is how many parameter values you try.

```
HoldOutSelect(S, V, Θ, A):
  for each θ in Θ:
    h_θ ← A(S; θ)
    err(θ) ← L_V(h_θ)
  θ̂ ← argmin_{θ in Θ} err(θ)
  return θ̂, h_θ̂, err(θ̂)

LearningCurve(S, V, A, fractions):
  for each f in fractions:
    P ← the first ⌈f·m⌉ examples of S
    h ← A(P)
    record (|P|, L_P(h), L_V(h))
  return the records
```

together with **KFoldSelect** from the cross-validation section.

**HoldOutSelect** trains \(|\Theta|\) times and evaluates each candidate on \(V\): time \(O(|\Theta|(\mathrm{Train}(m) + m_v c))\), and space for one candidate at a time plus \(|\Theta|\) numbers. Some learners hand you many candidates for the price of one. Boosting for \(T_{\max}\) rounds produces \(h_1, \dots, h_{T_{\max}}\) along the way, and the vote of each example can be updated one stump at a time, so all \(T_{\max}\) validation risks cost \(O(T_{\max} m_v)\) stump evaluations in total.

**KFoldSelect** trains \(k\) times per parameter value on \(m(k-1)/k\) examples, evaluates on \(m/k\) each time, and retrains once on all of \(S\): time \(O(|\Theta| k (\mathrm{Train}(m) + (m/k) c) + \mathrm{Train}(m))\), so \(k\) times the cost of hold-out (three parameter values and five folds mean fifteen training runs plus one), and leave-one-out costs \(m\) training runs per parameter value. Space is \(O(m)\) for the fold assignment, plus one hypothesis at a time.

**LearningCurve** with prefixes of 10%, 20%, …, 100% trains on \(0.1m + 0.2m + \dots + m = 5.5m\) examples in total, so about \(5.5\,\mathrm{Train}(m)\) for a learner linear in its input, and evaluates on each prefix and on \(V\).

**Build it yourself.** Using only the Python standard library, on the plain mail room:

```python
import random

def make_data(m, seed):
    rng = random.Random(seed)
    S = []
    for _ in range(m):
        x = tuple(rng.randint(0, 1) for _ in range(20))
        S.append((x, x[0] ^ (rng.random() < 0.1)))
    return S

S, V = make_data(200, 7), make_data(1000, 8)
THETAS = ("constant", "one_feature", "memorize")

def risk(h, T): ...
def A(S, theta): ...
def hold_out_select(S, V, thetas, A): ...
def k_fold_select(S, thetas, A, k): ...
def learning_curve(S, V, A, theta, fractions): ...

if __name__ == "__main__":
    best, h, err = hold_out_select(S, V, THETAS, A)
    assert best == "one_feature"
    errs = [round(err[t], 3) for t in THETAS]
    assert errs == [0.516, 0.114, 0.484]
    best, h, err = k_fold_select(S, THETAS, A, 5)
    assert best == "one_feature"
    errs = [round(err[t], 3) for t in THETAS]
    assert errs == [0.465, 0.13, 0.535]
    curve = learning_curve(S, V, A, "one_feature",
                           (0.1, 0.25, 0.5, 1.0))
    pairs = [(round(a, 3), round(b, 3)) for a, b in curve]
    assert pairs == [(0.15, 0.114), (0.16, 0.114),
                     (0.16, 0.114), (0.13, 0.114)]
    curve = learning_curve(S, V, A, "memorize", (0.1, 1.0))
    assert [round(a, 3) for a, _ in curve] == [0.0, 0.0]
```

1. `A(S, theta)` returns a hypothesis: for `"constant"`, the majority label of `S` (a tie goes to 0); for `"one_feature"`, \(h_j(x) = x_j\) with the smallest empirical risk, the smallest \(j\) among ties; for `"memorize"`, a lookup table built with `dict(S)`, answering 0 for unseen messages.
2. `hold_out_select` returns the winning parameter, its hypothesis, and a dictionary of validation risks; ties go to the earlier entry of `thetas`. `k_fold_select` returns the same three things, with the k-fold errors, using contiguous folds `S[i*n:(i+1)*n]` with `n = len(S) // k`, and retrains the winner on all of `S`.
3. `learning_curve` returns, for each fraction `f`, the pair (empirical risk on the prefix of `round(f * len(S))` examples, validation risk).
4. Run `k_fold_select` with `k = len(S)`, which is leave-one-out. Which parameter wins, and how does its error compare with five folds?
5. Extend `learning_curve` to the boosted filter of lesson 10 and reproduce the shape of the model-selection curve above.

## What people get wrong

**“The test set is clean, because we never trained on it.”** Choosing is a kind of training. Once a set has been used to pick among candidates, the winner’s score on it is biased low, and the single-hypothesis guarantee no longer applies. Either report with the margin for every candidate you compared, or report from a set nobody looked at.

**“Trying more candidates can only help.”** Each extra candidate costs a little margin, \(\log(2r/\delta)\) grows with \(r\), and a large enough search overfits the validation set itself. More candidates help only while the validation set is big enough to tell them apart.

**“The cross-validation error is the true risk of the hypothesis I ship.”** Each fold’s error is an honest estimate for a hypothesis trained on fewer examples, not for the final one. Usually the two are close; in general there is no guarantee, and the first problem below shows the estimate can be off by \(1/2\).

**“The empirical risk is small, so the class is rich enough.”** A class that can label any small training sequence perfectly gets empirical risk 0 whatever its approximation error. Only the learning curves, or more data, tell the two apart.

## Recap and the next bug

1. **Model selection** chooses among learners, or settings of one learner, from the data. Choosing by the smallest empirical risk always prefers the richest candidate, so something else must decide.
2. **Structural risk minimization** chooses by minimizing a bound, empirical risk plus a complexity term that grows with the class. It is sound, but often far too pessimistic.
3. A **validation set** drawn independently of a hypothesis estimates its true risk within \(\sqrt{\log(2/\delta)/(2m_v)}\), whatever the learner. The independence is the whole point.
4. Choosing among \(r\) candidates on a validation set costs a union bound: every candidate is within \(\sqrt{\log(2r/\delta)/(2m_v)}\), and the winner is within twice that of the best candidate.
5. **k-fold cross validation** reuses every example for both training and validation; it usually works, but carries no general guarantee. A final **test set**, touched once, gives the reported number its guarantee.
6. When learning fails, split the error into pieces you can measure. A large empirical risk proves a large approximation error (for ERM); a small one is ambiguous, and **learning curves** resolve it.

Your filter ships with an honest number this time, and the diagnosis tools are ready for the next disappointment. It comes soon. Your colleague in charge of the halfspace filter from lesson 9 brings you a learning curve whose empirical risk will not come down, even though the validation curve says the class should be good enough. You remember the escape hatch: maybe the learner is not really doing ERM. It isn’t. When the mail is not linearly separable, finding the halfspace with the fewest training mistakes is computationally hard, so the code runs a heuristic and hopes. Is there a way out? What if, instead of counting mistakes, the learner minimized a different loss, one shaped like a bowl, so that the bottom can be found efficiently and checked when found? Could a convex loss make learning both tractable and sound? That is the next lesson: convex learning problems.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A teammate compares 1,000 candidate filters on a validation set of 500 messages and reports the winner’s validation risk. With \(\delta = 0.05\), what margin can she honestly attach to it? How large would the validation set have to be for a margin of 0.03? And how large if she had only one candidate, fixed before seeing any data?
2. Each term \(L_{S_i}(h_{i,\theta})\) of the k-fold estimate is a hold-out estimate. Of which hypothesis, exactly? Using the steps of the hold-out proof, explain why each term on its own has a guarantee, and why the average does not give the hold-out guarantee for the hypothesis \(A(S; \theta^*)\) that you ship.
3. Two filters, each trained by ERM, each with learning curves at 200, 400, 800 and 1,600 examples. Filter X: empirical risk 0.00 at every size, validation risk 0.49, 0.50, 0.48, 0.50. Filter Y: empirical risk 0.31, 0.33, 0.34, 0.34, validation risk 0.37, 0.36, 0.35, 0.35. For each, what is the likely problem and what would you try first? For Y, what lower bound on the approximation error can you prove with 95% confidence from the last point?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (when leave-one-out fails).** Labels are pure coin flips: \(\mathbb P[y = 1] = \mathbb P[y = 0] = 1/2\), whatever the message. A learner looks only at the labels of its training sequence: if their sum is odd it returns the constant hypothesis \(h(x) = 1\), and if it is even, \(h(x) = 0\). Prove that the difference between the leave-one-out estimate and the true risk of the learner’s output is always exactly \(1/2\).
- **Problem 2 (one big class or many small ones?).** You have \(k\) hypothesis classes \(\mathcal H_1, \dots, \mathcal H_k\) and \(m\) i.i.d. examples, and you want to learn \(\mathcal H = \bigcup_{i=1}^k \mathcal H_i\). You can run ERM over \(\mathcal H\) on all \(m\) examples. Or you can split the examples into a training sequence of \((1 - \alpha)m\) and a validation set of \(\alpha m\), for some \(\alpha \in (0, 1)\), run ERM over each \(\mathcal H_i\) on the training sequence to get \(\hat h_1, \dots, \hat h_k\), and then run ERM over \(\{\hat h_1, \dots, \hat h_k\}\) on the validation set. Describe situations in which the first approach is better than the second, and situations in which the second is better than the first.
