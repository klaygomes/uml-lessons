# PAC-Bayes

## The committee nobody can certify

The filter has changed a great deal since the first lesson, and this week it changed again. Every message now arrives with a **junk score**, a number between 0 and 1 computed upstream by the scorer you built over the last few lessons: the higher the score, the more the message looks like junk. The last step of the filter is the simplest one in the whole system. It picks a number \(t\), the **threshold**, and says junk exactly when the score is at least \(t\). The team writes thresholds with nine decimal places, from 0.000000000 to 0.999999999, so there are a billion possible thresholds to choose from.

[[diagram:dial]]

Two things about this week are new, and both of them come from good instincts.

The first is trust. Long before this week’s labels arrived, the team wrote down which thresholds it believed in. Last year’s threshold was 0.60, tuned on mail from the senders everyone already knew to be bad, and in all the years of records the right threshold has never drifted more than 0.05 in a year. So the team put 99% of its trust on the band from 0.55 to 0.65, spread evenly, and the last 1% evenly over everything else. They also like short thresholds, the round ones you can write on a whiteboard, such as 0.60 rather than 0.613527118. Trust of this kind is exactly the inductive bias of lesson 1, written down as numbers before any data arrives.

The second is randomness. Remember the randomized filter of lesson 21, which answered by a coin flip so that a spammer could not predict it? The team has adopted the same habit. After looking at this week’s 650 labelled messages, they noticed that the threshold making the fewest mistakes on them sits near 0.62. Instead of shipping one threshold, they now ship a whole window of them: for every incoming message, the filter draws a threshold uniformly at random from the 40 million thresholds between 0.60 and 0.64, and uses that one. A spammer who probes the filter learns nothing sharp about where the line is. Each individual answer still comes from one readable rule, so any decision can be explained. The team calls the window **the committee**.

Then the reviewer from lesson 7 arrives with the same deal as before. The filter ships only with a certificate: a number that, except with probability \(\delta = 0.05\), bounds the true risk from above, the chance that the filter mislabels the next message. The certificate must be at most 0.13, the trust must have been written down before the labels were drawn (it was), and the label budget is the 650 messages already labelled.

So you try every certificate the course has built. On this week’s labels the committee errs on 2.94% of the examples, and the single best threshold on 1.69%.

- Lesson 4’s certificate for a finite class adds the same margin to every hypothesis in it, \(\sqrt{\log(2|\mathcal H|/\delta)/(2m)}\). Here \(|\mathcal H|\) is the number of hypotheses in the class, a billion; \(m = 650\) is the number of labelled examples; \(\log\) is the natural logarithm. Because the margin is the same for every threshold, it also covers any average of thresholds, the committee included. But it charges the full size of the class and ignores every bit of the team’s trust.
- Lesson 7’s certificate uses trust. A hypothesis \(h\) with trust \(w(h)\) gets the margin \(\sqrt{(\log(1/w(h)) + \log(2/\delta))/(2m)}\), which is small when the trust is large. But it certifies one hypothesis at a time, and here the trust is spread so thinly that each single threshold in the band gets only \(0.99/10^8 + 0.01/10^9 \approx 9.9 \times 10^{-9}\), less than one part in a hundred million.

```python
from math import log, sqrt
m, delta, N = 650, 0.05, 10**9
uc = sqrt(log(2 * N / delta) / (2 * m))
w = 0.99 / 10**8 + 0.01 / N
occam = sqrt((log(1 / w) + log(2 / delta)) / (2 * m))
print(round(uc, 3), round(0.0294 + uc, 3))
print(round(log(1 / w), 1), round(occam, 3))
print(round(0.0169 + occam, 3), round(0.0294 + occam, 3))
```

It prints `0.137 0.166`, then `18.4 0.13`, then `0.147 0.16`. Lesson 4’s margin is 0.137, so the committee gets a certificate of 0.166. A single threshold’s trust costs \(\log(1/w) = 18.4\) in lesson 7’s formula, its margin is 0.130, and the best single threshold gets 0.147; averaging lesson 7’s certificates over the committee gives 0.160. Every number is above 0.13. The reviewer does not sign.

Look at what went wrong. Lesson 4’s certificate can average over a mix, but it cannot use trust. Lesson 7’s can use trust, but only one hypothesis at a time, and the trust is spread so thinly over the band that no single threshold carries much of it. Yet the team’s trust is anything but thin: 99% of it sits in a band that holds the committee’s whole window. What the team needs is a certificate for a mix chosen after seeing the data, one that charges for the trust the mix actually uses, rather than for the trust of any one member.

How does trust written down before the data turn into a guarantee for a mix chosen after it? That is the question of this lesson, the last of the course, and its answer is called the **PAC-Bayes** bound.

## A small rulebook

A billion thresholds is too many to reason about by hand, so let’s keep only the five round ones: 0.50, 0.55, 0.60, 0.65 and 0.70. For each threshold \(t\), the hypothesis \(h_t\) says junk exactly when the score is at least \(t\). In symbols, writing \(s\) for a message’s score and \(\mathbb 1[\cdot]\) for the indicator (1 when the statement inside is true, 0 when it is false),

\[ h_t(s) = \mathbb 1[s \ge t]. \]

In plain words: **\(h_t\) says junk (1) when the score reaches the threshold, and keep (0) otherwise.** A message scored 0.63 is junk for \(h_{0.60}\) and keep for \(h_{0.65}\).

We need a world in which we know the truth, so that we can check every claim. Here is ours. Scores spread evenly over \([0, 1]\), the interval of numbers from 0 to 1: every score is equally likely. Users follow a true cut at 0.62, so a message is junk exactly when its score is at least 0.62, except that 2% of the labels are flipped, because users misclick or disagree. The learner knows none of this.

What is the true risk of \(h_t\)? Recall from lesson 1 that the **true risk** \(L_{\mathcal D}(h)\) is the chance that \(h\) mislabels a fresh message drawn from the distribution \(\mathcal D\) of messages. A message whose score lies between \(t\) and 0.62 gets the opposite of the true cut’s answer from \(h_t\), so \(h_t\) is wrong on it unless the label was flipped: chance 0.98. Everywhere else \(h_t\) agrees with the true cut, so it is wrong only on a flipped label: chance 0.02. The stretch between \(t\) and 0.62 has length \(|t - 0.62|\), the distance between the two numbers, so

\[ L_{\mathcal D}(h_t) = 0.02\,(1 - |t - 0.62|) + 0.98\,|t - 0.62| = 0.02 + 0.96\,|t - 0.62|. \]

In plain words: **each threshold pays the 2% of flipped labels everywhere, plus 96 points of error for every unit of distance to the true cut.** For \(h_{0.60}\) the distance is 0.02, so the true risk is \(0.02 + 0.96 \times 0.02 = 0.0392\). For \(h_{0.65}\) it is \(0.02 + 0.96 \times 0.03 = 0.0488\).

The team’s trust over the five thresholds peaks at 0.60, last year’s threshold, and falls off on both sides. This week’s 650 labels give each threshold an **empirical risk** \(L_S(h)\), the fraction of the labelled examples that \(h\) gets wrong (lesson 1). Here is the whole rulebook.

| threshold \(t\) | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|
| true risk \(L_{\mathcal D}(h_t)\) | 0.1352 | 0.0872 | 0.0392 | 0.0488 | 0.0968 |
| trust before the labels | 0.1 | 0.2 | 0.4 | 0.2 | 0.1 |
| empirical risk \(L_S(h_t)\), this week | 0.1446 | 0.0846 | 0.0415 | 0.0431 | 0.0985 |

Notice two things before we go on. The empirical risks track the true risks, but not exactly: on these labels 0.60 and 0.65 are almost tied. And the trust adds up to exactly 1.

## A filter that rolls a die

Suppose the team builds a small committee here too: 60% of the time it uses \(h_{0.60}\), and 40% of the time \(h_{0.65}\). Each message gets its own fresh draw. How often is this filter wrong?

Take one message. With probability 0.6 it is judged by \(h_{0.60}\), which is wrong on a random message with probability 0.0392; with probability 0.4 it is judged by \(h_{0.65}\), wrong with probability 0.0488. Adding the two cases, the chance of a mistake is \(0.6 \times 0.0392 + 0.4 \times 0.0488 = 0.0430\). The randomized filter’s error is the weighted average of its members’ errors, with the committee’s weights.

Now the names. A **posterior** is a probability distribution \(Q\) over the hypothesis class \(\mathcal H\): a number \(Q(h) \ge 0\) for each hypothesis, adding up to 1. Here \(Q(h_{0.60}) = 0.6\), \(Q(h_{0.65}) = 0.4\), and \(Q\) is 0 on the other three. The word comes from Bayesian statistics, but nothing Bayesian is required: a posterior is any distribution over \(\mathcal H\) you like, chosen however you like, after looking at the labels or not. The **Gibbs predictor** of \(Q\) is the randomized filter that, for each new message, draws a hypothesis \(h\) at random with probabilities \(Q\) and answers \(h(x)\).

To score a Gibbs predictor, we use the **loss** of lesson 3: \(\ell(h, z)\) is a number in \([0, 1]\) that says how badly \(h\) did on the example \(z = (x, y)\). For the filter it is the 0-1 loss, 1 for a mistake and 0 otherwise. The loss of \(Q\) on \(z\) is the average loss of the hypothesis it draws:

\[ \ell(Q, z) = \mathbb E_{h \sim Q}\big[\ell(h, z)\big] = \sum_{h \in \mathcal H} Q(h)\, \ell(h, z). \]

Read it from the inside out.

- \(\ell(h, z)\) is the loss of one fixed hypothesis \(h\) on the example \(z\): 1 if \(h\) gets it wrong, 0 if right.
- \(h \sim Q\) reads “\(h\) is drawn from \(Q\)”: the filter picks \(h\) at random, each hypothesis with probability \(Q(h)\).
- \(\mathbb E_{h \sim Q}[\cdot]\) is the expectation over that draw: the average of the bracket, each value weighted by the chance of drawing its \(h\). For a class we can list, it is the sum on the right.
- \(\sum_{h \in \mathcal H}\) adds over every hypothesis \(h\) in the class.

In plain words: **the loss of the randomized filter on one example is the chance that the hypothesis it draws gets that example wrong.** For a message scored 0.63 and labelled keep (one of the flipped ones), \(h_{0.60}\) says junk, wrong, and \(h_{0.65}\) says keep, right, so \(\ell(Q, z) = 0.6 \times 1 + 0.4 \times 0 = 0.6\).

Average that over a fresh example drawn from \(\mathcal D\) and you get the true risk of \(Q\); average it over the training sequence \(S = (z_1, \dots, z_m)\) and you get its empirical risk. Because an average of averages can be taken in either order (the two sums can be swapped, since all the terms are nonnegative and \(Q\) does not depend on the fresh example), both collapse to the same simple shape:

\[ L_{\mathcal D}(Q) = \mathbb E_{h \sim Q}\big[L_{\mathcal D}(h)\big], \qquad L_S(Q) = \mathbb E_{h \sim Q}\big[L_S(h)\big]. \]

- \(L_{\mathcal D}(h)\) is the true risk of one hypothesis, and \(L_S(h)\) its empirical risk, the fraction of the \(m\) examples it gets wrong.
- The outer \(\mathbb E_{h \sim Q}\) weights each hypothesis by its share of the posterior.

In plain words: **the risks of a Gibbs predictor are the posterior-weighted averages of its members’ risks.** On this week’s labels, \(L_S(Q) = 0.6 \times 0.0415 + 0.4 \times 0.0431 = 0.0422\), and the true risk is the 0.0430 we found above. Python can check both: the first snippet runs the randomized filter on 200,000 fresh messages and compares its mistake rate with the formula.

```python
import random
rng = random.Random(1)
rules = [0.50, 0.55, 0.60, 0.65, 0.70]
Q = [0, 0, 0.6, 0.4, 0]
L_D = [0.02 + 0.96 * abs(t - 0.62) for t in rules]
mistakes, n = 0, 200_000
for _ in range(n):
    s = rng.random()
    y = int(s >= 0.62) ^ (rng.random() < 0.02)
    t = rng.choices(rules, Q)[0]
    mistakes += int(s >= t) != y
L_D_Q = sum(q * l for q, l in zip(Q, L_D))
print(round(mistakes / n, 4), round(L_D_Q, 4))
```

It prints `0.0426 0.043`: the filter that rolls a die per message makes mistakes at the rate the weighted average predicts, up to simulation noise. (The `^` flips the true cut’s label when the 2% coin comes up.) The second snippet draws this week’s 650 labels and computes the empirical risks of the table and of the committee.

```python
import random
rng = random.Random(2026)
S = []
for _ in range(650):
    s = rng.random()
    y = int(s >= 0.62) ^ (rng.random() < 0.02)
    S.append((s, y))
rules = [0.50, 0.55, 0.60, 0.65, 0.70]
Q = [0, 0, 0.6, 0.4, 0]
L_S = [sum((s >= t) != y for s, y in S) / len(S)
       for t in rules]
L_S_Q = sum(q * l for q, l in zip(Q, L_S))
print([round(l, 4) for l in L_S], round(L_S_Q, 4))
```

It prints `[0.1446, 0.0846, 0.0415, 0.0431, 0.0985] 0.0422`, the last row of the table and the committee’s empirical risk.

[[diagram:gibbs]]

One consequence is worth saying out loud now, because it surprises people later. A Gibbs predictor is never better than its best member: an average of numbers is at least their minimum. The committee’s true risk, 0.0430, is above that of \(h_{0.60}\), 0.0392. Mixing does not lower the error. What it can lower, as we are about to see, is the price of the certificate.

## Trust before, belief after

The team’s trust is also a distribution over the class, and it gets its own name. A **prior** is a probability distribution \(P\) over \(\mathcal H\), a number \(P(h) \ge 0\) for each hypothesis with \(\sum_{h} P(h) = 1\), that is **fixed before the training sequence is drawn**. Its value \(P(h)\) is the **prior score** of \(h\). In the small rulebook, \(P(h_{0.60}) = 0.4\) and \(P(h_{0.50}) = 0.1\).

So the two distributions differ only in when they are chosen. The prior is written down before any labels exist, like lesson 7’s weights; the posterior is chosen afterwards, and may depend on the labels in any way at all. Lesson 7’s weight functions are almost priors already: they had to sum to at most 1, and a prior sums to exactly 1, so you can turn weights into a prior by dividing each one by their total, which only increases each weight.

[[diagram:prior-posterior]]

Why must the prior come first? You have seen the answer twice. In lesson 1, a hypothesis class chosen after looking at the labels can always be made to contain a hypothesis that fits them, coincidences and all. In lesson 7, weights tuned to the labels void the union bound. The prior is the same commitment in its most flexible form: instead of a class or a list of weights, a whole distribution of belief, and it is only worth something if it was made before the evidence.

The posterior has no such restriction, and that is the point of this lesson. The theorem we are about to prove holds for every posterior at once, including one picked by staring at the labels, as the committee was. The price is that the certificate charges for how far the posterior moved away from the prior. So we need a way to measure that distance.

## How far did the committee move?

The committee puts 0.6 on \(h_{0.60}\), where the prior put 0.4: it believes in that threshold 1.5 times more than the team did before the labels. On \(h_{0.65}\) it puts 0.4 against a prior 0.2, twice as much. On the other three it puts nothing, which costs nothing: moving belief away from a hypothesis is free. Moving belief towards one is what costs.

How should those ratios be combined? Ratios multiply, so we take logarithms, which turn products into sums: \(\log 1.5 = 0.405\) and \(\log 2 = 0.693\). Then we average them, weighting each by how often the committee actually uses that hypothesis: \(0.6 \times 0.405 + 0.4 \times 0.693 = 0.521\). That number is the relative entropy.

**Definition (relative entropy).** For distributions \(Q\) and \(P\) over a countable class \(\mathcal H\) (finite, or listable as \(h_1, h_2, \dots\)), the **relative entropy** of \(Q\) to \(P\) is

\[ D(Q \| P) = \mathbb E_{h \sim Q}\Big[\log \frac{Q(h)}{P(h)}\Big] = \sum_{h \,:\, Q(h) > 0} Q(h) \log \frac{Q(h)}{P(h)}, \]

and \(D(Q \| P) = +\infty\) if some \(h\) has \(Q(h) > 0\) but \(P(h) = 0\).

Read it from the inside out.

- \(Q(h)/P(h)\) is the factor by which the posterior’s belief in \(h\) exceeds the prior’s: 1.5 for \(h_{0.60}\) in the committee.
- \(\log\) is the natural logarithm, so a factor above 1 gives a positive number, a factor below 1 a negative one, and a factor of exactly 1 gives 0.
- \(\mathbb E_{h \sim Q}[\cdot]\) averages over a hypothesis drawn from the posterior. The sum on the right writes that average out: the colon in \(\sum_{h \,:\, Q(h) > 0}\) reads “such that”, so we add over the hypotheses \(h\) such that \(Q(h) > 0\). A hypothesis the posterior never draws contributes nothing.
- The double bar \(\|\) just separates the two arguments, and their order matters: the average is taken under the first one, \(Q\).
- The infinite case: if the posterior uses a hypothesis that the prior ruled out completely, the ratio is a division by zero and no finite price can pay for it.

In plain words: **the relative entropy is the posterior’s average log-factor of belief beyond the prior, averaged over how the posterior actually draws.** It was introduced in lesson 24, written \(D_{\mathrm{RE}}\) there, as the extra surprise of believing one distribution when the truth is another; it is also called the Kullback–Leibler divergence. For the committee, \(D(Q \| P) = 0.6 \log 1.5 + 0.4 \log 2 = 0.5205\).

Since \(D(Q\|P)\) is an expectation, you can also estimate it the way you would estimate any average: draw many hypotheses from \(Q\), compute \(\log(Q(h)/P(h))\) for each, and average. The snippet below computes it both ways.

```python
import random
from math import log
P = [0.1, 0.2, 0.4, 0.2, 0.1]
Q = [0, 0, 0.6, 0.4, 0]
D = sum(q * log(q / p) for q, p in zip(Q, P) if q > 0)
rng, n = random.Random(2), 100_000
draws = rng.choices(range(5), Q, k=n)
estimate = sum(log(Q[i] / P[i]) for i in draws) / n
print(round(D, 4), round(estimate, 4))
```

It prints `0.5205 0.5209`: the exact sum, and the average over 100,000 simulated draws from the committee.

Three facts make the relative entropy usable as a price.

**It is never negative, and it is 0 exactly when \(Q = P\).** Lesson 24 proved this for finite sets, and the proof carries over word for word to countable ones. Its engine is the inequality \(\log u \le u - 1\) for every \(u > 0\), with equality only at \(u = 1\): apply it to \(u = P(h)/Q(h)\) inside the sum. So the price is zero when you do not update your belief at all.

**A posterior that keeps the prior’s proportions on a set pays the log of one over the prior’s mass on that set.** Here is the precise version.

**Lemma (the restricted prior).** Let \(A \subseteq \mathcal H\) be a set of hypotheses with \(P(A) = \sum_{h \in A} P(h) > 0\), and let \(Q_A\) be the prior restricted to \(A\): \(Q_A(h) = P(h)/P(A)\) for \(h \in A\), and \(Q_A(h) = 0\) otherwise. Then

\[ D(Q_A \| P) = \log \frac{1}{P(A)}. \]

Here \(A \subseteq \mathcal H\) reads “\(A\) is a subset of \(\mathcal H\)”, and \(P(A)\) is the prior’s total trust on \(A\). The posterior \(Q_A\) throws away every hypothesis outside \(A\) and rescales the rest so they again add up to 1, keeping their proportions. In plain words: **if you keep only a set of hypotheses and believe in them in the prior’s proportions, you pay the log of one over the trust that set had.** Keep \(\{h_{0.60}, h_{0.65}\}\), which had trust \(0.4 + 0.2 = 0.6\): the posterior is \((2/3, 1/3)\) on the two, and the price is \(\log(1/0.6) = 0.5108\).

*Proof.* For every \(h \in A\), the ratio is the same number, \(Q_A(h)/P(h) = 1/P(A)\). So every term of the sum has the same logarithm, \(\log(1/P(A))\), and the weights \(Q_A(h)\) over \(A\) add up to 1. \(\blacksquare\)

```python
from math import log
P = [0.1, 0.2, 0.4, 0.2, 0.1]
A = [2, 3]
P_A = sum(P[i] for i in A)
Q = [P[i] / P_A if i in A else 0 for i in range(5)]
D = sum(q * log(q / p) for q, p in zip(Q, P) if q > 0)
print([round(q, 4) for q in Q])
print(round(D, 4), round(log(1 / P_A), 4))
print(round(log(1 / P[3]), 4))
```

It prints `[0, 0, 0.6667, 0.3333, 0]`, then `0.5108 0.5108`, then `1.6094`.

The last line is the third fact, and the most important special case. **A posterior that puts all its belief on one hypothesis pays \(\log(1/P(h))\).** That is the lemma with \(A = \{h\}\). For \(h_{0.65}\), with prior 0.2, the price is \(\log 5 = 1.6094\). This is exactly the cost \(\log(1/w(h))\) that lesson 7 charged a single hypothesis. The relative entropy extends that cost from one hypothesis to any mix of them.

Now go back to the committee on the dial. The team’s prior is flat inside the band [0.55, 0.65], so restricted to the window [0.60, 0.64] it is uniform on the window: that is exactly the committee, “every threshold in the window equally likely”. The lemma applies. The window’s share of the prior is \(0.99 \times 0.04/0.1\) from the band, plus \(0.01 \times 0.04\) from the thin layer of trust spread over everything, which is \(0.3964\). So the committee pays \(\log(1/0.3964) = 0.9253\). A single threshold in the band pays \(\log(1/(9.9 \times 10^{-9})) = 18.4\), the number that sank lesson 7’s certificate. Spreading the belief over the window is twenty times cheaper than pinning it to one threshold.

[[diagram:relative-entropy]]

## Try it: the window and its price

In the lab below, the team’s prior is drawn as a profile over the dial, and the posterior is a window you drag and stretch. Before you touch anything, predict: as you shrink the window from the whole band down to a single threshold around 0.62, what happens to the committee’s empirical risk, and what happens to its relative entropy?

[[lab]]

On this week’s labels, the whole band has empirical risk 0.0471 and relative entropy 0.009. The window [0.60, 0.64] has 0.0294 and 0.925; the window [0.615, 0.625] has 0.0201 and 2.31; a window 0.001 wide has 0.0169 and 4.61; a single threshold has 0.0169 and 18.4. One number falls as the posterior concentrates, the other rises. The certificate will need both.

## The PAC-Bayes theorem

Here is the guarantee. It reads like lesson 4’s and lesson 7’s, with the cost of the posterior in place of the cost of the class.

**Theorem (PAC-Bayes).** Let \(\mathcal D\) be any distribution over examples, \(\mathcal H\) a countable hypothesis class, \(\ell: \mathcal H \times Z \to [0, 1]\) a loss, \(P\) a prior over \(\mathcal H\), \(\delta \in (0, 1)\) and \(m \ge 2\). Then with probability at least \(1 - \delta\) over an i.i.d. training sequence \(S \sim \mathcal D^m\), every posterior \(Q\) over \(\mathcal H\), including any that depends on \(S\), satisfies

\[ L_{\mathcal D}(Q) \le L_S(Q) + \sqrt{\frac{D(Q \| P) + \log(m/\delta)}{2(m - 1)}}. \]

Read it from the inside out.

- \(\ell: \mathcal H \times Z \to [0, 1]\) says the loss takes a hypothesis and an example (from the set \(Z\) of all examples) and returns a number between 0 and 1; the 0-1 loss qualifies.
- \(S \sim \mathcal D^m\) reads “\(S\) is drawn from \(\mathcal D^m\)”: \(m\) examples, each drawn from \(\mathcal D\), independently of each other (i.i.d., lesson 1). The probability \(1 - \delta\) is over that draw.
- \(\delta \in (0, 1)\) says \(\delta\) is strictly between 0 and 1: the chance of failure you accept, 0.05 for the reviewer.
- \(D(Q\|P)\) is the price of the posterior; \(\log(m/\delta)\) is a price for the confidence, a little larger than lesson 4’s \(\log(2/\delta)\) because it grows with \(m\).
- Dividing by \(2(m - 1)\) and taking the square root turns the total price into a margin that shrinks like one over the square root of \(m\).
- “Every posterior \(Q\)” is the key phrase. The bad event, the unlucky training sequences, is fixed before any posterior is chosen, and outside it the inequality holds for all of them simultaneously. So you may look at \(S\), pick any \(Q\), and the certificate still holds.

In plain words: **except on a fraction \(\delta\) of unlucky training sequences, every randomized filter’s true risk is at most its empirical risk plus a margin that grows with how far its posterior moved from the prior.** The small committee: \(L_S(Q) = 0.0422\), \(D(Q\|P) = 0.5205\), \(\log(650/0.05) = \log 13{,}000 = 9.4727\), so the margin is \(\sqrt{(0.5205 + 9.4727)/1298} = 0.0877\) and the certificate is 0.1299.

```python
from math import log, sqrt
m, delta = 650, 0.05
L_S_Q, D = 0.0422, 0.5205
gap = sqrt((D + log(m / delta)) / (2 * (m - 1)))
print(round(log(m / delta), 4), round(gap, 4))
print(round(L_S_Q + gap, 4))
```

It prints `9.4727 0.0877`, then `0.1299`.

Now the real committee, on the dial. Its empirical risk is the average of \(L_S(h_t)\) over the thresholds in the window, which the snippet approximates on a grid of 4,001 thresholds; its relative entropy is the 0.9253 from the restricted-prior lemma.

```python
import random
from math import log, sqrt
rng, S = random.Random(2026), []
for _ in range(650):
    s = rng.random()
    S.append((s, int(s >= 0.62) ^ (rng.random() < 0.02)))
def L_S(t):
    return sum((s >= t) != y for s, y in S) / len(S)
window = [0.60 + k / 100_000 for k in range(4001)]
L_S_Q = sum(map(L_S, window)) / len(window)
P_window = 0.99 * 0.04 / 0.1 + 0.01 * 0.04
D = log(1 / P_window)
gap = sqrt((D + log(650 / 0.05)) / (2 * 649))
print(round(L_S_Q, 4), round(D, 4), round(L_S_Q + gap, 4))
```

It prints `0.0294 0.9253 0.1189`. The committee’s certificate is 0.119, under the reviewer’s 0.13. It is the first certificate in the course that both uses the team’s trust and covers a mix chosen after the labels. (Its true risk, which only we can see, is \(0.02 + 0.96 \times 0.01 = 0.0296\), since the average distance from a threshold in the window to 0.62 is 0.01.)

The other choices tell the story of the trade. Here they are, with the same function:

```python
from math import log, sqrt
def certificate(L, D, m=650, delta=0.05):
    return L + sqrt((D + log(m / delta)) / (2 * (m - 1)))
w = 0.99 / 10**8 + 0.01 / 10**9
print(round(certificate(0.0294, log(1 / 0.3964)), 4))
print(round(certificate(0.0294, log(1 / 0.04)), 4))
print(round(certificate(0.0169, log(1 / w)), 4))
print(round(certificate(0.0494, 0), 4))
```

It prints `0.1189`, `0.1283`, `0.1635` and `0.1348`. The second line is the same window under a flat prior, which trusts every threshold equally, so the window’s share of trust is just its width, 0.04, and its price \(\log 25 = 3.22\). The third is the single best threshold under the team’s prior: the lowest empirical risk, the highest price. The fourth is the prior itself used as the posterior, with an empirical risk of 0.0494: no price, but the band is too wide to be accurate. The best certificates sit in the middle, where the posterior has concentrated enough to be accurate but not so much that it pays for belief it cannot afford.

[[diagram:tradeoff]]

## Four tools for the proof

The proof uses four facts about averages. Two of them you met in lesson 4; the other two are new.

**Markov’s inequality** (lesson 4). For a random variable \(Z \ge 0\) and any \(a > 0\),

\[ \mathbb P[Z \ge a] \le \frac{\mathbb E[Z]}{a}. \]

\(\mathbb P[\cdot]\) is the probability of the event in brackets and \(\mathbb E[Z]\) the average of \(Z\). In plain words: **a nonnegative quantity is rarely many times its own average.** If \(Z\) averages 2, it reaches 40 at most a twentieth of the time. (The proof: \(Z \ge a \cdot \mathbb 1[Z \ge a]\) for every outcome; take averages.)

**Jensen’s inequality, two forms.** Lesson 14 met Jensen’s inequality for convex functions of finite averages. We need two cases of it, for any random variable that averages over a countable set, and both have one-line proofs.

\[ \mathbb E[\log Y] \le \log \mathbb E[Y] \quad (Y > 0), \qquad \big(\mathbb E[X]\big)^2 \le \mathbb E[X^2]. \]

On the left, \(Y\) is a positive random variable: the average of its logarithms is at most the logarithm of its average. On the right, \(X\) is any random variable with a finite average square: the square of its average is at most the average of its squares. In plain words: **taking the log after averaging gives more than averaging the logs, and squaring after averaging gives less than averaging the squares.** With \(Y\) equal to 3 with probability 0.6 and 0.5 with probability 0.4, the average of the logs is \(0.6\log 3 + 0.4 \log 0.5 = 0.3819\), and the log of the average is \(\log 2 = 0.6931\).

*Proof.* For the logarithm, let \(c = \mathbb E[Y]\), assumed finite (if it is infinite there is nothing to prove). For every \(y > 0\), lesson 24’s inequality \(\log u \le u - 1\) with \(u = y/c\) gives \(\log y \le \log c + (y - c)/c\): the logarithm lies below its tangent line at \(c\). Put \(y = Y\) and average both sides: \(\mathbb E[\log Y] \le \log c + (\mathbb E[Y] - c)/c = \log c\). For the square, the average of a square is never negative, so \(0 \le \mathbb E[(X - \mathbb E[X])^2] = \mathbb E[X^2] - 2\mathbb E[X]\,\mathbb E[X] + (\mathbb E[X])^2 = \mathbb E[X^2] - (\mathbb E[X])^2\). \(\blacksquare\)

```python
from math import log
Q = [0.6, 0.4]
Y = [3.0, 0.5]
avg_log = sum(q * log(y) for q, y in zip(Q, Y))
log_avg = log(sum(q * y for q, y in zip(Q, Y)))
X = [0.03, 0.01]
sq_avg = sum(q * x for q, x in zip(Q, X)) ** 2
avg_sq = sum(q * x * x for q, x in zip(Q, X))
print(round(avg_log, 4), round(log_avg, 4))
print(round(sq_avg, 6), round(avg_sq, 6))
```

It prints `0.3819 0.6931`, then `0.000484 0.00058`: both inequalities, on a two-point distribution.

**Hoeffding’s inequality, one side** (lesson 4). Fix a hypothesis \(h\) before seeing any data. Its losses on the \(m\) examples, \(\ell(h, z_1), \dots, \ell(h, z_m)\), are i.i.d. numbers in \([0, 1]\) with average \(L_{\mathcal D}(h)\), and \(L_S(h)\) is their sample average. Lesson 4 proved each tail of Hoeffding’s inequality separately before adding them; the lower one says that for every \(\varepsilon > 0\),

\[ \mathbb P_{S \sim \mathcal D^m}\big[L_{\mathcal D}(h) - L_S(h) \ge \varepsilon\big] \le e^{-2m\varepsilon^2}. \]

Here \(\varepsilon\) (epsilon) is how far the empirical risk falls below the true risk, and the subscript on \(\mathbb P\) says the randomness is the draw of \(S\). In plain words: **the chance that a fixed hypothesis looks better on the examples than it really is, by at least \(\varepsilon\), shrinks exponentially in \(m \varepsilon^2\).** For \(h_{0.60}\) at \(m = 650\) and \(\varepsilon = 0.02\), the bound is \(e^{-0.52} = 0.595\); the exact binomial chance is much smaller.

```python
from math import comb, exp
m, p = 650, 0.0392
def tail(eps):
    return sum(comb(m, k) * p**k * (1 - p)**(m - k)
               for k in range(m + 1) if p - k / m >= eps)
for eps in (0.01, 0.02, 0.03):
    print(eps, f"{tail(eps):.2e}", f"{exp(-2 * m * eps**2):.2e}")
```

It prints `0.01 7.38e-02 8.78e-01`, `0.02 2.07e-03 5.95e-01` and `0.03 6.77e-07 3.10e-01`: the exact tail, from counting \(k\) mistakes out of 650, always below Hoeffding’s bound.

**The layer-cake formula.** For a random variable \(Y \ge 0\),

\[ \mathbb E[Y] = \int_0^\infty \mathbb P[Y > t]\, dt. \]

The integral \(\int_0^\infty \dots\, dt\) is the area under the curve \(t \mapsto \mathbb P[Y > t]\), from \(t = 0\) to infinity. In plain words: **the average of a nonnegative quantity is the area under its tail.** Think of a staircase: \(Y\) is 1, 2 or 5 with probabilities 0.5, 0.3 and 0.2. Its average is \(0.5 + 0.6 + 1 = 2.1\). Its tail is 1 for \(t < 1\), then 0.5 up to 2, then 0.2 up to 5, then 0, and the area is \(1 + 0.5 + 0.2 \times 3 = 2.1\). Same area, sliced the other way. (In general: for every outcome, \(Y = \int_0^\infty \mathbb 1[Y > t]\, dt\), since the indicator is 1 exactly for \(t\) between 0 and \(Y\); averaging both sides and swapping the average with the integral, which is allowed because the integrand is nonnegative, gives the formula.)

```python
values, probs = [1.0, 2.0, 5.0], [0.5, 0.3, 0.2]
mean = sum(v * p for v, p in zip(values, probs))
dt, area = 0.001, 0.0
for k in range(6000):
    t = (k + 0.5) * dt
    area += dt * sum(p for v, p in zip(values, probs) if v > t)
print(round(mean, 3), round(area, 3))
```

It prints `2.1 2.1`.

## The proof

Here is the plan. We cannot apply Hoeffding to the posterior directly, because the posterior may depend on \(S\), and Hoeffding needs a hypothesis fixed in advance. The prior is fixed in advance. So we prove a single fact about the prior, that one particular average over the prior is rarely large, and then transfer it to every posterior at once with an inequality that holds for every training sequence, no probability involved. The transfer is where the relative entropy appears.

[[diagram:proof-map]]

Throughout, write the **gap** of a hypothesis as \(\Delta(h) = L_{\mathcal D}(h) - L_S(h)\), how much better it looks on the examples than it really is, and its **positive part** as \(\Delta^+(h) = \max\{\Delta(h), 0\}\), the gap when it is positive and 0 otherwise. For \(h_{0.65}\) on this week’s labels, \(\Delta = 0.0488 - 0.0431 = 0.0057\), so \(\Delta^+ = 0.0057\); for \(h_{0.50}\), \(\Delta = 0.1352 - 0.1446 < 0\), so \(\Delta^+ = 0\). We only need the upper side, because the theorem only bounds the true risk from above.

**Step 1: the positive gap of one fixed hypothesis has a Hoeffding tail.** Fix \(h\). For \(\varepsilon > 0\), the event \(\Delta^+(h) \ge \varepsilon\) is the same as \(\Delta(h) \ge \varepsilon\), because a positive part reaches a positive level exactly when the gap itself does. So the one-sided Hoeffding inequality gives \(\mathbb P_S[\Delta^+(h) \ge \varepsilon] \le e^{-2m\varepsilon^2}\) for every \(\varepsilon > 0\).

**Step 2: a moment bound.** The next step turns that tail into a bound on an average.

**Lemma (from tails to an exponential moment).** Let \(X \ge 0\) be a random variable with \(\mathbb P[X \ge \varepsilon] \le e^{-2m\varepsilon^2}\) for every \(\varepsilon > 0\), where \(m \ge 2\). Then

\[ \mathbb E\big[e^{2(m-1)X^2}\big] \le m. \]

Read it from the inside out: \(X^2\) is the square of \(X\), multiplied by \(2(m - 1)\) and raised as a power of \(e\), so large values of \(X\) are blown up enormously; the claim is that the average of this blown-up quantity is still at most \(m\). In plain words: **a quantity whose tail falls like \(e^{-2m\varepsilon^2}\) can be squared, scaled by almost \(2m\) and exponentiated, and its average is still at most \(m\).** The exponent \(2(m-1)\) is just short of the \(2m\) in the tail, and that small slack is what keeps the average finite. For \(m = 650\), \(\Delta^+(h_{0.60})\) qualifies by step 1, and the lemma promises an average of at most 650.

*Proof.* Let \(Y = e^{2(m-1)X^2}\). Since \(X^2 \ge 0\), \(Y \ge 1\), so \(\mathbb P[Y > t] = 1\) for every \(t < 1\), and the layer-cake formula gives

\[ \mathbb E[Y] = 1 + \int_1^\infty \mathbb P[Y > t]\, dt. \]

The 1 is the area of the first unit of the strip, where the tail is 1. Now change the variable of integration. Every \(t \ge 1\) can be written as \(t = e^{2(m-1)\varepsilon^2}\) for exactly one \(\varepsilon \ge 0\), and as \(\varepsilon\) runs from 0 to \(\infty\), \(t\) runs from 1 to \(\infty\), increasing. The derivative of \(t\) with respect to \(\varepsilon\) is \(4(m-1)\varepsilon\, e^{2(m-1)\varepsilon^2}\) (chain rule), so \(dt = 4(m-1)\varepsilon\, e^{2(m-1)\varepsilon^2} d\varepsilon\).

Next, the event. Since \(X \ge 0\) and squaring and exponentiating are increasing on nonnegative numbers, \(Y > t\) holds exactly when \(X > \varepsilon\). And \(\mathbb P[X > \varepsilon] \le \mathbb P[X \ge \varepsilon] \le e^{-2m\varepsilon^2}\) for \(\varepsilon > 0\), by the hypothesis. (The single point \(\varepsilon = 0\) has no width and does not affect the integral.) Putting the two together,

\[ \mathbb E[Y] \le 1 + \int_0^\infty 4(m-1)\varepsilon\, e^{2(m-1)\varepsilon^2} e^{-2m\varepsilon^2}\, d\varepsilon = 1 + (m-1) \int_0^\infty 4\varepsilon\, e^{-2\varepsilon^2}\, d\varepsilon. \]

The exponents combine as \(2(m-1)\varepsilon^2 - 2m\varepsilon^2 = -2\varepsilon^2\): this is where the slack between \(2(m-1)\) and \(2m\) is spent. Finally, \(4\varepsilon e^{-2\varepsilon^2}\) is the derivative of \(-e^{-2\varepsilon^2}\), so its integral from 0 to \(\infty\) is \(0 - (-1) = 1\). Hence \(\mathbb E[Y] \le 1 + (m - 1) = m\). \(\blacksquare\)

Two snippets check the lemma. The first evaluates \(1 + \int_0^\infty 4(m-1)\varepsilon e^{-2\varepsilon^2} d\varepsilon\) numerically for \(m = 650\); the second computes \(\mathbb E[e^{2(m-1)\Delta^+(h)^2}]\) exactly, by summing over the possible numbers of mistakes, for hypotheses with true risk 0.0392 (like \(h_{0.60}\)), 0.2 and 0.5, at \(m = 50\) and \(m = 650\).

```python
from math import exp
m, de = 650, 0.0001
total = sum(4 * (m - 1) * e * exp(-2 * e * e) * de
            for e in ((k + 0.5) * de for k in range(60_000)))
print(round(1 + total, 3))
```

It prints `650.0`.

```python
from math import comb, exp
def moment(m, p):
    total = 0.0
    for k in range(m + 1):
        gap = max(p - k / m, 0)
        p_k = comb(m, k) * p**k * (1 - p)**(m - k)
        total += p_k * exp(2 * (m - 1) * gap**2)
    return total
for p in (0.0392, 0.2, 0.5):
    print(p, round(moment(50, p), 3), round(moment(650, p), 3))
```

It prints `0.0392 1.032 1.04`, `0.2 1.242 1.304` and `0.5 2.212 3.853`. Real hypotheses sit far below \(m\). The lemma has to hold for every variable with a Hoeffding tail, including the worst one imaginable, so it is generous; we will pay for that generosity with the \(\log m\) in the theorem.

**Step 3: average over the prior.** Define, for each training sequence \(S\), the number

\[ \Phi(S) = \mathbb E_{h \sim P}\Big[e^{2(m-1)\Delta^+(h)^2}\Big] = \sum_{h \in \mathcal H} P(h)\, e^{2(m-1)\Delta^+(h)^2}. \]

\(\Phi\) (capital phi) is the prior-weighted average, over all hypotheses, of the blown-up gap of step 2; it depends on \(S\) through the empirical risks inside \(\Delta^+\). In plain words: **\(\Phi(S)\) measures how much the training sequence flatters the hypotheses the prior believes in.** Its average over training sequences is

\[ \mathbb E_S[\Phi(S)] = \sum_{h \in \mathcal H} P(h)\, \mathbb E_S\Big[e^{2(m-1)\Delta^+(h)^2}\Big] \le \sum_{h \in \mathcal H} P(h)\, m = m. \]

The equality moves the average over \(S\) inside the sum. This is the one place in the proof where the prior must be fixed before the data: each \(P(h)\) is a constant, not a random quantity, so it comes out of the average over \(S\) untouched, and the average of a sum of nonnegative terms is the sum of their averages (for a countable sum too, since the terms are nonnegative). The inequality applies step 2 to each fixed \(h\), and the last step uses \(\sum_h P(h) = 1\). Had the weights depended on \(S\), they could not have been pulled out, and the bound would say nothing. In the small rulebook, every term is at most about 1.04 (the snippet above), so \(\mathbb E_S[\Phi(S)]\) is near 1, far below 650.

**Step 4: Markov.** \(\Phi(S)\) is nonnegative, so Markov’s inequality with \(a = m/\delta\) gives

\[ \mathbb P_S\Big[\Phi(S) \ge \frac{m}{\delta}\Big] \le \frac{\mathbb E_S[\Phi(S)]}{m/\delta} \le \frac{m}{m/\delta} = \delta. \]

So the **good event** \(G = \{S : \Phi(S) < m/\delta\}\), the set of training sequences on which \(\Phi\) stays below \(m/\delta\), has probability at least \(1 - \delta\). With \(m = 650\) and \(\delta = 0.05\), the level is 13,000. Notice that no posterior has appeared yet. The good event is fixed before anyone chooses \(Q\).

**Step 5: the change of measure.** Now the transfer from the prior to every posterior. It is a fact about distributions, with no randomness in it at all.

**Lemma (change of measure).** Let \(P\) and \(Q\) be distributions over a countable \(\mathcal H\), and \(g: \mathcal H \to \mathbb R\) a bounded function. Then

\[ \mathbb E_{h \sim Q}\big[g(h)\big] - D(Q \| P) \le \log \mathbb E_{h \sim P}\big[e^{g(h)}\big]. \]

Read it from the inside out. \(g\) assigns a number \(g(h)\) to each hypothesis (\(\mathbb R\) is the set of real numbers). On the left, \(\mathbb E_{h \sim Q}[g(h)]\) is \(g\)’s average under the posterior, and from it we subtract the posterior’s price. On the right, \(e^{g(h)}\) is averaged under the prior, and the logarithm brings it back to the scale of \(g\). In plain words: **whatever a posterior gains on average over the prior, minus what it paid in relative entropy, is at most the log of the prior’s exponential average.** The right side does not mention \(Q\) at all, so one bound on it bounds every posterior at once. Take \(g = (1.0, 0.3, 2.0, 1.5, 0.2)\) on the five thresholds and the committee: the left side is \(0.6 \times 2.0 + 0.4 \times 1.5 - 0.5205 = 1.2795\), and the right side is \(\log(0.1e^{1} + 0.2e^{0.3} + 0.4e^{2} + 0.2e^{1.5} + 0.1e^{0.2}) = 1.5076\).

*Proof.* If \(D(Q\|P) = +\infty\), the left side is \(-\infty\) and there is nothing to prove. Otherwise every \(h\) with \(Q(h) > 0\) has \(P(h) > 0\). Write \(A = \{h : Q(h) > 0\}\). Put both terms on the left into one sum over \(A\), and use \(g(h) = \log e^{g(h)}\) and \(\log a - \log b = \log(a/b)\):

\[ \mathbb E_{Q}[g] - D(Q\|P) = \sum_{h \in A} Q(h)\Big[g(h) - \log\frac{Q(h)}{P(h)}\Big] = \sum_{h \in A} Q(h) \log\frac{e^{g(h)} P(h)}{Q(h)}. \]

The right side is \(\mathbb E_{h \sim Q}[\log Y(h)]\) for the positive quantity \(Y(h) = e^{g(h)}P(h)/Q(h)\). By Jensen’s inequality for the logarithm, it is at most \(\log \mathbb E_{h \sim Q}[Y(h)]\), and in that average the \(Q(h)\) cancels:

\[ \sum_{h \in A} Q(h) \log Y(h) \le \log \sum_{h \in A} Q(h)\, \frac{e^{g(h)}P(h)}{Q(h)} = \log \sum_{h \in A} P(h)\, e^{g(h)} \le \log \sum_{h \in \mathcal H} P(h)\, e^{g(h)}. \]

The last step adds back the hypotheses outside \(A\), whose terms are nonnegative, and uses that \(\log\) is increasing. The final sum is \(\mathbb E_{h \sim P}[e^{g(h)}]\), finite because \(g\) is bounded. \(\blacksquare\)

The inequality cannot be improved: the posterior \(Q^\star(h) = P(h)e^{g(h)}/\mathbb E_P[e^g]\), which tilts the prior towards hypotheses where \(g\) is large, makes \(Y\) constant, so Jensen’s step is an equality, and \(A\) is the whole support of \(P\), so the last step is too.

```python
from math import exp, log
P = [0.1, 0.2, 0.4, 0.2, 0.1]
g = [1.0, 0.3, 2.0, 1.5, 0.2]
def lhs(Q):
    D = sum(q * log(q / p) for q, p in zip(Q, P) if q > 0)
    return sum(q * x for q, x in zip(Q, g)) - D
rhs = log(sum(p * exp(x) for p, x in zip(P, g)))
Z = sum(p * exp(x) for p, x in zip(P, g))
tilted = [p * exp(x) / Z for p, x in zip(P, g)]
print(round(lhs([0, 0, 0.6, 0.4, 0]), 4), round(rhs, 4))
print(round(lhs(tilted), 4), [round(q, 3) for q in tilted])
```

It prints `1.2795 1.5076`, then `1.5076 [0.06, 0.06, 0.654, 0.198, 0.027]`: the committee stays below the bound, and the tilted posterior reaches it exactly.

[[diagram:change-of-measure]]

**Step 6: every posterior at once.** Fix any \(S\) in the good event \(G\), and apply the change of measure with \(g(h) = 2(m-1)\Delta^+(h)^2\), which is bounded (between 0 and \(2(m-1)\), since a gap between risks in \([0, 1]\) is at most 1). Its right side is exactly \(\log \Phi(S)\), which is below \(\log(m/\delta)\) on \(G\). So for every posterior \(Q\),

\[ 2(m-1)\, \mathbb E_{h \sim Q}\big[\Delta^+(h)^2\big] - D(Q\|P) \le \log \Phi(S) < \log\frac{m}{\delta}. \]

In plain words: **on a good training sequence, the posterior-average of the scaled squared gap exceeds the posterior’s price by less than \(\log(m/\delta)\).** This is where “every posterior, even one chosen after seeing \(S\)” is earned. The event \(G\) was fixed in step 4, before any \(Q\). Step 5 holds for every \(Q\) and every \(S\), with no probability in it. So whatever posterior you choose after looking at \(S\), even one chosen by an adversary who sees everything, it satisfies the inequality whenever \(S \in G\). (If \(D(Q\|P) = \infty\), the theorem’s right side is infinite and there is nothing to prove; from here on assume it is finite.)

**Step 7: back to the risks.** Divide by \(2(m-1)\) and move the price to the right: \(\mathbb E_Q[\Delta^+(h)^2] < (D(Q\|P) + \log(m/\delta))/(2(m-1))\). Jensen’s inequality for the square gives \((\mathbb E_Q[\Delta^+(h)])^2 \le \mathbb E_Q[\Delta^+(h)^2]\), and taking square roots of two nonnegative numbers keeps the order, so

\[ \mathbb E_{h \sim Q}\big[\Delta^+(h)\big] < \sqrt{\frac{D(Q\|P) + \log(m/\delta)}{2(m-1)}}. \]

Finally, a gap is at most its positive part, \(\Delta(h) \le \Delta^+(h)\), and by the formulas for the Gibbs predictor’s risks, \(L_{\mathcal D}(Q) - L_S(Q) = \mathbb E_Q[L_{\mathcal D}(h) - L_S(h)] = \mathbb E_Q[\Delta(h)] \le \mathbb E_Q[\Delta^+(h)]\). Combining the two lines gives the theorem on \(G\), and \(G\) has probability at least \(1 - \delta\). \(\blacksquare\)

## Watching the promise hold

A proof is a promise; let’s watch it be kept. Each simulated week below draws 100 labels (a smaller budget, so the margins are wider and failures have more room to happen), and only then chooses a posterior from the data: it tilts the prior towards thresholds with low empirical risk, \(Q(h) \propto P(h)e^{-50 L_S(h)}\), where \(\propto\) reads “proportional to” (the weights are rescaled to sum to 1). The section on the learning rule studies this kind of posterior properly; here it simply stands for “a posterior that depends on the labels”. The week fails if the true risk exceeds the certificate.

```python
import random
from math import exp, log, sqrt
rules = [0.50, 0.55, 0.60, 0.65, 0.70]
P = [0.1, 0.2, 0.4, 0.2, 0.1]
L_D = [0.02 + 0.96 * abs(t - 0.62) for t in rules]
m, rng, worst, fails = 100, random.Random(3), 0.0, 0
for week in range(2000):
    xs = [rng.random() for _ in range(m)]
    S = [(s, int(s >= 0.62) ^ (rng.random() < 0.02)) for s in xs]
    L_S = [sum((s >= t) != y for s, y in S) / m for t in rules]
    w = [p * exp(-50 * l) for p, l in zip(P, L_S)]
    Q = [x / sum(w) for x in w]
    D = sum(q * log(q / p) for q, p in zip(Q, P))
    gap = sqrt((D + log(m / 0.05)) / (2 * (m - 1)))
    excess = sum(q * (a - b) for q, a, b in zip(Q, L_D, L_S))
    fails += excess > gap
    worst = max(worst, excess / gap)
print(fails / 2000, round(worst, 2))
```

It prints `0.0 0.26`. In 2,000 weeks the certificate never failed, and the true risk never came within three quarters of the margin of breaking it. The theorem allows failure on 5% of weeks; it uses far less. Like every bound in this course, it holds for every distribution, and on a friendly one it is cautious.

## Old certificates, seen again

The theorem quietly contains the certificates of earlier lessons. Pick the prior and put the posterior on a single hypothesis, and the relative entropy becomes \(\log(1/P(h))\), the third fact above.

[[diagram:special-cases]]

**A finite class with no preference.** Let \(\mathcal H\) be finite, the prior uniform, \(P(h) = 1/|\mathcal H|\), and the posterior all on the hypothesis \(h_S\) returned by any learner, say ERM (lesson 2’s empirical risk minimization). Then \(D = \log|\mathcal H|\), the Gibbs predictor is \(h_S\) itself, and with probability at least \(1 - \delta\)

\[ L_{\mathcal D}(h_S) \le L_S(h_S) + \sqrt{\frac{\log|\mathcal H| + \log(m/\delta)}{2(m-1)}}. \]

\(|\mathcal H|\) is the number of hypotheses, and the rest is the theorem with \(D(Q\|P) = \log|\mathcal H|\). In plain words: **with no prior preference, the price of committing to one hypothesis is the log of the class size, as in lessons 2 and 4.** This is the same shape as lesson 4’s finite-class certificate \(\sqrt{\log(2|\mathcal H|/\delta)/(2m)}\), with \(\log(m/\delta)\) in place of \(\log(2/\delta)\) and \(m - 1\) in place of \(m\). For the billion thresholds at 650 labels the two margins are 0.1525 and 0.137. PAC-Bayes pays a little more, the \(\log m\) that the moment lemma’s generosity cost us.

**Description length.** Let \(d\) be a prefix-free description language for \(\mathcal H\), as in lesson 7, with \(|h|\) the length of \(h\)’s description in bits. By the Kraft inequality, \(K = \sum_h 2^{-|h|} \le 1\), so \(P(h) = 2^{-|h|}/K\) is a prior, and \(P(h) \ge 2^{-|h|}\). Put the posterior on any single \(h\): its price is \(\log(1/P(h)) \le |h| \log 2\), and the theorem gives, for every \(h \in \mathcal H\) at once,

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{|h|\log 2 + \log(m/\delta)}{2(m-1)}}. \]

In plain words: **a hypothesis that takes \(|h|\) bits to write down pays \(|h|\log 2\) for being chosen, which is Occam’s razor from lesson 7.** Lesson 7’s 7-bit decision list, with empirical risk 0.17 at 100 labels, got 0.401 from the MDL bound; here it gets 0.421.

```python
from math import log, sqrt
def pac_bayes_gap(D, m, delta=0.05):
    return sqrt((D + log(m / delta)) / (2 * (m - 1)))
m, N = 650, 10**9
finite = sqrt(log(2 * N / 0.05) / (2 * m))
print(round(pac_bayes_gap(log(N), m), 4), round(finite, 4))
occam = sqrt((7 + log(2 / 0.05)) / (2 * 100))
print(round(0.17 + pac_bayes_gap(7 * log(2), 100), 3),
      round(0.17 + occam, 3))
```

It prints `0.1525 0.137`, then `0.421 0.401`.

So for a single hypothesis, the old certificates are slightly sharper. PAC-Bayes earns its keep when the posterior spreads over many hypotheses that each have little prior trust but together have a lot. That is exactly the committee: 0.119 against the old 0.147, 0.160 and 0.166. A union bound pays for hypotheses one at a time; the relative entropy pays for belief as a whole.

## The PAC-Bayes rule

The theorem holds for every posterior simultaneously, so it suggests a learner: choose the posterior with the smallest certificate.

**The PAC-Bayes rule.** Given a prior \(P\) fixed in advance and a training sequence \(S\), return

\[ Q_S \in \operatorname*{argmin}_{Q} \bigg[ L_S(Q) + \sqrt{\frac{D(Q\|P) + \log(m/\delta)}{2(m-1)}} \bigg], \]

and predict with its Gibbs predictor. Here \(\operatorname{argmin}_Q\) is the set of posteriors where the bracket is smallest, and \(\in\) says we return one of them. In plain words: **pick the randomized filter whose certificate is lowest, trading empirical risk against the price of moving away from the prior.** It is the regularized risk minimization of lesson 13 in another guise: empirical risk plus a penalty, where the penalty measures distance from the prior instead of the size of a weight vector. And the certificate of the returned posterior is valid, even though the posterior was chosen by looking at \(S\), because the theorem covered every posterior in advance.

Searching over every distribution is a lot, so in practice you search a family. A natural one tilts the prior towards hypotheses with low empirical risk. For a number \(\lambda \ge 0\) (lambda, the **temperature dial**), the **Gibbs posterior** is

\[ Q_\lambda(h) = \frac{P(h)\, e^{-\lambda L_S(h)}}{Z_\lambda}, \qquad Z_\lambda = \sum_{h \in \mathcal H} P(h)\, e^{-\lambda L_S(h)}. \]

- \(e^{-\lambda L_S(h)}\) shrinks the prior score of \(h\) by a factor that is smaller the more mistakes \(h\) makes on \(S\).
- \(Z_\lambda\) is the total after shrinking, and dividing by it makes the posterior add up to 1.
- \(\lambda = 0\) gives back the prior; as \(\lambda\) grows, the posterior concentrates on the hypotheses with the lowest empirical risk.

In plain words: **the Gibbs posterior is the prior reweighted by an exponential penalty on empirical risk.** With \(\lambda = 30\) and this week’s risks, \(h_{0.60}\)’s score \(0.4\) becomes \(0.4 e^{-30 \times 0.0415} = 0.115\) before rescaling, and \(h_{0.50}\)’s \(0.1\) becomes \(0.1e^{-30 \times 0.1446} = 0.0013\).

Why this family? Because each of its members is the best possible at its own exchange rate between error and price.

**Proposition (the Gibbs posterior is optimal).** For every \(\lambda \ge 0\) and every posterior \(Q\),

\[ \lambda L_S(Q) + D(Q\|P) = D(Q\|Q_\lambda) - \log Z_\lambda \ge -\log Z_\lambda, \]

with equality exactly when \(Q = Q_\lambda\).

On the left, \(\lambda L_S(Q)\) is the posterior’s empirical risk, converted into the units of the price by the exchange rate \(\lambda\), plus its price. In the middle, \(D(Q\|Q_\lambda)\) is how far \(Q\) is from the Gibbs posterior, and \(\log Z_\lambda\) does not depend on \(Q\). In plain words: **among all posteriors, \(Q_\lambda\) minimizes empirical risk times \(\lambda\) plus relative entropy, and any other posterior pays extra exactly its distance to \(Q_\lambda\).** So no posterior has both a lower empirical risk and a lower price than a Gibbs posterior: if it did, its left side would be smaller, which the proposition forbids.

*Proof.* Assume \(D(Q\|P) < \infty\) (otherwise both sides are infinite). Expand \(D(Q\|Q_\lambda)\) using \(\log(Q_\lambda(h)) = \log P(h) - \lambda L_S(h) - \log Z_\lambda\):

\[ D(Q\|Q_\lambda) = \sum_{h \,:\, Q(h) > 0} Q(h)\Big[\log\frac{Q(h)}{P(h)} + \lambda L_S(h) + \log Z_\lambda\Big] = D(Q\|P) + \lambda L_S(Q) + \log Z_\lambda, \]

because the weights \(Q(h)\) add up to 1 and \(\sum_h Q(h)L_S(h) = L_S(Q)\). Rearranging gives the equality. The relative entropy is never negative and is 0 exactly when \(Q = Q_\lambda\). \(\blacksquare\)

This is the change of measure again, with \(g = -\lambda L_S\): the tilted posterior that made it an equality is the Gibbs posterior.

```python
from math import exp, log
P = [0.1, 0.2, 0.4, 0.2, 0.1]
L_S = [0.1446, 0.0846, 0.0415, 0.0431, 0.0985]
lam = 30
def kl(A, B):
    return sum(a * log(a / b) for a, b in zip(A, B) if a > 0)
Z = sum(p * exp(-lam * l) for p, l in zip(P, L_S))
Q_lam = [p * exp(-lam * l) / Z for p, l in zip(P, L_S)]
def objective(Q):
    return lam * sum(q * l for q, l in zip(Q, L_S)) + kl(Q, P)
Q = [0, 0, 0.6, 0.4, 0]
print(round(objective(Q), 4), round(kl(Q, Q_lam) - log(Z), 4))
print(round(objective(Q_lam), 4), round(-log(Z), 4))
```

It prints `1.7847 1.7847`, then `1.6483 1.6483`: the identity for the committee, and the minimum reached by \(Q_{30}\).

Here is the learner, scanning the dial \(\lambda\) over a grid and keeping the posterior with the lowest certificate, followed by the Gibbs predictor.

```
PACBayesRule(S, H, P, δ, Λ):
  m ← |S|
  for each h in H:
    L_S[h] ← (1/m) · Σ_i ℓ(h, z_i)
  best ← none
  for each λ in Λ:
    w[h] ← P(h) · exp(−λ · L_S[h])   for each h in H
    Q[h] ← w[h] / Σ_h' w[h']          for each h in H
    D ← Σ_{h : Q[h] > 0} Q[h] · log(Q[h] / P(h))
    c ← Σ_h Q[h] · L_S[h] + sqrt((D + log(m/δ)) / (2(m−1)))
    if best is none or c < best.c: best ← (λ, Q, c)
  return best

GibbsPredict(Q, x):
  draw h from H with probability Q[h]
  return h(x)
```

**Cost.** The empirical risks take \(O(m|\mathcal H|)\) loss evaluations, the dominant cost when the class is small. Each value of \(\lambda\) then costs \(O(|\mathcal H|)\) arithmetic, so a grid of \(G\) values costs \(O(G|\mathcal H|)\), and the whole rule \(O(m|\mathcal H| + G|\mathcal H|)\) time. Space is \(O(|\mathcal H|)\) for the risks and one posterior, plus the sequence itself. A Gibbs prediction draws from \(Q\) in \(O(|\mathcal H|)\) time by walking the cumulative sums; if you precompute those sums once, in \(O(|\mathcal H|)\), each draw is a binary search, \(O(\log|\mathcal H|)\), then one hypothesis evaluation. Choosing \(\lambda\) by looking at the certificates is legal: every \(Q_\lambda\) is a posterior, and the theorem covers all of them at once.

For a billion thresholds, a sum over the class is out of the question, but the dial has structure. With windows as posteriors, the empirical risk \(L_S(h_t)\) only changes when \(t\) crosses one of the \(m\) scores, so after sorting the scores in \(O(m\log m)\) time, the average over any window is a sum over at most \(m + 1\) flat pieces, \(O(m)\) per window, and the window’s price is one logarithm. A Gibbs prediction draws one number uniformly from the window, \(O(1)\). That is what the lab computes.

On the small rulebook, the scan over \(\lambda = 0, 10, \dots, 3000\) gives:

```python
from math import exp, log, sqrt
P = [0.1, 0.2, 0.4, 0.2, 0.1]
L_S = [0.1446, 0.0846, 0.0415, 0.0431, 0.0985]
m = 650
def certificate(lam):
    w = [p * exp(-lam * l) for p, l in zip(P, L_S)]
    Q = [x / sum(w) for x in w]
    D = sum(q * log(q / p) for q, p in zip(Q, P) if q > 0)
    L = sum(q * l for q, l in zip(Q, L_S))
    return L + sqrt((D + log(m / 0.05)) / (2 * (m - 1))), Q
lams = range(0, 3001, 10)
best = min(lams, key=lambda lam: certificate(lam)[0])
c, Q = certificate(best)
print(best, round(c, 4), [round(q, 3) for q in Q])
print(round(certificate(0)[0], 4), round(certificate(3000)[0], 4))
```

It prints `230 0.1297 [0.0, 0.0, 0.743, 0.257, 0.0]`, then `0.1519 0.1309`. The best posterior puts about three quarters on \(h_{0.60}\) and a quarter on \(h_{0.65}\), with certificate 0.1297. The prior itself (\(\lambda = 0\)) certifies only 0.1519; nearly all belief on \(h_{0.60}\) (\(\lambda = 3000\)) certifies 0.1309. The middle wins, a little, even in a rulebook of five.

**Build it yourself.** Using only the Python standard library:

```python
import random
from math import exp, log, sqrt

rng, S = random.Random(2026), []
for _ in range(650):
    s = rng.random()
    S.append((s, int(s >= 0.62) ^ (rng.random() < 0.02)))
rules = [0.50, 0.55, 0.60, 0.65, 0.70]
P = [0.1, 0.2, 0.4, 0.2, 0.1]

def empirical_risks(S, rules): ...
def kl(Q, P): ...
def gibbs_posterior(P, L_S, lam): ...
def certificate(Q, P, L_S, m, delta): ...
def pac_bayes_rule(P, L_S, m, delta, lams): ...
def gibbs_predict(Q, rules, s, rng): ...

if __name__ == "__main__":
    L_S = empirical_risks(S, rules)
    assert [round(l, 4) for l in L_S] == [
        0.1446, 0.0846, 0.0415, 0.0431, 0.0985]
    assert round(kl([0, 0, 0.6, 0.4, 0], P), 4) == 0.5205
    blind = [0, 0.5, 0.5, 0, 0]
    assert kl([0.5, 0.5, 0, 0, 0], blind) == float("inf")
    c = certificate([0, 0, 0.6, 0.4, 0], P, L_S, 650, 0.05)
    assert round(c, 4) == 0.1299
    lam, Q, c = pac_bayes_rule(P, L_S, 650, 0.05,
                               range(0, 3001, 10))
    assert lam == 230 and round(c, 4) == 0.1297
    r = random.Random(0)
    wrong = sum(gibbs_predict(Q, rules, s, r) != y for s, y in S)
    assert abs(wrong / len(S) - sum(
        q * l for q, l in zip(Q, L_S))) < 0.01
```

1. Implement the six functions. `kl` returns `float("inf")` when \(Q\) uses a hypothesis that \(P\) rules out; `pac_bayes_rule` returns `(lam, Q, c)` for the first \(\lambda\) in `lams` with the lowest certificate; `gibbs_predict` draws a threshold with probabilities `Q` and labels the score `s`.
2. Run the asserts. The last one checks that your Gibbs predictor’s mistake rate on the labels is close to \(L_S(Q)\).
3. Add the dial: a function that takes a window \([a, b]\) and the band prior and returns the window’s empirical risk (by sorting the scores) and its price \(\log(1/P([a, b]))\). Check it against the committee’s 0.0294 and 0.9253.
4. Predict before you run: if you double the budget to 1,300 labels and keep the same posterior, by roughly what factor does the margin shrink?

## Fix the bug

Back to the reviewer. You now hold the tool the team was missing, and the lab hands you the whole job.

The rules of the game are the reviewer’s. You write the prior first: flat over the dial, or a band of trust you place and size yourself, with whatever share of the trust you choose inside it. The prior locks. Then the week’s labels arrive, at most 650 of them, and you choose the posterior, a window on the dial, after looking at them. The lab computes the window’s empirical risk, its price under your prior, and the certificate at \(\delta = 0.05\). To ship, the certificate must be at most 0.13. And because the reviewer has learned to distrust a lucky week, your recipe (the same prior, and the same rule for placing the window) must certify at least 19 of 20 fresh weeks.

Remember what the team knows: last year’s threshold, tuned on mail from the known-bad senders, was 0.60, and the threshold has never drifted more than 0.05 in a year. Then think about the trade from the act 1 lab: a window too wide is inaccurate, a window too narrow is expensive.

[[lab-act2]]

## What people get wrong

**“The posterior has to be Bayes’ posterior.”** It doesn’t. The theorem holds for every distribution over the class, chosen in any way, and Bayes’ rule appears nowhere in the proof. The names come from the Bayesian habit of thinking “belief before, belief after”; the guarantee is pure PAC.

**“A mix of rules makes fewer mistakes than its best member.”** The Gibbs predictor’s true risk is a weighted average of its members’ true risks, so it is never below the best member’s. The committee’s 0.0296 is above 0.02, the risk of the perfect threshold. What a mix can lower is the certificate, because spreading belief over many hypotheses that the prior trusted together costs far less than pinning it on one.

**“If the data show the prior was wrong, fix the prior.”** Then it is no longer a prior. Step 3 of the proof pulled each \(P(h)\) out of an average over training sequences because it was a constant. A prior rewritten after seeing \(S\) is a function of \(S\), and the step breaks. The honest move is to keep the prior and let the posterior move, paying the relative entropy for it.

**“PAC-Bayes always beats the older bounds.”** On a single hypothesis it is slightly looser: \(\log(m/\delta)\) instead of \(\log(2/\delta)\), and \(m - 1\) instead of \(m\). It wins when belief is spread over many hypotheses with little individual trust, as on the dial.

**“A hypothesis the prior gave zero trust is merely expensive.”** It is impossible to use. If the posterior gives it any weight, the relative entropy is infinite and the certificate says nothing. A prior should give every hypothesis you might ever want to use some trust, which is why the team left a 1% layer under the whole dial.

## The end of the road

This is the last lesson, so let’s look back down the road the mail room travelled.

It started with a filter that memorized every label and guessed like a coin on everything else. Lesson 1 found the reason: a learner that is willing to believe anything learns nothing, and the cure was an **inductive bias**, a hypothesis class chosen before any data. Lesson 2 put a price on that choice, the logarithm of the class size, paid in labels. Lessons 3 to 6 made the promise exact and let the class grow without bound, as long as its VC dimension stayed finite, and lesson 5 proved that no learner escapes the need for a bias. Lesson 7 let the bias become graded, trusting some hypotheses more than others, and paid for each by its weight or its description length. Lesson 8 asked what learning costs in computation as well as in labels. Lessons 9 to 20 built the machinery: halfspaces, boosting, validation, convexity and regularization, stochastic gradient descent, margins and kernels, many classes and rankings, decision trees, nearest neighbours and neural networks, each with its own bias and its own price. Lesson 21 faced spammers who adapt, with a filter that keeps learning, message by message. Lessons 22 to 25 learned from mail nobody labelled, compressed it, modelled how it is produced, and chose and built its features. Lessons 26 to 30 sharpened the theory: measuring a class on the data at hand, covering it with a few representatives, proving the fundamental theorem in full, learning many classes at once, and rebuilding a rule from a handful of its own examples.

And today the filter is **certified**: the reviewer signs a guarantee for the committee that ships. It is **explainable**: each answer is made by one threshold you can read. And it is **adaptive**: next week the labels change, the posterior moves, and the certificate is recomputed, all under a prior written down in advance.

Look at what every one of those lessons had in common. Each began by committing to something before the data arrived: a class, a weight, a description language, a regularizer, a margin, a kernel, a prior. And each paid for that commitment in examples: the log of the class size, the VC dimension, the description length, the norm of the weights, and today the relative entropy. That is the one idea the whole course has been about. **Learning is a bias chosen before the data, paid for in examples.** The better your bias fits the world, the less it costs; and no learner, however clever, can do without one.

[[diagram:journey]]

Your filter ships. There is no next bug in this course. There are the problems below, which are the course’s last, and after them the whole field, which is still writing new chapters of exactly this story.

## Concept check

Write your answers down, with your reasoning, not just the answer.

1. After seeing the week’s labels, a teammate adds a new threshold, 0.30, to the class. It was not in the prior, so its prior trust is 0. She puts 10% of the committee on it. What does the certificate say? She then proposes to rewrite the prior so that 0.30 gets 1% of the trust. Which step of the proof does that break, and why?
2. In the proof, Markov’s inequality is applied to \(\Phi(S)\), an average over the prior, not to any average over the posterior. Explain why it could not be applied to the posterior directly, and point to the step at which the inequality becomes valid for every posterior at once. Then: why did the proof work with the positive part \(\Delta^+(h)\) instead of the gap \(\Delta(h)\) itself?
3. On the small rulebook, use a flat prior instead of the team’s, \(P(h) = 1/5\) for each threshold, and the posterior that puts one half on \(h_{0.60}\) and one half on \(h_{0.65}\). With this week’s 650 labels and \(\delta = 0.05\), compute \(D(Q\|P)\), \(L_S(Q)\) and the certificate. Then compute the certificate of the same posterior under the team’s prior. Which is lower, and why?

## End-of-course problems

These are the course’s last problems. Work on them with pencil and Python; hints are available when you are stuck, but no solutions.

1. **A two-sided certificate.** The proof bounded the true risk from above only. Let \(X\) be a random variable, of either sign, with \(\mathbb P[|X| \ge \varepsilon] \le 2e^{-2m\varepsilon^2}\) for every \(\varepsilon > 0\). Show that \(\mathbb E[e^{2(m-1)X^2}] \le 2m - 1\). Use it to prove that with probability at least \(1 - \delta\), every posterior satisfies \(|L_{\mathcal D}(Q) - L_S(Q)| \le \sqrt{(D(Q\|P) + \log((2m-1)/\delta))/(2(m-1))}\).
2. **A single hypothesis, compared.** Take a finite class, the uniform prior, and the posterior that puts all its belief on the hypothesis \(h_S\) a learner returns. Write down the certificate the theorem gives, and compare it with lesson 4’s finite-class certificate \(L_S(h_S) + \sqrt{\log(2|\mathcal H|/\delta)/(2m)}\). Prove that for every \(m \ge 2\) the PAC-Bayes margin is at least as large, and find out when the two are close.
3. **Occam for a committee.** Take a prefix-free description language and the prior \(P(h) = 2^{-|h|}/K\) from the section on old certificates. Let \(A\) be any set of hypotheses, chosen after seeing the labels, and \(Q_A\) the prior restricted to \(A\). Prove a certificate for the Gibbs predictor of \(Q_A\) whose price is at most the description length, in bits times \(\log 2\), of the *shortest* member of \(A\). What does this say about adding long hypotheses to a committee that already contains a short one?
