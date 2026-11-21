# Online Regret

## The newsletter that emptied the vote

Lesson 21a ended with a filter that shipped with a proved mistake bound, and with a newsletter. One user reported it as junk; an hour later her colleague kept the very same message. Let’s replay that Wednesday on the smallest class we have, the four one-feature hypotheses \(\mathcal H_1 = \{h_F, h_L, h_U, h_N\}\). A message is four bits FLUN (says “FREE”, has a link, unknown sender, sent at night), and \(h_F\) says junk exactly when the F bit is 1, and so on. The learner is Halving, which lets the hypotheses that still agree with every label vote, with ties going to junk. The newsletter is 0000: no “FREE”, no link, a known sender, sent in the day.

```python
H = "FLUN"
h = lambda name, x: int(x[H.index(name)])
halving = lambda V, x: int(2 * sum(h(n, x) for n in V)
                           >= len(V))
week = [("1100", 0), ("0010", 0), ("1111", 1), ("0001", 1),
        ("0000", 1), ("0000", 0), ("0100", 0), ("1000", 0),
        ("0010", 0), ("1001", 1)]
V, M = set(H), 0
for x, y in week:
    p = halving(V, x)
    M += p != y
    V = {n for n in V if h(n, x) == y}
    print(x, p, y, "".join(sorted(V, key=H.index)) or "-")
print(M, "mistakes")
```

It prints ten rows and then `7 mistakes`. Read the rows one at a time. Each shows the message, Halving’s prediction, the label, and the hypotheses still standing. The first four rounds are lesson 21a’s Halving table: two mistakes, and the survivors shrink to \(h_N\) alone (`1100 1 0 UN`, `0010 1 0 N`, then two rows ending in `N`). On round 5 the newsletter arrives, \(h_N\) says keep, so Halving says keep, and the user reports junk: a mistake, and \(h_N\) leaves too. The row reads `0000 0 1 -`, and the dash is an empty set. From then on the vote is zero against zero, a tie, and ties go to junk. Round 6 is the same newsletter, kept this time, and Halving calls it junk (`0000 1 0 -`). So is every keep message after it: rounds 7, 8 and 9 are all `1 0 -`. Seven mistakes in ten rounds, against a proved bound of two, and the count grows by one with every keep message for the rest of the week.

The other learners fare no better. SOA sees two empty sides, both of dimension \(-1\), a tie, and says junk; Consistent has no survivor to follow at all, so it crashes.

The proof did not lie. It promised at most two mistakes on any sequence that some hypothesis in \(\mathcal H_1\) labels perfectly, and this week no hypothesis does: no hypothesis in any class can call the same message junk on one round and keep on the next. Real mail is like this all the time. So here is the bug. Every guarantee you have rests on a perfect hypothesis, and the mail room will never have one. What can a filter promise when the best hypothesis makes mistakes too, and the adversary still reads every decision? This lesson answers that, and it ends with filters cheap enough to run on every message of the day.

## What you need from lesson 21a

Here is the toolkit, one line each, so you can check that nothing has slipped.

- **The round.** On round \(t\), a message \(x_t\) arrives, the learner announces a **prediction** \(p_t\) (1 for junk, 0 for keep), the true label \(y_t\) is revealed, and the learner may update. \(T\) is the number of rounds, and \([T] = \{1, \dots, T\}\) is the set of round numbers. A **mistake** is a round with \(p_t \neq y_t\).
- **The adversary.** Whoever chooses the messages, and from now on the labels too, possibly after seeing every past prediction. Its **opening trick**: send a message, wait for the prediction, and declare the opposite label, which wins every round when nothing restricts it.
- **Realizability and the mistake bound.** A sequence is realizable by a class \(\mathcal H\) when some \(h^\star \in \mathcal H\) has \(y_t = h^\star(x_t)\) on every round. The mistake bound \(M_A(\mathcal H)\) is the most mistakes any adversary can force on learner \(A\) with realizable sequences; it is a \(\sup\), “the largest value, allowed to be infinite”.
- **The version space** \(V_t\): the hypotheses that agree with every label of rounds \(1, \dots, t - 1\). For a message \(x\) and a label \(r\), \(V^{(r)} = \{h \in V : h(x) = r\}\) is the part of \(V\) that says \(r\) on \(x\); the superscript names a label, not a power.
- **The Littlestone dimension** \(\mathrm{Ldim}(\mathcal H)\): the depth of the deepest **shattered tree**, a complete binary tree of messages (go left after label 0, right after label 1) in which every path of labels is followed by some hypothesis. The empty class gets \(-1\).
- **The tree lower bound.** Every online learner can be forced into \(\mathrm{Ldim}(\mathcal H)\) mistakes: the adversary walks a shattered tree and always declares the opposite label.
- **SOA**, the Standard Optimal Algorithm: split the version space by what it says on \(x_t\), predict with the side of higher Littlestone dimension (ties go to 1), and keep the side the label confirms. It makes at most \(\mathrm{Ldim}(\mathcal H)\) mistakes, so \(\mathcal H\) is online learnable exactly when \(\mathrm{Ldim}(\mathcal H) < \infty\).
- **Three classes.** \(\mathcal H_1\) has dimension 2. The **rogue sender** class (a message is just its sender, and \(h_j\) says junk exactly when the sender is \(j\)) has dimension 1 however many senders there are. **Thresholds** \(h_\theta(x) = \mathbb 1[x > \theta]\) on a score in \([0, 1]\) (the indicator \(\mathbb 1[\cdot]\) is 1 when the statement inside is true) have VC dimension 1 but Littlestone dimension \(\infty\): the adversary plays binary search.

## When no hypothesis is perfect

Realizability was a comfortable assumption, and the spammers are not comfortable people. Real mail is noisy: some users report newsletters they subscribed to, a few junk messages look exactly like real ones, and no hypothesis in any class you can afford agrees with every label. Lesson 3 met the same problem in the batch world and answered it with **agnostic** learning: stop asking for a perfect hypothesis and ask instead to do almost as well as the best one in the class.

The online version of “almost as well as the best one” is called regret. The idea first, with numbers. Suppose that after 1,000 rounds your filter has made 130 mistakes. Looking back, you check every hypothesis in \(\mathcal H\) and find that the best of them, had you followed it from the first round, would have made 100. You are “sorry” about 30 mistakes. That 30 is your **regret** with respect to that hypothesis. If the worst-case regret grows more slowly than the number of rounds, say like \(\sqrt T\), then the regret per round goes to zero, and in the long run the filter is as good as the best hypothesis in hindsight, against any sequence at all.

In symbols, for an online learner \(A\), a hypothesis \(h\) and a number of rounds \(T\),

\[ \mathrm{Regret}_A(h, T) = \sup_{(x_1, y_1), \dots, (x_T, y_T)} \left[ \sum_{t=1}^T |p_t - y_t| - \sum_{t=1}^T |h(x_t) - y_t| \right], \qquad \mathrm{Regret}_A(\mathcal H, T) = \sup_{h \in \mathcal H} \mathrm{Regret}_A(h, T). \]

Read it from the inside out.

- \(|p_t - y_t|\) is the learner’s loss on round \(t\). With predictions and labels in \(\{0, 1\}\), it is 1 for a mistake and 0 otherwise.
- \(\sum_{t=1}^T\) adds that loss over the rounds \(1, \dots, T\): the first sum is the learner’s mistake count, and the second, with \(h(x_t)\) in place of \(p_t\), is the mistake count of the fixed hypothesis \(h\) on the same messages.
- The bracket is the difference: how many more mistakes the learner made than \(h\).
- \(\sup\) over the sequence \((x_1, y_1), \dots, (x_T, y_T)\) takes the worst sequence of \(T\) rounds for the learner. No realizability this time: any labels at all.
- \(\sup_{h \in \mathcal H}\) then takes the hypothesis that makes the learner look worst, which is the best hypothesis in hindsight.

In plain words: **the regret is how many more mistakes the learner makes than the best hypothesis in the class, on the worst sequence of \(T\) rounds.** With the numbers above, \(130 - 100 = 30\) on that sequence; the regret is the largest such difference any sequence can produce. The goal is a regret that grows **sublinearly** in \(T\), meaning \(\mathrm{Regret}_A(\mathcal H, T)/T \to 0\) as \(T \to \infty\).

## The flipping adversary wins again

Is sublinear regret possible? Not for any learner that decides deterministically, even for a class of two hypotheses. Take the smallest interesting class: \(h_0\) says keep on every message and \(h_1\) says junk on every message. The adversary plays lesson 21a’s opening trick again: it waits for the prediction and announces the opposite label. The learner makes \(T\) mistakes in \(T\) rounds.

And the best hypothesis in hindsight? Out of the \(T\) labels, either at least half are 0 or at least half are 1. Whichever label is in the majority, the constant hypothesis that always says it is wrong on at most \(T/2\) rounds. So the regret is at least \(T - T/2 = T/2\), which is linear in \(T\). This is **Cover’s impossibility result**.

[[diagram:cover]]

```python
def follow_the_majority(labels):
    return int(2 * sum(labels) > len(labels))

T, labels, mistakes = 1000, [], 0
for t in range(T):
    p = follow_the_majority(labels)
    y = 1 - p
    mistakes += p != y
    labels.append(y)
best = min(sum(labels), T - sum(labels))
print(mistakes, best, mistakes - best)
coin = sum(abs(0.5 - y) for y in labels)
print(coin, coin - best)
```

It prints `1000 500 500`: a learner that predicts the majority label so far makes 1,000 mistakes, the better constant hypothesis makes 500, and the regret is 500, half of \(T\). The second line is a hint of the way out. A learner that ignores everything and predicts junk with probability one half would expect to lose half a mistake per round, 500 here, the same as the best constant. Randomness is something the flipping adversary cannot read in advance. (The plain coin is not the answer, though: if every label were junk, the coin would still lose 500 while \(h_1\) lost nothing.)

The same thing happens with more hypotheses. Here is a deterministic learner with ten **experts** to choose from (we define experts properly below; for now think of ten hypotheses) that follows whichever one has the fewest mistakes so far. The adversary makes the one it picked wrong, every round:

```python
d, T = 10, 2000
totals, paid = [0] * d, 0
for t in range(T):
    pick = min(range(d), key=lambda i: totals[i])
    v = [int(i == pick) for i in range(d)]
    paid += v[pick]
    totals = [s + vi for s, vi in zip(totals, v)]
print(paid, min(totals), paid - min(totals))
```

It prints `2000 200 1800`: the learner pays on every one of the 2,000 rounds, the punishment is spread over all ten experts so the best of them pays only 200, and the regret is 1,800, which is \(0.9\,T\). Following the leader is the most natural deterministic strategy there is, and it is the one the adversary eats first.

## Let the learner flip a coin

So we give the learner a coin, and we change one rule of the game: the adversary must choose the label \(y_t\) without seeing the outcome of the learner’s coin on round \(t\). It may know the learner’s code, its probabilities, and every coin flip of earlier rounds. It just cannot see this round’s flip before it commits. For a spam filter that is realistic: the spammer sends the message before the filter decides.

Now the learner announces a number \(p_t \in [0, 1]\), the probability with which it will say junk, and then draws its actual label \(\hat y_t\), which is 1 with probability \(p_t\). How often is it wrong? If the true label is 1, it is wrong when it draws 0, with probability \(1 - p_t\). If the true label is 0, it is wrong when it draws 1, with probability \(p_t\). Both cases are the same formula:

\[ \mathbb P[\hat y_t \neq y_t] = |p_t - y_t|. \]

- \(\hat y_t\) (“y hat”) is the label the learner actually outputs after flipping its coin; \(p_t\) is now the coin’s bias, not a label.
- \(\mathbb P[\cdot]\) is the probability of the event in brackets, over the learner’s own coin only. The message and label are fixed when the coin is flipped.

In plain words: **a learner that says junk with probability \(p_t\) is wrong with probability equal to the distance between \(p_t\) and the true label.** With \(p_t = 0.3\): if the message is junk, it misses with probability \(0.7\); if it is keep, it raises a false alarm with probability \(0.3\).

```python
import random

rng, p, n = random.Random(4), 0.3, 100_000
for y in (1, 0):
    wrong = sum((rng.random() < p) != y for _ in range(n))
    print(y, round(wrong / n, 3), abs(p - y))
```

It prints `1 0.699 0.7` and `0 0.301 0.3`: the measured error rates over 100,000 coin flips land on \(|p - y|\). So the regret formula above does not change at all. We simply let \(p_t\) be any number in \([0, 1]\) and read \(\sum_t |p_t - y_t|\) as the learner’s **expected** number of mistakes, the expectation being over its own coins.

## Weighted-Majority

The learner that takes advantage of the coin is called **Weighted-Majority**, and it is easiest to describe in a slightly more general setting called **prediction with expert advice**. There are \(d\) **experts**: advisers whose advice the learner may follow. On each round, the learner chooses a probability distribution over the experts, a vector of weights \(\mathbf w^{(t)} = (w_1^{(t)}, \dots, w_d^{(t)})\) with every \(w_i^{(t)} \ge 0\) and \(\sum_i w_i^{(t)} = 1\), and follows expert \(i\) with probability \(w_i^{(t)}\). Then the adversary reveals a **cost** for every expert, a vector \(\mathbf v_t \in [0, 1]^d\), where \(v_{t,i}\) is what following expert \(i\) would have cost this round. The learner’s expected cost is the weighted average of the experts’ costs, \(\sum_i w_i^{(t)} v_{t,i} = \langle \mathbf w^{(t)}, \mathbf v_t \rangle\), using the inner product of lesson 9a.

For the filter, the experts will be hypotheses and the cost of an expert will be 1 if it got the message wrong and 0 if it got it right. But the general version is cleaner to analyse, so let’s stay with it.

The idea of Weighted-Majority, in words: trust every expert equally at first, and every time an expert costs something, shrink its trust by a factor that depends on how much it cost. Experts that keep being wrong fade away exponentially fast, while experts that are mostly right keep their weight. Because the learner always spreads its bets, the adversary cannot single out one choice to punish.

```
Weighted-Majority(d, T):
  η ← sqrt(2 log(d) / T)
  w̃^(1) ← (1, ..., 1)
  for t = 1, 2, ..., T:
    Z_t ← sum of w̃^(t)_i over i
    w^(t) ← w̃^(t) / Z_t
    choose expert i with probability w^(t)_i
    receive costs v_t in [0, 1]^d
    pay ⟨w^(t), v_t⟩
    for every i: w̃^(t+1)_i ← w̃^(t)_i · exp(−η v_{t,i})
```

The update is the heart of it:

\[ \tilde w_i^{(t+1)} = \tilde w_i^{(t)} e^{-\eta v_{t,i}}, \qquad \mathbf w^{(t)} = \frac{\tilde{\mathbf w}^{(t)}}{Z_t}, \quad Z_t = \sum_{i=1}^d \tilde w_i^{(t)}. \]

- \(\tilde w_i^{(t)}\) (“w tilde”) is expert \(i\)’s raw trust before round \(t\); it starts at 1.
- \(e^{-\eta v_{t,i}}\) is the shrink factor: 1 when the cost is 0, and \(e^{-\eta} \approx 1 - \eta\) when the cost is 1. The number \(\eta > 0\) is the **step size**, as in lessons 14a and 15b, and it sets how hard one bad round is punished. The pseudocode sets it to \(\sqrt{2 \log(d)/T}\), with \(\log\) the natural logarithm, as everywhere in this course.
- \(Z_t\) is the total raw trust, and dividing by it turns the raw trusts into weights that add up to 1: \(\mathbf w^{(t)}\) is a probability distribution.

In plain words: **multiply each expert’s trust by \(e^{-\eta}\) raised to its cost, then rescale so the trusts add up to one.** With two experts, \(\eta = 0.5\), and a first round where expert 1 costs 1 and expert 2 costs 0: the raw trusts become \(e^{-0.5} \approx 0.607\) and 1, so the weights become \(0.607/1.607 \approx 0.38\) and \(1/1.607 \approx 0.62\).

Let’s watch it against a hostile adversary. Ten experts, 2,000 rounds. On every round, the adversary looks at the weights and charges cost 1 to the five experts the learner currently trusts most, and cost 0 to the other five. A deterministic learner would be destroyed by this, as we saw. Weighted-Majority:

```python
from math import exp, log, sqrt

d, T = 10, 2000
eta = sqrt(2 * log(d) / T)
w_tilde, paid, totals = [1.0] * d, 0.0, [0.0] * d
for t in range(T):
    Z = sum(w_tilde)
    w = [wi / Z for wi in w_tilde]
    trusted = sorted(range(d), key=lambda i: -w[i])[:d // 2]
    v = [1.0 if i in trusted else 0.0 for i in range(d)]
    paid += sum(wi * vi for wi, vi in zip(w, v))
    totals = [s + vi for s, vi in zip(totals, v)]
    w_tilde = [wi * exp(-eta * vi) for wi, vi in zip(w_tilde, v)]
bound = sqrt(2 * log(d) * T)
print(round(paid, 1), min(totals), round(paid - min(totals), 1),
      round(bound, 1))
```

It prints `1012.0 1000.0 12.0 96.0`. The learner’s expected cost is 1,012, the best expert in hindsight paid 1,000, so the regret is 12, well under the 96 that the theorem below promises. Each line of the loop is a line of the pseudocode: `Z` is \(Z_t\), `w` is \(\mathbf w^{(t)}\), `paid` accumulates \(\langle \mathbf w^{(t)}, \mathbf v_t \rangle\), and the last line is the update.

[[diagram:weights]]

Before the proof, play with it. In the lab below you choose the adversary and the learner. The punisher is the adversary you just watched: every round it charges the five experts the learner trusts most. The steady adversary is the opposite of clever: expert 1 never costs anything, and the other nine always cost 1. The learners are follow-the-leader, the plain coin that keeps every weight at one tenth (Weighted-Majority with \(\eta = 0\)), and Weighted-Majority with a slider for \(\eta\). Watch the regret against each adversary as you move the slider, and look for the \(\eta\) that is safe against both.

[[lab]]

What the slider shows is two failures pulling in opposite directions. Here are both fights in code, for three step sizes:

```python
from math import exp

def regret(eta, adversary, d=10, T=2000):
    w, paid, totals = [1 / d] * d, 0.0, [0.0] * d
    for t in range(T):
        v = adversary(w)
        paid += sum(wi * vi for wi, vi in zip(w, v))
        totals = [s + vi for s, vi in zip(totals, v)]
        w = [wi * exp(-eta * vi) for wi, vi in zip(w, v)]
        Z = sum(w)
        w = [wi / Z for wi in w]
    return round(paid - min(totals), 1)

def punisher(w):
    top = sorted(range(10), key=lambda i: -w[i])[:5]
    return [float(i in top) for i in range(10)]

steady = lambda w: [0.0] + [1.0] * 9
for eta in (0.01, 0.048, 0.5):
    print(eta, regret(eta, punisher), regret(eta, steady))
```

It prints `0.01 2.5 230.7`, `0.048 12.0 48.4` and `0.5 122.5 5.1`. A timid \(\eta = 0.01\) shrugs off the punisher (regret 2.5) but takes hundreds of rounds to notice that expert 1 never errs, and the steady adversary bills it 230.7. A bold \(\eta = 0.5\) finds expert 1 almost at once (5.1) but lurches after every punishment, and the punisher bills it 122.5. The middle value, the one in the pseudocode, is never the best against either adversary and never bad against either. The theorem explains why, and it holds against every adversary at once, not just these two.

**Theorem (Weighted-Majority’s regret).** If \(T > 2 \log d\), Weighted-Majority with \(\eta = \sqrt{2 \log(d)/T}\) satisfies, for every sequence of cost vectors,

\[ \sum_{t=1}^T \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_{i \in [d]} \sum_{t=1}^T v_{t,i} \le \sqrt{2 \log(d) \, T}. \]

- The first sum is the learner’s total expected cost over \(T\) rounds.
- \(\sum_t v_{t,i}\) is expert \(i\)’s total cost, and \(\min_{i \in [d]}\) picks the cheapest expert in hindsight.

In plain words: **over any \(T\) rounds, Weighted-Majority pays at most \(\sqrt{2 \log(d)\, T}\) more than the best expert, whatever the adversary does.** With ten experts and 2,000 rounds, \(\sqrt{2 \cdot 2.303 \cdot 2000} \approx 96\), so at most about 4.8% of the rounds, and that fraction keeps shrinking like \(1/\sqrt T\).

*Proof.* The plan: watch the total raw trust \(Z_t\). We show it must fall quickly whenever the learner pays a lot (an upper bound), and it cannot fall below the trust of the best expert (a lower bound). Squeezed between the two, the learner’s cost cannot run far ahead of the best expert’s.

*Step 1: one round of shrinking.* Divide the new total by the old one:

\[ \log \frac{Z_{t+1}}{Z_t} = \log \sum_i \frac{\tilde w_i^{(t)}}{Z_t} e^{-\eta v_{t,i}} = \log \sum_i w_i^{(t)} e^{-\eta v_{t,i}}. \]

The first equality writes \(Z_{t+1} = \sum_i \tilde w_i^{(t)} e^{-\eta v_{t,i}}\), which is the update, and divides each term by \(Z_t\). The second recognizes \(\tilde w_i^{(t)}/Z_t\) as the weight \(w_i^{(t)}\). In words: **the log of how much the total trust shrank this round is the log of the weighted average of the shrink factors.** In the snippet above, on round 1 every weight is 0.1 and five experts shrink by \(e^{-0.048} \approx 0.953\), so the average shrink factor is \(0.5 + 0.5 \cdot 0.953 = 0.977\).

*Step 2: a quadratic bound on the exponential.* For \(0 \le a \le 1\), \(e^{-a} \le 1 - a + a^2/2\). Why: the difference \(g(a) = 1 - a + a^2/2 - e^{-a}\) has \(g(0) = 0\) and slope \(g'(a) = -1 + a + e^{-a}\), which is at least 0 because \(e^{-a} \ge 1 - a\) (the exponential lies above its tangent line at 0, the convexity of lesson 12a). So \(g\) never decreases from 0. Apply it with \(a = \eta v_{t,i}\), which is in \([0, 1]\) because \(v_{t,i} \in [0, 1]\) and \(\eta < 1\) (that is what \(T > 2 \log d\) buys: \(\eta^2 = 2 \log(d)/T < 1\)). Since the weights add up to 1,

\[ \log \frac{Z_{t+1}}{Z_t} \le \log \sum_i w_i^{(t)} \left( 1 - \eta v_{t,i} + \frac{\eta^2 v_{t,i}^2}{2} \right) = \log(1 - b), \qquad b = \sum_i w_i^{(t)} \left( \eta v_{t,i} - \frac{\eta^2 v_{t,i}^2}{2} \right). \]

Here \(\log\) is increasing, so a bigger inside gives a bigger log; and \(\sum_i w_i^{(t)} \cdot 1 = 1\) pulled the 1 out front. The new name \(b\) is just the weighted average of \(\eta v - \eta^2 v^2/2\). In words: **the shrink per round is at most \(\log(1 - b)\), where \(b\) is roughly \(\eta\) times the learner’s cost.**

*Step 3: \(\log(1 - b) \le -b\).* This holds for every \(b < 1\), because \(1 - b \le e^{-b}\) (the same tangent-line fact) and \(\log\) is increasing. Our \(b\) is less than 1 since each term \(\eta v - \eta^2 v^2/2 \le \eta < 1\). So

\[ \log \frac{Z_{t+1}}{Z_t} \le -\eta \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{\eta^2}{2} \sum_i w_i^{(t)} v_{t,i}^2 \le -\eta \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{\eta^2}{2}. \]

The first inequality expands \(-b\) into its two parts, the first being \(\eta\) times the learner’s expected cost. The second uses \(v_{t,i}^2 \le 1\) and weights that add up to 1. In words: **every round, the total trust shrinks (in log terms) by at least \(\eta\) times what the learner paid, minus a small \(\eta^2/2\).** On round 1 of the hostile run (\(\eta \approx 0.048\), the learner pays 0.5), the true shrink and the three bounds of Steps 2 and 3 read \(-0.023705 \le -0.023696 \le -0.023417 \le -0.022841\).

```python
from math import exp, log

grid = [k / 1000 for k in range(1001)]
print(all(exp(-a) <= 1 - a + a * a / 2 for a in grid))
print(all(log(1 - b) <= -b for b in grid[:-1] + [-0.5, -3]))
print(round(1 - 1 + 1 / 2 - exp(-1), 3))
```

It prints `True`, `True` and `0.132`: both inequalities hold on a fine grid (including some negative \(b\)), and at \(a = 1\) the quadratic is 0.132 above \(e^{-1}\), so the bound is not tight but is never wrong.

*Step 4: add up the rounds.* Summing Step 3 over \(t = 1, \dots, T\), the left side telescopes: \(\log(Z_2/Z_1) + \log(Z_3/Z_2) + \dots = \log Z_{T+1} - \log Z_1\). And \(Z_1 = d\), since every raw trust starts at 1. So

\[ \log Z_{T+1} - \log d \le -\eta \sum_{t=1}^T \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{T \eta^2}{2}. \]

In words: **the total trust has fallen by at least \(\eta\) times the learner’s total cost, minus \(T\eta^2/2\).**

*Step 5: the best expert holds the total up.* Unrolling the update, expert \(i\)’s raw trust after \(T\) rounds is \(\tilde w_i^{(T+1)} = e^{-\eta \sum_t v_{t,i}}\). The total is at least its largest term, so

\[ \log Z_{T+1} = \log \sum_i e^{-\eta \sum_t v_{t,i}} \ge \log \max_i e^{-\eta \sum_t v_{t,i}} = -\eta \min_i \sum_t v_{t,i}. \]

The last step: the largest \(e^{-\eta c}\) belongs to the smallest cost \(c\), and \(\log e^{-\eta c} = -\eta c\). In words: **the total trust is at least the trust of the best expert, which has only shrunk by its own cost.**

*Step 6: squeeze.* Steps 4 and 5 bound \(\log Z_{T+1}\) from both sides:

\[ -\eta \min_i \sum_t v_{t,i} - \log d \le -\eta \sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{T \eta^2}{2}. \]

Move the learner’s cost to the left and divide by \(\eta > 0\):

\[ \sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_i \sum_t v_{t,i} \le \frac{\log d}{\eta} + \frac{\eta T}{2}. \]

This holds for *every* \(\eta \in (0, 1)\), and it is the form we will use in the lab. Read it as two forces: a small \(\eta\) learns slowly (the first term, the cost of starting with no idea which expert is good), and a large \(\eta\) overreacts (the second term). With \(\eta = \sqrt{2 \log(d)/T}\), the first term is \(\log d \cdot \sqrt{T/(2 \log d)} = \sqrt{T \log(d)/2}\) and the second is \(\frac T2 \sqrt{2 \log(d)/T} = \sqrt{T \log(d)/2}\), and together they make \(2\sqrt{T \log(d)/2} = \sqrt{2 \log(d)\, T}\). \(\blacksquare\)

Here are Steps 3 to 5 checked on the same hostile run as before:

```python
from math import exp, log, sqrt

d, T = 10, 2000
eta = sqrt(2 * log(d) / T)
w_t, ok, paid, totals = [1.0] * d, True, 0.0, [0.0] * d
for t in range(T):
    Z = sum(w_t)
    w = [wi / Z for wi in w_t]
    trusted = sorted(range(d), key=lambda i: -w[i])[:d // 2]
    v = [1.0 if i in trusted else 0.0 for i in range(d)]
    wv = sum(wi * vi for wi, vi in zip(w, v))
    w_t = [wi * exp(-eta * vi) for wi, vi in zip(w_t, v)]
    ok &= log(sum(w_t) / Z) <= -eta * wv + eta ** 2 / 2
    paid += wv
    totals = [s + vi for s, vi in zip(totals, v)]
log_Z = log(sum(w_t))
print(ok, round(log_Z - log(d), 2),
      round(-eta * paid + T * eta ** 2 / 2, 2))
print(round(log_Z, 2), round(-eta * min(totals), 2))
```

It prints `True -47.99 -46.26` and `-45.68 -47.99`. Step 3 held on every round (`True`); the telescoped total \(\log Z_{T+1} - \log d = -47.99\) sits below Step 4’s right side, \(-46.26\); and \(\log Z_{T+1} = -45.68\) sits above Step 5’s floor, \(-47.99\).

And the choice of \(\eta\) really does minimize the two-force bound:

```python
from math import log, sqrt

d, T = 10, 2000
bound = lambda eta: log(d) / eta + eta * T / 2
etas = [k / 10_000 for k in range(1, 10_000)]
best = min(etas, key=bound)
print(best, round(bound(best), 2))
eta_star = sqrt(2 * log(d) / T)
print(round(eta_star, 4), round(sqrt(2 * log(d) * T), 2))
```

It prints `0.048 95.97` twice: a grid search over \(\eta\) finds the same minimum as the formula, at \(\eta \approx 0.048\), with value about 96.

What if \(T \le 2 \log d\)? Then \(\sqrt{2 \log(d)\, T} \ge T\), and the bound is true for free: no learner can pay more than \(T\) above anyone, since each round costs between 0 and 1. The condition only matters for the proof, not for the truth of the statement.

## From experts to hypotheses

Now back to classification. Let \(\mathcal H = \{h_1, \dots, h_d\}\) be finite, and make each hypothesis an expert: expert \(i\) advises \(h_i(x_t)\), and its cost is \(v_{t,i} = |h_i(x_t) - y_t|\), 1 if it is wrong and 0 if it is right. The learner predicts the weighted vote, \(p_t = \sum_i w_i^{(t)} h_i(x_t)\), as its probability of saying junk. For example, with weights 0.5, 0.3, 0.2 on three hypotheses that say junk, keep, junk, the learner says junk with probability \(0.5 + 0.2 = 0.7\).

Is the learner’s expected mistake \(|p_t - y_t|\) the same as Weighted-Majority’s cost \(\langle \mathbf w^{(t)}, \mathbf v_t \rangle\)? It is, and the reason is a sign argument:

\[ |p_t - y_t| = \Big| \sum_i w_i^{(t)} \big( h_i(x_t) - y_t \big) \Big| = \sum_i w_i^{(t)} \big| h_i(x_t) - y_t \big| = \langle \mathbf w^{(t)}, \mathbf v_t \rangle. \]

The first equality uses \(\sum_i w_i^{(t)} = 1\) to write \(y_t = \sum_i w_i^{(t)} y_t\). For the second: if \(y_t = 1\), every \(h_i(x_t) - y_t\) is 0 or \(-1\), all of one sign; if \(y_t = 0\), every one is 0 or \(+1\). The absolute value of a sum of terms of one sign is the sum of their absolute values. In words: **because all the hypotheses err in the same direction, the vote’s distance from the label is exactly the weighted fraction of hypotheses that are wrong.** In the example, if the message is keep, \(|0.7 - 0| = 0.7\), and the hypotheses that are wrong (the two saying junk) carry weight \(0.5 + 0.2 = 0.7\).

```python
import random

rng = random.Random(3)
for trial in range(10_000):
    d = rng.randint(1, 6)
    w = [rng.random() for _ in range(d)]
    w = [wi / sum(w) for wi in w]
    h = [rng.randint(0, 1) for _ in range(d)]
    y = rng.randint(0, 1)
    p = sum(wi * hi for wi, hi in zip(w, h))
    rhs = sum(wi * abs(hi - y) for wi, hi in zip(w, h))
    assert abs(abs(p - y) - rhs) < 1e-12
print("identity holds in 10,000 random checks")
```

It prints `identity holds in 10,000 random checks`. And \(\sum_t v_{t,i}\) is just the number of mistakes of \(h_i\). So Weighted-Majority’s theorem becomes, directly:

**Corollary (finite classes).** For a finite \(\mathcal H\) and \(T > 2 \log |\mathcal H|\), the weighted vote learner has, on every sequence,

\[ \sum_{t=1}^T |p_t - y_t| - \min_{h \in \mathcal H} \sum_{t=1}^T |h(x_t) - y_t| \le \sqrt{2 \log(|\mathcal H|) \, T}. \]

In plain words: **with a finite class, the randomized filter’s expected mistakes exceed those of the best hypothesis in hindsight by at most \(\sqrt{2 \log(|\mathcal H|)\, T}\).** For lesson 1’s ten one-feature hypotheses over 2,000 messages, that is the 96 we computed: under 5% extra mistakes per round, against the worst spammers.

What about an infinite class? The rogue-sender class from lesson 21a has one hypothesis per possible sender address, and there are infinitely many addresses, so \(\log |\mathcal H| = \infty\). But its Littlestone dimension is 1. Can we build a *finite* set of experts that, on whatever messages arrive, includes one that behaves exactly like the best hypothesis?

Here is the idea in words. SOA, run on messages labelled by some \(h\), makes at most \(L = \mathrm{Ldim}(\mathcal H)\) mistakes. If you knew *on which rounds* it would make them, you could reproduce \(h\) exactly: follow SOA’s guess, except flip it on those rounds. You don’t know the rounds, so create one expert for every possible set of at most \(L\) rounds. One of them has the right set.

```
Expert(i_1, ..., i_L), for rounds i_1 < ... < i_L:
  V_1 ← H
  for t = 1, 2, ..., T:
    receive x_t
    for r in {0, 1}: V_t^(r) ← {h in V_t : h(x_t) = r}
    guess ← argmax over r of Ldim(V_t^(r))   (ties: 0)
    if t is one of i_1, ..., i_L: predict 1 − guess
    else: predict guess
    V_{t+1} ← V_t^(prediction)
```

Each expert only needs the messages so far, never the labels, so it can advise in real time.

**Lemma (some expert copies each hypothesis).** Let \(\mathrm{Ldim}(\mathcal H) < \infty\) and fix any messages \(x_1, \dots, x_T\). For every \(h \in \mathcal H\) there are \(L \le \mathrm{Ldim}(\mathcal H)\) and rounds \(i_1 < \dots < i_L\) such that \(\mathrm{Expert}(i_1, \dots, i_L)\) predicts \(h(x_t)\) on every round \(t\).

*Proof.* Run SOA (with ties going to 0, as in the expert) on the sequence \((x_1, h(x_1)), \dots, (x_T, h(x_T))\). It is realizable, labelled by \(h\), so SOA makes at most \(\mathrm{Ldim}(\mathcal H)\) mistakes; let \(L\) be their number and \(i_1, \dots, i_L\) the rounds where they happen. Now compare SOA with \(\mathrm{Expert}(i_1, \dots, i_L)\), round by round. Suppose both have the same version space at the start of round \(t\) (true at \(t = 1\)). They then compute the same guess. On a round in the list, SOA’s guess was wrong, so \(h(x_t) = 1 - \text{guess}\), which is what the expert predicts; on any other round, SOA’s guess was right, so \(h(x_t) = \text{guess}\), which again is what the expert predicts. Either way the expert predicts \(h(x_t)\), and both then keep \(V_t^{(h(x_t))}\), so the version spaces agree at round \(t + 1\) too. By induction, the expert predicts \(h(x_t)\) on every round. \(\blacksquare\)

In plain words: **for any hypothesis, one of the experts is a perfect copy of it on the messages that actually arrived.** Here is the construction on \(\mathcal H_1\), for six messages: for each of the four hypotheses, find SOA’s mistake rounds, then check that the expert with those rounds reproduces the hypothesis.

```python
from functools import lru_cache
from itertools import product

X = ["".join(b) for b in product("01", repeat=4)]
H = frozenset(tuple(int(x[k]) for x in X) for k in range(4))
part = lambda V, i, r: frozenset(h for h in V if h[i] == r)

@lru_cache(None)
def ldim(V):
    splits = [(part(V, i, 0), part(V, i, 1)) for i in range(16)]
    return -1 if not V else max([1 + min(map(ldim, s))
                                 for s in splits if all(s)],
                                default=0)

def run(xs, flips=(), labels=None):
    V, out = H, []
    for t, x in enumerate(xs, 1):
        i = X.index(x)
        guess = int(ldim(part(V, i, 1)) > ldim(part(V, i, 0)))
        flip = t in flips
        out.append(labels[t - 1] if labels else guess ^ flip)
        flips = flips if out[-1] == guess else (*flips, t)
        V = part(V, i, out[-1])
    return out, flips

xs = ["1100", "0011", "1010", "0110", "1111", "0001"]
for h in sorted(H):
    labels = [h[X.index(x)] for x in xs]
    flips = run(xs, labels=labels)[1]
    print(flips, run(xs, flips)[0] == labels)
```

It prints `(1, 3) True`, `(1,) True`, `(3,) True` and `() True`: the four hypotheses need flips at rounds {1, 3}, {1}, {3} and none, never more than \(\mathrm{Ldim}(\mathcal H_1) = 2\) flips, and each expert copies its hypothesis exactly. The function `run` is SOA when given `labels` (it records where its guess was wrong) and the expert when given `flips`.

How many experts is that? One for each set of at most \(L = \mathrm{Ldim}(\mathcal H)\) rounds out of \(T\):

\[ d = \sum_{k=0}^{L} \binom{T}{k} \le \left( \frac{eT}{L} \right)^L \quad (1 \le L \le T). \]

- \(\binom{T}{k}\) (“\(T\) choose \(k\)”) is the number of ways to pick \(k\) rounds out of \(T\).
- The sum counts all the sets of size \(0, 1, \dots, L\).
- The inequality is the polynomial bound from Sauer’s lemma in lesson 6b, with \(T\) in place of \(m\) and \(L\) in place of \(d\).

In plain words: **the number of experts grows only polynomially in \(T\), like \(T^L\), not exponentially.** For the rogue sender, \(L = 1\), and over \(T = 10{,}000\) rounds there are \(1 + 10{,}000 = 10{,}001\) experts, below \(e \cdot 10{,}000 \approx 27{,}183\). So \(\log d \le L \log(eT/L) \le L \log(eT)\). Now run Weighted-Majority over these experts. For any \(h \in \mathcal H\), the lemma gives an expert that copies \(h\) on the messages that arrived, so the best expert made at most as many mistakes as \(h\), and a regret bound against the best expert is a regret bound against \(h\): \(\sqrt{2 L \log(eT)\, T}\).

```python
from math import comb, e, log, sqrt

T, L = 10_000, 1
experts = sum(comb(T, k) for k in range(L + 1))
print(experts, round((e * T / L) ** L))
regret = sqrt(2 * log(experts) * T)
print(round(regret, 1), round(regret / T, 4))
print(round(sqrt(2 * L * log(e * T) * T), 1))
```

It prints `10001 27183`, then `429.2 0.0429` (the regret bound with the exact expert count, about 4.3% of the rounds), then `451.9` (the slightly looser \(\sqrt{2L\log(eT)\,T}\)). An infinite class, an adversary, and still a guarantee.

**Theorem (online learnability with regret).** For every class \(\mathcal H\) there is an online learner whose predictions are probabilities \(p_t \in [0, 1]\) and that satisfies, for every \(h \in \mathcal H\) and every sequence,

\[ \sum_{t=1}^T |p_t - y_t| - \sum_{t=1}^T |h(x_t) - y_t| \le \sqrt{2 \min\{\log |\mathcal H|, \; \mathrm{Ldim}(\mathcal H) \log(eT)\} \, T}. \]

Moreover, no learner can guarantee an expected regret smaller than a constant times \(\sqrt{\mathrm{Ldim}(\mathcal H)\, T}\).

In plain words: **with a coin, every class of finite Littlestone dimension can be learned online against any sequence, with regret growing like \(\sqrt T\); and that is the best possible rate.** For the rogue sender, \(\log |\mathcal H| = \infty\), so the minimum picks \(1 \cdot \log(e \cdot 10{,}000) \approx 10.2\), and over 10,000 rounds the bound is the 451.9 printed above. The upper bound is the two constructions above: the finite one, and the Littlestone experts (when \(\mathrm{Ldim} = 0\), the class has a single behaviour and the bound is 0). We prove the upper bound only; the matching lower bound is a research result we state without proof. So the Littlestone dimension decides learnability online in both settings, realizable and not, just as the VC dimension did for the batch world in lesson 6a.

## Online convex optimization

Weighted-Majority needs a list of experts, which for big classes is a big list. In lessons 12a and 14a you met a different way to make learning tractable: convex losses, minimized by stepping against a subgradient. Does that survive an adversary? Yes, almost word for word.

The setting in words: each round, the learner commits to a weight vector, the adversary (adapting to everything so far) reveals an example, and the learner pays a loss that is convex in the vector, as in lesson 12a, over a convex set of allowed vectors.

**Online convex optimization.** A hypothesis class \(\mathcal H \subseteq \mathbb R^n\) that is a convex set (we write \(n\) for the number of coordinates, since \(d\) already counts experts), a domain \(Z\) of examples, and a loss \(\ell : \mathcal H \times Z \to \mathbb R\) with \(\ell(\cdot, z)\) convex for every \(z\). For \(t = 1, \dots, T\): the learner announces \(\mathbf w^{(t)} \in \mathcal H\), the environment answers with \(z_t \in Z\), and the learner pays \(\ell(\mathbf w^{(t)}, z_t)\).

Regret is measured against any fixed vector \(\mathbf w^\star\):

\[ \mathrm{Regret}_A(\mathbf w^\star, T) = \sum_{t=1}^T \ell(\mathbf w^{(t)}, z_t) - \sum_{t=1}^T \ell(\mathbf w^\star, z_t), \qquad \mathrm{Regret}_A(\mathcal H, T) = \sup_{\mathbf w^\star \in \mathcal H} \mathrm{Regret}_A(\mathbf w^\star, T). \]

- The first sum is the learner’s total loss with its changing vectors; the second is the total loss of the single vector \(\mathbf w^\star\), held fixed, on the same examples.
- \(\sup_{\mathbf w^\star \in \mathcal H}\) picks the comparison vector that makes the learner look worst: the best fixed vector in hindsight.

In plain words: **the regret is how much more loss the learner paid than the best single weight vector it could have used throughout.** For example, if over 100 rounds the learner pays 60 and the best fixed vector would have paid 52, the regret against it is 8.

The learner is **Online Gradient Descent**: lesson 14b’s gradient descent with projection, run on the losses as they arrive. Recall from lesson 14a that a **subgradient** \(\mathbf v\) of a convex \(f\) at \(\mathbf w\) is the slope of a supporting plane: \(f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \mathbf v \rangle\) for every \(\mathbf u\); that \(f\) is **\(\rho\)-Lipschitz** exactly when every subgradient has norm at most \(\rho\); and the **projection lemma**: projecting onto a convex set never increases the distance to any point of the set.

```
OnlineGradientDescent(η):
  w^(1) ← 0
  for t = 1, 2, ..., T:
    predict w^(t)
    receive z_t; let f_t(w) = ℓ(w, z_t)
    choose a subgradient v_t of f_t at w^(t)
    w^(t+½) ← w^(t) − η v_t
    w^(t+1) ← the point of H closest to w^(t+½)
```

The half step moves against the subgradient, and the projection pulls the vector back into \(\mathcal H\) if it stepped outside.

[[diagram:ogd]]

**Theorem (Online Gradient Descent’s regret).** For every \(\mathbf w^\star \in \mathcal H\),

\[ \mathrm{Regret}_A(\mathbf w^\star, T) \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

If every \(f_t\) is \(\rho\)-Lipschitz and \(\eta = 1/\sqrt T\), then \(\mathrm{Regret}_A(\mathbf w^\star, T) \le \frac12 (\|\mathbf w^\star\|^2 + \rho^2) \sqrt T\). If moreover every vector in \(\mathcal H\) has norm at most \(B\) and \(\eta = B/(\rho \sqrt T)\), then \(\mathrm{Regret}_A(\mathcal H, T) \le B \rho \sqrt T\).

- \(\|\mathbf w^\star\|\) is the length of the comparison vector: how far it is from the starting point \(\mathbf 0\).
- \(\|\mathbf v_t\|^2\) is the squared length of the step direction on round \(t\).
- The two terms are lesson 14a’s two forces again: a timid step size pays for the distance to cover, and a bold one pays for the size of its steps.

In plain words: **against any sequence of convex losses, Online Gradient Descent pays at most \(B\rho\sqrt T\) more than the best fixed vector, the same bound gradient descent had in lesson 14a, now with no distribution at all.** With \(B = 1\), \(\rho = 1\) and \(T = 10{,}000\) rounds, the regret is at most 100, one hundredth of a loss per round.

*Proof.* Fix \(\mathbf w^\star \in \mathcal H\). Track the squared distance to it, \(\|\mathbf w^{(t)} - \mathbf w^\star\|^2\), and see how much one round changes it. Add and subtract the half step:

\[ \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 = \Big( \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 \Big) + \Big( \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 \Big). \]

The first bracket is at most 0 by the projection lemma, since \(\mathbf w^\star \in \mathcal H\). For the second, \(\mathbf w^{(t+\frac12)} = \mathbf w^{(t)} - \eta \mathbf v_t\), and expanding \(\|\mathbf a - \eta\mathbf v\|^2 = \|\mathbf a\|^2 - 2\eta\langle \mathbf a, \mathbf v \rangle + \eta^2\|\mathbf v\|^2\) with \(\mathbf a = \mathbf w^{(t)} - \mathbf w^\star\) gives \(-2\eta \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle + \eta^2 \|\mathbf v_t\|^2\). Finally, the subgradient inequality at \(\mathbf w^{(t)}\) with \(\mathbf u = \mathbf w^\star\) reads \(f_t(\mathbf w^\star) \ge f_t(\mathbf w^{(t)}) + \langle \mathbf w^\star - \mathbf w^{(t)}, \mathbf v_t \rangle\), that is, \(\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \ge f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star)\). Together:

\[ \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 \le -2\eta \big( f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \big) + \eta^2 \|\mathbf v_t\|^2. \]

In words: **every round, the distance to the comparison vector shrinks by at least \(2\eta\) times the regret of that round, up to a penalty of \(\eta^2\|\mathbf v_t\|^2\).**

Sum over \(t = 1, \dots, T\). The left side telescopes to \(\|\mathbf w^{(T+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(1)} - \mathbf w^\star\|^2\). With \(\mathbf w^{(1)} = \mathbf 0\), moving terms across and dividing by \(2\eta\):

\[ \sum_{t=1}^T \big( f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \big) \le \frac{\|\mathbf w^\star\|^2 - \|\mathbf w^{(T+1)} - \mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2 \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

The left side is the regret, since \(f_t(\mathbf w) = \ell(\mathbf w, z_t)\), and dropping the subtracted squared distance only makes the right side bigger. That is the first bound. If each \(f_t\) is \(\rho\)-Lipschitz, then \(\|\mathbf v_t\| \le \rho\), so the sum is at most \(T\rho^2\): with \(\eta = 1/\sqrt T\) the bound is \(\frac{\sqrt T}{2}\|\mathbf w^\star\|^2 + \frac{\sqrt T}{2}\rho^2\), and with \(\|\mathbf w^\star\| \le B\) and \(\eta = B/(\rho\sqrt T)\) it is \(\frac{B\rho\sqrt T}{2} + \frac{B\rho\sqrt T}{2} = B\rho\sqrt T\). \(\blacksquare\)

The proof is lesson 14a’s step lemma with one thing missing: there is no expectation anywhere. Lesson 14a needed fresh random examples to turn the step lemma into a statement about the true risk. Here the step lemma itself is the result, and it holds for every sequence, because it never cared where the \(\mathbf v_t\) came from.

Let’s check it on a one-dimensional fight. The class is \(\mathcal H = [-1, 1]\), the loss is \(\ell(w, z) = |w - z|\) (convex and 1-Lipschitz in \(w\)), and the adversary always picks \(z_t\) at the end of the interval farthest from the learner’s current \(w\).

```python
from math import sqrt

B, rho, T = 1.0, 1.0, 10_000
eta = B / (rho * sqrt(T))
w, paid, zs = 0.0, 0.0, []
for t in range(T):
    z = -1.0 if w > 0 else 1.0
    paid += abs(w - z)
    v = 1.0 if w > z else -1.0
    w = max(-B, min(B, w - eta * v))
    zs.append(z)
grid = [k / 100 - 1 for k in range(201)]
best = min(sum(abs(u - z) for z in zs) for u in grid)
print(round(paid, 1), round(best, 1), round(paid - best, 1),
      B * rho * sqrt(T))
```

It prints `10050.0 10000.0 50.0 100.0`: the learner pays 10,050, the best fixed point in hindsight pays 10,000, and the regret of 50 is inside the promised \(B\rho\sqrt T = 100\). The line `w = max(-B, min(B, w - eta * v))` is the half step and the projection onto \([-1, 1]\) in one go. And the per-round inequality of the proof, checked against \(w^\star = 0.5\):

```python
from math import sqrt

B, T, w_star = 1.0, 10_000, 0.5
eta = B / sqrt(T)
w, ok, total = 0.0, True, 0.0
for t in range(T):
    z = -1.0 if w > 0 else 1.0
    v = 1.0 if w > z else -1.0
    w_next = max(-B, min(B, w - eta * v))
    drop = (w_next - w_star) ** 2 - (w - w_star) ** 2
    gap = abs(w - z) - abs(w_star - z)
    ok &= drop <= -2 * eta * gap + eta ** 2 * v * v + 1e-12
    total += gap
    w = w_next
print(ok, round(total, 1), round(w_star ** 2 / (2 * eta)
                                  + eta * T / 2, 1))
```

It prints `True 50.0 62.5`: the per-round inequality held on all 10,000 rounds, and the regret against \(w^\star = 0.5\), 50, sits under the first bound \(\|w^\star\|^2/(2\eta) + \eta T\rho^2/2 = 12.5 + 50 = 62.5\).

## The online Perceptron

The filter of lessons 9a and 15a was a halfspace: a weight vector \(\mathbf w\) over word counts, predicting junk when the score \(\langle \mathbf w, \mathbf x \rangle\) is positive. For halfspaces we switch labels to \(y \in \{-1, +1\}\), as lesson 9a did, with \(+1\) for junk. Can an online filter learn halfspaces against an adversary?

Not with a mistake bound in general. In two or more dimensions, halfspaces contain the thresholds in disguise: take messages \((s, 1, 0, \dots, 0)\) whose first coordinate is a score \(s\), and weight vectors \((-1, a, 0, \dots, 0)\); then \(\langle \mathbf w, \mathbf x \rangle = a - s\), which is positive when \(s < a\), a threshold on \(s\). Binary search on \(s\) shatters trees of every depth, so the Littlestone dimension of halfspaces is infinite, and by lesson 21a’s tree lower bound every online learner can be forced into a mistake on every round.

So we need to assume more about the sequence, and lessons 9a and 15a tell us what: a **margin**. If some \(\mathbf w^\star\) scores every message with \(y_t \langle \mathbf w^\star, \mathbf x_t \rangle \ge 1\), the messages sit at distance at least \(1/\|\mathbf w^\star\|\) from its boundary, and an adversary cannot binary-search ever closer to it.

The tool is the one lesson 12b introduced for exactly this purpose, a **convex surrogate loss**. The 0-1 loss, \(\mathbb 1[y\langle \mathbf w, \mathbf x \rangle \le 0]\) (1 on a mistake, 0 otherwise), is not convex. The **hinge loss** \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\) is convex and is never below the 0-1 loss. Online, we may even pick a different surrogate on each round, after seeing the learner’s current vector, because Online Gradient Descent’s bound holds for every sequence of convex losses, however they were chosen. So we use

\[ f_t(\mathbf w) = \begin{cases} \max\{0, 1 - y_t \langle \mathbf w, \mathbf x_t \rangle\} & \text{if } y_t \langle \mathbf w^{(t)}, \mathbf x_t \rangle \le 0, \\ 0 & \text{otherwise.} \end{cases} \]

- \(y_t \langle \mathbf w, \mathbf x_t \rangle\) is the **signed score** of lesson 9a: positive when \(\mathbf w\) labels the message correctly.
- The case split looks at the learner’s *current* vector \(\mathbf w^{(t)}\): on a round where it errs (signed score at most 0), the surrogate is the hinge loss; on a round where it is right, the surrogate is the zero function.

In plain words: **charge the hinge loss on the rounds where the filter got the message wrong, and nothing on the rounds where it got it right.** Both cases are convex. And on a mistake round, \(f_t(\mathbf w^{(t)}) \ge 1 - y_t\langle \mathbf w^{(t)}, \mathbf x_t \rangle \ge 1\), while on other rounds \(f_t(\mathbf w^{(t)}) = 0\), so \(f_t(\mathbf w^{(t)})\) is always at least the 0-1 loss of the learner. For example, if \(\mathbf w^{(t)}\) gives a junk message the score \(-0.4\), the round is a mistake and \(f_t(\mathbf w^{(t)}) = 1.4 \ge 1\).

Now run Online Gradient Descent on \(f_1, f_2, \dots\) with \(\mathcal H = \mathbb R^n\) (no projection needed). On a correct round, \(f_t\) is zero and \(\mathbf v_t = \mathbf 0\). On a mistake round, the hinge part \(1 - y_t\langle \mathbf w, \mathbf x_t \rangle\) is at least 1 at \(\mathbf w^{(t)}\), so the max is taken by the linear piece, whose gradient is \(-y_t \mathbf x_t\). So the update is

\[ \mathbf w^{(t+1)} = \begin{cases} \mathbf w^{(t)} + \eta\, y_t \mathbf x_t & \text{if } y_t \langle \mathbf w^{(t)}, \mathbf x_t \rangle \le 0, \\ \mathbf w^{(t)} & \text{otherwise,} \end{cases} \]

which reads: **on a mistake, add \(\eta\) times the message, signed by its label; otherwise leave the weights alone.** That is lesson 9a’s Perceptron update, with a step size. Does the step size matter? Write \(\mathcal M\) for the set of rounds with an update. Starting from \(\mathbf 0\), the weights are a sum of past updates, so the prediction on round \(t\) is

\[ \operatorname{sign}\langle \mathbf w^{(t)}, \mathbf x_t \rangle = \operatorname{sign}\Big( \eta \sum_{i \in \mathcal M,\, i < t} y_i \langle \mathbf x_i, \mathbf x_t \rangle \Big). \]

- \(\sum_{i \in \mathcal M, i < t}\) runs over the earlier rounds on which the learner updated.
- Each term is the past message’s similarity to the new one, \(\langle \mathbf x_i, \mathbf x_t \rangle\), signed by its label.
- \(\operatorname{sign}\) returns \(+1\) for a positive number and \(-1\) otherwise.

In plain words: **the prediction is a vote of the past mistakes, each weighted by how similar it is to the new message, and the step size \(\eta > 0\) multiplies every vote equally, so it never changes the outcome.** So the set of mistake rounds is the same for every \(\eta > 0\), and we may as well take \(\eta = 1\). That is the online **Perceptron**.

```python
import random

dot = lambda a, b: sum(ai * bi for ai, bi in zip(a, b))
sign = lambda s: 1 if s > 0 else -1
for eta in (1.0, 0.01):
    rng, w, M, signs = random.Random(8), [0.0] * 3, [], []
    for t in range(200):
        x = [rng.uniform(-1, 1) for _ in range(3)]
        y = sign(x[0] - x[1] + 0.3 * x[2])
        dual = eta * sum(yi * dot(xi, x) for xi, yi in M)
        assert abs(dot(w, x) - dual) < 1e-9
        signs.append(sign(dot(w, x)))
        if y * dot(w, x) <= 0:
            M.append((x, y))
            w = [wi + eta * y * xi for wi, xi in zip(w, x)]
    print(eta, len(M), sum(signs))
```

It prints `1.0 19 -2` and `0.01 19 -2`: on the same 200 messages, \(\eta = 1\) and \(\eta = 0.01\) update on the same 19 rounds and make identical predictions, and the `assert` confirms on every round that the score equals the vote over past mistakes.

How many mistakes can the adversary force? The idea first: with a margin, the analysis of Online Gradient Descent caps them at \((R\|\mathbf w^\star\|)^2\); without one, every message on which \(\mathbf w^\star\) falls short of the margin adds its hinge loss to the bill.

**Theorem (the online Perceptron’s mistake bound).** Run the Perceptron on any sequence \((\mathbf x_1, y_1), \dots, (\mathbf x_T, y_T)\), let \(R = \max_t \|\mathbf x_t\|\), and let \(\mathcal M\) be the rounds on which it updates. Then for every \(\mathbf w^\star\),

\[ |\mathcal M| \le \sum_{t \in \mathcal M} [1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+ + R\|\mathbf w^\star\| \sqrt{\sum_{t \in \mathcal M} [1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+} + R^2\|\mathbf w^\star\|^2. \]

In particular, if \(y_t \langle \mathbf w^\star, \mathbf x_t \rangle \ge 1\) for every \(t\), then \(|\mathcal M| \le R^2 \|\mathbf w^\star\|^2\).

- \([a]_+ = \max\{0, a\}\), so \([1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+\) is the hinge loss of the comparison vector on round \(t\): 0 when \(\mathbf w^\star\) scores the message with margin, positive when it falls short.
- The sum over \(t \in \mathcal M\) adds that shortfall over the Perceptron’s mistake rounds; call it \(C\).
- \(R\|\mathbf w^\star\|\) is lesson 9a’s \(RB\): message size times comparison size.

In plain words: **the Perceptron’s mistakes are at most the comparison vector’s total hinge shortfall, plus a correction, plus \((R\|\mathbf w^\star\|)^2\); and if the comparison vector separates every message with margin, the mistakes never exceed \((R\|\mathbf w^\star\|)^2\), however the adversary orders them.** With \(R = 1\), \(\|\mathbf w^\star\| = 4\) and a shortfall of \(C = 20\): at most \(20 + 4\sqrt{20} + 16 \approx 53.9\) mistakes.

*Proof.* Apply Online Gradient Descent’s first bound to the surrogates \(f_t\), with \(\mathbf v_t = -y_t\mathbf x_t\) on update rounds and \(\mathbf 0\) otherwise:

\[ \sum_{t=1}^T f_t(\mathbf w^{(t)}) - \sum_{t=1}^T f_t(\mathbf w^\star) \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

Three substitutions. First, \(f_t(\mathbf w^{(t)}) \ge 1\) on each update round and 0 elsewhere, so the first sum is at least \(|\mathcal M|\). Second, \(f_t(\mathbf w^\star)\) is the hinge shortfall on update rounds and 0 elsewhere, so the second sum is \(C\). Third, \(\|\mathbf v_t\|^2 = \|\mathbf x_t\|^2 \le R^2\) on update rounds and 0 elsewhere, so the last sum is at most \(|\mathcal M| R^2\). So, for every \(\eta > 0\),

\[ |\mathcal M| - C \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} |\mathcal M| R^2. \]

The run, and therefore \(|\mathcal M|\), is the same for every \(\eta\), so we may choose \(\eta\) after the fact (if \(|\mathcal M| = 0\) there is nothing to prove). The choice \(\eta = \|\mathbf w^\star\|/(R\sqrt{|\mathcal M|})\) makes both terms equal to \(\frac12 R\|\mathbf w^\star\|\sqrt{|\mathcal M|}\), so

\[ |\mathcal M| - R\|\mathbf w^\star\| \sqrt{|\mathcal M|} - C \le 0. \]

In words: **the mistake count, minus a square-root correction, minus the shortfall, is not positive.**

It remains to solve this for \(|\mathcal M|\). Write \(x = |\mathcal M|\), \(b = R\|\mathbf w^\star\|\), \(c = C\), all nonnegative, so \(x - b\sqrt x - c \le 0\). Put \(y = \sqrt x\): then \(y^2 - by - c \le 0\), and a parabola that opens upward is at most 0 only between its roots, so \(y \le \frac{b + \sqrt{b^2 + 4c}}{2}\). Squaring, \(x \le \frac{2b^2 + 4c + 2b\sqrt{b^2 + 4c}}{4} = \frac{b^2}{2} + c + \frac b2 \sqrt{b^2 + 4c}\). Finally \(\sqrt{b^2 + 4c} \le b + 2\sqrt c\) (square both sides: \(b^2 + 4c \le b^2 + 4b\sqrt c + 4c\)), so \(x \le \frac{b^2}{2} + c + \frac{b^2}{2} + b\sqrt c = c + b^2 + b\sqrt c\). That is the theorem. If \(\mathbf w^\star\) has margin on every message, \(c = 0\) and \(x \le b^2 = R^2\|\mathbf w^\star\|^2\). \(\blacksquare\)

```python
import random
from math import sqrt

rng = random.Random(5)
for trial in range(10_000):
    b, c = rng.uniform(0, 10), rng.uniform(0, 10)
    root = ((b + sqrt(b * b + 4 * c)) / 2) ** 2
    assert abs(root - b * sqrt(root) - c) < 1e-9
    assert root <= c + b * b + b * sqrt(c) + 1e-9
print("x <= c + b^2 + b*sqrt(c) in 10,000 random checks")
```

It prints `x <= c + b^2 + b*sqrt(c) in 10,000 random checks`: for random \(b\) and \(c\), the largest \(x\) with \(x - b\sqrt x - c \le 0\) (the square of the parabola’s right root) is below the claimed bound.

Now the whole chain on a run where the labels come from \(\mathbf u = (0.6, 0.8)\), the messages are drawn from the square \([-1, 1]^2\) with no margin promised, and the comparison vector is \(\mathbf w^\star = \mathbf u / 0.1\), so \(\|\mathbf w^\star\| = 10\):

```python
import random
from math import hypot, sqrt

rng, gamma, u = random.Random(2), 0.1, (0.6, 0.8)
dot = lambda a, b: a[0] * b[0] + a[1] * b[1]
w_star = (u[0] / gamma, u[1] / gamma)
w, M, R, hinge = (0.0, 0.0), 0, 0.0, 0.0
while M < 40:
    x = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    y = 1 if dot(u, x) > 0 else -1
    if y * dot(w, x) <= 0:
        M, R = M + 1, max(R, hypot(*x))
        hinge += max(0.0, 1 - y * dot(w_star, x))
        w = (w[0] + y * x[0], w[1] + y * x[1])
norm = hypot(*w_star)
b, c = R * norm, hinge
print(M, round(c, 1), round(M - b * sqrt(M) - c, 1))
print(round(c + b * sqrt(c) + b * b, 1))
```

It prints `40 22.0 -66.3` and `261.9`: after 40 mistakes, \(\mathbf w^\star\)’s shortfall on those rounds is 22.0, the quantity \(|\mathcal M| - R\|\mathbf w^\star\|\sqrt{|\mathcal M|} - C\) is \(-66.3\), negative as the proof requires, and the theorem’s bound is 261.9, far above 40. (Here \(R\) is the largest norm among the update rounds, which is all the proof uses.)

And against an adversary that always sends the message the current Perceptron gets most wrong, from a pool of 5,000 messages that \(\mathbf u\) separates with margin \(\gamma = 0.1\) (every \(|\langle \mathbf u, \mathbf x \rangle| \ge 0.1\), \(\|\mathbf x\| \le 1\)):

```python
import random
from math import hypot

rng, gamma, u = random.Random(2), 0.1, (0.6, 0.8)
dot = lambda a, b: a[0] * b[0] + a[1] * b[1]
pool = []
while len(pool) < 5000:
    x = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    if hypot(*x) <= 1 and abs(dot(u, x)) >= gamma:
        pool.append((x, 1 if dot(u, x) > 0 else -1))

def perceptron(eta):
    w, M = (0.0, 0.0), 0
    while True:
        bad = [(y * dot(w, x), x, y) for x, y in pool
               if y * dot(w, x) <= 0]
        if not bad:
            return M
        s, x, y = min(bad)
        M += 1
        w = (w[0] + eta * y * x[0], w[1] + eta * y * x[1])

print(perceptron(1.0), perceptron(0.01), round(1 / gamma ** 2))
```

It prints `28 28 100`. With \(\mathbf w^\star = \mathbf u/\gamma\), every message has \(y\langle \mathbf w^\star, \mathbf x \rangle \ge 1\), \(R \le 1\) and \(\|\mathbf w^\star\| = 1/\gamma = 10\), so the theorem promises at most \(R^2\|\mathbf w^\star\|^2 = (R/\gamma)^2 \le 100\) mistakes. This adversary manages 28, the same for both step sizes, and once no message in the pool is misclassified it has nothing left to send.

[[diagram:surrogate]]

Margin is the assumption that replaces realizability: a halfspace that labels every message correctly but with a tiny margin still leaves the Perceptron open to many mistakes, because the Littlestone dimension is infinite.

## The learners as code

Here are the three learners of this lesson side by side, with what each costs per round. Write \(n\) for the number of features of a message and \(d\) for the number of experts.

```
WeightedMajority(d, T):
  η ← sqrt(2 log(d) / T); w̃ ← (1, ..., 1)
  each round: w ← w̃ / sum(w̃); follow expert i with prob. w_i
              receive costs v; w̃_i ← w̃_i · exp(−η v_i) for all i

OnlineGradientDescent(η):
  w ← 0
  each round: predict w; receive z
              v ← a subgradient of ℓ(·, z) at w
              w ← projection onto H of (w − η v)

Perceptron:
  w ← 0
  each round: receive x; predict sign⟨w, x⟩; receive y
              if y⟨w, x⟩ ≤ 0: w ← w + y x
```

**Weighted-Majority** stores \(d\) weights and spends \(O(d)\) time per round (plus \(d\) hypothesis evaluations when the experts are hypotheses). With the Littlestone experts, \(d\) is about \(T^L\) and each expert runs SOA, so that construction proves learnability but is not a practical filter. **Online Gradient Descent** and the **Perceptron** keep one vector: \(O(n)\) space and \(O(n)\) time per round (\(O(s)\) for a message with \(s\) nonzero word counts), plus the projection, which for a ball of radius \(B\) is a rescaling in \(O(n)\). These last two are the ones a real mail system can afford at millions of messages a day. Compare lesson 21a’s learners, which all keep the version space: \(O(|\mathcal H|)\) per round for Consistent and Halving, and an exponential game-tree search for SOA.

**Build it yourself.** Using only the Python standard library:

```python
from math import exp, log, sqrt

costs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]] * 50
points = [((1, 0), 1), ((0, 1), -1), ((1, 1), 1),
          ((-1, 0.5), -1), ((0.5, -1), 1), ((0.2, 0.9), -1)]

def weighted_majority(costs, eta): ...
def perceptron(points, eta=1.0): ...

if __name__ == "__main__":
    paid, best = weighted_majority(costs, sqrt(2 * log(3) / 200))
    assert paid - best <= sqrt(2 * log(3) * 200)
    assert round(paid, 2) == 61.48 and best == 50
    assert perceptron(points) == perceptron(points, 0.01) == 4
    print("all asserts pass")
```

1. `weighted_majority(costs, eta)` runs the algorithm on the list of cost vectors, three experts over 200 rounds, and returns the learner’s total expected cost and the best expert’s total cost. Before you run it, work out the best expert’s total by hand: in every block of four rounds, how much does each expert pay?
2. `perceptron(points, eta)` presents the points in order, over and over, until a full pass makes no update, and returns the number of updates. Check that the answer does not depend on `eta`.
3. Then write the punisher and the steady adversary of the lab as functions of the current weights, and confirm that your `weighted_majority`, fed by them round by round, reproduces the regrets 12.0 and 48.4 at \(\eta = 0.048\) with ten experts and 2,000 rounds.

## Fix the bug

Back to the newsletter week. You now have learners whose promises survive mail that no hypothesis labels perfectly, and the lab below puts them against an adversary that plays as well as it can: in every round it computes the costs or the message that hurt your learner most, while obeying whatever rules the week sets.

Each week has a rule and a budget. In the first three weeks no hypothesis is perfect, and your regret against the best expert over the week must stay under a budget; the weeks grow from ten experts over 2,000 rounds to a thousand experts over 10,000. In the last, the filter is a halfspace over word counts, the spammers can only promise a margin, and you must certify the number of mistakes. For each week you choose the learner and its parameters, and the lab first checks that the bound you just proved, with your parameters, is within the budget. Only then does it run the week. A filter that happens to do well against a lazy adversary earns nothing; the adversary here is not lazy, and the certificate is what ships.

Before you start, work out what each budget allows. If the adversary sees your prediction before choosing the label, what is the least regret it can force on any deterministic learner over 2,000 rounds? Which \(\eta\) makes \(\log(d)/\eta + \eta T/2\) smallest, and what is the smallest value? For the halfspace week, which of the three terms of the Perceptron’s bound grows when you pick a comparison vector with a bigger norm, and which when you pick one with a bigger shortfall?

[[lab-act2]]

## What people get wrong

**“Randomizing just adds noise to the filter.”** Without it, no learner has sublinear regret even against two constant hypotheses: the adversary reads the prediction and flips it. The coin works because the adversary must commit to the label before the coin is flipped, and the guarantee is about the expected number of mistakes.

**“The Weighted-Majority bound needs the adversary to be random.”** The bound holds for every sequence of costs, including ones chosen after looking at the weights. The only expectation in it is over the learner’s own coin.

**“The Perceptron learns any separable stream with few mistakes.”** Only with a margin. Without one, the Littlestone dimension of halfspaces is infinite, and an adversary can binary-search the boundary forever. The bound \(R^2\|\mathbf w^\star\|^2\) is large exactly when the margin \(1/\|\mathbf w^\star\|\) is small.

## Recap and the next bug

1. A single contradictory label empties the version space, and every learner of lesson 21a is left with nothing to vote on. Without realizability, we measure regret against the best hypothesis in hindsight instead of counting mistakes.
2. Deterministic learners can be forced to regret \(T/2\) (Cover’s impossibility). With a coin the adversary cannot see, Weighted-Majority has regret at most \(\sqrt{2 \log(d)\, T}\), proved by squeezing the total trust \(Z_t\) between the learner’s cost and the best expert’s. For a general \(\eta\) the bound is \(\log d/\eta + \eta T/2\): learning slowly against overreacting.
3. With experts built from SOA, every class of finite Littlestone dimension has regret \(O(\sqrt{\mathrm{Ldim}(\mathcal H)\, T \log T})\), so the Littlestone dimension decides online learnability with and without a perfect hypothesis.
4. For convex losses, Online Gradient Descent has regret at most \(B\rho\sqrt T\) against any sequence: lesson 14a’s step lemma, with no expectations.
5. The Perceptron is Online Gradient Descent on hinge surrogates charged only on mistake rounds. Its mistakes are at most \(C + R\|\mathbf w^\star\|\sqrt C + R^2\|\mathbf w^\star\|^2\), and at most \(R^2\|\mathbf w^\star\|^2\) when \(\mathbf w^\star\) separates with margin.

The filter holds its ground against the spammers now, with a promise written down before each week starts. Then a different problem lands on your desk. The team of people who label messages has quit, all of them, on the same day. Thousands of messages keep arriving and not one of them has a label: no junk, no keep, nothing to count mistakes against. And the company still wants the mail sorted into kinds, newsletters with newsletters, invoices with invoices, whatever the kinds turn out to be. Every learner in this course so far has needed labels. What can a learner possibly do with none? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Weighted-Majority’s theorem assumes \(T > 2 \log d\). Point to the step of the proof that uses it, and say what goes wrong there without it. Then explain why the final inequality \(\sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_i \sum_t v_{t,i} \le \sqrt{2 \log(d)\, T}\) is still true when \(T \le 2 \log d\).
2. A week of mail has \(d = 100\) experts and \(T = 5{,}000\) rounds. Which \(\eta\) does Weighted-Majority use, what regret does it guarantee, and what is that per round? In a later week the filter is a halfspace with messages of norm at most \(R = 1\), and a comparison vector with \(\|\mathbf w^\star\| = 4\) has total hinge shortfall 9 over the stream. At most how many mistakes can the Perceptron make?

## Problems

Once the concept check is done, try this. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (not knowing the horizon).** Weighted-Majority and Online Gradient Descent both choose their step size from the number of rounds \(T\), which a mail filter running forever does not know. Suppose an algorithm has regret at most \(\alpha\sqrt T\) whenever it is told \(T\). Run it afresh on rounds \(1\), then \(2, 3\), then \(4, \dots, 7\), and in general on the \(2^m\) rounds \(t = 2^m, \dots, 2^{m+1} - 1\), each time telling it the length of the period. Show that the total regret after any \(T\) rounds is at most \(\frac{\sqrt 2}{\sqrt 2 - 1}\,\alpha\sqrt T\).
