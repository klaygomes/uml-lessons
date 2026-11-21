# Boosting and fresh mail

## A promise nobody can sign

The boosted filter shipped at the end of lesson 10a with a cheerful release note: “wrong on 0.9% of messages.” A week later a customer’s legal team asks to put that number into their contract, and then asks the obvious question. Which messages?

The 0.9% was measured on the twenty thousand messages the vote was trained on. It is the empirical risk. Every guarantee lesson 10a proved about the vote is about the empirical risk: the theorem drives it down exponentially, and says nothing whatsoever about the next message to arrive. You have met a hypothesis with a spotless empirical risk before. Lesson 1’s memorizer was right on every message it had seen and no better than a coin on every other.

So the manager rewrites the request. With probability at least 95%, the filter’s true risk on fresh mail must be at most 0.20. Not 0.9%. Just 0.20, but in writing, computed from the labelled messages alone, with nothing measured on mail you do not have.

The team’s first answer is more rounds, since more rounds kept pushing the empirical risk down. But every round also makes the vote a member of a bigger class, and lesson 4 taught you that the bigger the class, the further its empirical risks can stray from its true risks. So what does a vote of \(T\) stumps cost you on fresh mail, and which \(T\) can you sign for? That is this lesson’s hunt.

## What you need from lesson 10a

Here is the toolkit from lesson 10a, one reminder per line.

- **Labels and sign.** Junk is \(+1\) and keep is \(-1\), so \(\mathcal Y = \{-1, +1\}\), and a hypothesis \(h\) is right on \((x, y)\) exactly when \(y\, h(x) = +1\). The sign function answers only \(+1\) (positive) or \(-1\) (everything else): \(\operatorname{sign}(u) = +1\) if \(u > 0\) and \(-1\) otherwise, so \(\operatorname{sign}(0) = -1\).
- **Decision stumps.** A stump \(h_{j, \theta, b}(x) = b \cdot \operatorname{sign}(x_j - \theta)\) looks at count \(j\), says \(b\) if the count exceeds the threshold \(\theta\), and says \(-b\) otherwise. With a single feature, on the line, the stumps are the functions \(x \mapsto b \cdot \operatorname{sign}(x - \theta)\).
- **Base class.** A simple class \(B\), such as the stumps, whose ERM (empirical risk minimization, from lesson 2: return a hypothesis with the smallest empirical risk), written \(\mathrm{ERM}_B\), is fast. ERM over stumps finds the stump with the smallest weighted risk.
- **AdaBoost.** It runs \(T\) rounds. In round \(t\) it holds a weight on every example, the weak learner returns a hypothesis \(h_t \in B\) with weighted risk \(\varepsilon_t\) (the total weight of the examples \(h_t\) gets wrong), and \(h_t\) receives the vote \(w_t = \tfrac12\log(1/\varepsilon_t - 1)\). The output is \(h_s(x) = \operatorname{sign}\big(\sum_{t=1}^T w_t h_t(x)\big)\): each stump votes junk or keep with its own strength, and the side with more total strength wins.
- **The ten messages.** Two counts, “free” and “click”, and junk exactly when they add up to at least 4. Three rounds picked “junk if free \(> 0.5\)”, “junk if free \(> 2.5\)” and “junk if click \(> 3.5\)”, with votes 0.693, 0.973 and 1.282, and the vote got all ten right. Message 6, \((1, 4)\), got the answers \(+1, -1, +1\) and the score \(0.693 - 0.973 + 1.282 = 1.003 > 0\): junk.
- **The theorem.** If every round has an edge of \(\gamma\), meaning \(\varepsilon_t \le \tfrac12 - \gamma\), then the empirical risk of the vote, \(L_S(h_s)\), the fraction of the training examples it gets wrong, satisfies \(L_S(h_s) \le Z_T \le e^{-2\gamma^2 T}\). Here \(Z_T = \prod_{t=1}^T 2\sqrt{\varepsilon_t(1 - \varepsilon_t)}\) is the product of the rounds’ shrink factors (the capital pi multiplies over the rounds), and \(e^{-2\gamma^2 T}\) is \(e^{-2\gamma^2}\) multiplied by itself \(T\) times. \(Z_T\) falls every round; the empirical risk need not, and on the ten messages it sat at 0.2 after round 2.
- **The lab’s mail.** Ten word counts, each from 0 to 5: “free”, “win”, “click”, “offer”, “cash”, “now”, “prize”, “urgent”, “meeting” and “lunch”. A message is junk exactly when the first eight add up to at least 21. You train on 20,000 labelled messages, and the lab keeps 50,000 fresh ones aside. The best single stump is right 63% of the time.

Here is the three-stump vote once more, on message 6 and on message 4, \((1, 1)\). The list `psi` holds the three stumps’ answers, and `s` is the score:

```python
import math

w = [math.log(2), math.log(7) / 2, math.log(13) / 2]
hs = [lambda x: 1 if x[0] > 0.5 else -1,
      lambda x: 1 if x[0] > 2.5 else -1,
      lambda x: 1 if x[1] > 3.5 else -1]
for x in [(1, 4), (1, 1)]:
    psi = [ht(x) for ht in hs]
    s = sum(wt * p for wt, p in zip(w, psi))
    print(x, psi, round(s, 3), 1 if s > 0 else -1)
```

It prints `(1, 4) [1, -1, 1] 1.003 1` and `(1, 1) [1, -1, -1] -1.562 -1`: junk for message 6, keep for message 4, both correct.

## What the vote can express

AdaBoost does not return a stump. It returns a vote of \(T\) stumps, like the three-stump vote that got all ten messages right, and that vote lives in a bigger class. For a base class \(B\), define
\[ L(B, T) = \left\{ x \mapsto \operatorname{sign}\!\left( \sum_{t=1}^T w_t h_t(x) \right) : w \in \mathbb R^T,\ h_1, \dots, h_T \in B \right\}. \]
The braces collect every function of that shape such that the votes \(w = (w_1, \dots, w_T)\) are any \(T\) real numbers (\(w \in \mathbb R^T\)) and the \(h_t\) are any \(T\) members of \(B\). In plain words: **every vote of \(T\) base hypotheses, with any votes you like.** The three-stump vote is one member of \(L(B, 3)\).

Read it in two stages. First, the \(T\) base hypotheses turn a message \(x\) into a vector of \(T\) labels, \(\psi(x) = (h_1(x), \dots, h_T(x)) \in \{\pm 1\}^T\) (psi; \(\{\pm 1\}^T\) is the set of lists of \(T\) entries, each \(\pm 1\)). Message 6 became \(\psi = (1, -1, 1)\) in the snippet above. Then a homogeneous halfspace from lesson 9a, the hypothesis \(\operatorname{sign}(\langle w, \psi \rangle)\) that multiplies entries pairwise, adds, and takes the sign, labels that vector; \(0.693 - 0.973 + 1.282 = 1.003\) was exactly that. A vote of stumps is a halfspace on top of stumps.

How much richer is the vote than one stump? On the line, take the stumps \(B = \{x \mapsto b \cdot \operatorname{sign}(x - \theta)\}\), and compare \(L(B, T)\) with **piecewise constant** hypotheses: cut the line at \(\theta_1 < \theta_2 < \dots < \theta_{r-1}\), set \(\theta_0 = -\infty\) and \(\theta_r = +\infty\), and give each piece \((\theta_{i-1}, \theta_i]\) its own label \(\alpha_i \in \{\pm 1\}\):
\[ g_r(x) = \sum_{i=1}^r \alpha_i\, \mathbb 1[x \in (\theta_{i-1}, \theta_i]]. \]
Here \(\alpha_i\) (alpha) is the label of piece \(i\), and \(\mathbb 1[x \in (\theta_{i-1}, \theta_i]]\) is 1 exactly when \(x\) lies in piece \(i\), just above one cut and up to and including the next. Since \(x\) lies in exactly one piece, all terms but one are 0. In plain words: **find the piece that \(x\) falls in and return its label.** With cuts at 2 and 4 and labels \(-1, +1, -1\), the point 3 lies in \((2, 4]\) and gets \(+1\):

```python
import math

def g(x, thetas, alphas):
    th = [-math.inf] + thetas + [math.inf]
    return sum(a for i, a in enumerate(alphas, 1)
               if th[i - 1] < x <= th[i])

print([g(x, [2, 4], [-1, 1, -1]) for x in (1, 2, 3, 4, 5)])
```

It prints `[-1, -1, 1, 1, -1]`: each cut belongs to the piece on its left.

Call \(G_r\) the class of such hypotheses with at most \(r\) pieces. The main case is the alternating one, \(\alpha_i = (-1)^i\), where every cut flips the label: \((-1)^i\) is \(-1\) for odd \(i\) and \(+1\) for even \(i\), exactly the labels above. Here is how to write it as a vote of \(T\) stumps:
\[ h(x) = \operatorname{sign}\!\left( \sum_{t=1}^T w_t \operatorname{sign}(x - \theta_{t-1}) \right), \qquad w_1 = \tfrac{(-1)^T}{2}, \quad w_t = (-1)^t \text{ for } t \ge 2. \]
Stump \(t\) switches from \(-1\) to \(+1\) at cut \(\theta_{t-1}\), and the votes alternate in sign, so each cut a point passes swings the score by \(\pm 2\). In plain words: **a vote of stumps, one per cut, with alternating votes, plus a half-vote offset that makes the score’s sign alternate.** For \(T = 4\) the votes are \(\tfrac12, 1, -1, 1\), and the scores on the four pieces are \(-0.5, 1.5, -0.5, 1.5\), signs \(-, +, -, +\). The first term uses \(\theta_0 = -\infty\), so it is a constant \(+1\) that acts like an offset. On the whole line no stump is constant, but on any finite set of points (and for word counts, which are never negative) a stump with its threshold below every point is constant, which is all we will need. Problem 1 asks you to prove that \(h = g_T\); before you do, here is a check for \(T = 1\) to 7, with cuts at 2, 4, 6, and so on. The list `w` is the votes, `th` puts a threshold below every point in front of the cuts, and `piece` finds the piece index \(i\) of \(g_T\):

```python
def sign(u):
    return 1 if u > 0 else -1

def staircase(T, thetas):
    w = [(-1) ** T / 2] + [(-1) ** t for t in range(2, T + 1)]
    th = [-1] + thetas
    return lambda x: sign(sum(w[t] * sign(x - th[t])
                              for t in range(T)))

for T in range(1, 8):
    thetas = [2 * t for t in range(1, T)]
    h = staircase(T, thetas)
    piece = lambda x: 1 + sum(x > th for th in thetas)
    assert all(h(x) == (-1) ** piece(x) for x in range(2 * T))
print("h equals g for T = 1, ..., 7")
```

It prints `h equals g for T = 1, ..., 7`. (The sign of \(w_1\) matters. With \(w_1 = +\tfrac12\) for every \(T\), the vote is right for even \(T\) and says \(+1\) everywhere for odd \(T\). Try it.)

[[diagram:staircase]]

So \(T\) stumps can draw a staircase with \(T\) steps. That has a consequence for the VC dimension (lesson 6a: the size of the largest set of points a class can shatter, that is, label in every possible way). Take any \(T\) distinct points on the line and any of their \(2^T\) labellings. Cut between neighbours whose labels differ: that gives at most \(T\) pieces whose labels alternate. If the first piece is \(-1\), the staircase with that many stumps draws it; if it is \(+1\), flip the sign of every vote; and any stumps left over get vote 0. So on those points every labelling is a vote of \(T\) stumps. Hence
\[ \operatorname{VCdim}(L(B, T)) \ge T. \]
\(\operatorname{VCdim}(\cdot)\) is the VC dimension of the class in the brackets. In plain words: **votes of \(T\) stumps on the line can label some \(T\) points in every possible way.** Three stumps can label any three points on the line in all \(2^3 = 8\) ways. The exact value turns out to depend on the parity of \(T\): it is \(T + 1\) when \(T\) is odd and \(T\) when \(T\) is even. You will see why in a moment, and it is a nice puzzle on its own. What matters is the trend: every extra round buys the vote more expressive power.

That power is a double-edged sword, and lesson 5 named both edges. A richer class lowers the **approximation error**, the true risk of the best hypothesis in the class. A richer class also raises the **estimation error**, the gap between that best and what the learner finds from finitely many examples. AdaBoost gives you a single dial for this tradeoff: the number of rounds \(T\). The empirical risk falls as \(T\) grows. How fast does the class grow?

## How big the class gets

Each round adds one base hypothesis, with its own \(d\) degrees of freedom, plus one vote, so expect about \(T(d + 1)\), a little more for the logarithm.

**Lemma (the VC dimension of a vote).** Let \(B\) be a base class with \(d = \operatorname{VCdim}(B)\) (a VC dimension, not the number of words), and assume \(T \ge 3\) and \(d \ge 3\). Then
\[ \operatorname{VCdim}(L(B, T)) \le T(d + 1)\big(3 \log(T(d + 1)) + 2\big). \]
Inside the big brackets, \(T(d + 1)\) is the number of rounds times one more than the base class’s VC dimension; the logarithm of that number, times 3, plus 2, multiplies it. In plain words: **up to the logarithm, the VC dimension of the vote is at most \(T\) times that of the base class.** With \(d = 3\) and \(T = 10\), \(T(d+1) = 40\) and the bound is \(40 \cdot (3 \log 40 + 2) \approx 523\); the snippet after the proof prints it.

We need two tools from earlier lessons. **Sauer’s lemma** (lesson 6b): if a class has VC dimension \(d\), then on any set of \(m > d + 1\) points it induces at most \((em/d)^d\) different labellings. That is \(e\) times \(m/d\), raised to the power \(d\): for \(d = 3\) and \(m = 20\) points it allows at most about \(18.1^3 \approx 5{,}950\) labellings, far fewer than the \(2^{20} \approx 1{,}000{,}000\) possible ones. And **homogeneous halfspaces** in \(\mathbb R^T\) have VC dimension \(T\) (lesson 9a), so on \(m > T + 1\) points of \(\mathbb R^T\) they induce at most \((em/T)^T\) labellings.

*Proof.* Let \(C = \{x_1, \dots, x_m\}\) be a set that \(L(B, T)\) shatters. We count the labellings of \(C\) that \(L(B, T)\) can produce, and compare with \(2^m\).

**A small case first.** The right side of the lemma is larger than both \(d + 1\) and \(T + 1\): with \(d, T \ge 3\) we have \(T(d+1) \ge 3(d + 1)\) and \(T(d+1) \ge 4T\), and the factor \(3\log(T(d+1)) + 2\) is at least \(3 \log 12 + 2 > 1\). So if \(m \le \max(d, T) + 1\), the lemma already holds. Assume from now on that \(m > d + 1\) and \(m > T + 1\), so both versions of Sauer’s lemma apply.

**Count the labellings.** A hypothesis in \(L(B, T)\) labels \(C\) in two stages. First it picks \(h_1, \dots, h_T \in B\). What matters for \(C\) is only how each \(h_t\) labels \(C\), and by Sauer’s lemma \(B\) has at most \((em/d)^d\) labellings of \(C\). Choosing \(T\) of them, in order, can be done in at most \((em/d)^{dT}\) ways. Second, once the \(T\) labellings are fixed, the vectors \(\psi(x_1), \dots, \psi(x_m) \in \mathbb R^T\) are fixed, and the halfspace with weights \(w\) labels those \(m\) vectors in at most \((em/T)^T\) ways. Every labelling of \(C\) by \(L(B, T)\) arises from some choice in stage one followed by some choice in stage two, so the number of labellings is at most
\[ (em/d)^{dT}(em/T)^T. \]
In plain words: **the choices of base labellings, times the choices of halfspace labelling.** The exponent \(dT\) is \(d\) taken \(T\) times, one Sauer count per round. With \(d = 3\) and \(T = 10\), this count is still at least \(2^m\) at \(m = 307\) points and falls below it at 308; the `log_count` snippet after the proof finds that crossover.

**Simplify.** Since \(d \ge 3 > e\) we have \(e/d < 1\), so \((em/d)^{dT} < m^{dT}\). Likewise \(T \ge 3\) gives \((em/T)^T < m^T\). So the count is less than \(m^{(d+1)T}\).

**Use the shattering.** \(C\) is shattered, so all \(2^m\) labellings occur, and
\[ 2^m < m^{(d+1)T}. \]
In plain words: **every possible labelling must be one the class can produce.** The left side doubles with each extra point; the right side grows only like a fixed power of \(m\), so for large \(m\) this must fail. Taking logarithms, \(m \log 2 < (d + 1)T \log m\), that is,
\[ m < a \log m, \qquad a = \frac{(d + 1)T}{\log 2}. \]
With \(d = 3\) and \(T = 10\), \(a = 40/0.693 \approx 57.7\), and \(m = 307\) passes, since \(57.7 \cdot \log 307 \approx 330.5\).

**Turn \(m < a \log m\) into a bound on \(m\).** We use a small fact about logarithms: for \(a > 0\), if \(x \ge 2a \log a\) then \(x \ge a \log x\). Here is why. If \(a \le \sqrt e\), the function \(x - a \log x\) has its minimum at \(x = a\) (its derivative \(1 - a/x\) changes sign there), where it equals \(a(1 - \log a) \ge 0\); so \(x \ge a \log x\) for every \(x > 0\). If \(a > \sqrt e\), then \(2 \log a > 1\), so \(2a \log a > a\), and \(x - a\log x\) is increasing for \(x > a\). At \(x = 2a\log a\) it equals \(a \log a - a \log(2 \log a) = a \log\!\big(a / (2 \log a)\big)\), which is nonnegative because \(a \ge 2 \log a\) for every \(a > 0\) (the function \(a - 2\log a\) has derivative \(1 - 2/a\), so its minimum is at \(a = 2\), where it is \(2 - 2\log 2 \approx 0.61 > 0\)). So for \(x \ge 2a\log a\) the function is at least its value there, which is at least 0.

Read it backwards. Since \(m < a \log m\), the conclusion fails for \(x = m\), so the hypothesis must fail too: \(m < 2a \log a\). Now unpack \(a\):
\[ 2a \log a = \frac{2}{\log 2}(d+1)T \left( \log((d+1)T) + \log\frac{1}{\log 2} \right). \]
(The log of the fraction \(a\) splits into the log of its top plus the log of \(1/\log 2\).) For \(d = 3\), \(T = 10\) this is about 468.
With \(2/\log 2 \approx 2.885 \le 3\) and \(\log(1/\log 2) \approx 0.367\), this is at most \((d + 1)T\big(2.885 \log((d+1)T) + 1.06\big) \le (d + 1)T\big(3 \log((d+1)T) + 2\big)\). So every shattered set has fewer elements than the bound. \(\blacksquare\)

Here are the numbers for \(d = 3\) and \(T = 10\): the largest \(m\) for which the count still allows \(2^m\) labellings, the intermediate bound \(2a\log a\), and the lemma’s bound. Then the lemma’s bound at \(T = 40\) and 100. The function `log_count` is the logarithm of \((em/d)^{dT}(em/T)^T\), so comparing it with `m * math.log(2)` is the test \(2^m \le\) count.

```python
import math

def log_count(m, d, T):
    return d * T * math.log(math.e * m / d) + T * math.log(
        math.e * m / T)

d, T = 3, 10
m = max(m for m in range(T + 2, 2000)
        if m * math.log(2) <= log_count(m, d, T))
a = (d + 1) * T / math.log(2)
lemma = T * (d + 1) * (3 * math.log(T * (d + 1)) + 2)
print(m, round(2 * a * math.log(a)), round(lemma))
for T in (40, 100):
    print(T, round(T * 4 * (3 * math.log(T * 4) + 2)))
```

It prints `307 468 523`, then `40 2756` and `100 7990`. Each step of the proof loosens the bound a little, and the shape survives: roughly linear in \(T\).

Now the parity puzzle from the last section. Put \(T\) stumps on the line, all with genuine thresholds between \(T + 1\) points, and fold each stump’s sign \(b\) into its vote, so every stump says \(+1\) above its threshold. Left of every threshold, every stump says \(-1\) and the score is \(-\sum_t w_t\); right of every threshold it is \(+\sum_t w_t\). So the two ends always get opposite signs (or both \(-1\), if the votes sum to zero). With two stumps and three points \(x_1 < x_2 < x_3\), the labelling \(+, -, +\) needs two sign changes, so both thresholds must sit between the points, and then it needs both ends positive, which is impossible; and three points are not shattered. With an odd number of stumps the two ends of an alternating labelling have opposite signs anyway, which is why the parity of \(T\) decides the exact VC dimension.

Now play with both ideas yourself. In the lab, you put a few messages on a line, give them labels, and ask whether some vote of \(T\) stumps can draw that labelling. Before you start, a prediction: of the eight ways to label three points, how many can a vote of two stumps draw? Then switch tabs and watch the lemma’s count of labellings race against \(2^m\) as you add rounds.

[[lab]]

## A promise about fresh mail

The empirical risk falls with \(T\). The class grows with \(T\). To turn that into a promise about fresh mail, we need a bound with real numbers in it, and our mail room has a feature that makes one easy: there are only finitely many possible messages. Ten counts, each from 0 to 5, give \(|\mathcal X| = 6^{10} = 60{,}466{,}176\) messages.

On a finite domain, every class is finite once you count functions rather than formulas. Two hypotheses that agree on every possible message have the same true risk and the same empirical risk, so only the distinct functions matter. How many functions does \(L(B, T)\) contain? Count as in the proof above, with \(C = \mathcal X\):

- A stump on one count from 0 to 5 can cut in 5 genuine places, with 2 signs, and there are 10 counts. Add the two constant stumps (a threshold below 0 or above 5), and \(B\) contains \(2 \cdot 10 \cdot 5 + 2 = 102\) distinct functions on \(\mathcal X\).
- Choosing \(T\) of them, in order: at most \(102^T\) ways.
- For each choice, a homogeneous halfspace labels the \(|\mathcal X|\) vectors \(\psi(x)\) in at most \((e|\mathcal X|/T)^T\) ways, by Sauer’s lemma.

So \(|L(B, T)| \le 102^T (e|\mathcal X|/T)^T\) as a set of functions on \(\mathcal X\). The bars count again: \(|L(B, T)|\) is how many distinct functions the class holds, and \(|\mathcal X|\) how many messages there are. In plain words: **at most 102 choices per round for the stumps, times Sauer’s count for the halfspace on top.** For \(T = 1\) the logarithm of the bound is \(\log 102 + 1 + \log 6^{10} \approx 4.6 + 1 + 17.9 = 23.5\); the `log_H` line of the snippet below computes exactly this.

Now bring in lesson 4. For a finite class \(\mathcal H\) and \(m\) i.i.d. examples, with probability at least \(1 - \delta\), every \(h \in \mathcal H\) satisfies
\[ |L_{\mathcal D}(h) - L_S(h)| \le \sqrt{\frac{\log(2|\mathcal H|/\delta)}{2m}}. \]
Careful with the bars: around \(L_{\mathcal D}(h) - L_S(h)\), a number, they mean absolute value, the size of the difference whichever way it points; around \(\mathcal H\), a set, they count its members. \(L_{\mathcal D}(h)\) is the true risk, shortened from \(L_{(\mathcal D, f)}(h)\). Inside the root, the log of the class size, adjusted for confidence, is divided by twice the sample size. In plain words: **on every hypothesis of the class at once, the empirical risk is within this distance of the true risk.** With the 102 stumps alone, \(m\) = 20,000 and \(\delta = 0.05\):

```python
import math

m, delta, size = 20_000, 0.05, 102
print(round(math.sqrt(math.log(2 * size / delta) / (2 * m)), 4))
```

It prints `0.0144`: a single stump’s empirical risk is within about 1.4 points of its true risk. That was uniform convergence for finite classes, from Hoeffding’s inequality (lesson 4: an average of independent bounded draws rarely strays far from its mean) and the union bound. One more detail. We will try every \(T\) from 1 to 100 and keep the best, so we need the promise to hold for all hundred classes at once. Give each class confidence \(\delta/100\); by the union bound, all hundred promises hold together with probability at least \(1 - \delta\). That is the same trick lesson 7a used to spread confidence over a sequence of classes. So, with probability at least \(1 - \delta\), for every \(T \le 100\) and every vote \(h\) of \(T\) stumps,
\[ L_{\mathcal D}(h) \le L_S(h) + \operatorname{gap}(T), \qquad \operatorname{gap}(T) = \sqrt{\frac{\log|L(B, T)| + \log(200/\delta)}{2m}}. \]
It is the previous formula with \(|\mathcal H|\) replaced by \(|L(B, T)|\), its log split in two, and \(\delta\) by \(\delta/100\), which turns \(2/\delta\) into \(200/\delta\); only the upward direction is kept. In plain words: **the true risk of any vote of \(T\) stumps is at most its empirical risk plus a gap that grows with \(T\).** In the snippet below, `log_H` is the log of the count and `log_terms` the top of the fraction. We call \(L_S(h) + \operatorname{gap}(T)\) the **certified risk**: a number you compute from the training sequence alone, which the true risk exceeds only with probability at most \(\delta\).

With \(m\) = 20,000 and \(\delta = 0.05\) (here `k` is the 10 words and `K` the cap of 5):

```python
import math

m, delta, k, K = 20_000, 0.05, 10, 5
log_B = math.log(2 * k * K + 2)
log_X = k * math.log(K + 1)

def gap(T):
    log_H = T * (log_B + 1 + log_X - math.log(T))
    log_terms = log_H + math.log(2 * 100 / delta)
    return math.sqrt(log_terms / (2 * m))

for T in (1, 10, 20, 40, 60, 100):
    print(T, round(gap(T), 3))
```

It prints

```
1 0.028
10 0.074
20 0.102
40 0.142
60 0.171
100 0.218
```

The gap grows like \(\sqrt{T}\), give or take a logarithm. The empirical risk shrinks with \(T\). Their sum must have a sweet spot.

Here is one run of AdaBoost with stumps on 20,000 messages from the lab’s mail, where junk means the first eight counts add up to at least 21. The last column is the error on 50,000 fresh messages, which a real mail room would not have; the lab can afford it.

| rounds \(T\) | \(L_S(h_s)\) | bound \(Z_T\) | gap\((T)\) | certified risk | fresh error |
|---|---|---|---|---|---|
| 1 | 0.368 | 0.965 | 0.028 | 0.397 | 0.377 |
| 10 | 0.158 | 0.599 | 0.074 | 0.233 | 0.155 |
| 20 | 0.106 | 0.446 | 0.102 | 0.208 | 0.103 |
| 30 | 0.067 | 0.349 | 0.124 | 0.191 | 0.069 |
| 40 | 0.033 | 0.290 | 0.142 | 0.175 | 0.032 |
| 60 | 0.023 | 0.233 | 0.171 | 0.195 | 0.024 |
| 80 | 0.006 | 0.196 | 0.196 | 0.203 | 0.006 |
| 100 | 0.009 | 0.173 | 0.218 | 0.227 | 0.009 |

[[diagram:tradeoff]]

Read it column by column. The empirical risk falls fast and then crawls; it even goes up between 80 and 100 rounds, the same wobble lesson 10a saw on the ten messages. The bound \(Z_T\) sits above it every time, and falls every time, as lesson 10a’s theorem promised. The gap climbs steadily. The certified risk, their sum, falls to its lowest, about 0.170 at 39 rounds (0.175 at 40), and climbs again. In this run, any \(T\) from 23 to 64 certifies a true risk of at most 0.20.

And the fresh error? It keeps falling all the way to 80 rounds. The certificate holds for *every* distribution and *every* vote in the class, so it charges you for the class’s size even when, as here, the extra rounds happen not to hurt. It is what you can promise in writing, not a prediction.

## Fix the bug

Back to the inbox. Your manager wants the filter shipped with a written promise: with probability at least 95%, its true risk on fresh mail is at most 0.20. You have the twenty thousand labelled messages and the stump learner. The only dial is the number of rounds.

In the lab you set \(T\), from 1 to 100, and train. You see the empirical risk, the bound \(Z_T\) falling under it, the gap growing, and the certified risk. Too few rounds, and the vote can’t express the true labelling: the empirical risk alone is above 0.20. Too many, and the class is so large that the gap alone eats your margin. Find a \(T\) whose certified risk is at most 0.20, and ship.

[[lab-act2]]

## What people get wrong

**“More rounds means a better filter.”** More rounds means a lower empirical risk and a bigger class, whose VC dimension grows roughly linearly with \(T\). In the lab’s run, 100 rounds have a smaller empirical risk than 40, and a certified risk of 0.227 instead of 0.175.

**“The theorem says AdaBoost learns.”** The theorem from lesson 10a is about the empirical risk only. It says nothing about fresh mail. That part needs the size of \(L(B, T)\), and the two together make the tradeoff you just navigated.

**“The certified risk is what the filter will score on fresh mail.”** It is a ceiling, not a forecast. At 80 rounds the certified risk is 0.203 and the fresh error in the lab is 0.006. The certificate has to hold for every distribution and every vote in the class, so it charges you for the class’s size even when the extra rounds happen not to hurt.

## Faces: boosting in the wild

The same recipe, counts plus stumps plus boosting, made one of the best-known detectors in computer vision: the face detector of Viola and Jones. The task is to tell whether a small grey image, 24 by 24 pixels, shows a human face. So \(\mathcal X\) is the set of \(24 \times 24\) real matrices of grey levels, and the goal is a hypothesis \(h: \mathcal X \to \{\pm 1\}\).

Each base hypothesis has the form \(h(x) = f(g(x))\), where \(f\) is a decision stump on a single number and \(g\) turns the image into that number. Every \(g\) is described by two things:

- an axis-aligned rectangle \(R\) inside the image, and
- a type, one of four masks, A to D.

To compute \(g\), stretch the mask to fill \(R\), add up the pixel values in each of the mask’s regions, and take a difference. Types A and B split the rectangle into two equal halves, side by side or one above the other, and subtract one sum from the other. Type C splits it into three strips and subtracts the two outer sums from the centre one. Type D splits it into four quarters and takes the difference between the diagonal pairs.

[[diagram:haar]]

How many functions \(g\) are there? A rectangle is fixed by its two horizontal and two vertical edges, so there are at most \(24^4\) rectangles and \(4 \cdot 24^4\) functions. Counting more carefully, the edges sit on 25 grid lines in each direction:

```python
rectangles = (25 * 24 // 2) ** 2
print(rectangles, 24 ** 4, 4 * rectangles)
```

It prints `90000 331776 360000`. So the weak learner is the stump learner of lesson 10a, run over a few hundred thousand features: compute every \(g\) on every training image once, then find the best stump on the weighted images in each round. The first step is made fast by a preprocessing trick, the **integral image**, which lets every rectangle sum be computed in constant time. Problem 2 asks you to invent it.

And what does AdaBoost pick? The first feature it chooses compares the band across the eyes with the band across the upper cheeks just below: the eye region is often darker than the cheeks. The second compares the eye regions with the bridge of the nose between them. Nobody told the learner that faces have eyes. It found the two cheapest rules of thumb that beat a coin, then the next one, on the faces the first two missed.

## Recap and the next bug

1. AdaBoost returns a member of \(L(B, T)\), a halfspace on top of \(T\) base hypotheses: the stumps turn a message into \(\psi(x) = (h_1(x), \dots, h_T(x))\), and the votes label that vector.
2. Votes of \(T\) stumps on the line draw staircases with \(T\) steps, so \(\operatorname{VCdim}(L(B, T)) \ge T\); the exact value is \(T + 1\) for odd \(T\) and \(T\) for even \(T\), because the two ends of the line always get opposite signs or both \(-1\).
3. In the other direction, \(\operatorname{VCdim}(L(B, T)) \le T(d + 1)\big(3\log(T(d + 1)) + 2\big)\): up to a logarithm, \(T\) times the base class’s VC dimension. The proof applies Sauer’s lemma twice, once to the stumps and once to the halfspace on top.
4. On the mail room’s finite domain, the same count bounds the number of distinct votes, and the finite-class bound with a union bound over \(T = 1, \dots, 100\) gives the **certified risk** \(L_S(h) + \operatorname{gap}(T)\), which the true risk exceeds with probability at most \(\delta\).
5. The number of rounds is a dial for the tradeoff between approximation and estimation error. On the lab’s mail, the empirical risk keeps falling while the certified risk is U-shaped, lowest at about 0.170 at 39 rounds, 0.175 at 40 and 0.227 at 100.

Your filter ships with its promise, and the inbox is quiet. Then the questions start. Why stumps, and not small trees two levels deep? Why these ten words, and not fifty? Why 40 rounds? The certificate you used covered your choice of \(T\), because you paid for all hundred values up front. But for every other knob you turned, you looked at the same twenty thousand labels, tried several settings, and kept whichever looked best on them. Each choice tuned on the same labels makes those labels look better than fresh mail will. Pay for every knob up front, and the certificate becomes too loose to say anything. So how do you choose the rounds, the depth and the features without fooling yourself? That is the next lesson: model selection and validation.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Put four messages on the line at counts 1, 2, 3 and 4, and take three stumps with thresholds 1.5, 2.5 and 3.5, each saying \(+1\) above its threshold, with votes \(1, -1, 1\). Compute the four scores and the labelling the vote draws. Then explain why no vote of three stumps can label five points \(+, -, +, -, +\), and why no vote of two stumps can draw your four-point labelling.
2. The certificate spread \(\delta = 0.05\) over 100 values of \(T\). Suppose you want the freedom to try every \(T\) from 1 to 1,000 instead, so each class gets \(\delta/1{,}000\). Compute \(\operatorname{gap}(40)\) both ways, to four decimal places, with \(m\) = 20,000. Does the passing window from the lab change? What does this tell you about the price of a union bound over many choices?
3. Your stump learner has an edge of \(\gamma = 0.05\) on every reweighting, and you have \(m\) = 20,000 examples. After how many rounds does lesson 10a’s theorem guarantee that the empirical risk is exactly 0? For that number of rounds, evaluate the bound on \(\operatorname{VCdim}(L(B, T))\) from this lesson with \(d = 3\), and compare it with \(m\). What does the comparison tell you about using a VC-dimension argument to promise anything about fresh mail at that point?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (staircases from stumps).** Prove that the vote \(h(x) = \operatorname{sign}\big(\sum_{t=1}^T w_t \operatorname{sign}(x - \theta_{t-1})\big)\), with \(\theta_0 = -\infty\), \(w_1 = (-1)^T/2\) and \(w_t = (-1)^t\) for \(t \ge 2\), equals the piecewise constant hypothesis with labels \(\alpha_i = (-1)^i\) on the pieces \((\theta_{i-1}, \theta_i]\).
- **Problem 2 (the integral image).** Let \(A\) be a \(24 \times 24\) matrix of grey levels. Its integral image \(I(A)\) is the matrix \(B\) with \(B_{i,j} = \sum_{i' \le i,\ j' \le j} A_{i',j'}\): entry \((i, j)\) adds up every grey level in rows 1 to \(i\) and columns 1 to \(j\), the rectangle above and to the left (for \(A\) with rows \(1, 2\) and \(3, 4\), \(B\) has rows \(1, 3\) and \(4, 10\)). Show that \(I(A)\) can be computed from \(A\) in time linear in the size of \(A\), and that, given \(I(A)\), every Viola and Jones feature can be computed in constant time, that is, in a time that does not depend on the size of its rectangle.
