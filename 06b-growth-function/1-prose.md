# The Growth Function

## Five million clicks, and no certificate

Your interval filter shipped with its certificate: 148 labels, true risk at most 5%, with confidence 95%. Then the product team put a junk button in every inbox. Labels used to cost a reviewer’s afternoon; now five million clicks arrive every week, for free.

Two things came with them. Clicks are careless: about one in twenty is wrong, a newsletter marked junk in a hurry or a scam kept by mistake. And marketing doubled its newsletters, which now take every score above 0.8. So the junk your users mean is the interval of scores from 0.55 to 0.8, and their clicks agree with it 95% of the time.

You run ERM for intervals on the clicks, the maximum-subarray scan from lesson 6a, and the interval it returns lands within a whisker of \([0.55, 0.8]\). Then you take it to the reviewer, and the review goes badly.

Your certificate came from the strip argument, and the strip argument needs every label to be right. One careless junk click on a newsletter at 0.97 stretches the tightest interval all the way up to 0.97, and with five million clicks no interval agrees with every label, so the strips pin nothing. You reach for the fundamental theorem instead: intervals have VC dimension 2, so ERM is an agnostic PAC learner for them. “Which arrow gives you that?” asks the reviewer. The arrow from a finite VC dimension to uniform convergence, the one lesson 6a took on trust. “Then prove it. And tell me how many clicks it takes to promise a true risk within 10 points of the best interval, with confidence 90%. The log holds five million a week.”

Is any number of clicks enough, for a class with infinitely many rules and labels that are sometimes wrong? And if so, does that number fit in a week? This lesson proves the missing arrow, and then turns its proof into a number of clicks.

## What you need from lesson 6a

A short reminder, one idea per line, each with the example lesson 6a used.

- A **threshold** \(h_t(x) = \mathbb 1[x > t]\) says junk (1) when the spam score \(x\) is above the cut-off \(t\) and keep (0) otherwise; the **indicator** \(\mathbb 1[\cdot]\) turns “yes” into 1 and “no” into 0. All thresholds together form the class \(\mathcal H_{\mathrm{thr}}\). An **interval** rule \(h_{a,b}\) says junk when \(a \le x \le b\), and **unions of intervals**, \(\mathcal H_{\cup}\), say junk inside any of finitely many bands.
- The **restriction** \(\mathcal H_C\) of a class \(\mathcal H\) to a finite set of points \(C\) is the set of distinct label vectors, the **labellings**, that the class produces on \(C\). Thresholds on the two scores \(\{0.3, 0.7\}\) produce \((0, 0)\), \((0, 1)\) and \((1, 1)\): three of the four, never \((1, 0)\).
- A class **shatters** \(C\) when it produces all \(2^{|C|}\) labellings, where \(|C|\) is how many points \(C\) has. Thresholds shatter \(\{0.3\}\) but not \(\{0.3, 0.7\}\).
- The **VC dimension** \(\mathrm{VCdim}(\mathcal H)\) is the size of the largest set the class shatters: 1 for thresholds, 2 for intervals, infinite for unions of intervals.
- The **true risk** \(L_{\mathcal D}(h)\) is the chance that \(h\) mislabels a fresh message drawn from \(\mathcal D\), the distribution of arriving mail; the **empirical risk** \(L_S(h)\) is the share of the training sequence \(S\), its \(m\) labelled examples, that \(h\) gets wrong. \(S \sim \mathcal D^m\) says the \(m\) examples are drawn i.i.d., independently and each from \(\mathcal D\). The **0-1 loss** charges 1 for a wrong label and 0 for a right one. ERM returns a rule \(h_S\) with the fewest training mistakes.
- A training sequence is **\(\varepsilon\)-representative** when every rule’s gap between training and true risk, \(|L_S(h) - L_{\mathcal D}(h)|\), is at most \(\varepsilon\) (epsilon). A class has the **uniform convergence property** when, for every \(\mathcal D\), at least \(m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) examples are \(\varepsilon\)-representative with probability at least \(1 - \delta\) (delta, the failure chance you tolerate). On 200 uniform scores, the worst gap over 101 thresholds was 0.035.
- A class is **agnostic PAC learnable** when, with no assumption that any rule is perfect, some learner gets within \(\varepsilon\) of the best rule in the class, \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\), with probability at least \(1 - \delta\).
- **The fundamental theorem of statistical learning** says six properties of a class are the same property: 1, uniform convergence; 2, any ERM rule is a successful agnostic PAC learner; 3, agnostic PAC learnable; 4, PAC learnable; 5, any ERM rule is a successful PAC learner; 6, a finite VC dimension. Lesson 6a proved every arrow but one: 6 implies 1.
- The arrow 1 implies 2 comes from lesson 4: on an \(\varepsilon/2\)-representative training sequence, ERM’s true risk is within \(\varepsilon\) of the best rule in the class.
- The **union bound**: the chance that at least one of several bad events happens is at most the sum of their chances.

The plan for 6 implies 1 has two parts. First, **Sauer’s lemma**: a class of VC dimension \(d\), however large, produces only polynomially many labellings of \(m\) points, roughly \(m^d\) instead of \(2^m\). Second, a generalization of lesson 4: uniform convergence holds whenever a class has few labellings in this sense, which we will call a small **effective size**.

## Counting labellings: the growth function

We need a name for “how many labellings of \(m\) points, at most”. For thresholds and two points, every pair of distinct scores gets the same three labellings we found on \(\{0.3, 0.7\}\), so the answer for \(m = 2\) is 3. For a class where the choice of points matters, we take the best choice, the set on which the class is most flexible.

**Definition (growth function).** The **growth function** of \(\mathcal H\) is \(\tau_{\mathcal H}: \mathbb N \to \mathbb N\),

\[ \tau_{\mathcal H}(m) = \max_{C \subset \mathcal X : |C| = m} |\mathcal H_C|. \]

From the inside out:

- \(|\mathcal H_C|\) is how many distinct labellings the class produces on the set \(C\).
- Under \(\max\), the condition \(C \subset \mathcal X : |C| = m\) reads “over every set \(C\) of domain points such that \(C\) has \(m\) points”, and \(\max\) keeps the largest count.
- \(\tau\) is the Greek letter tau, and \(\tau_{\mathcal H}: \mathbb N \to \mathbb N\) says the growth function takes a whole number of points (\(\mathbb N\), the natural numbers) and returns a whole number of labellings.

In plain words: **\(\tau_{\mathcal H}(m)\) is the largest number of distinct labellings that \(\mathcal H\) produces on any set of \(m\) points.** For thresholds, \(\tau(2) = 3\). In the snippet below, `tau` counts \(|\mathcal H_C|\) on one set, which on the line is already the maximum, for a reason given at the end of this paragraph. Always \(\tau_{\mathcal H}(m) \le 2^m\), and if \(m \le \mathrm{VCdim}(\mathcal H)\), then some set of size \(m\) is shattered, so \(\tau_{\mathcal H}(m) = 2^m\). The interesting question is what happens past the VC dimension. Let’s just count, for thresholds and intervals. On the line, only the order of the points matters, so the points \(0, 1, \dots, m-1\) are as good as any \(m\) distinct scores.

The snippet also computes one more count, which will turn out to be the answer. The **binomial coefficient** \(\binom{m}{i}\), read “\(m\) choose \(i\)”, is the number of ways to pick \(i\) of \(m\) points: from four messages you can pick two in \(\binom{4}{2} = 6\) ways, one in \(\binom{4}{1} = 4\) ways, none in \(\binom{4}{0} = 1\) way. The sum \(\sum_{i=0}^{d} \binom{m}{i}\) adds those counts for \(i = 0, 1, \dots, d\): the number of ways to pick at most \(d\) of the \(m\) points. With \(m = 4\) and \(d = 1\) it is \(1 + 4 = 5\). In the code, `comb(m, i)` is \(\binom{m}{i}\) and `sauer(m, d)` is the sum:

```python
from math import comb

def tau(H, m):
    C = list(range(m))
    return len({tuple(h(c) for c in C) for h in H})

def sauer(m, d):
    return sum(comb(m, i) for i in range(d + 1))

for m in range(1, 9):
    cuts = [c - 0.5 for c in range(m + 2)]
    thr = [lambda x, t=t: x > t for t in cuts]
    itv = [lambda x, a=a, b=b: a < x < b
           for a in cuts for b in cuts if a < b]
    print(m, tau(thr, m), sauer(m, 1),
          tau(itv, m), sauer(m, 2), 2 ** m)
```

It prints

```
1 2 2 2 2 2
2 3 3 4 4 4
3 4 4 7 7 8
4 5 5 11 11 16
5 6 6 16 16 32
6 7 7 22 22 64
7 8 8 29 29 128
8 9 9 37 37 256
```

Read the columns as: \(m\), \(\tau_{\mathrm{thr}}(m)\), \(\sum_{i \le 1} \binom{m}{i}\), \(\tau_{\mathrm{int}}(m)\), \(\sum_{i \le 2} \binom{m}{i}\), \(2^m\). Thresholds produce \(m + 1\) labellings (a run of keeps, then a run of junks, split at one of \(m + 1\) places). Intervals produce \(\binom{m}{2} + m + 1\) (one block of junk, chosen by its two ends, or a single point, or nothing). Both grow like a polynomial of degree equal to the VC dimension, while \(2^m\) runs away. And both match the binomial sums exactly. That is no coincidence.

[[diagram:growth]]

## Sauer’s lemma

**Sauer’s lemma.** Let \(\mathcal H\) be a class with \(\mathrm{VCdim}(\mathcal H) \le d < \infty\). Then for every \(m\),

\[ \tau_{\mathcal H}(m) \le \sum_{i=0}^{d} \binom{m}{i}. \]

The left side is the growth function; the right side is the count of ways to pick at most \(d\) of the \(m\) points, the `sauer` column of the table. In plain words: **a class that shatters no more than \(d\) points can label \(m\) points in no more ways than there are ways to choose at most \(d\) of them.** For intervals, \(d = 2\), and \(m = 8\): \(1 + 8 + 28 = 37\), the last row of the table, against \(2^8 = 256\).

In particular, if \(1 \le d \le m\), then \(\tau_{\mathcal H}(m) \le (em/d)^d\). This simpler bound is a power with fixed exponent \(d\), a polynomial in \(m\), with \(e \approx 2.718\) again; for \(m = 8\) and \(d = 2\) it is \((4e)^2 \approx 118\), looser than 37 but far below 256. The last snippet of this section checks it.

It is also called the Sauer-Shelah-Perles lemma: Sauer, Shelah with Perles, and Vapnik and Chervonenkis each found it independently.

The proof is a beautiful piece of counting, and the trick is to prove something stronger. Instead of counting labellings against a formula, count them against the shattered subsets. Back to thresholds on \(\{0.3, 0.7\}\): they make 3 labellings. Which parts of the set do they shatter? The **empty set**, written \(\emptyset\), the set with no points, counts as shattered, since its single labelling (the empty one) is produced by any hypothesis. Each single point is shattered. The pair is not. That is 3 shattered subsets, the same as the number of labellings. The stronger claim says the labellings can never outnumber the shattered subsets.

**Stronger claim.** For every class \(\mathcal H\) (non-empty) and every finite set \(C = \{c_1, \dots, c_m\}\),

\[ |\mathcal H_C| \le \left| \{ B \subseteq C : \mathcal H \text{ shatters } B \} \right|. \]

On the left, the number of labellings of \(C\). On the right, the braces collect every subset \(B\) of \(C\) (\(\subseteq\) allows \(B\) to be all of \(C\), or empty) such that the class shatters \(B\), and the outer bars count them. In plain words: **a class labels a set in at most as many ways as the set has shattered pieces.** Here is the count for thresholds on \(\{0.3, 0.7\}\), with one cut-off per gap as before:

```python
from itertools import combinations

C = [0.3, 0.7]
H = [lambda x, t=t: int(x > t) for t in (0.0, 0.5, 1.0)]
HC = {tuple(h(c) for c in C) for h in H}

def shatters(B):
    labels = {tuple(h(c) for c in B) for h in H}
    return len(labels) == 2 ** len(B)

subsets = [B for k in range(3) for B in combinations(C, k)]
print(len(HC), [B for B in subsets if shatters(B)])
```

It prints `3 [(), (0.3,), (0.7,)]`: three labellings, three shattered subsets, the empty one first.

*Why the stronger claim gives Sauer’s lemma.* If \(\mathrm{VCdim}(\mathcal H) \le d\), no subset of size above \(d\) is shattered, so the shattered subsets of \(C\) are among the subsets of size \(0, 1, \dots, d\), of which there are \(\sum_{i=0}^{d} \binom{m}{i}\). Apply this to the set \(C\) that achieves the maximum in \(\tau_{\mathcal H}(m)\).

*Proof of the stronger claim,* by induction on \(m\), for all classes at once.

*Base case, \(m = 1\).* Either every hypothesis gives \(c_1\) the same label, so \(|\mathcal H_C| = 1\) and the only shattered subset is \(\emptyset\); or both labels occur, so \(|\mathcal H_C| = 2\) and both \(\emptyset\) and \(\{c_1\}\) are shattered. Equality either way.

*Inductive step.* Assume the claim for every class and every set of fewer than \(m\) points. Fix \(\mathcal H\) and \(C = \{c_1, \dots, c_m\}\), and let \(C' = \{c_2, \dots, c_m\}\), the set without its first point. Split the labellings in \(\mathcal H_C\) by their **tail**, the labels on \(C'\):

\[ Y_0 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ or } (1, y_2, \dots, y_m) \in \mathcal H_C \}, \]

\[ Y_1 = \{ (y_2, \dots, y_m) : (0, y_2, \dots, y_m) \in \mathcal H_C \text{ and } (1, y_2, \dots, y_m) \in \mathcal H_C \}. \]

Both sets are built the same way. Each vector \((y_2, \dots, y_m)\) is a tail, the labels on every point but the first; the condition after the colon asks whether that tail appears in \(\mathcal H_C\) with first label 0, with first label 1, or with both. The only difference is the word: “or” in \(Y_0\), “and” in \(Y_1\). In plain words: **\(Y_0\) holds every tail that occurs, and \(Y_1\) the tails that occur with both labels on \(c_1\).** On \(\{0.3, 0.7\}\), with \(c_1 = 0.3\), the labellings are \((0, 0)\), \((0, 1)\) and \((1, 1)\). The tails are \((0)\) and \((1)\), so \(Y_0\) has both. Only the tail \((1)\) comes with both first labels, from \((0, 1)\) and \((1, 1)\), so \(Y_1 = \{(1)\}\).

```python
HC = {(0, 0), (0, 1), (1, 1)}
Y0 = {v[1:] for v in HC}
Y1 = {v[1:] for v in HC if (1 - v[0],) + v[1:] in HC}
print(sorted(Y0), sorted(Y1), len(Y0) + len(Y1))
```

It prints `[(0,), (1,)] [(1,)] 3`. In the code, `v[1:]` is the tail, and the test for `Y1` asks whether the vector with its first label flipped is also present. The sizes add up to \(|\mathcal H_C| = 3\), which is Step 1.

*Step 1: \(|\mathcal H_C| = |Y_0| + |Y_1|\).* Group the labellings in \(\mathcal H_C\) by tail. A tail in \(Y_0\) but not \(Y_1\) occurs with exactly one label on \(c_1\), so it accounts for one labelling. A tail in \(Y_1\) accounts for two. Since \(Y_1 \subseteq Y_0\), the total is \(|Y_0 \setminus Y_1| + 2|Y_1| = |Y_0| + |Y_1|\). Here \(Y_0 \setminus Y_1\), “\(Y_0\) minus \(Y_1\)”, is the set of tails in \(Y_0\) but not in \(Y_1\); in the example, \(\{(0)\}\), one labelling, plus twice the one tail in \(Y_1\), gives 3.

*Step 2: bound \(Y_0\).* \(Y_0\) is exactly the restriction of \(\mathcal H\) to \(C'\): the tails that occur are the labellings of \(C'\). By the induction hypothesis on \(C'\),

\[ |Y_0| = |\mathcal H_{C'}| \le |\{ B \subseteq C' : \mathcal H \text{ shatters } B \}| = |\{ B \subseteq C : c_1 \notin B,\ \mathcal H \text{ shatters } B \}|. \]

The notation is the stronger claim’s, applied to \(C'\); the last equality just renames subsets of \(C'\) as subsets of \(C\) that leave out \(c_1\), with \(\notin\) read “is not in”. In the example, \(C' = \{0.7\}\) has two shattered subsets, \(\emptyset\) and \(\{0.7\}\), against \(|Y_0| = 2\).

*Step 3: bound \(Y_1\) through a smaller class.* Let \(\mathcal H'\) be the hypotheses that have a **partner**: another hypothesis in \(\mathcal H\) that agrees with it on \(C'\) and disagrees on \(c_1\). In the example, the cut-off 0.5 labels \((0, 1)\) and the cut-off 0.0 labels \((1, 1)\): same tail, different first label, so each is the other’s partner. The cut-off 1.0 labels \((0, 0)\), and nothing labels \((1, 0)\), so it has no partner. In symbols,

\[ \mathcal H' = \{ h \in \mathcal H : \exists h' \in \mathcal H,\ h'(c_1) = 1 - h(c_1),\ h'(c_i) = h(c_i) \text{ for } i = 2, \dots, m \}. \]

From the inside out: \(h'(c_i) = h(c_i)\) for \(i = 2, \dots, m\) says \(h'\) copies \(h\) on every point of \(C'\); \(h'(c_1) = 1 - h(c_1)\) says it flips the label on \(c_1\), since \(1 - 0 = 1\) and \(1 - 1 = 0\); \(\exists h' \in \mathcal H\) reads “there exists some \(h'\) in the class” with both properties; and the braces keep every \(h\) for which such an \(h'\) exists. In plain words: **\(\mathcal H'\) is the set of rules that have a twin differing only at the first point.**

```python
C, ts = [0.3, 0.7], (0.0, 0.5, 1.0)
lab = {t: tuple(int(c > t) for c in C) for t in ts}

def has_partner(t):
    return any(lab[u][0] == 1 - lab[t][0] and
               lab[u][1:] == lab[t][1:] for u in ts)

H1 = [t for t in ts if has_partner(t)]
print(H1, {lab[t][1:] for t in H1})
```

It prints `[0.0, 0.5] {(1,)}`: two cut-offs have partners, and their tails make exactly \(Y_1\). The `any` is the \(\exists\).

A tail is in \(Y_1\) exactly when it is the tail of a hypothesis with a partner, so \(Y_1 = \mathcal H'_{C'}\). (If no hypothesis has a partner, \(Y_1\) is empty and there is nothing to bound.) Otherwise, by the induction hypothesis, now applied to the class \(\mathcal H'\) on \(C'\),

\[ |Y_1| = |\mathcal H'_{C'}| \le |\{ B \subseteq C' : \mathcal H' \text{ shatters } B \}|. \]

Same notation as Step 2, now for the partner class. In the example, \(\mathcal H'\) labels 0.7 only as junk, so it shatters only \(\emptyset\): one subset, against \(|Y_1| = 1\).

*Step 4: each set shattered by \(\mathcal H'\) grows by \(c_1\).* Suppose \(\mathcal H'\) shatters \(B \subseteq C'\). Take any labelling of \(B \cup \{c_1\}\), the set \(B\) with the point \(c_1\) added (\(\cup\) reads “together with”). Some \(g \in \mathcal H'\) produces its part on \(B\). The partner \(g'\) of \(g\) agrees with \(g\) on all of \(C'\), hence on \(B\), and gives \(c_1\) the other label; and \(g'\) is itself in \(\mathcal H'\), with \(g\) as its partner. One of \(g, g'\) gives \(c_1\) the label we want. So \(\mathcal H'\) shatters \(B \cup \{c_1\}\), and since \(\mathcal H' \subseteq \mathcal H\), so does \(\mathcal H\). The map \(B \mapsto B \cup \{c_1\}\) is one-to-one from subsets of \(C'\) to subsets of \(C\) that contain \(c_1\), so

\[ |Y_1| \le |\{ B \subseteq C : c_1 \in B,\ \mathcal H \text{ shatters } B \}|. \]

Here \(B \mapsto B \cup \{c_1\}\) is the rule that sends each \(B\) to that bigger set. In the example, \(\mathcal H'\) shatters \(\emptyset\), which grows into \(\{0.3\}\), and thresholds do shatter \(\{0.3\}\): one shattered subset containing \(c_1\), against \(|Y_1| = 1\).

*Step 5: add.* The shattered subsets of \(C\) split into those without \(c_1\) (Step 2) and those with it (Step 4), so

\[ |\mathcal H_C| = |Y_0| + |Y_1| \le |\{ B \subseteq C : \mathcal H \text{ shatters } B \}|. \qquad \blacksquare \]

In the example that is \(3 = 2 + 1 \le 2 + 1 = 3\).

[[diagram:sauer-split]]

Why does Step 3 switch to \(\mathcal H'\) instead of using \(\mathcal H\) again? Because \(\mathcal H\) itself may shatter sets of \(C'\) that it cannot extend by \(c_1\). The partner class keeps only the hypotheses that come in flip pairs at \(c_1\), and those can always extend.

The stronger claim says something about every set of labellings, not only those from tidy classes, so we can test it on random ones. Each trial picks a random set \(\mathcal H_C\) of labellings of five points and checks Step 1 and the claim:

```python
from itertools import combinations, product
import random

def shattered(HC, B):
    patterns = {tuple(v[i] for i in B) for v in HC}
    return len(patterns) == 2 ** len(B)

rng, cube = random.Random(1), list(product([0, 1], repeat=5))
for trial in range(2000):
    HC = set(rng.sample(cube, rng.randint(1, 32)))
    Y0 = {v[1:] for v in HC}
    Y1 = {v[1:] for v in HC if (1 - v[0],) + v[1:] in HC}
    n_sh = sum(shattered(HC, B) for k in range(6)
               for B in combinations(range(5), k))
    assert len(HC) == len(Y0) + len(Y1) <= n_sh
print(len(HC), len(Y0), len(Y1), n_sh)
```

The `Y0` and `Y1` lines are the same two lines as in the small example above, and `n_sh` counts the right side of the stronger claim. All 2,000 assertions hold, and it prints the last trial, `23 14 9 26`: 23 labellings, which split into 14 tails plus 9 doubled tails, against 26 shattered subsets.

**The polynomial bound.** It remains to show \(\sum_{i=0}^{d} \binom{m}{i} \le (em/d)^d\) when \(1 \le d \le m\). Since \(d/m \le 1\), raising it to a smaller power makes it bigger, so \((d/m)^d \le (d/m)^i\) for every \(i \le d\). Hence

\[ \left(\frac{d}{m}\right)^d \sum_{i=0}^{d} \binom{m}{i} \le \sum_{i=0}^{d} \binom{m}{i} \left(\frac{d}{m}\right)^i \le \sum_{i=0}^{m} \binom{m}{i} \left(\frac{d}{m}\right)^i = \left(1 + \frac{d}{m}\right)^m \le e^d. \]

Read it left to right, writing \(q = d/m\). The first expression is the Sauer sum shrunk by \(q^d\). The second gives each term its own, larger, factor \(q^i\). The third lets \(i\) run on to \(m\), which only adds non-negative terms. The **binomial theorem** says that sum is exactly \((1 + q)^m\): multiply out \(m\) copies of \(1 + q\), and choosing \(q\) from \(i\) of the brackets can be done in \(\binom{m}{i}\) ways. Finally \(1 + q \le e^q\), raised to the power \(m\), gives \(e^{qm} = e^d\). Multiplying both ends by \((m/d)^d\) gives the bound. In plain words: **shrinking the Sauer sum by \((d/m)^d\) leaves something no bigger than \(e^d\), so the sum is at most \((em/d)^d\).** With the interval numbers \(m = 8\), \(d = 2\), so \(q = 1/4\):

```python
from math import comb, e

m, d = 8, 2
q = d / m
left = q ** d * sum(comb(m, i) for i in range(d + 1))
mid = sum(comb(m, i) * q ** i for i in range(d + 1))
full = sum(comb(m, i) * q ** i for i in range(m + 1))
print(round(left, 4), mid, round(full, 4),
      round((1 + q) ** m, 4), round(e ** d, 4))
```

It prints `2.3125 4.75 5.9605 5.9605 7.3891`: each link of the chain in order, the third and fourth equal as the binomial theorem promises. Multiplying 7.39 back by \(4^2 = 16\) gives the 118 from before, against the true 37. A quick check over every \(1 \le d \le m < 200\), and a look at the sizes involved for \(d = 2\):

```python
from math import comb, e

def sauer(m, d):
    return sum(comb(m, i) for i in range(d + 1))

ratio = max(sauer(m, d) / (e * m / d) ** d
            for m in range(1, 200) for d in range(1, m + 1))
print(round(ratio, 4))
for m in (10, 100, 1000):
    poly = round((e * m / 2) ** 2)
    print(m, sauer(m, 2), poly, len(str(2 ** m)))
```

It prints `0.7358`, so the sum never exceeds the bound, and then `10 56 185 4`, `100 5051 18473 31`, `1000 500501 1847264 302`. At \(m = 1{,}000\), a class of VC dimension 2 produces at most 500,501 labellings, under the bound of about 1.8 million, while \(2^{1000}\) has 302 digits. That gap is the whole story.

Now it’s your turn to count labellings, and to watch them branch.

[[lab]]

## Uniform convergence from a small effective size

Now the second part of the plan. Lesson 4 proved uniform convergence for a finite class by a union bound over its \(|\mathcal H|\) hypotheses, paying \(\log |\mathcal H|\). The idea here is to pay only for the labellings, \(\log \tau_{\mathcal H}\). Throughout, \(\mathcal D\) is a distribution over examples \(z = (x, y) \in \mathcal X \times \{0, 1\}\), the loss is \(\ell(h, z) = \mathbb 1[h(x) \ne y]\), and we assume \(\mathcal X\) has at least \(2m\) points (true for the score line). The loss \(\ell\) (a script l) is the 0-1 loss as a formula: the indicator of a wrong label. With \(t^\star = 0.6\), the threshold \(h_{0.5}\) on the example \(z = (0.55, 0)\) says junk while the truth says keep, so \(\ell(h_{0.5}, z) = 1\); averaged over \(S\), these losses give \(L_S(h)\).

Two more symbols, both from lesson 2. \(\mathbb E[\cdot]\) is the **expected value**, the long-run average of a random number; with a subscript such as \(\mathbb E_{S \sim \mathcal D^m}\), it says which draw is being averaged over. And \(\sup_{h \in \mathcal H}\), the **supremum**, is the largest value over all rules in the class, or, when no single rule attains it, the value the rules creep up to; for our purposes, read it as “the worst case over the class”. In lesson 6a’s uniform-convergence check, the `max` over 101 thresholds was a stand-in for a supremum over all of them.

**Theorem (uniform convergence from a small growth function).** For every class \(\mathcal H\), every distribution \(\mathcal D\) and every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every \(h \in \mathcal H\) satisfies

\[ |L_{\mathcal D}(h) - L_S(h)| \le \frac{4 + 2\sqrt{\log \tau_{\mathcal H}(2m)}}{\delta \sqrt{2m}}. \]

From the inside out. \(\tau_{\mathcal H}(2m)\) is the growth function at twice the training size, the number of labellings that matters; its log, square-rooted, is the price of the class; adding 4 and doubling come from the proof; and the bottom, \(\delta \sqrt{2m}\), shrinks the whole thing as labels grow and inflates it as you demand more confidence. On the left is the gap for one rule, and “every \(h\)” makes it a bound on the worst gap. In plain words: **with probability at least \(1 - \delta\), no rule’s training score is off by more than a quantity that depends on the class only through how many ways it labels \(2m\) points.** For thresholds, \(\tau(2m) = 2m + 1\); with \(m = 200\) and \(\delta = 0.1\):

```python
from math import log, sqrt

m, delta = 200, 0.1
tau = 2 * m + 1
expected = (4 + 2 * sqrt(log(tau))) / sqrt(2 * m)
print(tau, round(expected, 4), round(expected / delta, 4))
```

It prints `401 0.4448 4.4483`. The middle number is the bound before dividing by \(\delta\), and the last is the theorem’s bound: 4.45, more than 1, so at 200 labels it says nothing, since a gap between two risks never exceeds 1. The bound shrinks like \(\sqrt{\log m / m}\), so it does bite eventually; the end of the next section finds where.

The proof bounds the expected worst gap, \(\mathbb E_{S \sim \mathcal D^m}\left[\sup_{h \in \mathcal H} |L_{\mathcal D}(h) - L_S(h)|\right]\), by \((4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/\sqrt{2m}\), the middle number above, and then converts expectation into probability. It is the deepest proof in the lesson, so let’s take it in seven steps, each saying what changes and why.

*Step 1: from expectation to probability.* Let \(Z = \sup_{h} |L_{\mathcal D}(h) - L_S(h)|\), a non-negative random variable. **Markov’s inequality** says that for a non-negative \(Z\) and any \(a > 0\), \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\). (Proof: \(Z \ge a \cdot \mathbb 1[Z \ge a]\) pointwise; take expectations.) With \(a = \mathbb E[Z]/\delta\), the probability that \(Z \ge \mathbb E[Z]/\delta\) is at most \(\delta\). So a bound on \(\mathbb E[Z]\), divided by \(\delta\), holds with probability at least \(1 - \delta\). (If \(\mathbb E[Z] = 0\), then \(Z = 0\) with probability 1 and there is nothing to prove.) It remains to bound \(\mathbb E[Z]\).

In Markov’s inequality, \(\mathbb E[Z]/a\) is the average measured in units of the level \(a\). In plain words: **a non-negative quantity can rarely be many times its own average.** If a gap averages 0.04, it can reach 0.4 at most a tenth of the time. Here is one rule’s gap, for the threshold \(h_{0.5}\) with \(t^\star = 0.6\), whose true risk is 0.1, over 20,000 training sequences of 30 uniform scores:

```python
import random

rng, m, delta = random.Random(0), 30, 0.1
Z = [abs(sum(0.5 < rng.random() <= 0.6 for _ in range(m)) / m
         - 0.1) for _ in range(20_000)]
EZ = sum(Z) / len(Z)
share = sum(z >= EZ / delta for z in Z) / len(Z)
print(round(EZ, 4), round(EZ / delta, 4), share)
```

It prints `0.0427 0.4273 0.0`. The gap averages 0.043, Markov allows it to reach ten times that, 0.43, at most 10% of the time, and in fact it never did. Markov is safe but generous, and that generosity is the \(1/\delta\) in the theorem.

*Step 2: a ghost sample.* The trouble with \(L_{\mathcal D}(h)\) is that it involves the whole distribution. Replace it with something that looks like \(L_S\). Imagine a second, independent training sequence \(S' = (z'_1, \dots, z'_m) \sim \mathcal D^m\), a **ghost sample** that nobody ever collects. Each \(\ell(h, z'_i)\) has mean \(L_{\mathcal D}(h)\), so by linearity of expectation \(L_{\mathcal D}(h) = \mathbb E_{S'}[L_{S'}(h)]\). Then

\[ |L_{\mathcal D}(h) - L_S(h)| = \left| \mathbb E_{S'}\left[ L_{S'}(h) - L_S(h) \right] \right| \le \mathbb E_{S'}\left| L_{S'}(h) - L_S(h) \right| \le \mathbb E_{S'}\left[ \sup_{h' \in \mathcal H} |L_{S'}(h') - L_S(h')| \right]. \]

Left to right: the gap we care about; the true risk rewritten as the average ghost score, \(\mathbb E_{S'}\) being the average over all the ghost samples you might draw; then “the absolute value of an average is at most the average of absolute values”; then the inside can only grow when replaced by its worst case over the class, the prime in \(h'\) only giving that worst-case rule its own name. The right side no longer depends on \(h\), so it also bounds the supremum over \(h\) of the left side. In plain words: **the gap between true and training risk is at most the average, over imaginary second samples, of the worst gap between two training scores.** For the threshold \(h_{0.5}\) (true risk 0.1) and one real sample of 30 scores:

```python
import random

rng, m = random.Random(0), 30
wrong = lambda x: 0.5 < x <= 0.6
L = lambda xs: sum(map(wrong, xs)) / m
S = [rng.random() for _ in range(m)]
ghosts = [L([rng.random() for _ in range(m)])
          for _ in range(20_000)]
mean = sum(ghosts) / len(ghosts)
spread = sum(abs(g - L(S)) for g in ghosts) / len(ghosts)
print(L(S), round(mean, 4), round(abs(mean - L(S)), 4),
      round(spread, 4))
```

It prints `0.1 0.1001 0.0001 0.0427`. The ghost scores average 0.1001, the true risk, as linearity promised; this real sample happens to have no gap at all; and the average absolute gap to a ghost, 0.043, is larger, as the first inequality allows.

Taking \(\mathbb E_S\) of both sides, and writing each score as an average of losses,

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

Nothing new here but the notation \(\sum_{i=1}^{m}\), “add up for \(i = 1, 2, \dots, m\)”: \(L_{S'}(h) - L_S(h)\) is the average, over positions \(i\), of the ghost’s loss minus the real loss at that position, and \(\mathbb E_{S, S'}\) averages over both samples.

*Step 3: random signs.* The \(2m\) examples \(z_1, \dots, z_m, z'_1, \dots, z'_m\) are i.i.d. So if we swap \(z_i\) with \(z'_i\) for some indices \(i\), the joint distribution of the \(2m\) examples does not change, and neither does the expectation of any function of them. Swapping at index \(i\) flips the sign of the \(i\)-th term \(\ell(h, z'_i) - \ell(h, z_i)\). So for every sign vector \(\sigma \in \{\pm 1\}^m\),

\[ \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right] = \mathbb E_{S, S'}\left[ \sup_{h} \left| \frac{1}{m} \sum_{i} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

Here \(\sigma\) (sigma) is a list of \(m\) signs, each \(+1\) or \(-1\), and \(\{\pm 1\}^m\) is the set of all \(2^m\) such lists; \(\sigma_i = -1\) means “swap the real and ghost example at position \(i\)”, which turns \(\ell(h, z'_i) - \ell(h, z_i)\) into its negative. In plain words: **because the real and ghost examples are interchangeable, swapping any of them leaves the average worst gap unchanged.** A world small enough to average exactly: scores 0.55, 0.65 or 0.9, equally likely, \(t^\star = 0.6\), two rules (\(t = 0.5\) and \(t = 0.7\)) and \(m = 2\), so there are four sign vectors:

```python
from itertools import product

pts, t_star, m = (0.55, 0.65, 0.9), 0.6, 2
H = [lambda x, t=t: (x > t) != (x > t_star) for t in (0.5, 0.7)]

def value(sigma):
    total = 0
    for z in product(pts, repeat=2 * m):
        S, G = z[:m], z[m:]
        total += max(abs(sum(s * (h(g) - h(x))
                             for s, x, g in zip(sigma, S, G))) / m
                     for h in H)
    return round(total / len(pts) ** (2 * m), 4)

print([value(s) for s in product((1, -1), repeat=m)])
```

It prints `[0.4938, 0.4938, 0.4938, 0.4938]`: the same value for every sign vector. Each `h` returns the loss itself, `max` is the supremum over the two rules, and the loop over all \(3^4\) equally likely draws is \(\mathbb E_{S, S'}\).

The right side has the same value for all \(2^m\) sign vectors, so it also equals their average. Draw \(\sigma\) uniformly, each \(\sigma_i\) an independent fair \(\pm 1\) coin. Averaging over the finitely many \(\sigma\) and swapping the order of the two averages (linearity of expectation),

\[ \mathbb E_S[Z] \le \mathbb E_{S, S'}\, \mathbb E_{\sigma}\left[ \sup_{h \in \mathcal H} \left| \frac{1}{m} \sum_{i=1}^{m} \sigma_i \left( \ell(h, z'_i) - \ell(h, z_i) \right) \right| \right]. \]

The only new piece is \(\mathbb E_\sigma\), the average over the fair coin flips, taken inside the average over the two samples; in the small world above it is the average of four equal numbers, 0.4938.

[[diagram:ghost-sample]]

*Step 4: the infinite class becomes finite.* Now fix \(S\) and \(S'\), and let \(C\) be the set of instances appearing in them, at most \(2m\) points. Each loss \(\ell(h, z_i) = \mathbb 1[h(x_i) \ne y_i]\) depends on \(h\) only through its labels on \(C\). So the expression inside the supremum depends on \(h\) only through its labelling in \(\mathcal H_C\), and the supremum over the infinite class is a maximum over at most \(|\mathcal H_C|\) distinct values. How large can \(|\mathcal H_C|\) be? If \(|C| < 2m\), add points of \(\mathcal X\) to reach a set \(C^+\) of exactly \(2m\) points; each labelling of \(C\) is the shortening of some labelling of \(C^+\), so \(|\mathcal H_C| \le |\mathcal H_{C^+}| \le \tau_{\mathcal H}(2m)\). This is where the growth function enters, and it is the heart of the matter: once \(S\) and \(S'\) are fixed, an infinite class acts like a finite one.

*Step 5: Hoeffding and the union bound.* For each labelling, pick one hypothesis \(h\) that produces it, and write

\[ \theta_h = \frac{1}{m} \sum_{i=1}^{m} \sigma_i c_i, \qquad c_i = \ell(h, z'_i) - \ell(h, z_i) \in \{-1, 0, 1\}. \]

Here \(c_i\) is the loss difference at position \(i\): 1 if \(h\) is wrong on the ghost example and right on the real one, \(-1\) the other way round, 0 if it does the same on both. \(\theta_h\) (theta) multiplies each difference by its random sign and averages. In plain words: **\(\theta_h\) is \(h\)’s ghost-minus-real gap after a fair coin has decided, position by position, which example counts as ghost.** If \(m = 4\), \(c = (1, 0, -1, 1)\) and the signs come up \((+1, -1, -1, +1)\), then \(\theta_h = (1 + 0 + 1 + 1)/4 = 0.75\).

With \(S, S'\) fixed, the \(c_i\) are fixed numbers and only the signs are random. The terms \(\sigma_i c_i\) are independent, each has mean 0, and each lies in \([-1, 1]\). **Hoeffding’s inequality** from lesson 4 says that an average of \(m\) independent variables, each confined to a range of width \(w\), is more than \(\rho\) (rho, a level you choose) from its mean with probability at most \(2\exp(-2m\rho^2/w^2)\), where \(\exp(u)\) is another way to write \(e^u\): the more variables and the narrower their range, the less likely a big miss. (Lesson 4 stated it for identically distributed variables, but its proof multiplies one bound per variable and never uses that they share a distribution, so it holds for independent ones.) Here the range \([-1, 1]\) has width \(w = 2\), and \(2\rho^2/4 = \rho^2/2\), so

\[ \mathbb P_\sigma\left[ |\theta_h| > \rho \right] \le 2 \exp\left( -\frac{m \rho^2}{2} \right). \]

A union bound over the at most \(|\mathcal H_C|\) chosen hypotheses gives

\[ \mathbb P_\sigma\left[ \max_{h} |\theta_h| > \rho \right] \le 2 |\mathcal H_C| \exp\left( -\frac{m \rho^2}{2} \right) \quad \text{for every } \rho > 0. \]

In both displays \(|\theta_h| > \rho\) is the event that the signed gap overshoots the level, and \(\mathbb P_\sigma\) says only the coins are random. The union bound pays once per labelling, the factor \(|\mathcal H_C|\), and \(\max_h\) is the largest over the chosen rules. In plain words: **one rule’s signed gap rarely exceeds \(\rho\), and the chance that any of the \(|\mathcal H_C|\) rules does is at most that many times larger.** For thresholds on 400 scores, \(|\mathcal H_C| = 401\), with \(m = 200\), and the worst-spread case where every \(c_i = 1\):

```python
import math
import random

rng, m, n_labellings = random.Random(0), 200, 401
c = [1] * m
theta = lambda: sum(rng.choice((-1, 1)) * ci for ci in c) / m
draws = [abs(theta()) for _ in range(20_000)]
for rho in (0.2, 0.35):
    one = 2 * math.exp(-m * rho ** 2 / 2)
    seen = sum(x > rho for x in draws) / len(draws)
    print(rho, seen, f"{one:.2g}", f"{n_labellings * one:.2g}")
```

It prints `0.2 0.0032 0.037 15`, then `0.35 0.0 9.6e-06 0.0038`. At \(\rho = 0.2\) the observed overshoot rate, 0.3%, sits under Hoeffding’s 3.7%, but the union bound over 401 labellings gives 15, which says nothing. At \(\rho = 0.35\) the union bound is down to 0.4%: paying for 401 labellings costs only a modestly higher level, because the exponential falls so fast.

*Step 6: from a tail bound to an expectation.* We need \(\mathbb E_\sigma[\max_h |\theta_h|]\). A **tail bound** says how rarely a quantity exceeds each level \(\rho\); the snippet above showed one falling from 15 to 0.004 as \(\rho\) rose from 0.2 to 0.35. If the chance of exceeding a level collapses past some point, the quantity cannot average much more than that point. Here is a small lemma that makes this precise.

**Tail lemma.** Let \(X \ge 0\) be a random variable, and suppose \(\mathbb P[X > \rho] \le 2b\, e^{-\rho^2/a^2}\) for all \(\rho \ge 0\), with \(a > 0\) and \(b \ge e\). Then \(\mathbb E[X] \le a(1 + \sqrt{\log b})\).

Matched against Step 5, \(a\) sets the scale of the fall and \(b\) is the number of labellings. In plain words: **if the tail falls off like a bell curve of width \(a\), inflated by a factor \(b\), the average is at most \(a\) times \(1 + \sqrt{\log b}\).** With \(a = 0.1\) and \(b = 401\), that is \(0.1 \times (1 + 2.45) \approx 0.345\).

*Proof.* For a non-negative variable, \(\mathbb E[X] = \int_0^\infty \mathbb P[X > \rho]\, d\rho\). The integral sign \(\int_0^\infty \dots d\rho\) is the area under the curve of \(\mathbb P[X > \rho]\) as \(\rho\) runs from 0 upward; think of it as adding the chance of passing each tiny level step. A die roll shows why it gives the average: it passes level 0 for sure, level 1 with chance \(5/6\), and so on, and \(1 + 5/6 + 4/6 + 3/6 + 2/6 + 1/6 = 3.5\), the average roll. Split at \(\rho_0 = a\sqrt{\log b}\). Below \(\rho_0\), bound the probability by 1, which contributes \(\rho_0\). Above \(\rho_0\), use the tail bound together with \(1 \le \rho/\rho_0\):

\[ \int_{\rho_0}^\infty 2b\, e^{-\rho^2/a^2}\, d\rho \le \frac{2b}{\rho_0} \int_{\rho_0}^\infty \rho\, e^{-\rho^2/a^2}\, d\rho = \frac{2b}{\rho_0} \cdot \frac{a^2}{2} e^{-\rho_0^2/a^2} = \frac{2b}{\rho_0} \cdot \frac{a^2}{2b} = \frac{a}{\sqrt{\log b}} \le a, \]

where the last step uses \(\log b \ge 1\). Adding the two parts gives \(a\sqrt{\log b} + a\). \(\blacksquare\)

The display reads left to right: the area under the tail bound past \(\rho_0\); the same with an extra factor \(\rho/\rho_0 \ge 1\), which makes the area exactly computable; the computed area, because \(-\tfrac{a^2}{2} e^{-\rho^2/a^2}\) has slope \(\rho\, e^{-\rho^2/a^2}\) and vanishes as \(\rho\) grows, so the area is its value at infinity, 0, minus its value at \(\rho_0\); and the simplification, using \(e^{-\rho_0^2/a^2} = e^{-\log b} = 1/b\). In plain words: **past \(\rho_0\) the tail is so thin that its whole area is at most \(a\).** Numerically, with \(a = 0.1\) and \(b = 401\), adding up thin slices of width 0.00001:

```python
from math import exp, log, sqrt

a, b, step = 0.1, 401, 1e-5
rho0 = a * sqrt(log(b))
tail = sum(2 * b * exp(-((rho0 + k * step) / a) ** 2) * step
           for k in range(100_000))
print(round(rho0, 4), round(tail, 4), round(a / sqrt(log(b)), 4),
      round(rho0 + a, 4))
```

It prints `0.2448 0.0381 0.0408 0.3448`: the split point, the area past it, the bound \(a/\sqrt{\log b}\) on that area, and the lemma’s bound \(\rho_0 + a\) on the whole average.

Apply it with \(a^2 = 2/m\) and \(b = \max(|\mathcal H_C|, e)\); the tail bound of Step 5 still holds with this larger \(b\). Then \(\sqrt{\log b} \le 1 + \sqrt{\log |\mathcal H_C|}\), and \(a = \sqrt{2/m} = 2/\sqrt{2m}\), so

\[ \mathbb E_\sigma\left[ \max_h |\theta_h| \right] \le \frac{2}{\sqrt{2m}} \left( 2 + \sqrt{\log |\mathcal H_C|} \right) \le \frac{4 + 2\sqrt{\log \tau_{\mathcal H}(2m)}}{\sqrt{2m}}. \]

This is the lemma’s \(a(1 + \sqrt{\log b})\) with the values plugged in, then Step 4’s \(|\mathcal H_C| \le \tau_{\mathcal H}(2m)\). With \(m = 200\) and 401 labellings it is the 0.4448 printed earlier, and the simulation below meets that number again as `bound`.

*Step 7: put it together.* The bound of Step 6 holds for every fixed \(S, S'\), so it survives the outer expectation in Step 3: \(\mathbb E_S[Z] \le (4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/\sqrt{2m}\). Step 1 divides by \(\delta\). \(\blacksquare\)

Steps 4 to 6 are easy to watch for thresholds. Draw \(2m = 400\) scores, label them by \(t^\star = 0.6\), split them into \(S\) and \(S'\), and estimate \(\mathbb E_\sigma[\max_h |\theta_h|]\) by drawing random signs:

```python
import math
import random

rng, m, t_star = random.Random(0), 200, 0.6
x = [rng.random() for _ in range(2 * m)]
loss_at = lambda t: [int((xi > t) != (xi > t_star)) for xi in x]
HC = {tuple(loss_at(t)) for t in sorted(x) + [-1.0]}

def max_theta(sigma):
    return max(abs(sum(s * (v[m + i] - v[i])
                       for i, s in enumerate(sigma))) / m
               for v in HC)

draws = [max_theta([rng.choice((-1, 1)) for _ in range(m)])
         for _ in range(300)]
bound = math.sqrt(2 / m) * (2 + math.sqrt(math.log(len(HC))))
mean = sum(draws) / len(draws)
print(len(HC), round(mean, 4), round(bound, 4))
```

It prints `401 0.0845 0.4448`. An infinite class, but on these 400 scores only \(401 = \tau_{\mathrm{thr}}(400)\) distinct loss vectors, exactly as Step 4 promised. The average largest \(|\theta_h|\) is about 0.085, comfortably under the bound of 0.445. The bound is valid, and loose.

## Closing the circle: finite VC dimension gives uniform convergence

Now combine the two halves. Let \(\mathrm{VCdim}(\mathcal H) = d\). If \(d = 0\), no single point is shattered, so all hypotheses agree on every point: \(\mathcal H\) is one function, a finite class, and lesson 4 already gives uniform convergence. So let \(d \ge 1\). For \(2m \ge d\), Sauer’s lemma gives \(\tau_{\mathcal H}(2m) \le (2em/d)^d\), and the theorem gives, with probability at least \(1 - \delta\), for all \(h\),

\[ |L_S(h) - L_{\mathcal D}(h)| \le \frac{4 + 2\sqrt{d \log(2em/d)}}{\delta \sqrt{2m}}. \]

It is the theorem’s bound with Sauer’s \((2em/d)^d\) in place of \(\tau_{\mathcal H}(2m)\), and the log of a \(d\)-th power is \(d\) times the log. In plain words: **for a class of VC dimension \(d\), every training score is within roughly \(\sqrt{d \log m / m}/\delta\) of the truth.** For thresholds, \(d = 1\), at \(m = 200\) and \(\delta = 0.1\), it gives 4.64, a little above the 4.45 from the exact growth function, since Sauer’s simpler bound is looser.

Write \(u = d \log(2em/d)\). When \(u \ge 4\), we have \(4 \le 2\sqrt u\), so the numerator is at most \(4\sqrt u\) and the bound is at most \(\frac{1}{\delta}\sqrt{8u/m}\). This is at most \(\varepsilon\) when

\[ m \ge \frac{8 d \log(2em/d)}{(\delta\varepsilon)^2} = \alpha \log m + \alpha \log(2e/d), \qquad \alpha = \frac{8d}{(\delta\varepsilon)^2}. \]

The left inequality is “the bound \(\sqrt{8u/m}/\delta\) is at most \(\varepsilon\)”, squared and solved for \(m\). The right side splits \(\log(2em/d)\) into \(\log m + \log(2e/d)\), and \(\alpha\) (alpha) is just a name for the coefficient. In plain words: **enough labels means \(m\) beats a multiple of its own logarithm plus a constant.** For thresholds at \(\varepsilon = \delta = 0.1\): \(\alpha = 8/0.0001 = 80{,}000\).

The unknown \(m\) sits on both sides. A small inequality, which we’ll call the **log lemma**, unties it: if \(\alpha \ge 1\) and \(\beta \ge 0\), then \(m \ge 4\alpha \log(2\alpha) + 2\beta\) implies \(m \ge \alpha \log m + \beta\). Here \(\beta\) (beta) stands for the constant term. In plain words: **a number at least about \(4\alpha \log \alpha\) outgrows \(\alpha\) times its own log.** With \(\alpha = 2\) and \(\beta = 1\): \(8 \log 4 + 2 \approx 13.1\), and indeed \(13.1 \ge 2 \log 13.1 + 1 \approx 6.1\).

*Proof of the log lemma.* First, for any \(\gamma > 0\), \(x \ge 2\gamma \log \gamma\) implies \(x \ge \gamma \log x\). If \(\gamma \le e\), then \(x - \gamma \log x\) is smallest at \(x = \gamma\), where it equals \(\gamma(1 - \log \gamma) \ge 0\), so the conclusion holds for every \(x > 0\). If \(\gamma > e\), then \(x - \gamma \log x\) increases for \(x > \gamma\), and \(2\gamma\log\gamma > \gamma\); at \(x = 2\gamma \log \gamma\) it equals \(\gamma(\log \gamma - \log(2\log\gamma)) \ge 0\), because \(\gamma \ge 2 \log \gamma\) for every \(\gamma > 0\). Now apply this with \(\gamma = 2\alpha\): \(m \ge 4\alpha \log(2\alpha)\) gives \(m \ge 2\alpha \log m\), that is, \(m/2 \ge \alpha \log m\). Also \(m \ge 2\beta\), that is, \(m/2 \ge \beta\), because \(4\alpha\log(2\alpha) \ge 0\) when \(\alpha \ge 1\). Add the two halves. \(\blacksquare\)

With \(\beta = \alpha \log(2e/d)\) when \(d \le 5\), and \(\beta = 0\) when \(d \ge 6\) (then \(\log(2e/d) < 0\) and that term only helps), we get:

\[ m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le \frac{32 d}{(\delta\varepsilon)^2} \log\left( \frac{16 d}{(\delta\varepsilon)^2} \right) + \frac{16 d}{(\delta\varepsilon)^2} \max\left\{ \log\frac{2e}{d},\, 0 \right\}. \]

This is the log lemma’s \(4\alpha \log(2\alpha) + 2\beta\) written out: \(4\alpha = 32d/(\delta\varepsilon)^2\), \(2\alpha = 16d/(\delta\varepsilon)^2\), and \(\max\{\cdot, 0\}\) keeps \(\log(2e/d)\) when it is positive and replaces it by 0 otherwise. In plain words: **a class of VC dimension \(d\) gets uniform convergence from about \(d \log(d/(\delta\varepsilon)^2)/(\delta\varepsilon)^2\) labels, times a constant.** For thresholds at \(\varepsilon = \delta = 0.1\):

```python
from math import e, log, sqrt

d, eps, delta = 1, 0.1, 0.1

def bound(m):
    u = d * log(2 * e * m / d)
    return (4 + 2 * sqrt(u)) / (delta * sqrt(2 * m))

alpha = 8 * d / (delta * eps) ** 2
beta = alpha * max(log(2 * e / d), 0)
m = 4 * alpha * log(2 * alpha) + 2 * beta
print(round(bound(200), 4), round(alpha), round(beta))
print(round(m), m >= alpha * log(m) + beta, round(bound(m), 4))
```

It prints `4.6441 80000 135452`, then `4105441 True 0.0427`. The first line is the Sauer-based bound at 200 labels, then \(\alpha\) and \(\beta\). The second is the guaranteed sample size, about 4.1 million, the log lemma’s condition holding there, and the bound at that size, 0.043, safely under \(\varepsilon = 0.1\).

Two loose ends. Such an \(m\) is far larger than \(d\), so \(2m \ge d\) holds. And \(m/d \ge 32 \log 16 > 88\), so \(2em/d > 478\), \(\log(2em/d) > 6\), and \(u \ge 4\) holds too. So every class of finite VC dimension has the uniform convergence property, which is \(6 \Rightarrow 1\), and the circle of equivalences is closed. \(\blacksquare\)

How good is this bound in practice? For thresholds we know the growth function exactly, \(\tau(2m) = 2m + 1\), so let’s ask the theorem directly for the smallest \(m\) that brings its bound under \(\varepsilon = 0.1\) at \(\delta = 0.1\). The function `theorem_bound` is the theorem’s right-hand side, the same formula as the 200-label snippet earlier:

```python
from math import log, sqrt

def theorem_bound(m, delta, tau_2m):
    return (4 + 2 * sqrt(log(tau_2m))) / (delta * sqrt(2 * m))

eps = delta = 0.1
m = 1
while theorem_bound(m, delta, 2 * m + 1) > eps:
    m *= 2
lo, hi = m // 2, m
while hi - lo > 1:
    mid = (lo + hi) // 2
    ok = theorem_bound(mid, delta, 2 * mid + 1) <= eps
    lo, hi = (lo, mid) if ok else (mid, hi)
print(hi)
```

It prints `662296`. The two-strip argument of lesson 6a needed 30. The general proof pays twice: Markov’s inequality puts \(1/\delta\) where a \(\log(1/\delta)\) would do, and the ghost-sample and random-sign steps give away constant factors. That is why the quantitative version of the fundamental theorem, stated in lesson 6a and proved in a later lesson, matters for budgets. But the proof above answers the yes-or-no question completely, and that was the reviewer’s first question: *is any number of clicks enough?* For any class of finite VC dimension, yes. The second question, how many, is where we go next.

## Fix the bug

Back to the click log. You now own a theorem that works with careless labels: it holds for every distribution \(\mathcal D\), and nothing in its proof asked for a rule with zero mistakes. Let’s turn it into a number of clicks.

First, what does each rule really cost in this world? Say a rule disagrees with the junk your users mean, the interval \([0.55, 0.8]\), on a stretch of scores that carries probability \(w\). Where the rule agrees with that interval, it is wrong only when the click is wrong, 5% of the time. Where it disagrees, it is wrong whenever the click is right, 95% of the time. Adding the two parts,

\[ L_{\mathcal D}(h) = 0.05\,(1 - w) + 0.95\, w = 0.05 + 0.9\, w. \]

From the inside out:

- \(w\) is the share of arriving mail on which \(h\) and the interval \([0.55, 0.8]\) give different labels, and \(1 - w\) is the share on which they agree.
- 0.05 and 0.95 are the chances that a click contradicts \(h\) on each part, so each product is the share of mail where \(h\) is wrong for that reason.
- The right side collects the terms: \(0.05 - 0.05w + 0.95w = 0.05 + 0.9w\).

In plain words: **a rule pays 5% for the careless clicks, plus 0.9 for every unit of probability on which it departs from the junk your users mean.** The best interval is \([0.55, 0.8]\) itself, with \(w = 0\) and true risk 0.05. The best threshold says junk above 0.55 and is wrong about the newsletters above 0.8, so \(w = 0.2\) and its true risk is 0.23. No threshold does better: one with its cut-off below 0.8 calls every newsletter junk, so \(w \ge 0.2\), and one with its cut-off at 0.8 or above calls all of \([0.55, 0.8]\) keep, so \(w \ge 0.25\). Unions of intervals wrap a tiny band around every junk click and call almost everything else keep, so they disagree on nearly the whole interval, \(w = 0.25\), and their true risk is 0.275:

```python
def risk(w):
    return 0.05 * (1 - w) + 0.95 * w

for name, w in (("interval [0.55, 0.8]:", 0.0),
                ("threshold at 0.55:", 0.2),
                ("unions of intervals:", 0.25)):
    print(name, round(risk(w), 4))
```

It prints each name with its risk: `0.05`, `0.23` and `0.275`. The function `risk` is the first form of the formula, term by term. So the reviewer’s target, a true risk within \(\varepsilon = 0.1\) of the best interval, means a true risk of at most 0.15, with confidence \(1 - \delta = 90\%\).

[[diagram:noisy-clicks]]

Now the certificate. The theorem says that, with probability at least \(1 - \delta\), every rule’s gap between training and true risk is at most its bound. If that bound is at most \(\varepsilon/2 = 0.05\), the clicks are \(0.05\)-representative, and the arrow 1 implies 2 puts ERM within \(\varepsilon = 0.1\) of the best rule in the class. For intervals, the condition reads

\[ \frac{4 + 2\sqrt{\log \tau_{\mathrm{int}}(2m)}}{\delta \sqrt{2m}} \le \frac{\varepsilon}{2}, \qquad \tau_{\mathrm{int}}(2m) = \binom{2m}{2} + 2m + 1. \]

From the inside out:

- \(\tau_{\mathrm{int}}(2m)\) is the growth function of intervals at \(2m\) points. The second formula is the interval count from the growth table, one block of junk chosen by its two ends, or a single point, or nothing, with \(2m\) points in place of \(m\).
- The left side is the theorem’s bound with that count plugged in, and \(\delta\) is 0.1.
- The right side, \(\varepsilon/2\), is half the tolerance, because the arrow 1 implies 2 doubles the gap on its way to ERM’s true risk.

In plain words: **certify a number of clicks when the theorem’s worst gap, computed with the exact number of interval labellings of \(2m\) points, is at most half the tolerance.** Try a million clicks. Then \(2m = 2{,}000{,}000\), the count is \(\tau_{\mathrm{int}} = 2{,}000{,}001{,}000{,}001\), its log is about 28.32 and the square root of that about 5.32, so the top is \(4 + 10.64 = 14.64\). The bottom is \(0.1 \times \sqrt{2{,}000{,}000} \approx 141.42\). The bound is about 0.1035, twice what we need. The snippet below computes it, then searches, as the 662,296 snippet did, for the smallest certified number of clicks for thresholds (whose count at \(n\) points is \(n + 1\)) and for intervals; `bound` is the theorem’s right-hand side and `tau` holds the two growth functions:

```python
from math import comb, log, sqrt

def bound(m, delta, tau_2m):
    return (4 + 2 * sqrt(log(tau_2m))) / (delta * sqrt(2 * m))

tau = {"thresholds": lambda n: n + 1,
       "intervals": lambda n: comb(n, 2) + n + 1}
eps = delta = 0.1
m = 10 ** 6
print(round(bound(m, delta, tau["intervals"](2 * m)), 4))
for name, t in tau.items():
    lo, hi = 1, 2 ** 23
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok = bound(mid, delta, t(2 * mid)) <= eps / 2
        lo, hi = (lo, mid) if ok else (mid, hi)
    print(name, hi)
print(round(2 * sqrt(log(2)) / delta, 4))
```

It prints `0.1035`, then `thresholds 2825554`, `intervals 4623683`, and `16.6511`. That last number is for unions of intervals. They shatter every set, so \(\tau_{\cup}(2m) = 2^{2m}\), and \(\log 2^{2m} = 2m \log 2\). Put that into the bound and the \(\sqrt{2m}\) cancels from its second part: the bound is \(4/(\delta\sqrt{2m}) + 2\sqrt{\log 2}/\delta\), and the second term alone is 16.65 at \(\delta = 0.1\), whatever \(m\) is. No number of clicks brings it near 0.05.

Now the three classes sort themselves, exactly as the VC dimension said they would.

- **Unions of intervals**, VC dimension infinite: no certificate at any number of clicks. And their ERM, which wraps a band around every junk click, sits at true risk 0.275.
- **Thresholds**, VC dimension 1: certified at 2,825,554 clicks. But certified *what*? That ERM lands within 0.1 of the best threshold, 0.23, so at most 0.33. That is the rat’s failure from lesson 1 again: a class that misses the truth stays wrong, however well it is certified.
- **Intervals**, VC dimension 2: certified at 4,623,683 clicks, inside the week’s five million. With probability at least 90%, ERM’s true risk is at most \(0.05 + 0.1 = 0.15\).

Here is one run at the certified size, with the maximum-subarray scan of ERM-Interval from lesson 6a. Each click is junk with weight \(+1\) or keep with weight \(-1\); a junk click is one where “inside \([0.55, 0.8]\)” and “the click is careless” disagree, which is what the `!=` says; and the true risk is the formula above, with \(w\) the two slivers between the learned ends and the true ones:

```python
import random

rng, m, a, b, flip = random.Random(0), 4_623_683, 0.55, 0.8, 0.05
xs = sorted(rng.random() for _ in range(m))
best, lo, hi, run, start = 0, None, None, 0, 0.0
for x in xs:
    w = 1 if (a <= x <= b) != (rng.random() < flip) else -1
    run, start = (run + w, start) if run > 0 else (w, x)
    if run > best:
        best, lo, hi = run, start, x
w = abs(lo - a) + abs(hi - b)
print(f"{lo:.6f} {hi:.6f} {0.05 + 0.9 * w:.6f}")
```

It prints `0.550000 0.800000 0.050000`: the learned ends match the true ones to six decimals, and the true risk is the 5% of careless clicks that no rule can avoid. The certificate promised at most 0.15; the run did far better. It would have done far better with much less, too: in five runs of a thousand clicks each, ERM never did worse than 0.053. But a good run is not a certificate. The reviewer signs for 4,623,683 clicks, because that is the number the proof can stand behind.

In the lab below you face the reviewer again. Each round the hidden junk interval changes. You choose the class and the number of clicks, and the lab tells you whether your choice is certified before it trains. Watch what a rich class, a poor class and a short log each do.

[[lab-act2]]

## What people get wrong

**“The proven bound tells me how many labels to buy.”** The proof in this lesson certifies that some finite number exists, and its number, 662,296 for thresholds at \(\varepsilon = \delta = 0.1\), is a wild overestimate of the 30 a direct argument gives. Use a direct argument where you have one, and the quantitative version of the fundamental theorem for the shape of the dependence.

**“Uniform convergence needs a finite class.”** Thresholds are infinite, yet on 400 fixed scores they leave only 401 distinct loss vectors, and the proof never needs more than that. Once the real and ghost samples are fixed, what a class costs is how many ways it labels \(2m\) points, \(\tau_{\mathcal H}(2m)\), not how many rules it lists.

## Recap and the next bug

1. The **growth function** \(\tau_{\mathcal H}(m)\) is the largest number of labellings a class produces on any \(m\) points. It equals \(2^m\) up to the VC dimension; past it, thresholds give \(m + 1\) and intervals \(\binom{m}{2} + m + 1\).
2. **Sauer’s lemma**: a class of VC dimension \(d\) has \(\tau_{\mathcal H}(m) \le \sum_{i \le d} \binom{m}{i} \le (em/d)^d\), polynomial in \(m\). Its proof shows something stronger, that a class labels a set in at most as many ways as the set has shattered subsets, by splitting off one point and counting tails.
3. **Uniform convergence from a small growth function**: with probability at least \(1 - \delta\), every rule’s gap is at most \((4 + 2\sqrt{\log \tau_{\mathcal H}(2m)})/(\delta\sqrt{2m})\). The proof runs through Markov’s inequality, a ghost sample, random signs, Hoeffding’s inequality and the union bound over the finitely many labellings of the \(2m\) points, and the tail lemma.
4. Together they close the circle: a finite VC dimension gives uniform convergence, so all six statements of the fundamental theorem are equivalent. The proof’s numbers are generous: 662,296 labels for thresholds where a direct argument needs 30.
5. On noisy clicks, intervals are certified at 4,623,683 clicks to land within 0.1 of the best interval. Thresholds can be certified, but only near a best threshold that is itself wrong on 23% of mail, and unions of intervals can never be certified.

Your filter ships with a certificate. Then a teammate comes to you with a better idea. Scores are blunt, she says; what you want are **decision lists**: rules like “if the score is above 0.9 and the sender is unknown, junk; else if the message has no link, keep; else if ...; else keep”. Short lists for simple weeks, long lists when the spammers get clever, and the list can grow as long as it needs to. You check: lists of length at most \(k\) form a class of finite VC dimension, for each \(k\). But she wants no cap on \(k\), and the class of all lists, of every length, shatters sets of every size. Its VC dimension is infinite, and today’s theorem says no learner can PAC learn it. Yet a learner that prefers short lists and accepts a long one only when the data demand it seems perfectly sensible. Can a class like that be learned at all, in some weaker but still honest sense? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof of Sauer’s lemma, explain in your own words why \(|\mathcal H_C| = |Y_0| + |Y_1|\), and why every set \(B \subseteq C'\) shattered by the partner class \(\mathcal H'\) gives a shattered set \(B \cup \{c_1\}\). What would go wrong in Step 4 if we used \(\mathcal H\) in place of \(\mathcal H'\)?
2. In lesson 6a’s concept check you found the VC dimension of \(\mathcal H_2\), “junk if the score lies in one of at most two intervals”: it is 4. Compute \(\tau_{\mathcal H_2}(5)\) and \(\tau_{\mathcal H_2}(10)\) and compare them with Sauer’s bound \(\sum_{i \le d} \binom{m}{i}\) and with \((em/d)^d\).
3. For unions of intervals, the theorem’s bound never falls below about 16.65 at \(\delta = 0.1\). Does that prove that unions of intervals lack the uniform convergence property? If not, what does prove it?

## Problems

Once the concept check is done, try this. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (when the counts are tight).** For a finite set \(A\), Sauer’s lemma came as a chain of two inequalities: the number of labellings \(|\mathcal H_A|\) is at most the number of subsets of \(A\) that \(\mathcal H\) shatters (the stronger claim), and that is at most \(\sum_{i \le d} \binom{|A|}{i}\), where \(d\) is the VC dimension of \(\mathcal H\). Find classes for all four combinations: both inequalities equalities, only the first strict, only the second strict, and both strict.
