# Uniform Convergence

## The dashboard that always looks better than production

Since the last lesson, the team has grown up. Nobody pretends any more that one hypothesis will agree with every label: users disagree, people click in a hurry, and the promise you now make is the agnostic one, a true risk close to the best in the class. The mail room this year has the thirty features from lesson 2, and the class the team works with is the simplest one: thirty one-feature hypotheses, \(h_1, \dots, h_{30}\), where \(h_j\) says junk exactly when feature \(j\) is present.

To keep everyone honest, someone built a dashboard. Every Monday, users label a fresh batch of 60 messages. The dashboard has one row per hypothesis, and each row shows that hypothesis’s empirical risk on this week’s 60 examples: the fraction it gets wrong. Then the team does the obvious thing. They ship the hypothesis at the top of the dashboard, the one with the smallest empirical risk. That is ERM, empirical risk minimization from lesson 2, run by hand, every week.

A month later, production reports back. Production measures the shipped hypothesis on the thousands of messages users labelled after it went live, which is so many that its number is, for all practical purposes, the true risk. And production is disappointed. Not once, but every single week. On average, the dashboard said the shipped hypothesis would be wrong on 18% of messages. Production says 28%.

You check the code. The counting is right: each row really is the fraction of this week’s examples that its hypothesis gets wrong. The examples are drawn fairly, nobody peeks, and no row is computed on the wrong data. Every number on the dashboard is correct, and yet the number you act on is wrong by ten points, always in the same direction.

How can a dashboard where every row is computed correctly be systematically misleading? And how many labels would you need before you could trust it? That is this lesson.

## A world we can see into

The team cannot see the true risks; that is the whole problem. But we can build a world where we know them, and watch the dashboard from above.

In this world, every message is junk or keep with equal chance, and each of the thirty features is a noisy signal of the label. Feature \(j\) agrees with the label except on a fraction \(r_j\) of messages, and these disagreements happen independently from feature to feature. So the true risk of \(h_j\) is exactly \(r_j\). We spread the thirty rows evenly: the first row is wrong on 25% of messages, and each row after it is wrong half a percentage point more often than the one before, so row 2 is at 25.5% and row 3 at 26%. As a formula,

\[ r_j = 0.25 + 0.005\,(j - 1), \qquad j = 1, \dots, 30. \]

Read it from the inside out. The subscript \(j\) is the row number, so \(r_j\) is “the true risk of row \(j\)”, and the list after the comma, \(j = 1, \dots, 30\), says the formula is applied once to each of the thirty rows. Inside the brackets, \(j - 1\) counts how many rows sit above row \(j\). Multiplying by 0.005 turns that count into half a point per row, and adding 0.25 starts the first row at 25%. In plain words: **each row is wrong half a point more often than the row above it, starting from 25%.** For row 11, ten rows sit above it, so \(r_{11} = 0.25 + 0.005 \times 10 = 0.30\).

```python
r = [0.25 + 0.005 * (j - 1) for j in range(1, 31)]
print(r[0], round(r[10], 3), round(r[29], 3))
```

It prints `0.25 0.3 0.395`. Python counts list positions from 0, so `r[10]` is row 11 and `r[29]` is row 30.

So \(h_1\) is the best in the class with true risk 0.25, \(h_{11}\) has 0.30, and \(h_{30}\), the worst, has 0.395. The team does not know any of this. They only see the dashboard.

Before we look at the whole dashboard, let’s look at one row on its own.

## One row on its own is honest

Take \(h_{11}\), whose true risk is 0.30, and watch its row for eight weeks. Each week, 60 new examples arrive, and the row shows how many of them \(h_{11}\) gets wrong.

| week | mistakes out of 60 | empirical risk |
|---|---|---|
| 1 | 17 | 0.283 |
| 2 | 24 | 0.400 |
| 3 | 18 | 0.300 |
| 4 | 14 | 0.233 |
| 5 | 19 | 0.317 |
| 6 | 12 | 0.200 |
| 7 | 20 | 0.333 |
| 8 | 20 | 0.333 |

The row wobbles a lot. Week 6 made \(h_{11}\) look like a 20% hypothesis, and week 2 made it look like a 40% one. But notice which way it wobbles: both ways. Add up the mistakes and you get 144 out of 480 examples, exactly 0.300. A single row has no lean. It is a noisy but fair estimate of its own true risk.

That fairness has a name and a one-line proof, and it is worth having both. Recall from lesson 3 that a **loss function** \(\ell(h, z)\) scores hypothesis \(h\) on one example \(z\). For junk filtering it is the 0-1 loss: 1 when \(h\) gets the example wrong, 0 when it gets it right. The true risk is the expected loss on a fresh example, the average over every example that could arrive, each weighted by how often it arrives. The empirical risk is the average loss on the training sequence. For \(h_{11}\), a fresh example is a mistake with probability 0.30, so its true risk is 0.30; in week 1 it made 17 mistakes on 60 examples, so its empirical risk was \(17/60 \approx 0.283\). In symbols:

\[ L_{\mathcal D}(h) = \mathbb E_{z \sim \mathcal D}[\ell(h, z)], \qquad L_S(h) = \frac{1}{m} \sum_{i=1}^m \ell(h, z_i). \]

Read the two halves from the inside out.

- An example is a pair \(z = (x, y)\): a message \(x\) and its label \(y\). The loss \(\ell(h, z)\) is the number that scores \(h\) on that one example.
- The 0-1 loss is written \(\mathbb 1[h(x) \ne y]\), with the indicator notation of lesson 1: \(\mathbb 1[\text{statement}]\) is 1 when the statement is true and 0 when it is false.
- \(\mathcal D\) is, as in lesson 1, the distribution of examples: how often each (message, label) pair arrives. \(z \sim \mathcal D\) reads “\(z\) is drawn from \(\mathcal D\)”, and the \(\mathbb E\) in front, the expectation, averages the loss over those draws, weighting each possible example by its probability.
- On the right, \(z_1, \dots, z_m\) are the \(m\) examples of the training sequence \(S\). The sum \(\sum_{i=1}^m\) adds the loss of example \(i\) for each \(i\) from 1 to \(m\), and \(\frac1m\) turns that total into an average.

In plain words: **the true risk is the average loss on a fresh example, and the empirical risk is the average loss on the examples you have.** Here is week 1 of \(h_{11}\) in Python, with each message shrunk to the one feature bit that \(h_{11}\) reads: 9 false alarms, 8 missed junk messages and 43 correct labels.

```python
def h(x):
    return x

def loss(h, z):
    x, y = z
    return int(h(x) != y)

S = ([(1, 0)] * 9 + [(0, 1)] * 8
     + [(1, 1)] * 25 + [(0, 0)] * 18)
L_S = sum(loss(h, z) for z in S) / len(S)
L_D = 1 * 0.30 + 0 * 0.70
print(len(S), round(L_S, 3), L_D)
```

It prints `60 0.283 0.3`. The last line before the print is the expectation spelled out: each value of the loss times its probability, added up.

Now fix a hypothesis \(h\) before the examples are drawn, and ask what its row shows on average over many weeks. For \(h_{11}\), each of the 60 examples is a mistake with chance 0.30, so on average a week holds \(60 \times 0.30 = 18\) mistakes, and \(18/60 = 0.30\): exactly its true risk. In general, each \(z_i\) is drawn from \(\mathcal D\), so each \(\ell(h, z_i)\) has expectation \(L_{\mathcal D}(h)\). The expectation of a sum is the sum of the expectations (this is **linearity of expectation**, and it needs no independence at all), so

\[ \mathbb E_{S \sim \mathcal D^m}[L_S(h)] = \frac{1}{m} \sum_{i=1}^m \mathbb E[\ell(h, z_i)] = L_{\mathcal D}(h). \]

Here \(S \sim \mathcal D^m\) is lesson 1’s notation for a random training sequence: its \(m\) examples are drawn i.i.d. from \(\mathcal D\), that is, each from the same \(\mathcal D\) and no draw influencing another. \(\mathbb E_{S \sim \mathcal D^m}\) averages over every training sequence that could arrive. Read the chain left to right. The first expression is the row’s dashboard number averaged over all possible weeks. The middle one moves the expectation inside the sum, one example at a time. Each of those \(m\) expectations is \(L_{\mathcal D}(h)\), so the sum is \(m\,L_{\mathcal D}(h)\), and \(\frac1m\) brings it back to \(L_{\mathcal D}(h)\). In plain words: **averaged over all possible weeks, a fixed row shows exactly its true risk.** The eight weeks in the table, 144 mistakes in 480 examples, landed right on it.

An estimate whose expectation equals the thing it estimates is called **unbiased**. Every row of the dashboard, looked at alone, is unbiased.

Here is the same claim run 10,000 times instead of eight. Under the 0-1 loss, the number of mistakes in a week is a sum of 60 independent coin flips that each come up “mistake” with probability 0.30, which Python can draw directly. Each entry of `k` below is one week’s mistakes, so `k / m` is that week’s \(L_S(h_{11})\), and averaging over weeks is the \(\mathbb E_S\) on the left of the chain. Being off by more than 0.1 means missing the mean of 18 by more than 6 mistakes, and counting mistakes keeps the test exact.

```python
import random

rng, r, m, weeks = random.Random(0), 0.30, 60, 10_000
k = [rng.binomialvariate(m, r) for _ in range(weeks)]
print(round(sum(k) / m / weeks, 4))
print(round(sum(abs(x - m * r) > 6 for x in k) / weeks, 4))
```

It prints `0.2995` and `0.0635`. The average over many weeks sits on 0.30, as the proof says. The second number is the one to remember: in about 6% of weeks, this row is off by more than 10 points. For one row, that is an occasional bad week. The dashboard has thirty rows.

## The top of the dashboard is crowded with lucky rows

Here is one typical week, the whole dashboard sorted the way the team sees it, smallest empirical risk first. We show the top eight rows, and next to each one the true risk that only we can see. The last column is the row’s **luck** for the week: its empirical risk minus its true risk, negative when the row looks better than it is.

| rank | hypothesis | mistakes out of 60 | empirical risk | true risk | luck |
|---|---|---|---|---|---|
| 1 | \(h_7\) | 11 | 0.183 | 0.280 | −0.097 |
| 2 | \(h_2\) | 14 | 0.233 | 0.255 | −0.022 |
| 3 | \(h_6\) | 14 | 0.233 | 0.275 | −0.042 |
| 4 | \(h_{14}\) | 14 | 0.233 | 0.315 | −0.082 |
| 5 | \(h_{17}\) | 15 | 0.250 | 0.330 | −0.080 |
| 6 | \(h_5\) | 16 | 0.267 | 0.270 | −0.003 |
| 7 | \(h_{15}\) | 16 | 0.267 | 0.320 | −0.053 |
| 8 | \(h_3\) | 17 | 0.283 | 0.260 | +0.023 |

Before you read on, look at the luck column. What do the top seven rows have in common?

Every one of them is lucky. The winner, \(h_7\), had a great week: it looks like an 18% hypothesis and is really a 28% one. Meanwhile the best hypothesis in the class, \(h_1\), had a bad week, 19 mistakes, and sits at rank 14 with an empirical risk of 0.317. The team ships \(h_7\), promises 18%, and production finds 28%. That is the bug, in one week.

Nothing here is a coincidence of this particular week. Ranking by empirical risk sorts the rows by true risk plus luck, and the top of that ranking is exactly where rows with large negative luck collect. Each row is a fair estimate, but **the row you pick for being smallest is not**. This is called the **winner’s curse**: when you select the best-looking of many noisy estimates, the estimate of the winner is biased toward looking good. The name comes from auctions, where the bidder who wins a contested oil lease is usually the one who overestimated it most.

[[diagram:winners-curse]]

We can prove the direction of the bias in general. In what follows, \(\mathcal H\) is the hypothesis class, as in lesson 1, here the thirty rows, and \(h_S\) is the hypothesis ERM returns on the training sequence \(S\), the shipped row. Its dashboard number is the smallest on the dashboard, 0.183 in the typical week, which we write \(L_S(h_S) = \min_{h \in \mathcal H} L_S(h)\): read \(\min_{h \in \mathcal H}\) as “the smallest value, over every \(h\) in \(\mathcal H\), of”.

**Proposition (the winner’s curse).** Let \(\mathcal H\) be a finite class, \(\mathcal D\) any distribution, and \(h_S\) any ERM output on \(S \sim \mathcal D^m\). Then

\[ \mathbb E_S\big[L_S(h_S)\big] \;\le\; \min_{h \in \mathcal H} L_{\mathcal D}(h) \;\le\; \mathbb E_S\big[L_{\mathcal D}(h_S)\big]. \]

Read it from the inside out. In the middle, \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\) is the best true risk in the class, a fixed number that does not change from week to week: 0.25 here, the true risk of \(h_1\). On the left, \(L_S(h_S)\) is the number the dashboard shows for the shipped row, and \(\mathbb E_S\) averages it over all possible weeks. On the right, \(L_{\mathcal D}(h_S)\) is what production finds for the shipped row, averaged the same way. The two \(\le\) signs put the best in the class between them. In words: **on average, the dashboard shows the winner at or below the best true risk in the class, and production finds the winner at or above it.** The typical week already has this order: 0.183 shown, 0.25 best, 0.280 found. The snippet after the proof pushes 10,000 weeks through all three terms.

*Proof.* Start with the left inequality. Pick any fixed hypothesis \(h' \in \mathcal H\). Since \(h_S\) minimizes the empirical risk, \(L_S(h_S) \le L_S(h')\) holds for every training sequence \(S\). An inequality that holds for every outcome survives taking expectations, so \(\mathbb E[L_S(h_S)] \le \mathbb E[L_S(h')]\). The right side is \(L_{\mathcal D}(h')\), because a fixed row is unbiased. This is true for every \(h'\) in the class, so it is true for the one with the smallest true risk.

The right inequality is even simpler. Whatever \(S\) is, \(h_S\) is a member of \(\mathcal H\), so its true risk is at least the smallest true risk in \(\mathcal H\). Take expectations again. \(\blacksquare\)

Notice exactly where unbiasedness was used: for a *fixed* \(h'\), chosen before the examples. The proof never claims \(\mathbb E[L_S(h_S)] = \mathbb E[L_{\mathcal D}(h_S)]\), and it would be false, because \(h_S\) is chosen by looking at \(S\). Here are the three quantities for our dashboard, over 10,000 simulated weeks:

```python
import random

rng, m, weeks = random.Random(0), 60, 10_000
r = [0.25 + 0.005 * j for j in range(30)]
shown = shipped = 0.0
for _ in range(weeks):
    L_S = [rng.binomialvariate(m, p) / m for p in r]
    h_S = min(range(30), key=lambda j: (L_S[j], rng.random()))
    shown += L_S[h_S]
    shipped += r[h_S]
print(round(shown / weeks, 4), min(r), round(shipped / weeks, 4))
```

It prints `0.1791 0.25 0.2755`: the dashboard promises 18%, the best in the class is 25%, and production gets 28%, in the order the proposition says: `shown` adds up \(L_S(h_S)\), `min(r)` is the middle term, and `shipped` adds up \(L_{\mathcal D}(h_S)\). (The `rng.random()` in the key breaks ties at random, so a tie never quietly favours a better row.)

The proposition tells you which way the dashboard errs. It does not tell you by how much, and it gives you no way to make the error small. For that we need a stronger idea. Before we build it, play with the curse yourself. In the lab, you set how many rows the dashboard has and how many examples each week brings. Predict first: if the team grows the dashboard from 30 rows to 465, still with 60 examples a week, does the winner’s empirical risk go up, go down, or stay the same? And its true risk?

[[lab]]

## A dashboard you can trust row by row

Watch what the lab did as you raised the number of examples. The luck in every row shrank, and once every row’s luck was small, the winner’s luck had nowhere to hide. That is the whole idea of this lesson, and it deserves a definition. Pick a small tolerance, say 0.1, and suppose that this week every row’s luck, the winner’s included, lies between −0.1 and +0.1. Then no row, whichever one you ship, can look more than 0.1 better or worse than it is. We call the tolerance \(\varepsilon\) (the Greek letter epsilon), the accuracy parameter of lesson 3.

**Definition (ε-representative sample).** A training sequence \(S\) is **\(\varepsilon\)-representative** (with respect to the set \(Z\) of all possible examples, the domain, a hypothesis class \(\mathcal H\), a loss function \(\ell\) and a distribution \(\mathcal D\)) if

\[ \forall h \in \mathcal H, \quad |L_S(h) - L_{\mathcal D}(h)| \le \varepsilon. \]

Read it from the inside out. \(L_S(h) - L_{\mathcal D}(h)\) is row \(h\)’s luck this week. The bars \(|\cdot|\) around a number mean its absolute value, its size with the sign dropped (around a set, as in lesson 1, the same bars mean how many elements it has; here they wrap a number). So \(|\dots| \le \varepsilon\) says the luck is at most \(\varepsilon\) in either direction. The symbol \(\forall\) reads “for every”, so \(\forall h \in \mathcal H\) demands this of every row at once. In plain words: **this week, no row of the dashboard is off from its true risk by more than \(\varepsilon\).**

In the typical week above, the largest luck in absolute value, over all thirty rows, was 0.132 (row \(h_{18}\), which looked much worse than it is), so that training sequence was 0.132-representative but not 0.1-representative. Here are three of that week’s rows in Python: the lucky winner \(h_7\), the lucky \(h_{14}\), and the unlucky \(h_{18}\), each as (mistakes, true risk).

```python
rows = {7: (11, 0.280), 14: (14, 0.315), 18: (28, 0.335)}
luck = {j: k / 60 - r for j, (k, r) in rows.items()}
worst = max(abs(v) for v in luck.values())
print(round(worst, 3), worst <= 0.1, worst <= 0.15)
```

It prints `0.132 False True`. The worst luck of the three, which is also the worst of all thirty, is 0.132, so the week fails the test at \(\varepsilon = 0.1\) and passes it at 0.15.

[[diagram:representative]]

Why is this the right condition? Because it survives selection. The winner’s curse happens because the row you pick depends on the luck. A promise about *every* row cannot be dodged that way: whichever row ERM picks, the promise was already made about it. Concretely, if every row is within 0.1 this week, the shipped row’s true risk is at most its dashboard number plus 0.1. And the best row, say at 0.25, shows at most 0.35, so the shipped row shows at most 0.35 too and is truly at most 0.45: within twice the tolerance of the best.

The lemma below needs one new piece of notation. \(\operatorname{argmin}_{h \in \mathcal H} L_S(h)\) is not the smallest value but the set of hypotheses that achieve it, the rows tied at the top of the dashboard, and \(h_S \in \operatorname{argmin}\) says the shipped row is one of them.

**Lemma (a representative sample makes ERM good).** If \(S\) is \(\varepsilon\)-representative, then every ERM output \(h_S \in \operatorname{argmin}_{h \in \mathcal H} L_S(h)\) satisfies

\[ L_{\mathcal D}(h_S) \le L_S(h_S) + \varepsilon \qquad \text{and} \qquad L_{\mathcal D}(h_S) \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + 2\varepsilon. \]

Read the first inequality as: production’s number for the shipped row is at most the dashboard’s number plus \(\varepsilon\). Read the second as: it is at most the best true risk in the class plus \(2\varepsilon\). In plain words: **when every row is within \(\varepsilon\), the shipped hypothesis is at most \(\varepsilon\) worse than its row says and at most \(2\varepsilon\) worse than the best.** Take the typical week, which was 0.132-representative. The dashboard showed \(h_7\) at 0.183, so the lemma promises a true risk of at most \(0.183 + 0.132 = 0.315\), and at most \(0.25 + 2 \times 0.1317 \approx 0.513\). Production found 0.280, inside both. The snippet after the proof checks each link.

*Proof.* The first inequality is the definition applied to the row \(h = h_S\). That is legitimate even though \(h_S\) depends on \(S\): the definition quantifies over every \(h \in \mathcal H\), and \(h_S\) is one of them.

For the second, take any \(h \in \mathcal H\) and chain three steps:

\[ L_{\mathcal D}(h_S) \;\le\; L_S(h_S) + \varepsilon \;\le\; L_S(h) + \varepsilon \;\le\; L_{\mathcal D}(h) + \varepsilon + \varepsilon. \]

The first step is representativeness for \(h_S\). The second holds because \(h_S\) has the smallest empirical risk, so it is at most that of \(h\). The third is representativeness again, this time for \(h\). The chain holds for every \(h \in \mathcal H\), in particular for the one with the smallest true risk. \(\blacksquare\)

In a row of \(\le\) signs, each link is one step and the two ends are what you keep. Here is the chain on the typical week, with \(\varepsilon\) the week’s worst luck and \(h\) the best row \(h_1\), which showed 19 mistakes:

```python
eps = 28 / 60 - 0.335
LS_hS, LD_hS = 11 / 60, 0.280
LS_h1, LD_h1 = 19 / 60, 0.250
chain = [LD_hS, LS_hS + eps, LS_h1 + eps, LD_h1 + 2 * eps]
print([round(c, 3) for c in chain])
print(all(a <= b for a, b in zip(chain, chain[1:])))
```

It prints `[0.28, 0.315, 0.448, 0.513]` and `True`: every link holds, and the two ends are the lemma’s second promise.

These are two promises the team can finally keep, one about the dashboard and one about the choice. To be within \(\varepsilon\) of the best, an \(\varepsilon/2\)-representative sample is enough: to ship within 0.05 of the best, make every row accurate to 0.025. The factor 2 is honest, too: one \(\varepsilon\) is the winner’s possible luck, the other is the best hypothesis’s possible bad luck, and both can happen in the same week.

## The uniform convergence property

The lemma is about one fixed training sequence. But the training sequence is random. So the real question is: how many examples do we need so that, with high probability, the sample we get is representative? For instance: how many examples a week make the 30-row dashboard 0.05-representative in at least 95 weeks out of 100, whatever the world looks like? The 5 bad weeks in 100 we are willing to tolerate are written \(\delta = 0.05\) (the Greek letter delta), the confidence parameter of lesson 3.

**Definition (uniform convergence).** A hypothesis class \(\mathcal H\) has the **uniform convergence property** (with respect to a domain \(Z\) and a loss function \(\ell\)) if there exists a function \(m^{\mathrm{UC}}_{\mathcal H}: (0, 1)^2 \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\) and for every probability distribution \(\mathcal D\) over \(Z\), if \(S\) is a training sequence of \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta)\) examples drawn i.i.d. from \(\mathcal D\), then with probability at least \(1 - \delta\), \(S\) is \(\varepsilon\)-representative.

This is the densest sentence in the lesson, so let’s take it apart.

- \(m^{\mathrm{UC}}_{\mathcal H}\) is a function. The superscript UC stands for uniform convergence, and the subscript says which class it belongs to.
- \((0, 1)^2 \to \mathbb N\) is its type: it takes a pair of numbers, each strictly between 0 and 1 (that is what \((0, 1)^2\) means), and returns a natural number, an element of \(\mathbb N = \{1, 2, 3, \dots\}\), here a count of examples. The pair is \((\varepsilon, \delta)\).
- “For every probability distribution \(\mathcal D\) over \(Z\)”: over all the (message, label) pairs, the same function must work whichever \(\mathcal D\) the world uses.
- “Drawn i.i.d.” is the \(S \sim \mathcal D^m\) of before, and “with probability at least \(1 - \delta\)” is over that random draw.

In plain words: **there is a recipe that turns \((\varepsilon, \delta)\) into a number of examples, and that many examples make every row \(\varepsilon\)-accurate in all but a \(\delta\) fraction of weeks, whatever the world.** For the 30-row dashboard and \(\varepsilon = \delta = 0.05\), this lesson will show that 1,419 examples are enough, so \(m^{\mathrm{UC}}_{\mathcal H}(0.05, 0.05) \le 1{,}419\). The last snippet of “From one row to every row” runs exactly this definition on our world: it draws 10,000 weeks of 1,419 examples and counts the weeks that are not 0.05-representative.

Like the sample complexity of lesson 3, \(m^{\mathrm{UC}}_{\mathcal H}\) is taken to be the minimal such function: how many examples you need to make the whole dashboard trustworthy, probably. And the word **uniform** carries two meanings at once. One sample size works for all members of \(\mathcal H\) together, and it works for every distribution \(\mathcal D\), which nobody tells you in advance.

Recall the target from lesson 3. A class is **agnostic PAC learnable** with a loss \(\ell\) if there are a function \(m_{\mathcal H}\) and an algorithm such that, for every \(\varepsilon, \delta\) and every \(\mathcal D\) over \(Z\), on \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples the algorithm returns \(h \in \mathcal H\) with \(L_{\mathcal D}(h) \le \min_{h' \in \mathcal H} L_{\mathcal D}(h') + \varepsilon\), with probability at least \(1 - \delta\).

The symbols are lesson 3’s: \(m_{\mathcal H}\), without the UC, is the sample complexity of learning, \(h\) is what the algorithm returns, and \(\min_{h' \in \mathcal H} L_{\mathcal D}(h')\) is the best in the class, written with \(h'\) so it doesn’t collide with the returned \(h\). In plain words: **enough examples let the algorithm return, in all but a \(\delta\) fraction of weeks, a hypothesis at most \(\varepsilon\) worse than the best.** With \(\varepsilon = 0.1\) in our world, that means a shipped true risk of at most \(0.25 + 0.1 = 0.35\). The same last snippet counts the weeks where the shipped row misses such a promise, in its variable `worse`.

**Corollary (uniform convergence is enough for learning).** If \(\mathcal H\) has the uniform convergence property with a function \(m^{\mathrm{UC}}_{\mathcal H}\), then \(\mathcal H\) is agnostic PAC learnable with sample complexity

\[ m_{\mathcal H}(\varepsilon, \delta) \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta), \]

and ERM is a successful agnostic PAC learner for \(\mathcal H\).

The display compares two functions at different arguments. On the left is the number of examples learning needs at accuracy \(\varepsilon\); on the right is the number uniform convergence needs at the tighter accuracy \(\varepsilon/2\), with the same \(\delta\). In plain words: **to learn within \(\varepsilon\) of the best, make every row accurate to \(\varepsilon/2\).** For the 30-row dashboard with \(\varepsilon = \delta = 0.05\), that means rows accurate to 0.025, which the formula later in this lesson prices at 5,673 examples, the second column of the `m_uc` snippet there.

*Proof.* Fix \(\varepsilon\) and \(\delta\), and draw \(m \ge m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\) examples. By uniform convergence, the event “\(S\) is \(\varepsilon/2\)-representative” has probability at least \(1 - \delta\). On that event, the lemma with \(\varepsilon/2\) in place of \(\varepsilon\) gives \(L_{\mathcal D}(h_S) \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). So the good outcome contains an event of probability at least \(1 - \delta\), and therefore has probability at least \(1 - \delta\) itself. This holds for every \(\mathcal D\), with ERM as the algorithm. \(\blacksquare\)

So learning has been reduced to a question about averages: how fast do all the rows of a dashboard settle down at once? We answer it in two steps, the same two steps as lesson 2’s proof. First, how far can a single row stray? Then, how do we pay for all the rows together?

## How far can one average stray?

A single row is an average of \(m\) independent, identically distributed numbers, the losses of its \(m\) examples: \(L_S(h) = \frac{1}{m}\sum_i \theta_i\), where \(\theta_i = \ell(h, z_i)\) (the Greek letter theta) is just a short name for example \(i\)’s loss. You may know the **law of large numbers**: as \(m\) goes to infinity, such an average converges to its expectation. True, and useless to the team. It is an asymptotic statement, so it says nothing about the gap at \(m = 60\) or at any other finite \(m\). We need a **measure concentration inequality**: a bound on the probability that an average lands far from its expectation, valid at every finite \(m\).

A few words first. A **random variable** is a number whose value depends on chance, like the number of mistakes this week. Its **variance** measures how widely it spreads around its mean: take how far each outcome lands from the mean, square it, and average. One 0-1 loss of \(h_{11}\) is 1 with probability 0.3 and 0 with probability 0.7, so its mean is 0.3; it misses the mean by 0.7 with probability 0.3 and by 0.3 with probability 0.7, so its variance is \(0.7^2 \times 0.3 + 0.3^2 \times 0.7 = 0.147 + 0.063 = 0.21\). In symbols, \(\operatorname{Var}[Z] = \mathbb E[(Z - \mathbb E[Z])^2]\). Read it from the inside out: \(\mathbb E[Z]\) is the mean, \(Z - \mathbb E[Z]\) is how far this outcome lands from it, squaring makes every distance positive, and the outer \(\mathbb E\) averages the squares. In plain words: **variance is the average squared miss.**

```python
p = 0.3
mean = 1 * p + 0 * (1 - p)
var = (1 - mean)**2 * p + (0 - mean)**2 * (1 - p)
print(round(mean, 2), round(var, 2))
```

It prints `0.3 0.21`.

We build three inequalities, each from the one before, and each much stronger than the last. The first uses only the mean. If \(h_{11}\) averages 18 mistakes a week, it cannot have 24 or more in more than three weeks out of four, because those weeks alone would push the average above \(\frac34 \times 24 = 18\).

**Markov’s inequality.** If \(Z \ge 0\) and \(a > 0\), then \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\).

Here \(Z \ge 0\) says the variable is never negative, \(a\) is any positive threshold, \(\mathbb P[Z \ge a]\) is the probability that \(Z\) reaches it, and \(\mathbb E[Z]/a\) is the mean divided by the threshold. In plain words: **a variable that is never negative reaches a threshold \(a\) at most a fraction mean-over-\(a\) of the time.** With \(Z\) the week’s mistakes, mean 18 and \(a = 24\), the bound is \(18/24 = 0.75\). The exact binomial answer is much smaller:

```python
from math import comb

m, p, a = 60, 0.3, 24
def pmf(k):
    return comb(m, k) * p**k * (1 - p)**(m - k)
EZ = sum(k * pmf(k) for k in range(m + 1))
tail = sum(pmf(k) for k in range(a, m + 1))
print(round(EZ, 6), round(tail, 4), round(EZ / a, 4))
```

It prints `18.0 0.0632 0.75`: the mean computed from the distribution itself, the exact chance of 24 or more mistakes, and Markov’s bound. Markov knows only the mean, so it is crude, but it is the seed of everything below.

*Proof.* Compare two random variables outcome by outcome: \(a \cdot \mathbb 1[Z \ge a] \le Z\). When \(Z \ge a\), the left side is \(a\), which is at most \(Z\). Otherwise the left side is 0, which is at most \(Z\) because \(Z \ge 0\). Take expectations of both sides: the expectation of an indicator is the probability of its event, so \(a\,\mathbb P[Z \ge a] \le \mathbb E[Z]\). Divide by \(a\). \(\blacksquare\)

The second inequality also uses the spread. A variable with a small variance rarely strays far in either direction: a week of \(h_{11}\) seldom misses its mean of 18 by 6 mistakes or more.

**Chebyshev’s inequality.** For any \(Z\) with finite variance and any \(a > 0\), \(\mathbb P[|Z - \mathbb E Z| \ge a] \le \operatorname{Var}[Z]/a^2\).

Here \(\mathbb E Z\) is shorthand for \(\mathbb E[Z]\), \(|Z - \mathbb E Z|\) is the distance from the mean in either direction, and \(\operatorname{Var}[Z]/a^2\) is the variance over the squared distance. In plain words: **a variable lands at least \(a\) away from its mean with probability at most its variance over \(a^2\).** The week’s mistake count has variance \(60 \times 0.21 = 12.6\) (the next paragraph shows why variances of independent draws add up), so missing the mean by 6 or more, which is 0.1 in empirical risk, has probability at most \(12.6/36 = 0.35\).

```python
from math import comb

m, p, a = 60, 0.3, 6
def pmf(k):
    return comb(m, k) * p**k * (1 - p)**(m - k)
mu = m * p
var = sum((k - mu)**2 * pmf(k) for k in range(m + 1))
tail = sum(pmf(k) for k in range(m + 1) if abs(k - mu) >= a)
print(round(var, 4), round(tail, 4), round(var / a**2, 4))
```

It prints `12.6 0.12 0.35`: the variance, the exact chance of missing the mean by 6 or more, and Chebyshev’s bound. Since 24 or more mistakes is one way of missing the mean by 6, Chebyshev’s 0.35 also covers Markov’s event, and beats its 0.75, because it knows the variance too.

*Proof.* The events \(|Z - \mathbb E Z| \ge a\) and \((Z - \mathbb E Z)^2 \ge a^2\) are the same event. Apply Markov to the non-negative variable \((Z - \mathbb E Z)^2\) with threshold \(a^2\); its expectation is \(\operatorname{Var}[Z]\). \(\blacksquare\)

Averaging calms a variable down. One loss of \(h_{11}\) has variance 0.21, but the average of a week’s 60 losses has variance only \(0.21/60 = 0.0035\). In general, for an average of \(m\) i.i.d. variables \(Z_1, \dots, Z_m\) with mean \(\mu\) (the Greek letter mu), the variance shrinks like \(1/m\). Expand \(\operatorname{Var}[\sum_i Z_i] = \sum_{i,j} \mathbb E[(Z_i - \mu)(Z_j - \mu)]\). For \(i \ne j\), independence lets the expectation of the product split into a product of expectations, \(\mathbb E[Z_i - \mu]\,\mathbb E[Z_j - \mu] = 0 \cdot 0\). Only the \(m\) diagonal terms survive, so \(\operatorname{Var}[\sum_i Z_i] = m \operatorname{Var}[Z_1]\), and dividing the sum by \(m\) divides the variance by \(m^2\):

\[ \operatorname{Var}\Big[\frac{1}{m}\sum_{i=1}^m Z_i\Big] = \frac{\operatorname{Var}[Z_1]}{m}, \qquad \mathbb P\Big[\Big|\frac{1}{m}\sum_{i=1}^m Z_i - \mu\Big| \ge \varepsilon\Big] \le \frac{\operatorname{Var}[Z_1]}{m \varepsilon^2}. \]

Read the first identity from the inside out: \(\sum_{i=1}^m Z_i\) adds the \(m\) variables, \(\frac1m\) makes the sum an average, and \(\operatorname{Var}\) measures how much that average wobbles. The second display is Chebyshev applied to the average, with \(a = \varepsilon\) and the shrunken variance. In plain words: **averaging \(m\) independent draws divides the variance by \(m\), so the chance of being \(\varepsilon\) off falls like \(1/m\).**

```python
import random

rng, m, p, weeks = random.Random(0), 60, 0.3, 100_000
avgs = [rng.binomialvariate(m, p) / m for _ in range(weeks)]
mean = sum(avgs) / weeks
var = sum((v - mean)**2 for v in avgs) / weeks
print(round(var, 5), round(p * (1 - p) / m, 5))
```

It prints `0.00351 0.0035`: the variance of 100,000 simulated weekly averages, against the formula. For \(h_{11}\)’s row, with \(m = 60\) and \(\varepsilon = 0.1\), the second display gives \(0.21/0.6 = 0.35\), the same bound as before, now read in empirical risk; it is the `chebyshev` column of the snippet after Hoeffding’s inequality.

That bound shrinks like \(1/m\). For one row, fine. For a dashboard, it is the wrong shape, as you’ll see when we add up the rows: we will need a bound that shrinks *exponentially* in \(m\). That is Hoeffding’s inequality, and its engine is a small lemma about one bounded variable.

The lemma looks not at a variable itself but at \(e\), the number 2.718…, raised to a multiple of it. Here is why we want that strange-looking quantity: powers turn a sum into a product, as in \(e^{2 + 3} = e^2 \cdot e^3\), and products of independent variables are easy to take expectations of. Hold that thought. For a variable that averages 0, the lemma caps that average using only the width of its range. Take the loss of \(h_{11}\) minus its mean 0.3: it is 0.7 with probability 0.3 and −0.3 with probability 0.7, so it averages 0 and lives in a range of width 1. The average of \(e\) to that value is \(0.3\,e^{0.7} + 0.7\,e^{-0.3} \approx 1.1227\), and the lemma caps it at \(e^{1/8} \approx 1.1331\).

**Hoeffding’s lemma.** Let \(X\) take values in \([a, b]\) with \(\mathbb E[X] = 0\). Then for every \(\lambda > 0\),

\[ \mathbb E\big[e^{\lambda X}\big] \le e^{\lambda^2 (b - a)^2 / 8}. \]

Read it from the inside out. \(X\) is centred, with mean 0, and lives in \([a, b]\), a range of width \(b - a\). \(\lambda\) (the Greek letter lambda) is a positive dial that multiplies \(X\); we will tune it later. \(e^{\lambda X}\) is \(e\) raised to the power \(\lambda X\), and \(\mathbb E\) averages it over the outcomes of \(X\). The right side depends only on the dial and the width. In plain words: **for a centred variable, the average of \(e^{\lambda X}\) is capped by a number that knows only \(\lambda\) and the width of the range.** The example above is \(\lambda = 1\), \(a = -0.3\), \(b = 0.7\).

*Proof.* If \(a = b\), then \(X = 0\) and both sides equal 1, so assume \(a < b\). Since \(X\) has mean 0 and lives in \([a, b]\), we have \(a \le 0 \le b\).

*Step 1, convexity.* The function \(x \mapsto e^{\lambda x}\) is convex: its graph lies below every chord. Every \(x \in [a, b]\) can be written as \(x = \alpha a + (1 - \alpha) b\) with \(\alpha = \frac{b - x}{b - a} \in [0, 1]\), and the chord from \(a\) to \(b\) gives

\[ e^{\lambda x} \le \frac{b - x}{b - a}\, e^{\lambda a} + \frac{x - a}{b - a}\, e^{\lambda b}. \]

The two fractions are weights that add up to 1: \(\frac{b - x}{b - a}\) grows as \(x\) moves toward \(a\), and \(\frac{x - a}{b - a}\) as it moves toward \(b\), so the right side is the straight line from the curve’s value at \(a\) to its value at \(b\). In plain words: **between \(a\) and \(b\), the curve \(e^{\lambda x}\) stays below the straight line joining its two ends.** With \(a = -0.3\), \(b = 0.7\) and \(\lambda = 1\), at \(x = 0\) the curve is \(e^0 = 1\) and the line is \(0.7\,e^{-0.3} + 0.3\,e^{0.7} \approx 1.1227\).

```python
from math import exp

a, b, lam = -0.3, 0.7, 1.0
def chord(x):
    w = (b - x) / (b - a)
    return w * exp(lam * a) + (1 - w) * exp(lam * b)
for x in (-0.3, 0.0, 0.7):
    print(x, round(exp(lam * x), 4), round(chord(x), 4))
```

It prints `-0.3 0.7408 0.7408`, `0.0 1.0 1.1227` and `0.7 2.0138 2.0138`: curve and line meet at the two ends, and the line is above in between.

*Step 2, take expectations.* This holds for every value \(X\) can take, so it holds in expectation. The right side is linear in \(x\), and \(\mathbb E[X] = 0\), so

\[ \mathbb E\big[e^{\lambda X}\big] \le \frac{b}{b - a}\, e^{\lambda a} - \frac{a}{b - a}\, e^{\lambda b}. \]

This is the chord evaluated at the mean of \(X\), which is 0. Our centred loss only ever sits at an end, where curve and line meet, so for it the bound is exact: 1.1227 again.

*Step 3, change variables.* Let \(p = \frac{-a}{b - a} \in [0, 1]\) and \(h = \lambda (b - a)\). Then \(1 - p = \frac{b}{b - a}\) and \(\lambda a = -hp\), so the right side equals \(e^{-hp}(1 - p + p e^{h})\), which is \(e^{L(h)}\) with

\[ L(h) = -hp + \log\big(1 - p + p e^{h}\big). \]

(Check the second term: \(p\,e^{-hp} e^{h} = p\,e^{\lambda a + \lambda(b - a)} = p\,e^{\lambda b}\), and \(p = \frac{-a}{b - a}\).) It remains to show \(L(h) \le h^2/8\), because then \(e^{L(h)} \le e^{\lambda^2 (b - a)^2/8}\).

Here \(\log\) is the natural logarithm, the inverse of raising \(e\) to a power, so \(e^{L(h)}\) undoes it. This \(h\) is a plain number, not a hypothesis; the letter is borrowed for the proof only. For our loss, \(p = 0.3\), and at \(h = 1\) we get \(L(1) \approx 0.116\), below \(1/8 = 0.125\).

*Step 4, three facts about \(L\).* First, \(L(0) = 0 + \log 1 = 0\). Second, with \(q(h) = \frac{p e^h}{1 - p + p e^h}\), the derivative is \(L'(h) = -p + q(h)\), and \(q(0) = p\), so \(L'(0) = 0\). Third, the quotient rule gives \(q'(h) = \frac{p e^h (1 - p + p e^h) - p e^h \cdot p e^h}{(1 - p + p e^h)^2} = \frac{p e^h}{1 - p + p e^h} \cdot \frac{1 - p}{1 - p + p e^h}\), and the two factors are \(q(h)\) and \(1 - q(h)\), so \(q' = q(1 - q)\) and \(L''(h) = q(1 - q)\). Since \(q \in [0, 1]\), and a product \(q(1 - q)\) of two non-negative numbers summing to 1 is at most \(1/4\), we get \(L''(h) \le 1/4\) for every \(h\).

*Step 5, Taylor.* Taylor’s theorem with the Lagrange remainder says that a smooth function equals its value at 0, plus its slope at 0 times \(h\), plus half its curvature somewhere along the way times \(h^2\). With \(L'\) the slope, \(L''\) the curvature, and \(\xi\) (the Greek letter xi) a point between 0 and \(h\) that the theorem promises without naming,

\[ L(h) = L(0) + L'(0)\,h + \tfrac12 L''(\xi)\,h^2 \le 0 + 0 + \tfrac12 \cdot \tfrac14\, h^2 = \frac{h^2}{8}. \qquad \blacksquare \]

Two checks in Python. The first evaluates both sides of the lemma for the centred loss above at five values of \(\lambda\). The second scans \(h^2/8 - L(h)\) over a grid of \(h\) and \(p\) and reports the smallest value.

```python
from math import exp

p = 0.3
for lam in (0.5, 1, 2, 4, 8):
    mgf = (1 - p) * exp(-lam * p) + p * exp(lam * (1 - p))
    print(lam, round(mgf, 4), round(exp(lam**2 / 8), 4))
```

It prints `0.5 1.0282 1.0317`, `1 1.1227 1.1331`, `2 1.6007 1.6487`, `4 5.1442 7.3891` and `8 81.1914 2980.958`. The bound is tight near 0 and loose for large \(\lambda\), and the proof only ever needs the small-\(\lambda\) end.

```python
from math import exp, log

def L(h, p):
    return -h * p + log(1 - p + p * exp(h))

gap = min(h * h / 8 - L(h, p)
          for p in (0.1, 0.3, 0.5, 0.9)
          for h in (k / 10 for k in range(-80, 81)))
print(round(gap, 6))
```

It prints `0.0`: the gap is never negative, and it touches zero only at \(h = 0\), where both sides are 0.

Now the payoff. Hoeffding’s inequality says that when an average is built from \(m\) independent draws that each stay inside a range of width 1, the chance that it misses its mean by more than \(\varepsilon\) is at most twice \(e^{-2m\varepsilon^2}\). For \(h_{11}\)’s row at 60 examples and \(\varepsilon = 0.1\), that is at most \(2e^{-1.2} \approx 0.60\).

**Hoeffding’s inequality.** Let \(Z_1, \dots, Z_m\) be i.i.d. random variables with \(\mathbb E[Z_i] = \mu\) and \(\mathbb P[a \le Z_i \le b] = 1\). Then for every \(\varepsilon > 0\),

\[ \mathbb P\Big[\Big|\frac{1}{m}\sum_{i=1}^m Z_i - \mu\Big| > \varepsilon\Big] \le 2 \exp\big(-2 m \varepsilon^2 / (b - a)^2\big). \]

Read it from the inside out. \(\frac{1}{m}\sum_{i=1}^m Z_i\) is the average of the \(m\) draws, a row of the dashboard when \(Z_i\) is example \(i\)’s loss. Subtracting \(\mu\) and taking \(|\cdot|\) gives how far the average lands from its mean, and the event inside \(\mathbb P[\dots]\) is that this distance exceeds \(\varepsilon\). The condition \(\mathbb P[a \le Z_i \le b] = 1\) just says every draw lands in \([a, b]\). On the right, \(\exp(t)\) is another way to write \(e^t\), used when the exponent is too long to sit up high, and dividing by \((b - a)^2\) rescales for ranges wider or narrower than 1. With width 1 the right side is the \(2e^{-2m\varepsilon^2}\) of the last paragraph, and the 0.60 is the `hoeffding` column of the snippet after the proof.

*Proof.* Centre the variables: \(X_i = Z_i - \mu\), and \(\bar X = \frac1m \sum_i X_i\). Each \(X_i\) has mean 0 and lives in \([a - \mu, b - \mu]\), an interval of the same width \(b - a\).

*Step 1, the upper tail through Markov.* Fix any \(\lambda > 0\). Since \(t \mapsto e^{\lambda t}\) is increasing, the event \(\bar X \ge \varepsilon\) is the same event as \(e^{\lambda \bar X} \ge e^{\lambda \varepsilon}\). The variable \(e^{\lambda \bar X}\) is non-negative, so Markov applies:

\[ \mathbb P[\bar X \ge \varepsilon] = \mathbb P\big[e^{\lambda \bar X} \ge e^{\lambda \varepsilon}\big] \le e^{-\lambda \varepsilon}\, \mathbb E\big[e^{\lambda \bar X}\big]. \]

\(\bar X \ge \varepsilon\) is the upper half of the event we want, and the display is Markov with \(Z = e^{\lambda \bar X}\) and threshold \(e^{\lambda \varepsilon}\), and \(e^{-\lambda\varepsilon}\) is one over that threshold.

*Step 2, independence turns the sum into a product.* The symbol \(\prod_{i=1}^m\) is the product counterpart of \(\sum\): it multiplies the \(m\) factors instead of adding them. Since \(\bar X = \sum_i X_i/m\) and a power of a sum is a product of powers, \(e^{\lambda \bar X} = \prod_i e^{\lambda X_i / m}\). Each factor is a function of \(Z_i\) alone, and functions of independent variables are independent, so the expectation of the product is the product of the expectations:

\[ \mathbb E\big[e^{\lambda \bar X}\big] = \prod_{i=1}^m \mathbb E\big[e^{(\lambda/m) X_i}\big]. \]

This is the step that uses independence, and the only one. In plain words: **for independent draws, the average of a product is the product of the averages.** With two centred losses of \(h_{11}\) and \(\lambda = 1\), there are four outcomes to average over, and Python can check both sides:

```python
from itertools import product
from math import exp

p, lam, m = 0.3, 1.0, 2
X = {1 - p: p, 0 - p: 1 - p}
one = sum(q * exp(lam / m * x) for x, q in X.items())
both = sum(X[x1] * X[x2] * exp(lam * (x1 + x2) / m)
           for x1, x2 in product(X, repeat=2))
print(round(both, 6), round(one**m, 6))
```

It prints `1.057228 1.057228`: averaging over all four outcomes gives the same number as squaring the one-draw average.

*Step 3, Hoeffding’s lemma on each factor*, with \(\lambda/m\) in place of \(\lambda\): each factor is at most \(e^{\lambda^2 (b - a)^2/(8 m^2)}\). Multiplying \(m\) of them,

\[ \mathbb P[\bar X \ge \varepsilon] \le \exp\Big(-\lambda \varepsilon + \frac{\lambda^2 (b - a)^2}{8m}\Big). \]

The exponent is step 1’s \(-\lambda\varepsilon\) plus \(m\) copies of \(\lambda^2(b-a)^2/(8m^2)\), since multiplying powers of \(e\) adds their exponents.

*Step 4, choose \(\lambda\).* The bound holds for every \(\lambda > 0\), so pick the one that makes the exponent smallest. It is a parabola in \(\lambda\), \(-\lambda\varepsilon + c\lambda^2\) with \(c = (b - a)^2/(8m)\); its slope \(-\varepsilon + 2c\lambda\) is zero at \(\lambda = \varepsilon/(2c) = 4 m \varepsilon/(b - a)^2\). Substituting, the exponent becomes \(-\frac{4m\varepsilon^2}{(b-a)^2} + \frac{2m\varepsilon^2}{(b-a)^2} = -\frac{2m\varepsilon^2}{(b-a)^2}\). For \(h_{11}\)’s row, with \(m = 60\), \(\varepsilon = 0.1\) and width 1, the best dial is \(\lambda = 24\) and the exponent is \(-2.4 + 1.2 = -1.2\).

*Step 5, the lower tail.* The variables \(-Z_i\) are also i.i.d., with mean \(-\mu\), in an interval of width \(b - a\). The same four steps give \(\mathbb P[\bar X \le -\varepsilon] \le e^{-2m\varepsilon^2/(b-a)^2}\).

*Step 6, combine.* The event \(|\bar X| > \varepsilon\) is contained in the union of \(\bar X \ge \varepsilon\) and \(\bar X \le -\varepsilon\). By the union bound from lesson 2 (the chance that at least one of several events happens is at most the sum of their chances), its probability is at most the sum of the two, \(2e^{-2m\varepsilon^2/(b-a)^2}\). \(\blacksquare\)

Let’s compare the two bounds with the truth for our row, \(p = 0.3\) and \(\varepsilon = 0.1\). For the 0-1 loss, the number of mistakes is binomial, so the exact probability of being off by more than 0.1 is a finite sum we can compute with fractions, no rounding.

```python
from fractions import Fraction
from math import comb, exp

p, eps = Fraction(3, 10), Fraction(1, 10)
def exact(m):
    return float(sum(comb(m, k) * p**k * (1 - p)**(m - k)
                     for k in range(m + 1)
                     if abs(Fraction(k, m) - p) > eps))
for m in (60, 200, 500):
    chebyshev = float(p * (1 - p) / (m * eps**2))
    hoeffding = 2 * exp(-2 * m * float(eps)**2)
    print(m, f"{exact(m):.1e} {chebyshev:.1e} {hoeffding:.1e}")
```

It prints

```
60 6.6e-02 3.5e-01 6.0e-01
200 1.5e-03 1.0e-01 3.7e-02
500 9.3e-07 4.2e-02 9.1e-05
```

Here is the surprise. At \(m = 60\), Hoeffding is the *worse* of the two bounds: 0.60 against Chebyshev’s 0.35. Hoeffding uses only the range of the loss, while Chebyshev also uses its variance. But look down the columns. Chebyshev creeps down from 0.35 to 0.042; Hoeffding plunges to 0.00009. Each extra example multiplies Hoeffding’s bound by the same factor, \(e^{-2\varepsilon^2} = e^{-0.02}\) here, so it falls exponentially. That shape is what pays for a whole dashboard, as we are about to see. (The exact column falls faster still. Both bounds are cautious; they have to hold for every distribution with that range.)

[[diagram:concentration]]

## From one row to every row

Now we put the two pieces together: Hoeffding for each row, the union bound for the dashboard. With \(\varepsilon = 0.05\), Hoeffding says each row is off by more than 0.05 with chance at most \(2e^{-2m \times 0.0025}\). The union bound says the chance that some row of thirty is off is at most thirty times that. So choose \(m\) to push thirty times that below the 5% of bad weeks we tolerate: 1,419 examples do it, as we are about to compute.

**Theorem (finite classes have uniform convergence).** Let \(\mathcal H\) be a finite hypothesis class, \(Z\) a domain, and \(\ell: \mathcal H \times Z \to [0, 1]\) a loss function. Then \(\mathcal H\) has the uniform convergence property with

\[ m^{\mathrm{UC}}_{\mathcal H}(\varepsilon, \delta) \le \left\lceil \frac{\log(2|\mathcal H|/\delta)}{2 \varepsilon^2} \right\rceil, \]

and therefore \(\mathcal H\) is agnostic PAC learnable by ERM with

\[ m_{\mathcal H}(\varepsilon, \delta) \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta) \le \left\lceil \frac{2 \log(2|\mathcal H|/\delta)}{\varepsilon^2} \right\rceil. \]

Read the first bound from the inside out.

- The type \(\ell: \mathcal H \times Z \to [0, 1]\) says the loss takes a hypothesis and an example and returns a number between 0 and 1; the 0-1 loss is one such.
- \(|\mathcal H|\), bars around a set, is the number of rows.
- \(2|\mathcal H|/\delta\) grows with the class and with the confidence you demand, and \(\log\), the natural logarithm, shrinks it drastically.
- Dividing by \(2\varepsilon^2\) charges for accuracy: halve \(\varepsilon\) and you pay four times as much.
- The brackets \(\lceil \cdot \rceil\) are the ceiling: round up to the next whole number, since you cannot label part of an example.

The second bound is the first evaluated at \(\varepsilon/2\): halving \(\varepsilon\) quarters \(\varepsilon^2\), which turns \(2\varepsilon^2\) in the denominator into \(\varepsilon^2/2\), the same as a 2 on top over \(\varepsilon^2\). In plain words: **for a finite class with losses between 0 and 1, \(\log(2|\mathcal H|/\delta)/(2\varepsilon^2)\) examples, rounded up, make every row \(\varepsilon\)-accurate with probability \(1 - \delta\), and four times as many make ERM \(\varepsilon\)-close to the best.** For the 30-row dashboard with \(\varepsilon = \delta = 0.05\): \(\log 1200 \approx 7.09\), divided by \(2 \times 0.0025 = 0.005\), is 1,418.02, rounded up to 1,419. The `m_uc` snippet after the proof computes exactly this.

*Proof.* Fix \(\varepsilon\), \(\delta\) and any distribution \(\mathcal D\) over \(Z\). We want the representative training sequences to have probability at least \(1 - \delta\). To write that down we need one more piece of notation: for a set \(A\) of training sequences, \(\mathcal D^m(A)\) is the probability that \(S \sim \mathcal D^m\) lands in \(A\). With lesson 1’s set-builder braces, where the colon reads “such that”, the goal is \(\mathcal D^m(\{S : \forall h \in \mathcal H,\ |L_S(h) - L_{\mathcal D}(h)| \le \varepsilon\}) \ge 1 - \delta\). Equivalently, the complement, the set of training sequences that are *not* \(\varepsilon\)-representative, must have probability at most \(\delta\).

*Step 1, name the bad event row by row.* A training sequence fails to be representative exactly when at least one row is off by more than \(\varepsilon\):

\[ \{S : \exists h \in \mathcal H,\ |L_S(h) - L_{\mathcal D}(h)| > \varepsilon\} = \bigcup_{h \in \mathcal H} \{S : |L_S(h) - L_{\mathcal D}(h)| > \varepsilon\}. \]

Two symbols are new. \(\exists\) reads “there exists”, so the left side is the set of training sequences such that some row is off by more than \(\varepsilon\). \(\bigcup_{h \in \mathcal H}\) is a union: everything that belongs to at least one of the sets after it, one set per row. In plain words: **a week is bad exactly when it is bad for at least one row.** The typical week is in the bad set for \(\varepsilon = 0.1\), because it is in row \(h_{18}\)’s, whose luck was 0.132.

*Step 2, the union bound.* The class is finite, so this is a finite union, and its probability is at most the sum:

\[ \mathcal D^m\big(\{S : \exists h \in \mathcal H,\ |L_S(h) - L_{\mathcal D}(h)| > \varepsilon\}\big) \le \sum_{h \in \mathcal H} \mathcal D^m\big(\{S : |L_S(h) - L_{\mathcal D}(h)| > \varepsilon\}\big). \]

The union bound asks nothing about how the events overlap. The rows of a real dashboard are anything but independent (they are computed on the same examples), and that is fine.

Here \(\sum_{h \in \mathcal H}\) adds one term per row, thirty terms for our dashboard. In plain words: **the chance that some row is bad is at most the sum of each row’s chance of being bad.** In our world with \(m = 60\) and \(\varepsilon = 0.1\), Python can estimate both sides, and the total that Hoeffding allows in the next step:

As in the one-row snippet, “off by more than 0.1” is tested as “more than 6 mistakes from the mean”, with the true risks as exact fractions, \(r_j = (49 + j)/200\).

```python
import random
from fractions import Fraction
from math import exp

rng, m, weeks = random.Random(0), 60, 10_000
r = [Fraction(50 + j, 200) for j in range(30)]
per_row, any_row = [0] * 30, 0
for _ in range(weeks):
    k = [rng.binomialvariate(m, float(p)) for p in r]
    bad = [abs(x - m * p) > 6 for x, p in zip(k, r)]
    per_row = [c + b for c, b in zip(per_row, bad)]
    any_row += any(bad)
print(round(any_row / weeks, 3), round(sum(per_row) / weeks, 3))
print(round(30 * 2 * exp(-2 * m * 0.1**2), 2))
```

It prints `0.952 2.792` and then `18.07`: some row is bad in 95% of weeks, the thirty rows’ chances add up to 2.79, and Hoeffding’s total is 18. All three are true, and the last two are useless at 60 examples, since a bound above 1 says nothing. The union bound pays for overlaps it never looks at, and Hoeffding pays for distributions it never sees. Step 5 is where we choose \(m\) to bring the last number below \(\delta\).

*Step 3, Hoeffding for one fixed row.* Fix one \(h\) and set \(\theta_i = \ell(h, z_i)\). Because \(h\) is fixed in advance, before the sample is drawn, each \(\theta_i\) is a function of \(z_i\) alone, so \(\theta_1, \dots, \theta_m\) are i.i.d., because \(z_1, \dots, z_m\) are. Each has expectation \(L_{\mathcal D}(h)\), and each lies in \([0, 1]\), the range of the loss. Their average is \(L_S(h)\). Hoeffding with \(a = 0\), \(b = 1\) gives

\[ \mathcal D^m\big(\{S : |L_S(h) - L_{\mathcal D}(h)| > \varepsilon\}\big) \le 2 e^{-2 m \varepsilon^2}. \]

It is Hoeffding with \(\mu = L_{\mathcal D}(h)\), written as \(\mathcal D^m\) of a set: 0.60 per row at \(m = 60\) and \(\varepsilon = 0.1\).

*Step 4, add up.* Each of the \(|\mathcal H|\) terms of the sum is at most \(2e^{-2m\varepsilon^2}\), so the probability of a non-representative sample is at most \(2 |\mathcal H| e^{-2 m \varepsilon^2}\).

*Step 5, solve for \(m\).* We need \(2|\mathcal H| e^{-2m\varepsilon^2} \le \delta\). Divide by \(2|\mathcal H|\), and take logarithms (both sides are positive and \(\log\) is increasing): \(-2m\varepsilon^2 \le \log(\delta / 2|\mathcal H|) = -\log(2|\mathcal H|/\delta)\). Dividing by \(-2\varepsilon^2\) flips the inequality: \(m \ge \log(2|\mathcal H|/\delta)/(2\varepsilon^2)\). The smallest integer with that property is the ceiling in the statement. For the 30-row dashboard at \(m = 1{,}419\) and \(\varepsilon = 0.05\), the total is \(60\,e^{-7.095} \approx 0.0498\), just under \(\delta = 0.05\).

*Step 6, learning.* The agnostic PAC bound is the corollary of the last section, applied with \(\varepsilon/2\) as worked out above. \(\blacksquare\)

Step 3 is where the winner’s curse is finally defeated, and it is worth seeing why it could not be skipped. You might hope to apply Hoeffding straight to the shipped row, with \(\theta_i = \ell(h_S, z_i)\). But \(h_S\) was chosen by looking at all of \(z_1, \dots, z_m\), so those \(\theta_i\) are not independent draws with mean \(L_{\mathcal D}(h_S)\): the sample that judges \(h_S\) also picked it. The proof never does that. It applies Hoeffding only to hypotheses fixed in advance, every one of them, and then the union bound turns “every row is fine” into a statement that no choice can escape.

[[diagram:union]]

What does that cost? Watch the shape of the bound: \(|\mathcal H|\) enters only through \(\log |\mathcal H|\). That is the exponential tail paying off. Doubling the dashboard adds \(\log 2/(2\varepsilon^2)\) examples, not twice as many. Here is the bill for three dashboards, with \(\varepsilon = 0.05\) and \(\delta = 0.05\): 30 one-feature hypotheses, the 465 hypotheses that use one or two of the thirty features, and the 4,525 that use up to three. The first column is the uniform convergence number for \(\varepsilon\); the second is the agnostic PAC number, which asks for \(\varepsilon/2\).

```python
from math import ceil, log

def m_uc(size, eps, delta):
    return ceil(log(2 * size / delta) / (2 * eps**2))

for size in (30, 465, 4525):
    print(size, m_uc(size, 0.05, 0.05), m_uc(size, 0.025, 0.05))
```

The function `m_uc` is the first bound, term by term: `log(2 * size / delta)` on top, `2 * eps**2` below, and `ceil` for the brackets. It prints `30 1419 5673`, `465 1967 7865` and `4525 2422 9686`. Fifteen times more rows cost 1.4 times more examples; 150 times more rows cost 1.7 times more.

Does the promise hold in our world? Give the 30-row dashboard 1,419 examples a week, and count, over 10,000 weeks, how often some row is off by more than 0.05, and how often the shipped hypothesis is more than \(2\varepsilon = 0.1\) worse than the best.

```python
import random

rng, m, weeks, eps = random.Random(0), 1419, 10_000, 0.05
r = [0.25 + 0.005 * j for j in range(30)]
off = worse = 0
for _ in range(weeks):
    L_S = [rng.binomialvariate(m, p) / m for p in r]
    h_S = min(range(30), key=lambda j: (L_S[j], rng.random()))
    off += max(abs(a - b) for a, b in zip(L_S, r)) > eps
    worse += r[h_S] > min(r) + 2 * eps
print(off / weeks, worse / weeks)
```

It prints `0.0019 0.0`. The theorem allowed 5% of weeks to be non-representative; our world used 0.19%. The bound is safe, and it is also generous: it has to hold for every distribution, and it pays separately for every row even though the rows share their examples. In this world, a bit under 1,000 examples would already have done. The theorem cannot know that, because it cannot see \(\mathcal D\).

## Infinite classes, the engineer’s way

The theorem needs a finite class. Many classes we care about are not finite. Take the domain \(\mathcal X = \mathbb R\), the real numbers (a single number per message, say its length), labels \(\pm 1\), and the **threshold** hypotheses: pick a cutoff, say 500 characters, and call every longer message \(+1\) and every shorter one \(-1\). Written with the cutoff \(\theta\), these are \(h_\theta(x) = \operatorname{sign}(x - \theta)\), where \(\operatorname{sign}\) gives \(+1\) for a positive number and \(-1\) for a negative one; with \(\theta = 500\), a message of length 620 gets \(\operatorname{sign}(120) = +1\). There is one hypothesis for every real \(\theta\), infinitely many, and \(\log |\mathcal H|\) is infinite.

But your computer never stores a real number. It stores \(\theta\) as a 64-bit floating-point number, and there are at most \(2^{64}\) of those. So the class you actually learn has at most \(2^{64}\) members. More generally, a class described by \(d\) numbers is, in practice, a class of at most \(2^{64 d}\) hypotheses. Call this the **discretization trick**. Plugging it into the agnostic PAC bound,

\[ \frac{2 \log(2 \cdot 2^{64d}/\delta)}{\varepsilon^2} = \frac{128\,d \log 2 + 2\log(2/\delta)}{\varepsilon^2} \le \frac{128\,d + 2\log(2/\delta)}{\varepsilon^2}, \]

where the last step uses \(\log 2 < 1\) to get a cleaner formula.

Read it from the inside out. \(2^{64d}\) is how many different settings \(d\) numbers of 64 bits each can take, so it stands in for \(|\mathcal H|\) in the agnostic PAC bound. The logarithm turns powers into products, \(\log(2^{64d}) = 64d \log 2\), and turns the product and quotient inside into a sum, \(\log(2 \cdot 2^{64d}/\delta) = 64d\log 2 + \log(2/\delta)\). The 2 in front doubles both terms, which gives the \(128\,d\log 2\). In plain words: **a class described by \(d\) floating-point numbers needs at most \((128d + 2\log(2/\delta))/\varepsilon^2\) examples, a cost that grows only linearly in \(d\).** For one threshold, \(d = 1\), with \(\varepsilon = 0.1\) and \(\delta = 0.05\): \((128 + 2\log 40)/0.01 \approx (128 + 7.38)/0.01 \approx 13{,}538\). In the snippet, `size` is \(2^{64d}\), the first print is the left side of the display and the second is the right.

```python
from math import log

d, eps, delta = 1, 0.1, 0.05
size = 2 ** (64 * d)
print(round(2 * log(2 * size / delta) / eps**2))
print(round((128 * d + 2 * log(2 / delta)) / eps**2))
```

It prints `9610` and `13538`: about ten thousand examples for one threshold to within 0.1 of the best, with 95% confidence, and the cleaner formula rounds that up to about thirteen and a half thousand. It is a rough estimate, and it has an embarrassing feature. The answer depends on how your machine stores numbers, which has nothing to do with learning. A 32-bit machine would “need” only about half as many examples for the same class.

[[diagram:discretization]]

Lesson 6 replaces this trick with a measure of a class’s size that does not care about bits.

## The dashboard as code

Here is the procedure the team should have run from the start. The new line is the first one: the number of examples is computed from the promise, before any data arrives.

```
ShipFromDashboard(H, ε, δ):
  m ← ⌈ log(2|H|/δ) / (2ε²) ⌉
  S ← m fresh labelled examples z_1, ..., z_m
  for each h in H:
    dashboard[h] ← (1/m) · Σ_{i=1..m} ℓ(h, z_i)
  h_S ← an h with the smallest dashboard[h]
  return h_S, with the promises
    L_D(h_S) ≤ dashboard[h_S] + ε
    L_D(h_S) ≤ min_{h ∈ H} L_D(h) + 2ε
    each holding with probability at least 1 − δ
```

**Time.** Filling the dashboard takes \(m |\mathcal H|\) loss evaluations; if one evaluation costs \(c\), that is \(O(m |\mathcal H| c)\) time, and with \(m\) from the first line, \(O\big(|\mathcal H| \log(|\mathcal H|/\delta)\, c / \varepsilon^2\big)\). Notice the two costs pulling apart. The number of examples grows like \(\log |\mathcal H|\), but the computation grows like \(|\mathcal H|\). The 30-row dashboard at 1,419 examples is 42,570 evaluations; the 4,525-row one at 2,422 examples is 10,959,550. Statistically, big classes are cheap. Computationally, they are not, and lesson 8 is about that gap.

**Space.** \(O(m)\) for the examples and \(O(|\mathcal H|)\) for the dashboard. If you only need the winner, you can stream through \(\mathcal H\) keeping the best so far, and the dashboard shrinks to \(O(1)\) extra space; the team keeps all rows because they like to look at them, which is how the winner’s curse fooled them in the first place.

**Build it yourself.** The starter data is the noisy mail room of lesson 3: four features FLUN, every message equally likely, and the probability that a message is junk is 0.9 when it says “FREE”, 0.7 for the two newsletters 0110 and 0111, and 0.1 otherwise. The twelve examples are lesson 3’s.

```python
import itertools
import random

S = [("1101", 1), ("0000", 0), ("0110", 1), ("1010", 1),
     ("0011", 0), ("0110", 0), ("1100", 0), ("0101", 0),
     ("1111", 1), ("0001", 0), ("0111", 1), ("1000", 1)]
X = ["".join(b) for b in itertools.product("01", repeat=4)]
H = [lambda x, j=j: int(x[j]) for j in range(4)]

def eta(x):
    if x[0] == "1":
        return 0.9
    return 0.7 if x in ("0110", "0111") else 0.1

def empirical_risk(h, S): ...
def true_risk(h): ...
def erm(H, S): ...
def is_representative(H, S, eps): ...
def m_uc(size, eps, delta): ...
def draw(m, rng): ...

if __name__ == "__main__":
    true = [round(true_risk(h), 3) for h in H]
    assert true == [0.175, 0.425, 0.425, 0.5]
    seen = [round(empirical_risk(h, S), 4) for h in H]
    assert seen == [0.25, 0.4167, 0.3333, 0.5]
    assert erm(H, S) == 0
    assert is_representative(H, S, 0.1)
    assert not is_representative(H, S, 0.09)
    assert m_uc(4, 0.1, 0.05) == 254
    assert m_uc(30, 0.05, 0.05) == 1419
```

1. Implement the six functions. `true_risk` averages, over the 16 messages, the probability that \(h\) is wrong there: \(\eta(x)\) when \(h\) says keep, \(1 - \eta(x)\) when it says junk. `erm` returns the index of a hypothesis with the smallest empirical risk. `draw` returns \(m\) examples drawn i.i.d. from the noisy mail room.
2. Which row decides that the twelve examples are 0.1-representative but not 0.09-representative? What is its luck?
3. Draw 2,000 training sequences of 254 examples each and measure the fraction that are not 0.1-representative. The theorem promises at most 5%. Predict, then measure.
4. Reproduce the winner’s curse table: for the 30-row world of this lesson, simulate a week, sort the dashboard, and print the luck of the top rows.

## Fix the bug

Back to Monday. Your manager wants two lines on every release note, and wants them to be true: “production will find this hypothesis at most \(\varepsilon\) worse than the dashboard says” and “it is at most \(2\varepsilon\) worse than the best hypothesis on the dashboard”, both with confidence \(1 - \delta\). The team agrees on \(\varepsilon = 0.05\) and \(\delta = 0.05\).

In the lab, you set the number of examples each week. Labels cost people’s time, so there is a budget. Each round, the dashboard changes size: the team keeps adding hypotheses, from 30 rows to 465 to 4,525. For each round, choose the smallest number of examples that makes the whole dashboard \(\varepsilon\)-representative with probability at least \(1 - \delta\), then run a few hundred weeks and watch the shipped hypothesis in production. Before the first round, a prediction: when the dashboard grows fifteenfold, from 30 rows to 465, how many more examples will you need?

[[lab-act2]]

## What people get wrong

**“Every row is an unbiased estimate, so the winner’s row is too.”** Unbiasedness is a property of a hypothesis fixed before the data. The winner is chosen by the data, and the proposition of this lesson shows its empirical risk is, on average, at most the best true risk in the class, while its true risk is at least that. In our world: 0.18 shown, 0.28 delivered.

**“More hypotheses on the dashboard can only help: more choice, better winner.”** More choice also means more chances for a lucky row to win. In the lab, with true risks spread evenly from 0.25 to 0.40 and 60 examples a week, growing the dashboard from 30 rows to 465 pushes the winner’s empirical risk down from about 0.18 to about 0.13 while its true risk stays near 0.27. The cure costs examples, but only \(\log |\mathcal H|\) of them.

**“The theorem says I need 1,419 examples.”** It says 1,419 examples are *enough*, for every distribution. It is a sufficient sample size, not a necessary one. In our particular world, about 900 would have kept the non-representative weeks under 5%. You use the theorem because you cannot see \(\mathcal D\), not because it is tight.

**“Uniform convergence is just the law of large numbers.”** The law of large numbers is about one average, and only in the limit. Uniform convergence is about all the averages in the class at once, at a finite \(m\), with a number attached. That is why we needed Hoeffding’s exponential tail and the union bound, not just the limit.

## Where this comes from

Classes for which uniform convergence holds are also called Glivenko-Cantelli classes, after Valery Glivenko and Francesco Cantelli, who proved the first uniform convergence result in the 1930s. The connection between uniform convergence and learnability was studied in depth by Vapnik. Lesson 6 will show that for binary classification, uniform convergence is not only sufficient for learnability but also necessary. For more general learning problems that is not the case.

## Recap and the next bug

1. A single row of the dashboard, a hypothesis fixed before the data, is an unbiased estimate of its true risk. The row you pick for being smallest is not: on average it shows at most the best true risk in the class and delivers at least that. This is the **winner’s curse**.
2. A training sequence is **\(\varepsilon\)-representative** when every hypothesis’s empirical risk is within \(\varepsilon\) of its true risk. On such a sequence, ERM ships a hypothesis whose true risk is at most its empirical risk plus \(\varepsilon\), and at most the best in the class plus \(2\varepsilon\).
3. A class has the **uniform convergence property** when some sample size, the same for every distribution, makes the sample \(\varepsilon\)-representative with probability at least \(1 - \delta\). Uniform convergence implies agnostic PAC learnability by ERM, with \(m_{\mathcal H}(\varepsilon, \delta) \le m^{\mathrm{UC}}_{\mathcal H}(\varepsilon/2, \delta)\).
4. **Hoeffding’s inequality** bounds how far an average of \(m\) i.i.d. bounded variables can stray, with a tail \(2e^{-2m\varepsilon^2/(b-a)^2}\) that falls exponentially in \(m\). Its proof is Markov’s inequality, applied to \(e^{\lambda \bar X}\), plus independence and Hoeffding’s lemma.
5. Hoeffding for each fixed hypothesis plus the union bound give every finite class uniform convergence with \(m \ge \log(2|\mathcal H|/\delta)/(2\varepsilon^2)\), and agnostic PAC learnability with \(2\log(2|\mathcal H|/\delta)/\varepsilon^2\). The class size enters only through its logarithm. The discretization trick stretches this, roughly, to classes described by \(d\) floating-point numbers.

Your dashboard is honest now. Then, at the Friday demo, a colleague stands up with a bolder plan. Why choose a class at all? Why 30 rows, or 4,525? Her “universal filter” has no bias whatsoever: it considers every possible way of labelling messages, and she says that, given enough data, it will learn whatever rule junk follows. You run her class through this lesson’s formula, with \(\varepsilon = \delta = 0.05\). With thirty yes/no features there are \(2^{2^{30}}\) hypotheses, and making that dashboard 0.05-representative asks for about 149 billion examples, more than a hundred times the number of distinct messages that even exist. But a bound that is too big only shows that this proof fails. It does not show that her filter fails. Could a learner with no bias at all succeed, given enough data? Or is there a reason, deeper than any bound, why no such filter can exist? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A teammate proposes a shortcut: “Skip the union bound. The shipped hypothesis’s empirical risk is an average of \(m\) losses in \([0, 1]\), so Hoeffding alone gives \(\mathbb P[|L_S(h_S) - L_{\mathcal D}(h_S)| > \varepsilon] \le 2e^{-2m\varepsilon^2}\), however big the dashboard is.” Which hypothesis of Hoeffding’s inequality fails, and why? Then use the lab to find a setting where the shortcut’s promise is visibly broken.
2. The lemma says an \(\varepsilon\)-representative sample makes ERM at most \(2\varepsilon\) worse than the best. (a) Build a dashboard of two hypotheses and describe a training sequence that is 0.05-representative, on which ERM may ship a hypothesis exactly 0.10 worse than the best. So is the factor 2 ever reached? (b) If the dashboard has a single row, what do the lemma and the theorem say, and is there any winner’s curse?
3. The team’s dashboard has 4,525 hypotheses and they now want \(\delta = 0.01\). How many examples make the dashboard 0.05-representative? How many does the theorem ask for to ship a hypothesis within 0.05 of the best? And if they doubled the dashboard to 9,050 rows, how many more examples would the first number need?
