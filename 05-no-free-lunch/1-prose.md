# No Free Lunch

## The universal filter

Theo from the platform team has been reading your design notes, and he is not impressed. “All this care about hypothesis classes,” he says, scrolling past lesson 1’s one-feature hypotheses and lesson 3’s agnostic guarantees. “You keep choosing what the learner is allowed to believe. Why decide in advance? I’ve built a filter with no bias at all. It considers every hypothesis there is, every possible way of labelling every possible message. It never rules anything out, so it can never be wrong about what junk looks like. Give it enough labelled mail and it will learn any junk rule the spammers come up with, including ones none of us has imagined.”

He wants to replace the course’s careful classes with his **universal filter**, and the pitch is attractive. Every class you have chosen so far was a bet, and lesson 1 showed you what happens when the bet is wrong: the rat, whose bias blinds it to a world where a sound causes sickness. Theo’s filter makes no bet, so it can never lose one.

You have a nagging feeling that lesson 1 already answered him. It proved that when a learner considers every hypothesis, the examples leave every unseen message perfectly balanced: exactly half of the consistent hypotheses call it junk. But read lesson 1 again and you will find that it stopped short. It said that a learner with no bias has nothing to go on, and it also admitted that a learner whose tie-breaker happens to match the truth would get lucky. Theo is not claiming his filter is lucky. He is claiming that some clever tie-breaker, some rule for labelling unseen messages, works well on every task. Maybe there is one, and we just haven’t found it.

So here is the question this lesson settles. Is there a **universal learner**: one learner, and one number of examples \(m\), such that on every learning task, given \(m\) examples, it probably returns a hypothesis with small true risk? If there is, Theo wins and the course can stop worrying about bias. If there isn’t, we owe him a proof, and we owe ourselves an account of what a bias costs and what it buys.

## Where we stand

A short recap of the tools we already own, since the proof will lean on every one of them, and since this lesson uses their symbols on almost every line, we read each formula again.

Since lesson 3 a **learning task** is the way labelled mail arrives: some messages come often, some rarely, and each comes with its label. Formally it is a distribution \(\mathcal D\) over pairs \((x, y)\) in \(\mathcal X \times \{0, 1\}\). Here \(\mathcal X\) is the **domain**, the set of every message that could ever arrive, and \(\mathcal X \times \{0, 1\}\) is the set of all pairs of a message and a label, with 1 for junk and 0 for keep. The **true risk** of a hypothesis \(h\) is the chance that it mislabels the next pair to arrive:

\[ L_{\mathcal D}(h) = \mathbb P_{(x,y) \sim \mathcal D}[h(x) \neq y]. \]

Read it from the inside out.

- \(h(x) \neq y\) is the event that \(h\)’s label for the message \(x\) differs from its true label \(y\): a mistake.
- \((x,y) \sim \mathcal D\) reads “the pair \((x, y)\) is drawn from \(\mathcal D\)”: a fresh labelled message arrives at random.
- \(\mathbb P[\dots]\) is the probability of the event inside the brackets, and its subscript says what is random.
- \(L\) stands for loss, and the subscript \(\mathcal D\) in \(L_{\mathcal D}\) says the score is taken over everything \(\mathcal D\) can send, not over the examples we happen to hold.

In plain words: **the true risk of \(h\) is the probability that \(h\) gets a fresh, randomly drawn example wrong.** Take the four-message world of the next section: the messages 00, 01, 10 and 11, each equally likely, labelled junk exactly when the first bit (FREE) is on. Let \(h\) say junk exactly when the second bit (a link) is on. The two disagree on 01 and 10, two of the four, so \(L_{\mathcal D}(h) = 2/4 = 0.5\).

Counting a mistake as 1 and a correct label as 0 is the **0-1 loss**, the only loss in this lesson. The training sequence \(S = ((x_1, y_1), \dots, (x_m, y_m))\) is \(m\) labelled examples drawn **i.i.d.**, independently and identically distributed: each from the same \(\mathcal D\), and no draw influencing another. We write that as \(S \sim \mathcal D^m\), where the power \(m\) on \(\mathcal D\) names the whole \(m\)-fold draw.

Lesson 2 studied **ERM**, the learner that returns a hypothesis with the fewest mistakes on \(S\). Run over a finite class \(\mathcal H\) that contains the truth, ERM reaches true risk at most \(\varepsilon\) with probability at least \(1 - \delta\) once

\[ m \ge \frac{\log(|\mathcal H|/\delta)}{\varepsilon}. \]

Here \(|\mathcal H|\) is how many hypotheses the class holds, \(\varepsilon\) is the error we are willing to tolerate, \(\delta\) is the chance of failure we are willing to tolerate, and \(\log\) is the natural logarithm, base \(e \approx 2.718\). Read from the inside: divide the class size by \(\delta\), take the log, and divide by \(\varepsilon\). In plain words: **enough examples means the log of the class size over the failure chance, divided by the tolerated error.** Lesson 3 called a class with a guarantee of this kind **PAC learnable**. Lesson 4 dropped the assumption that the class contains the truth: with

\[ m \ge \frac{2\log(2|\mathcal H|/\delta)}{\varepsilon^2} \]

examples, ERM comes within \(\varepsilon\) of the **best in the class**, \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\). The \(\min\) with a subscript reads “the smallest value as \(h\) ranges over \(\mathcal H\)”, so this is the smallest true risk any member of the class achieves. The new formula has the same log, a 2 inside it and a 2 in front, and a square on \(\varepsilon\). In plain words: **without the truth in the class, getting within \(\varepsilon\) of the best member costs about \(1/\varepsilon^2\) examples instead of \(1/\varepsilon\).** For the class of all 16 labellings of the four messages, with \(\varepsilon = \delta = 0.1\), the first formula asks for 51 examples and the second for 1,154. Here are the true risk and both formulas in Python:

```python
import math

C = ["00", "01", "10", "11"]
f = dict(zip(C, (0, 0, 1, 1)))
h = dict(zip(C, (0, 1, 0, 1)))
L_D = sum(h[x] != f[x] for x in C) / len(C)
size, eps, delta = 2 ** 4, 0.1, 0.1
m_realizable = math.log(size / delta) / eps
m_agnostic = 2 * math.log(2 * size / delta) / eps ** 2
print(L_D, math.ceil(m_realizable), math.ceil(m_agnostic))
```

It prints `0.5 51 1154`. Every one of those guarantees starts with “fix a class”. Theo’s question is whether we needed to.

## A duel on four messages

Let’s shrink the mail room until we can check everything by hand. Keep only two features, F (the subject says “FREE”) and L (the message has a link), so there are four possible messages: 00, 01, 10 and 11. Each training message is drawn uniformly from these four, and we draw \(m = 2\) of them.

Now set up a duel. Theo picks a learner. You pick the **target**, the labelling function \(f\) that decides which of the four messages are junk. Then the training sequence is drawn, his learner returns a hypothesis, and we score its true risk. You are allowed to know his learner before you pick, which is the whole point: you are looking for a task on which it fails.

Theo’s first learner is lesson 1’s memorizer: it returns the label of every message it has seen, and says keep for every message it hasn’t. Which target would you pick against it? Think about it before you read on.

The memorizer’s weakness is its default. It says keep for every unseen message, so pick the target that labels every message junk. Then the memorizer is right on what it has seen and wrong on everything else, and its true risk is simply the fraction of the four messages it hasn’t seen. There are \(4 \times 4 = 16\) equally likely training sequences, and they come in two kinds.

| training sequence | how many | unseen messages | true risk |
|---|---|---|---|
| the same message twice, like (01, 01) | 4 | 3 | 3/4 |
| two different messages, like (01, 10) | 12 | 2 | 2/4 |

Averaging over the 16, the expected true risk is \((4 \cdot \tfrac34 + 12 \cdot \tfrac24)/16 = 9/16 = 0.5625\). Worse than a coin.

Read that from the inside out. Each fraction is a true risk from the table, and the number in front of it is how many training sequences produce it, so \(4 \cdot \tfrac34 + 12 \cdot \tfrac24\) adds up the true risk over all 16 sequences, 3 plus 6, which is 9. Dividing by 16 turns that total into an average, and since the 16 sequences are equally likely, that average is the expected value. In plain words: **the memorizer’s expected true risk on this target is its true risk averaged over the 16 equally likely training sequences.** The snippet a little further down computes exactly this average in its function `expected_risk`, which loops over the 16 sequences and the four messages and divides by \(16 \cdot 4\).

Theo shrugs and patches it. His second learner fills in unseen messages by majority: if most of the labels it has seen are junk, it says junk for the unseen ones, and otherwise keep. Against “everything is junk” it is now perfect, since it sees two junk labels and copies them everywhere. So you change targets. Try “junk exactly when the subject says FREE”: 00 and 01 are keep, 10 and 11 are junk. Now whenever both training messages share a label, majority copies that label onto the unseen messages of the other kind, and gets them wrong. The bookkeeping is a little longer, so let Python do it. The snippet scores both learners on all 16 possible targets, writing a target as the tuple \((f(00), f(01), f(10), f(11))\), and prints each learner’s average over the targets, its worst score, and the target that achieves it.

```python
import itertools

C = ["00", "01", "10", "11"]
targets = list(itertools.product([0, 1], repeat=4))
sequences = list(itertools.product(C, repeat=2))

def memorizer(S):
    return lambda x: dict(S).get(x, 0)

def majority(S):
    vote = int(2 * sum(y for _, y in S) > len(S))
    return lambda x: dict(S).get(x, vote)

def expected_risk(A, f):
    g = dict(zip(C, f))
    return sum(A([(x, g[x]) for x in xs])(c) != g[c]
               for xs in sequences for c in C) / (16 * 4)

for A in (memorizer, majority):
    r = {f: expected_risk(A, f) for f in targets}
    worst = max(r, key=r.get)
    print(sum(r.values()) / 16, r[worst], worst)
```

It prints

```
0.28125 0.5625 (1, 1, 1, 1)
0.28125 0.375 (0, 0, 1, 1)
```

The memorizer’s worst target is “everything is junk”, at 0.5625, the number we computed by hand. Majority’s worst is “junk exactly when FREE”, at 0.375. Majority is the better learner in the worst case, but you still found a task where it is wrong more than a third of the time.

Now look at the first column. Both learners score exactly 0.28125 when you average over all 16 targets. Two different learners, with different weaknesses and different worst cases, and the same average to five decimal places. Is that a coincidence?

## Lab: beat the adversary

Try to break the pattern yourself. The lab below has a slightly larger world: three features, F, L and U, so eight messages, and \(m = 4\) training messages drawn uniformly. You choose how Theo’s learner labels the messages it hasn’t seen: always keep, always junk, the majority of the labels it has seen, or the label of the nearest seen message (the one that differs in the fewest features). The lab then plays the adversary for you. It computes the expected true risk on every one of the \(2^8 = 256\) targets, exactly, by going through all \(8^4 = 4{,}096\) training sequences, and it draws the 256 results as a strip, with the average marked and the worst target highlighted.

Before you press anything, predict: which of the four rules has the lowest average over all 256 targets?

[[lab]]

## Why every learner has the same average

They tie. Every rule averages 0.2931 over the 256 targets, and every rule has a target that pushes it well above a quarter. Here is why, and you have seen the trick before.

Fix one training sequence, say it contains the messages 000, 011 and 101 (one of them twice), and fix one message it doesn’t contain, say 110. Now take any target \(f\) and its **twin**: the target that agrees with \(f\) everywhere except at 110, where its label is flipped. On the training sequence, the twins are indistinguishable. They give the same labels to 000, 011 and 101, so the learner receives exactly the same examples under both, and so it returns exactly the same hypothesis. That hypothesis says one thing about 110, either junk or keep. One twin says junk there and the other says keep, so exactly one of the twins catches the learner out.

[[diagram:pairing]]

That is lesson 1’s flip map, the one that paired consistent hypotheses by flipping one unseen label, now applied to targets instead of hypotheses. Every target has exactly one twin at 110, so the 256 targets split into 128 twin pairs, and in every pair the learner is wrong at 110 exactly once. Averaged over all targets, the learner is wrong at 110 exactly half the time. And nothing in that argument cared what the learner does. Keep, junk, majority, nearest, a neural network, a crystal ball: as long as it is a function of the examples it sees, the twins defeat it at every unseen message half the time.

So, averaged over all targets, the expected true risk of any learner that gets its seen messages right is half the expected fraction of messages it hasn’t seen. (A learner that also errs on seen messages only does worse.) That fraction depends on how many distinct messages turn up in the training sequence, and not on the learner at all, which is why every rule in the lab got the same number.

Two more facts finish the argument, and they are the heart of this lesson.

First, if the training sequence has \(m\) messages and the domain has \(2m\), then at most \(m\) distinct messages can appear, so at least half are unseen. Half of “at least half” is at least a quarter. Averaged over all targets, every learner’s expected true risk is at least \(1/4\).

Second, **the maximum is at least the average**. If the average over targets is at least \(1/4\), then some single target has expected risk at least \(1/4\), since not every number in a list can be below the list’s average. That target is a task on which the learner fails.

[[diagram:averaging]]

The quarter is a floor, not the exact value. The exact average is half the expected fraction of unseen messages, so let’s work that fraction out. In the duel, one draw lands on the message 01 with chance \(1/4\) and misses it with chance \(3/4\). The two draws are independent, so the chance that both miss it is \(3/4 \cdot 3/4 = 9/16\). Every message has that same chance of going unseen, so on average a fraction \(9/16\) of the four messages is unseen, and half of that is \(9/32 = 0.28125\), the duel’s average. In general, when each of the \(m\) draws is uniform over \(2m\) messages, a given message is missed by all of them with probability

\[ \left(1 - \frac{1}{2m}\right)^m. \]

Read it from the inside out. \(\frac{1}{2m}\) is the chance that one uniform draw lands on the given message, so \(1 - \frac{1}{2m}\) is the chance that it misses. The power \(m\) multiplies that chance by itself once per draw, which is how independent chances combine. In plain words: **the chance that a given message is never drawn is the chance that one draw misses it, multiplied together over all \(m\) draws.** With \(m = 2\) it is \((3/4)^2 = 0.5625\), as above. In the snippet below, `formula` is this expression divided by 2, and `average_over_targets` is the brute force it should match. Let’s check them against each other, using the memorizer as the learner, for domains of 2, 4, 6 and 8 messages.

```python
import itertools

def average_over_targets(m):
    C = range(2 * m)
    targets = list(itertools.product([0, 1], repeat=2 * m))
    sequences = list(itertools.product(C, repeat=m))
    wrong = sum(f[c] != (f[c] if c in xs else 0)
                for f in targets for xs in sequences for c in C)
    return wrong / (len(targets) * len(sequences) * len(C))

for m in (1, 2, 3, 4):
    formula = (1 - 1 / (2 * m)) ** m / 2
    brute = average_over_targets(m)
    print(m, round(brute, 4), round(formula, 4))
print(round(0.5 * 2.718281828 ** -0.5, 4))
```

It prints

```
1 0.25 0.25
2 0.2812 0.2812
3 0.2894 0.2894
4 0.2931 0.2931
0.3033
```

The brute force matches the formula: 0.28125 is the four-message duel and 0.2931 is the lab. As \(m\) grows the average climbs toward \(e^{-1/2}/2 \approx 0.3033\), never below a quarter. Here \(e \approx 2.718\) is the base of the natural logarithm, and \(e^{-1/2} = 1/\sqrt e \approx 0.6065\) is where \((1 - \frac{1}{2m})^m\) settles as \(m\) grows: a miss chance that creeps toward 1, raised to a power that grows just as fast, tends to this fixed number. In plain words: **when the sample and the domain grow together, the average over targets settles at half of \(e^{-1/2}\).** The last line of the snippet computes exactly that, \(0.5 \cdot 2.718281828^{-0.5}\), and prints 0.3033. A learner that has seen half the domain is, averaged over all the ways the world could be, wrong on more than a quarter of it.

This is the **No-Free-Lunch theorem**, which lesson 1 promised. Let’s state it properly and prove it properly.

## The formal statements

**Setting.** We are doing binary classification with the 0-1 loss over a domain \(\mathcal X\). A learning task is a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\), and \(L_{\mathcal D}(h) = \mathbb P_{(x,y) \sim \mathcal D}[h(x) \neq y]\). A learner \(A\) is a deterministic algorithm: it maps each training sequence \(S\) to a hypothesis \(A(S): \mathcal X \to \{0, 1\}\), so the same \(S\) always gives the same hypothesis. The notation \(A(S): \mathcal X \to \{0, 1\}\) reads “\(A(S)\) is a function that takes a message in \(\mathcal X\) and returns 0 or 1”: the arrow points from what goes in to what comes out. A **universal learner** would be a learner \(A\) and a training sequence size \(m\) such that for every task \(\mathcal D\), given \(S \sim \mathcal D^m\), \(A(S)\) has small true risk with high probability.

Before the statement, here is what it says in words. The averaging argument found, for every learner, a target that pushes its expected true risk to at least a quarter. An expectation is a statement about all training sequences at once, and you only ever get one. So the theorem trades the quarter for a pair of numbers about the one sequence you get: with a chance of at least one in seven, the learner is wrong on at least an eighth of the mail. Step 7 of the proof shows where the \(1/8\) and the \(1/7\) come from.

**No-Free-Lunch theorem.** Let \(A\) be any learner for binary classification with the 0-1 loss over a domain \(\mathcal X\), and let \(m\) be any training sequence size smaller than \(|\mathcal X|/2\). Then there is a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\) such that:

1. there is a function \(f: \mathcal X \to \{0, 1\}\) with \(L_{\mathcal D}(f) = 0\), and
2. with probability at least \(1/7\) over the choice of \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \ge 1/8\).

Read the order of the quantifiers carefully, because it carries the whole meaning. The learner comes first; the task is then chosen to defeat it. The task is perfectly realizable, since \(f\) labels it with zero error, so a different learner (ERM over the one-hypothesis class \(\{f\}\), which simply returns \(f\)) would solve it with no examples at all. What fails is not the task. It is the attempt to solve every task with one learner.

Now read the statement symbol by symbol.

- \(|\mathcal X|\) is how many messages the domain holds, so \(m < |\mathcal X|/2\) says the training sequence is shorter than half the domain. With the lab’s eight messages, \(m\) can be 1, 2 or 3.
- \(f: \mathcal X \to \{0, 1\}\) is a labelling of every message, and \(L_{\mathcal D}(f) = 0\) says it never makes a mistake under \(\mathcal D\): the task is **realizable**, perfectly solved by one function.
- \(A(S)\) is the hypothesis the learner returns on the training sequence \(S\), and \(L_{\mathcal D}(A(S)) \ge 1/8\) is the event that it is wrong on at least an eighth of the mail.
- “With probability at least \(1/7\) over the choice of \(S \sim \mathcal D^m\)” says that, weighting each training sequence by its chance of being drawn, at least a seventh of the weight sits on sequences where that event happens.

In plain words: **for every learner and every sample shorter than half the domain, there is a task that one labelling solves perfectly, on which the learner is wrong on at least an eighth of the mail at least one time in seven.** The duel shows it in miniature. Its \(m = 2\) is exactly half of its four messages, the boundary case that the proof itself works with, since all the proof needs is \(2m\) messages to spread the tasks over. Against the memorizer, the target “everything is junk” leaves at least two of the four messages unseen on every training sequence, so the true risk is at least \(1/2\), far above \(1/8\), every time:

```python
import itertools

C, m = ["00", "01", "10", "11"], 2
f = dict(zip(C, (1, 1, 1, 1)))

def memorizer(S):
    return lambda x: dict(S).get(x, 0)

hits = 0
for xs in itertools.product(C, repeat=m):
    h = memorizer([(x, f[x]) for x in xs])
    risk = sum(h(c) != f[c] for c in C) / len(C)
    hits += risk >= 1 / 8
print(hits / len(C) ** m, hits / len(C) ** m >= 1 / 7)
```

It prints `1.0 True`: the bad event has probability 1, comfortably at least \(1/7\).

*Proof.* We go in seven steps. Steps 1 and 2 set up a family of tasks; steps 3 to 6 show that, on average over that family, the learner’s expected true risk is at least \(1/4\); step 7 turns that average into the probability statement.

**Step 1: a family of tasks on a finite piece of the domain.** Since \(m < |\mathcal X|/2\), the domain has at least \(2m\) elements, so we can pick a subset \(C \subseteq \mathcal X\) with \(|C| = 2m\) (the symbol \(\subseteq\) reads “is a subset of”). The intuition is the one from the lab: a learner that observes at most half of \(C\) has no information about the labels of the rest. There are \(T = 2^{2m}\) functions from \(C\) to \(\{0, 1\}\), since each of the \(2m\) messages gets one of two labels; call them \(f_1, \dots, f_T\). In the duel, \(C\) is the four messages and \(T = 2^4 = 16\), the 16 targets of the first snippet.

Each target becomes a task in the simplest possible way: pick a message uniformly from \(C\), and label it by the target. For the duel’s target “junk exactly when FREE”, that task sends the pair (10, junk) a quarter of the time and never sends (10, keep). Written out, for each \(i\), let \(\mathcal D_i\) be the distribution over \(C \times \{0, 1\}\) given by

\[ \mathcal D_i(\{(x, y)\}) = \begin{cases} 1/|C| & \text{if } y = f_i(x), \\ 0 & \text{otherwise.} \end{cases} \]

- \(\mathcal D_i(\{(x, y)\})\) is the probability that task \(i\) sends exactly the pair \((x, y)\).
- The large brace lists **cases**: use the top line when its condition holds, and the bottom line otherwise.
- \(1/|C|\) is one message’s share when all \(|C|\) messages are equally likely, and \(y = f_i(x)\) says the pair carries the label the target gives it.

In plain words: **task \(i\) sends every message of \(C\) equally often, always with the label \(f_i\) gives it.** Clearly \(L_{\mathcal D_i}(f_i) = 0\). Under \(\mathcal D_i\), a hypothesis \(h\) makes a mistake exactly when the message drawn is one where \(h\) and \(f_i\) disagree, so its true risk is the fraction of \(C\) on which they disagree:

\[ L_{\mathcal D_i}(h) = \frac{1}{2m} \sum_{x \in C} \mathbb 1[h(x) \neq f_i(x)]. \]

- \(\mathbb 1[\cdot]\), the indicator from lesson 2, is 1 when the statement inside is true and 0 when it is false, so \(\mathbb 1[h(x) \neq f_i(x)]\) is 1 exactly when \(h\) gets \(x\) wrong.
- \(\sum_{x \in C}\) adds that up over every message \(x\) in \(C\), which counts the disagreements.
- \(\frac{1}{2m}\) divides the count by the number of messages, \(|C| = 2m\).

In plain words: **under task \(i\), the true risk of \(h\) is the fraction of the \(2m\) messages on which \(h\) and \(f_i\) disagree.** With \(f_i\) “junk exactly when FREE” and \(h\) “junk exactly when there is a link”, they disagree on 01 and 10, so the risk is \(2/4 = 0.5\), the number from the recap. The snippet builds \(\mathcal D_i\) as a table of probabilities, then computes the risk twice: by adding the probabilities of the pairs \(h\) gets wrong, and by the counting formula.

```python
C, m = ["00", "01", "10", "11"], 2
f_i = dict(zip(C, (0, 0, 1, 1)))
h = dict(zip(C, (0, 1, 0, 1)))
D_i = {(x, y): (1 / len(C) if y == f_i[x] else 0)
       for x in C for y in (0, 1)}
print(sum(D_i.values()), D_i[("10", 1)], D_i[("10", 0)])
by_pairs = sum(p for (x, y), p in D_i.items() if h[x] != y)
by_count = sum(h[x] != f_i[x] for x in C) / (2 * m)
print(by_pairs, by_count)
```

It prints `1.0 0.25 0` and `0.5 0.5`: the probabilities add up to 1, (10, junk) gets a quarter and (10, keep) nothing, and the two ways of computing the risk agree.

**Step 2: the expectation is an average over sequences.** There are \(k = (2m)^m\) sequences of \(m\) messages from \(C\), since each of the \(m\) positions holds one of \(2m\) messages; in the duel that is \(4^2 = 16\). Call them \(S_1, \dots, S_k\). If \(S_j = (x_1, \dots, x_m)\), write \(S_j^i = ((x_1, f_i(x_1)), \dots, (x_m, f_i(x_m)))\) for the same messages labelled by \(f_i\). The superscript \(i\) here is a tag, not a power: it says which target did the labelling. When the task is \(\mathcal D_i\), the training sequence is always one of \(S_1^i, \dots, S_k^i\). Each \(x_t\) is uniform on \(C\) and the draws are independent, so the probability of any particular sequence of messages is the product \((1/2m)^m = 1/k\): all \(k\) are equally likely. An expectation over finitely many equally likely outcomes is their plain average, so

\[ \mathbb E_{S \sim \mathcal D_i^m}\left[L_{\mathcal D_i}(A(S))\right] = \frac1k \sum_{j=1}^{k} L_{\mathcal D_i}(A(S_j^i)). \]

- \(\mathbb E_{S \sim \mathcal D_i^m}[\cdot]\) is the expectation: the average value of the bracket over random training sequences drawn from task \(i\), each weighted by its probability.
- \(A(S_j^i)\) is the hypothesis the learner returns on the \(j\)-th sequence of messages labelled by the \(i\)-th target, and \(L_{\mathcal D_i}\) scores it on task \(i\).
- \(\sum_{j=1}^{k}\) adds over \(j = 1, 2, \dots, k\), every sequence once, and \(\frac1k\) divides by how many there are.

In plain words: **the expected true risk on task \(i\) is the plain average of the true risk over all \(k\) equally likely training sequences.** For the memorizer on “everything is junk” that is the average of four risks of \(3/4\) and twelve of \(2/4\), the 0.5625 we computed by hand. The function `expected_risk` in the duel’s snippet is this formula line for line: it runs the learner on each \(S_j^i\), scores the result, and divides by \(k = 16\) (and by 4 for the messages inside the true risk).

**Step 3: maximum, average, minimum.** Write \([T]\) for the list of target numbers \(\{1, \dots, T\}\), as lesson 1 wrote \([m]\) for positions. Then \(\max_{i \in [T]}\) reads “the largest value as \(i\) runs over the targets”, and \(\min_{j \in [k]}\) “the smallest value as \(j\) runs over the sequences”. In the duel, the memorizer’s largest expected risk over the 16 targets was 0.5625 and its average over them 0.28125. We want to show that \(\max_{i \in [T]}\) of that expectation is at least \(1/4\), for every learner. Two elementary facts carry the step: the maximum of a list of numbers is at least their average (each number is at most the maximum, so their sum is at most \(T\) times it), and the average is at least the minimum (by the same argument turned upside down). Between them we swap the order of two finite sums, which changes nothing:

\[ \max_{i \in [T]} \frac1k \sum_{j=1}^{k} L_{\mathcal D_i}(A(S_j^i)) \;\ge\; \frac1T \sum_{i=1}^{T} \frac1k \sum_{j=1}^{k} L_{\mathcal D_i}(A(S_j^i)) \;=\; \frac1k \sum_{j=1}^{k} \frac1T \sum_{i=1}^{T} L_{\mathcal D_i}(A(S_j^i)) \;\ge\; \min_{j \in [k]} \frac1T \sum_{i=1}^{T} L_{\mathcal D_i}(A(S_j^i)). \]

Read it left to right, as four numbers.

- The first is the worst target’s expected true risk: for each \(i\), the average over sequences, and then the largest.
- The second averages that same expected risk over all targets instead, with \(\frac1T \sum_{i=1}^{T}\), so it can only be smaller or equal.
- The third adds exactly the same \(T \cdot k\) risks in the other order: for each sequence, average over targets, then average over sequences. Same numbers, same total.
- The fourth keeps only the kindest sequence: the one whose average over targets is smallest, which no average over sequences can fall below.

In plain words: **the worst target is at least as bad as the average target, the average can be taken in either order, and an average over sequences is at least its kindest sequence.** The snippet fills a grid of true risks for the memorizer in the duel, one row per target and one column per sequence, and reads the four numbers off it:

```python
import itertools

C, m = ["00", "01", "10", "11"], 2
F = list(itertools.product([0, 1], repeat=4))
Ss = list(itertools.product(C, repeat=m))

def L(f, xs):
    seen = dict((x, f[C.index(x)]) for x in xs)
    return sum(seen.get(c, 0) != f[C.index(c)] for c in C) / 4

grid = [[L(f, xs) for xs in Ss] for f in F]
by_target = [sum(row) / len(Ss) for row in grid]
by_seq = [sum(col) / len(F) for col in zip(*grid)]
print(max(by_target), sum(by_target) / len(F),
      sum(by_seq) / len(Ss), min(by_seq))
```

It prints `0.5625 0.28125 0.28125 0.25`. The kindest sequences are those with two different messages, where two of the four are unseen and each is wrong for half the targets, a quarter in all.

The right-hand end fixes one sequence of messages \(S_j\) and averages over all targets. That is exactly what the twins argument handles. It remains to show that, for every \(j\), this average over targets is at least \(1/4\).

**Step 4: keep only the unseen messages.** Fix \(j\), and let \(v_1, \dots, v_p\) be the messages of \(C\) that do not appear in \(S_j\). Since \(S_j\) has \(m\) entries, it contains at most \(m\) distinct messages, so \(p \ge 2m - m = m\). In the duel, the sequence (01, 01) leaves \(v_1, v_2, v_3 = 00, 10, 11\) unseen, so \(p = 3\). For every \(h: C \to \{0, 1\}\) and every \(i\), dropping the seen messages from the sum in step 1 can only lower it, since every term is 0 or 1, and \(\frac{1}{2m} \ge \frac{1}{2p}\) because \(p \ge m\):

\[ L_{\mathcal D_i}(h) = \frac{1}{2m} \sum_{x \in C} \mathbb 1[h(x) \neq f_i(x)] \;\ge\; \frac{1}{2m} \sum_{r=1}^{p} \mathbb 1[h(v_r) \neq f_i(v_r)] \;\ge\; \frac{1}{2p} \sum_{r=1}^{p} \mathbb 1[h(v_r) \neq f_i(v_r)]. \]

The first expression is the true risk from step 1. The second keeps the same \(\frac{1}{2m}\) but sums only over the unseen messages, \(r = 1\) to \(p\), so it counts only the mistakes \(h\) makes on \(v_1, \dots, v_p\). The third divides that count by the larger \(2p\), so it is smaller again. In plain words: **the true risk is at least half the fraction of unseen messages that \(h\) gets wrong.** Take the memorizer on (01, 01) with the target “everything is junk”: it says keep on all three unseen messages, so the three expressions are \(3/4\), \(3/4\) and \(3/6\).

```python
C, m = ["00", "01", "10", "11"], 2
f_i = dict(zip(C, (1, 1, 1, 1)))
xs = ("01", "01")
h = {c: (f_i[c] if c in xs else 0) for c in C}
v = [c for c in C if c not in xs]
p = len(v)
full = sum(h[x] != f_i[x] for x in C) / (2 * m)
unseen = sum(h[x] != f_i[x] for x in v)
print(v, p, full, unseen / (2 * m), unseen / (2 * p))
```

It prints `['00', '10', '11'] 3 0.75 0.75 0.5`.

**Step 5: from the average over unseen messages to the worst one.** Apply step 4 with \(h = A(S_j^i)\), average over \(i\), and swap the two finite sums again. Then use “an average is at least the minimum” once more, this time over \(r\). One new piece of notation appears: \(A(S_j^i)(v_r)\) is the learner’s hypothesis on \(S_j^i\), applied to the unseen message \(v_r\), so it is the label the learner guesses there. The fraction of targets that catch the learner out at \(v_r\) we call its **catch rate** at \(v_r\).

\[ \frac1T \sum_{i=1}^{T} L_{\mathcal D_i}(A(S_j^i)) \;\ge\; \frac{1}{2p} \sum_{r=1}^{p} \frac1T \sum_{i=1}^{T} \mathbb 1[A(S_j^i)(v_r) \neq f_i(v_r)] \;\ge\; \frac12 \cdot \min_{r \in [p]} \frac1T \sum_{i=1}^{T} \mathbb 1[A(S_j^i)(v_r) \neq f_i(v_r)]. \]

The \(\frac12\) comes from writing \(\frac{1}{2p} \sum_{r=1}^{p}\) as \(\frac12\) times the average over \(r\). The inner \(\frac1T \sum_{i=1}^{T} \mathbb 1[\dots]\) is the catch rate at \(v_r\): add 1 for every target the guess disagrees with, divide by \(T\). Then \(\min_{r \in [p]}\) picks the unseen message with the lowest catch rate. In plain words: **for a fixed sequence, the true risk averaged over targets is at least half the catch rate at the hardest-to-catch unseen message.** For the memorizer on (01, 01), the left end is \(3/8\), the middle drops the seen message’s share and gives \(\frac16 (\frac12 + \frac12 + \frac12) = \frac14\), and every catch rate is \(\frac12\), so the right end is \(\frac12 \cdot \frac12 = \frac14\):

```python
import itertools

C, m, xs = ["00", "01", "10", "11"], 2, ("01", "01")
F = list(itertools.product([0, 1], repeat=4))
v = [c for c in C if c not in xs]

def wrong(f, c):
    g = dict(zip(C, f))
    h = {x: (g[x] if x in xs else 0) for x in C}
    return h[c] != g[c]

left = sum(sum(wrong(f, c) for c in C) / 4 for f in F) / 16
per_r = [sum(wrong(f, r) for f in F) / 16 for r in v]
print(left, sum(per_r) / (2 * len(v)), 0.5 * min(per_r))
```

It prints `0.375 0.25 0.25`.

**Step 6: the twins.** Fix \(r\). Pair each target \(f_i\) with the target \(f_{i'}\) that agrees with it everywhere on \(C\) except at \(v_r\). Flipping the same label twice gives back the original, so this pairing splits \(f_1, \dots, f_T\) into \(T/2\) disjoint pairs. Since \(v_r\) does not appear in \(S_j\), the twins label every message of \(S_j\) identically, so \(S_j^i = S_j^{i'}\), and because \(A\) is a function of its input, \(A(S_j^i) = A(S_j^{i'})\). Call that hypothesis \(h\). The bit \(h(v_r)\) equals exactly one of \(f_i(v_r)\) and \(f_{i'}(v_r)\), since those two differ, so

\[ \mathbb 1[A(S_j^i)(v_r) \neq f_i(v_r)] + \mathbb 1[A(S_j^{i'})(v_r) \neq f_{i'}(v_r)] = 1. \]

The first indicator asks whether the learner’s guess at \(v_r\) is wrong for target \(f_i\), the second whether the same guess is wrong for its twin \(f_{i'}\) (the prime on \(i'\) just marks the twin’s number). In plain words: **of each pair of twins, exactly one catches the learner at \(v_r\).** In the duel, take (01, 01), \(v_r = 00\), the target “everything is junk” and its twin “junk everywhere except 00”. The memorizer sees (01, junk) twice under both and guesses keep at 00, so it is wrong for the first and right for the twin: \(1 + 0 = 1\).

Summing over the \(T/2\) pairs gives \(T/2\), so for every \(r\),

\[ \frac1T \sum_{i=1}^{T} \mathbb 1[A(S_j^i)(v_r) \neq f_i(v_r)] = \frac12. \]

In plain words: **at every unseen message, exactly half the targets catch the learner, whatever the learner is.** That is the catch rate of \(\frac12\) the last snippet found at all three unseen messages, and the machine check below confirms it for an arbitrary learner.

Chaining steps 3, 5 and 6: for every \(j\) the average over targets is at least \(\frac12 \cdot \frac12 = \frac14\), so the minimum over \(j\) is at least \(1/4\), and therefore

\[ \max_{i \in [T]} \mathbb E_{S \sim \mathcal D_i^m}\left[L_{\mathcal D_i}(A(S))\right] \ge \frac14. \]

Every symbol here has been read already: **some target’s task gives the learner an expected true risk of at least a quarter**, as the memorizer’s 0.5625 on “everything is junk” illustrates.

So far \(A\) was a learner that receives examples from \(C\) and returns a function on \(C\). Now take any learner \(A'\) over the whole domain \(\mathcal X\). Under any \(\mathcal D_i\), every training sequence lies in \(C\), and the true risk of \(A'(S)\) depends only on its values on \(C\), because \(\mathcal D_i\) puts no probability outside \(C\). So define \(A(S)\) as \(A'(S)\) restricted to \(C\); the bound applies to it, and the risks agree. Let \(i^\star\) be an index achieving the maximum, set \(\mathcal D = \mathcal D_{i^\star}\), and extend \(f_{i^\star}\) to all of \(\mathcal X\) in any way, say with keep outside \(C\); call the result \(f\). Then \(L_{\mathcal D}(f) = 0\), which is part 1, and

\[ \mathbb E_{S \sim \mathcal D^m}\left[L_{\mathcal D}(A'(S))\right] \ge \frac14. \]

This is the previous display with the worst task named \(\mathcal D\) and the learner allowed to live on all of \(\mathcal X\); the star in \(i^\star\) only marks the chosen index.

Here is a machine check of step 6, the one step that does real work. The learner below is as arbitrary as I could make it: for each training sequence, it labels every message by a coin flip seeded with that sequence, so it is deterministic but follows no pattern at all. For every one of the 16 sequences and every message the sequence misses, the snippet collects the fraction of the 16 targets that catch the learner out at that message.

```python
import itertools, random

C, m = ["00", "01", "10", "11"], 2
targets = list(itertools.product([0, 1], repeat=4))

def A(S):
    rng = random.Random(str(S))
    return {c: rng.randint(0, 1) for c in C}

averages = set()
for xs in itertools.product(C, repeat=m):
    for v in set(C) - set(xs):
        wrong = [A([(x, f[C.index(x)]) for x in xs])[v]
                 != f[C.index(v)] for f in targets]
        averages.add(sum(wrong) / len(targets))
print(averages)
```

It prints `{0.5}`. Every sequence, every unseen message, the same answer: exactly half the targets catch it.

**Step 7: from an average to a probability.** Let \(\theta = L_{\mathcal D}(A'(S))\). It is a random variable, because \(S\) is random, and it takes values in \([0, 1]\), with \(\mathbb E[\theta] \ge 1/4\). Let \(q = \mathbb P[\theta \ge 1/8]\). Split the expectation according to whether \(\theta\) is at least \(1/8\). On the event \(\theta \ge 1/8\), which has probability \(q\), the value is at most 1. On its complement, which has probability \(1 - q\), the value is below \(1/8\). So

\[ \frac14 \le \mathbb E[\theta] \le 1 \cdot q + \frac18 (1 - q). \]

- \(\theta\) is the true risk of the one hypothesis you actually get, random because \(S\) is, and \(\mathbb E[\theta]\) is its expected value.
- \(q\) is the probability of the bad event \(\theta \ge 1/8\), so \(1 \cdot q\) is the most the bad sequences can add to the average: their weight times the largest risk possible.
- \(\frac18 (1 - q)\) is the most the good sequences can add: their weight times a risk below an eighth.

In plain words: **the average risk can be no more than the bad sequences’ most plus the good sequences’ most, and that ceiling must still reach a quarter.** For the memorizer on “everything is junk”, every sequence is bad, so \(q = 1\) and the ceiling is \(1\), far above \(\frac14\). The ceiling is tightest when \(q\) is as small as the quarter allows, which is the case the snippet after the proof builds.

Multiply by 8 to get \(2 \le 8q + 1 - q\), so \(7q \ge 1\) and \(q \ge 1/7\). That is part 2. (The same bound follows from **Markov’s inequality**, which says that a nonnegative random variable \(Z\) has \(\mathbb P[Z \ge a] \le \mathbb E[Z]/a\) for every \(a > 0\). Apply it to \(Z = 1 - \theta\) with \(a = 7/8\): \(\mathbb P[\theta \le 1/8] \le (3/4)/(7/8) = 6/7\).) \(\blacksquare\)

The \(1/7\) is not slack in the argument; it can be reached. A risk that equals 1 with probability \(1/7\) and sits just under \(1/8\) otherwise has mean exactly a quarter:

```python
mean_at_least, cutoff = 1 / 4, 1 / 8
q = (mean_at_least - cutoff) / (1 - cutoff)
print(round(q, 6), round(1 / 7, 6))
theta = [(1.0, q), (cutoff - 1e-9, 1 - q)]
print(round(sum(v * p for v, p in theta), 6))
```

It prints `0.142857 0.142857` and `0.25`. Knowing only the mean, you cannot promise more than \(1/7\).

Notice what the theorem does and doesn’t say. It does not say every learner is bad on every task: the memorizer is perfect on any task whose target says keep everywhere. It does not say learning is impossible: the task it builds is solved by a learner that knows to look for \(f\). It says that for each learner, separately, a bad task exists, and that the bad task needs nothing exotic, just a uniform distribution on \(2m\) messages and a labelling the learner didn’t expect.

## Theo’s filter, on an infinite domain

Theo’s filter does not work on four messages, or eight, or 256. It works on real messages, and a real message is text of any length. There are infinitely many possible messages, so \(\mathcal X\) is infinite. And “consider every hypothesis” means his hypothesis class is the set of all functions from \(\mathcal X\) to \(\{0, 1\}\). What does the theorem say about that?

Recall lesson 3’s definition. A class \(\mathcal H\) is **PAC learnable** if there are a function \(m_{\mathcal H}: (0,1)^2 \to \mathbb N\) and a learner such that for every \(\varepsilon, \delta \in (0, 1)\), every distribution over \(\mathcal X\) and every labelling \(f\) for which some hypothesis in \(\mathcal H\) has zero true risk, the learner, run on \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples, returns a hypothesis with true risk at most \(\varepsilon\), with probability at least \(1 - \delta\).

**Corollary (the class of all functions is not learnable).** Let \(\mathcal X\) be an infinite domain and let \(\mathcal H\) be the set of all functions from \(\mathcal X\) to \(\{0, 1\}\). Then \(\mathcal H\) is not PAC learnable.

*Proof.* Suppose, for contradiction, that it is. Choose \(\varepsilon < 1/8\) and \(\delta < 1/7\). By the definition there is a learner \(A\) and an integer \(m = m_{\mathcal H}(\varepsilon, \delta)\) such that on every realizable task, with probability at least \(1 - \delta\), \(L_{\mathcal D}(A(S)) \le \varepsilon\) when \(S\) has \(m\) examples. Equivalently, the probability that \(L_{\mathcal D}(A(S)) > \varepsilon\) is at most \(\delta\).

Since \(\mathcal X\) is infinite, \(|\mathcal X| > 2m\), so the No-Free-Lunch theorem applies to \(A\) and \(m\). It gives a task \(\mathcal D\) with a labelling \(f\) of zero true risk. Every function is in \(\mathcal H\), so \(f \in \mathcal H\) and the task is realizable with respect to \(\mathcal H\): the definition’s promise covers it. But the theorem says that with probability at least \(1/7\), \(L_{\mathcal D}(A(S)) \ge 1/8\), and \(1/8 > \varepsilon\), so with probability at least \(1/7 > \delta\), \(L_{\mathcal D}(A(S)) > \varepsilon\). The promise says that probability is at most \(\delta\). Contradiction. \(\blacksquare\)

That is the answer to Theo. There is no “enough labelled mail”. Whatever number of labels he names, the domain has more than twice that many messages, and there is a junk rule, one that some hypothesis labels perfectly, on which his filter has a one-in-seven chance or more of being wrong on at least an eighth of all messages.

How do we escape? By doing exactly what the course has been doing: using prior knowledge about the task to rule out the distributions that would defeat us, and writing that knowledge down as a restricted hypothesis class. Which raises the real question. If the class of all functions is too rich, how rich should a class be? On one side, we want the class to contain the truth, or at least something with small true risk. On the other, we have just seen that we cannot take the richest class there is. Lesson 1 named the two ways to get this wrong, the rat’s and the pigeon’s. Now we can measure them.

## Two ways to be wrong

Take any class \(\mathcal H\), and let \(h_S\) be the hypothesis ERM returns, that is, \(h_S \in \operatorname{argmin}_{h \in \mathcal H} L_S(h)\). Split its true risk into two parts:

\[ L_{\mathcal D}(h_S) = \varepsilon_{\mathrm{app}} + \varepsilon_{\mathrm{est}}, \qquad \varepsilon_{\mathrm{app}} = \min_{h \in \mathcal H} L_{\mathcal D}(h), \qquad \varepsilon_{\mathrm{est}} = L_{\mathcal D}(h_S) - \varepsilon_{\mathrm{app}}. \]

The decomposition is trivially true, since the second part is defined as whatever is left over. What makes it worth writing down is that each part has its own cause and its own cure.

[[diagram:decomposition]]

The **approximation error** \(\varepsilon_{\mathrm{app}}\) is the true risk of the best hypothesis in the class. It measures how much we lose by restricting ourselves to \(\mathcal H\) at all, which is to say how much inductive bias we have, and it is the rat’s failure: a class that misses the truth. It is a property of the class and the task alone. It does not depend on \(m\) or on which examples arrived, and no number of examples changes it. When the task is realizable by \(\mathcal H\) it is zero. When it isn’t, it can be large, and it is never below the Bayes risk, lesson 3’s floor that no hypothesis at all can beat. (Some authors call only the excess over the Bayes risk the approximation error; we keep the full minimum.) Enlarging the class can only help it: if \(\mathcal H \subseteq \mathcal H'\), then \(\min_{h \in \mathcal H'} L_{\mathcal D}(h) \le \min_{h \in \mathcal H} L_{\mathcal D}(h)\), because the minimum over a bigger set is taken over more candidates, including all the old ones.

The **estimation error** \(\varepsilon_{\mathrm{est}}\) is how far ERM’s choice falls short of the best in the class. It is never negative, since \(h_S\) is itself in \(\mathcal H\) and so its true risk is at least the class minimum. It exists because ERM ranks hypotheses by empirical risk, which is only an estimate of true risk, so the hypothesis that wins on the examples is only an estimate of the hypothesis that wins on \(\mathcal D\). This is the pigeon’s failure: a class so large for the examples that a coincidence can beat the truth. It depends on \(m\) and on the size of the class, and lesson 4 already measured it.

**Proposition (estimation error of a finite class).** Let \(\mathcal H\) be finite, let \(\delta \in (0, 1)\), and let \(S \sim \mathcal D^m\). Then with probability at least \(1 - \delta\),

\[ \varepsilon_{\mathrm{est}} \le \sqrt{\frac{2 \log(2|\mathcal H|/\delta)}{m}}. \]

*Proof.* Set \(\varepsilon = \sqrt{2\log(2|\mathcal H|/\delta)/m}\), so that \(m = 2\log(2|\mathcal H|/\delta)/\varepsilon^2\). Lesson 4 showed, using Hoeffding’s inequality for each fixed hypothesis and the union bound across the class, that the probability of some \(h \in \mathcal H\) having \(|L_S(h) - L_{\mathcal D}(h)| > \varepsilon/2\) is at most \(2|\mathcal H| \exp(-2m(\varepsilon/2)^2) = 2|\mathcal H| \exp(-m\varepsilon^2/2)\). With our \(m\), the exponent is \(-\log(2|\mathcal H|/\delta)\), so the bound equals \(\delta\). So with probability at least \(1 - \delta\), the training sequence is \(\varepsilon/2\)-representative: every hypothesis in the class has empirical risk within \(\varepsilon/2\) of its true risk.

On that event, let \(h^\star\) be a best hypothesis in the class, so \(L_{\mathcal D}(h^\star) = \varepsilon_{\mathrm{app}}\). Then

\[ L_{\mathcal D}(h_S) \le L_S(h_S) + \tfrac{\varepsilon}{2} \le L_S(h^\star) + \tfrac{\varepsilon}{2} \le L_{\mathcal D}(h^\star) + \varepsilon. \]

The first and third steps use representativeness, and the middle step uses that \(h_S\) minimizes empirical risk. Subtracting \(\varepsilon_{\mathrm{app}}\) gives \(\varepsilon_{\mathrm{est}} \le \varepsilon\). \(\blacksquare\)

The bound grows with the logarithm of the class size and shrinks like \(1/\sqrt m\). The logarithm is generous, but notice what it is a logarithm of. For Theo’s class on a domain of \(N\) messages, \(|\mathcal H| = 2^N\), so \(\log|\mathcal H| = N \log 2\): the bound needs \(m\) to be large compared with the number of possible messages, which is the No-Free-Lunch theorem seen from the other side.

## A ladder of classes

Let’s watch the two errors trade against each other in a world we can compute exactly. Give each message eight features, so the domain has \(2^8 = 256\) messages, drawn uniformly. The first four are lesson 1’s, and four new ones join them. Here is the truth, which the learner doesn’t know: each feature carries some points, and a message is junk exactly when its points add up to at least 26, half of the 52 available.

| feature | meaning | points |
|---|---|---|
| F | the subject says “FREE” | 16 |
| L | it has a link | 12 |
| U | the sender is unknown | 8 |
| N | it was sent at night | 6 |
| A | it has an attachment | 4 |
| R | the reply-to address differs from the sender | 3 |
| M | it went to many recipients | 2 |
| I | it is mostly images | 1 |

Every feature matters a little, and the ones further down matter less. Now build a **nested family** of classes, each containing the one before. Let \(\mathcal H_k\) be the set of all hypotheses that look only at the first \(k\) features: a **\(k\)-feature table**, which assigns a label to each of the \(2^k\) possible settings of those features, called **cells**, and gives that label to every message in the cell. So \(\mathcal H_0\) has two members (always keep, always junk), \(\mathcal H_1\) has four, and in general \(|\mathcal H_k| = 2^{2^k}\). At the top of the ladder, \(\mathcal H_8\) is every hypothesis on the domain, all \(2^{256}\) of them. That top rung is Theo’s universal filter.

\[ \mathcal H_0 \subseteq \mathcal H_1 \subseteq \cdots \subseteq \mathcal H_8 = \{\text{all } h: \{0,1\}^8 \to \{0,1\}\}. \]

What does ERM over \(\mathcal H_k\) do? The empirical risk splits into one sum per cell, and each cell’s sum depends only on the label the table gives that cell. Every combination of cell labels is in the class, so we can minimize each cell separately, and within a cell the fewest mistakes come from the majority label of the examples that fell in it. So ERM over \(\mathcal H_k\) is a **table learner**: majority vote within each cell, with ties and empty cells labelled keep (any label is a minimizer there; keep is our tie-break). At \(k = 8\) every cell is a single message, and the table learner is lesson 1’s memorizer.

The approximation error comes from the same cell-by-cell reasoning, applied to the truth instead of the examples. The best table puts the majority true label in each cell, and it is wrong on the cell’s minority. Add the minorities up:

```python
import itertools

POINTS = (16, 12, 8, 6, 4, 3, 2, 1)
X = list(itertools.product([0, 1], repeat=8))

def f(x):
    return int(sum(p * b for p, b in zip(POINTS, x)) >= 26)

def approximation_error(k):
    cells = {}
    for x in X:
        cells.setdefault(x[:k], []).append(f(x))
    wrong = 0
    for labels in cells.values():
        wrong += min(sum(labels), len(labels) - sum(labels))
    return wrong / len(X)

print(sum(map(f, X)))
print([approximation_error(k) * 256 for k in range(9)])
```

It prints `132` (132 of the 256 messages are junk) and `[124.0, 48.0, 48.0, 26.0, 18.0, 12.0, 8.0, 4.0, 0.0]`: the best \(k\)-feature table is wrong on that many messages out of 256. The approximation error falls from \(124/256 \approx 0.48\) for the constant hypotheses to zero for Theo’s class, the only one that contains the truth. (Adding L alone doesn’t help, since 48 stays 48; it only pays off once U joins it.)

So far Theo is winning: his class is the only one with no approximation error at all. Now give the table learner a label budget of \(m = 50\) and compute its expected true risk on each rung. The column below was computed exactly, by summing over how many examples land in each cell and how their votes split; the simulation after it checks the numbers.

| \(k\) | \(\lvert\mathcal H_k\rvert\) | \(\varepsilon_{\mathrm{app}}\) | expected \(L_{\mathcal D}(h_S)\) | expected \(\varepsilon_{\mathrm{est}}\) |
|---|---|---|---|---|
| 0 | \(2\) | 0.484 | 0.499 | 0.015 |
| 1 | \(2^2\) | 0.188 | 0.188 | 0.000 |
| 2 | \(2^4\) | 0.188 | 0.210 | 0.023 |
| 3 | \(2^8\) | 0.102 | 0.120 | 0.018 |
| 4 | \(2^{16}\) | 0.070 | **0.105** | 0.035 |
| 5 | \(2^{32}\) | 0.047 | 0.156 | 0.109 |
| 6 | \(2^{64}\) | 0.031 | 0.256 | 0.225 |
| 7 | \(2^{128}\) | 0.016 | 0.353 | 0.338 |
| 8 | \(2^{256}\) | 0 | 0.424 | 0.424 |

Read down the last two columns. The approximation error falls on every rung, and the estimation error rises, slowly at first and then fast, because each rung doubles the number of cells and 50 examples spread over 256 cells leave most cells empty. The total, their sum, is lowest at \(k = 4\): a 16-cell table on F, L, U and N, wrong about 10.5% of the time. Theo’s filter, at the top, is wrong 42.4% of the time, every bit of it estimation error.

Here is the same claim as an experiment: draw 4,000 training sequences of 50 examples, run the table learner, and average its exact true risk over all 256 messages.

```python
import itertools, random

POINTS = (16, 12, 8, 6, 4, 3, 2, 1)
X = list(itertools.product([0, 1], repeat=8))
def f(x):
    return int(sum(p * b for p, b in zip(POINTS, x)) >= 26)

def table_erm(S, k):
    cells = {}
    for x, y in S:
        cells.setdefault(x[:k], []).append(y)
    return {c: int(2 * sum(ys) > len(ys))
            for c, ys in cells.items()}

rng, m, trials = random.Random(1), 50, 4000
for k in (1, 3, 4, 5, 8):
    total = 0
    for _ in range(trials):
        S = [(x, f(x)) for x in rng.choices(X, k=m)]
        h = table_erm(S, k)
        total += sum(h.get(x[:k], 0) != f(x) for x in X) / len(X)
    print(k, round(total / trials, 3))
```

It prints `1 0.188`, `3 0.119`, `4 0.105`, `5 0.155` and `8 0.424`, matching the exact column to within simulation noise.

[[diagram:tradeoff]]

What about the bound from the proposition? It holds for every distribution, which makes it cautious, and here is how cautious:

```python
import math

def estimation_bound(k, m, delta=0.1):
    log_size = 2 ** k * math.log(2)
    return math.sqrt(2 * (log_size + math.log(2 / delta)) / m)

for m in (50, 5000):
    bounds = [estimation_bound(k, m) for k in (1, 4, 8)]
    print(m, [round(b, 3) for b in bounds])
```

It prints `50 [0.419, 0.751, 2.687]` and `5000 [0.042, 0.075, 0.269]`. At 50 examples the bound is useless, above 1 for most rungs, since a risk can never exceed 1. At 5,000 it says something, and its shape is the shape of the table: it grows with \(\sqrt{2^k}\), the square root of the log of the class size. The bound tells you which way the estimation error moves; the exact numbers came from knowing \(\mathcal D\).

## The bias-complexity tradeoff

Put it together and you have the **bias-complexity tradeoff**. Choose a very rich class and the approximation error shrinks, but the estimation error may grow, because a rich class has room for a coincidence to beat the truth: that is **overfitting**, the pigeon’s failure. Choose a very small class and the estimation error shrinks, but the approximation error may grow, because the truth, or anything close to it, may not be in the class: that is **underfitting**, the rat’s failure. The best class sits between them, and where it sits depends on how many examples you have. In our eight-feature world, recomputing the table for other budgets, the best rung climbs with the budget:

| label budget \(m\) | 20 | 50 | 100 | 300 | 500 | 1,000 |
|---|---|---|---|---|---|---|
| best \(k\) | 3 | 4 | 5 | 6 | 7 | 8 |
| its expected true risk | 0.162 | 0.105 | 0.079 | 0.040 | 0.026 | 0.010 |

Theo’s filter does eventually win, at 1,000 labels, which is nearly four labels for every possible message. In a domain of 256 messages, that is affordable. In the real one, where messages are unbounded text, it never is.

Is there a perfect class? Yes, in principle: the class containing only the Bayes optimal predictor from lesson 3. Its approximation error is the Bayes risk, the best possible, and its estimation error is zero, since there is nothing to choose. But building it requires knowing \(\mathcal D\), and if we knew \(\mathcal D\) we wouldn’t need to learn. Learning theory asks how rich a class can be while the estimation error stays reasonable. Practice spends much of its effort designing classes whose approximation error is small for the domain at hand. The point of prior knowledge is to land somewhere that both errors are tolerable: you don’t know the junk rule, but you know enough about junk to guess which features carry it.

One honest caveat about the lab you are about to use. It shows you the approximation error because it knows the truth. In the real mail room you don’t, and picking the rung from data alone is a problem of its own: lesson 7 handles it by spreading confidence across the ladder, and lesson 11 by holding out examples to validate.

## The algorithms

Two algorithms carry this lesson. The first is the table learner, ERM over \(\mathcal H_k\). The second is the adversary from the proof, written as code: given a learner, it finds the task that defeats it.

```
TableERM(S, k):
  votes ← empty hash map from cells to (junk count, total)
  for i = 1..m:
    c ← first k features of x_i
    votes[c].junk += y_i;  votes[c].total += 1
  table ← for each cell c in votes: 1 if 2·junk > total else 0
  return h(x) = table[first k features of x] if present, else 0

Adversary(A, C, m):
  best ← none
  for each target f: C → {0, 1}:
    total ← 0
    for each sequence (x_1, ..., x_m) in C^m:
      h ← A(((x_1, f(x_1)), ..., (x_m, f(x_m))))
      total += (number of c in C with h(c) ≠ f(c)) / |C|
    if total / |C|^m > risk of best: best ← (f, total / |C|^m)
  return best
```

**TableERM** reads each example once and hashes a prefix of \(k\) bits, so it takes \(O(mk)\) time. It stores at most one entry per occupied cell, so \(O(k \cdot \min(m, 2^k))\) space. A prediction costs \(O(k)\) expected time. The cost does not grow with the class size \(2^{2^k}\), which is the one piece of good news about a class this rich: searching it is easy, and it is only the statistics that are hard.

**Adversary** is the proof of the No-Free-Lunch theorem made executable, and its cost shows why the proof averages instead of searching. With \(|C| = 2m\), it runs the learner on \(2^{2m} \cdot (2m)^m\) training sequences and evaluates each returned hypothesis on \(2m\) messages, so it takes \(O(2^{2m} (2m)^{m} (t_A + m))\) time, where \(t_A\) is the learner’s running time, and \(O(m)\) space beyond what the learner itself uses. At \(m = 4\) that is already over a million runs of the learner, which is what the lab did for you. The proof never finds the worst target. It only shows that the average is high, and the average is enough.

**Build it yourself.** Using only the Python standard library:

```python
import itertools

POINTS = (16, 12, 8, 6, 4, 3, 2, 1)
MESSAGES = [
    "11010000", "11010000", "11010001", "00000000", "11000011",
    "01100101", "10101111", "10110010", "11011101", "00000111",
    "11011010", "01101101", "11000110", "00100111", "10001100",
    "01001100", "11010111", "11111001", "01100111", "01011111",
    "11001111", "11111111", "01110001", "11010111", "00010000",
    "10000010", "10011110", "00101111", "01010111", "10111100",
    "11000101", "01011110", "11111011", "11101111", "01001010",
    "01011010", "01101111", "00011110", "11100011", "00011111",
    "10110101", "11100111", "01000100", "01101100", "00001100",
    "01000001", "01001001", "01111100", "10101001", "10011100"]

def f(x):
    return int(sum(p * int(b) for p, b in zip(POINTS, x)) >= 26)

def memorizer(S):
    table = dict(S)
    return lambda x: table.get(x, 0)

S = [(x, f(x)) for x in MESSAGES]

def table_learner(S, k): ...
def true_risk(h, f): ...
def approximation_error(k, f): ...
def adversary(A, C, m): ...

if __name__ == "__main__":
    app = [approximation_error(k, f) * 256 for k in range(9)]
    assert app == [124, 48, 48, 26, 18, 12, 8, 4, 0]
    assert true_risk(table_learner(S, 4), f) == 20 / 256
    assert true_risk(table_learner(S, 8), f) == 102 / 256
    risks = [true_risk(table_learner(S, k), f) for k in range(9)]
    assert min(range(9), key=risks.__getitem__) == 4
    C2 = ["00", "01", "10", "11"]
    assert adversary(memorizer, C2, 2) == ((1, 1, 1, 1), 0.5625)
    C3 = ["000", "001", "010", "011", "100", "101"]
    assert adversary(memorizer, C3, 3)[1] >= 1 / 4
```

1. Implement the four functions. A message is a string of eight bits; `table_learner` returns a function from messages to labels (majority per cell, ties and empty cells keep); `true_risk` averages mistakes over all 256 messages; `adversary` takes a learner (a function from a list of `(x, y)` pairs to a hypothesis), the list `C` and `m`, and returns the target, as a tuple of labels in the order of `C`, with the highest expected true risk, together with that risk.
2. Print the true risk of `table_learner(S, k)` for every \(k\) on this one training sequence of 50 examples, and compare it with the expected values in the table of “A ladder of classes”. Which rungs did this sequence treat kindly?
3. Run `adversary` on `table_learner` with \(k = 1\) over the four two-bit messages. Which target defeats it, and is its failure approximation error, estimation error, or both?
4. Draw 2,000 training sequences for each budget \(m\) in 20, 50, 100 and 300, and reproduce the row “best \(k\)” of the tradeoff table by simulation.

## Fix the bug

Theo isn’t convinced by a proof. He wants to see the filter lose, and he wants you to show him what to ship instead.

In the lab below, the mail room is the eight-feature world, with its points rule hidden from the learner but known to the lab. Each round gives you a label budget, and you choose the rung \(k\) of the ladder, from the constant hypotheses at \(k = 0\) to Theo’s universal filter at \(k = 8\). The lab shows the approximation error of every rung, because it knows the truth. What it doesn’t show is the estimation error: to see that, you train, and watch how the true risk of \(h_S\) scatters from one training sequence to the next. Your goal is the rung whose expected true risk is lowest for the budget, within a hundredth.

Before your first round, predict: with a budget of 50 labels, on which rung will the universal filter’s expected true risk land, compared with the best rung?

[[lab-act2]]

## What people get wrong

**“No free lunch means learning is impossible.”** It means learning is impossible *without prior knowledge*. Every task the theorem builds is solved easily by a learner that expected it. The theorem is an argument for choosing a class, not against learning.

**“So every learner is equally good.”** Only on average over *all* targets, each counted equally. The real mail room is not a uniformly random labelling of all messages: junk has structure, and a class that matches that structure beats Theo’s filter by a factor of four at 50 labels. Prior knowledge is exactly the claim that your task is not a uniformly random target.

**“A bigger class is safer, because it is more likely to contain the truth.”** It is more likely to contain the truth, which lowers the approximation error. But it also contains more impostors, which raises the estimation error. In the ladder, the class that contains the truth comes last at 50 labels.

**“The approximation error is bad luck with the data.”** No. It is a property of the class and the task, fixed before any example arrives. More examples shrink the estimation error and leave the approximation error exactly where it was, which is why the rat in a world of clicks never recovers.

## Recap and the next bug

1. A universal learner would succeed on every task with one fixed number of examples. The No-Free-Lunch theorem says none exists: for every learner and every \(m < |\mathcal X|/2\) there is a realizable task on which, with probability at least \(1/7\), its true risk is at least \(1/8\).
2. The proof averages over all \(2^{2m}\) targets on \(2m\) messages. Twin targets, which differ only at one unseen message, hand the learner identical examples, so every unseen message is wrong for exactly half of them. At least half the messages are unseen, so the average risk is at least \(1/4\); the maximum is at least the average; and splitting the expectation at \(1/8\) turns it into the probability.
3. On an infinite domain, the class of all functions is not PAC learnable. Theo’s “enough labelled mail” does not exist.
4. ERM’s true risk splits into approximation error, the best risk in the class (the rat’s failure, fixed by the class), and estimation error, ERM’s shortfall from it (the pigeon’s failure, shrinking with \(m\) and growing with \(\log|\mathcal H|\)).
5. Choosing a class is the bias-complexity tradeoff: too rich overfits, too poor underfits, and the best rung climbs as the label budget grows.

Theo has gone quiet, and the team has moved on to a new idea. Instead of tables over yes/no features, they want the filter to compute one continuous spam score per message, a real number, and learn a threshold: junk exactly when the score is at least \(\theta\). The class is \(\{h_\theta : \theta \in \mathbb R\}\), and it has one hypothesis for every real number, so it is infinite. Every guarantee you own runs through \(\log|\mathcal H|\), and for this class that is infinite too; the bound says nothing at all. By this lesson’s logic, an infinite class should be Theo’s filter all over again. And yet, when you try it, thresholds learn from a few dozen labels and stay right on fresh messages. What is it about thresholds that a count of hypotheses misses? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A mail room has only \(|\mathcal X| = 10\) possible messages, and Theo offers \(m = 10\) labels drawn uniformly. Does the No-Free-Lunch theorem apply? If it doesn’t, does that mean the memorizer (keep on unseen messages) succeeds on every task here? Find a task and compute its expected true risk exactly.
2. In step 6 of the proof, the twins differ only at an unseen message \(v_r\). Suppose instead you paired targets that differ at a message that does appear in \(S_j\). Which line of the argument breaks, and why? Separately, step 4 replaced \(\frac{1}{2m}\) by \(\frac{1}{2p}\): where does \(p \ge m\) come from, and what would the final constant be if every training sequence happened to contain \(m\) distinct messages?
3. In the eight-feature world at \(m = 50\), the class \(\mathcal H_3\) has approximation error \(26/256\) and expected true risk \(0.1196\). What is its expected estimation error? How many examples does the proposition’s bound need before it guarantees \(\varepsilon_{\mathrm{est}} \le 0.05\) with \(\delta = 0.1\) for \(\mathcal H_3\), and how many for Theo’s \(\mathcal H_8\)?
