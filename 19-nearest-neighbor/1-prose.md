# Nearest Neighbour

## Mail that matches no rule

Every filter you have built so far began the same way: before seeing a single label, you fixed a hypothesis class, and the learner searched it. Halfspaces in lesson 9, boosted votes of stumps in lesson 10, decision trees in lesson 18. Each class was a bet about what junk looks like, placed in advance.

This month the bets stopped paying. New kinds of mail keep arriving that match no rule anyone wrote and no hypothesis anyone trained: parcel-delivery notices that are half genuine and half phishing, and supplier invoices that are half real and half fraud. The filters shrug at them. So the team reached for the oldest trick there is. When a new message arrives, find the labelled messages that look most like it, and give it their label. No class, no search, no training. Just “this one looks like those, and those were junk.”

By now every message in the mail room is described by thirty numbers, which the team calls **scores**, each scaled to lie between 0 and 1. Two of them matter for this story. The **payment score** says how much a message talks about paying (0 for a lunch invitation, near 1 for “settle the attached invoice today”), and the **sender score** says how unfamiliar the sender is (0 for your manager, near 1 for an address nobody has ever written to). The other twenty-eight scores cover everything else the team ever measured: link share, word counts, the hour it was sent, and so on.

For parcel notices, the team compared messages on the two scores the parcel dashboard happens to show, payment and sender. For invoices, a different engineer compared them on all thirty scores, on the reasonable-sounding principle that more information can’t hurt. A month later the numbers came in. On parcel notices, the copy-your-neighbours filter was wrong on 14% of messages. On invoices it was wrong on 32%, more than twice as often, and closer to a coin than anyone was comfortable with.

Here is the part that should bother you. When we dig into the data later in this lesson, it will turn out that both kinds of mail follow exactly the same pattern of junk and keep. The only difference is how many scores the comparison used. So why would *more* information make the filter *worse*? And how many labelled messages would you need before “label it like its neighbours” comes with a guarantee?

This lesson answers both. The method is called **nearest neighbour**, and it is the first learner in this course that does not search a hypothesis class at all. That makes it a strange and interesting test of everything lesson 1 told you about bias.

## Similar messages, similar labels

To say “the most similar message” we need a way to measure how far apart two messages are. Take a new invoice with payment score 0.70 and sender score 0.70, and a labelled one with 0.65 and 0.65 (you will meet both again in a moment). Draw them as two dots on a square whose sides are the two scores: they are 0.05 apart along each side, and the ruler between them reads about 0.071.

That ruler, made general, is a metric. Recall that the **domain** \(\mathcal X\) is the set of all messages that could ever arrive, and \(\mathbb R\) is the set of real numbers. A **metric** on \(\mathcal X\) is a function \(\rho: \mathcal X \times \mathcal X \to \mathbb R\) (the Greek letter rho) that takes a pair of messages and returns the distance between them; \(\mathcal X \times \mathcal X\) is the set of all such pairs. When each message is a list of \(d\) scores, a point of \(\mathbb R^d\), the natural choice is the **Euclidean distance**,

\[ \rho(x, x') = \|x - x'\| = \sqrt{\textstyle\sum_{i=1}^d (x_i - x'_i)^2}. \]

Read it from the inside out.

- \(x_i\) is score number \(i\) of message \(x\), and \(x'_i\) (read “x prime i”) is the same score of the other message \(x'\).
- \(x_i - x'_i\) is the gap on that one score, and squaring it makes every gap positive.
- \(\sum_{i=1}^d\) adds those squared gaps for \(i = 1, 2, \dots, d\), one per score.
- The square root undoes the squaring, so the answer is in score units again.
- \(\|x - x'\|\), with double bars, is the **norm** of the difference: its length. It is a second name for the same number.

In plain words: **the distance between two messages is the straight-line length you would measure with a ruler between their points.** For our two invoices both gaps are 0.05, the squares are 0.0025 each, their sum is 0.005, and its square root is 0.071.

```python
import math

x, u = (0.70, 0.70), (0.65, 0.65)
gaps = [round(a - b, 2) for a, b in zip(x, u)]
squares = [g ** 2 for g in gaps]
rho = math.sqrt(sum(squares))
print(gaps, round(sum(squares), 4), round(rho, 3))
print(round(math.dist(x, u), 3))
```

It prints `[0.05, 0.05] 0.005 0.071` and `0.071`: the hand computation, and Python’s own `math.dist`, which the rest of the lesson uses. Throughout this lesson the domain is \(\mathcal X = [0, 1]^d\), the set of lists of \(d\) scores each between 0 and 1, and \(\rho\) is the Euclidean distance.

Let’s try the idea on paper first. Here are eight invoices your users labelled, described by just the payment score \(x_1\) and the sender score \(x_2\). Message G is the interesting one: a real invoice from a supplier you use, with a moderately high payment score and a slightly unfamiliar sending address.

| message | payment \(x_1\) | sender \(x_2\) | label |
|---|---|---|---|
| A | 0.90 | 0.80 | junk |
| B | 0.85 | 0.40 | junk |
| C | 0.70 | 0.90 | junk |
| D | 0.20 | 0.20 | keep |
| E | 0.30 | 0.70 | keep |
| F | 0.10 | 0.50 | keep |
| G | 0.65 | 0.65 | keep |
| H | 0.40 | 0.10 | keep |

Six new invoices arrive. For each one, we sort the eight labelled messages by distance and look at the closest. Before you read the table, guess the label of q3, which sits at (0.70, 0.70), right next to G.

| new | scores | three nearest (distance) | 1 nearest says | 3 nearest vote |
|---|---|---|---|---|
| q1 | (0.90, 0.55) | B 0.158, A 0.250, G 0.269 | junk | junk (2 of 3) |
| q2 | (0.15, 0.30) | D 0.112, F 0.206, H 0.320 | keep | keep (0 of 3) |
| q3 | (0.70, 0.70) | G 0.071, C 0.200, A 0.224 | keep | junk (2 of 3) |
| q4 | (0.50, 0.30) | H 0.224, D 0.316, B 0.364 | keep | keep (1 of 3) |
| q5 | (0.75, 0.20) | B 0.224, H 0.364, G 0.461 | junk | keep (1 of 3) |
| q6 | (0.45, 0.85) | E 0.212, C 0.255, G 0.283 | keep | keep (1 of 3) |

Notice q3 and q5. If you trust only the single closest message, q3 inherits G’s “keep”, even though everything else around it is junk. If you let three neighbours vote, q3 becomes junk, but q5 flips the other way: its closest message B is junk, while its second and third neighbours outvote B. Neither choice is right everywhere. The number of neighbours you consult is a real decision, and we will see what it trades.

Now the general rule. What we did for q3 was to number the eight labelled messages A to H as 1 to 8, then list those numbers again in order of distance from q3: G (number 7) first, then C (3), then A (1), and so on. The rest of the lesson needs a name for that list.

Recall that the **training sequence** is the list of labelled examples, \(S = ((x_1, y_1), \dots, (x_m, y_m))\): message \(x_i\) with its label \(y_i\), for \(m\) examples. For each message \(x \in \mathcal X\) (read \(\in\) as “in”), let \(\pi_1(x), \dots, \pi_m(x)\) (the Greek letter pi) be the indices \(1, \dots, m\) reordered by distance to \(x\), closest first:

\[ \rho(x, x_{\pi_i(x)}) \le \rho(x, x_{\pi_{i+1}(x)}) \quad \text{for all } i < m. \]

- \(\pi_i(x)\) is a position in the training sequence: the number of the \(i\)-th closest message to \(x\).
- \(x_{\pi_i(x)}\) is a subscript inside a subscript: the training message sitting at that position.
- The inequality says each message on the list is no further from \(x\) than the next one.

In plain words: **\(\pi_1(x), \pi_2(x), \dots\) lists the training messages from nearest to furthest.** For q3, \(\pi_1 = 7\), \(\pi_2 = 3\), \(\pi_3 = 1\), with distances 0.071, 0.200 and 0.224.

```python
import math

X = [(0.9, 0.8), (0.85, 0.4), (0.7, 0.9), (0.2, 0.2),
     (0.3, 0.7), (0.1, 0.5), (0.65, 0.65), (0.4, 0.1)]
x = (0.7, 0.7)
rho = lambda i: math.dist(x, X[i - 1])
pi = sorted(range(1, 9), key=lambda i: (rho(i), i))
print(pi)
print([round(rho(i), 3) for i in pi])
```

It prints `[7, 3, 1, 2, 5, 6, 8, 4]` and `[0.071, 0.2, 0.224, 0.335, 0.4, 0.632, 0.671, 0.707]`: every distance is at most the next. The sort key `(rho(i), i)` is the tie rule: when two training messages are at exactly the same distance, the smaller index goes first, so the reordering is well defined. Then \(x_{\pi_1(x)}\) is the **nearest neighbour** of \(x\) in \(S\), and \(x_{\pi_1(x)}, \dots, x_{\pi_k(x)}\) are its \(k\) nearest neighbours.

**The k-NN rule.** For a number \(k\), the **k-NN rule** for binary classification returns, for every \(x \in \mathcal X\), the majority label among \(\{y_{\pi_i(x)} : i \le k\}\), the labels of the \(k\) nearest neighbours (the braces collect them, and the colon reads “such that”). For q3 with \(k = 3\) that is the labels of G, C and A: keep, junk, junk, so the majority says junk. With \(k = 1\) it is the **1-NN rule**,

\[ h_S(x) = y_{\pi_1(x)}. \]

Here \(h_S\) is the hypothesis the rule builds from \(S\) (the subscript records which training sequence), and \(y_{\pi_1(x)}\) is the label of the nearest neighbour. In plain words: **the 1-NN rule copies the label of the single closest labelled message.** For q3 that is \(y_7\), G’s “keep”. (For binary labels, an odd \(k\) avoids tied votes.) The 1-NN rule carves the domain into cells, one per training message: a cell holds every point closer to its training message than to any other, and every point in it gets that message’s label. This picture of the domain is called a **Voronoi tessellation**.

[[diagram:voronoi]]

Here is the rule as code, run on the table above.

```python
import math

S = [((0.9, 0.8), 1), ((0.85, 0.4), 1), ((0.7, 0.9), 1),
     ((0.2, 0.2), 0), ((0.3, 0.7), 0), ((0.1, 0.5), 0),
     ((0.65, 0.65), 0), ((0.4, 0.1), 0)]
Q = [(0.9, 0.55), (0.15, 0.3), (0.7, 0.7),
     (0.5, 0.3), (0.75, 0.2), (0.45, 0.85)]

def knn(S, x, k):
    near = sorted(S, key=lambda z: math.dist(z[0], x))[:k]
    return int(sum(y for _, y in near) > k / 2)

print([knn(S, x, 1) for x in Q])
print([knn(S, x, 3) for x in Q])
```

It prints `[1, 0, 0, 0, 1, 0]` and `[1, 0, 1, 0, 0, 0]`: the two columns of the table, with 1 for junk.

The same idea works for numbers instead of labels. Recall that \(\mathcal Y\) is the **label set**, so far \(\{0, 1\}\) with 1 for junk. If \(\mathcal Y = \mathbb R\), say a junk score you want to predict, the natural prediction is the average target of the \(k\) nearest neighbours, \(h_S(x) = \frac1k \sum_{i=1}^k y_{\pi_i(x)}\): add up the targets of neighbours 1 to \(k\), and divide by \(k\). For q1 with \(k = 3\) the neighbours B, A, G carry 1, 1, 0, and the average is \(2/3\).

More generally, any recipe that turns the \(k\) nearest examples into a prediction will do. Write it as a function \(\phi\) (the Greek letter phi), with \(\phi: (\mathcal X \times \mathcal Y)^k \to \mathcal Y\): its input is \(k\) message-label pairs, its output a label. The k-NN rule with respect to \(\phi\) is

\[ h_S(x) = \phi\big( (x_{\pi_1(x)}, y_{\pi_1(x)}), \dots, (x_{\pi_k(x)}, y_{\pi_k(x)}) \big). \]

Inside the brackets sit the nearest example, the second nearest, up to the \(k\)-th, each as a (message, label) pair; \(\phi\) reads them and answers. In plain words: **find the \(k\) nearest labelled examples and let one fixed recipe turn them into a prediction.** Majority vote and the plain average are two choices of \(\phi\), and on q1 they answer 1 and \(2/3\):

```python
import math

S = [((0.9, 0.8), 1), ((0.85, 0.4), 1), ((0.7, 0.9), 1),
     ((0.2, 0.2), 0), ((0.3, 0.7), 0), ((0.1, 0.5), 0),
     ((0.65, 0.65), 0), ((0.4, 0.1), 0)]
x, k = (0.9, 0.55), 3
near = sorted(S, key=lambda z: math.dist(z[0], x))[:k]
ys = [y for _, y in near]
mean = lambda ys: sum(ys) / len(ys)
vote = lambda ys: int(mean(ys) > 1 / 2)
print(ys, round(mean(ys), 3), vote(ys))
```

It prints `[1, 1, 0] 0.667 1`. A weighted average whose weights depend on the neighbours’ distances is a third choice of \(\phi\). For the rest of the lesson we stay with binary labels and the 0-1 loss.

## No class to search, but still a bias

Pause on what just happened, because it is new. ERM, structural risk minimization, boosting and every learner since lesson 2 chose a hypothesis from a class \(\mathcal H\) fixed in advance. The k-NN rule picks a label for any new message without searching any predefined class of functions. It stores the training sequence and, at prediction time, looks things up.

That should ring an alarm. Lesson 1’s memorizer also stored the training sequence and looked things up, and it learned nothing. What is different? The memorizer answered only for messages it had seen exactly, and fell back on a blind “keep” for everything else. The nearest neighbour rule answers for unseen messages by borrowing from seen ones that are *close*. That borrowing is generalization, and it rests on an assumption: that the scores describing a message are relevant to its label in such a way that nearby messages are likely to have the same label.

So the bias has not disappeared. It has moved out of a hypothesis class and into the distance. If the scores you compare on are the wrong ones, or if junk changes abruptly between two messages that look nearly identical, nearest neighbour is the pigeon of lesson 1 all over again. The rest of this lesson makes “nearby messages have similar labels” precise, proves what it buys, and shows what it costs.

## Where it works and where it fails

Let’s reproduce the bug. Here is the secret behind both kinds of mail, the one I promised in the opening. A message is a point \(x \in [0, 1]^{30}\), thirty scores, with every score drawn independently and uniformly between 0 and 1. Users do not all agree: show the same invoice to ten of them and perhaps seven click “junk”. Call the chance that a message \(x\) gets labelled junk \(\eta(x)\) (the Greek letter eta); for that invoice \(\eta(x) = 0.7\). In this world it depends only on the payment score \(x_1\) and the sender score \(x_2\):

\[ \eta(x) = \min\{1, \max\{0,\; 2x_1 + 2x_2 - 1.5\}\}. \]

- Inside: \(2x_1 + 2x_2 - 1.5\) grows with both scores, and can fall below 0 or rise above 1.
- \(\max\{0, \cdot\}\) takes the larger of 0 and that number, so anything negative becomes 0.
- \(\min\{1, \cdot\}\) takes the smaller of 1 and the result, so anything above 1 becomes 1. Together they **clip** the number into \([0, 1]\), as a probability must be.

In plain words: **the chance of junk rises in a straight line with the payment and sender scores, held between 0 and 1.** A message at payment 0.6 and sender 0.5 gives \(1.2 + 1.0 - 1.5 = 0.7\), nothing to clip, so \(\eta = 0.7\); one at (0.2, 0.3) gives \(-0.5\), clipped to 0; one at (0.9, 0.8) gives 1.9, clipped to 1.

```python
def eta(x):
    t = 2 * x[0] + 2 * x[1] - 1.5
    return min(1.0, max(0.0, t))

for x in [(0.2, 0.3), (0.6, 0.5), (0.9, 0.8)]:
    t = 2 * x[0] + 2 * x[1] - 1.5
    print(x, round(t, 2), round(eta(x), 2))
```

It prints `(0.2, 0.3) -0.5 0.0`, `(0.6, 0.5) 0.7 0.7` and `(0.9, 0.8) 1.9 1.0`. So a message with low payment and sender scores (\(x_1 + x_2 \le 0.75\)) is always keep, one with high scores (\(x_1 + x_2 \ge 1.25\)) is always junk, and in between there is a band where users disagree. The other twenty-eight scores carry no information about the label at all.

Now run the copy-your-neighbours filter twice on the same 1,000 labelled messages: once comparing only the first two scores, as the parcel team did, and once comparing all thirty, as the invoice team did.

```python
import math, random

def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))
def draw(m, rng):
    xs = [[rng.random() for _ in range(30)] for _ in range(m)]
    return [(x, int(rng.random() < eta(x))) for x in xs]
def nn_label(S, x, d):
    return min(S, key=lambda z: math.dist(z[0][:d], x[:d]))[1]

rng = random.Random(1)
S, test = draw(1000, rng), draw(1000, rng)
for d in (2, 30):
    err = sum(nn_label(S, x, d) != y for x, y in test) / 1000
    print(d, round(err, 2))
```

It prints `2 0.14` and `30 0.32`. Same messages, same labels, same rule; the only change is which scores enter the distance. That is the bug, reproduced.

You can already guess the mechanism. For two independent uniform scores, the average squared difference is \(1/6\). With thirty scores, the two that matter contribute about \(2/6\) to the squared distance on average, and the twenty-eight that don’t contribute about \(28/6\). The nearest neighbour in thirty dimensions is chosen mostly by noise, so it is often a message whose payment and sender scores look nothing like the new one’s. We will make this precise, but first, play with it.

In the lab below you control the number of labelled messages \(m\), the number of scores the distance uses (the first \(d\) of the thirty), and \(k\). You see the filter’s measured true risk, along with the cells of the 1-NN rule when \(d = 2\).

[[lab]]

## Labels that change slowly

To prove anything, we have to say what “nearby messages have similar labels” means. Recall from lesson 3 that the labels need not be a function of the message: users disagree. Examples come from a **distribution** \(\mathcal D\) over \(\mathcal X \times \mathcal Y\), with \(\mathcal Y = \{0, 1\}\): a rule saying how likely each (message, label) pair is. Ignoring the labels leaves its **marginal** \(\mathcal D_x\), how likely each message is. And for each message there is the **conditional probability of junk**, the \(\eta\) we just used:

\[ \eta(x) = \mathbb P[y = 1 \mid x]. \]

\(\mathbb P[\cdot]\) is the probability of the event inside, here “the label \(y\) is 1, junk”. The vertical bar reads “given”: we look only at examples whose message is \(x\). In plain words: **\(\eta(x)\) is the fraction of users who would call message \(x\) junk.** At (0.6, 0.5) it is 0.7, and drawing ten thousand labels for that one message shows it:

```python
import random

def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))

rng, x = random.Random(6), (0.6, 0.5)
ys = [int(rng.random() < eta(x)) for _ in range(10_000)]
print(round(eta(x), 2), round(sum(ys) / len(ys), 3))
```

It prints `0.7 0.701`.

What is the best any filter could do on that message? Saying junk, it is wrong whenever the user says keep, 30% of the time; saying keep, it is wrong 70% of the time. The best choice is the more likely label, and it still errs with the less likely probability, 0.3. The **Bayes optimal predictor** \(f_{\mathcal D}\) makes that best choice everywhere: it says junk exactly when \(\eta(x) \ge 1/2\), and lesson 3 proved that no hypothesis has smaller true risk. Recall that the **true risk** \(L_{\mathcal D}(h)\) of a hypothesis \(h\) is the chance that it mislabels a fresh example drawn from \(\mathcal D\). The true risk of \(f_{\mathcal D}\), the **Bayes risk**, is

\[ L_{\mathcal D}(f_{\mathcal D}) = \mathbb E_{x \sim \mathcal D_x}\big[ \min\{\eta(x), 1 - \eta(x)\} \big]. \]

- \(\min\{\eta(x), 1 - \eta(x)\}\) is the smaller of the two label probabilities at \(x\): the error of the best choice there.
- \(x \sim \mathcal D_x\) reads “\(x\) is drawn from \(\mathcal D_x\)”: a fresh message arrives at random.
- \(\mathbb E[\cdot]\), the **expectation**, is the average of the bracket over that random message, each message weighted by how likely it is.

In plain words: **the Bayes risk is the unavoidable error, averaged over all messages.** At (0.6, 0.5) the unavoidable error is \(\min\{0.7, 0.3\} = 0.3\). Messages are uniform on the square here, so the expectation is a plain average over a fine grid of messages:

```python
def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))

n = 400
grid = [((i + 0.5) / n, (j + 0.5) / n)
        for i in range(n) for j in range(n)]
risk = [min(eta(x), 1 - eta(x)) for x in grid]
print(round(sum(risk) / len(risk), 4), round(11 / 96, 4))
```

It prints `0.1146 0.1146`: the invoice world’s Bayes risk is \(11/96\), a number we will use often.

Now the assumption. In the invoice world, nudge the message at (0.6, 0.5) to (0.62, 0.52). The distance moved is \(\sqrt{0.02^2 + 0.02^2} \approx 0.028\), and \(\eta\) moves from 0.70 to 0.78, a change of 0.08: about 2.83 times the distance. Close messages, close probabilities of junk, with a fixed exchange rate between the two. That exchange rate is the constant in the definition. A function \(\eta: \mathcal X \to [0, 1]\) is **c-Lipschitz**, for a constant \(c > 0\), if for all \(x, x' \in \mathcal X\),

\[ |\eta(x) - \eta(x')| \le c\, \|x - x'\|. \]

The single bars \(|\cdot|\) are the absolute value, the size of a change ignoring its sign; the double bars are the distance from before. In plain words: **if two messages are close, their probabilities of junk are close: moving a distance \(r\) through the domain changes \(\eta\) by at most \(cr\).** A small \(c\) means junk changes gently from message to message; a large \(c\) means a tiny change in the scores can flip the odds. Lesson 12 used the same word for losses; here it is a property of the world, not of anything we choose.

[[diagram:lipschitz]]

Our invoice world’s \(\eta\) is \(2\sqrt 2\)-Lipschitz. Here is why. Inside the clip, the expression \(2x_1 + 2x_2 - 1.5\) changes by \(2(x'_1 - x_1) + 2(x'_2 - x_2)\), and by the Cauchy–Schwarz inequality that is at most \(\|(2, 2)\| \cdot \|x' - x\| = 2\sqrt 2\, \|x' - x\|\) in absolute value. The clip \(t \mapsto \min\{1, \max\{0, t\}\}\) never moves two numbers further apart than they were, so it keeps the bound. The twenty-eight noise scores do not enter \(\eta\) at all, so the same constant works in thirty dimensions. Let’s check by brute force: the ratio we computed for one nudge above, now for many pairs of nearby messages, keeping the largest.

```python
import math, random

def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))

rng, worst = random.Random(0), 0.0
nudge = lambda t: t + rng.uniform(-0.01, 0.01)
for _ in range(200_000):
    x = (rng.random(), rng.random())
    u = (nudge(x[0]), nudge(x[1]))
    if math.dist(x, u) > 0:
        worst = max(worst, abs(eta(x) - eta(u)) / math.dist(x, u))
print(round(worst, 3), round(2 * math.sqrt(2), 3))
```

It prints `2.828 2.828`: the worst ratio found is the constant, reached when you step diagonally inside the band where users disagree.

Two more conventions before the proofs. The learner’s output depends on the random training sequence: a different thousand labelled invoices gives a different 1-NN rule, with a different true risk. So we measure it on average over many draws of the training sequence. Recall that \(S \sim \mathcal D^m\) means the \(m\) examples of \(S\) are drawn independently from \(\mathcal D\). The guarantee will bound \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(h_S)]\): **the true risk of the 1-NN rule, averaged over the random training sequence.** The simulation in Step 1 will put it at about 0.152 for a thousand invoices on two scores. And the setting throughout is \(\mathcal X = [0, 1]^d\), \(\mathcal Y = \{0, 1\}\), the 0-1 loss \(\ell(h, (x, y)) = \mathbb 1[h(x) \neq y]\), and Euclidean distance. The **indicator** \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false, so the 0-1 loss charges 1 for a mistake and 0 otherwise.

The guarantee comes in three steps. Step 1 turns the error of the 1-NN rule into a distance: how far a fresh message is from its nearest neighbour. Step 2 is a counting fact about regions that a random sample misses. Step 3 combines them to bound that distance, and hence the error.

## Step 1: from mistakes to distances

**Lemma (error through distance).** Let \(\mathcal X = [0, 1]^d\), \(\mathcal Y = \{0, 1\}\), and let \(\mathcal D\) be a distribution over \(\mathcal X \times \mathcal Y\) whose conditional probability \(\eta\) is c-Lipschitz. Let \(S \sim \mathcal D^m\) and let \(h_S\) be the 1-NN rule on \(S\). Then

\[ \mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(h_S)] \le 2\, L_{\mathcal D}(f_{\mathcal D}) + c\, \mathbb E_{S \sim \mathcal D^m,\, x \sim \mathcal D_x}\big[ \|x - x_{\pi_1(x)}\| \big]. \]

Before the proof, read what it says, piece by piece.

- On the left, the average true risk of the 1-NN rule, as just defined.
- \(2\, L_{\mathcal D}(f_{\mathcal D})\) is twice the Bayes risk, a price we will understand in a moment.
- \(\|x - x_{\pi_1(x)}\|\) is the distance from a message \(x\) to its nearest neighbour in \(S\).
- The expectation under it draws both at once: the training sequence \(S\), and a fresh message \(x \sim \mathcal D_x\). So it is the average nearest-neighbour distance, and \(c\) times it is what Lipschitz lets \(\eta\) drift over that distance.

In plain words: **the 1-NN rule’s average risk is at most twice the unavoidable error, plus \(c\) times the average distance to the nearest neighbour.** If labelled messages crowd the domain densely, that distance is small, and the second term vanishes. In the invoice world with a thousand labels on two scores, the average distance comes out near 0.016, so the bound is \(2 \times 0.115 + 2.83 \times 0.016 \approx 0.275\); the simulation after the proof computes both sides.

*Proof.* We go in five steps.

**Step 1a: the expected risk is one probability.** For a fixed \(S\), the true risk is \(L_{\mathcal D}(h_S) = \mathbb P_{(x, y) \sim \mathcal D}[h_S(x) \neq y]\), a probability over a fresh example drawn independently of \(S\). Averaging a conditional probability over the thing it conditions on gives the joint probability (the law of total expectation, applied to the indicator \(\mathbb 1[h_S(x) \neq y]\)). So \(\mathbb E_S[L_{\mathcal D}(h_S)]\) is the probability of the following event: draw \(S\) and a fresh example \((x, y)\), and find that \(y\) differs from \(y_{\pi_1(x)}\), the label of the nearest neighbour.

**Step 1b: draw the messages first, the labels second.** Every example \((x_i, y_i)\) is drawn from \(\mathcal D\), which is the same as drawing \(x_i \sim \mathcal D_x\) and then \(y_i\) with \(\mathbb P[y_i = 1 \mid x_i] = \eta(x_i)\). So we may run the experiment in two stages. First draw the unlabelled messages \(x_1, \dots, x_m\) and the fresh \(x\), all independently from \(\mathcal D_x\). That alone fixes the nearest neighbour, because \(\pi_1(x)\) depends on distances and not on labels; call it \(x' = x_{\pi_1(x)}\). Then draw the labels. Because the \(m + 1\) examples are independent, each label depends only on its own message: given all the messages, \(y\) is a coin with probability \(\eta(x)\) of being 1, the label \(y'\) of \(x'\) is a coin with probability \(\eta(x')\), and the two coins are independent (they belong to different examples). Therefore

\[ \mathbb E_S[L_{\mathcal D}(h_S)] = \mathbb E_{x_1, \dots, x_m,\, x}\Big[ \mathbb P_{y \sim \eta(x),\; y' \sim \eta(x')}[\, y \neq y' \,] \Big], \]

where \(y \sim p\) means a label that is 1 with probability \(p\) and 0 otherwise. Read it from the inside: the inner \(\mathbb P\) flips the two coins for fixed messages and asks how often they disagree; the outer \(\mathbb E\), with subscripts naming what is random, averages that over the messages. In plain words: **the expected risk is the average, over random messages, of the chance that two coins, one biased by \(x\) and one by its nearest neighbour, come up different.**

**Step 1c: two coins that disagree.** Fix two messages \(x, x'\) and write \(p = \eta(x)\), \(q = \eta(x')\). Two independent coins disagree when the first is 0 and the second is 1, or the other way round:

\[ \mathbb P[y \neq y'] = q(1 - p) + (1 - q)p. \]

Now the one trick of the proof: write \(q = p - (p - q)\) in the first term and \(1 - q = (1 - p) + (p - q)\) in the second, and expand.

\[ q(1 - p) + (1 - q)p = p(1-p) - (p - q)(1 - p) + (1 - p)p + (p - q)p = 2p(1 - p) + (p - q)(2p - 1). \]

The first part, \(2p(1-p)\), is what you would pay if the neighbour’s coin were exactly as biased as yours, \(q = p\). The second part is the penalty for the neighbour being a different message. In plain words: **two coins disagree with probability \(2p(1-p)\), plus a correction proportional to how different their biases are.** Take the message at (0.6, 0.5), so \(p = 0.7\), and a neighbour with \(q = 0.6\). The left side is \(0.6 \times 0.3 + 0.4 \times 0.7 = 0.46\); the right is \(0.42 + 0.1 \times 0.4 = 0.46\). Flipping the coins agrees:

```python
import random

p, q = 0.7, 0.6
differ = q * (1 - p) + (1 - q) * p
split = 2 * p * (1 - p) + (p - q) * (2 * p - 1)
rng, n = random.Random(7), 100_000
flips = sum((rng.random() < p) != (rng.random() < q)
            for _ in range(n))
print(round(differ, 2), round(split, 2), round(flips / n, 3))
```

It prints `0.46 0.46 0.458`.

**Step 1d: Lipschitz bounds the penalty.** Since \(0 \le p \le 1\), we have \(|2p - 1| \le 1\). Since \(\eta\) is c-Lipschitz, \(|p - q| \le c\|x - x'\|\). So the penalty is at most \(|p - q| \cdot |2p - 1| \le c\|x - x'\|\), and

\[ \mathbb P_{y \sim \eta(x),\; y' \sim \eta(x')}[y \neq y'] \le 2\eta(x)(1 - \eta(x)) + c\,\|x - x'\|. \]

For our pair, 0.46 is indeed at most \(0.42 + 0.1\): the penalty 0.04 is below \(|p - q| = 0.1\). Put this into the formula of Step 1b, with \(x' = x_{\pi_1(x)}\), and use linearity of expectation (the average of a sum is the sum of the averages). The first term depends only on \(x\), so its expectation is over \(x \sim \mathcal D_x\) alone:

\[ \mathbb E_S[L_{\mathcal D}(h_S)] \le \mathbb E_x\big[ 2\eta(x)(1 - \eta(x)) \big] + c\, \mathbb E_{S, x}\big[\|x - x_{\pi_1(x)}\|\big]. \]

**Step 1e: compare with the Bayes risk.** For any \(p \in [0, 1]\), \(\min\{p, 1-p\} \ge p(1-p)\), because \(p(1-p) = \min\{p, 1-p\} \cdot \max\{p, 1-p\}\) and the max is at most 1. Taking expectations, \(L_{\mathcal D}(f_{\mathcal D}) = \mathbb E_x[\min\{\eta(x), 1 - \eta(x)\}] \ge \mathbb E_x[\eta(x)(1 - \eta(x))]\), so the first term is at most \(2L_{\mathcal D}(f_{\mathcal D})\). At \(p = 0.7\): \(\min\{0.7, 0.3\} = 0.3\), at least \(0.7 \times 0.3 = 0.21\). \(\blacksquare\)

The coin snippet checked one pair; the algebra of Steps 1c to 1e is worth checking on a whole grid of values for \(p\) and \(q\), since a sign slip there would sink the whole argument.

```python
grid = [i / 100 for i in range(101)]
gap = 0.0
for p in grid:
    for q in grid:
        differ = q * (1 - p) + (1 - q) * p
        split = 2 * p * (1 - p) + (p - q) * (2 * p - 1)
        gap = max(gap, abs(differ - split))
        assert differ <= 2 * p * (1 - p) + abs(p - q) + 1e-12
    assert min(p, 1 - p) >= p * (1 - p)
print(gap < 1e-12)
```

It prints `True`: the identity holds on all 10,201 pairs, and both inequalities never fail.

Now let’s see the lemma at work in the invoice world, with the two scores that matter (\(d = 2\)) and \(m = 1{,}000\). The code below follows the proof exactly: it draws the messages, finds the nearest neighbour, and adds up the disagreement probability of Step 1c instead of flipping the coins. The Bayes risk of this world is \(11/96 \approx 0.115\), as the grid average showed earlier.

```python
import math, random

def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))
def point(rng):
    return (rng.random(), rng.random())

rng, m, runs = random.Random(2), 1000, 2000
err = dist = 0.0
for _ in range(runs):
    S = [point(rng) for _ in range(m)]
    x = point(rng)
    nn = min(S, key=lambda s: math.dist(s, x))
    p, q = eta(x), eta(nn)
    err += q * (1 - p) + (1 - q) * p
    dist += math.dist(x, nn)
c, bayes = 2 * math.sqrt(2), 11 / 96
print(round(err / runs, 3), round(2 * bayes + c * dist / runs, 3))
```

It prints `0.152 0.275`. The expected true risk of the 1-NN rule is about 0.152, and the lemma’s bound is 0.275. Where does the gap come from? Almost all of it is Step 1e. In this world \(\mathbb E_x[2\eta(x)(1 - \eta(x))] \approx 0.151\), so the first inequality, the Lipschitz one, is nearly tight: the 1-NN rule really does pay about \(2\eta(1-\eta)\) per message. What we gave away is the replacement of \(\eta(1-\eta)\) by \(\min\{\eta, 1-\eta\}\), which is exact only where \(\eta\) is 0, \(1/2\) or 1. Hold on to that: “twice the Bayes risk” is a ceiling, and real 1-NN rules usually sit below it.

## Step 2: boxes nobody sampled

The lemma leaves one quantity to control: the expected distance from a fresh message to its nearest neighbour. The plan is to chop the domain into small boxes. If the fresh message lands in a box that contains a training message, its nearest neighbour is at most a box-width away. The danger is landing in a box that the training sequence missed entirely. So we need to know how much probability the missed boxes carry, and that is a clean fact about random samples, true for any collection of sets.

**Lemma (missed mass).** Let \(C_1, \dots, C_r\) be subsets of a domain \(\mathcal X\), and let \(S\) be a sequence of \(m\) points drawn i.i.d. from a distribution \(\mathcal D\) over \(\mathcal X\). Then

\[ \mathbb E_{S \sim \mathcal D^m}\Big[ \sum_{i \,:\, C_i \cap S = \emptyset} \mathbb P[C_i] \Big] \le \frac{r}{me}. \]

Read it from the inside out.

- \(\mathbb P[C_i]\) is the probability \(\mathcal D\) gives to set \(C_i\): the chance that one random point lands in it.
- \(C_i \cap S\) (read \(\cap\) as “and also in”) is the points of \(S\) inside \(C_i\), and \(\emptyset\) is the empty set. So \(C_i \cap S = \emptyset\) means that none of the \(m\) points landed in \(C_i\): the set was **missed**.
- The sum runs over the indices \(i\) such that \(C_i\) was missed, adding their probabilities: the **missed mass**.
- The outer expectation averages the missed mass over the random draw of \(S\).
- On the right, \(e \approx 2.718\) is the base of the natural logarithm.

In plain words: **on average, the sets a random sample misses carry at most \(r/(me)\) of the probability.** With 25 sets and 40 points, that is \(25/(40 \times 2.718) \approx 0.23\).

*Proof.* **Linearity.** Write the random sum with an indicator for each set: \(\sum_{i : C_i \cap S = \emptyset} \mathbb P[C_i] = \sum_{i=1}^r \mathbb P[C_i] \cdot \mathbb 1[C_i \cap S = \emptyset]\). The numbers \(\mathbb P[C_i]\) are fixed, so by linearity of expectation the expected sum is \(\sum_{i=1}^r \mathbb P[C_i] \cdot \mathbb P_S[C_i \cap S = \emptyset]\).

**One set at a time.** Fix \(i\). A single point misses \(C_i\) with probability \(1 - \mathbb P[C_i]\). The \(m\) points are independent, so the probability that all of them miss is the product, \((1 - \mathbb P[C_i])^m\). Now use \(1 - a \le e^{-a}\), which holds for every real \(a\) because \(e^{-a}\) is convex and \(1 - a\) is its tangent line at \(a = 0\). Raising both sides (both nonnegative here) to the power \(m\) gives \((1 - \mathbb P[C_i])^m \le e^{-\mathbb P[C_i]\, m}\). For a set of probability \(1/40\) and 40 points, the left side is \((39/40)^{40} \approx 0.363\) and the right is \(e^{-1} \approx 0.368\).

**The worst size of a set.** So each term is at most \(a e^{-ma}\) with \(a = \mathbb P[C_i]\). How large can \(a e^{-ma}\) be? Its derivative is \(e^{-ma}(1 - ma)\), positive for \(a < 1/m\) and negative for \(a > 1/m\), so the maximum over \(a \ge 0\) is at \(a = 1/m\), where the value is \(\frac{1}{m} e^{-1} = \frac{1}{me}\). Each of the \(r\) terms is at most \(1/(me)\), and the sum is at most \(r/(me)\). \(\blacksquare\)

The middle step says something you can feel. A set with a lot of probability is almost certainly hit, so it rarely counts as missed. A set with tiny probability can be missed, but it carries almost nothing. The dangerous sets are the ones of size about \(1/m\): likely enough to matter, unlikely enough to be skipped. Here is a test with the 25 sets and 40 points from above: ten sets have exactly that dangerous probability \(1/40\), and fifteen share the rest. The code mirrors the lemma: `hit` is the set of indices some point landed in, the inner sum adds the probabilities of the others, and the loop averages over draws of \(S\).

```python
import math, random

r, m = 25, 40
P = [1 / m] * 10 + [0.05] * 15
exact = sum(p * (1 - p) ** m for p in P)
rng, runs, missed = random.Random(3), 20_000, 0.0
for _ in range(runs):
    hit = set(rng.choices(range(r), weights=P, k=m))
    missed += sum(p for i, p in enumerate(P) if i not in hit)
print(round(sum(P), 6), round(missed / runs, 3), round(exact, 3))
print(round(r / (m * math.e), 3))
a = [i / 10000 for i in range(1, 10000)]
print(round(max(t * math.exp(-m * t) for t in a), 4),
      round(1 / (m * math.e), 4))
```

It prints `1.0 0.187 0.187`, then `0.23`, then `0.0092 0.0092`. The probabilities sum to 1; the simulated missed mass is 0.187, matching the exact \(\sum_i \mathbb P[C_i](1 - \mathbb P[C_i])^m\); the lemma’s bound is 0.23; and the largest value of \(a e^{-40a}\) on a fine grid is \(1/(40e)\).

## Step 3: how far is the nearest neighbour?

Now the boxes. Picture the invoice square cut by a 5 by 5 grid into 25 small squares of side 0.2. The new invoice q3 at (0.70, 0.70) sits in the fourth column and fourth row, the square from 0.6 to 0.8 on both scores, and so does G at (0.65, 0.65): same box, so q3’s nearest neighbour is at most one box diagonal away.

In general, pick a whole number \(T\) and let \(\varepsilon = 1/T\) (the Greek letter epsilon, here a box side). Cut each axis of \([0, 1]^d\) into \(T\) intervals of length \(\varepsilon\), and take all products. Recall that \([T]\) is the list \(1, \dots, T\); so \([T]^d\) is the set of all lists of \(d\) such numbers, and for every choice \(\alpha = (\alpha_1, \dots, \alpha_d) \in [T]^d\) (the Greek letter alpha) there is a box

\[ C_\alpha = \{ x : x_j \in [(\alpha_j - 1)/T,\; \alpha_j/T] \text{ for every } j \}. \]

- \(\alpha_j\) says which of the \(T\) slices to take along score \(j\).
- \([(\alpha_j - 1)/T,\; \alpha_j/T]\) is that slice: the interval from its left edge to its right edge.
- The braces collect every message \(x\) such that each score \(x_j\) lies in its slice.

In plain words: **box \(\alpha\) holds the messages whose score \(j\) falls in slice \(\alpha_j\), for every score at once.** For q3 and G, \(\alpha = (4, 4)\), the slice \([0.6, 0.8]\) on both scores.

```python
import math

T = 5
for x in [(0.7, 0.7), (0.65, 0.65), (0.15, 0.3)]:
    alpha = tuple(max(1, math.ceil(t * T)) for t in x)
    lo = [round((a - 1) / T, 1) for a in alpha]
    hi = [round(a / T, 1) for a in alpha]
    print(x, alpha, lo, hi)
print(T ** 2, round(math.sqrt(2) / T, 3))
```

It prints `(0.7, 0.7) (4, 4) [0.6, 0.6] [0.8, 0.8]`, the same box for `(0.65, 0.65)`, then `(0.15, 0.3) (1, 2) [0.0, 0.2] [0.2, 0.4]` for q2, and finally `25 0.283`. There are \(r = T^d\) boxes covering the whole cube, 25 here, and the last number is a box diagonal, which we need next.

[[diagram:boxes]]

Two distance facts do all the geometric work. Two points in the same box differ by at most \(\varepsilon\) in each of the \(d\) coordinates, so they are at most \(\sqrt{d \varepsilon^2} = \sqrt d\, \varepsilon\) apart: the diagonal of the box, \(\sqrt 2 \times 0.2 \approx 0.283\) on our grid. Any two points of the cube are at most \(\sqrt d\) apart: the diagonal of the cube, about 1.414 for the square.

**Bounding the distance for a fixed \(S\).** Fix \(S\) and draw a fresh \(x\). If some box containing \(x\) also contains a training message, then that message is within \(\sqrt d\, \varepsilon\) of \(x\), and the nearest neighbour is at least as close. Otherwise every box containing \(x\) was missed by \(S\), so \(x\) lies in the union of the missed boxes, and its nearest neighbour is still within \(\sqrt d\) (nothing in the cube is further). Splitting the expectation over these two cases, and bounding the probability of the first case by 1,

\[ \mathbb E_x\big[\|x - x_{\pi_1(x)}\|\big] \le \sqrt d\, \varepsilon + \sqrt d \cdot \mathbb P_x\Big[ x \in \bigcup_{i : C_i \cap S = \emptyset} C_i \Big] \le \sqrt d\, \varepsilon + \sqrt d \sum_{i : C_i \cap S = \emptyset} \mathbb P[C_i], \]

where the last step is the union bound from lesson 2 (the chance of landing in at least one of several sets is at most the sum of their chances).

- \(\mathbb E_x\) averages over the fresh message alone; \(S\) is held fixed.
- \(\bigcup_{i : C_i \cap S = \emptyset} C_i\) is the **union** of the missed boxes: every point lying in at least one of them.
- \(\mathbb P_x[x \in \dots]\) is the chance the fresh message lands there.

In plain words: **the average nearest-neighbour distance is at most one box diagonal, plus a cube diagonal times the missed mass.** On the 5 by 5 grid, if a sample of 40 points happened to miss boxes carrying 0.1 of the probability, the bound would be \(0.283 + 1.414 \times 0.1 \approx 0.42\).

**Averaging over \(S\).** Take the expectation over \(S \sim \mathcal D^m\); the boxes are fixed sets and the training messages are i.i.d. from \(\mathcal D_x\), so the missed-mass lemma applies with \(r = T^d\):

\[ \mathbb E_{S, x}\big[\|x - x_{\pi_1(x)}\|\big] \le \sqrt d \Big( \varepsilon + \frac{T^d}{me} \Big). \]

In plain words: **the average nearest-neighbour distance is at most a cube diagonal times the box side plus the average missed mass.** With \(m = 1{,}000\) on the 5 by 5 grid: \(\sqrt 2\,(0.2 + 25/(1000e)) = 1.414 \times 0.209 \approx 0.296\). The distance snippet below computes exactly this as its middle column.

**Choosing the box size.** Small boxes make \(\varepsilon\) small but there are more of them to miss; big boxes do the opposite. Let \(u = m^{1/(d+1)}\), the \((d+1)\)-th root of \(m\), so that \(u^{d+1} = m\). For \(m = 1{,}000\) and \(d = 2\), \(u = 10\).

If \(u \ge 2\), choose \(T = \lceil u/2 \rceil\), where \(\lceil \cdot \rceil\) rounds up to a whole number; for \(u = 10\) that is the \(T = 5\) we have been using. Then \(T \ge u/2\), so \(\varepsilon = 1/T \le 2/u = 2m^{-1/(d+1)}\). Also \(T \le u/2 + 1 \le u\) (because \(u \ge 2\)), so \(T^d \le u^d = m^{d/(d+1)}\) and \(\frac{T^d}{me} \le \frac{m^{d/(d+1)}}{me} = \frac{1}{e} m^{-1/(d+1)}\). Together,

\[ \mathbb E_{S, x}\big[\|x - x_{\pi_1(x)}\|\big] \le \sqrt d \Big(2 + \frac1e\Big) m^{-1/(d+1)} \le 4\sqrt d\, m^{-1/(d+1)}. \]

If \(u < 2\), then \(m^{-1/(d+1)} > 1/2\), so \(4\sqrt d\, m^{-1/(d+1)} > 2\sqrt d\), and the inequality holds trivially, since no distance in the cube exceeds \(\sqrt d\).

Combining with the first lemma gives the main result.

**Theorem (the 1-NN guarantee).** Let \(\mathcal X = [0, 1]^d\), \(\mathcal Y = \{0, 1\}\), and let \(\mathcal D\) be a distribution over \(\mathcal X \times \mathcal Y\) whose conditional probability \(\eta\) is c-Lipschitz. Let \(h_S\) be the 1-NN rule on \(S \sim \mathcal D^m\). Then

\[ \mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(h_S)] \le 2\, L_{\mathcal D}(f_{\mathcal D}) + 4c\sqrt d\; m^{-1/(d+1)}. \]

*Proof.* Insert the distance bound into the error-through-distance lemma. \(\blacksquare\)

Every symbol here has been walked through already; the new piece is \(m^{-1/(d+1)}\), one over the \((d+1)\)-th root of \(m\), which shrinks as labels pile up, and slowly when \(d\) is large. In plain words: **the 1-NN rule’s average risk is at most twice the Bayes risk, plus an excess that shrinks like one over the \((d+1)\)-th root of the number of labels.** In the invoice world on two scores with a million labels, \(m^{-1/3} = 0.01\), the excess is \(4 \times 2.83 \times 1.414 \times 0.01 = 0.16\), and the bound is \(0.229 + 0.16 = 0.389\); a snippet below tabulates it.

(The constant 4 is generous: the argument gives \(2 + 1/e \approx 2.37\). We keep 4 because it is round and every later number uses it.)

How good is the distance bound in practice? Here is the real average nearest-neighbour distance in two dimensions, next to the box bound \(\sqrt d(\varepsilon + T^d/(me))\) with our choice of \(T\), and the final \(4\sqrt d\, m^{-1/(d+1)}\).

```python
import math, random

def mean_nn_dist(m, d, rng, queries=300):
    S = [[rng.random() for _ in range(d)] for _ in range(m)]
    total = 0.0
    for _ in range(queries):
        x = [rng.random() for _ in range(d)]
        total += min(math.dist(s, x) for s in S)
    return total / queries

rng, d = random.Random(4), 2
for m in (100, 1000, 10000):
    T = math.ceil(m ** (1 / (d + 1)) / 2)
    boxes = math.sqrt(d) * (T ** d / (m * math.e) + 1 / T)
    final = 4 * math.sqrt(d) * m ** (-1 / (d + 1))
    print(m, round(mean_nn_dist(m, d, rng), 4),
          round(boxes, 3), round(final, 3))
```

It prints `100 0.0549 0.518 1.219`, `1000 0.0159 0.296 0.566` and `10000 0.005 0.135 0.263`. The bound is true and it is loose, by a factor that grows with \(m\): in two dimensions the real distance shrinks roughly like \(m^{-1/2}\), and the bound only like \(m^{-1/3}\). A box argument pays for boxes, not for the actual geometry. Keep that in mind when we turn the bound into label budgets. The exponent will matter more than the constant, and we will see shortly that the dimension in the exponent is not an artefact.

What does the theorem promise in the invoice world with two scores? There \(c = 2\sqrt 2\), \(d = 2\), and \(2L_{\mathcal D}(f_{\mathcal D}) = 22/96\).

```python
import math

c, d, bayes = 2 * math.sqrt(2), 2, 11 / 96
for m in (10**3, 10**5, 10**6, 10**7):
    excess = 4 * c * math.sqrt(d) * m ** (-1 / (d + 1))
    print(m, round(2 * bayes, 3), round(excess, 3),
          round(2 * bayes + excess, 3))
```

It prints `1000 0.229 1.6 1.829`, `100000 0.229 0.345 0.574`, `1000000 0.229 0.16 0.389` and `10000000 0.229 0.074 0.303`. At a thousand labels the promise is empty (a risk above 1 promises nothing); at a million it says 0.389; the measured risk, remember, was 0.14 at a thousand. A guarantee is a promise for *every* world with this Lipschitz constant and dimension, including far nastier ones than ours, and it has to pay for all of them.

One more reading of the theorem, the classical one. Fix the distribution and let \(m\) grow. The second term goes to zero, so the expected risk of the 1-NN rule ends up at most twice the Bayes risk. No class was searched, nothing about \(\mathcal D\) was known except the Lipschitz constant, and yet the rule gets within a factor of two of the best anyone could do.

## More neighbours: the k-NN guarantee

Why ever consult more than one neighbour? Look back at Step 1c: the 1-NN rule copies a single noisy coin, \(y'\), and pays \(2\eta(1-\eta)\) even when the neighbour is in exactly the right place. A majority vote of \(k\) coins with the same bias lands on the more likely label far more often than one coin does. That is the intuition behind the extension to \(k\) neighbours.

**Theorem (the k-NN guarantee).** Under the same assumptions, let \(h_S\) be the k-NN rule on \(S \sim \mathcal D^m\), with \(k \ge 10\). Then

\[ \mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(h_S)] \le \Big(1 + \sqrt{8/k}\Big) L_{\mathcal D}(f_{\mathcal D}) + \big(6c\sqrt d + k\big)\, m^{-1/(d+1)}. \]

The proof follows the same plan with two changes, and you will build it yourself in this lesson’s problems. First, a box is now dangerous not when it is empty but when it holds fewer than \(k\) training messages, and a concentration bound replaces the simple “all \(m\) points miss” computation. Second, the single coin of Step 1c becomes a majority vote of \(k\) coins, and a lemma about majority votes replaces the factor 2.

Here is what the new multiplier on the Bayes risk looks like (with the factor 2 of the 1-NN rule for comparison).

```python
import math

for k in (1, 10, 50, 100, 1000):
    factor = 2 if k == 1 else 1 + math.sqrt(8 / k)
    print(k, round(factor, 3))
```

It prints `1 2`, `10 1.894`, `50 1.4`, `100 1.283` and `1000 1.089`. As \(k\) grows, the multiplier slides towards 1, so with enough data the k-NN rule approaches the Bayes risk itself. But the price is the \(k\) in the second term: each extra neighbour adds \(m^{-1/(d+1)}\) to the bound, because consulting more neighbours means reaching further away. Choosing \(k\) is a tradeoff, and it depends on how many labels you have.

## The curse of dimensionality

Back to the bug. Both guarantees have a second term that shrinks like \(m^{-1/(d+1)}\). How many labels does it take to make the 1-NN term at most some accuracy \(\varepsilon\)? Rearranging \(4c\sqrt d\, m^{-1/(d+1)} \le \varepsilon\) gives \(m^{1/(d+1)} \ge 4c\sqrt d/\varepsilon\), and raising both sides to the power \(d+1\),

\[ m \ge \left( \frac{4c\sqrt d}{\varepsilon} \right)^{d+1}. \]

The dimension sits in the exponent. Here is that requirement for \(c = 2\) and \(\varepsilon = 0.1\), next to a second column, \((c+1)^d/2\), that we will explain in a moment.

```python
import math

c, eps = 2, 0.1
for d in (1, 2, 3, 5, 10, 30):
    upper = (4 * c * math.sqrt(d) / eps) ** (d + 1)
    lower = (c + 1) ** d / 2
    print(d, f"{upper:.2g}", f"{lower:.2g}")
```

It prints

```
1 6.4e+03 1.5
2 1.4e+06 4.5
3 3.7e+08 14
5 3.3e+13 1.2e+02
10 2.7e+26 3e+04
30 7.8e+81 1e+14
```

At one score, 6,400 labels. At two, well over a million. At ten scores, about \(10^{26}\), and at thirty, about \(10^{82}\): numbers no mail room, and no planet, will ever label. The exponential growth of the required training sequence with the dimension is called the **curse of dimensionality**.

A sceptic should object here. This is an upper bound from a proof that we have just seen is loose. Maybe a cleverer proof, or a cleverer rule, escapes the exponential. It doesn’t, and the second column is the reason.

**Theorem (the curse is real).** Let \(c \ge 1\) be a whole number, and let \(A\) be any learner. For every training sequence size \(m \le (c+1)^d/2\), there is a distribution \(\mathcal D\) over \([0, 1]^d \times \{0, 1\}\) whose conditional probability \(\eta\) is c-Lipschitz and whose Bayes risk is 0, such that \(\mathbb E_{S \sim \mathcal D^m}[L_{\mathcal D}(A(S))] \ge 1/4\), and with probability at least \(1/7\), \(L_{\mathcal D}(A(S)) \ge 1/8\).

*Proof.* **A grid of far-apart points.** Let \(G\) be the grid of points in \([0, 1]^d\) whose coordinates all lie in \(\{0, 1/c, 2/c, \dots, 1\}\). Each coordinate has \(c + 1\) choices, so \(|G| = (c+1)^d\). Two distinct grid points differ in at least one coordinate, by at least \(1/c\), so they are at least \(1/c\) apart.

**On the grid, every labelling is Lipschitz.** Let \(\eta: G \to [0, 1]\) be any function at all. For distinct \(g, g' \in G\), \(|\eta(g) - \eta(g')| \le 1 = c \cdot \frac1c \le c\|g - g'\|\). So every function on the grid, including every labelling \(G \to \{0, 1\}\), is c-Lipschitz.

**Run the No-Free-Lunch argument on the grid.** Lesson 5’s proof of the No-Free-Lunch theorem begins by choosing a set \(C\) of \(2m\) domain points, and builds its hard tasks on \(C\): uniform over \(C\), with labels given by a function \(f_i: C \to \{0, 1\}\). Since \(2m \le (c+1)^d\), we can choose \(C \subseteq G\). Every task in that family has deterministic labels, so its Bayes risk is 0, and its conditional probability on \(C\) is \(\eta = f_i\), which is c-Lipschitz by the previous step. The proof shows that for any learner, one of these tasks gives expected true risk at least \(1/4\), and true risk at least \(1/8\) with probability at least \(1/7\).

**Extending \(\eta\) to the whole cube.** The assumption asks for \(\eta\) on all of \([0, 1]^d\). Since the task puts no probability off \(C\), any extension describes the same task, and one extension is c-Lipschitz: \(\tilde\eta(x) = \min\{1, \min_{g \in C} (f_i(g) + c\|x - g\|)\}\). Each \(f_i(g) + c\|x - g\|\) is c-Lipschitz in \(x\), and taking minima, including with the constant 1, keeps that property. At a point \(g' \in C\), the term \(g = g'\) gives \(f_i(g')\), every other term is at least \(c \cdot \frac1c = 1\), so \(\tilde\eta(g') = f_i(g')\). \(\blacksquare\)

So the second column of the table, \((c+1)^d/2\) with \(c = 2\), is a number of labels with which *no learner whatsoever*, nearest neighbour or otherwise, can be guaranteed to beat a quarter risk on every 2-Lipschitz world. At \(d = 30\) that is about \(10^{14}\) labels. The upper bound and the lower bound disagree enormously in their bases, but they agree on the shape: exponential in \(d\).

[[diagram:curse]]

Now we can explain the invoices. In thirty dimensions there is a more direct symptom of the curse: everything is far from everything. Here is the average distance from a fresh point to its nearest neighbour among 1,000 uniform points, next to its average distance to a typical point, as the dimension grows.

```python
import math, random

rng, m = random.Random(5), 1000
for d in (1, 2, 10, 30):
    S = [[rng.random() for _ in range(d)] for _ in range(m)]
    near = far = 0.0
    for _ in range(200):
        x = [rng.random() for _ in range(d)]
        ds = [math.dist(s, x) for s in S]
        near += min(ds) / 200
        far += sum(ds) / m / 200
    print(d, round(near, 4), round(far, 3), round(near / far, 3))
```

It prints

```
1 0.0005 0.336 0.001
2 0.0156 0.526 0.03
10 0.499 1.271 0.393
30 1.4726 2.224 0.662
```

In one dimension, the nearest of a thousand messages is a thousand times closer than a typical one. In thirty, the nearest neighbour is two thirds as far away as a message picked at random. “Nearest” has almost stopped meaning anything. The invoice team’s filter was copying labels from messages that were not similar in any way that matters, which is why it was wrong on 32% of them.

And notice the subtler point. In the invoice world \(\eta\) depends on two scores and its Lipschitz constant is \(2\sqrt 2\) in thirty dimensions just as in two. Nothing about the labels got harder. What got harder is the space the neighbours live in. The nearest neighbour rule searches no hypothesis class, yet it carries prior knowledge all the same: its success depends on the assumption that the dimension and the Lipschitz constant of \(\eta\) are not too high. Choosing which scores to compare on is choosing that dimension. This is why, in practice, nearest neighbour usually runs after a step that reduces the number of scores; lesson 23 is about how to do that well.

## Nearest neighbours as code

Here is the k-NN rule as pseudocode. The training sequence holds \(m\) messages of \(d\) scores each.

```
KNN-Train(S):
  store S

KNN-Predict(S, x, k):
  for i = 1..m:
    dist_i ← ρ(x, x_i)
  N ← the indices of the k smallest dist_i
      (ties broken by smaller index)
  return the majority label among {y_i : i ∈ N}
```

For regression, replace the last line by the average of \(\{y_i : i \in N\}\), or by any \(\phi\) of the \(k\) neighbours.

**Cost.** Training is nothing more than storing the training sequence: \(O(md)\) space, and no time beyond copying. Nearest neighbour is a learning-by-memorization rule, and the whole training sequence must be kept. Prediction is where the work is: computing all \(m\) distances costs \(\Theta(md)\), selecting the \(k\) smallest with a heap of size \(k\) costs \(O(m \log k)\), and the vote costs \(O(k)\), so one prediction is \(O(md + m\log k)\) time and \(O(k)\) extra space. For the 1-NN rule it is \(\Theta(md)\). Compare that with a halfspace from lesson 9, which predicts in \(O(d)\) no matter how many examples trained it. Nearest neighbour moves all the cost from training to prediction, and every single prediction pays it.

**Faster search.** When \(d\) is small, data structures from computational geometry can answer a nearest-neighbour query in time polynomial in \(d\) times \(\log m\), instead of scanning. The catch is space: roughly \(m^{O(d)}\), which is impractical once \(d\) is more than a handful. The way out is to ask for less. An **r-approximate search** is a procedure guaranteed to return a training message within distance at most \(r\) times the distance to the true nearest neighbour. Three popular methods are **kd-trees**, **ball trees** and **locality-sensitive hashing**. A kd-tree, the one you will build below, splits the training messages in half by one score at a time, cycling through the scores, and at query time skips any half that cannot contain anything closer than the best message found so far. In low dimension the skipping saves most of the work. In high dimension, by the curse you just met, almost nothing can be skipped, because almost everything is about equally far.

**Build it yourself.** Using only the Python standard library:

```python
import heapq
import math
import random

S = [((0.9, 0.8), 1), ((0.85, 0.4), 1), ((0.7, 0.9), 1),
     ((0.2, 0.2), 0), ((0.3, 0.7), 0), ((0.1, 0.5), 0),
     ((0.65, 0.65), 0), ((0.4, 0.1), 0)]
Q = [(0.9, 0.55), (0.15, 0.3), (0.7, 0.7),
     (0.5, 0.3), (0.75, 0.2), (0.45, 0.85)]

def eta(x):
    return min(1.0, max(0.0, 2 * x[0] + 2 * x[1] - 1.5))

def draw(m, d, seed):
    rng = random.Random(seed)
    xs = [tuple(rng.random() for _ in range(d)) for _ in range(m)]
    return [(x, int(rng.random() < eta(x))) for x in xs]

def knn(S, x, k): ...
def knn_regress(S, x, k): ...
def build_kd(points): ...
def nearest_kd(tree, x): ...

def test_risk(train, test, k):
    return sum(knn(train, x, k) != y for x, y in test) / len(test)

if __name__ == "__main__":
    assert [knn(S, x, 1) for x in Q] == [1, 0, 0, 0, 1, 0]
    assert [knn(S, x, 3) for x in Q] == [1, 0, 1, 0, 0, 0]
    assert round(knn_regress(S, Q[0], 3), 3) == 0.667
    train, test = draw(2000, 2, 1), draw(1000, 2, 2)
    assert round(test_risk(train, test, 1), 3) == 0.161
    assert round(test_risk(train, test, 15), 3) == 0.122
    wide_train, wide_test = draw(2000, 30, 1), draw(1000, 30, 2)
    assert round(test_risk(wide_train, wide_test, 1), 3) == 0.325
    tree = build_kd([x for x, _ in train])
    for x, _ in test[:300]:
        scan = min((p for p, _ in train),
                   key=lambda p: math.dist(p, x))
        assert nearest_kd(tree, x) == scan
    print("all asserts pass")
```

1. Implement `knn` with a heap (`heapq.nsmallest`), so that one prediction costs \(O(md + m\log k)\), and `knn_regress` as the average label of the \(k\) nearest neighbours. The first three asserts are the worked example: the two columns of the table, and q1’s three-neighbour average \(2/3\).
2. Run the risk asserts. Notice that in two dimensions fifteen neighbours beat one (0.122 against 0.161, with a Bayes risk of 0.115), while in thirty dimensions one neighbour is wrong on almost a third of the messages.
3. Implement `build_kd` (sort by score `depth % d`, put the median at the node, recurse on the halves) and an exact `nearest_kd` that descends towards the query first and visits the far half only when the splitting plane is closer than the best distance so far. The last assert checks it against a full scan.
4. Count the distance computations `nearest_kd` makes per query on 2,000 points in \(d = 2\), and again in \(d = 30\) (build the tree on `wide_train`). Predict both numbers first.

## Fix the bug

Time to fix the invoice filter properly, with a guarantee rather than a hope. The company’s “report junk” button gives you a year of clicks on invoices: a budget of 1,000,000 labels. Your manager wants a certificate: a proven bound, from this lesson’s theorems, saying that the expected true risk on invoices is at most 0.40. Not a measurement on one test set; a promise that holds for every world with the same Bayes risk and Lipschitz constant.

You choose two things. First, how many scores to keep: the team has ranked them, payment score first, sender score second, then the other twenty-eight, and you keep the first \(d\). Second, \(k\): either the 1-NN rule, with the 1-NN guarantee, or a k-NN rule with \(k \ge 10\), with the k-NN guarantee. To evaluate either guarantee you need the Bayes risk and the Lipschitz constant of the scores you keep. The data team supplies both. Keeping only the payment score, users’ junk probability given that one score has Bayes risk \(25/96 \approx 0.260\) and Lipschitz constant 2. Keeping two or more scores, the Bayes risk is \(11/96 \approx 0.115\) and the constant is \(2\sqrt 2\), because the scores past the second carry nothing.

Here is the certificate as code, evaluated at two settings your colleagues suggested.

```python
import math

def certificate(d, k, m):
    bayes = 25 / 96 if d == 1 else 11 / 96
    c = 2.0 if d == 1 else 2 * math.sqrt(2)
    shrink = m ** (-1 / (d + 1))
    if k == 1:
        return 2 * bayes + 4 * c * math.sqrt(d) * shrink
    assert k >= 10
    slack = (6 * c * math.sqrt(d) + k) * shrink
    return (1 + math.sqrt(8 / k)) * bayes + slack

print(round(certificate(30, 1, 10**6), 1))
print(round(certificate(1, 10, 10**6), 3))
```

It prints `39.9` and `0.515`. All thirty scores and one neighbour: a “risk at most 39.9”, which certifies nothing. One score and ten neighbours: 0.515, also too high. Before you open the lab, work out which \(d\) and \(k\) can meet 0.40 within the budget, and how many labels each choice would need. Then watch what happens to that number as you add scores.

[[lab-act2]]

A word on the target. A certified 0.40 sounds poor next to the 0.14 the filter actually measures. That gap is the price of a worst-case promise, and you have seen where it comes from: the factor 2 in Step 1e and the boxes in Step 3. The lab shows the measured risk too, and you will find settings that measure better than the one you can certify, for instance more neighbours. Measuring is not certifying, as lesson 11 taught you, and only the certificate survives next month’s new kind of mail.

## What people get wrong

**“Nearest neighbour has no inductive bias, because there is no hypothesis class.”** The bias moved into the distance. The rule works when nearby messages have similar labels, in a space whose dimension is small enough for neighbours to be near. The curse-is-real theorem shows that without such assumptions no rule, nearest neighbour included, can learn from fewer than exponentially many labels.

**“More scores can only help.”** On invoices, adding twenty-eight irrelevant scores more than doubled the error, from 0.14 to 0.32, while leaving \(\eta\) and the Bayes risk untouched. Irrelevant scores add distance without adding information, and the training sequence needed grows exponentially with every score you keep.

**“The 1-NN rule converges to the Bayes risk.”** The guarantee is at most twice the Bayes risk as \(m\) grows. Getting all the way down needs a vote of many neighbours, which is what the k-NN guarantee’s multiplier \(1 + \sqrt{8/k}\) describes.

**“The bound \(m^{-1/(d+1)}\) is how fast the error really falls.”** It is an upper bound from a box argument, and in the simulations the real distances fell faster. What is not an artefact is the exponential dependence on \(d\): the lower bound proves it for every learner.

**“A bigger \(k\) is always safer.”** Larger \(k\) shrinks the multiplier on the Bayes risk but reaches further for its neighbours, and the k-NN guarantee charges \(k\, m^{-1/(d+1)}\) for that. With a fixed budget there is a best \(k\), and it can be 1.

## Recap and the next bug

1. The **k-NN rule** labels a message by the majority label of its \(k\) nearest training messages under a metric \(\rho\). It searches no hypothesis class; it stores the training sequence and pays \(\Theta(md)\) per prediction.
2. Its bias is the assumption that \(\eta\) is **c-Lipschitz**: nearby messages have similar probabilities of junk.
3. The 1-NN rule’s expected risk is at most twice the Bayes risk plus \(c\) times the expected nearest-neighbour distance, because two independent coins disagree with probability \(2p(1-p)\) plus a Lipschitz penalty.
4. Covering the cube with boxes and bounding the missed mass by \(r/(me)\) gives \(\mathbb E_S[L_{\mathcal D}(h_S)] \le 2L_{\mathcal D}(f_{\mathcal D}) + 4c\sqrt d\, m^{-1/(d+1)}\). With \(k \ge 10\) neighbours the multiplier becomes \(1 + \sqrt{8/k}\) and the excess \((6c\sqrt d + k)m^{-1/(d+1)}\).
5. The **curse of dimensionality**: the labels needed grow exponentially with the dimension, and a grid plus the No-Free-Lunch theorem shows that no learner escapes it.

Your invoice filter now keeps two scores, consults one neighbour, and comes with a certificate. But look at what it took. Somebody had to invent the payment score and the sender score, and then somebody had to rank thirty hand-made scores so you could throw twenty-eight away. Every filter in this course has stood on scores like these: word counts, sender reputations, link shares, all designed by people who guessed what might matter. The next kind of mail may not be captured by any score anyone has thought of. Could the system learn its own features, straight from the raw messages, and learn the filter on top of them at the same time? That is the next lesson: neural networks.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Suppose the labels are deterministic everywhere, \(\eta(x) \in \{0, 1\}\) for every \(x \in [0, 1]^d\), and \(\eta\) is c-Lipschitz for some \(c\). What does that force \(\eta\) to be, and what does the 1-NN guarantee then say? The hard tasks in the curse-is-real theorem also have deterministic labels and c-Lipschitz \(\eta\). Why is that not a contradiction?
2. In Step 1b of the error-through-distance proof, where exactly is independence used, and what would go wrong if the nearest neighbour \(\pi_1(x)\) were chosen by looking at the training labels? Then: in a world where \(\eta(x) = 0.1\) for every message, what is the Bayes risk, what does the 1-NN rule pay as \(m\) grows (use Step 1c with \(q = p\)), and what does the theorem’s first term allow?
3. Take \(c = 3\) and \(d = 8\). How many labels does the 1-NN guarantee ask for to make its second term at most \(\varepsilon = 0.1\)? Up to what training sequence size does the curse-is-real theorem say that some 3-Lipschitz task defeats every learner? Explain why the two numbers do not contradict each other.

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions. Together they prove the k-NN guarantee.

- **Problem 1 (boxes with too few points).** Let \(C_1, \dots, C_r\) be subsets of a domain \(\mathcal X\), and let \(S\) be a sequence of \(m\) points drawn i.i.d. from a distribution \(\mathcal D\) over \(\mathcal X\). Prove that for every \(k \ge 2\),
  \[ \mathbb E_{S \sim \mathcal D^m}\Big[ \sum_{i \,:\, |C_i \cap S| < k} \mathbb P[C_i] \Big] \le \frac{2rk}{m}, \]
  where \(|C_i \cap S|\) is the number of points of \(S\) that fall in \(C_i\).
- **Problem 2 (moving a coin’s bias).** Fix \(p, p' \in [0, 1]\) and a label \(y' \in \{0, 1\}\). Show that \(\mathbb P_{y \sim p}[y \neq y'] \le \mathbb P_{y \sim p'}[y \neq y'] + |p - p'|\).
- **Problem 3 (the k-NN guarantee).** Prove the k-NN guarantee of this lesson, for \(k \ge 10\). You may use, without proof, the following fact about majority votes: if \(Z_1, \dots, Z_k\) are independent coins with \(\mathbb P[Z_i = 1] = p_i\), \(p = \frac1k\sum_i p_i\) and \(p' = \frac1k \sum_i Z_i\), then
  \[ \mathbb E_{Z_1, \dots, Z_k}\, \mathbb P_{y \sim p}\big[ y \neq \mathbb 1[p' > 1/2] \big] \le \Big(1 + \sqrt{8/k}\Big)\, \mathbb P_{y \sim p}\big[ y \neq \mathbb 1[p > 1/2] \big]. \]
  Use boxes of side \(\varepsilon\) as in Step 3, Problem 1 for the boxes with fewer than \(k\) training messages, and Problem 2 to compare the neighbours’ coins with the coin at \(x\). It is enough to prove the bound when \(m^{1/(d+1)}/2\) is a whole number, so that the box side \(\varepsilon = 2m^{-1/(d+1)}\) divides the cube exactly.
