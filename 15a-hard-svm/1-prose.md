# Hard-SVM and the Margin

## The filter that grazes

Your colleague’s complaint from the end of lesson 14 comes with an example. On that quiet week the labelled mail could be split perfectly by a line, and on separable mail the retraining job takes the shortcut of lesson 9a: it runs the Perceptron, which stops at the first line that makes no training mistakes, and operations likes it because the mistake bound says in advance how long the job can take. The example comes from the facilities team, who write in again, confused this time rather than annoyed.

On Monday, “free lunch at the meeting, meeting room 4” arrived in everyone’s inbox, exactly as it should. On Tuesday the same invitation went out with one small edit, “free lunch and free parking at the meeting, meeting room 4”, and the filter sent it to junk. One extra word. The message is obviously still a lunch invitation, and a filter that weighs the evidence should not change its mind over a single “free”.

You pull up the hypothesis the Perceptron returned and the training messages it was given, and the picture explains everything. The separating line does separate the training messages, every one of them. But it passes a hair’s breadth from the lunch invitation. Monday’s message sat on the keep side by a sliver; Tuesday’s extra “free” nudged it a short distance, and that was enough to cross.

So the training sequence was fine and the learner was correct by the rules we gave it. It found a consistent halfspace, and lesson 9a never asked for more. The trouble is that there are infinitely many consistent halfspaces, and the Perceptron returns whichever one its updates happen to land on. Some of them leave plenty of room around every message. Some graze. Which one *should* a learner return? And while we are at it, the team has a second worry: next month’s labels will contain a few messages that no halfspace can get right, because people disagree about what is junk. What does the learner do when the mail is only almost separable?

This lesson answers the first question, and lesson 15b answers the second. Together the answers have a name you have probably heard, the **support vector machine**, and the theory in this lesson already delivers something lesson 9a could not: a number of labels that does not grow with the size of the vocabulary.

## How close is close?

Let’s look at the actual messages. As in lesson 9a, a message is a vector of word counts, here over a vocabulary of two words so you can draw it: \(x_1\) counts “free” and \(x_2\) counts “meeting”. Labels are \(+1\) for junk and \(-1\) for keep. These are the eight messages the Perceptron trained on, in the order it scanned them.

| message | gist | “free” \(x_1\) | “meeting” \(x_2\) | label \(y\) |
|---|---|---|---|---|
| A | FREE trial, FREE setup, FREE support, FREE month, FREE gift: book a meeting, any meeting | 5 | 2 | junk, \(+1\) |
| B | FREE gift, FREE delivery, FREE | 3 | 0 | junk, \(+1\) |
| C | free phone, free case, free charger, free delivery, no meeting needed | 4 | 1 | junk, \(+1\) |
| D | free prize, free cruise, free spa, free! | 4 | 0 | junk, \(+1\) |
| E | see you tomorrow | 0 | 0 | keep, \(-1\) |
| F | meeting moved to Tuesday | 0 | 1 | keep, \(-1\) |
| G | free lunch at the meeting, meeting room 4 | 1 | 2 | keep, \(-1\) |
| H | free pizza after the meeting; meeting notes; next meeting Friday | 1 | 3 | keep, \(-1\) |

The Perceptron of lesson 9a appends a constant 1 to every message, so its weights are \((b, w_1, w_2)\) with the bias term first, starts from zeros, and adds \(y_i \mathbf x_i\) (the bold \(\mathbf x_i\) is message \(i\)’s whole vector \((1, x_1, x_2)\), and multiplying by its label \(y_i\) flips it for keep messages) for the first message it gets wrong until there is none. Before you run it, a question: the four keep messages all sit near the origin, and the junk messages are far out along the “free” axis. Would you expect the line it returns to leave a lot of room on both sides?

```python
S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
score = lambda w, x: w[0] + w[1] * x[0] + w[2] * x[1]
w, updates = [0, 0, 0], ""
while True:
    bad = [i for i, (x, y) in enumerate(S)
           if y * score(w, x) <= 0]
    if not bad:
        break
    x, y = S[bad[0]]
    w = [w[0] + y, w[1] + y * x[0], w[2] + y * x[1]]
    updates += "ABCDEFGH"[bad[0]]
norm = (w[1] ** 2 + w[2] ** 2) ** 0.5
print(w, updates)
print([round(abs(score(w, x)) / norm, 3) for x, _ in S])
print([score(w, x) for x in [(1, 0), (2, 2), (2, 3)]])
```

It prints `[-3, 4, -1] AEEFG`, then `[3.638, 2.183, 2.91, 3.153, 0.728, 0.97, 0.243, 0.485]`, then `[1, 3, 2]`.

Five updates (on A, E twice, F and G) and the result is the score \(4x_1 - x_2 - 3\). The second line is the distance from each message to the line \(4x_1 - x_2 - 3 = 0\), computed as the score without its sign divided by the length of the weight pair, \(\sqrt{4^2 + 1^2} = \sqrt{17}\); the next section names the symbols and proves the formula. Look at G, the lunch invitation: 0.243. One extra “free” moves a message one full unit along the \(x_1\) axis, much further than 0.243, and the third line shows what happens. Tuesday’s invitation, \((2, 2)\), scores 3: junk. And it is not alone. Add one “free” to E (“see you tomorrow, free for coffee?”, \((1, 0)\)) or to H (“free pizza and free drinks after the meeting”, \((2, 3)\)) and both become junk too. Three of the four keep messages are one word away from the junk folder.

[[diagram:graze]]

The fix is not to stop using halfspaces. It is to stop accepting just *any* consistent halfspace, and to ask for the one that leaves the most room. To say that precisely we need a way to measure room, which means measuring the distance from a point to a hyperplane.

## The distance to a hyperplane

First, the notation we will use for the whole lesson, read once on the Perceptron’s line. That line weighs “free” by 4, “meeting” by \(-1\), adds \(-3\), and calls the message junk when the total is positive. A halfspace is named by a pair \((\mathbf w, b)\): it labels \(\mathbf x\) with \(\operatorname{sign}(\langle \mathbf w, \mathbf x \rangle + b)\), and its boundary is the hyperplane \(\{\mathbf v : \langle \mathbf w, \mathbf v \rangle + b = 0\}\).

- \(\mathbf w = (w_1, \dots, w_d)\) is the weight vector, one weight per word, and \(b\) is the bias term, a number added at the end. Here \(\mathbf w = (4, -1)\) and \(b = -3\).
- \(\langle \mathbf w, \mathbf x \rangle = w_1 x_1 + \dots + w_d x_d\) is the inner product of lesson 9a: multiply matching coordinates and add.
- \(\langle \mathbf w, \mathbf x \rangle + b\) is the **score**; \(\operatorname{sign}\) turns a positive score into \(+1\) (junk) and a negative one into \(-1\) (keep).
- In \(\{\mathbf v : \dots\}\) the braces collect points and the colon reads “such that”: the set of points \(\mathbf v\) whose score is exactly 0. That set is the line you have been looking at.
- \(\|\mathbf w\| = \sqrt{w_1^2 + \dots + w_d^2}\) is the length of \(\mathbf w\).

In plain words: **a halfspace scores a message by weighing its counts and adding a constant, and labels it by the sign of the score; the hyperplane is where the score is zero.** For G, \((1, 2)\): \(\langle \mathbf w, \mathbf x \rangle = 4 - 2 = 2\), the score is \(2 - 3 = -1\), the sign says keep, and \(\|\mathbf w\| = \sqrt{17} \approx 4.123\).

```python
import math

w, b, x = (4, -1), -3, (1, 2)
inner = w[0] * x[0] + w[1] * x[1]
score = inner + b
sign = 1 if score > 0 else -1
norm = math.sqrt(w[0] ** 2 + w[1] ** 2)
w10, b10 = (40, -10), -30
score10 = w10[0] * x[0] + w10[1] * x[1] + b10
print(inner, score, sign, round(norm, 3), score10)
```

It prints `2 -1 -1 4.123 -10`. The last number is the score of G under \((40, -10)\) and \(-30\), the same pair times 10: ten times bigger, same sign. Multiplying \(\mathbf w\) and \(b\) by the same positive number changes neither the labels nor the hyperplane, so we are free to scale; the convenient scale is \(\|\mathbf w\| = 1\), a **unit** vector.

The **distance** between a point \(\mathbf x\) and the hyperplane is the length of the shortest gap between \(\mathbf x\) and a point on it:

\[ \operatorname{dist}(\mathbf x, (\mathbf w, b)) = \min \{ \|\mathbf x - \mathbf v\| \; : \; \langle \mathbf w, \mathbf v \rangle + b = 0 \}. \]

From the inside out: \(\mathbf x - \mathbf v\) is the arrow from a point \(\mathbf v\) to the message, and \(\|\mathbf x - \mathbf v\|\) is its length. The colon restricts \(\mathbf v\) to the hyperplane. The braces collect one length per such point, and \(\min\) picks the smallest. In plain words: **the distance is how far the message is from the nearest point of the line.** For G and the Perceptron’s line, the points of the line are \((v_1, 4v_1 - 3)\); try them all on a fine grid and keep the shortest gap:

```python
import math

w, b, x = (4, -1), -3, (1, 2)
score = lambda v: w[0] * v[0] + w[1] * v[1] + b
line = [(t / 1000, 4 * t / 1000 - 3) for t in range(-5000, 5001)]
on_line = all(abs(score(v)) < 1e-9 for v in line)
dist = min(math.dist(x, v) for v in line)
norm = math.sqrt(w[0] ** 2 + w[1] ** 2)
print(on_line, round(dist, 4), round(abs(score(x)) / norm, 4))
```

It prints `True 0.2425 0.2425`: every grid point is on the line, the nearest is 0.2425 away, and \(|\text{score}|/\|\mathbf w\| = 1/\sqrt{17}\) gives the same number. The claim says this is no accident.

**Claim (distance to a hyperplane).** If \(\|\mathbf w\| = 1\), the distance between \(\mathbf x\) and the hyperplane of \((\mathbf w, b)\) is \(|\langle \mathbf w, \mathbf x \rangle + b|\).

*Proof.* The idea is to walk from \(\mathbf x\) straight along \(\mathbf w\), the direction perpendicular to the hyperplane, until we hit it, and then to show that no other point of the hyperplane is closer.

First, a point on the hyperplane at exactly that distance. Let \(s = \langle \mathbf w, \mathbf x \rangle + b\) and take \(\mathbf v = \mathbf x - s \mathbf w\). Then \(\mathbf v\) is on the hyperplane, because

\[ \langle \mathbf w, \mathbf v \rangle + b = \langle \mathbf w, \mathbf x \rangle - s \|\mathbf w\|^2 + b = \langle \mathbf w, \mathbf x \rangle + b - s = 0, \]

using linearity of the inner product (\(\langle \mathbf w, \mathbf x - s\mathbf w \rangle = \langle \mathbf w, \mathbf x \rangle - s\langle \mathbf w, \mathbf w \rangle\)) and \(\langle \mathbf w, \mathbf w \rangle = \|\mathbf w\|^2 = 1\). In words: stepping back from \(\mathbf x\) by its own score along the unit \(\mathbf w\) lowers the score by exactly that much, to 0. Its distance from \(\mathbf x\) is \(\|\mathbf x - \mathbf v\| = \|s \mathbf w\| = |s| \, \|\mathbf w\| = |s|\). So the distance is at most \(|s|\).

Second, no point of the hyperplane is closer. Take any \(\mathbf u\) with \(\langle \mathbf w, \mathbf u \rangle + b = 0\). Write \(\mathbf x - \mathbf u = (\mathbf x - \mathbf v) + (\mathbf v - \mathbf u)\) and expand the square:

\[ \|\mathbf x - \mathbf u\|^2 = \|\mathbf x - \mathbf v\|^2 + \|\mathbf v - \mathbf u\|^2 + 2 \langle \mathbf x - \mathbf v, \mathbf v - \mathbf u \rangle \ge \|\mathbf x - \mathbf v\|^2 + 2 \langle \mathbf x - \mathbf v, \mathbf v - \mathbf u \rangle. \]

This is the square of a sum of two arrows, \(\|\mathbf a + \mathbf c\|^2 = \|\mathbf a\|^2 + \|\mathbf c\|^2 + 2\langle \mathbf a, \mathbf c \rangle\), the vector form of \((a + c)^2\). The inequality drops the nonnegative term \(\|\mathbf v - \mathbf u\|^2\). Now the cross term vanishes: \(\mathbf x - \mathbf v = s \mathbf w\), so \(\langle \mathbf x - \mathbf v, \mathbf v - \mathbf u \rangle = s \langle \mathbf w, \mathbf v - \mathbf u \rangle = s (\langle \mathbf w, \mathbf v \rangle - \langle \mathbf w, \mathbf u \rangle) = s(-b - (-b)) = 0\), because both \(\mathbf v\) and \(\mathbf u\) lie on the hyperplane. So \(\|\mathbf x - \mathbf u\| \ge \|\mathbf x - \mathbf v\| = |s|\) for every \(\mathbf u\) on the hyperplane, and the minimum is exactly \(|s|\). \(\blacksquare\)

[[diagram:distance]]

When \(\|\mathbf w\| \ne 1\), divide by it first: the distance is \(|\langle \mathbf w, \mathbf x \rangle + b| / \|\mathbf w\|\), the formula the first snippet used and the grid search above confirmed on G.

## The margin, and the widest separator

Now the measure of room. For a training sequence \(S = ((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m))\) with \(\mathbf x_i \in \mathbb R^d\) and \(y_i \in \{\pm 1\}\), recall from lesson 9a that \(S\) is **separable** when some halfspace gets every example right, that is, some \((\mathbf w, b)\) has

\[ y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \quad \text{for all } i \in [m]. \]

Here \([m]\) is the list of positions \(\{1, \dots, m\}\), and \(y_i\) is example \(i\)’s label, \(+1\) or \(-1\). Multiplying the score by the label gives the **signed score**: positive exactly when the score has the label’s sign, that is, when the halfspace gets example \(i\) right. In plain words: **some halfspace gives every training message a positive signed score.** For G under the Perceptron’s line, the score is \(-1\) and the label \(-1\), so the signed score is \(+1\): right. The first snippet’s loop tests exactly this, `y * score(w, x) <= 0`, and stops when no message fails it.

Every such halfspace has **empirical risk** 0 (the fraction of training messages it gets wrong, lesson 1), so every one of them is a valid output of **ERM**, the learner of lesson 2 that returns a hypothesis with the fewest training mistakes. The **margin** of a hyperplane with respect to \(S\) is the smallest distance between a training message and the hyperplane. By the claim, for a separating \((\mathbf w, b)\) with \(\|\mathbf w\| = 1\), it is \(\min_i |\langle \mathbf w, \mathbf x_i \rangle + b|\), and since every signed score is positive, that is the same as \(\min_i y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). The subscript in \(\min_i\) says what varies: compute the expression for each example \(i\) and keep the smallest. In plain words: **the margin is the signed score of the worst-placed message, once \(\mathbf w\) is a unit vector.** The Perceptron’s hyperplane has margin 0.243 on our eight messages, set by G: the smallest entry of the first snippet’s second line, which divides by \(\|\mathbf w\|\) to make the vector unit.

Why care about the margin? Because a hyperplane with a large margin still separates the training messages after you perturb each one a little: every message can move by anything less than the margin, in any direction, without crossing. A grazing hyperplane survives almost no perturbation at all. Later in this lesson we will also see that a large margin buys a small true risk, regardless of the dimension. That is the real prize.

**Definition (hard-SVM).** **Hard-SVM** is the learning rule that returns, among all halfspaces that separate \(S\), one with the largest margin:

\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} |\langle \mathbf w, \mathbf x_i \rangle + b| \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i. \]

When \(S\) is separable, this is the same as dropping the constraint and maximizing the smallest signed score directly,

\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} y_i (\langle \mathbf w, \mathbf x_i \rangle + b), \]

and proving that the two agree is this lesson’s problem at the end. We will work with the second form.

Read it from the inside out. The inner part, \(\min_{i \in [m]} y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\), is the margin of one candidate, as above. The subscript under \(\operatorname{argmax}\) says which candidates compete: every pair \((\mathbf w, b)\) such that \(\mathbf w\) is a unit vector. And \(\operatorname{argmax}\) returns not the largest value but the candidate that achieves it. In plain words: **among all unit halfspaces, return the one whose worst-placed message is as far as possible.** The Perceptron’s unit pair, \((4, -1, -3)/\sqrt{17}\), scores 0.243 in this contest. The snippet after the next lemma runs the whole contest by brute force.

“Maximize a minimum over a sphere” is an awkward thing to hand a computer. The standard trick is to fix the scale differently: instead of fixing \(\|\mathbf w\| = 1\) and maximizing the smallest score, fix the smallest score at 1 and minimize \(\|\mathbf w\|\). A **quadratic program** is an optimization problem whose objective is a convex quadratic function and whose constraints are linear inequalities; solvers for them are standard and run in polynomial time.

```
Hard-SVM(S):
  solve the quadratic program
    (w0, b0) = argmin ||w||^2
               over (w, b)
               subject to y_i (⟨w, x_i⟩ + b) ≥ 1 for all i
  return ŵ = w0 / ||w0||,  b̂ = b0 / ||w0||
```

The program reads: among all pairs \((\mathbf w, b)\) such that every signed score is at least 1, find one with the smallest squared length \(\|\mathbf w\|^2\) (squaring removes the square root and changes nothing about which pair is shortest); then rescale it to unit length, the hats on \(\hat{\mathbf w}, \hat b\) marking the returned pair. For the pair \(\mathbf w_0 = (\tfrac23, -\tfrac13)\), \(b_0 = -1\) that we will meet shortly, \(\|\mathbf w_0\|^2 = \tfrac49 + \tfrac19 = \tfrac59\).

Why should the shortest \(\mathbf w\) give the widest margin? A message’s score is its distance to the hyperplane times \(\|\mathbf w\|\), so a short \(\mathbf w\) can only give every message a score of 1 if every message is far away.

**Lemma (hard-SVM finds the widest separator).** Suppose \(S\) is separable and contains both labels, and let \((\mathbf w_0, b_0)\) solve the quadratic program. Then for every \((\mathbf w, b)\) with \(\|\mathbf w\| = 1\) and \(\gamma = \min_i y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0\),

\[ \min_{i} y_i (\langle \hat{\mathbf w}, \mathbf x_i \rangle + \hat b) \ge \gamma. \]

Here \(\gamma\) (gamma) is the margin of a competitor, and the left side is the margin of hard-SVM’s output. In plain words: **no separating unit halfspace has a wider margin than the one hard-SVM returns.** With the Perceptron as the competitor, \(\gamma = 0.243\), and the left side will turn out to be 1.342. So \((\hat{\mathbf w}, \hat b)\) has the largest margin of any separating halfspace: it solves the max-min problem.

*Proof.* First, \(\mathbf w_0 \ne \mathbf 0\), so the output is defined: if \(\mathbf w_0 = \mathbf 0\), the constraints would read \(y_i b_0 \ge 1\) for all \(i\), forcing every \(y_i\) to have the sign of \(b_0\), but \(S\) contains both labels.

Now take any unit \((\mathbf w, b)\) with smallest signed score \(\gamma > 0\). Dividing every inequality \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge \gamma\) by the positive number \(\gamma\) gives

\[ y_i \left( \left\langle \frac{\mathbf w}{\gamma}, \mathbf x_i \right\rangle + \frac{b}{\gamma} \right) \ge 1 \quad \text{for all } i, \]

so the pair \((\mathbf w / \gamma, b / \gamma)\) satisfies the program’s constraints. (For the Perceptron’s unit pair, dividing by 0.243 stretches its length from 1 to about 4.12, and G’s signed score from 0.243 to exactly 1.) The program’s solution has the smallest norm among all pairs that do, so \(\|\mathbf w_0\| \le \|\mathbf w / \gamma\| = 1/\gamma\).

Finally, the output is \((\mathbf w_0, b_0)\) divided by \(\|\mathbf w_0\|\), so for every \(i\),

\[ y_i (\langle \hat{\mathbf w}, \mathbf x_i \rangle + \hat b) = \frac{y_i (\langle \mathbf w_0, \mathbf x_i \rangle + b_0)}{\|\mathbf w_0\|} \ge \frac{1}{\|\mathbf w_0\|} \ge \gamma. \]

The first inequality is the program’s constraint, the second is the bound we just proved (on our messages, \(1/\|\mathbf w_0\| = 1.342 \ge 0.243\)). Since \(\|\hat{\mathbf w}\| = 1\), this says the output’s margin is at least the margin of every separating unit halfspace. \(\blacksquare\)

Notice what the proof also tells you: the largest margin is exactly \(1 / \|\mathbf w_0\|\). On our eight messages the program’s solution turns out to be \(\mathbf w_0 = (\tfrac23, -\tfrac13)\), \(b_0 = -1\), the score \((2x_1 - x_2)/3 - 1\). Let’s check that it satisfies every constraint, read off its margin, and compare it with a brute-force search over every direction of \(\mathbf w\). For a unit direction \(\mathbf u\), the best bias term puts the hyperplane halfway between the lowest-scoring junk message and the highest-scoring keep message, so the best margin in that direction is half the gap between them.

```python
import math
from fractions import Fraction as Fr

S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
w0, b0 = (Fr(2, 3), Fr(-1, 3)), Fr(-1)
print([str(y * (w0[0] * x[0] + w0[1] * x[1] + b0)) for x, y in S])
print(round(1 / math.sqrt(w0[0] ** 2 + w0[1] ** 2), 4))

def best_margin(theta):
    u = (math.cos(theta), math.sin(theta))
    p = [y * (u[0] * x[0] + u[1] * x[1]) for x, y in S]
    return (min(p[:4]) + min(p[4:])) / 2

grid = [2 * math.pi * k / 100_000 for k in range(100_000)]
print(round(max(best_margin(t) for t in grid), 4))
```

It prints `['5/3', '1', '4/3', '5/3', '1', '4/3', '1', '4/3']`, then `1.3416`, then `1.3416`. Every signed score is at least 1, so the constraints hold; the margin is \(1/\|\mathbf w_0\| = 3/\sqrt 5 \approx 1.342\); and a search over 100,000 directions finds nothing wider. The Perceptron left 0.243. Hard-SVM leaves more than five times as much.

Three messages sit exactly at score 1, which means exactly at distance 1.342 from the line: B, E and G. Those are the messages that hold the hyperplane in place, and they will get a name in lesson 15b. And Tuesday’s lunch invitation? Its score is \((4 - 2)/3 - 1 = -\tfrac13\): keep. So are the nudged versions of E and H, at \(-\tfrac13\) and \(-\tfrac23\).

[[diagram:margin-band]]

Before going further, try it with your hands. In the lab below you can rerun the Perceptron in different orders of the eight messages and watch the margin it happens to leave, then switch on hard-SVM and press “add one word” to see which messages survive the nudge.

[[lab]]

## Without the bias term

It is often more convenient to work with **homogeneous** halfspaces, those with no bias term, \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\), whose hyperplane passes through the origin. Hard-SVM for homogeneous halfspaces is the program

\[ \min_{\mathbf w} \|\mathbf w\|^2 \quad \text{subject to} \quad y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1 \text{ for all } i. \]

It is the program you just read with \(b\) gone: **the shortest weight vector that gives every message a signed score of at least 1.**

As in lesson 9a, a halfspace with a bias term becomes a homogeneous one after appending a constant 1 to every message: \((1, \mathbf x)\) in \(\mathbb R^{d+1}\), with the bias term as the first weight. But there is one difference worth noticing. The program with a bias term minimizes \(\|\mathbf w\|^2\) and leaves \(b\) free, while the homogeneous program on the lifted vectors minimizes \(b^2 + \|\mathbf w\|^2\): it **regularizes** the bias term too, preferring small \(b\). In general the two programs can return different halfspaces. In practice, regularizing \(b\) usually makes no significant difference to the number of examples needed, and the homogeneous form is much easier to analyze, so from here on the theory is homogeneous.

On our eight messages the two even agree: the homogeneous program on the lifted vectors returns the same \((b, w_1, w_2) = (-1, \tfrac23, -\tfrac13)\), as the solver of lesson 15b will confirm. The margin is now measured in \(\mathbb R^3\), where the length includes the bias term, so it is \(1/\sqrt{1 + \tfrac49 + \tfrac19} = 3/\sqrt{14} \approx 0.802\) rather than 1.342. Same hyperplane in the plane, different ruler.

## Why a wide margin needs fewer labels

Here is the deeper reason to want a wide margin, and it is the reason this lesson exists.

Lesson 9a showed that homogeneous halfspaces in \(\mathbb R^d\) (the space of vectors with \(d\) coordinates, one per word) have VC dimension \(d\): they can label some \(d\) messages in every possible way, and no \(d + 1\). By the fundamental theorem of lesson 6a, the number of examples needed to learn them grows with \(d\), and, worse, for an accuracy target \(\varepsilon\) (epsilon, the true risk you are willing to tolerate), if the number of examples is much smaller than \(d/\varepsilon\), then *no* learner can guarantee true risk at most \(\varepsilon\) for every distribution. For a filter with two words that is nothing. For a filter that tracks fifty thousand words it is a disaster: the label budget would have to grow with the vocabulary.

The way out is to assume something more about the distribution. There is no contradiction with the lower bound here: the lower bound says that *some* distributions need that many examples, and we are about to rule those distributions out with an extra assumption. The assumption is that the data are separable with a margin. But there is a catch to deal with first.

Suppose \(S\) is separable with margin \(\gamma\). Multiply every message by 10. The same hyperplane (if it is homogeneous) still separates, and every distance is now ten times bigger, so the new training sequence is separable with margin \(10\gamma\). By scaling the data, you can make the margin as large as you like, and a measure of difficulty that changes when you change units is useless. The fix is to measure the margin relative to the size of the messages.

```python
import math

S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X = [(1,) + x for x, _ in S]
w0 = (-1, 2 / 3, -1 / 3)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
for alpha in (1, 10, 0.1):
    Xa = [[alpha * c for c in x] for x in X]
    gamma = min(y * dot(w0, x) for x, (_, y) in zip(Xa, S))
    gamma /= math.sqrt(dot(w0, w0))
    rho = max(math.sqrt(dot(x, x)) for x in Xa)
    print(alpha, round(gamma, 4), round(rho, 4),
          round((rho / gamma) ** 2, 2))
```

It prints `1 0.8018 5.4772 46.67`, `10 8.0178 54.7723 46.67` and `0.1 0.0802 0.5477 46.67`. Scaling the lifted messages by \(\alpha\) multiplies the margin by \(\alpha\) and the length of the longest message by \(\alpha\), and their ratio does not move.

[[diagram:scale]]

**Definition (separability with a \((\gamma, \rho)\)-margin).** Let \(\mathcal D\) be a distribution over \(\mathbb R^d \times \{\pm 1\}\). We say \(\mathcal D\) is **separable with a \((\gamma, \rho)\)-margin** if there is \((\mathbf w^\star, b^\star)\) with \(\|\mathbf w^\star\| = 1\) such that, with probability 1 over \((\mathbf x, y) \sim \mathcal D\), both \(y (\langle \mathbf w^\star, \mathbf x \rangle + b^\star) \ge \gamma\) and \(\|\mathbf x\| \le \rho\). It is separable with a \((\gamma, \rho)\)-margin **using a homogeneous halfspace** if this holds with \(b^\star = 0\).

Reading it piece by piece: \(\mathcal D\) is the unknown source of labelled mail, and \((\mathbf x, y) \sim \mathcal D\) reads “a message and its label drawn from \(\mathcal D\)”. The star on \(\mathbf w^\star\) marks one fixed, ideal separator that we never get to see. “With probability 1” means the two inequalities hold for every message that can actually arrive. The first says that message’s signed score under the ideal separator is at least \(\gamma\) (gamma), so it sits at least \(\gamma\) from the hyperplane; the second says its length is at most \(\rho\) (rho). In plain words: **some unit separator keeps every possible message at least \(\gamma\) from the boundary, and no message is longer than \(\rho\).** For the uniform distribution on our eight lifted messages, the scaling snippet’s first line gives \(\gamma = 0.8018\) and \(\rho = 5.4772\) (that is \(\sqrt{30}\), the length of A’s lifted vector \((1, 5, 2)\)).

The number \(\rho\) is the **radius**: every message lies in the ball of radius \(\rho\). (Lesson 9a called the radius of a training sequence \(R\); we switch to \(\rho\) here, because it will soon play a second role.) Scaling the data multiplies both \(\gamma\) and \(\rho\), so the ratio \(\rho / \gamma\) is the honest measure of difficulty.

**Theorem (hard-SVM’s sample complexity depends on the margin, not the dimension).** Let \(\mathcal D\) be a distribution over \(\mathbb R^d \times \{\pm 1\}\) that is separable with a \((\gamma, \rho)\)-margin using a homogeneous halfspace. Then, with probability at least \(1 - \delta\) over a training sequence \(S \sim \mathcal D^m\), the homogeneous hard-SVM hypothesis has 0-1 true risk at most

\[ \sqrt{\frac{4 (\rho / \gamma)^2}{m}} + \sqrt{\frac{2 \log(2/\delta)}{m}}. \]

First the notation. \(S \sim \mathcal D^m\) reads “\(S\) is \(m\) examples drawn independently from \(\mathcal D\)”, and \(\delta\) (delta) is the small chance we allow the training sequence to be unlucky, as in every probably approximately correct guarantee since lesson 2. The **0-1 true risk** is the chance that the returned halfspace mislabels a fresh message from \(\mathcal D\). Inside the first square root sits the difficulty ratio \(\rho/\gamma\), squared, divided by the number of labels; inside the second, \(\log(2/\delta)\), which grows only slowly as you demand more confidence. In plain words: **with high confidence, hard-SVM’s error rate on new mail is at most a difficulty term plus a confidence term, and both shrink like one over the square root of the number of labels.** With \(\rho/\gamma = 10\), \(m = 1000\) and \(\delta = 0.05\): \(\sqrt{400/1000} \approx 0.632\), plus \(\sqrt{2 \log 40 / 1000} \approx 0.086\), about 0.718, the first line `hard_svm_bound` prints below.

This is the heart of the lesson: the dimension \(d\) appears nowhere, so a filter over fifty thousand words gets the same guarantee as one over two words with the same ratio \(\rho/\gamma\). Setting the bound equal to \(\varepsilon\), the number of labels needed grows like \((\rho/\gamma)^2/\varepsilon^2\), with no \(d\). (Here \(\log\) is the natural logarithm, as everywhere in this course.)

The proof needs a tool we have not built yet, the Rademacher complexity of a class, and lesson 26 proves this theorem in full. Here is the bound as the label count grows, still with \(\rho/\gamma = 10\) (the closest message a tenth of the radius from the best separator) and \(\delta = 0.05\):

```python
import math

def hard_svm_bound(rho_over_gamma, m, delta):
    estimation = math.sqrt(4 * rho_over_gamma ** 2 / m)
    confidence = math.sqrt(2 * math.log(2 / delta) / m)
    return estimation + confidence

for m in (1_000, 10_000, 100_000):
    print(m, round(hard_svm_bound(10, m, 0.05), 4))
```

It prints `1000 0.7183`, `10000 0.2272` and `100000 0.0718`. The bound shrinks like \(1/\sqrt m\), and none of those numbers would change if the vocabulary grew a thousandfold.

The ratio \(\rho/\gamma\) has met you before. Lesson 9a proved that on separable data the Perceptron makes at most \((R/\gamma^\star)^2\) updates, where \(\gamma^\star\) is the best margin of a homogeneous separator. In this lesson’s letters that is \((\rho/\gamma)^2\), the very quantity that controls hard-SVM’s sample complexity. For our lifted eight messages it is 46.67 (from the scaling snippet), and the Perceptron used 5. The same ratio governs how fast the Perceptron finishes and how many examples a large-margin separator needs. That is not a coincidence: both are statements that wide, well-scaled margins make halfspaces easy.

## A margin is a safety distance

Back to Tuesday’s lunch invitation. We have been saying, loosely, that a wide margin lets every message move a little without crossing. Before we lean on that to fix the bug, let’s make it a claim with a proof, because it is exactly the promise the facilities team is about to ask for.

Picture an **edit** as an arrow \(\mathbf e\) added to a message: adding one “free” to G is the arrow \(\mathbf e = (1, 0)\), which carries \((1, 2)\) to \((2, 2)\). Its length \(\|\mathbf e\|\) measures how big the edit is, and any edit that adds or removes a single word has length exactly 1. The question is how far the edit can push the signed score down. Here is the answer.

**Claim (a margin is a safety distance).** Let \((\mathbf w, b)\) with \(\|\mathbf w\| = 1\) separate \(S\) with margin \(\gamma > 0\). Then for every example \(i\) and every edit \(\mathbf e\),

\[ y_i (\langle \mathbf w, \mathbf x_i + \mathbf e \rangle + b) \ge \gamma - \|\mathbf e\|. \]

So every edit shorter than \(\gamma\) leaves every training message strictly on its own side.

Read it from the inside out. \(\mathbf x_i + \mathbf e\) is the edited message. \(\langle \mathbf w, \mathbf x_i + \mathbf e \rangle + b\) is its score under the unit halfspace, and multiplying by the label \(y_i\) gives its signed score, positive when the halfspace still gets it right. On the right, \(\gamma\) is the margin, the smallest signed score before any edit, and \(\|\mathbf e\|\) is the length of the edit. In plain words: **an edit can lower a message’s signed score by at most its own length, so a message that started at least \(\gamma\) from the boundary is still at least \(\gamma - \|\mathbf e\|\) from it afterwards.** For G under hard-SVM’s unit line, \(\gamma = 1.342\) and the one-word edit has length 1, so the edited G keeps a signed score of at least \(0.342\).

*Proof.* The inner product is linear, so the edited signed score splits into the old one plus the edit’s share:

\[ y_i (\langle \mathbf w, \mathbf x_i + \mathbf e \rangle + b) = y_i (\langle \mathbf w, \mathbf x_i \rangle + b) + y_i \langle \mathbf w, \mathbf e \rangle. \]

The first term is at least \(\gamma\), because \(\gamma\) is the smallest signed score over the training sequence. For the second, \(|y_i| = 1\), and the Cauchy–Schwarz inequality of lesson 9a, \(|\langle \mathbf a, \mathbf c \rangle| \le \|\mathbf a\| \, \|\mathbf c\|\), gives \(|\langle \mathbf w, \mathbf e \rangle| \le \|\mathbf w\| \, \|\mathbf e\| = \|\mathbf e\|\), since \(\mathbf w\) is a unit vector. So the second term is at least \(-\|\mathbf e\|\), and the sum is at least \(\gamma - \|\mathbf e\|\). When \(\|\mathbf e\| < \gamma\), that is positive. \(\blacksquare\)

Let’s push G through it term by term: its signed score before the edit, the edit’s share \(y \langle \mathbf w, \mathbf e \rangle\), the signed score after, and the floor \(\gamma - \|\mathbf e\|\) the claim promises.

```python
import math

w, b = (2 / math.sqrt(5), -1 / math.sqrt(5)), -3 / math.sqrt(5)
x, y, e = (1, 2), -1, (1, 0)
score = lambda v: w[0] * v[0] + w[1] * v[1] + b
gamma = y * score(x)
edited = (x[0] + e[0], x[1] + e[1])
push = y * (w[0] * e[0] + w[1] * e[1])
length = math.hypot(*e)
after = y * score(edited)
print(round(gamma, 3), round(push, 3), round(after, 3))
print(round(gamma - length, 3), abs(push) <= length)
```

It prints `1.342 -0.894 0.447`, then `0.342 True`. The extra “free” pushes G’s signed score down by 0.894, less than the edit’s length of 1, and G lands at 0.447, comfortably above the floor of 0.342 and on the keep side. The edit’s share is less than its full length because the arrow \((1, 0)\) does not point straight at the line; an edit aimed exactly along \(-y_i \mathbf w\) would use its whole length, and an edit of length \(\gamma\) aimed that way lands exactly on the hyperplane. So no halfspace can promise more than its margin.

Now all eight messages and all four one-word edits (one more or one fewer “free”, one more or one fewer “meeting”), under hard-SVM’s line and under the Perceptron’s. The function returns the margin and the smallest signed score after any single edit:

```python
import math

S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
edits = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def worst_after_edit(w, b):
    n = math.hypot(*w)
    s = lambda x: (w[0] * x[0] + w[1] * x[1] + b) / n
    gamma = min(y * s(x) for x, y in S)
    after = min(y * s((x[0] + e[0], x[1] + e[1]))
                for x, y in S for e in edits)
    return round(gamma, 3), round(after, 3)

print(worst_after_edit((2, -1), -3))
print(worst_after_edit((4, -1), -3))
```

It prints `(1.342, 0.447)` and `(0.243, -0.728)`. (Some of those edits remove a word the message does not have, such as a “meeting” from B; they are arrows in the plane all the same, and the claim covers them.) Under hard-SVM no single-word edit brings any message below 0.447, as the claim guaranteed once the margin passed 1. Under the Perceptron’s line, the margin is 0.243, the claim promises nothing for an edit of length 1, and indeed one edit drives a signed score to \(-0.728\): a message on the wrong side.

[[diagram:safety]]

## Fix the bug

Time to ship. The facilities team and operations sit down together and agree on a written promise for the separator that goes into the nightly job: **every labelled message sits more than one word from the boundary.** By the claim, that settles Tuesday once and for all. A one-word edit has length 1, so if the margin is more than 1, no message in the training sequence can be pushed across by adding or removing a single word.

Why write the promise as a distance, instead of just trying the edits? In the production filter a message has fifty thousand words it could gain or lose, the closest direction to the boundary is usually some mix of many words, and next week’s complaint will be about an edit nobody thought to test. The margin is one number that answers for every edit of length below it, in every direction at once. And a wide margin is exactly what the sample-complexity theorem rewards, so asking for one costs nothing on new mail.

In the lab below, each round brings one week of labelled mail on the two words, so you can see it, starting with our eight messages. You steer the separator with two controls: its direction, the angle of \(\mathbf w\), and its offset, the bias term \(b\) of the unit pair. The lab draws a circle of radius 1, one word, around every message, computes the margin exactly, and names the message that sets it. A round passes only when the line separates and its margin is more than 1. Trying edits does not count: a line can dodge every edit you draw and still pass closer than one word to a message, and the promise is about distance. Three weeks in a row, and the filter ships.

[[lab-act2]]

What you just did by hand is what hard-SVM does by solving its quadratic program. By the lemma, its line has the largest margin of any separator, so whenever *some* line can keep the promise, hard-SVM’s line keeps it, with nothing to steer. On our eight messages it leaves 1.342, and the one-word promise holds with room to spare.

## What people get wrong

**“Every consistent halfspace is as good as any other.”** They all have empirical risk 0, but they are not equally robust, and they do not carry the same guarantee. The Perceptron’s line on our eight messages left 0.243; hard-SVM’s left 1.342. The three keep messages that a one-word nudge sends to junk under the Perceptron’s line all stay keep under hard-SVM.

**“A bigger margin is always better, so rescale the data.”** Multiplying every message by 10 multiplies the margin by 10 and changes nothing real. The meaningful quantity is \(\rho/\gamma\), which scaling leaves alone. That is why the radius is part of every margin bound.

**“If none of the edits I tried flips a message, the line is safe.”** A line can dodge every edit on your list and still pass closer than one word to a message, because the direction straight toward the boundary is rarely the direction of a single word. On our eight messages, point the weight vector 50 degrees below the “free” axis and centre the line between the two groups: none of the four one-word edits of any message crosses it, yet A and E sit only 0.841 from it. The safety distance is the margin, and only the margin covers the edits you did not think of.

## Recap and the next bug

1. The **margin** of a separating hyperplane is the distance from it to the closest training message; for a unit \(\mathbf w\) that distance is \(|\langle \mathbf w, \mathbf x \rangle + b|\). An edit can lower a signed score by at most its own length, so every training message survives every edit shorter than the margin.
2. **Hard-SVM** returns the separating halfspace with the largest margin. It is the quadratic program “minimize \(\|\mathbf w\|^2\) subject to every signed score being at least 1”, and the largest margin is \(1/\|\mathbf w_0\|\).
3. Margins only mean something relative to the radius \(\rho\) of the messages. If the distribution is separable with a \((\gamma, \rho)\)-margin, hard-SVM needs a number of examples that grows with \((\rho/\gamma)^2\), with no dependence on the dimension.

The filter ships with its promise, and the facilities team gets its lunch invitations. A month later the nightly job runs hard-SVM on the new labels and, in the morning, there is no new filter, only a line in the log: the quadratic program has no feasible point. Among the new labels is a free consultation with exactly the counts of the lunch invitation, which your users called junk, and a farewell party full of free cake, which they called keep. No line separates the month’s mail, so hard-SVM has nothing to return. Real labels are noisy and people disagree, so this will happen again every month. What does a learner that loves wide margins do when the mail is only almost separable, and does its guarantee survive? That is lesson 15b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A colleague multiplies every lifted message vector (including the constant 1) by 10, “to make the margin bigger”. For homogeneous hard-SVM on our eight messages, what happens to the margin, to the radius \(\rho\), to the hypothesis returned (as a labelling of messages), and to the bound of the hard-SVM sample-complexity theorem? Then explain why the definition of separability with a \((\gamma, \rho)\)-margin needs \(\rho\) at all.
2. Hard-SVM’s line on the eight messages has margin 1.342. Does the safety-distance claim promise that B, “FREE gift, FREE delivery, FREE”, stays junk if one “FREE” is swapped for a “meeting”? Work out the edited message’s score. What does this tell you about how much a margin can promise?
3. The lemma that hard-SVM finds the widest separator assumes that \(S\) contains both labels. Give a separable training sequence with junk messages only, say what the quadratic program returns, and explain why hard-SVM’s output is then undefined. Where else does the proof need the competitor’s margin \(\gamma\) to be positive?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (two forms of hard-SVM).** Show that, on a separable training sequence, the rule
\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} |\langle \mathbf w, \mathbf x_i \rangle + b| \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i \]
is the same as \(\operatorname{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \min_{i \in [m]} y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). Start from the set \(G = \{(\mathbf w, b) : y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i\}\).
