# Support Vector Machines

## The filter that grazes

The halfspace filter of lesson 9 has been in production for weeks. Every night the Perceptron retrains it on the new labels, it separates every labelled message perfectly, and operations is happy because the mistake bound tells them in advance how long the nightly job can take. Then the facilities team writes in again, and this time they are confused rather than annoyed.

On Monday, “free lunch at the meeting, meeting room 4” arrived in everyone’s inbox, exactly as it should. On Tuesday the same invitation went out with one small edit, “free lunch and free parking at the meeting, meeting room 4”, and the filter sent it to junk. One extra word. The message is obviously still a lunch invitation, and a filter that weighs the evidence should not change its mind over a single “free”.

You pull up the hypothesis the Perceptron returned and the training messages it was given, and the picture explains everything. The separating line does separate the training messages, every one of them. But it passes a hair’s breadth from the lunch invitation. Monday’s message sat on the keep side by a sliver; Tuesday’s extra “free” nudged it a short distance, and that was enough to cross.

So the training sequence was fine and the learner was correct by the rules we gave it. It found a consistent halfspace, and lesson 9 never asked for more. The trouble is that there are infinitely many consistent halfspaces, and the Perceptron returns whichever one its updates happen to land on. Some of them leave plenty of room around every message. Some graze. Which one *should* a learner return? And while we are at it, the team has a second worry: next month’s labels will contain a few messages that no halfspace can get right, because people disagree about what is junk. What does the learner do when the mail is only almost separable?

This lesson answers both questions. The answer has a name you have probably heard, the **support vector machine**, and the theory behind it delivers something lesson 9 could not: a number of labels that does not grow with the size of the vocabulary.

## How close is close?

Let’s look at the actual messages. As in lesson 9, a message is a vector of word counts, here over a vocabulary of two words so you can draw it: \(x_1\) counts “free” and \(x_2\) counts “meeting”. Labels are \(+1\) for junk and \(-1\) for keep. These are the eight messages the Perceptron trained on, in the order it scanned them.

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

The Perceptron of lesson 9 appends a constant 1 to every message, so its weights are \((b, w_1, w_2)\) with the bias term first, starts from zeros, and adds \(y_i \mathbf x_i\) (the bold \(\mathbf x_i\) is message \(i\)’s whole vector \((1, x_1, x_2)\), and multiplying by its label \(y_i\) flips it for keep messages) for the first message it gets wrong until there is none. Before you run it, a question: the four keep messages all sit near the origin, and the junk messages are far out along the “free” axis. Would you expect the line it returns to leave a lot of room on both sides?

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

Five updates (on A, E twice, F and G) and the result is the score \(4x_1 - x_2 - 3\). The second line is the distance from each message to the line \(4x_1 - x_2 - 3 = 0\), computed as \(|\text{score}| / \|(4, -1)\|\), where the bars \(|\cdot|\) drop the sign and the double bars \(\|(4, -1)\| = \sqrt{4^2 + 1^2} = \sqrt{17}\) give the length of the weight pair; we will prove in a moment that this is the right formula. Look at G, the lunch invitation: 0.243. One extra “free” moves a message one full unit along the \(x_1\) axis, much further than 0.243, and the third line shows what happens. Tuesday’s invitation, \((2, 2)\), scores 3: junk. And it is not alone. Add one “free” to E (“see you tomorrow, free for coffee?”, \((1, 0)\)) or to H (“free pizza and free drinks after the meeting”, \((2, 3)\)) and both become junk too. Three of the four keep messages are one word away from the junk folder.

[[diagram:graze]]

The fix is not to stop using halfspaces. It is to stop accepting just *any* consistent halfspace, and to ask for the one that leaves the most room. To say that precisely we need a way to measure room, which means measuring the distance from a point to a hyperplane.

## The distance to a hyperplane

First, the notation we will use for the whole lesson, read once on the Perceptron’s line. That line weighs “free” by 4, “meeting” by \(-1\), adds \(-3\), and calls the message junk when the total is positive. A halfspace is named by a pair \((\mathbf w, b)\): it labels \(\mathbf x\) with \(\operatorname{sign}(\langle \mathbf w, \mathbf x \rangle + b)\), and its boundary is the hyperplane \(\{\mathbf v : \langle \mathbf w, \mathbf v \rangle + b = 0\}\).

- \(\mathbf w = (w_1, \dots, w_d)\) is the weight vector, one weight per word, and \(b\) is the bias term, a number added at the end. Here \(\mathbf w = (4, -1)\) and \(b = -3\).
- \(\langle \mathbf w, \mathbf x \rangle = w_1 x_1 + \dots + w_d x_d\) is the inner product of lesson 9: multiply matching coordinates and add.
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

When \(\|\mathbf w\| \ne 1\), divide by it first: the distance is \(|\langle \mathbf w, \mathbf x \rangle + b| / \|\mathbf w\|\), which is the formula the first snippet used. Here is the claim tested in three dimensions: a random unit \(\mathbf w\), a random \(b\) and \(\mathbf x\), the foot point \(\mathbf v\), and ten thousand other random points of the hyperplane, none of which may be closer.

```python
import math, random

rng = random.Random(0)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
for _ in range(3):
    w = [rng.gauss(0, 1) for _ in range(3)]
    w = [c / math.sqrt(dot(w, w)) for c in w]
    b, x = rng.uniform(-2, 2), [rng.uniform(-3, 3) for _ in "abc"]
    s = dot(w, x) + b
    v = [xi - s * wi for xi, wi in zip(x, w)]
    Z = [[rng.gauss(0, 3) for _ in "abc"] for _ in range(10_000)]
    U = [[vi + zi - dot(z, w) * wi
          for vi, zi, wi in zip(v, z, w)] for z in Z]
    nearest = min(math.dist(x, u) for u in U)
    print(round(abs(s), 4), round(math.dist(x, v), 4),
          abs(dot(w, v) + b) < 1e-12, nearest >= math.dist(x, v))
```

It prints `1.1182 1.1182 True True`, `3.0192 3.0192 True True` and `1.3828 1.3828 True True`: the formula matches the distance to \(\mathbf v\), \(\mathbf v\) is on the hyperplane, and no other sampled point of the hyperplane beats it. (Each `u` is `v` plus a random vector with its component along `w` removed, which keeps it on the hyperplane.)

## The margin, and the widest separator

Now the measure of room. For a training sequence \(S = ((\mathbf x_1, y_1), \dots, (\mathbf x_m, y_m))\) with \(\mathbf x_i \in \mathbb R^d\) and \(y_i \in \{\pm 1\}\), recall from lesson 9 that \(S\) is **separable** when some halfspace gets every example right, that is, some \((\mathbf w, b)\) has

\[ y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \quad \text{for all } i \in [m]. \]

Here \([m]\) is the list of positions \(\{1, \dots, m\}\), and \(y_i\) is example \(i\)’s label, \(+1\) or \(-1\). Multiplying the score by the label gives the **signed score**: positive exactly when the score has the label’s sign, that is, when the halfspace gets example \(i\) right. In plain words: **some halfspace gives every training message a positive signed score.** For G under the Perceptron’s line, the score is \(-1\) and the label \(-1\), so the signed score is \(+1\): right. The first snippet’s loop tests exactly this, `y * score(w, x) <= 0`, and stops when no message fails it.

Every such halfspace has **empirical risk** 0 (the fraction of training messages it gets wrong, lesson 1), so every one of them is an ERM hypothesis. The **margin** of a hyperplane with respect to \(S\) is the smallest distance between a training message and the hyperplane. By the claim, for a separating \((\mathbf w, b)\) with \(\|\mathbf w\| = 1\), it is \(\min_i |\langle \mathbf w, \mathbf x_i \rangle + b|\), and since every signed score is positive, that is the same as \(\min_i y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). The subscript in \(\min_i\) says what varies: compute the expression for each example \(i\) and keep the smallest. In plain words: **the margin is the signed score of the worst-placed message, once \(\mathbf w\) is a unit vector.** The Perceptron’s hyperplane has margin 0.243 on our eight messages, set by G: the smallest entry of the first snippet’s second line, which divides by \(\|\mathbf w\|\) to make the vector unit.

Why care about the margin? Because a hyperplane with a large margin still separates the training messages after you perturb each one a little: every message can move by anything less than the margin, in any direction, without crossing. A grazing hyperplane survives almost no perturbation at all. Later in this lesson we will also see that a large margin buys a small true risk, regardless of the dimension. That is the real prize.

**Definition (hard-SVM).** **Hard-SVM** is the learning rule that returns, among all halfspaces that separate \(S\), one with the largest margin:

\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} |\langle \mathbf w, \mathbf x_i \rangle + b| \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i. \]

When \(S\) is separable, this is the same as dropping the constraint and maximizing the smallest signed score directly,

\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} y_i (\langle \mathbf w, \mathbf x_i \rangle + b), \]

and proving that the two agree is this lesson’s first problem. We will work with the second form.

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

Why should the shortest \(\mathbf w\) give the widest margin? Think of the constraint as demanding a score of at least 1 from every message. The score of \(\mathbf x\) is its distance to the hyperplane times \(\|\mathbf w\|\), so a short \(\mathbf w\) can only reach a score of 1 on every message if every message is far away. Minimizing the length of \(\mathbf w\) pushes the messages as far from the hyperplane as they can be.

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

Three messages sit exactly at score 1, which means exactly at distance 1.342 from the line: B, E and G. Those are the messages that hold the hyperplane in place, and they will get a name later in this lesson. And Tuesday’s lunch invitation? Its score is \((4 - 2)/3 - 1 = -\tfrac13\): keep. So are the nudged versions of E and H, at \(-\tfrac13\) and \(-\tfrac23\).

[[diagram:margin-band]]

Before going further, try it with your hands. In the lab below you can rerun the Perceptron in different orders of the eight messages and watch the margin it happens to leave, then switch on hard-SVM and press “add one word” to see which messages survive the nudge.

[[lab]]

## Without the bias term

It is often more convenient to work with **homogeneous** halfspaces, those with no bias term, \(\mathbf x \mapsto \operatorname{sign}(\langle \mathbf w, \mathbf x \rangle)\), whose hyperplane passes through the origin. Hard-SVM for homogeneous halfspaces is the program

\[ \min_{\mathbf w} \|\mathbf w\|^2 \quad \text{subject to} \quad y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1 \text{ for all } i. \]

It is the program you just read with \(b\) gone: **the shortest weight vector that gives every message a signed score of at least 1.**

As in lesson 9, a halfspace with a bias term becomes a homogeneous one after appending a constant 1 to every message: \((1, \mathbf x)\) in \(\mathbb R^{d+1}\), with the bias term as the first weight. But there is one difference worth noticing. The program with a bias term minimizes \(\|\mathbf w\|^2\) and leaves \(b\) free, while the homogeneous program on the lifted vectors minimizes \(b^2 + \|\mathbf w\|^2\): it **regularizes** the bias term too, preferring small \(b\). In general the two programs can return different halfspaces. In practice, regularizing \(b\) usually makes no significant difference to the number of examples needed, and the homogeneous form is much easier to analyze, so from here on the theory is homogeneous.

On our eight messages the two even agree: the homogeneous program on the lifted vectors returns the same \((b, w_1, w_2) = (-1, \tfrac23, -\tfrac13)\), as the solver in the duality section will confirm. The margin is now measured in \(\mathbb R^3\), where the length includes the bias term, so it is \(1/\sqrt{1 + \tfrac49 + \tfrac19} = 3/\sqrt{14} \approx 0.802\) rather than 1.342. Same hyperplane in the plane, different ruler.

## Why a wide margin needs fewer labels

Here is the deeper reason to want a wide margin, and it is the reason this lesson exists.

Lesson 9 showed that homogeneous halfspaces in \(\mathbb R^d\) (the space of vectors with \(d\) coordinates, one per word) have VC dimension \(d\): they can label some \(d\) messages in every possible way, and no \(d + 1\). By the fundamental theorem of lesson 6, the number of examples needed to learn them grows with \(d\), and, worse, for an accuracy target \(\varepsilon\) (epsilon, the true risk you are willing to tolerate), if the number of examples is much smaller than \(d/\varepsilon\), then *no* learner can guarantee true risk at most \(\varepsilon\) for every distribution. For a filter with two words that is nothing. For a filter that tracks fifty thousand words it is a disaster: the label budget would have to grow with the vocabulary.

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

The number \(\rho\) is the **radius**: every message lies in the ball of radius \(\rho\). (Lesson 9 called the radius of a training sequence \(R\); we switch to \(\rho\) here, because it will soon play a second role.) Scaling the data multiplies both \(\gamma\) and \(\rho\), so the ratio \(\rho / \gamma\) is the honest measure of difficulty.

**Theorem (hard-SVM’s sample complexity depends on the margin, not the dimension).** Let \(\mathcal D\) be a distribution over \(\mathbb R^d \times \{\pm 1\}\) that is separable with a \((\gamma, \rho)\)-margin using a homogeneous halfspace. Then, with probability at least \(1 - \delta\) over a training sequence \(S \sim \mathcal D^m\), the homogeneous hard-SVM hypothesis has 0-1 true risk at most

\[ \sqrt{\frac{4 (\rho / \gamma)^2}{m}} + \sqrt{\frac{2 \log(2/\delta)}{m}}. \]

Read it slowly, because it is the heart of the lesson. The dimension \(d\) does not appear anywhere. A filter over fifty thousand words and a filter over two words get the same guarantee, as long as the mail is separable with the same ratio \(\rho/\gamma\). Setting the bound equal to \(\varepsilon\), the number of labels needed grows like \((\rho/\gamma)^2/\varepsilon^2\), with no \(d\). (Here \(\log\) is the natural logarithm, as everywhere in this course.)

The proof needs a tool we have not built yet, the Rademacher complexity of a class, and lesson 26 proves this theorem in full. What we can do now is feel the numbers. Take \(\rho/\gamma = 10\), so the closest message to the best separator is a tenth of the radius away, and \(\delta = 0.05\):

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

The ratio \(\rho/\gamma\) has met you before. Lesson 9 proved that on separable data the Perceptron makes at most \((R/\gamma^\star)^2\) updates, where \(\gamma^\star\) is the best margin of a homogeneous separator. In this lesson’s letters that is \((\rho/\gamma)^2\), the very quantity that controls hard-SVM’s sample complexity. For our lifted eight messages it is 46.67 (from the scaling snippet), and the Perceptron used 5. The same ratio governs how fast the Perceptron finishes and how many examples a large-margin separator needs. That is not a coincidence: both are statements that wide, well-scaled margins make halfspaces easy.

## Mail that is almost separable

Now the team’s second worry. Next month’s labels include two new messages.

| message | gist | “free” \(x_1\) | “meeting” \(x_2\) | label \(y\) |
|---|---|---|---|---|
| I | free consultation: book a meeting, meeting link inside | 1 | 2 | junk, \(+1\) |
| J | free cake, free coffee, free hugs: Ana’s farewell meeting | 3 | 1 | keep, \(-1\) |

Look at I. It has exactly the counts of G, the lunch invitation, but your users called it junk. No hypothesis that sees only these two counts can label both correctly, halfspace or not. The ten messages are not separable, the hard-SVM program has no feasible point at all, and hard-SVM returns nothing. We need a version that tolerates some violations.

The hard program demands \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1\) for every \(i\). A natural relaxation lets each constraint be violated, at a price. Introduce a **slack variable** \(\xi_i \ge 0\) for each example and ask only for \(y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1 - \xi_i\). The slack \(\xi_i\) measures how much the constraint for example \(i\) is violated. Then minimize the norm of \(\mathbf w\) (which stands for the margin) and the average slack (which stands for the violations) together, with a parameter \(\lambda > 0\) setting the exchange rate between them.

**Definition (soft-SVM).** Given \(S\) and a parameter \(\lambda > 0\), **soft-SVM** solves

\[ \min_{\mathbf w, b, \boldsymbol \xi} \left( \lambda \|\mathbf w\|^2 + \frac{1}{m} \sum_{i=1}^m \xi_i \right) \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) \ge 1 - \xi_i \text{ and } \xi_i \ge 0 \text{ for all } i, \]

and returns \((\mathbf w, b)\).

A large \(\lambda\) makes the norm expensive: the learner prefers a short \(\mathbf w\), a wide margin, and pays for it with more violations. A small \(\lambda\) makes violations expensive: the learner fits the examples harder and accepts a narrower margin.

The slack variables are a modelling device, and they disappear on inspection. Recall the **hinge loss** you met in lesson 12,

\[ \ell^{\text{hinge}}((\mathbf w, b), (\mathbf x, y)) = \max\{0, \, 1 - y (\langle \mathbf w, \mathbf x \rangle + b)\}, \]

and write \(L_S^{\text{hinge}}((\mathbf w, b))\) for its average over the training sequence. It is zero for an example with signed score at least 1, and grows linearly as the score falls below 1.

**Claim (slack is hinge loss).** Soft-SVM is the same problem as

\[ \min_{\mathbf w, b} \left( \lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}((\mathbf w, b)) \right). \]

*Proof.* Fix \((\mathbf w, b)\) and minimize the soft-SVM objective over the slacks alone. The objective is a sum, and each \(\xi_i\) appears in exactly one term and one pair of constraints, so we can minimize each \(\xi_i\) separately. Write \(s_i = y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). The constraints on \(\xi_i\) are \(\xi_i \ge 0\) and \(\xi_i \ge 1 - s_i\), so \(\xi_i \ge \max\{0, 1 - s_i\}\), and the smallest allowed value is exactly \(\max\{0, 1 - s_i\} = \ell^{\text{hinge}}((\mathbf w, b), (\mathbf x_i, y_i))\). Plugging in the best slacks turns the objective into \(\lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}((\mathbf w, b))\). Minimizing over \((\mathbf w, b, \boldsymbol\xi)\) is the same as first minimizing over \(\boldsymbol\xi\) for each \((\mathbf w, b)\) and then over \((\mathbf w, b)\), so the two problems have the same minimizers in \((\mathbf w, b)\) and the same optimal value. \(\blacksquare\)

Here is the inner minimization done by brute force for four random signed scores, searching \(\xi\) on a grid of step 0.001:

```python
import random

rng = random.Random(0)
for _ in range(4):
    margin = round(rng.uniform(-2, 3), 2)
    grid = [k / 1000 for k in range(5001)]
    feasible = [xi for xi in grid if margin >= 1 - xi]
    print(margin, min(feasible), round(max(0.0, 1 - margin), 2))
```

It prints `2.22 0.0 0.0`, `1.79 0.0 0.0`, `0.1 0.9 0.9` and `-0.71 1.71 1.71`: the smallest feasible slack is the hinge loss every time.

So soft-SVM is **regularized loss minimization**, the paradigm of lesson 13: minimize the empirical risk under some loss plus a penalty \(\lambda \|\mathbf w\|^2\) on the norm. It has a bias, in the sense of lesson 1, toward short weight vectors, which is to say toward wide margins. As before, the homogeneous version drops \(b\) (or lifts the messages) and solves

\[ \min_{\mathbf w} \left( \lambda \|\mathbf w\|^2 + L_S^{\text{hinge}}(\mathbf w) \right), \qquad L_S^{\text{hinge}}(\mathbf w) = \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\}. \]

Let’s run it on the ten messages, lifted, with \(\lambda = 0.1\). The minimizer turns out to be \(\mathbf w = (-\tfrac12, \tfrac12, -\tfrac12)\), the score \((x_1 - x_2 - 1)/2\); the solver behind that claim appears in the duality section. Compare its objective with the hard-SVM separator of the eight messages, which ignores I and J.

```python
S10 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
       ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1),
       ((1, 2), 1), ((3, 1), -1)]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
X = [(1,) + x for x, _ in S10]
Y = [y for _, y in S10]

def soft_svm_objective(w, lam):
    hinge = [max(0.0, 1 - y * dot(w, x)) for x, y in zip(X, Y)]
    return lam * dot(w, w) + sum(hinge) / len(X), hinge

for w in [(-0.5, 0.5, -0.5), (-1, 2 / 3, -1 / 3)]:
    value, hinge = soft_svm_objective(w, 0.1)
    print(round(value, 4), [round(h, 2) for h in hinge])
```

It prints `0.475 [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 1.5]` and then `0.5222 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.67]`.

Read the slacks of the soft-SVM solution. A, B, C, D, F, G and H pay nothing. E pays 0.5: it is on the right side but inside the margin. I pays 2 and J pays 1.5: both are on the wrong side. Total hinge \(4/10 = 0.4\), plus \(0.1 \times \|\mathbf w\|^2 = 0.1 \times 0.75 = 0.075\), gives 0.475. The old hard-SVM separator has no slack on the eight original messages, but its longer \(\mathbf w\) (\(\|\mathbf w\|^2 = 14/9\)) costs \(0.156\) and its slacks on I and J cost 0.367, for 0.522. Soft-SVM accepts a little violation on E to buy a shorter vector.

[[diagram:slack]]

## What soft-SVM costs in labels

Hard-SVM needed the data to be separable with a margin. Soft-SVM needs no such thing, and it comes with its own guarantee, which we can now prove from tools you already have. Three of them, recalled in a line each:

- A function \(f\) is **\(\rho\)-Lipschitz** if \(|f(\mathbf w) - f(\mathbf u)| \le \rho \|\mathbf w - \mathbf u\|\) for all \(\mathbf w, \mathbf u\) (lesson 12): it cannot change faster than rate \(\rho\).
- The **Cauchy–Schwarz inequality**, \(|\langle \mathbf a, \mathbf b \rangle| \le \|\mathbf a\| \|\mathbf b\|\) (proved in lesson 9).
- The **regularized loss minimization guarantee** of lesson 13: if for every example \(z\) the loss \(\ell(\cdot, z)\) is convex and \(\rho\)-Lipschitz, then the rule \(A(S) = \operatorname{argmin}_{\mathbf w} \left( L_S(\mathbf w) + \lambda \|\mathbf w\|^2 \right)\) satisfies, for every vector \(\mathbf u\),

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}(A(S)) \right] \le L_{\mathcal D}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}. \]

That guarantee is an **oracle inequality**: think of \(\mathbf u\) as a good hypothesis an oracle knows about, and the bound says how close the learner gets to it on average. To use it for soft-SVM we need the hinge loss to be convex and Lipschitz.

Convexity is quick. For a fixed example \((\mathbf x, y)\), the map \(\mathbf w \mapsto 1 - y \langle \mathbf w, \mathbf x \rangle\) is affine, and the hinge loss is the pointwise maximum of it and the constant 0. A maximum of two convex functions is convex: for \(\theta \in [0, 1]\), each of the two functions at \(\theta \mathbf w + (1-\theta) \mathbf u\) is at most \(\theta\) times its value at \(\mathbf w\) plus \((1-\theta)\) times its value at \(\mathbf u\), and each of those is at most the same combination of the maxima.

**Claim (the hinge loss is Lipschitz).** For a fixed example \((\mathbf x, y)\) with \(y \in \{\pm 1\}\), the function \(f(\mathbf w) = \max\{0, 1 - y \langle \mathbf w, \mathbf x \rangle\}\) is \(\|\mathbf x\|\)-Lipschitz.

*Proof.* First a fact about the function \(t \mapsto \max\{0, t\}\): for all real \(a\) and \(c\), \(|\max\{0, a\} - \max\{0, c\}| \le |a - c|\). Check the cases. If both are at most 0, the left side is 0. If both are positive, the two sides are equal. If \(a > 0 \ge c\), the left side is \(a\), and \(a \le a - c\) because \(c \le 0\); the case \(c > 0 \ge a\) is the same with the roles swapped.

Now take \(a = 1 - y \langle \mathbf w, \mathbf x \rangle\) and \(c = 1 - y \langle \mathbf u, \mathbf x \rangle\). Then \(a - c = y \langle \mathbf u - \mathbf w, \mathbf x \rangle\), and

\[ |f(\mathbf w) - f(\mathbf u)| \le |y| \, |\langle \mathbf u - \mathbf w, \mathbf x \rangle| \le \|\mathbf u - \mathbf w\| \, \|\mathbf x\|, \]

using \(|y| = 1\) and Cauchy–Schwarz. \(\blacksquare\)

(Lesson 14 offers a second route: every subgradient of \(f\) has the form \(\alpha \mathbf x\) with \(|\alpha| \le 1\), so its length is at most \(\|\mathbf x\|\), and a convex function whose subgradients are all at most \(\rho\) long is \(\rho\)-Lipschitz.)

The ratio \(|f(\mathbf w) - f(\mathbf u)| / (\|\mathbf x\| \|\mathbf w - \mathbf u\|)\) should never exceed 1, and it should get close to 1 when both \(\mathbf w\) and \(\mathbf u\) sit on the sloped part of the hinge and \(\mathbf w - \mathbf u\) points along \(\mathbf x\):

```python
import math, random

rng = random.Random(0)
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
hinge = lambda w, x, y: max(0, 1 - y * dot(w, x))
worst = 0
for _ in range(100_000):
    x = [rng.uniform(-2, 2) for _ in range(3)]
    w = [rng.uniform(-2, 2) for _ in range(3)]
    u = [rng.uniform(-2, 2) for _ in range(3)]
    y = rng.choice((1, -1))
    gap = abs(hinge(w, x, y) - hinge(u, x, y))
    ratio = gap / (math.sqrt(dot(x, x)) * math.dist(w, u))
    worst = max(worst, ratio)
print(round(worst, 4))
```

It prints `0.9999`: over a hundred thousand random triples, the ratio comes within a hair of 1 and never passes it.

So when every message lies in the ball of radius \(\rho\), the hinge loss is convex and \(\rho\)-Lipschitz on that domain, and this is the second role of \(\rho\): the radius of the messages is the Lipschitz constant of the loss. Now the guarantee.

**Theorem (soft-SVM’s sample complexity depends on the norm, not the dimension).** Let \(\mathcal X = \{\mathbf x \in \mathbb R^d : \|\mathbf x\| \le \rho\}\), let \(\mathcal D\) be a distribution over \(\mathcal X \times \{\pm 1\}\), and let \(A(S)\) be the homogeneous soft-SVM solution on \(S \sim \mathcal D^m\) with parameter \(\lambda\). Then for every \(\mathbf u\),

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{\text{hinge}}(A(S)) \right] \le L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}, \]

and the same bound holds for the expected 0-1 true risk \(\mathbb E_S [L_{\mathcal D}^{0-1}(A(S))]\). Moreover, for every \(B > 0\), choosing \(\lambda = \sqrt{2\rho^2 / (B^2 m)}\) gives

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{0-1}(A(S)) \right] \le \mathop{\mathbb E}_{S \sim \mathcal D^m} \left[ L_{\mathcal D}^{\text{hinge}}(A(S)) \right] \le \min_{\mathbf w : \|\mathbf w\| \le B} L_{\mathcal D}^{\text{hinge}}(\mathbf w) + \sqrt{\frac{8 \rho^2 B^2}{m}}. \]

*Proof.* *The first bound.* Soft-SVM is regularized loss minimization with the hinge loss, which we just showed is convex and \(\rho\)-Lipschitz for every example in \(\mathcal X \times \{\pm 1\}\). The guarantee of lesson 13 applies word for word with \(L_{\mathcal D} = L_{\mathcal D}^{\text{hinge}}\).

*From hinge to 0-1.* For every example, the 0-1 loss is at most the hinge loss. Write \(s = y \langle \mathbf w, \mathbf x \rangle\). If \(s \le 0\), the halfspace errs or sits on the boundary, the 0-1 loss is at most 1, and the hinge loss is \(1 - s \ge 1\). If \(s > 0\), the 0-1 loss is 0, and the hinge loss is nonnegative. So \(L_{\mathcal D}^{0-1}(\mathbf w) \le L_{\mathcal D}^{\text{hinge}}(\mathbf w)\) for every \(\mathbf w\), taking expectations over \(\mathcal D\) on both sides of a pointwise inequality, and then over \(S\) as well.

*Choosing \(\lambda\).* Fix \(B > 0\) and any \(\mathbf u\) with \(\|\mathbf u\| \le B\). Then \(\lambda \|\mathbf u\|^2 \le \lambda B^2\), so the first bound gives \(L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda B^2 + 2\rho^2/(\lambda m)\). With \(\lambda = \sqrt{2\rho^2/(B^2 m)}\), the two terms that depend on \(\lambda\) become

\[ \lambda B^2 = \sqrt{\frac{2\rho^2 B^4}{B^2 m}} = \sqrt{\frac{2 \rho^2 B^2}{m}}, \qquad \frac{2\rho^2}{\lambda m} = \frac{2\rho^2}{m} \sqrt{\frac{B^2 m}{2 \rho^2}} = \sqrt{\frac{4 \rho^4 B^2 m}{2 \rho^2 m^2}} = \sqrt{\frac{2\rho^2 B^2}{m}}. \]

They are equal, which is no accident: this \(\lambda\) is exactly where the increasing term \(\lambda B^2\) and the decreasing term \(2\rho^2/(\lambda m)\) cross, and for a sum \(a\lambda + c/\lambda\) with \(a, c > 0\) the crossing point is the minimum. Their sum is \(2\sqrt{2\rho^2 B^2/m} = \sqrt{8 \rho^2 B^2/m}\). The bound holds for every \(\mathbf u\) in the ball of radius \(B\), so it holds for the best one, which gives the minimum over \(\|\mathbf w\| \le B\). \(\blacksquare\)

Two remarks on what this says. First, it is a bound on the *expected* true risk, averaged over the random training sequence, not a high-probability bound like the hard-SVM theorem; Markov’s inequality turns the one into the other, at some cost in the constants. Second, look at what controls it: the radius \(\rho\), the norm budget \(B\) (which plays the part of \(1/\gamma\), since a separator of norm \(B\) at score level 1 has margin \(1/B\)), and, when the mail is not separable, the best hinge risk available within that norm. No \(d\).

Let’s put numbers in. Messages of radius \(\rho = 5\), and a comparison separator \(\mathbf u\) of norm \(B = 2\) with hinge risk 0.04:

```python
import math

rho, B, hinge_u = 5, 2, 0.04

def soft_svm_bound(lam, m):
    return hinge_u + lam * B ** 2 + 2 * rho ** 2 / (lam * m)

for m in (10_000, 50_000):
    lam = math.sqrt(2 * rho ** 2 / (B ** 2 * m))
    grid = min(soft_svm_bound(k / 1e5, m)
               for k in range(1, 10**4))
    closed = hinge_u + math.sqrt(8 * rho ** 2 * B ** 2 / m)
    print(m, round(lam, 4), round(soft_svm_bound(lam, m), 4),
          round(grid, 4), round(closed, 4))
```

It prints `10000 0.0354 0.3228 0.3228 0.3228` and `50000 0.0158 0.1665 0.1665 0.1665`. The formula’s \(\lambda\), a brute-force search over \(\lambda\), and the closed form all agree: with 50,000 labels the expected 0-1 true risk is certified below 0.1665.

A word on the radius. The requirement that every message lies in a ball of radius \(\rho\) is not a technicality. We saw that without a constraint on scale, any margin can be made as large as you like by multiplying the messages by a big number. The radius is what gives the norm of \(\mathbf w\), and the margin, a meaning.

## Margins against dimension

How much does dropping \(d\) actually buy? Compare the two kinds of bound. The VC route of lesson 6, for homogeneous halfspaces with VC dimension \(d\), makes the estimation error of an ERM hypothesis shrink like \(\sqrt{d/m}\). The margin route makes it shrink like \(\sqrt{\rho^2 B^2/m}\). Which is smaller depends on the problem, so let’s look at a realistic one.

Say you want to tell whether a short document is about sports. Represent it as a **bag of words**: fix a dictionary of \(d\) words, and let \(x_i = 1\) if word \(i\) appears in the document and 0 otherwise, so \(\mathbf x \in \{0, 1\}^d\). Then \(\|\mathbf x\|^2\) is the number of distinct dictionary words in the document, and \(\rho^2\) is the largest such number. A halfspace assigns a weight to every word, and it is natural to assume that positive and negative weights on a few dozen words decide the topic with reasonable accuracy, so \(B^2\) under 100 is plausible. Overall, \(\rho^2 B^2\) below 10,000 is a reasonable assumption. A typical dictionary, on the other hand, is far bigger than 10,000 words; English has more than 100,000.

```python
import math

d, rho2_B2 = 100_000, 10_000
for m in (10**5, 10**6, 10**7):
    vc_term = math.sqrt(d / m)
    margin_term = math.sqrt(rho2_B2 / m)
    print(m, round(vc_term, 3), round(margin_term, 3))
```

It prints `100000 1.0 0.316`, `1000000 0.316 0.1` and `10000000 0.1 0.032`. At every \(m\), the margin term is \(\sqrt{10}\) times smaller, and to reach the same value the VC route needs ten times as many examples: an order of magnitude, for the same halfspaces, just from asking for a wide margin. (The constants in front differ between the two bounds, so read this as a comparison of how they scale, not of their exact values.) And the gap only widens as the dictionary grows, because the margin term does not see the dictionary at all.

[[diagram:bounds]]

It is not a free lunch, though, and you know by now (lesson 5) that there is none. One can build problems where the margin bound is worse than the VC bound: the mail might be separable only by a hyperplane that passes close to many messages. Preferring a large margin is another **inductive bias**. It can sharply decrease the estimation error, and it can also increase the approximation error, when the truth is a halfspace with a narrow margin and the learner, steered toward short \(\mathbf w\), never reaches it.

## The ramp loss

One more subtlety about the soft-SVM guarantee. Its approximation term is the best *hinge* risk, \(\min_{\|\mathbf w\| \le B} L_{\mathcal D}^{\text{hinge}}(\mathbf w)\), while a VC bound measures the approximation error with the 0-1 loss. Since the hinge loss is at least the 0-1 loss, the hinge approximation term can be larger, and the 0-1 approximation error never exceeds it.

Could we get the nice \(\sqrt{\rho^2 B^2/m}\) estimation term with the 0-1 loss itself? No. The 0-1 loss is **scale insensitive**: multiplying \(\mathbf w\) by any positive number changes no label and no 0-1 loss, so the norm of \(\mathbf w\), and the margin, mean nothing to it. A norm-based bound needs a loss that notices scale. The **ramp loss** is a compromise that notices scale and is otherwise much closer to the 0-1 loss:

\[ \ell^{\text{ramp}}(\mathbf w, (\mathbf x, y)) = \min\{1, \ell^{\text{hinge}}(\mathbf w, (\mathbf x, y))\} = \min\{1, \max\{0, 1 - y \langle \mathbf w, \mathbf x \rangle\}\}. \]

As a function of the signed score \(s = y\langle \mathbf w, \mathbf x \rangle\):

```python
zero_one = lambda s: 1 if s <= 0 else 0
hinge = lambda s: max(0.0, 1.0 - s)
ramp = lambda s: min(1.0, hinge(s))
for s in (-2, -0.5, 0, 0.5, 1, 2):
    print(s, zero_one(s), hinge(s), ramp(s))
```

It prints `-2 1 3.0 1.0`, `-0.5 1 1.5 1.0`, `0 1 1.0 1.0`, `0.5 0 0.5 0.5`, `1 0 0.0 0.0` and `2 0 0.0 0.0`.

On a mistake (\(s \le 0\)) the ramp loss charges exactly 1, like the 0-1 loss, while the hinge loss keeps growing. On a comfortable correct answer (\(s \ge 1\)) both charge 0. The ramp and 0-1 losses differ only on examples that are correct but not by a significant margin, \(0 < s < 1\). Generalization bounds for the ramp loss come with lesson 26.

[[diagram:hinge]]

So why does SVM use the hinge loss and not the ramp? Computation. The hinge loss is convex, so minimizing it is efficient: every local minimum is global, and the methods of lessons 12 to 14 apply. The ramp loss is not convex (it is flat for very negative \(s\), then slopes, then is flat again), and minimizing it is computationally intractable.

## Support vectors

Where does the name come from? Go back to the eight separable messages and the three that sat exactly at score 1: B, E and G. The claim behind the name is that hard-SVM’s weight vector is **supported** by those messages alone: it is a linear combination of them, and the other five could move around (as long as they stay outside the margin) without changing it.

We work with homogeneous hard-SVM, \(\mathbf w_0 = \operatorname{argmin}_{\mathbf w} \|\mathbf w\|^2\) subject to \(y_i \langle \mathbf w, \mathbf x_i \rangle \ge 1\) for all \(i\). Two facts first, so that \(\mathbf w_0\) is well defined when \(S\) is separable. It **exists**: take any feasible \(\mathbf w'\); the feasible vectors with \(\|\mathbf w\| \le \|\mathbf w'\|\) form a closed and bounded set (each constraint is a closed halfspace), it is nonempty, and a continuous function on a nonempty closed bounded set attains its minimum. It is **unique**: if \(\mathbf w\) and \(\mathbf w'\) were two different minimizers, both of norm \(r\), their midpoint would be feasible (average the two constraint inequalities) and, by the identity \(\|\tfrac{\mathbf w + \mathbf w'}{2}\|^2 = \tfrac12 \|\mathbf w\|^2 + \tfrac12 \|\mathbf w'\|^2 - \tfrac14 \|\mathbf w - \mathbf w'\|^2\), it would have squared norm \(r^2 - \tfrac14 \|\mathbf w - \mathbf w'\|^2 < r^2\), contradicting minimality.

**Theorem (support vectors).** Let \(\mathbf w_0\) be the homogeneous hard-SVM solution on a separable \(S\), and let \(I = \{ i : |\langle \mathbf w_0, \mathbf x_i \rangle| = 1\}\). Then there are coefficients \(\alpha_i\), \(i \in I\), with

\[ \mathbf w_0 = \sum_{i \in I} \alpha_i \mathbf x_i. \]

The messages \(\{\mathbf x_i : i \in I\}\) are called the **support vectors**.

Since \(\mathbf w_0\) is feasible, \(y_i \langle \mathbf w_0, \mathbf x_i \rangle \ge 1 > 0\) for every \(i\), so \(|\langle \mathbf w_0, \mathbf x_i \rangle| = y_i \langle \mathbf w_0, \mathbf x_i \rangle\). Thus \(I\) is the set of examples whose constraint holds with equality, the ones at distance exactly \(1/\|\mathbf w_0\|\) from the hyperplane, and every other example has \(y_i \langle \mathbf w_0, \mathbf x_i \rangle > 1\).

The standard proof applies a general tool from optimization, the **Fritz John conditions**: if \(\mathbf w^\star\) minimizes a differentiable \(f\) subject to \(g_i(\mathbf w) \le 0\) for differentiable \(g_1, \dots, g_m\), then there is \(\boldsymbol\alpha \in \mathbb R^m\) with \(\nabla f(\mathbf w^\star) + \sum_{i \in I} \alpha_i \nabla g_i(\mathbf w^\star) = \mathbf 0\), where \(I = \{i : g_i(\mathbf w^\star) = 0\}\) is the set of constraints that are tight at the optimum. With \(f(\mathbf w) = \|\mathbf w\|^2\) and \(g_i(\mathbf w) = 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\), the gradients are \(2\mathbf w_0\) and \(-y_i \mathbf x_i\), and solving for \(\mathbf w_0\) gives the theorem. We will not prove the Fritz John conditions here. Instead, here is a direct proof that uses nothing but the geometry of the inner product.

*Proof.* Let \(V\) be the span of the support vectors, the set of all linear combinations of \(\{\mathbf x_i : i \in I\}\). Every vector splits uniquely into a part in \(V\) and a part perpendicular to every vector of \(V\) (its orthogonal projection onto \(V\) and the remainder), so write \(\mathbf w_0 = \mathbf p + \mathbf r\) with \(\mathbf p \in V\) and \(\langle \mathbf r, \mathbf x_i \rangle = 0\) for every \(i \in I\). The theorem says exactly that \(\mathbf r = \mathbf 0\). Suppose not; we will build a feasible vector shorter than \(\mathbf w_0\), which is impossible.

For a number \(t\) with \(0 < t \le 1\), consider \(\mathbf w_t = \mathbf w_0 - t \mathbf r\), which shaves off part of the component that the support vectors cannot see.

*The support vectors do not notice.* For \(i \in I\), \(y_i \langle \mathbf w_t, \mathbf x_i \rangle = y_i \langle \mathbf w_0, \mathbf x_i \rangle - t y_i \langle \mathbf r, \mathbf x_i \rangle = 1 - 0 = 1\). Their constraints still hold.

*The others have room to spare.* For \(i \notin I\), we know \(y_i \langle \mathbf w_0, \mathbf x_i \rangle > 1\). Then \(y_i \langle \mathbf w_t, \mathbf x_i \rangle \ge y_i \langle \mathbf w_0, \mathbf x_i \rangle - t |\langle \mathbf r, \mathbf x_i \rangle|\), and this is still at least 1 as long as \(t |\langle \mathbf r, \mathbf x_i \rangle| \le y_i \langle \mathbf w_0, \mathbf x_i \rangle - 1\), a positive number. There are finitely many examples, so we can choose one \(t \in (0, 1]\) small enough for all of them at once. Then \(\mathbf w_t\) is feasible.

*But it is shorter.* Expand the square:

\[ \|\mathbf w_t\|^2 = \|\mathbf w_0\|^2 - 2t \langle \mathbf w_0, \mathbf r \rangle + t^2 \|\mathbf r\|^2. \]

Since \(\mathbf p\) lies in \(V\) and \(\mathbf r\) is perpendicular to every vector of \(V\), \(\langle \mathbf p, \mathbf r \rangle = 0\), so \(\langle \mathbf w_0, \mathbf r \rangle = \langle \mathbf p + \mathbf r, \mathbf r \rangle = \|\mathbf r\|^2\). Hence \(\|\mathbf w_t\|^2 = \|\mathbf w_0\|^2 - (2t - t^2) \|\mathbf r\|^2\), and for \(0 < t \le 1\) we have \(2t - t^2 = t(2 - t) > 0\), so \(\|\mathbf w_t\|^2 < \|\mathbf w_0\|^2\). A feasible vector shorter than the minimizer is a contradiction, so \(\mathbf r = \mathbf 0\) and \(\mathbf w_0 = \mathbf p \in V\). \(\blacksquare\)

One edge case falls out for free: \(I\) cannot be empty. If it were, \(V = \{\mathbf 0\}\) and the theorem would give \(\mathbf w_0 = \mathbf 0\), which violates every constraint.

[[diagram:support]]

On our eight lifted messages, the support vectors are B, E and G, and the next section’s solver finds the coefficients. Here they are, so you can check by hand: in the form \(\mathbf w_0 = \sum_i \alpha_i y_i \mathbf x_i\) that the next section derives, \(\alpha_B = \tfrac{5}{18}\), \(\alpha_E = \tfrac{10}{9}\), \(\alpha_G = \tfrac16\), and

\[ \tfrac{5}{18}(1, 3, 0) - \tfrac{10}{9}(1, 0, 0) - \tfrac16 (1, 1, 2) = \left(\tfrac{5 - 20 - 3}{18}, \tfrac{15 - 3}{18}, -\tfrac{6}{18}\right) = \left(-1, \tfrac23, -\tfrac13\right). \]

Five messages, the ones far from the line, contribute nothing.

## Duality

Many properties of SVMs were historically discovered by studying a different optimization problem, the **dual** of hard-SVM. Our development never needed it, but it explains where the coefficients \(\alpha_i\) come from, and it is the door to the next lesson, so let’s derive it for the homogeneous program.

*Step 1: hide the constraints inside a max.* Define

\[ g(\mathbf w) = \max_{\boldsymbol\alpha \in \mathbb R^m : \boldsymbol\alpha \ge \mathbf 0} \; \sum_{i=1}^m \alpha_i (1 - y_i \langle \mathbf w, \mathbf x_i \rangle). \]

If \(\mathbf w\) satisfies every constraint, each \(1 - y_i \langle \mathbf w, \mathbf x_i \rangle\) is at most 0, so every term with \(\alpha_i \ge 0\) is at most 0, and the maximum is 0, reached at \(\boldsymbol\alpha = \mathbf 0\). If some constraint fails, its \(1 - y_i \langle \mathbf w, \mathbf x_i \rangle\) is positive, and sending that \(\alpha_i\) to infinity (with the others at 0) makes the sum as large as you like, so \(g(\mathbf w) = \infty\). So \(g\) is 0 on the feasible set and \(\infty\) off it, and hard-SVM is the unconstrained problem \(\min_{\mathbf w} \left( \|\mathbf w\|^2 + g(\mathbf w) \right)\).

*Step 2: a min of a max.* Halving the objective does not change the minimizer, and \(\tfrac12 g = g\) because \(g\) only takes the values 0 and \(\infty\). So hard-SVM has the same solution as

\[ \min_{\mathbf w} \; \max_{\boldsymbol\alpha \ge \mathbf 0} \left( \tfrac12 \|\mathbf w\|^2 + \sum_{i=1}^m \alpha_i (1 - y_i \langle \mathbf w, \mathbf x_i \rangle) \right). \]

*Step 3: swap.* Exchanging the min and the max can only lower the value:

\[ \min_{\mathbf w} \max_{\boldsymbol\alpha \ge \mathbf 0} \Big( \cdots \Big) \;\ge\; \max_{\boldsymbol\alpha \ge \mathbf 0} \min_{\mathbf w} \Big( \cdots \Big). \]

This inequality, true for any function of two variables, is called **weak duality**, and proving it is one of this lesson’s problems. For hard-SVM something stronger holds, **strong duality**: the two sides are equal. We state strong duality without proof; it is a standard result of convex optimization. The right-hand side is the **dual problem**.

*Step 4: solve the inner minimization.* For fixed \(\boldsymbol\alpha\), the inner function of \(\mathbf w\) is \(\tfrac12 \|\mathbf w\|^2\) plus a linear function, unconstrained and differentiable, and convex (a convex quadratic plus a linear term). A convex differentiable function is minimized exactly where its gradient vanishes, and the gradient is \(\mathbf w - \sum_i \alpha_i y_i \mathbf x_i\). So

\[ \mathbf w = \sum_{i=1}^m \alpha_i y_i \mathbf x_i. \]

The optimal weight vector is a combination of the training messages, a fact the next lesson builds on.

*Step 5: substitute.* Put \(\mathbf w = \sum_j \alpha_j y_j \mathbf x_j\) back in. The sum \(\sum_i \alpha_i y_i \langle \mathbf w, \mathbf x_i \rangle = \langle \mathbf w, \sum_i \alpha_i y_i \mathbf x_i \rangle = \|\mathbf w\|^2\), so the objective becomes \(\tfrac12 \|\mathbf w\|^2 + \sum_i \alpha_i - \|\mathbf w\|^2 = \sum_i \alpha_i - \tfrac12 \|\mathbf w\|^2\). Expanding \(\|\mathbf w\|^2 = \langle \sum_i \alpha_i y_i \mathbf x_i, \sum_j \alpha_j y_j \mathbf x_j \rangle\) by bilinearity gives the dual problem:

\[ \max_{\boldsymbol\alpha \in \mathbb R^m : \boldsymbol\alpha \ge \mathbf 0} \left( \sum_{i=1}^m \alpha_i - \frac12 \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \langle \mathbf x_j, \mathbf x_i \rangle \right). \]

Look at what it needs: only the inner products \(\langle \mathbf x_j, \mathbf x_i \rangle\) between messages, never the coordinates of a message on their own. Hold on to that; it is the whole point of the next lesson.

[[diagram:duality]]

The dual is easy to attack one coordinate at a time. Fix every \(\alpha_j\) except \(\alpha_i\); the dual objective is then a downward parabola in \(\alpha_i\), maximized at \(\alpha_i + (1 - y_i \langle \mathbf w, \mathbf x_i \rangle)/\|\mathbf x_i\|^2\) with \(\mathbf w = \sum_j \alpha_j y_j \mathbf x_j\), and clipping at 0 respects \(\alpha_i \ge 0\). Sweeping over the coordinates again and again climbs to the maximum. (This method, dual coordinate ascent, is a standard solver; it is the one that produced the numbers we have been quoting.) At the end, compare the primal value \(\tfrac12 \|\mathbf w_0\|^2\) with the dual value:

```python
S = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
     ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
X, Y = [(1,) + x for x, _ in S], [y for _, y in S]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
alpha, w = [0.0] * 8, [0.0] * 3
for sweep in range(5000):
    for i in range(8):
        gap = 1 - Y[i] * dot(w, X[i])
        new = max(0.0, alpha[i] + gap / dot(X[i], X[i]))
        w = [c + (new - alpha[i]) * Y[i] * xi
             for c, xi in zip(w, X[i])]
        alpha[i] = new
print([round(a, 4) for a in alpha])
print([round(c, 4) for c in w])
primal, dual = dot(w, w) / 2, sum(alpha) - dot(w, w) / 2
print(round(primal, 4), round(dual, 4))
```

It prints `[0.0, 0.2778, 0.0, 0.0, 1.1111, 0.0, 0.1667, 0.0]`, then `[-1.0, 0.6667, -0.3333]`, then `0.7778 0.7778`. The coefficients are \(\tfrac{5}{18}\), \(\tfrac{10}{9}\) and \(\tfrac16\) on B, E and G and zero elsewhere, exactly the support vectors; the weight vector is the hard-SVM solution of the homogeneous section; and primal and dual values agree at \(\tfrac79\), as strong duality promises.

The same method solves soft-SVM, with one change. Repeating the derivation for the soft-SVM program with its slack variables gives a dual whose only difference is a ceiling: each coefficient must stay between 0 and \(1/(2\lambda m)\), so the update clips at both ends. That is the solver behind the soft-SVM solution \((-\tfrac12, \tfrac12, -\tfrac12)\) we quoted for the ten messages.

## Soft-SVM by stochastic gradient descent

Quadratic-program solvers and dual coordinate ascent both work, but for the real mail system, with millions of messages and fifty thousand words, there is a much simpler method you already own. Lesson 14 showed how to solve regularized loss minimization with stochastic gradient descent. Let’s apply it to soft-SVM, in the homogeneous form

\[ \min_{\mathbf w} \left( \frac{\lambda}{2} \|\mathbf w\|^2 + \frac1m \sum_{i=1}^m \max\{0, 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\} \right). \]

(Notice the \(\lambda/2\). It makes the algebra below cleaner. Running this method with parameter \(\lambda\) solves soft-SVM as we defined it with parameter \(\lambda/2\); to get soft-SVM with parameter \(\lambda\), run it with \(2\lambda\).)

Recall the method from lesson 14. The objective \(f(\mathbf w) = \tfrac{\lambda}{2}\|\mathbf w\|^2 + L_S(\mathbf w)\) is \(\lambda\)-strongly convex. At step \(t\), pick an example uniformly at random from \(S\) and a subgradient \(\mathbf v_t\) of its loss at \(\mathbf w^{(t)}\); then \(\lambda \mathbf w^{(t)} + \mathbf v_t\) is, in expectation, a subgradient of \(f\), and the update with step size \(1/(\lambda t)\) is

\[ \mathbf w^{(t+1)} = \mathbf w^{(t)} - \frac{1}{\lambda t} \left( \lambda \mathbf w^{(t)} + \mathbf v_t \right) = \left(1 - \frac1t\right) \mathbf w^{(t)} - \frac{1}{\lambda t} \mathbf v_t, \]

starting from \(\mathbf w^{(1)} = \mathbf 0\).

*The update in closed form.* We claim \(\mathbf w^{(t+1)} = -\frac{1}{\lambda t} \sum_{j=1}^t \mathbf v_j\) for every \(t \ge 1\). For \(t = 1\), the factor \(1 - 1/1 = 0\) kills \(\mathbf w^{(1)}\) and leaves \(-\mathbf v_1/\lambda\). If it holds for \(t - 1\), that is \(\mathbf w^{(t)} = -\frac{1}{\lambda(t-1)} \sum_{j<t} \mathbf v_j\), then

\[ \mathbf w^{(t+1)} = \frac{t-1}{t} \cdot \frac{-1}{\lambda (t-1)} \sum_{j<t} \mathbf v_j - \frac{1}{\lambda t} \mathbf v_t = -\frac{1}{\lambda t} \sum_{j \le t} \mathbf v_j. \]

So the weights at step \(t\) are just the running sum of the subgradients, scaled. Check it against the step-by-step update with fifty random subgradients:

```python
import random

rng, lam = random.Random(0), 0.2
v = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(50)]
w = [0.0, 0.0, 0.0]
for t in range(1, 51):
    g = [lam * wi + vi for wi, vi in zip(w, v[t - 1])]
    w = [wi - gi / (lam * t) for wi, gi in zip(w, g)]
closed = [-sum(col) / (lam * 50) for col in zip(*v)]
print([round(c, 6) for c in w])
print([round(c, 6) for c in closed])
```

Both lines print `[-0.685902, -0.263015, -0.231963]`.

*The subgradient of the hinge loss.* Lesson 14 computed it: at \(\mathbf w\), for the example \((\mathbf x, y)\), take \(\mathbf v = \mathbf 0\) if \(y \langle \mathbf w, \mathbf x \rangle \ge 1\) (the flat part) and \(\mathbf v = -y\mathbf x\) otherwise (the sloped part). Write \(\boldsymbol\theta^{(t)} = -\sum_{j<t} \mathbf v_j\) for the running sum. Then \(\boldsymbol\theta\) changes only on examples with signed score below 1, by adding \(y_i \mathbf x_i\), and the weights are \(\boldsymbol\theta\) scaled down. Matching the indices exactly, the closed form gives \(\mathbf w^{(t)} = \boldsymbol\theta^{(t)}/(\lambda (t-1))\) for \(t \ge 2\). The standard form of the algorithm, below, divides by \(\lambda t\) instead: it is defined already at \(t = 1\), and the two differ by the factor \((t-1)/t\), which tends to 1. That is the whole algorithm.

```
SGD-SoftSVM(S, λ, T):
  θ ← (0, ..., 0) in R^d
  total ← (0, ..., 0) in R^d
  for t = 1..T:
    w ← θ / (λ t)
    total ← total + w
    choose i uniformly at random from {1, ..., m}
    if y_i ⟨w, x_i⟩ < 1:
      θ ← θ + y_i x_i
  return w̄ = total / T
```

Look at the update line. It is the Perceptron’s update, \(y_i \mathbf x_i\), triggered not only on mistakes but on every example with signed score below 1: soft-SVM by SGD is a Perceptron that insists on a margin and shrinks its weights by \(1/(\lambda t)\) as it goes.

[[diagram:sgd]]

How good is \(\bar{\mathbf w}\)? Lesson 14’s analysis for a \(\lambda\)-strongly convex objective with a \(\rho\)-Lipschitz loss gives

\[ \mathbb E[f(\bar{\mathbf w})] - \min_{\mathbf w} f(\mathbf w) \le \frac{4\rho^2}{\lambda T} (1 + \log T). \]

Let’s run it on the ten almost-separable messages, lifted, with \(\lambda = 0.2\) and \(T = 100{,}000\). By the remark above, that solves soft-SVM with parameter 0.1, whose minimizer we quoted as \((-\tfrac12, \tfrac12, -\tfrac12)\) with objective 0.475.

```python
import random

S10 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
       ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1),
       ((1, 2), 1), ((3, 1), -1)]
X, Y = [(1,) + x for x, _ in S10], [y for _, y in S10]
dot = lambda u, v: sum(a * b for a, b in zip(u, v))
rng, lam, T = random.Random(0), 0.2, 100_000
theta, total = [0.0] * 3, [0.0] * 3
for t in range(1, T + 1):
    w = [c / (lam * t) for c in theta]
    total = [s + c for s, c in zip(total, w)]
    i = rng.randrange(len(X))
    if Y[i] * dot(w, X[i]) < 1:
        theta = [c + Y[i] * xi for c, xi in zip(theta, X[i])]
w_bar = [s / T for s in total]
hinge = sum(max(0, 1 - y * dot(w_bar, x)) for x, y in zip(X, Y))
print([round(c, 3) for c in w_bar])
print(round(lam / 2 * dot(w_bar, w_bar) + hinge / len(X), 4))
```

It prints `[-0.518, 0.502, -0.484]` and `0.4752`: within 0.0002 of the optimum 0.475. The guarantee, with \(\rho^2 = 30\) (the longest lifted message is A, \((1, 5, 2)\)), promises an expected gap of at most \(4 \cdot 30 \cdot (1 + \log 100{,}000)/(0.2 \cdot 100{,}000) \approx 0.075\). As with the Perceptron’s mistake bound, the guarantee is a worst case, and on friendly data the method does far better.

## SVMs as code

Here are the two learners side by side, homogeneous, with messages already lifted if a bias term is wanted, so they live in \(\mathbb R^d\).

```
HardSVM(S):
  w0 ← argmin ||w||^2 subject to y_i ⟨w, x_i⟩ ≥ 1 for all i
  return h(x) = sign(⟨w0, x⟩)

SGD-SoftSVM(S, λ, T):
  as in the previous section
  return h(x) = sign(⟨w̄, x⟩)
```

**HardSVM** hands the work to a quadratic-program solver, which runs in time polynomial in \(m\) and \(d\); which polynomial depends on the solver. Through the dual, one sweep of dual coordinate ascent costs \(O(md)\) time (each of the \(m\) coordinate steps computes one inner product and updates \(\mathbf w\)), with \(O(m + d)\) memory for \(\boldsymbol\alpha\) and \(\mathbf w\) beyond the data. It needs separable data; on anything else the program is infeasible.

**SGD-SoftSVM** does \(O(d)\) work per step: one inner product, at most one vector addition, and one addition to the running total. That is \(O(Td)\) time and \(O(d)\) memory beyond the data, for any \(m\): the number of steps is set by the accuracy you want, through the bound above, not by the number of examples. It never needs separability. Predicting a new message costs \(O(d)\) for either learner.

**Build it yourself.** Using only the Python standard library:

```python
import math
import random

S8 = [((5, 2), 1), ((3, 0), 1), ((4, 1), 1), ((4, 0), 1),
      ((0, 0), -1), ((0, 1), -1), ((1, 2), -1), ((1, 3), -1)]
S10 = S8 + [((1, 2), 1), ((3, 1), -1)]

def lift(x): ...
def hinge_risk(w, S): ...
def soft_svm_objective(w, S, lam): ...
def sgd_soft_svm(S, lam, T, rng): ...
def hard_svm(S, sweeps=5000): ...
def plane_margin(w, S): ...
def at_or_inside_margin(w, S, tol=1e-6): ...

if __name__ == "__main__":
    w, alpha = hard_svm(S8)
    assert all(abs(a - b) < 1e-3
               for a, b in zip(w, (-1, 2 / 3, -1 / 3)))
    assert at_or_inside_margin(w, S8) == [1, 4, 6]
    support = [i for i, a in enumerate(alpha) if a > 1e-6]
    assert support == [1, 4, 6]
    assert abs(plane_margin(w, S8) - 3 / math.sqrt(5)) < 1e-3
    gap = plane_margin((-3, 4, -1), S8) - 1 / math.sqrt(17)
    assert abs(gap) < 1e-9
    hinge = hinge_risk((-1, 2 / 3, -1 / 3), S10)
    assert abs(hinge - 11 / 30) < 1e-9
    w_soft = (-0.5, 0.5, -0.5)
    value = soft_svm_objective(w_soft, S10, 0.1)
    assert abs(value - 0.475) < 1e-9
    inside = at_or_inside_margin(w_soft, S10)
    assert inside == [0, 1, 2, 4, 5, 6, 8, 9]
    w = sgd_soft_svm(S10, 0.1, 200_000, random.Random(0))
    assert soft_svm_objective(w, S10, 0.1) < 0.48
```

1. `lift(x)` puts a constant 1 in front of the counts; every weight vector is \((b, w_1, w_2)\). `hinge_risk` averages the hinge loss over `S` and `soft_svm_objective` returns \(\lambda \|\mathbf w\|^2\) plus it, with the bias term inside the norm.
2. `sgd_soft_svm(S, lam, T, rng)` must minimize `soft_svm_objective` with parameter `lam`, so it runs the SGD procedure with \(2\lambda\). It returns \(\bar{\mathbf w}\) as a list.
3. `hard_svm` runs dual coordinate ascent on the lifted messages and returns `(w, alpha)`. `plane_margin(w, S)` is the smallest signed distance in the count plane, \(\min_i y_i(\langle \mathbf w, (1, \mathbf x_i) \rangle) / \|(w_1, w_2)\|\). `at_or_inside_margin` lists the indices with \(y_i \langle \mathbf w, (1, \mathbf x_i) \rangle \le 1\) (up to `tol`).
4. On `S10`, `at_or_inside_margin` returns message A (index 0), whose signed score is exactly 1, but the dual coefficient of A at the soft-SVM solution is 0. Explain why “on the margin” and “has a nonzero coefficient” are not the same thing.
5. Run `sgd_soft_svm` on `S10` for \(\lambda\) from 0.001 to 1 and plot the lifted margin \(1/\|\mathbf w\|\) (bias term included) and the hinge risk against \(\lambda\). Which way does each move, and why? Then plot the plane margin \(1/\|(w_1, w_2)\|\) as well. It does not move in one direction: explain why, remembering which program regularizes the bias term.

## Fix the bug

Time to ship. The production filter tracks a vocabulary of \(d = 50{,}000\) words, each a yes/no feature as in lesson 1: \(x_j = 1\) when word \(j\) appears in the message. Messages are short, and none contains more than 25 distinct vocabulary words, so every message vector has length at most \(\rho = 5\). The mail is almost separable: a few messages are mislabelled or genuinely ambiguous, like the consultation that looks exactly like a lunch invitation.

Operations has the same demand as in lesson 9: a written guarantee, computed before training, that fits the label budget of 50,000 labels. The guarantee they want is a certified expected 0-1 true risk of at most 0.18.

First, what does the VC route offer? Homogeneous halfspaces over 50,000 words have VC dimension 50,000, and the fundamental theorem of lesson 6 says that with far fewer than \(d/\varepsilon\) labels, no learner can promise accuracy \(\varepsilon\) on every distribution. Fifty thousand labels is a single multiple of \(d\), before any accuracy factor. That route is closed.

The margin route is open, because the soft-SVM theorem has no \(d\) in it. In the lab below, the team has measured three candidate comparison separators on a large pool of labelled mail: a wide one (norm \(B = \sqrt2\), hinge risk 0.12), a middle one (\(B = 2\), hinge risk 0.04) and a tight one (\(B = 4\), hinge risk 0.01). You choose a comparator, the regularization parameter \(\lambda\) and the number of labels \(m\), and the lab computes the certificate

\[ L_{\mathcal D}^{\text{hinge}}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}. \]

Only a certificate at most 0.18, within the budget, ships the filter. Then the lab trains soft-SVM by SGD with your \(\lambda\) on your \(m\) labels and shows its error on fresh mail, but the training run does not count: what ships is the certificate, because tomorrow’s training sequence will be different. Before you touch the sliders, work out which comparator can possibly make it, and roughly where \(\lambda\) must be.

[[lab-act2]]

## What people get wrong

**“Every consistent halfspace is as good as any other.”** They all have empirical risk 0, but they are not equally robust, and they do not carry the same guarantee. The Perceptron’s line on our eight messages left 0.243; hard-SVM’s left 1.342. The three keep messages that a one-word nudge sends to junk under the Perceptron’s line all stay keep under hard-SVM.

**“A bigger margin is always better, so rescale the data.”** Multiplying every message by 10 multiplies the margin by 10 and changes nothing real. The meaningful quantity is \(\rho/\gamma\), or \(\rho B\), which scaling leaves alone. That is why the radius is part of every margin bound.

**“The margin bound beats the VC bound.”** It scales better when \(\rho^2 B^2\) is much smaller than \(d\), as for short documents over a large dictionary. It can be worse on mail that is separable only with a narrow margin, and the hinge loss in its approximation term can exceed the 0-1 error. Preferring a large margin is an inductive bias, with the usual price.

**“Every message on the margin pulls on the separator.”** The weight vector is a combination of messages whose constraints are tight (for soft-SVM, messages on or inside the margin), but being on the margin does not force a nonzero coefficient. In the ten-message soft-SVM example, message A sits exactly at score 1, yet its dual coefficient is 0: removing it would change nothing.

**“Soft-SVM with a tiny \(\lambda\) is hard-SVM.”** On some samples it is, but no single \(\lambda\) works for every separable sample. Whether the two always agree for one well-chosen \(\lambda\) is one of this lesson’s problems, and the answer may surprise you.

## Recap and the next bug

1. The **margin** of a separating hyperplane is the distance from it to the closest training message; for a unit \(\mathbf w\) that distance is \(|\langle \mathbf w, \mathbf x \rangle + b|\). A large margin means the separator survives perturbations of every message.
2. **Hard-SVM** returns the separating halfspace with the largest margin. It is the quadratic program “minimize \(\|\mathbf w\|^2\) subject to every signed score being at least 1”, and the largest margin is \(1/\|\mathbf w_0\|\).
3. Margins only mean something relative to the radius \(\rho\) of the messages. If the distribution is separable with a \((\gamma, \rho)\)-margin, hard-SVM needs a number of examples that grows with \((\rho/\gamma)^2\), with no dependence on the dimension.
4. **Soft-SVM** allows violations through slack variables, and is exactly regularized loss minimization with the **hinge loss**. Because the hinge loss is convex and \(\rho\)-Lipschitz, its expected 0-1 true risk is at most the best hinge risk of norm \(B\) plus \(\sqrt{8\rho^2 B^2/m}\), again with no \(d\).
5. The hard-SVM weight vector is a combination of the **support vectors**, the messages on the margin. The **dual problem** finds the coefficients and involves the messages only through their inner products.
6. Soft-SVM is solved by SGD in \(O(Td)\) time: a Perceptron-like update \(\boldsymbol\theta \leftarrow \boldsymbol\theta + y_i \mathbf x_i\) whenever the signed score is below 1, with weights \(\boldsymbol\theta/(\lambda t)\), averaged.

The filter ships with its certificate, and the facilities team gets its lunch invitations. Then the data team brings you something unsettling. They took a month of junk your users labelled and looked at the words, and junk is often marked not by single words but by combinations: “free” alone is innocent, “meeting” alone is innocent, but “free” together with “click” and without “meeting” is almost always junk. No halfspace on word counts can express “this word and that word, but not that one”. The obvious repair is to add a feature for every pair of words. But with 50,000 words that is about 1.25 billion pairs, every message becomes a vector in a space of over a billion dimensions, and just writing one down is out of the question. The margin bound does not care about the dimension, so statistically you could afford that space. But can a linear method work in that space without ever building it? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. A colleague multiplies every lifted message vector (including the constant 1) by 10, “to make the margin bigger”. For homogeneous hard-SVM on our eight messages, what happens to the margin, to the radius \(\rho\), to the hypothesis returned (as a labelling of messages), and to the bound of the hard-SVM sample-complexity theorem? Then explain why the definition of separability with a \((\gamma, \rho)\)-margin needs \(\rho\) at all.
2. In the proof of the support-vector theorem, the vector \(\mathbf w_t = \mathbf w_0 - t\mathbf r\) must stay feasible. Why must \(t\) be small, and where exactly does the proof use the fact that there are finitely many examples? Why does the proof need \(t \le 1\) rather than any positive \(t\)? And why can the set of support vectors never be empty?
3. Messages have radius \(\rho = 5\), and some separator \(\mathbf u\) of norm \(B = 2\) has hinge risk 0.04. With \(m = 20{,}000\) labels, which \(\lambda\) does the soft-SVM theorem tell you to use, and what expected 0-1 true risk does it certify? How many labels would you need to certify 0.15?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (two forms of hard-SVM).** Show that, on a separable training sequence, the rule
\[ \operatorname*{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \; \min_{i \in [m]} |\langle \mathbf w, \mathbf x_i \rangle + b| \quad \text{subject to} \quad y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i \]
is the same as \(\operatorname{argmax}_{(\mathbf w, b) : \|\mathbf w\| = 1} \min_{i \in [m]} y_i (\langle \mathbf w, \mathbf x_i \rangle + b)\). Start from the set \(G = \{(\mathbf w, b) : y_i (\langle \mathbf w, \mathbf x_i \rangle + b) > 0 \text{ for all } i\}\).
- **Problem 2 (hard versus soft).** Prove or refute: there is a \(\lambda > 0\) such that for every sample \(S\) of \(m > 1\) examples that is separable by homogeneous halfspaces, hard-SVM and soft-SVM with parameter \(\lambda\) return exactly the same weight vector.
- **Problem 3 (weak duality).** Prove that for any function \(f\) of two variables \(x \in \mathcal X\) and \(y \in \mathcal Y\),
\[ \min_{x \in \mathcal X} \max_{y \in \mathcal Y} f(x, y) \ge \max_{y \in \mathcal Y} \min_{x \in \mathcal X} f(x, y). \]
