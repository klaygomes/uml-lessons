# Kernel Methods

## The junk that hides in pairs

The filter still runs the way lesson 15 shipped it. Every message becomes a vector of yes/no word features over the 50,000-word vocabulary, the labels are \(+1\) for junk and \(-1\) for keep, and soft-SVM finds a halfspace with a large margin. It weighs the evidence word by word, it does not care how large the vocabulary is, and the margin bound told you, before shipping, roughly how many labels it needed.

But the data team’s finding from the end of lesson 15, that junk is often marked by combinations of words, has stopped being a curiosity. A wave of phishing arrives, and the filter lets almost all of it through.

The new messages say things like “You are a winner, claim your reward” or “Verify your password today”. Look at the words one at a time and none of them is suspicious. The facilities team announces the “winner” of the office bake-off every month. People “claim” expenses all the time. IT asks staff to “verify” their laptops, and the helpdesk resets “password”s daily. What marks the junk is not any single word but a **combination**: “winner” together with “claim”, or “verify” together with “password”.

So what? A halfspace adds up evidence; surely it can add up two words. Let’s try. Take a vocabulary of five words, winner, claim, verify, password and lunch, and four messages that all mention lunch.

| message | winner | claim | verify | password | lunch | label |
|---|---|---|---|---|---|---|
| A | 1 | 1 | 0 | 0 | 1 | junk |
| B | 0 | 0 | 1 | 1 | 1 | junk |
| C | 1 | 0 | 1 | 0 | 1 | keep |
| D | 0 | 1 | 0 | 1 | 1 | keep |

Recall lesson 9’s halfspace: a weight vector \(\mathbf w\) with one weight per word, a bias term \(b\), and the score \(\langle \mathbf w, \mathbf x \rangle + b\), where the inner product \(\langle \mathbf w, \mathbf x \rangle = \sum_j w_j x_j\) multiplies each weight by its word’s count and adds. The message is called junk when the score is positive. Before you read on, try to find \(\mathbf w\) and \(b\) with \(\langle \mathbf w, \mathbf x \rangle + b > 0\) on A and B and \(\le 0\) on C and D. Give winner a big weight and C becomes junk. Give claim a big weight and D does. You will run out of ideas quickly, and for a good reason.

Add the two junk vectors, and add the two keep vectors:

```python
words = ["winner", "claim", "verify", "password", "lunch"]

def vec(*ws):
    return [1 if w in ws else 0 for w in words]

junk = [vec("winner", "claim", "lunch"),
        vec("verify", "password", "lunch")]
keep = [vec("winner", "verify", "lunch"),
        vec("claim", "password", "lunch")]
print([sum(col) for col in zip(*junk)])
print([sum(col) for col in zip(*keep)])
```

It prints `[1, 1, 1, 1, 2]` twice. The two pairs have the same sum. Now suppose some halfspace separated them. Adding its two inequalities on the junk messages gives \(\langle \mathbf w, \mathbf x_A + \mathbf x_B \rangle + 2b > 0\), and adding its two inequalities on the keep messages gives \(\langle \mathbf w, \mathbf x_C + \mathbf x_D \rangle + 2b \le 0\). The left-hand sides are the same number, because \(\mathbf x_A + \mathbf x_B = \mathbf x_C + \mathbf x_D\). A number cannot be both positive and at most zero, so no halfspace on these features separates the four messages. It is not that soft-SVM is searching badly: the hypothesis class itself cannot express the rule. In the language of lesson 5, the approximation error (the error of the best hypothesis in the class) is the problem, and no number of labels fixes that.

Your colleague has the obvious fix. If junk is marked by pairs of words, give the filter pairs of words as features: one new coordinate for every pair, equal to 1 when both words appear. And since next month’s junk might be marked by three words, why not triples too? Then a halfspace over the new coordinates can say “winner and claim” directly.

It is a good idea, and it is the idea of this whole lesson. But count the coordinates first. With a vocabulary of \(n = 50{,}000\) words, the number of products of at most \(k\) words, including the empty product (a constant), is \(\binom{n+k}{k}\). Read \(\binom{n+k}{k}\) as “\(n + k\) choose \(k\)”, the number of ways to pick \(k\) things out of \(n + k\); a product of at most \(k\) words is \(k\) slots, each filled by a word or left blank, order ignored. So: **the number of word products of size at most \(k\) is \(n + k\) choose \(k\).** With the five words of the table and \(k = 2\), that is \(\binom{7}{2} = 21\): the constant, 5 single words, 5 squares and 10 pairs. The snippet mirrors it with `comb(n + k, k)` and charges 8 bytes per weight:

```python
from math import comb

n = 50_000
for k in (2, 3):
    dim = comb(n + k, k)
    print(k, f"{dim:.3e}", f"{8 * dim / 1e9:,.1f} GB")
```

It prints `2 1.250e+09 10.0 GB` and `3 2.084e+13 166,686.7 GB`. With pairs, the weight vector alone has the 1.25 billion coordinates the data team warned about, 10 gigabytes that every SGD step would have to touch. With triples it has 21 trillion, about 167 terabytes: the team cannot even store it.

So here is the bug. The features that would make the junk separable are too many to write down. The fix is one of the most elegant ideas in the subject: use those features, and never build them.

## Lifting points so a line can split them

Let’s start smaller than mail, on the real line, where you can see everything. Take the 21 points \(-10, -9, \dots, 9, 10\), and label a point \(+1\) when \(|x| > 2\) and \(-1\) otherwise. The positive points sit on both sides, the negative ones in the middle. A halfspace on the line is a threshold, “positive to the right of \(t\)” or “positive to the left of \(t\)”, so no halfspace gets this right.

Now send every point \(x\) to the pair \(\psi(x) = (x, x^2)\) in the plane. The points land on a parabola. The middle points, with \(|x| \le 2\), have \(x^2 \le 4\), and the outer ones have \(x^2 \ge 9\). A horizontal line between heights 4 and 9 splits them: take \(\mathbf w = (0, 1)\) and \(b = 5\), and predict \(h(x) = \operatorname{sign}(\langle \mathbf w, \psi(x) \rangle - b)\). Here \(\operatorname{sign}(s)\) is \(+1\) when \(s > 0\) and \(-1\) otherwise, and \(\langle \mathbf w, \psi(x) \rangle = 0 \cdot x + 1 \cdot x^2\) is just the height \(x^2\). In plain words: **a point is positive when its height on the parabola is above 5.** At \(x = 3\) the height is 9 and the score \(9 - 5 = 4\), positive; at \(x = 2\) the score is \(4 - 5 = -1\), negative. In the snippet, `h` is this formula term by term.

[[diagram:lift]]

```python
xs = range(-10, 11)
label = {x: 1 if abs(x) > 2 else -1 for x in xs}

def psi(x):
    return (x, x * x)

w, b = (0, 1), 5

def h(x):
    score = sum(wi * pi for wi, pi in zip(w, psi(x))) - b
    return 1 if score > 0 else -1

print(all(h(x) == label[x] for x in xs))
print(min(abs(psi(x)[1] - b) for x in xs))
```

It prints `True` and `1`: every point is labelled correctly, and none sits closer than 1 to the line \(x^2 = 5\). The classifier is linear in the plane and quite nonlinear on the line.

That move has a name. A map \(\psi: \mathcal X \to \mathcal F\) (read the arrow as “takes a point of \(\mathcal X\), returns a point of \(\mathcal F\)”) from the domain set, lesson 1’s set of all possible inputs, into a vector space is an **embedding**, and its range \(\mathcal F\) is the **feature space**. Each coordinate of \(\psi(x)\) is a new feature computed from the message, such as “winner and claim both appear”. The paradigm has four steps.

1. Given the domain set \(\mathcal X\) and the task, choose an embedding \(\psi: \mathcal X \to \mathcal F\). Usually \(\mathcal F = \mathbb R^n\) for some \(n\), but we will soon need spaces of infinite dimension too.
2. Given the training sequence \(S = ((x_1, y_1), \dots, (x_m, y_m))\), form the **image sequence** \(\hat S = ((\psi(x_1), y_1), \dots, (\psi(x_m), y_m))\).
3. Train a halfspace \(h\) on \(\hat S\).
4. Predict the label of a new \(x\) as \(h(\psi(x))\).

Does learning in the feature space measure the right thing? Yes, and it is worth one line of care. A distribution \(\mathcal D\) over \(\mathcal X \times \mathcal Y\) induces an **image distribution** \(\mathcal D_\psi\) over \(\mathcal F \times \mathcal Y\): draw \((x, y) \sim \mathcal D\) and output \((\psi(x), y)\). Formally \(\mathcal D_\psi(A) = \mathcal D(\{(x, y) : (\psi(x), y) \in A\})\) for every (measurable) set \(A\). Then for every hypothesis \(h\) on the feature space,

\[ L_{\mathcal D_\psi}(h) = \mathbb P_{(x, y) \sim \mathcal D}\left[ h(\psi(x)) \ne y \right] = L_{\mathcal D}(h \circ \psi), \]

because both sides are the probability of the same event, just described in two spaces. Read it from the inside out.

- In \(\mathcal D_\psi(A)\), the braces \(\{(x, y) : \dots\}\) collect the pairs such that their image lands in \(A\), so a set of feature-space pairs gets the probability its original pairs had.
- \((x, y) \sim \mathcal D\) reads “a message and its label drawn from \(\mathcal D\)”, and \(\mathbb P[\dots]\) is the probability of the event in the brackets: embed the message, let \(h\) label it, and get it wrong.
- \(L_{\mathcal D_\psi}(h)\) is lesson 1’s true risk, scored in feature space; \(h \circ \psi\), “\(h\) after \(\psi\)”, is the filter that embeds first and then applies \(h\), and \(L_{\mathcal D}(h \circ \psi)\) is its true risk on messages.

In plain words: **the halfspace’s error rate in feature space is the filter’s error rate on messages.** Make the 21 points of the line equally likely and lower the line to height 3. Then \(x = \pm 2\), at height 4, are called positive and are wrong, and both sides equal \(2/21\). The snippet builds \(\mathcal D_\psi\) by pushing each probability to its image, then scores both sides:

```python
from fractions import Fraction

xs = range(-10, 11)
D = {(x, 1 if abs(x) > 2 else -1): Fraction(1, 21) for x in xs}
psi = lambda x: (x, x * x)
h = lambda z: 1 if z[1] - 3 > 0 else -1
D_psi = {}
for (x, y), p in D.items():
    D_psi[(psi(x), y)] = D_psi.get((psi(x), y), 0) + p
lhs = sum(p for (z, y), p in D_psi.items() if h(z) != y)
rhs = sum(p for (x, y), p in D.items() if h(psi(x)) != y)
print(lhs, rhs)
```

It prints `2/21 2/21`. Nothing is lost in translation.

You have seen one embedding before. In lesson 9, polynomial regression was least squares on \(\psi(x) = (1, x, x^2, \dots, x^k)\): a polynomial of degree \(k\) in \(x\) is a linear function of \(\psi(x)\). The same works with many variables. A polynomial of degree \(k\) from \(\mathbb R^n\) to \(\mathbb R\) is a weighted sum of monomials, products of at most \(k\) of the coordinates:

\[ p(\mathbf x) = \sum_{r=0}^{k} \; \sum_{J \in [n]^r} w_J \prod_{i=1}^{r} x_{J_i}. \]

Read it from the inside out.

- \([n]\) is the list of positions \(\{1, \dots, n\}\), and \([n]^r\) is the set of all lists \(J = (J_1, \dots, J_r)\) of \(r\) positions, repeats allowed; for \(r = 0\) it holds only the empty list.
- \(x_{J_i}\) is the coordinate at the \(i\)-th position of \(J\), and the big pi \(\prod_{i=1}^{r}\) multiplies the \(r\) of them; the empty product is 1, the constant term.
- \(w_J\) is the weight of that monomial, and the two sums add the weighted monomials over every length \(r\) from 0 to \(k\) and every list of that length.

In plain words: **a polynomial of degree \(k\) is a weighted sum of products of at most \(k\) coordinates.** With two words, winner (1) and claim (2), \(k = 2\), weight \(-1\) on the empty list and 1 on each of \((1, 2)\) and \((2, 1)\), you get \(p(\mathbf x) = 2 x_1 x_2 - 1\): 1 on a message with both words, \(-1\) on the other three kinds. The two sums are the two `for` clauses, and `prod` is the big pi:

```python
from itertools import product
from math import prod

n, k = 2, 2
w = {(): -1, (1, 2): 1, (2, 1): 1}

def p(x):
    return sum(w.get(J, 0) * prod(x[j - 1] for j in J)
               for r in range(k + 1)
               for J in product(range(1, n + 1), repeat=r))

print([p(x) for x in [(1, 1), (1, 0), (0, 1), (0, 0)]])
```

It prints `[1, -1, -1, -1]`. Give \(\psi(\mathbf x)\) one coordinate for each index tuple \(J\), equal to the monomial \(\prod_i x_{J_i}\), and then \(p(\mathbf x) = \langle \mathbf w, \psi(\mathbf x) \rangle\). Halfspaces over this \(\psi\) are exactly the polynomial threshold rules of degree \(k\), and that includes “junk when winner and claim both appear”.

So the plan is sound in principle. In practice it raises two problems, and they are the same two that lesson 15 opened with.

The first is **statistical**. Halfspaces in \(\mathbb R^n\) have VC dimension \(n + 1\) (lesson 6’s largest number of points a class labels in every possible way, computed for halfspaces in lesson 9), so a feature space with trillions of coordinates seems to demand trillions of labels. Lesson 15 already answered this one: a large margin, or equivalently a small norm \(\|\mathbf w\| = \sqrt{\sum_j w_j^2}\), the length of the weight vector, gives a sample complexity that does not depend on the dimension at all. We will put numbers on it near the end of this lesson.

The second is **computational**: computing in the feature space is too expensive, and even writing down \(\mathbf w\) can be impossible. That is the bug from the opening, and the rest of this lesson takes it apart.

## A worked example: pairs make the mail separable

Before we fight the cost, let’s make sure pairs really do the job. Go back to the mail. Number the words so that winner is 1, claim is 2, verify is 3 and password is 4, and write \(x_j \in \{0, 1\}\) for “word \(j\) appears”. Let the subject line of every message carry exactly three words from a vocabulary of \(d\) words. (The team’s preprocessor reduces each subject line to its three key words; that keeps the numbers small enough to check by hand, and nothing below depends on \(d\).) The junk rule is

\[ y = +1 \iff (x_1 = 1 \text{ and } x_2 = 1) \text{ or } (x_3 = 1 \text{ and } x_4 = 1). \]

Here \(\iff\) reads “exactly when”, and each bracket asks whether both words of a pair appear: **junk exactly when winner and claim both appear, or verify and password both do.** “winner claim lunch” is junk; “winner verify lunch” is keep. The function `y` in the snippet below is this rule.

Use the degree-2 embedding. Prepend a constant \(x_0 = 1\), and give \(\psi(\mathbf x)\) one coordinate for each ordered pair \(J = (J_1, J_2) \in \{0, 1, \dots, d\}^2\), equal to \(x_{J_1} x_{J_2}\). So the coordinate \((0, 0)\) is the constant 1, the coordinates \((0, j)\) and \((j, 0)\) are \(x_j\), and the coordinate \((1, 2)\) is \(x_1 x_2\), “winner and claim”.

Now write down a weight vector \(\mathbf u\) by hand: put 1 on the four coordinates \((1, 2), (2, 1), (3, 4), (4, 3)\), put \(-1\) on \((0, 0)\), and 0 everywhere else. Then

\[ \langle \mathbf u, \psi(\mathbf x) \rangle = 2 x_1 x_2 + 2 x_3 x_4 - 1. \]

Each pair product sits in two coordinates, \((1, 2)\) and \((2, 1)\), which is where the 2s come from, and the \(-1\) is the weight on the constant. In plain words: **the score is 1 or 3 when a junk pair is present and \(-1\) otherwise.** For “winner claim lunch”, \(x_1 x_2 = 1\) and \(x_3 x_4 = 0\), so the score is \(2 - 1 = 1\). On a junk message at least one product is 1, so the score is at least 1. On a keep message both products are 0, so the score is exactly \(-1\). Every message gets \(y \langle \mathbf u, \psi(\mathbf x) \rangle \ge 1\): the label times the score is at least 1, so the message is on its correct side with score to spare. That is lesson 9’s rescaled separability condition.

Here is the check over all 56 three-word subject lines from an 8-word vocabulary, building \(\psi\) explicitly:

```python
import itertools, math

d, s, k = 8, 3, 2
coords = list(itertools.product(range(d + 1), repeat=k))
u = {J: 0 for J in coords}
u[(0, 0)] = -1
for J in [(1, 2), (2, 1), (3, 4), (4, 3)]:
    u[J] = 1

def score(x):
    z = (1,) + x
    return sum(u[J] * math.prod(z[j] for j in J) for J in coords)

def y(x):
    return 1 if (x[0] and x[1]) or (x[2] and x[3]) else -1

msgs = [tuple(int(i in c) for i in range(d))
        for c in itertools.combinations(range(d), s)]
print(len(msgs), min(y(x) * score(x) for x in msgs))
print(sum(v * v for v in u.values()), (1 + s) ** k)
```

Here `score` builds \(\langle \mathbf u, \psi(\mathbf x) \rangle\) coordinate by coordinate, and the last line prints \(\|\mathbf u\|^2\), the sum of the squared weights. It prints `56 1` and `5 16`. All 56 messages are on the right side with \(y \langle \mathbf u, \psi(\mathbf x) \rangle \ge 1\), and \(\|\mathbf u\|^2 = 5\). Remember the 5. (The 16 will matter later, too.) Notice also what the vector \(\mathbf u\) did not need: weights on the 81 coordinates of this small example, or on the 2.5 billion of a 50,000-word vocabulary. Only five of its coordinates are nonzero. The learner, however, does not know which five, and that is where the cost comes from.

## Where the weight vector lives

Here is a question that sounds idle and turns out to be the key. When soft-SVM finishes, where is its weight vector? It lives in the feature space, which may have trillions of dimensions. But is it anywhere in particular?

Think about the training sequence. The learner only ever sees \(m\) vectors, \(\psi(x_1), \dots, \psi(x_m)\). A direction in feature space that is perpendicular to all of them is invisible to the training data: moving \(\mathbf w\) along it changes no score \(\langle \mathbf w, \psi(x_i) \rangle\). It only makes \(\mathbf w\) longer. And soft-SVM dislikes long vectors. So you would expect it never to bother with such directions, and to keep \(\mathbf w\) inside the span of the training vectors. That hunch is a theorem.

First, the setting. Lesson 15’s learners all minimize something of the form

\[ \min_{\mathbf w} \; f\big(\langle \mathbf w, \psi(x_1) \rangle, \dots, \langle \mathbf w, \psi(x_m) \rangle\big) + R(\|\mathbf w\|), \]

where \(f: \mathbb R^m \to \mathbb R\) is any function at all and \(R: \mathbb R_+ \to \mathbb R\) is **nondecreasing**: \(a \le a'\) implies \(R(a) \le R(a')\). Read it from the inside out.

- \(\langle \mathbf w, \psi(x_i) \rangle\) is the score of training message \(i\) in feature space, one for each of the \(m\) examples.
- \(f\) takes those \(m\) scores (it may also look at the labels, which are fixed) and returns one number, the data term; that is what \(f: \mathbb R^m \to \mathbb R\) says.
- \(R\) turns the length \(\|\mathbf w\|\) into a penalty, the regularizer of lesson 13; \(\mathbb R_+\), the nonnegative numbers, is where lengths live.
- \(\min_{\mathbf w}\) asks for the smallest total over all weight vectors, and an optimal solution is a \(\mathbf w\) that attains it.

In plain words: **score the training messages, add a penalty that grows with the length of \(\mathbf w\), and make the total as small as possible.** Soft-SVM for homogeneous halfspaces is the case \(R(a) = \lambda a^2\) and \(f(a_1, \dots, a_m) = \frac{1}{m} \sum_i \max\{0, 1 - y_i a_i\}\), the average hinge loss of lessons 12 and 15, which charges nothing when the label times the score is at least 1 and charges the shortfall otherwise. On the line, with \(\psi(x) = (1, x, x^2)\), the eleven points \(-5, \dots, 5\) and \(\lambda = 0.1\), the weight vector \((-5, 0, 1)\) scores \(x^2 - 5\) and clears every point by at least 1, so \(f = 0\), but \(R = 0.1 \times 26 = 2.6\). Halve it to \((-2.5, 0, 0.5)\): the points \(\pm 2\) now score \(-0.5\) and pay 0.5 each, so \(f = 1/11\), while \(R\) drops to 0.65. The shorter vector wins:

```python
S = [(x, 1 if abs(x) > 2 else -1) for x in range(-5, 6)]
psi = lambda x: (1, x, x * x)
dot = lambda a, b: sum(i * j for i, j in zip(a, b))
lam = 0.1

def objective(w):
    a = [dot(w, psi(x)) for x, _ in S]
    hinge = [max(0, 1 - y * ai) for (_, y), ai in zip(S, a)]
    f, R = sum(hinge) / len(S), lam * dot(w, w)
    return round(f, 4), round(R, 4), round(f + R, 4)

print(objective((-5, 0, 1)), objective((-2.5, 0, 0.5)))
```

It prints `(0.0, 2.6, 2.6) (0.0909, 0.65, 0.7409)`: data term, penalty and total for each. Hard-SVM is the case \(R(a) = a^2\) and \(f(a_1, \dots, a_m) = 0\) if there is a \(b\) with \(y_i(a_i + b) \ge 1\) for every \(i\), and \(f = \infty\) otherwise.

We also need the space to have a geometry. A **Hilbert space** is a vector space with an inner product \(\langle \cdot, \cdot \rangle\) (symmetric, linear in each argument, and \(\langle \mathbf v, \mathbf v \rangle > 0\) for \(\mathbf v \ne 0\)) that is also **complete**: every Cauchy sequence (one whose terms eventually stay as close to each other as you like) converges to a point of the space. The norm is \(\|\mathbf v\| = \sqrt{\langle \mathbf v, \mathbf v \rangle}\), the length; in \(\mathbb R^3\), \(\|(1, 0, 2)\| = \sqrt{1 + 0 + 4} = \sqrt 5\). Every \(\mathbb R^n\) with the usual dot product is a Hilbert space, and so are some spaces of infinite dimension that we will meet shortly. The one fact we need from the geometry is the orthogonal decomposition onto a finite-dimensional subspace, and we will prove it inline.

**Representer theorem.** Let \(\psi\) map \(\mathcal X\) into a Hilbert space, and suppose the problem above has an optimal solution. Then it has an optimal solution of the form \(\mathbf w = \sum_{i=1}^m \alpha_i \psi(x_i)\) for some \(\boldsymbol\alpha \in \mathbb R^m\), a list of \(m\) coefficients, one per training message. In plain words: **some best weight vector is a weighted sum of the embedded training messages.** In the \(\mathbb R^3\) example after the proof, it is \(2\psi_1 - \psi_2\), so \(\boldsymbol\alpha = (2, -1)\).

*Proof.* Let \(\mathbf w^\star\) be an optimal solution, and let \(M\) be the span of \(\psi(x_1), \dots, \psi(x_m)\), the set of all their weighted sums, a subspace of dimension \(r \le m\).

Step one splits \(\mathbf w^\star\) into a part inside \(M\) and a part perpendicular to it. Run Gram–Schmidt on \(\psi(x_1), \dots, \psi(x_m)\), dropping any vector that depends on the earlier ones; this standard procedure gives an **orthonormal basis** \(\mathbf e_1, \dots, \mathbf e_r\) of \(M\), vectors of length 1, each perpendicular to the others, whose weighted sums are exactly \(M\). Set \(\mathbf p = \sum_{k=1}^r \langle \mathbf w^\star, \mathbf e_k \rangle \mathbf e_k\), the shadow of \(\mathbf w^\star\) on \(M\) (along each basis direction, step as far as \(\mathbf w^\star\) reaches along it), and \(\mathbf u = \mathbf w^\star - \mathbf p\). Then for each \(k\), \(\langle \mathbf u, \mathbf e_k \rangle = \langle \mathbf w^\star, \mathbf e_k \rangle - \langle \mathbf w^\star, \mathbf e_k \rangle = 0\), so \(\mathbf u\) is perpendicular to every basis vector, hence to every vector in \(M\), and in particular \(\langle \mathbf u, \psi(x_i) \rangle = 0\) for all \(i\). And \(\mathbf p \in M\), so \(\mathbf p = \sum_{i} \alpha_i \psi(x_i)\) for some coefficients \(\alpha_i\). We have written

\[ \mathbf w^\star = \sum_{i=1}^m \alpha_i \psi(x_i) + \mathbf u, \qquad \langle \mathbf u, \psi(x_i) \rangle = 0 \text{ for all } i. \]

In plain words: **the best weight vector is a part built from the training messages plus a part \(\mathbf u\) that no training message can see.** In the \(\mathbb R^3\) example below, the first part is \(2\psi_1 - \psi_2\) and \(\mathbf u\) is a cross product.

Step two compares \(\mathbf w = \mathbf w^\star - \mathbf u = \mathbf p\) with \(\mathbf w^\star\). The regularizer first. Since \(\langle \mathbf p, \mathbf u \rangle = 0\),

\[ \|\mathbf w^\star\|^2 = \langle \mathbf p + \mathbf u, \mathbf p + \mathbf u \rangle = \|\mathbf p\|^2 + 2 \langle \mathbf p, \mathbf u \rangle + \|\mathbf u\|^2 = \|\mathbf w\|^2 + \|\mathbf u\|^2, \]

expanding the inner product of a sum term by term and dropping the cross term, which is 0. In plain words: **removing a perpendicular part lowers the squared length by exactly that part’s squared length** (the \(20 = 14 + 6\) printed below). So \(\|\mathbf w\| \le \|\mathbf w^\star\|\), and because \(R\) is nondecreasing, \(R(\|\mathbf w\|) \le R(\|\mathbf w^\star\|)\).

Step three handles the data term. For every \(i\), by linearity of the inner product,

\[ \langle \mathbf w, \psi(x_i) \rangle = \langle \mathbf w^\star - \mathbf u, \psi(x_i) \rangle = \langle \mathbf w^\star, \psi(x_i) \rangle - 0, \]

so **removing \(\mathbf u\) changes no training score** (the `[8, 2]` printed twice below): the \(m\) arguments of \(f\) are exactly the same at \(\mathbf w\) and at \(\mathbf w^\star\), and so is the value of \(f\).

Adding the two parts, the objective at \(\mathbf w\) is at most the objective at \(\mathbf w^\star\). Since \(\mathbf w^\star\) is optimal, the objective at \(\mathbf w\) cannot be smaller either, so \(\mathbf w\) is also optimal, and it has the promised form. \(\blacksquare\)

[[diagram:representer]]

Notice the two places the proof leaned on its assumptions. The data term must see \(\mathbf w\) only through the numbers \(\langle \mathbf w, \psi(x_i) \rangle\), or removing \(\mathbf u\) might change it. And \(R\) must not reward length, or removing \(\mathbf u\) might cost something. If \(R\) is strictly increasing, the argument says more: any optimal solution with \(\mathbf u \ne 0\) could be strictly improved, so every optimal solution lies in the span.

Small enough to watch? In \(\mathbb R^3\), take two training vectors \(\psi_1 = (1, 0, 2)\) and \(\psi_2 = (0, 1, 1)\). Their cross product is perpendicular to both, so it plays \(\mathbf u\). Build \(\mathbf w^\star = 2\psi_1 - \psi_2 + \mathbf u\) and strip \(\mathbf u\) off:

```python
def dot(a, b):
    return sum(i * j for i, j in zip(a, b))

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])

psi1, psi2 = (1, 0, 2), (0, 1, 1)
u = cross(psi1, psi2)
w_star = [2 * a - b + c for a, b, c in zip(psi1, psi2, u)]
w = [a - c for a, c in zip(w_star, u)]
print([dot(w_star, p) for p in (psi1, psi2)])
print([dot(w, p) for p in (psi1, psi2)])
print(dot(w_star, w_star), dot(w, w), dot(u, u))
```

It prints `[8, 2]` twice, then `20 14 6`. The scores on the training vectors did not move, the squared norm fell from 20 to 14, and the 6 that disappeared is exactly \(\|\mathbf u\|^2\), Pythagoras at work.

## The kernel trick

The representer theorem lets us stop searching over \(\mathbf w\), a vector with trillions of coordinates, and search over \(\boldsymbol\alpha\), a vector with \(m\) coordinates. Let’s rewrite everything in terms of \(\boldsymbol\alpha\) and see what the computer actually needs.

Write \(\mathbf w = \sum_{j=1}^m \alpha_j \psi(x_j)\). Then, by linearity, each score is

\[ \langle \mathbf w, \psi(x_i) \rangle = \Big\langle \sum_j \alpha_j \psi(x_j), \psi(x_i) \Big\rangle = \sum_{j=1}^m \alpha_j \langle \psi(x_j), \psi(x_i) \rangle, \]

because the inner product passes through a weighted sum. In plain words: **the score of message \(i\) is a weighted sum of its inner products with the training messages.** The squared norm is

\[ \|\mathbf w\|^2 = \Big\langle \sum_i \alpha_i \psi(x_i), \sum_j \alpha_j \psi(x_j) \Big\rangle = \sum_{i,j=1}^m \alpha_i \alpha_j \langle \psi(x_i), \psi(x_j) \rangle. \]

Here \(\sum_{i,j=1}^m\) runs over every pair of positions: **the squared length is a weighted sum over all pairs of training messages.** Look at what survived. Neither formula mentions a coordinate of \(\psi\). Both use only inner products between pairs of embedded messages. That function deserves a name. Given an embedding \(\psi\) into a Hilbert space, its **kernel** is

\[ K(x, x') = \langle \psi(x), \psi(x') \rangle. \]

You can read \(K\) as a similarity between messages: large when their embeddings point the same way. And for a training sequence, the \(m \times m\) matrix of all these similarities, \(G_{i,j} = K(x_i, x_j)\) in row \(i\) and column \(j\), is the **Gram matrix**.

With this notation, the general problem becomes a problem over \(\mathbb R^m\):

\[ \min_{\boldsymbol\alpha \in \mathbb R^m} \; f\Big( \sum_{j} \alpha_j K(x_j, x_1), \dots, \sum_j \alpha_j K(x_j, x_m) \Big) + R\Big( \sqrt{\textstyle\sum_{i,j} \alpha_i \alpha_j K(x_j, x_i)} \Big). \]

It is the old objective with each score replaced by its kernel sum and the length by the square root of the double sum, and \(\min_{\boldsymbol\alpha \in \mathbb R^m}\) now searches over \(m\) numbers: **the same problem, written with coefficients and kernel values instead of a weight vector.** To solve it, you never touch the feature space. You need the Gram matrix and nothing else. For soft-SVM in particular, \(\sum_j \alpha_j K(x_j, x_i)\) is the \(i\)-th entry of the vector \(G\boldsymbol\alpha\) (row \(i\) of \(G\) times \(\boldsymbol\alpha\), entry by entry, added up), and \(\sum_{i,j} \alpha_i \alpha_j G_{i,j} = \boldsymbol\alpha^\top G \boldsymbol\alpha\), where \(^\top\), the transpose, lays the column \(\boldsymbol\alpha\) on its side, so the problem reads

\[ \min_{\boldsymbol\alpha \in \mathbb R^m} \; \lambda\, \boldsymbol\alpha^\top G \boldsymbol\alpha + \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i (G\boldsymbol\alpha)_i\}. \]

In plain words: **soft-SVM’s penalty and hinge losses, all read off the Gram matrix.** The \(\mathbb R^3\) example below pushes numbers through it. This can be written as a quadratic program (a quadratic objective under linear constraints) and solved efficiently, and later in this lesson you will build an even simpler solver. Once \(\boldsymbol\alpha\) is learned, predicting the label of a new message also needs only the kernel:

\[ \langle \mathbf w, \psi(x) \rangle = \sum_{j=1}^m \alpha_j \langle \psi(x_j), \psi(x) \rangle = \sum_{j=1}^m \alpha_j K(x_j, x). \]

In plain words: **a new message is scored by its weighted similarity to the training messages.** Go back to the \(\mathbb R^3\) example, where \(\mathbf w = 2\psi_1 - \psi_2\), so \(\boldsymbol\alpha = (2, -1)\), and give \(\psi_1\) the label \(+1\) and \(\psi_2\) the label \(-1\). The Gram matrix has rows \((5, 2)\) and \((2, 2)\); \(G\boldsymbol\alpha = (8, 2)\), the scores from before; and \(\boldsymbol\alpha^\top G \boldsymbol\alpha = 14\), the squared norm from before, with no coordinate of \(\mathbf w\) in sight. With \(\lambda = 0.1\) the hinge losses are 0 and 3, so the soft-SVM objective is \(1.4 + 1.5 = 2.9\). A new embedded message \((1, 1, 1)\) scores \(2 \cdot 3 - 1 \cdot 2 = 4\):

```python
def dot(a, b):
    return sum(i * j for i, j in zip(a, b))

psis, ys, alpha = [(1, 0, 2), (0, 1, 1)], [1, -1], [2, -1]
G = [[dot(p, q) for q in psis] for p in psis]
Ga = [sum(a * g for a, g in zip(alpha, row)) for row in G]
aGa = sum(alpha[i] * alpha[j] * G[i][j]
          for i in range(2) for j in range(2))
hinge = [max(0, 1 - y * s) for y, s in zip(ys, Ga)]
print(G, Ga, aGa, round(0.1 * aGa + sum(hinge) / 2, 4))
psi_new = (1, 1, 1)
print(sum(a * dot(p, psi_new) for a, p in zip(alpha, psis)))
```

It prints `[[5, 2], [2, 2]] [8, 2] 14 2.9` and then `4`, which is also \(\langle (2, -1, 3), (1, 1, 1) \rangle\) computed the long way.

That is the **kernel trick**: run a halfspace learner in the feature space using only kernel values, never the embedded vectors. The weight vector is stored implicitly, as \(m\) coefficients plus the \(m\) training messages. The trick pays off exactly when the feature space is huge but \(K\) is cheap to compute. Is that ever the case? Constantly.

[[diagram:kernel-trick]]

## Polynomial kernels

The **polynomial kernel of degree \(k\)** is

\[ K(\mathbf x, \mathbf x') = (1 + \langle \mathbf x, \mathbf x' \rangle)^k. \]

Here \(\langle \mathbf x, \mathbf x' \rangle\) is the ordinary inner product of the two input vectors and \(k\) a whole-number power: **add 1 to the inner product and raise the result to the power \(k\).** For subject lines as 0/1 word vectors, the inner product counts shared words: “winner claim lunch” and “winner claim report” share 2, so the degree-2 kernel is \((1 + 2)^2 = 9\). Computing it costs one inner product in \(\mathbb R^n\), then a power: \(O(n)\) time, in lesson 8’s big-O notation, which names a growth rate (here, in proportion to \(n\)) and hides the constant. The claim is that it is the inner product of an embedding into all monomials of degree at most \(k\). Let’s prove it.

*Proof.* Set \(x_0 = x'_0 = 1\), so that \(1 + \langle \mathbf x, \mathbf x' \rangle = \sum_{j=0}^n x_j x'_j\). The power is a product of \(k\) copies of this sum:

\[ (1 + \langle \mathbf x, \mathbf x' \rangle)^k = \Big( \sum_{j=0}^n x_j x'_j \Big) \cdots \Big( \sum_{j=0}^n x_j x'_j \Big). \]

Multiply out. Each term of the expansion picks one index from each of the \(k\) factors, that is, a tuple \(J = (J_1, \dots, J_k) \in \{0, 1, \dots, n\}^k\), and contributes the product of what it picked:

\[ = \sum_{J \in \{0, \dots, n\}^k} \prod_{i=1}^k x_{J_i} x'_{J_i} = \sum_{J \in \{0, \dots, n\}^k} \Big( \prod_{i=1}^k x_{J_i} \Big) \Big( \prod_{i=1}^k x'_{J_i} \Big). \]

Here \(\{0, \dots, n\}^k\) is the set of all lists of \(k\) indices, and \(\prod_{i=1}^k\) multiplies over the \(k\) picks: **multiplying out gives one term per list of picks.** With \(n = 1\) and \(k = 2\) the lists are \((0,0), (0,1), (1,0), (1,1)\), and \((1 + x_1 x'_1)^2 = 1 + x_1 x'_1 + x_1 x'_1 + (x_1 x'_1)^2\). The last step only regroups the factors of \(\mathbf x\) and of \(\mathbf x'\). Now define \(\psi: \mathbb R^n \to \mathbb R^{(n+1)^k}\) with one coordinate per tuple \(J\), equal to \(\prod_{i=1}^k x_{J_i}\). The sum is exactly \(\langle \psi(\mathbf x), \psi(\mathbf x') \rangle\). \(\blacksquare\)

Because \(x_0 = 1\), a tuple that contains some zeros produces a monomial of lower degree, so \(\psi\) contains every monomial of degree at most \(k\) (several times, in different orders). A halfspace over \(\psi\) is therefore a polynomial threshold rule of degree \(k\) on the original space. The kernel costs \(O(n)\); the feature space has \((n+1)^k\) coordinates. The snippet’s `psi` builds one coordinate per list with `itertools.product` and multiplies its entries with `math.prod`, and `K` is the kernel formula as written:

```python
import itertools, math, random

def psi(x, k):
    z = (1,) + tuple(x)
    return [math.prod(z[j] for j in J)
            for J in itertools.product(range(len(z)), repeat=k)]

def K(x, xp, k):
    return (1 + sum(a * b for a, b in zip(x, xp))) ** k

rng = random.Random(0)
x = [rng.uniform(-1, 1) for _ in range(3)]
xp = [rng.uniform(-1, 1) for _ in range(3)]
for k in (2, 3):
    explicit = sum(a * b for a, b in zip(psi(x, k), psi(xp, k)))
    print(k, len(psi(x, k)), round(explicit, 6),
          round(K(x, xp, k), 6))
```

It prints `2 16 0.503674 0.503674` and `3 64 0.357457 0.357457`. For three input coordinates, the explicit embedding has 16 and then 64 coordinates, and the two computations agree to every printed digit. For the mail, with 50,000 words and \(k = 3\), the explicit route is the 167 terabytes from the opening, and the kernel is still one count of shared words.

## The Gaussian kernel

Here is a stranger one. On the real line, consider the embedding with one coordinate for every integer \(n \ge 0\):

\[ \psi(x)_n = \frac{1}{\sqrt{n!}} e^{-x^2/2} x^n. \]

Here \(n!\), “\(n\) factorial”, is \(1 \cdot 2 \cdots n\) (with \(0! = 1\)), \(e^{-x^2/2}\) is the exponential function at \(-x^2/2\), a factor shared by every coordinate, and \(x^n\) is the \(n\)-th power: **coordinate \(n\) is \(x^n\), shrunk by \(\sqrt{n!}\) and by the common factor \(e^{-x^2/2}\).** At \(x = 0.7\), coordinate 0 is \(e^{-0.245} \approx 0.783\) and coordinate 1 is 0.7 times that, about 0.548. That is infinitely many coordinates. Can you even take an inner product? Let’s try:

\[ \langle \psi(x), \psi(x') \rangle = \sum_{n=0}^\infty \frac{1}{\sqrt{n!}} e^{-x^2/2} x^n \cdot \frac{1}{\sqrt{n!}} e^{-x'^2/2} x'^n = e^{-\frac{x^2 + x'^2}{2}} \sum_{n=0}^\infty \frac{(x x')^n}{n!}. \]

The sum \(\sum_{n=0}^\infty\) runs over every coordinate, without end. The first step multiplies matching coordinates and pulls the two exponentials, which do not depend on \(n\), out of the sum. The remaining sum is the power series of the exponential function, a fact from calculus, so it equals \(e^{x x'}\). Combining exponents, \(-\frac{x^2 + x'^2}{2} + x x' = -\frac{x^2 - 2 x x' + x'^2}{2} = -\frac{(x - x')^2}{2}\), so

\[ \langle \psi(x), \psi(x') \rangle = e^{-\frac{(x - x')^2}{2}}. \]

An infinite-dimensional inner product that costs one subtraction and one exponential. In the snippet, `psi` builds the first coordinates term by term, and the partial sums close in on the closed form:

```python
import math

def psi(x, terms):
    return [math.exp(-x * x / 2) * x ** n
            / math.sqrt(math.factorial(n)) for n in range(terms)]

x, xp = 0.7, -0.4
for terms in (2, 5, 10, 20):
    s = sum(a * b for a, b in zip(psi(x, terms), psi(xp, terms)))
    print(terms, round(s, 8))
print(round(math.exp(-(x - xp) ** 2 / 2), 8))
```

It prints `2 0.52021969`, `5 0.54608432`, `10 0.54607443`, `20 0.54607443`, and then the closed form `0.54607443`. Ten coordinates already match to eight digits.

More generally, for a scale \(\sigma > 0\), the **Gaussian kernel** is

\[ K(\mathbf x, \mathbf x') = e^{-\frac{\|\mathbf x - \mathbf x'\|^2}{2\sigma}}. \]

(It is also called the RBF kernel, for radial basis function. Watch the convention: we divide by \(2\sigma\); many references divide by \(2\sigma^2\), which only renames the parameter.) Here \(\|\mathbf x - \mathbf x'\|^2\) is the squared distance between the two inputs and \(\sigma > 0\) a scale you choose. Read it as a similarity: **it is 1 for identical messages and decays exponentially with their squared distance, so \(\sigma\) sets what “close” means.** For subject lines as 0/1 vectors, the squared distance counts the words in one line but not the other, the symmetric difference, written \(a \triangle b\) (Python’s `a ^ b`). “winner claim lunch” and “winner claim report” differ in two words, so with \(\sigma = 1\) the kernel is \(e^{-1} \approx 0.37\):

```python
import math

def K(a, b, sigma):
    return math.exp(-len(a ^ b) / (2 * sigma))

x = {"winner", "claim", "lunch"}
for other in ({"winner", "claim", "report"},
              {"verify", "password", "report"}):
    print(len(x ^ other),
          [round(K(x, other, s), 4) for s in (0.25, 1, 4)])
```

It prints `2 [0.0183, 0.3679, 0.7788]` and `6 [0.0, 0.0498, 0.4724]`. A small \(\sigma\) calls even a one-word swap far; a large one calls even disjoint lines fairly close. The same computation in many variables gives an embedding with a coordinate \(\frac{1}{\sqrt{n!}} e^{-\|\mathbf x\|^2/2} \prod_{i=1}^n x_{J_i}\) for every \(n\) and every monomial, so a halfspace over it can express any polynomial predictor at all.

Wait. The class of all polynomial threshold rules has infinite VC dimension. Shouldn’t lesson 6 forbid learning with it? There is no contradiction, and the reason is the thread running through lessons 15 and 16: the sample complexity of a large-margin learner depends on the margin in the feature space, not on its dimension. If the data happen to be separable with a large margin in this feature space, few labels suffice. If not, the margin can be arbitrarily small, and the bound says nothing. Large margins are the luck you hope for, not something the kernel promises.

Now it is your turn to play. In the lab below you can lift the points of the line onto the parabola, switch kernels on a small mail world, and watch both the Gram matrix and the number of coefficients the learner keeps.

[[lab]]

## Kernels are prior knowledge

Choosing an embedding is choosing a hypothesis class, so it is choosing an inductive bias, the idea from lesson 1. Every kernel says something about which messages should get the same label.

If you believe the positive examples are the ones inside some ellipse, use all monomials of degree at most 2, or simply the degree-2 polynomial kernel: an ellipse is a degree-2 threshold rule. If you believe junk is marked by pairs of words, the same kernel says it. The Gaussian kernel says something different: messages that are close should get the same label. On subject lines, “close” means “share most of their words”. Is that a good belief about pair-marked junk? Look at three subject lines: \(\{\)winner, claim, lunch\(\}\), \(\{\)winner, claim, report\(\}\) and \(\{\)winner, verify, lunch\(\}\). The first differs from each of the other two in exactly one word, so the Gaussian kernel finds it equally similar to both. But the second is junk like the first, and the third is keep. The kernel’s idea of similarity is blind to the very distinction that matters. You will see what that costs, in labels, in the lab at the end of this lesson.

Here is a cleaner example, from a different domain. To spot a virus, look for a **signature**, a short string of characters whose presence in a file means trouble. Let \(\mathcal X_d\) be the strings of length at most \(d\) over an alphabet \(\Sigma\), and let the class be \(\mathcal H = \{h_v : v \in \mathcal X_d\}\), the set of all \(h_v\) with \(v\) a string of length at most \(d\), where \(h_v(x) = 1\) if \(v\) is a substring of \(x\) and \(-1\) otherwise. Embed each string \(x\) as a vector \(\psi(x) \in \{0, 1\}^{|\mathcal X_d|}\), a list of 0s and 1s with one entry for each of the \(|\mathcal X_d|\) possible strings \(v\), equal to 1 when \(v\) is a substring of \(x\). Each \(h_v\) is then a halfspace over \(\psi\) with a short weight vector: put weight 2 on coordinate \(v\), 0 elsewhere, and use the bias term \(-1\). The score is \(2 - 1 = 1\) when \(v\) is a substring of \(x\) and \(0 - 1 = -1\) when it is not, so every string sits at distance at least 1 from the boundary in the rescaled sense of lesson 9, with a weight vector of norm 2 whatever \(d\) is. A string of length \(d\) has \(d\) substrings starting at its first letter, \(d - 1\) at its second, and so on, at most \(d + (d - 1) + \dots + 1 = d(d+1)/2\) in all, and the squared norm of a 0/1 vector counts its 1s, so \(\|\psi(x)\| \le \sqrt{d(d+1)/2} = O(d)\): **the embedded string is never longer than about \(d\).** The margin bound of lesson 15 then gives a sample complexity polynomial in \(d\).

But the dimension \(|\mathcal X_d|\) is exponential in \(d\), so running SVM on \(\psi\) directly is hopeless. The kernel, though, is just “how many substrings do these two strings share?”, which takes polynomial time in \(d\):

```python
def subs(x):
    return {x[i:j] for i in range(len(x))
            for j in range(i + 1, len(x) + 1)}

def K(x, xp):
    return len(subs(x) & subs(xp))

a, b = "freeprize", "prizefree"
space = sorted(subs(a) | subs(b))
psi_a = [int(v in subs(a)) for v in space]
psi_b = [int(v in subs(b)) for v in space]
print(K(a, b), sum(p * q for p, q in zip(psi_a, psi_b)))
print(K(a, a), len(a) * (len(a) + 1) // 2)
print(f"{sum(26 ** n for n in range(1, 11)):.2e}")
```

It prints `22 22`, `42 45` and then `1.47e+14`. The two strings share 22 nonempty substrings, and the explicit inner product agrees. \(K(a, a) = \|\psi(a)\|^2\) is 42, under the 45 that nine letters allow, because some substrings, such as “e”, repeat. (The snippet only builds the coordinates that these two strings use; the full feature space for 26 letters and \(d = 10\) has about \(1.47 \times 10^{14}\) of them, the second number.) Notice what else happened: strings are not vectors at all, yet we are learning halfspaces over them. Embeddings let you use the whole machinery of halfspaces on domains that have no coordinates of their own.

[[diagram:prior]]

## Which functions are kernels?

So far we started from an embedding and computed its kernel. In practice you often go the other way: you dream up a similarity function \(K: \mathcal X \times \mathcal X \to \mathbb R\) that encodes your prior knowledge, and you want to know whether some embedding stands behind it. If none does, the whole derivation above, which needed \(K(x, x') = \langle \psi(x), \psi(x') \rangle\), collapses. Is there a test you can run on \(K\) alone?

There is, and it is about Gram matrices. A symmetric \(m \times m\) matrix \(G\) is **positive semidefinite** when \(\boldsymbol\alpha^\top G \boldsymbol\alpha = \sum_{i,j} \alpha_i \alpha_j G_{i,j} \ge 0\) for every \(\boldsymbol\alpha \in \mathbb R^m\). A symmetric function \(K\) is positive semidefinite when every Gram matrix it produces is: for every \(m\) and every \(x_1, \dots, x_m \in \mathcal X\), the matrix \(G_{i,j} = K(x_i, x_j)\) is positive semidefinite. The quadratic form \(\boldsymbol\alpha^\top G \boldsymbol\alpha\) is the double sum we met as \(\|\mathbf w\|^2\): pick a weight \(\alpha_i\) per point and add \(\alpha_i \alpha_j G_{i,j}\) over all pairs. In plain words: **no choice of weights makes the double sum negative.** The function `quad` in the snippet below is this sum, term by term.

**Characterization of kernels.** A symmetric function \(K: \mathcal X \times \mathcal X \to \mathbb R\) is the inner product of some embedding into a Hilbert space if and only if it is positive semidefinite.

Before the proof, let’s see the test bite. On the points 1 and 2 of the real line, try the product \(K(x, x') = x x'\), which is plainly an inner product in \(\mathbb R^1\), and the maximum \(K(x, x') = \max\{x, x'\}\), which looks like a harmless similarity:

```python
import random

def quad(G, a):
    n = len(a)
    return sum(a[i] * G[i][j] * a[j]
               for i in range(n) for j in range(n))

pts = [1, 2]
for name, K in (("product", lambda p, q: p * q), ("max", max)):
    G = [[K(p, q) for q in pts] for p in pts]
    print(name, G, quad(G, [1, -1]))

rng = random.Random(0)
xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(6)]
G = [[(1 + sum(a * b for a, b in zip(p, q))) ** 2 for q in xs]
     for p in xs]
worst = min(quad(G, [rng.gauss(0, 1) for _ in xs])
            for _ in range(10_000))
print(worst >= 0)
```

It prints `product [[1, 2], [2, 4]] 1`, then `max [[1, 2], [2, 2]] -1`, then `True`. The maximum fails on just two points: \(\boldsymbol\alpha = (1, -1)\) gives \(1 - 2 - 2 + 2 = -1 < 0\), so by the theorem no embedding has \(\max\) as its inner product, however clever. The degree-2 polynomial kernel, on six random points in \(\mathbb R^3\), survives 10,000 random \(\boldsymbol\alpha\) (a spot check, not a proof; the proof is the polynomial-kernel computation above together with the easy direction below).

[[diagram:gram]]

*Proof, the easy direction.* Suppose \(K(x, x') = \langle \psi(x), \psi(x') \rangle\). Then \(K\) is symmetric because the inner product is. Fix points \(x_1, \dots, x_m\) and any \(\boldsymbol\alpha\). Using bilinearity to pull the sums inside,

\[ \boldsymbol\alpha^\top G \boldsymbol\alpha = \sum_{i,j} \alpha_i \alpha_j \langle \psi(x_i), \psi(x_j) \rangle = \Big\langle \sum_i \alpha_i \psi(x_i), \sum_j \alpha_j \psi(x_j) \Big\rangle = \Big\| \sum_i \alpha_i \psi(x_i) \Big\|^2 \ge 0. \]

In plain words: **the double sum is the squared length of a vector, and squared lengths are never negative.** For the product kernel on 1 and 2 with \(\boldsymbol\alpha = (1, -1)\), the vector is \(1 - 2 = -1\), and its square is the `1` printed above.

*Proof, the interesting direction.* Now suppose \(K\) is symmetric and positive semidefinite. We have to invent a space and an embedding. The idea is bold: let each point be represented by its own row of similarities. For \(x \in \mathcal X\), let \(\psi(x)\) be the function \(K(\cdot, x)\) (the dot marks the empty slot), which sends \(x'\) to \(K(x', x)\). These are elements of the vector space of all functions from \(\mathcal X\) to \(\mathbb R\). Let \(V\) be the set of their finite linear combinations, functions of the form \(f = \sum_i \alpha_i K(\cdot, x_i)\).

On \(V\), define

\[ \Big\langle \sum_i \alpha_i K(\cdot, x_i), \; \sum_j \beta_j K(\cdot, x'_j) \Big\rangle = \sum_{i,j} \alpha_i \beta_j K(x_i, x'_j). \]

In plain words: **the inner product of two combinations is the kernel summed over every pair of their points, weighted by both coefficients.** The function `inner` in the snippet at the end of this proof is this formula term by term. We must check four things: that this is well defined, symmetric, bilinear, and positive definite.

*Well defined.* A function in \(V\) can have many representations as a combination, so we must check that the formula does not depend on which one we pick. Write \(f = \sum_i \alpha_i K(\cdot, x_i)\) and \(g = \sum_j \beta_j K(\cdot, x'_j)\). Then \(f(x'_j) = \sum_i \alpha_i K(x'_j, x_i) = \sum_i \alpha_i K(x_i, x'_j)\), using symmetry, so the formula equals \(\sum_j \beta_j f(x'_j)\). That expression depends on \(f\) only through its values, not through the \(\alpha_i\) and \(x_i\). By the same computation with the roles swapped, the formula also equals \(\sum_i \alpha_i g(x_i)\), which depends on \(g\) only through its values. So the number is the same for every pair of representations.

*Symmetric and bilinear.* Swapping \(f\) and \(g\) swaps the order inside \(K(x_i, x'_j)\), and \(K\) is symmetric. Linearity in each argument is immediate from the formula.

*Positive definite.* For \(f = \sum_i \alpha_i K(\cdot, x_i)\), \(\langle f, f \rangle = \sum_{i,j} \alpha_i \alpha_j K(x_i, x_j) = \boldsymbol\alpha^\top G \boldsymbol\alpha \ge 0\), by the assumption on \(K\). It remains to show that \(\langle f, f \rangle = 0\) forces \(f = 0\). Two small facts do it. The first is the **reproducing property**: taking \(g = K(\cdot, x)\), a single term with coefficient 1, the formula gives

\[ \langle f, K(\cdot, x) \rangle = \sum_i \alpha_i K(x_i, x) = f(x). \]

In plain words: **to evaluate \(f\) at \(x\), take its inner product with \(K(\cdot, x)\).** The second is the Cauchy–Schwarz inequality \(|\langle f, g \rangle| \le \sqrt{\langle f, f \rangle} \sqrt{\langle g, g \rangle}\), which holds for any symmetric bilinear form with \(\langle h, h \rangle \ge 0\). (For every real \(t\), \(0 \le \langle f + t g, f + t g \rangle = \langle f, f \rangle + 2t \langle f, g \rangle + t^2 \langle g, g \rangle\). If \(\langle g, g \rangle > 0\), a quadratic in \(t\) that never goes negative has discriminant \(4\langle f, g \rangle^2 - 4 \langle f, f \rangle \langle g, g \rangle \le 0\). If \(\langle g, g \rangle = 0\), the expression is linear in \(t\) and stays nonnegative for all \(t\) only if \(\langle f, g \rangle = 0\).) Put them together:

\[ |f(x)| = |\langle f, K(\cdot, x) \rangle| \le \sqrt{\langle f, f \rangle} \sqrt{K(x, x)}. \]

So **a function of length 0 is 0 everywhere.** If \(\langle f, f \rangle = 0\), then \(f(x) = 0\) for every \(x\), so \(f\) is the zero function.

So \(V\) is an inner product space, and by construction

\[ \langle \psi(x), \psi(x') \rangle = \langle K(\cdot, x), K(\cdot, x') \rangle = K(x, x'). \]

To be a Hilbert space in the strict sense, \(V\) must also be complete. The standard completion adds the limits of its Cauchy sequences, extends the inner product to them, and changes none of the values above. And nothing this lesson does needs more: the representer theorem only ever projects onto the span of \(m\) training vectors, where completeness is automatic. \(\blacksquare\)

The reproducing property is worth seeing in numbers, because it is the whole trick in one line: evaluating a function in \(V\) is taking an inner product with \(K(\cdot, x)\). Here \(f = K(\cdot, 0) - 2K(\cdot, 1) + 0.5 K(\cdot, 2.5)\) for the Gaussian kernel with \(\sigma = 1\):

```python
import math

def K(x, xp):
    return math.exp(-(x - xp) ** 2 / 2)

centers, alpha = [0.0, 1.0, 2.5], [1.0, -2.0, 0.5]

def f(x):
    return sum(a * K(c, x) for a, c in zip(alpha, centers))

def inner(alpha, xs, beta, xps):
    return sum(a * b * K(x, xp) for a, x in zip(alpha, xs)
               for b, xp in zip(beta, xps))

x = 1.7
print(round(f(x), 10),
      round(inner(alpha, centers, [1.0], [x]), 10))
norm_f = math.sqrt(inner(alpha, centers, alpha, centers))
print(abs(f(x)) <= norm_f * math.sqrt(K(x, x)))
```

It prints `-0.9665884814 -0.9665884814` and `True`: the value of \(f\) at 1.7 equals its inner product with \(K(\cdot, 1.7)\), and the Cauchy–Schwarz bound from the proof holds.

## Soft-SVM with kernels, trained by SGD

We could hand the \(\boldsymbol\alpha\) form of soft-SVM to a quadratic-programming solver. There is a simpler road, and it reuses something you already built. In lesson 15 you solved soft-SVM in the form

\[ \min_{\mathbf w} \; \frac{\lambda}{2} \|\mathbf w\|^2 + \frac{1}{m} \sum_{i=1}^m \max\{0, 1 - y_i \langle \mathbf w, \mathbf x_i \rangle\} \]

with SGD, lesson 14’s stochastic gradient descent, which steps against the slope of one random example’s loss at a time. The \(\frac{\lambda}{2}\) is a convenience of SGD: it is the \(\lambda \|\mathbf w\|^2\) form with \(\lambda\) halved, so the two line vectors from the representer section now cost \(1.3\) and \(0.0909 + 0.325 = 0.4159\). The algorithm had this shape, where \([m]\) is the positions \(1, \dots, m\):

```
SGD-SoftSVM(S, λ, T):
  θ ← 0
  for t = 1..T:
    w⁽ᵗ⁾ ← θ / (λ t)
    choose i uniformly at random from [m]
    if y_i ⟨w⁽ᵗ⁾, x_i⟩ < 1:
      θ ← θ + y_i x_i
  return w̄ = (1/T) Σ_t w⁽ᵗ⁾
```

Look at the only thing that ever gets added to \(\boldsymbol\theta\): a training vector, times \(\pm 1\). Run it on the image sequence, with \(\psi(x_i)\) in place of \(\mathbf x_i\), and \(\boldsymbol\theta\) is always a combination of \(\psi(x_1), \dots, \psi(x_m)\), so it can be stored as \(m\) coefficients. The representer theorem promised that a solution lives in the span; SGD never even leaves it.

So keep two vectors in \(\mathbb R^m\), where a superscript \((t)\) marks the value at step \(t\): \(\boldsymbol\beta^{(t)}\), the coefficients of \(\boldsymbol\theta^{(t)} = \sum_j \beta_j^{(t)} \psi(x_j)\), and \(\boldsymbol\alpha^{(t)}\), the coefficients of \(\mathbf w^{(t)} = \sum_j \alpha_j^{(t)} \psi(x_j)\).

```
KernelSGD-SoftSVM(S, K, λ, T):
  β ← 0 ∈ ℝᵐ;  ᾱ ← 0 ∈ ℝᵐ
  for t = 1..T:
    α ← β / (λ t)
    ᾱ ← ᾱ + α / T
    choose i uniformly at random from [m]
    if y_i Σ_j α_j K(x_j, x_i) < 1:
      β_i ← β_i + y_i
  return ᾱ      (predict on x by the sign of Σ_j ᾱ_j K(x_j, x))
```

Every other coordinate of \(\boldsymbol\beta\) stays as it was. The output stands for \(\bar{\mathbf w} = \sum_j \bar\alpha_j \psi(x_j)\), with \(\bar{\boldsymbol\alpha} = \frac1T \sum_{t=1}^T \boldsymbol\alpha^{(t)}\), the bar marking an average over the \(T\) steps.

[[diagram:kernel-sgd]]

**Kernel SGD computes the same hypothesis.** Run SGD-SoftSVM on the image sequence and KernelSGD-SoftSVM on \(S\) with the kernel of \(\psi\), using the same random choices of \(i\). Then the outputs are equal: \(\hat{\mathbf w} = \bar{\mathbf w}\), where \(\hat{\mathbf w}\) is the output of SGD-SoftSVM and \(\bar{\mathbf w} = \sum_j \bar\alpha_j \psi(x_j)\).

*Proof.* We show by induction on \(t\) that \(\boldsymbol\theta^{(t)} = \sum_j \beta_j^{(t)} \psi(x_j)\) for every \(t\).

For \(t = 1\), both sides are 0.

Suppose it holds at step \(t\). The two algorithms set \(\mathbf w^{(t)} = \boldsymbol\theta^{(t)}/(\lambda t)\) and \(\boldsymbol\alpha^{(t)} = \boldsymbol\beta^{(t)}/(\lambda t)\), so dividing the hypothesis by \(\lambda t\) gives \(\mathbf w^{(t)} = \sum_j \alpha_j^{(t)} \psi(x_j)\). Both pick the same \(i\). The feature-space test computes

\[ y_i \langle \mathbf w^{(t)}, \psi(x_i) \rangle = y_i \Big\langle \sum_j \alpha_j^{(t)} \psi(x_j), \psi(x_i) \Big\rangle = y_i \sum_{j=1}^m \alpha_j^{(t)} K(x_j, x_i), \]

which is exactly the kernel test, by the same pass-through-the-sum step as in the kernel trick: **the two tests compute the same number**, so both algorithms take the same branch. If neither updates, both sides of the claim are unchanged. If both update,

\[ \boldsymbol\theta^{(t+1)} = \boldsymbol\theta^{(t)} + y_i \psi(x_i) = \sum_{j} \beta_j^{(t)} \psi(x_j) + y_i \psi(x_i) = \sum_j \beta_j^{(t+1)} \psi(x_j), \]

since \(\boldsymbol\beta^{(t+1)}\) differs from \(\boldsymbol\beta^{(t)}\) only by \(y_i\) in coordinate \(i\): **adding \(y_i \psi(x_i)\) to \(\boldsymbol\theta\) is adding \(y_i\) to coefficient \(i\).** That completes the induction. It follows that \(\mathbf w^{(t)} = \sum_j \alpha_j^{(t)} \psi(x_j)\) for every \(t\), and averaging over \(t\), by linearity, \(\hat{\mathbf w} = \frac1T \sum_t \mathbf w^{(t)} = \sum_j \bar\alpha_j \psi(x_j) = \bar{\mathbf w}\). \(\blacksquare\)

Let’s watch the two runs side by side on the line example, with \(\psi(x) = (1, x, x^2)\) and the kernel \(K(x, x') = \langle \psi(x), \psi(x') \rangle\). One detail matters: the comparison \(< 1\) is a knife edge, and floating-point rounding can push the two runs to different sides of it when a score lands exactly on 1. Exact fractions remove that doubt. In the snippet, `theta` follows the feature-space test and `beta` the kernel test, side by side.

```python
import random
from fractions import Fraction

S = [(x, 1 if abs(x) > 2 else -1) for x in range(-5, 6)]
psi = lambda x: (1, x, x * x)
dot = lambda a, b: sum(i * j for i, j in zip(a, b))
K = lambda x, xp: dot(psi(x), psi(xp))
lam, T, m, rng = Fraction(1, 10), 10_000, len(S), random.Random(0)
theta, beta, ws, alphas = [0, 0, 0], [0] * m, [], []
for t in range(1, T + 1):
    i = rng.randrange(m)
    x, y = S[i]
    ws.append([v / (lam * t) for v in theta])
    alphas.append([v / (lam * t) for v in beta])
    if y * dot(ws[-1], psi(x)) < 1:
        theta = [p + y * q for p, q in zip(theta, psi(x))]
    a = alphas[-1]
    if y * sum(a[j] * K(S[j][0], x) for j in range(m)) < 1:
        beta[i] += y
wbar = [sum(c) / T for c in zip(*ws)]
abar = [sum(c) / T for c in zip(*alphas)]
back = [sum(abar[j] * psi(S[j][0])[c] for j in range(m))
        for c in range(3)]
print(wbar == back, [round(float(v), 4) for v in wbar])
```

It prints `True [-1.3161, -0.0179, 0.2995]`: after 10,000 steps, the explicit average weight vector and the one rebuilt from \(\bar{\boldsymbol\alpha}\) are exactly equal, as fractions. Read the weights as a rule: \(0.2995 x^2 - 0.0179 x - 1.3161 > 0\) holds roughly when \(x^2 > 4.4\), so it is positive for \(|x| \ge 3\) and negative for \(|x| \le 2\). The learner found the parabola. (Patience is part of the deal: the average carries SGD’s noisy early steps, and with only 300 steps the same run still mislabels three of the eleven points.)

**Cost.** Let \(c_K\) be the time to evaluate the kernel once: \(O(n)\) for the polynomial and Gaussian kernels on \(\mathbb R^n\), and \(O(s)\) for subject lines of \(s\) words stored as sets.

- *Time per step.* The test needs \(\sum_j \alpha_j K(x_j, x_i)\). Only the coordinates with \(\beta_j \ne 0\) contribute, and at step \(t\) at most \(\min(t, m)\) of them are nonzero, so the test costs \(O(\min(t, m)\, c_K)\). Updating \(\bar{\boldsymbol\alpha}\) the naive way costs \(O(m)\) per step. Since \(\bar\alpha_j = \frac{1}{\lambda T} \sum_t \beta_j^{(t)}/t\) and \(\beta_j\) changes only when example \(j\) is updated, you can instead settle coordinate \(j\) lazily at each of its updates, adding its old value times a difference of harmonic numbers (\(H_n = 1 + \frac12 + \dots + \frac1n\)), \(\sum_{t'=t_0}^{t-1} 1/t'\), which is \(O(1)\) per update plus \(O(m)\) at the end. All of \(T\) steps: \(O(T \min(T, m)\, c_K)\).
- *With a cached Gram matrix.* Precompute all \(m^2\) kernel values in \(O(m^2 c_K)\) time and \(O(m^2)\) space; then each test is \(O(m)\). For \(m = 16{,}000\) messages, the Gram matrix has 256 million entries, about 2 GB at 8 bytes each. Caching is a trade, not a free lunch.
- *Space without the cache.* \(O(m)\) for \(\boldsymbol\beta\) and \(\bar{\boldsymbol\alpha}\), plus the training messages themselves.
- *Prediction.* \(O(r\, c_K)\) per new message, where \(r\) is the number of nonzero \(\bar\alpha_j\), at most \(m\).

The lazy formula is easier to trust with numbers. Since \(\alpha_j^{(t)} = \beta_j^{(t)}/(\lambda t)\), averaging gives \(\bar\alpha_j = \frac{1}{\lambda T} \sum_{t=1}^T \beta_j^{(t)}/t\): **the average coefficient is the history of \(\beta_j\), each step divided by its step number.** Follow one coordinate for \(T = 6\) steps with \(\lambda = 0.1\), where \(\beta_j\) is 0 at step 1, 1 at steps 2 to 4 and 2 at steps 5 and 6. Grouping the steps where it held still gives \(1 \cdot (\frac12 + \frac13 + \frac14) + 2 \cdot (\frac15 + \frac16)\), two differences of harmonic numbers:

```python
from fractions import Fraction as F

lam, T = F(1, 10), 6
beta = [0, 1, 1, 1, 2, 2]
direct = sum(b / (lam * t) for t, b in enumerate(beta, 1)) / T
H = lambda a, b: sum(F(1, t) for t in range(a, b + 1))
lazy = (1 * H(2, 4) + 2 * H(5, 6)) / (lam * T)
print(direct, lazy)
```

It prints `109/36 109/36`.

Compare with SGD on explicit features of dimension \(D\): \(O(D)\) per step and \(O(D)\) space. The kernel version wins whenever \(D\) is much larger than \(m\), which for triples of 50,000 words means always. It does have one cost the explicit version lacks: prediction grows with the number of training messages the hypothesis keeps.

**Build it yourself.** Using only the Python standard library, with subject lines as sets of word indices (0 winner, 1 claim, 2 verify, 3 password, 4 lunch, 5 meeting, 6 invoice, 7 report):

```python
import itertools
import math
import random

JUNK = [{0, 1, 4}, {2, 3, 4}, {0, 1, 5}, {2, 3, 7}, {0, 1, 2}]
KEEP = [{0, 2, 4}, {1, 3, 4}, {0, 3, 7}, {1, 2, 6},
        {4, 5, 7}, {0, 4, 5}, {3, 6, 7}]
S = ([(frozenset(x), 1) for x in JUNK]
     + [(frozenset(x), -1) for x in KEEP])
ALL = [frozenset(c) for c in itertools.combinations(range(8), 3)]

def label(x):
    return 1 if {0, 1} <= x or {2, 3} <= x else -1

def poly_kernel(a, b, k): ...
def gaussian_kernel(a, b, sigma): ...
def gram(xs, K): ...
def kernel_sgd_svm(S, K, lam, T, rng): ...
def predict(alpha, S, K, x): ...
def train_mistakes(alpha, S, K): ...

if __name__ == "__main__":
    K1 = lambda a, b: poly_kernel(a, b, 1)
    K2 = lambda a, b: poly_kernel(a, b, 2)
    assert K2(frozenset({0, 1}), frozenset({1, 2})) == 4
    assert math.isclose(gaussian_kernel(S[0][0], S[3][0], 1.0),
                        math.exp(-3))
    G = gram([x for x, _ in S], K2)
    assert all(G[i][j] == G[j][i]
               for i in range(12) for j in range(12))
    a2 = kernel_sgd_svm(S, K2, 0.0137, 2000, random.Random(0))
    a1 = kernel_sgd_svm(S, K1, 0.0137, 2000, random.Random(0))
    assert train_mistakes(a2, S, K2) == 0
    assert train_mistakes(a1, S, K1) >= 1
    assert round(sum(a2), 4) == -0.3
    unseen = [predict(a2, S, K2, x) != label(x) for x in ALL]
    assert sum(unseen) == 3
```

1. Implement the six functions. `poly_kernel` is \((1 + |a \cap b|)^k\) and `gaussian_kernel` is \(e^{-|a \triangle b|/(2\sigma)}\), the set forms of the two kernels from earlier. `kernel_sgd_svm` follows KernelSGD-SoftSVM exactly: at each step, first compute \(\boldsymbol\alpha\) and add \(\boldsymbol\alpha/T\) to \(\bar{\boldsymbol\alpha}\), then draw `i = rng.randrange(m)`, then test and update. Return \(\bar{\boldsymbol\alpha}\) as a list. `predict` returns \(+1\) when \(\sum_j \bar\alpha_j K(x_j, x) > 0\) and \(-1\) otherwise.
2. The odd \(\lambda = 0.0137\) is on purpose. With a round value such as 0.1, some step lands exactly on the boundary \(y_i \sum_j \alpha_j K(x_j, x_i) = 1\), and the order in which your code adds floating-point numbers decides the branch. Try 0.1 and see whether your `sum(a2)` still matches.
3. Why must `train_mistakes(a1, S, K1)` be at least 1, whatever the learner does? Find the four messages in `S` that make it so.
4. Train with the Gaussian kernel for \(\sigma \in \{0.25, 1, 4\}\) and count mistakes on `ALL`. Which \(\sigma\) does worst, and why, in terms of what “close” means?
5. Replace the cached Gram matrix by kernel evaluations only for the nonzero \(\beta_j\), and time both versions for \(m = 2{,}000\) random subject lines.

## How many labels? Margins in feature space

The kernel trick solved the computational half of the bug. Now the statistical half: is it reasonable to learn in a feature space with over a billion coordinates from a budget of labels you can afford? Lesson 15 gave the tool, and it transfers to kernels without a change.

Recall it. Let every message satisfy \(\|\psi(x)\| \le \rho\), so \(\rho\) is lesson 15’s radius, and let \(A(S)\) be the solution of soft-SVM, \(\min_{\mathbf w} \lambda \|\mathbf w\|^2 + L^{\text{hinge}}_S(\mathbf w)\), on the image sequence. Then for every vector \(\mathbf u\) in the feature space,

\[ \mathop{\mathbb E}_{S \sim \mathcal D^m}\big[ L^{0-1}_{\mathcal D}(A(S)) \big] \le L^{\text{hinge}}_{\mathcal D}(\mathbf u) + \lambda \|\mathbf u\|^2 + \frac{2\rho^2}{\lambda m}. \]

Read it from the inside out.

- \(A(S)\) is the hypothesis soft-SVM returns on the training sequence \(S\), and \(L^{0-1}_{\mathcal D}(A(S))\) is its true risk with the 0-1 loss: the probability it mislabels a fresh message.
- Under \(\mathbb E\), “the average value of”, \(S \sim \mathcal D^m\) says the training sequence is \(m\) examples drawn i.i.d. from \(\mathcal D\), so the average runs over lucky and unlucky archives alike.
- On the right, \(L^{\text{hinge}}_{\mathcal D}(\mathbf u)\) is the expected hinge loss of a comparison vector \(\mathbf u\), \(\lambda \|\mathbf u\|^2\) its penalty, and \(\frac{2\rho^2}{\lambda m}\) a price that falls as the number of labels \(m\) grows.

In plain words: **on average, the filter’s error rate is at most any comparison vector’s hinge loss, plus its penalty, plus a term that shrinks with more labels.** Suppose a separator \(\mathbf u\) with norm \(B\) has \(y \langle \mathbf u, \psi(x) \rangle \ge 1\) on every message. Every hinge loss of \(\mathbf u\) is then \(\max\{0, 1 - y\langle \mathbf u, \psi(x) \rangle\} = 0\), so

\[ \mathop{\mathbb E}\big[ L^{0-1}_{\mathcal D}(A(S)) \big] \le \lambda B^2 + \frac{2\rho^2}{\lambda m}. \]

Which \(\lambda\) makes the right side smallest? For positive numbers \(a\) and \(b\), \(a + b \ge 2\sqrt{ab}\), because \(a + b - 2\sqrt{ab} = (\sqrt a - \sqrt b)^2 \ge 0\), with equality when \(a = b\). Here \(ab = \lambda B^2 \cdot \frac{2\rho^2}{\lambda m} = \frac{2\rho^2 B^2}{m}\), whatever \(\lambda\) is, so the right side is at least \(2\sqrt{2\rho^2 B^2/m}\), and setting \(\lambda B^2 = \frac{2\rho^2}{\lambda m}\), that is \(\lambda = \sqrt{2\rho^2/(B^2 m)}\), achieves it:

\[ \mathop{\mathbb E}\big[ L^{0-1}_{\mathcal D}(A(S)) \big] \le \sqrt{\frac{8 \rho^2 B^2}{m}}. \]

In plain words: **the expected error shrinks like one over the square root of the number of labels, at a rate set by \(\rho B\).** No dimension anywhere. To guarantee an expected true risk of at most \(\varepsilon\), the error rate you are willing to accept, solve \(\sqrt{8\rho^2B^2/m} \le \varepsilon\) for \(m\): it is enough that \(m \ge 8\rho^2 B^2/\varepsilon^2\). And both quantities can be read through the kernel: \(\rho^2 = \max_x K(x, x)\), the largest squared length \(\|\psi(x)\|^2\) over all messages, and a separator written as \(\mathbf u = \sum_i \gamma_i \psi(z_i)\) has \(B^2 = \sum_{i,j} \gamma_i \gamma_j K(z_i, z_j)\). As in lesson 9, any separator with margin 1 everywhere is a **certificate**: its norm bounds the labels you need. You never need the best one, just one you can check.

Now the mail. For the degree-2 polynomial kernel on three-word subject lines, \(K(x, x) = (1 + 3)^2 = 16\), so \(\rho = 4\); that was the 16 printed earlier. And the hand-built \(\mathbf u\) is a certificate with \(B^2 = 5\). Push them through: with \(m = 16{,}000\), the best \(\lambda\) is \(\sqrt{2 \cdot 16/(5 \cdot 16{,}000)} = 0.02\), the two terms \(\lambda B^2\) and \(2\rho^2/(\lambda m)\) are both 0.1, and their sum is the bound 0.2:

```python
import math

rho, B, m = 4, math.sqrt(5), 16_000
lam = math.sqrt(2 * rho ** 2 / (B ** 2 * m))
terms = [lam * B ** 2, 2 * rho ** 2 / (lam * m)]
print(round(lam, 4), [round(t, 4) for t in terms],
      round(sum(terms), 4),
      round(math.sqrt(8 * rho ** 2 * B ** 2 / m), 4))
```

It prints `0.02 [0.1, 0.1] 0.2 0.2`. What about degree \(k\)? On 0/1 vectors a word times itself is itself, so the monomial \(x_1 x_2\) appears in every coordinate whose index tuple uses exactly the indices 1 and 2, padded with zeros or repeats. Counting the \(k\)-tuples over \(\{0, 1, 2\}\) that contain both a 1 and a 2 by inclusion and exclusion gives \(c_k = 3^k - 2 \cdot 2^k + 1\): all \(3^k\) lists, minus the \(2^k\) that miss 1 and the \(2^k\) that miss 2, plus the all-zeros list, which was taken away twice. For \(k = 2\), \(9 - 8 + 1 = 2\): the lists \((1, 2)\) and \((2, 1)\). The snippet’s `c` is this count. Spread a weight of \(2/c_k\) over those coordinates, the same for \(x_3 x_4\), and put \(-1\) on the all-zero tuple: the score is still \(2x_1x_2 + 2x_3x_4 - 1\), and the squared norm is \(1 + 2 \cdot c_k (2/c_k)^2 = 1 + 8/c_k\). Meanwhile \(\rho^2 = (1 + 3)^k\) grows. The labels needed for \(\varepsilon = 0.2\):

```python
import math

s, eps = 3, 0.2

def labels(rho_B):
    return math.ceil(8 * rho_B ** 2 / eps ** 2)

for k in (2, 3, 4):
    c = 3 ** k - 2 * 2 ** k + 1
    B = math.sqrt(1 + 8 / c)
    rho = (1 + s) ** (k / 2)
    print(k, c, round(B, 3), round(rho * B, 3), labels(rho * B))
```

It prints `2 2 2.236 8.944 16000`, `3 12 1.291 10.328 21334` and `4 50 1.077 17.233 59392`. Degree 2 certifies an expected true risk of at most 0.2 with 16,000 labels. Higher degrees have smaller certificates \(B\) but a larger radius \(\rho\), and the product grows. A richer kernel is not a free upgrade.

Two things about these numbers. First, compare them with counting dimensions. Halfspaces over pairs of a 50,000-word vocabulary live in a space of about \(1.25 \times 10^9\) dimensions, so a guarantee that grows with the VC dimension, like those of lesson 6, needs labels in proportion to that. The margin certificate does not mention the vocabulary at all: 16,000 labels for 1,000 words, 50,000 words or a million. Second, the guarantee is an upper bound in expectation, and a loose one. The filter will usually do far better than 0.2; what the bound gives you is a promise you can make before you train, the kind operations asked for in lesson 9.

One honest caveat. The bound is about the exact soft-SVM solution, and KernelSGD-SoftSVM approximates it, with an optimization error that lesson 14 showed shrinks as the number of steps \(T\) grows. In the lab, \(T\) is large enough for that gap to be small, but it is not part of the certificate.

## Fix the bug

Time to ship. In the lab below, the filter reads three-word subject lines from a vocabulary of 1,000 words, and this month’s junk is marked by winner with claim, or verify with password. The archive holds 20,000 labelled subject lines, which is your label budget. You choose a kernel (linear, polynomial of degree 2, 3 or 4, or Gaussian with a choice of \(\sigma\)) and how many labels to use. The lab sets \(\lambda\) from the bound and trains KernelSGD-SoftSVM.

The goal is a filter whose expected true risk is certified to be at most 0.2 by the margin bound, using a certificate the lab can check, within the budget, and whose measured error on fresh subject lines agrees. A filter that happens to do well without a certificate does not count: next month’s junk will not ask whether you were lucky.

Before you train, work out each kernel’s label count from the table above, and ask what the linear kernel can possibly certify. For the Gaussian kernel, think about the three subject lines from “Kernels are prior knowledge” before you trust it.

[[lab-act2]]

## What people get wrong

**“The kernel trick makes the feature space small.”** The feature space is exactly as large as before, possibly infinite. What shrinks is what you store and compute: \(m\) coefficients and \(m\) training messages instead of a weight vector, and kernel values instead of coordinates. The price moves to \(m\): the Gram matrix has \(m^2\) entries and each prediction touches up to \(m\) training messages.

**“Any similarity function is a kernel.”** Only the positive semidefinite ones. \(\max\{x, x'\}\) looks like a perfectly reasonable similarity and fails on two points. Plug a non-kernel into soft-SVM and the “norm” \(\boldsymbol\alpha^\top G \boldsymbol\alpha\) can go negative; the objective is then no longer the problem you meant, and every guarantee you derived for it is gone.

**“A Gaussian kernel can separate anything, so it is the safe default.”** With a small enough \(\sigma\), the Gram matrix of distinct messages is close to the identity and any labelling is separable. That is exactly the memorizer of lesson 1 in a new coat: separable with a norm that grows with the number of messages, so the margin bound needs more labels than there are messages (the first concept-check question asks you to show it). The kernel is prior knowledge, and the wrong prior costs labels.

**“The representer theorem says the solution uses only a few training messages.”** It says the solution is a combination of the training messages, possibly all \(m\) of them. Which coefficients are zero depends on the loss; for hard-SVM, lesson 15’s support vectors are the ones that remain.

## Recap and the next bug

1. An **embedding** \(\psi\) sends messages into a **feature space** where a halfspace can express nonlinear rules, such as “winner and claim both appear”. The true risk is preserved: \(L_{\mathcal D_\psi}(h) = L_{\mathcal D}(h \circ \psi)\).
2. The **representer theorem**: whenever the objective sees \(\mathbf w\) only through \(\langle \mathbf w, \psi(x_i) \rangle\) plus a nondecreasing function of \(\|\mathbf w\|\), some optimal \(\mathbf w\) is a combination \(\sum_i \alpha_i \psi(x_i)\) of the training vectors.
3. The **kernel trick**: rewritten in \(\boldsymbol\alpha\), learning and prediction need only the **kernel** \(K(x, x') = \langle \psi(x), \psi(x') \rangle\), through the **Gram matrix**. The polynomial kernel \((1 + \langle \mathbf x, \mathbf x' \rangle)^k\) costs \(O(n)\) for a space of \((n+1)^k\) coordinates; the Gaussian kernel costs \(O(n)\) for an infinite one.
4. A kernel is **prior knowledge**: the polynomial kernel believes in combinations of words, the Gaussian kernel in nearness, the substring kernel in signatures.
5. A symmetric \(K\) is a kernel exactly when every Gram matrix is **positive semidefinite**.
6. **KernelSGD-SoftSVM** stores \(\boldsymbol\beta \in \mathbb R^m\) instead of \(\boldsymbol\theta\) and computes exactly the same hypothesis as SGD in the feature space. In lesson 10, boosting handled a huge feature space with a weak learner that hands it one good coordinate at a time; kernels handle it by never needing coordinates at all.
7. The margin bound \(\sqrt{8 \rho^2 B^2/m}\) does not depend on the dimension, and any separator with margin 1 is a certificate for it.

Your filter is catching the pair-marked junk again, and the certificate is pinned above your desk. Then the product team walks over with a survey. Users are grateful, they say, but junk or keep is not what they actually want. They want their mail sorted into folders: work, personal, promotions and junk. And within the inbox, they want the important messages first. A single yes/no filter cannot do either: it has two answers, and folders need four, while an ordering needs a score for every message that is right relative to the others. Can the halfspaces, margins and kernels you have built answer questions with more than two answers? That is the next lesson: multiclass, ranking and complex prediction problems.

## Concept check

Write your answers down before moving on, with your reasoning, not just the answer.

1. Take the Gaussian kernel on the three-word subject lines of a 1,000-word vocabulary, and let \(\sigma \to 0\). What does the Gram matrix of \(N\) distinct subject lines tend to? Show that, in the limit, any labelling of them is separable, and find the smallest norm \(B\) of a separator with \(y \langle \mathbf w, \psi(x) \rangle \ge 1\) on all of them. If the \(N\) lines are the whole domain, how many labels does the margin bound then need for \(\varepsilon = 0.2\), and how does that compare with \(N\)? Which learner from lesson 1 have you rebuilt?
2. In the proof of the representer theorem, point to the exact step that uses “\(R\) is nondecreasing” and the exact step that uses “the data term sees \(\mathbf w\) only through \(\langle \mathbf w, \psi(x_i) \rangle\)”. Then give a small example in \(\mathbb R^2\) with one training vector, \(\psi(x_1) = (1, 0)\), where \(R\) is not nondecreasing and no optimal solution lies in the span of \(\psi(x_1)\).
3. A colleague proposes \(K(\mathbf x, \mathbf x') = \|\mathbf x - \mathbf x'\|^2\) as a kernel: “it measures how different two messages are, that’s a similarity in reverse”. Compute its Gram matrix on \(\mathbf x_1 = (0, 0)\) and \(\mathbf x_2 = (1, 0)\), find an \(\boldsymbol\alpha\) with \(\boldsymbol\alpha^\top G \boldsymbol\alpha < 0\), and say what that means for any embedding your colleague hopes to find.

## Problems

These are this lesson’s own problems, to work on after the concept check. Hints are available on request; try each one for a while first.

1. **A kernel Perceptron.** Rewrite the Perceptron of lesson 9 so that it touches the messages only through a kernel \(K\). What does it store, what does each update cost, and how does it predict?
2. **The minimum is a kernel.** Let \(N\) be a positive integer, and for \(x, x' \in \{1, \dots, N\}\) let \(K(x, x') = \min\{x, x'\}\). Prove that \(K\) is a valid kernel by finding an explicit embedding \(\psi: \{1, \dots, N\} \to \mathbb R^n\), for some \(n\), with \(K(x, x') = \langle \psi(x), \psi(x') \rangle\). (Contrast it with \(\max\), which you saw fail.)
3. **The nearest average.** A simple learner labels a new message by whichever class has the closer average in feature space. With \(\mathbf c_{+} = \frac{1}{m_+} \sum_{i : y_i = +1} \psi(x_i)\) and \(\mathbf c_{-}\) defined the same way (both classes nonempty), it predicts \(+1\) when \(\|\psi(x) - \mathbf c_+\| \le \|\psi(x) - \mathbf c_-\|\) and \(-1\) otherwise. Show that this is the halfspace \(\operatorname{sign}(\langle \mathbf w, \psi(x) \rangle + b)\) with \(\mathbf w = \mathbf c_+ - \mathbf c_-\) and \(b = \frac12 (\|\mathbf c_-\|^2 - \|\mathbf c_+\|^2)\), and then compute it using only the kernel.
