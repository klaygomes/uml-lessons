# A Gentle Start

## Thirty features and a few hundred labels

Your filter works again. The one-feature learner followed the spammers from one feature to the next, and the inbox is quiet. Then your manager drops by with two pieces of news.

The first is that the real mail system does not have four features, or ten. It has thirty. Every message comes in as thirty yes/no answers. Does the subject say "FREE"? Is there a link? Is the sender unknown? Was it sent at night? Is there an attachment? And twenty-five more like those.

The second is about money. Every label is a person reading a message and clicking "junk" or "keep", and those people have better things to do. You can afford a few hundred labels. Not thousands.

So you face two questions, and lesson 1 left both of them open. Which hypothesis classes can you afford with a few hundred examples? And when the learner returns a hypothesis that fits all your examples, how sure can you be that its true risk is small?

In lesson 1 we answered a toy version of the second question. In a world of sixteen messages, we computed exactly how often a wrong one-feature hypothesis survives \(m\) random examples. That calculation leaned on everything being uniform and independent, and the real mail room is neither. Today we answer it for any finite hypothesis class and for any way messages might arrive. By the end you will have the first real theorem of this course, proved line by line, and a formula that turns a label budget into a promise.

First, though, we need a shared language. We have been using some of it already.

## The language of the course

Most notation in learning theory is either standard or defined on the spot, and a notation table is something you consult, not something you read. But a few conventions will be on every page from here on, so let's meet them through the mail room.

**Vectors and their entries.** A message with thirty features is a vector \(x \in \{0,1\}^{30}\). The convention is lowercase letters for scalars and abstract objects, and switches to boldface when it wants to stress that something is a vector. We will keep writing a plain \(x\), as in lesson 1. The \(j\)-th entry is \(x_j\), so \(x_1 = 1\) means the subject says "FREE". With a whole sequence of messages \(x_1, \dots, x_m\), we write \(x_{i,j}\) for feature \(j\) of the \(i\)-th message. You met that in lesson 1's pseudocode.

**Uppercase means a collection.** Capital letters are sets, sequences or matrices, and context tells you which. \(S\) is a sequence, \(\mathcal H\) is a set of hypotheses. Calligraphic capitals name the big sets of the framework: the domain set \(\mathcal X\), the label set \(\mathcal Y\), and the **examples domain** \(\mathcal Z = \mathcal X \times \mathcal Y\). That last one is the set of all possible message-and-label pairs. An example \(z = (x, y)\) is one element of \(\mathcal Z\).

**Distributions and sampling.** \(\mathcal D\) is a distribution over some set, usually \(\mathcal X\) or \(\mathcal Z\). In the mail room, \(\mathcal D\) says how often each of the \(2^{30}\) possible messages arrives. Writing \(z \sim \mathcal D\) means \(z\) is drawn according to \(\mathcal D\). For a yes/no question about \(z\), we write

\[ \mathbb P_{z \sim \mathcal D}[f(z)] = \mathcal D(\{ z : f(z) = \text{true} \}), \]

the probability mass of the set of \(z\) where the answer is yes. \(\mathbb E_{z \sim \mathcal D}[f(z)]\) is the expected value of a number \(f(z)\). And \(S \sim \mathcal D^m\) means the \(m\) examples are drawn independently from the same \(\mathcal D\). We will define \(\mathcal D^m\) properly in a few sections, because a whole proof rests on it.

**Small tools.** \([n]\) is the set \(\{1, \dots, n\}\). The indicator \(\mathbb 1[\text{statement}]\) is 1 when the statement is true and 0 when it is false. And \(\log\) always means the natural logarithm, base \(e\). That matters today, because the formula we are heading for has a \(\log\) in it.

**The loss function.** One more symbol won't be needed until lesson 3, the **loss function** \(\ell : \mathcal H \times \mathcal Z \to \mathbb R_+\). It scores how badly a hypothesis does on one example. Its value is a non-negative real number. For our junk-or-keep problem the natural choice is \(\ell(h, (x, y)) = \mathbb 1[h(x) \neq y]\), one point per mistake. The empirical risk is then just the average loss over the examples. Lesson 3 lets the loss be other things, such as a squared distance.

**Norms.** For a vector \(x\), \(\|x\|\) is its Euclidean length, \(\sqrt{\sum_j x_j^2}\). We will also use \(\|x\|_1 = \sum_j |x_j|\), \(\|x\|_\infty = \max_j |x_j|\), and \(\|x\|_0\), the number of nonzero entries. For a yes/no message, \(\|x\|_1\) and \(\|x\|_0\) are both the number of features that are on, and \(\|x\|\) is the square root of that number. Norms start to matter later in the course, when instances become real-valued vectors, as they do for the linear predictors of lesson 9.

**Asymptotic notation.** We will avoid it where we can, but it is handy for summarizing results, and its definitions are the ones you know from algorithms. For \(f, g : \mathbb R \to \mathbb R_+\):

- \(f = O(g)\) if there are \(x_0, \alpha \in \mathbb R_+\) with \(f(x) \le \alpha g(x)\) for all \(x > x_0\);
- \(f = \Omega(g)\) is the same with \(\ge\);
- \(f = o(g)\) if for every \(\alpha > 0\) there is an \(x_0\) beyond which \(f(x) \le \alpha g(x)\), and \(\omega\) is defined the same way with \(\ge\);
- \(f = \Theta(g)\) means \(f = O(g)\) and \(g = O(f)\);
- \(f = \tilde O(g)\) means \(f = O(g \log^k g)\) for some \(k \in \mathbb N\): big-O that is allowed to hide logarithmic factors.

One last convention. We write \(\min\) and \(\max\) even where a mathematician would insist on \(\inf\) and \(\sup\), because the minimum might not be reached. The distinction rarely matters in this course. Keep it in mind when a "minimum" looks suspicious.

## The framework: who knows what

Now the real content: the **statistical learning framework**. Its whole point is to say precisely what the learner knows and what it doesn't.

Here is a story far from the mail room. You have just arrived on a small Pacific island where papayas are a big part of the diet, and you have never tasted one. You decide to judge them by two features, color and softness, and you taste a few to learn which ones are good. We will come back to the papayas twice. Your situation in the mail room is the same, with messages for papayas and junk for "not tasty". Here are the pieces.

**The learner's input.**

- The **domain set** \(\mathcal X\): every message that could arrive. Its elements are also called **instances**, and \(\mathcal X\) the instance space.
- The **label set** \(\mathcal Y\). For now it has two elements, \(\{0, 1\}\), with 1 for junk.
- The **training data** \(S = ((x_1, y_1), \dots, (x_m, y_m))\), a finite sequence of labelled messages. This is our training sequence, and it is the only thing the learner ever sees. Many texts call it the training set, or the sample, but it is always a sequence: the same example can appear twice, and some learners care about the order.

**The learner's output** is a hypothesis \(h : \mathcal X \to \mathcal Y\). Other names for it are predictor and classifier, and we'll keep saying hypothesis. \(A(S)\) is the hypothesis that learner \(A\) returns on \(S\).

**How the data is made.** This is the new part, so read it slowly. The framework assumes two hidden objects.

1. A probability distribution \(\mathcal D\) over \(\mathcal X\), which decides how often each message arrives. It can be anything at all.
2. A **labelling function** \(f : \mathcal X \to \mathcal Y\), the "correct" label of every message, so that \(y_i = f(x_i)\) for every example.

Each example is made in two steps: draw a message \(x_i\) from \(\mathcal D\), then label it with \(f\). The learner knows neither \(\mathcal D\) nor \(f\). Finding \(f\) is the whole job.

**How success is measured.** The true risk of \(h\) is the probability that \(h\) gets a fresh message wrong:

\[ L_{\mathcal D, f}(h) = \mathbb P_{x \sim \mathcal D}\left[ h(x) \neq f(x) \right] = \mathcal D(\{ x : h(x) \neq f(x) \}). \]

The subscript says the risk depends on both hidden objects. When they are clear we drop it and write \(L_{\mathcal D}(h)\). Other names for it are generalization error, risk and true error. The letter \(L\) is for loss.

Notice what is missing from this list. In lesson 1 we made \(\mathcal D\) uniform, so that true risk was a count out of 16. That was a convenience, and it is gone now. The framework insists that the learner is blind to \(\mathcal D\) and to \(f\), and that the only way it can interact with the world is through \(S\). Any guarantee we prove has to work without knowing what \(\mathcal D\) looks like.

[[diagram:framework]]

## Empirical risk minimization

The learner wants a hypothesis with small true risk, but it cannot compute true risk, because that needs \(\mathcal D\) and \(f\). What it can compute is the empirical risk, the fraction of the training examples \(h\) gets wrong:

\[ L_S(h) = \frac{\left| \{ i \in [m] : h(x_i) \neq y_i \} \right|}{m}. \]

Other names for it are training error and empirical error.

The training sequence is the learner's only snapshot of the world, so it makes sense to look for a hypothesis that does well on it. Returning a hypothesis \(h\) that minimizes \(L_S(h)\) is called **Empirical Risk Minimization**, or **ERM**. You will meet it again, in one form or another, in almost every lesson of this course.

Let's run it by hand. We go back to lesson 1's tiny world of four features, F, L, U, N, because thirty features won't fit on a page. But this week the spammers are cleverer: a message is junk exactly when the subject says "FREE" *and* the sender is unknown. A single feature no longer captures that, so we give the learner a bigger class: every one-feature hypothesis and every two-feature hypothesis,

\[ \mathcal H_{\le 2} = \{ h_j \} \cup \{ h_{jk} : j < k \}, \qquad h_{jk}(x) = x_j \wedge x_k, \]

so \(h_{FU}\) says "junk exactly when the subject says FREE and the sender is unknown". That is 4 one-feature hypotheses plus 6 pairs, ten in all, and the truth \(h_{FU}\) is one of them.

Your users label eight messages.

| i | message FLUN | label |
|---|---|---|
| 1 | 0001 | keep |
| 2 | 1010 | junk |
| 3 | 1111 | junk |
| 4 | 1100 | keep |
| 5 | 1000 | keep |
| 6 | 1001 | keep |
| 7 | 0101 | keep |
| 8 | 1110 | junk |

ERM scores every hypothesis in the class against these eight examples. Before you read the table, try one yourself: which examples does \(h_F\) get wrong? It says junk whenever F is on, so it is wrong on 1100, 1000 and 1001, examples 4, 5 and 6.

| hypothesis | wrong on examples | \(L_S(h)\) | \(L_{\mathcal D}(h)\), uniform \(\mathcal D\) |
|---|---|---|---|
| \(h_F\) | 4, 5, 6 | 3/8 | 4/16 |
| \(h_L\) | 2, 4, 7 | 3/8 | 8/16 |
| \(h_U\) | none | 0 | 4/16 |
| \(h_N\) | 1, 2, 6, 7, 8 | 5/8 | 8/16 |
| \(h_{FL}\) | 2, 4 | 2/8 | 4/16 |
| \(h_{FU}\) | none | 0 | 0 |
| \(h_{FN}\) | 2, 6, 8 | 3/8 | 4/16 |
| \(h_{LU}\) | 2 | 1/8 | 4/16 |
| \(h_{LN}\) | 2, 7, 8 | 3/8 | 6/16 |
| \(h_{UN}\) | 2, 8 | 2/8 | 4/16 |

Two hypotheses tie at empirical risk 0: the truth, \(h_{FU}\), and our old friend \(h_U\), "unknown sender means junk". Look at the examples and you can see why. Every message here with an unknown sender also happens to say "FREE", so nothing in \(S\) separates the two. The learner returns one of them, and it has no way of knowing which is right. If it returns \(h_U\), its true risk is \(4/16 = 0.25\): it wrongly flags the four messages with an unknown sender and no "FREE" in the subject.

Hold on to two lessons from this table. First, the set of minimizers can have more than one member, so a theorem about ERM has to cover *every* hypothesis ERM might return, not just a lucky one. Second, the last column is invisible to the learner. We can only fill it in because we built this world ourselves.

Here is the same ERM in a few lines of Python. Each hypothesis is named by the features it requires, and the last line prints the argmin.

```python
S = [("0001", 0), ("1010", 1), ("1111", 1), ("1100", 0),
     ("1000", 0), ("1001", 0), ("0101", 0), ("1110", 1)]
F = "FLUN"
names = ["F", "L", "U", "N", "FL", "FU", "FN", "LU", "LN", "UN"]
h = lambda n, x: int(all(x[F.index(c)] == "1" for c in n))
L_S = {n: sum(h(n, x) != y for x, y in S) / len(S) for n in names}
print(L_S)
print([n for n in names if L_S[n] == min(L_S.values())])
```

It prints the ten empirical risks of the table, then `['U', 'FU']`: the tie.

## Something may go wrong: overfitting

ERM sounds so natural that it's worth seeing it fail before we go any further.

Back to the papayas. Picture a gray square of area 2, with a smaller blue square of area 1 inside it. Papayas are spread uniformly over the gray square, and a papaya is tasty exactly when it lands in the blue one. So half of all papayas are tasty.

[[diagram:overfit]]

Now consider this hypothesis:

\[ h_S(x) = \begin{cases} y_i & \text{if } x = x_i \text{ for some } i \in [m], \\ 0 & \text{otherwise.} \end{cases} \]

You know it. It is lesson 1's memorizer, word for word: repeat the label of any message you have seen, say "keep" (or "not tasty") for everything else.

Whatever the training sequence, \(L_S(h_S) = 0\). No hypothesis can do better than zero, so ERM is entirely free to return it. And its true risk? It says "tasty" on at most \(m\) points. A finite set of points has area zero, so under the uniform distribution the probability of ever meeting one of those papayas again is zero. The memorizer says "not tasty" on essentially every papaya, which is wrong on the whole blue square: \(L_{\mathcal D}(h_S) = 1/2\).

You can watch it happen. The snippet draws 1,000 papayas, builds the memorizer from them, and scores it on its own examples and on 100,000 fresh ones.

```python
import random
r = random.Random(0)
side, inner = 2 ** 0.5, 1.0
lo, hi = (side - inner) / 2, (side + inner) / 2
f = lambda x: int(lo <= x[0] <= hi and lo <= x[1] <= hi)
draw = lambda: (r.uniform(0, side), r.uniform(0, side))
S = {x: f(x) for x in (draw() for _ in range(1000))}
h_S = lambda x: S.get(x, 0)
print(sum(h_S(x) != y for x, y in S.items()) / len(S))
test = [draw() for _ in range(100000)]
print(sum(h_S(x) != f(x) for x in test) / len(test))
```

It prints `0.0`, then `0.49943`. Perfect on the training sequence, a coin on the world.

That is **overfitting**: a hypothesis that does excellently on the training sequence and very poorly on the world. It fits the training data "too well", a little like a person who has a perfect, detailed explanation for every single thing they did, which tends to raise suspicion.

In the mail room the picture is the same. With thirty features there are \(2^{30} = 1{,}073{,}741{,}824\) possible messages. Suppose they all arrive equally often and junk means "FREE" and unknown sender, which is a quarter of them. A memorizer that has seen 300 labelled messages has seen at most a 0.00003% sliver of the world, and its true risk is at least 0.2499997. Consistent, and nearly as bad as it can be.

## ERM with an inductive bias

We don't give up on ERM. Instead, we look for conditions under which ERM is safe. Safe means that when the ERM hypothesis does well on the training sequence, it is also very likely to do well on \(\mathcal D\).

The fix is the one you met in lesson 1. Before seeing any data, the learner chooses a hypothesis class \(\mathcal H\), and ERM is only allowed to search inside it:

\[ \mathrm{ERM}_{\mathcal H}(S) \in \operatorname*{argmin}_{h \in \mathcal H} L_S(h). \]

Here \(\operatorname{argmin}\) is the set of hypotheses in \(\mathcal H\) that reach the smallest empirical risk, and the \(\in\) says ERM returns some member of that set, any one. The table above computed exactly this set for \(\mathcal H_{\le 2}\): \(\{h_U, h_{FU}\}\).

Restricting the learner this way biases it toward certain hypotheses, and the restriction is called an **inductive bias**, the same term as in lesson 1. Since \(\mathcal H\) is chosen before any data arrives, it should come from prior knowledge about the problem. For the papayas, a good choice is the class of axis-aligned rectangles in the color-softness plane, and ERM over it does not overfit. (Proving that is problem 3 at the end of this lesson.) The memorizer shows the opposite. A class containing every hypothesis that says "tasty" on a finite set of points is too big for ERM to be safe.

That raises a fundamental question of learning theory: over which hypothesis classes does \(\mathrm{ERM}_{\mathcal H}\) avoid overfitting? And there is a tension built into it. A smaller class protects you better against overfitting, but it is a stronger bias, and a stronger bias is more likely to leave the truth out. That is lesson 1's pigeon-and-rat tradeoff again, and lesson 5 makes it exact.

## Finite hypothesis classes

The simplest restriction you can put on a class is a cap on its size. This section shows that if \(\mathcal H\) is finite, \(\mathrm{ERM}_{\mathcal H}\) does not overfit, as long as the training sequence is large enough, and "large enough" depends on \(|\mathcal H|\).

Finite sounds restrictive. It isn't, really. Think of the class of every hypothesis that can be written as a C++ program of at most \(10^9\) bits. That class is finite, and it is enormous. Even the papayas' axis-aligned rectangles, an infinite class, become finite once you store the corner coordinates as 64-bit floating-point numbers, as every real program does.

Here is a menu of finite classes for the thirty-feature mail room.

- **One feature**, \(h_j(x) = x_j\): 30 hypotheses.
- **One or two features**, \(\mathcal H_{\le 2}\): 30 singles plus \(\binom{30}{2} = 435\) pairs, 465 hypotheses.
- **Conjunctions of literals**: rules like "FREE and a link and *not* sent at night", where each feature can be required on, required off, or ignored. That is \(3^{30} + 1 \approx 2.06 \times 10^{14}\) hypotheses (the extra one is the hypothesis that never says junk).
- **Anything goes**: every function from messages to labels, \(2^{2^{30}}\) hypotheses. The memorizer lives here.

All four are finite. The question is which ones a few hundred labels can pay for. To answer it, we need two assumptions about the world.

## Two assumptions: realizability and i.i.d.

The first assumption is about the class. For now we take the simplest case.

**Definition (the realizability assumption).** There exists \(h^\star \in \mathcal H\) with \(L_{(\mathcal D, f)}(h^\star) = 0\).

In words: the class contains a hypothesis that is never wrong, apart from a set of messages that has probability zero. In the tiny world, \(\mathcal H_{\le 2}\) contains \(h_{FU}\), so realizability holds. The one-feature class does not contain it, so for that class realizability fails.

Realizability has an immediate consequence. Since \(h^\star\) errs only on a set of probability zero, a random training sequence almost surely has no example from that set. So \(L_S(h^\star) = 0\) with probability 1. The smallest empirical risk in \(\mathcal H\) is therefore 0, and every ERM hypothesis is consistent: \(L_S(h_S) = 0\). (Strictly, this holds with probability 1, and from now on we leave that qualifier out.) The empirical risk of \(h_S\) is settled, then. What we want to know is its true risk.

The second assumption is about the training sequence. Any guarantee has to depend on how \(S\) relates to \(\mathcal D\). If your users only ever labelled messages from their friends, \(S\) would tell you nothing about junk. The standard assumption is the following.

**The i.i.d. assumption.** The examples are **independently and identically distributed** according to \(\mathcal D\): every \(x_i\) is freshly drawn from \(\mathcal D\), independently of the others, and then labelled by \(f\). We write this \(S \sim \mathcal D^m\). Here \(\mathcal D^m\) is the probability over \(m\)-tuples of messages that you get by drawing each entry from \(\mathcal D\), independently of the other entries.

A good picture for this is a window. \(S\) is a window through which the learner sees part of \(\mathcal D\) and \(f\), and the larger \(S\) is, the more faithfully it shows them.

Now the uncomfortable part. \(S\) is random, so \(h_S\) is random, so its true risk \(L_{(\mathcal D, f)}(h_S)\) is a random variable. No amount of cleverness makes it certain that \(S\) leads to a good hypothesis, because \(S\) might be wildly unrepresentative. Back to the papayas: suppose 70% of the island's papayas are tasty, and every papaya you happened to taste was not. Then ERM may well return the constant hypothesis "never tasty", which is wrong 70% of the time. Unlucky, yes, but possible: with three random papayas it happens with probability \(0.3^3 = 0.027\).

So any promise has to come with two numbers.

- The **accuracy parameter** \(\varepsilon\). If \(L_{(\mathcal D, f)}(h_S) > \varepsilon\) we call it a failure of the learner. If \(L_{(\mathcal D, f)}(h_S) \le \varepsilon\), the hypothesis is approximately correct.
- The **confidence parameter** \(1 - \delta\). Here \(\delta\) bounds the probability of drawing a training sequence that leads to failure.

The promise we are after reads: with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), the true risk of \(h_S\) is at most \(\varepsilon\). In the mail room, take \(\varepsilon = 0.05\) and \(\delta = 0.01\). That would mean at most 5% of fresh messages mislabelled, and at most a 1% chance that your users' labels led you astray.

## Misleading training sequences

How can ERM fail? Let's catch it in the act. Since \(h_S\) is consistent, a failure means that ERM returned a consistent hypothesis whose true risk is above \(\varepsilon\). For that to happen, such a hypothesis has to exist in the first place.

So we name two sets. The **bad hypotheses** are the ones with true risk above \(\varepsilon\):

\[ \mathcal H_B = \{ h \in \mathcal H : L_{(\mathcal D, f)}(h) > \varepsilon \}. \]

And a training sequence is **misleading** if some bad hypothesis is consistent with it. The labels are fixed by \(f\), so all the randomness is in the messages, \(S|_x = (x_1, \dots, x_m)\). The misleading set is therefore a set of message tuples:

\[ M = \{ S|_x : \exists h \in \mathcal H_B, \ L_S(h) = 0 \}. \]

You may also see them called misleading samples. On a misleading training sequence, a bad hypothesis looks exactly as good as the truth.

The tiny-world table showed one. Take \(\varepsilon = 0.2\). Then every hypothesis except \(h_{FU}\) is bad, since all nine wrong ones have true risk 0.25 or more. And the eight examples above form a misleading sequence, because the bad hypothesis \(h_U\) is consistent with them.

[[diagram:misleading]]

Here is the picture to keep in mind. Every point in the big circle is one possible tuple of \(m\) messages. Each oval is the set of tuples on which one particular bad hypothesis is consistent. \(M\) is the union of the ovals. Any training sequence outside every oval is safe: no bad hypothesis survives it, so ERM cannot return one.

Before the proof, get a feel for how big \(M\) is. In the lab, the world has thirty features, \(\mathcal D\) is uniform, the truth is "FREE and unknown sender", and the class is \(\mathcal H_{\le 2}\). You choose \(m\), and the lab draws thousands of training sequences and counts the misleading ones.

[[lab]]

## The proof: one bad hypothesis at a time

Here is the theorem. Call it **the finite-class guarantee**: it is the first real guarantee in the course.

**The finite-class guarantee.** Let \(\mathcal H\) be a finite hypothesis class. Let \(\delta \in (0, 1)\) and \(\varepsilon > 0\), and let \(m\) be an integer with

\[ m \ge \frac{\log(|\mathcal H| / \delta)}{\varepsilon}. \]

Then for any labelling function \(f\) and any distribution \(\mathcal D\) for which realizability holds, with probability at least \(1 - \delta\) over the choice of an i.i.d. training sequence \(S\) of size \(m\), every ERM hypothesis \(h_S\) satisfies \(L_{(\mathcal D, f)}(h_S) \le \varepsilon\).

We'll prove the bound it rests on, one step at a time:

\[ \mathcal D^m(\{ S|_x : L_{(\mathcal D, f)}(h_S) > \varepsilon \}) \le |\mathcal H| \, e^{-\varepsilon m}. \]

Fix \(f\), \(\mathcal D\), \(\varepsilon\) and \(m\), and assume realizability.

**Step 1: every failure happens on a misleading sequence.** Suppose \(L_{(\mathcal D, f)}(h_S) > \varepsilon\). Then \(h_S\) is in \(\mathcal H_B\), by definition. Realizability gives \(L_S(h_S) = 0\). So \(h_S\) is a bad hypothesis that is consistent with \(S\), and \(S|_x \in M\). We have shown

\[ \{ S|_x : L_{(\mathcal D, f)}(h_S) > \varepsilon \} \subseteq M. \]

A set can't have more probability than a set that contains it, so the failure probability is at most \(\mathcal D^m(M)\). Notice that we never asked *which* consistent hypothesis ERM picked. That is why the guarantee holds for every ERM hypothesis, ties included.

**Step 2: split \(M\) by hypothesis.** A tuple is misleading when at least one bad hypothesis survives it, so \(M\) is the union of one set per bad hypothesis. In symbols:

\[ M = \bigcup_{h \in \mathcal H_B} \{ S|_x : L_S(h) = 0 \}. \]

Each set in the union is one oval in the figure. Combining with step 1 gives

\[ \mathcal D^m(\{ S|_x : L_{(\mathcal D, f)}(h_S) > \varepsilon \}) \le \mathcal D^m\Big( \bigcup_{h \in \mathcal H_B} \{ S|_x : L_S(h) = 0 \} \Big). \]

**Step 3: the union bound.** The ovals overlap, and working out the exact area of their union is hard. So we overestimate it by adding up their areas, which counts the overlaps more than once.

**The union bound.** For any two sets \(A, B\) and any distribution \(\mathcal D\), \(\mathcal D(A \cup B) \le \mathcal D(A) + \mathcal D(B)\).

It is a basic property of probability, and the reason is short. \(A \cup B\) is the disjoint union of \(A\) and \(B \setminus A\), so \(\mathcal D(A \cup B) = \mathcal D(A) + \mathcal D(B \setminus A)\), and \(\mathcal D(B \setminus A) \le \mathcal D(B)\) because \(B \setminus A\) sits inside \(B\).

We need it for \(|\mathcal H_B|\) sets, not two. Apply it repeatedly: \(\mathcal D(A_1 \cup \dots \cup A_k) \le \mathcal D(A_1 \cup \dots \cup A_{k-1}) + \mathcal D(A_k)\), and so on down to a single set.

Here are two real ovals from the tiny world: \(A\), the training sequences of 5 messages on which \(h_U\) is consistent, and \(B\), the same for \(h_F\). The snippet estimates \(\mathcal D^m(A \cup B)\) and \(\mathcal D^m(A) + \mathcal D^m(B)\).

```python
import random
r = random.Random(2)
X = [format(v, "04b") for v in range(16)]
f = lambda x: x[0] == x[2] == "1"
h_U = lambda x: x[2] == "1"
h_F = lambda x: x[0] == "1"
m, runs = 5, 200000
A = B = AorB = 0
for _ in range(runs):
    S = r.choices(X, k=m)
    a = all(h_U(x) == f(x) for x in S)
    b = all(h_F(x) == f(x) for x in S)
    A, B, AorB = A + a, B + b, AorB + (a or b)
print(round(AorB / runs, 4), round((A + B) / runs, 4))
```

It prints `0.4439` and `0.475`. The exact values are 0.4434 and 0.4746, and the gap, 0.0313, is the overlap \(\mathcal D^m(A \cap B) = 0.5^5\), counted twice by the sum.

With the union bound in hand, the failure probability is at most a sum, one term per oval:

\[ \mathcal D^m(\{ S|_x : L_{(\mathcal D, f)}(h_S) > \varepsilon \}) \le \sum_{h \in \mathcal H_B} \mathcal D^m(\{ S|_x : L_S(h) = 0 \}). \]

The problem has shrunk to one bad hypothesis at a time.

**Step 4: one bad hypothesis, through independence.** Fix one \(h \in \mathcal H_B\). It is consistent exactly when it agrees with \(f\) on every message: \(h(x_i) = f(x_i)\) for all \(i\). Under \(\mathcal D^m\) the messages are drawn independently, so the probability that all \(m\) agreements happen is the product of the \(m\) separate probabilities. This is the only place the proof uses the i.i.d. assumption:

\[ \mathcal D^m(\{ S|_x : L_S(h) = 0 \}) = \mathcal D^m(\{ S|_x : \forall i, \ h(x_i) = f(x_i) \}) = \prod_{i=1}^m \mathcal D(\{ x_i : h(x_i) = f(x_i) \}). \]

**Step 5: each factor is small because \(h\) is bad.** One draw agrees with \(f\) exactly when it does not land where \(h\) is wrong, so

\[ \mathcal D(\{ x : h(x) = f(x) \}) = 1 - L_{(\mathcal D, f)}(h) \le 1 - \varepsilon. \]

The inequality holds because \(h \in \mathcal H_B\) means \(L_{(\mathcal D, f)}(h) > \varepsilon\). All \(m\) factors are the same, so the product is at most \((1 - \varepsilon)^m\).

Steps 4 and 5 are easy to check by simulation. Take the bad hypothesis \(h_U\) from the tiny world, whose true risk is 0.25, and draw 200,000 training sequences of 10 messages.

```python
import math, random
r = random.Random(1)
X = [format(v, "04b") for v in range(16)]
f = lambda x: x[0] == x[2] == "1"
h_U = lambda x: x[2] == "1"
L = sum(h_U(x) != f(x) for x in X) / 16
m, runs = 10, 200000
S_all = (r.choices(X, k=m) for _ in range(runs))
hits = sum(all(h_U(x) == f(x) for x in S) for S in S_all)
print(L, round(hits / runs, 4))
print(round((1 - L) ** m, 4), round(math.exp(-L * m), 4))
```

It prints `0.25`, then about `0.0558` for the simulated chance that \(h_U\) survives, then the product \(0.75^{10} = 0.0563\), then \(e^{-2.5} = 0.0821\), which is where step 6 is heading.

**Step 6: trade the power for an exponential.** Now we use the inequality

\[ 1 - \varepsilon \le e^{-\varepsilon}. \]

It holds for every real number, and the proof is a little calculus. Let \(g(a) = e^{-a} - (1 - a)\). Then \(g(0) = 0\), and \(g'(a) = 1 - e^{-a}\), which is negative for \(a < 0\) and positive for \(a > 0\). So \(g\) falls until 0 and rises after it, its smallest value is \(g(0) = 0\), and \(g(a) \ge 0\) everywhere. Geometrically, the line \(1 - a\) is the tangent to the curve \(e^{-a}\) at \(a = 0\), and the curve bends upward away from it.

A few values make it concrete, and the `assert` checks the inequality at each one, negative \(a\) included.

```python
import math
for eps in [-0.5, 0, 0.05, 0.2, 0.5, 1, 2]:
    assert 1 - eps <= math.exp(-eps)
    print(eps, round(1 - eps, 4), round(math.exp(-eps), 4))
```

At \(\varepsilon = 0.05\) it prints 0.95 against 0.9512, and at 0.5 it prints 0.5 against 0.6065. The two sides agree at 0 and drift apart as \(\varepsilon\) grows.

[[diagram:tangent]]

To raise both sides to the power \(m\), we need both to be non-negative, since \(t \mapsto t^m\) only preserves order for \(t \ge 0\). The right side, \(e^{-\varepsilon}\), always is. The left side is too whenever \(\mathcal H_B\) has any members at all: a bad hypothesis has \(\varepsilon < L_{(\mathcal D, f)}(h) \le 1\), so \(1 - \varepsilon > 0\). (If \(\varepsilon \ge 1\), there are no bad hypotheses and the failure probability is 0.) So for every \(h \in \mathcal H_B\), we get

\[ \mathcal D^m(\{ S|_x : L_S(h) = 0 \}) \le (1 - \varepsilon)^m \le e^{-\varepsilon m}. \]

Each oval has probability at most \(e^{-\varepsilon m}\), and every oval shrinks exponentially as \(m\) grows.

**Step 7: add up the ovals.** Put the step 6 bound into each of the \(|\mathcal H_B|\) terms of the step 3 sum:

\[ \mathcal D^m(\{ S|_x : L_{(\mathcal D, f)}(h_S) > \varepsilon \}) \le |\mathcal H_B| \, e^{-\varepsilon m} \le |\mathcal H| \, e^{-\varepsilon m}. \]

The last step simply uses \(\mathcal H_B \subseteq \mathcal H\). We don't know how many hypotheses are bad, since that depends on \(\mathcal D\), but there can't be more of them than there are hypotheses.

**Step 8: solve for \(m\).** We want the failure probability to be at most \(\delta\), so it is enough that \(|\mathcal H| e^{-\varepsilon m} \le \delta\). Take logarithms, which preserves order, and rearrange:

\[ |\mathcal H| e^{-\varepsilon m} \le \delta \iff \log|\mathcal H| - \varepsilon m \le \log \delta \iff m \ge \frac{\log(|\mathcal H| / \delta)}{\varepsilon}. \]

That is the guarantee's condition. In code, for the class of 465 hypotheses at \(\varepsilon = 0.05\) and \(\delta = 0.01\), the snippet finds the smallest such \(m\) and checks that it is exactly where the bound drops to \(\delta\):

```python
import math
H, eps, delta = 465, 0.05, 0.01
m = math.ceil(math.log(H / delta) / eps)
bound = lambda m: H * math.exp(-eps * m)
print(m, bound(m) <= delta, bound(m - 1) <= delta)
```

It prints `215 True False`: at 215 examples \(|\mathcal H| e^{-\varepsilon m} \le \delta\), and at 214 it is not yet.

When the condition holds, the probability of failure is at most \(\delta\), so with probability at least \(1 - \delta\), \(L_{(\mathcal D, f)}(h_S) \le \varepsilon\). \(\blacksquare\)

Look at what the proof never used. It never asked what \(\mathcal D\) looks like, or what \(f\) is beyond being realizable, or which consistent hypothesis ERM returned. That is why the guarantee holds for every distribution at once.

In one sentence: for large enough \(m\), \(\mathrm{ERM}_{\mathcal H}\) over a finite class is **probably** (with confidence \(1 - \delta\)) **approximately** (up to an error of \(\varepsilon\)) **correct**. The next lesson turns that phrase into a definition.

## Worked example 2: where the slack goes

Every step of the proof gave up a little accuracy. Let's measure how much, in the tiny world, where we can compute everything exactly. The setting is the one from the ERM table: \(\mathcal D\) uniform on the 16 messages, truth \(h_{FU}\), class \(\mathcal H_{\le 2}\) with \(|\mathcal H| = 10\), and \(\varepsilon = 0.2\). Nine hypotheses are bad: six with true risk 0.25, one with 0.375 and two with 0.5.

Each column below is the bound after one more step of the proof. Read a row from left to right and watch each number grow.

| \(m\) | exact \(\mathcal D^m(M)\) | step 3: \(\sum_{h \in \mathcal H_B} (1 - L_{\mathcal D}(h))^m\) | step 5: \(\lvert\mathcal H_B\rvert (1 - \varepsilon)^m\) | step 6: \(\lvert\mathcal H_B\rvert e^{-\varepsilon m}\) | step 7: \(\lvert\mathcal H\rvert e^{-\varepsilon m}\) |
|---|---|---|---|---|---|
| 5 | 0.7514 | 1.5817 | 2.9491 | 3.3109 | 3.6788 |
| 10 | 0.2692 | 0.3489 | 0.9664 | 1.2180 | 1.3534 |
| 15 | 0.0737 | 0.0811 | 0.3167 | 0.4481 | 0.4979 |
| 20 | 0.0184 | 0.0191 | 0.1038 | 0.1648 | 0.1832 |
| 25 | 0.0045 | 0.0045 | 0.0340 | 0.0606 | 0.0674 |
| 30 | 0.0011 | 0.0011 | 0.0111 | 0.0223 | 0.0248 |
| 35 | 0.0003 | 0.0003 | 0.0037 | 0.0082 | 0.0091 |

You can reproduce the row for \(m = 20\) yourself. The snippet builds the class, finds the nine bad hypotheses, estimates \(\mathcal D^m(M)\) from 100,000 training sequences, and prints three of the bounds.

```python
import math, random
r = random.Random(3)
X = [format(v, "04b") for v in range(16)]
f = lambda x: x[0] == x[2] == "1"
H = [(a,) for a in range(4)]
H += [(a, b) for a in range(4) for b in range(a + 1, 4)]
h = lambda c, x: all(x[j] == "1" for j in c)
L = {c: sum(h(c, x) != f(x) for x in X) / 16 for c in H}
eps, m, runs = 0.2, 20, 100000
H_B = [c for c in H if L[c] > eps]
ok = lambda c, S: all(h(c, x) == f(x) for x in S)
draws = (r.choices(X, k=m) for _ in range(runs))
M = sum(any(ok(c, S) for c in H_B) for S in draws)
print(len(H_B), round(M / runs, 4))
print(round(sum((1 - L[c]) ** m for c in H_B), 4),
      round(len(H_B) * (1 - eps) ** m, 4),
      round(len(H) * math.exp(-eps * m), 4))
```

It prints `9 0.0184`, matching the exact 0.0184, then 0.0191, 0.1038 and 0.1832, as in the table.

Three things are worth noticing.

The union bound, step 3, costs a lot when \(m\) is small, because the ovals overlap heavily. At \(m = 5\) it says "at most 1.58", which is no information at all about a probability. Once \(m\) is large the ovals are small and rarely overlap, and the sum is almost exact.

The biggest loss is step 5, which treats every bad hypothesis as if its true risk were exactly \(\varepsilon = 0.2\). The real bad hypotheses are worse than that, wrong at least 25% of the time, so they die faster than the bound assumes. The proof can't use that, because it doesn't know \(\mathcal D\).

And the guarantee's own answer: \(\log(10 / 0.01) / 0.2 = 34.5\), so \(m = 35\) is enough for \(\delta = 0.01\). At 35 examples the bound says 0.0091, safely under 0.01, while the true chance of a misleading sequence is 0.00025. The bound is loose in this world. It has to be, since it must also hold in the worst world: one where a single bad hypothesis has true risk just above \(\varepsilon\). There, step 3 is exact, step 5 is nearly exact, and step 6 is all that remains.

## What a label costs

Now we can answer your manager. The finite-class guarantee says how many examples are enough for any finite class, and the striking part is where \(|\mathcal H|\) appears: inside a logarithm. Doubling the class adds only \(\log 2 / \varepsilon\) examples. A class gets expensive only when its size grows exponentially.

Take \(\varepsilon = 0.05\) and \(\delta = 0.01\), and the menu from before.

| class | \(\lvert\mathcal H\rvert\) | \(\log \lvert\mathcal H\rvert\) | examples enough for the guarantee |
|---|---|---|---|
| one feature | 30 | 3.40 | 161 |
| one or two features | 465 | 6.14 | 215 |
| conjunctions of literals | \(3^{30} + 1 \approx 2.06 \times 10^{14}\) | 32.96 | 752 |
| anything goes | \(2^{2^{30}}\) | about \(7.44 \times 10^8\) | about \(1.49 \times 10^{10}\) |

The table comes from a few lines of arithmetic. The snippet works with \(\log|\mathcal H|\) directly, since \(2^{2^{30}}\) is too large to write down.

```python
import math
def m_needed(log_H, eps, delta):
    return math.ceil((log_H + math.log(1 / delta)) / eps)
eps, delta = 0.05, 0.01
log_H = {"one feature": math.log(30),
         "one or two features": math.log(30 + math.comb(30, 2)),
         "conjunctions of literals": math.log(3 ** 30 + 1),
         "anything goes": 2 ** 30 * math.log(2)}
for name, v in log_H.items():
    print(name, round(v, 2), m_needed(v, eps, delta))
```

It prints 161, 215, 752 and 14,885,222,452.

With a budget of a few hundred labels, the first two are affordable and the last two are not. Conjunctions of literals cost about 750 labels. Anything goes asks for about fifteen billion, more than ten times the number of distinct messages that exist. And the C++ class from earlier, programs of at most \(10^9\) bits? Its log-size is about \(6.93 \times 10^8\), almost as hopeless as anything goes. That's what such a broad bias costs.

[[diagram:budget]]

But affordable is not the same as safe. The guarantee assumes realizability. If the spammers' rule is "FREE and unknown sender", the one-feature class does not contain it, and the guarantee promises nothing about it. In fact, every one-feature hypothesis then has true risk at least 0.25, under the uniform distribution. You will see what that looks like in the lab: after enough examples, ERM over that class finds no consistent hypothesis at all. That is lesson 1's rat failure again. A class too small to contain the truth can't be rescued with labels.

So the decision has two halves. First, choose a class that contains the truth, or that you have good reason to believe does. Second, choose the smallest such class, because its log-size is what you pay for in labels.

## Fix the bug

The spammers have moved again. This week's rule uses two features, and you don't know which. Your manager has set a target for accuracy and confidence, and you have a budget of 300 labels.

In the lab you choose the hypothesis class the learner searches and how many labels to buy. The lab runs ERM on a fresh training sequence and tells you what happened. It also tells you whether the finite-class guarantee backs your choice. Your goal: a choice the guarantee covers, within budget. Each "Try another" brings a new rule and a new target.

[[lab-act2]]

## ERM over a finite class, as code

For a finite class, ERM is a brute-force search: score every hypothesis on the training sequence and keep one with the fewest mistakes.

```
ERM(H, S):
  best ← none
  best_errors ← m + 1
  for h in H:
    errors ← 0
    for i = 1..m:
      if h(x_i) ≠ y_i: errors ← errors + 1
    if errors < best_errors:
      best ← h
      best_errors ← errors
  return best
```

Ties go to the first hypothesis found, which is one legitimate member of the argmin. Under realizability, the loop could stop at the first \(h\) with zero errors.

**Time.** The search makes \(m \cdot |\mathcal H|\) hypothesis evaluations. If one evaluation costs \(T\), the total is \(O(m \, |\mathcal H| \, T)\). For a pair hypothesis \(T = O(1)\): read two bits. So ERM over the 465 hypotheses of \(\mathcal H_{\le 2}\) with 215 examples is 99,975 evaluations, nothing at all. For conjunctions of literals, it would be about \(1.5 \times 10^{17}\) evaluations: the class is affordable in theory, hopeless as brute force. Whether a cleverer search exists is a separate question, and lesson 8 is about exactly that tension.

**Space.** The loop keeps one best hypothesis and one counter, so beyond the input it uses \(O(1)\) words plus whatever it takes to represent one hypothesis. This assumes the class can be enumerated one hypothesis at a time, as the singles and pairs can with two nested loops. Storing \(\mathcal H\) as a list would cost \(O(|\mathcal H|)\) instead.

**Build it yourself.** Using only the Python standard library:

```python
def erm(H, S): ...
def pair_class(d): ...
def true_risk_uniform(h, f, d): ...
def labels_needed(size_H, eps, delta): ...
```

1. Implement `erm`, where `H` is a list of functions and `S` is a list of `(x, y)` pairs with `x` a tuple of bits. Return every member of the argmin, not just one.
2. Implement `pair_class(d)`, which returns the singles and pairs, and check that it has 465 members for \(d = 30\).
3. Reproduce the ERM table of the tiny world, including the tie between \(h_U\) and \(h_{FU}\).
4. For \(d = 4\), truth \(h_{FU}\), and \(\varepsilon = 0.2\), estimate \(\mathcal D^m(M)\) by simulation for \(m = 5, 10, \dots, 35\), and compare with the "exact" column of worked example 2.
5. Implement `labels_needed` and reproduce the label-cost table. Then make the simulation in step 4 work for \(d = 30\). (A message fits in one integer, and a pair hypothesis is a two-bit mask.) Find the smallest \(m\) at which fewer than 1% of training sequences are misleading. Compare it with the guarantee's 215.

## What people get wrong

**"The guarantee says I need 215 labels, so with 200 the learner fails."** The condition is sufficient, not necessary. Below 215 the guarantee simply says nothing. In the lab's uniform world, misleading sequences are already rare at 30 or 40 examples. The 215 buys a promise that holds for *every* distribution, including ones you would never guess.

**"A bigger class needs proportionally more data."** The cost grows with \(\log |\mathcal H|\), not with \(|\mathcal H|\). Going from 30 hypotheses to 465, fifteen times as many, raises the requirement from 161 to 215. What breaks the budget is a class whose size is exponential in the number of features, such as \(3^{30}\) or \(2^{2^{30}}\).

**"The guarantee is about the hypothesis I got."** It is about the procedure. The probability is over the draw of \(S\). Think of all the training sequences your users might have labelled: at most a \(\delta\) fraction of them lead ERM to a hypothesis with true risk above \(\varepsilon\). Once you hold one particular \(h_S\), the guarantee can't tell you whether your \(S\) was one of the unlucky ones.

**"\(\varepsilon\) and \(\delta\) are interchangeable knobs."** They have very different prices. From the 215 at \(\varepsilon = 0.05\) and \(\delta = 0.01\), halving \(\varepsilon\) doubles the requirement to 430. Making \(\delta\) ten times smaller only adds \(\log 10 / \varepsilon \approx 46\) examples, giving 261. Confidence is cheap; accuracy is not.

## Recap and the next bug

1. The statistical learning framework: messages come from an unknown distribution \(\mathcal D\) and are labelled by an unknown function \(f\). The learner sees only \(S\), and is judged by its true risk \(L_{(\mathcal D, f)}(h)\).
2. ERM returns a hypothesis with the smallest empirical risk. Without a restriction it can overfit, as the memorizer does, with \(L_S = 0\) and \(L_{\mathcal D} = 1/2\) in the papaya example.
3. \(\mathrm{ERM}_{\mathcal H}\) searches only a class fixed in advance: an inductive bias.
4. Under realizability and i.i.d. sampling, ERM over a finite class fails only on misleading training sequences. The union bound and \(1 - \varepsilon \le e^{-\varepsilon}\) bound their probability by \(|\mathcal H| e^{-\varepsilon m}\).
5. So \(m \ge \log(|\mathcal H| / \delta) / \varepsilon\) examples make ERM probably approximately correct. Labels pay for the logarithm of the class size, but only if the class contains the truth.

The filter ships, and for a while it's quiet. Then the complaints change character. One user marks the weekly newsletter as junk. Another user, reading the very same newsletter, marks it keep. Both are right, for themselves. Now look at your training sequence: the same message appears twice, with two different labels. No hypothesis in any class can agree with both, so no hypothesis is consistent with every example, and realizability is dead. ERM will still return something, but every step of today's proof started from \(L_S(h_S) = 0\). What should a learner return when no hypothesis fits every example, and what can anyone promise about it? That is the next lesson.

## Concept check

Answer these in the chat before we move on to lesson 3. Write down your reasoning, not just the answer.

1. Your class is the thirty one-feature hypotheses, and this week's junk rule is "FREE and unknown sender". You buy 161 labels, which is exactly what the guarantee asks for at \(\varepsilon = 0.05\) and \(\delta = 0.01\) for a class of 30. ERM returns some hypothesis. What can you say about its true risk, and what does the guarantee promise? Which step of the proof breaks, and how could you notice the problem from the training sequence alone?
2. In step 4 the proof writes the probability that a bad hypothesis is consistent as a product of \(m\) factors. Which assumption allows that, and what would go wrong if your users chose which messages to label, say, by labelling only messages that arrived from a newsletter they read? Separately: in step 7 the proof replaces \(|\mathcal H_B|\) by \(|\mathcal H|\). Why can't it keep \(|\mathcal H_B|\)?
3. With the class \(\mathcal H_{\le 2}\) of 465 hypotheses, how many examples does the guarantee ask for at \(\varepsilon = 0.02\) and \(\delta = 0.05\)? Is that within a budget of 300? If not, which is the cheaper way to get back under budget: relax \(\varepsilon\) to 0.05, or relax \(\delta\) to 0.5?
