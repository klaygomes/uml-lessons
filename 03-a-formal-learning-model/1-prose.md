# A Formal Learning Model

## The learner that gave up

Last time you gave the filter a real guarantee. You fixed a finite hypothesis class in advance and ran ERM on it: empirical risk minimization, which returns the hypothesis in the class with the fewest mistakes on the labelled examples. The finite-class bound told you how many labelled examples to ask for so that, with high probability, the returned hypothesis had small true risk, meaning a small chance of mislabelling the next message to arrive. You shipped it, and for a while the mail room was quiet.

Then two things happened in the same week.

First, the newsletters. The company’s design team signs up for a weekly digest that arrives with a link, from a sender nobody has in their contacts. Half the design team reads it every Monday. Most people in finance click “junk” the moment it lands. Same message, same four features, opposite labels. One person’s newsletter is another person’s junk.

Second, the honest mistakes. People label in a hurry. Somebody marks a “FREE” prize draw as keep because they were scrolling past it with their coffee in the other hand. Nobody is lying; they are just people.

The learner from last lesson doesn’t cope. Its proof leaned on one assumption, **realizability**: some hypothesis in the class has true risk 0, so it agrees with every example. Now no hypothesis does. Not one of the four one-feature hypotheses. Not even the memorizer: the same message turns up once as junk and once as keep, and a hypothesis has to give it one label. A learner written to return a consistent hypothesis, one with no mistakes on the examples, now returns nothing at all. A learner that breaks ties on its own returns something arbitrary.

So what *should* a learner return when the labels themselves disagree? And, the harder question, what could you possibly promise about it? This lesson answers both, and along the way it turns lesson 2’s single result into the definition everything after it is built on.

## What the last learner promised

Before we break the old promise, let’s write it down carefully, because the new one will be built in its image.

Recall the setting of lesson 2. Messages arrive from a distribution \(\mathcal D\) over the domain set \(\mathcal X\), the set of every possible message (16 of them in lesson 1’s four-feature world). A labelling function \(f: \mathcal X \to \{0, 1\}\) gives each message its true label. The learner sees a training sequence \(S\) of \(m\) examples drawn i.i.d. (independently, each from the same distribution), which we write \(S \sim \mathcal D^m\), and returns a hypothesis. (The tilde \(\sim\) reads “is drawn from”, and \(\mathcal D^m\) means \(m\) independent draws from \(\mathcal D\), one per example.) Its true risk is \(L_{(\mathcal D, f)}(h)\), the probability that \(h\) disagrees with \(f\) on a fresh message; the subscript \((\mathcal D, f)\) names the two things that risk depends on, where messages come from and who labels them.

Lesson 2 proved a guarantee for ERM over a finite class \(\mathcal H\) when realizability holds. Pick a tolerated true risk, say 0.1, and a tolerated chance of failure, say 0.05. Once the number of examples passed a threshold, 44 for lesson 1’s four one-feature hypotheses, the returned hypothesis had true risk at most 0.1 on all but 5% of training sequences. The threshold grows slowly with the size of the class and quickly as the tolerance shrinks; its exact formula comes back in the next section. Call the tolerated true risk \(\varepsilon\) (the Greek letter epsilon) and the tolerated failure chance \(\delta\) (delta). Then the guarantee reads: true risk at most \(\varepsilon\), with probability at least \(1 - \delta\). That is two promises nested inside each other, and their names give the whole theory its title.

- **Approximately correct.** The returned hypothesis is not required to be perfect, only to have true risk at most \(\varepsilon\). The number \(\varepsilon\) is the **accuracy parameter**.
- **Probably.** Even that is not promised every time, only with probability at least \(1 - \delta\) over the random draw of the training sequence. The number \(\delta\) is the **confidence parameter**. (Lesson 2 gave that name to \(1 - \delta\), the chance of success. From here on it goes to \(\delta\), the allowed chance of failure. The promise is the same either way, and this lesson always states it in full: with probability at least \(1 - \delta\).)

Probably approximately correct: **PAC**.

To turn that into a definition, we need one more ingredient: a recipe that, told only \(\varepsilon\) and \(\delta\), answers “how many examples?”. For the one-feature class, the recipe answers 44 when you ask for \(\varepsilon = 0.1\) and \(\delta = 0.05\). The definition calls that recipe \(m_{\mathcal H}\), with the class as a subscript because each class gets its own recipe.

**Definition (PAC learnability).** A hypothesis class \(\mathcal H\) is **PAC learnable** if there exist a function \(m_{\mathcal H}: (0, 1)^2 \to \mathbb N\) and a learning algorithm with the following property. For every \(\varepsilon, \delta \in (0, 1)\), for every distribution \(\mathcal D\) over \(\mathcal X\), and for every labelling function \(f: \mathcal X \to \{0, 1\}\), if the realizability assumption holds with respect to \(\mathcal H\), \(\mathcal D\), \(f\), then when the algorithm runs on \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples generated by \(\mathcal D\) and labelled by \(f\), it returns a hypothesis \(h\) such that, with probability at least \(1 - \delta\) over the choice of the examples, \(L_{(\mathcal D, f)}(h) \le \varepsilon\).

The definition is one long sentence, so let’s take its notation apart before its logic.

- \((0, 1)\) is the open interval: every number strictly between 0 and 1. The exponent in \((0, 1)^2\) means a pair of such numbers, here \((\varepsilon, \delta)\).
- \(\mathbb N\) is the natural numbers \(1, 2, 3, \dots\), the possible counts of examples.
- The arrow in \(m_{\mathcal H}: (0, 1)^2 \to \mathbb N\) reads “takes … and returns …”: \(m_{\mathcal H}\) takes a pair \((\varepsilon, \delta)\) and returns a whole number of examples.
- \(\varepsilon, \delta \in (0, 1)\) reads “\(\varepsilon\) and \(\delta\) belong to the open interval”: neither may be 0, neither may be 1.
- \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) says the learner gets at least as many examples as the function asks for.
- \(L_{(\mathcal D, f)}(h) \le \varepsilon\) is the “approximately correct” part, and “with probability at least \(1 - \delta\)” wraps it in the “probably” part. That probability is over the random draw \(S \sim \mathcal D^m\).

In plain words: **there is a recipe that turns any accuracy and confidence you ask for into a number of examples, and with that many examples the learner is, on all but a \(\delta\) share of training sequences, within \(\varepsilon\) of perfect, whatever \(\mathcal D\) and \(f\) turn out to be.** With \(\varepsilon = 0.1\) and \(\delta = 0.05\), the recipe for lesson 1’s one-feature class says 44 examples, and the promise reads: at least 95 runs in 100 return a hypothesis whose true risk is at most 0.1. The simulation in the next section runs exactly that test, line by line: `misses` counts the runs where the true risk exceeds \(\varepsilon\), and the final `assert` checks that their share stays at or below \(\delta\).

Read the quantifiers slowly, because they are where the strength lives. The function \(m_{\mathcal H}\) gets to see \(\varepsilon\) and \(\delta\), and nothing else. It is chosen before anyone tells you what \(\mathcal D\) is or which \(f\) labels the messages, and it has to work for all of them. That is the distribution-free ambition from lesson 1, now written as a definition.

[[diagram:pac]]

Why two parameters, and not just “the learner returns the right hypothesis”? There are two reasons, and you have already watched both happen. The training sequence is random, so there is always a small chance it is useless. Imagine a training sequence that contains only one message, drawn over and over again. No learner can do anything with that, which is why \(\delta\) can never be zero. And even when the training sequence does reflect \(\mathcal D\) faithfully, it is finite, so some fine details of \(\mathcal D\) will be missing from it. The accuracy parameter forgives the learner for small mistakes on those details, which is why \(\varepsilon\) can never be zero either.

## Sample complexity

The function \(m_{\mathcal H}\) is the **sample complexity** of learning \(\mathcal H\): how many examples guarantee a probably approximately correct answer. It depends on \(\varepsilon\), on \(\delta\), and on properties of the class. If one function works, any larger function works too, so there are many valid choices. We pin it down by taking the minimal one. For each \(\varepsilon\) and \(\delta\), \(m_{\mathcal H}(\varepsilon, \delta)\) is the smallest integer that meets the requirements of PAC learning with that accuracy and confidence.

With that, lesson 2’s result becomes a one-line statement.

**The finite-class guarantee.** Every finite hypothesis class is PAC learnable with sample complexity

\[ m_{\mathcal H}(\varepsilon, \delta) \le \left\lceil \frac{\log(|\mathcal H|/\delta)}{\varepsilon} \right\rceil. \]

Read it from the inside out.

- \(|\mathcal H|\) is how many hypotheses the class has: the vertical bars around a set mean “how many elements”.
- \(|\mathcal H|/\delta\) divides that count by the allowed failure probability. A smaller \(\delta\) makes this ratio bigger.
- \(\log\) is the natural logarithm, the power you must raise \(e \approx 2.718\) to in order to get its input. It is what keeps the bound gentle: multiplying the ratio by 10 only adds about 2.3 to its log.
- Dividing by \(\varepsilon\) scales that up: asking for ten times the accuracy costs ten times the examples.
- The brackets \(\lceil \cdot \rceil\) are the ceiling, “round up to the next whole number”, because you cannot collect 43.8 examples.
- \(\le\) says this is an upper bound on the minimal sample complexity, not its exact value.

In plain words: **the number of examples you need is at most the log of “class size over failure chance”, divided by the accuracy, rounded up.** With \(|\mathcal H| = 4\), \(\delta = 0.05\) and \(\varepsilon = 0.1\): the ratio is \(4/0.05 = 80\), its log is about 4.38, divided by 0.1 that is 43.8, and the ceiling makes it 44. The first three lines of the snippet below are this formula term by term, `math.ceil(math.log(H / delta) / eps)`, and the first number it prints is that 44.

*Proof.* Take ERM over \(\mathcal H\) as the algorithm. Lesson 2’s bound says that every integer \(m \ge \log(|\mathcal H|/\delta)/\varepsilon\) meets the requirement. The smallest such integer is the ceiling on the right, and the minimal function can only be smaller. \(\blacksquare\)

Put numbers in. In lesson 1’s tiny world, the class of one-feature hypotheses has \(|\mathcal H| = 4\). Ask for \(\varepsilon = 0.1\) and \(\delta = 0.05\), and the bound says 44 labelled examples are enough. The real mail system has 30 features, so a one-feature class has 30 members, and the bound grows only to 64. That gentle growth comes from the logarithm: doubling the class adds a fixed number of examples instead of doubling them.

Here is the bound, and then the claim itself, simulated in lesson 1’s world. There are four features, all messages are equally likely, and a message is junk exactly when the subject says “FREE”. ERM over the one-feature class picks one of its consistent hypotheses at random, as lesson 1’s ConsistentFeature did. A miss is a run whose true risk is above \(\varepsilon\). An impostor, as in lesson 1, is a wrong feature that happens to agree with every example; it has true risk 0.5, so every impostor is a miss.

```python
import math, random
eps, delta = 0.1, 0.05
for H in (4, 30):
    print(H, math.ceil(math.log(H / delta) / eps))
rng = random.Random(1)
def erm(S):
    H = [j for j in range(4) if all(x[j] == y for x, y in S)]
    return rng.choice(H)
for m in (6, 44):
    misses = 0
    for _ in range(20000):
        xs = [rng.choices((0, 1), k=4) for _ in range(m)]
        S = [(x, x[0]) for x in xs]
        misses += (0.0 if erm(S) == 0 else 0.5) > eps
    print(m, round(misses / 20000, 3))
assert misses / 20000 <= delta
```

It prints 44 and 64. Then come the miss rates over 20,000 runs: 0.023 (about 2.3%) with only 6 examples, and 0.0 with the 44 the bound asks for, well under \(\delta = 0.05\). The bound is safe, and here it is generous. A wrong feature agrees with the true label on a random message half the time, so it survives 44 examples with chance \(1/2\) multiplied by itself 44 times, written \(2^{-44}\) (about one in 18 trillion). There are 3 wrong features, so an impostor survives with probability at most \(3 \cdot 2^{-44}\).

Finite classes are not the only learnable ones. Some infinite classes are PAC learnable too, such as circles centred at the origin, where a hypothesis says 1 inside the circle. What really decides learnability is not finiteness but a combinatorial measure called the VC dimension, which arrives in lesson 6.

## Where realizability breaks

Now back to the bad week. Here is a training sequence of twelve examples from the mail room, in lesson 1’s notation. Each message is four bits FLUN (says “FREE”, has a link, unknown sender, sent at night), followed by its label. Three rows deserve a mark. Rows 3 and 6 are the same newsletter, 0110, labelled by two different people. Row 7 is the coffee-in-hand mistake: a “FREE” message labelled keep.

The last four columns grade each one-feature hypothesis. The hypothesis \(h_F\) says junk exactly when the F bit is 1, \(h_L\) when the L bit is 1, and so on; in symbols \(h_j(x) = x_j\), “the label is feature \(j\)’s bit”. A cross marks a mistake. Before reading them, guess which hypothesis will make the fewest mistakes, and whether any will make none.

| i | message FLUN | label | \(h_F\) | \(h_L\) | \(h_U\) | \(h_N\) |
|---|---|---|---|---|---|---|
| 1 | 1101 | junk | | | ✗ | |
| 2 | 0000 | keep | | | | |
| 3 | 0110 | junk (newsletter, reader A) | ✗ | | | ✗ |
| 4 | 1010 | junk | | ✗ | | ✗ |
| 5 | 0011 | keep | | | ✗ | ✗ |
| 6 | 0110 | keep (newsletter, reader B) | | ✗ | ✗ | |
| 7 | 1100 | keep (honest mistake) | ✗ | ✗ | | |
| 8 | 0101 | keep | | ✗ | | ✗ |
| 9 | 1111 | junk | | | | |
| 10 | 0001 | keep | | | | ✗ |
| 11 | 0111 | junk | ✗ | | | |
| 12 | 1000 | junk | | ✗ | ✗ | ✗ |

A reminder from lesson 1: \(L_S(h)\), the empirical risk, is the fraction of the examples in \(S\) that \(h\) gets wrong, so here it is the number of crosses in a column divided by 12. Count the crosses: \(L_S(h_F) = 3/12 = 0.25\), \(L_S(h_U) = 4/12 \approx 0.33\), \(L_S(h_L) = 5/12 \approx 0.42\), \(L_S(h_N) = 6/12 = 0.5\). No hypothesis in the class is consistent.

It is worse than that. Look at rows 3 and 6. Every hypothesis, of any kind, gives 0110 one label, so it is wrong on one of those two rows. No function from \(\mathcal X\) to \(\{0, 1\}\) has empirical risk below \(1/12\) on this training sequence. Realizability isn’t merely false for our small class. It is false for every class.

Here is the encouraging part. ERM, as lesson 2 defined it, never asked for a consistent hypothesis. It asks for one that minimizes \(L_S\), and that still makes sense: it returns \(h_F\), with three mistakes out of twelve. The code fix is one line. Instead of “return a hypothesis with \(L_S(h) = 0\), or fail”, write “return a hypothesis with the smallest \(L_S(h)\)”.

What breaks is the promise. Write \(h_S\) for the hypothesis ERM returns on the training sequence \(S\). Lesson 2’s proof ran like this. ERM returns a hypothesis with \(L_S(h_S) = 0\), so if \(h_S\) is bad, some bad hypothesis looked perfect on \(S\), and that is unlikely. Every link in that chain needed \(L_S(h_S) = 0\). Now ERM returns a hypothesis with \(L_S(h_S) = 0.25\), and the chain has nothing to hold on to. We need a new account of where labels come from, a new benchmark for what “good” means, and eventually a new proof. This lesson supplies the first two. The proof is the cliffhanger.

## Labels from a distribution

Where do these labels come from, if not from a single function \(f\)? The answer is to let the label be random too.

Picture each example as a two-step draw. First a message arrives, drawn from a distribution over \(\mathcal X\). Then someone labels it, and the label is itself a draw: for the newsletter, junk for seven readers in ten and keep for the other three. What gets drawn is a pair, a message together with its label, like (0110, junk). With \(\mathcal Y = \{0, 1\}\) the set of labels, the set of all such pairs is written \(\mathcal X \times \mathcal Y\); the \(\times\) reads “paired with”, so here there are \(16 \cdot 2 = 32\) pairs. Formally, from now on \(\mathcal D\) is a probability distribution over \(\mathcal X \times \mathcal Y\), a **joint distribution** over messages and labels. It is also called the data-labels generating distribution. It has two parts, one for each step of the draw:

- the **marginal distribution** of unlabelled messages, written \(\mathcal D_x\): how often each message arrives, so \(\mathcal D_x(x)\) is the probability that the next message is \(x\);
- the **conditional probability** of the label given the message, \(\mathcal D((x, y) \mid x)\): how likely a message \(x\) is to be labelled junk. The bar \(\mid\) reads “given”: we only look at draws where this particular message arrived, and ask about its label.

For binary labels we’ll shorten the second part to one number per message, the share of labellers who call it junk, 0.7 for the newsletter:

\[ \eta(x) = \mathbb P_{(x, y) \sim \mathcal D}[\, y = 1 \mid x \,], \]

the probability that a message \(x\) gets labelled junk. (Many texts write \(\mathbb P[y = 1 \mid x]\) in full; \(\eta\) just saves ink.)

Read it from the inside out.

- \(y = 1\) is the event “the label is junk”.
- The bar makes it \(y = 1 \mid x\): “the label is junk, given that the message is \(x\)”.
- \((x, y) \sim \mathcal D\) says the pair is drawn from the joint distribution, message and label together; \(\sim\) reads “drawn from”, as before.
- \(\mathbb P[\cdot]\) is the probability of the event inside the brackets.
- \(\eta\) is the Greek letter eta, just a name for this function of \(x\).

In plain words: **\(\eta(x)\) is the probability that message \(x\), once it has arrived, gets labelled junk.** For the newsletter 0110, \(\eta = 0.7\): of ten readers, seven click junk. In Python, a label for the newsletter is a biased coin that comes up 1 with probability \(\eta\), and the share of 1s over many coins comes out close to \(\eta\):

```python
import random
eta_newsletter = 0.7
rng = random.Random(7)
labels = [int(rng.random() < eta_newsletter)
          for _ in range(10000)]
print(eta_newsletter, sum(labels) / len(labels))
```

It prints `0.7 0.6992`: in 10,000 labellings of the newsletter, 69.92% said junk.

Lesson 2’s world, where \(f\) decides every label, is the special case where \(\eta(x)\) is always exactly 0 or exactly 1; realizability asks, on top of that, for \(f\) to agree with some hypothesis in the class. Disagreement and honest mistakes are what you get when \(\eta(x)\) sits strictly in between.

Here is the noisy mail room we’ll use for the rest of the lesson. As in lesson 1, all 16 messages are equally likely, so \(\mathcal D_x\) is uniform. The labels follow three kinds of message.

| kind | messages | how many | \(\eta(x)\) | what is going on |
|---|---|---|---|---|
| “FREE” | every message starting with 1 | 8 | 0.9 | junk, but one person in ten labels it keep by mistake |
| newsletter | 0110 and 0111 (link, unknown sender, no “FREE”) | 2 | 0.7 | seven in ten readers call it junk, three call it keep |
| everything else | the other six | 6 | 0.1 | keep, but one person in ten labels it junk by mistake |

[[diagram:joint]]

The true risk needs a new definition, because there is no \(f\) to compare with. We redefine it as the probability that a hypothesis disagrees with the label of a fresh example, message and label drawn together. Take the newsletter: when it arrives, \(h_F\) says keep, and seven labellers in ten say junk, so on that message \(h_F\) is wrong with probability 0.7. The new true risk weighs chances like that over every message that might arrive:

\[ L_{\mathcal D}(h) = \mathbb P_{(x, y) \sim \mathcal D}[\, h(x) \ne y \,] = \mathcal D(\{(x, y) : h(x) \ne y\}). \]

Read it from the inside out.

- \(h(x) \ne y\) is the event “the hypothesis’s label differs from the label actually drawn”: a mistake.
- \(\mathbb P_{(x, y) \sim \mathcal D}[\dots]\) is its probability when the pair \((x, y)\) is drawn from \(\mathcal D\). Two things are random now: which message arrives and which label it gets.
- The second form says the same thing about a set. The braces \(\{(x, y) : h(x) \ne y\}\) collect every pair such that \(h\) gets it wrong (read the colon as “such that”), and \(\mathcal D(\dots)\) is the total probability \(\mathcal D\) gives to that set.

In plain words: **the chance that \(h\) mislabels a freshly drawn message, measured against the label it is actually given.** For \(h_F\), the set of mistakes holds 16 of the 32 pairs, one per message: (newsletter, junk) is in it, with probability \(1/16 \cdot 0.7\), and so is every “FREE” message paired with keep, each with probability \(1/16 \cdot 0.1\). Adding up all 16 gives 0.175. The snippet builds that set and sums it:

```python
from itertools import product
X = list(product((0, 1), repeat=4))
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
def D(x, y):
    return (eta(x) if y == 1 else 1 - eta(x)) / 16
h_F = lambda x: x[0]
pairs = [(x, y) for x in X for y in (0, 1)]
wrong = [(x, y) for x, y in pairs if h_F(x) != y]
print(len(pairs), len(wrong), round(sum(D(*z) for z in wrong), 3))
```

It prints `32 16 0.175`: 32 possible pairs, 16 of them mistakes for \(h_F\), with total probability 0.175. The next paragraphs do the same sum by hand.

The subscript is now just \(\mathcal D\), because \(\mathcal D\) carries the labels as well as the messages. The empirical risk keeps its old definition, \(L_S(h) = |\{i \in [m] : h(x_i) \ne y_i\}|/m\): \([m]\) is the list of positions 1 to \(m\), the braces collect the positions \(i\) such that \(h\) is wrong on example \(i\), the bars count them, and dividing by \(m\) makes the count a fraction, 3/12 for \(h_F\) on the table above. Here is a neat observation: \(L_S(h)\) is the true risk of \(h\) under the distribution that picks one of the \(m\) examples of \(S\) uniformly at random: each of our twelve rows has probability \(1/12\), three of them are \(h_F\)‘s mistakes, and \(3 \cdot 1/12 = 0.25\). The empirical risk is a true risk, just for a tiny, lopsided world made of your training sequence.

Let’s compute the true risk of \(h_F\) in the noisy mail room. On the 8 “FREE” messages it says junk, and the label is keep one time in ten, so it errs with probability 0.1 on each. On the two newsletters it says keep, and the label is junk seven times in ten, so it errs with probability 0.7. On the other six it says keep and errs with probability 0.1. Averaging over the 16 equally likely messages:

\[ L_{\mathcal D}(h_F) = \frac{8 \cdot 0.1 + 2 \cdot 0.7 + 6 \cdot 0.1}{16} = \frac{2.8}{16} = 0.175. \]

Each product on top is “how many messages of that kind” times “how often \(h_F\) errs on one of them”: \(8 \cdot 0.1 = 0.8\) for the “FREE” messages, \(2 \cdot 0.7 = 1.4\) for the newsletters, \(6 \cdot 0.1 = 0.6\) for the rest. Their total, 2.8, is the expected number of mistakes if each of the 16 messages arrived once, and dividing by 16 turns it into a chance per message, because each message arrives with probability \(1/16\). In plain words: **the true risk is the average, over equally likely messages, of the chance of a mistake on each.**

The same calculation gives \(L_{\mathcal D}(h_L) = L_{\mathcal D}(h_U) = 0.425\) and \(L_{\mathcal D}(h_N) = 0.5\). So \(h_F\) is the best hypothesis in the one-feature class, and even the best one is wrong 17.5% of the time. Notice too that on our twelve examples \(h_F\) scored 0.25, not 0.175. The empirical risk is an estimate of the true risk, and with twelve examples it is a rough one.

In code, the noisy mail room is one function, and the true risk is an exact average over the 16 messages:

```python
from itertools import product
X = list(product((0, 1), repeat=4))
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
def L_D(h):
    risks = [(1 - eta(x)) if h(x) else eta(x) for x in X]
    return sum(risks) / len(X)
for j, name in enumerate("FLUN"):
    print(name, round(L_D(lambda x: x[j]), 3))
```

It prints 0.175, 0.425, 0.425 and 0.5.

Time to play with that. The lab below draws training sequences from the noisy mail room. You choose how many examples, and you watch each one-feature hypothesis’s empirical risk wander around its true risk, along with the hypothesis ERM picks.

[[lab]]

## The best anyone can do

Here is a question the realizable world never raised. In the noisy mail room, even a perfect learner, one that somehow knew \(\mathcal D\) exactly, could not get every label right. The newsletter is junk for seven readers and keep for three; whatever you say, someone disagrees. So what is the smallest true risk that *any* hypothesis can reach?

Think about one message at a time. For the newsletter 0110, saying junk is wrong with probability \(1 - 0.7 = 0.3\), and saying keep is wrong with probability \(0.7\). The choice is easy: say junk. For a “FREE” message, junk is wrong with probability 0.1, keep with 0.9: say junk. For the other six, say keep. In each case you pick the label that is more likely, and you are wrong only with the smaller of the two probabilities.

That rule has a name.

**The Bayes optimal predictor.** For a distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\),

\[ f_{\mathcal D}(x) = \begin{cases} 1 & \text{if } \mathbb P[y = 1 \mid x] \ge 1/2, \\ 0 & \text{otherwise.} \end{cases} \]

Read it from the inside out.

- \(\mathcal X \times \{0, 1\}\) is \(\mathcal X \times \mathcal Y\) with the labels spelled out: every message paired with 0 (keep) or 1 (junk).
- \(\mathbb P[y = 1 \mid x]\) is \(\eta(x)\), the chance that message \(x\) is labelled junk.
- \(\ge 1/2\) asks whether junk is at least as likely as keep.
- The big brace lists cases: output 1 if the condition on its row holds, and 0 “otherwise”, meaning in every other case.
- The subscript in \(f_{\mathcal D}\) says this predictor is built from \(\mathcal D\) itself, so it changes when the distribution does.

In plain words: **for each message, predict the label that most labellers would give it.** In our shorthand: predict junk exactly when \(\eta(x) \ge 1/2\). The newsletter has \(\eta = 0.7 \ge 0.5\), so \(f_{\mathcal D}\) says junk; the message 0011 has \(\eta = 0.1\), so it says keep.

```python
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
def f_D(x):
    return 1 if eta(x) >= 1 / 2 else 0
for x in ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)):
    print(x, eta(x), f_D(x))
```

It prints one line per kind of message: `(1, 0, 0, 0) 0.9 1`, `(0, 1, 1, 0) 0.7 1` and `(0, 0, 1, 1) 0.1 0`, so junk, junk, keep.

We’ll call its true risk, \(L_{\mathcal D}(f_{\mathcal D})\), the **Bayes risk**: the chance that even this best guess mislabels a fresh message.

[[diagram:bayes]]

Here is the whole calculation for the noisy mail room.

| kind | how many | \(\eta(x)\) | wrong if it says junk | wrong if it says keep | \(f_{\mathcal D}\) says | \(h_F\) says |
|---|---|---|---|---|---|---|
| “FREE” | 8 | 0.9 | 0.1 | 0.9 | junk | junk |
| newsletter | 2 | 0.7 | 0.3 | 0.7 | junk | keep |
| everything else | 6 | 0.1 | 0.9 | 0.1 | keep | keep |

\[ L_{\mathcal D}(f_{\mathcal D}) = \frac{8 \cdot 0.1 + 2 \cdot 0.3 + 6 \cdot 0.1}{16} = \frac{2}{16} = 0.125. \]

It has the same shape as the sum for \(h_F\): each product is a count of messages times the chance of a mistake on one of them, now always the smaller of the two “wrong if” columns. Only the middle term changed, \(2 \cdot 0.3 = 0.6\) instead of \(2 \cdot 0.7 = 1.4\), because \(f_{\mathcal D}\) says junk to the newsletters. In plain words: **the Bayes risk averages, over messages, the smaller of the two ways to be wrong.** In Python, `min(e, 1 - e)` is that smaller chance:

```python
kinds = [("FREE", 8, 0.9), ("newsletter", 2, 0.7),
         ("everything else", 6, 0.1)]
terms = [n * min(e, 1 - e) for _, n, e in kinds]
print([round(t, 2) for t in terms], round(sum(terms) / 16, 3))
```

It prints `[0.8, 0.6, 0.6] 0.125`: the three products on top of the fraction, then the Bayes risk.

The Bayes optimal predictor junks “FREE” messages and newsletters, and keeps everything else. In feature language: junk when F is on, or when L and U are both on. That is not a one-feature hypothesis. The best one-feature hypothesis, \(h_F\), gets the newsletters wrong, and pays for it: 0.175 against 0.125.

The claim is that \(f_{\mathcal D}\) is optimal: no hypothesis \(g: \mathcal X \to \{0, 1\}\) has smaller true risk. The proof is short, and important enough to do in full. We’ll prove it for a finite domain like ours, where every probability is a sum. The general case is the same argument with integrals.

**Claim.** For every distribution \(\mathcal D\) over \(\mathcal X \times \{0, 1\}\) and every \(g: \mathcal X \to \{0, 1\}\), \(L_{\mathcal D}(f_{\mathcal D}) \le L_{\mathcal D}(g)\).

*Proof.* Split the true risk of any \(g\) by which message arrived. This is the law of total probability: first draw \(x\) from \(\mathcal D_x\), then the label.

\[ L_{\mathcal D}(g) = \sum_{x \in \mathcal X} \mathcal D_x(x) \; \mathbb P[\, y \ne g(x) \mid x \,]. \]

The \(\sum\) (a capital sigma) means “add up”, and the \(x \in \mathcal X\) under it says over what: one term for every message \(x\) in the domain. Each term multiplies the chance that \(x\) arrives, \(\mathcal D_x(x)\), by the chance that \(g\) is wrong given that it did. In plain words: **the true risk is the arrival-weighted average of the mistake chance on each message.** For \(g = h_F\), the newsletter’s term is \(1/16 \cdot 0.7\), and all 16 terms add up to the 0.175 we computed. The function `L_D` in the brute-force snippet below is this sum: its list `risks` holds the 16 mistake chances, and dividing the total by `len(X)` multiplies each by \(\mathcal D_x(x) = 1/16\).

Now look inside the sum at one message. If \(g(x) = 1\), the label disagrees when \(y = 0\), which has probability \(1 - \eta(x)\). If \(g(x) = 0\), it disagrees when \(y = 1\), which has probability \(\eta(x)\). Either way,

\[ \mathbb P[\, y \ne g(x) \mid x \,] \in \{\eta(x),\ 1 - \eta(x)\}, \quad \text{so} \quad \mathbb P[\, y \ne g(x) \mid x \,] \ge \min\{\eta(x),\ 1 - \eta(x)\}. \]

Here \(\in\) reads “is one of”, the braces list the two possibilities, and \(\min\{a, b\}\) is the smaller of \(a\) and \(b\). In plain words: **whatever \(g\) says, its mistake chance on a message is one of the two numbers, so it is at least the smaller one.** On the newsletter the two numbers are 0.7 and 0.3; \(h_F\) gets 0.7, which is indeed at least 0.3.

That inequality holds for every hypothesis. Next, check that \(f_{\mathcal D}\) reaches it with equality. If \(\eta(x) \ge 1/2\), then \(f_{\mathcal D}(x) = 1\), its conditional risk is \(1 - \eta(x)\), and \(1 - \eta(x) \le 1/2 \le \eta(x)\), so that is the minimum. If \(\eta(x) < 1/2\), then \(f_{\mathcal D}(x) = 0\), its conditional risk is \(\eta(x)\), and \(\eta(x) < 1/2 < 1 - \eta(x)\), so again that is the minimum. Hence, for every message,

\[ \mathbb P[\, y \ne f_{\mathcal D}(x) \mid x \,] = \min\{\eta(x),\ 1 - \eta(x)\} \le \mathbb P[\, y \ne g(x) \mid x \,]. \]

This reuses the same pieces and says: **on every message, the Bayes optimal predictor’s mistake chance is the smallest one available, so no \(g\) can do better there** (on the newsletter, 0.3 against \(h_F\)’s 0.7).

Finally, multiply both sides by \(\mathcal D_x(x) \ge 0\), which keeps the inequality, and sum over \(x\). The left side becomes \(L_{\mathcal D}(f_{\mathcal D})\) and the right side becomes \(L_{\mathcal D}(g)\). \(\blacksquare\)

The proof covers every distribution at once. For our one distribution you can also check the claim by brute force. Compute \(f_{\mathcal D}\), then try all \(2^{16} = 65{,}536\) hypotheses and keep the smallest true risk.

```python
from itertools import product
X = list(product((0, 1), repeat=4))
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
def L_D(g):
    risks = [(1 - eta(x)) if g[x] else eta(x) for x in X]
    return sum(risks) / len(X)
f_D = {x: int(eta(x) >= 0.5) for x in X}
bayes_risk = L_D(f_D)
labelings = product((0, 1), repeat=16)
every_g = (dict(zip(X, labels)) for labels in labelings)
best = min(L_D(g) for g in every_g)
print(round(bayes_risk, 3), round(best, 3))
assert abs(best - bayes_risk) < 1e-12
```

It prints 0.125 twice. Not one of the 65,536 hypotheses beats the Bayes optimal predictor.

Notice what kind of proof that was. There is no clever global trick, just a decision made separately for every message, each one locally best. A message where \(\eta(x) = 1/2\) exactly is a coin, and either label is equally good there; the rule as written happens to say junk.

So the Bayes risk is a floor, and \(\min\{\eta(x), 1 - \eta(x)\}\), averaged over messages, is how high that floor sits. It is zero exactly when every label is certain. If, on top of that, the class contains \(f_{\mathcal D}\), we are back in realizability’s world.

Then why not just return \(f_{\mathcal D}\)? Because computing it needs \(\eta\), and \(\eta\) is part of \(\mathcal D\), which the learner never sees. All it has is the training sequence.

## A promise you can keep

So what should the new promise say? Two natural candidates fail before we land on the right one.

“True risk at most \(\varepsilon\)” is impossible. In the noisy mail room every hypothesis has true risk at least 0.125, so asking for \(\varepsilon = 0.1\) asks for the impossible, however many examples arrive.

“True risk within \(\varepsilon\) of the Bayes risk” is the next idea, and it is also out of reach. Once we make no prior assumptions about \(\mathcal D\), no algorithm can be guaranteed to find a hypothesis as good as the Bayes optimal one. The proof is the No-Free-Lunch theorem of lesson 5, the formal version of lesson 1’s lesson that learning needs a bias.

What is left is to compare the learner with a benchmark it could actually hope to match: the best hypothesis in a class you fixed in advance. In the noisy mail room, the best one-feature hypothesis is \(h_F\), with true risk 0.175. With \(\varepsilon = 0.05\), the learner would be asked to land at \(0.175 + 0.05 = 0.225\) or below, on all but a \(\delta\) share of training sequences. The definition says exactly that, for every distribution at once.

**Definition (Agnostic PAC learnability).** A hypothesis class \(\mathcal H\) is **agnostic PAC learnable** if there exist a function \(m_{\mathcal H}: (0, 1)^2 \to \mathbb N\) and a learning algorithm with the following property. For every \(\varepsilon, \delta \in (0, 1)\) and for every distribution \(\mathcal D\) over \(\mathcal X \times \mathcal Y\), when the algorithm runs on \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples generated by \(\mathcal D\), it returns a hypothesis \(h\) such that, with probability at least \(1 - \delta\) over the choice of the \(m\) training examples,

\[ L_{\mathcal D}(h) \le \min_{h' \in \mathcal H} L_{\mathcal D}(h') + \varepsilon. \]

Read it from the inside out.

- \(h'\) (“h prime”) is a second name for a hypothesis, so it doesn’t clash with \(h\), the one the learner returned.
- \(L_{\mathcal D}(h')\) is the true risk of \(h'\).
- \(\min_{h' \in \mathcal H}\) means “the smallest value as \(h'\) runs over every hypothesis in \(\mathcal H\)”: the true risk of the best in the class.
- \(+\, \varepsilon\) adds the forgiven slack on top, and \(\le\) says the learner’s true risk must not exceed that total.

In plain words: **the returned hypothesis is at most \(\varepsilon\) worse than the best hypothesis in the class.** With the one-feature class the right side is \(\min\{0.175, 0.425, 0.425, 0.5\} + 0.05 = 0.225\). The simulation a little further down mirrors it term by term: `min(L_D)` is the minimum over the class, `+ eps` is the slack, and a run counts as a miss when `L_D[j]`, the returned hypothesis’s true risk, is above that.

Two words in the name are doing work. “Agnostic” because the learner no longer assumes the class contains a perfect hypothesis; it doesn’t need to believe anything about \(\mathcal D\). And the guarantee is relative: success means “not much worse than the best in \(\mathcal H\)”, not “small in absolute terms”. Be candid about what that means: the strength of such a promise depends entirely on the choice of the class.

In the noisy mail room with the one-feature class, three numbers now line up, and you should be able to say what each one is.

[[diagram:ladder]]

- The Bayes risk, 0.125. No hypothesis can beat it.
- The best in the class, \(\min_{h \in \mathcal H} L_{\mathcal D}(h) = L_{\mathcal D}(h_F) = 0.175\). The gap of 0.05 between these two is the price of the bias: the class cannot express “junk when L and U are both on”.
- The learner’s own true risk. An agnostic PAC learner with \(\varepsilon = 0.05\) promises \(L_{\mathcal D}(h_S) \le 0.175 + 0.05 = 0.225\), with probability at least \(1 - \delta\).

Here is that promise put to the test with \(m = 50\) examples and \(\delta = 0.05\). Each run draws a training sequence from the noisy mail room and runs ERM over the one-feature class. It counts a miss when the returned hypothesis is more than \(\varepsilon\) above the best in the class.

```python
import random
from itertools import product
X = list(product((0, 1), repeat=4))
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
L_D = [sum((1 - eta(x)) if x[j] else eta(x) for x in X) / 16
       for j in range(4)]
eps, delta, m = 0.05, 0.05, 50
rng = random.Random(3)
misses = 0
for _ in range(20000):
    xs = rng.choices(X, k=m)
    S = [(x, int(rng.random() < eta(x))) for x in xs]
    j = min(range(4), key=lambda j: sum(x[j] != y for x, y in S))
    misses += L_D[j] > min(L_D) + eps
print(round(min(L_D), 3), round(misses / 20000, 3))
assert misses / 20000 <= delta
```

It prints 0.175 for the best in the class and a miss rate of 0.006, about 0.6%, under \(\delta\). Notice that the code measures against \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\), never against zero or against the Bayes risk.

When realizability does hold, the best in the class has true risk 0, and the agnostic promise collapses to the PAC promise, \(L_{\mathcal D}(h) \le \varepsilon\). In that sense agnostic PAC learning generalizes PAC learning.

## Beyond junk and keep

So far a hypothesis has answered one yes/no question. The same framework covers far more, once we stop hard-wiring “count the wrong labels” into the definition of risk.

**Multiclass classification.** Suppose the mail room adds folders: keep, junk, newsletters, invoices. Now \(\mathcal Y\) is a set of several labels. Sorting documents by topic (news, sports, biology, medicine) is the same kind of problem, with features such as counts of key words, the document’s size or its origin. Nothing else in the framework changes. An example is still a pair \((x, y)\), a hypothesis is still a function from \(\mathcal X\) to \(\mathcal Y\), and success is still measured by the probability of a wrong label.

**Regression.** Now suppose the mail room wants to predict a number, say how many minutes a message will sit unread. The labels are real numbers, and in this case \(\mathcal Y\) is often called the target set. Other examples are tomorrow’s temperature at 9 p.m., or a baby’s birth weight predicted from three ultrasound measurements, where \(\mathcal X \subseteq \mathbb R^3\). Here “wrong or right” is the wrong question, since a prediction of 41 minutes for a message read after 40 is nearly right. Instead, square the gap: 41 against 40 costs \(1^2 = 1\), and 41 against 50 costs \(9^2 = 81\). Squaring makes every gap count as positive and makes big misses cost much more than small ones.

The label is random, though. Say the same message sits 40 minutes for half its readers and 50 for the other half. Then the cost of predicting 41 is 1 half the time and 81 the other half, and its **expected value**, the average you would see over many draws with each outcome weighted by its probability, is \(0.5 \cdot 1 + 0.5 \cdot 81 = 41\). So we measure a hypothesis by its expected squared difference from the label:

\[ L_{\mathcal D}(h) = \mathbb E_{(x, y) \sim \mathcal D}\left[ (h(x) - y)^2 \right]. \]

Read it from the inside out.

- \(h(x) - y\) is the gap between the prediction and the actual label.
- The exponent \(^2\) squares it.
- \(\mathbb E_{(x, y) \sim \mathcal D}[\cdot]\) is the expected value of what is inside, over pairs drawn from \(\mathcal D\): each possible pair’s squared gap, weighted by that pair’s probability, all added up.

In plain words: **the average squared miss, over messages and their labels as they actually arrive.** For the half-40, half-50 message, predicting 41 scores 41, and predicting 45, right in the middle, scores \(0.5 \cdot 25 + 0.5 \cdot 25 = 25\), which is better.

```python
outcomes = [(40, 0.5), (50, 0.5)]
def L_D(c):
    return sum(p * (c - y) ** 2 for y, p in outcomes)
for c in (41, 45):
    print(c, L_D(c))
```

It prints `41 41.0` and `45 25.0`. Each pair in `outcomes` is a label and its probability, and the sum weighs each squared gap by that probability.

Two tasks, two ways to score a hypothesis. So we generalize the scoring itself.

**Loss functions.** The idea is a price tag. Hand it a hypothesis and one example, and it says how much it cost to use that hypothesis on that example. For the junk filter the price is 1 for a wrong label and 0 for a right one, so \(h_F\) on row 7 of the table (1100, labelled keep) costs 1. For regression it is the squared gap, so predicting 41 for a message read after 50 costs 81. Formally, take a set \(\mathcal H\) (our hypotheses) and a domain \(Z\) of examples. A **loss function** is any function

\[ \ell: \mathcal H \times Z \to \mathbb R_+, \]

from a hypothesis and an example to a non-negative number. The letter \(\ell\) is a script lowercase L, for loss. \(\mathcal H \times Z\) is the set of pairs (hypothesis, example), the arrow reads “to”, and \(\mathbb R_+\) is the non-negative real numbers, 0 and up. In plain words: **a loss function takes a hypothesis and an example and returns a cost that is never negative.** So \(\ell(h, z)\) is how much it costs to use \(h\) on \(z\). For prediction problems \(Z = \mathcal X \times \mathcal Y\). We deliberately allow any \(Z\), though, because in unsupervised problems such as clustering an example is not a pair at all.

With a loss in hand, both risks are averages of the loss. The **true risk** is the expected loss on a fresh example, and the **empirical risk** is the average loss over the training sequence. For \(h_F\) with the 1-or-0 price, the twelve examples cost 1 three times and 0 nine times, an average of 0.25, and the expected cost on a fresh example from the noisy mail room is 0.175. In symbols, with the training sequence written \(S = (z_1, \dots, z_m)\), one \(z_i\) per example:

\[ L_{\mathcal D}(h) = \mathbb E_{z \sim \mathcal D}[\, \ell(h, z) \,], \]

\[ L_S(h) = \frac{1}{m} \sum_{i=1}^m \ell(h, z_i). \]

Read them from the inside out.

- \(\ell(h, z)\) is the cost of using \(h\) on one example \(z\).
- \(\mathbb E_{z \sim \mathcal D}[\cdot]\) averages that cost over a fresh example drawn from \(\mathcal D\), each possible example weighted by its probability.
- \(\sum_{i=1}^m\) adds one term for each position \(i\) from 1 up to \(m\): the costs of all the training examples.
- \(\frac{1}{m}\) divides the total by \(m\), turning it into an average.

In plain words: **the true risk is the expected cost on the next example, and the empirical risk is the average cost on the examples you have.** The snippet a little further down computes the second one, `sum(loss(h, z) for z in S) / len(S)`, on the twelve examples. Here is the first one, the weighted sum over all 32 pairs of the noisy mail room:

```python
from itertools import product
X = list(product((0, 1), repeat=4))
def eta(x):
    return 0.9 if x[0] else (0.7 if x[1] and x[2] else 0.1)
def D(z):
    x, y = z
    return (eta(x) if y == 1 else 1 - eta(x)) / 16
def loss_01(h, z):
    x, y = z
    return int(h(x) != y)
Z = [(x, y) for x in X for y in (0, 1)]
h_F = lambda x: x[0]
print(round(sum(D(z) * loss_01(h_F, z) for z in Z), 3))
```

It prints 0.175, the same true risk we found by summing the probabilities of \(h_F\)’s mistakes. The next paragraphs explain why the two must agree.

The two losses we have met so far are both instances.

- The **0-1 loss**, \(\ell_{0-1}(h, (x, y)) = 0\) if \(h(x) = y\) and 1 if \(h(x) \ne y\), for binary and multiclass classification. The subscript \(0-1\) is only a name. On row 7, \(h_F(1100) = 1\) and \(y = 0\), so the loss is 1.
- The **square loss**, \(\ell_{sq}(h, (x, y)) = (h(x) - y)^2\), for regression; the subscript \(sq\) stands for square. On the same row it is \((1 - 0)^2 = 1\).

Is the new definition of true risk consistent with the old one? For the 0-1 loss, yes, thanks to a small fact worth keeping. Average a list of 0s and 1s, and you get the share of 1s: the average of 1, 0, 0, 0 is 0.25, and so is the share of 1s. The same holds for expected values. A random variable \(\alpha\) that only takes the values 0 and 1 has \(\mathbb E[\alpha] = \mathbb P[\alpha = 1]\), because \(\mathbb E[\alpha] = 0 \cdot \mathbb P[\alpha = 0] + 1 \cdot \mathbb P[\alpha = 1]\) and the first term vanishes. The 0-1 loss is exactly such a variable, so its expected value is the probability of a wrong label. The expected-loss definition gives back exactly the true risk we started with. On the newsletter, \(h_F\)‘s loss is 1 with probability 0.7, so its expected loss is \(0 \cdot 0.3 + 1 \cdot 0.7 = 0.7\):

```python
alpha = [1, 0, 0, 0]
print(sum(alpha) / len(alpha), alpha.count(1) / len(alpha))
p_wrong = 0.7
E_alpha = 0 * (1 - p_wrong) + 1 * p_wrong
print(E_alpha, p_wrong)
```

It prints `0.25 0.25` and `0.7 0.7`: the average of the list is its share of 1s, and the expected 0-1 loss is the chance of a mistake.

In code, a loss function is just a function of a hypothesis and an example, and the empirical risk is its average over the training sequence. Here are both losses on the twelve examples from earlier:

```python
S = [("1101", 1), ("0000", 0), ("0110", 1), ("1010", 1),
     ("0011", 0), ("0110", 0), ("1100", 0), ("0101", 0),
     ("1111", 1), ("0001", 0), ("0111", 1), ("1000", 1)]
def h_F(x):
    return int(x[0])
def loss_01(h, z):
    x, y = z
    return int(h(x) != y)
def loss_sq(h, z):
    x, y = z
    return (h(x) - y) ** 2
def L_S(h, loss):
    return sum(loss(h, z) for z in S) / len(S)
print(L_S(h_F, loss_01), L_S(h_F, loss_sq))
assert L_S(h_F, loss_01) == L_S(h_F, loss_sq) == 0.25
```

It prints 0.25 twice, the three crosses in the \(h_F\) column. The next paragraph says why they agree.

[[diagram:losses]]

Here is a small observation. Suppose a hypothesis only ever outputs 0 or 1, and the label is 0 or 1. Then the square loss and the 0-1 loss are the same number, since \((1 - 0)^2 = 1\) and \(0^2 = 0\). The two losses only part ways when a hypothesis may answer with something in between, like “0.7 junk”. What should such a hypothesis say for the newsletter? That is the third question of the concept check.

Now the most general definition of all, the one every later lesson means when it says “PAC”.

**Definition (Agnostic PAC learnability for general loss functions).** A hypothesis class \(\mathcal H\) is agnostic PAC learnable with respect to a set \(Z\) and a loss function \(\ell: \mathcal H \times Z \to \mathbb R_+\) if there exist a function \(m_{\mathcal H}: (0, 1)^2 \to \mathbb N\) and a learning algorithm with the following property. For every \(\varepsilon, \delta \in (0, 1)\) and every distribution \(\mathcal D\) over \(Z\), when the algorithm runs on \(m \ge m_{\mathcal H}(\varepsilon, \delta)\) i.i.d. examples generated by \(\mathcal D\), it returns \(h \in \mathcal H\) such that, with probability at least \(1 - \delta\) over the choice of the \(m\) training examples,

\[ L_{\mathcal D}(h) \le \min_{h' \in \mathcal H} L_{\mathcal D}(h') + \varepsilon, \qquad \text{where } L_{\mathcal D}(h) = \mathbb E_{z \sim \mathcal D}[\, \ell(h, z) \,]. \]

Every piece of this display has been read already: it is the agnostic promise, **at most \(\varepsilon\) worse than the best in the class**, with the true risk now meaning expected loss. With the 0-1 loss and the one-feature class it is the same 0.225 ceiling as before.

Two fine points are worth a sentence each.

**Measurability.** For \(L_{\mathcal D}(h)\) to be an expected value at all, each function \(z \mapsto \ell(h, z)\) (read “the function that sends each example \(z\) to its cost”) has to be a random variable, something \(\mathcal D\) can assign probabilities to. So we assume it is measurable with respect to the \(\sigma\)-algebra on which \(\mathcal D\) is defined, the collection of events that \(\mathcal D\) gives a probability. For a finite domain like ours this holds automatically, and you can safely read past it.

**Proper versus representation-independent learning.** The definition requires the returned hypothesis to lie in \(\mathcal H\). Sometimes it is natural to allow a hypothesis from a larger set \(\mathcal H' \supseteq \mathcal H\) (the symbol reads “contains”: every member of \(\mathcal H\) is also in \(\mathcal H'\)), as long as it still satisfies \(L_{\mathcal D}(h') \le \min_{h \in \mathcal H} L_{\mathcal D}(h) + \varepsilon\). That is **representation-independent** learning; insisting on \(\mathcal H\) is **proper** learning. The first is sometimes called “improper”, although there is nothing improper about it.

## The learner as code

What is a PAC learner, as a piece of software? Two things shipped together: an algorithm \(A\) that maps a training sequence to a hypothesis, and a function \(m_{\mathcal H}(\varepsilon, \delta)\) that tells you how many examples to collect. In use, the order is: choose \(\varepsilon\) and \(\delta\), compute \(m = m_{\mathcal H}(\varepsilon, \delta)\), have people label \(m\) messages drawn as they arrive, run \(A\), ship the hypothesis. Notice that \(A\) never sees \(\varepsilon\) or \(\delta\). They only decide how much data it gets.

For a finite class the algorithm is ERM, now with a general loss. It walks through the class and keeps the hypothesis with the smallest empirical risk.

```
ERM(H, loss, S):
  best ← +infinity
  for h in H:
    total ← 0
    for i = 1..m:
      total ← total + loss(h, z_i)
    if total < best:
      best ← total
      h_best ← h
  return h_best
```

Dividing by \(m\) doesn’t change which hypothesis wins, so the code compares totals. Ties go to the first hypothesis found, which is one legal choice: ERM may return any member of the argmin, the set of hypotheses tied for the smallest empirical risk.

**Cost.** Count the work. The loop prices every hypothesis on every example, so with the twelve examples and the four one-feature hypotheses it calls the loss \(12 \cdot 4 = 48\) times. If one loss evaluation costs \(c\) time, the total is \(m \cdot |\mathcal H| \cdot c\), up to the small bookkeeping around it. Computer science writes this \(O(m \cdot |\mathcal H| \cdot c)\): the big \(O\) reads “grows at most in proportion to”, so doubling \(m\) at most doubles the time, give or take a fixed factor. In plain words: **ERM’s time is the number of examples times the size of the class times the price of one loss.** The count is easy to check:

```python
rows = ("1101 1 0000 0 0110 1 1010 1 0011 0 0110 0 "
        "1100 0 0101 0 1111 1 0001 0 0111 1 1000 1").split()
S = [(rows[i], int(rows[i + 1])) for i in range(0, 24, 2)]
calls = 0
def loss(j, z):
    global calls
    calls += 1
    return int(int(z[0][j]) != z[1])
best = min(range(4), key=lambda j: sum(loss(j, z) for z in S))
print(len(S), calls, "FLUN"[best])
```

It prints `12 48 F`: twelve examples, 48 loss evaluations, and ERM picks \(h_F\). Beyond the input it needs only \(O(1)\) extra space, meaning a fixed amount that does not grow with \(m\) or \(|\mathcal H|\), for the running total and the best so far. For the one-feature class over \(d\) features, \(c = O(1)\) and \(|\mathcal H| = d\), so ERM costs \(O(md)\). That is the same as lesson 1’s ConsistentFeature, and it never has to return FAIL.

Brute force is not always the smart way to run ERM. Take the class of *all* hypotheses on the 16 messages, \(2^{16} = 65{,}536\) of them. Looping over them would be silly, because the 0-1 loss splits message by message. The empirical risk is a sum over distinct messages, and each hypothesis chooses each message’s label independently. So ERM over all hypotheses is a tally: count junk and keep votes for every message seen, return the majority label, and pick a default for ties and unseen messages. That is \(O(m)\) expected time with a hash map. It is also lesson 1’s memorizer, repaired so that it can cope with disagreement. On the twelve examples above it reaches \(L_S = 1/12\), the lowest possible, and its true risk is 0.35, double that of \(h_F\). It memorized the coffee-in-hand mistake as if it were the truth, and it says keep for the three “FREE” messages it never saw. Overfitting, now with noisy labels.

What about the Bayes optimal predictor, as code? If \(\eta\) were known, it would be a lookup: \(O(|\mathcal X|)\) to tabulate, \(O(1)\) per message. Its cost is not computational. Its cost is that \(\eta\) is never known.

One more point about cost matters later. Valiant’s original definition of PAC learning also required the learner’s runtime to be polynomial. This course separates the statistical question (how many examples?) from the computational one (how much time?), and returns to the second in lesson 8.

**Build it yourself.** Using only the Python standard library:

```python
def eta(x): ...            # the noisy mail room: 0.9, 0.7 or 0.1
def true_risk(h): ...      # exact, summed over all 16 messages
def bayes(x): ...          # 1 if eta(x) >= 0.5 else 0
def draw(m, rng): ...      # m i.i.d. examples (x, y) from D
def erm(H, S): ...         # 0-1 loss, ties go to the first
```

1. Reproduce the true risks of this lesson: 0.125 for \(f_{\mathcal D}\), 0.175 for \(h_F\), 0.425 for \(h_L\) and \(h_U\), 0.5 for \(h_N\).
2. Reproduce the twelve-row table: empirical risks 0.25, 0.42, 0.33, 0.5.
3. Implement the tally version of ERM over all hypotheses and confirm \(L_S = 1/12\) and true risk 0.35 on the twelve examples, with “keep” as the default.
4. For each \(m\) in 10, 20, 50, 100, 200, draw 20,000 training sequences. Estimate the probability that ERM over the one-feature class returns a hypothesis with true risk above \(0.175 + 0.05\). Before you run it, predict the shape.
5. Try to use the finite-class guarantee to predict the answer to step 4. Explain why it does not apply.

## Three classes, one budget

Now let’s put three classes side by side in the noisy mail room. Each is a different bias.

- The **one-feature class**: 4 hypotheses. Best true risk 0.175. It cannot express the newsletter rule.
- **Any rule on F, L and U**: every hypothesis that ignores the night bit, \(2^8 = 256\) of them. Its best is the Bayes optimal predictor itself, 0.125, because \(\eta\) never depends on N. This is the right bias for this world.
- **Anything goes**: all \(2^{16} = 65{,}536\) hypotheses. Its best is also 0.125.

For each class and several values of \(m\), we drew 20,000 training sequences and ran ERM with the 0-1 loss (ties and unseen messages go to keep). Then we computed the exact true risk of each returned hypothesis. The first number in each cell is the average true risk. The second is how often the learner missed the agnostic target, that is, the fraction of runs with \(L_{\mathcal D}(h_S) > \min_{h \in \mathcal H} L_{\mathcal D}(h) + 0.05\).

| \(m\) | one-feature | any rule on F, L, U | anything goes |
|---|---|---|---|
| 20 | 0.191, misses 6.3% | 0.216, misses 55% | 0.308, misses 97% |
| 50 | 0.177, misses 0.6% | 0.146, misses 10% | 0.194, misses 50% |
| 100 | 0.175, misses 0.01% | 0.130, misses 0.6% | 0.147, misses 8.9% |
| 200 | 0.175, misses 0% | 0.126, misses 0.01% | 0.131, misses 0.2% |

Read the table along each column first. Every class gets close to its own best, and the bigger the class, the more examples that takes. Then read across the rows. At \(m = 20\) the tiny class wins outright. By \(m = 50\) the 256-hypothesis class has overtaken it, and by \(m = 100\) even “anything goes” is ahead. The one-feature class is stuck at 0.175 forever, because no number of examples can buy back what its bias left out.

This is the rat and the pigeon from lesson 1, with numbers. A small class learns its best quickly, but its best may be poor. A large class has a better best, but needs more examples to find it. Lesson 5 gives the two gaps their names. And notice which class does best overall: the one whose bias is *right*, which ignores exactly the feature that carries no information.

## Fix the bug

Back to the mail room. The newsletter fight and the hurried clicks aren’t going away, so the filter has to live with them. Your manager asks for a promise in writing, in the form you now know how to state. With probability at least 95%, the shipped filter’s true risk must be at most 0.05 above the best hypothesis in whatever class you choose. That is \(\varepsilon = 0.05\) and \(\delta = 0.05\).

Labels cost people’s time, so the fewer you ask for, the better. In the lab you choose the class and the label budget. The lab redraws the training sequence many times, so you can see how often your choice misses the target. On every run it shows three numbers: the learner’s true risk, the best in your class, and the Bayes risk. Find the smallest budget that keeps the miss rate under 5%. Then ask yourself whether the cheapest promise is also the best filter.

[[lab-act2]]

## What people get wrong

**“Agnostic PAC with \(\varepsilon = 0.01\) means the filter is wrong at most 1% of the time.”** It means the filter’s true risk is at most 0.01 above the best hypothesis in the class. In the noisy mail room, with the one-feature class, that is up to 18.5%. The promise is relative to the class, and the class may be weak.

**“If I had enough data, I could reach zero true risk.”** Not when labels disagree. The Bayes risk is a floor under every hypothesis, and in the noisy mail room it is 0.125. Three readers in ten will always call the newsletter keep, and one hurried click in ten will always go the wrong way.

**“Minimizing empirical risk is the goal.”** It is the method, not the goal. ERM over all 65,536 hypotheses reached the smallest possible empirical risk on our twelve examples, 1/12, and had true risk 0.35, double that of \(h_F\). The goal is a small true risk. Empirical risk is only useful as long as it tracks true risk, and that is exactly what the next lesson must show.

**“PAC learnable means the learner finds the best hypothesis.”** It finds a hypothesis that is probably approximately as good. Both words stay in. Neither \(\varepsilon\) nor \(\delta\) can be zero, for the two reasons we met at the start. A training sequence can be useless with small but positive probability, and a finite one always misses fine details.

## Where these definitions come from

PAC learning was introduced by Valiant (1984), who won the 2010 Turing Award for it. Valiant’s definition required the sample complexity to be polynomial in \(1/\varepsilon\), \(1/\delta\) and the size of the hypotheses’ representation, and the runtime to be polynomial too. Lesson 6 will show that whenever a problem is PAC learnable at all, its sample complexity depends polynomially on \(1/\varepsilon\) and \(\log(1/\delta)\). The agnostic formalization is due to Haussler (1992), and the general setting with loss functions follows Vapnik and Chervonenkis (1971) and Vapnik’s later work. From here on, “PAC learning” means the most general model, agnostic with a general loss, and the loss is understood from context. When the realizable case is meant, we will say so. Lesson 7 discusses other notions of learnability.

## Recap and the next bug

1. **PAC learnability** asks for a sample complexity \(m_{\mathcal H}(\varepsilon, \delta)\) that does not depend on \(\mathcal D\) or \(f\). After that many examples, the learner’s true risk is at most \(\varepsilon\) with probability at least \(1 - \delta\). Every finite class is PAC learnable, with \(m_{\mathcal H}(\varepsilon, \delta) \le \lceil \log(|\mathcal H|/\delta)/\varepsilon \rceil\).
2. Realizable labels are rare. The agnostic model lets \(\mathcal D\) be a distribution over messages *and* labels, and redefines the true risk as \(\mathbb P_{(x, y) \sim \mathcal D}[h(x) \ne y]\).
3. The **Bayes optimal predictor** labels each message with its more likely label. No hypothesis has smaller true risk, by a message-by-message argument. It needs \(\mathcal D\), so no learner can use it directly.
4. **Agnostic PAC learnability** promises a true risk within \(\varepsilon\) of the best in the class, \(\min_{h \in \mathcal H} L_{\mathcal D}(h)\). That is not within \(\varepsilon\) of zero, and not within \(\varepsilon\) of the Bayes risk.
5. A **loss function** \(\ell: \mathcal H \times Z \to \mathbb R_+\) turns every task, from junk filtering to folders to regression, into one definition. True risk is expected loss, empirical risk is average loss, and ERM minimizes the second.

Your filter now ships with an honest promise. Look closely at what the lab did to give it, though. It knew \(\mathcal D\), so it could redraw the training sequence thousands of times and count the misses. The real mail room grants you one training sequence and never shows you \(\mathcal D\). In the realizable world, lesson 2 got around this by proving a bound. Can we do the same here?

Look at what would be enough. Suppose every hypothesis in the class had its empirical risk close to its true risk, all at once, on the same training sequence. Then the hypothesis with the smallest empirical risk would have nearly the smallest true risk too. On the twelve examples, \(h_F\) was off by 0.075. Was that bad luck, or normal? When are all the empirical risks close to their true risks at the same time, and can we prove it directly, with a formula for \(m\) and no simulation? That is the next lesson: uniform convergence.

## Concept check

Answer these in the chat before we move on to lesson 4. Show your reasoning.

1. Suppose \(\eta(x) = 1/2\) for every message: users split evenly on everything. What is the Bayes risk? Which hypotheses are Bayes optimal? What does an agnostic PAC learner for the one-feature class promise in this world, and is the promise worth anything?
2. A colleague writes: “Our learner is agnostic PAC for the one-feature class with \(\varepsilon = 0.02\) and \(\delta = 0.01\). So, given enough examples, at most 2% of messages will be mislabelled, guaranteed.” Find every mistake in that sentence, and give the correct statement for the noisy mail room with numbers.
3. A hypothesis may now answer any number \(c \in [0, 1]\) (any number from 0 to 1, both ends included), and we score it with the square loss. For the newsletter 0110, with \(\eta = 0.7\), which \(c\) minimizes the expected square loss on that message, and what is the minimum? Under the 0-1 loss, with answers in \(\{0, 1\}\), which answer is best? Why do the two losses pick different answers?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (monotonicity of the sample complexity).** Show that \(m_{\mathcal H}(\varepsilon, \delta)\) never increases when \(\varepsilon\) grows, and never increases when \(\delta\) grows.
- **Problem 2 (Boolean conjunctions).** Messages are \(d\) bits, and a hypothesis is an AND of features or their negations, such as “F and not N”. Prove the class is PAC learnable, bound its sample complexity, and give an ERM algorithm whose runtime is polynomial in \(d \cdot m\).
- **Problem 3 (harder).** A probabilistic hypothesis says junk with probability \(h(x) \in [0, 1]\), and its loss is \(|h(x) - y|\). Show that the Bayes optimal predictor still has the smallest risk, even against these.
