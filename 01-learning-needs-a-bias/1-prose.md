# Learning Needs a Bias

## The filter that learned nothing

Welcome to the mail room. Your company gets thousands of messages a day, and a good share of them are junk: prize draws, fake invoices, “FREE” in capital letters. Your job, across this whole course, is to build the learner behind the mail filter. People in the company will label some messages by hand, clicking “junk” or “keep”. The learner looks at those labelled messages and comes up with a way to label all the others.

You shipped version one last week. It was simple, and you were rather proud of it. It remembers every message a user has ever labelled, and when one of those messages arrives again, it gives back the user’s label. When a message it has never seen arrives, it plays safe and says “keep”.

The first report looked fantastic. On the messages your users labelled, the filter is right 100% of the time. Not one mistake.

Then the complaints started. The inbox is full of junk. When you check, the filter is right on new messages about as often as a coin would be. It has been shown hundreds of examples, it gets every one of them right, and yet it has learned nothing at all about junk.

How can a program be perfect on everything it was shown and useless on everything else? And what would it even mean to do better? This lesson is about that question, and it turns out to be the first question of all of learning theory.

## A rat and a pigeon

Let’s start not with computers but with two animals. They are worth meeting before we touch any math.

The first is a rat. Rats are wary of food they don’t know. When a rat comes across something new, it first eats only a very small amount. If that food makes it ill, the rat ties the illness to the food, and from then on it won’t eat it. This is called **bait shyness**. Notice what the rat is doing with its experience. It isn’t just remembering one bad meal: it avoids new pieces of food it has never met, as long as they smell and taste like the one that made it sick. Using past examples to label new ones is called **generalization**, and it is the heart of learning.

Now hold your version one up against the rat. A filter that stores the messages users marked as junk, and flags a new message only when it matches a stored one exactly, is **learning by memorization**. It looks a little like the rat, since it does learn from experience, but it is missing the part that matters: it has nothing to say about a message it hasn’t seen. A filter that generalizes would do something else. It would scan the junk your users labelled for a set of words whose appearance signals junk, and then use those words to label messages it has never seen.

Generalizing has a catch, though. Going from particular examples to a general rule is **inductive reasoning**, and it can lead to false conclusions. That brings us to the second animal.

The second is a pigeon. The psychologist B. F. Skinner put hungry pigeons in a cage and set a feeder to deliver food at regular intervals, with no reference whatsoever to what the birds were doing. When the first food arrived, each pigeon happened to be doing *something*: pecking, say, or turning its head. The bird tied the food to that action and did it more. That made it more likely to be doing the same thing when the next food arrived, which strengthened the link further. Skinner called the result superstition. The pigeons generalized too, but from a coincidence.

So both animals learned from examples, and both arrived at a rule that fits what they saw. One rule is useful and the other is nonsense. So here is the obvious question: what made the rat’s learning more successful than the pigeon’s?

Hold on to that question. Your version-one filter doesn’t generalize at all, which is bad enough. But by the end of this lesson you’ll see that it fails for the same reason the pigeon does, and you’ll know what the rat has that the pigeon lacks.

## A world small enough to hold in your hand

The real mail room has far too many possible messages to reason about, so let’s shrink it. Suppose every message is described by just four yes/no **features**:

- F: the subject says “FREE”.
- L: the message has a link.
- U: the sender is unknown.
- N: it was sent at night.

We write a message as four bits in that order. So 1011 means: says “FREE”, no link, unknown sender, sent at night. With four bits there are exactly \(2^4 = 16\) possible messages, and we’ll call the set of all of them \(\mathcal X\), the **domain set**. Every message gets a label from \(\mathcal Y = \{0, 1\}\), where 1 means junk and 0 means keep.

Now a secret, known to us but not to the learner: in this little world, a message is junk exactly when its subject says “FREE”. The other three features are pure noise. We’ll write this true labelling as \(f\), so \(f(x)\) is the first bit of \(x\).

Your users label eight messages. Each labelled message is an **example**, written \(z = (x, y)\): a message \(x\) and its label \(y\). The examples, in the order the learner sees them, form the **training sequence** \(S = (z_1, \dots, z_8)\). (People often say “training set”, but it is a sequence: order matters and repeats are allowed.) Here it is.

| i | message FLUN | label |
|---|---|---|
| 1 | 1111 | junk |
| 2 | 0000 | keep |
| 3 | 1110 | junk |
| 4 | 1011 | junk |
| 5 | 0001 | keep |
| 6 | 0100 | keep |
| 7 | 1101 | junk |
| 8 | 0010 | keep |

What the learner hands back is a **hypothesis**: a rule that gives a label to every possible message, not just the ones it has seen. Formally it is a function \(h: \mathcal X \to \mathcal Y\). (It is also called a predictor or a classifier; we’ll stick with hypothesis.) A **learner** is the algorithm that turns a training sequence into a hypothesis.

Version one is a learner we’ll call the **memorizer**. Its hypothesis is: if \(x\) is one of the training messages, return its label; otherwise return 0.

Let’s grade it. The eight messages it has not seen are 0011, 0101, 0110, 0111, 1000, 1001, 1010 and 1100. Before you read on, how many of those eight does it get wrong?

Here are all 16 messages at once, with the eight your users labelled outlined, the true label of each, and the memorizer’s mistakes marked.

[[diagram:messages]]

The memorizer says “keep” for all eight. Four of them start with a 0 and really are keep messages, so those are right by luck. The other four, 1000, 1001, 1010 and 1100, say “FREE” and are junk. Wrong, wrong, wrong, wrong. So on the unseen messages it is right exactly half the time, which is the coin from the opening.

We need names for the two scores. The **empirical risk** \(L_S(h)\) is the fraction of the training examples that \(h\) gets wrong. The **true risk** \(L_{\mathcal D}(h)\) is the fraction of all messages \(h\) gets wrong, weighted by how often each one actually arrives. In this tiny world we’ll assume every one of the 16 messages is equally likely, so the true risk is just “wrong ones out of 16”. (The \(\mathcal D\) stands for that distribution over messages; the next lesson makes it precise.) For the memorizer, \(L_S = 0\) and \(L_{\mathcal D} = 4/16 = 0.25\).

A hypothesis with empirical risk 0, one that agrees with every example, is called **consistent**. The memorizer is consistent. Clearly, being consistent is not enough.

## How many explanations fit?

Here is the question that cracks this open. The memorizer’s hypothesis is one consistent hypothesis. How many others are there?

A hypothesis on our tiny world is a choice of label for each of the 16 messages, so there are \(2^{16} = 65{,}536\) hypotheses in total. Watch what each example does to that count. The first example, 1111 is junk, rules out every hypothesis that says keep on 1111. That’s exactly half of them, leaving 32,768. The second example halves it again. After all eight examples, each pinning down one message, the survivors number

\[ 2^{16 - 8} = 2^8 = 256. \]

Two hundred and fifty-six hypotheses fit your users’ labels perfectly. They all agree on the eight seen messages, and between them they cover every possible way of labelling the eight unseen ones. One of the 256 is the truth. One is the memorizer. One labels every unseen message exactly backwards.

Now pick any unseen message, say 1000. How many of the 256 consistent hypotheses call it junk? Exactly 128. And the other 128 call it keep. The same is true for every unseen message. The examples have told you nothing at all about the messages you have not seen. A learner that is willing to consider every hypothesis has no grounds whatsoever for preferring one answer to the other.

[[diagram:funnel]]

The funnel above shows the two paths. Down the left, examples halve the count one at a time and still leave 256. Down the right, something else cuts the count first. Let’s see what.

Try it yourself in a bigger world, before we name the idea. The lab below has ten features, so 1,024 possible messages, and a hidden junk feature you don’t know. You choose how many examples the learners see, and you watch two of them: the memorizer, and a learner that only believes in one-feature hypotheses. There are ten of those, \(h_1, \dots, h_{10}\), and \(h_j\) says junk exactly when feature \(j\) is present. A wrong hypothesis that still agrees with every example so far is an **impostor**. We’ll study this learner properly in a moment; for now, watch its impostors.

[[lab]]

## What the rat knows

The answer comes from experiments by the psychologists Garcia and Koelling, and it is wonderfully specific. Suppose the unpleasant thing that follows a new food is not nausea but an electric shock. Then the rats do not learn to avoid the food, even after repeated trials. Or suppose nausea follows not a new taste but a sound. Then the rats do not learn to avoid the sound. They readily connect a taste with nausea, and they ignore other coincidences in time, however often those coincidences repeat.

Is that a flaw? Not in the rat’s world. There, it is very unlikely that a sound causes nausea, so a rat that never even considers that explanation loses nothing, and it gains a great deal: a coincidence of that kind can never fool it. The rat comes with **built-in** prior knowledge about which explanations are worth taking seriously.

That is the central idea of this lesson. The **inductive bias** of a learner is the prior knowledge it brings before seeing any data, knowledge that biases what it is able to learn. The pigeon has none to speak of: it is willing to adopt any explanation at all for the arrival of food, so whatever it happened to be doing becomes the explanation.

How do you build a bias into an algorithm? The next lesson makes this fully precise, but we can borrow its tool now. You fix, in advance, a **hypothesis class** \(\mathcal H\): the set of hypotheses the learner is allowed to return. The learner looks at the examples and returns a hypothesis from \(\mathcal H\) that fits them, and it never strays outside.

Your memorizer is the pigeon. Strictly speaking it does make a choice, since it always says keep for a message it hasn’t seen, but that choice is a tie-breaker, not knowledge: nothing in it has anything to do with junk. As far as the examples are concerned, every one of the 65,536 hypotheses was on the table.

A bias has a price, and it’s worth stating plainly. The stronger the prior knowledge, the easier it is to learn from further examples. But the stronger it is, the less flexible the learner becomes, because it is committed in advance to its assumptions. And some bias is unavoidable: lesson 5 proves a precise version of that claim, the **No-Free-Lunch theorem**.

## A bias, one example at a time

Let’s give the learner a bias and watch it work. The new hypothesis class has only four members, one per feature:

\[ \mathcal H_1 = \{ h_F, h_L, h_U, h_N \}, \qquad h_j(x) = x_j, \]

so \(h_U\) says “junk exactly when the sender is unknown”. Our learner, call it **consistent-feature**, keeps the hypotheses in \(\mathcal H_1\) that agree with every example and returns one of them, picked at random. It has to pick blindly: it doesn’t know which feature is the real one, and listing F first is just how we humans wrote the class down. The truth, \(h_F\), is in the class, so at least one consistent hypothesis always exists.

Feed it the same eight examples, one at a time, and track which of the four hypotheses survive. Notice two things as you go down the table: how quickly the class shrinks compared with 65,536, and which impostor hangs on the longest.

| i | message FLUN | label | survivors in \(\mathcal H_1\) | survivors among all hypotheses |
|---|---|---|---|---|
| 1 | 1111 | junk | F, L, U, N | 32,768 |
| 2 | 0000 | keep | F, L, U, N | 16,384 |
| 3 | 1110 | junk | F, L, U | 8,192 |
| 4 | 1011 | junk | F, U | 4,096 |
| 5 | 0001 | keep | F, U | 2,048 |
| 6 | 0100 | keep | F, U | 1,024 |
| 7 | 1101 | junk | F | 512 |
| 8 | 0010 | keep | F | 256 |

After eight examples the biased learner has exactly one survivor, \(h_F\), the truth. Its true risk is 0 on all 16 messages, including the eight it never saw. The unbiased learner still has 256 candidates and no way to choose.

But look at rows 4 to 6. For three whole examples, “unknown sender means junk” fit every label perfectly. If your users had stopped at six examples, the learner would have had two consistent hypotheses and a coin flip between them. Half the time it would return \(h_U\), which is wrong on half of all messages: \(h_U\) and \(h_F\) disagree whenever the U bit and the F bit differ, which is 8 of the 16. That is a superstition, alive inside a perfectly good bias. A bias makes learning possible. It does not make coincidences impossible; only more examples do that.

So there are two quite different ways for a learner to fail, and it’s worth naming them now, because the rest of the course keeps returning to them.

- **The pigeon’s failure: a class too large for the examples.** So many hypotheses fit the examples that a coincidence can win. The memorizer is the extreme case, and \(h_U\) after six examples is a mild one. More examples, or a smaller class, cure it.
- **The rat’s failure: a class that misses the truth.** Put the rat in a world where sickness really is signalled by a click, and its bias guarantees it will never learn that. No number of examples helps, because the right hypothesis was never allowed in.

Lesson 5 turns these two failures into a precise tradeoff.

I should confess that I chose these eight examples so that the impostor lasts. If your users picked messages at random, how often would a coincidence like that survive? That we can compute exactly, and it is where the math begins.

## The formal statements

Let’s collect the definitions, then prove two facts. Everything below lives in our tiny world, but every step is written so it carries over to the general setting of the next lesson.

**Setting.** The domain set is \(\mathcal X = \{0,1\}^4\) with \(|\mathcal X| = 16\). The label set is \(\mathcal Y = \{0, 1\}\). The true labelling is \(f(x) = x_F\). The distribution \(\mathcal D\) is uniform on \(\mathcal X\). An example is \(z_i = (x_i, y_i)\) with \(y_i = f(x_i)\), and a training sequence is \(S = (z_1, \dots, z_m)\). A learner is an algorithm \(A\), and \(A(S)\) is the hypothesis it returns on \(S\). When \(S\) is random, we write \(S \sim \mathcal D^m\): the messages \(x_1, \dots, x_m\) are drawn **independently and identically distributed** (i.i.d.) from \(\mathcal D\), meaning each is drawn from the same \(\mathcal D\) and no draw influences another.

**Risks.** These two formulas are the two scores every later lesson keeps comparing, so let’s read them slowly. Here is the first one, the empirical risk of a hypothesis \(h: \mathcal X \to \mathcal Y\):

\[ L_S(h) = \frac{1}{m} \left| \{ i \in [m] : h(x_i) \neq y_i \} \right|. \]

Read it from the inside out.

- \([m]\) is shorthand for the list of positions \(\{1, 2, \dots, m\}\): example 1, example 2, up to example \(m\).
- \(h(x_i) \neq y_i\) asks one yes/no question about example \(i\): does the hypothesis’s label for message \(x_i\) differ from the label \(y_i\) your users gave it? “Yes” means a mistake.
- The braces \(\{ i \in [m] : \dots \}\) collect every position where the answer is yes. Read the colon as “such that”: the set of positions \(i\) such that \(h\) gets example \(i\) wrong.
- The vertical bars \(|\cdot|\) around a set mean “how many elements it has”. So \(|\{\dots\}|\) is the number of mistakes.
- Dividing by \(m\) turns that count into a fraction, so \(L_S(h)\) is always between 0 and 1.
- The subscript \(S\) in \(L_S\) says which messages are being scored: the training sequence and nothing else.

In plain words: **the fraction of your labelled examples that \(h\) gets wrong.** Take the eight examples of the table and \(h_U\), “unknown sender means junk”. It is right on examples 1 to 6, then misses example 7 (1101: known sender, but junk) and example 8 (0010: unknown sender, but keep). The set of mistakes is \(\{7, 8\}\), its size is 2, and \(L_S(h_U) = 2/8 = 0.25\).

The second formula is the true risk:

\[ L_{(\mathcal D, f)}(h) = \mathbb P_{x \sim \mathcal D}\left[ h(x) \neq f(x) \right]. \]

- \(f\) is the true labelling, the rule your users follow in their heads; \(f(x)\) is the label message \(x\) really deserves.
- \(x \sim \mathcal D\) reads “\(x\) is drawn from \(\mathcal D\)”: a fresh message arrives at random, with each message as likely as \(\mathcal D\) says.
- \(\mathbb P[\dots]\) is the probability of the event inside the brackets. Here the event is \(h(x) \neq f(x)\): \(h\) gets that fresh message wrong.

In plain words: **the chance that \(h\) mislabels the next message to arrive.** In our tiny world \(\mathcal D\) is uniform, so every message has probability \(1/16\) and the probability is just a count: \(L_{\mathcal D}(h)\) is the number of messages \(h\) gets wrong, divided by 16. The hypothesis \(h_U\) disagrees with the truth on 8 of the 16 messages, so its true risk is \(8/16 = 0.5\).

The two formulas have the same shape on purpose. The first counts mistakes on the messages you happened to label; the second weighs mistakes over every message that could ever arrive. The whole course is about when the first is a trustworthy estimate of the second. Here they are side by side in Python, on the eight examples from the table above:

```python
from itertools import product

f = lambda x: x[0]
h_U = lambda x: x[2]
bits = ["1111", "0000", "1110", "1011",
        "0001", "0100", "1101", "0010"]
S = [(tuple(map(int, b)), int(b[0])) for b in bits]
mistakes = [i for i, (x, y) in enumerate(S, 1) if h_U(x) != y]
L_S = len(mistakes) / len(S)
X = list(product([0, 1], repeat=4))
L_D = sum(h_U(x) != f(x) for x in X) / len(X)
print(mistakes, L_S, L_D)
```

It prints `[7, 8] 0.25 0.5`: \(h_U\) looks only a little wrong on the examples, and is in fact wrong on half of all mail.

The true risk depends on both \(\mathcal D\) and \(f\), which is why it is written \(L_{(\mathcal D, f)}\). When \(f\) is clear from context, as it is throughout this lesson, we shorten it to \(L_{\mathcal D}(h)\). You will also see it called the generalization error or the true error. A hypothesis is consistent with \(S\) when \(L_S(h) = 0\): it makes no mistakes on the examples.

**Proposition 1 (the examples say nothing about unseen messages).** Let \(\mathcal X\) be any finite domain, with \(|\mathcal X|\) elements, and let \(S\) contain \(k\) distinct messages. Assume the labels in \(S\) don’t conflict: a message that appears more than once always carries the same label. Then exactly \(2^{|\mathcal X|-k}\) hypotheses \(h: \mathcal X \to \{0,1\}\) are consistent with \(S\). Moreover, for every message \(x'\) not in \(S\), exactly half of them have \(h(x') = 1\).

*Proof.* A hypothesis is a free choice of one label for each of the \(|\mathcal X|\) messages. Being consistent forces the label of each of the \(k\) messages in \(S\) (and because the labels don’t conflict, each forced label is well defined). The other \(|\mathcal X| - k\) labels are still free, with two options each, so there are \(2^{|\mathcal X|-k}\) consistent hypotheses.

For the second claim, fix an unseen \(x'\) and let \(\varphi\) be the map that flips a hypothesis’s label at \(x'\) and leaves every other label alone. Since \(x'\) is not in \(S\), flipping it changes nothing on the examples, so \(\varphi\) sends consistent hypotheses to consistent hypotheses. Applying \(\varphi\) twice gives back the original hypothesis, so \(\varphi\) is its own inverse, and therefore a bijection from the consistent set to itself. Finally, \(\varphi\) sends every consistent hypothesis with \(h(x') = 1\) to one with \(h(x') = 0\) and vice versa, so a bijection pairs the two groups off and they have the same size, \(2^{|\mathcal X|-k-1}\) each. \(\blacksquare\)

[[diagram:flip]]

You don’t have to take the counting on trust. Python can try all 65,536 hypotheses, keep the consistent ones, and ask them about 1000:

```python
import itertools

X = list(itertools.product([0, 1], repeat=4))
S = {(1, 1, 1, 1): 1, (0, 0, 0, 0): 0, (1, 1, 1, 0): 1,
     (1, 0, 1, 1): 1, (0, 0, 0, 1): 0, (0, 1, 0, 0): 0,
     (1, 1, 0, 1): 1, (0, 0, 1, 0): 0}

def consistent(h):
    return all(h[X.index(x)] == y for x, y in S.items())

every_h = itertools.product([0, 1], repeat=len(X))
H = [h for h in every_h if consistent(h)]
print(len(H))
print(sum(h[X.index((1, 0, 0, 0))] for h in H))
```

It prints 256 and 128: a hypothesis here is just a tuple of 16 labels, and the proposition is a statement about how many tuples survive the filter.

With \(|\mathcal X| = 16\) and \(k = 8\), that is the 256 and the 128 we counted by hand. Notice what the proposition does and doesn’t say. It says the examples alone leave every unseen label perfectly balanced, so a learner with no bias has nothing to go on. It does not yet say that every learner must fail; a learner whose tie-breaker happens to match the truth would get lucky. Turning “nothing to go on” into “for every learner there is a world where it fails” is the **No-Free-Lunch theorem** of lesson 5, and this counting is its seed.

**Proposition 2 (how long a coincidence survives).** Let \(S \sim \mathcal D^m\) in our tiny world. For each wrong hypothesis \(h_j \in \{h_L, h_U, h_N\}\),

\[ \mathbb P_{S \sim \mathcal D^m}\left[ L_S(h_j) = 0 \right] = 2^{-m}, \]

and the probability that at least one of the three survives is

\[ \mathbb P\left[ \exists j \in \{L, U, N\} : L_S(h_j) = 0 \right] = 1 - \left(1 - 2^{-m}\right)^3 \le 3 \cdot 2^{-m}. \]

*Proof.* On one message \(x\), the hypothesis \(h_j\) agrees with the truth exactly when \(x_j = x_F\).

Under the uniform \(\mathcal D\), the four bits of \(x\) are independent fair coins, so \(\mathbb P[x_j = x_F] = 1/2\). Equivalently, \(L_{\mathcal D}(h_j) = 1/2\): 8 of the 16 messages, as we saw for \(h_U\).

\(h_j\) survives \(S\) only if it agrees on all \(m\) messages. The messages are independent, so the probabilities multiply: \(\mathbb P[L_S(h_j) = 0] = (1/2)^m = 2^{-m}\).

For the three together, we need a little more care, in two steps.

First, within one message. Define three agreement bits, \(b_j = \mathbb 1[x_j = x_F]\), using indicator notation: \(\mathbb 1[\text{statement}]\) is 1 when the statement is true and 0 when it is false. Fix the value of \(x_F\). Then each \(b_j\) is a function of \(x_j\) alone, and \(x_L, x_U, x_N\) are independent fair coins, so \(b_L, b_U, b_N\) are independent fair coins. That is a statement about the case \(x_F = 0\) and, separately, the case \(x_F = 1\). But the answer is the same joint distribution in both cases, so it doesn’t depend on \(x_F\), and averaging over \(x_F\) gives it back unchanged: without any conditioning, \(b_L, b_U, b_N\) are independent fair coins.

Second, across messages. The \(m\) messages are drawn independently, so the bits of different messages are independent of each other. Together with the first step, all \(3m\) bits \(b_j^{(i)}\), for \(j \in \{L, U, N\}\) and \(i \in [m]\), are mutually independent. The event “\(h_j\) survives” is “\(b_j^{(i)} = 1\) for every \(i\)”, which depends only on the \(m\) bits in row \(j\). Events built from disjoint groups of independent bits are independent, so the three survival events are independent.

So the probability that none of them survives is \((1 - 2^{-m})^3\), and its complement is the probability that at least one survives.

For the bound, use \((1 - a)^3 \ge 1 - 3a\) for \(0 \le a \le 1\), which holds because \((1-a)^3 - (1 - 3a) = 3a^2 - a^3 = a^2(3 - a) \ge 0\). \(\blacksquare\)

Put numbers in. With six random examples, the chance that some impostor is still consistent is \(1 - (63/64)^3 \approx 4.6\%\). With eight, it is \(1.167\%\), and the bound \(3/256\) is \(1.172\%\), almost exactly the same. Each extra example roughly halves the danger.

Here is the same claim as an experiment: draw random training sequences, check whether any impostor survives, and compare with the exact formula and the bound.

```python
import itertools
import random

X = list(itertools.product([0, 1], repeat=4))

def impostor_survives(m, rng):
    S = [rng.choice(X) for _ in range(m)]
    return any(all(x[j] == x[0] for x in S) for j in (1, 2, 3))

rng, trials = random.Random(0), 100_000
for m in (6, 8):
    hits = sum(impostor_survives(m, rng) for _ in range(trials))
    seen = hits / trials
    exact = 1 - (1 - 2**-m) ** 3
    print(m, round(seen, 4), round(exact, 4), round(3 * 2**-m, 4))
```

It prints `6 0.0471 0.0461 0.0469` and `8 0.0118 0.0117 0.0117`: the simulation lands on the exact value, and the bound sits just above it. Notice what `impostor_survives` checks: hypothesis \(h_j\) survives when bit \(j\) equals the F bit on every message, which is exactly the event \(L_S(h_j) = 0\).

Look at the shape of that bound: the number of wrong hypotheses in the class, times a factor that shrinks exponentially with \(m\). A smaller class means fewer impostors to rule out, so fewer examples are needed. The rat’s class is tiny, so one meal will do. The next lesson proves the general version of this for any finite class and any distribution, using the **union bound**: the probability that at least one of several events happens is at most the sum of their probabilities. (You can see it in the \(3 \cdot 2^{-m}\) above: three impostors, each surviving with probability \(2^{-m}\).) That becomes the first real theorem of the course.

## The two learners as code

Here are both learners in pseudocode. Each takes a training sequence \(S\) of \(m\) examples over \(d\) features.

```
Memorize(S):
  table ← empty hash map
  for i = 1..m:
    table[x_i] ← y_i
  return h(x) = table[x] if x in table, else 0

ConsistentFeature(S):
  alive ← {1, ..., d}
  for i = 1..m:
    for j in alive:
      if x_{i,j} ≠ y_i: remove j from alive
  if alive is empty: return FAIL
  j ← a uniformly random element of alive
  return h(x) = x_j
```

Here \(x_{i,j}\) is feature \(j\) of the \(i\)-th message.

**Memorize** takes \(O(md)\) time to build, since hashing each \(d\)-bit message costs \(O(d)\), and \(O(md)\) space. Each prediction costs \(O(d)\) expected time. It is fast and it is cheap, and it fails for reasons that have nothing to do with computation. It considers all \(2^{2^d}\) hypotheses, and Proposition 1 says the examples leave every unseen label balanced among them, so whatever it says off the training sequence is decided by its tie-breaker, not by the data.

**ConsistentFeature** takes \(O(md)\) time and \(O(d)\) space for the set of survivors. It is the simplest instance of a pattern you will see in every lesson from here on: search a fixed class \(\mathcal H\) for a hypothesis that does well on \(S\). The next lesson names it empirical risk minimization. For a finite class, the brute-force version costs \(O(m \cdot |\mathcal H|)\) hypothesis evaluations, which is fine for four hypotheses and hopeless for \(2^{2^d}\). How the statistical cost and the computational cost of a class pull against each other is the subject of lesson 8.

**Build it yourself.** Using only the Python standard library:

```python
import itertools
import random

S = [((1, 1, 1, 1), 1), ((0, 0, 0, 0), 0), ((1, 1, 1, 0), 1),
     ((1, 0, 1, 1), 1), ((0, 0, 0, 1), 0), ((0, 1, 0, 0), 0),
     ((1, 1, 0, 1), 1), ((0, 0, 1, 0), 0)]

def f(x):
    return x[0]

def memorize(S): ...
def consistent_feature(S, d, rng): ...
def true_risk(h, f, d): ...

if __name__ == "__main__":
    assert true_risk(memorize(S), f, 4) == 0.25
    h = consistent_feature(S, 4, random.Random(0))
    assert true_risk(h, f, 4) == 0.0
    h = consistent_feature(S[:6], 4, random.Random(1))
    assert true_risk(h, f, 4) in (0.0, 0.5)
```

1. Implement the three functions, where `S` is a list of `(x, y)` pairs, `x` is a tuple of `d` bits, and `true_risk` averages mistakes over all \(2^d\) messages.
2. Reproduce the survivor table from “A bias, one example at a time” and the memorizer’s true risk of 0.25.
3. For \(d = 4\), draw \(S\) at random 100,000 times for each \(m\) from 1 to 10, and measure how often some wrong one-feature hypothesis is still consistent. Plot it against \(1 - (1 - 2^{-m})^3\).
4. Repeat for \(d = 10\). Predict the formula first.

## Why learn at all?

Before we fix the filter, a fair question from an engineer: why learn the rule? You could write “junk if the subject says FREE” and ship it. There are two good reasons to learn instead of program, and the mail room has both.

The first is that the task is **too complex to program**, and that comes in two kinds. Some tasks we do every day without being able to explain how: driving, speech recognition, image understanding. You cannot write down the rules because you do not know them, yet a learner trained on enough examples can do well. Other tasks are beyond human capabilities altogether, because they mean finding meaningful patterns in data sets too large and too complex for anyone to read. Think of astronomical data, turning medical archives into medical knowledge, weather prediction, genomic data, Web search engines and electronic commerce.

The second reason is **adaptivity**. A program, once written, stays as it was written. But many tasks change over time, or from one user to another. Think of programs that decode handwriting and adapt to each writer, junk filters that adapt as junk changes, and speech recognizers that adapt to each voice. A learner can be trained again on fresh examples, which a hand-written rule cannot.

That second reason is about to bite.

## Fix the bug

The spammers noticed. This week, junk no longer says “FREE”. It is marked by some other feature, and nobody knows which. A hand-written rule like “junk if the subject says FREE” would be dead in the water. A learner can follow them, if you give it the right bias and enough examples.

In the lab below you choose two things: the hypothesis class the learner searches, and how many examples your users label. Every label costs a person a click, so you have a budget. Each round, the spammers’ feature changes. Your goal is a filter whose true risk is under 1% on fresh messages, and not by luck: the chance that a wrong hypothesis is still consistent must be under 1% too.

Two of the classes on offer are new. One is narrower: one feature, but only among the first three. Small is good, right up until the junk feature isn’t one of them.

The other is bigger. Besides the ten one-feature hypotheses, it also contains every pair \(h_j \wedge h_k\), which says junk only when features \(j\) and \(k\) are both present. That makes 10 single features plus \(\binom{10}{2} = 45\) pairs, so 55 hypotheses. It contains the truth, but its impostors are not all alike, and that matters. A pair without the junk feature is wrong on half the messages, like a single impostor. But a pair that includes it, say the junk feature is \(j^*\) and the pair is \(h_{j^*} \wedge h_k\), disagrees with the truth only when \(j^*\) is present and \(k\) is absent: a quarter of the messages. So it survives each example with probability \(3/4\), not \(1/2\), and Proposition 2’s \(2^{-m}\) is too optimistic for it. The argument still works if you give each impostor its own survival chance. There are 9 of the slow kind and 45 of the fast kind, so

\[ \mathbb P[\text{some impostor survives}] \le 9 \left(\tfrac{3}{4}\right)^m + 45 \cdot 2^{-m}. \]

```python
for m in (10, 20):
    print(m, round(9 * 0.75**m + 45 * 0.5**m, 4))
```

It prints `10 0.5508` and `20 0.0286`: even at 20 labels, almost a 3% chance of a coincidence. Work out the label count each class needs for 1%, and compare it with the budget, before you train.

[[lab-act2]]

## What people get wrong

**“It’s right on all the training data, so it works.”** Being consistent says nothing on its own. Proposition 1 shows that for every unseen message, half of the consistent hypotheses are wrong. The memorizer is consistent and has true risk 0.25.

**“A bias is a flaw to remove.”** It is the opposite. Without a bias, learning is impossible: the examples cannot tell the consistent hypotheses apart. The real danger is a *wrong* bias, a class that does not contain anything close to the truth. The rat’s blindness to sounds is the right bias in its own world, where sounds don’t cause nausea. Move it to a world where they do, and the same bias becomes a wrong one.

**“With a good bias, a few examples are enough.”** A good bias makes few examples enough *with high probability*, not always. In the tiny world, with six random examples, an impostor survives about 4.6% of the time. The guarantee is always “probably”, and the next lessons make that word exact.

## Kinds of learning

Let’s step back from the mail room and see where this problem sits. Learning problems differ along four dimensions. It’s worth knowing where we stand, because the answer changes the math.

[[diagram:taxonomy]]

**Supervised or unsupervised?** Think of learning as using experience to gain expertise. In **supervised** learning, the experience carries extra information that the test examples lack: your users’ junk and keep clicks. The environment acts as a teacher that supervises, and the learner’s job is to supply the missing label for messages that arrive without one. In **unsupervised** learning there is no such distinction between training and test data. The learner gets messages with no labels and has to come up with a summary of them. A classic example is anomaly detection: flagging the unusual messages. Clustering, grouping similar items together, is another.

In between sits **reinforcement learning**. Here the training examples carry more information than the test examples, yet the learner must predict even more information at test time. Chess is the classic example. Training gives you whole games labelled with who eventually won. What the learner has to produce is a value for each board position, something no example states directly.

**Active or passive?** A **passive** learner simply observes what the environment provides, and your spam filter is the typical case: it waits for users to label whatever messages they happen to label. An **active** learner interacts with the environment during training, for example by asking users to label messages that the learner itself has chosen, or even composed.

**How helpful is the teacher?** A teacher can be helpful, as with a child learning or a student in a class. Nature, which a scientist learns from, is passive and indifferent: the examples come out at random. This is the **statistical** setting, and it’s what \(S \sim \mathcal D^m\) will mean. And a teacher can be adversarial, as with spammers, or with fraudsters trying to fool a fraud detector. The adversarial teacher also serves as a worst case: a learner that succeeds against an adversary is guaranteed to succeed with any teacher at all.

**Batch or online?** In **batch** learning, the learner gets a large body of training data first and must produce its hypothesis afterwards, like a data miner analysing a big archive. In **online** learning, the learner has to respond throughout the learning process, like a stockbroker who must make decisions every day using only the experience gathered so far. Expertise built online is usually costlier, since early mistakes are paid for along the way.

This course mostly studies **supervised statistical batch learning with a passive learner**, such as learning to predict patients’ outcomes from archives of past cases labelled with how each one turned out. Online learning comes back in lesson 21, and unsupervised learning, in particular clustering, in lesson 22.

## Where learning theory sits

Last, where does the subject sit among its neighbours? Machine learning is naturally a subfield of computer science, and it shares common ground with mathematics, statistics, information theory, game theory and optimization. Two relationships matter most.

**Artificial intelligence.** In a sense machine learning can be viewed as a branch of AI, since turning experience into expertise is a mark of intelligence. But the goal is different from building an automated imitation of intelligent behaviour. Learning is meant to complement human intelligence and to use the strengths of computers, which often means doing what people cannot, such as finding patterns in data far outside the scope of human perception.

**Statistics.** Both learn from data, and they share a great deal, but the emphasis differs in four ways.

1. A statistician often starts with a hypothesis and tests it on data (for instance, whether smoking is linked to heart disease). Machine learning tries to generate the hypothesis from the data itself.
2. Algorithmic questions are pivotal in machine learning. A learner is a program that has to run on a computer, so its computational efficiency matters as much as its statistical guarantee.
3. Traditional statistics often studies what happens as the sample grows without end. Machine learning theory asks for bounds at a specific, finite sample size \(m\), exactly the kind of statement Proposition 2 made.
4. Statistics often assumes the data come from a prescribed model, such as a particular family of distributions. Machine learning theory tries to assume as little as possible about \(\mathcal D\), ideally nothing: a **distribution-free** guarantee. Lesson 5 shows how far that ambition can go.

## Recap and the next bug

1. A learner turns a training sequence \(S\) into a hypothesis \(h\). Its empirical risk \(L_S(h)\) is how often it errs on the examples, and its true risk \(L_{\mathcal D}(h)\) is how often it errs on fresh messages.
2. A learner that considers every hypothesis learns nothing about unseen messages: exactly half of the consistent hypotheses say junk for each of them (Proposition 1).
3. An inductive bias, a hypothesis class fixed before seeing data, is what makes learning from finitely many examples possible. The rat has one; the pigeon does not.
4. A bias does not rule out coincidences. In a small class they die off exponentially fast with \(m\) (Proposition 2).
5. We learn instead of program when the rule is too complex to write or the world keeps changing. This course focuses on supervised statistical batch learning with a passive learner, and also covers online learning and clustering.

Your filter works again. Then your manager drops by. The real mail system has thirty features, not ten, and the people labelling messages are expensive: you can afford a few hundred labels, not thousands. Which hypothesis classes can you afford with that budget? And when the learner picks a hypothesis that fits all your examples, how sure can you be that its true risk is small? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The rat’s bias lets it learn from a single meal. Describe a world in which that same bias makes the rat do *worse* than a learner with no bias at all. Which of the two failure modes from this lesson is it, and why can more examples not rescue it?
2. In the proof of Proposition 1, the map \(\varphi\) flips a hypothesis’s label at one unseen message \(x'\). Explain why \(\varphi\) sends consistent hypotheses to consistent hypotheses, and why it is a bijection. Then: what happens to the count \(2^{|\mathcal X|-k}\) if \(S\) contains the same message twice, once labelled junk and once labelled keep?
3. Now there are \(d = 20\) features, \(\mathcal D\) is uniform on \(\{0,1\}^{20}\), and the truth is one of the twenty one-feature hypotheses. Extend Proposition 2 to this case. What is the smallest \(m\) for which your bound guarantees that some wrong one-feature hypothesis survives with probability at most 0.001? For comparison, after that many examples, what fraction of all possible messages can the memorizer possibly have seen?
