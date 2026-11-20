# Generative Models

## The filter that was sure of everything

For twenty-three lessons, every filter you built did the same kind of thing. It looked at labelled messages and drew a boundary: a threshold, a halfspace, a tree of questions, a margin, a nearest neighbour. Junk on one side, keep on the other. The learner never asked where the messages came from. It only asked where to put the line.

This week a colleague, Ines, walks over with a different idea. “Stop drawing boundaries,” she says. “Junk is written by people who want something from you. Keep mail is written by colleagues who want lunch. Each kind of message is produced in its own way, word by word. So let’s learn *how each kind is produced*. When a new message arrives, we ask: which of the two producers most likely wrote this?”

It is a lovely idea, and she has already built it. For each word in the dictionary, she counts how often it appears in junk and how often in keep. Those fractions become her estimates of how each producer behaves. A new message gets a score under each producer, and the filter picks the larger one.

On Monday the filter is brilliant on ordinary mail. On Tuesday the bug reports arrive, and they are strange. A message that reads like a textbook prize draw, “free”, “click”, “claim your prize”, was filed as keep with probability exactly 1. Not 0.99. One. Another message, containing the word “winner”, did not get filed at all: the filter threw an exception and the message sat in limbo. When you look closer, the pattern is always the same. Every message that misbehaves contains some word the filter had never seen in one of the two piles. One unseen word, and the filter goes from “probably” to absolute certainty, or to a crash.

How can counting words, the most honest estimate imaginable, produce a filter that is certain of nonsense? And what would it mean to count *correctly*? This lesson builds the answer from the ground up. Along the way you will meet the most popular estimation principle in all of statistics, and find out that it is a learning rule you already know in disguise.

## Telling the story of a message

Let’s name the two approaches before we look at Ines’s numbers.

Every filter in lessons 2 to 23 followed the **discriminative** approach: the goal is a hypothesis that labels messages accurately, and nothing more. We never tried to say how likely a given message is to arrive; we only asked which side of a boundary it lands on. Lesson 3 made one big promise about this approach: it is **distribution free**, meaning the guarantees hold for every distribution \(\mathcal D\) over messages and labels, with no assumption at all about its shape.

The **generative** approach makes the opposite bet. It assumes the distribution \(\mathcal D\) has a specific form described by a handful of numbers, called **parameters**, and its goal is to estimate those parameters from the training sequence. A distribution described this way is a **parametric model**, and estimating its parameters from data is called **parametric density estimation**. Ines’s model has one parameter per word and per kind of message: the probability that a message of that kind contains that word.

Why would anyone take the generative route, when it means assuming more? A principle usually credited to the statistician Vladimir Vapnik warns against it: *when solving a given problem, try to avoid a more general problem as an intermediate step.* Learning the whole distribution is a more general problem than learning to label, and usually a harder one. But there are good reasons to try anyway. If you do learn the distribution, you can predict with the Bayes optimal predictor of lesson 3. (A reminder: that is the hypothesis that gives every message whichever label is more probable for it, and no hypothesis has smaller true risk.) Sometimes estimating the parameters is computationally easier than finding a good discriminative hypothesis. And sometimes there is no single task yet: you want a model of the mail you can query later for questions nobody has asked, or one a person can read and understand.

[[diagram:generative]]

So the question of this lesson is: how should you estimate the parameters, and what can you promise about the result?

## Ines’s filter, by hand

Let’s rebuild Ines’s filter on a dictionary small enough to hold in your head. It has seven words: “free”, “prize”, “click”, “invoice”, “meeting”, “lunch” and “winner”. A message is described by which of these words it contains, one yes/no feature per word, just as messages were vectors of yes/no features in lesson 1. Label 1 is junk and 0 is keep.

Your users labelled eight messages, four of each kind:

| message | words | label |
|---|---|---|
| 1 | free, prize, click | junk |
| 2 | free, click | junk |
| 3 | prize, click | junk |
| 4 | free, prize | junk |
| 5 | meeting, lunch | keep |
| 6 | invoice, meeting | keep |
| 7 | lunch, click | keep |
| 8 | free, lunch, meeting | keep |

Ines estimates, for each word and each label, the fraction of messages with that label that contain the word. Call the estimate for word \(j\) and label \(y\) by the name \(\hat\theta_{j,y}\). The hat on top is the usual mark for “estimated from data”, and \(\theta\) (the Greek letter theta) is the usual letter for a parameter. Count before you read on: what is \(\hat\theta_{\text{invoice},1}\), the fraction of the junk messages that contain “invoice”?

None of the four junk messages contains it, so the fraction is \(0/4 = 0\). Here is the full table.

| word | in junk \(\hat\theta_{j,1}\) | in keep \(\hat\theta_{j,0}\) |
|---|---|---|
| free | 3/4 | 1/4 |
| prize | 3/4 | 0 |
| click | 3/4 | 1/4 |
| invoice | 0 | 2/4 |
| meeting | 0 | 3/4 |
| lunch | 0 | 3/4 |
| winner | 0 | 0 |

Now her scoring recipe. A producer of junk writes a message by flipping one biased coin per word: word \(j\) goes in with probability \(\theta_{j,1}\), and stays out with probability \(1 - \theta_{j,1}\), each coin independent of the others. So the probability that the junk producer writes exactly the message \(x\) is a product of seven factors, one per word. Writing \(x_j = 1\) when word \(j\) is in the message and \(x_j = 0\) when it is not:

\[ \mathbb P[x \mid y] = \prod_{j=1}^{d} \theta_{j,y}^{\,x_j} \, (1 - \theta_{j,y})^{1 - x_j}. \]

This is the first formula of the lesson, so let’s take it apart from the inside out.

- \(d\) is the number of words in the dictionary, here 7, and \(j\) runs over them: \(j = 1\) is “free”, \(j = 2\) is “prize”, and so on.
- \(\theta_{j,y}\) is the probability that a message with label \(y\) contains word \(j\). The two subscripts say which word and which label.
- \(\theta_{j,y}^{\,x_j} (1 - \theta_{j,y})^{1 - x_j}\) is a compact way to write a choice. If \(x_j = 1\), the exponents are 1 and 0, and since anything to the power 0 is 1, the factor is \(\theta_{j,y}\): the chance the word went in. If \(x_j = 0\), it is \(1 - \theta_{j,y}\): the chance it stayed out.
- \(\prod_{j=1}^{d}\) (a capital Greek pi) means “multiply together, for \(j\) from 1 to \(d\)”. It is to multiplication what \(\sum\) is to addition. We may multiply because the coins are independent: the probability that several independent things all happen is the product of their probabilities.
- \(\mathbb P[x \mid y]\) is read “the probability of \(x\) given \(y\)”: the vertical bar means “given”, so this is the chance of writing exactly message \(x\) *if* the writer is the producer of kind \(y\).

In one sentence: **the chance that the producer of kind \(y\) writes exactly this message is the product, over every dictionary word, of the chance that it put that word in or left it out, as the message says.**

Push message 2, “free, click”, through the junk producer. “free” is in: \(3/4\). “prize” is out: \(1 - 3/4 = 1/4\). “click” is in: \(3/4\). The four words with \(\hat\theta = 0\) are out, each with probability \(1 - 0 = 1\). The product is \(\tfrac34 \cdot \tfrac14 \cdot \tfrac34 = 9/64 \approx 0.1406\).

To decide, Ines also needs the share of junk in the mail, \(\mathbb P[y = 1]\). Users labelled four of each, so she takes \(1/2\). Then she compares the two producers with **Bayes’ rule**, the identity that turns “how likely is this message under each kind” into “how likely is each kind given this message”:

\[ \mathbb P[y = 1 \mid x] = \frac{\mathbb P[y = 1]\, \mathbb P[x \mid y = 1]}{\mathbb P[y = 1]\, \mathbb P[x \mid y = 1] + \mathbb P[y = 0]\, \mathbb P[x \mid y = 0]}. \]

- \(\mathbb P[y = 1]\) is the share of junk in all mail before you read a word, and \(\mathbb P[y = 0] = 1 - \mathbb P[y = 1]\) the share of keep.
- \(\mathbb P[y = 1]\,\mathbb P[x \mid y = 1]\) is the probability that a random message is junk *and* reads exactly like \(x\): first pick the producer, then the producer writes \(x\). The second product in the denominator is the same for keep.
- The denominator adds the two ways \(x\) could have been produced, so it is the total probability of seeing \(x\) at all.
- \(\mathbb P[y = 1 \mid x]\), on the left, is the probability that the message is junk, given that it reads exactly like \(x\).

In one sentence: **the chance that this message is junk is the junk producer’s share of all the ways this message could have come about.** For “free, click”, the junk side is \(\tfrac12 \cdot 0.1406 = 0.0703\), the keep side works out to \(0.0015\), and the filter says junk with probability \(0.0703/(0.0703 + 0.0015) \approx 0.98\). That is a sensible answer.

Now try it on two of the messages from the bug reports: “free, click, meeting”, and “winner, click”. Before you run it, predict: what will each side be?

```python
import math
W = ["free", "prize", "click", "invoice",
     "meeting", "lunch", "winner"]
junk = [{"free", "prize", "click"}, {"free", "click"},
        {"prize", "click"}, {"free", "prize"}]
keep = [{"meeting", "lunch"}, {"invoice", "meeting"},
        {"lunch", "click"}, {"free", "lunch", "meeting"}]
def freq(msgs):
    return {w: sum(w in x for x in msgs) / len(msgs) for w in W}
def P(x, th):
    return math.prod(th[w] if w in x else 1 - th[w] for w in W)
th1, th0 = freq(junk), freq(keep)
for x in ({"free", "click"}, {"free", "click", "meeting"},
          {"winner", "click"}):
    p1, p0 = 0.5 * P(x, th1), 0.5 * P(x, th0)
    try:
        print(round(p1, 4), round(p0, 4), round(p1 / (p1 + p0), 2))
    except ZeroDivisionError as err:
        print(round(p1, 4), round(p0, 4), "crash:", err)
```

It prints:

```
0.0703 0.0015 0.98
0.0 0.0013 0.0
0.0 0.0 crash: float division by zero
```

There is the bug, reproduced by hand. “free, click, meeting” has two strong junk words, but “meeting” never appeared in the four junk messages, so \(\hat\theta_{\text{meeting},1} = 0\), the junk producer’s product contains a factor 0, and the filter concludes the message is junk with probability exactly 0. And “winner” never appeared anywhere, so *both* producers give the message probability 0, and Bayes’ rule divides zero by zero.

Notice what went wrong, because it is subtle. The arithmetic is fine. The fractions are honest counts. What failed is the *step from counts to probabilities*: four junk messages without “meeting” are weak evidence that junk is unlikely to say “meeting”, and Ines’s estimate turned them into a proof that junk can never say it. A single factor of 0 then overrides any amount of evidence from every other word.

Explore it in a bigger mail room. The lab below has a dictionary of 200 words, each with its own true frequency in junk and in keep, many of them rare. You choose how many labelled messages of each kind the filter sees, and watch how often a fresh message makes Ines’s filter certain or makes it crash.

[[lab]]

With 100 labelled messages of each kind, about a quarter of fresh messages crash the filter and almost half more get an answer of exactly 0 or exactly 1. Rare words are everywhere in real mail, and every one of them is a trap. To fix this properly, we need to understand what Ines was actually computing. It has a name.
