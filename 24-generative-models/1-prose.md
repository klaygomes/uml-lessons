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
        post = p1 / (p1 + p0)
        print(round(p1, 4), round(p0, 4), round(post, 2))
    except ZeroDivisionError as err:
        print(round(p1, 4), round(p0, 4), "crash:", err)
```

It prints:

```
0.0703 0.0015 0.98
0.0 0.0044 0.0
0.0 0.0 crash: division by zero
```

There is the bug, reproduced by hand. “free, click, meeting” has two strong junk words, but “meeting” never appeared in the four junk messages, so \(\hat\theta_{\text{meeting},1} = 0\), the junk producer’s product contains a factor 0, and the filter concludes the message is junk with probability exactly 0. And “winner” never appeared anywhere, so *both* producers give the message probability 0, and Bayes’ rule divides zero by zero.

Notice what went wrong, because it is subtle. The arithmetic is fine. The fractions are honest counts. What failed is the *step from counts to probabilities*: four junk messages without “meeting” are weak evidence that junk is unlikely to say “meeting”, and Ines’s estimate turned them into a proof that junk can never say it. A single factor of 0 then overrides any amount of evidence from every other word.

Explore it in a bigger mail room. The lab below has a dictionary of 200 words, each with its own true frequency in junk and in keep, many of them rare. You choose how many labelled messages of each kind the filter sees, and watch how often a fresh message makes Ines’s filter certain or makes it crash.

[[lab]]

With 100 labelled messages of each kind, about a quarter of fresh messages crash the filter and almost half more get an answer of exactly 0 or exactly 1. Rare words are everywhere in real mail, and every one of them is a trap. To fix this properly, we need to understand what Ines was actually computing. It has a name.

## Which parameter makes the data most likely?

Forget the filter for a moment and look at one word in one pile: “prize” in the four junk messages. Message 1 contains it, message 2 does not, messages 3 and 4 do. Write that as the pattern 1, 0, 1, 1.

Now play a game. Suppose the true probability that a junk message contains “prize” were some number \(\theta\). How likely would it be to see exactly this pattern? Each message is an independent coin, so the chance is \(\theta\) for message 1, times \(1 - \theta\) for message 2, times \(\theta\) twice more. Try a few values:

| \(\theta\) | chance of the pattern 1, 0, 1, 1 |
|---|---|
| 0.25 | 0.0117 |
| 0.50 | 0.0625 |
| 0.75 | 0.1055 |
| 0.90 | 0.0729 |

Some values of \(\theta\) make what you actually saw much more plausible than others. A world where “prize” turns up in a quarter of junk would rarely produce three hits out of four; a world where it turns up in three quarters produces them most often of the four we tried. The idea of this section is to choose the parameter that makes the observed data most likely. That is exactly what Ines did, without saying so: \(3/4\) is the winner. Let’s make it precise.

**The model.** We observe one yes/no value per example, \(x \in \{0, 1\}\), and the model says \(x = 1\) with probability \(\theta\), for an unknown \(\theta\) between 0 and 1. Its probability of each value is

\[ P_\theta[x] = \theta^{x} (1 - \theta)^{1 - x}. \]

- \(x\) is one observation: 1 if the word is in the message, 0 if not.
- \(\theta \in [0, 1]\) is the parameter, the probability of a 1. The symbol \(\in\) reads “is a member of”, and \([0, 1]\) is the set of numbers from 0 to 1, ends included.
- The subscript in \(P_\theta\) says which parameter the probability is computed under: the same \(x\) gets different probabilities under different \(\theta\).
- The exponent trick is the one from Ines’s recipe: for \(x = 1\) the right side is \(\theta\), for \(x = 0\) it is \(1 - \theta\).

In one sentence: **under parameter \(\theta\), a 1 has probability \(\theta\) and a 0 has probability \(1 - \theta\).** This distribution is called the **Bernoulli distribution** with parameter \(\theta\). With \(\theta = 0.75\), \(P_\theta[1] = 0.75\) and \(P_\theta[0] = 0.25\).

**The likelihood.** The training sequence is now just the observations, \(S = (x_1, \dots, x_m)\), drawn independently. A reminder from lesson 1: \(m\) is the number of examples, and the examples are i.i.d., each drawn from the same distribution with no draw influencing another. Multiply the per-example probabilities, and collect the factors. If \(k = \sum_{i=1}^m x_i\) is the number of 1s, there are \(k\) factors \(\theta\) and \(m - k\) factors \(1 - \theta\):

\[ P_\theta[S] = \prod_{i=1}^{m} \theta^{x_i} (1 - \theta)^{1 - x_i} = \theta^{k} (1 - \theta)^{m - k}. \]

- \(\prod_{i=1}^m\) multiplies one factor per example, \(i\) from 1 to \(m\). Independence of the examples is what allows the product.
- \(k = \sum_i x_i\) adds up the observations, so it counts the 1s; \(m - k\) counts the 0s.
- The right side just groups equal factors: the order of a product does not matter.

In one sentence: **the chance of the whole training sequence under \(\theta\) is \(\theta\) once per 1 times \(1 - \theta\) once per 0.** Seen as a function of \(\theta\) with the data held fixed, \(P_\theta[S]\) is called the **likelihood** of \(\theta\). For the “prize” pattern, \(m = 4\), \(k = 3\), and at \(\theta = 0.75\) the likelihood is \(0.75^3 \cdot 0.25 = 0.1055\), the third row of the table.

Products of many probabilities are tiny numbers, and computers dislike tiny numbers: a product of a thousand factors of about 0.5 is smaller than the smallest positive floating-point number, so it rounds to 0. The fix is to take logarithms, which turn products into sums. Throughout this lesson \(\log\) is the natural logarithm, base \(e\). Because \(\log\) is increasing (a bigger input always gives a bigger output), whichever \(\theta\) maximizes the likelihood also maximizes its logarithm. The **log-likelihood** of \(S\) given \(\theta\) is

\[ L(S; \theta) = \log P_\theta[S] = k \log \theta + (m - k) \log(1 - \theta). \]

- \(L(S; \theta)\) is a number that depends on the data \(S\) and on the parameter \(\theta\); the semicolon separates the data from the parameter. (This \(L\) has no subscript, unlike the risks \(L_S\) and \(L_{\mathcal D}\); the context keeps them apart.)
- \(\log\) of a product is the sum of the logs, and \(\log a^n = n \log a\), so \(\theta^k\) becomes \(k \log \theta\) and \((1-\theta)^{m-k}\) becomes \((m-k)\log(1-\theta)\).
- Both logs are of numbers at most 1, so both terms are at most 0: a log-likelihood is never positive for yes/no data.

In one sentence: **the log-likelihood adds \(\log \theta\) for each 1 and \(\log(1 - \theta)\) for each 0.** For “prize”, \(L(S; 0.75) = 3 \log 0.75 + \log 0.25 = -0.863 - 1.386 = -2.249\).

**Definition (maximum likelihood estimator).** The **maximum likelihood estimator** is a parameter that maximizes the log-likelihood,

\[ \hat\theta \in \operatorname*{argmax}_{\theta} L(S; \theta). \]

- \(\operatorname*{argmax}_\theta\) of a function is the set of inputs \(\theta\) at which the function reaches its largest value (“arg” for argument: not the largest value itself, but where it happens).
- The \(\in\) says \(\hat\theta\) is one of them: if several parameters tie, any of them counts.

In one sentence: **the estimate is whichever parameter makes the training sequence most likely.** We abbreviate it MLE.

```python
import math
S = [1, 0, 1, 1]
def P(S, th):
    return math.prod(th**x * (1 - th)**(1 - x) for x in S)
def L(S, th):
    k, m = sum(S), len(S)
    return k * math.log(th) + (m - k) * math.log(1 - th)
for th in (0.25, 0.5, 0.75, 0.9):
    print(th, round(P(S, th), 4), round(L(S, th), 3))
grid = [i / 1000 for i in range(1, 1000)]
print(max(grid, key=lambda th: L(S, th)))
```

It prints the table above with the log-likelihoods beside it (\(-4.447\), \(-2.773\), \(-2.249\), \(-2.619\)), and then `0.75`: a search over a thousand values of \(\theta\) lands on Ines’s fraction. That is no accident.

[[diagram:likelihood]]

**Proposition (the Bernoulli MLE is the average).** For \(S = (x_1, \dots, x_m)\) with \(k\) ones, the maximum likelihood estimator is \(\hat\theta = k/m = \frac{1}{m}\sum_{i=1}^m x_i\).

*Proof.* First the ordinary case, \(0 < k < m\). Then \(L(S; \theta)\) is finite for \(0 < \theta < 1\) and falls to \(-\infty\) at both ends, because \(\log \theta \to -\infty\) as \(\theta \to 0\) and \(\log(1-\theta) \to -\infty\) as \(\theta \to 1\), each multiplied by a positive count. So the maximum is inside the interval, where the derivative is zero. The derivative of \(\log \theta\) is \(1/\theta\), and of \(\log(1 - \theta)\) it is \(-1/(1 - \theta)\), so

\[ \frac{d}{d\theta} L(S; \theta) = \frac{k}{\theta} - \frac{m - k}{1 - \theta}. \]

Here \(\frac{d}{d\theta}\) is the derivative with respect to \(\theta\): the slope of the log-likelihood as \(\theta\) changes. **The slope is the pull of the ones toward bigger \(\theta\) minus the pull of the zeros toward smaller \(\theta\).** At \(\theta = 0.5\) with \(k = 3\), \(m = 4\), it is \(6 - 2 = 4\): still climbing.

Setting the slope to 0 and multiplying both sides by \(\theta(1 - \theta)\) gives \(k(1 - \theta) = (m - k)\theta\), that is \(k - k\theta = m\theta - k\theta\), so \(\theta = k/m\). This is the only point with zero slope. It is a maximum, not a minimum: the second derivative, \(-k/\theta^2 - (m - k)/(1 - \theta)^2\), is negative everywhere, so the slope only ever decreases, from positive before \(k/m\) to negative after it.

The edge cases. If \(k = 0\), the log-likelihood is \(m \log(1 - \theta)\), which is largest at \(\theta = 0\), where it equals 0 (the term \(k \log \theta\) is absent because \(k = 0\); by the usual convention \(0 \cdot \log 0 = 0\)). So \(\hat\theta = 0 = k/m\). If \(k = m\), symmetrically \(\hat\theta = 1 = k/m\). \(\blacksquare\)

Read that edge case again, because it is our bug. When a word never appears among \(m\) junk messages, the most likely world, given only those messages, is one where junk *never* contains it. Maximum likelihood is not making an arithmetic slip: it is answering the question it was asked.

**How accurate is it?** The estimate \(\hat\theta\) is an average of \(m\) independent coins, each equal to 1 with probability \(\theta\). An estimator is **unbiased** when its expected value, averaged over the random training sequence, equals the parameter. This one is: by linearity of expectation, \(\mathbb E_S[\hat\theta] = \frac1m \sum_i \mathbb E[x_i] = \frac1m \cdot m\theta = \theta\). (A reminder: \(\mathbb E\) is the expected value, the probability-weighted average, and the subscript \(S\) says the randomness is in the training sequence.)

Unbiased says the estimate is right on average; it does not say how far one estimate strays. For that, recall Hoeffding’s inequality from lesson 4: for an average \(\bar Z\) of \(m\) i.i.d. variables with values in \([0, 1]\) and mean \(\mu\), \(\mathbb P[|\bar Z - \mu| > \varepsilon] \le 2e^{-2m\varepsilon^2}\). Set the right side equal to a confidence parameter \(\delta\) and solve for \(\varepsilon\): \(2e^{-2m\varepsilon^2} = \delta\) gives \(\varepsilon = \sqrt{\log(2/\delta)/(2m)}\). So with probability at least \(1 - \delta\) over the choice of \(S\),

\[ |\hat\theta - \theta| \le \sqrt{\frac{\log(2/\delta)}{2m}}. \]

- \(|\hat\theta - \theta|\) is the distance between the estimate and the truth; the vertical bars around a number mean its absolute value.
- \(\delta\) is the confidence parameter of lesson 3: the probability we allow for an unlucky training sequence.
- Inside the root, \(\log(2/\delta)\) grows only slowly as you demand more confidence, and the \(2m\) in the denominator shrinks the bound as you add examples.

In one sentence: **with probability at least \(1 - \delta\), the counted fraction is within \(\sqrt{\log(2/\delta)/(2m)}\) of the true one.** With \(m = 500\) messages and \(\delta = 0.05\), that is \(\sqrt{\log 40/1000} \approx 0.061\).

```python
import math, random
m, delta, theta = 500, 0.05, 0.3
eps = math.sqrt(math.log(2 / delta) / (2 * m))
rng, runs, misses = random.Random(0), 2000, 0
for _ in range(runs):
    k = sum(rng.random() < theta for _ in range(m))
    misses += abs(k / m - theta) > eps
print(round(eps, 3), misses / runs)
```

It prints `0.061 0.0035`: the bound promised at most 5% misses, and the simulation saw 0.35%. Hoeffding is honest, and a little pessimistic.

So for estimating *the frequency of a word*, maximum likelihood is fine: close to the truth with high probability. Keep that in mind. The trouble in Ines’s filter is not that \(\hat\theta\) is far from \(\theta\). It is what happens when a tiny error near 0 gets fed into a logarithm, and we are about to see exactly how.

## Measured values: densities

Not every feature is a yes/no word. The mail room also records numbers: the length of a message, the time it was sent. Take the **length score** of a message, the natural logarithm of its number of characters. It is a real number, and for a real number something odd happens: the probability that a score equals 3.1 exactly, to infinitely many decimals, is 0. So “the probability of the data” is 0 for every parameter, and the likelihood we just defined is useless.

The fix is to use the **density** instead. A density \(p\) is a nonnegative function whose area over any interval is the probability of landing in that interval; near a point \(x\), the probability of landing in a tiny interval of width \(w\) is about \(p(x) \cdot w\). So a density compares how likely different points are, just as probabilities do for yes/no data. For an i.i.d. sample \(S = (x_1, \dots, x_m)\) from a density \(P_\theta\), define the log-likelihood as the sum of the logs of the density values:

\[ L(S; \theta) = \log \prod_{i=1}^{m} P_\theta(x_i) = \sum_{i=1}^{m} \log P_\theta(x_i). \]

- \(P_\theta(x_i)\) is the density of the model with parameter \(\theta\) at the observed value \(x_i\).
- The product is the joint density of an independent sample, and the log turns it into a sum.

In one sentence: **the log-likelihood of measured data adds up the log of the model’s density at each observed value.** The maximum likelihood estimator is again a maximizer of \(L(S; \theta)\). To keep notation light, from here on we write \(\mathbb P[X = x]\), or \(P_\theta[x]\), for both the probability of \(x\) (yes/no data) and the density at \(x\) (measured data).

**The Gaussian model.** Lengths of keep messages cluster around a typical value, with fewer messages the further you go. The usual model for such a bell-shaped spread is the **Gaussian** (or normal) distribution with parameters \(\theta = (\mu, \sigma)\), whose density is

\[ P_\theta(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right). \]

- \(\mu\) (mu) is the centre of the bell, the mean; \(\sigma\) (sigma) is its width, the standard deviation, and \(\sigma^2\) the variance.
- \((x - \mu)^2\) is the squared distance from the centre; dividing by \(2\sigma^2\) measures that distance in units of the width.
- \(\exp(u) = e^u\). With a minus sign in front, the density is largest at \(x = \mu\) and falls off quickly with distance.
- \(\frac{1}{\sigma\sqrt{2\pi}}\) is the constant that makes the total area under the curve equal to 1.

In one sentence: **the Gaussian density is a bell centred at \(\mu\) with width \(\sigma\), falling off with the squared distance from the centre.** At the centre with \(\sigma = 1\), the density is \(1/\sqrt{2\pi} \approx 0.399\).

Take the log of the density, add over the sample, and the log-likelihood is

\[ L(S; \mu, \sigma) = -\frac{1}{2\sigma^2}\sum_{i=1}^{m} (x_i - \mu)^2 - m\log\left(\sigma\sqrt{2\pi}\right). \]

- \(\log\) undoes \(\exp\), so each example contributes \(-(x_i - \mu)^2/(2\sigma^2)\), plus \(\log\) of the constant, which is \(-\log(\sigma\sqrt{2\pi})\).
- The \(m\) copies of that constant add up to the last term.
- The first term is a penalty for points far from \(\mu\); the second is a penalty for a wide bell.

In one sentence: **a Gaussian explains the data well when the points are close to its centre and its bell is no wider than it needs to be.** Five keep messages with length scores 3.1, 2.4, 4.0, 3.5 and 2.0 give, for \(\mu = 3\) and \(\sigma = 1\), the log-likelihood \(-\frac12(0.01 + 0.36 + 1 + 0.25 + 1) - 5\log\sqrt{2\pi} = -1.31 - 4.595 = -5.905\).

**Proposition (the Gaussian MLE).** If not all \(x_i\) are equal, the maximum likelihood estimates are the sample mean and the root of the average squared deviation:

\[ \hat\mu = \frac1m \sum_{i=1}^{m} x_i, \qquad \hat\sigma = \sqrt{\frac1m \sum_{i=1}^{m} (x_i - \hat\mu)^2}. \]

- \(\hat\mu\) is the plain average of the observations.
- \(\hat\sigma^2\) is the average of the squared distances from that average; \(\hat\sigma\) is its square root.

In one sentence: **fit the bell’s centre at the average and its width at the typical distance from the average.** For the five scores, \(\hat\mu = 15/5 = 3.0\) and \(\hat\sigma^2 = 2.62/5 = 0.524\), so \(\hat\sigma \approx 0.724\).

*Proof.* Two steps: first the centre for any fixed width, then the width.

Fix \(\sigma\). Only the first term depends on \(\mu\), and it is largest when \(\sum_i (x_i - \mu)^2\) is smallest. Write \(x_i - \mu = (x_i - \hat\mu) + (\hat\mu - \mu)\) and square:

\[ \sum_{i=1}^{m} (x_i - \mu)^2 = \sum_{i=1}^{m} (x_i - \hat\mu)^2 + 2(\hat\mu - \mu)\sum_{i=1}^{m}(x_i - \hat\mu) + m(\hat\mu - \mu)^2. \]

The middle sum is \(\sum_i x_i - m\hat\mu = 0\), by the definition of \(\hat\mu\). So **the total squared distance to any centre \(\mu\) is the total squared distance to the average plus \(m\) times the squared gap between \(\mu\) and the average.** It is smallest exactly at \(\mu = \hat\mu\), whatever \(\sigma\) is. (With the five scores and \(\mu = 3.5\): \(2.62 + 5 \cdot 0.25 = 3.87\).)

Now put \(\mu = \hat\mu\) and write \(A = \sum_i (x_i - \hat\mu)^2\), which is positive because not all \(x_i\) are equal. The log-likelihood as a function of \(\sigma > 0\) is \(g(\sigma) = -A/(2\sigma^2) - m\log\sigma - m\log\sqrt{2\pi}\), with derivative

\[ g'(\sigma) = \frac{A}{\sigma^3} - \frac{m}{\sigma} = \frac{A - m\sigma^2}{\sigma^3}. \]

**The slope is positive while \(\sigma^2\) is below \(A/m\) and negative after,** so \(g\) climbs and then falls, and its maximum is at \(\sigma^2 = A/m\), which is \(\hat\sigma^2\). \(\blacksquare\)

(Why the condition “not all equal”? If every \(x_i\) is the same, \(A = 0\) and \(g(\sigma) = -m\log\sigma - \text{const}\) grows without limit as \(\sigma \to 0\): an infinitely thin bell on a single point. There is no maximum. Hold on to this; it is the continuous cousin of our zero-probability bug.)

```python
import math
xs = [3.1, 2.4, 4.0, 3.5, 2.0]
def L(mu, sigma):
    sq = sum((x - mu)**2 for x in xs)
    c = len(xs) * math.log(sigma * math.sqrt(2 * math.pi))
    return -sq / (2 * sigma**2) - c
mu_hat = sum(xs) / len(xs)
s_hat = math.sqrt(sum((x - mu_hat)**2 for x in xs) / len(xs))
print(round(L(3, 1), 3), mu_hat, round(s_hat, 3))
best = max(((L(mu_hat + a / 100, s_hat + b / 100), a, b)
            for a in range(-20, 21) for b in range(-20, 21)))
print(round(best[0], 3), best[1:])
```

It prints `-5.905 3.0 0.724`, then `-5.479 (0, 0)`: around the formulas, nudging the centre or the width in any direction only lowers the log-likelihood.

One more property you will prove yourself at the end of the lesson. The centre estimate \(\hat\mu\) is unbiased. The width estimate is not: on average, \(\hat\sigma^2\) comes out a little too small. Maximum likelihood estimates are not always unbiased.

## Maximum likelihood is ERM in disguise

Here is the surprise this lesson promised. Go back to lesson 3’s general setting: a loss function \(\ell(h, z)\) says how badly hypothesis \(h\) does on example \(z\), the empirical risk is the average loss on the training sequence, and **ERM**, empirical risk minimization, returns the hypothesis with the smallest empirical risk.

Now let the “hypothesis” be a parameter \(\theta\) and the example be an observation \(x\). How badly does \(\theta\) do on \(x\)? A natural answer is: by how *surprised* the model is to see \(x\). If \(P_\theta[x]\) is close to 1, the model expected it, and the loss should be near 0. If \(P_\theta[x]\) is tiny, the model was badly wrong, and the loss should be large. The negative logarithm does exactly this. Under \(\theta = 0.75\) for “prize”, a message with “prize” costs \(-\log 0.75 = 0.288\), and a message without it costs \(-\log 0.25 = 1.386\).

**Definition (log loss).** The **log loss** of parameter \(\theta\) on observation \(x\) is

\[ \ell(\theta, x) = -\log P_\theta[x]. \]

- \(\ell\) is the loss function of lesson 3, now with a parameter in the hypothesis slot and an observation (no label) in the example slot.
- \(P_\theta[x]\) is the probability (or density) the model gives to what actually happened.
- The minus sign turns the log, which is negative for probabilities below 1, into a nonnegative cost.

In one sentence: **the log loss is how surprised the model was by what it saw.** A probability of 1 costs 0; a probability of 0 costs \(-\log 0 = +\infty\). Because we use the natural logarithm, log losses are measured in units called **nats**; with base-2 logarithms they would be bits, as in lesson 18.

Since the empirical risk with this loss is \(\frac1m\sum_i(-\log P_\theta[x_i]) = -\frac1m L(S;\theta)\), minimizing one is maximizing the other:

\[ \operatorname*{argmin}_{\theta} \sum_{i=1}^{m} \left(-\log P_\theta[x_i]\right) = \operatorname*{argmax}_{\theta} \sum_{i=1}^{m} \log P_\theta[x_i]. \]

- \(\operatorname*{argmin}\) is the set of parameters where the left sum is smallest, just as \(\operatorname*{argmax}\) is where the right sum is largest.
- The left sum is \(m\) times the empirical risk with the log loss; the right sum is the log-likelihood.
- A sum and its negative are smallest and largest at the same places.

In one sentence: **the maximum likelihood estimator is exactly ERM with the log loss.** For “prize”, the empirical risk at \(\theta = 0.75\) is \(2.249/4 = 0.562\), and no other \(\theta\) does better. You met one instance of this in lesson 9: logistic regression is maximum likelihood for a model of the label given the message.

```python
import math
S = [1, 0, 1, 1]
def loss(th, x):
    return -math.log(th if x == 1 else 1 - th)
def risk_S(th):
    return sum(loss(th, x) for x in S) / len(S)
print(round(loss(0.75, 1), 3), round(loss(0.75, 0), 3))
grid = [i / 1000 for i in range(1, 1000)]
best = min(grid, key=risk_S)
print(best, round(risk_S(best), 3))
```

It prints `0.288 1.386`, then `0.75 0.562`.

**The true risk.** ERM is only as good as the true risk it is chasing, so let’s compute it. Suppose the observations really come from some distribution \(P\), which need not be of the form \(P_\theta\) at all. The true risk of \(\theta\) is its expected log loss on a fresh observation, and it splits into two pieces. First the idea, with numbers. Say the word truly appears with probability 0.3, and the model uses \(\theta = 0.5\). Then the expected surprise is \(0.3 \cdot (-\log 0.5) + 0.7 \cdot (-\log 0.5) = 0.693\). Even the perfect model, \(\theta = 0.3\), would be surprised on average by \(-0.3\log 0.3 - 0.7\log 0.7 = 0.611\): the data is random, and some surprise is unavoidable. The difference, \(0.082\), is the price of using the wrong parameter. In symbols:

\[ \mathbb E_{x \sim P}\left[\ell(\theta, x)\right] = \sum_x P[x]\log\frac{P[x]}{P_\theta[x]} + \sum_x P[x]\log\frac{1}{P[x]} = D_{\mathrm{RE}}(P \,\|\, P_\theta) + H(P). \]

- \(x \sim P\) reads “\(x\) is drawn from \(P\)”, and \(\mathbb E_{x \sim P}\) averages over that draw: each possible \(x\) weighted by its true probability \(P[x]\).
- The first equality adds and subtracts \(\log P[x]\) inside \(-\log P_\theta[x]\): \(-\log P_\theta[x] = \log\frac{P[x]}{P_\theta[x]} + \log\frac{1}{P[x]}\).
- \(H(P) = \sum_x P[x]\log\frac{1}{P[x]}\) is the **entropy** of \(P\): the average surprise of the true distribution about itself. It does not depend on \(\theta\). (Lesson 18 used the same quantity in base 2 to choose splits.)
- \(D_{\mathrm{RE}}(P \| P_\theta) = \sum_x P[x]\log\frac{P[x]}{P_\theta[x]}\) is the **relative entropy** of \(P\) to \(P_\theta\), also called the Kullback–Leibler divergence: the extra surprise from believing \(P_\theta\) when the truth is \(P\). The double bar \(\|\) only separates its two arguments, and the order matters.
- Terms with \(P[x] = 0\) are left out of both sums: an outcome that never happens costs nothing on average.

In one sentence: **the true risk of a parameter is the unavoidable randomness of the data plus the extra surprise caused by the model being wrong.** In the example, \(0.693 = 0.611 + 0.082\).

The entropy is the floor nobody can go below, as long as the relative entropy is never negative. It isn’t:

**Proposition (relative entropy is nonnegative).** For distributions \(P\) and \(Q\) on a finite set, \(D_{\mathrm{RE}}(P\|Q) \ge 0\), with equality exactly when \(Q = P\).

*Proof.* The engine is a one-line inequality: \(\log t \le t - 1\) for every \(t > 0\), with equality only at \(t = 1\). To see it, let \(g(t) = t - 1 - \log t\). Its derivative \(1 - 1/t\) is negative for \(t < 1\) and positive for \(t > 1\), so \(g\) falls to its minimum at \(t = 1\), where \(g(1) = 0\), and is positive everywhere else.

Now sum over the outcomes with \(P[x] > 0\). If \(Q[x] = 0\) for one of them, the term \(\log(P[x]/Q[x])\) is \(+\infty\), and there is nothing to prove. Otherwise apply the inequality with \(t = Q[x]/P[x]\):

\[ -D_{\mathrm{RE}}(P\|Q) = \sum_{x : P[x] > 0} P[x]\log\frac{Q[x]}{P[x]} \le \sum_{x : P[x] > 0} P[x]\left(\frac{Q[x]}{P[x]} - 1\right) = \sum_{x : P[x] > 0} Q[x] - 1 \le 0. \]

- The colon in \(\sum_{x : P[x] > 0}\) reads “such that”: add over the outcomes \(x\) such that \(P[x] > 0\).
- The middle step multiplies out: \(P[x] \cdot Q[x]/P[x] = Q[x]\), and the \(P[x]\) add up to 1.
- The last step holds because \(Q\)’s total over some outcomes is at most its total over all, which is 1.

In one sentence: **minus the relative entropy is at most \(Q\)’s total mass on the outcomes \(P\) can produce, minus 1, which is never positive.** Equality needs both inequalities to be equalities: \(Q[x]/P[x] = 1\) wherever \(P[x] > 0\), and \(Q\) putting no mass elsewhere. That is \(Q = P\). \(\blacksquare\)

```python
import math
P = {1: 0.3, 0: 0.7}
def Q(th):
    return {1: th, 0: 1 - th}
def risk(th):
    return sum(-P[x] * math.log(Q(th)[x]) for x in P)
H = sum(P[x] * math.log(1 / P[x]) for x in P)
def D(th):
    return sum(P[x] * math.log(P[x] / Q(th)[x]) for x in P)
for th in (0.1, 0.3, 0.5):
    print(th, round(risk(th), 3), round(H, 3), round(D(th), 3))
```

It prints `0.1 0.765 0.611 0.154`, `0.3 0.611 0.611 0.0` and `0.5 0.693 0.611 0.082`: in every row the risk is the entropy plus the relative entropy, and the relative entropy vanishes only at the truth.

So the true risk is smallest when \(P_\theta = P\). If the data really has the assumed parametric form, the right parameter brings the risk down to the entropy. If it does not, even the best parameter pays a relative entropy forever, no matter how much data you collect. That is the inductive bias of a generative model, in the sense of lesson 1: the parametric form is a hypothesis class, chosen before seeing any data, and it can miss the truth.

[[diagram:logloss]]

## How well does maximum likelihood generalize?

Lesson 2 asked of every learner: trained on a finite training sequence, how good is its true risk? For a generative model the natural yardstick is the one we just built, the expected log loss. Since the entropy is a floor nobody can beat, what matters is the **excess risk**: the true risk of the estimate minus the true risk of the best parameter. Let’s compute it for two models, one where maximum likelihood behaves beautifully and one where it fails outright.

**A well-behaved case: the centre of a bell.** Suppose the length scores of keep messages really are Gaussian with width \(\sigma = 1\) and an unknown centre \(\mu^\star\) (the star marks the true value). We estimate the centre by \(\hat\mu\), the average. If the estimate is off by 0.2, how much extra surprise does that cost on a fresh message? Intuition says something like the square of the miss, and the exact answer is \(\tfrac12 \cdot 0.2^2 = 0.02\). Here is the calculation, with \(\ell(\mu, x) = -\log P_\mu(x)\) the log loss of the unit-width Gaussian:

\[ \begin{aligned} \mathbb E_{x \sim N(\mu^\star, 1)}\left[\ell(\hat\mu, x) - \ell(\mu^\star, x)\right] &= \mathbb E\left[-\tfrac12 (x - \mu^\star)^2 + \tfrac12 (x - \hat\mu)^2\right] \\ &= \mathbb E\left[x(\mu^\star - \hat\mu) + \tfrac12\hat\mu^2 - \tfrac12(\mu^\star)^2\right] \\ &= \mu^\star(\mu^\star - \hat\mu) + \tfrac12\hat\mu^2 - \tfrac12(\mu^\star)^2 = \tfrac12(\hat\mu - \mu^\star)^2. \end{aligned} \]

- \(N(\mu^\star, 1)\) is the Gaussian with centre \(\mu^\star\) and width 1, and \(x \sim N(\mu^\star, 1)\) is a fresh score drawn from it. The estimate \(\hat\mu\) is held fixed: it was computed from the training sequence, and the fresh \(x\) is independent of it.
- First line: \(\ell(\hat\mu, x) - \ell(\mu^\star, x) = \log P_{\mu^\star}(x) - \log P_{\hat\mu}(x)\). Both densities have the same constant \(\frac{1}{\sqrt{2\pi}}\), which cancels, leaving the two exponents.
- Second line: expand both squares. The \(x^2\) terms cancel, \(-\tfrac12(-2x\mu^\star) + \tfrac12(-2x\hat\mu) = x(\mu^\star - \hat\mu)\), and the constants are \(\tfrac12\hat\mu^2 - \tfrac12(\mu^\star)^2\).
- Third line: the only random thing left is \(x\), and \(\mathbb E[x] = \mu^\star\). Collect: \(\tfrac12\hat\mu^2 - \hat\mu\mu^\star + \tfrac12(\mu^\star)^2\) is \(\tfrac12(\hat\mu - \mu^\star)^2\).

In one sentence: **using a centre that is off by some amount costs, on every fresh message on average, half the square of that amount.** And how far off is \(\hat\mu\)? It is an average of \(m\) independent scores of variance 1, and lesson 4 showed that such an average has variance \(1/m\). Since \(\hat\mu\) is unbiased, its variance *is* \(\mathbb E[(\hat\mu - \mu^\star)^2]\), so **the expected excess risk is exactly \(1/(2m)\).** With Chebyshev’s inequality from lesson 4, \(\mathbb P[(\hat\mu - \mu^\star)^2 \ge 1/(m\delta)] \le \delta\), so with probability at least \(1 - \delta\) the excess risk is below \(1/(2m\delta)\). Ten messages give an expected excess of \(0.05\).

```python
import random
rng, mu_star, m, runs = random.Random(1), 3.0, 10, 20000
total = 0.0
for _ in range(runs):
    mu_hat = sum(rng.gauss(mu_star, 1) for _ in range(m)) / m
    total += 0.5 * (mu_hat - mu_star)**2
print(round(total / runs, 3), 1 / (2 * m))
```

It prints `0.051 0.05`: twenty thousand simulated training sequences land on \(1/(2m)\).

**A failing case: a rare word.** Now the yes/no model, for a word that junk messages contain with a small probability \(\theta^\star\). Say “winner” is in one junk message out of 500, so \(\theta^\star = 0.002\), and you have \(m = 200\) junk messages. What is the chance that “winner” appears in none of them? Each message misses it with probability 0.998, independently, so the chance is \(0.998^{200} \approx 0.67\). Two times out of three, maximum likelihood sets \(\hat\theta = 0\).

And then the true risk is infinite. The next junk message contains “winner” with probability \(\theta^\star\), and when it does, the log loss is \(-\log \hat\theta = -\log 0 = +\infty\):

\[ \mathbb E_{x \sim \theta^\star}\left[\ell(\hat\theta, x)\right] = \theta^\star \log\frac{1}{\hat\theta} + (1 - \theta^\star)\log\frac{1}{1 - \hat\theta} = \theta^\star \log\frac10 = \infty. \]

- \(x \sim \theta^\star\) is shorthand for “\(x\) is drawn from the Bernoulli distribution with parameter \(\theta^\star\)”.
- The two terms are the two possible observations, each weighted by its true probability: a 1 with probability \(\theta^\star\), costing \(\log(1/\hat\theta)\), and a 0 with probability \(1 - \theta^\star\), costing \(\log(1/(1 - \hat\theta))\).
- With \(\hat\theta = 0\), the second term is \(\log 1 = 0\), and the first is a positive number times \(\log(1/0) = +\infty\).

In one sentence: **an estimate of exactly 0 for something that does happen makes the expected surprise infinite.** It does not matter that \(\hat\theta = 0\) is within 0.002 of the truth, a superb estimate by Hoeffding’s standards. The log loss magnifies errors near 0 without limit.

How often does this happen? The chance that all \(m\) observations are 0 is \((1 - \theta^\star)^m\). A clean lower bound: for \(0 \le \theta \le 1/2\),

\[ (1 - \theta^\star)^m \ge e^{-2\theta^\star m}. \]

- \(e^{-2\theta^\star m}\) is easier to reason about than the power: it says the chance shrinks exponentially in the number of examples, at rate \(2\theta^\star\).

In one sentence: **the chance of never seeing a rare word is at least \(e\) to the minus twice the expected number of times you would see it.** To prove it, take logs: we need \(m\log(1 - \theta) \ge -2\theta m\), so it is enough that \(h(\theta) = \log(1 - \theta) + 2\theta \ge 0\). At \(\theta = 0\), \(h(0) = 0\), and its derivative \(-1/(1 - \theta) + 2\) is nonnegative while \(1 - \theta \ge 1/2\), so \(h\) only grows on \([0, 1/2]\).

Now ask when the bound is at least a half: \(e^{-2\theta^\star m} \ge 1/2\) exactly when \(m \le \log 2/(2\theta^\star)\). **Whenever the training sequence has at most \(\log 2/(2\theta^\star)\) examples, maximum likelihood has infinite true risk with probability at least one half.** For “winner”, that is any \(m\) up to 173.

```python
import math
theta, m = 0.002, 200
p_zero = (1 - theta)**m
print(round(p_zero, 3), round(math.exp(-2 * theta * m), 3))
print(math.floor(math.log(2) / (2 * theta)))
print(round((1 - theta)**173, 3))
```

It prints `0.67 0.449`, then `173` and `0.707`: at 173 examples the all-zero chance is still 71%.

This is Ines’s bug, now as a theorem. A dictionary has thousands of rare words. For each one, the counts say 0 with substantial probability, and each such 0 is a factor that can wipe out a whole producer’s probability. Maximum likelihood **overfits**, in exactly the sense of lesson 2: it fits the training sequence perfectly (it makes the observed data as likely as possible) while its true risk is as bad as it can be.

[[diagram:zero]]

## The fix: pseudo-counts

Lesson 13 cured overfitting with a regularizer, a penalty that pulls the learner away from extreme answers. Let’s do the same here, in the most down-to-earth way possible. Before counting, pretend you have already seen \(\alpha\) junk messages that contain the word and \(\alpha\) that do not. These imaginary messages are **pseudo-counts**. With \(\alpha = 1\), the estimate for “winner” in the four junk messages becomes \((0 + 1)/(4 + 2) = 1/6\) instead of 0, and the estimate for “free” becomes \((3 + 1)/6 = 2/3\) instead of \(3/4\). No word can ever get an estimate of 0 or 1 again.

**Definition (the add-\(\alpha\) estimator).** For \(\alpha \ge 0\) and a training sequence with \(k\) ones among \(m\) observations,

\[ \hat\theta_\alpha = \frac{k + \alpha}{m + 2\alpha}. \]

- \(k + \alpha\) is the number of ones, real and imaginary; \(m + 2\alpha\) is the number of observations, real and imaginary.
- \(\alpha = 0\) gives back maximum likelihood, \(k/m\). \(\alpha = 1\) is called **add-one smoothing**, or **Laplace’s rule**.
- Whatever the data, \(\hat\theta_\alpha\) lies between \(\alpha/(m + 2\alpha)\) (no ones at all) and \((m + \alpha)/(m + 2\alpha)\) (all ones).

In one sentence: **estimate the frequency as if you had also seen \(\alpha\) imaginary examples of each kind.** With \(m = 4\), \(k = 0\), \(\alpha = 1\): \(1/6 \approx 0.167\).

This is regularized maximum likelihood, not just a hack. Add to the empirical log loss a penalty that charges the parameter for being near 0 or near 1:

\[ \frac1m\sum_{i=1}^{m}\log\frac{1}{P_\theta[x_i]} + \frac{\alpha}{m}\left(\log\frac1\theta + \log\frac{1}{1-\theta}\right). \]

- The first part is the empirical risk with the log loss, as before.
- The penalty \(\log(1/\theta) + \log(1/(1-\theta))\) is exactly the log loss of \(\theta\) on two imaginary observations, one 1 and one 0; it is huge near either end.
- The weight \(\alpha/m\) says how much the penalty counts, relative to one real example.

In one sentence: **the regularized objective is the average log loss on the real examples plus \(\alpha\) imaginary ones and \(\alpha\) imaginary zeros.** Multiply it by \(m\): you get \(-(k + \alpha)\log\theta - (m - k + \alpha)\log(1 - \theta)\), the negative log-likelihood of a sequence with \(k + \alpha\) ones and \(m - k + \alpha\) zeros. For \(\alpha > 0\) both counts are positive, and the proof of the Bernoulli MLE (which used only that the counts are positive numbers, never that they are whole) puts the unique minimizer at \((k + \alpha)/(m + 2\alpha)\). So the add-\(\alpha\) estimator is ERM with a regularizer, the pattern of lesson 13.

```python
import math
k, m, alpha = 0, 4, 1
def objective(th):
    real = -(k * math.log(th) + (m - k) * math.log(1 - th))
    fake = alpha * (math.log(1 / th) + math.log(1 / (1 - th)))
    return (real + fake) / m
grid = [i / 1000 for i in range(1, 1000)]
theta_alpha = (k + alpha) / (m + 2 * alpha)
print(min(grid, key=objective), round(theta_alpha, 3))
```

It prints `0.167 0.167`.

What does this buy? Here is the guarantee, the heart of the fix. It has three parts: the estimate stays accurate, the expected excess risk becomes finite, and without pseudo-counts it cannot be.

**Theorem (the pseudo-count guarantee).** Let \(x_1, \dots, x_m\) be i.i.d. Bernoulli with parameter \(\theta^\star \in [0, 1]\), and let \(\hat\theta = \hat\theta_\alpha\). Write the excess risk of a parameter \(q\) as \(\Delta(q) = \mathbb E_{x \sim \theta^\star}[\ell(q, x)] - \mathbb E_{x \sim \theta^\star}[\ell(\theta^\star, x)]\).

(a) For every \(\alpha \ge 0\) and \(\delta \in (0, 1)\), with probability at least \(1 - \delta\),

\[ |\hat\theta - \theta^\star| \le \sqrt{\frac{\log(2/\delta)}{2m}} + \frac{\alpha}{m + 2\alpha}. \]

(b) For every \(\alpha > 0\),

\[ \mathbb E_S\left[\Delta(\hat\theta)\right] \le \frac{m\theta^\star(1-\theta^\star) + \alpha^2}{\alpha(m + \alpha)} \le \frac{m/4 + \alpha^2}{\alpha(m + \alpha)}. \]

(c) For \(\alpha = 0\) and \(0 < \theta^\star < 1\), \(\mathbb P_S[\Delta(\hat\theta) = \infty] \ge (1 - \theta^\star)^m > 0\), so \(\mathbb E_S[\Delta(\hat\theta)] = \infty\).

Let’s read the three formulas before proving them.

- In (a), the first term is Hoeffding’s accuracy from before; the second, \(\alpha/(m + 2\alpha)\), is the most the imaginary examples can drag the estimate. **The estimate is within the usual Hoeffding distance of the truth, plus a small pull from the pseudo-counts that fades as \(m\) grows.** With \(m = 600\), \(\delta = 0.05\), \(\alpha = 1\): \(0.0554 + 0.0017 = 0.0571\).
- In (b), \(\mathbb E_S\) averages over the random training sequence, and \(\Delta(\hat\theta)\) is the extra expected log loss of the estimate on a fresh observation. The numerator grows with how random the word is (\(\theta^\star(1-\theta^\star)\) is largest at \(1/2\), where it equals \(1/4\), which gives the second bound); the denominator grows with \(\alpha\). **With pseudo-counts, the expected excess risk is finite for every word, however rare, and we can say how big it can be.** For “winner” (\(\theta^\star = 0.002\), \(m = 200\), \(\alpha = 1\)): \((0.399 + 1)/201 \approx 0.007\).
- In (c), \(\mathbb P_S\) is the probability over the training sequence. **Without pseudo-counts, every word that is neither impossible nor certain has infinite expected excess risk.**

*Proof of (a).* Write \(\bar x = k/m\) for the plain average, so \(\hat\theta = (m\bar x + \alpha)/(m + 2\alpha)\). The trick is to compare \(\hat\theta\) with its own expectation, then the expectation with the truth.

The expectation: \(\mathbb E[k] = m\theta^\star\), so \(\mathbb E[\hat\theta] = (m\theta^\star + \alpha)/(m + 2\alpha)\). Its distance from the truth, the **bias**, is

\[ \mathbb E[\hat\theta] - \theta^\star = \frac{m\theta^\star + \alpha - (m + 2\alpha)\theta^\star}{m + 2\alpha} = \frac{\alpha(1 - 2\theta^\star)}{m + 2\alpha}, \]

and since \(|1 - 2\theta^\star| \le 1\), its size is at most \(\alpha/(m + 2\alpha)\). **The pseudo-counts pull the average estimate toward \(1/2\), by at most \(\alpha/(m + 2\alpha)\).** For “winner” with \(m = 200\), \(\alpha = 1\): \(0.996/202 \approx 0.0049\).

The fluctuation: \(\hat\theta - \mathbb E[\hat\theta] = \frac{m}{m + 2\alpha}(\bar x - \theta^\star)\), and the factor \(m/(m + 2\alpha)\) is at most 1, so \(|\hat\theta - \mathbb E\hat\theta| \le |\bar x - \theta^\star|\), which Hoeffding bounds by \(\sqrt{\log(2/\delta)/(2m)}\) with probability at least \(1 - \delta\).

Put them together with the triangle inequality, \(|a + b| \le |a| + |b|\), applied to \(\hat\theta - \theta^\star = (\hat\theta - \mathbb E\hat\theta) + (\mathbb E\hat\theta - \theta^\star)\). \(\blacksquare\)

*Proof of (b).* Three steps: the excess risk is a relative entropy, the relative entropy is at most a squared distance divided by \(q(1-q)\), and the pseudo-counts keep \(q(1-q)\) away from 0.

Step 1. By the decomposition of the previous section, the true risk of \(q\) is \(H + D_{\mathrm{RE}}(\theta^\star\|q)\), and of \(\theta^\star\) itself it is \(H + 0\). So \(\Delta(q) = D_{\mathrm{RE}}(\theta^\star \| q)\), where for yes/no distributions we write the parameters in place of the distributions.

Step 2. Fix \(p = \theta^\star\) and look at the true risk as a function of the parameter, \(f(q) = -p\log q - (1-p)\log(1-q)\) for \(0 < q < 1\). Its derivatives are

\[ f'(q) = -\frac{p}{q} + \frac{1-p}{1-q} = \frac{q - p}{q(1-q)}, \qquad f''(q) = \frac{p}{q^2} + \frac{1-p}{(1-q)^2} \ge 0. \]

- \(f'(q)\) is the slope of the true risk as the parameter moves; over a common denominator, the numerator \(-p(1 - q) + (1 - p)q\) simplifies to \(q - p\).
- \(f''(q)\), the second derivative, is a sum of nonnegative terms.

In one sentence: **the true risk slopes toward the truth, and it curves upward everywhere.** A function whose second derivative is nonnegative is convex, and lesson 12 showed that a differentiable convex function lies above each of its tangent lines: \(f(p) \ge f(q) + f'(q)(p - q)\). (If \(p\) is 0 or 1, \(f\) is still defined there by continuity and the same holds.) Rearranged,

\[ \Delta(q) = f(q) - f(p) \le f'(q)(q - p) = \frac{(q - p)^2}{q(1 - q)}. \]

**The excess risk of a parameter is at most its squared distance from the truth, divided by \(q(1-q)\).** For \(p = 0.3\), \(q = 0.5\): the excess is \(0.082\), and the bound is \(0.04/0.25 = 0.16\). Notice the denominator: near 0 or 1 it collapses, which is the whole story of the bug.

Step 3. With pseudo-counts, \(q = \hat\theta\) lies in the interval from \(a = \alpha/(m + 2\alpha)\) to \(1 - a\). The function \(t(1 - t)\) is a downward parabola, so on an interval it is smallest at an end, and both ends give \(a(1-a) = \alpha(m + \alpha)/(m + 2\alpha)^2\). So, for every training sequence,

\[ \Delta(\hat\theta) \le \frac{(\hat\theta - \theta^\star)^2 (m + 2\alpha)^2}{\alpha(m + \alpha)}. \]

Now average over \(S\). The expected squared distance splits into variance plus squared bias: writing \(\hat\theta - \theta^\star = (\hat\theta - \mathbb E\hat\theta) + (\mathbb E\hat\theta - \theta^\star)\) and squaring, the cross term has expectation 0 because \(\mathbb E[\hat\theta - \mathbb E\hat\theta] = 0\). The variance of \(\hat\theta = (k + \alpha)/(m + 2\alpha)\) is \(\operatorname{Var}[k]/(m + 2\alpha)^2 = m\theta^\star(1 - \theta^\star)/(m + 2\alpha)^2\), since \(k\) is a sum of \(m\) independent coins, each of variance \(\theta^\star(1 - \theta^\star)\). The squared bias is \(\alpha^2(1 - 2\theta^\star)^2/(m + 2\alpha)^2 \le \alpha^2/(m + 2\alpha)^2\). So

\[ \mathbb E_S\left[(\hat\theta - \theta^\star)^2\right] \le \frac{m\theta^\star(1-\theta^\star) + \alpha^2}{(m + 2\alpha)^2}, \]

and multiplying by \((m + 2\alpha)^2/(\alpha(m + \alpha))\) gives the first bound in (b). The second uses \(\theta^\star(1 - \theta^\star) \le 1/4\). \(\blacksquare\)

*Proof of (c).* With \(\alpha = 0\), \(\hat\theta = 0\) whenever all \(m\) observations are 0, which has probability \((1 - \theta^\star)^m\), positive because \(\theta^\star < 1\). On that event the excess risk is infinite, as computed for “winner”, because \(\theta^\star > 0\). An expectation that puts positive probability on \(+\infty\) is \(+\infty\). \(\blacksquare\)

Let’s check (b) for “winner” by computing the expectation exactly, adding over every possible count \(k\) weighted by its binomial probability:

```python
import math
def D(p, q):
    if q in (0, 1):
        return math.inf
    return (p * math.log(p / q)
            + (1 - p) * math.log((1 - p) / (1 - q)))
def expected_excess(p, m, a):
    w = [math.comb(m, k) * p**k * (1 - p)**(m - k)
         for k in range(m + 1)]
    return sum(w[k] * D(p, (k + a) / (m + 2 * a))
               for k in range(m + 1) if w[k] > 0)
p, m = 0.002, 200
for a in (0, 1):
    top = m * p * (1 - p) + a * a
    bound = top / (a * (m + a)) if a else math.inf
    print(a, round(expected_excess(p, m, a), 4), round(bound, 4))
```

It prints `0 inf inf` and `1 0.0026 0.007`: no pseudo-counts, infinite risk; one pseudo-count each way, an expected excess of 0.0026 nats, safely under the bound of 0.007.

[[diagram:pseudocounts]]

There is a price, and (a) shows it. The pull \(\alpha/(m + 2\alpha)\) is a bias toward \(1/2\). One pseudo-count is a whisper next to 600 real messages; a hundred pseudo-counts would drown a rare word’s real frequency. And (b) shows that too few pseudo-counts are dangerous in the other direction: as \(\alpha\) shrinks toward 0, the bound \((m/4 + \alpha^2)/(\alpha(m + \alpha))\) grows like \(1/(4\alpha)\). Remember this when you fix the filter: the guarantee needs both halves.

## A second road to the same place: Bayesian reasoning

Add-one smoothing has a second, independent justification, and it comes from a different philosophy of what a parameter is.

So far we have treated \(\theta\) as a fixed number that we simply do not know. This is the **frequentist** view. The **Bayesian** view says: if we are uncertain about \(\theta\), let’s describe that uncertainty with probabilities too. Before seeing any data, the learner chooses a **prior distribution** \(\mathbb P[\theta]\) over the possible parameter values, saying how plausible it finds each. For example, before reading any mail you might say every frequency of “winner” in junk, from 0 to 1, is equally plausible: the **uniform prior**, with density 1 on \([0, 1]\).

Given \(\theta\), the observations are generated as before: \(\mathbb P[X = x \mid \theta] = \theta^x(1-\theta)^{1-x}\). Together, prior and model give a complete distribution over a single observation \(X\): first draw \(\theta\) from the prior, then \(X\) given \(\theta\). Under the uniform prior, how likely is a 1? Averaging \(\theta\) over the prior gives \(1/2\). In general,

\[ \mathbb P[X = x] = \sum_\theta \mathbb P[\theta]\,\mathbb P[X = x \mid \theta], \qquad \text{or} \qquad \mathbb P[X = x] = \int \mathbb P[\theta]\,\mathbb P[X = x \mid \theta]\,d\theta. \]

- The sum runs over the possible parameter values, each weighted by its prior probability; when \(\theta\) is continuous, the prior is a density and the sum becomes an integral \(\int \dots d\theta\), the area under the curve.
- \(\mathbb P[\theta]\,\mathbb P[X = x \mid \theta]\) is the probability of “the parameter is \(\theta\) and the observation is \(x\)”, by the definition of conditional probability.

In one sentence: **before any data, the chance of an observation is the model’s chance averaged over the prior.** Under the uniform prior, \(\mathbb P[X = 1] = \int_0^1 \theta\,d\theta = 1/2\).

Here is the twist. For a frequentist, the training sequence and a fresh observation are independent: they are separate draws from one fixed distribution. For a Bayesian, they are *not*, because both depend on the same random \(\theta\). Seeing many 0s makes small \(\theta\) more plausible, which makes the next observation more likely to be 0 too. They are independent only **conditionally on \(\theta\)**: once \(\theta\) is fixed, the draws do not influence each other. So the Bayesian’s question is: given the training sequence \(S\), what is the probability of the next observation? Split by the value of \(\theta\):

\[ \mathbb P[X = x \mid S] = \sum_\theta \mathbb P[X = x \mid \theta, S]\,\mathbb P[\theta \mid S] = \sum_\theta \mathbb P[X = x \mid \theta]\,\mathbb P[\theta \mid S]. \]

- \(\mathbb P[\theta \mid S]\) is the **posterior distribution**: how plausible each parameter is after seeing \(S\).
- The first equality is the law of total probability, applied inside the world where \(S\) is known: average over the possible \(\theta\).
- The second drops \(S\) from \(\mathbb P[X = x \mid \theta, S]\), because given \(\theta\), the fresh \(X\) is independent of \(S\).

In one sentence: **the prediction is the model’s prediction averaged over how plausible each parameter is after the data.** This is called the **predictive probability**.

The posterior comes from Bayes’ rule, the same identity Ines used for labels, now applied to parameters:

\[ \mathbb P[\theta \mid S] = \frac{\mathbb P[S \mid \theta]\,\mathbb P[\theta]}{\mathbb P[S]} = \frac{1}{\mathbb P[S]}\prod_{i=1}^{m}\mathbb P[X = x_i \mid \theta]\;\mathbb P[\theta]. \]

- \(\mathbb P[S \mid \theta]\) is the likelihood of \(\theta\), which factors into a product because the observations are independent given \(\theta\).
- \(\mathbb P[S]\) is the total probability of the data, averaged over the prior; it does not depend on \(\theta\), so it only rescales the posterior to total 1.

In one sentence: **after the data, each parameter’s plausibility is its prior plausibility times its likelihood, rescaled.** For a word seen in 0 of 4 junk messages, under the uniform prior, the posterior density is proportional to \((1 - \theta)^4\): a parameter of 0.1 is now \(0.9^4/0.5^4 \approx 10.5\) times as plausible as 0.5.

Now the payoff. Put the uniform prior in, with \(k\) ones among \(m\) observations, and compute the predictive probability of a 1.

**Proposition (Laplace’s rule from a uniform prior).** Under the uniform prior, \(\mathbb P[X = 1 \mid S] = \dfrac{k + 1}{m + 2}\).

*Proof.* Combining the last two displays, with \(\mathbb P[\theta] = 1\),

\[ \mathbb P[X = 1 \mid S] = \frac{\int_0^1 \theta \cdot \theta^k(1-\theta)^{m-k}\,d\theta}{\int_0^1 \theta^k(1-\theta)^{m-k}\,d\theta} = \frac{I(k + 1, m - k)}{I(k, m - k)}, \]

where \(I(a, b) = \int_0^1 \theta^a(1 - \theta)^b\,d\theta\). The denominator is \(\mathbb P[S]\), the integral of prior times likelihood. **The predictive probability of a 1 is the likelihood-weighted average of \(\theta\).**

So we need \(I(a, b)\) for whole numbers \(a, b \ge 0\). **Claim:** \(I(a, b) = \dfrac{a!\,b!}{(a + b + 1)!}\), where \(n! = 1 \cdot 2 \cdots n\) and \(0! = 1\). For instance \(I(0, 0) = 1\), the area of the unit square, and \(I(1, 1) = \int_0^1 \theta(1-\theta)\,d\theta = 1/6\), which the formula gives as \(1 \cdot 1/3! = 1/6\).

Integration by parts, with \(u = (1 - \theta)^b\) and \(dv = \theta^a d\theta\), so \(v = \theta^{a+1}/(a + 1)\), gives for \(b \ge 1\):

\[ I(a, b) = \left[\frac{\theta^{a+1}}{a+1}(1-\theta)^b\right]_0^1 + \frac{b}{a + 1}\int_0^1 \theta^{a+1}(1-\theta)^{b-1}\,d\theta = \frac{b}{a+1}\,I(a + 1, b - 1). \]

The bracket is 0 at both ends (at 1 because of \((1-\theta)^b\) with \(b \ge 1\), at 0 because of \(\theta^{a+1}\)). **Each step moves one power from \(1 - \theta\) to \(\theta\), at the price of the factor \(b/(a + 1)\).** Repeat \(b\) times: \(I(a, b) = \frac{b}{a+1}\cdot\frac{b-1}{a+2}\cdots\frac{1}{a+b}\,I(a + b, 0) = \frac{a!\,b!}{(a+b)!}\cdot\frac{1}{a+b+1}\), since \(I(a+b, 0) = 1/(a + b + 1)\). That proves the claim.

Finally, \(\dfrac{I(k+1, m-k)}{I(k, m-k)} = \dfrac{(k+1)!\,(m-k)!}{(m+2)!}\cdot\dfrac{(m+1)!}{k!\,(m-k)!} = \dfrac{k+1}{m+2}\). \(\blacksquare\)

That is add-one smoothing, exactly. A word never seen in 200 junk messages gets predictive probability \(1/202 \approx 0.00495\), not 0. The pseudo-counts that we introduced as a regularizer are what a Bayesian with a uniform prior predicts. The prior biases the prediction toward its own average, \(1/2\), and the data overwhelms it as \(m\) grows.

```python
import math
def I(a, b, n=100000):
    h = 1 / n
    return sum(((i + .5) * h)**a * (1 - (i + .5) * h)**b
               for i in range(n)) * h
print(round(I(1, 1), 5), round(1 / math.factorial(3), 5))
k, m = 0, 200
print(round(I(k + 1, m - k) / I(k, m - k), 5),
      round((k + 1) / (m + 2), 5))
```

It prints `0.16667 0.16667` and `0.00495 0.00495`: a numerical integral (the midpoint rule with 100,000 slices) agrees with the formula.

**Maximum a posteriori.** Integrals like these rarely have closed forms. A popular shortcut keeps a single parameter: the one that maximizes the posterior,

\[ \hat\theta_{\mathrm{MAP}} \in \operatorname*{argmax}_\theta \mathbb P[\theta \mid S] = \operatorname*{argmax}_\theta \left(\log \mathbb P[S \mid \theta] + \log \mathbb P[\theta]\right). \]

- The equality drops \(\mathbb P[S]\), which does not depend on \(\theta\), and takes logs, which keeps the maximizer.
- So MAP is maximum likelihood plus a bonus \(\log\mathbb P[\theta]\) from the prior: a regularizer once more.

In one sentence: **the maximum a posteriori estimate is the parameter that best balances fitting the data against the prior’s preferences.** With a uniform prior the bonus is constant, and MAP is plain maximum likelihood. With a strong prior, it can overrule weak data. Suppose the security team believes junk makes up more than half of all mail, and encodes this as a prior on the junk share \(\theta\) with density 1.6 for \(\theta \ge 0.5\) and 0.4 below: four times as dense on the upper half, total area \(0.5 \cdot 1.6 + 0.5 \cdot 0.4 = 1\). With 3 junk messages out of 10, maximum likelihood says 0.3. MAP compares \(0.4 \cdot 0.3^3 \cdot 0.7^7 \approx 0.00089\), the best below one half, with \(1.6 \cdot 0.5^{10} \approx 0.00156\), the best at or above one half, and answers 0.5. With 30 out of 100, the data wins and MAP says 0.3.

```python
def prior(th):
    return 1.6 if th >= 0.5 else 0.4
def posterior(th, k, m):
    return prior(th) * th**k * (1 - th)**(m - k)
grid = [i / 1000 for i in range(1, 1000)]
for k, m in ((3, 10), (30, 100)):
    print(k, m, max(grid, key=lambda th: posterior(th, k, m)))
```

It prints `3 10 0.5` and `30 100 0.3`.

[[diagram:bayes]]

## Naive Bayes

Now we can rebuild Ines’s filter on solid ground. Messages are vectors \(x = (x_1, \dots, x_d)\) of yes/no word features, labels are \(y \in \{0, 1\}\), and the best possible hypothesis is the Bayes optimal predictor, which picks the label that is more probable given the message:

\[ h_{\mathrm{Bayes}}(x) = \operatorname*{argmax}_{y \in \{0, 1\}} \mathbb P[Y = y \mid X = x]. \]

- \(X\) and \(Y\) are the random message and label of a fresh example; \(\mathbb P[Y = y \mid X = x]\) is the probability that the label is \(y\) given that the message is \(x\).
- \(\operatorname*{argmax}_{y \in \{0,1\}}\) picks whichever of the two labels gives the larger value.

In one sentence: **label each message with whichever label is more probable for it.** This is the predictor of lesson 3, whose true risk is the smallest possible.

Why not estimate \(\mathbb P[Y = 1 \mid X = x]\) directly, one number per message? Count them. With \(d\) yes/no words there are \(2^d\) possible messages, so \(2^d\) numbers to estimate. Ines’s seven words already mean 128 numbers; a dictionary of 200 words means \(2^{200} \approx 1.6 \times 10^{60}\). Most messages would never appear in any training sequence, and the number of examples needed would grow exponentially with the number of words.

The **naive Bayes** assumption cuts this down with one bold simplification, the one Ines made with her independent coins: **given the label, the words are independent of each other.** Knowing that a junk message contains “free” tells you nothing extra about whether it contains “click”. Formally,

\[ \mathbb P[X = x \mid Y = y] = \prod_{j=1}^{d} \mathbb P[X_j = x_j \mid Y = y]. \]

- \(X_j\) is the random yes/no feature for word \(j\), and \(\mathbb P[X_j = x_j \mid Y = y]\) is the chance that a message of kind \(y\) has the value \(x_j\) for word \(j\): \(\theta_{j,y}\) if \(x_j = 1\), and \(1 - \theta_{j,y}\) if \(x_j = 0\).
- The product over \(j\) is valid exactly because of the assumption: independent events multiply.

In one sentence: **within each kind of message, the probability of the whole message is the product of the probabilities of its words, one by one.** This is Ines’s recipe from the start of the lesson; “free, click” under junk gave \(9/64\).

[[diagram:naive-bayes]]

The assumption is called naive because it is usually false: “claim” and “prize” travel together. But it turns an impossible estimation problem into an easy one. Plug it into Bayes’ rule. The denominator of Bayes’ rule, \(\mathbb P[X = x]\), is the same for both labels, so it cannot change which label wins, and we may drop it:

\[ h_{\mathrm{Bayes}}(x) = \operatorname*{argmax}_{y \in \{0,1\}} \mathbb P[Y = y]\,\mathbb P[X = x \mid Y = y] = \operatorname*{argmax}_{y \in \{0,1\}} \mathbb P[Y = y]\prod_{j=1}^{d}\mathbb P[X_j = x_j \mid Y = y]. \]

- The first equality is Bayes’ rule with the shared denominator removed: the numerators are compared directly.
- The second plugs in the naive Bayes assumption.

In one sentence: **pick the label whose prior share times the product of its word probabilities is larger.** For “free, click” with Ines’s numbers: \(0.5 \cdot 0.1406 = 0.0703\) against \(0.0015\), junk.

Now count parameters: one prior share \(\mathbb P[Y = 1]\), plus \(\theta_{j,y}\) for \(d\) words and 2 labels. That is \(2d + 1\), instead of \(2^d\). For 200 words, 401 numbers. The generative assumption has done what an inductive bias always does: it bought a small number of parameters with a strong belief about the world.

When the parameters are estimated by maximum likelihood (the counted fractions, as Ines did), the result is the classical **naive Bayes classifier**. You now know that this is exactly the version that crashes, and that the right estimator for each \(\theta_{j,y}\) is \(\hat\theta_\alpha\) with \(\alpha > 0\). In what follows the **naive Bayes filter** means naive Bayes with add-\(\alpha\) estimates.

Rerun Ines’s two bug-report messages with add-one estimates, which we can now do with confidence:

```python
import math
W = ["free", "prize", "click", "invoice",
     "meeting", "lunch", "winner"]
junk = [{"free", "prize", "click"}, {"free", "click"},
        {"prize", "click"}, {"free", "prize"}]
keep = [{"meeting", "lunch"}, {"invoice", "meeting"},
        {"lunch", "click"}, {"free", "lunch", "meeting"}]
def freq(msgs, a):
    m = len(msgs)
    return {w: (sum(w in x for x in msgs) + a) / (m + 2 * a)
            for w in W}
def P(x, th):
    return math.prod(th[w] if w in x else 1 - th[w] for w in W)
th1, th0 = freq(junk, 1), freq(keep, 1)
for x in ({"free", "click", "meeting"}, {"winner", "click"}):
    p1, p0 = 0.5 * P(x, th1), 0.5 * P(x, th0)
    print(sorted(x), round(p1 / (p1 + p0), 3))
```

It prints `['click', 'free', 'meeting'] 0.556` and `['click', 'winner'] 0.758`. The junk-looking message with “meeting” in it is now junk with probability 0.556: “meeting” is real evidence for keep and “free, click” real evidence for junk, and four examples per pile can only weigh them so finely. That is a hedge, not a certainty. “winner, click” no longer crashes: “winner” is equally unknown to both piles, so it cancels out, and “click” tips the balance toward junk.

**A guarantee for the whole filter.** The pseudo-count theorem speaks about one parameter. The filter has \(2d\) word parameters, and we want all of them to behave at once. Lesson 2’s union bound does it: the probability that at least one of several bad events happens is at most the sum of their probabilities. Suppose your users label \(m\) messages of each kind, and the share of junk \(\mathbb P[Y = 1]\) is known from traffic logs, so only the \(2d\) word frequencies are estimated. Apply part (a) of the theorem to each of them with confidence \(\delta/(2d)\) instead of \(\delta\). Each fails with probability at most \(\delta/(2d)\), so all \(2d\) hold together with probability at least \(1 - \delta\):

\[ \mathbb P_S\left[\forall j \in [d],\ \forall y \in \{0,1\}:\ |\hat\theta_{j,y} - \theta^\star_{j,y}| \le \sqrt{\frac{\log(4d/\delta)}{2m}} + \frac{\alpha}{m + 2\alpha}\right] \ge 1 - \delta. \]

- \(\forall\) reads “for every”, and \([d]\) is the set of word positions \(\{1, \dots, d\}\), as \([m]\) was in lesson 1.
- \(\theta^\star_{j,y}\) is the true frequency of word \(j\) in messages of kind \(y\), and \(\hat\theta_{j,y}\) its add-\(\alpha\) estimate from the \(m\) messages of that kind.
- \(\log(4d/\delta)\) is \(\log(2/\delta')\) with \(\delta' = \delta/(2d)\): the price of the union bound is a logarithm of the number of parameters.

In one sentence: **with probability at least \(1 - \delta\), every word frequency in both piles is estimated within the same distance of the truth.** Part (b), meanwhile, needs no union bound, because it is a statement about each word’s expected excess risk: **every word, seen or unseen, costs at most \((m/4 + \alpha^2)/(\alpha(m + \alpha))\) nats of expected excess log loss.** With \(d = 200\), \(\delta = 0.05\), \(m = 600\) and \(\alpha = 1\): every frequency within \(0.0915\), and every word’s expected excess at most \(0.2512\) nats.

## Naive Bayes as code

Training only counts. Prediction adds logarithms instead of multiplying probabilities, so that 200 small factors do not underflow to 0, and compares the two sums.

```
NaiveBayesTrain(S, d, α):
  for y in {0, 1}:
    m_y ← number of examples in S with label y
    for j = 1..d:
      k ← number of examples with label y and x_j = 1
      θ[j, y] ← (k + α) / (m_y + 2α)
  return θ

NaiveBayesPredict(θ, π, x):
  for y in {0, 1}:
    s[y] ← log(π_y)
    for j = 1..d:
      s[y] ← s[y] + log(θ[j, y]) if x_j = 1
                    else log(1 − θ[j, y])
  return 1 if s[1] > s[0] else 0
```

Here \(\pi_1 = \mathbb P[Y = 1]\) and \(\pi_0 = 1 - \pi_1\) are the prior shares, known or estimated by counting. The probability of junk, if you want it rather than the label, is \(1/(1 + e^{s[0] - s[1]})\), which is Bayes’ rule divided through by its numerator.

**Training** reads every feature of every example once: \(O(md)\) time, and \(O(d)\) space for the \(2d\) counts. **Prediction** is \(O(d)\) time per message. A message usually contains only a few dictionary words, and that can be exploited: precompute \(B_y = \sum_j \log(1 - \theta_{j,y})\) once, and then for each word present add \(\log\theta_{j,y} - \log(1 - \theta_{j,y})\). Prediction then costs time proportional to the number of words in the message. Notice also what \(\alpha > 0\) does for the code: every \(\theta\) is strictly between 0 and 1, so every \(\log\) is finite, and the crash is impossible by construction rather than by luck.

Compare this with every learner since lesson 9. There is no optimization at all: no gradient, no iterations, no convergence to worry about. That is the computational argument for generative models from the start of the lesson, in its purest form.

**Build it yourself.** Using only the Python standard library:

```python
import math

W = ["free", "prize", "click", "invoice",
     "meeting", "lunch", "winner"]
S = [({"free", "prize", "click"}, 1), ({"free", "click"}, 1),
     ({"prize", "click"}, 1), ({"free", "prize"}, 1),
     ({"meeting", "lunch"}, 0), ({"invoice", "meeting"}, 0),
     ({"lunch", "click"}, 0), ({"free", "lunch", "meeting"}, 0)]

def train(S, words, alpha): ...
def junk_probability(theta, prior, x, words): ...

if __name__ == "__main__":
    theta = train(S, W, 1)
    assert theta[("winner", 1)] == 1 / 6
    assert theta[("free", 0)] == 2 / 6
    p = junk_probability(theta, 0.5, {"winner", "click"}, W)
    assert round(p, 3) == 0.758
    p = junk_probability(theta, 0.5, {"lunch", "meeting"}, W)
    assert round(p, 3) == 0.008
    theta0 = train(S, W, 0)
    p = junk_probability(theta0, 0.5, {"free", "click"}, W)
    assert round(p, 3) == 0.98
```

1. Implement `train`, returning a dictionary keyed by `(word, label)`, and `junk_probability`, working in logarithms and returning \(\mathbb P[Y = 1 \mid x]\).
2. With `alpha = 0`, decide what `junk_probability` should do on “winner, click”. Raise a clear exception? Return `None`? Write down why each choice is worse than smoothing.
3. Build the 200-word world of the lab: word \(j\) (from 0) with rank \(r = j + 1\) has frequency \(0.6\,r^{-0.8}\) in junk and \(0.15\,r^{-0.8}\) in keep when \(j\) is even, the other way round when \(j\) is odd. Draw training sequences and fresh messages, and measure how often maximum likelihood crashes for \(m = 10, 30, 100, 300, 1000\) messages per kind.
4. Implement the sparse prediction trick and check it returns the same probabilities.

## Fix the bug

Ines is back, and she has the 200-word dictionary from the lab, with its long tail of rare words. Users will label up to 600 messages of each kind; the traffic logs say half of all mail is junk, so the prior share is known. Your manager will ship the filter only with a certificate from this lesson’s theorems, holding for every possible set of true word frequencies, not just for the frequencies in this week’s mail:

1. **Accurate:** with probability at least 95%, every one of the 400 word frequencies is estimated within 0.1 of the truth.
2. **Safe on unseen words:** every word, including words never seen in training, has expected excess log loss at most 0.4 nats.

You choose two things: the number \(m\) of labelled messages of each kind, and the estimator, given by its pseudo-count \(\alpha\). Here are both certificates as code, so you can work out the settings before you try them:

```python
import math
d, delta = 200, 0.05
def accuracy(m, a):
    return math.sqrt(math.log(4 * d / delta) / (2 * m)) \
        + a / (m + 2 * a)
def excess(m, a):
    return math.inf if a == 0 else (m / 4 + a * a) / (a * (m + a))
for m, a in ((600, 0), (600, 0.5), (600, 1), (400, 1), (600, 20)):
    print(m, a, round(accuracy(m, a), 4), round(excess(m, a), 4))
```

It prints:

```
600 0 0.0898 inf
600 0.5 0.0906 0.5004
600 1 0.0915 0.2512
400 1 0.1125 0.2519
600 20 0.1211 0.0444
```

Read the rows before you open the lab. Maximum likelihood is accurate and has no certificate at all on unseen words. Half a pseudo-count is not enough. Too few labels break the accuracy certificate, and so do too many pseudo-counts.

In the lab, each round draws a fresh set of true frequencies and a fresh training sequence. A setting that happens to classify this round’s test messages well does not pass; only a setting whose certificates meet both targets does.

[[lab-act2]]

## Another assumption: Gaussian clouds

Naive Bayes is one generative assumption. Here is a second, for measured features, and it leads somewhere you have been before.

Suppose the mail room scores every message on two measured features: the **capitals score** \(x_1\) (how shouty the subject is) and the **link score** \(x_2\) (how many links, scaled). Plot keep messages and they form an elliptical cloud around \((1, 1)\); plot junk and they form a cloud of the same shape around \((3, 2)\). In both clouds the two scores tend to rise together: a shouty message tends to have more links.

To describe a cloud like this we need the Gaussian in \(d\) dimensions. Its centre is a vector \(\mu \in \mathbb R^d\). Its shape is a **covariance matrix** \(\Sigma\), a \(d \times d\) table: on the diagonal, the variance of each score (its spread); off the diagonal, the **covariance** of each pair of scores, positive when they tend to rise together. For our clouds,

\[ \Sigma = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix}: \]

each score has variance 1, and the two scores have covariance 0.5. The density of the Gaussian with centre \(\mu\) and covariance \(\Sigma\) is

\[ \mathbb P[X = x] = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}\exp\left(-\frac12 (x - \mu)^\top \Sigma^{-1}(x - \mu)\right). \]

- \(x - \mu\) is the offset of the message from the centre, a column vector; \((x - \mu)^\top\) is the same offset written as a row (the \(\top\) means transpose).
- \(\Sigma^{-1}\) is the inverse matrix of \(\Sigma\), the matrix with \(\Sigma^{-1}\Sigma = I\), the identity. The product \((x - \mu)^\top\Sigma^{-1}(x - \mu)\) is a single number: a squared distance in which directions where the cloud is wide count less, and directions where it is narrow count more. With \(\Sigma = I\) it is the ordinary squared distance \(\|x - \mu\|^2\).
- \(|\Sigma|\) is the determinant of \(\Sigma\), a measure of the cloud’s volume; with the factor \((2\pi)^{d/2}\) it makes the total volume under the density equal to 1.

In one sentence: **a Gaussian cloud is densest at its centre and thins out with the stretched squared distance from it.** For \(d = 1\) this is the bell from before. At the centre of our clouds, the density is \(1/(2\pi\sqrt{0.75}) \approx 0.184\).

**Linear discriminant analysis** (LDA) makes three generative assumptions: the two labels are equally likely, \(\mathbb P[Y = 1] = \mathbb P[Y = 0] = 1/2\); given the label \(y\), the message is Gaussian with centre \(\mu_y\); and the covariance \(\Sigma\) is the same for both labels. Under these assumptions, what does the Bayes optimal predictor look like?

It predicts junk when \(\mathbb P[Y = 1]\,\mathbb P[X = x \mid Y = 1] > \mathbb P[Y = 0]\,\mathbb P[X = x \mid Y = 0]\), that is, when the ratio of the two sides is above 1, that is, when its logarithm is above 0. That logarithm is the **log-likelihood ratio**:

\[ \log\left(\frac{\mathbb P[Y = 1]\,\mathbb P[X = x \mid Y = 1]}{\mathbb P[Y = 0]\,\mathbb P[X = x \mid Y = 0]}\right) = \frac12(x - \mu_0)^\top\Sigma^{-1}(x - \mu_0) - \frac12(x - \mu_1)^\top\Sigma^{-1}(x - \mu_1). \]

- On the left, the prior shares are equal and cancel, and so do the normalizing constants \((2\pi)^{d/2}|\Sigma|^{1/2}\), because \(\Sigma\) is shared.
- What is left is the log of a ratio of two exponentials, which is the difference of their exponents.

In one sentence: **the evidence for junk is how much closer the message is to the junk centre than to the keep centre, in stretched distance.** For the message \(x = (3, 2)\), sitting on the junk centre, its stretched squared distance to the keep centre is \((2, 1)\Sigma^{-1}(2, 1)^\top = 4\) and to the junk centre 0, so the ratio’s log is \(\frac12 \cdot 4 = 2\): junk is \(e^2 \approx 7.4\) times as likely.

Now expand both quadratic forms. Each is \(x^\top\Sigma^{-1}x - 2\mu_y^\top\Sigma^{-1}x + \mu_y^\top\Sigma^{-1}\mu_y\), where the middle term combines \(x^\top\Sigma^{-1}\mu_y\) and \(\mu_y^\top\Sigma^{-1}x\), which are equal because \(\Sigma\), and hence \(\Sigma^{-1}\), is symmetric. The \(x^\top\Sigma^{-1}x\) terms cancel between the two halves, and the log-likelihood ratio is \(\langle w, x\rangle + b\) with

\[ w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad b = \frac12\left(\mu_0^\top\Sigma^{-1}\mu_0 - \mu_1^\top\Sigma^{-1}\mu_1\right). \]

- \(\langle w, x\rangle = \sum_j w_j x_j\) is the inner product of lesson 9: a weighted sum of the scores.
- \(w\) points from the keep centre to the junk centre, then is reshaped by \(\Sigma^{-1}\) to account for the clouds’ correlations.
- \(b\) is a constant: it puts the boundary halfway between the clouds, in stretched distance.

In one sentence: **under the LDA assumptions, the Bayes optimal predictor is a halfspace: junk exactly when a weighted sum of the scores exceeds a threshold.** For our clouds, \(\Sigma^{-1} = \frac{1}{0.75}\begin{pmatrix}1 & -0.5\\ -0.5 & 1\end{pmatrix}\), \(\mu_1 - \mu_0 = (2, 1)\), so \(w = (2, 0)\), and \(b = \frac12(\frac43 - \frac{28}{3}) = -4\). Junk exactly when \(2x_1 - 4 > 0\), that is, when the capitals score exceeds 2.

[[diagram:lda]]

Look at that \(w\): the link score gets weight 0, even though junk has more links on average. The reason is the correlation. Given the capitals score, the extra links of junk are just what the shared cloud shape predicts from its shoutiness, so they carry no further evidence. A rule that weighs each score by its own difference in means would get this wrong.

```python
import math
mu0, mu1 = (1, 1), (3, 2)
inv = [[1 / .75, -.5 / .75], [-.5 / .75, 1 / .75]]
def quad(v):
    return sum(v[i] * inv[i][j] * v[j]
               for i in range(2) for j in range(2))
def log_ratio(x):
    d0 = [x[i] - mu0[i] for i in range(2)]
    d1 = [x[i] - mu1[i] for i in range(2)]
    return 0.5 * quad(d0) - 0.5 * quad(d1)
dm = [mu1[i] - mu0[i] for i in range(2)]
w = [sum(inv[i][j] * dm[j] for j in range(2)) for i in range(2)]
b = 0.5 * (quad(mu0) - quad(mu1))
pts = [(a / 4, c / 4)
       for a in range(-8, 24) for c in range(-8, 24)]
print([round(v, 6) for v in w], round(b, 6),
      round(log_ratio(mu1), 3))
print(max(abs(log_ratio(x) - (w[0] * x[0] + w[1] * x[1] + b))
          for x in pts) < 1e-9)
```

It prints `[2.0, 0.0] -4.0 2.0` and `True`: on 1,024 grid points, the difference of stretched distances and the linear rule agree.

So a generative assumption has rederived the hypothesis class of lesson 9. The difference is in how the halfspace is chosen. LDA does not search for a good \(w\); it estimates the centres and the covariance, and computes \(w\) and \(b\) from them. The usual estimates are the maximum likelihood ones: each centre is the average of its class, and the shared covariance is the average, over all \(m\) examples, of the outer product of each example’s offset from its own class centre.

```
LDATrain(S):
  for y in {0, 1}:
    μ[y] ← average of the x_i with y_i = y
  Σ ← (1/m) Σ_i (x_i − μ[y_i]) (x_i − μ[y_i])ᵀ
  w ← solve Σ w = μ[1] − μ[0]
  b ← ½ (μ[0]ᵀ Σ⁻¹ μ[0] − μ[1]ᵀ Σ⁻¹ μ[1])
  return (w, b)

LDAPredict(w, b, x):
  return 1 if ⟨w, x⟩ + b > 0 else 0
```

**Cost.** The centres take \(O(md)\) time, the covariance \(O(md^2)\) (one \(d \times d\) outer product per example), and solving the linear system \(O(d^3)\) by Gaussian elimination; space is \(O(d^2)\) for \(\Sigma\). Prediction is \(O(d)\), like any halfspace. For \(b\), note that \(\mu_y^\top\Sigma^{-1}\mu_y\) can reuse the solved system: \(b = -\tfrac12\langle w, \mu_1 + \mu_0\rangle\), because \(\mu_0^\top\Sigma^{-1}\mu_0 - \mu_1^\top\Sigma^{-1}\mu_1 = -(\mu_1 - \mu_0)^\top\Sigma^{-1}(\mu_1 + \mu_0)\) by symmetry of \(\Sigma^{-1}\).

**Build it yourself.** In the standard library, for \(d = 2\):

```python
S = [((2, 2), 0), ((0, 0), 0), ((2, 1), 0), ((0, 1), 0),
     ((4, 3), 1), ((2, 1), 1), ((4, 2), 1), ((2, 2), 1)]

def lda_train(S): ...
def lda_predict(w, b, x): ...

if __name__ == "__main__":
    w, b = lda_train(S)
    assert [round(v, 6) for v in w] == [2.0, 0.0]
    assert round(b, 6) == -4.0
    assert lda_predict(w, b, (3, 2)) == 1
    assert lda_predict(w, b, (1, 1)) == 0
```

1. Implement `lda_train` (class means, pooled covariance, a 2 by 2 inverse) and `lda_predict`.
2. Draw 2,000 messages from the two clouds of this section (use `random.gauss` on independent coordinates and mix them to get covariance 0.5) and check that the estimated \(w\) approaches \((2, 0)\) as \(m\) grows.
3. Break the shared-covariance assumption: give junk a wider cloud. Compare LDA’s error with the Bayes optimal predictor’s, computed from the true densities.

## Hidden causes: latent variables

So far every example came with its label. Now look at the unlabelled archive: a million messages nobody ever clicked on. The product team suspects there are two kinds of message in it, newsletters and personal notes, with different length scores, but no message says which kind it is. The kind is a **latent variable**: part of the story of how the message was produced, but never observed.

A model with a latent variable tells the production story in two steps. First, choose the kind: kind \(y\) with probability \(c_y\), for \(y\) in \(\{1, \dots, k\}\). Second, write a message of that kind: draw \(x\) from a Gaussian with centre \(\mu_y\) and covariance \(\Sigma_y\). With two kinds, a newsletter share of \(c_2 = 0.3\), and centres 1 and 4 on the length score, you get a two-humped density. The probability (density) of a message adds over the kinds that could have produced it:

\[ \mathbb P[X = x] = \sum_{y=1}^{k}\mathbb P[Y = y]\,\mathbb P[X = x \mid Y = y] = \sum_{y=1}^{k} c_y\,\frac{1}{(2\pi)^{d/2}|\Sigma_y|^{1/2}}\exp\left(-\frac12(x - \mu_y)^\top\Sigma_y^{-1}(x - \mu_y)\right). \]

- \(Y\) is the latent kind and \(c_y = \mathbb P[Y = y]\) its share; the \(c_y\) are nonnegative and add up to 1.
- The first equality is the law of total probability: a message arises by some kind, so add the chances over the kinds.
- Each term is a kind’s share times its Gaussian cloud, the density from the previous section with its own centre and covariance.

In one sentence: **a mixture of Gaussians is a weighted sum of bells, one per hidden kind.** At \(x = 1\), with \(c = (0.7, 0.3)\), centres 1 and 4 and width 1: \(0.7 \cdot 0.399 + 0.3 \cdot 0.0044 \approx 0.281\).

Let \(\theta\) collect all the parameters (the shares, centres and covariances). For an i.i.d. sample \(S = (x_1, \dots, x_m)\), the log-likelihood is

\[ L(\theta) = \sum_{i=1}^{m}\log \mathbb P_\theta[X = x_i] = \sum_{i=1}^{m}\log\left(\sum_{y=1}^{k}\mathbb P_\theta[X = x_i, Y = y]\right). \]

- \(\mathbb P_\theta[X = x_i, Y = y]\) is the joint probability that the kind is \(y\) and the message is \(x_i\): \(c_y\) times the \(y\)-th bell at \(x_i\).
- The outer sum adds one log per example; the inner sum, inside the log, adds over the kinds.

In one sentence: **the log-likelihood adds, over messages, the log of the total chance of each message over all the kinds that might have produced it.** Maximum likelihood wants the \(\theta\) that maximizes it.

The sum inside the logarithm is the trouble. Without it, the log of each joint probability would split into simple pieces, and maximizing would be as easy as for LDA: if we knew each message’s kind, we would just average each kind’s messages. With it, there is no closed form, and in many cases the optimization is computationally hard. The **Expectation-Maximization** algorithm, **EM**, due to Dempster, Laird and Rubin, is an iterative way to search for a maximum. It may stop at a local maximum rather than the global one, but it often works well in practice.

EM’s idea is a chicken-and-egg loop. If we knew the kinds, finding \(\theta\) would be easy. If we knew \(\theta\), we could compute, for each message, how probable each kind is. So alternate. To say it precisely, replace “the kind of message \(i\)” by a **soft assignment**: a probability vector \(Q_{i,1}, \dots, Q_{i,k}\) over the kinds for each message, collected in an \(m \times k\) matrix \(Q\) whose rows add up to 1. Then define the expected complete log-likelihood:

\[ F(Q, \theta) = \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\log\left(\mathbb P_\theta[X = x_i, Y = y]\right). \]

- For each message \(i\) and kind \(y\), \(\log\mathbb P_\theta[X = x_i, Y = y]\) is what the log-likelihood would be if we knew that message \(i\) has kind \(y\).
- \(Q_{i,y}\) weights it by how much we currently believe that. So \(F\) is the expected log-likelihood of a labelled data set whose labels are drawn from the rows of \(Q\).
- The sum over kinds is now *outside* the log.

In one sentence: **\(F\) is the log-likelihood we would have if the hidden kinds were known, averaged over our current beliefs about them.** With one message \(x = 1\), the belief \(Q = (0.9, 0.1)\), and the mixture above: \(0.9\log(0.7 \cdot 0.399) + 0.1\log(0.3 \cdot 0.0044) \approx -1.81\).

We assume, as is true for Gaussian mixtures, that for any fixed \(Q\) with rows adding to 1, maximizing \(F(Q, \theta)\) over \(\theta\) is tractable. EM builds a sequence \((Q^{(1)}, \theta^{(1)}), (Q^{(2)}, \theta^{(2)}), \dots\); the superscript \((t)\) is the iteration number. Each iteration does two steps:

\[ \text{E step:}\quad Q^{(t+1)}_{i,y} = \mathbb P_{\theta^{(t)}}[Y = y \mid X = x_i], \qquad \text{M step:}\quad \theta^{(t+1)} \in \operatorname*{argmax}_\theta F(Q^{(t+1)}, \theta). \]

- The **E step** (expectation) sets each soft assignment to the probability of each kind given the message, under the current parameters, by Bayes’ rule. It defines the new expected log-likelihood \(F(Q^{(t+1)}, \cdot)\).
- The **M step** (maximization) finds the parameters that maximize that expected log-likelihood, which is tractable by assumption.

In one sentence: **guess the hidden kinds from the parameters, then refit the parameters to the guessed kinds, and repeat.** The starting values are usually random, and the loop stops when the log-likelihood stops improving noticeably.

## Why EM never goes downhill

An iterative algorithm with no guarantee is a hope. EM comes with one: the log-likelihood never decreases. The proof is short once we see EM as climbing a different function, one coordinate at a time.

Define, with the convention \(0 \log 0 = 0\),

\[ G(Q, \theta) = F(Q, \theta) - \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\log Q_{i,y} = \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\log\frac{\mathbb P_\theta[X = x_i, Y = y]}{Q_{i,y}}. \]

- The subtracted double sum is minus the sum of the entropies of the rows of \(Q\): a bonus for soft assignments that are uncertain.
- The second form combines the two logs into one; terms with \(Q_{i,y} = 0\) are left out.

In one sentence: **\(G\) is the expected complete log-likelihood plus the entropy of the beliefs.** Let \(\mathcal Q\) be the set of \(m \times k\) matrices with entries in \([0, 1]\) whose rows add up to 1, the allowed soft assignments.

**Lemma (EM is alternate maximization).** The two EM steps are \(Q^{(t+1)} \in \operatorname*{argmax}_{Q \in \mathcal Q} G(Q, \theta^{(t)})\) and \(\theta^{(t+1)} \in \operatorname*{argmax}_\theta G(Q^{(t+1)}, \theta)\). Moreover, \(G(Q^{(t+1)}, \theta^{(t)}) = L(\theta^{(t)})\).

*Proof.* The M step first. The entropy term of \(G\) does not involve \(\theta\), so for fixed \(Q^{(t+1)}\), maximizing \(G\) over \(\theta\) is the same as maximizing \(F\). That is the M step.

The E step. Fix any \(\theta\) and any \(Q \in \mathcal Q\). We show \(G(Q, \theta) \le L(\theta)\), and then that the E step’s choice reaches it.

For the inequality, fix a message \(i\) and write \(a_y = \mathbb P_\theta[X = x_i, Y = y]\) and \(A = \sum_y a_y = \mathbb P_\theta[X = x_i]\). Using \(\log t \le t - 1\) from the relative-entropy proof, with \(t = a_y/(Q_{i,y}A)\), on the kinds with \(Q_{i,y} > 0\):

\[ \sum_{y : Q_{i,y} > 0} Q_{i,y}\log\frac{a_y}{Q_{i,y}A} \le \sum_{y : Q_{i,y} > 0} Q_{i,y}\left(\frac{a_y}{Q_{i,y}A} - 1\right) = \sum_{y : Q_{i,y} > 0}\frac{a_y}{A} - 1 \le 0. \]

- The left side is message \(i\)’s part of \(G\), minus \(\log A\) (because the \(Q_{i,y}\) add up to 1, \(\sum_y Q_{i,y}\log A = \log A\)).
- The middle step multiplies out: each \(Q_{i,y}\) cancels, and the \(Q_{i,y}\) themselves add up to 1.
- The last step holds because the \(a_y\) over some kinds add up to at most \(A\).

In one sentence: **any soft assignment explains message \(i\) at most as well as the true total chance of message \(i\).** This is Jensen’s inequality for the logarithm, in the special form we need. Adding over \(i\): \(G(Q, \theta) \le \sum_i \log\mathbb P_\theta[X = x_i] = L(\theta)\).

For the equality, put \(Q_{i,y} = \mathbb P_\theta[Y = y \mid X = x_i]\). By the definition of conditional probability, \(\mathbb P_\theta[X = x_i, Y = y] = \mathbb P_\theta[Y = y \mid X = x_i]\,\mathbb P_\theta[X = x_i]\), so the ratio inside the log of \(G\) is \(\mathbb P_\theta[X = x_i]\) for every \(y\), and

\[ G(Q, \theta) = \sum_{i=1}^{m}\log\mathbb P_\theta[X = x_i]\sum_{y=1}^{k}\mathbb P_\theta[Y = y \mid X = x_i] = \sum_{i=1}^{m}\log\mathbb P_\theta[X = x_i] = L(\theta), \]

since conditional probabilities over all kinds add up to 1. **With the E step’s beliefs, \(G\) equals the log-likelihood exactly.** (For a Gaussian mixture every joint probability is positive, so no terms are dropped.) So the E step’s \(Q\) attains the upper bound \(L(\theta)\): it maximizes \(G(\cdot, \theta)\) over \(\mathcal Q\), and \(G(Q^{(t+1)}, \theta^{(t)}) = L(\theta^{(t)})\). \(\blacksquare\)

Here is the lemma on the one-message example from the previous section, \(x = 1\) with shares \((0.7, 0.3)\) and centres 1 and 4: the mixture density, \(F\) and \(G\) for the belief \((0.9, 0.1)\), and \(G\) for the E step’s belief, against the log-likelihood.

```python
import math
c, mu, x = (0.7, 0.3), (1.0, 4.0), 1.0
def bell(x, m):
    return math.exp(-(x - m)**2 / 2) / math.sqrt(2 * math.pi)
a = [c[y] * bell(x, mu[y]) for y in (0, 1)]
L = math.log(sum(a))
def F(Q):
    return sum(Q[y] * math.log(a[y]) for y in (0, 1))
def G(Q):
    return F(Q) - sum(q * math.log(q) for q in Q if q > 0)
post = [v / sum(a) for v in a]
print(round(sum(a), 3), round(F((0.9, 0.1)), 2))
print(round(G((0.9, 0.1)), 3), round(G(post), 3), round(L, 3))
```

It prints `0.281 -1.81`, then `-1.485 -1.271 -1.271`: the guessed belief gives a \(G\) below the log-likelihood, and the E step’s belief, \((0.995, 0.005)\), meets it exactly.

**Theorem (EM never decreases the likelihood).** For every \(t\), \(L(\theta^{(t+1)}) \ge L(\theta^{(t)})\).

*Proof.* Three links, each from the lemma:

\[ L(\theta^{(t+1)}) = G(Q^{(t+2)}, \theta^{(t+1)}) \ge G(Q^{(t+1)}, \theta^{(t+1)}) \ge G(Q^{(t+1)}, \theta^{(t)}) = L(\theta^{(t)}). \]

- The first equality is the lemma’s “moreover” at iteration \(t+1\): the E step’s beliefs make \(G\) equal to \(L\).
- The first inequality: \(Q^{(t+2)}\) maximizes \(G(\cdot, \theta^{(t+1)})\) over \(\mathcal Q\), so it does at least as well as the particular matrix \(Q^{(t+1)}\).
- The second inequality: \(\theta^{(t+1)}\) maximizes \(G(Q^{(t+1)}, \cdot)\), so it does at least as well as \(\theta^{(t)}\).
- The last equality is the lemma’s “moreover” at iteration \(t\).

In one sentence: **each E step lifts \(G\) up to the log-likelihood, each M step lifts \(G\) further, so the log-likelihood can only climb.** \(\blacksquare\)

[[diagram:em]]

Since the log-likelihood never decreases, and (for a mixture where it is bounded above) a nondecreasing bounded sequence converges, the values \(L(\theta^{(t)})\) settle down. What the theorem does not promise is that they settle at the *global* maximum: EM climbs, and a hill can be a local one.

## Soft k-means

Let’s write EM out for a mixture of \(k\) Gaussians in which every covariance is the identity, \(\Sigma_y = I\), so the parameters are the shares \(c\) and the centres \(\mu_1, \dots, \mu_k\).

**E step.** By Bayes’ rule, the probability of kind \(y\) given message \(x_i\) is the joint probability divided by the total over kinds. The Gaussian normalizing constant \((2\pi)^{d/2}\) is the same for every kind and cancels:

\[ \mathbb P_{\theta^{(t)}}[Y = y \mid X = x_i] = \frac{1}{Z_i}\,c^{(t)}_y\exp\left(-\frac12\left\|x_i - \mu^{(t)}_y\right\|^2\right). \]

- \(\|x_i - \mu_y\|^2\) is the ordinary squared distance from message \(i\) to centre \(y\) (the stretched distance with \(\Sigma = I\)).
- \(Z_i\) is the **normalization factor**, \(\sum_{y'} c_{y'}\exp(-\frac12\|x_i - \mu_{y'}\|^2)\), which makes the probabilities for message \(i\) add up to 1.

In one sentence: **each message belongs to each kind in proportion to the kind’s share times how close the message is to its centre.** A message at length score 2.5, halfway between centres 1 and 4 with shares 0.7 and 0.3, belongs to the first kind with probability \(0.7/(0.7 + 0.3) = 0.7\), since the distances are equal.

**M step.** Write \(Q_{i,y}\) for the E step’s probabilities. Dropping constants that do not involve \(c\) or \(\mu\), the M step maximizes

\[ \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\left(\log c_y - \frac12\|x_i - \mu_y\|^2\right). \]

- Each term is \(\log\mathbb P_\theta[X = x_i, Y = y]\), which is \(\log c_y - \frac12\|x_i - \mu_y\|^2 - \frac d2\log 2\pi\), with the last constant dropped.

In one sentence: **reward shares that are large where the beliefs are large, and centres that are close to the messages believed to belong to them.** The centres and the shares appear in separate terms, so they can be optimized separately.

For a centre \(\mu_y\), set the gradient to zero. The gradient of \(-\frac12\|x_i - \mu_y\|^2\) with respect to \(\mu_y\) is \(x_i - \mu_y\), so the condition is \(\sum_i Q_{i,y}(x_i - \mu_y) = 0\); the objective is a concave quadratic in \(\mu_y\), so this point is the maximum. For the shares, the constraint that \(c\) is a probability vector needs more care, and you will prove the answer in the problems. The results are

\[ \mu_y = \frac{\sum_{i=1}^{m} Q_{i,y}\,x_i}{\sum_{i=1}^{m} Q_{i,y}}, \qquad c_y = \frac{\sum_{i=1}^{m} Q_{i,y}}{\sum_{y'=1}^{k}\sum_{i=1}^{m} Q_{i,y'}}. \]

- \(\mu_y\) is a weighted average of all the messages, each weighted by how much it belongs to kind \(y\).
- \(\sum_i Q_{i,y}\) is the total belief in kind \(y\), a “soft count” of its messages; \(c_y\) is that soft count as a share of all \(m\) (the denominator equals \(m\), since every row of \(Q\) adds up to 1).

In one sentence: **move each centre to the weighted average of the messages, and set each share to the kind’s soft count over \(m\).** If three messages at 0, 1 and 2 belong to kind 1 with weights 1, 1 and 0.5, the new centre is \((0 + 1 + 1)/2.5 = 0.8\).

Compare this with the k-means algorithm of lesson 22. There, each message is assigned to its nearest centre, and each centre moves to the average of its assigned messages. Here, each message is assigned to every centre with a probability, and each centre moves to a weighted average over the entire sample. That is why EM for this model is called **soft k-means**.

[[diagram:soft-kmeans]]

```
SoftKMeans(x_1..x_m, k, T):
  choose initial c and μ_1..μ_k
  repeat T times (or until L stops improving):
    for i = 1..m:                      ▷ E step
      for y = 1..k:
        Q[i, y] ← c_y · exp(−½‖x_i − μ_y‖²)
      divide row i of Q by its sum
    for y = 1..k:                      ▷ M step
      n_y ← Σ_i Q[i, y]
      μ_y ← (Σ_i Q[i, y] x_i) / n_y
      c_y ← n_y / m
  return c, μ
```

**Cost.** Each E step computes \(mk\) squared distances in \(d\) dimensions, \(O(mkd)\) time; each M step is another weighted pass, \(O(mkd)\). So one iteration costs \(O(mkd)\) time, the same as one k-means iteration, and the memory is \(O(mk + kd)\) for \(Q\) and the centres (the data itself aside). In practice the exponentials are computed after subtracting each row’s largest exponent, so that \(\exp\) of a very negative number does not underflow to 0 for every kind at once.

Here is EM on the unlabelled archive, in one dimension: 140 personal notes around length score 1 and 60 newsletters around 4, with the kinds hidden. The snippet records the log-likelihood before each E step and checks that it never decreases.

```python
import math, random
rng = random.Random(4)
xs = [rng.gauss(1, 1) for _ in range(140)]
xs += [rng.gauss(4, 1) for _ in range(60)]
c, mu, lls = [0.5, 0.5], [0.0, 0.5], []
def joint(x, y):
    g = math.exp(-(x - mu[y])**2 / 2) / math.sqrt(2 * math.pi)
    return c[y] * g
for t in range(40):
    tot = [joint(x, 0) + joint(x, 1) for x in xs]
    lls.append(sum(math.log(v) for v in tot))
    Q = [[joint(x, y) / v for y in (0, 1)]
         for x, v in zip(xs, tot)]
    n = [sum(q[y] for q in Q) for y in (0, 1)]
    mu = [sum(q[y] * x for q, x in zip(Q, xs)) / n[y]
          for y in (0, 1)]
    c = [n[y] / len(xs) for y in (0, 1)]
print([round(v, 1) for v in lls[:4]], round(lls[-1], 1))
print(all(b >= a - 1e-9 for a, b in zip(lls, lls[1:])))
print([round(v, 2) for v in mu], [round(v, 2) for v in c])
```

It prints `[-711.7, -414.6, -396.7, -387.4] -364.0`, the log-likelihood climbing fast and then settling, then `True`, then `[1.18, 3.98] [0.72, 0.28]`: centres and shares close to the hidden truth (1 and 4, 0.7 and 0.3), found without a single label.

**Build it yourself.** In the standard library, for messages in \(\mathbb R^d\):

```python
import math

X = [(0.0, 0.0), (0.5, 0.2), (0.1, 0.6), (4.0, 4.1),
     (3.6, 4.4), (4.3, 3.8), (0.3, 0.1), (4.1, 4.0)]

def soft_kmeans(X, mu, c, iters): ...
def log_likelihood(X, mu, c): ...

if __name__ == "__main__":
    mu0, c0 = [(1.0, 0.0), (2.0, 3.0)], [0.5, 0.5]
    lls = []
    for t in range(1, 21):
        mu, c = soft_kmeans(X, mu0, c0, t)
        lls.append(log_likelihood(X, mu, c))
    assert all(b >= a - 1e-12 for a, b in zip(lls, lls[1:]))
    assert [round(v, 2) for v in sorted(c)] == [0.5, 0.5]
    assert round(lls[-1], 2) == -20.65
```

1. Implement `soft_kmeans` (run `iters` EM iterations from the given start, identity covariances) and `log_likelihood` (the mixture log-likelihood with the \((2\pi)^{d/2}\) constant included).
2. Start both centres at the same point. What happens, and which line of the monotonicity proof tells you EM can never escape?
3. Replace the E step by a hard assignment to the most probable kind, and check that you have rebuilt k-means from lesson 22. Does the log-likelihood still never decrease?

## What people get wrong

**“A probability of 0 is just a very small probability.”** In a product it is not small, it is a veto: one factor of 0 wipes out every other word’s evidence, and in a log it is minus infinity. The rare-word theorem shows that with \(m \le \log 2/(2\theta^\star)\) examples, maximum likelihood produces such a veto with probability at least one half, and its true risk is then infinite.

**“Smoothing is a hack that makes the numbers look nicer.”** Add-\(\alpha\) is regularized maximum likelihood, and add-one is exactly the predictive probability of a Bayesian with a uniform prior. It comes with a theorem: finite expected excess risk for every word, at the cost of a bias of at most \(\alpha/(m + 2\alpha)\).

**“More pseudo-counts are always safer.”** They make the log-loss certificate smaller but pull every estimate toward \(1/2\). In the fix, 20 pseudo-counts broke the accuracy certificate at 600 labels per kind.

**“Naive Bayes assumes the words are independent.”** It assumes they are independent *given the label*. “free” and “prize” are strongly correlated across all mail, simply because both are common in junk; the assumption is only that, within junk, one does not make the other more likely. That assumption is still usually false, and naive Bayes works anyway surprisingly often, because a classifier only needs the right side of the comparison, not the right probabilities.

**“EM finds the maximum likelihood estimate.”** It never decreases the likelihood, which is all the theorem says. It can stop at a local maximum, and a bad start (two identical centres, for instance) can keep it there forever.

**“Maximum likelihood is a different principle from ERM.”** It is ERM with the log loss. Everything you know about ERM applies, including that it overfits when the class is too rich for the data, and that a regularizer helps.

## Recap and the next bug

1. The **generative** approach assumes the data comes from a **parametric model** and estimates its **parameters**; the discriminative approach of the earlier lessons only learns to label.
2. The **maximum likelihood estimator** picks the parameter that makes the training sequence most likely. For a yes/no feature it is the counted fraction \(k/m\); for a Gaussian, the sample mean and the root of the average squared deviation.
3. Maximum likelihood is **ERM with the log loss** \(-\log P_\theta[x]\). Its true risk is the **entropy** of the data plus the **relative entropy** from the truth to the model, which is zero only when the model is right.
4. For a Gaussian centre the expected excess risk is \(1/(2m)\). For a rare yes/no feature, maximum likelihood sets the estimate to 0 with probability at least one half when \(m \le \log 2/(2\theta^\star)\), and its true risk is infinite: the bug.
5. **Pseudo-counts**, \(\hat\theta_\alpha = (k + \alpha)/(m + 2\alpha)\), are regularized maximum likelihood and, for \(\alpha = 1\), the Bayesian prediction under a uniform prior. They keep the estimate within \(\sqrt{\log(2/\delta)/(2m)} + \alpha/(m + 2\alpha)\) of the truth and the expected excess risk below \((m/4 + \alpha^2)/(\alpha(m + \alpha))\).
6. **Naive Bayes** assumes the features are independent given the label, cutting \(2^d\) parameters to \(2d + 1\); trained by counting in \(O(md)\), it predicts in \(O(d)\). **LDA** assumes Gaussian classes with a shared covariance and produces a halfspace.
7. **EM** handles latent variables by alternating soft assignments and refits; it is alternate maximization of \(G\), so the log-likelihood never decreases. With identity covariances it is **soft k-means**.

Ines’s filter now ships with a certificate, and it is fast. But look at what it is made of. It keeps a parameter for every one of the 200 words in the dictionary, and the real dictionary has tens of thousands. Most of those words are noise: they appear about as often in junk as in keep, and every one of them adds a little random evidence to every decision, and two more numbers to estimate. And some of the most useful signals are not words at all: the ratio of links to text, a sender’s history, “prize” and “claim” appearing together. Which features should the filter keep, and could it build better ones than the ones it was given? That is the next lesson: feature selection and generation.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. The word “unsubscribe” appears in every one of the 300 junk messages your users labelled. What is its maximum likelihood estimate in junk, and what happens to Ines’s filter on a junk message that does not contain it? What does add-one smoothing estimate instead? Which part of the pseudo-count theorem covers this case, and why does it not matter which end of \([0, 1]\) the true frequency is close to?
2. In the proof that EM never decreases the likelihood, the chain has two equalities and two inequalities. Suppose your M step is only approximate: it returns some \(\theta^{(t+1)}\) with \(F(Q^{(t+1)}, \theta^{(t+1)}) \ge F(Q^{(t+1)}, \theta^{(t)})\), but not the maximizer. Does the theorem still hold? Which link of the chain changes, and why is it still valid? What if instead the E step were approximate?
3. A word appears in junk with true frequency \(\theta^\star = 0.001\). (a) Up to what number of junk messages \(m\) does the rare-word theorem guarantee that maximum likelihood has infinite true risk with probability at least one half? (b) With \(m = 1{,}000\) and add-one smoothing, what does the pseudo-count theorem bound the expected excess risk by? (c) With \(m = 1{,}000\) and \(\delta = 0.05\), how far from \(\theta^\star\) can the add-one estimate be, by part (a) of the theorem, and why is that bound so much worse than the bias alone?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (a biased width).** Let \(x_1, \dots, x_m\) be i.i.d. Gaussian with centre \(\mu\) and variance \(\sigma^2\), and let \(\hat\sigma^2 = \frac1m\sum_i(x_i - \hat\mu)^2\) be the maximum likelihood estimate of the variance. Prove that \(\mathbb E[\hat\sigma^2] = \frac{m-1}{m}\sigma^2\), so the estimator is biased, and find a multiple of \(\hat\sigma^2\) that is unbiased.
- **Problem 2 (the shares of soft k-means).** For nonnegative weights \(\nu_1, \dots, \nu_k\), not all zero, consider maximizing \(\sum_{y=1}^k \nu_y\log c_y\) over probability vectors \(c\) with every \(c_y > 0\). (i) Check that the share part of the soft k-means M step has this form, with \(\nu_y = \sum_i Q_{i,y}\). (ii) Show that \(c^\star = \nu/\sum_y \nu_y\) is a probability vector. (iii) Show that the problem is equivalent to minimizing the relative entropy \(D_{\mathrm{RE}}(c^\star\|c)\) over the same \(c\). (iv) Conclude, using a property of the relative entropy proved in this lesson, that \(c^\star\) is the solution.
