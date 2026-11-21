# Discriminant Analysis and EM

## The loop that made things worse

Last lesson’s certificate came with a condition that was easy to miss: every message in it carried its label. Your users labelled 600 messages of each kind, and the theorems did the rest. But labels are the expensive part of the whole business. Meanwhile the product team has been saving something else, an archive of a million messages that nobody ever labelled, and they have a hunch: two kinds of message are hiding in it, newsletters and personal notes, with different length scores. (A length score, remember, is the natural logarithm of a message’s number of characters.) If a filter could learn from the archive alone, labels would stop being a bottleneck.

Ines takes the hunch seriously and builds the obvious loop. Start with a guess for the two kinds: where each one is centred on the length score, and what share of the archive it makes up. Then repeat two steps. First, give every message to whichever kind finds it more probable. Second, refit: move each kind’s centre to the average score of the messages it just received, and set its share to the fraction of messages it received. She keeps one number on her screen, the archive’s log-likelihood: how likely the current model finds all the messages together, measured in logs, as in lesson 24a. Higher is better, and a loop that keeps refitting to the data ought to push it up.

She tries it on a test archive of 200 messages, 140 personal notes centred on a length score of 1 and 60 newsletters centred on 4, with the kinds hidden from the loop. Before the first round the log-likelihood is \(-711.7\). After round one it is \(-422.2\), a huge improvement. Then the dashboard turns red. After round two it is \(-423.2\), then \(-424.3\), then \(-426.5\), and by round forty it has sunk to \(-433.7\). The model gets worse every round it is given. Look at where it ends up: the centres are at \(-1.1\) and \(2.04\), nowhere near the 1 and 4 that made the archive, and one of the two kinds claims 98% of all messages.

A colleague shows her a second loop that differs from the first by one small change. On the very same archive it climbs every single round, to \(-364.0\), and lands on centres 1.18 and 3.98. There is a third oddity, too: when Ines starts both centres at the same score, her loop never pulls them apart.

Why does one sensible-sounding loop go downhill while its cousin never does? What is the one change that makes the cousin safe, and what can you promise about it? To find out you need two ideas. The first is how a generative model describes measured scores that move together, which gives a classifier you will recognise. The second is how it describes a kind of message that nobody ever observes.

## What you need from lesson 24a

This lesson leans on a few ideas from the first half. Here they are, one line each, so you can check them against what you remember.

- A **generative** model describes how each kind of message is produced by a few numbers, its **parameters**, and learning means estimating them from the training sequence.
- The **log-likelihood** of a training sequence under parameters \(\theta\) adds up, over the examples, the log of the probability (for measured values, the density) the model gave each one. The **maximum likelihood estimator** is the \(\theta\) that makes it largest. For a Gaussian, the best centre is the average of the scores and the best width is the root of the average squared distance from that average.
- Maximum likelihood is **ERM** (empirical risk minimization, from lesson 3) with the **log loss** \(-\log P_\theta[x]\), the model’s surprise at what it saw. The extra surprise caused by a wrong model is the **relative entropy**, which is never negative; its proof rested on the inequality \(\log t \le t - 1\), which we use again below.
- **Bayes’ rule** turns “how likely is this message under each kind” into “how likely is each kind given this message”: a kind’s share times its chance of writing the message, divided by the total over the kinds. The **Bayes optimal predictor** gives each message whichever label that makes more probable.
- **Naive Bayes** assumes the words are independent *given the label*, which cuts \(2^d\) parameters to \(2d + 1\).

## Another assumption: Gaussian clouds

Naive Bayes, from lesson 24a, is one generative assumption. Here is a second, for measured features, and it leads somewhere you have been before.

Suppose the mail room scores every message on two measured features: the **capitals score** \(x_1\) (how shouty the subject is) and the **link score** \(x_2\) (how many links, scaled). Plot keep messages and they form an elliptical cloud around \((1, 1)\); plot junk and they form a cloud of the same shape around \((3, 2)\). In both clouds the two scores tend to rise together: a shouty message tends to have more links.

To describe a cloud like this we need the Gaussian in \(d\) dimensions. Its centre is a vector \(\mu \in \mathbb R^d\). Its shape is a **covariance matrix** \(\Sigma\), a \(d \times d\) table: on the diagonal, the variance of each score (its spread); off the diagonal, the **covariance** of each pair of scores, positive when they tend to rise together. For our clouds,

\[ \Sigma = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix}: \]

each score has variance 1, and the two scores have covariance 0.5. The density of the Gaussian with centre \(\mu\) and covariance \(\Sigma\) is

\[ \mathbb P[X = x] = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}\exp\left(-\frac12 (x - \mu)^\top \Sigma^{-1}(x - \mu)\right). \]

- \(x - \mu\) is the offset of the message from the centre, a column vector; \((x - \mu)^\top\) is the same offset written as a row (the \(\top\) means transpose).
- \(\Sigma^{-1}\) is the inverse matrix of \(\Sigma\), the matrix with \(\Sigma^{-1}\Sigma = I\), the identity. The product \((x - \mu)^\top\Sigma^{-1}(x - \mu)\) is a single number: a squared distance in which directions where the cloud is wide count less, and directions where it is narrow count more. With \(\Sigma = I\) it is the ordinary squared distance \(\|x - \mu\|^2\).
- \(|\Sigma|\) is the determinant of \(\Sigma\), a measure of the cloud’s volume; with the factor \((2\pi)^{d/2}\) it makes the total volume under the density equal to 1.

In one sentence: **a Gaussian cloud is densest at its centre and thins out with the stretched squared distance from it.** For \(d = 1\) this is the bell of lesson 24a. At the centre of our clouds, the density is \(1/(2\pi\sqrt{0.75}) \approx 0.184\).

**Linear discriminant analysis** (LDA) makes three generative assumptions: the two labels are equally likely, \(\mathbb P[Y = 1] = \mathbb P[Y = 0] = 1/2\); given the label \(y\), the message is Gaussian with centre \(\mu_y\); and the covariance \(\Sigma\) is the same for both labels. Under these assumptions, what does the Bayes optimal predictor (the one that gives each message its more probable label) look like?

It predicts junk when \(\mathbb P[Y = 1]\,\mathbb P[X = x \mid Y = 1] > \mathbb P[Y = 0]\,\mathbb P[X = x \mid Y = 0]\), that is, when the ratio of the two sides is above 1, that is, when its logarithm is above 0. That logarithm is the **log-likelihood ratio**:

\[ \log\left(\frac{\mathbb P[Y = 1]\,\mathbb P[X = x \mid Y = 1]}{\mathbb P[Y = 0]\,\mathbb P[X = x \mid Y = 0]}\right) = \frac12(x - \mu_0)^\top\Sigma^{-1}(x - \mu_0) - \frac12(x - \mu_1)^\top\Sigma^{-1}(x - \mu_1). \]

- On the left, the prior shares are equal and cancel, and so do the normalizing constants \((2\pi)^{d/2}|\Sigma|^{1/2}\), because \(\Sigma\) is shared.
- What is left is the log of a ratio of two exponentials, which is the difference of their exponents.

In one sentence: **the evidence for junk is how much closer the message is to the junk centre than to the keep centre, in stretched distance.** For the message \(x = (3, 2)\), sitting on the junk centre, its stretched squared distance to the keep centre is \((2, 1)\Sigma^{-1}(2, 1)^\top = 4\) and to the junk centre 0, so the ratio’s log is \(\frac12 \cdot 4 = 2\): junk is \(e^2 \approx 7.4\) times as likely.

Now expand both stretched distances. Each is \(x^\top\Sigma^{-1}x - 2\mu_y^\top\Sigma^{-1}x + \mu_y^\top\Sigma^{-1}\mu_y\), where the middle term combines \(x^\top\Sigma^{-1}\mu_y\) and \(\mu_y^\top\Sigma^{-1}x\), which are equal because \(\Sigma\), and hence \(\Sigma^{-1}\), is symmetric. The \(x^\top\Sigma^{-1}x\) terms cancel between the two halves, and the log-likelihood ratio is \(\langle w, x\rangle + b\) with

\[ w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad b = \frac12\left(\mu_0^\top\Sigma^{-1}\mu_0 - \mu_1^\top\Sigma^{-1}\mu_1\right). \]

- \(\langle w, x\rangle = \sum_j w_j x_j\) is the inner product of lesson 9a: a weighted sum of the scores.
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

So a generative assumption has rederived the hypothesis class of lesson 9a, the halfspaces. The difference is in how the halfspace is chosen. LDA does not search for a good \(w\); it estimates the centres and the covariance, and computes \(w\) and \(b\) from them. The usual estimates are the maximum likelihood ones: each centre is the average of its class, and the shared covariance is the average, over all \(m\) examples, of the outer product of each example’s offset \(v = x_i - \mu_{y_i}\) from its own class centre. The outer product \(vv^\top\) is the \(d \times d\) matrix with entry \(v_j v_k\) in row \(j\), column \(k\), as in lesson 23a; for the offset \((1, 0)\) it is \(\begin{pmatrix}1 & 0\\ 0 & 0\end{pmatrix}\).

```
LDATrain(S):
  for y in {0, 1}:
    μ[y] ← average of the x_i with y_i = y
  Σ ← (1/m) · sum over i of (x_i − μ[y_i])(x_i − μ[y_i])ᵀ
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
- The first equality is the law of total probability: a message arises by some kind, so add the chances over the kinds. Each term is then a share times a Gaussian cloud of the previous section.

In one sentence: **a mixture of Gaussians is a weighted sum of bells, one per hidden kind.** At \(x = 1\), with \(c = (0.7, 0.3)\), centres 1 and 4 and width 1: \(0.7 \cdot 0.399 + 0.3 \cdot 0.0044 \approx 0.281\).

Let \(\theta\) collect all the parameters (the shares, centres and covariances). For an i.i.d. sample \(S = (x_1, \dots, x_m)\), the log-likelihood (the sum of the log-probabilities from lesson 24a) is

\[ L(\theta) = \sum_{i=1}^{m}\log \mathbb P_\theta[X = x_i] = \sum_{i=1}^{m}\log\left(\sum_{y=1}^{k}\mathbb P_\theta[X = x_i, Y = y]\right). \]

- \(\mathbb P_\theta[X = x_i, Y = y]\) is the joint probability that the kind is \(y\) and the message is \(x_i\): \(c_y\) times the \(y\)-th bell at \(x_i\).
- The outer sum adds one log per example; the inner sum, inside the log, adds over the kinds.

In one sentence: **the log-likelihood adds, over messages, the log of the total chance of each message over all the kinds that might have produced it.** For the single message \(x = 1\) in the mixture above, \(L = \log 0.281 \approx -1.27\). Maximum likelihood wants the \(\theta\) that maximizes it.

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

An iterative algorithm with no guarantee is a hope. EM comes with one: the log-likelihood never decreases. The proof is short once we see EM as climbing a different function, one coordinate at a time. The plan: build a function \(G\) of the beliefs and the parameters that is never above the log-likelihood, touches it whenever the beliefs are the E step’s, and goes up at every step. Then the log-likelihood, which \(G\) touches after every E step, must go up too.

Define, with the convention \(0 \log 0 = 0\),

\[ G(Q, \theta) = F(Q, \theta) - \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\log Q_{i,y} = \sum_{i=1}^{m}\sum_{y=1}^{k} Q_{i,y}\log\frac{\mathbb P_\theta[X = x_i, Y = y]}{Q_{i,y}}. \]

- The subtracted double sum is minus the sum of the entropies of the rows of \(Q\): a bonus for soft assignments that are uncertain. (The entropy of a row is its average surprise, as in lesson 24a: 0 for a row that is certain, largest for a row that is spread evenly.)
- The second form combines the two logs into one; terms with \(Q_{i,y} = 0\) are left out.

In one sentence: **\(G\) is the expected complete log-likelihood plus the entropy of the beliefs.** Let \(\mathcal Q\) be the set of \(m \times k\) matrices with entries in \([0, 1]\) whose rows add up to 1, the allowed soft assignments.

**Lemma (EM is alternate maximization).** The two EM steps are \(Q^{(t+1)} \in \operatorname*{argmax}_{Q \in \mathcal Q} G(Q, \theta^{(t)})\) and \(\theta^{(t+1)} \in \operatorname*{argmax}_\theta G(Q^{(t+1)}, \theta)\). Moreover, \(G(Q^{(t+1)}, \theta^{(t)}) = L(\theta^{(t)})\).

*Proof.* The M step first. The entropy term of \(G\) does not involve \(\theta\), so for fixed \(Q^{(t+1)}\), maximizing \(G\) over \(\theta\) is the same as maximizing \(F\). That is the M step.

The E step. Fix any \(\theta\) and any \(Q \in \mathcal Q\). We show \(G(Q, \theta) \le L(\theta)\), and then that the E step’s choice reaches it.

For the inequality, fix a message \(i\) and write \(a_y = \mathbb P_\theta[X = x_i, Y = y]\) and \(A = \sum_y a_y = \mathbb P_\theta[X = x_i]\). Using \(\log t \le t - 1\) from the relative-entropy proof of lesson 24a, with \(t = a_y/(Q_{i,y}A)\), on the kinds with \(Q_{i,y} > 0\):

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

**E step.** By Bayes’ rule (lesson 24a’s identity from “how likely is this message under each kind” to “how likely is each kind given this message”), the probability of kind \(y\) given message \(x_i\) is the joint probability divided by the total over kinds. The Gaussian normalizing constant \((2\pi)^{d/2}\) is the same for every kind and cancels:

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

## Poking at both loops

You now hold both loops: Ines’s, which gives each message wholly to one kind, and soft k-means, which gives each message to both kinds in proportion. They differ in a single line. The lab below runs either one on a fresh archive of 200 length scores, from a starting guess that you choose, and draws two things: the log-likelihood after every round as a line, where a climbing line is a model that is improving, and the two fitted bells over the scores. “Resample” draws a new archive, and “Pin this” keeps one run on screen, dashed, so that you can lay a second run beside it.

Make the prediction first, then count: how often does the line dip, and which loop never dips?

[[lab]]

On 200 archives drawn like this one, with the starting guess of the story (centres 0 and 0.5, shares one half each), Ines’s loop dipped at least once on 133 of them, and the soft loop on none. Started with both centres at the same score, neither loop ever separated the kinds. The next section explains both facts.

## Why Ines’s loop went downhill

Let’s put the two loops side by side in code, on the same archive as the soft k-means run above (140 personal notes around 1, 60 newsletters around 4). The only difference is one flag that turns each row of \(Q\) into an all-or-nothing row: 1 on the kind the message finds more probable, 0 on the other.

```python
import math, random
rng = random.Random(4)
xs = [rng.gauss(1, 1) for _ in range(140)]
xs += [rng.gauss(4, 1) for _ in range(60)]
def joint(x, y, c, mu):
    g = math.exp(-(x - mu[y])**2 / 2) / math.sqrt(2 * math.pi)
    return c[y] * g
def loop(hard):
    c, mu, lls = [0.5, 0.5], [0.0, 0.5], []
    for t in range(40):
        tot = [joint(x, 0, c, mu) + joint(x, 1, c, mu)
               for x in xs]
        lls.append(sum(math.log(v) for v in tot))
        Q = [[joint(x, y, c, mu) / v for y in (0, 1)]
             for x, v in zip(xs, tot)]
        if hard:
            Q = [[float(q[0] >= q[1]), float(q[0] < q[1])]
                 for q in Q]
        n = [sum(q[y] for q in Q) for y in (0, 1)]
        mu = [sum(q[y] * x for q, x in zip(Q, xs)) / n[y]
              for y in (0, 1)]
        c = [n[y] / len(xs) for y in (0, 1)]
    return lls, mu, c
for name, hard in (("soft", False), ("hard", True)):
    lls, mu, c = loop(hard)
    print(name, [round(v, 1) for v in lls[:5]], round(lls[-1], 1))
    print("   ", [round(v, 2) for v in mu],
          [round(v, 2) for v in c])
```

It prints:

```
soft [-711.7, -414.6, -396.7, -387.4, -380.7] -364.0
    [1.18, 3.98] [0.72, 0.28]
hard [-711.7, -422.2, -423.2, -424.3, -426.5] -433.7
    [-1.1, 2.04] [0.02, 0.98]
```

The soft loop climbs from \(-711.7\) to \(-364.0\) and lands near the truth. Ines’s loop jumps to \(-422.2\) in the first round, and from then on every round loses ground until it settles at \(-433.7\), with centres \(-1.1\) and \(2.04\) and a kind that holds 2% of the messages.

[[diagram:downhill]]

**Which link of the proof breaks?** Go back to the theorem that EM never decreases the likelihood. Its proof was a chain of two equalities and two inequalities, and the equalities needed \(Q\) to be the E step’s posterior: only then does \(G\) touch the log-likelihood \(L\). A hard assignment is not the posterior, unless the posterior is already certain for every message, and it leaves a gap that we can measure exactly. For the all-or-nothing matrix \(Q^{\mathrm{hard}}\) built from the current parameters \(\theta\),

\[ L(\theta) = G(Q^{\mathrm{hard}}, \theta) + \tau(\theta), \qquad \tau(\theta) = \sum_{i=1}^{m} \log\frac{1}{\max_{y}\,\mathbb P_\theta[Y = y \mid X = x_i]}. \]

- \(Q^{\mathrm{hard}}\) has, in row \(i\), a 1 at the kind that \(\theta\) finds most probable for message \(x_i\) and a 0 at every other kind (a tie goes to the first kind).
- \(\max_y \mathbb P_\theta[Y = y \mid X = x_i]\) is the probability the model gives to its best guess for message \(i\): 1 when the model is certain, one half when two kinds tie.
- \(\log\frac{1}{\cdot}\) of that is the model’s surprise at its own best guess: 0 when it is certain, \(\log 2 \approx 0.693\) when two kinds tie. The sum \(\tau\) adds these up over the messages, and we call it the **rounding tax**; it is never negative.
- The equation holds because a row with a single 1 has no entropy, so message \(i\)’s part of \(G\) is just \(\log\mathbb P_\theta[X = x_i, Y = \hat y_i]\), where \(\hat y_i\) is the kind it was given. By the definition of conditional probability that equals \(\log\mathbb P_\theta[X = x_i] + \log\max_y\mathbb P_\theta[Y = y \mid X = x_i]\), the message’s part of \(L\) minus its part of the tax.

In one sentence: **the log-likelihood is what \(G\) records for the all-or-nothing assignment, plus a rounding tax: for each message, the log of one over how sure the model was of its best guess.** Take one message with length score 2.5, shares \((0.7, 0.3)\) and centres 1 and 4. It sits exactly between them, so the posterior is \((0.7, 0.3)\), the tax is \(\log(1/0.7) = 0.357\), and the log-likelihood \(-2.044\) exceeds the all-or-nothing \(G\), \(-2.401\), by exactly that.

```python
import math
c, mu, x = (0.7, 0.3), (1.0, 4.0), 2.5
def bell(x, m):
    return math.exp(-(x - m)**2 / 2) / math.sqrt(2 * math.pi)
a = [c[y] * bell(x, mu[y]) for y in (0, 1)]
post = [v / sum(a) for v in a]
L = math.log(sum(a))
G_hard = math.log(a[0])
print(round(L, 3), round(G_hard, 3))
print(round(math.log(1 / max(post)), 3), round(L - G_hard, 3))
```

It prints `-2.044 -2.401` and `0.357 0.357`.

With the soft assignment the tax is zero, because the posterior makes \(G\) equal to \(L\) exactly. That is the E step’s whole job. Now redo the proof’s chain for a hard round. The M step still maximizes \(G(Q^{\mathrm{hard}}, \cdot)\), so it raises \(G\) by some gain \(\gamma \ge 0\) (it can always keep the old parameters). And the inequality \(G \le L\) held for every matrix \(Q\), hard ones included. So

\[ L(\theta^{(t+1)}) \;\ge\; G(Q^{\mathrm{hard}}, \theta^{(t+1)}) \;=\; G(Q^{\mathrm{hard}}, \theta^{(t)}) + \gamma \;=\; L(\theta^{(t)}) - \tau(\theta^{(t)}) + \gamma. \]

- The first step is the inequality \(G \le L\), at the new parameters.
- The second is the definition of the gain \(\gamma\): the amount by which the M step raised \(G\).
- The third is the display above, at the old parameters.

In one sentence: **after a hard round the log-likelihood is guaranteed to be at least where it was, minus the rounding tax, plus what the M step gained, so it is guaranteed not to fall only when the gain covers the tax.** With a tax of 2 nats and a gain of 0.5, the promise is only \(L - 1.5\): the likelihood may fall by as much as 1.5 nats, or may rise. Here are the real numbers for round two of Ines’s loop on the archive:

```python
import math, random
rng = random.Random(4)
xs = [rng.gauss(1, 1) for _ in range(140)]
xs += [rng.gauss(4, 1) for _ in range(60)]
def joint(x, y, c, mu):
    g = math.exp(-(x - mu[y])**2 / 2) / math.sqrt(2 * math.pi)
    return c[y] * g
def hard_round(c, mu):
    hat = [int(joint(x, 1, c, mu) > joint(x, 0, c, mu))
           for x in xs]
    n = [hat.count(0), hat.count(1)]
    mu = [sum(x for x, h in zip(xs, hat) if h == y) / n[y]
          for y in (0, 1)]
    return [n[0] / len(xs), n[1] / len(xs)], mu, hat
def L(c, mu):
    return sum(math.log(joint(x, 0, c, mu) + joint(x, 1, c, mu))
               for x in xs)
def G(hat, c, mu):
    return sum(math.log(joint(x, h, c, mu))
               for x, h in zip(xs, hat))
c1, mu1, _ = hard_round([0.5, 0.5], [0.0, 0.5])
c2, mu2, hat = hard_round(c1, mu1)
tax = L(c1, mu1) - G(hat, c1, mu1)
gain = G(hat, c2, mu2) - G(hat, c1, mu1)
print(round(L(c1, mu1), 2), round(tax, 2), round(gain, 2))
print(round(L(c1, mu1) - tax + gain, 2), round(L(c2, mu2), 2))
```

It prints `-422.15 22.37 0.17` and `-444.35 -423.21`. Before round two the log-likelihood was \(-422.15\); the tax was 22.37 nats and the M step won back only 0.17; so the chain promises no better than \(-444.35\), which is no promise at all. The real value, \(-423.21\), is a fall of 1.06 nats, comfortably inside what the chain allows. The chain does not say how far a hard loop will fall. It only withdraws the guarantee. On this archive every round from the second to the ninth lost ground, and after that the loop stood still.

**The third oddity.** Why did the loop with both centres at the same score never separate them? Call that common score \(s\). Then both bells give every message the same density, so in the E step the bells cancel in Bayes’ rule, and each message’s posterior is simply the shares: \(Q_{i,y} = c_y\) for every message \(i\). The M step then moves each centre to \(\sum_i c_y x_i / \sum_i c_y\), which is the plain average of all the scores (the share \(c_y\) factors out of top and bottom), and each share to \(\sum_i c_y / m = c_y\), unchanged. Both centres land on the overall average, still on top of each other, and the shares never move: a fixed point. **A start where the kinds cannot be told apart gives every message the same belief, and the refit then treats the kinds identically.** Here it is on the archive, starting at 2.0 with shares \((0.6, 0.4)\):

```python
import math, random
rng = random.Random(4)
xs = [rng.gauss(1, 1) for _ in range(140)]
xs += [rng.gauss(4, 1) for _ in range(60)]
c, mu = [0.6, 0.4], [2.0, 2.0]
def joint(x, y):
    g = math.exp(-(x - mu[y])**2 / 2) / math.sqrt(2 * math.pi)
    return c[y] * g
for t in range(3):
    tot = [joint(x, 0) + joint(x, 1) for x in xs]
    print(round(sum(math.log(v) for v in tot), 2),
          [round(v, 2) for v in mu], [round(v, 2) for v in c])
    Q = [[joint(x, y) / v for y in (0, 1)]
         for x, v in zip(xs, tot)]
    n = [sum(q[y] for q in Q) for y in (0, 1)]
    mu = [sum(q[y] * x for q, x in zip(Q, xs)) / n[y]
          for y in (0, 1)]
    c = [n[y] / len(xs) for y in (0, 1)]
```

It prints:

```
-438.12 [2.0, 2.0] [0.6, 0.4]
-438.05 [1.98, 1.98] [0.6, 0.4]
-438.05 [1.98, 1.98] [0.6, 0.4]
```

The log-likelihood rises by 0.07 in the first round, as both centres move from 2.0 to the average of all scores, 1.98, and then it is flat for good. The theorem is not violated, since the log-likelihood never fell. It simply promised nothing about where the climb ends.

## Fix the bug

Ines has the safe loop, and the product team wants to ship it on the live archive. Your manager will accept it only with a certificate, held to this lesson’s theorems and not to this week’s luck. It has three parts:

1. **Never worse:** every round’s E step gives each message its exact posterior over the two kinds, and every round’s M step leaves the expected log-likelihood \(F\) at least as high as it found it. By this lesson’s theorem the archive’s log-likelihood then never falls, on any archive.
2. **Settled:** within a budget of 40 rounds, the last round raises the log-likelihood by less than 0.01 nats, on each of 40 test archives.
3. **A real fit:** on each of the 40 test archives, the final log-likelihood is at most 1 nat below that of the true shares and centres. (An honest fit scores above the truth, and the test archives come from a generator whose truth is known.)

The 40 test archives are 200-score archives like the one above, drawn with seeds 0 to 39, and every run starts from the same guess: centres 0 and 0.5, shares one half each.

You choose three things. The **E step**: the exact posterior, Ines’s hard assignment, or the posterior computed as if both shares were one half. The **M step**: the exact maximizer; a half-way step, which moves every parameter half the distance from its old value to the maximizer; refitting only the centres and keeping the shares; or the exact maximizer plus a random jitter of up to 0.3 on each centre. And the number of rounds \(T\), at most 40. Here are the three parts as code, for some of the settings, so that you can work out which to try before you open the lab:

```python
import math, random
def bell(x, m):
    return math.exp(-(x - m)**2 / 2) / math.sqrt(2 * math.pi)
def archive(seed):
    rng = random.Random(seed)
    xs = [rng.gauss(1, 1) for _ in range(140)]
    return xs + [rng.gauss(4, 1) for _ in range(60)]
def L(xs, c, mu):
    f = lambda x: sum(c[y] * bell(x, mu[y]) for y in (0, 1))
    return sum(math.log(f(x)) for x in xs)
def run(xs, soft, half, share, T):
    c, mu, lls = [0.5, 0.5], [0.0, 0.5], []
    for t in range(T + 1):
        lls.append(L(xs, c, mu))
        Q = [[c[y] * bell(x, mu[y]) for y in (0, 1)] for x in xs]
        Q = [[v / sum(q) for v in q] for q in Q]
        if not soft:
            Q = [[float(q[0] >= q[1]), float(q[0] < q[1])]
                 for q in Q]
        n = [sum(q[y] for q in Q) for y in (0, 1)]
        new_mu = [sum(q[y] * x for q, x in zip(Q, xs)) / n[y]
                  for y in (0, 1)]
        new_c = [v / len(xs) for v in n] if share else c
        w = 0.5 if half else 1.0
        mu = [(1 - w) * a + w * b for a, b in zip(mu, new_mu)]
        c = [(1 - w) * a + w * b for a, b in zip(c, new_c)]
    return lls
def report(name, soft, half, share, T):
    falls, gain, fit = 0, 0.0, 1e9
    for seed in range(40):
        xs = archive(seed)
        lls = run(xs, soft, half, share, T)
        falls += any(b < a - 1e-9 for a, b in zip(lls, lls[1:]))
        gain = max(gain, lls[-1] - lls[-2])
        true = L(xs, [0.7, 0.3], [1, 4])
        fit = min(fit, lls[-1] - true)
    print(name, T, falls, round(gain, 4), round(fit, 2))
report("soft+exact", True, False, True, 18)
report("soft+exact", True, False, True, 19)
report("soft+half", True, True, True, 36)
report("soft+half", True, True, True, 37)
report("soft+centres", True, False, False, 40)
report("hard+exact", False, False, True, 40)
```

Each row names the E step and the M step, then gives \(T\), how many of the 40 archives had a round where the log-likelihood fell, the worst last-round gain, and the worst fit (final log-likelihood minus the truth’s):

```
soft+exact 18 0 0.0129 0.05
soft+exact 19 0 0.0085 0.05
soft+half 36 0 0.0123 0.05
soft+half 37 0 0.0092 0.05
soft+centres 40 0 0.0 -15.5
hard+exact 40 22 0.0 -104.97
```

Read the rows before you open the lab. The exact steps need 19 rounds to settle on every archive, and 18 are not enough. The half-way step never lets the likelihood fall, but it is slow: it needs 37 rounds. Refitting only the centres is safe and settles quickly, and it fits 15.5 nats worse than the truth, because the shares can never move. And a hard assignment fails the first part on 22 of the 40 archives.

In the lab, each round draws a fresh archive for the readouts under the certificate. A setting that happens to climb on this round’s archive does not pass; only a setting that meets all three parts on the 40 test archives does.

[[lab-act2]]

## What people get wrong

**“EM finds the maximum likelihood estimate.”** The theorem only says it never goes downhill. It can stop at a local maximum, and a bad start (two identical centres, for instance) can keep it there forever.

**“Giving each message wholly to its most probable kind is just a cheaper version of the same loop.”** It is a different loop, and the theorem does not cover it. The soft assignment is what makes \(G\) touch the log-likelihood after every E step; an all-or-nothing assignment leaves a rounding tax, up to \(\log 2\) nats per message for two kinds. On the archive of this lesson that loop fell from \(-422.2\) to \(-433.7\) while the soft loop climbed to \(-364.0\).

**“LDA weighs each score by how far apart its averages are.”** The weights are \(\Sigma^{-1}(\mu_1 - \mu_0)\), not \(\mu_1 - \mu_0\). In this lesson’s clouds junk has an average link score one higher than keep, and the link score still gets weight 0, because the capitals score already explains it.

## Recap and the next bug

1. A Gaussian cloud in \(d\) dimensions has a centre \(\mu\) and a **covariance matrix** \(\Sigma\); its density thins out with the stretched squared distance \((x - \mu)^\top\Sigma^{-1}(x - \mu)\).
2. **LDA** assumes equally likely labels, Gaussian classes and one shared covariance. The Bayes optimal predictor is then a halfspace, with \(w = \Sigma^{-1}(\mu_1 - \mu_0)\) and \(b = -\frac12\langle w, \mu_0 + \mu_1\rangle\). It is trained from class averages and a pooled covariance in \(O(md + md^2 + d^3)\) time and predicts in \(O(d)\).
3. A **latent variable** is part of the production story that is never observed. A **mixture of Gaussians** is a weighted sum of bells, one per hidden kind, and its log-likelihood has a sum inside the log, so it has no closed-form maximum.
4. **EM** alternates an E step (the soft assignment, the posterior of each kind given each message) and an M step (refit the parameters to maximize \(F\)). It is alternate maximization of \(G\), and \(G\) touches the log-likelihood after every E step, so the log-likelihood never decreases. It can stop at a local maximum, and identical centres never separate.
5. With identity covariances, EM is **soft k-means**: weighted averages for the centres and soft counts for the shares, \(O(mkd)\) per iteration.
6. Replace the soft assignment by a hard one and \(G\) no longer touches the log-likelihood: \(L = G + \tau\), the rounding tax. A round then promises only \(L - \tau + \gamma\), which is why Ines’s loop went downhill.
7. A loop worth shipping needs an exact E step, an M step that never lowers \(F\), a log-likelihood that has settled within budget, and a fit close to the truth. Never getting worse is not the same as being right: freezing the shares is safe, and wrong.

Ines’s filter now ships with a certificate, and it is fast. But look at what it is made of. It keeps a parameter for every one of the 200 words in the dictionary, and the real dictionary has tens of thousands. Most of those words are noise: they appear about as often in junk as in keep, and every one of them adds a little random evidence to every decision, and two more numbers to estimate. And some of the most useful signals are not words at all: the ratio of links to text, a sender’s history, “prize” and “claim” appearing together. Which features should the filter keep, and could it build better ones than the ones it was given? That is the next lesson: feature selection and generation.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof that EM never decreases the likelihood, the chain has two equalities and two inequalities. Suppose your M step is only approximate: it returns some \(\theta^{(t+1)}\) with \(F(Q^{(t+1)}, \theta^{(t+1)}) \ge F(Q^{(t+1)}, \theta^{(t)})\), but not the maximizer. Does the theorem still hold? Which link of the chain changes, and why is it still valid? What if instead the E step were approximate?
2. A colleague fits the two clouds of this lesson but ignores the correlation between the scores: she treats the capitals score and the link score as independent bells of variance 1, which is the same as putting \(\Sigma = I\) in the LDA formulas. (a) What \(w\) and \(b\) does she get? (b) A message has capitals score 2.3 and link score 0.5. What does each rule say about it, and which one agrees with the Bayes optimal predictor under the true clouds? (c) Without computing anything, which rule do you expect to make more mistakes, and what does the weight that LDA gives the link score tell you about why?
3. A round of Ines’s loop has a rounding tax of 2 nats, and its M step gains 0.5 nats in \(G\). What can you say about the change in the log-likelihood? What if the tax were 0.3 nats? And why can a round of the soft loop never pay any tax?

## Problems

Once the concept check is done, try this one. Hints come one at a time in the chat, and no full solution.

- **Problem 1 (the shares of soft k-means).** For nonnegative weights \(\nu_1, \dots, \nu_k\), not all zero, consider maximizing \(\sum_{y=1}^k \nu_y\log c_y\) over probability vectors \(c\) with every \(c_y > 0\). (i) Check that the share part of the soft k-means M step has this form, with \(\nu_y = \sum_i Q_{i,y}\). (ii) Show that \(c^\star = \nu/\sum_y \nu_y\) is a probability vector. (iii) Show that the problem is equivalent to minimizing the relative entropy \(D_{\mathrm{RE}}(c^\star\|c)\) over the same \(c\). (iv) Conclude, using a property of the relative entropy proved in lesson 24a, that \(c^\star\) is the solution.
