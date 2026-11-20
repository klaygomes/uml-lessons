# Description Length and Occam’s Razor

@@HEAD@@

## Lists by the bit: minimum description length

SRM spreads the budget over classes. There is a sharper version: spread it over individual hypotheses. Our \(\mathcal H\) is countable (each list is a finite object), so we can list its members as \(h_1, h_2, \dots\) and write it as a union of one-member classes, \(\mathcal H = \bigcup_n \{h_n\}\). Then each hypothesis gets its own weight \(w(h)\), say 1/128 for the list “unknown sender, junk; otherwise keep”.

A single fixed hypothesis has uniform convergence with no union bound at all. Its mistakes are \(\mathbb 1[h(x_i) \ne y_i]\), where the **indicator** \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false. These are i.i.d. numbers in \([0, 1]\) (the numbers from 0 to 1, ends included) with mean \(L_{\mathcal D}(h)\), so by Hoeffding’s inequality (lesson 4), \(\mathbb P[|L_S(h) - L_{\mathcal D}(h)| > \varepsilon] \le 2e^{-2m\varepsilon^2}\). Here \(\mathbb P\) is the probability over the training sequence, and \(e^{-2m\varepsilon^2}\) is \(e \approx 2.718\) raised to the power \(-2m\varepsilon^2\), which shrinks very fast as \(m\) grows. In plain words: **the chance that one fixed list’s score on \(m\) labels is off by more than \(\varepsilon\) falls exponentially in \(m\varepsilon^2\).** At \(m = 300\) and \(\varepsilon = 0.1\) it is at most \(2e^{-6} = 0.005\). Setting this to \(\delta\) and solving for \(\varepsilon\) gives the gap \(\sqrt{\log(2/\delta)/(2m)}\), which at \(\delta = 0.05\) is \(\sqrt{3.689/600} = 0.078\):

```python
import math

m, eps, delta = 300, 0.1, 0.05
print(round(2 * math.exp(-2 * m * eps**2), 5))
print(round(math.sqrt(math.log(2 / delta) / (2 * m)), 4))
```

It prints `0.00496` and `0.0784`. Giving the list \(h\) the share \(w(h)\delta\) instead of \(\delta\) turns \(\log(2/\delta)\) into \(\log(2/\delta) - \log w(h)\), and the SRM rule becomes

\[ \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{-\log w(h) + \log(2/\delta)}{2m}} \bigg]. \]

Read it from the inside out. A weight is below 1, so its logarithm is negative and \(-\log w(h) = \log(1/w(h))\) is a positive cost: the smaller the weight, the bigger the cost. Add the confidence cost \(\log(2/\delta)\), divide by \(2m\), take the square root, and add the empirical risk; the argmin picks the lists where that total is smallest. In plain words: **return a list with the smallest empirical risk plus a gap that grows as its weight shrinks.** With weight 1/128 the cost is \(\log 128 = 4.852\); at 100 labels that list has empirical risk 0.17, so its bracket is \(0.17 + \sqrt{(4.852 + 3.689)/200} = 0.3767\).

```python
import math

LS, m, delta = 0.17, 100, 0.05
w = 2.0 ** -7
inside = -math.log(w) + math.log(2 / delta)
print(round(-math.log(w), 3), round(math.log(2 / delta), 3))
print(round(LS + math.sqrt(inside / (2 * m)), 4))
```

It prints `4.852 3.689` and `0.3767`.

Now the prior knowledge is entirely in the weights. Where do good weights come from? From writing hypotheses down.

**Description languages.** Fix the alphabet \(\Sigma = \{0, 1\}\) (the capital Greek sigma names the set of symbols we may write). A **string** is a finite sequence of symbols, such as \(\sigma = (0, 1, 1, 1, 0)\), of length \(|\sigma| = 5\): around a string, the bars mean its length, a third meaning after “how many” and “absolute value”. The set of all strings, of every finite length, is \(\Sigma^*\), the star meaning “any number of symbols”. A **description language** for \(\mathcal H\) is a function \(d: \mathcal H \to \Sigma^*\) that maps each hypothesis \(h\) to a string \(d(h)\), its description. Its length is written \(|h|\), the **description length** of \(h\). The language is **prefix-free** if for every two distinct \(h, h'\), \(d(h)\) is not a prefix of \(d(h')\): no description is exactly the first \(|h|\) symbols of a longer one.

Here is a prefix-free language for decision lists. First, a code for feature numbers: write \(j\) in binary, with \(b\) digits, and put \(b - 1\) zeros in front. So 1 is `1`, 2 is `010`, 3 is `011`, 4 is `00100`, 5 is `00101`, and feature \(j\) costs \(2\lfloor \log_2 j \rfloor + 1\) bits. Read that inside out: \(\log_2 j\) is the power of 2 that gives \(j\), the brackets \(\lfloor \cdot \rfloor\) round it down to a whole number, and that whole number plus 1 is how many binary digits \(j\) has; doubling it and adding 1 counts the digits plus the leading zeros. In plain words: **a feature number costs twice its binary digit count, minus one.** For \(j = 5\), \(\log_2 5 = 2.32\), rounded down to 2, so \(2 \times 2 + 1 = 5\) bits, and indeed `00101` has five. A reader can always tell where a feature number ends: count the leading zeros, then read one more digit than that. Then a list is written test by test: a `1` meaning “a test follows”, the feature’s code, and the test’s label bit; and at the end a `0` meaning “no more tests”, followed by the default label bit.

The list “unknown sender, junk; otherwise keep”, \(((3, 1); 0)\), is written `1 011 1 0 0`, which is 7 bits. “Everything is junk”, \((();1)\), is `01`, which is 2 bits. The three-test list from the opening takes 15 bits and the five-test list \(f\) takes 29.

```python
import math

def gamma(j):
    b = bin(j)[2:]
    return "0" * (len(b) - 1) + b

def code(tests, default):
    body = "".join("1" + gamma(j) + str(b) for j, b in tests)
    return body + "0" + str(default)

for j in range(1, 6):
    print(j, gamma(j), 2 * math.floor(math.log2(j)) + 1)
f = [(3, 1), (1, 0), (2, 1), (5, 0), (4, 1)]
print(code([(3, 1)], 0), code([], 1))
print(len(code(f[:3], 0)), len(code(f, 0)))
```

It prints the five feature codes with lengths 1, 3, 3, 5, 5, matching the formula, then `1011100 01` and `15 29`. `bin(j)[2:]` is \(j\) in binary, and `gamma` puts the zeros in front.

Why is this prefix-free? Read any codeword from the left, the way a decoder would. Each flag bit says whether a test follows, each feature code announces its own length, and after the end flag exactly one bit remains. So the decoder knows where the codeword ends from the symbols alone. If one codeword were a prefix of another, longer one, decoding the longer one would reach that point and stop, so the longer one could have nothing after it, which is a contradiction.

Now give each codeword the weight 1 over 2 to the power of its length: “everything is junk”, 2 bits, gets 1/4; the 7-bit list gets 1/128. Can a prefix-free code overspend, so that these weights add to more than 1? Try the small prefix-free set `0`, `10`, `110`, `111`: \(1/2 + 1/4 + 1/8 + 1/8 = 1\), exactly the whole budget. Shorten any word and you create a prefix. That is the general rule. (In this lemma \(S\) names a set of strings, not a training sequence.)

**Lemma (Kraft inequality).** If \(S \subseteq \{0, 1\}^*\) is a prefix-free set of strings, then

\[ \sum_{\sigma \in S} \frac{1}{2^{|\sigma|}} \le 1. \]

Read it from the inside out: \(|\sigma|\) is the length of the string \(\sigma\), \(1/2^{|\sigma|}\) halves once per symbol, and \(\sum_{\sigma \in S}\) adds that weight over every string in the set. In plain words: **in a prefix-free code, the weights 1/2 per symbol, multiplied along each codeword, add up to at most 1.** Drop prefix-freeness and it fails: `0`, `01`, `1` has `0` as a prefix of `01`, and its sum is \(1/2 + 1/4 + 1/2 = 1.25\).

```python
S = ["0", "10", "110", "111"]
clash = any(a != b and b.startswith(a) for a in S for b in S)
print(clash, sum(1 / 2 ** len(s) for s in S))
T = ["0", "01", "1"]
print(sum(1 / 2 ** len(s) for s in T))
```

It prints `False 1.0` and `1.25`.

*Proof.* Start with a finite \(S\), and let \(L\) be the length of its longest string. Toss a fair coin \(L\) times, with faces 0 and 1. For each \(\sigma \in S\), let \(E_\sigma\) be the event “the first \(|\sigma|\) tosses spell out \(\sigma\)”. The tosses are independent and each matches its symbol with probability 1/2, so \(\mathbb P[E_\sigma] = 2^{-|\sigma|}\).

Now the key step. For two distinct \(\sigma, \sigma' \in S\), the events \(E_\sigma\) and \(E_{\sigma'}\) cannot both happen: if they did, both strings would be beginnings of the same toss sequence, so the shorter one would be a prefix of the longer one (or they would be equal), which prefix-freeness forbids. So the events are pairwise disjoint, and the probability of their union is the sum of their probabilities. A probability is at most 1:

\[ \sum_{\sigma \in S} 2^{-|\sigma|} = \mathbb P\Big[\bigcup_{\sigma \in S} E_\sigma\Big] \le 1. \]

This is the lemma’s sum, written with a negative exponent, equal to the chance that some codeword is spelled out by the tosses. For `0`, `10`, `110`, `111`, the toss sequences starting 0, 10, 110 and 111 cover every outcome exactly once, so the chance is 1.

For an infinite \(S\), every finite subset is prefix-free too, so every partial sum of the series is at most 1, and so is the series. \(\blacksquare\)

[[diagram:kraft-tree]]

Let’s check it on the list code: every list with at most two tests over features 1 to 8. The `gamma` and `code` functions are the ones from the code-length snippet above.

```python
import itertools

def gamma(j):
    b = bin(j)[2:]
    return "0" * (len(b) - 1) + b

def code(tests, default):
    body = "".join("1" + gamma(j) + str(b) for j, b in tests)
    return body + "0" + str(default)

T = [(j, b) for j in range(1, 9) for b in (0, 1)]
C = [code(t, d) for k in range(3)
     for t in itertools.product(T, repeat=k) for d in (0, 1)]
clash = any(a != b and b.startswith(a) for a in C for b in C)
print(len(C), clash, sum(2 ** -len(c) for c in C))
```

It prints `546 False 0.8181228637695312`: 546 codewords, no codeword is a prefix of another, and their Kraft sum is about 0.818. Include more features and more tests and the sum creeps up towards 1, and over every list it is exactly 1. Here is why, in three sums. First, the feature code: \(\sum_j 2^{-(2\lfloor \log_2 j \rfloor + 1)} = \sum_{k \ge 0} 2^k \cdot 2^{-(2k+1)} = 1\). The left sum runs over every feature \(j\); the middle one groups the features by \(k + 1\), their number of binary digits. There are \(2^k\) numbers with \(k + 1\) digits (feature 1 alone, then 2 and 3, then 4 to 7), each costing \(2k + 1\) bits, so each group adds \(2^k / 2^{2k+1} = 1/2^{k+1}\): \(1/2 + 1/4 + 1/8 + \cdots = 1\). Second, one test, with its flag bit, its feature code and its label bit, contributes a factor \(2 \cdot \frac14 \cdot 1 = \frac12\): two choices of label, \(\frac14\) for the two single bits, and 1 for the feature code. The end, a flag and a default bit, contributes \(2 \cdot \frac14 = \frac12\) too. Third, a list with \(k\) tests gets \((1/2)^k \cdot \frac12\), and \(\sum_{k \ge 0} (1/2)^k \cdot \frac12 = 1\). In plain words: **the code spends exactly the whole budget: no prefix-free code could make lists any cheaper overall.**

```python
import math

feature = sum(2.0 ** -(2 * math.floor(math.log2(j)) + 1)
              for j in range(1, 2**20))
one_test = 2 * feature * 0.25
total = sum(one_test**k * 0.5 for k in range(200))
print(round(feature, 6), round(one_test, 6), round(total, 6))
```

It prints `0.999999 0.5 0.999999`: features up to about a million already reach 1 within a millionth.

So any prefix-free language hands you a weight function for free: \(w(h) = 2^{-|h|}\). Kraft says it sums to at most 1.

**Theorem (the MDL bound).** Let \(\mathcal H\) be a hypothesis class and \(d: \mathcal H \to \{0, 1\}^*\) a prefix-free description language for it. Then for every sample size \(m\), every \(\delta \in (0, 1)\) and every distribution \(\mathcal D\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ \forall h \in \mathcal H, \quad L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}}. \]

It is the per-hypothesis SRM bound with the weight cost \(-\log w(h)\) replaced by \(|h|\), the description length in bits. In plain words: **except with probability \(\delta\), every list’s true risk is at most its empirical risk plus a gap that grows with the square root of its length in bits.** The 7-bit list at 100 labels, empirical risk 0.17: \(0.17 + \sqrt{(7 + 3.689)/200} = 0.17 + 0.231 = 0.401\). The snippet after the MDL rule below computes this bound as `mdl_bound`.

*Proof.* Distinct hypotheses get distinct descriptions (a string is a prefix of itself, so equal descriptions would break prefix-freeness), so \(\{d(h) : h \in \mathcal H\}\) is a prefix-free set, and the Kraft inequality gives \(\sum_h 2^{-|h|} \le 1\). That also makes \(\mathcal H\) countable, since there are only countably many strings. List its members as \(h_1, h_2, \dots\), take \(\mathcal H_n = \{h_n\}\) and \(w(n) = 2^{-|h_n|}\). Each singleton has the uniform convergence property with gap \(\varepsilon_n(m, \delta) = \sqrt{\log(2/\delta)/(2m)}\), by Hoeffding as above. The SRM bound theorem then says that with probability at least \(1 - \delta\), every \(h\) has

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{\log(2^{|h|}) + \log(2/\delta)}{2m}}. \]

Here \(\log(2^{|h|})\) is \(\log(1/w(n))\), the weight cost from before. Finally, \(\log(2^{|h|}) = |h| \log 2 < |h|\), because \(\log 2 \approx 0.693\), and making the numerator larger only loosens the bound. For the 7-bit list, \(7 \log 2 = 4.852\), the cost the earlier snippet printed, and the theorem rounds it up to 7. \(\blacksquare\)

The bound suggests its learner: trade empirical risk against description length. That is the **minimum description length** (MDL) rule:

\[ \mathrm{MDL}(S) \in \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}} \bigg]. \]

The bracket is the MDL bound of one list. In plain words: **MDL ships the list whose fit plus a cost for its length in bits is smallest.** At 100 labels on our stream that is the 7-bit list, with value 0.4012, beating “everything is junk”, whose 2 bits do not make up for its worse fit.

In the snippet, `mdl_bound` is the bracket, term by term. Two lists with the same empirical risk 0.12 at \(m = 300\), one of 15 bits and one of 60; and how many examples a 29-bit list needs before its complexity term drops to 0.05.

```python
import math

def mdl_bound(LS, bits, m, delta=0.05):
    return LS + math.sqrt((bits + math.log(2 / delta)) / (2 * m))

print(round(mdl_bound(0.12, 15, 300), 4))
print(round(mdl_bound(0.12, 60, 300), 4))
need = (29 + math.log(2 / 0.05)) / (2 * 0.05**2)
print(math.ceil(need))
```

It prints `0.2965`, `0.4458` and `6538`. The last line sets the complexity term \(\sqrt{(29 + \log 40)/(2m)}\) equal to 0.05 and solves for \(m\). Same fit, very different certificates. On our stream, MDL ships 2 bits (“everything is junk”) at 30 labels, the 7-bit list “unknown sender, junk; otherwise keep” at 100 labels, the 15-bit three-test list at 300 and 1,000 labels, and the 29-bit list \(f\) at 3,000. The same staircase as SRM, with steps in different places, because it is a different prior.

The language need not be ours. Take the class of every filter you could write in some programming language, and describe each program by its source file in bits, preceded by that file’s length written in the feature-number code. The length tells a reader where the description ends, so this is a prefix-free description language too, and the MDL bound applies to it: it covers every program you could ever write.

## Occam’s razor, and its catch

Read the MDL bound as a philosophical statement. Two hypotheses with the same empirical risk get different guarantees, and the shorter one gets the better one. In words:

> A short explanation (that is, a hypothesis with a short description) tends to be more valid than a long explanation.

That is **Occam’s razor**, after William of Ockham, a fourteenth-century English logician who is believed to have been the first to state it explicitly. The MDL bound is one justification of it, made precise: the longer a hypothesis’s description, the more examples it must fit before its small empirical risk certifies a small true risk.

Now a puzzle, and it’s a good one. Description length depends on the language. Suppose \(h'\) is much shorter than \(h\) in our list code, and both have the same empirical risk; MDL says to prefer \(h'\). But we could have chosen a different prefix-free language, one that writes \(h\) in 3 bits and \(h'\) in 100,000. Under that language, MDL prefers \(h\). They are the same two lists. Which is it?

The answer is that no hypothesis is more trustworthy than another in itself. What matters is the order of events: the language must be chosen before the training sequence is seen. Recall the simplest case. If you commit to one hypothesis before seeing any data, Hoeffding gives you \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{\log(2/\delta)/(2m)}\), whatever that hypothesis is, however strange. Choosing a description language, or any weighting of hypotheses, is a weaker form of committing in advance. Instead of putting the whole commitment on one hypothesis, you spread it over many. As long as you spread it independently of the training sequence, the guarantee holds. And just as the single hypothesis could be anything, so can the language.

What breaks the guarantee is choosing the language after looking at the labels, giving the list that happens to fit a 3-bit code. Then the weights depend on \(S\), the bad events in the union bound are no longer the ones you budgeted for, and the proof says nothing. Your teammate cannot pick her weights after the run.

[[diagram:commit-order]]

@@CODE@@

## Weaker still: consistency

There is an even weaker way to define learning: let the sample size depend on the distribution as well.

A word of warning first, because of a name clash. In lesson 1, a *hypothesis* was consistent when it agreed with every example. Here, **consistency** is a property of a *learning rule*, borrowed from statistics, and it has nothing to do with fitting the examples. Same word, different idea.

In words first: a consistent learning rule, given enough labels, gets within \(\varepsilon\) of any hypothesis you name, but “enough” may depend on the mail stream itself, so on a stream it has never met you cannot say in advance how many. The memorizer on a stream dominated by a few repeated messages needs a hundred labels; on a stream of a million equally likely messages it needs millions.

**Definition (consistency).** Let \(Z\) be a domain set, \(\mathcal P\) a set of probability distributions over \(Z\), and \(\mathcal H\) a hypothesis class. A learning rule \(A\) is **consistent** with respect to \(\mathcal H\) and \(\mathcal P\) if there is a function \(m^{\mathrm{CON}}_{\mathcal H}: (0,1)^2 \times \mathcal H \times \mathcal P \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\), every \(h \in \mathcal H\) and every \(\mathcal D \in \mathcal P\), if \(m \ge m^{\mathrm{CON}}_{\mathcal H}(\varepsilon, \delta, h, \mathcal D)\) then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon\). If \(\mathcal P\) is the set of all distributions, \(A\) is **universally consistent** with respect to \(\mathcal H\).

Here \(Z\) plays the role of message-label pairs, \(\mathcal X \times \{0,1\}\), and \(\mathcal P\) is the family of streams you are willing to face. The only change from nonuniform learnability is the extra \(\times \mathcal P\) in the signature: the sample size may also read the distribution. In plain words: **for each stream and each hypothesis there is some number of labels beyond which the rule competes with that hypothesis on that stream.** The two streams just mentioned get two very different entries of \(m^{\mathrm{CON}}\), as the snippet below shows.

A nonuniform learner for \(\mathcal H\) is universally consistent for \(\mathcal H\) (take \(m^{\mathrm{CON}}\) to ignore \(\mathcal D\)). The converse fails, and the counterexample is an old friend: lesson 1’s memorizer. Let Memorize return, on a message \(x\), the majority label among the copies of \(x\) in the training sequence, and a fixed default label if \(x\) never appeared. Over any countable domain with a finite label set, Memorize is universally consistent for the class of all classifiers. Yet over an infinite domain the class of all classifiers is not nonuniformly learnable, as we saw. So consistency is strictly weaker. (It is also unsettling: lesson 1’s cautionary tale meets the definition.)

Why does Memorize succeed eventually? Take the noise-free case. It errs only on messages it has never seen, so its true risk is at most the probability mass of the unseen messages, whose average is

\[ \mathbb E_{S}\Big[\sum_{x} \mathcal D(x)\, \mathbb 1[x \notin S]\Big] = \sum_x \mathcal D(x)\, \mathbb P[x \notin S] = \sum_x \mathcal D(x) \big(1 - \mathcal D(x)\big)^m. \]

Read it from the inside out. \(\mathcal D(x)\) is the probability that message \(x\) arrives. \(\mathbb 1[x \notin S]\) is 1 when \(x\) is not among the training messages (\(\notin\) reads “is not in”) and 0 otherwise, so the sum \(\sum_x\) over every message adds up the probability of the unseen ones: the **missing mass**. \(\mathbb E_S\) averages that over random training sequences. Linearity of expectation (the average of a sum is the sum of the averages) moves the average inside the sum, the average of an indicator is the probability of its statement, and \(x\) is missed by all \(m\) independent draws with probability \((1 - \mathcal D(x))^m\), one factor per draw. In plain words: **on average, the unseen mail weighs the sum, over messages, of each message’s probability times its chance of being missed \(m\) times in a row.** If message 1 arrives half the time, at \(m = 10\) it contributes \(0.5 \times 0.5^{10} = 0.0005\); message 4, at \(1/16\), contributes \(0.0625 \times (15/16)^{10} = 0.033\); message 8, at \(1/256\), only 0.004 because it is so rare.

```python
def contribution(q, m):
    return q * (1 - q) ** m

for x in (1, 4, 8):
    print(x, round(contribution(2.0 ** -x, 10), 5))
```

It prints `1 0.00049`, `4 0.03278` and `8 0.00376`. The snippet below adds these terms up, as `missing_mass`.

For any fixed \(\mathcal D\) on a countable domain this goes to 0 as \(m\) grows. Each term shrinks to 0 and never exceeds \(\mathcal D(x)\). So pick a target, say 0.01, and set aside the finitely many messages that carry all but 0.01 of the probability: the rest add at most 0.01, whatever \(m\) is, and the finitely many terms left each vanish as \(m\) grows. But how fast depends entirely on \(\mathcal D\). Compare a stream where message \(x\) arrives with probability \(2^{-x}\) with one spread evenly over a million messages:

```python
def missing_mass(p, m):
    return sum(q * (1 - q) ** m for q in p)

geometric = [2.0 ** -x for x in range(1, 200)]
for m in (10, 100, 1000):
    g = round(missing_mass(geometric, m), 4)
    print(m, g, round((1 - 1e-6) ** m, 4))
```

It prints `10 0.1312 1.0`, `100 0.0143 0.9999` and `1000 0.0014 0.999`. Same learner, same \(m\): on the first stream it has seen nearly everything that matters, and on the second it has seen almost nothing. Consistency promises that each stream is eventually conquered, but it never says when. You will prove the consistency of Memorize yourself in the end-of-lesson problems.

**No-Free-Lunch, revisited.** Lesson 5 proved that no learner learns the class of all classifiers over an infinite domain. Memorize is consistent for exactly that class. Why is that not a contradiction? Look at the order of the quantifiers. No-Free-Lunch says: for every learner and every training sequence size \(m\), there exist a distribution and a labelling function on which the learner, given \(m\) examples, is likely to fail. Consistency of Memorize says: for every distribution and labelling function, there exists a size \(m\), depending on both, beyond which Memorize is likely to succeed. No-Free-Lunch fixes \(m\) first and then picks a bad world; consistency fixes the world first and then picks \(m\).

[[diagram:quantifiers]]

## Which notion of learnability do you want?

We now have three definitions: PAC (one sample size for every hypothesis and every distribution), nonuniform (the size may depend on the hypothesis), and consistency (it may depend on the distribution too). Which is right depends on what you want a definition for. Let’s go through the questions an engineer would ask.

**What is the risk of the hypothesis I learned?** PAC and nonuniform learning both bound the true risk of the returned hypothesis by its empirical risk plus a computable term. Consistency gives no such bound. (You can always estimate the risk on held-out labelled messages, a validation set, which lesson 11 covers.)

**How many examples do I need to be as good as the best hypothesis in the class?** PAC gives a crisp answer before you start. Nonuniform learning does not: the number depends on the best hypothesis, which you don’t know. Consistency does not either, and the number depends on \(\mathcal D\) as well. In this sense PAC is the only useful definition. Keep in mind, though, that even PAC only promises to approach the best in the class: if the class has a large approximation error, the risk may still be large. And when a learner fails, PAC lets you bound the estimation error and so blame the approximation error and change the class; a nonuniform learner lets you change the weights; a consistent learner tells you nothing about which part failed, or how many more examples would help.

**How do I learn, and how do I say what I know?** PAC learning, through No-Free-Lunch, shows prior knowledge is necessary, gives a crisp way to state it (choose a class) and a generic learner for it (ERM). Nonuniform learning also gives a crisp way to state prior knowledge (weights over classes or hypotheses) and a generic learner (SRM). SRM is especially good at **model selection**, when you are unsure which class to use. Fit lists by ERM with cap 6 and you overfit, as the 30-label table showed; cap them at 1 and you underfit; SRM over all sizes picks the size from the data, paying only a small extra in estimation error and the loss of knowing in advance how many examples you need. Consistency gives no learning paradigm and no way to state prior knowledge at all. Memorize needs none.

**Should I prefer a consistent learner?** It sounds desirable to be consistent for every function from \(\mathcal X\) to \(\mathcal Y\): with enough data you would eventually match the Bayes optimal predictor (lesson 3: the best possible predictor of all, with true risk 0.1 on our stream). The argument has two problems. First, on the distributions you actually meet, the consistent learner may need more examples than you will ever have. Second, consistency is cheap to add. Over a countable domain with a finite label set, take any nonuniform learner: run it, compute its bound, and if the bound is small enough, return its hypothesis; otherwise fall back to Memorize. The result is consistent for every classifier. A property that easy to bolt on is a poor reason to choose one learner over another.

@@TAIL@@
