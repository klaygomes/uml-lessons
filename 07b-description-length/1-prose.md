# Description Length and Occam’s Razor

## Two tests, priced like a septillion lists

The help desk wants your filter. Its inbox gets one kind of junk, phishing, and the team has just added the newest feature to the catalogue: feature 20, P, the message asks for a password. Features 7 to 19 were added for the main inbox, for things like newsletters and calendar invites, and they never show up in the help desk’s mail. On the help desk’s mail, junk is easy to describe:

```
if it asks for a password:     junk
else if the sender is unknown: junk
otherwise:                     keep
```

Two tests. You know the drill from lesson 7a. The reviewer wants a certificate of at most 0.20 at confidence \(\delta = 0.05\), with the weights written into the change before any labels are drawn, and the help desk can afford 1,500 labels. You run SRM, the structural risk minimization of lesson 7a, which ships the list whose empirical risk plus its own gap is smallest, with the weights the reviewer already signed, \(w(n) = 6/(\pi^2 n^2)\), and SRM does its job: the list it ships labels every message exactly as the one above. Then the certificate comes back. 0.2381. Rejected.

Here is the SRM table on those 1,500 labels. For each size it shows the best empirical risk \(L_S\), the fraction of the 1,500 labels that the best list of at most that size gets wrong, then the gap SRM gives that size, and their sum, the bound. Before you read it, guess: the right list fits the labels far better than anything else, so what is holding its bound up?

| size \(n\) | best \(L_S\) | gap | bound |
|---|---|---|---|
| 1 | 0.3087 | 0.0446 | 0.3533 |
| 3 | 0.2840 | 0.0618 | 0.3458 |
| 6 | 0.2753 | 0.0810 | 0.3564 |
| 7 to 19 | 0.2753 | 0.0869 to 0.1458 | 0.3622 to 0.4211 |
| 20 | 0.0880 | 0.1501 | **0.2381** |

Read it top to bottom. A list’s size is the larger of its number of tests and its highest feature number, so sizes 1 to 19 cannot test feature 20. Without the password test, the best a list can do is a true risk, the chance of mislabelling the next message, of 0.3: “unknown sender, junk; otherwise keep”, for example, misses the quarter of messages that ask for a password from a known sender. On the labels, the best of them is wrong on more than a quarter. At size 20 the fit is excellent, 0.088, just under the 10% of labels that users flip. The fit is not the problem. The gap is.

Recall from lesson 7a the gap that SRM gives every list of size \(n\), for a finite class \(\mathcal H_n\) and a weight function \(w\):

\[ \varepsilon_n(m, w(n)\delta) = \sqrt{\frac{\log\big(2|\mathcal H_n|/(w(n)\delta)\big)}{2m}}. \]

Read it from the inside out. \(|\mathcal H_n|\) is how many lists have size at most \(n\). \(w(n)\delta\) is the share of the confidence budget \(\delta\) that size \(n\) was given, so dividing by it makes the inside bigger as the share gets smaller. The logarithm is the natural one; dividing by \(2m\) shrinks the whole thing as labels come in, and the square root turns it into a distance on the scale of a risk. In plain words: **a list’s gap is set by how many lists share its size and how small that size’s share of the budget is; the list itself never enters.** For the help desk list, \(n = 20\): \(\mathcal H_{20}\) holds about \(8.41 \times 10^{24}\) lists, the signed weight is \(w(20) = 6/(400\pi^2) = 0.00152\), and the logarithm comes to \(58.085 + 6.489 + 2.996 = 67.57\) (the class, the weight, the budget). Divided by \(2 \times 1{,}500 = 3{,}000\) that is 0.0225, and the square root is 0.1501.

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def gap(n, m, share):
    return math.sqrt(math.log(2 * size_H(n) / share) / (2 * m))

n, m, delta = 20, 1500, 0.05
w = 6 / (math.pi**2 * n**2)
print(f"{size_H(n):.3e}", round(w, 5))
print(round(gap(n, m, w * delta), 4), round(gap(n, m, delta), 4))
print(round(132 / m + gap(n, m, w * delta), 4))
```

It prints `8.412e+24 0.00152`, then `0.1501 0.1427`, then `0.2381`: the right list wrong on 132 of the 1,500 labels, plus its gap. The second number on the middle line is what you would get by betting the whole budget on size 20 before seeing any data. Even that gives \(0.088 + 0.1427 = 0.2307\), still over the bar, and no other size can help, since none of them fits better than 0.2753. So on these labels, no way of sharing the budget across sizes gets the certificate down to 0.20. With the signed weights, this draw would need about 3,300 labels.

That is the bug. SRM charges a list for the company it keeps. The help desk list says very little, two tests, but it shares its size with 8.4 septillion other lists, almost all of them long monsters, and its certificate is priced as if it could be any of them. What we want is a certificate that charges a list for what it actually says. By the end of this lesson you will have one, you will see why it is a precise version of Occam’s razor, and you will see the catch hidden in the razor.

## What you need from lesson 7a

Here are the pieces this lesson reuses, in one place.

- A **decision list** \(h = ((j_1, b_1), \dots, (j_k, b_k); b_0)\) is a chain of tests read from the top: test \(i\) looks at feature \(j_i\), and if that feature is present the list answers \(b_i\) (1 for junk, 0 for keep); if no test fires, the default \(b_0\) answers. The help desk list is \(((20, 1), (3, 1); 0)\). Features are numbered in the order the team added them: 1 is F (“FREE”), 2 is L (a link), 3 is U (unknown sender), 4 is N (sent at night), 5 is A (an attachment), 6 is R (a reply).
- The **size** of a list is the larger of its number of tests and its highest feature number, and at least 1. \(\mathcal H_n\) is the class of lists of size at most \(n\), and \(n(h)\) is the size of \(h\). Counting ordered features and labels for each number of tests, and doubling for the default, gives \(|\mathcal H_n| = 2\sum_{k=0}^{n} 2^k\, n!/(n-k)!\): 158 lists for \(n = 3\), 151,946 for \(n = 6\).
- The **main inbox’s stream**: features 1 to 6 each present with probability 1/2, independently, and later features never; labels follow \(f = ((3, 1), (1, 0), (2, 1), (5, 0), (4, 1); 0)\), with 10% of them flipped. The **help desk’s stream** is built the same way, with features 1 to 6 and 20 live and labels following the help desk list. On both streams, a list that disagrees with the truth on a fraction \(p\) of messages has **true risk** \(L_{\mathcal D}(h) = 0.1 + 0.8p\), the chance that it mislabels the next message to arrive. Its **empirical risk** \(L_S(h)\) is the fraction of the \(m\) labelled messages, the training sequence \(S\), that it gets wrong. **ERM** returns a list with the smallest empirical risk in its class. The true risk of a learned list splits into the **approximation error**, the best true risk in the class it came from, plus the **estimation error**, how far it falls short of that best (lesson 5).
- **Hoeffding’s inequality** (lesson 4): the average of \(m\) independent numbers between 0 and 1 rarely strays far from its mean. **The union bound**: the chance that at least one of several bad events happens is at most the sum of their chances.
- A **weight function** \(w\) gives each size \(n\) a fraction \(w(n)\) of the confidence budget \(\delta\), with \(\sum_n w(n) \le 1\); the signed one is \(w(n) = 6/(\pi^2 n^2)\).
- **The SRM bound**: with probability at least \(1 - \delta\), every list has \(L_{\mathcal D}(h) \le L_S(h) + \varepsilon_{n(h)}(m, w(n(h))\delta)\), with the gap above. **SRM** ships a list with the smallest such bound. On the main inbox it ships the three-test list \(((3, 1), (1, 0), (2, 1); 0)\) at 300 labels, with bound 0.2715, and \(f\) at 1,000 labels, with bound 0.1938.
- A class is **agnostic PAC learnable** (lesson 3) when one number of labels \(m_{\mathcal H}(\varepsilon, \delta)\), fixed before any data, gets the learner within the accuracy \(\varepsilon\) of the best hypothesis in the class, except with probability at most the confidence \(\delta\), on every distribution. It has the **uniform convergence property** (lesson 4) when enough labels put every hypothesis’s empirical risk within \(\varepsilon\) of its true risk at once, and the fundamental theorem (lesson 6a) says both hold exactly when the VC dimension is finite.
- A class is **nonuniformly learnable** when there is a function \(m^{\mathrm{NUL}}_{\mathcal H}: (0,1)^2 \times \mathcal H \to \mathbb N\), taking the two dials \(\varepsilon, \delta\) and a hypothesis \(h\) and returning a number of labels, possibly different for each \(h\) but never depending on the distribution, beyond which the learner is within \(\varepsilon\) of \(h\)’s true risk except with probability \(\delta\). SRM learns every countable union of classes with the uniform convergence property this way, and the nonuniformly learnable classes are exactly the countable unions of agnostic PAC learnable classes. The class of all functions over an infinite domain is not one of them.

## Lists by the bit: minimum description length

SRM spreads the budget over classes, and that is the help desk’s whole problem. There is a sharper version: spread it over individual hypotheses. Our \(\mathcal H\) is countable (each list is a finite object), so we can list its members as \(h_1, h_2, \dots\) and write it as a union of one-member classes, \(\mathcal H = \bigcup_n \{h_n\}\). Then each hypothesis gets its own weight \(w(h)\), say 1/128 for the list “unknown sender, junk; otherwise keep”.

A single fixed hypothesis has uniform convergence with no union bound at all. Its mistakes are \(\mathbb 1[h(x_i) \ne y_i]\), where the **indicator** \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false. These are i.i.d. numbers in \([0, 1]\) (the numbers from 0 to 1, ends included) with mean \(L_{\mathcal D}(h)\), so by Hoeffding’s inequality (lesson 4), \(\mathbb P[|L_S(h) - L_{\mathcal D}(h)| > \varepsilon] \le 2e^{-2m\varepsilon^2}\). Here \(\mathbb P\) is the probability over the training sequence, and \(e^{-2m\varepsilon^2}\) is \(e \approx 2.718\) raised to the power \(-2m\varepsilon^2\), which shrinks very fast as \(m\) grows. In plain words: **the chance that one fixed list’s score on \(m\) labels is off by more than \(\varepsilon\) falls exponentially in \(m\varepsilon^2\).** At \(m = 300\) and \(\varepsilon = 0.1\) it is at most \(2e^{-6} = 0.005\). Setting this to \(\delta\) and solving for \(\varepsilon\) gives the gap \(\sqrt{\log(2/\delta)/(2m)}\), which at \(\delta = 0.05\) is \(\sqrt{3.689/600} = 0.078\):

```python
import math

m, eps, delta = 300, 0.1, 0.05
print(round(2 * math.exp(-2 * m * eps**2), 5))
print(round(math.sqrt(math.log(2 / delta) / (2 * m)), 4))
```

It prints `0.00496` and `0.0784`. Giving the list \(h\) the share \(w(h)\delta\) instead of \(\delta\) turns \(\log(2/\delta)\) into \(\log(2/\delta) - \log w(h)\), and the SRM rule becomes

\[ \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{-\log w(h) + \log(2/\delta)}{2m}} \bigg]. \]

Read it from the inside out. A weight is below 1, so its logarithm is negative and \(-\log w(h) = \log(1/w(h))\) is a positive cost: the smaller the weight, the bigger the cost. Add the confidence cost \(\log(2/\delta)\), divide by \(2m\), take the square root, and add the empirical risk; the argmin picks the lists where that total is smallest. In plain words: **return a list with the smallest empirical risk plus a gap that grows as its weight shrinks.** With weight 1/128 the cost is \(\log 128 = 4.852\); on the main inbox, at 100 labels, that list has empirical risk 0.17, so its bracket is \(0.17 + \sqrt{(4.852 + 3.689)/200} = 0.3767\).

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

The list “unknown sender, junk; otherwise keep”, \(((3, 1); 0)\), is written `1 011 1 0 0`, which is 7 bits. “Everything is junk”, \((();1)\), is `01`, which is 2 bits. The three-test list from lesson 7a’s opening, \(((3, 1), (1, 0), (2, 1); 0)\), takes 15 bits, the five-test list \(f\) takes 29, and the help desk list \(((20, 1), (3, 1); 0)\) takes 18, nine of them for feature 20 alone.

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
print(len(code([(20, 1), (3, 1)], 0)), gamma(20))
```

It prints the five feature codes with lengths 1, 3, 3, 5, 5, matching the formula, then `1011100 01`, `15 29` and `18 000010100`. `bin(j)[2:]` is \(j\) in binary, and `gamma` puts the zeros in front.

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

There is a picture worth keeping here. Read the tosses as the binary digits of a number between 0 and 1, so that `10` followed by anything is a number from 0.5 up to 0.75. The sequences that start with \(\sigma\) then fill a stretch of that line of width \(2^{-|\sigma|}\): `0` owns the first half, `10` the stretch from 0.5 to 0.75. Prefix-free codewords own stretches that never overlap, and stretches that never overlap inside a line of length 1 add up to at most 1.

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

It is the per-hypothesis SRM bound with the weight cost \(-\log w(h)\) replaced by \(|h|\), the description length in bits. In plain words: **except with probability \(\delta\), every list’s true risk is at most its empirical risk plus a gap that grows with the square root of its length in bits.** The 7-bit list on the main inbox at 100 labels, empirical risk 0.17: \(0.17 + \sqrt{(7 + 3.689)/200} = 0.17 + 0.231 = 0.401\). The snippet after the MDL rule below computes this bound as `mdl_bound`.

*Proof.* Distinct hypotheses get distinct descriptions (a string is a prefix of itself, so equal descriptions would break prefix-freeness), so \(\{d(h) : h \in \mathcal H\}\) is a prefix-free set, and the Kraft inequality gives \(\sum_h 2^{-|h|} \le 1\). That also makes \(\mathcal H\) countable, since there are only countably many strings. List its members as \(h_1, h_2, \dots\), take \(\mathcal H_n = \{h_n\}\) and \(w(n) = 2^{-|h_n|}\). Each singleton has the uniform convergence property with gap \(\varepsilon_n(m, \delta) = \sqrt{\log(2/\delta)/(2m)}\), by Hoeffding as above. The SRM bound theorem of lesson 7a then says that with probability at least \(1 - \delta\), every \(h\) has

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{\log(2^{|h|}) + \log(2/\delta)}{2m}}. \]

Here \(\log(2^{|h|})\) is \(\log(1/w(n))\), the weight cost from before. Finally, \(\log(2^{|h|}) = |h| \log 2 < |h|\), because \(\log 2 \approx 0.693\), and making the numerator larger only loosens the bound. For the 7-bit list, \(7 \log 2 = 4.852\), the cost the earlier snippet printed, and the theorem rounds it up to 7. \(\blacksquare\)

The bound suggests its learner: trade empirical risk against description length. That is the **minimum description length** (MDL) rule:

\[ \mathrm{MDL}(S) \in \operatorname*{argmin}_{h \in \mathcal H} \bigg[ L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}} \bigg]. \]

The bracket is the MDL bound of one list. In plain words: **MDL ships the list whose fit plus a cost for its length in bits is smallest.** At 100 labels on the main inbox’s stream that is the 7-bit list, with value 0.4012, beating “everything is junk”, whose 2 bits do not make up for its worse fit.

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

It prints `0.2965`, `0.4458` and `6538`. The last line sets the complexity term \(\sqrt{(29 + \log 40)/(2m)}\) equal to 0.05 and solves for \(m\). Same fit, very different certificates. On the main inbox’s stream, MDL ships 2 bits (“everything is junk”) at 30 labels, the 7-bit list “unknown sender, junk; otherwise keep” at 100 labels, the 15-bit three-test list at 300 and 1,000 labels, and the 29-bit list \(f\) at 3,000. The same staircase as SRM, with steps in different places, because it is a different prior.

The language need not be ours. Take the class of every filter you could write in some programming language, and describe each program by its source file in bits, preceded by that file’s length written in the feature-number code. The length tells a reader where the description ends, so this is a prefix-free description language too, and the MDL bound applies to it: it covers every program you could ever write.

Now try it yourself. In the lab, build a list test by test and watch its codeword grow bit by bit, while the stretch of the line from 0 to 1 that the codeword owns halves with every bit. Then slide the label count and compare the two gaps the list can get: the size-based one from SRM with the signed weights, and the bit-based one from MDL. Before you start, a prediction. The help desk list has two tests, size 20 and 18 bits. At 1,500 labels, which of its two gaps is smaller, and by roughly how much?

[[lab]]

**Back to the help desk.** Here are the two gaps, for the help desk list and, for comparison, for \(f\) on the main inbox, both at 1,500 labels:

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def srm_gap(n, m, delta=0.05):
    share = 6 / (math.pi**2 * n**2) * delta
    return math.sqrt(math.log(2 * size_H(n) / share) / (2 * m))

def mdl_gap(bits, m, delta=0.05):
    return math.sqrt((bits + math.log(2 / delta)) / (2 * m))

print(round(srm_gap(20, 1500), 4), round(mdl_gap(18, 1500), 4))
print(round(srm_gap(5, 1500), 4), round(mdl_gap(29, 1500), 4))
```

It prints `0.1501 0.085`, then `0.0749 0.1044`. For the help desk list the bits are far cheaper. SRM charges it for its whole size class: the logarithm in its gap is 67.57, as we computed in the opening. MDL charges it for what it says: \(18 + \log 40 = 21.69\). For \(f\) it is the other way round: its size-5 class is small, and its 29 bits are expensive. Neither gap is the better one everywhere. Each is a bet placed before the data, and the help desk’s bet is the list code.

[[diagram:two-prices]]

## Occam’s razor, and its catch

Read the MDL bound as a philosophical statement. Two hypotheses with the same empirical risk get different guarantees, and the shorter one gets the better one. In words:

> A short explanation (that is, a hypothesis with a short description) tends to be more valid than a long explanation.

That is **Occam’s razor**, after William of Ockham, a fourteenth-century English logician who is believed to have been the first to state it explicitly. The MDL bound is one justification of it, made precise: the longer a hypothesis’s description, the more examples it must fit before its small empirical risk certifies a small true risk.

Now a puzzle, and it’s a good one. Description length depends on the language. Suppose \(h'\) is much shorter than \(h\) in our list code, and both have the same empirical risk; MDL says to prefer \(h'\). But we could have chosen a different prefix-free language, one that writes \(h\) in 3 bits and \(h'\) in 100,000. Under that language, MDL prefers \(h\). They are the same two lists. Which is it?

The answer is that no hypothesis is more trustworthy than another in itself. What matters is the order of events: the language must be chosen before the training sequence is seen. Recall the simplest case. If you commit to one hypothesis before seeing any data, Hoeffding gives you \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{\log(2/\delta)/(2m)}\), whatever that hypothesis is, however strange. Choosing a description language, or any weighting of hypotheses, is a weaker form of committing in advance. Instead of putting the whole commitment on one hypothesis, you spread it over many. As long as you spread it independently of the training sequence, the guarantee holds. And just as the single hypothesis could be anything, so can the language.

What breaks the guarantee is choosing the language after looking at the labels, giving the list that happens to fit a 3-bit code. Then the weights depend on \(S\), the bad events in the union bound are no longer the ones you budgeted for, and the proof says nothing. Your teammate cannot pick her weights after the run.

[[diagram:commit-order]]

## MDL as code

Here are the two learners in pseudocode, as the theory states them: SRM from lesson 7a, beside MDL.

```
SRM(S, δ):
  prior knowledge: H = H_1 ∪ H_2 ∪ ..., each with gap ε_n
                   weights w(1), w(2), ... with Σ w(n) ≤ 1
  return h in argmin over h ∈ H of
         L_S(h) + ε_{n(h)}(m, w(n(h)) · δ)

MDL(S, δ):
  prior knowledge: H countable, described by a prefix-free
                   language, |h| the length of d(h)
  return h in argmin over h ∈ H of
         L_S(h) + sqrt((|h| + log(2/δ)) / (2m))
```

Both take an argmin over an infinite class, so both need a way to stop. Lesson 7a stopped SRM with **SRM-nested**: when the classes are nested and the gap \(g_n = \varepsilon_n(m, w(n)\delta)\) does not decrease with \(n\), run ERM in \(\mathcal H_1\), then in \(\mathcal H_2\), and so on, keep the best bound found so far, and stop as soon as \(g_n\) alone reaches it. An empirical risk is never below 0, so from then on no larger size can win. MDL stops the same way: enumerate hypotheses in order of description length, and stop at the first length \(\ell\) with \(\sqrt{(\ell + \log(2/\delta))/(2m)} \ge\) the best bound so far.

```
MDL-ordered(S, δ):
  best ← +∞
  for ℓ = 1, 2, 3, ...:
    g ← sqrt((ℓ + log(2/δ)) / (2m))
    if g ≥ best: stop
    for each h with |h| = ℓ:
      if L_S(h) + g < best: best ← L_S(h) + g; ship ← h
  return ship
```

**Cost.** Recall that \(O(\cdot)\) means “grows at most in proportion to”: \(O(m)\) time doubles at most when \(m\) doubles, ignoring constant factors. There are at most \(2^\ell\) strings of length \(\ell\), so MDL up to length \(\ell^*\) examines at most \(2 + 4 + \cdots + 2^{\ell^*} < 2^{\ell^* + 1}\) codewords. Each is decoded and evaluated on \(m\) messages in \(O(\ell^*)\) time per message, so the whole search takes \(O(m \ell^* 2^{\ell^*})\) time. In plain words: **the work is labels times code length times the number of codewords tried, and the longest length dominates.** Lesson 7a’s SRM-nested had the same shape, \(O(m N |\mathcal H_N|)\) up to the stopping size \(N\).

Lesson 7a stopped SRM at size 6 on the main inbox, and that is exact there: only features 1 to 6 ever appear, so a list in \(\mathcal H_n\) for \(n > 6\) behaves on the data like a list in \(\mathcal H_6\) (its tests on later features never fire). The learner on the real stream cannot know that. How far does the stopping rule have to go when it doesn’t?

```python
import math

def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))

def eps(n, m, delta):
    w = 6 / (math.pi**2 * n**2)
    budget = w * delta
    return math.sqrt(math.log(2 * size_H(n) / budget) / (2 * m))

for m, best in ((300, 0.2715), (1000, 0.1938), (3000, 0.151)):
    n = 1
    while eps(n, m, 0.05) < best:
        n += 1
    print(m, n, f"{size_H(n):.2e}")
```

The `while` loop is the stopping rule: grow \(n\) until the gap alone reaches the best bound. It prints `300 14 4.71e+15`, `1000 22 1.55e+28` and `3000 37 6.24e+54`. With 1,000 labels and the best bound at 0.1938, the rule cannot stop before size 22, and \(\mathcal H_{22}\) holds about \(1.55 \times 10^{28}\) lists. MDL is no better: at 1,000 labels its best bound on the main inbox is 0.2227, so it cannot stop before a length \(\ell\) with \(\ell \ge 2m \cdot 0.2227^2 - \log 40\):

```python
import math

m, best = 1000, 0.2227
ell = 2 * m * best**2 - math.log(40)
top = math.ceil(ell)
print(round(ell, 1), top, f"{2.0 ** (top + 1):.2e}")
```

It prints `95.5 96 1.58e+29`: it must examine codes of up to 96 bits, up to about \(1.6 \times 10^{29}\) codewords. Keep those numbers in mind.

**Build it yourself.** Using only the standard library. Start from your solution to lesson 7a’s build-it challenge: `predict`, `draw`, `lists` and `true_risk` carry over unchanged.

```python
import itertools
import math
import random
from collections import Counter

TRUTH = (((3, 1), (1, 0), (2, 1), (5, 0), (4, 1)), 0)
HELP = (((20, 1), (3, 1)), 0)

def predict(h, x): ...
def draw(m, seed, d=6, noise=0.1): ...
def lists(n): ...
def true_risk(h, d=6, noise=0.1): ...
def code_len(h): ...
def mdl(S, delta=0.05, n_max=6): ...

if __name__ == "__main__":
    assert code_len(TRUTH) == 29 and code_len(HELP) == 18
    assert sum(2.0 ** -code_len(h) for h in lists(4)) <= 1
    value, h = mdl(draw(100, 7))
    assert code_len(h) == 7 and round(value, 4) == 0.4012
    value, h = mdl(draw(1000, 7))
    assert code_len(h) == 15 and round(value, 4) == 0.2227
    assert round(true_risk(h), 3) == 0.125
    print("all asserts pass")
```

1. `code_len` is the description length of the list code from “Lists by the bit”.
2. `mdl` searches all lists over features 1 to `n_max` and returns `(value, h)`. That is exact on this stream: a test on a later feature never fires, so it only adds bits.
3. Reproduce MDL’s staircase on the main inbox: 2 bits at 30 labels, 7 at 100, 15 at 300 and 1,000, and 29 at 3,000.
4. Harder: the help desk. Draw features 1 to 6 and then feature 20, in that order, from one generator, label by `HELP`, and search the lists on those seven features. Brute force is slow, with over two million lists, so share prefixes: recurse over the tests, carrying along the messages on which no test has fired yet. With seed 7 and 1,500 labels, MDL should ship an 18-bit list.

## Fix the bug

Back to the help desk. The reviewer offers the same deal as for the main inbox, with the help desk’s budget: the shipped list needs a certificate of at most 0.20 at \(\delta = 0.05\), the weights or the description language are written into the change before any labels are drawn, and the help desk can label 1,500 messages.

In the lab below you choose how to share out the confidence budget. By size, with a weight function: the signed one, \(2^{-n}\), everything on size 20, or an even 1/20 for each of sizes 1 to 20. Or by bits, with MDL and the list code. Then you choose how many labels to buy, your choice locks, the labels are drawn, and the lab shows the candidates’ fit, gap and bound, and which list ships. After it ships, you get to see next week’s help desk inbox: the list’s true risk.

Before your first try, a prediction. You switch to MDL but buy only 300 labels. What ships, and does the reviewer sign?

[[lab-act2]]

## What people get wrong

**“Shorter descriptions are truer.”** Description length is relative to a language. The razor is about committing before you look, not about any intrinsic simplicity.

**“Pick the language that makes the winner short.”** A language, like a weight function, encodes prior knowledge, and it must be fixed before the training sequence is seen. A language tuned after looking at the labels voids the union bound, exactly like the language puzzle in “Occam’s razor, and its catch”. The help desk’s certificate is honest because the list code was written down before its labels existed, and it would have priced any other list just as fairly.

**“MDL is always better than SRM.”** On the help desk it is. On the main inbox at 1,000 labels, SRM with the signed weights certifies \(f\) at 0.1938, while MDL’s best certificate is 0.2227, for the three-test list: \(f\)’s 29 bits cost more than its share of size 5. Weights by size and weights by bits are two different priors, and each wins on the streams it fits.

## Weaker still: consistency

There is an even weaker way to define learning: let the sample size depend on the distribution as well.

A word of warning first, because of a name clash. In lesson 1, a *hypothesis* was consistent when it agreed with every example. Here, **consistency** is a property of a *learning rule*, borrowed from statistics, and it has nothing to do with fitting the examples. Same word, different idea.

In words first: a consistent learning rule, given enough labels, gets within \(\varepsilon\) of any hypothesis you name, but “enough” may depend on the mail stream itself, so on a stream it has never met you cannot say in advance how many. The memorizer on a stream dominated by a few repeated messages needs a hundred labels; on a stream of a million equally likely messages it needs millions.

**Definition (consistency).** Let \(Z\) be a domain set, \(\mathcal P\) a set of probability distributions over \(Z\), and \(\mathcal H\) a hypothesis class. A learning rule \(A\) is **consistent** with respect to \(\mathcal H\) and \(\mathcal P\) if there is a function \(m^{\mathrm{CON}}_{\mathcal H}: (0,1)^2 \times \mathcal H \times \mathcal P \to \mathbb N\) such that for every \(\varepsilon, \delta \in (0, 1)\), every \(h \in \mathcal H\) and every \(\mathcal D \in \mathcal P\), if \(m \ge m^{\mathrm{CON}}_{\mathcal H}(\varepsilon, \delta, h, \mathcal D)\) then with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), \(L_{\mathcal D}(A(S)) \le L_{\mathcal D}(h) + \varepsilon\). If \(\mathcal P\) is the set of all distributions, \(A\) is **universally consistent** with respect to \(\mathcal H\).

Here \(Z\) plays the role of message-label pairs, \(\mathcal X \times \{0,1\}\), and \(\mathcal P\) is the family of streams you are willing to face. The only change from nonuniform learnability is the extra \(\times \mathcal P\) in the signature: the sample size may also read the distribution. In plain words: **for each stream and each hypothesis there is some number of labels beyond which the rule competes with that hypothesis on that stream.** The two streams just mentioned get two very different entries of \(m^{\mathrm{CON}}\), as the snippet below shows.

A nonuniform learner for \(\mathcal H\) is universally consistent for \(\mathcal H\) (take \(m^{\mathrm{CON}}\) to ignore \(\mathcal D\)). The converse fails, and the counterexample is an old friend: lesson 1’s memorizer. Let Memorize return, on a message \(x\), the majority label among the copies of \(x\) in the training sequence, and a fixed default label if \(x\) never appeared. Over any countable domain with a finite label set, Memorize is universally consistent for the class of all classifiers. Yet over an infinite domain the class of all classifiers is not nonuniformly learnable, as we saw at the end of lesson 7a. So consistency is strictly weaker. (It is also unsettling: lesson 1’s cautionary tale meets the definition.)

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

**How do I learn, and how do I say what I know?** PAC learning, through No-Free-Lunch, shows prior knowledge is necessary, gives a crisp way to state it (choose a class) and a generic learner for it (ERM). Nonuniform learning also gives a crisp way to state prior knowledge (weights over classes or hypotheses) and a generic learner (SRM). SRM is especially good at **model selection**, when you are unsure which class to use. Fit lists by ERM with cap 6 and you overfit, as lesson 7a’s 30-label table showed; cap them at 1 and you underfit; SRM over all sizes picks the size from the data, paying only a small extra in estimation error and the loss of knowing in advance how many examples you need. Consistency gives no learning paradigm and no way to state prior knowledge at all. Memorize needs none.

**Should I prefer a consistent learner?** It sounds desirable to be consistent for every function from \(\mathcal X\) to \(\mathcal Y\): with enough data you would eventually match the Bayes optimal predictor (lesson 3: the best possible predictor of all, with true risk 0.1 on our stream). The argument has two problems. First, on the distributions you actually meet, the consistent learner may need more examples than you will ever have. Second, consistency is cheap to add. Over a countable domain with a finite label set, take any nonuniform learner: run it, compute its bound, and if the bound is small enough, return its hypothesis; otherwise fall back to Memorize. The result is consistent for every classifier. A property that easy to bolt on is a poor reason to choose one learner over another.

## Recap and the next bug

1. A description language is prefix-free when no description is the beginning of another. The Kraft inequality then says the weights \(2^{-|h|}\) add up to at most 1, so any prefix-free language is a weight function, fixed before the data.
2. With those weights the SRM bound becomes the MDL bound, \(L_S(h) + \sqrt{(|h| + \log(2/\delta))/(2m)}\), and MDL ships the list that minimizes it: Occam’s razor, valid because the language is fixed before the data.
3. SRM prices a list by its size class, MDL by its own bits. The help desk list has size 20 but only 18 bits, so at 1,500 labels its size-based gap is 0.1501 and its bit-based gap 0.085. For \(f\) on the main inbox it is the other way round. Neither prior wins everywhere.
4. MDL runs by enumerating codes in order of length and stopping once the gap alone reaches the best bound; the work is labels times length times the number of codewords tried.
5. Consistency lets the sample size depend on the distribution too. Memorize is universally consistent over countable domains, which shows how weak that promise is.

The reviewer signs the help desk’s change too. Your filters now ship with certificates, the five-test list \(f\) on the main inbox and the help desk’s list by its bits, and your teammate is happy. Then the operations team runs the main filter on the real mail stream, where hundreds of features are in use, and the training job never finishes. Of course it doesn’t: with 1,000 labels, SRM’s stopping rule has to examine lists up to size 22, about \(1.55 \times 10^{28}\) of them, and MDL has to reach codes of 96 bits. The statistics are fine; a thousand labels are enough. The computation is hopeless. So far we have only asked how many examples a learner needs. When is learning hard because of the computing time, not the data? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof of the Kraft inequality, point to the exact sentence that uses prefix-freeness. Then check the claim fails without it: compute the Kraft sum of the code \(\{0, 01, 1\}\), and name two events in the proof that overlap. Finally, suppose the list code dropped its end flag, so a list is written as its tests followed by the default bit. Find two lists whose codes break prefix-freeness.
2. On 300 labels, one list has 15 bits and empirical risk 0.12; another has 29 bits and empirical risk 0.10. Which does MDL prefer at \(\delta = 0.05\), and by how much? Assuming the two empirical risks stay where they are, what is the smallest \(m\) at which MDL prefers the 29-bit list?

