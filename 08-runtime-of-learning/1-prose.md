# The Runtime of Learning

## The job that never finishes

Lesson 7 ended with a filter that, on paper, does everything right. Your teammate’s decision lists may grow to any size, and structural risk minimization keeps them honest: it shares the confidence budget out across the sizes, gives every list a gap that depends on its size, and returns the list with the smallest bound. The reviewer signed. The five-test list shipped with a certificate.

Then the operations team ran the same training job on the real mail stream, where hundreds of features are in use, and it never finished.

You already know why, because lesson 7 did the sum. SRM’s stopping rule runs ERM (empirical risk minimization, lesson 2’s rule: return a hypothesis with the fewest mistakes on the training examples) over the lists of size at most 1, then size at most 2, and so on, and it may stop only once the gap for the next size reaches the best bound found so far. With 1,000 labels the best bound was 0.1938, and the gap does not reach it before size 22. As in lesson 7, \(\mathcal H_n\) is the class of decision lists of size at most \(n\), so \(\mathcal H_{22}\) is every list with at most 22 tests, each on a different one of features 1 to 22, and it holds about \(1.55 \times 10^{28}\) of them. On a toy stream with six features the rule could stop at size 6; on the real stream it cannot know that no later feature matters.

How long is “never”? Suppose the job does the simplest thing, checking every list of each size against every one of the 1,000 examples, on a machine that does a billion such checks a second. To price that, we need lesson 7’s count of lists, so here it is again, read slowly:

\[ |\mathcal H_n| = 2 \sum_{k=0}^{n} 2^k \, \frac{n!}{(n-k)!}. \]

Read it from the inside out.

- \(k\) is the number of tests in a list, and the sum adds up the lists of each length \(k\) from 0 to \(n\).
- \(n!/(n-k)!\) is \(n \times (n-1) \times \dots \times (n-k+1)\), the number of ways to choose \(k\) different features in order: \(n\) choices for the first test, \(n-1\) for the second, and so on. The exclamation mark is the factorial, \(n! = 1 \times 2 \times \dots \times n\).
- \(2^k\) is the label each of the \(k\) tests outputs when it fires, junk or keep, two choices per test.
- The factor 2 in front is the default label at the end of the list.
- The bars \(|\cdot|\) mean “how many”: \(|\mathcal H_n|\) is the number of lists in the class.

In plain words: **the number of decision lists of size at most \(n\) is the sum, over every length, of the ways to order the tests times the ways to label them, doubled for the default.** With \(n = 2\) features: one empty list, \(2 \times 2 = 4\) lists of one test, \(4 \times 2 = 8\) lists of two, so \(2 \times (1 + 4 + 8) = 26\) lists. In the snippet below, `size_H(n)` is this formula term by term, with `math.perm(n, k)` standing for \(n!/(n-k)!\); `size_H(2)` returns 26.

Here is the cost of finishing every size up to \(n\).

```python
import math
def size_H(n):
    return 2 * sum(2**k * math.perm(n, k) for k in range(n + 1))
m, rate, day = 1000, 1e9, 24 * 3600
for n in (10, 11, 12, 22):
    evals = m * sum(size_H(k) for k in range(1, n + 1))
    print(n, f"{size_H(n):.3g}", f"{evals / rate / day:.3g}")
```

It prints `10 1.23e+10 0.149`, `11 2.7e+11 3.27`, `12 6.47e+12 78.1` and `22 1.55e+28 1.84e+17`: the size, the number of lists of that size or less, and the days needed. Size 10 is done in under four hours. Size 11 takes a long weekend, size 12 most of a season. Size 22, where the stopping rule is allowed to stop, needs \(1.84 \times 10^{17}\) days, about \(5 \times 10^{14}\) years.

Notice which number is the problem. The statistics are fine: lesson 7’s guarantee says a thousand labels are enough. Each extra size adds only a logarithm’s worth to the gap, while the number of lists to check grows by a factor of twenty or more. Every lesson so far asked how many examples learning needs. Nobody asked how long it takes to use them.

That is this lesson. We will ask when a learning problem is computationally easy, when it is hard, and, the twist that rescues the filter, when changing the hypothesis class makes a hard problem easy.

## Two prices for every class

Every hypothesis class has two prices. The first is the one you have been paying since lesson 2: the **sample complexity**, how many examples the learner needs before its guarantee holds. That is a price in information. The second is the **computational complexity**: how much computation the learner does to turn those examples into a hypothesis, and to use the hypothesis on a new message. That is a price in time.

How do you measure time without a stopwatch? The seconds an algorithm takes depend on the machine: the clock rate of its processor, its memory, the language. So computer science measures time asymptotically. When we say merge sort sorts \(n\) items in time \(O(n \log n)\), we mean this: on any machine that implements some accepted abstract model of computation, there are constants \(c\) and \(n_0\), which may depend on the machine, such that for every \(n > n_0\), sorting any \(n\) items takes at most \(c \, n \log n\) seconds. The machine changes the constant and nothing else.

That sentence is dense, so let’s take it apart.

- \(O(\cdot)\), read “big O of”, names a growth rate and hides the constant in front.
- \(n\) is the number of items, and \(\log n\) is its logarithm (base 2 here; another base only changes \(c\)), so \(n \log n\) grows a little faster than \(n\).
- \(c\) is the constant a particular machine brings: a slow machine has a large \(c\), a fast one a small \(c\).
- \(n_0\) is a threshold: the promise only has to hold for inputs larger than \(n_0\), because small inputs are cheap anyway.
- “For every \(n > n_0\)” and “any \(n\) items” say the bound holds for every large size and for the worst arrangement of that size.

In plain words: **past some size, the time never exceeds a fixed multiple of \(n \log n\).** For 8 items, \(8 \log_2 8 = 24\), and merge sort makes at most 24 comparisons. Here is merge sort counting its comparisons against \(n \log_2 n\), with \(c = 1\).

```python
import math, random
def sort(a, count):
    if len(a) < 2:
        return a
    half = len(a) // 2
    left, right = sort(a[:half], count), sort(a[half:], count)
    out = []
    while left and right:
        count[0] += 1
        side = left if left[0] <= right[0] else right
        out.append(side.pop(0))
    return out + left + right
for n in (8, 1000, 10000):
    count = [0]
    sort(random.Random(0).sample(range(n), n), count)
    print(n, count[0], round(n * math.log2(n)))
```

It prints `8 15 24`, `1000 8709 9966` and `10000 120501 132877`: the comparisons, then \(n \log_2 n\), and the first never passes the second.

A task is called **feasible**, or **efficiently computable**, when some algorithm does it in time \(O(p(n))\) for a polynomial \(p\), a sum of powers of \(n\) with constant coefficients and exponents, such as \(n^3\) or \(5n^2 + n\). Polynomial is the line between “buy a faster machine” and “no machine will do”. Doubling \(n\) multiplies \(n^3\) by 8. It multiplies \(2^n\) by \(2^n\).

```python
n = 10
print((2 * n) ** 3 // n ** 3, 2 ** (2 * n) // 2 ** n)
```

It prints `8 1024`: going from 10 to 20 features multiplies \(n^3\) by 8 and \(2^n\) by \(2^{10} = 1024\).

All of that assumes we know what \(n\), the input size, is. For sorting it is obvious: the length of the list, or the number of bits it takes to write down. For learning it is not obvious at all, and this is the first real subtlety of the lesson.

Why not the number of examples \(m\), the length of the training sequence? Because that would be pointless. Give a learner far more examples than its sample complexity, and it can simply ignore the extra ones. A larger training sequence never makes learning harder, so the time we allow a learner should not grow just because we handed it more data. Measuring runtime against \(m\) would reward a learner for wasting examples.

Instead, we measure runtime against the natural parameters of the problem: the accuracy \(\varepsilon\) (as since lesson 3, the extra true risk we tolerate above the best in the class, such as 0.05), the confidence \(\delta\) (the chance we allow the guarantee to fail, such as 0.01), the number of features, and measures of how complex the hypothesis class is. To get an asymptotic question at all, we build a **sequence of problems**. Take axis aligned rectangles, which you will meet properly in a moment. Fix \(\varepsilon\) and \(\delta\) and let the dimension run through \(n = 2, 3, 4, \dots\): that is one sequence. Or fix the dimension and \(\delta\), and let the accuracy run through \(\varepsilon = 1/2, 1/3, \dots\): that is another. Once you fix the sequence, you can ask how the runtime grows along it.

## A learner that cheats

Before the formal definition, one more trap. Here is a learner that runs in constant time on every training sequence, whatever the class.

It does no work at all. It returns, as its hypothesis, a little program that stores the training sequence, and whenever someone asks it for the label of a message \(x\), runs ERM on the stored examples and applies the result to \(x\). The learner itself always returns the same thing and never touches the data. Constant time!

Of course nothing has been gained. The work moved from training to prediction: every message that arrives at the filter now triggers a full ERM search. This is the **lazy learner**, and the rule that defeats it is simple. We require that the hypothesis a learner returns can predict the label of a new example in time no greater than the learner’s own training time. A learner may not hide its work inside its output.

[[diagram:lazy]]

## The formal definition

Now we can say exactly what “learning is fast” means. We count **operations** of an abstract machine, such as a Turing machine (or a Turing machine over the reals, when the inputs are real numbers), and we assume that every machine implementing it does each operation in at most some constant number of seconds.

**Definition (the runtime of a learner).** Recall that a learning task is a triple \((\mathcal Z, \mathcal H, \ell)\): a domain of examples, a benchmark hypothesis class, and a loss function.

1. Let \(f : (0,1)^2 \to \mathbb N\) be a function, \((\mathcal Z, \mathcal H, \ell)\) a learning task and \(A\) a learning algorithm. We say \(A\) **solves the learning task in time** \(O(f)\) if there is a constant \(c\) such that for every distribution \(\mathcal D\) over \(\mathcal Z\) and every \(\varepsilon, \delta \in (0, 1)\), when \(A\) has access to examples drawn i.i.d. from \(\mathcal D\):
   - \(A\) stops after at most \(c f(\varepsilon, \delta)\) operations;
   - the hypothesis \(h_A\) it returns can predict the label of a new example in at most \(c f(\varepsilon, \delta)\) operations;
   - with probability at least \(1 - \delta\) over the examples \(A\) receives, \(L_{\mathcal D}(h_A) \le \min_{h' \in \mathcal H} L_{\mathcal D}(h') + \varepsilon\).
2. Let \((\mathcal Z_n, \mathcal H_n, \ell_n)_{n=1}^\infty\) be a sequence of learning tasks, and \(A\) an algorithm designed for tasks of this form. Given \(g : \mathbb N \times (0,1)^2 \to \mathbb N\), we say the **runtime** of \(A\) with respect to the sequence is \(O(g)\) if for every \(n\), \(A\) solves task \(n\) in time \(O(f_n)\), where \(f_n(\varepsilon, \delta) = g(n, \varepsilon, \delta)\).

We call \(A\) **efficient** with respect to the sequence if its runtime is \(O(p(n, 1/\varepsilon, 1/\delta))\) for some polynomial \(p\).

Read the three bullets again with the lazy learner in mind. The first bounds training, the second bounds prediction by the same budget, and the third is the agnostic PAC promise of lesson 3. Notice also what “has access to examples” implies: the learner draws its own examples, and drawing one costs at least one operation. So an efficient learner can only ever use polynomially many examples. Efficient learning implies a polynomial sample complexity, never the other way round.

The definition makes one thing very plain: whether a learning problem is “efficiently solvable” depends on which sequence you embed it in. Take finite classes. Lesson 2 and lesson 4 showed that ERM over \(\mathcal H\) learns with \(m_{\mathcal H}(\varepsilon, \delta)\) of order \(\log(|\mathcal H|/\delta)/\varepsilon^2\) examples in general (and \(\log(|\mathcal H|/\delta)/\varepsilon\) under realizability). If evaluating one hypothesis on one example takes constant time, ERM by exhaustive search runs in time \(O(|\mathcal H| \, m_{\mathcal H}(\varepsilon, \delta))\). For one fixed finite class that is polynomial in \(1/\varepsilon\) and \(1/\delta\). For a sequence with \(|\mathcal H_n| = n\), it is still efficient. For a sequence with \(|\mathcal H_n| = 2^n\), the sample complexity is still polynomial in \(n\), but the search is exponential.

Here are both sequences side by side, using the realizable bound with \(\varepsilon = 0.05\), \(\delta = 0.01\). The columns are \(n\), the search work when \(|\mathcal H_n| = n\), the examples needed when \(|\mathcal H_n| = 2^n\), and the search work for that class.

```python
import math
eps, delta = 0.05, 0.01
def m_H(size):
    return math.ceil(math.log(size / delta) / eps)
for n in (10, 20, 40, 80):
    m_small, m_big = m_H(n), m_H(2 ** n)
    print(n, m_small * n, m_big, f"{m_big * 2 ** n:.2g}")
```

It prints `10 1390 231 2.4e+05`, `20 3060 370 3.9e+08`, `40 6640 647 7.1e+14` and `80 14400 1202 1.5e+27`. Look at the third column: going from 10 to 80 multiplies the examples needed by about five. The fourth column multiplies by about \(10^{22}\). That is the training job from the opening in miniature: a class whose size is exponential in \(n\) is cheap in examples and ruinous in search.

[[diagram:two-clocks]]

## ERM by brute force

Let’s state the most natural learner precisely, because every algorithm in this lesson is a cleverer version of it. Given a class \(\mathcal H\), a domain \(\mathcal Z\) and a loss \(\ell\), the rule \(\mathrm{ERM}_{\mathcal H}\) is: on a finite input sequence \(S \in \mathcal Z^m\), return some \(h \in \mathcal H\) that minimizes the empirical risk

\[ L_S(h) = \frac{1}{|S|} \sum_{z \in S} \ell(h, z). \]

For binary classification, lesson 6 showed that if learning is possible at all, ERM does it. So the question “how fast can we learn?” starts as “how fast can we run ERM?”.

Restricting yourself to a finite class sounds mild. The class of every hypothesis a C++ program of at most 10,000 bits can compute is finite. So is any class described by finitely many parameters, if you store each parameter with finitely many bits: rectangles with corners written to three decimal places, say. And the sample complexity of a finite class barely notices its size, because it pays only \(\log |\mathcal H|\). For the C++ class, \(|\mathcal H| = 2^{10{,}000}\) and \(\log |\mathcal H| = 10{,}000 \log 2 \approx 6{,}931\), a perfectly reasonable number of examples.

The obvious ERM for a finite class is **exhaustive search**: compute \(L_S(h)\) for every \(h \in \mathcal H\) and return one with the smallest value. You built it in lesson 2. Here it is again with its cost written in.

```
ExhaustiveERM(H, S):
  best ← none; best_errors ← m + 1
  for h in H:
    errors ← 0
    for i = 1..m:
      if h(x_i) ≠ y_i: errors ← errors + 1
    if errors < best_errors:
      best ← h; best_errors ← errors
  return best
```

**Time.** If evaluating \(\ell(h, z)\) on one example takes constant time \(k\), the search takes \(k \, |\mathcal H| \, m\). With \(m\) set to the sample complexity, \(k \, |\mathcal H| \log(|\mathcal H|/\delta)/\varepsilon\) under realizability. **Space.** \(O(1)\) words beyond the input and one stored hypothesis, if \(\mathcal H\) can be enumerated one member at a time.

The linear dependence on \(|\mathcal H|\) is the whole problem. Take a sequence with \(\log |\mathcal H_n| = n\), such as C++ programs of at most \(n\) bits: the sample complexity grows like \(n\), the search like \(2^n\). That is exactly the job that would never finish.

Here is a class you know well, which shows the gap at real scale: conjunctions over the thirty features of the real mail system (a conjunction is an AND of features or their negations; lesson 3’s second problem introduced them, and the next sections make them precise). There are \(3^{30} + 1\) of them.

```python
import math
eps, delta, rate = 0.05, 0.01, 1e9
year = 365.25 * 24 * 3600
n = 30
size = 3 ** n + 1
m = math.ceil(math.log(size / delta) / eps)
print(m, f"{size * m:.3g}", round(size * m / rate / year, 1))
print(m * 2 * n)
```

It prints `752 1.55e+17 4.9` and `45120`. Exhaustive search over conjunctions needs 752 examples and \(1.55 \times 10^{17}\) evaluations: 4.9 years at a billion a second. The second number is a preview. An algorithm you will build in a few minutes finds the same answer in 45,120 steps.

That is the key point of this section, and it is easy to miss. The inefficiency of one algorithm, exhaustive search, says nothing yet about the problem. Maybe there is a smarter way to find an ERM hypothesis. For some classes there is, and the next three sections are about classes where we can find it.

**Build it yourself.** Using only the standard library, write `all_conjunctions(n)`, returning every conjunction over \(n\) features as a frozenset of literals \((i, v)\) (the literal is satisfied when feature \(i\) equals \(v\)), plus the all-negative conjunction containing every literal, and `exhaustive_erm(H, S)`, returning the best conjunction and the number of evaluations it made. The starter data is the mail-room training sequence you will meet in the conjunctions section.

```python
import itertools

S = [((1, 1, 0, 0), 1), ((0, 1, 1, 0), 0), ((1, 1, 1, 0), 1),
     ((1, 0, 0, 1), 0), ((0, 0, 1, 0), 0), ((1, 0, 1, 0), 1),
     ((0, 1, 0, 1), 0), ((1, 0, 1, 1), 0)]

def predict(c, x):
    return int(all(x[i] == v for i, v in c))

def all_conjunctions(n): ...
def exhaustive_erm(H, S): ...

if __name__ == "__main__":
    H = all_conjunctions(4)
    assert len(H) == 82
    best, evaluations = exhaustive_erm(H, S)
    assert evaluations == 82 * 8
    assert best == {(0, 1), (3, 0)}
```

## Rectangles: draw the tightest box

Lesson 6 met axis aligned rectangles as a class with VC dimension \(2n\) in \(\mathbb R^n\). They suit features that are numbers rather than yes/no answers. Suppose two of the filter’s inputs are scores: \(x_1\), the share of capital letters in the subject, and \(x_2\), the share of the body taken up by links. This season’s junk sits in a middle band on both: enough capitals to shout, not so many that a simple threshold catches it.

Formally, the class of axis aligned rectangles in \(\mathbb R^n\) is

\[ \mathcal H_n = \{ h_{(a_1, \dots, a_n, b_1, \dots, b_n)} : \forall i,\ a_i \le b_i \}, \qquad h_{(a, b)}(x) = \begin{cases} 1 & \text{if } \forall i,\ x_i \in [a_i, b_i] \\ 0 & \text{otherwise.} \end{cases} \]

Assume realizability: some rectangle labels every example correctly. Can you find one with zero empirical risk, quickly? Before you read on, here are eight examples. Where would you put the box?

| i | capitals \(x_1\) | links \(x_2\) | label |
|---|---|---|---|
| 1 | 0.35 | 0.25 | junk |
| 2 | 0.60 | 0.50 | junk |
| 3 | 0.50 | 0.30 | junk |
| 4 | 0.75 | 0.40 | junk |
| 5 | 0.10 | 0.30 | keep |
| 6 | 0.90 | 0.50 | keep |
| 7 | 0.50 | 0.80 | keep |
| 8 | 0.40 | 0.10 | keep |

The answer is the laziest box that could possibly work: the **tightest box** around the junk examples. For each coordinate \(i\), set

\[ a_i = \min\{ x_i : (x, 1) \in S \}, \qquad b_i = \max\{ x_i : (x, 1) \in S \}. \]

Here that gives \([0.35, 0.75] \times [0.25, 0.5]\), and none of the four keep messages falls inside it.

```python
S = [((0.35, 0.25), 1), ((0.6, 0.5), 1), ((0.5, 0.3), 1),
     ((0.75, 0.4), 1), ((0.1, 0.3), 0), ((0.9, 0.5), 0),
     ((0.5, 0.8), 0), ((0.4, 0.1), 0)]
pos = [x for x, y in S if y == 1]
box = [(min(p[i] for p in pos), max(p[i] for p in pos))
       for i in range(2)]
def h(x):
    return int(all(a <= xi <= b for (a, b), xi in zip(box, x)))
print(box)
print(sum(h(x) != y for x, y in S))
```

It prints `[(0.35, 0.75), (0.25, 0.5)]` and `0`.

[[diagram:tight-box]]

**Claim (the tightest box is an ERM in the realizable case).** If some rectangle is consistent with \(S\) and \(S\) has at least one positive example, the tightest box has empirical risk 0, and it can be computed in time \(O(nm)\).

*Proof.* Junk examples first. Each positive \(x\) has \(a_i \le x_i \le b_i\) in every coordinate, because \(a_i\) and \(b_i\) are the smallest and largest \(i\)-th coordinates among the positives, and \(x\) is one of them. So the box labels every positive example 1.

Now keep examples. By realizability there is a rectangle \(h^\star = \prod_i [a_i^\star, b_i^\star]\) that labels every example correctly. It labels every positive 1, so every positive lies in it: \(a_i^\star \le x_i \le b_i^\star\) for every positive \(x\) and every \(i\). Taking the minimum and the maximum over the positives gives \(a_i^\star \le a_i\) and \(b_i \le b_i^\star\). So the tightest box sits inside \(h^\star\). A negative example lies outside \(h^\star\), because \(h^\star\) labels it 0, so it lies outside the smaller box too, and the box labels it 0.

For the time, each \(a_i\) and \(b_i\) is a minimum or a maximum over at most \(m\) numbers, so \(O(m)\) each, and there are \(2n\) of them: \(O(nm)\) in all. \(\blacksquare\)

If \(S\) has no positive example at all, any rectangle that contains none of the training messages is consistent, for instance a single point away from them all. Code usually returns “nothing is junk” in that case.

```
TightestBox(S):
  P ← { x : (x, 1) in S }
  if P is empty: return the all-keep hypothesis
  for i = 1..n:
    a_i ← min over x in P of x_i
    b_i ← max over x in P of x_i
  return h(x) = 1 if a_i ≤ x_i ≤ b_i for all i, else 0
```

**Time** \(O(nm)\) to train, \(O(n)\) to predict. **Space** \(O(n)\) for the \(2n\) bounds.

## Rectangles with noisy labels

Now drop realizability. Users mislabel things, and the band is not perfectly rectangular. ERM must now find the rectangle with the fewest mistakes, not one with none. How hard can that be?

Hard, it turns out, in a precise sense. To say it, we need three standard names from complexity theory. A **decision problem** asks a yes/no question about an input, such as “does this graph have a proper colouring with three colours?”. **P** is the class of decision problems some algorithm answers in time polynomial in the input size. **NP** is the class whose yes answers come with a certificate that can be checked in polynomial time (a colouring, for the graph question). A problem is **NP-hard** when every problem in NP reduces to it in polynomial time, so a polynomial algorithm for it would put all of NP in P. Whether \(P = NP\) is the most famous open question in computer science, and almost everyone believes the answer is no.

**Fact (agnostic ERM is NP-hard for rectangles).** For many common classes, including axis aligned rectangles, solving the ERM problem in the agnostic setting is NP-hard. In most cases it is NP-hard even to find some \(h \in \mathcal H\) whose empirical risk is at most a constant \(c > 1\) times that of the best. So unless \(P = NP\), no algorithm whose runtime is polynomial in both \(m\) and \(n\) is guaranteed to find an ERM hypothesis.

We won’t prove this one; it is a research result, and the proof is a delicate reduction. But here is the surprise: for any *fixed* dimension \(n\), there is an agnostic learner whose runtime is polynomial in \(1/\varepsilon\) and \(1/\delta\). Its dependence on \(n\) is exponential, which is exactly what the hardness result allows.

The idea comes from the tightest box. A box built from a set of points is pinned down by at most \(2n\) of them: one point achieving each minimum and one achieving each maximum. So try every subset of at most \(2n\) training messages, build the tightest box around each, and keep the box with the fewest mistakes.

**Claim (subset search is an exact ERM).** Let \(S\) be any training sequence in \(\mathbb R^n\). Among the tightest boxes of all subsets of at most \(2n\) training messages, together with the all-keep hypothesis, there is one whose empirical risk is the minimum over all rectangles.

*Proof.* Take any rectangle \(h\). We will find a candidate on the list that labels every training message exactly as \(h\) does, so it has the same empirical risk. Apply this to an ERM rectangle and you are done.

Let \(T\) be the set of training messages that \(h\) labels 1 (whatever their labels in \(S\)). If \(T\) is empty, \(h\) labels every training message 0, just like the all-keep hypothesis.

Otherwise let \(B\) be the tightest box around \(T\). Every message in \(T\) lies in \(B\), by the same argument as before, so \(B\) labels it 1, like \(h\). Every message of \(T\) lies inside the rectangle \(h\), so the minimum and maximum of each coordinate over \(T\) lie inside \(h\)’s interval for that coordinate, which means \(B \subseteq h\). A training message not in \(T\) lies outside \(h\), hence outside \(B\), so \(B\) labels it 0, like \(h\).

Finally, \(B\) is also the tightest box of a small subset. For each coordinate, pick one message of \(T\) that achieves the minimum and one that achieves the maximum. That is at most \(2n\) messages, call them \(T'\), and the tightest box around \(T'\) has the same \(2n\) bounds as \(B\), so it is \(B\). Since \(T'\) is a subset of the training messages with \(|T'| \le 2n\), \(B\) is on the list. \(\blacksquare\)

Here is the claim as an experiment: nine messages with random positions and random labels, ERM by subset search against ERM by brute force over every rectangle whose sides sit at data coordinates.

```python
import itertools, random
rng = random.Random(6)
S = [((rng.random(), rng.random()), rng.randint(0, 1))
     for _ in range(9)]
xs, empty = [x for x, _ in S], sum(y for _, y in S)
def errs(B):
    return sum(all(a <= v <= b for (a, b), v in zip(B, x)) != y
               for x, y in S)
subsets = [c for k in range(1, 5)
           for c in itertools.combinations(xs, k)]
tight = [[(min(c), max(c)) for c in zip(*T)] for T in subsets]
sides = [[(a, b) for a in c for b in c if a <= b]
         for c in zip(*xs)]
grid = list(itertools.product(*sides))
print(len(tight), len(grid))
print(min([empty] + [errs(B) for B in tight]),
      min([empty] + [errs(B) for B in grid]))
```

It prints `255 2025` and `2 2`: 255 candidate boxes from subsets, 2,025 from the grid, and the same best number of mistakes.

How many subsets are there? The number of subsets of at most \(K\) elements of an \(m\)-element set is \(\sum_{k=0}^{K} \binom{m}{k}\). Each such subset can be written as a sorted list of its indices, padded with zeros to length \(K\): a \(K\)-tuple over \(\{0, 1, \dots, m\}\). Different subsets give different tuples, so there are at most \((m + 1)^K\) of them. With \(K = 2n\), and \(O(nm)\) work to build and score each box, the runtime is

\[ O\!\left( nm \, (m+1)^{2n} \right) = m^{O(n)}. \]

```python
from math import comb
for m in (100, 1000):
    for n in (1, 2, 3):
        subsets = sum(comb(m, k) for k in range(2 * n + 1))
        bound = (m + 1) ** (2 * n)
        print(m, n, f"{subsets:.3g}", f"{bound:.3g}")
```

It prints, for \(m = 100\): \(5.05 \times 10^3\), \(4.09 \times 10^6\) and \(1.27 \times 10^9\) subsets for \(n = 1, 2, 3\), each under its bound (\(1.02 \times 10^4\), \(1.04 \times 10^8\), \(1.06 \times 10^{12}\)). For \(m = 1{,}000\): \(5.01 \times 10^5\), \(4.16 \times 10^{10}\) and \(1.38 \times 10^{15}\). Polynomial in \(m\) for each fixed \(n\), with the degree growing with \(n\).

For a fixed \(n\), the sample complexity of rectangles is polynomial in \(1/\varepsilon\) and \(\log(1/\delta)\) (lesson 6’s fundamental theorem, with VC dimension \(2n\)), so this learner runs in time polynomial in \(1/\varepsilon\) and \(1/\delta\). There is no contradiction with the hardness fact: that one forbids a runtime polynomial in \(n\) as well, and ours is exponential in \(n\).

[[diagram:subset-box]]

```
SubsetBoxERM(S):
  best ← all-keep hypothesis; best_errors ← number of junk in S
  for every subset T of S's messages with 1 ≤ |T| ≤ 2n:
    B ← TightestBox of T (every message in T treated as junk)
    if errors of B on S < best_errors:
      best ← B; best_errors ← errors of B on S
  return best
```

**Time** \(O(nm \, (m+1)^{2n})\). **Space** \(O(n)\) beyond the input, enumerating subsets one at a time.

**Build it yourself.** Write `tight_box(S)` (return `None` when there is no junk example) and `agnostic_box(S)`, returning the best box and its number of mistakes. The noisy sequence adds one far-away junk message and one keep message inside the band.

```python
import itertools

S = [((0.35, 0.25), 1), ((0.6, 0.5), 1), ((0.5, 0.3), 1),
     ((0.75, 0.4), 1), ((0.1, 0.3), 0), ((0.9, 0.5), 0),
     ((0.5, 0.8), 0), ((0.4, 0.1), 0)]
noisy = S + [((0.95, 0.95), 1), ((0.55, 0.35), 0)]

def predict(box, x):
    return int(box is not None and
               all(a <= v <= b for (a, b), v in zip(box, x)))

def errors(box, S):
    return sum(predict(box, x) != y for x, y in S)

def tight_box(S): ...
def agnostic_box(S): ...

if __name__ == "__main__":
    box = tight_box(S)
    assert box == [(0.35, 0.75), (0.25, 0.5)]
    assert errors(box, S) == 0
    assert tight_box([(x, 0) for x, _ in S]) is None
    assert errors(tight_box(noisy), noisy) == 3
    box, err = agnostic_box(noisy)
    assert err == 2 and errors(box, noisy) == 2
```

Notice the third-to-last assert. With noise, the tightest box is no longer an ERM: one far-away junk label stretches it over three keep messages.

## Conjunctions: cross out what junk contradicts

Back to yes/no features. A **literal** is a feature or its negation: \(x_i\), satisfied when feature \(i\) is 1, or \(\neg x_i\), satisfied when it is 0. A **Boolean conjunction** over \(\mathcal X = \{0,1\}^n\) is an AND of literals,

\[ h(x) = x_{i_1} \wedge \dots \wedge x_{i_k} \wedge \neg x_{j_1} \wedge \dots \wedge \neg x_{j_r}, \]

for some indices \(i_1, \dots, i_k, j_1, \dots, j_r \in [n]\), which says 1 exactly when \(x_{i_1} = \dots = x_{i_k} = 1\) and \(x_{j_1} = \dots = x_{j_r} = 0\). In the mail room, “F and not N” (says “FREE”, sent in the daytime) is a conjunction. Let \(\mathcal H_C^n\) be the class of all of them.

How big is it? In a conjunction, each feature either appears, or appears negated, or does not appear: three choices per feature, \(3^n\) in all. Add the all-negative formula, which labels everything 0 (for instance \(x_1 \wedge \neg x_1\)), and

\[ |\mathcal H_C^n| \le 3^n + 1. \]

**Sample complexity.** By lesson 2’s finite-class bound, ERM over \(\mathcal H_C^n\) learns in the realizable case with \(m \ge \log(|\mathcal H_C^n|/\delta)/\varepsilon\) examples. When \(n \ge 2\) and \(\delta \le 1/2\), this is at most \(n \log(3/\delta)/\varepsilon\), a simpler expression worth having. Here is why, one inequality at a time:

\[ \log\frac{3^n + 1}{\delta} \le \log\frac{2 \cdot 3^n}{\delta} = n \log 3 + \log 2 + \log\frac1\delta \le n \log 3 + 2 \log \frac 1\delta \le n \log 3 + n \log \frac1\delta. \]

The first step is \(3^n + 1 \le 2 \cdot 3^n\), true because \(1 \le 3^n\). The middle equality splits the logarithm of a product into a sum. The next step uses \(\log 2 \le \log(1/\delta)\), which is exactly \(\delta \le 1/2\). The last uses \(2 \le n\) and \(\log(1/\delta) \ge 0\). The right side is \(n \log(3/\delta)\); divide by \(\varepsilon\) and you have the bound. With thirty features, \(\varepsilon = 0.05\) and \(\delta = 0.01\), the exact bound is the 752 you saw earlier, and the simpler \(n \log(3/\delta)/\varepsilon\) is 3,422: safe, and loose.

So conjunctions are cheap in examples. Are they cheap in time? Exhaustive search says 4.9 years. Here is the trick that brings it down to a blink.

### Worked example: crossing out literals

Start with the most restrictive conjunction there is: all \(2n\) literals at once, \(h_0 = x_1 \wedge \neg x_1 \wedge \dots \wedge x_n \wedge \neg x_n\). It contains a feature and its negation, so it says keep to everything. Then go through the junk examples. Each junk message \(v\) falsifies some literals of the current conjunction, and those literals cannot be in the truth, because the truth says junk to \(v\). Cross them out. Keep examples need no action at all.

Here is a mail-room training sequence over FLUN, labelled by the truth “F and not N”. Before reading the table, guess: after which example does the learner find the truth, and what does it believe before that?

| i | message FLUN | label | conjunction after the example | true risk |
|---|---|---|---|---|
| 0 | | | all eight literals | 0.25 |
| 1 | 1100 | junk | F, L, ¬U, ¬N | 0.1875 |
| 2 | 0110 | keep | F, L, ¬U, ¬N | 0.1875 |
| 3 | 1110 | junk | F, L, ¬N | 0.125 |
| 4 | 1001 | keep | F, L, ¬N | 0.125 |
| 5 | 0010 | keep | F, L, ¬N | 0.125 |
| 6 | 1010 | junk | F, ¬N | 0 |
| 7 | 0101 | keep | F, ¬N | 0 |
| 8 | 1011 | keep | F, ¬N | 0 |

The true risk column assumes all 16 messages are equally likely, as in lesson 1, and counts the messages where the current conjunction and the truth disagree. Notice the direction of every mistake along the way: the learner only ever says keep to a junk message, never junk to a keep message. After example 3 it believes “F and L and not N”, stricter than the truth, so it misses 1000 and 1010. After example 6 it has the truth.

```python
names = "FLUN"
S = [("1100", 1), ("0110", 0), ("1110", 1), ("1001", 0),
     ("0010", 0), ("1010", 1), ("0101", 0), ("1011", 0)]
h = {(i, v) for i in range(4) for v in "01"}
def show(h):
    return " ".join(("" if v == "1" else "not ") + names[i]
                    for i, v in sorted(h))
for x, y in S:
    if y == 1:
        h = {(i, v) for i, v in h if x[i] == v}
        print(x, show(h))
print(all(all(x[i] == v for i, v in h) == bool(y) for x, y in S))
```

It prints `1100 F L not U not N`, `1110 F L not N`, `1010 F not N` and `True`: one line per junk example, and the final conjunction is consistent with all eight.

[[diagram:elimination]]

### Why it works

Let \(v_1, \dots, v_{m_+}\) be the positive messages of \(S\), in order. Define \(h_0\) as the conjunction of all \(2n\) literals, and \(h_{i+1}\) as \(h_i\) with every literal that \(v_{i+1}\) does not satisfy deleted. The learner returns \(h_{m_+}\).

**Claim (crossing out literals is an ERM in the realizable case).** If some conjunction is consistent with \(S\), then \(h_{m_+}\) is consistent with \(S\). It is computed in time \(O(nm)\).

*Proof.* We prove one thing by induction and then read off the claim.

The thing: for every \(i \le m_+\), \(h_i\) is the **most restrictive** conjunction that labels \(v_1, \dots, v_i\) positively. That is, \(h_i\) labels them 1, and every conjunction \(g\) that labels them 1 also labels 1 every message that \(h_i\) labels 1.

For \(i = 0\) there is nothing to label, and \(h_0\) labels no message 1 at all, so it is trivially the most restrictive.

Now suppose it holds for \(i\). Every literal left in \(h_{i+1}\) was satisfied by \(v_{i+1}\) (otherwise it would have been deleted), and by \(v_1, \dots, v_i\) (it survived those steps), so \(h_{i+1}\) labels \(v_1, \dots, v_{i+1}\) positively. Now take any conjunction \(g\) that labels \(v_1, \dots, v_{i+1}\) positively. If \(g\) is the all-negative formula it labels nothing 1, so it is not this \(g\); hence every literal of \(g\) is satisfied by each \(v_j\), for \(j \le i + 1\). A literal of \(g\) was never deleted, because deletion only removes literals some \(v_j\) falsifies. And every literal of \(g\) was in \(h_0\), which had them all. So the literals of \(g\) are a subset of the literals of \(h_{i+1}\). A message satisfying all of \(h_{i+1}\)’s literals satisfies all of \(g\)’s. That is the induction step.

Now the claim. By realizability, some conjunction \(f \in \mathcal H_C^n\) is consistent with \(S\), so it labels every positive message 1. By what we just proved with \(i = m_+\), every message \(h_{m_+}\) labels 1, \(f\) labels 1 too. Turn that around: every message \(f\) labels 0, \(h_{m_+}\) labels 0. The negative examples of \(S\) are labelled 0 by \(f\), so \(h_{m_+}\) labels them 0. The positive examples it labels 1 by construction. So \(L_S(h_{m_+}) = 0\), and \(h_{m_+}\) is a legal ERM hypothesis.

For the time: each example is read once, and for a positive one we check at most \(2n\) literals, so \(O(nm)\). \(\blacksquare\)

The induction is the kind of argument worth testing on everything small. Below, two thousand random realizable training sequences in the four-feature world: each time, the learner’s output must be consistent, and every consistent conjunction among all 82 must say junk wherever the output does.

```python
import itertools, random
rng = random.Random(0)
X = list(itertools.product((0, 1), repeat=4))
lits = [(i, v) for i in range(4) for v in (0, 1)]
C = [set(c) for k in range(5)
     for c in itertools.combinations(lits, k)
     if len({i for i, _ in c}) == k] + [set(lits)]
def h(c, x):
    return all(x[i] == v for i, v in c)
for _ in range(2000):
    f, m = rng.choice(C), rng.randint(0, 10)
    S = [(x, h(f, x)) for x in rng.choices(X, k=m)]
    out = set(lits)
    for x in (x for x, y in S if y):
        out = {(i, v) for i, v in out if x[i] == v}
    fits = [g for g in C if all(h(g, x) == y for x, y in S)]
    assert out in fits
    assert all(h(g, x) for g in fits for x in X if h(out, x))
print(len(C), "ok")
```

It prints `82 ok`.

```
CrossOutLiterals(S, n):
  h ← { x_1, ¬x_1, ..., x_n, ¬x_n }
  for i = 1..m:
    if y_i = 1:
      remove from h every literal that x_i does not satisfy
  return the conjunction of the literals in h
```

**Time** \(O(nm)\) to train, \(O(n)\) to predict. **Space** \(O(n)\): one bit per literal. For thirty features and the 752 examples the guarantee asks for, that is at most \(752 \times 60 = 45{,}120\) literal checks, against exhaustive search’s \(1.55 \times 10^{17}\) evaluations.

**Build it yourself.** Add `eliminate(S, n)` to your file from “ERM by brute force”, returning a frozenset of literals, and check it against the exhaustive search.

```python
import random

def eliminate(S, n): ...

if __name__ == "__main__":
    best, _ = exhaustive_erm(all_conjunctions(4), S)
    assert eliminate(S, 4) == best
    assert eliminate([], 3) == {(i, v) for i in range(3)
                                for v in (0, 1)}
    rng = random.Random(0)
    for _ in range(1000):
        f = rng.choice(all_conjunctions(6))
        xs = [tuple(rng.randint(0, 1) for _ in range(6))
              for _ in range(rng.randint(0, 30))]
        T = [(x, predict(f, x)) for x in xs]
        h = eliminate(T, 6)
        assert all(predict(h, x) == y for x, y in T)
```

**And without realizability?** Same story as rectangles. Unless \(P = NP\), no algorithm whose runtime is polynomial in \(m\) and \(n\) is guaranteed to find an ERM conjunction when no conjunction is consistent. Crossing out literals still runs, but one mislabelled junk message can delete a literal of the truth forever.

Try both learners in the lab. A hidden conjunction labels a stream of messages; you set the number of features and the number of examples, and watch the conjunction shrink while two counters run: the literal checks crossing out costs, and the evaluations exhaustive search would cost on the same examples.

[[lab]]

## Three kinds of junk

The conjunction filter ships, and for a while it holds. Then the security team walks in with a chart. There are now three kinds of junk: prize draws (“FREE” and a link), night blasts (unknown sender and sent at night), and fake invoices (a link from an unknown sender, no “FREE” in sight). Each kind is a conjunction. Junk is any of them.

That is a disjunction of conjunctions. A **3-term DNF formula** (disjunctive normal form) over \(\{0,1\}^n\) is a hypothesis

\[ h(x) = A_1(x) \vee A_2(x) \vee A_3(x), \]

where each \(A_i\) is a Boolean conjunction. It says 1 if at least one of the three conjunctions says 1, and 0 if all three say 0. Let \(\mathcal H_{3DNF}^n\) be the class of all of them. In the mail room, the security team’s rule is \((F \wedge L) \vee (U \wedge N) \vee (L \wedge U \wedge \neg F)\).

**Sample complexity.** Each term is one of at most \(3^n + 1\) conjunctions, so \(|\mathcal H_{3DNF}^n| \le (3^n + 1)^3\), and the same calculation as for conjunctions bounds the sample complexity of ERM in the realizable case by about \(3n \log(3/\delta)/\varepsilon\). Three times the conjunctions, still linear in \(n\). Statistically, this class is no harder than its parts.

Computationally, it is a different animal. To say how, we need two more definitions.

A learner is **proper** if the hypothesis it returns always belongs to the class \(\mathcal H\) we are learning. \(\mathrm{ERM}_{\mathcal H}\) is proper by definition: it returns a member of \(\mathcal H\).

**RP** (randomized polynomial time) is the class of decision problems for which some algorithm that may flip coins runs in time polynomial in the input size on every input, always answers no when the right answer is no, and answers yes with probability at least \(1/2\) when the right answer is yes. Clearly \(P \subseteq RP\), and it is known that \(RP \subseteq NP\). Nobody knows whether either inclusion is strict, but it is widely believed that NP is strictly larger than RP: NP-hard problems are believed to have no randomized polynomial algorithm.

**Fact (proper learning of 3-term DNF is hard).** Unless \(RP = NP\), there is no polynomial time algorithm that properly learns the sequence of 3-term DNF problems in which the \(n\)-th problem has dimension \(n\). In particular, \(\mathrm{ERM}_{\mathcal H_{3DNF}^n}\) is proper, so it cannot be implemented in polynomial time either.

The proof reduces graph 3-colouring, a classic NP-hard problem, to properly PAC learning 3-term DNF: an efficient proper learner would colour graphs. We won’t prove it here. The second problem at the end of this lesson builds the same kind of reduction for a related class, so you will see the machinery with your own hands.

Notice how small the change was. Conjunctions: easy. Three conjunctions joined by OR: hard. A slight generalization of the class has made ERM intractable, even in the realizable case where conjunctions were a blink.

## Change the representation

So the security team’s rule cannot be learned properly in polynomial time, unless something everyone believes false is true. Does that mean the filter cannot learn it?

Read the fact again. It forbids a fast learner that returns a 3-term DNF formula. Nobody said the filter must return one. What we actually want is a hypothesis with small true risk. If a hypothesis outside the class predicts just as well, who cares how it is written?

A learner that may return a hypothesis outside \(\mathcal H\), from some larger class, is called **representation independent** (lesson 3 mentioned the idea in passing). The plan: find a bigger class that contains every 3-term DNF and whose ERM is easy.

The bigger class comes from a piece of Boolean algebra. OR distributes over AND, just as multiplication distributes over addition, so

\[ A_1 \vee A_2 \vee A_3 = \bigwedge_{u \in A_1,\ v \in A_2,\ w \in A_3} (u \vee v \vee w), \]

where \(u \in A_1\) means “\(u\) is one of the literals of \(A_1\)”. The right side is an AND of clauses, each an OR of three literals: a **3-CNF formula** (conjunctive normal form).

**Claim (3-term DNF as 3-CNF).** For every message \(x\), the two sides agree.

*Proof.* If the left side is 1, some term is true, say \(A_1(x) = 1\). Then every literal \(u \in A_1\) is satisfied, so every clause \(u \vee v \vee w\) is satisfied through its \(u\), and the right side is 1. The same works for \(A_2\) through \(v\) and for \(A_3\) through \(w\).

If the left side is 0, every term is false, so each term has a literal that \(x\) does not satisfy: pick \(u^\star \in A_1\), \(v^\star \in A_2\), \(w^\star \in A_3\), all false at \(x\). The clause \(u^\star \vee v^\star \vee w^\star\) is one of the clauses on the right, and it is false, so the AND is 0. \(\blacksquare\)

There is one corner the proof steps around. A term with no literals at all is the empty conjunction, which is always true. Then the left side is always 1; on the right, the index set of the AND is empty, and an empty AND is 1 by convention. They still agree. The concept check asks you to make sure you believe this.

```python
import itertools, random
rng = random.Random(2)
X = list(itertools.product((0, 1), repeat=4))
lits = [(i, v) for i in range(4) for v in (0, 1)]
def sat(u, x):
    return x[u[0]] == u[1]
checked = 0
for _ in range(500):
    A = [rng.sample(lits, rng.randint(0, 3)) for _ in range(3)]
    for x in X:
        dnf = any(all(sat(u, x) for u in Ai) for Ai in A)
        cnf = all(sat(u, x) or sat(v, x) or sat(w, x)
                  for u in A[0] for v in A[1] for w in A[2])
        assert dnf == cnf
        checked += 1
print(checked)
```

It prints `8000`: 500 random triples of terms (some empty, some containing a feature and its negation), each checked on all 16 messages.

### New features for old

Now the trick that turns this identity into an algorithm. There are \(2n\) literals, so \((2n)^3\) ordered triples \((u, v, w)\) of them. Define a map

\[ \psi : \{0,1\}^n \to \{0,1\}^{(2n)^3}, \qquad \psi(x)_{(u,v,w)} = \mathbb 1[x \text{ satisfies } u \vee v \vee w]. \]

Each coordinate of \(\psi(x)\) is a new yes/no feature of the message: “does it satisfy this clause?”. With four features there are 512 of them; with thirty, 216,000.

In the new features, a 3-CNF formula is a conjunction. The formula \(\bigwedge (u \vee v \vee w)\) over the triples in the identity says 1 at \(x\) exactly when every one of those coordinates of \(\psi(x)\) is 1. So for every 3-term DNF \(h\) over \(\{0,1\}^n\) there is a conjunction \(g\) over \(\{0,1\}^{(2n)^3}\) with \(g(\psi(x)) = h(x)\) for every \(x\): the same truth table.

That makes the learner almost embarrassing to write. Map every example \((x, y)\) to \((\psi(x), y)\). If the labels come from a 3-term DNF, the mapped examples are labelled by a conjunction, so realizability holds in the new space. Run crossing out literals there, and return \(x \mapsto g(\psi(x))\).

[[diagram:nested]]

**Runtime.** Computing \(\psi(x)\) takes \(O(n^3)\) operations. Crossing out literals in dimension \(d = (2n)^3\) takes \(O(dm) = O(n^3 m)\). Predicting takes \(O(n^3)\), within the training budget, so the lazy-learner rule is respected.

**Sample complexity.** It is the sample complexity of conjunctions in dimension \(d = (2n)^3\): at most \(d \log(3/\delta)/\varepsilon = 8 n^3 \log(3/\delta)/\varepsilon\), of order \(n^3 \log(1/\delta)/\varepsilon\). Polynomial in \(n\), \(1/\varepsilon\) and \(1/\delta\), so the whole procedure is efficient.

That is the theorem of this section, and it is worth stating plainly. **3-term DNF is efficiently learnable** in the realizable case, by ERM over a larger class (conjunctions over the clause features, equivalently 3-CNF formulas), even though, unless \(RP = NP\), it is not efficiently learnable by any proper learner. Hardness of ERM over \(\mathcal H\) does not mean \(\mathcal H\) is hard to learn.

Why did it work? We swapped a class where ERM is hard for a larger class with more structure. The larger class is easier to search, because its ERM is crossing out literals, and it costs more examples, because it is larger. That is the trade, and you will feel it in the lab.

Let’s watch it learn the security team’s rule, with four noise features added: \(n = 8\), 256 equally likely messages of which 128 are junk, and 100 random examples.

```python
import itertools, random
n = 8
lits = [(i, v) for i in range(n) for v in (0, 1)]
def psi(x):
    s = [x[i] == v for i, v in lits]
    return [int(a or b or c) for a in s for b in s for c in s]
def junk(x):
    A1, A2 = x[0] and x[1], x[2] and x[3]
    return int(A1 or A2 or (x[1] and x[2] and not x[0]))
X = list(itertools.product((0, 1), repeat=n))
S = random.Random(1).choices(X, k=100)
h = {(j, v) for j in range((2 * n) ** 3) for v in (0, 1)}
for z in (psi(x) for x in S if junk(x)):
    h = {(j, v) for j, v in h if z[j] == v}
def h_S(x):
    z = psi(x)
    return int(all(z[j] == v for j, v in h))
print(len(h), len(set(S)))
print(sum(h_S(x) != junk(x) for x in X) / len(X))
```

It prints `942 81` and `0.0`. The 100 examples cover 81 distinct messages, the learned conjunction keeps 942 of the 8,192 literals over the clause features, and its true risk is 0 on all 256 messages, including the 175 it never saw. For comparison, lesson 1’s memorizer, given the same 100 examples, would be wrong on 0.324 of all messages. With only the first 25 examples, the 3-CNF learner is wrong on 0.402: a larger class needs its examples.

```
Learn3TermDNF(S, n):
  T ← [ (ψ(x_i), y_i) for i = 1..m ]
  g ← CrossOutLiterals(T, (2n)^3)
  return h(x) = g(ψ(x))
```

**Time** \(O(n^3 m)\) to train, \(O(n^3)\) to predict. **Space** \(O(n^3)\) for the literals of \(g\), plus \(O(n^3)\) for one mapped message at a time.

**Build it yourself.** Write `psi(x)` (ordered triples, with repetition, so \((2n)^3\) coordinates) and `learn_3dnf(S)`, returning the hypothesis and its set of literals over the clause features.

```python
import itertools, random

n = 8
X = list(itertools.product((0, 1), repeat=n))

def junk(x):
    return int((x[0] and x[1]) or (x[2] and x[3])
               or (x[1] and x[2] and not x[0]))

S = [(x, junk(x)) for x in random.Random(1).choices(X, k=100)]

def psi(x): ...
def learn_3dnf(S): ...

if __name__ == "__main__":
    assert len(psi(X[0])) == 4096
    h_S, h = learn_3dnf(S)
    assert len(h) == 942
    assert all(h_S(x) == y for x, y in S)
    assert sum(h_S(x) != junk(x) for x in X) == 0
    h_small, _ = learn_3dnf(S[:25])
    assert sum(h_small(x) != junk(x) for x in X) == 103
```

## Fix the bug

Back to the real mail system: thirty features, and the security team’s three kinds of junk. Your manager wants the usual promise, \(\varepsilon = 0.05\) and \(\delta = 0.01\), and two budgets. Labels: the company’s junk and keep clicks for a year, about five million. Compute: one week on the machine that does a billion operations a second, about \(6 \times 10^{14}\) operations.

You have three ways to write down the junk rule. You can learn it as a 3-term DNF, properly, by exhaustive search. You can learn it as a 3-CNF, through the clause features. Or you can drop all bias and learn any rule at all, lesson 1’s memorizer, which is fast since it only stores examples in a hash table. Here are the guarantee’s label counts and the runtime in seconds for each, at a billion operations a second.

```python
import math
n, eps, delta, rate = 30, 0.05, 0.01, 1e9
d = (2 * n) ** 3
classes = {"3-term DNF": (3 * n * math.log(3), 3 ** (3 * n)),
           "3-CNF": (d * math.log(3), 2 * d),
           "all rules": (2 ** n * math.log(2), n)}
for name, (log_size, work) in classes.items():
    m = math.ceil((log_size + math.log(1 / delta)) / eps)
    print(name, m, f"{m * work / rate:.2g}")
```

It prints `3-term DNF 2070 1.8e+37`, `3-CNF 4746098 2.1e+03` and `all rules 14885222452 4.5e+02`. Read it as a table of two prices.

| representation | labels the guarantee needs | runtime |
|---|---|---|
| 3-term DNF, exhaustive search | 2,070 | \(1.8 \times 10^{37}\) s, about \(6 \times 10^{29}\) years |
| 3-CNF, crossing out literals | 4,746,098 | about 35 minutes |
| any rule (memorizer) | about \(1.5 \times 10^{10}\) | about 8 minutes |

The proper class is a bargain in labels and impossible in time. The unbiased class is fast and impossible in labels. The 3-CNF class pays about two thousand times more labels than the proper class, and gets a runtime you can wait for. Only one row fits both budgets.

In the lab, you pick a representation and a number of labels for this week’s junk rule. Two meters show what the guarantee needs and what the search costs. Your goal: a filter whose true risk is at most 0.05 with probability at least 0.99, within both budgets. Not by luck: the guarantee must cover it. A fourth option is on the menu too, a single conjunction, cheap on both meters. Think about what happens to its guarantee before you pick it.

[[lab-act2]]

## When no representation helps

The 3-CNF trick shows that hardness of ERM is not hardness of learning. So how could anyone ever prove that a learning problem is hard, for every learner, whatever it returns?

One route goes through cryptography. In a sense, cryptography is the opposite of learning. A learner tries to uncover the rule behind the examples it sees. A cryptographer tries to make sure that nobody can uncover a secret, even with access to partial information about it. So a result saying some system is secure should translate into a result saying some task is unlearnable.

There is a catch. Nobody can currently prove that any cryptographic protocol is unbreakable. Even \(P \ne NP\) is not enough to prove it, although for most common cryptographic scenarios it is necessary. So cryptography starts from **cryptographic assumptions**, and the more these are relied on, the more we trust them, or at least, the more we believe algorithms that refute them are hard to find. Hardness of learning built this way is conditional on the same assumptions.

**Definition (one-way function).** A one-way function is a function \(f : \{0,1\}^n \to \{0,1\}^n\) (formally a sequence of them, one for each \(n\)) that is easy to compute and hard to invert: \(f\) can be computed in time \(\mathrm{poly}(n)\), but for every randomized polynomial time algorithm \(A\) and every polynomial \(p(\cdot)\),

\[ \mathbb P\left[ f(A(f(x))) = f(x) \right] < \frac{1}{p(n)}, \]

where the probability is over \(x\) drawn uniformly from \(\{0,1\}^n\) and over the coins of \(A\). In words: given \(f(x)\), no fast algorithm can find any preimage of it, except with a vanishing chance.

**Definition (trapdoor one-way function).** A one-way function \(f\) is a trapdoor function if for some polynomial \(p\), for every \(n\) there is a bit string \(s_n\) of length at most \(p(n)\), the **secret key**, and a polynomial time algorithm that, for every \(n\) and every \(x \in \{0,1\}^n\), on input \((f(x), s_n)\) outputs \(x\). Hard to invert, unless you hold the key; then easy. Such functions are parameterized by their keys.

Here is a toy version, the textbook RSA function on tiny numbers (far too small to be secure, but the shape is right): computing \(f\) is one modular power, inverting with the key is another, and inverting without it is a search.

```python
N, e, key = 3233, 17, 2753
def f(x):
    return pow(x, e, N)
def invert_with_key(c):
    return pow(c, key, N)
def invert_by_search(c):
    for steps, x in enumerate(range(N), start=1):
        if f(x) == c:
            return x, steps
c = f(2024)
print(c, invert_with_key(c), invert_by_search(c))
```

It prints `1414 2024 (2024, 2025)`. The key inverts in one step; the search needed 2,025. A real key has thousands of bits, and the search space doubles with every bit.

[[diagram:trapdoor]]

**Claim (trapdoor functions give an unlearnable class).** Let \(\mathcal F_n\) be a family of trapdoor functions over \(\{0,1\}^n\), computable by a fixed polynomial time algorithm from the key and the input. Consider the class of their inverses, \(\mathcal H_{\mathcal F_n} = \{ f^{-1} : f \in \mathcal F_n \}\). Its sample complexity is polynomial in \(n\), yet no efficient learner exists for it, as long as the functions really are one-way.

*Argument.* The statistical half first. Each \(f^{-1}\) is determined by a secret key of at most \(p(n)\) bits, so the class can be parameterized by keys, and \(|\mathcal H_{\mathcal F_n}| \le 2^{p(n)}\) (counting keys up to that length gives at most \(2^{p(n)+1}\), the same order). By the finite-class bound, its sample complexity is of order \(p(n) + \log(1/\delta)\) over \(\varepsilon\): polynomial in \(n\). Information is not the obstacle.

Now the computational half. Suppose, for contradiction, that \(L\) learns \(\mathcal H_{\mathcal F_n}\) efficiently. Here is how to invert some \(f \in \mathcal F_n\) without its key. Draw a polynomial number of strings \(x_1, \dots, x_m\) uniformly from \(\{0,1\}^n\), and compute \(f(x_1), \dots, f(x_m)\); that is polynomial time, because \(f\) is easy to compute. The pairs \((f(x_i), x_i)\) are a training sequence for the target \(f^{-1}\), drawn i.i.d. from the distribution of \(f(x)\) with \(x\) uniform. Feed them to \(L\) with, say, \(\varepsilon = \delta = 1/4\). With probability at least \(3/4\), it returns \(h\) with \(\mathbb P_x[h(f(x)) \ne x] \le 1/4\), with respect to that same distribution. Now take the challenge \(f(x)\), with \(x\) fresh and uniform, and answer \(h(f(x))\). With probability at least \(3/4 \cdot 3/4 = 9/16\), the answer is \(x\) itself, and then \(f(h(f(x))) = f(x)\).

That procedure runs in polynomial time and inverts \(f\) with probability at least \(9/16\), for every \(n\). The one-way property, with the constant polynomial \(p(n) = 2\), says the probability must be below \(1/2\). Contradiction. So no efficient learner exists. \(\blacksquare\)

Two remarks keep this honest. First, the argument assumes trapdoor one-way functions exist, which is a cryptographic assumption, not a theorem. Second, notice what the key did: it never entered the inversion. It only guaranteed the class was small, so that the hardness is purely computational. Through reductions, the same style of argument shows that the class of functions computed by small Boolean circuits is not efficiently learnable, even in the realizable case.

## What people get wrong

**“ERM over this class is NP-hard, so the class cannot be learned efficiently.”** Hardness of ERM rules out efficient *proper* learning (under \(RP \ne NP\)), nothing more. 3-term DNF is the counterexample: hard for any proper learner, easy through 3-CNF.

**“Realizable or agnostic, it’s the same problem.”** Statistically, yes: a class is learnable in both settings exactly when its VC dimension is finite. Computationally, the difference is immense. For rectangles and conjunctions, realizable ERM takes \(O(nm)\) time, and agnostic ERM is NP-hard.

**“A smaller class is always cheaper.”** Cheaper in labels, yes. But 3-term DNF is smaller than 3-CNF and, for proper learners, far more expensive in time. The two prices can point in opposite directions.

**“The runtime should be measured against the number of examples.”** A learner can always ignore surplus examples, so measuring against \(m\) would reward waste. Runtime is measured against \(n\), \(1/\varepsilon\) and \(1/\delta\), along a sequence of problems.

**“My learner trains in constant time.”** Check its hypothesis. If predicting one label runs a search, the work has only moved. The definition bounds prediction by the training budget.

## Recap and the next bug

1. Every class has two prices: the sample complexity (how many examples) and the runtime (how much computation). Lessons 2 to 7 priced the first; this lesson prices the second.
2. Runtime is measured in operations, asymptotically, along a sequence of problems, as a function of \(n\), \(1/\varepsilon\) and \(1/\delta\). A learner is efficient when that function is polynomial, and its hypothesis must predict within the same budget.
3. Exhaustive ERM over a finite class costs \(O(|\mathcal H| \, m)\): fine for small classes, exponential when \(\log |\mathcal H_n|\) grows like \(n\), although the sample complexity stays polynomial.
4. In the realizable case, ERM is fast for rectangles (the tightest box) and conjunctions (crossing out literals), both \(O(nm)\). In the agnostic case it is NP-hard for both; for rectangles in a fixed dimension, subset search runs in \(m^{O(n)}\).
5. Proper learning of 3-term DNF is hard unless \(RP = NP\), but learning it through 3-CNF is efficient: \(O(n^3 m)\) time for \(O(n^3)\) times more examples. Changing the representation trades labels for time.
6. To show a problem is hard for every learner, cryptographic assumptions help: if trapdoor one-way functions exist, their inverses form a class that is small but not efficiently learnable.

Your filter learns the three kinds of junk in about half an hour. Then the analytics team sends a request. They have messages described by real-valued scores, dozens of them, and they want the filter to weigh them all: a little for each capital letter, more for each link, less for a known sender, and junk when the total passes a threshold. That is a huge, infinite class with real geometry. Rectangles had geometry too, and their agnostic ERM was NP-hard. Conjunctions were fast only because the problem was realizable. Is there a rich class, with real geometry, where finding the best hypothesis is fast? That is the next lesson: linear predictors.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. You run crossing out literals on a training sequence that contains no junk example at all. What does it return, what does that hypothesis say about every message, and is it consistent with the training sequence? Now suppose the sequence is not realizable: one message, 1100, appears twice, once labelled junk and once keep. Does the learner notice? What is the empirical risk of what it returns?
2. In the proof that a 3-term DNF equals the 3-CNF formula \(\bigwedge (u \vee v \vee w)\), the second direction picks a false literal from each term. Which step breaks if one of the terms is the empty conjunction, and why is the identity still true? What if one term is the all-negative formula \(x_1 \wedge \neg x_1\)?
3. Learning a 3-term DNF through 3-CNF costs \((2n)^3 \log 3\) in the logarithm of the class size, and the memorizer costs \(2^n \log 2\). For \(n = 4\), which needs fewer examples according to the guarantee? Find the smallest \(n\) from which the 3-CNF route needs fewer. What does your answer say about the 0.402 true risk the 3-CNF learner had after 25 examples in the eight-feature world?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (intervals, agnostic, in quadratic time).** Intervals on the line are rectangles with \(n = 1\): \(h_{a,b}(x) = 1\) exactly when \(a \le x \le b\). Subset search over pairs of examples gives an agnostic ERM in time \(O(m^3)\). Give one that runs in time \(O(m^2)\). (Think of dynamic programming.)
- **Problem 2 (intersections of halfspaces are hard to fit).** A halfspace in \(\mathbb R^n\) labels \(x\) by the sign of \(\langle w, x \rangle + b\), for a weight vector \(w \in \mathbb R^n\) and a number \(b\) (lesson 9 studies them). Let \(\mathcal H_k^n\) be the intersections of \(k\) halfspaces. Given a graph with vertices \(v_1, \dots, v_n\), build a training sequence: for each vertex \(v_i\), the unit vector \(e_i\) labelled negative; for each edge \((v_i, v_j)\), the point \((e_i + e_j)/2\) labelled positive. Prove that some \(h \in \mathcal H_k^n\) has zero empirical risk on it exactly when the graph can be coloured with \(k\) colours so that no edge joins two vertices of the same colour, and conclude that ERM over \(\mathcal H_k^n\) is NP-hard for every \(k \ge 3\).
- **Problem 3 (harder: efficient proper learning puts ERM in RP).** Suppose \(A\) properly PAC learns \(\mathcal H\) in time polynomial in \(n\), \(1/\varepsilon\) and \(1/\delta\). Show that the decision problem “is there an \(h \in \mathcal H\) with zero empirical risk on this \(S\)?” is in RP. Conclude that if that problem is NP-hard, there is no polynomial time proper PAC learner for \(\mathcal H\) unless \(NP = RP\).
