# Neural Networks

## The junk nobody had a feature for

Your invoice filter from lesson 19 works, and it comes with a certificate. But remember what it stood on: two scores that somebody invented, picked out of thirty that somebody else designed. Look back over the whole course and the pattern is the same. The one-feature hypotheses of lesson 1 looked at bits someone chose (“FREE”, a link, an unknown sender). The halfspaces of lesson 9 weighed word counts. The trees of lesson 18 asked questions about features a person had written down. Every filter so far is only as good as the imagination of whoever built its features.

This week that imagination ran out. A new wave of junk is arriving, and every filter you own lets it through. You pull a few hundred of the messages and stare at them. No word gives them away. No sender list catches them. Then a colleague plots two numbers the mail server records for every message anyway, before anyone designs anything: the time of day it was sent and its length. We’ll call these **raw signals**, and scale each one to lie between −1 and 1, so a message becomes a point \(u = (u_1, u_2)\) in a square. Here \(u_1\) is the send time (−1 is midnight at the start of the day, 1 is midnight at the end) and \(u_2\) is the length (−1 is the shortest message, 1 the longest).

On that plot the new junk sits in a slanted triangle, with corners at \((-0.6, -0.6)\), \((0.6, -0.2)\) and \((-0.2, 0.6)\). It covers 16% of the square. Inside the triangle, everything is junk; outside, everything is keep, apart from the 2% of messages your users label carelessly.

Why does every old filter miss it? A hand-made feature such as “sent in the morning” or “longer than average” cuts the square with a vertical or horizontal line. A halfspace on the two raw signals, lesson 9’s tool, cuts it with one slanted line. A triangle is neither: it is the region inside *three* lines at once. The filter you would need has to invent three new features, “on the correct side of line 1”, “of line 2”, “of line 3”, and then combine them. Nobody thought of those features, and next month’s junk will need different ones.

So here is this lesson’s question. Could a filter build its own features, straight from the raw signals, and learn the junk rule on top of them, all from labelled examples? That is what a neural network does. By the end you will know what such a network can express, how many labels it costs, why training it is hard in the worst case, and how to train it anyway with stochastic gradient descent and an algorithm called backpropagation.

## Features that build themselves

Let’s build the triangle detector by hand first, with real numbers, and only then give the parts their names.

Start with one of the three lines. The first edge of the triangle is where \(3u_1 - u_2 + 1.2\) equals 0, and the triangle lies on the side where that number is positive. So here is a tiny machine: take the two raw signals, multiply them by 3 and −1, add 1.2, and answer +1 if the total is positive and −1 if it is not. On the message \(u = (0.3, -0.1)\), the total is \(3 \cdot 0.3 - (-0.1) + 1.2 = 0.9 + 0.1 + 1.2 = 2.2\), which is positive, so the machine says +1. That machine is a halfspace, exactly as in lesson 9, and its answer is a brand new feature of the message: “on the inner side of edge 1”.

Build two more machines for the other two edges, with weights \((-1, 3)\) and a constant 1.2, and weights \((-1, -1)\) and a constant 0.4. Now each message has three new features, each +1 or −1. A message is in the triangle exactly when all three are +1. One more machine finishes the job: add the three features, subtract 2, and answer by the sign. If all three are +1 the total is \(3 - 2 = 1\), positive. If even one is −1, the total is at most \(1 + 1 - 1 - 2 = -1\), negative. So the last machine computes “all three are +1”, the logical AND.

Before you run it, predict the answer for \((0.5, 0.5)\): a message sent in the afternoon, a bit longer than average. Which of the three features does it fail?

```python
def sign(a):
    return 1 if a > 0 else -1

hidden = [(3, -1, 1.2), (-1, 3, 1.2), (-1, -1, 0.4)]

def network(u1, u2):
    o = [sign(w1 * u1 + w2 * u2 + b) for w1, w2, b in hidden]
    return o, sign(sum(o) - 2)

for u in [(0, 0), (0.3, -0.1), (-0.4, -0.4),
          (0.5, 0.5), (-0.8, 0), (-0.9, -0.9)]:
    print(u, *network(*u))
```

It prints

```
(0, 0) [1, 1, 1] 1
(0.3, -0.1) [1, 1, 1] 1
(-0.4, -0.4) [1, 1, 1] 1
(0.5, 0.5) [1, 1, -1] -1
(-0.8, 0) [-1, 1, 1] -1
(-0.9, -0.9) [-1, -1, 1] -1
```

Read the rows as: the message, its three new features, and the final label (+1 junk, −1 keep). The first three messages sit inside the triangle and pass all three tests. The message \((0.5, 0.5)\) fails the third: \(-0.5 - 0.5 + 0.4 = -0.6\). It is past the long edge. The message \((-0.8, 0)\), early and of middling length, fails the first test, and \((-0.9, -0.9)\), very early and very short, fails two.

Notice what happened. Four tiny machines, each doing nothing smarter than “weigh, add, take the sign”, did what no single halfspace can. The first three turned two raw signals into three useful features, and the last one combined them. Nobody designed “inner side of edge 1” as a mail feature. It came out of the weights. And if the weights could be *learned* from labelled messages, the features would build themselves.

[[diagram:hidden-features]]

Try it in the lab before we formalize anything. You drag three lines across the square of raw signals; each line is one feature machine. A last machine combines them, and you choose its weights. Predict first how many lines the triangle needs, then find out what a second triangle costs.

[[lab]]

## The layered graph

Now the names. Each tiny machine is a **neuron**. A neuron receives some numbers, multiplies each by its own **weight**, adds them up, and passes the total through a fixed scalar function \(\sigma: \mathbb R \to \mathbb R\), read “sigma, a function from real numbers to real numbers”, called the **activation function**. We will use three activation functions. Here they are in words first, with the total \(a = 0.4\) as an example:

- the **sign** function answers +1 when \(a\) is positive and −1 otherwise, so \(\operatorname{sign}(0.4) = 1\) and \(\operatorname{sign}(0) = -1\) (the same convention as lesson 9);
- the **threshold** function \(\mathbb 1[a > 0]\) answers 1 when \(a\) is positive and 0 otherwise (recall from lesson 1 that \(\mathbb 1[\text{statement}]\) is 1 when the statement is true and 0 when it is false), so it gives 1 at 0.4;
- the **sigmoid** function \(\sigma(a) = 1/(1 + e^{-a})\) is a smooth version of the threshold: it climbs from 0 to 1, is exactly \(1/2\) at \(a = 0\), and at \(a = 0.4\) gives \(1/(1 + e^{-0.4}) \approx 0.599\). It borrows the letter \(\sigma\), so whenever \(\sigma\) gets a formula, it means the sigmoid.

```python
import math

def sign(a):
    return 1 if a > 0 else -1

def threshold(a):
    return 1 if a > 0 else 0

def sigmoid(a):
    return 1 / (1 + math.exp(-a))

for a in (-2.0, 0.0, 0.4, 3.0):
    print(a, sign(a), threshold(a), round(sigmoid(a), 3))
```

It prints `-2.0 -1 0 0.119`, `0.0 -1 0 0.5`, `0.4 1 1 0.599` and `3.0 1 1 0.953`. Far from 0, the sigmoid is nearly the threshold; near 0 it bends smoothly instead of jumping. That smoothness will matter when we train.

The neurons are wired together. Picture a diagram where each neuron is a dot and each wire is an arrow from the neuron whose output it carries to the neuron that receives it. In graph language this is a **directed graph** \(G = (V, E)\): \(V\) is the set of dots, called nodes (here, neurons), and \(E\) is the set of arrows, called **edges**, each an ordered pair \((v, v')\) meaning “the output of \(v\) feeds \(v'\)”. We only allow graphs with no cycles, where you can never follow arrows around a loop back to where you started: a **feedforward** network. The weights form a **weight function** \(w: E \to \mathbb R\), one real number per edge.

To keep the bookkeeping simple, the neurons come in **layers**. The set \(V\) splits into disjoint, nonempty groups \(V_0, V_1, \dots, V_T\), and every edge goes from some layer \(V_{t-1}\) to the next one, \(V_t\). Our triangle detector has three layers:

- \(V_0\), the **input layer**, has one neuron per raw signal plus one **constant neuron**. For an input with \(n\) signals, \(V_0\) has \(n + 1\) neurons: neuron \(i\) simply outputs signal \(x_i\), and the last one always outputs 1. That constant is how the numbers 1.2, 1.2 and 0.4 entered the triangle machines: they are the weights on the edges from the constant neuron.
- \(V_1\) holds the three feature machines and, again, a constant neuron. It is a **hidden layer**: its outputs are neither the input nor the answer. In general, layers \(V_1, \dots, V_{T-1}\) are the hidden layers.
- \(V_T\), the **output layer**, here a single neuron whose output is the network’s answer.

A small word on the constant neuron of a hidden layer. It has no incoming edges, and a neuron with no incoming edges receives the total 0, so by the definition it outputs the constant \(\sigma(0)\): −1 for sign, \(1/2\) for the sigmoid. Any nonzero constant does the job of a bias, because you can scale or negate its outgoing weights. For readability we will write every hidden constant neuron as outputting 1. In the triangle detector, the “−2” in the last machine is the weight −2 on the edge from that constant.

The number of layers after the input, \(T\), is the **depth** of the network. The number of neurons, \(|V|\), is its **size** (the bars \(|\cdot|\) mean “how many elements”). The number of neurons in the widest layer, \(\max_t |V_t|\), is its **width**. The triangle detector has depth 2, size \(3 + 4 + 1 = 8\) and width 4.

Now let’s write down the calculation every neuron does. Name the \(i\)-th neuron of layer \(t\) as \(v_{t,i}\), and its output, when the network is fed the input \(x\), as \(o_{t,i}(x)\). Feeding the input means \(o_{0,i}(x) = x_i\) for the signals and \(o_{0,n+1}(x) = 1\) for the constant. Then the layers are computed one after another, bottom to top. The total that neuron \(v_{t+1,j}\) receives, called its **input** and written \(a_{t+1,j}(x)\), and its output are

\[ a_{t+1,j}(x) = \sum_{r \,:\, (v_{t,r},\, v_{t+1,j}) \in E} w\big((v_{t,r}, v_{t+1,j})\big)\, o_{t,r}(x), \qquad o_{t+1,j}(x) = \sigma\big(a_{t+1,j}(x)\big). \]

Read it from the inside out.

- \(o_{t,r}(x)\) is the output of neuron \(r\) of the layer below, already computed.
- \((v_{t,r}, v_{t+1,j})\) is the edge from that neuron to ours, and \(w(\cdot)\) of that edge is its weight.
- The sum runs over the \(r\) such that \((v_{t,r}, v_{t+1,j}) \in E\), read “the edge from \(v_{t,r}\) to \(v_{t+1,j}\) is in the graph”: only neurons wired to ours count.
- \(a_{t+1,j}(x)\) is therefore the weighted sum of the outputs wired into \(v_{t+1,j}\), and \(o_{t+1,j}(x)\) applies the activation to it.

In plain words: **each neuron adds up the outputs wired into it, each times the weight on its wire, and passes the total through the activation function.** In the triangle detector, take neuron \(v_{1,1}\) and the message \((0.3, -0.1)\): the outputs below are \(0.3\), \(-0.1\) and the constant 1, the weights on the three wires are 3, −1 and 1.2, so \(a_{1,1} = 3 \cdot 0.3 + (-1)(-0.1) + 1.2 \cdot 1 = 2.2\) and \(o_{1,1} = \operatorname{sign}(2.2) = 1\). That was the first row of the table above.

Here is the same formula with the sigmoid, on a smaller network we will reuse when we train: two raw signals, two hidden sigmoid neurons plus a constant, one sigmoid output. Each list `W0[i]` holds the weights on the wires into hidden neuron \(i\) (from \(u_1\), \(u_2\) and the constant), and `W1` the weights into the output (from the two hidden neurons and the hidden constant).

```python
import math

def sigma(a):
    return 1 / (1 + math.exp(-a))

W0 = [[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]]
W1 = [[2.0, -1.0, 0.5]]
x, y = [0.5, -0.5], 1.0

o0 = x + [1.0]
a1 = [sum(w * o for w, o in zip(row, o0)) for row in W0]
o1 = [sigma(a) for a in a1] + [1.0]
a2 = sum(w * o for w, o in zip(W1[0], o1))
o2 = sigma(a2)
print([round(a, 4) for a in a1], [round(o, 4) for o in o1])
print(round(a2, 4), round(o2, 4), round((o2 - y) ** 2 / 2, 5))
```

It prints `[1.0, -0.5] [0.7311, 0.3775, 1.0]` and then `1.5846 0.8299 0.01448`. Line by line: `o0` is \(o_0\), the input layer with the constant appended; `a1` is the list of \(a_{1,j}\), each a weighted sum; `o1` applies \(\sigma\) and appends the hidden constant; `a2` and `o2` repeat the step for the output. (Ignore the last number for now; it is the loss, which we meet when we train.)

[[diagram:layered-graph]]

## A network as a hypothesis class

Once the graph \((V, E)\), the activation \(\sigma\) and the weights \(w\) are fixed, the network is a function: it takes the \(n\) signals of the input layer (that is \(|V_0| - 1\) numbers, since one input neuron is the constant) and returns the outputs of the top layer. Write that function \(h_{V,E,\sigma,w}\).

The triangle detector and an “either” detector can share a graph. Keep the three feature machines, but give the last machine the weight +2 on the constant instead of −2. Now the total is \(o_1 + o_2 + o_3 + 2\), which is positive unless all three features are −1. Same wires, different weights, different hypothesis. That is the whole idea of learning a network: the wiring is fixed in advance, and learning chooses the weights.

The triple \((V, E, \sigma)\) is called the **architecture** of the network. Each architecture defines a hypothesis class (lesson 1’s word for the set of hypotheses a learner may return, fixed before it sees data):

\[ \mathcal H_{V,E,\sigma} = \{ h_{V,E,\sigma,w} \;:\; w \text{ is a mapping from } E \text{ to } \mathbb R \}. \]

Reading it: the braces with a colon are set-builder notation, “the set of all \(h_{V,E,\sigma,w}\) such that \(w\) is a mapping from \(E\) to \(\mathbb R\)”. The subscripts \(V, E, \sigma\) are fixed for the whole class; only \(w\) varies. In plain words: **the class of a network is every function you can get from one fixed wiring and one fixed activation by choosing the weights.** The triangle architecture has \(9 + 4 = 13\) edges (three wires into each of three feature neurons, four into the output), so a hypothesis in its class is named by 13 real numbers.

```python
def sign(a):
    return 1 if a > 0 else -1

def h(w, u):
    hidden, out = w
    o = [sign(a * u[0] + b * u[1] + c) for a, b, c in hidden]
    o = o + [1]
    return sign(sum(v * x for v, x in zip(out, o)))

hidden = [(3, -1, 1.2), (-1, 3, 1.2), (-1, -1, 0.4)]
w_and = (hidden, (1, 1, 1, -2))
w_or = (hidden, (1, 1, 1, 2))
for u in [(0, 0), (0.5, 0.5), (-0.9, -0.9)]:
    print(u, h(w_and, u), h(w_or, u))
print(sum(len(r) for r in hidden) + len(w_and[1]))
```

It prints `(0, 0) 1 1`, `(0.5, 0.5) -1 1`, `(-0.9, -0.9) -1 1` and `13`. The function `h` is the architecture; `w_and` and `w_or` are two weight functions; the two columns are two members of the same class. The last line counts the edges, which is the number of weights.

Choosing a class means choosing an inductive bias, so lesson 5’s three questions apply: how close can the class come to the truth (its approximation error), how many labels does it need (its estimation error), and can we actually find good weights (the computation)? The rest of the lesson takes them in that order.

## What a network can express

Let’s start with the most generous question: which Boolean functions can a sign network compute? A **Boolean function** here is any rule that takes \(n\) bits, each written as +1 or −1, and returns +1 or −1: a function from \(\{\pm 1\}^n\) to \(\{\pm 1\}\). There are \(2^n\) possible inputs, and a function picks one of two outputs for each, so there are \(2^{2^n}\) such functions: 256 when \(n = 3\). This is not as narrow as it sounds. A computer stores every real number in \(b\) bits, so whatever it computes from \(n\) real numbers is really a Boolean-valued function of \(nb\) bits.

**Claim (depth 2 computes every Boolean function).** For every \(n\) there is a graph \((V, E)\) of depth 2 such that \(\mathcal H_{V,E,\operatorname{sign}}\) contains every function from \(\{\pm 1\}^n\) to \(\{\pm 1\}\).

The idea is the one the triangle used, taken to the extreme: give the hidden layer one neuron per input that should be labelled +1, each neuron firing on its own input and nowhere else, and let the output neuron fire if any of them fires.

*Proof.* Take \(|V_0| = n + 1\), \(|V_1| = 2^n + 1\) and \(|V_2| = 1\), with every possible edge between adjacent layers. Fix a function \(f\), and let \(u_1, \dots, u_k\) be the inputs it labels +1.

First, a fact about \(\pm 1\) vectors. The inner product \(\langle x, u \rangle = \sum_{i=1}^n x_i u_i\) adds \(n\) terms, each \(+1\) where \(x\) and \(u\) agree and \(-1\) where they differ. If \(x = u\), it equals \(n\). If \(x \neq u\), at least one term is −1, so it is at most \((n - 1) - 1 = n - 2\).

So for each \(i\) the neuron

\[ g_i(x) = \operatorname{sign}\big(\langle x, u_i \rangle - n + 1\big) \]

sees the total \(n - n + 1 = 1\) when \(x = u_i\), and at most \(n - 2 - n + 1 = -1\) otherwise. It outputs +1 exactly on \(u_i\). Its weights are the coordinates of \(u_i\) on the edges from the signals and \(-n + 1\) on the edge from the constant. Give hidden neurons \(1, \dots, k\) these weights. Give the remaining hidden neurons weights that make them harmless (for instance all zero, so they output \(\operatorname{sign}(0) = -1\), and weight 0 on their edges into the output), and let the last hidden neuron be the constant 1, using weight 1 on its edge from the input constant and 0 elsewhere.

Second, the output neuron computes

\[ f(x) = \operatorname{sign}\Big( \sum_{i=1}^k g_i(x) + k - 1 \Big). \]

If \(x\) is one of the \(u_i\), exactly one \(g_i\) is +1 and the other \(k - 1\) are −1, so the sum is \(1 - (k - 1) = 2 - k\) and the total is \(2 - k + k - 1 = 1\): output +1. If \(x\) is none of them, all \(k\) are −1, and the total is \(-k + k - 1 = -1\): output −1. (If \(k = 0\), the empty sum is 0 and the total is −1, which is right for the function that is −1 everywhere.) The weight on each edge from \(g_i\) is 1, and the weight from the hidden constant is \(k - 1\). \(\blacksquare\)

Reading the two formulas. In the first, \(\langle x, u_i \rangle\) is the agreement score between the input and the stored pattern \(u_i\), and subtracting \(n - 1\) sets the bar just below a perfect match. In the second, \(\sum_{i=1}^k\) adds the \(k\) detector outputs, and adding \(k - 1\) sets the bar so that one +1 is enough. In plain words: **one hidden neuron per junk input recognises that input exactly, and the output neuron says junk if any of them recognises it.** With \(n = 3\) and \(u_1 = (1, 1, -1)\): on \(x = u_1\) the score is 3 and the neuron sees \(3 - 2 = 1\); on \(x = (1, 1, 1)\) the score is 1 and it sees \(1 - 2 = -1\).

Python can build this network for all 256 functions on three bits and check every one on every input:

```python
import itertools

def sign(a):
    return 1 if a > 0 else -1

n = 3
cube = list(itertools.product([-1, 1], repeat=n))

def depth_two(f):
    us = [u for u in cube if f[u] == 1]
    def h(x):
        g = [sign(sum(a * b for a, b in zip(x, u)) - n + 1)
             for u in us]
        return sign(sum(g) + len(us) - 1)
    return h

count = 0
for labels in itertools.product([-1, 1], repeat=len(cube)):
    f = dict(zip(cube, labels))
    h = depth_two(f)
    count += all(h(x) == f[x] for x in cube)
print(count, 2 ** len(cube))
```

It prints `256 256`. The list `g` is \(g_1(x), \dots, g_k(x)\) term by term, and the last line of `h` is the output formula.

The output neuron of that proof computed an OR, and the triangle’s output computed an AND. Both are instances of one small lemma.

**Lemma (a sign neuron can compute AND and OR).** Let a sign neuron have \(k\) incoming edges from neurons whose outputs are in \(\{\pm 1\}\), plus one edge from a constant neuron. With suitable weights it computes the conjunction (AND) of its inputs, and with others the disjunction (OR):

\[ \bigwedge_{i=1}^k x_i = \operatorname{sign}\Big(1 - k + \sum_{i=1}^k x_i\Big), \qquad \bigvee_{i=1}^k x_i = \operatorname{sign}\Big(k - 1 + \sum_{i=1}^k x_i\Big). \]

Here \(\bigwedge\) means “all of them are +1” and \(\bigvee\) means “at least one is +1”, with +1 playing true and −1 false.

*Proof.* If all inputs are +1, the sum is \(k\). If exactly \(j \ge 1\) of them are −1, the sum is \(k - 2j \le k - 2\). For the AND, all +1 gives \(1 - k + k = 1 > 0\), and any −1 gives at most \(1 - k + k - 2 = -1\). For the OR, the mirror image: all −1 gives \(k - 1 - k = -1\), and any +1 gives at least \(k - 1 + (2 - k) = 1\). \(\blacksquare\)

In plain words: **an AND neuron sets its bar so high that one −1 sinks it; an OR neuron sets it so low that one +1 lifts it.** The triangle used the AND with \(k = 3\): \(\operatorname{sign}(1 - 3 + \sum x_i) = \operatorname{sign}(\sum x_i - 2)\). And negation is one weight of −1. Checking every input for \(k\) from 1 to 8:

```python
import itertools

def sign(a):
    return 1 if a > 0 else -1

checked = 0
for k in range(1, 9):
    for x in itertools.product([-1, 1], repeat=k):
        conj = 1 if all(v == 1 for v in x) else -1
        disj = 1 if any(v == 1 for v in x) else -1
        assert sign(1 - k + sum(x)) == conj
        assert sign(k - 1 + sum(x)) == disj
        checked += 1
print(checked)
```

It prints `510`: all \(2 + 4 + \dots + 256\) inputs pass both formulas.

[[diagram:boolean-depth-two]]

AND, OR and NOT are the gates of a **Boolean circuit**, a network of logic gates, and every program can be unrolled into one. That gives a striking fact about efficient programs. We state it without proof, because the proof is a standard result about circuits rather than about learning.

**Fact (efficient programs are small networks).** If a function on \(n\) bits can be computed by a Turing machine (the formal model of a computer program) in at most \(T(n)\) steps, then it can be computed by a sign network whose size is at most \(c\,T(n)^2 + b\), for constants \(b\) and \(c\) that do not depend on \(n\).

The intuition: lay out the computer’s memory as a layer of neurons, one layer per time step; each step is a simple operation on the memory, and the lemma says such operations are a layer of AND, OR and NOT neurons. So every predictor that runs in polynomial time lives in some network class of polynomial size. Hold on to this. It means networks of modest size have almost no approximation error for anything we could hope to compute. The catch comes later.

## Seeing it in the plane

Back to the square of raw signals, where the pictures live. In a depth 2 sign network, each hidden neuron is a halfspace: it says +1 on one side of a line. If the output neuron computes the AND of the hidden features, the network says +1 exactly on the region inside every one of those lines. Such a region, an intersection of finitely many halfspaces, is a **convex polytope** (in the plane, a convex polygon). The triangle is a polytope with three faces. With \(k\) feature neurons plus the constant, a depth 2 network can carve any convex polytope with \(k\) faces.

What if the junk sits in two separate triangles, or a shape with a dent? Add one more layer. Let layer 1 hold the lines, layer 2 hold one AND neuron per polytope, and the output neuron take the OR of those. Now the network says +1 on a **union of polytopes**, which can approximate any shape you can draw.

Here is a depth 3 network for the union of the triangle and its mirror image through the centre of the square (the mirror triangle uses the same three lines with the signal weights negated). We compare it with the true region on a grid of \(201 \times 201\) messages:

```python
def sign(a):
    return 1 if a > 0 else -1

T1 = [(3, -1, 1.2), (-1, 3, 1.2), (-1, -1, 0.4)]
T2 = [(-w1, -w2, b) for w1, w2, b in T1]

def score(w, u):
    return w[0] * u[0] + w[1] * u[1] + w[2]

def inside(T, u):
    return all(score(w, u) > 0 for w in T)

def depth_three(u):
    o1 = [sign(score(w, u)) for w in T1 + T2]
    o2 = [sign(sum(o1[:3]) - 2), sign(sum(o1[3:]) - 2)]
    return sign(sum(o2) + 1)

grid = [(i / 100 - 1, j / 100 - 1)
        for i in range(201) for j in range(201)]
truth = [1 if inside(T1, u) or inside(T2, u) else -1
         for u in grid]
print(sum(depth_three(u) != t for u, t in zip(grid, truth)))
print(round(sum(t == 1 for t in truth) / len(grid), 2))
```

It prints `0` mistakes, and `0.22`: the six-pointed region covers 22% of the square. Layer 1 is `o1`, six features; layer 2 is `o2`, two ANDs (\(\sum - 2\) is the AND with \(k = 3\)); the output is the OR with \(k = 2\) (\(\sum + 1\)).

[[diagram:polytopes]]

So networks are wonderfully expressive. How much does that cost in labels?

## How many labels a network needs

Lesson 6 answered “how many labels” for any class through one number. Recall that a class **shatters** a set of \(m\) points when it produces all \(2^m\) labellings of them, and its **VC dimension** is the size of the largest set it shatters. The fundamental theorem of lesson 6 says that a class of binary hypotheses is learnable exactly when its VC dimension is finite, and that the number of labels grows in proportion to it. So we want the VC dimension of a sign network with a single output neuron.

A guess first. A network is named by \(|E|\) numbers, and a halfspace named by \(d\) numbers has VC dimension \(d\) (lesson 9). So perhaps the VC dimension is about \(|E|\). That is nearly right: we will prove at most a constant times \(|E| \log |E|\).

The tool is the **growth function** of lesson 6, which counts labellings. For a set \(C\) of \(m\) points, \(\mathcal H_C\) is the set of distinct ways the hypotheses of \(\mathcal H\) label the points of \(C\) (the restrictions of \(\mathcal H\) to \(C\)), and

\[ \tau_{\mathcal H}(m) = \max_{C \subset \mathcal X \,:\, |C| = m} |\mathcal H_C|. \]

As a reminder of the notation: \(C \subset \mathcal X\) is a set of messages, \(|C| = m\) says it has \(m\) of them, \(|\mathcal H_C|\) is how many distinct labellings the class produces on it, and \(\max\) takes the largest count over every such set. In plain words: **the most labellings the class can produce on any \(m\) points.** Thresholds on a line, for instance, produce \(m + 1\) labellings of \(m\) points: with \(m = 5\), six. A class shatters some set of \(m\) points exactly when \(\tau_{\mathcal H}(m) = 2^m\).

We need one small extension. The layers of a network output vectors of \(\pm 1\), not single labels, so we allow the hypotheses to take values in any finite set \(\mathcal Y\), and keep the same definition: \(\mathcal H_C\) is the set of distinct lists of values on \(C\). Two conventions keep the counting honest. A set with fewer than \(m\) points gets at most \(\tau_{\mathcal H}(m)\) labellings, because adding points to a set never reduces the number of labellings. And if the whole domain has fewer than \(m\) points, read \(\tau_{\mathcal H}(m)\) as the count on the whole domain.

The plan is to break the network into pieces whose growth functions we know, one neuron at a time, and multiply. Two lemmas say that multiplying is allowed.

**Lemma (growth of a product).** For \(i = 1, 2\), let \(\mathcal F_i\) be a class of functions from \(\mathcal X\) to a finite set \(\mathcal Y_i\). Let \(\mathcal H = \mathcal F_1 \times \mathcal F_2\) be the class of all functions \(h(x) = (f_1(x), f_2(x))\) with \(f_1 \in \mathcal F_1\) and \(f_2 \in \mathcal F_2\): two hypotheses run side by side. Then \(\tau_{\mathcal H}(m) \le \tau_{\mathcal F_1}(m)\, \tau_{\mathcal F_2}(m)\).

*Proof.* Fix any \(C\) with \(|C| = m\). The labelling that \(h = (f_1, f_2)\) produces on \(C\) is fully determined by the pair (labelling of \(f_1\) on \(C\), labelling of \(f_2\) on \(C\)). Write \(\mathcal F_{1,C}\) for the set of labellings \(\mathcal F_1\) produces on \(C\), and likewise \(\mathcal F_{2,C}\). There are at most \(|\mathcal F_{1,C}|\) choices for the first and \(|\mathcal F_{2,C}|\) for the second, so \(|\mathcal H_C| \le |\mathcal F_{1,C}|\,|\mathcal F_{2,C}| \le \tau_{\mathcal F_1}(m)\,\tau_{\mathcal F_2}(m)\). This holds for every \(C\), so it holds for the maximum. \(\blacksquare\)

**Lemma (growth of a composition).** Let \(\mathcal F_1\) be a class of functions from \(\mathcal X\) to a finite set \(\mathcal Z\), and \(\mathcal F_2\) a class from \(\mathcal Z\) to a finite set \(\mathcal Y\). Let \(\mathcal H = \mathcal F_2 \circ \mathcal F_1\) be the class of all \(h(x) = f_2(f_1(x))\): one hypothesis fed into another. Then \(\tau_{\mathcal H}(m) \le \tau_{\mathcal F_2}(m)\, \tau_{\mathcal F_1}(m)\).

*Proof.* Fix \(C = \{c_1, \dots, c_m\}\). Sort the hypotheses \(f_2 \circ f_1\) into groups by the list \((f_1(c_1), \dots, f_1(c_m))\). There are at most \(\tau_{\mathcal F_1}(m)\) groups. Inside one group that list is fixed; call the set of distinct points in it \(C'\), a subset of \(\mathcal Z\) with at most \(m\) points. The labelling of \(f_2 \circ f_1\) on \(C\) is read off from the labelling of \(f_2\) on \(C'\), so one group contributes at most \(|\mathcal F_{2,C'}| \le \tau_{\mathcal F_2}(m)\) labellings. Add over the groups: \(|\mathcal H_C| \le \tau_{\mathcal F_1}(m)\,\tau_{\mathcal F_2}(m)\). \(\blacksquare\)

In plain words: **running classes side by side, or one after another, multiplies their counts at worst.** On five points with thresholds (six labellings each), and the four functions from \(\{0, 1\}\) to \(\{0, 1\}\) as the second stage:

```python
from itertools import product

C = range(5)
F1 = [lambda x, t=t: int(x > t) for t in range(-1, 5)]
F2 = [lambda z, a=a, b=b: (a, b)[z]
      for a, b in product([0, 1], repeat=2)]
pairs = {tuple((f(x), g(x)) for x in C) for f in F1 for g in F1}
comps = {tuple(g(f(x)) for x in C) for f in F1 for g in F2}
print(len(F1), len(pairs), len(F1) ** 2)
print(len(comps), len(F2) * len(F1))
```

It prints `6 36 36` and `10 24`. The product of two threshold classes reaches its bound, 36. The composition gives 10 labellings against a bound of 24: the bound is a ceiling, not a prediction.

**Theorem (VC dimension of sign networks).** Let \((V, E)\) be a layered graph with a single output neuron and at least one edge. The VC dimension of \(\mathcal H_{V,E,\operatorname{sign}}\) is at most \(4\alpha \ln(2\alpha) + 2\alpha + 2\), where \(\alpha = |E|/\ln 2\) and \(\ln\) is the natural logarithm. In particular it is \(O(|E| \log |E|)\).

Here \(O(\cdot)\) means “at most a constant times”, as in every complexity statement of this course.

*Proof.* Write \(\mathcal H\) for the class.

*Step 1, cut the network into layers.* For each layer \(t \in \{1, \dots, T\}\), let \(\mathcal H^{(t)}\) be the class of all maps from the outputs of layer \(t - 1\) to the outputs of layer \(t\) that you get by choosing the weights on the edges between them. Its values are lists of \(\pm 1\), a finite set. Running the network is running the layers in turn, so \(\mathcal H = \mathcal H^{(T)} \circ \cdots \circ \mathcal H^{(1)}\). Applying the composition lemma \(T - 1\) times,

\[ \tau_{\mathcal H}(m) \le \prod_{t=1}^T \tau_{\mathcal H^{(t)}}(m). \]

*Step 2, cut each layer into neurons.* Within layer \(t\), the neurons work side by side, each with its own weights: \(\mathcal H^{(t)} = \mathcal H^{(t,1)} \times \cdots \times \mathcal H^{(t,|V_t|)}\), where \(\mathcal H^{(t,i)}\) holds the functions neuron \(i\) can compute from the layer below. The product lemma gives \(\tau_{\mathcal H^{(t)}}(m) \le \prod_i \tau_{\mathcal H^{(t,i)}}(m)\).

*Step 3, one neuron.* Let \(d_{t,i}\) be the number of edges into neuron \(v_{t,i}\), its **in-degree**. The neuron computes \(\operatorname{sign}(\langle w, z \rangle)\), where \(z\) is the list of the \(d_{t,i}\) outputs wired into it: a homogeneous halfspace in \(\mathbb R^{d_{t,i}}\), whose VC dimension is \(d_{t,i}\) (lesson 9). Sauer’s lemma (lesson 6) bounds its growth function. When \(1 \le d_{t,i} \le m\), it gives \(\tau \le (em/d_{t,i})^{d_{t,i}} \le (em)^{d_{t,i}}\). When \(m < d_{t,i}\), the trivial bound gives \(\tau \le 2^m < e^m \le (em)^{d_{t,i}}\): since \(em \ge e\) and \(d_{t,i} > m\), we have \((em)^{d_{t,i}} \ge e^{d_{t,i}} \ge e^m\). When \(d_{t,i} = 0\), as for a constant neuron, the neuron computes one function, so \(\tau = 1 = (em)^0\). In every case,

\[ \tau_{\mathcal H^{(t,i)}}(m) \le (em)^{d_{t,i}}. \]

*Step 4, multiply.* Every edge enters exactly one neuron, so the in-degrees add up to the number of edges, \(\sum_{t,i} d_{t,i} = |E|\). Putting Steps 1 to 3 together,

\[ \tau_{\mathcal H}(m) \le \prod_{t,i} (em)^{d_{t,i}} = (em)^{\sum_{t,i} d_{t,i}} = (em)^{|E|}. \]

Read the last chain: \(\prod_{t,i}\) multiplies over every neuron of every layer, \(e \approx 2.718\) is Euler’s number, and powers of the same base multiply by adding exponents. In plain words: **a network with \(|E|\) edges can label \(m\) points in at most \((em)^{|E|}\) ways, a polynomial in \(m\) whose degree is the number of weights.** Take the triangle architecture: in-degrees 3, 3, 3 for the feature neurons, 0 for the hidden constant, 4 for the output, 13 in all. At \(m = 20\):

```python
from math import e, prod

m = 20
in_degrees = [3, 3, 3, 0, 4]
per_neuron = [(e * m / d) ** d if d else 1 for d in in_degrees]
loose = [(e * m) ** d for d in in_degrees]
print(sum(in_degrees))
print(f"{prod(per_neuron):.2e} {prod(loose):.2e}")
print(f"{(e * m) ** sum(in_degrees):.2e} {2 ** m:.2e}")
```

It prints `13`, then `7.19e+15 3.62e+22`, then `3.62e+22 1.05e+06`. The Sauer bounds per neuron multiply to \(7.2 \times 10^{15}\); the looser \((em)^{d}\) version to \((em)^{13} = 3.6 \times 10^{22}\). At 20 points that is still far above \(2^{20} \approx 10^6\), so the bound cannot yet rule out shattering 20 points. It wins only further out, because \(2^m\) grows exponentially and \((em)^{13}\) only polynomially.

*Step 5, where the bound beats \(2^m\).* Suppose \(\mathcal H\) shatters some set of \(m\) points. Then \(\tau_{\mathcal H}(m) = 2^m\), so

\[ 2^m \le (em)^{|E|} \quad\Longrightarrow\quad m \le |E| \log_2(em). \]

The arrow comes from taking \(\log_2\) of both sides (\(\log_2\) is increasing, so the inequality keeps its direction) and using \(\log_2(a^b) = b \log_2 a\). In plain words: **a set the network shatters can have at most about \(|E|\) times a logarithm many points.**

*Step 6, untie \(m\).* The unknown \(m\) sits on both sides. Switch to natural logarithms: \(\log_2(em) = (1 + \ln m)/\ln 2\), so a shattered \(m\) satisfies \(m \le \alpha \ln m + \alpha\) with \(\alpha = |E|/\ln 2 \ge 1\). Lesson 6’s **log lemma** says: if \(\alpha \ge 1\) and \(\beta \ge 0\), then \(m \ge 4\alpha \ln(2\alpha) + 2\beta\) implies \(m \ge \alpha \ln m + \beta\). Use it with \(\beta = \alpha + 1\): any \(m \ge 4\alpha\ln(2\alpha) + 2\alpha + 2\) has \(m \ge \alpha \ln m + \alpha + 1 > \alpha \ln m + \alpha\), so it cannot be shattered. Hence the VC dimension is below \(4\alpha\ln(2\alpha) + 2\alpha + 2\), which is \(O(|E| \log |E|)\). \(\blacksquare\)

Here is Step 5 solved exactly by search, next to \(|E| \log_2 |E|\) and the closed form of Step 6:

```python
from math import e, log, log2

def vc_cap(N):
    m = 1
    while m <= N * log2(e * m):
        m += 1
    return m - 1

for N in (9, 13, 100, 1000):
    a = N / log(2)
    lemma = 4 * a * log(2 * a) + 2 * a + 2
    print(N, vc_cap(N), round(N * log2(N)), round(lemma))
```

It prints `9 67 29 197`, `13 106 48 311`, `100 1162 664 3560` and `1000 15348 9966 48866`. Reading the triangle row: no set of more than 106 messages can be shattered by the 13-weight architecture, and the closed form, which is easier to quote, says 311. Both grow like \(|E| \log |E|\).

[[diagram:growth-layers]]

Combine this with the fundamental theorem of lesson 6: a sign network is learnable with a number of labels proportional to \(|E| \log |E|\) (plus \(\log(1/\delta)\)), divided by \(\varepsilon^2\) in the agnostic case. Here \(\varepsilon\) is the accuracy and \(\delta\) the allowed failure probability, as always. The price in labels grows only a little faster than the number of weights.

## Most functions need exponentially many neurons

We can now settle what the depth 2 construction left open. It used \(2^n + 1\) hidden neurons. Was that wasteful? No, and the VC bound is exactly the tool that shows it.

**Theorem (universality costs exponential size).** Let \(s(n)\) be the smallest size \(|V|\) of a graph for which \(\mathcal H_{V,E,\operatorname{sign}}\) contains every function from \(\{\pm 1\}^n\) to \(\{\pm 1\}\). Then \(s(n)\) is exponential in \(n\): every such graph has

\[ |E| \ge \frac{2^n}{n + \log_2 e} \qquad\text{and}\qquad |V| \ge \sqrt{\frac{2^n}{n + \log_2 e}}. \]

*Proof.* If the class contains every Boolean function, it produces every labelling of the \(2^n\) inputs, so it shatters a set of \(m = 2^n\) points. Step 5 of the VC proof gives \(2^n \le |E| \log_2(e \cdot 2^n) = |E|(n + \log_2 e)\), using \(\log_2(e \cdot 2^n) = \log_2 e + n\). Divide by \(n + \log_2 e\) for the first bound. For the second, an edge is an ordered pair of neurons, so \(|E| \le |V|^2\), and \(|V| \ge \sqrt{|E|}\). \(\blacksquare\)

Reading it: \(2^n\) is the number of inputs, \(n + \log_2 e \approx n + 1.44\) is the logarithm that Step 5 charges, and the square root converts edges into neurons. In plain words: **a network that can compute every Boolean function on \(n\) bits needs about \(2^n/n\) weights, and at least the square root of that in neurons.** At \(n = 20\): \(2^{20}/21.44 \approx 48{,}902\) edges and 222 neurons, at least.

```python
from math import ceil, e, log2, sqrt

for n in (10, 20, 30, 40):
    m = 2 ** n
    edges = ceil(m / log2(e * m))
    nodes = ceil(sqrt(edges))
    print(n, edges, nodes, m + n + 3)
```

It prints `10 90 10 1037`, `20 48902 222 1048599`, `30 34149167 5844 1073741857` and `40 26530891070 162884 1099511627819`. The last column is the size of the depth 2 construction, \((n + 1) + (2^n + 1) + 1\). Every 10 extra bits multiply the lower bound on neurons by more than 20 and on edges by more than 500: exponential growth, whatever the depth.

So there is no free universal network. The Fact about efficient programs says the functions you can compute quickly *do* have small networks; this theorem says most functions do not. The same holds for sigmoid networks that approximate smooth functions to a fixed precision, which is the second problem at the end of the lesson.

## Sigmoid networks and the bits of a weight

The VC proof used, in Step 3, that every neuron outputs one of two values. A sigmoid neuron outputs any number between 0 and 1, so the proof does not apply, and the answer really is different: there are sigmoid networks whose VC dimension grows like \(|E|^2\), the square of the number of weights. The best known upper bound is of order \(|V|^2 |E|^2\), with a proof beyond this course. Real numbers can hide a lot of information.

But your computer never stores a real number. Lesson 4’s **discretization trick** says: a class described by numbers that each take one of finitely many values is a finite class. Suppose every weight is stored with \(b\) bits, so it takes one of \(2^b\) values. Then a network with \(|E|\) weights has at most \((2^b)^{|E|} = 2^{b|E|}\) possible weight settings, so the class has at most \(2^{b|E|}\) hypotheses, whatever the activation.

A finite class \(\mathcal H\) cannot shatter more than \(\log_2 |\mathcal H|\) points, because shattering \(m\) points takes \(2^m\) different hypotheses. So with \(b\)-bit weights the VC dimension is at most \(b|E|\): for the triangle architecture with 8-bit weights, \(8 \cdot 13 = 104\). Linear in the number of edges, even with sigmoids.

And for a finite class, lesson 4 gave us something better than a VC argument: a bound with explicit constants. Recall what it said. Fix the class before seeing data; then, with probability at least \(1 - \delta\) over the training sequence, *every* hypothesis has empirical risk within a margin of its true risk, at once, and the margin shrinks like \(1/\sqrt m\). The empirical risk \(L_S(h)\) is the fraction of the \(m\) labelled examples \(h\) gets wrong, and the true risk \(L_{\mathcal D}(h)\) is its chance of mislabelling the next message (lesson 1). Lesson 4 wrote the margin \(\varepsilon\) and solved for \(m \ge \ln(2|\mathcal H|/\delta)/(2\varepsilon^2)\). Solve the same inequality for \(\varepsilon\) instead, and put in \(\ln |\mathcal H| \le b|E| \ln 2\). With probability at least \(1 - \delta\), every network \(h\) in the class satisfies

\[ L_{\mathcal D}(h) \;\le\; L_S(h) + \sqrt{\frac{b\,|E| \ln 2 + \ln(2/\delta)}{2m}}. \]

Reading from the inside: \(b|E|\ln 2\) is the natural logarithm of the number of networks, \(\ln(2/\delta)\) pays for the confidence, dividing by \(2m\) and taking the square root gives the margin, and adding it to the empirical risk gives a ceiling on the true risk. The 0-1 loss lies in \([0, 1]\), which is what lesson 4’s argument needs. In plain words: **whatever network you pick, by whatever method, its true risk is at most its training error plus a margin that grows with the number of edges and shrinks with the number of labels.** We’ll call the right-hand side the network’s **certificate**. With the triangle architecture, \(b = 8\), \(\delta = 0.05\) and \(m = 12{,}000\) labels, the margin is \(\sqrt{(72.08 + 3.69)/24{,}000} \approx 0.056\).

That “by whatever method” is the important part. The guarantee holds for every hypothesis in the class simultaneously, so it does not care how the weights were found: by a clever algorithm, by a heuristic, by trying five runs and keeping the best. We are about to need exactly that freedom, because finding good weights turns out to be the hard part.

Here is the margin for networks with \(k\) feature neurons on our two raw signals. Such a network has \(3k\) edges into the hidden layer (two signals and a constant into each) and \(k + 1\) into the output, so \(|E| = 4k + 1\). The last column is how much training error a network can afford if its certificate is to stay under 0.10.

```python
from math import log, sqrt

m, b, delta = 12_000, 8, 0.05
for k in (1, 2, 3, 4, 5, 8):
    E = 3 * k + (k + 1)
    gap = sqrt((b * E * log(2) + log(2 / delta)) / (2 * m))
    print(k, E, round(gap, 4), round(0.10 - gap, 4))
```

It prints

```
1 5 0.0362 0.0638
2 9 0.0473 0.0527
3 13 0.0562 0.0438
4 17 0.0639 0.0361
5 21 0.0708 0.0292
8 33 0.0882 0.0118
```

Read it with the triangle in mind. With one or two feature neurons the margin is small, but the network cannot carve a triangle, so its training error will be large. With eight, it can carve anything nearby, but it would need a training error of 1.2%, below the 2% your careless users guarantee. Somewhere in between is the size that pays for itself. Keep this table; you will need it in the lab.

## The price: training is hard

So far networks look like the perfect class. Polynomial-size networks express everything that can be computed efficiently, so their approximation error is as small as one could hope, and their label cost grows with the number of edges. What is the catch?

The catch is computation. Recall from lesson 8 that a problem is **NP-hard** when a polynomial-time algorithm for it would give polynomial-time algorithms for every problem in NP, which almost everyone believes impossible. And recall that ERM, empirical risk minimization, means returning a hypothesis in the class with the smallest empirical risk.

**Fact (ERM for small networks is NP-hard).** Let \(k \ge 3\). For every \(n\), let \((V, E)\) be the layered graph with \(n\) input neurons (plus the constant), one hidden layer of \(k + 1\) neurons, one of them the constant, and a single output neuron. Implementing the ERM rule for \(\mathcal H_{V,E,\operatorname{sign}}\) is NP-hard.

The third problem at the end of the lesson asks you to prove it. The key is a class you met in lesson 8’s problems: intersections of \(k\) halfspaces. By the AND lemma, this network contains every intersection of \(k\) halfspaces, and fitting those exactly encodes colouring a graph with \(k\) colours, a classic NP-hard problem. Notice how small the network is: three feature neurons, the triangle architecture itself.

Could we escape? Two natural ideas, and the evidence against each:

- **Settle for nearly the best weights.** Even finding weights whose empirical risk is close to the minimum is known to be computationally infeasible.
- **Use a bigger network, or another activation.** Lesson 8 showed that a hard class can sometimes be learned through a larger, easier one. But under a widely believed cryptographic assumption, intersections of halfspaces cannot be learned efficiently by *any* learner, whatever hypotheses it outputs. Any network class that contains them inherits the hardness.

So no efficient algorithm is guaranteed to train a network well in the worst case. What people do instead is run stochastic gradient descent and hope. Lesson 14 proved that SGD succeeds when the loss is convex; for networks it is far from convex, so there is no such guarantee. Yet on many practical tasks it finds good weights. The certificate of the last section is what lets us use the result safely: we do not need SGD to be optimal, only to land on a network whose training error is small, and then the certificate tells us what that is worth on fresh mail.

## Training by stochastic gradient descent

Gradient descent needs a loss that changes smoothly with the weights. The sign activation is no good: nudge a weight a little and the output either does not change at all or jumps. So for training we switch to the sigmoid, its smooth cousin, and to a smooth loss.

Concretely, the network now has \(n\) inputs and \(k\) outputs, and we write \(\mathbf w \in \mathbb R^{|E|}\) for the weights listed as one long vector, so \(h_{\mathbf w}: \mathbb R^n \to \mathbb R^k\) is the function the network computes. A sigmoid output lies between 0 and 1, so the label goes back to the course’s 1 for junk and 0 for keep. On one example \((x, y)\), the network predicts \(h_{\mathbf w}(x)\) and pays the **squared loss**, written \(\Delta\) (capital delta), half the squared distance between prediction and target. On the small network from “The layered graph”, the message \((0.5, -0.5)\) with label \(y = 1\) gets the prediction 0.8299, so the loss is \(\tfrac12 (0.8299 - 1)^2 \approx 0.01448\): the last number that snippet printed. Averaging the loss over the distribution of messages gives the risk of the weights:

\[ \Delta(h_{\mathbf w}(x), y) = \tfrac12 \|h_{\mathbf w}(x) - y\|^2, \qquad L_{\mathcal D}(\mathbf w) = \mathop{\mathbb E}_{(x,y) \sim \mathcal D} \big[ \Delta(h_{\mathbf w}(x), y) \big]. \]

Symbol by symbol: \(\|\cdot\|\) is the length of a vector (for one output, the absolute value), so \(\|h_{\mathbf w}(x) - y\|^2\) is the squared distance between prediction and label. The notation \((x, y) \sim \mathcal D\) reads “the example \((x, y)\) is drawn from \(\mathcal D\)”, and \(\mathbb E[\dots]\) is the expected value, the average over those draws. In plain words: **the risk of a weight vector is the average squared miss of the network on a random message.** For our small network, with untrained weights, on the triangle world (no careless labels for this example), we estimate the average by drawing 100,000 messages:

```python
import math, random

def sigma(a):
    return 1 / (1 + math.exp(-a))

def h(W0, W1, u):
    o0 = list(u) + [1.0]
    o1 = [sigma(sum(w * o for w, o in zip(r, o0))) for r in W0]
    return sigma(sum(w * o for w, o in zip(W1, o1 + [1.0])))

def inside(u):
    s = [3 * u[0] - u[1] + 1.2, -u[0] + 3 * u[1] + 1.2]
    return min(s + [-u[0] - u[1] + 0.4]) > 0

W0 = [[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]]
W1 = [2.0, -1.0, 0.5]
rng = random.Random(0)
total, n = 0.0, 100_000
for _ in range(n):
    u = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    total += (h(W0, W1, u) - inside(u)) ** 2 / 2
print(round(total / n, 3))
```

It prints `0.241`. Training will push that number down.

SGD from lesson 14, in one breath: start somewhere, then repeatedly draw one example, compute the gradient of the loss on it, and take a small step against it. For networks four things change.

**First, start at random, near zero, not at zero.** If every weight starts at 0 in a fully connected layered network, all the hidden neurons of a layer compute the same function, receive the same gradient, and stay identical forever: the layer behaves like a single neuron. Random starting weights break the tie. Here is the trap, run on three hidden neurons, with the label kept short: junk when the message is inside the long edge of the triangle.

```python
import math, random

def sigma(a):
    return 1 / (1 + math.exp(-a))

def sgd(W0, W1, steps, rng, eta=0.5):
    for _ in range(steps):
        u = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        y = float(-u[0] - u[1] + 0.4 > 0)
        o0 = u + [1.0]
        a1 = [sum(w * o for w, o in zip(r, o0)) for r in W0]
        o1 = [sigma(a) for a in a1] + [1.0]
        o2 = sigma(sum(w * o for w, o in zip(W1, o1)))
        d2 = (o2 - y) * o2 * (1 - o2)
        for i in range(len(W0)):
            d1 = W1[i] * d2 * o1[i] * (1 - o1[i])
            W0[i] = [w - eta * d1 * o for w, o in zip(W0[i], o0)]
        W1[:] = [w - eta * d2 * o for w, o in zip(W1, o1)]
    return W0

rng = random.Random(0)
zero = sgd([[0.0] * 3 for _ in range(3)], [0.0] * 4, 2000, rng)
print([[round(w, 3) for w in r] for r in zero])
```

It prints `[[-2.46, -2.323, 0.377], [-2.46, -2.323, 0.377], [-2.46, -2.323, 0.377]]`: after 2,000 steps, three copies of one neuron. (The update lines are backpropagation, derived in the next section.) The hope with random starts goes further. SGD can get stuck in a **local minimum** (lesson 12): weights where every small step raises the loss, although weights far away do better. Run SGD several times from different random points, and one of the runs may reach a good one.

The other three changes:

- **A step size that changes,** \(\eta_i\) at step \(i\), as in lesson 14’s variants. Without convexity the schedule matters more, and in practice it is tuned by trial and error.
- **Output the best iterate,** judged on a validation set (lesson 11), rather than the average of lesson 14.
- **Optionally, a penalty on large weights,** with a parameter \(\lambda > 0\): minimize \(L_{\mathcal D}(\mathbf w) + \frac{\lambda}{2}\|\mathbf w\|^2\), the regularization of lesson 13.

The penalty’s gradient is \(\lambda \mathbf w\), so the step from iterate \(\mathbf w^{(i)}\) (the weights after \(i - 1\) steps), with \(\mathbf v_i\) the gradient of the loss on the example drawn, is

\[ \mathbf w^{(i+1)} = \mathbf w^{(i)} - \eta_i \big( \mathbf v_i + \lambda \mathbf w^{(i)} \big). \]

Reading it: \(\mathbf v_i + \lambda \mathbf w^{(i)}\) is the gradient of the penalized loss on this example; multiplying by \(\eta_i\) sets the step length; subtracting moves downhill. In plain words: **step against the example’s gradient, and also shrink every weight a little toward zero.** For the output weights of our small network, \((2, -1, 0.5)\), the gradient computed in the next section is \((-0.0176, -0.0091, -0.024)\). With \(\eta = 0.5\) and \(\lambda = 0.01\):

```python
w = [2.0, -1.0, 0.5]
v = [-0.0176, -0.0091, -0.024]
eta, lam = 0.5, 0.01
w_next = [wi - eta * (vi + lam * wi) for wi, vi in zip(w, v)]
print([round(x, 4) for x in w_next])
```

It prints `[1.9988, -0.9905, 0.5095]`. The first weight went *down* although its gradient is negative: the shrinkage \(0.5 \cdot 0.01 \cdot 2 = 0.01\) outweighed the push \(0.5 \cdot 0.0176 = 0.0088\).

Everything here is cheap except one line: computing \(\mathbf v_i\), the gradient of the loss with respect to every weight in the network. That is backpropagation.

## Backpropagation

Let’s first do it by hand, on the small network and the message \((0.5, -0.5)\) labelled junk, \(y = 1\). We want to know, for each weight, how fast the loss changes when that weight moves.

Start at the top. The loss is \(\tfrac12(o_2 - y)^2\), so it changes at rate \(o_2 - y = 0.8299 - 1 = -0.1701\) per unit change of the output \(o_2\). The output is \(\sigma(a_2)\), and \(\sigma\) changes at rate \(\sigma(a_2)(1 - \sigma(a_2)) = 0.8299 \cdot 0.1701 = 0.1412\) per unit of \(a_2\). And \(a_2\) is a weighted sum, so it changes at rate \(o_{1,j}\) per unit of the weight on the wire from hidden neuron \(j\). Multiply the three rates: the weight from hidden neuron 1 (output 0.7311) has slope \(-0.1701 \cdot 0.1412 \cdot 0.7311 \approx -0.0176\). The loss falls if that weight grows, which makes sense: hidden neuron 1 is fairly active, and we want a higher output.

Now one layer down. How much does the loss care about hidden neuron 1’s output? It reaches the loss only through \(a_2\), with weight 2, so its rate is \(2 \cdot (-0.1701 \cdot 0.1412) \approx -0.0480\). That number is passed backwards, and the same three-factor product gives the slopes of the weights into hidden neuron 1. This passing of rates from the top layer down is why the algorithm is called **backpropagation**.

To make this exact for any layered network, we need three tools from calculus, then one recursion.

**Partial derivatives and the Jacobian.** For a function \(f: \mathbb R^n \to \mathbb R\), the **partial derivative** with respect to coordinate \(i\) at \(\mathbf w\) is the ordinary derivative you get by freezing every other coordinate and moving only \(w_i\): the derivative at 0 of \(g(a) = f(w_1, \dots, w_i + a, \dots, w_n)\). The **gradient** lists all \(n\) partial derivatives. For a function with several outputs, \(f: \mathbb R^n \to \mathbb R^m\), the **Jacobian** at \(\mathbf w\), written \(J_{\mathbf w}(f)\), is the \(m \times n\) table whose entry in row \(i\), column \(j\) is the partial derivative of output \(i\) with respect to input \(j\). With one output, the Jacobian is the gradient written as a row.

Two Jacobians we will use. First, a linear map \(f(\mathbf w) = A\mathbf w\), for a matrix \(A\) with \(m\) rows and \(n\) columns, has Jacobian \(A\) itself: output \(i\) is \(\sum_j A_{ij} w_j\), whose derivative in \(w_j\) is \(A_{ij}\). Second, applying the sigmoid to each coordinate of a vector \(\boldsymbol\theta\), written \(\sigma(\boldsymbol\theta)\), has a diagonal Jacobian: output \(i\) depends only on \(\theta_i\), with derivative \(\sigma'(\theta_i)\), and every off-diagonal entry is 0. We write that matrix \(\operatorname{diag}(\sigma'(\boldsymbol\theta))\). The derivative of the sigmoid has a neat form:

\[ \sigma'(a) = \frac{1}{(1 + e^{a})(1 + e^{-a})} = \sigma(a)\big(1 - \sigma(a)\big). \]

To see the equality, differentiate \(\sigma(a) = (1 + e^{-a})^{-1}\) to get \(e^{-a}/(1 + e^{-a})^2\). Then note \(1 - \sigma(a) = e^{-a}/(1 + e^{-a})\), so \(\sigma(a)(1 - \sigma(a)) = e^{-a}/(1 + e^{-a})^2\) too. And multiplying out \((1 + e^a)(1 + e^{-a}) = 2 + e^a + e^{-a} = (1 + e^{-a})^2 e^{a}\) gives the middle form. In plain words: **the sigmoid’s slope is its value times one minus its value**, largest (\(1/4\)) at \(a = 0\) and tiny far away. At \(a_2 \approx 1.5846\) it is \(0.8299 \cdot 0.1701 = 0.1412\), the number we used by hand.

```python
import math

def sigma(a):
    return 1 / (1 + math.exp(-a))

for a in (-2.0, 0.0, 1.0, 1.5846):
    left = sigma(a) * (1 - sigma(a))
    right = 1 / ((1 + math.exp(a)) * (1 + math.exp(-a)))
    print(a, round(left, 4), round(right, 4))
```

It prints `-2.0 0.105 0.105`, `0.0 0.25 0.25`, `1.0 0.1966 0.1966` and `1.5846 0.1412 0.1412`.

**The chain rule.** For \(g: \mathbb R^k \to \mathbb R^n\) and \(f: \mathbb R^n \to \mathbb R^m\), the composition \(f \circ g\) (first \(g\), then \(f\)) has Jacobian

\[ J_{\mathbf w}(f \circ g) = J_{g(\mathbf w)}(f)\; J_{\mathbf w}(g). \]

The subscripts say where each Jacobian is taken: \(g\)’s at the point \(\mathbf w\), and \(f\)’s at the point \(g(\mathbf w)\) where \(f\) is actually evaluated. The product is a matrix product. In plain words: **rates of change along a pipeline multiply**, exactly as we multiplied three rates by hand. For example, with \(g(\mathbf w) = A\mathbf w\) followed by the sigmoid, \(J_{\mathbf w}(\sigma \circ g) = \operatorname{diag}(\sigma'(A\mathbf w))\, A\). Python can check that against finite differences, nudging each coordinate by \(10^{-6}\):

```python
import math

def sigma(a):
    return 1 / (1 + math.exp(-a))

A = [[1.0, -2.0], [0.5, 3.0]]
w = [0.3, -0.1]

def F(w):
    return [sigma(r[0] * w[0] + r[1] * w[1]) for r in A]

Aw = [r[0] * w[0] + r[1] * w[1] for r in A]
chain = [[sigma(Aw[i]) * (1 - sigma(Aw[i])) * A[i][j]
          for j in range(2)] for i in range(2)]
h = 1e-6
numeric = [[(F([w[0] + h * (j == 0), w[1] + h * (j == 1)])[i]
             - F(w)[i]) / h for j in range(2)] for i in range(2)]
print([[round(c, 4) for c in r] for r in chain])
print([[round(c, 4) for c in r] for r in numeric])
```

Both lines print `[[0.235, -0.47], [0.1243, 0.7458]]`.

**Matrices for the layers.** Write layer \(t\) as \(V_t = \{v_{t,1}, \dots, v_{t,k_t}\}\), so \(k_t = |V_t|\). Collect the weights between layers \(t\) and \(t + 1\) into a matrix \(W_t\) with \(k_{t+1}\) rows and \(k_t\) columns, whose entry \(W_{t,i,j}\) is the weight of the edge from \(v_{t,j}\) up to \(v_{t+1,i}\). If that edge is not in \(E\), put a **phantom edge** of weight 0 there. Phantom edges change nothing: a partial derivative freezes every other weight, and a weight that stays 0 contributes nothing to any sum. So we may pretend every pair of neurons in adjacent layers is wired. Now a layer’s inputs are \(\mathbf a_t = W_{t-1} \mathbf o_{t-1}\), and its outputs are \(\mathbf o_t = \sigma(\mathbf a_t)\). (Constant neurons receive no weights and need no gradient; they just sit in \(\mathbf o_{t-1}\) with value 1.)

**The weights of one layer.** Freeze every weight except those in \(W_{t-1}\). Then \(\mathbf o_{t-1}\), the outputs of the layer below, are fixed numbers. Let \(\ell_t\) be the loss as a function of the outputs of layer \(t\): “feed these outputs through the rest of the network and score the result”. The loss, as a function of \(W_{t-1}\) alone, is \(\ell_t(\sigma(W_{t-1}\mathbf o_{t-1}))\). Here is a bookkeeping trick: \(W_{t-1}\mathbf o_{t-1}\) is linear in the weights, and if we list the weights row by row as one long vector \(\mathbf w_{t-1}\), it equals \(O_{t-1}\mathbf w_{t-1}\), where row \(i\) of the matrix \(O_{t-1}\) holds the outputs \(\mathbf o_{t-1}\) in the columns of the weights into neuron \(i\), and zeros elsewhere. In plain words: **neuron \(i\)’s row of \(O_{t-1}\) multiplies only neuron \(i\)’s weights, by the values on their wires.** In the small network, \(\mathbf o_0 = (0.5, -0.5, 1)\) and six weights:

```python
W0 = [[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]]
o0 = [0.5, -0.5, 1.0]
k, n = len(W0), len(o0)
w = [x for row in W0 for x in row]
O = [[o0[c - i * n] if i * n <= c < (i + 1) * n else 0.0
      for c in range(k * n)] for i in range(k)]
for row in O:
    print(row)
print([sum(p * q for p, q in zip(r, w)) for r in O])
print([sum(p * q for p, q in zip(r, o0)) for r in W0])
```

It prints the rows `[0.5, -0.5, 1.0, 0.0, 0.0, 0.0]` and `[0.0, 0.0, 0.0, 0.5, -0.5, 1.0]`, then `[1.0, -0.5]` twice: \(O_0\mathbf w_0\) and \(W_0\mathbf o_0\) are the same inputs \(\mathbf a_1\). The chain rule twice, with the two Jacobians above, gives

\[ J_{\mathbf w_{t-1}}(\text{loss}) = J_{\mathbf o_t}(\ell_t)\, \operatorname{diag}(\sigma'(\mathbf a_t))\, O_{t-1}. \]

Name the first factor \(\boldsymbol\delta_t = J_{\mathbf o_t}(\ell_t)\): the row of rates at which the loss changes with each output of layer \(t\). The entry of the product belonging to the weight \(W_{t-1,i,j}\) picks out \(\delta_{t,i}\), then \(\sigma'(a_{t,i})\), then the \(j\)-th entry of \(\mathbf o_{t-1}\). Writing \(\partial\,\text{loss}/\partial W_{t-1,i,j}\) for the partial derivative of the loss with respect to that one weight,

\[ \frac{\partial\, \text{loss}}{\partial W_{t-1,i,j}} = \delta_{t,i}\; \sigma'(a_{t,i})\; o_{t-1,j}. \]

In plain words: **the slope of a weight is how much the loss cares about the neuron it feeds, times that neuron’s sigmoid slope, times the value travelling along the wire.** The three factors are the three rates we multiplied by hand: \(-0.1701 \cdot 0.1412 \cdot 0.7311\).

**The rates, from the top down.** At the top, \(\ell_T(\mathbf u) = \tfrac12\|\mathbf u - y\|^2\), whose gradient is \(\mathbf u - y\), so \(\boldsymbol\delta_T = \mathbf o_T - y\). Below the top, the loss of layer \(t\)’s outputs is the loss of what they produce one layer up: \(\ell_t(\mathbf u) = \ell_{t+1}(\sigma(W_t \mathbf u))\). The chain rule, at \(\mathbf u = \mathbf o_t\), where \(W_t\mathbf o_t = \mathbf a_{t+1}\) and \(\sigma(\mathbf a_{t+1}) = \mathbf o_{t+1}\), gives

\[ \boldsymbol\delta_T = \mathbf o_T - y, \qquad \boldsymbol\delta_t = \boldsymbol\delta_{t+1}\, \operatorname{diag}(\sigma'(\mathbf a_{t+1}))\, W_t, \quad\text{that is,}\quad \delta_{t,i} = \sum_{j=1}^{k_{t+1}} W_{t,j,i}\, \delta_{t+1,j}\, \sigma'(a_{t+1,j}). \]

Reading the sum: neuron \(i\) of layer \(t\) feeds every neuron \(j\) of the layer above along the weight \(W_{t,j,i}\); each of those passes back its own rate \(\delta_{t+1,j}\) times its sigmoid slope; and neuron \(i\) adds up what comes back along its outgoing wires. In plain words: **a neuron’s rate is the sum, over the neurons it feeds, of their rate times their slope times the connecting weight.** By hand we had \(\delta_{1,1} = 2 \cdot (-0.1701) \cdot 0.1412 \approx -0.0480\).

So the algorithm is: a forward pass computes every \(\mathbf a_t\) and \(\mathbf o_t\) from the bottom up; a backward pass computes every \(\boldsymbol\delta_t\) from the top down; then each partial derivative is one product of three numbers. Here is all of it for our small network:

```python
import math

def sigma(a):
    return 1 / (1 + math.exp(-a))

def d_sigma(a):
    return sigma(a) * (1 - sigma(a))

W0 = [[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]]
W1 = [2.0, -1.0, 0.5]
o0, y = [0.5, -0.5, 1.0], 1.0
a1 = [sum(w * o for w, o in zip(row, o0)) for row in W0]
o1 = [sigma(a) for a in a1] + [1.0]
a2 = sum(w * o for w, o in zip(W1, o1))
delta2 = sigma(a2) - y
delta1 = [W1[i] * delta2 * d_sigma(a2) for i in range(2)]
g1 = [delta2 * d_sigma(a2) * o for o in o1]
g0 = [[delta1[i] * d_sigma(a1[i]) * o for o in o0]
      for i in range(2)]
print(round(delta2, 4), [round(d, 4) for d in delta1])
print([round(g, 4) for g in g1])
print([[round(g, 4) for g in row] for row in g0])
```

It prints

```
-0.1701 [-0.048, 0.024]
[-0.0176, -0.0091, -0.024]
[[-0.0047, 0.0047, -0.0094], [0.0028, -0.0028, 0.0056]]
```

`delta2` is \(\boldsymbol\delta_2\), `delta1` is the recursion for \(\boldsymbol\delta_1\), and `g1`, `g0` apply the three-factor product to every weight. Is it right? The definition of a partial derivative says: nudge one weight, rerun the network, and measure. Nudging the second weight into hidden neuron 1, and the second output weight:

```python
import math

def sigma(a):
    return 1 / (1 + math.exp(-a))

def loss(W0, W1, o0=(0.5, -0.5, 1.0), y=1.0):
    o1 = [sigma(sum(w * o for w, o in zip(r, o0))) for r in W0]
    o1 = o1 + [1.0]
    out = sigma(sum(w * o for w, o in zip(W1, o1)))
    return (out - y) ** 2 / 2

W0 = [[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]]
W1 = [2.0, -1.0, 0.5]
h = 1e-6
W0_plus = [[1.0, -1.0 + h, 0.0], [0.5, 0.5, -0.5]]
W1_plus = [2.0, -1.0 + h, 0.5]
print(round((loss(W0_plus, W1) - loss(W0, W1)) / h, 4))
print(round((loss(W0, W1_plus) - loss(W0, W1)) / h, 4))
```

It prints `0.0047` and `-0.0091`, the same as backpropagation’s entries. The difference is speed. Nudging needs one full forward pass per weight, so a gradient costs \(|E|\) forward passes. Backpropagation gets all \(|E|\) partial derivatives from one forward and one backward pass.

[[diagram:backprop]]

## Networks as code

Here are the three algorithms together. The network has layers \(V_0, \dots, V_T\) with \(k_t\) neurons each, and weight matrices \(W_0, \dots, W_{T-1}\).

```
Forward(x, W):
  o_0 ← x
  for t = 1..T:
    for i = 1..k_t:
      a_{t,i} ← Σ_j W_{t-1,i,j} · o_{t-1,j}
      o_{t,i} ← σ(a_{t,i})
  return all a_t, o_t

Backpropagation(x, y, W):
  run Forward(x, W)
  δ_T ← o_T − y
  for t = T−1 down to 1:
    for i = 1..k_t:
      δ_{t,i} ← Σ_j W_{t,j,i} · δ_{t+1,j} · σ'(a_{t+1,j})
  for each edge (v_{t-1,j}, v_{t,i}) in E:
    partial derivative ← δ_{t,i} · σ'(a_{t,i}) · o_{t-1,j}
  return the vector of partial derivatives

SGDForNetworks(S, (V, E), σ, τ, η_1..η_τ, λ):
  w^(1) ← random, close to 0
  for i = 1..τ:
    draw (x, y) from S
    v_i ← Backpropagation(x, y, w^(i))
    w^(i+1) ← w^(i) − η_i (v_i + λ w^(i))
  return the w^(i) that does best on a validation set
```

**Cost.** The forward pass touches each edge once, a multiply and an add, and applies \(\sigma\) once per neuron: \(O(|V| + |E|)\) time. The backward pass touches each edge once more for the \(\delta\)s and once more for the partial derivatives: also \(O(|V| + |E|)\). So a full gradient costs a small constant times one forward pass, against \(|E|\) forward passes for nudging. (With phantom edges and full matrices, read \(|E|\) as \(\sum_t k_t k_{t+1}\).) Space is \(O(|V|)\) for the stored \(\mathbf a_t\), \(\mathbf o_t\) and \(\boldsymbol\delta_t\), plus \(O(|E|)\) for the weights and the gradient. SGD for \(\tau\) steps costs \(O(\tau(|V| + |E|))\) time and \(O(|V| + |E|)\) space, plus whatever it takes to score the iterates on the validation set. Nothing here is slow. What is not guaranteed is the *result*: SGD may stop at a poor local minimum, which is why it is restarted.

**Build it yourself.** Using only the standard library, implement a layered sigmoid network with any number of layers. A network `Ws` is a list of weight matrices; `Ws[t][i]` lists the weights into neuron \(i\) of layer \(t + 1\), with the last entry for the constant neuron of layer \(t\). Every layer except the output gets a constant 1 appended to its outputs.

```python
import math, random

def sigma(a): ...
def forward(Ws, x): ...
def backprop(Ws, x, y): ...
def sgd(S, Ws, steps, eta0, rng): ...

def inside(u):
    s = [3 * u[0] - u[1] + 1.2, -u[0] + 3 * u[1] + 1.2]
    return min(s + [-u[0] - u[1] + 0.4]) > 0

def loss(Ws, x, y):
    o = forward(Ws, x)[1][-1]
    return sum((p - q) ** 2 for p, q in zip(o, y)) / 2

def init(sizes, rng):
    return [[[rng.uniform(-0.5, 0.5) for _ in range(n + 1)]
             for _ in range(k)] for n, k in zip(sizes, sizes[1:])]

def risk(Ws, S):
    return sum((forward(Ws, x)[1][-1][0] > 0.5) != y[0]
               for x, y in S) / len(S)

if __name__ == "__main__":
    WE2 = [[[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]],
           [[2.0, -1.0, 0.5]]]
    out = forward(WE2, [0.5, -0.5])[1][-1][0]
    assert round(out, 4) == 0.8299
    Ws = init([2, 4, 3, 2], random.Random(7))
    x, y, h = [0.3, -0.8], [1.0, 0.0], 1e-6
    g = backprop(Ws, x, y)
    for t, W in enumerate(Ws):
        for i, row in enumerate(W):
            for j in range(len(row)):
                row[j] += h
                up = loss(Ws, x, y)
                row[j] -= h
                slope = (up - loss(Ws, x, y)) / h
                assert abs(slope - g[t][i][j]) < 1e-5
    rng = random.Random(0)
    S = []
    for _ in range(2000):
        u = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        S.append((u, [float(inside(u))]))
    runs = [sgd(S, init([2, 3, 1], rng), 60_000, 1.0, rng)
            for _ in range(5)]
    best = min(risk(Ws, S) for Ws in runs)
    assert best <= 0.08
    print("all asserts pass", round(best, 3))
```

1. `forward(Ws, x)` returns two lists, the inputs \(\mathbf a_t\) and the outputs \(\mathbf o_t\) of every layer, with `o[0]` the input plus the constant.
2. `backprop(Ws, x, y)` returns gradients shaped exactly like `Ws`, for the squared loss and any number of outputs. The second assert checks every entry against nudging, on a network with two hidden layers and two outputs.
3. `sgd` draws a random example per step, uses the step size \(\eta_i = \eta_0/\sqrt{1 + i/m}\) with \(m = |S|\), updates `Ws` in place and returns it. The last assert trains five networks with three feature neurons on 2,000 triangle messages and asks that the best one mislabel at most 8% of them; a correct implementation lands near 2%.
4. Then switch the triangle for the two-triangle star of “Seeing it in the plane”. How many feature neurons does depth 2 need now, and does depth 3 need fewer?

## Fix the bug

Back to the new wave. Your users can label 12,000 messages this month, and every message comes with its two raw signals. Remember the world: junk fills the triangle, 16% of the mail, and about 2% of labels are careless. A filter that says keep to everything is wrong on \(0.16 \cdot 0.98 + 0.84 \cdot 0.02 \approx 17.4\%\) of messages. Your manager wants a filter whose true risk is at most 10%, **certified**: with confidence 95% (\(\delta = 0.05\)), the certificate from “Sigmoid networks and the bits of a weight” must come out at 0.10 or less. A low error on one test set does not count. Measuring is not certifying, as lesson 11 taught you.

In the lab you choose the number of feature neurons \(k\) in a depth 2 sigmoid network, and how many labels to spend. The lab trains your network with SGD and backpropagation, from random starts, exactly as above. It then rounds every weight to 8 bits (one of 256 values between −16 and 15.875, in steps of 1/8), so that the network belongs to the finite class the certificate covers. It reports the training error \(L_S\), the margin, and the certificate. You may train again from new random starts and keep the run with the lowest training error. That is allowed, and it is worth seeing why: the certificate holds for every network in the class at once, so it is not fooled by choosing among runs.

Before you start, use the table of margins. Which values of \(k\) can carve the triangle at all? Which leave enough room under 0.10 for a training error of about 3 to 4%, given that 2% of labels are wrong no matter what? And what happens to the instinct “bigger network, better filter”?

[[lab-act2]]

## What people get wrong

**“A network with one hidden layer can compute anything, so depth and size don’t matter.”** It can compute every Boolean function, but the size that takes is exponential: at least about \(2^n/n\) edges for \(n\) bits, by the VC bound. What polynomial-size networks can compute is everything that runs in polynomial time, which is plenty, but it is a statement about size.

**“The label cost of a network grows with the number of neurons.”** It grows with the number of *edges*, the weights. For sign networks the VC dimension is \(O(|E| \log |E|)\). Two networks with the same number of neurons can differ enormously in edges.

**“The VC bound covers sigmoid networks too.”** Its proof needs each neuron’s output to take finitely many values. Sigmoid networks with real weights can have VC dimension of order \(|E|^2\). What rescues the engineer is the discretization trick: \(b\)-bit weights give at most \(2^{b|E|}\) networks and VC dimension at most \(b|E|\).

**“Gradient descent finds the best weights.”** Lesson 14’s guarantee needs a convex loss, and the loss of a network is not convex. Worse, finding the ERM weights is NP-hard even for three feature neurons, and is believed hard even approximately. SGD is a heuristic that often works; the certificate is what tells you what the result is worth.

**“Start every weight at zero, as lesson 14 did.”** In a fully connected layer, zero starts make all neurons of a layer identical forever, so the layer acts like one neuron. Start at small random values.

**“Choosing the best of several runs is cheating.”** Not when the guarantee is uniform over the class: the certificate bounds the true risk of every network in the class at once, including the one you picked.

## Recap and the next bug

1. A **feedforward neural network** is a layered directed graph of **neurons**, with a **weight** on every edge and an **activation function** \(\sigma\) (sign, threshold or sigmoid). Each neuron computes \(o = \sigma(\sum w\,o_{\text{below}})\). The **hidden layers** compute features nobody designed.
2. An **architecture** \((V, E, \sigma)\) defines the hypothesis class \(\mathcal H_{V,E,\sigma}\): every choice of weights.
3. Depth 2 sign networks compute every Boolean function, AND and OR are single neurons, and a program running in time \(T(n)\) fits in a network of size \(O(T(n)^2)\). In the plane, depth 2 carves convex polytopes and depth 3 carves unions of them.
4. Cutting a sign network into layers and neurons multiplies growth functions: \(\tau_{\mathcal H}(m) \le (em)^{|E|}\), so the VC dimension is \(O(|E| \log |E|)\). Hence most Boolean functions need networks of exponential size. With \(b\)-bit weights, any activation, at most \(b|E|\), and an explicit certificate.
5. ERM for networks is NP-hard, even with three feature neurons. In practice networks are trained by **SGD** from random starts, with **backpropagation** computing the whole gradient in one forward and one backward pass, \(O(|V| + |E|)\) time.

Your new filter catches the triangle, with a certificate to show for it. Then the junk changes again. And again, the very next morning. The spammers have noticed the filter: they now send their mail one message at a time, watch what your filter decides, and change tactics after every decision. No fixed distribution describes them any more, and every guarantee in this course so far began with “draw the examples i.i.d. from \(\mathcal D\)”. Can any learner keep up when the examples are chosen by an adversary? That is the next lesson: online learning.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Take a depth 2 sigmoid network with three feature neurons and every weight set to 0, and run SGD with backpropagation. Explain, using the backpropagation formulas, why after any number of steps all three feature neurons have identical incoming weights. Then: why can SGD not be run at all on a network with the sign activation?
2. In Step 3 of the VC proof, the bound \(\tau \le (em)^{d}\) for one neuron is claimed in three cases: \(1 \le d \le m\), \(m < d\), and \(d = 0\). Justify each. Then name the step that breaks for sigmoid neurons with real weights, and say exactly why.
3. A network has \(n = 10\) raw signals (plus the constant), one hidden layer of 20 sign neurons plus a constant neuron, and one output neuron, fully connected between adjacent layers. How many edges does it have? What is the largest \(m\) with \(m \le |E| \log_2(em)\), the bound on a shattered set from Step 5? If instead the weights are stored with 8 bits each, what is the VC bound, and how many labels make the certificate’s margin at most 0.05 with \(\delta = 0.05\)?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (networks approximate every Lipschitz function).** A function \(f: [-1, 1]^n \to [-1, 1]\) is \(\rho\)-Lipschitz when \(|f(x) - f(x')| \le \rho \|x - x'\|\) for all \(x, x'\). Fix \(\varepsilon > 0\). Construct a sigmoid network \(N: [-1, 1]^n \to [-1, 1]\) with \(|f(x) - N(x)| \le \varepsilon\) for every \(x \in [-1, 1]^n\). (Think of the boxes of lesson 19.)
- **Problem 2 (smooth universality also costs exponential size).** Fix \(\varepsilon \in (0, 1)\). Let \(s(n)\) be the smallest size of a graph whose sigmoid class can approximate, to within \(\varepsilon\), every 1-Lipschitz function from \([-1, 1]^n\) to \([-1, 1]\). Prove that \(s(n)\) is exponential in \(n\). You may use the fact that a sigmoid network whose weights are stored with a bounded number of bits has VC dimension at most a constant times its number of edges.
- **Problem 3 (training a tiny network is NP-hard).** Prove the fact from “The price: training is hard”: for \(k \ge 3\), ERM for sign networks with one hidden layer of \(k\) neurons plus a constant, and one output, is NP-hard. Build on lesson 8’s second problem about intersections of halfspaces.
