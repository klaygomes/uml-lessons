# Decision Trees

## The filter nobody can read

The letter from the compliance office is polite and completely firm. From next quarter, every filtering decision must be explainable to an auditor. Not “the system is accurate on average”, but this message, on this day, went to the junk folder because of these reasons, written so that a person can read them, check them, and object to them.

You open the filter you have spent the last several lessons building and try to write the explanation for one message. It went to junk because its junk score beat the scores of the other three folders by 0.12. Each score is one block of the weight vector applied to the message’s word counts, and each block has a few thousand entries: 0.31 on “free”, minus 0.07 on “meeting”, 0.004 on “Tuesday”, minus 0.0009 on “regards”, and so on, for every word in the vocabulary. Each entry nudges a score a little. None of them is a reason anybody can read. The auditor looks at your printout of four blocks of numbers and asks the obvious question: “So why was it junk?”

You can’t answer. The filter is accurate, it comes with a certificate, and it is unreadable. That is the bug of this lesson.

What would a readable filter look like? Probably like what you would write yourself if someone asked you to hand-code one: “If the sender is unknown, look at the subject; if it says FREE, junk. If it doesn’t, check for a link…” A cascade of yes/no questions, each about one feature, ending in a label. That object has a name, the **decision tree**, and this lesson is about learning one: how many examples a tree needs, how to grow one greedily, how to cut it back so that its certificate holds and an auditor can still read it, how to handle features that are numbers instead of yes/no, and what you give up when you replace one tree by a forest of them.

## A tree is a stack of readable rules

Let’s go back to the four features of lesson 1, written as four bits FLUN: F, the subject says “FREE”; L, the message has a link; U, the sender is unknown; N, it was sent at night. Labels are back to 1 for junk and 0 for keep, after the \(\pm 1\) and the folders of the last few lessons.

Here is a tree an auditor would be happy with.

[[diagram:tree]]

To label a message, you start at the top, at the **root**, and answer its question: is the sender unknown? If not, you follow the “no” branch and land on a box that just says keep. That is a **leaf**: it carries a label, and it is where the journey ends. If the sender is unknown, you follow the “yes” branch to another question, “does the subject say FREE?”, and so on, until you reach a leaf. The questions sit at the **internal nodes**, and each internal node has exactly two children, one for “yes” and one for “no”.

The beauty of this object is that every leaf is a readable rule. Read the path from the root to a leaf and you get a sentence. The four paths of this tree say:

- the sender is known: keep;
- the sender is unknown and the subject says FREE: junk;
- the sender is unknown, no FREE, but there is a link: junk;
- the sender is unknown, no FREE, no link: keep.

That is exactly the explanation compliance asked for. For any message, the auditor can follow its path and read off the reasons, one question at a time.

Let’s make it formal. We take the domain to be \(\mathcal X = \{0, 1\}^d\): every message is a vector of \(d\) yes/no features, and \(x_i\) is feature \(i\). As in lesson 1, \(\mathcal X\) is the set of all messages and \(\mathcal Y\) the set of labels. The braces \(\{0, 1\}\) hold the two possible answers to one question, and the exponent \(d\) means “\(d\) answers in a row”, so \(\{0,1\}^d\) is the set of all strings of \(d\) bits. With FLUN, \(d = 4\) and there are \(2^4 = 16\) possible messages; 1110 is the message with F, L and U present and N absent, so \(x_3 = 1\) says its sender is unknown. And \([d]\) is shorthand for the feature numbers \(\{1, \dots, d\}\).

**Definition (decision tree).** A **decision tree** over \(\{0,1\}^d\) is a finite rooted binary tree in which every internal node is labelled by a feature index \(i \in [d]\) and has exactly two children, a “yes” child and a “no” child, and every leaf is labelled by a value in \(\mathcal Y = \{0, 1\}\). The hypothesis \(h: \mathcal X \to \{0, 1\}\) it defines works like this: start at the root; at an internal node labelled \(i\), go to the “yes” child if \(x_i = 1\) and to the “no” child if \(x_i = 0\); when you reach a leaf, return its label. The test at a node, \(\mathbb 1[x_i = 1]\), is called its **splitting rule**. The **size** of a tree is its number of nodes, \(n\), counting both internal nodes and leaves, and its **depth** is the largest number of internal nodes on a root-to-leaf path.

Let’s read the notation in that definition.

- \(i \in [d]\) reads “\(i\) is one of the feature numbers 1 to \(d\)”: each question is about one feature.
- \(h: \mathcal X \to \{0, 1\}\) says \(h\) is a function that takes a message and returns a label.
- \(\mathbb 1[\cdot]\) is the **indicator**: it is 1 when the statement inside the brackets is true and 0 when it is false. So \(\mathbb 1[x_i = 1]\) is just the answer to “is feature \(i\) present?”, written as a bit.

In plain words: **a decision tree labels a message by answering one yes/no question per internal node, starting at the root, until it reaches a leaf, whose label it returns.** Push the message 1110 through the auditor’s tree: the root asks about U, and \(\mathbb 1[x_3 = 1] = 1\), so we go “yes”; the next node asks about F, again 1, “yes” again, and we land on the leaf junk. Here is the same walk in Python, with the tree written as nested triples (question, “yes” child, “no” child) and a leaf as a bare label:

```python
tree = ("U", ("F", 1, ("L", 1, 0)), 0)
idx = {"F": 0, "L": 1, "U": 2, "N": 3}

def h(t, x):
    while not isinstance(t, int):
        i, yes, no = t
        t = yes if x[idx[i]] == 1 else no
    return t

print(h(tree, (1, 1, 1, 0)), h(tree, (0, 1, 1, 1)),
      h(tree, (1, 1, 0, 0)), h(tree, (0, 0, 1, 0)))
```

It prints `1 1 0 0`: 1110 is junk by FREE, 0111 is junk by its link, 1100 is kept because the sender is known, and 0010 is kept because it has neither FREE nor a link.

The tree in the picture has three internal nodes (U, F, L) and four leaves, so \(n = 7\), and its depth is 3.

Why insist that every internal node has two children? A node with only one child asks a question and then ignores the answer. You can always replace it by its child without changing what the tree predicts, so we lose nothing by forbidding it, and it makes the counting below much cleaner.

When features are numbers rather than bits, say the hour a message was sent, the natural splitting rule is a threshold, \(\mathbb 1[x_i < \theta]\) for a feature \(i\) and a cut-off number \(\theta\) (the Greek letter theta): the question “is the hour below 6?” is \(\mathbb 1[x_i < 6]\), which is 1 for a message sent at 2.0 and 0 for one sent at 9.0. That is lesson 10’s decision stump, used as a question inside a tree. We will come back to real-valued features near the end. Until then, everything is yes/no.

## How big can a tree get?

A tree is a hypothesis, so the first question from lesson 2 onward is always the same: how large is the class, and how many examples does it need?

Start with threshold splits on real-valued features, the most generous setting. Take \(k\) messages and one numeric feature on which their values are all different, say \(1, 2, 3, 4, 5\) for \(k = 5\). Build a “chain” tree: the root asks “is the value below 1.5?”, and its “yes” child is a leaf; its “no” child asks “is it below 2.5?”, with a leaf on the “yes” side, and so on, until the last question “is it below 4.5?” has a leaf on each side. That is \(k - 1 = 4\) internal nodes and \(k = 5\) leaves, and each of the five messages lands in its own leaf. Now any labelling of the five can be produced by writing the right label into each leaf.

```python
import itertools

points = [1.0, 2.0, 3.0, 4.0, 5.0]
cuts = [1.5, 2.5, 3.5, 4.5]

def chain(labels):
    def h(x):
        for theta, y in zip(cuts, labels):
            if x < theta:
                return y
        return labels[-1]
    return h

k = len(points)
ok = all(tuple(chain(ys)(x) for x in points) == ys
         for ys in itertools.product([0, 1], repeat=k))
print(k, len(cuts) + 1, ok)
```

It prints `5 5 True`: with five leaves, all 32 labellings of the five messages are produced. Recall from lesson 6 that a class **shatters** a set when it produces every labelling of it, and that the **VC dimension** is the size of the largest set it shatters. The argument works for any \(k\), so:

**Proposition (trees with \(k\) leaves shatter \(k\) points).** For every \(k \ge 1\), threshold trees with \(k\) leaves shatter some set of \(k\) messages. So the class of threshold trees of unrestricted size has infinite VC dimension, and by lesson 6’s theorem it is not PAC learnable: no finite number of examples is enough to guarantee, for every distribution, a small true risk with high confidence.

*Proof.* Pick \(k\) messages whose values on one feature are distinct, and sort them so that \(v_1 < v_2 < \dots < v_k\). Put the cuts \(\theta_j = (v_j + v_{j+1})/2\) for \(j = 1, \dots, k - 1\): cut number \(j\) sits halfway between the \(j\)-th smallest value and the next one, which is how the snippet got its cuts 1.5, 2.5, 3.5, 4.5 from the values 1 to 5. Then build the chain tree whose \(j\)-th internal node asks \(\mathbb 1[x < \theta_j]\), with a leaf on its “yes” side and the next internal node (or, after \(\theta_{k-1}\), a final leaf) on its “no” side. Message \(j\) answers “no” to the questions \(\theta_1, \dots, \theta_{j-1}\), since \(v_j > \theta_{j'}\) for \(j' < j\), and “yes” to \(\theta_j\), since \(v_j < \theta_j\). So it stops at the \(j\)-th leaf, and message \(k\) reaches the final leaf. The \(k\) messages land in \(k\) different leaves, and writing \(y_j\) into leaf \(j\) produces any labelling \((y_1, \dots, y_k)\) we want. Since this works for every \(k\), no finite number bounds the VC dimension. \(\blacksquare\)

With yes/no features the class is finite, since there are only finitely many functions from \(\{0,1\}^d\) to \(\{0,1\}\), but it is still enormous. Any such function at all can be written as a tree: ask about feature 1 at the root, about feature 2 at both children, and so on down to feature \(d\), giving a full tree with \(2^d\) leaves, one per message, and write the function’s value into each leaf. That tree has depth \(d\) (so \(d + 1\) levels of nodes), and proving that it works is the first problem at the end of this lesson. The consequence is that trees over \(\{0,1\}^d\) shatter the whole domain, so their VC dimension is \(2^d\). With the 29 features our auditors approved, that is \(2^{29} = 536{,}870{,}912\). By the fundamental theorem of lesson 6, the number of examples needed to PAC learn the class grows linearly with that. Hopeless.

So we are back where lesson 7 left us: a class that is too large to learn as a whole, which is the union of small classes that are each easy to learn. Small trees are few; big trees are many. What we need is a way to say “a small tree that fits the data is trustworthy, and a big one needs more evidence”, and lesson 7 built exactly that tool.

## Paying for size: the description length of a tree

Recall lesson 7’s **minimum description length** bound. A **description language** for a class \(\mathcal H\) maps each hypothesis \(h\) to a binary string \(d(h)\), its description, whose length is the **description length** \(|h|\). The language is **prefix-free** when no description is a prefix of another. The MDL bound then says: for every distribution \(\mathcal D\), every \(m\) and every \(\delta \in (0,1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\),

\[ \forall h \in \mathcal H, \qquad L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{|h| + \log(2/\delta)}{2m}}, \]

where \(\log\) is the natural logarithm. Let’s read it from the inside out, since it is the certificate this whole lesson leans on.

- \(m\) is the number of labelled examples, and \(S \sim \mathcal D^m\) reads “\(S\) is drawn from \(\mathcal D^m\)”: \(m\) examples, each drawn independently from the same distribution \(\mathcal D\) over messages and their labels.
- \(\delta\) (delta) is the chance we allow the certificate to fail, and \(\delta \in (0,1)\) says it is strictly between 0 and 1. The phrase “with probability at least \(1 - \delta\)” says that for all but a fraction \(\delta\) of the possible draws of \(S\), what follows holds.
- \(|h|\) is the number of bits in \(h\)’s description. The bars mean “how many”, here how many bits, and not an absolute value.
- \(\log(2/\delta)\) is the price of confidence: the smaller \(\delta\), the bigger this term.
- The fraction adds the two and divides by \(2m\), so more data shrinks it; the square root \(\sqrt{\cdot}\) turns it into the **penalty**.
- \(L_S(h)\) is the empirical risk, the fraction of the \(m\) examples \(h\) gets wrong, and \(L_{\mathcal D}(h)\) is the true risk, the chance that \(h\) mislabels a fresh message drawn from \(\mathcal D\).
- \(\forall h \in \mathcal H\) reads “for every \(h\) in \(\mathcal H\)”: one draw of \(S\) makes the inequality true for all hypotheses at once.

In plain words: **except on a fraction \(\delta\) of unlucky training sequences, every hypothesis’s true risk is at most its training error plus a penalty that grows with its description length and shrinks with the number of examples.** Say some hypothesis has a 40-bit description and gets 10% of 1,000 examples wrong, and we want \(\delta = 0.05\). Then \(\log(2/0.05) = \log 40 \approx 3.689\), the penalty is \(\sqrt{43.689/2000} \approx 0.148\), and the certificate reads \(L_{\mathcal D}(h) \le 0.248\).

```python
import math

LS, bits, m, delta = 0.1, 40, 1000, 0.05
log_term = math.log(2 / delta)
penalty = math.sqrt((bits + log_term) / (2 * m))
print(round(log_term, 3), round(penalty, 3))
print(round(LS + penalty, 3))
```

It prints `3.689 0.148` and `0.248`. Remember the 40 bits; they will come back as a tree.

The prior knowledge sits in the language: hypotheses with short descriptions get small penalties. And lesson 7’s one warning: the language must be fixed before the training sequence is seen.

So all we need is a prefix-free language in which small trees have short descriptions. Here is one. Write a tree with \(n\) nodes as \(n + 1\) **blocks** of equal length. The first \(n\) blocks list the nodes in **preorder**: the root first, then the whole “yes” subtree in preorder, then the whole “no” subtree in preorder. The last block marks the end. Each block says one of the following:

- an internal node with splitting rule \(\mathbb 1[x_i = 1]\), for some \(i \in [d]\) (that is \(d\) options);
- a leaf with value 1;
- a leaf with value 0;
- end of the code.

That makes \(d + 3\) options per block. How many bits does a block need? With \(b\) bits you can write \(2^b\) different strings, so we need the smallest \(b\) with \(2^b \ge d + 3\): for \(d = 4\) there are 7 options, 2 bits give only 4 strings and 3 bits give 8, so \(b = 3\). In symbols,

\[ b = \lceil \log_2(d + 3) \rceil, \qquad \text{description length} = (n + 1)\, b. \]

Here \(\log_2\) is the base-2 logarithm, the power you must raise 2 to: \(\log_2 8 = 3\), and \(\log_2 7 \approx 2.807\). The brackets \(\lceil \cdot \rceil\) are the **ceiling**, rounding up to the next whole number, so \(\lceil 2.807 \rceil = 3\). In plain words: **each block gets just enough bits to name one of the \(d + 3\) options, and a tree with \(n\) nodes is \(n + 1\) blocks.** The auditor’s tree, with \(n = 7\) and \(d = 4\), takes \(8 \times 3 = 24\) bits.

Let’s write down the auditor’s tree. With \(d = 4\) there are 7 options, so blocks have 3 bits. Number the symbols F, L, U, N as 0 to 3, “leaf 1” as 4, “leaf 0” as 5 and “end” as 6. The preorder of the tree is U, then its “yes” subtree (F, leaf 1, L, leaf 1, leaf 0), then its “no” subtree (leaf 0).

```python
import math

d = 4
b = math.ceil(math.log2(d + 3))
F, L, U, N, LEAF1, LEAF0, END = range(d + 3)
tree = [U, F, LEAF1, L, LEAF1, LEAF0, LEAF0]
code = "".join(format(s, f"0{b}b") for s in tree + [END])
print(b, len(code))
print(code)
```

It prints `3 24` and `010000100001100101101110`: eight blocks of three bits, 24 bits for the whole tree. The line `b = math.ceil(math.log2(d + 3))` is the ceiling formula term by term, and the length of `code` is \((n + 1) b\).

[[diagram:code]]

Two things need checking before lesson 7’s bound applies: no description is a proper prefix of another, and two different trees never get the same description.

**Lemma (the tree code is prefix-free).** For every \(d\), the preorder code with an end block assigns distinct trees distinct descriptions, and no description is a proper prefix of another.

*Proof.* The proper-prefix part is quick. The end symbol appears in a description exactly once, as its last block. If a description \(c\) were a proper prefix of a longer description \(c'\), then \(c'\) would contain \(c\)’s end block at a position before its own last block, which never happens.

For the other part, we show that the blocks of a tree’s preorder can be read back in only one way. Read the blocks from left to right and keep a counter of **open slots**, starting at 1 for the root. A block that names an internal node fills one slot and opens two (its children), so the counter goes up by 1. A leaf block fills one slot and opens none, so the counter goes down by 1. The claim is: if we start reading the preorder of a subtree when the counter is \(c_0 \ge 1\), then the counter stays at least \(c_0\) until the subtree’s last block, and that last block brings it to exactly \(c_0 - 1\).

We prove the claim by induction on the size of the subtree. A single leaf is one block taking \(c_0\) to \(c_0 - 1\). For an internal node, its own block takes the counter to \(c_0 + 1\). By the induction hypothesis applied to the “yes” subtree, starting from \(c_0 + 1\), the counter stays at least \(c_0 + 1\) inside it and ends at \(c_0\). Then the “no” subtree starts at \(c_0\), stays at least \(c_0\) and ends at \(c_0 - 1\). At no point before the last block did the counter drop below \(c_0\), which proves the claim.

Now suppose two trees have the same description. Their first blocks agree, so their roots agree. If the root is a leaf, both trees are that leaf. If it is an internal node, the claim says the “yes” subtree’s code ends at the first moment after the root block that the counter falls back to \(c_0 = 1\), and that moment is the same for both descriptions since they are the same string. So the two “yes” subtrees have the same description, and so do the two “no” subtrees, which are what remains before the end block. By induction on size, the subtrees are equal, so the trees are equal. \(\blacksquare\)

Notice that the counter argument shows something a little stronger: a tree’s preorder already announces its own end, because the counter first hits 0 exactly at its last node. The end block is a convenience that buys the one-line proof of the first part, at the price of one block.

Here is the lemma on a small case. The snippet lists every tree with at most 7 nodes over \(d = 2\) features, writes each in preorder with an end block, and checks for prefix clashes and duplicate descriptions.

```python
import itertools

def trees(n, d):
    if n == 1:
        return [("1",), ("0",)]
    out = []
    for a in range(1, n - 1, 2):
        for l, r in itertools.product(trees(a, d),
                                      trees(n - 1 - a, d)):
            out += [(f"x{i}",) + l + r for i in range(1, d + 1)]
    return out

C = [t + ("end",) for n in (1, 3, 5, 7) for t in trees(n, 2)]
clash = any(a != b and b[:len(a)] == a for a in C for b in C)
print(len(C), clash, len(set(C)) == len(C))
```

It prints `714 False True`: 714 trees, no description a prefix of another, all descriptions distinct.

There is one subtlety left. A hypothesis is a function, and many trees compute the same function (the “ignored question” trees, for a start). So define the description of a function \(h\) as the code of a smallest tree that computes it, breaking ties by any fixed rule chosen in advance. Different functions then get codes of different trees, which are distinct and prefix-free by the lemma. Now we can state the certificate.

**Theorem (the description-length bound for trees).** Let \(\mathcal X = \{0,1\}^d\) and \(b = \lceil \log_2(d + 3) \rceil\). For every distribution \(\mathcal D\) over \(\mathcal X \times \{0,1\}\), every \(m\) and every \(\delta \in (0, 1)\), with probability at least \(1 - \delta\) over \(S \sim \mathcal D^m\), every decision tree \(h\) with \(n\) nodes satisfies

\[ L_{\mathcal D}(h) \le L_S(h) + \sqrt{\frac{(n + 1)\, b + \log(2/\delta)}{2m}}. \]

This is the MDL bound we just read, with the tree’s bit count \((n + 1) b\) in the place of \(|h|\), so nothing new needs naming. In plain words: **a tree’s true risk is at most its training error plus a penalty paid per node.** With the 29 approved features, \(b = \lceil \log_2 32 \rceil = 5\), so a 7-node tree costs \(8 \times 5 = 40\) bits: the 40 bits of the earlier snippet, which certified 0.248 at 1,000 examples. The `tree_bound` snippet below computes exactly this right-hand side.

*Proof.* Let \(\mathcal H\) be the set of functions computed by decision trees over \(\{0,1\}^d\), described as above by a smallest tree. By the lemma this description language is prefix-free, and it is fixed before any data is seen: it depends only on \(d\). So lesson 7’s MDL bound applies. With probability at least \(1 - \delta\), every \(h \in \mathcal H\) satisfies \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{(|h| + \log(2/\delta))/(2m)}\). Take any tree with \(n\) nodes and let \(h\) be its function. A smallest tree for \(h\) has \(n' \le n\) nodes, so \(|h| = (n' + 1) b \le (n + 1) b\). The square root grows with its numerator, so replacing \(|h|\) by \((n + 1)b\) can only loosen the bound. The bound then holds for every tree simultaneously, on the same event of probability at least \(1 - \delta\). \(\blacksquare\)

Two words in that statement do a lot of work. “Every” means the bound holds for all trees at once, so it still applies to a tree you chose after looking at \(S\), for instance the one your algorithm grew. And \(L_{\mathcal D}\) is the true risk with respect to any distribution, including noisy labels, so nothing here assumes that some tree is perfect.

The bound performs a tradeoff, and it is the whole story of this lesson in one formula. Bigger trees can fit the training sequence better, so their empirical risk \(L_S(h)\) is smaller, but they pay through \(n\). Smaller trees pay little for size but may fit worse. The prior knowledge we are betting on is that some tree is both small and accurate.

Let’s put numbers in. Our auditors approved \(d = 29\) features, so \(d + 3 = 32\) and each block is exactly 5 bits. Here is the bound for two trees grown on 1,000 labelled messages, with \(\delta = 0.05\): a big tree of 321 nodes that fits every example, and a small one of 7 nodes with empirical risk 0.096. Then the same small tree with 1,500 labels, and the number of labels a 7-node tree needs before its square-root term drops to 0.15.

```python
import math

def tree_bound(LS, n, d, m, delta=0.05):
    bits = (n + 1) * math.ceil(math.log2(d + 3))
    return LS + math.sqrt((bits + math.log(2 / delta)) / (2 * m))

print(round(tree_bound(0.0, 321, 29, 1000), 3))
print(round(tree_bound(0.096, 7, 29, 1000), 3))
print(round(tree_bound(0.096, 7, 29, 1500), 3))
need = (8 * 5 + math.log(2 / 0.05)) / (2 * 0.15 ** 2)
print(math.ceil(need))
```

It prints `0.898`, `0.244`, `0.217` and `971`. The big tree is perfect on its training sequence and certified for almost nothing: a true risk of at most 0.898 is not a promise anyone wants. The small tree is wrong on almost one message in ten and certified at 0.244. Where do these two trees come from? From the algorithm we are about to build, run on one draw of the lab’s mail.

## Why not just minimize the bound?

The bound suggests its own learner: search over all trees for the one with the smallest right-hand side. Lesson 7 called this the MDL rule. For trees, it is out of reach. Finding the tree that minimizes this bound is computationally hard: unless \(\mathrm P = \mathrm{NP}\) (lesson 8’s shorthand for “every problem whose answers are easy to check is also easy to solve”, which almost nobody believes), no algorithm solves it in time polynomial in \(n\), \(d\) and \(m\). We met this situation in lesson 8, where the statistics were fine and the computation was not.

So practical tree learners give up on the optimum and build the tree **greedily**: one node at a time, making the locally best choice at each node and never revisiting it. Such algorithms cannot promise the best tree, but they tend to work reasonably well in practice. And the certificate does not care how the tree was found: since the bound holds for every tree at once, any tree a heuristic produces gets an honest certificate.

Here is the general scheme. Start with a tree that is a single leaf, the root, labelled by the majority vote of all the training labels. Then, again and again, look at one leaf and consider splitting it on a feature: replacing it by an internal node whose two children are new leaves. Measure the improvement the split brings with some **gain** measure, and either perform the split with the largest gain or decide not to split that leaf at all.

The whole design question is: what should “gain” measure?

## Measuring a split

We need a world small enough to do with a pencil. Our users label one message of each of the 16 kinds FLUN, and this time the truth is the auditor’s tree from the picture: junk exactly when the sender is unknown and the message says FREE or has a link. Six of the 16 are junk: the ones with \(U = 1\) and at least one of \(F, L\). Eight of the 16 have an unknown sender, and all six junk messages are among them.

Some notation first. Those counts are fractions of the training sequence: 6 of the 16 messages are junk, a fraction \(6/16 = 0.375\); among the 8 with an unknown sender, 6 are junk, a fraction \(6/8 = 0.75\). For a training sequence \(S\) and an event about an example, write \(\mathbb P_S[\,\cdot\,]\) for its probability when the example is drawn uniformly from \(S\): just the fraction of the examples of \(S\) for which it holds. So \(\mathbb P_S[y = 1]\) is the fraction of junk in \(S\), here 0.375. The vertical bar in \(\mathbb P_S[y = 1 \mid x_i = 1]\) reads “given”: it is the fraction of junk among the examples with feature \(i\) present only, and for \(i = U\) it is 0.75.

**Gain from training error.** The most direct idea: a split is good if it reduces the training error. A leaf predicts the majority label of the examples that reach it, so if a fraction \(a\) of them are junk, it is wrong on whichever fraction is the minority. At the root of our tiny world, \(a = 0.375\): the majority is keep, and the leaf is wrong on the six junk messages, a fraction 0.375. Had \(a\) been 0.75, the majority would be junk and the leaf wrong on 0.25. Write this as

\[ C(a) = \min\{a, 1 - a\}, \]

where \(\min\{\cdot\}\) is the smaller of the numbers in the braces. In plain words: **a leaf whose junk fraction is \(a\) mislabels the smaller of the two groups.**

Before the split, the leaf’s training error is \(C(\mathbb P_S[y = 1])\). After splitting on feature \(i\), the examples with \(x_i = 1\) go to one new leaf and those with \(x_i = 0\) to the other, each with its own majority, so the error becomes

\[ \mathbb P_S[x_i = 1]\, C\big(\mathbb P_S[y = 1 \mid x_i = 1]\big) + \mathbb P_S[x_i = 0]\, C\big(\mathbb P_S[y = 1 \mid x_i = 0]\big). \]

Read it from the inside out. Each \(C(\mathbb P_S[y = 1 \mid \dots])\) is the error rate of one new leaf, computed from its own junk fraction. Each is multiplied by \(\mathbb P_S[x_i = \dots]\), the fraction of the examples that land in that leaf, so a leaf that receives few examples counts for little. Adding the two gives the error over the whole of \(S\). In plain words: **after the split, the training error is the error of each new leaf, weighted by how many examples it gets.** Split the root on U: half the messages go each way; the unknown side is 0.75 junk, error \(C(0.75) = 0.25\); the known side has no junk, error 0. The total is \(0.5 \times 0.25 + 0.5 \times 0 = 0.125\), which is 2 mistakes out of 16.

The **gain** is the difference:

\[ \mathrm{Gain}(S, i) = C\big(\mathbb P_S[y = 1]\big) - \Big( \mathbb P_S[x_i = 1]\, C\big(\mathbb P_S[y = 1 \mid x_i = 1]\big) + \mathbb P_S[x_i = 0]\, C\big(\mathbb P_S[y = 1 \mid x_i = 0]\big) \Big). \]

In plain words: **the gain of a feature is how much the training error drops when we split on it.** For U it is \(0.375 - 0.125 = 0.25\). Here are the three pieces in Python, with \(p\) the junk fraction before, \(q\) the fraction with \(U = 1\), and \(p_1\), \(p_0\) the junk fractions on the two sides:

```python
C = lambda a: min(a, 1 - a)
p, q, p1, p0 = 6 / 16, 8 / 16, 6 / 8, 0 / 8
before = C(p)
after = q * C(p1) + (1 - q) * C(p0)
print(before, after, before - after)
```

It prints `0.375 0.125 0.25`.

If no example has \(x_i = 1\), the conditional probability is undefined, but its weight \(\mathbb P_S[x_i = 1]\) is 0, so the term is dropped. Same for \(x_i = 0\).

**Information gain.** Keep the formula and replace \(C\) by a measure of how uncertain the label is in a leaf. A leaf where every message is junk, or every message is keep, holds no uncertainty at all; a leaf that is half junk is as uncertain as a fair coin. The standard measure is the **entropy**, counted in bits: 0 for a pure leaf, 1 for the fair coin, and in between otherwise. For a yes/no outcome with probability \(a\),

\[ C(a) = -a \log_2 a - (1 - a) \log_2 (1 - a), \qquad \text{with } 0 \log_2 0 = 0. \]

Read it from the inside. \(\log_2 a\) is the base-2 logarithm we met in the code length; for a fraction \(a < 1\) it is negative, so \(-\log_2 a\) is positive, and it is large when \(a\) is small: a rare outcome is a big surprise. Each outcome’s surprise is weighted by how often it happens, and the two are added. The convention \(0 \log_2 0 = 0\) says an outcome that never happens adds nothing. In plain words: **entropy is the average surprise of the label, in bits.** At the root, \(a = 0.375\) gives \(0.375 \times 1.415 + 0.625 \times 0.678 \approx 0.954\) bits: nearly a fair coin. The gain is then how much uncertainty the split removes. (Changing the base of the logarithm multiplies every gain by the same constant, so it never changes which feature wins. We use base 2.)

**Gini index.** Another way to measure how mixed a leaf is: draw two of its examples independently. The first is junk and the second keep with probability \(a(1 - a)\), and the other way round with the same probability. Replace \(C\) by the sum,

\[ C(a) = 2a(1 - a). \]

In plain words: **the Gini index is the chance that two examples drawn from the leaf disagree.** At the root it is \(2 \times 0.375 \times 0.625 \approx 0.469\).

```python
import math

def H(a):
    return sum(-p * math.log2(p) for p in (a, 1 - a) if p > 0)

gini = lambda a: 2 * a * (1 - a)
for a in (0.0, 0.5, 0.375, 0.75):
    print(a, round(H(a), 3), round(gini(a), 3))
```

It prints `0.0 0.0 0.0`, `0.5 1.0 0.5`, `0.375 0.954 0.469` and `0.75 0.811 0.375`: both are 0 on a pure leaf and largest on the fair coin. The `if p > 0` is the convention \(0 \log_2 0 = 0\). Put these into the gain snippet in place of `C` and the U split’s information gain is \(0.954 - 0.5 \times 0.811 \approx 0.549\), its Gini gain \(0.469 - 0.5 \times 0.375 \approx 0.281\).

Let’s try all three, on every feature, at the root and one level down.

Before reading on, look at the root. Which feature would you split on, and does training error agree with you?

```python
import itertools, math

def H(a):
    return sum(-p * math.log2(p) for p in (a, 1 - a) if p > 0)

C = {"err": lambda a: min(a, 1 - a), "info": H,
     "gini": lambda a: 2 * a * (1 - a)}
X = itertools.product([0, 1], repeat=4)
S = [(x, int(x[2] and (x[0] or x[1]))) for x in X]
P = lambda T: sum(y for _, y in T) / len(T)

def gain(S, i, c):
    parts = [[z for z in S if z[0][i] == v] for v in (1, 0)]
    after = sum(len(T) / len(S) * c(P(T)) for T in parts if T)
    return c(P(S)) - after

for T in (S, [z for z in S if z[0][2] == 1]):
    for name, c in C.items():
        print(name, [round(gain(T, i, c), 3) for i in range(4)])
```

It prints six lines, the gains of F, L, U, N in that order:

```
err [0.0, 0.0, 0.25, 0.0]
info [0.049, 0.049, 0.549, 0.0]
gini [0.031, 0.031, 0.281, 0.0]
err [0.0, 0.0, 0.0, 0.0]
info [0.311, 0.311, 0.0, 0.0]
gini [0.125, 0.125, 0.0, 0.0]
```

At the root, all three agree: split on U. Training error sees a gain of 0.25, since before the split the majority label is keep and 6 of 16 are wrong, and after it the “unknown sender” side has majority junk with 2 of 8 wrong, while the other side is all keep: \(6/16 - 2/16 = 0.25\). Notice, though, that training error gives F and L a gain of exactly 0, while the other two measures see they are worth something.

Now the second three lines, at the node reached by \(U = 1\): eight messages, six junk. Here training error is stuck. Every feature has gain 0. Splitting on F sends four messages to an all-junk leaf and four to a leaf that is half junk, so the number of training mistakes stays at 2: none on the FREE side, and two on the other side whichever label its leaf gets. Training error cannot tell F, which matters, from N, which is pure noise. Information gain and the Gini index both see at once that F and L are useful and N is not.

Hold on to that observation. It is about to become a theorem.

## The greedy grower: ID3

Now the algorithm. It is a recursive version of the general scheme, a popular one known as **ID3** (“Iterative Dichotomizer 3”). The first call is \(\mathrm{ID3}(S, [d])\): the whole training sequence and all features.

```
ID3(S, A)
  input: training sequence S, set of features A ⊆ [d]
  if every example in S has label 1: return a leaf 1
  if every example in S has label 0: return a leaf 0
  if A is empty: return a leaf labelled by the majority of S
  j ← argmax over i in A of Gain(S, i)
  T1 ← ID3({(x, y) in S : x_j = 1}, A \ {j})
  T2 ← ID3({(x, y) in S : x_j = 0}, A \ {j})
  return the tree whose root asks x_j = 1?,
         with T1 on the yes branch and T2 on the no branch
```

A word on the notation. \(A \subseteq [d]\) says \(A\) is some of the feature numbers, the questions still allowed. The arrow \(\leftarrow\) stores a value. “argmax over \(i\) in \(A\)” returns not the largest gain but the feature that achieves it: at the root of the tiny world the information gains of F, L, U, N are 0.049, 0.049, 0.549 and 0, so the argmax is U. The braces \(\{(x, y) \in S : x_j = 1\}\) are the examples of \(S\) such that feature \(j\) is present, the 8 unknown-sender messages when \(j = U\), and \(A \setminus \{j\}\) is \(A\) with \(j\) removed, here \(\{F, L, N\}\).

Three details are worth a moment. First, a feature, once used on a path, is removed from \(A\) for the rest of that path: asking the same question twice would be pointless, since every example below the node has the same answer. Second, if every example reaching a node has \(x_j = 1\), the “no” child gets an empty \(S\), and the first test passes vacuously (every example of an empty set has label 1), so it becomes a leaf 1 that no training example reaches. Third, ties in the argmax and in the majority need a rule; we break argmax ties toward the lowest feature index and majority ties toward junk.

Run ID3 with any of the three gains on the 16 messages and it rebuilds the auditor’s tree exactly: U at the root, then F, then L, 7 nodes. But break the stuck tie at the \(U = 1\) node toward N instead, as an unlucky rule could, and the training-error version wastes its split on N, then has to ask F and L separately on both sides of it. It still ends with empirical risk 0, but with 13 nodes instead of 7. Its description grows from 24 bits to 42, and every extra bit weakens its certificate.

**Runtime.** Recall from lesson 3 that \(O(\cdot)\) reads “grows at most in proportion to”, and here \(|S|\) is how many examples \(S\) holds. \(\mathrm{Gain}(S, i)\) needs only four counts (how many examples have each combination of \(x_i\) and \(y\)), so it costs \(O(|S|)\). A node that reaches the argmax with sample \(S_v\) and feature set \(A\) costs \(O(|S_v| \cdot |A|)\) for all the gains, plus \(O(|S_v|)\) to split the sample. The nodes at the same depth receive disjoint parts of \(S\), so one level of the tree costs \(O(md)\) in total. Every level removes a feature, so the depth is at most \(d\), and ID3 runs in \(O(md^2)\) time. For space: every internal node’s sample contains both labels, so it has at least two examples, and the internal nodes at one depth have disjoint samples. So there are at most \(m/2\) internal nodes per level and at most \(md/2\) in total, and storing the tree and the data takes \(O(md)\) space. Predicting a label walks one path, \(O(d)\).

That is cheap. What it returns is another matter.

[[lab]]

In the lab, ID3 grows trees on mail with 29 approved features and noisy labels: the truth is the auditor’s tree, but each label has been flipped with probability 0.1, because people misclick. Before you touch it, predict: ID3 is given 1,000 labelled messages and returns a tree with empirical risk 0. About what is its true risk? Then move the slider and watch the tree’s size and its two risks.

## Why the smooth gains are better

The worked example showed training error getting stuck. Let’s prove that this is not bad luck, and that information gain and the Gini index can’t get stuck that way.

Write \(q = \mathbb P_S[x_i = 1]\), \(p_1 = \mathbb P_S[y = 1 \mid x_i = 1]\) and \(p_0 = \mathbb P_S[y = 1 \mid x_i = 0]\). By the law of total probability, the junk fraction before the split is the weighted average of the two after it:

\[ p = \mathbb P_S[y = 1] = q\, p_1 + (1 - q)\, p_0. \]

Take the stuck node of the tiny world, the 8 unknown-sender messages, and split it on F. Half have FREE, so \(q = 0.5\); all 4 of those are junk, \(p_1 = 1\); 2 of the other 4 are junk, \(p_0 = 0.5\). Then \(0.5 \times 1 + 0.5 \times 0.5 = 0.75\), and indeed 6 of the 8 are junk. In plain words: **the junk fraction of a node is the average of its children’s junk fractions, weighted by their sizes.** So every gain in this lesson has the same shape,

\[ \mathrm{Gain}(S, i) = C\big(q p_1 + (1 - q) p_0\big) - \big( q\, C(p_1) + (1 - q)\, C(p_0) \big), \]

the value of \(C\) at an average minus the average of the values. Inside the first \(C\) sits the parent’s junk fraction, 0.75; inside the parentheses, each child’s score weighted by its size. In plain words: **the gain is how much better the parent’s score is than the average score of its children.** For the F split, training error gives \(C(0.75) - (0.5 \times 0 + 0.5 \times 0.5) = 0.25 - 0.25 = 0\), stuck; the entropy gives \(0.811 - 0.5 \times 1 = 0.311\).

```python
import math

def H(a):
    return sum(-p * math.log2(p) for p in (a, 1 - a) if p > 0)

C = {"err": lambda a: min(a, 1 - a), "info": H,
     "gini": lambda a: 2 * a * (1 - a)}
q, p1, p0 = 4 / 8, 4 / 4, 2 / 4
p = q * p1 + (1 - q) * p0
for name, c in C.items():
    g = c(p) - (q * c(p1) + (1 - q) * c(p0))
    print(name, p, round(g, 3))
```

It prints `err 0.75 0.0`, `info 0.75 0.311` and `gini 0.75 0.125`, the F column of the second three lines of the earlier output.

That shape, value at an average minus average of values, is exactly the quantity that **concavity** controls. A function \(C\) on \([0,1]\) (the numbers from 0 to 1, ends included) is concave if \(C(q a + (1-q) a') \ge q C(a) + (1-q) C(a')\) for all \(a, a', q \in [0,1]\): the chord between two points of its graph lies below the graph. With the entropy, \(a = 1\), \(a' = 0.5\) and \(q = 0.5\), the left side is \(H(0.75) = 0.811\) and the right side is \(0.5 \times 0 + 0.5 \times 1 = 0.5\): the curve is above the chord, and the gap 0.311 is the gain we just computed. It is **strictly concave** if the inequality is strict whenever \(a \ne a'\) and \(0 < q < 1\).

[[diagram:gain-curves]]

**Proposition (the three gain measures).**

1. Information gain and the Gini index are upper bounds of the training error: for every \(a \in [0, 1]\), \(2a(1-a) \ge \min\{a, 1-a\}\) and \(-a\log_2 a - (1-a)\log_2(1-a) \ge \min\{a, 1-a\}\).
2. All three gains are never negative.
3. The entropy and the Gini index are strictly concave, so their gains are strictly positive whenever \(0 < q < 1\) and \(p_1 \ne p_0\): whenever the split separates the examples into two groups with different junk fractions. The training-error gain is exactly 0 whenever \(p_1\) and \(p_0\) are on the same side of \(1/2\) (both at most \(1/2\), or both at least \(1/2\)), however different they are.

*Proof.* Part 1. All three functions are symmetric under \(a \mapsto 1 - a\) (replacing \(a\) by \(1 - a\), swapping the roles of junk and keep, leaves them unchanged), so it is enough to take \(a \le 1/2\), where \(\min\{a, 1-a\} = a\). For the Gini index, \(1 - a \ge 1/2\), so \(2a(1-a) \ge 2a \cdot \tfrac12 = a\). For the entropy \(H\), we use concavity, proved in part 3: \(H(0) = 0\) and \(H(1/2) = 1\), and \(a = (1 - 2a) \cdot 0 + 2a \cdot \tfrac12\) is a weighted average of 0 and \(1/2\) with weights \(1 - 2a\) and \(2a\). So \(H(a) \ge (1 - 2a) H(0) + 2a H(1/2) = 2a \ge a\).

Part 3 first, since part 2 uses it. The second derivative measures how fast the slope changes; when it is negative the slope keeps falling, so the graph bends downward and every chord lies below it. For a function with a second derivative, a negative second derivative everywhere on \((0, 1)\) makes it strictly concave there. The Gini index \(2a - 2a^2\) has second derivative \(-4\). The entropy has first derivative \(\log_2\frac{1-a}{a}\) and second derivative \(-\frac{1}{a(1-a)\ln 2}\), where \(\ln 2 \approx 0.693\) is the natural logarithm of 2; at \(a = 0.75\) it is \(-1/(0.1875 \times 0.693) \approx -7.7\), and it is negative on all of \((0,1)\), the numbers strictly between 0 and 1; it is also continuous at 0 and 1, so strict concavity extends to \([0,1]\). Then, with \(0 < q < 1\) and \(p_1 \ne p_0\), strict concavity says \(C(q p_1 + (1-q) p_0) > q C(p_1) + (1-q) C(p_0)\), which is exactly “gain \(> 0\)”.

For the training error, \(C(a) = \min\{a, 1-a\}\) is the minimum of two linear functions, and a minimum of linear functions is concave: for each of the two lines \(\ell\), \(\ell(q a + (1-q)a') = q\ell(a) + (1-q)\ell(a') \ge q C(a) + (1-q) C(a')\), so the smaller of the two lines is also at least that. But on \([0, 1/2]\) it is the single line \(C(a) = a\). If \(p_1, p_0 \le 1/2\), then \(p \le 1/2\) too, and \(C(p) = p = q p_1 + (1-q) p_0 = q C(p_1) + (1-q) C(p_0)\): the gain is exactly 0. The case \(p_1, p_0 \ge 1/2\) is the same with the line \(1 - a\).

Part 2. Each \(C\) is concave, so \(C\) at the average is at least the average of \(C\): every gain is at least 0. When \(q \in \{0, 1\}\), one side is empty and the gain is \(C(p) - C(p) = 0\). \(\blacksquare\)

In words: training error only notices a split when it flips a majority. A split that turns a 75% junk leaf into a 100% junk leaf and a 50% one is real progress, and the next split will finish the job, but training error scores it zero. The smooth measures reward every move toward purer leaves. How often does training error go blind? Draw random splits and see.

```python
import math, random

def H(a):
    return sum(-p * math.log2(p) for p in (a, 1 - a) if p > 0)

err = lambda a: min(a, 1 - a)
gini = lambda a: 2 * a * (1 - a)
grid = [k / 1000 for k in range(1001)]
print(all(H(a) >= err(a) and gini(a) >= err(a) for a in grid))

rng, trials = random.Random(0), 100_000
for c in (err, H, gini):
    gains = []
    for _ in range(trials):
        q, p1, p0 = rng.random(), rng.random(), rng.random()
        p = q * p1 + (1 - q) * p0
        gains.append(c(p) - q * c(p1) - (1 - q) * c(p0))
    print(min(gains) > -1e-12, sum(g < 1e-12 for g in gains))
```

It prints `True`, then `True 49934`, `True 0` and `True 0`. Both smooth measures sit above the training error on the whole grid. No gain is ever negative. And training error scores 49,934 of 100,000 random splits as exactly worthless, about half, as you would expect: \(p_1\) and \(p_0\) land on the same side of \(1/2\) half the time. The smooth measures never do.

On the lab’s mail this shows up in the size of what ID3 grows. On 300 noisy labels, the information-gain and Gini versions of ID3 both grow trees of 107 nodes; the training-error version grows 173.

## Pruning

Still, look back at the lab. Even with information gain, ID3 grows trees with hundreds of nodes. Why? Because it keeps splitting until each leaf is pure or it runs out of features, and with noisy labels, purity means memorizing the noise. Each misclicked label gets its own little branch. The tree ends with empirical risk 0 and, as you saw, a true risk around 0.22, more than twice the 0.1 that the noise forces on any filter. And the description-length bound, which is a certificate an auditor could actually rely on, sees through it: 0.898.

There are two common remedies. One is to stop early: limit the number of iterations, or the depth, so the tree never gets big. The other is to grow the full tree and then **prune** it: walk back over it and cut off branches that do not pay for themselves, aiming for a much smaller tree with nearly the same empirical risk. By the bound, making \(n\) much smaller without raising \(L_S\) by much is exactly what lowers the certificate.

Pruning is usually done bottom-up, from the leaves toward the root, and needs a yardstick: a function \(f(T, m)\) that estimates or bounds the true risk of a tree \(T\) learned from \(m\) examples. The description-length bound is a natural choice. Here is the generic procedure.

```
Prune(T, f, m)
  input: a tree T, an estimate or bound f(T, m) of the
         true risk of a tree learned from m examples
  for each node j of T, in a bottom-up walk (leaves first):
    candidates:
      T with node j replaced by a leaf 1
      T with node j replaced by a leaf 0
      T with node j replaced by its yes subtree
      T with node j replaced by its no subtree
      T unchanged
    T ← the candidate with the smallest f(candidate, m)
  return T
```

[[diagram:prune]]

Replacing a node by one of its subtrees deserves a word. It means deleting the question at node \(j\) and letting every message that reaches \(j\) continue into one chosen branch, whatever its answer. Sometimes the question at the top of a branch is noise and the useful questions are below it; this move keeps them.

Here is what pruning does on one draw of the lab’s mail with 300 labels and \(\delta = 0.05\), using the description-length bound as \(f\). ID3 with information gain grows 107 nodes, empirical risk 0, bound 0.952, true risk 0.220. Pruning cuts it to 7 nodes, 4 leaves, empirical risk 0.100, bound 0.370, and true risk 0.100. Those 7 nodes are the auditor’s tree: U, then F, then L. Pruning threw away a hundred nodes of memorized noise and found the truth underneath.

Is the certificate of the pruned tree legitimate, even though pruning looked at \(S\) to decide what to cut? Yes, and it is worth saying why: the bound holds for every tree at once, on one event of probability at least \(1 - \delta\), and the code was fixed before \(S\) was seen. Choosing which tree to report after seeing \(S\) is fine. Choosing the code after seeing \(S\) would not be.

**Runtime.** The walk visits each of the \(n\) nodes once and evaluates five candidates there. With the bound as \(f\), evaluating a candidate means computing its empirical risk, routing all \(m\) examples down paths of length at most \(d\), plus counting its nodes, so \(O(md + n)\) per candidate. Pruning a tree of \(n\) nodes costs \(O(n(md + n))\) time and \(O(n + md)\) space. Since ID3’s tree has \(n = O(md)\) nodes, this is polynomial, if not elegant.

## Real-valued features: thresholds

Not every feature is yes/no. The hour a message was sent, its length, the number of links: these are numbers, and the natural question about a number is a threshold, \(\mathbb 1[x_i < \theta]\). Do we need a new algorithm? No. We can turn each numeric feature into a handful of yes/no features and run ID3 unchanged.

The trick is that, on a training sequence, a threshold only matters through which examples fall below it. Take the hour of eight training messages: 23.5, 2.0, 14.25, 9.0, 2.0, 19.75, 9.0 and 3.5. Sorted, the distinct values are 2.0, 3.5, 9.0, 14.25, 19.75, 23.5. Any threshold between 3.5 and 9.0, whether 4, 7 or 8.99, splits the eight messages the same way. So it is enough to keep one threshold per gap between neighbouring values, say the midpoint, plus one below everything and one above.

```python
hours = [23.5, 2.0, 14.25, 9.0, 2.0, 19.75, 9.0, 3.5]
v = sorted(set(hours))
mids = [(a + b) / 2 for a, b in zip(v, v[1:])]
thetas = [v[0] - 1] + mids + [v[-1] + 1]

def split(theta):
    return tuple(h < theta for h in hours)

built = {split(t) for t in thetas}
grid = [k / 100 for k in range(-100, 2600)]
print(len(thetas), len(built))
print(all(split(t) in built for t in grid))
```

It prints `7 7` and `True`: seven thresholds producing seven different splits, and every threshold on a fine grid from \(-1\) to 26 produces one of those seven.

Formally, let \(x_1, \dots, x_m\) be the training messages. For each real-valued feature \(i\), sort them so that \(x_{1,i} \le \dots \le x_{m,i}\), and use the conventions \(x_{0,i} = -\infty\) and \(x_{m+1,i} = +\infty\). For each \(j = 0, \dots, m\) with \(x_{j,i} < x_{j+1,i}\), pick a threshold \(\theta_{j,i}\) strictly inside \((x_{j,i}, x_{j+1,i})\), and define the yes/no feature \(\mathbb 1[x_i < \theta_{j,i}]\). That makes at most \(m + 1\) new features per original feature, so at most \(d(m+1)\) in all, about \(dm\). Then run ID3 on them.

The double subscript \(x_{j,i}\) is “feature \(i\) of the \(j\)-th message in sorted order”, \(\pm\infty\) are stand-ins smaller and larger than every value, and \((x_{j,i}, x_{j+1,i})\) is the open gap between two neighbours, endpoints excluded. The condition \(x_{j,i} < x_{j+1,i}\) skips gaps of width zero, which tied values create. In plain words: **sort the training values, and put one threshold in every gap between neighbours that are actually different, including the gap before the first and after the last.** With the eight hours, \(m = 8\) allows up to 9 gaps, but the two 2.0s and the two 9.0s close two of them, leaving 7, as the earlier snippet found. Here is the formal version, sentinels included:

```python
hours = [23.5, 2.0, 14.25, 9.0, 2.0, 19.75, 9.0, 3.5]
inf = float("inf")
x = [-inf] + sorted(hours) + [inf]
m = len(hours)
gaps = [(x[j], x[j + 1]) for j in range(m + 1)
        if x[j] < x[j + 1]]
print(m + 1, len(gaps))
print(gaps[0], gaps[2])
```

It prints `9 7`, then `(-inf, 2.0) (3.5, 9.0)`: the first gap lies below everything, and the third is the one where 4, 7 and 8.99 all live.

**Proposition (thresholds reduce to yes/no features).** For every tree \(T\) whose splitting rules are thresholds \(\mathbb 1[x_i < \theta]\) on the original features, there is a tree \(T'\) over the constructed yes/no features with the same shape, the same number of nodes, and the same label on every training message. In particular, \(L_S(T') = L_S(T)\).

*Proof.* Take a node of \(T\) with rule \(\mathbb 1[x_i < \theta]\). Let \(j\) be the number of training messages whose feature \(i\) is below \(\theta\). In the sorted order those are exactly the first \(j\), so \(x_{j,i} < \theta \le x_{j+1,i}\) (with the conventions at the ends). In particular \(x_{j,i} < x_{j+1,i}\), so the constructed threshold \(\theta_{j,i}\) exists. Now take any training message \(x_t\). Either \(x_{t,i} \le x_{j,i}\), and then \(x_{t,i} < \theta\) and \(x_{t,i} < \theta_{j,i}\); or \(x_{t,i} \ge x_{j+1,i}\), and then \(x_{t,i} \ge \theta\) and \(x_{t,i} > \theta_{j,i}\). Every value falls into one of these two cases, because the values are sorted and \(j\) counts exactly the ones below \(\theta\). So the rules \(\mathbb 1[x_i < \theta]\) and \(\mathbb 1[x_i < \theta_{j,i}]\) agree on every training message.

Build \(T'\) by replacing each node’s rule this way and keeping all leaves. A training message answers every question of \(T'\) exactly as it answers the corresponding question of \(T\), so it follows the same path and reaches the same leaf. Same shape, same number of nodes, same label on every training message. \(\blacksquare\)

**Runtime.** Done naively, the reduction is expensive. There are about \(dm\) constructed features, and computing the gain of each costs \(O(m)\), so evaluating every split at one node takes \(O(dm^2)\) operations. A cleverer implementation brings it to \(O(dm \log m)\), using the same idea as the fast search for the best decision stump in lesson 10: sort, then sweep.

```
BestThreshold(S, i, C)
  sort the examples of S by feature i: x_(1),i ≤ ... ≤ x_(m),i
  left_n ← 0, left_junk ← 0, total_junk ← number of junk in S
  best ← (−∞, none)
  for t = 1..m−1:
    left_n ← left_n + 1
    left_junk ← left_junk + y_(t)
    if x_(t),i < x_(t+1),i:
      q ← left_n / m
      p1 ← left_junk / left_n
      p0 ← (total_junk − left_junk) / (m − left_n)
      g ← C(total_junk / m) − q·C(p1) − (1 − q)·C(p0)
      if g > best.gain: best ← (g, midpoint of the two values)
  return best
```

Each candidate threshold’s gain comes from running counts in \(O(1)\), so after the \(O(m \log m)\) sort, one sweep is \(O(m)\). Over \(d\) features that is \(O(dm \log m)\) per node, and \(O(m)\) extra space for the sorted order. (The thresholds below everything and above everything put all examples on one side and have gain 0, so the sweep can skip them.)

## Build it yourself

Here is the whole pipeline, from scratch, in the standard library. The starter code contains the lab’s mail world: 29 yes/no features, each present with probability 1/2 independently, the auditor’s tree as the truth on features 0, 1, 2 (F, L, U), and each label flipped with probability 0.1. It also contains the 16-message tiny world and the eight hours.

```python
import itertools
import math
import random

d = 29

def f(x):
    return int(x[2] == 1 and (x[0] == 1 or x[1] == 1))

def draw(m, rng, eta=0.1):
    S = []
    for _ in range(m):
        x = tuple(int(rng.random() < 0.5) for _ in range(d))
        y = f(x) if rng.random() >= eta else 1 - f(x)
        S.append((x, y))
    return S

tiny = [(x, int(x[2] and (x[0] or x[1])))
        for x in itertools.product([0, 1], repeat=4)]
hours = [23.5, 2.0, 14.25, 9.0, 2.0, 19.75, 9.0, 3.5]

def gain(S, i, C): ...
def id3(S, A, C): ...
def nodes(T): ...
def encode(T, d): ...
def tree_bound(T, S, d, delta): ...
def prune(T, S, d, delta): ...
def thresholds(values): ...

if __name__ == "__main__":
    T = id3(tiny, list(range(4)), "info")
    assert nodes(T) == 7 and len(encode(T, 4)) == 24
    S = draw(300, random.Random(0))
    T = id3(S, list(range(d)), "info")
    assert nodes(T) == 107
    P = prune(T, S, d, 0.05)
    assert nodes(P) == 7
    assert round(tree_bound(P, S, d, 0.05), 2) == 0.37
    assert len(thresholds(hours)) == 7
```

1. Implement `gain` for the three measures, selected by the names `"err"`, `"info"` and `"gini"`, and `id3` exactly as in the pseudocode, breaking argmax ties toward the lowest feature index and majority ties toward 1. Represent a tree however you like.
2. Implement `encode` with the block code of this lesson (feature \(i\) as the number \(i\), leaf 1 as \(d\), leaf 0 as \(d+1\), end as \(d+2\), each in \(\lceil\log_2(d+3)\rceil\) bits, in preorder with the “yes” subtree first), `tree_bound` with the description-length bound, and `prune` with the generic procedure, the bound as \(f\), and the unchanged tree preferred on ties.
3. Implement `thresholds`, returning one threshold below all values, one per gap between distinct sorted values, and one above. All asserts should pass.
4. Write a function that computes a tree’s true risk exactly in this world, using \(L_{\mathcal D}(h) = 0.1 + 0.8\, \mathbb P[h(x) \ne f(x)]\) (why does that formula hold?) and the fact that each leaf’s region has probability \(2^{-\text{depth}}\). Check that the unpruned tree of step 1 has true risk about 0.22 and the pruned one exactly 0.1.
5. Rerun steps 1 and 2 with training error as the gain. How big is the unpruned tree, and what does pruning return?

## Fix the bug

Back to the compliance office. The auditors set two conditions on the filter they will sign off. First, it must be a single tree they can read: at most 8 leaves, so at most 8 rules. Second, it must come with a certificate: with probability at least 0.95, its true risk is at most 0.25, certified by the description-length bound, not by a hopeful look at fresh mail. Your users can label at most 2,000 messages.

In the lab you choose the gain measure, a depth limit or none, whether to prune, and how many labels to buy. Before you start, use the bound to work out roughly how many labels a 7-node tree with empirical risk around 0.1 needs: the snippet in “Paying for size” has the arithmetic, and 971 is the point where the square-root term alone reaches 0.15. Leave yourself a margin, because the empirical risk of the tree you get will wobble around 0.1 from one draw to the next.

[[lab-act2]]

## Random forests: accuracy for readability

The auditors signed. Before we leave trees, though, there is another way to fight overfitting that is worth knowing, partly because it is popular and partly because of what it costs.

Instead of one tree, grow many and let them vote. A **random forest** is a hypothesis built from a collection of decision trees. Each tree is grown by running an algorithm \(A\) on the training sequence \(S\) together with an extra random vector \(\theta\), with a fresh, independent \(\theta\) for every tree. The forest predicts by a majority vote of its trees.

To pin down a forest, you choose \(A\) and the distribution of \(\theta\). Here is one common choice. Generate \(\theta\) in two parts. First, draw a new training sequence \(S'\) of size \(m'\) by sampling from \(S\) uniformly **with replacement**: some examples appear several times, others not at all. Second, draw a sequence of feature subsets \(I_1, I_2, \dots\), each a uniformly random set of \(k\) features. Then grow a tree with ID3 on \(S'\), except that at the \(t\)-th split the algorithm may only choose among the features in \(I_t\), picking the one with the highest gain there. Intuitively, if \(k\) is small, this restriction may prevent overfitting: no single tree can chase every coincidence, because at most splits the coincidence’s feature isn’t on offer. And because each tree sees a different resample, their mistakes differ, and the vote can cancel some of them.

How different are the resamples? With \(m' = m\), each draw misses a given example with probability \(1 - 1/m\), and it is missed by all \(m\) independent draws with probability \((1 - 1/m)^m\), the miss chance multiplied by itself \(m\) times. So a resample contains a fraction

\[ 1 - (1 - 1/m)^m \approx 1 - 1/e \]

of the distinct examples, where \(e \approx 2.718\) is the base of the natural logarithm and \(\approx\) reads “is about”, the approximation getting better as \(m\) grows. In plain words: **a resample of size \(m\) leaves out about 37% of the examples and repeats others.** With \(m = 1{,}000\), \((0.999)^{1000} \approx 0.368\), so about 632 distinct examples make it in. The snippet draws the resamples and computes the formula side by side.

```python
import random

rng, m, trials = random.Random(0), 1000, 2000
seen = 0
for _ in range(trials):
    boot = [rng.randrange(m) for _ in range(m)]
    seen += len(set(boot)) / m
print(round(seen / trials, 3), round(1 - (1 - 1 / m) ** m, 3))
```

It prints `0.633 0.632`: each tree trains on about 63% of the distinct examples, some of them twice.

```
RandomForest(S, T, m', k)
  for t = 1..T:
    S' ← m' examples drawn from S uniformly with replacement
    tree_t ← ID3 on S', where each split picks the best feature
             among k features drawn uniformly at random from
             those still available on that path
  return h(x) = majority vote of tree_1(x), ..., tree_T(x)
```

**Runtime.** The resample costs \(O(m')\). Growing one tree is ID3 with only \(k\) gains per node instead of up to \(d\): \(O(m'k)\) per level and at most \(d\) levels, so \(O(m'kd)\). The forest costs \(O(Tm'kd)\) time to train, \(O(Td)\) to predict one label, and space for \(T\) trees, up to \(O(Tm'd)\) nodes.

On one draw of the lab’s mail with 500 labels, the unpruned ID3 tree has 153 nodes and true risk 0.224. A forest of 25 trees with \(k = 10\) has an estimated true risk of 0.104, almost the noise floor of 0.1, without any pruning. It also has 3,301 nodes. Hand that to the auditor. Each of the 25 trees is readable on its own, after a fashion, but the forest’s answer is a vote among 25 paths, and “junk because 14 of 25 trees said so” is not a reason anyone can check. A random forest trades readability for accuracy. In this world the trade buys nothing, since the pruned single tree also reaches 0.100 with 7 nodes; the truth here is a small tree, and pruning finds it. When the truth is not a small tree, forests are often the more accurate choice, and then you have to decide which one the compliance office wants more.

**Build it.** Add a `forest(S, d, C, trees, k, rng)` and a `vote(F, x)` to your build-it file. On `draw(500, random.Random(0))`, with 25 trees, \(k = 10\) and information gain, estimate \(\mathbb P[h(x) \ne f(x)]\) on 4,000 fresh messages for the forest and for the single unpruned tree, and check:

```python
assert forest_disagreement < 0.05 and tree_disagreement > 0.1
```

(The reference implementation gets about 0.005 and 0.154.)

## What people get wrong

**“ID3 finds the best tree.”** It finds a tree, greedily, one node at a time, and never revisits a choice. Finding the tree that minimizes the description-length bound is computationally hard, and greedy choices can be badly wrong: the second problem below has four examples on which ID3, limited to depth 2, is stuck with a training error of at least 1/4, while another depth-2 tree makes no mistakes at all.

**“A tree with zero empirical risk is a good tree.”** On noisy mail, zero empirical risk means the tree memorized the noise. On the lab’s mail, the full ID3 tree with zero empirical risk has true risk about 0.22, and its certificate is near 0.9. The bound charges for every node, and that charge is the honest one.

**“Pruning with the bound uses the data twice, so the certificate is void.”** The bound holds for every tree at once, on one event of probability \(1 - \delta\), so it covers whichever tree pruning picks. What would void it is changing the description language after seeing the data.

**“Training error is the natural gain, the others are tricks.”** Training error is blind to every split that does not flip a majority, which on random splits is about half of them. Information gain and the Gini index are concave upper bounds of the training error that reward every step toward purer leaves.

## Recap and the next bug

1. A **decision tree** labels a message by following yes/no questions from the root to a leaf. Every root-to-leaf path is a readable rule, which is the whole reason auditors like them.
2. Trees of unrestricted size are too rich: threshold trees with \(k\) leaves shatter \(k\) points, and over \(\{0,1\}^d\) trees can express every function, so their VC dimension is \(2^d\).
3. A prefix-free code with \((n + 1)\lceil\log_2(d+3)\rceil\) bits for a tree of \(n\) nodes turns lesson 7’s MDL bound into a certificate for every tree at once: \(L_{\mathcal D}(h) \le L_S(h) + \sqrt{((n+1)b + \log(2/\delta))/(2m)}\).
4. Minimizing that bound is computationally hard, so **ID3** grows a tree greedily, splitting on the feature with the largest gain, in \(O(md^2)\) time. Information gain and the Gini index are strictly concave and see every useful split; training error misses any split that leaves both majorities unchanged.
5. **Pruning** walks the tree bottom-up and replaces nodes by leaves or subtrees whenever that lowers the bound, trading a little empirical risk for many fewer nodes.
6. Real-valued features reduce to yes/no features with one threshold per gap between sorted training values; a sort-and-sweep finds the best threshold in \(O(dm\log m)\) per node.
7. A **random forest** votes over trees grown on resamples with random feature subsets. It can be more accurate than one tree, and it is far harder to read.

The compliance office is happy for exactly one month. Then the mail changes again. New kinds of message keep arriving, newsletters from tools nobody has heard of, invoices in formats the auditors never imagined, and they match no existing rule: they wander down the tree to whatever leaf is nearest, for reasons that have nothing to do with them. Writing more rules is a losing race. So a colleague suggests the simplest idea of all, one that needs no rules and no tree: to label a new message, find the labelled messages most similar to it, and copy their label. Does that work? And when does it fail? That’s the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Your training sequence contains just two examples, the same message 1011 labelled once junk and once keep, and \(d = 4\). Trace ID3 with information gain (ties toward the lowest feature, majority ties toward junk). What tree does it return, how many nodes does it have, and what is its empirical risk? What is the smallest possible empirical risk of any tree on this sequence, and what does pruning with the description-length bound return?
2. Lesson 7 warned that choosing a description language after seeing the labels breaks the MDL guarantee. Pruning chooses a tree after seeing the labels. Explain precisely why the pruned tree’s certificate is still valid. Then say which step of the description-length theorem’s proof would fail if, after seeing \(S\), you shortened the blocks by dropping the features that no tree you grew ever used.
3. With \(d = 29\), \(\delta = 0.05\) and \(m = 1{,}500\), compare two pruned trees: one with 11 nodes and empirical risk 0.08, one with 7 nodes and empirical risk 0.10. Which has the smaller bound, and by how much? Assuming both empirical risks stay where they are, from which \(m\) on does the 11-node tree win?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (every function is a tree).** Show that every function \(h: \{0,1\}^d \to \{0,1\}\) can be computed by a decision tree of depth at most \(d\) (so at most \(d + 1\) levels of nodes) whose internal nodes ask questions of the form “is \(x_i = 0\)?”. Conclude that the class of all decision trees over \(\{0,1\}^d\) has VC dimension exactly \(2^d\).
- **Problem 2 (the greedy grower can be fooled).** Take \(\mathcal X = \{0,1\}^3\) and the training sequence \(((1,1,1), 1)\), \(((1,0,0), 1)\), \(((1,1,0), 0)\), \(((0,0,1), 0)\). You want a tree of depth 2: every message may be asked at most two questions of the form “is \(x_i = 0\)?”. Run ID3 with information gain up to depth 2 (choose the root and its children by the algorithm, then stop and label each leaf by the majority of its examples), where ties between features may be broken in any way. Show that the resulting tree has training error at least \(1/4\), whatever the tie-breaking. Then find a tree of depth 2 with training error 0.
