# Clustering

## Mail with no labels

For twenty-one lessons, someone has been clicking. Every filter you built learned from messages a person had labelled junk or keep, folder or no folder, and every guarantee you proved was a statement about how far those labels could be trusted.

On Monday the labelling team quit. All of them, at once, for better pay somewhere with fewer prize draws. The mail has not stopped. Thousands of messages arrive every day, and not one of them carries a label. The company still wants the same thing it always wanted: sort the mail into its kinds, newsletters with newsletters, invoices with invoices, scams with scams, personal mail with personal mail, so that each kind can be routed to the right desk.

So you try. You take the scores you already compute for every message, and you ask three different engineers to group the same twelve messages into two kinds. They come back with three different answers. Each one is defended with a perfectly reasonable argument, each one looks sensible on a plot, and when the manager asks “which one is right?”, nobody can say. There is no label to check against. There is no empirical risk to compute. The thing every previous lesson leaned on, a number that says how wrong you are, has simply gone.

This lesson is about grouping without labels, which is called **clustering**. We will build the main families of clustering algorithms, prove what each one does and does not guarantee, and then prove something uncomfortable: that no clustering method can have all three of a short list of properties that every reasonable person would want. By the end you will be able to answer the manager honestly, which is not the same as answering with one grouping.

## Three answers to one question

Let’s look at the twelve messages. Each is described by two of the scores the mail room computes: the **link share** (how much of the message is links) and the **payment score** (how much it talks about paying). Both are numbers between 0 and 1, and to keep the arithmetic clean we will write them in tenths, so a link share of 3 means 0.3. The twelve messages sit in two rows:

| message | link share | payment score |
|---|---|---|
| 1 to 6 | 0, 1, 2, 3, 4, 5 | 0 |
| 7 to 12 | 0, 1, 2, 3, 4, 5 | 2 |

So message 1 is at \((0, 0)\), message 6 at \((5, 0)\), message 7 at \((0, 2)\) and message 12 at \((5, 2)\). Neighbours in a row are 1 apart. The two rows are 2 apart. Before reading on, if you had to split these twelve into two groups, which split would you choose?

[[diagram:two-answers]]

The first engineer says: two messages that are very close must end up in the same group. Every message is 1 away from its neighbour in the row, and nothing in the other row is closer than 2. So each row is one group, and the answer is “bottom row, top row”: messages 1 to 6 and messages 7 to 12.

The second engineer says: two messages that are far apart must not end up in the same group. Message 1 at \((0, 0)\) and message 6 at \((5, 0)\) are 5 apart, far more than anything across the gap between the rows. So she cuts the picture down the middle: the left half (link share 0 to 2, both rows) and the right half (link share 3 to 5, both rows).

The third engineer used yet another rule and got eight messages in one group and four in the other. We will meet his rule in a moment.

The first two engineers are both following half of the same intuitive sentence: **similar things should share a group, and dissimilar things should be in different groups.** The trouble is that the two halves pull against each other. Here is the precise reason. “Is similar to” is not a transitive relation: message 1 is similar to message 2, which is similar to message 3, and so on to message 6, but message 1 is not similar to message 6. “Shares a group with”, on the other hand, is transitive: if 1 shares a group with 2 and 2 with 3, then 1 shares a group with 3. So a long chain of messages, each close to the next, forces a choice. Keep every close pair together and you must put the whole chain in one group, ends included, and then two dissimilar messages share a group. Separate the ends and you must cut the chain somewhere, separating two similar neighbours.

The second difficulty is deeper, and it is the one that hurts in the mail room. In every earlier lesson there was a ground truth: a label your users gave, against which a hypothesis could be scored. Clustering is **unsupervised** learning, as lesson 1 called it: there are no labels to predict, only messages to organise. Even if you knew the distribution of incoming mail perfectly, it would not tell you what the “correct” grouping is. The same mail could be grouped by sender, by topic or by tone, and each grouping could be exactly what some desk needs, just as recordings of speech can be grouped by the speaker’s accent or by what is being said, and paintings by subject or by style.

## What a clustering is

Before we can build anything, we have to say what goes in and what comes out.

**Input: a distance function.** A clustering method receives a finite set \(\mathcal X\) of messages, and a way to say how far apart two of them are: messages 1 and 7 are 2 apart, messages 1 and 6 are 5 apart, and every message is 0 away from itself. Formally, a **distance function** is a function

\[ d : \mathcal X \times \mathcal X \to \mathbb R_+ \]

with \(d(x, y) = d(y, x)\) and \(d(x, x) = 0\) for every \(x, y \in \mathcal X\).

Let’s read it.

- \(\mathcal X\) is the set of messages to be grouped; in the opening, the twelve messages.
- \(\mathcal X \times \mathcal X\) is the set of all ordered pairs \((x, y)\) of messages. The arrow says that \(d\) takes one such pair and returns a number.
- \(\mathbb R_+\) is the set of non-negative real numbers, so a distance is never negative.
- \(d(x, y) = d(y, x)\) is **symmetry**: the distance from \(x\) to \(y\) is the distance from \(y\) to \(x\).
- \(d(x, x) = 0\): every message is at distance 0 from itself.

Often, though not always, \(d\) also satisfies the **triangle inequality** \(d(x, z) \le d(x, y) + d(y, z)\): going via \(y\) is never shorter than going straight. In plain words: **a distance function gives every pair of messages a non-negative number, the same in both directions, and zero for a message and itself.** On the twelve messages we use the ordinary straight-line distance, \(d(x, y) = \|x - y\|\), the square root of the summed squared differences of the two scores (lesson 9a used the same norm \(\|\cdot\|\) for vectors). Messages 1 and 7 are at \(\|(0, 0) - (0, 2)\| = 2\); messages 1 and 6 are at 5.

Instead of a distance, a method may receive a **similarity function** \(s : \mathcal X \times \mathcal X \to [0, 1]\), symmetric, with \(s(x, x) = 1\): large means alike. Some methods also take a number \(k\), the number of groups wanted.

**Output: a partition.** What comes out is a split of the messages into groups, where every message lands in exactly one group, the way the first engineer put messages 1 to 6 in one group and 7 to 12 in the other. Formally, a **clustering** of \(\mathcal X\) is a partition \(C = (C_1, \dots, C_k)\): a list of sets with

\[ \bigcup_{i=1}^{k} C_i = \mathcal X \quad \text{and} \quad C_i \cap C_j = \emptyset \text{ for all } i \ne j. \]

- Each \(C_i\) is a **cluster**: a set of messages, one group.
- \(\bigcup_{i=1}^k C_i\) is the union of the \(k\) clusters, everything that is in at least one of them. Setting it equal to \(\mathcal X\) says that every message lands in some cluster.
- \(C_i \cap C_j\) is the set of messages in both \(C_i\) and \(C_j\), and \(\emptyset\) is the empty set. “For all \(i \ne j\)” says that two different clusters never share a message.

In plain words: **a clustering puts every message into exactly one group.** The first engineer’s answer is \(C_1 = \{1, \dots, 6\}\), \(C_2 = \{7, \dots, 12\}\): the union is all twelve, and the two rows share nothing.

```python
X = set(range(1, 13))
rows = [set(range(1, 7)), set(range(7, 13))]
halves = [{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}]
bad = [{1, 2, 3}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}]

def is_partition(C, X):
    union = set().union(*C)
    disjoint = all(not (A & B) for i, A in enumerate(C)
                   for B in C[i + 1:])
    return union == X and disjoint

print([is_partition(C, X) for C in (rows, halves, bad)])
```

It prints `[True, True, False]`: both engineers’ answers are partitions, and a grouping that puts message 3 in two places is not.

Two other kinds of output exist. A **soft clustering** gives each message a list of probabilities \((p_1(x), \dots, p_k(x))\), where \(p_i(x)\) is the probability that \(x\) belongs to cluster \(C_i\). And a **dendrogram** (Greek *dendron*, tree, and *gramma*, drawing) is a whole hierarchy of clusterings at once: a tree whose leaves are the single messages, whose root is all of \(\mathcal X\), and whose every inner node is the union of its children. The next section builds one.

## Linkage: merging the closest clusters

The simplest family of clustering methods works bottom up. Start with every message in its own cluster. Then, round after round, merge the two clusters that are closest to each other. Each round there is one cluster fewer, and if you never stop, everything ends up in one cluster. Methods of this kind are called **agglomerative**, and they leave two decisions open: how to measure the distance between two clusters, and when to stop.

**Three ways to measure the distance between clusters.** We have \(d\) between messages; we need a distance \(D(A, B)\) between two sets of messages \(A\) and \(B\). Look at every pair made of one message from each cluster. You can then measure the two clusters by their closest such pair, by the average of all such pairs, or by their farthest such pair. With one message at 0 and a cluster of two at 6 and 8, the cross pairs are 6 and 8 apart, so the three answers are 6, 7 and 8. In symbols, the three common choices are:

\[ D_{\text{single}}(A, B) = \min\{ d(x, y) : x \in A,\ y \in B \}, \]

\[ D_{\text{average}}(A, B) = \frac{1}{|A|\,|B|} \sum_{x \in A,\ y \in B} d(x, y), \]

\[ D_{\text{max}}(A, B) = \max\{ d(x, y) : x \in A,\ y \in B \}. \]

Read them from the inside out.

- \(d(x, y)\) with \(x \in A,\ y \in B\) is the distance of one cross pair: \(x\) taken from cluster \(A\), \(y\) from cluster \(B\). “\(\in\)” reads “is a member of”.
- The braces \(\{ d(x, y) : x \in A,\ y \in B \}\) collect all of those cross distances; read the colon as “such that”. With \(|A| = 2\) and \(|B| = 3\) there are six of them.
- \(\min\) takes the smallest of them and \(\max\) the largest.
- \(|A|\) is the number of messages in \(A\), so \(|A|\,|B|\) is the number of cross pairs. The sum \(\sum_{x \in A,\ y \in B}\) adds up every cross distance, and dividing by the number of pairs gives their average.

In plain words: **single linkage measures two clusters by their closest pair, average linkage by the average pair, and max linkage by their farthest pair.** Take \(A = \{0\}\) and \(B = \{6, 8\}\), messages at those positions on a single score. The cross distances are 6 and 8, so single linkage says 6, average linkage 7, and max linkage 8.

```python
from statistics import mean

def cross(A, B):
    return [abs(x - y) for x in A for y in B]

A, B = [0], [6, 8]
print(min(cross(A, B)), mean(cross(A, B)), max(cross(A, B)))
```

It prints `6 7 8`.

**A worked dendrogram.** Take five messages \(a, b, c, d, e\) on a single score, at positions 0, 6, 8, 11 and 13. Single linkage first merges the closest pair of single messages. \(b\) and \(c\) are 2 apart, and so are \(d\) and \(e\); nothing else is that close. So round one merges \(\{b, c\}\) and round two merges \(\{d, e\}\), both at distance 2. Now there are three clusters, \(\{a\}\), \(\{b, c\}\) and \(\{d, e\}\). The closest pair between \(\{b, c\}\) and \(\{d, e\}\) is \(c\) and \(d\), at 3, while \(\{a\}\) is 6 away from \(b\). So round three merges \(\{b, c, d, e\}\) at 3, and round four adds \(a\) at 6.

| round | merged | single | average | max |
|---|---|---|---|---|
| 1 | \(\{b\}, \{c\}\) | 2 | 2 | 2 |
| 2 | \(\{d\}, \{e\}\) | 2 | 2 | 2 |
| 3 | \(\{b, c\}, \{d, e\}\) | 3 | 5 | 7 |
| 4 | \(\{a\}, \{b, c, d, e\}\) | 6 | 9.5 | 13 |

The other two linkages happen to merge in the same order here, but at different heights: average linkage joins \(\{b, c\}\) and \(\{d, e\}\) at the mean of 3, 5, 5 and 7, which is 5. Drawn as a tree, with each merge as a node at the height where it happened, this is the dendrogram.

[[diagram:dendrogram]]

**When to stop.** A dendrogram is not yet a partition. To get one, you cut it with a **stopping rule**. Three are common.

- **Fixed number of clusters:** fix \(k\) and stop as soon as there are \(k\) clusters. With \(k = 2\), the five messages become \(\{a\}\) and \(\{b, c, d, e\}\).
- **Distance upper bound:** fix \(r > 0\) and stop as soon as every between-cluster distance is larger than \(r\). With single linkage and \(r = 2.5\), merging stops after round two: \(\{a\}, \{b, c\}, \{d, e\}\).
- **Scaled distance upper bound:** the same, with \(r = \alpha \max\{ d(x, y) : x, y \in \mathcal X \}\) for a fixed \(\alpha < 1\), a fraction of the largest distance in the data. With \(\alpha = 0.4\), the largest distance is 13 (from \(a\) to \(e\)), so \(r = 5.2\), and merging stops before \(a\) joins at 6: \(\{a\}, \{b, c, d, e\}\).

Back to the opening. The first engineer was running single linkage with \(k = 2\). In the twelve messages every row neighbour is at 1 and the rows are at 2, so single linkage merges each row completely before it ever crosses the gap, and stops at the two rows. The third engineer ran max linkage, which on this input ends with eight messages against four.

**Single linkage is a famous algorithm in disguise.** Picture the complete graph whose \(m\) vertices are the \(m\) messages, with an edge between every pair, each edge \(\{x, y\}\) weighted by \(d(x, y)\). A **spanning tree** is a set of \(m - 1\) edges that connects all \(m\) vertices without a cycle, and a **minimum spanning tree** is one whose total weight is as small as possible. **Kruskal’s algorithm** builds one: go through the edges from lightest to heaviest, and keep an edge whenever its two ends are not yet connected by the edges kept so far.

**Proposition (single linkage is Kruskal’s algorithm).** Run single linkage to the end, and for each merge of \(A\) and \(B\) record a closest pair \(\{x, y\}\), \(x \in A\), \(y \in B\). The recorded edges are exactly the edges Kruskal’s algorithm keeps (with ties broken the same way), so they form a minimum spanning tree.

*Proof.* We show, by induction on the round, that the clusters of single linkage are always the connected pieces of the edges recorded so far, and that each round records the edge Kruskal would keep next.

At the start, no edge is recorded and every cluster is one message: the pieces are the single vertices.

Suppose that before some round, the clusters are exactly the connected pieces of the recorded edges. An edge \(\{x, y\}\) joins two different pieces exactly when \(x\) and \(y\) are in different clusters, and those are exactly the edges Kruskal may still keep: an edge inside one piece would close a cycle, an edge across two pieces would not. Single linkage picks the two clusters with the smallest \(D_{\text{single}}(A, B)\), which is the smallest cross distance, so it records the lightest edge whose ends lie in different pieces. That is Kruskal’s next edge. Merging \(A\) and \(B\) joins their two pieces with that edge, so after the round the clusters are again the connected pieces.

After \(m - 1\) rounds, the recorded edges are Kruskal’s, and Kruskal’s algorithm returns a minimum spanning tree, a standard fact from any algorithms course (its proof is the exchange argument: any lighter tree would have to cross some cut more cheaply than the edge Kruskal chose there). \(\blacksquare\)

So a fixed-\(k\) cut of the single-linkage dendrogram is what you get by deleting the \(k - 1\) heaviest edges of a minimum spanning tree. That is also why single linkage follows chains so faithfully: a chain of close neighbours is a path of light edges, and a minimum spanning tree never refuses one.

## Linkage as code

Here are the linkage methods in pseudocode, with both stopping rules as options.

```
Linkage(X, d, D, stop):
  clusters ← [{x} for x in X]
  dendrogram ← empty list
  while |clusters| > 1:
    if stop is "k clusters" and |clusters| = k: break
    (A, B) ← the pair of clusters with smallest D(A, B)
    if stop is "bound r" and D(A, B) > r: break
    append (D(A, B), A, B) to dendrogram
    replace A and B in clusters by A ∪ B
  return clusters, dendrogram
```

Done naively, each round compares every pair of clusters, and computing one \(D(A, B)\) from scratch costs \(|A||B|\) distance lookups. The standard trick is to keep an \(m \times m\) table of current cluster distances. Finding the smallest entry costs \(O(m^2)\). After merging \(A\) and \(B\), the new cluster’s distance to any other cluster \(E\) follows from the old ones in \(O(1)\): \(\min\) of the two for single linkage, \(\max\) for max linkage, and the size-weighted average \((|A| D(A, E) + |B| D(B, E))/(|A| + |B|)\) for average linkage. So each of the at most \(m - 1\) rounds costs \(O(m^2)\), and linkage clustering runs in \(O(m^3)\) time and \(O(m^2)\) space, after \(O(m^2 n)\) time to compute the distances of \(m\) messages with \(n\) scores each. Single linkage alone can do better: Prim’s algorithm (another classic minimum-spanning-tree method, which grows a single tree outward from one vertex, always adding the lightest edge that leaves it) finds a minimum spanning tree of a complete graph in \(O(m^2)\) time, and by the proposition that is the whole dendrogram.

**Build it yourself.** Using only the standard library:

```python
import math

P = [(x, 0) for x in range(6)] + [(x, 2) for x in range(6)]
line = [(0,), (6,), (8,), (11,), (13,)]

def linkage(points, how, k=None, r=None): ...

if __name__ == "__main__":
    rows = [list(range(6)), list(range(6, 12))]
    assert linkage(P, "single", k=2)[0] == rows
    groups, merges = linkage(line, "single")
    assert [round(v, 1) for v, A, B in merges] == [2, 2, 3, 6]
    groups, merges = linkage(line, "average")
    assert [v for v, A, B in merges] == [2, 2, 5, 9.5]
    three = [[0], [1, 2], [3, 4]]
    assert linkage(line, "single", r=2.5)[0] == three
```

1. Implement `linkage(points, how, k, r)` returning the sorted list of clusters (each a sorted list of indices) and the list of merges `(D, A, B)`, with `how` one of `"single"`, `"average"`, `"max"`. Break ties toward the pair of clusters that appears first in your list.
2. Make the asserts pass, then print the max-linkage merge heights of `line` and check them against the table: 2, 2, 7, 13.
3. Check the proposition: for 200 random sets of 9 points in the unit square, the merge heights of single linkage sum to the weight of the minimum spanning tree that Prim’s algorithm finds.

## Clustering as optimization

The second family turns clustering into an optimization problem. Fix a **cost function** \(G\) that takes the input \((\mathcal X, d)\) and a proposed clustering \(C = (C_1, \dots, C_k)\) and returns a non-negative number, and look for the clustering with the smallest cost. Now “which grouping is right?” has an answer, at the price of having chosen \(G\).

**The k-means cost.** Suppose the messages live inside a larger space \(\mathcal X'\) where centres can sit (for messages described by \(n\) scores, \(\mathcal X' = \mathbb R^n\), every possible list of \(n\) numbers). Each cluster is represented by one point of \(\mathcal X'\), its **centroid** or centre, and the cost charges every message the squared distance to the centre of its cluster: a message 1.5 away from its centre pays \(1.5^2 = 2.25\), a message 2.5 away pays 6.25. Add the charges up, after placing every centre where the total is smallest:

\[ G_{k\text{-means}}((\mathcal X, d), (C_1, \dots, C_k)) = \min_{\mu_1, \dots, \mu_k \in \mathcal X'} \sum_{i=1}^{k} \sum_{x \in C_i} d(x, \mu_i)^2. \]

Read it from the inside out.

- \(\mu_i\) (the Greek letter mu) is a candidate centre for cluster \(C_i\), a point of \(\mathcal X'\).
- \(d(x, \mu_i)^2\) is the squared distance from message \(x\) to that centre.
- \(\sum_{x \in C_i}\) adds this up over the messages of cluster \(C_i\): the cluster’s total squared spread around \(\mu_i\).
- \(\sum_{i=1}^k\) adds the clusters together.
- \(\min_{\mu_1, \dots, \mu_k \in \mathcal X'}\) chooses the \(k\) centres that make the total as small as possible. The cost of a clustering is the total you get with the best centres.

In plain words: **the k-means cost of a grouping is the total squared distance from each message to the centre of its group, with each centre placed as well as possible.** For the bottom row of the opening, messages at \((0,0)\) to \((5,0)\), the best centre turns out to be \((2.5, 0)\), and the squared distances are \(6.25, 2.25, 0.25, 0.25, 2.25, 6.25\), which add to 17.5. The top row is the same, so the first engineer’s grouping costs 35. The second engineer’s halves cost 10 each, 20 in total. By this cost, she wins. Here is the same computation in code, placing each centre at the cluster’s mean (the next section proves that the mean is the best place):

```python
P = [(x, 0) for x in range(6)] + [(x, 2) for x in range(6)]

def kmeans_cost(clusters):
    total = 0
    for C in clusters:
        mu = [sum(v) / len(C) for v in zip(*C)]
        total += sum((a - mu[0])**2 + (b - mu[1])**2
                     for a, b in C)
    return total

rows = [P[:6], P[6:]]
halves = [[p for p in P if p[0] < 3], [p for p in P if p[0] >= 3]]
print(kmeans_cost(rows), kmeans_cost(halves))
```

It prints `35.0 20.0`: the rows cost 35, the halves 20.

The k-means cost has a natural reading in communication. If every message must be sent as one of \(k\) prototypes, the index of the nearest centre, then the k-means cost measures the total distortion of that compression.

**Relatives.** Change the centre set or the penalty and you get other costs:

- **k-medoids**: the same formula, but the centres must be messages: \(\min_{\mu_1, \dots, \mu_k \in \mathcal X}\).
- **k-median**: centres among the messages, and the penalty is the distance itself, \(d(x, \mu_i)\), not its square. Placing \(k\) fire stations among the houses of a town to minimize the total distance from a house to its nearest station is a k-median problem.
- More generally, any increasing function \(f : \mathbb R_+ \to \mathbb R_+\) gives a **center-based** cost \(G_f = \min_{\mu_1, \dots, \mu_k \in \mathcal X'} \sum_{i=1}^k \sum_{x \in C_i} f(d(x, \mu_i))\), with \(\mathcal X'\) either \(\mathcal X\) itself or a larger space. Here \(f\) turns a message’s distance to its centre into its charge: \(f(u) = u^2\) with centres anywhere gives the k-means cost (a message 1.5 away pays 2.25), and \(f(u) = u\) with centres among the messages gives k-median (it pays 1.5). A center-based cost is decided by \(k\) centres, and the clustering assigns each message to its closest centre.

Not every cost is center-based. The **sum of in-cluster distances**,

\[ G_{\text{SOD}}((\mathcal X, d), (C_1, \dots, C_k)) = \sum_{i=1}^k \sum_{x, y \in C_i} d(x, y), \]

adds, for each cluster, the distance of every ordered pair of its messages: \(x, y \in C_i\) runs over all pairs inside cluster \(C_i\), each pair in both orders (and \(x = y\) adds 0). In plain words: **the sum of in-cluster distances adds up how far apart the members of each group are, with no centre anywhere.** With \(C_1 = \{0, 6, 8\}\) on a single score, the ordered pairs give \(2 \times (6 + 8 + 2) = 32\).

```python
from itertools import permutations

C1 = [0, 6, 8]
print(sum(abs(x - y) for x, y in permutations(C1, 2)))
```

It prints `32`.

Here is the bad news, stated without proof: finding a clustering that minimizes the k-means cost is NP-hard, and even NP-hard to approximate within some constant factor, and the same holds for most of these costs. (Lesson 8 explained what NP-hard means for learning.) So in practice “k-means” usually names not the cost, but a fast algorithm that tries to lower it. Before we meet it, one small fact about centres.

## The best centre is the mean

Here is the idea in numbers first. The bottom row’s total squared spread around its mean \((2.5, 0)\) is 17.5. Slide the centre 1.5 to the left, to \((1, 0)\), and the spread becomes 31, which is exactly \(17.5 + 6 \times 1.5^2\): six messages, each charged the squared distance the centre moved. That extra charge is always (number of messages) times (distance moved, squared), and it is zero only if the centre does not move.

**Proposition (the mean minimizes the squared spread).** Let \(C\) be a finite set of points in \(\mathbb R^n\) and let

\[ \bar x = \frac{1}{|C|} \sum_{x \in C} x \]

be its mean. Then for every \(\mu \in \mathbb R^n\),

\[ \sum_{x \in C} \|x - \mu\|^2 = \sum_{x \in C} \|x - \bar x\|^2 + |C|\, \|\bar x - \mu\|^2. \]

Let’s name every piece first.

- \(\bar x\) (“x bar”) adds the points of \(C\) coordinate by coordinate and divides by how many there are: the ordinary average point.
- \(\|x - \mu\|^2\) is the squared straight-line distance from \(x\) to a candidate centre \(\mu\), the sum of the squared coordinate differences.
- The left side is the total squared spread of \(C\) around \(\mu\). The first term on the right is the spread around the mean, and does not depend on \(\mu\). The last term is how far \(\mu\) is from the mean, squared, counted once for every point.

In plain words: **moving the centre away from the mean adds exactly (number of points) times (squared distance moved) to the spread.** Since that added term is never negative and is zero only at \(\mu = \bar x\), the mean is the one and only best centre. For the bottom row, \(\bar x = (2.5, 0)\) and the spread around it is 17.5; move the centre to \(\mu = (1, 0)\), 1.5 away, and the spread becomes \(17.5 + 6 \times 2.25 = 31\).

*Proof.* Write \(x - \mu = (x - \bar x) + (\bar x - \mu)\) and expand the square of a sum, \(\|u + v\|^2 = \|u\|^2 + 2\langle u, v \rangle + \|v\|^2\), where \(\langle u, v\rangle\) is the inner product (lesson 9a):

\[ \|x - \mu\|^2 = \|x - \bar x\|^2 + 2 \langle x - \bar x,\ \bar x - \mu \rangle + \|\bar x - \mu\|^2. \]

Now add this over all \(x \in C\). The first terms add to \(\sum_{x \in C} \|x - \bar x\|^2\). The last term does not depend on \(x\), so it is added \(|C|\) times. In the middle, \(\bar x - \mu\) does not depend on \(x\) either, so the inner product can be pulled out of the sum, since it is linear in its first argument: the middle terms add to \(2 \langle \sum_{x \in C}(x - \bar x),\ \bar x - \mu \rangle\). And \(\sum_{x \in C} (x - \bar x) = \sum_{x \in C} x - |C| \bar x = 0\), by the definition of \(\bar x\). So the middle vanishes, which is the claim. \(\blacksquare\)

```python
row = [(x, 0) for x in range(6)]

def spread(C, mu):
    return sum((a - mu[0])**2 + (b - mu[1])**2 for a, b in C)

x_bar = (sum(a for a, b in row) / 6, sum(b for a, b in row) / 6)
mu = (1, 0)
gap = (x_bar[0] - mu[0])**2 + (x_bar[1] - mu[1])**2
print(x_bar, spread(row, x_bar), spread(row, mu))
print(spread(row, x_bar) + len(row) * gap)
```

It prints `(2.5, 0.0) 17.5 31` and `31.0`: the spread around \((1, 0)\) is the spread around the mean plus six times \(1.5^2\).

So the centres inside the k-means cost can be dropped: for a fixed clustering, the best centres are the means \(\mu(C_i) = \frac{1}{|C_i|}\sum_{x \in C_i} x\), and

\[ G(C_1, \dots, C_k) = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu(C_i)\|^2. \]

In words: **the k-means cost of a grouping is the total squared distance from each message to the mean of its group.** That is how we computed 35 and 20 above.

## The k-means algorithm

Here is the algorithm everyone means when they say “k-means”. It is also called **Lloyd’s algorithm**, after Stuart Lloyd, who proposed it in 1957. It alternates two easy steps: given centres, the best groups are “each message to its nearest centre”; given groups, the best centres are the means.

```
KMeans(X ⊂ R^n, k):
  choose initial centres μ_1, ..., μ_k (say, k random messages)
  repeat until the clusters stop changing:
    assign:  for each i in [k],
             C_i ← {x ∈ X : i = argmin_j ||x − μ_j||}
    update:  for each i in [k],
             μ_i ← (1 / |C_i|) · Σ_{x ∈ C_i} x
  return C_1, ..., C_k
```

Two pieces of notation. \([k]\) is the list \(\{1, \dots, k\}\), and \(\operatorname{argmin}_j \|x - \mu_j\|\) is the index \(j\) whose centre is nearest to \(x\). If two centres are equally near, a tie rule decides, and the choice matters, as you are about to see; unless we say otherwise, ties go to the smaller index \(j\). If a cluster comes out empty, its mean is undefined, and we keep its old centre.

**A worked run.** Take the twelve messages with \(k = 2\) and start the centres at messages 1 and 2, \(\mu_1 = (0, 0)\) and \(\mu_2 = (1, 0)\). Two numbers are tracked each iteration: the **assignment cost**, the total squared distance of every message to the centre it was just assigned to (old centres), and \(G\), the cost after the centres moved to the means.

| iteration | cluster 1 after assign | assignment cost | new centres | \(G\) |
|---|---|---|---|---|
| 1 | link share 0 | 84 | \((0, 1)\), \((3, 1)\) | 32 |
| 2 | link share 0, 1 | 26 | \((0.5, 1)\), \((3.5, 1)\) | 23 |
| 3 | link share 0, 1, 2 | 23 | \((1, 1)\), \((4, 1)\) | 20 |
| 4 | no change | 20 | | 20 |

Notice the zigzag: 84, then 32, 26, 23, 23, 20, 20. Every number is at most the one before it. The run ends at the halves, cost 20, the best possible. Look closely at iteration 3, though. The two messages with link share 2 are exactly 1.5 from both centres, \((0.5, 1)\) and \((3.5, 1)\): a tie. The smaller-index rule sends them to cluster 1, and the run reaches 20. A rule that keeps a tied message where it already is would stop right there, at cost 23, which is also a place where neither step changes anything. Now start instead at messages 1 and 7, \((0, 0)\) and \((0, 2)\). The first assignment puts each row with the centre in its own row, the means move to \((2.5, 0)\) and \((2.5, 2)\), and nothing ever changes again. The run stops at the rows, cost 35, and has no idea that 20 exists.

```python
P = [(x, 0) for x in range(6)] + [(x, 2) for x in range(6)]

def sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

def kmeans(P, mu):
    lab, log = None, []
    while True:
        new = [min((0, 1), key=lambda j: sq(p, mu[j])) for p in P]
        log.append(sum(sq(p, mu[j]) for p, j in zip(P, new)))
        if new == lab:
            return log
        lab = new
        C = [[p for p, j in zip(P, lab) if j == i]
             for i in (0, 1)]
        mu = [tuple(sum(v) / len(c) for v in zip(*c)) for c in C]
        log.append(sum(sq(p, mu[j]) for p, j in zip(P, lab)))

print(kmeans(P, [(0, 0), (1, 0)]), kmeans(P, [(0, 0), (0, 2)]))
```

It prints `[84, 32.0, 26.0, 23.0, 23.0, 20.0, 20.0] [110, 35.0, 35.0]`: the whole zigzag of the table, and a second run that settles at 35. (Python’s `min` returns the first of equally near centres, which is the smaller-index rule, so iteration 3 breaks its tie as the table does.)

[[diagram:zigzag]]

Why must the numbers only ever go down? That is the main fact about the algorithm.

**Proposition (k-means never increases its cost).** Let \(C^{(t-1)} = (C_1^{(t-1)}, \dots, C_k^{(t-1)})\) be the clusters after iteration \(t - 1\), with centres \(\mu_i^{(t-1)} = \mu(C_i^{(t-1)})\), and let \(C^{(t)}\) be the clusters after the assignment step of iteration \(t\). Then

\[ G(C^{(t)}) \le \sum_{i=1}^k \sum_{x \in C_i^{(t)}} \|x - \mu_i^{(t-1)}\|^2 \le G(C^{(t-1)}). \]

The superscript \((t)\) is just a time stamp: \(C_i^{(t)}\) is cluster \(i\) after iteration \(t\). The middle quantity is the assignment cost of the table: new clusters, old centres. So the statement says: **the cost after iteration \(t\) is at most the assignment cost, which is at most the cost after iteration \(t - 1\).** In the table, iteration 2 gives \(23 \le 26 \le 32\).

*Proof.* Two inequalities, one per step of the algorithm.

The left inequality is about the update step. By definition, \(G(C^{(t)})\) is the minimum, over all choices of \(k\) centres, of \(\sum_i \sum_{x \in C_i^{(t)}} \|x - \mu_i\|^2\). The old centres \(\mu_1^{(t-1)}, \dots, \mu_k^{(t-1)}\) are one particular choice, and a minimum is at most the value at any particular choice. (By the previous proposition the minimum is attained at the means, which is exactly where the update step moves the centres.)

The right inequality is about the assignment step. Fix the old centres. The assignment cost is a sum with one term per message, \(\|x - \mu_{j(x)}^{(t-1)}\|^2\), where \(j(x)\) is the cluster \(x\) is assigned to. The assignment step gives each message its nearest old centre, so each term is at most the term for any other choice of cluster, and in particular at most \(\|x - \mu_{i}^{(t-1)}\|^2\) for the cluster \(i\) that held \(x\) after iteration \(t - 1\). Adding these per-message inequalities over all messages,

\[ \sum_{i=1}^k \sum_{x \in C_i^{(t)}} \|x - \mu_i^{(t-1)}\|^2 \le \sum_{i=1}^k \sum_{x \in C_i^{(t-1)}} \|x - \mu_i^{(t-1)}\|^2. \]

The right-hand side is the old clusters measured from their own means, which is \(G(C^{(t-1)})\) by the mean proposition. An empty old cluster contributes nothing to either side, so keeping its old centre changes nothing. \(\blacksquare\)

Two consequences, and one honest limit.

**It stops.** Suppose ties are broken in favour of the cluster a message is already in. Then a message changes cluster only when some other centre is strictly nearer, so whenever the clusters change, the right inequality is strict and \(G\) strictly drops. A strictly dropping cost can never revisit a clustering, and there are finitely many clusterings of \(m\) messages, so the algorithm stops after finitely many iterations. With the smaller-index rule, the same argument goes through if you also stop as soon as \(G\) fails to drop strictly, which is what the build-it challenge below does. Either way, the proof gives no useful bound on how many iterations: the number of clusterings is astronomically large.

**Where it stops is a fixed point.** At the end, every message is already with its nearest centre and every centre is already the mean of its cluster: neither step can lower the cost on its own. That is all the guarantee says.

**The limit.** A fixed point need not be good. The second run above stopped at cost 35 when 20 was available, and the ratio between where k-means stops and the best possible cost can be made as large as you like: put four messages at the corners of a rectangle \(w\) wide and \(h\) tall, with \(w\) much larger than \(h\), start the centres at the two left corners, and k-means stops at the top and bottom pairs, cost \(w^2\), while the left and right pairs cost \(h^2\). The first problem at the end of this lesson shows that a fixed point need not even be a local minimum of the cost as a function of the centres. The usual remedy is cheap: run k-means several times from different random starts and keep the lowest cost. On the twelve messages, starting from two random messages, 1,000 runs ended at cost 20 in 692 runs, at 23 in 208 and at the rows, 35, in 100.

**Cost.** One iteration computes the distance from each of the \(m\) messages to each of the \(k\) centres, \(n\) scores each, so the assignment step costs \(O(mkn)\) time, and the update step \(O(mn)\). Space is \(O(mn)\) for the messages plus \(O(kn)\) for the centres. With \(T\) iterations and \(R\) restarts, \(O(RTmkn)\) in total.

**Build it yourself.**

```python
import random

P = [(x, 0) for x in range(6)] + [(x, 2) for x in range(6)]

def kmeans(points, centres): ...
def best_of(points, k, restarts, rng): ...

if __name__ == "__main__":
    lab, mu, log = kmeans(P, [(0, 0), (1, 0)])
    assert log == [32, 23, 20]
    assert all(b <= a for a, b in zip(log, log[1:]))
    lab, mu, log = kmeans(P, [(0, 0), (0, 2)])
    assert log == [35] and lab == [0] * 6 + [1] * 6
    cost, lab = best_of(P, 2, 20, random.Random(0))
    assert cost == 20
```

1. Implement `kmeans(points, centres)` for any \(k\) and any number of scores, returning the final labels, the final centres and the list of \(G\) values after each update step. Break ties toward the smaller index, keep the old centre of an empty cluster, and stop when an assignment changes nothing or \(G\) fails to drop strictly.
2. Implement `best_of`, which starts from `k` distinct random messages `restarts` times and returns the lowest final cost with its labels.
3. Add an assertion inside `kmeans` that checks both inequalities of the proposition at every iteration, then run it on 1,000 random point sets. It should never fire.

## Try it: three methods, one batch of mail

Before going further, play with the methods you have. The lab holds a fresh day of mail: about a hundred messages on two scores, link share and payment score. Behind the scenes they come from four kinds, newsletters, invoices, scams and personal mail, plus a handful of odd invoices that look half like scams. You never see the kinds unless you ask to peek. Choose a method and a number of groups, and compare what each method draws.

[[lab]]

## Spectral clustering: cutting a graph

The costs so far concentrate on one half of the intuitive sentence: messages in a cluster should be close to their centre. The next family concentrates on the other half: messages in different clusters should be dissimilar.

**The similarity graph.** Put the \(m\) messages \(x_1, \dots, x_m\) at the vertices of a graph and give the edge between \(x_r\) and \(x_s\) a weight \(W_{r,s} = s(x_r, x_s)\), their similarity. The weights form a symmetric \(m \times m\) matrix \(W\), written \(W \in \mathbb R^{m, m}\). A common choice turns distances into similarities: identical messages get weight 1, row neighbours a bit more than a third, and messages across the gap almost nothing.

\[ W_{r,s} = \exp\!\left( -\frac{d(x_r, x_s)^2}{\sigma^2} \right). \]

- \(d(x_r, x_s)^2\) is the squared distance between messages \(r\) and \(s\).
- \(\sigma\) (sigma) is a length scale you choose: it says what counts as “near”.
- \(\exp(-u) = e^{-u}\) is 1 at \(u = 0\) and falls toward 0 as \(u\) grows.

In plain words: **the edge weight is 1 for identical messages and fades toward 0 as they move apart, at a pace set by \(\sigma\).** With \(\sigma = 1\), two row neighbours (distance 1) get \(e^{-1} \approx 0.368\), and messages 1 and 7 across the gap (distance 2) get \(e^{-4} \approx 0.018\).

```python
import math

def weight(p, q, sigma):
    d2 = (p[0] - q[0])**2 + (p[1] - q[1])**2
    return math.exp(-d2 / sigma**2)

print(round(weight((0, 0), (1, 0), 1), 3),
      round(weight((0, 0), (0, 2), 1), 3))
```

It prints `0.368 0.018`.

**Cuts.** Given a clustering \(C_1, \dots, C_k\) of the vertices, the **cut** adds up the weight of every edge that leaves a cluster:

\[ \operatorname{cut}(C_1, \dots, C_k) = \sum_{i=1}^k \sum_{r \in C_i,\ s \notin C_i} W_{r,s}. \]

- The inner sum runs over pairs \((r, s)\) with \(r\) inside cluster \(C_i\) and \(s\) outside it (\(\notin\) reads “is not a member of”): the edges leaving \(C_i\).
- The outer sum adds this over the clusters. An edge between \(C_1\) and \(C_2\) leaves both, so it is counted twice, once from each side. That changes no comparison.

In plain words: **the cut is the total similarity you throw away by separating the clusters.** Minimizing it is the **mincut** problem. For \(k = 2\) it can be solved efficiently, but it has a fatal habit. Here is a seven-vertex graph to see it: two triangles \(\{0, 1, 2\}\) and \(\{3, 4, 5\}\) with edges of weight 1, a bridge from 2 to 3 of weight 0.2, and one straggler, vertex 6, hanging off vertex 5 by an edge of weight 0.15.

[[diagram:graph-cut]]

Cutting the bridge costs \(2 \times 0.2 = 0.4\). Cutting off the straggler alone costs \(2 \times 0.15 = 0.3\). Mincut picks the straggler: one message against six. That is not a grouping into kinds; it is an outlier report. Mincut will often do exactly this, slicing off a single weakly attached vertex.

The fix is to charge small clusters for being small: divide the weight leaving each cluster by the cluster’s size, so a cluster of one pays its leaving weight in full and a cluster of six pays a sixth.

\[ \operatorname{RatioCut}(C_1, \dots, C_k) = \sum_{i=1}^k \frac{1}{|C_i|} \sum_{r \in C_i,\ s \notin C_i} W_{r,s}. \]

The only change is the factor \(\frac{1}{|C_i|}\): the weight leaving each cluster is divided by that cluster’s size. In plain words: **RatioCut is the weight leaving each cluster, per member, added over the clusters.** A cluster of one pays its full leaving weight; a cluster of six pays a sixth. Isolating the straggler now costs \(0.15/1 + 0.15/6 = 0.175\), while cutting the bridge costs \(0.2/3 + 0.2/4 \approx 0.117\), and the bridge wins.

```python
E = {(0, 1): 1, (0, 2): 1, (1, 2): 1, (3, 4): 1, (3, 5): 1,
     (4, 5): 1, (2, 3): 0.2, (5, 6): 0.15}
W = [[0] * 7 for _ in range(7)]
for (r, s), w in E.items():
    W[r][s] = W[s][r] = w

def leaving(C):
    return sum(W[r][s] for r in C for s in range(7) if s not in C)

def costs(*clusters):
    cut = sum(leaving(C) for C in clusters)
    ratio = sum(leaving(C) / len(C) for C in clusters)
    return round(cut, 3), round(ratio, 3)

print(costs({6}, {0, 1, 2, 3, 4, 5}))
print(costs({0, 1, 2}, {3, 4, 5, 6}))
```

It prints `(0.3, 0.175)` and `(0.4, 0.117)`: mincut prefers the straggler, RatioCut the bridge. A search over all 63 two-cluster splits confirms that these are the best split under each cost.

Balancing has a price: minimizing RatioCut is computationally hard. Spectral clustering is a way around that, and it starts from a matrix.

**The graph Laplacian.** The **degree** of a vertex is the total weight of its edges: vertex 2 touches 0, 1 and 3 with weights 1, 1 and 0.2, so its degree is 2.2. In symbols, the degree of vertex \(r\) is \(D_{r,r} = \sum_{s=1}^m W_{r,s}\). The **degree matrix** \(D\) is the diagonal matrix with these degrees on the diagonal and zeros elsewhere, and the **unnormalized graph Laplacian** is

\[ L = D - W. \]

So \(L_{r,r}\) is the degree of \(r\) minus \(W_{r,r}\) (which is 0 here, since no vertex has an edge to itself), and \(L_{r,s} = -W_{r,s}\) off the diagonal. In plain words: **the Laplacian has each vertex’s total edge weight on the diagonal and minus the edge weights everywhere else.** In the seven-vertex graph, vertex 6 has one edge, so \(L_{6,6} = 0.15\) and \(L_{6,5} = -0.15\); vertex 2 has edges to 0, 1 and 3, so \(L_{2,2} = 2.2\). Each row of \(L\) adds to 0, because the diagonal entry is exactly the sum of the weights subtracted in the rest of the row.

(The letter \(D\) has now been used for the between-cluster distance of linkage and for the degree matrix; which is meant is always clear from whether a matrix is in sight. And \(L\) here is a matrix, not a risk \(L_{\mathcal D}\).)

Why this matrix? Because of one identity. Give every vertex a number, say 1 on the first triangle and 0 everywhere else. Each edge whose two ends got different numbers is a disagreement, and we charge it the edge weight times the squared difference. Here only the bridge disagrees, so the total charge is \(0.2 \times 1^2 = 0.2\). The Laplacian computes exactly this total.

**Proposition (the Laplacian measures disagreement along edges).** For every vector \(v \in \mathbb R^m\),

\[ v^\top L v = \frac12 \sum_{r, s} W_{r,s} (v_r - v_s)^2. \]

- \(v\) gives each vertex a number; \(v_r\) is the number at vertex \(r\).
- \(v^\top L v = \sum_{r,s} v_r L_{r,s} v_s\) is the quadratic form of \(L\): one number built from \(v\) and \(L\). (\(v^\top\) is \(v\) written as a row.)
- On the right, \((v_r - v_s)^2\) is how much the numbers at the two ends of edge \(\{r, s\}\) disagree, and \(W_{r,s}\) weights that by the edge’s similarity. The sum over all ordered pairs \((r, s)\) sees every edge twice, which the \(\frac12\) undoes.

In plain words: **\(v^\top L v\) adds, over all edges, the edge weight times the squared difference of the numbers at its ends.** It is small exactly when strongly connected vertices get nearly equal numbers. Give the seven vertices \(v = (1, 1, 1, 0, 0, 0, 0)\): the only edge whose ends differ is the bridge, so \(v^\top L v = 0.2 \times 1^2 = 0.2\).

*Proof.* Expand the right-hand side with \((v_r - v_s)^2 = v_r^2 - 2 v_r v_s + v_s^2\):

\[ \frac12 \sum_{r,s} W_{r,s}(v_r - v_s)^2 = \frac12 \Big( \sum_{r,s} W_{r,s} v_r^2 - 2 \sum_{r,s} W_{r,s} v_r v_s + \sum_{r,s} W_{r,s} v_s^2 \Big). \]

In the first sum, do the inner sum over \(s\) first: \(\sum_s W_{r,s} = D_{r,r}\), so the first sum is \(\sum_r D_{r,r} v_r^2\). The third sum is the same with the roles of \(r\) and \(s\) swapped, and \(W\) is symmetric, so it is \(\sum_s D_{s,s} v_s^2\), the same number. So the bracket is \(2 \sum_r D_{r,r} v_r^2 - 2 \sum_{r,s} W_{r,s} v_r v_s\), and half of it is

\[ \sum_r D_{r,r} v_r^2 - \sum_{r,s} v_r W_{r,s} v_s = v^\top D v - v^\top W v = v^\top (D - W) v = v^\top L v, \]

using that \(v^\top D v = \sum_r D_{r,r} v_r^2\) for a diagonal \(D\). \(\blacksquare\)

Two facts fall out at once. Every \(v^\top L v\) is a sum of non-negative terms, so \(v^\top L v \ge 0\). And the all-ones vector \(\mathbf 1\) gives \(v_r - v_s = 0\) on every edge, so \(\mathbf 1^\top L \mathbf 1 = 0\); in fact \(L \mathbf 1 = 0\), since each row of \(L\) adds to 0. Recall from lesson 9b that a nonzero vector \(u\) which a matrix only stretches, \(L u = \lambda u\) for some number \(\lambda\), is an **eigenvector**, and \(\lambda\) its **eigenvalue**. So \(\mathbf 1\) is an eigenvector of \(L\) with eigenvalue 0. And no eigenvalue of \(L\) is negative: if \(L u = \lambda u\) then \(0 \le u^\top L u = \lambda \|u\|^2\).

The same charge, summed over clusters, gives RatioCut. Turn each cluster into a column of numbers: \(1/\sqrt{|C_j|}\) on its members and 0 elsewhere, so the first triangle becomes \(1/\sqrt 3\) on vertices 0, 1, 2. The charge of that column is the bridge weight 0.2 divided by 3, which is exactly the first triangle’s share of RatioCut.

**Proposition (RatioCut as a trace).** Let \(C_1, \dots, C_k\) be a clustering of the \(m\) vertices and let \(H \in \mathbb R^{m, k}\) be the matrix with entries

\[ H_{i,j} = \frac{1}{\sqrt{|C_j|}} \, \mathbb 1[i \in C_j]. \]

Then the columns of \(H\) are orthonormal, and \(\operatorname{RatioCut}(C_1, \dots, C_k) = \operatorname{trace}(H^\top L H)\).

- \(H\) has one row per vertex and one column per cluster.
- \(\mathbb 1[i \in C_j]\) is the indicator from lesson 1: 1 if vertex \(i\) is in cluster \(j\), 0 otherwise. So column \(j\) is zero outside \(C_j\) and equal to \(1/\sqrt{|C_j|}\) on it.
- **Orthonormal columns** means each column has length 1 and any two different columns have inner product 0.
- \(H^\top L H\) is a \(k \times k\) matrix, and the **trace** of a square matrix is the sum of its diagonal entries.

In plain words: **write each cluster as a scaled 0/1 column, and the sum of the Laplacian’s quadratic forms on those columns is exactly RatioCut.** For the bridge split, column 1 is \((1, 1, 1, 0, 0, 0, 0)/\sqrt 3\) and column 2 is \((0, 0, 0, 1, 1, 1, 1)/2\), and the trace comes out at 0.117, the RatioCut we computed.

*Proof.* Call the columns \(h_1, \dots, h_k\). Column \(h_j\) has \(|C_j|\) entries equal to \(1/\sqrt{|C_j|}\) and zeros elsewhere, so \(\|h_j\|^2 = |C_j| \cdot \frac{1}{|C_j|} = 1\). Two different columns are nonzero on disjoint sets of rows, because the clusters do not overlap, so every product \(h_{i,j} h_{i,j'}\) is 0 and their inner product is 0. The columns are orthonormal.

The \((j, j)\) diagonal entry of \(H^\top L H\) is row \(j\) of \(H^\top\), times \(L\), times column \(j\) of \(H\), that is \(h_j^\top L h_j\). So \(\operatorname{trace}(H^\top L H) = \sum_{j=1}^k h_j^\top L h_j\).

Now apply the identity to \(v = h_j\). For an edge with both ends in \(C_j\), or both ends outside, \(h_{r,j} - h_{s,j} = 0\). For an ordered pair with \(r \in C_j\) and \(s \notin C_j\), or the other way round, the difference is \(\pm 1/\sqrt{|C_j|}\) and its square is \(1/|C_j|\). Each edge leaving \(C_j\) appears as two ordered pairs, which cancels the \(\frac12\):

\[ h_j^\top L h_j = \frac12 \cdot 2 \sum_{r \in C_j,\ s \notin C_j} W_{r,s} \cdot \frac{1}{|C_j|} = \frac{1}{|C_j|} \sum_{r \in C_j,\ s \notin C_j} W_{r,s}. \]

Adding over \(j\) gives RatioCut. \(\blacksquare\)

```python
W = [[0] * 7 for _ in range(7)]
for r, s, w in [(0, 1, 1), (0, 2, 1), (1, 2, 1), (3, 4, 1),
                (3, 5, 1), (4, 5, 1), (2, 3, .2), (5, 6, .15)]:
    W[r][s] = W[s][r] = w
L = [[(sum(W[r]) if r == s else 0) - W[r][s] for s in range(7)]
     for r in range(7)]

def quad(v):
    return sum(v[r] * L[r][s] * v[s]
               for r in range(7) for s in range(7))

h1 = [3**-.5] * 3 + [0] * 4
h2 = [0] * 3 + [.5] * 4
print(round(quad([1, 1, 1, 0, 0, 0, 0]), 3),
      round(quad(h1) + quad(h2), 3))
```

It prints `0.2 0.117`: the quadratic form of the triangle indicator is the bridge weight, and the trace of \(H^\top L H\) is the RatioCut of the bridge split.

**The relaxation.** So minimizing RatioCut means minimizing \(\operatorname{trace}(H^\top L H)\) over matrices \(H\) with orthonormal columns whose entries are either 0 or \(1/\sqrt{|C_j|}\). That last condition makes it a search over integer-like patterns, which we cannot do efficiently. So we drop it, and minimize \(\operatorname{trace}(H^\top L H)\) over all \(m \times k\) matrices with orthonormal columns. That relaxed problem has a clean answer, which lesson 23a proves (it is the same fact that makes principal component analysis work): take as columns the eigenvectors of \(L\) belonging to its \(k\) smallest eigenvalues.

The relaxed solution is no longer a clustering, so one more step turns it back into one. Each vertex \(r\) gets the \(r\)-th row of that matrix, a point in \(\mathbb R^k\), and k-means clusters those points.

```
UnnormalizedSpectralClustering(W ∈ R^{m,m}, k):
  L ← D − W, with D the diagonal matrix of row sums of W
  U ← the m × k matrix whose columns are eigenvectors of L
      for its k smallest eigenvalues
  v_1, ..., v_m ← the rows of U
  run k-means on v_1, ..., v_m
  return the k-means clusters, with vertex r in the cluster of v_r
```

Why should the rows of \(U\) be easy for k-means? Look at the ideal case. If \(U\) were the \(H\) of the trace proposition, the row of a vertex in \(C_j\) would be zero except for \(1/\sqrt{|C_j|}\) in position \(j\): every vertex of a cluster would sit on the same point, and different clusters on different axes. Here is one case where that is exactly what happens.

**Proposition (separate pieces are free).** Suppose the clustering \(C_1, \dots, C_k\) has no edges between clusters: \(W_{r,s} = 0\) whenever \(r\) and \(s\) are in different clusters. Then its matrix \(H\) achieves \(\operatorname{trace}(H^\top L H) = 0\), the smallest value any matrix with orthonormal columns can achieve, and each column \(h_j\) is an eigenvector of \(L\) with eigenvalue 0.

*Proof.* RatioCut adds only weights of edges between clusters, all 0 here, so by the trace proposition \(\operatorname{trace}(H^\top L H) = 0\). For any \(H'\) with orthonormal columns, the trace is a sum of quadratic forms \(h'^\top L h' \ge 0\), so it is at least 0. And for each column, \(h_j^\top L h_j = 0\) means every term \(W_{r,s}(h_{r,j} - h_{s,j})^2\) of the identity is 0. Then \((L h_j)_r = D_{r,r} h_{r,j} - \sum_s W_{r,s} h_{s,j} = \sum_s W_{r,s}(h_{r,j} - h_{s,j})\), since \(D_{r,r} = \sum_s W_{r,s}\), and this has every term equal to 0 too: wherever \(W_{r,s} > 0\), \(h_{r,j} = h_{s,j}\). So \(L h_j = 0 = 0 \cdot h_j\). \(\blacksquare\)

When the pieces are joined by weak edges, as the triangles are by the bridge, the second-smallest eigenvector is no longer exactly a scaled indicator, but it stays close to one, and its signs give the split. You can watch that happen with a few lines of power iteration: repeatedly multiply by \(cI - L\), where \(I\) is the identity matrix (1 on the diagonal, 0 elsewhere) and \(c = 5\) (for a large enough \(c\), this turns the smallest eigenvalues of \(L\) into the largest), subtract the part along \(\mathbf 1\) (its eigenvalue 0 becomes the largest, 5, so it would otherwise win), and normalize.

```python
import math

W = [[0] * 7 for _ in range(7)]
for r, s, w in [(0, 1, 1), (0, 2, 1), (1, 2, 1), (3, 4, 1),
                (3, 5, 1), (4, 5, 1), (2, 3, .2), (5, 6, .15)]:
    W[r][s] = W[s][r] = w
deg = [sum(row) for row in W]
v = [math.sin(r + 1) for r in range(7)]
for _ in range(3000):
    Lv = [deg[r] * v[r] - sum(W[r][s] * v[s] for s in range(7))
          for r in range(7)]
    v = [5 * v[r] - Lv[r] for r in range(7)]
    v = [x - sum(v) / 7 for x in v]
    v = [x / math.sqrt(sum(y * y for y in v)) for x in v]
print([round(x, 2) for x in v])
```

It prints `[0.42, 0.42, 0.38, -0.18, -0.22, -0.24, -0.6]`. (An eigenvector with every sign flipped is still an eigenvector, so another start could print the same numbers with opposite signs, which describes the same split.) The first triangle is positive and everything else negative: spectral clustering with \(k = 2\) cuts the bridge, which is the RatioCut optimum, and leaves the straggler where it hangs.

**Cost.** Building \(W\) takes \(O(m^2 n)\) time and \(O(m^2)\) space, and so does \(L\). A full symmetric eigendecomposition takes \(O(m^3)\) time with standard methods; k-means on \(m\) rows of length \(k\) then costs \(O(mk^2)\) per iteration. The \(m^2\) memory is what limits spectral clustering in practice.

## The information bottleneck

The last method on our list comes from information theory, and it fits mail unusually well. Treat each message as a **bag of words**: the set of dictionary words that appear in it. Here are four messages from the unlabelled pile:

| message | words |
|---|---|
| \(n_1\) | news, unsubscribe |
| \(n_2\) | news, unsubscribe, link |
| \(i_1\) | invoice, pay |
| \(i_2\) | invoice, pay, due |

Turn this into probability. Let \(x\) be a message picked uniformly at random, so each has probability \(1/4\), and let \(y\) be a word picked uniformly from that message. A clustering is a third random variable \(C\), the cluster of \(x\), which may even be random: the method chooses the probabilities \(p(C \mid x)\) of each cluster given each message.

To score it we need a measure of how much one random quantity tells you about another. If the cluster is the kind of the message, two kinds, equally likely, then learning which message it was settles the kind completely: one yes/no question’s worth of information, 1 bit. If the cluster were a coin flip unrelated to the message, learning the message would tell you nothing, 0 bits. That measure is the **mutual information** of two random variables \(a\) and \(b\):

\[ I(a; b) = \sum_{\alpha} \sum_{\beta} p(\alpha, \beta) \log_2 \frac{p(\alpha, \beta)}{p(\alpha)\, p(\beta)}. \]

- \(\alpha\) runs over the values \(a\) can take and \(\beta\) over the values of \(b\).
- \(p(\alpha, \beta)\) is the probability that \(a = \alpha\) and \(b = \beta\) together; \(p(\alpha)\) and \(p(\beta)\) are the probabilities of each alone.
- The ratio \(p(\alpha, \beta)/(p(\alpha) p(\beta))\) is 1 when the two values occur together exactly as often as independence would predict, and its logarithm is then 0. Terms with \(p(\alpha, \beta) = 0\) count as 0.
- With base-2 logarithms, as for the entropy of lesson 18, the unit is the **bit**.

In plain words: **mutual information is how many bits knowing one variable tells you, on average, about the other.** It is 0 when they are independent. If \(C\) is the kind of a uniformly random message, with two equally likely kinds, then knowing \(x\) tells you \(C\) exactly, and \(I(x; C) = 1\) bit.

Now the idea. A good grouping should forget as much as possible about which message it was (that is what grouping means), while remembering as much as possible about the words. One cluster for everything forgets all, and tells you nothing about the words; every message in its own cluster remembers all, and compresses nothing. The **information bottleneck** objective balances the two, choosing the soft assignment by

\[ \min_{p(C \mid x)} \; I(x; C) - \beta\, I(C; y), \]

where \(\beta > 0\) is a trade-off you choose (a fresh use of the Greek letter beta, unrelated to the values \(\beta\) in the sum above). Read the two terms as two wishes that pull apart.

- \(I(x; C)\) small: the clusters should forget as much as possible about which message it was. One cluster for everything forgets all, \(I(x; C) = 0\). This is compression.
- \(I(C; y)\) large: the cluster should still tell you as much as possible about the words. This is relevance.

In plain words: **squeeze the messages into as few distinguishable groups as you can, while keeping as much as you can of what the groups say about the words.** Here are four deterministic clusterings of the four messages, scored with \(\beta = 2\):

```python
from math import log2
from collections import Counter

docs = [["news", "unsubscribe"], ["news", "unsubscribe", "link"],
        ["invoice", "pay"], ["invoice", "pay", "due"]]

def mi(joint):
    pa, pb = Counter(), Counter()
    for (a, b), p in joint.items():
        pa[a] += p
        pb[b] += p
    return sum(p * log2(p / (pa[a] * pb[b]))
               for (a, b), p in joint.items())

for C in ("0000", "0011", "0101", "0123"):
    xc, cy = Counter(), Counter()
    for x, words in enumerate(docs):
        xc[x, C[x]] += 1 / 4
        for w in words:
            cy[C[x], w] += 1 / 4 / len(words)
    i_xc, i_cy = abs(round(mi(xc), 3)), abs(round(mi(cy), 3))
    print(C, i_xc, i_cy, round(i_xc - 2 * i_cy, 3))
```

Each line gives the clustering, \(I(x; C)\), \(I(C; y)\) and the objective. It prints

```
0000 0.0 0.0 0.0
0011 1.0 1.0 -1.0
0101 1.0 0.191 0.618
0123 2.0 1.191 -0.382
```

[[diagram:bottleneck]]

Read the four lines in order: one cluster for everything (`0000`), the two kinds (`0011`), a mixed pairing (`0101`), and every message alone (`0123`). The kinds cost 1 bit of compression and keep 1 bit about the words, the best value at \(\beta = 2\). The mixed pairing pays the same bit and keeps only 0.191 bits. Keeping every message apart keeps the most, 1.191 bits, but pays 2 bits for it. Turn \(\beta\) up and relevance wins: from \(\beta = 1/0.191 \approx 5.2\) on, the singletons beat the kinds. Turn it below 1 and one cluster for everything wins. So, once again, the answer depends on a number someone has to choose.

The bottleneck generalizes the statistician’s **minimal sufficient statistic**: a summary of the data that keeps everything relevant to the question and nothing else. Solving the optimization is hard in general, and the methods proposed for it resemble the expectation-maximization procedure of lesson 24b. We will not build one.

## What is clustering, really?

Step back. We have linkage methods with three linkages and three stopping rules, costs and an algorithm for them, graph cuts, and an information-theoretic objective, and on the twelve messages they disagree. Is there anything that every clustering method *should* do, whatever the method? One way to answer is to write down properties and ask which methods have them.

Throughout this section, a **clustering function** \(F\) takes any finite set \(\mathcal X\) with a dissimilarity \(d\) on it and returns a partition \(F(\mathcal X, d)\) of \(\mathcal X\). A **dissimilarity** here is a distance function as defined earlier, with one extra requirement: \(d(x, y) > 0\) whenever \(x \ne y\). Three properties sound like plain good sense.

**Scale invariance.** Measuring the twelve messages in tenths or in hundredths multiplies every distance by 10; the groups should not care. Formally, for every \(\mathcal X\), every \(d\) and every \(\alpha > 0\),

\[ F(\mathcal X, d) = F(\mathcal X, \alpha d), \qquad \text{where } (\alpha d)(x, y) = \alpha\, d(x, y). \]

\(\alpha d\) is the same dissimilarity with every value multiplied by \(\alpha\). In plain words: **changing the units of distance must not change the clustering.** Scores in tenths or in hundredths, the twelve messages are the same twelve messages: multiplying every distance by 10 must give the same groups.

**Richness.** For every finite \(\mathcal X\) and every partition \(C\) of \(\mathcal X\) into non-empty sets, there is some dissimilarity \(d\) with \(F(\mathcal X, d) = C\). In plain words: **every grouping is possible, if the distances call for it.** The method has no built-in preference for, say, two groups.

**Consistency.** Let \(d\) and \(d'\) be dissimilarities on \(\mathcal X\) such that, for all \(x, y \in \mathcal X\):

- if \(x, y\) are in the same cluster of \(F(\mathcal X, d)\), then \(d'(x, y) \le d(x, y)\);
- if \(x, y\) are in different clusters of \(F(\mathcal X, d)\), then \(d'(x, y) \ge d(x, y)\).

Then \(F(\mathcal X, d') = F(\mathcal X, d)\). The two conditions describe a **consistent change** of \(d\) with respect to the clustering \(F(\mathcal X, d)\): members of a cluster move closer together or stay put, members of different clusters move further apart or stay put. In plain words: **if the clusters you found get tighter and further apart, you must find the same clusters.** If the two rows of the opening were found as clusters, pushing the rows from 2 apart to 4 apart must not change them.

Each property is the kind of thing a manager would write into a requirements document without blinking. Scale invariance says units don’t matter. Richness says the data, not the method, decides the grouping. Consistency is the only one that speaks to the intuitive sentence itself: it says that more evidence for a grouping should never overturn it.

Before the theorem, the idea of the proof in one breath. Richness gives one dissimilarity that makes everything a singleton and another that makes something else. Blow the second one up by a scale factor until every pair is at least as far apart as in the first. Scale invariance says the blown-up version still gives the something-else; consistency says that pushing singletons apart keeps them singletons. Both cannot be true.

**Theorem (no clustering function has all three).** There is no clustering function \(F\) that satisfies scale invariance, richness and consistency.

*Proof.* Suppose, for contradiction, that \(F\) has all three. Pick any finite \(\mathcal X\) with at least two elements, so that it has at least two different partitions: all singletons, and everything in one cluster.

Step 1, two dissimilarities from richness. Richness applied to the all-singletons partition gives a \(d_1\) with \(F(\mathcal X, d_1) = \{\{x\} : x \in \mathcal X\}\): every message alone. Richness applied to any other partition gives a \(d_2\) with \(F(\mathcal X, d_2) \ne F(\mathcal X, d_1)\).

Step 2, scale \(d_2\) up past \(d_1\). Let

\[ \alpha = \max_{x \ne y} \frac{d_1(x, y)}{d_2(x, y)}, \]

the largest ratio over pairs of distinct elements. It exists, because \(\mathcal X\) is finite, and it is a positive number, because dissimilarities between distinct elements are positive (this is where that requirement is used). Set \(d_3 = \alpha d_2\). For every pair of distinct \(x, y\), \(\alpha \ge d_1(x, y)/d_2(x, y)\), so \(d_3(x, y) = \alpha d_2(x, y) \ge d_1(x, y)\). For \(x = y\) both are 0.

Step 3, scale invariance. \(d_3\) is \(d_2\) in other units, so \(F(\mathcal X, d_3) = F(\mathcal X, d_2)\).

Step 4, consistency. Compare \(d_3\) with \(d_1\). In \(F(\mathcal X, d_1)\) every message is alone, so there is no pair of distinct elements in the same cluster, and the first condition of a consistent change asks nothing (pairs \(x = x\) have \(d_3 = d_1 = 0\)). Every pair of distinct elements is in different clusters, and for those the condition asks \(d_3(x, y) \ge d_1(x, y)\), which step 2 guaranteed. So \(d_3\) is a consistent change of \(d_1\), and consistency gives \(F(\mathcal X, d_3) = F(\mathcal X, d_1)\).

Step 5, the contradiction. Steps 3 and 4 give \(F(\mathcal X, d_2) = F(\mathcal X, d_3) = F(\mathcal X, d_1)\), but step 1 chose \(d_2\) with \(F(\mathcal X, d_2) \ne F(\mathcal X, d_1)\). So no such \(F\) exists. \(\blacksquare\)

[[diagram:impossibility]]

It is worth running the proof against a real clustering function to see exactly where it bites. Take single linkage with a scaled distance upper bound, \(\alpha = 0.5\): merge as long as the closest clusters are at most half the largest distance apart. Three messages \(a, b, c\). Under \(d_1\) all three pairs are at distance 2, the bound is 1, and nothing merges: singletons. Under \(d_2\), with \(d_2(a, b) = 1\), \(d_2(b, c) = 1.6\) and \(d_2(a, c) = 2.6\), the bound is 1.3, so \(a\) and \(b\) merge and \(c\) stays alone.

```python
from itertools import combinations as pairs

def F(d, frac=0.5):
    r, C = frac * max(d.values()), [{x} for x in "abc"]
    while True:
        D = {(i, j): min(d["".join(sorted(x + y))]
                         for x in C[i] for y in C[j])
             for i, j in pairs(range(len(C)), 2)}
        if not D or min(D.values()) > r:
            return sorted("".join(sorted(c)) for c in C)
        i, j = min(D, key=D.get)
        C[i] |= C.pop(j)

d1 = {"ab": 2.0, "bc": 2.0, "ac": 2.0}
d2 = {"ab": 1.0, "bc": 1.6, "ac": 2.6}
alpha = max(d1[p] / d2[p] for p in d1)
d3 = {p: alpha * d2[p] for p in d2}
print(F(d1), F(d2), F(d3), alpha)
print(all(d3[p] >= d1[p] for p in d1))
```

It prints `['a', 'b', 'c'] ['ab', 'c'] ['ab', 'c'] 2.0` and `True`: step 2 found \(\alpha = 2\), so \(d_3\) puts the pairs at 2, 3.2 and 5.2, each at least 2. This function is scale invariant (the bound scales with the data) and its output under \(d_3\) matches \(d_2\). But \(d_3\) is a consistent change of \(d_1\), every pair moved apart or stayed, and the output changed from singletons to \(\{a, b\}, \{c\}\). The property that gave way is consistency: raising the largest distance raised the bound, and that merged a pair that had not moved at all.

**No single bad property.** The theorem does not say that one of the three is to blame. For every *pair* of the three properties, there are natural clustering functions with both; you can even get each pair from single linkage alone, just by changing the stopping rule among the three you met. Working out which rule gives which pair is the third problem at the end of this lesson. What is known about the costs: a method that minimizes a center-based cost, such as the k-means cost, fails consistency on some inputs (the counterexamples use general dissimilarities, not only straight-line distances), while minimizing the sum of in-cluster distances satisfies it. We state these two without proof.

**Ways out.** The theorem is about three exact properties, and loosening any of them reopens the door.

- If the method takes the number of clusters \(k\) as an input, richness is too much to ask: it can never produce a different number of groups. The natural replacement is **\(k\)-richness**: every partition into exactly \(k\) non-empty sets is possible. Scale invariance, \(k\)-richness and consistency can all hold at once.
- Or relax consistency. Call two clusterings \(C\) and \(C'\) **compatible** if every cluster of one is either inside a cluster of the other, contains it, or is disjoint from it, the way any two cuts of one dendrogram are. **Refinement consistency** asks only that, after a consistent change, the new clustering be compatible with the old one. Many common clustering functions satisfy refinement consistency, scale invariance and richness together.

So what does the theorem mean for the mail room? That there is no ideal clustering function, one that every task would want. Every clustering function gives up something that sounds desirable, and the choice of method has to be made from what the specific task needs. This is the clustering face of the No-Free-Lunch theorem of lesson 5: there, no learner wins on every learnable task, so a learner must bring prior knowledge in the form of a hypothesis class. Here, no clustering function has every reasonable property, so whoever asks for a clustering must bring prior knowledge in the form of the properties, or the cost, they actually want.

## Fix the bug

So you go back to the manager, and instead of three plots you bring a question: what, exactly, do the desks need? After some argument, the answer comes back in three lines.

1. Exactly four groups, one per desk.
2. The grouping must not depend on units: the scores are about to be re-scaled from tenths to hundredths, and nobody wants the desks reshuffled by a change of units.
3. Each group should be as tight as possible around its centre, and you must hand in an audit log showing that your method never made the grouping worse on that measure, and stopped only where no single step could improve it.

Each line is now a property you can prove, not a picture you hope looks right. In the lab you choose a method and its settings, and it runs on five fresh batches of mail in a row. For each batch, the lab checks every line: it counts the groups, it reruns your method on the same messages with every score multiplied by 10 and compares, and it checks your log step by step. A method passes only if it meets all three lines on all five batches. Average linkage often draws four groups that look exactly like the four kinds. Does it meet line 3?

[[lab-act2]]

Here is why the right choice cannot fail. Line 3 asks for the k-means cost and a log that never increases: that is precisely the proposition that k-means never increases its cost, checked step by step, and the fixed-point guarantee at the end. Line 1 is \(k = 4\). Line 2 follows from how k-means computes: multiply every message by 10, and if the starting centres are the same messages, every distance is multiplied by 10, so every “nearest centre” is the same, every mean is multiplied by 10, and the run makes exactly the same assignments. Notice what the manager did *not* ask for, and what you therefore did not promise: that the four groups are the four kinds. No method could promise that without labels. With several restarts, k-means often finds them, and the lab lets you peek, but that is a pleasant observation about this mail, not a guarantee.

## What people get wrong

**“k-means finds the best clustering.”** It finds a fixed point of two alternating steps, and the cost never goes up on the way. On the twelve messages, the start at messages 1 and 7 stops at cost 35 while 20 is possible, and a stretched rectangle of four messages makes the gap as large as you like. Restarts help; they do not certify.

**“If a clustering looks right on the plot, the method is right.”** A method that looks right on this batch has made no promise about the next one. Average linkage drew the four kinds on 25 of 30 test batches of the lab’s mail, but it minimizes no stated cost and keeps no log, so nothing it does can be checked against what the desks asked for.

**“Single linkage is the natural choice: close things belong together.”** Close things do belong together under single linkage, so a chain of close messages, like the odd invoices between invoices and scams, glues two kinds into one. On the lab’s mail, single linkage with four groups typically returns one giant group and a few leftovers.

**“There must be a correct clustering; we just haven’t found the right algorithm.”** Without labels, even a perfect knowledge of the mail distribution does not single out one grouping, and the impossibility theorem shows that not even a short list of reasonable properties can be met at once. Choosing the method is choosing what you want.

## Recap and the next bug

1. **Clustering** partitions a set \(\mathcal X\) with a distance function \(d\) into groups, without labels. It is ill-defined in two ways: similarity is not transitive while sharing a group is, and there is no ground truth to score against.
2. **Linkage** methods merge the closest clusters, measured by their closest pair (single), average pair or farthest pair (max), and turn the resulting dendrogram into a partition with a stopping rule. Single linkage is Kruskal’s algorithm, so its merges form a minimum spanning tree. Cost \(O(m^3)\) time and \(O(m^2)\) space, \(O(m^2)\) for single linkage via Prim.
3. **Cost minimization** picks the partition of smallest cost, for example the k-means cost, the total squared distance to the group means. Minimizing it is NP-hard. The **k-means algorithm** alternates assignment and update; its cost never increases, it stops at a fixed point, and that fixed point can be far from optimal.
4. **Spectral clustering** relaxes RatioCut, which equals \(\operatorname{trace}(H^\top L H)\) for the unnormalized Laplacian \(L = D - W\), to an eigenvector problem, then runs k-means on the rows of the eigenvectors.
5. The **information bottleneck** trades compression \(I(x; C)\) against relevance \(I(C; y)\).
6. **No clustering function** is scale invariant, rich and consistent at once. Every pair is possible, and so is the triple with \(k\)-richness. Like a hypothesis class, a clustering method has to be chosen from prior knowledge of the task.

The desks are happy for a week. Then the data team, eager to help, replaces the two scores with something richer: for every message, the count of every word in a 50,000-word dictionary. Every message is now a point in 50,000 dimensions. k-means slows to a crawl, the distance between any two messages starts to look like the distance between any other two, and the clusters turn noisy. Surely those 50,000 counts don’t carry 50,000 independent pieces of information. Can every message be squeezed into a handful of numbers without losing what makes messages different from each other? That’s the next lesson: dimensionality reduction.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Three messages sit on one score at 0, 2 and 10. You run k-means with \(k = 2\), but start the centres at 20 and 30. Trace the algorithm, using the rule that an empty cluster keeps its old centre. Where does it stop, and at what cost? What is the best possible k-means cost here? Does the proposition that k-means never increases its cost still hold when a cluster is empty, and which line of its proof needs the empty-cluster rule?
2. In the proof of the impossibility theorem, \(d_1\) was chosen so that \(F(\mathcal X, d_1)\) puts every element alone. Suppose instead you picked a \(d_1\) whose clustering had a cluster \(\{x, y\}\) with two elements. Which condition of a consistent change would \(d_3 = \alpha d_2\) now have to satisfy for the pair \(x, y\), and why can step 2 no longer guarantee it? Separately: where exactly does the proof use that dissimilarities are positive between distinct elements?
3. Four vertices in a path, 0 – 1 – 2 – 3, with edge weights \(W_{0,1} = 0.3\), \(W_{1,2} = 0.35\), \(W_{2,3} = 1\) and no other edges. List the cut and the RatioCut of the three two-cluster splits that cut a single edge. Which split does mincut choose, and which does RatioCut choose? For the RatioCut winner, write down its \(H\) and check that \(\operatorname{trace}(H^\top L H)\) gives the same number.

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (k-means can stop where no local minimum is).** Take the messages at \(1, 2, 3, 4\) on one score, \(k = 2\), and start the centres at \(2\) and \(4\). Break ties in the assignment step toward the smaller index \(j\) among the nearest centres. Run k-means to its end, then show that the point where it stops is not a local minimum of the cost viewed as a function of the two centres: find centres arbitrarily close to the final ones with strictly lower cost.
- **Problem 2 (a fast 2-approximation for the widest cluster).** Given a finite \(\mathcal X\) with a distance function \(d\) satisfying the triangle inequality, and \(k\), the **k-diameter** cost of a clustering is \(\max_j \operatorname{diam}(C_j)\), where \(\operatorname{diam}(C_j) = \max_{x, x' \in C_j} d(x, x')\) (0 for a cluster with fewer than two elements). Minimizing it is NP-hard. Consider the **farthest-first** algorithm: pick any \(x \in \mathcal X\) as \(\mu_1\); for \(j = 2, \dots, k\), let \(\mu_j\) be a message whose distance to its nearest already-chosen centre is largest; then put each message in the cluster of its nearest centre. Prove that its k-diameter cost is at most twice the smallest possible.
- **Problem 3 (single linkage under three stopping rules).** For single linkage with (a) a fixed number of clusters \(k \ge 1\) and (b) a fixed distance upper bound \(r > 0\), find which of scale invariance, richness and consistency hold, with proofs or counterexamples. Then (c) show that for each pair of the three properties there is a stopping rule for single linkage under which both properties of the pair hold.
