# Online Learning

## The spammers are watching

For twenty lessons the filter has lived a sheltered life. Every guarantee you proved began the same way: the messages arrive at random from some fixed distribution \(\mathcal D\), each one drawn independently of the others, and the learner gets a whole training sequence \(S \sim \mathcal D^m\) before it has to say anything. Under that promise you learned to count how many labels buy how much accuracy, from finite classes in lesson 2 to neural networks in lesson 20.

This week the promise broke. The spammers have noticed the filter. They no longer send a big batch of junk and wait. They send one message, see whether it got through, and then decide what to send next. If “FREE” in the subject gets caught, the next message says “F.R.E.E”. If the filter lets a message with a link through, the next hundred all have links. Every decision the filter makes is information for them, and they use it straight away.

Your monitoring shows the damage. The filter is retrained every night on the day’s labelled mail, and every morning it starts losing again, because whatever it learned overnight is exactly what the spammers stop doing. There is no distribution here. The mail today depends on what the filter did yesterday, and on what it did five minutes ago. Not one of the guarantees from lessons 2 to 20 applies, because each of them assumed that the examples are drawn independently from one fixed \(\mathcal D\), and an opponent who reacts to you is about as far from that as anything can be.

So what can a learner promise when the mail is chosen by someone who reads its decisions? It sounds hopeless. It is not, and this lesson is about what you can still prove.

## One message at a time

First, the shape of the new problem. Messages arrive one at a time. For each one, the filter must decide on the spot, junk or keep. Only afterwards does it find out the right answer, because a user opens the message and either reports it as junk or doesn’t. Then the next message arrives.

Each pass through that loop is a **round**. On round \(t\):

1. a message \(x_t\) arrives (the spammers chose it);
2. the learner announces a **prediction** \(p_t\), its label for \(x_t\), with 1 for junk and 0 for keep, as always;
3. the true label \(y_t\) is revealed;
4. the learner may update itself before round \(t + 1\).

A learner that works this way is an **online learner**, and the setting is **online learning**. Lesson 1 named it, with the stockbroker who must decide every day using only what happened so far; now we study it. There is no separate training phase and test phase. Every message is first a test (the learner must predict) and then a training example (the learner sees the label). The score is the number of **mistakes**, the rounds where \(p_t \neq y_t\). Each mistake is a junk message in someone’s inbox or a real message in the junk folder, so the filter is paying for its education as it goes.

[[diagram:protocol]]

Notice what is missing: no \(\mathcal D\). We make no assumption about where the messages come from. They may be fixed in advance, random, or chosen by an opponent who watches the learner’s every move and changes tactics after each decision. We call that opponent the **adversary**. A guarantee that holds against the adversary holds against anything, which is exactly what a filter facing real spammers needs.

Can anything be guaranteed against such an opponent? Let’s try the simplest possible attack before anything else. The adversary sends the *same* message every round, waits for the prediction, and then declares the opposite label to be the true one. Whatever the learner does, it is wrong.

```python
def learner(history, x):
    past = [y for (u, y) in history if u == x]
    return int(2 * sum(past) >= len(past)) if past else 0

history, mistakes, T = [], 0, 100
for t in range(T):
    x = (1, 0, 1, 1)
    p = learner(history, x)
    y = 1 - p
    mistakes += p != y
    history.append((x, y))
print(mistakes, "mistakes in", T, "rounds")
```

It prints `100 mistakes in 100 rounds`. The learner here is a sensible one: it predicts whatever label message \(x\) got most often in the past. It does not matter. The adversary reads `p` and sets `y = 1 - p`, so every round is a mistake. Replace `learner` with anything you like that returns 0 or 1, and the count stays at 100.

So with no restriction at all, the adversary wins every round. To say anything, we have to restrict the adversary somehow, the same way lesson 1 had to restrict the learner. The question is which restriction is both realistic and enough.

## Realizability buys a mistake bound

The attack above works because the adversary’s labels make no sense: the same message is junk on Monday and keep on Tuesday. Real junk has a rule behind it. So here is the first restriction, and you have met it before. In lesson 2 it was called **realizability**: the labels come from some hypothesis in a class \(\mathcal H\) that the learner knows in advance.

Online, it reads like this. Before the game starts, the learner is told \(\mathcal H\). The adversary still chooses every message, and in whatever order and with whatever cunning it likes. It may even put off deciding which hypothesis is “the truth” until the end, as long as, at the end, some single \(h^\star \in \mathcal H\) agrees with every label it gave. So all labels satisfy \(y_t = h^\star(x_t)\). Under that rule, the attack with one repeated message dies: once the adversary has called \(x\) junk, it must call \(x\) junk forever, because one hypothesis cannot give one message two labels.

Let’s use the smallest class we know, the one-feature class of lesson 1. Messages are four bits FLUN (says “FREE”, has a link, unknown sender, sent at night), and the class has four hypotheses, \(\mathcal H_1 = \{h_F, h_L, h_U, h_N\}\), where \(h_F\) says junk exactly when the F bit is 1, and so on. The adversary has picked one of the four as the truth, or will pick one by the end. How many mistakes can it force?

That number, the worst the adversary can do, is how we will measure online learners. Here it is in words first: run the learner on a sequence of examples and count its mistakes; then take the worst sequence the adversary could choose, over every length and every \(h^\star\) in the class. If that worst count is finite, the learner has a promise to make: “whatever you send me, I will make at most this many mistakes, ever.”

Now the notation. A sequence of \(T\) labelled messages is written \(S = ((x_1, y_1), \dots, (x_T, y_T))\), where \(T\) is the number of rounds. We call \(S\) **realizable by \(\mathcal H\)** when some \(h^\star \in \mathcal H\) has \(y_t = h^\star(x_t)\) for every round \(t\). For an online learner \(A\),

\[ M_A(S) = \left| \{ t \in [T] : p_t \neq y_t \} \right|, \qquad M_A(\mathcal H) = \sup_{S} M_A(S). \]

Read it from the inside out.

- \(p_t\) is the prediction \(A\) made on round \(t\), and \(y_t\) the label revealed afterwards. The test \(p_t \neq y_t\) asks one yes/no question: was round \(t\) a mistake?
- \([T]\) is the set of round numbers \(\{1, 2, \dots, T\}\), as \([m]\) was the set of example positions in lesson 1.
- The braces \(\{ t \in [T] : \dots \}\) collect the rounds where the answer is yes. The colon reads “such that”: the rounds \(t\) such that \(A\) got round \(t\) wrong.
- The bars \(|\cdot|\) mean “how many”. So \(M_A(S)\) is the number of mistakes \(A\) makes on the sequence \(S\). No division this time: online, we count mistakes rather than average them.
- \(\sup_S\) (“supremum over \(S\)”) means: the largest value over every realizable sequence \(S\), of any length \(T\), labelled by any \(h^\star \in \mathcal H\). It is a maximum that is allowed to be infinite, or to be approached without being reached.

In plain words: **\(M_A(S)\) is how many mistakes learner \(A\) makes on one sequence, and \(M_A(\mathcal H)\) is the most mistakes any adversary can force on it, as long as the labels follow some hypothesis in \(\mathcal H\).** For a tiny number: if the learner predicts “junk” on every message and the adversary sends 1000, 0100, 0010 (all keep) and then 0001 (junk), labelled by \(h_N\), then \(M_A(S) = 3\).

A bound \(M_A(\mathcal H) \le B < \infty\) is a **mistake bound** (here \(B\) is just a number, not the norm bound of lessons 12 to 15). A class \(\mathcal H\) is **online learnable** if some online learner \(A\) has a mistake bound for it. For now we ignore running time completely, as lesson 3 did at first with PAC learning: the learners in the first half of this lesson may be slow, and only the last two learners will be efficient.

## Consistent: keep the survivors

Lesson 2 taught us a learner that works in the batch world: return any hypothesis that is consistent with every example (under realizability, that is what empirical risk minimization, ERM, returns). The online version is just as natural. Keep the set of hypotheses that have agreed with every label so far; when a message arrives, pick any one of them and predict with it. Call it **Consistent**.

The set of hypotheses still consistent with everything seen is the **version space**. On round 1 it is the whole class. After each round, every hypothesis that disagreed with the revealed label leaves.

Let’s play it against the adversary on \(\mathcal H_1\). Suppose Consistent picks the first surviving hypothesis in the order F, L, U, N. The adversary knows this (it may know the learner’s code; only the truth is hidden from the learner). Before reading the table, ask yourself what message the adversary should send first, to make \(h_F\) wrong.

| t | message FLUN | Consistent predicts with | prediction \(p_t\) | label \(y_t\) | version space after |
|---|---|---|---|---|---|
| 1 | 1000 | \(h_F\) | 1 | 0 | L, U, N |
| 2 | 0100 | \(h_L\) | 1 | 0 | U, N |
| 3 | 0010 | \(h_U\) | 1 | 0 | N |
| 4 | 0001 | \(h_N\) | 1 | 1 | N |
| 5 | 1111 | \(h_N\) | 1 | 1 | N |
| 6 | 0000 | \(h_N\) | 0 | 0 | N |

The trick is to send a message on which only the chosen hypothesis says junk, and then call it keep. Three rounds, three mistakes, each removing exactly one hypothesis. The labels are realizable, by \(h_N\), so the adversary played fair. After that, only \(h_N\) is left and Consistent never errs again.

Let’s write the version space down. Call it \(V_t\), the hypotheses still standing at the start of round \(t\). Then \(V_1 = \mathcal H\) and

\[ V_{t+1} = \{ h \in V_t : h(x_t) = y_t \}. \]

- \(V_t\) is a set of hypotheses; \(h \in V_t\) reads “\(h\) is one of them”.
- \(h(x_t) = y_t\) asks whether \(h\) agrees with the label just revealed.
- The braces and colon read “the hypotheses \(h\) in \(V_t\) such that \(h\) agrees with round \(t\)”.

In plain words: **after each round, keep exactly the hypotheses that got that round right.** In the table, \(V_1 = \{h_F, h_L, h_U, h_N\}\); round 1 reveals that 1000 is keep, and only \(h_F\) says junk on 1000, so \(V_2 = \{h_L, h_U, h_N\}\). Here is the table again as code, with the version space printed after each round:

```python
H = "FLUN"
h = lambda name, x: int(x[H.index(name)])
consistent = lambda V, x: h(min(V, key=H.index), x)

def play(learn, stream):
    V, M = set(H), 0
    for x, y in stream:
        p = learn(V, x)
        M += p != y
        V = {n for n in V if h(n, x) == y}
        print(x, p, y, "".join(sorted(V, key=H.index)))
    return M

s1 = [("1000", 0), ("0100", 0), ("0010", 0), ("0001", 1),
      ("1111", 1), ("0000", 0)]
print(play(consistent, s1))
```

It prints the six rows of the table (`1000 1 0 LUN`, `0100 1 0 UN`, `0010 1 0 N`, then three rows ending in `N`) and finally `3`, the mistake count \(M_A(S)\). The line `V = {n for n in V if h(n, x) == y}` is the version-space formula, symbol for symbol.

Could the adversary have done better than three? Here is the argument that it couldn’t, in words first. Every mistake throws out at least one hypothesis: the one Consistent predicted with was wrong, so it leaves. And the truth \(h^\star\) never leaves, because it is never wrong. So the version space starts with \(|\mathcal H|\) hypotheses, loses at least one per mistake, and always keeps at least one.

**Claim (Consistent’s mistake bound).** For a finite class \(\mathcal H\), \(M_{\text{Consistent}}(\mathcal H) \le |\mathcal H| - 1\).

*Proof.* Run Consistent on any realizable sequence and let \(M\) be its number of mistakes. On a mistake round, the hypothesis \(h\) that Consistent used predicted \(p_t = h(x_t) \neq y_t\), so \(h \notin V_{t+1}\): the version space shrinks by at least one. On other rounds it does not grow, since \(V_{t+1} \subseteq V_t\). So after all \(T\) rounds, \(|V_{T+1}| \le |\mathcal H| - M\).

The truth \(h^\star\) has \(h^\star(x_t) = y_t\) on every round, so by induction it is in every \(V_t\), and \(|V_{T+1}| \ge 1\). Putting the two together, \(1 \le |\mathcal H| - M\), which rearranges to \(M \le |\mathcal H| - 1\). The sequence was arbitrary, so the supremum obeys the same bound. \(\blacksquare\)

With four hypotheses, that is at most 3, which the adversary reached. For lesson 1’s ten one-feature hypotheses it is 9, and for a class of a million hypotheses it is 999,999. A bound that grows linearly with the size of the class is a poor promise. Can a learner do better by choosing more carefully which survivor to follow?

## Halving: let the survivors vote

Here is a better idea. Instead of following one survivor, let all of them vote. If more survivors say junk than keep, predict junk; otherwise predict keep. Call it **Halving**, and when the vote is tied, let it predict junk.

Why the name? Think about what a mistake means for a voting learner. If Halving is wrong, the majority was wrong, and every hypothesis in the majority leaves the version space. At least half of the survivors are gone in one round. That is the whole idea, and it is worth seeing with numbers before any symbols.

Before reading the next table, try to play adversary yourself. Against Halving on \(\mathcal H_1\), what should the first message be? Sending 1000 again is a waste: three survivors say keep and one says junk, so Halving predicts keep, and calling it keep is no mistake, while calling it junk is a mistake that leaves only \(h_F\). The adversary wants a tie, so that whatever it says, it throws out only half.

| t | message FLUN | votes junk / keep | prediction \(p_t\) | label \(y_t\) | survivors after |
|---|---|---|---|---|---|
| 1 | 1100 | 2 / 2 | 1 | 0 | 2 (\(h_U, h_N\)) |
| 2 | 0010 | 1 / 1 | 1 | 0 | 1 (\(h_N\)) |
| 3 | 1111 | 1 / 0 | 1 | 1 | 1 |
| 4 | 0001 | 1 / 0 | 1 | 1 | 1 |
| 5 | 0000 | 0 / 1 | 0 | 0 | 1 |
| 6 | 1001 | 1 / 0 | 1 | 1 | 1 |

Two mistakes, and the count of survivors goes 4, 2, 1. Each mistake halved it.

```python
H = "FLUN"
h = lambda name, x: int(x[H.index(name)])
halving = lambda V, x: int(2 * sum(h(n, x) for n in V)
                           >= len(V))
V, M = set(H), 0
s2 = [("1100", 0), ("0010", 0), ("1111", 1), ("0001", 1),
      ("0000", 0), ("1001", 1)]
for x, y in s2:
    p = halving(V, x)
    M += p != y
    V = {n for n in V if h(n, x) == y}
    print(x, p, y, len(V))
print(M)
```

It prints `1100 1 0 2`, `0010 1 0 1`, four rows with no mistake and one survivor, and then `2`. The line `2 * sum(...) >= len(V)` is the vote: junk when at least half of the survivors say junk, so a tie goes to junk.

Now the general rule, again in words before symbols. The prediction is the label with more votes among the survivors. If that prediction is wrong, the survivors that voted for it all leave, and they were at least half. So every mistake at least halves the version space, while the truth never leaves. Starting from \(|\mathcal H|\), you can only halve about \(\log_2 |\mathcal H|\) times before you are down to one.

The prediction rule, written out, is

\[ p_t = \operatorname*{argmax}_{r \in \{0, 1\}} \left| \{ h \in V_t : h(x_t) = r \} \right| \quad (\text{ties go to } 1). \]

- \(r\) ranges over the two possible labels, 0 (keep) and 1 (junk).
- \(\{ h \in V_t : h(x_t) = r \}\) is the set of survivors that say \(r\) on this message, and \(|\cdot|\) counts them: the number of votes for \(r\).
- \(\operatorname{argmax}_{r}\) means “the value of \(r\) that makes the expression largest”: not the largest vote count itself, but the label that has it.

In plain words: **predict the label that most survivors give this message.** In row 1, \(V_1\) has four survivors; \(h_F\) and \(h_L\) say 1 on 1100 and \(h_U\), \(h_N\) say 0; the votes are 2 and 2, a tie, so \(p_1 = 1\).

**Theorem (Halving’s mistake bound).** For a finite class \(\mathcal H\), \(M_{\text{Halving}}(\mathcal H) \le \log_2 |\mathcal H|\).

*Proof.* Take any realizable sequence and suppose Halving errs on round \(t\). The hypotheses that said \(p_t\) on \(x_t\) were wrong, so all of them leave; what remains, \(V_{t+1}\), is the set that said \(y_t = 1 - p_t\). Since \(p_t\) won the vote (or tied), the losing side had at most half the votes: \(|V_{t+1}| \le |V_t|/2\). On rounds without a mistake, \(|V_{t+1}| \le |V_t|\).

So if \(M\) mistakes happen in \(T\) rounds, the size has been halved at least \(M\) times, and the truth is still in it:

\[ 1 \le |V_{T+1}| \le |\mathcal H| \, 2^{-M}. \]

- \(|V_{T+1}|\) is how many hypotheses survive all \(T\) rounds.
- \(2^{-M} = 1/2^M\) is what \(M\) halvings do to a count: multiply it by one half, \(M\) times.
- The left inequality is the truth surviving; the right one is the halving.

In plain words: **the survivors never run out, and every mistake at least halves them.** In the table, \(|\mathcal H_1| = 4\) and \(M = 2\), so \(4 \cdot 2^{-2} = 1\), and indeed exactly one survivor is left: the inequality is tight. Multiplying both sides by \(2^M\) gives \(2^M \le |\mathcal H|\), and taking \(\log_2\) of both sides gives \(M \le \log_2 |\mathcal H|\). \(\blacksquare\)

For four hypotheses, \(\log_2 4 = 2\), which the adversary reached. For a million hypotheses it is about 19.9, so at most 19 mistakes, instead of Consistent’s 999,999. Exponentially better, by changing only *which* survivor we listen to.

You don’t have to trust either table to show the worst case. Python can play every possible adversary against each learner, trying every message and both labels in every round, and report the most mistakes any of them can force. That is \(M_A(\mathcal H_1)\) itself, the supremum, computed by brute force:

```python
from functools import lru_cache
from itertools import product

X, H = list(product([0, 1], repeat=4)), "FLUN"
consistent = lambda V, x: x[H.index(min(V, key=H.index))]
halving = lambda V, x: int(2 * sum(x[H.index(h)] for h in V)
                           >= len(V))

@lru_cache(None)
def worst(V, learn):
    games = [(learn(V, x) != y) + worst(W, learn)
             for x in X for y in (0, 1)
             for W in [frozenset(h for h in V
                                 if x[H.index(h)] == y)]
             if W and W != V]
    return max(games, default=0)

print(worst(frozenset(H), consistent),
      worst(frozenset(H), halving))
```

It prints `3 2`. The function `worst` is the adversary: from version space `V`, it tries each message `x` and each label `y` that keeps some hypothesis alive (`if W`), adds one if the learner got that round wrong, and recurses. Rounds where every survivor agrees teach nothing and cost nothing, so it skips them (`W != V`). The `max` is the \(\sup\) of the definition.

Notice something that separates the online world from the batch world. In lesson 2, *any* consistent hypothesis was a fine output. Online, choosing an arbitrary consistent hypothesis can cost exponentially more mistakes than choosing well.

[[diagram:version-space]]

## The adversary’s plan is a tree

Halving’s bound depends on \(|\mathcal H|\), the number of hypotheses. Lesson 6 taught us to distrust counts: the class of thresholds is infinite, yet it is learnable from a few labels. Is there a better measure of how hard a class is online? To find it, look at the game from the adversary’s side.

The adversary wants to force a mistake on every one of the first few rounds. That part is easy: it waits for the prediction and announces the opposite label. The hard part is staying honest, so that at the end some hypothesis in \(\mathcal H\) agrees with every label it announced. Which message should it send on round 2? That depends on what happened on round 1, which depends on the learner. So the adversary’s plan is not a list of messages. It is a branching plan: “send \(v_1\) first; if the label turns out 0, send this next; if 1, send that.”

A branching plan is a **binary tree**. Each node holds a message. The adversary starts at the top node, the **root**, and sends its message. When the label is revealed, it moves to the left child if the label was 0 and to the right child if it was 1, and sends that node’s message next. The **depth** of the tree is the number of steps along a path from the root to the bottom. A tree in which every path has the same depth \(d\) and every node above the bottom has two children is **complete**, and it has \(2^d - 1\) nodes that hold messages: 1 for depth 1, 3 for depth 2.

Here is one for \(\mathcal H_1\), of depth 2. The root holds 1100, on which \(h_F\) and \(h_L\) say junk, \(h_U\) and \(h_N\) keep. Its left child, used after the label 0, holds 0010: the survivors are then \(h_U\) and \(h_N\), and 0010 splits them. Its right child, used after the label 1, holds 1000: the survivors are \(h_F\) and \(h_L\), and 1000 splits them. Every one of the four label paths (0 then 0, 0 then 1, 1 then 0, 1 then 1) is followed exactly by one hypothesis. So whatever the learner predicts, the adversary can announce the opposite label twice and still have a hypothesis that agrees. That is the depth-2 game from the Halving table, drawn as a plan.

[[diagram:shattered-tree]]

To write this down we need names for the nodes. Number them like a heap in a computer science course: the root is node 1, and node \(i\) has left child \(2i\) and right child \(2i + 1\). So the messages are \(v_1, v_2, \dots, v_{2^d - 1}\). If \(i_t\) is the node used on round \(t\), then \(i_1 = 1\) and \(i_{t+1} = 2 i_t + y_t\): double to go down a level, and add the label to go right on a 1. Unrolled,

\[ i_t = 2^{t-1} + \sum_{j=1}^{t-1} y_j \, 2^{t-1-j}. \]

- \(2^{t-1}\) is the number of the leftmost node on level \(t\) (levels counted from 1 at the root).
- \(\sum_{j=1}^{t-1}\) adds one term for each earlier round \(j = 1, \dots, t - 1\).
- \(y_j \, 2^{t-1-j}\) adds the label of round \(j\), worth more the earlier it was: the labels so far, read as a binary number, are the offset from the left of the level.

In plain words: **the node used on round \(t\) is found by writing the labels so far as a binary number and adding it to the first node number on level \(t\).** With labels \(y_1 = 1\), \(y_2 = 0\): \(i_1 = 1\), and \(i_2 = 2 + 1 \cdot 1 = 3\), the right child, holding 1000 in our tree.

```python
from itertools import product

def path(ys):
    i = [1]
    for y in ys[:-1]:
        i.append(2 * i[-1] + y)
    return i

closed = lambda ys, t: 2 ** (t - 1) + sum(
    ys[j - 1] * 2 ** (t - 1 - j) for j in range(1, t))
print(path((1, 0)))
print(all(path(ys)[t - 1] == closed(ys, t)
          for ys in product([0, 1], repeat=5)
          for t in range(1, 6)))
v = {1: "1100", 2: "0010", 3: "1000"}
h = {n: (lambda x, k=k: int(x[k])) for k, n in enumerate("FLUN")}
print(all(any(all(h[n](v[i]) == y for i, y in zip(path(ys), ys))
              for n in h) for ys in product([0, 1], repeat=2)))
```

It prints `[1, 3]` (the path for labels 1 then 0), `True` (the recursion and the closed form agree on every path of depth 5) and `True` (our depth-2 tree for \(\mathcal H_1\) works: for each of the four label paths, some hypothesis agrees with all of it).

That last check is the property we need, and it gets a name.

**Definition (shattered tree).** A complete binary tree of depth \(d\) with messages \(v_1, \dots, v_{2^d - 1}\) is **shattered** by \(\mathcal H\) if for every sequence of labels \((y_1, \dots, y_d) \in \{0, 1\}^d\) there is some \(h \in \mathcal H\) with \(h(v_{i_t}) = y_t\) for every \(t \in [d]\), where \(i_t\) is the node the labels lead to.

In plain words: **whichever way the labels go down the tree, some hypothesis in the class agrees with all of them.** The word is borrowed from lesson 6, where a set of messages was shattered when the class could label it in every possible way; here it is a tree instead of a set, so later messages may depend on earlier labels.

**Definition (Littlestone dimension).** The **Littlestone dimension** of \(\mathcal H\), written \(\mathrm{Ldim}(\mathcal H)\), is the largest depth \(d\) of a tree shattered by \(\mathcal H\). If there are shattered trees of every depth, \(\mathrm{Ldim}(\mathcal H) = \infty\).

The tree of depth 0 has no nodes, and the one label sequence of length 0 is matched by any hypothesis at all, so every nonempty class has \(\mathrm{Ldim} \ge 0\). Our tree shows \(\mathrm{Ldim}(\mathcal H_1) \ge 2\).

Try it before we prove anything. In the lab below, you play the adversary against Consistent, Halving, and a third learner we will meet shortly, on a class you choose, and you see how many mistakes you can force while keeping the labels honest. Watch your own moves: they will trace out a tree.

[[lab]]

## No learner beats the tree

The first fact about the Littlestone dimension is the one the adversary’s story already told us.

**Lemma (the tree lower bound).** For every online learner \(A\), \(M_A(\mathcal H) \ge \mathrm{Ldim}(\mathcal H)\).

*Proof.* Let \(T = \mathrm{Ldim}(\mathcal H)\), assumed finite for now, and take a tree of depth \(T\) shattered by \(\mathcal H\). The adversary walks it: on round \(t\) it sends \(x_t = v_{i_t}\), waits for \(p_t\), and announces \(y_t = 1 - p_t\). Every one of the \(T\) rounds is a mistake.

Is the sequence realizable? The labels \((y_1, \dots, y_T)\) form one particular path, and the tree is shattered, so some \(h \in \mathcal H\) has \(h(v_{i_t}) = y_t\) for every \(t\). That \(h\) is a legitimate truth. So this realizable sequence costs \(A\) exactly \(T\) mistakes, and the supremum \(M_A(\mathcal H)\) is at least \(T\). If \(\mathrm{Ldim}(\mathcal H) = \infty\), run the same argument with a shattered tree of each depth \(T\), so \(M_A(\mathcal H) \ge T\) for every \(T\), which means \(M_A(\mathcal H) = \infty\). \(\blacksquare\)

The learner can be as clever as it likes; it can even know the tree. It does not help, because the adversary moves after the learner has predicted.

## Three classes, three dimensions

Let’s compute the dimension of a few classes, because the answers are surprising.

**A finite class.** A tree of depth \(d\) has \(2^d\) label paths, and each path needs its own hypothesis (two different paths disagree on some node they both visit, so no hypothesis follows both). So a shattered tree of depth \(d\) needs \(|\mathcal H| \ge 2^d\), that is, \(\mathrm{Ldim}(\mathcal H) \le \log_2 |\mathcal H|\). The tree lower bound and Halving’s bound give the same thing from the other side: \(\mathrm{Ldim}(\mathcal H) \le M_{\text{Halving}}(\mathcal H) \le \log_2 |\mathcal H|\).

**One rogue sender.** Suppose the junk this week comes from a single sender, and you don’t know which. Number the possible senders \(1, \dots, d\), let a message be just its sender, and let \(h_j\) say junk exactly when the sender is \(j\). This is the class of **singletons**, with \(d\) hypotheses, so \(\log_2 d\) can be large. Yet its Littlestone dimension is 1. A tree of depth 1 is just a root: send sender 1, and \(h_1\) says junk while \(h_2\) says keep, so depth 1 is shattered. For depth 2, the root holds some sender \(j\); on the label 1 branch, only \(h_j\) is left, and one hypothesis cannot be split, so no depth-2 tree is shattered. The dimension can be far below \(\log_2 |\mathcal H|\).

**Thresholds on a score.** Lesson 6 met thresholds on a spam score in \([0, 1]\): for each cut-off \(\theta\) (“theta”), the hypothesis \(h_\theta(x) = \mathbb 1[x > \theta]\) says junk when the score \(x\) is above \(\theta\). (The indicator \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false, so \(h_{0.5}(0.7) = 1\), junk, and \(h_{0.5}(0.2) = 0\), keep.) Its VC dimension is 1, so in the batch world it is learnable from a handful of labels. Online, the adversary plays binary search. It sends score \(1/2\). Whatever the learner predicts, it announces the opposite. If it said junk, the threshold must be below \(1/2\), so it next sends \(1/4\); if keep, the threshold is at least \(1/2\), so it sends \(3/4\). And so on, halving the interval of thresholds that agree with everything so far. Every round is a mistake, and because the interval never becomes empty, some threshold agrees with all the labels, at every depth. So \(\mathrm{Ldim} = \infty\), and *no* online learner has a mistake bound for thresholds.

```python
from fractions import Fraction

def learner(x, lo, hi):
    return int(x > (2 * lo + hi) / 3)

lo, hi, stream, mistakes = Fraction(0), Fraction(1), [], 0
for t in range(20):
    x = (lo + hi) / 2
    p = learner(x, lo, hi)
    y = 1 - p
    mistakes += p != y
    stream.append((x, y))
    lo, hi = (lo, x) if y == 1 else (x, hi)
t_star = (lo + hi) / 2
print(mistakes, all(int(x > t_star) == y for x, y in stream))
```

It prints `20 True`: twenty mistakes in twenty rounds, and the threshold \(\theta^\star\) in the middle of the final interval (`t_star`) agrees with every label. The learner here predicts with a threshold one third of the way into the current interval; swap in any rule you like. The exact fractions matter: with real numbers the binary search never runs out of room, and that is the density the proof uses.

Wow. A class that is easy for the batch filter is impossible for the online filter. The difference is that the batch filter got its examples at random, and an adversary is not random.

On a finite grid of scores the picture is in between. With 7 possible scores there are 8 thresholds and the dimension is 3; with 15 scores, 4. The next snippet computes Littlestone dimensions by brute force, using a recursion we prove in the next section; here it checks the three examples.

```python
from functools import lru_cache
from itertools import product

@lru_cache(None)
def ldim(V):
    if not V:
        return -1
    n = len(next(iter(V)))
    splits = [[frozenset(h for h in V if h[x] == r)
               for r in (0, 1)] for x in range(n)]
    return max([1 + min(map(ldim, s)) for s in splits
                if all(s)], default=0)

X = list(product([0, 1], repeat=4))
one_feature = {tuple(x[j] for x in X) for j in range(4)}
singletons = {tuple(int(x == j) for x in range(8))
              for j in range(8)}
thresholds = lambda n: {tuple(int(x > t) for x in range(n))
                        for t in range(-1, n)}
print(ldim(frozenset(one_feature)), ldim(frozenset(singletons)),
      ldim(frozenset(thresholds(7))),
      ldim(frozenset(thresholds(15))))
```

It prints `2 1 3 4`: \(\mathcal H_1\) has dimension 2 (so our tree was the deepest possible), eight singletons have dimension 1 while \(\log_2 8 = 3\), and thresholds on 7 and 15 grid points have dimensions 3 and 4, the depth of binary search. Each hypothesis is stored as the tuple of its labels on all messages.

## The dimension splits at the root

To build the best learner, we need one structural fact. Pick a message \(x\) and split a class \(V\) by what its hypotheses say on \(x\):

\[ V^{(r)} = \{ h \in V : h(x) = r \}, \qquad r \in \{0, 1\}. \]

- \(V^{(1)}\) is the part of \(V\) that says junk on \(x\), and \(V^{(0)}\) the part that says keep. The superscript \((r)\) names the label, not a power.

In plain words: **split the class into the hypotheses that call \(x\) junk and those that call it keep.** For \(\mathcal H_1\) and \(x = 1100\): \(V^{(1)} = \{h_F, h_L\}\) and \(V^{(0)} = \{h_U, h_N\}\).

Now look at a shattered tree with root \(x\). Every label path starts with \(y_1\). The paths that start with 0 go into the left subtree, and they must be matched by hypotheses that also say 0 on \(x\), that is, by \(V^{(0)}\). So the left subtree is a tree of depth \(d - 1\) shattered by \(V^{(0)}\), and the right subtree is one shattered by \(V^{(1)}\). The converse holds too: put any message \(x\) at a root, hang a tree shattered by \(V^{(0)}\) on the left and one shattered by \(V^{(1)}\) on the right, both of depth \(d - 1\), and the result is a tree of depth \(d\) shattered by \(V\). (If the two subtrees have different depths, cut the deeper one down: the top part of a shattered tree is still shattered.)

**Lemma (the root split).** For any class \(V\) with \(\mathrm{Ldim}(V) \ge 1\),

\[ \mathrm{Ldim}(V) = \max_{x} \; \Big( 1 + \min\big\{ \mathrm{Ldim}(V^{(0)}), \mathrm{Ldim}(V^{(1)}) \big\} \Big), \]

where the maximum runs over messages \(x\) on which both parts are nonempty, and \(V^{(0)}, V^{(1)}\) are the parts for that \(x\).

- \(\min\{\cdot, \cdot\}\) takes the smaller of the two dimensions: a tree of depth \(d\) needs *both* subtrees to have depth \(d - 1\), so the weaker side decides.
- The \(1 +\) counts the root.
- \(\max_x\) lets the adversary choose the best root.

In plain words: **the deepest tree puts at its root the message that splits the class into two parts that are both as rich as possible, and is one deeper than the poorer part.** For \(\mathcal H_1\) and root 1100, both parts have two hypotheses and dimension 1, so the formula gives \(1 + 1 = 2\). The `ldim` function above is this lemma, word for word: `splits` lists \((V^{(0)}, V^{(1)})\) for every message, `if all(s)` keeps those where both are nonempty, and `1 + min(map(ldim, s))` is the bracket. (It also returns 0 when no message splits the class and \(-1\) for the empty class, a convention that makes the next section’s comparisons work.)

*Proof.* Both directions are the two paragraphs before the lemma. If a tree of depth \(d \ge 1\) is shattered by \(V\), with root \(x\), its two subtrees of depth \(d - 1\) are shattered by \(V^{(0)}\) and \(V^{(1)}\), so both parts are nonempty and the bracket for that \(x\) is at least \(d\). Conversely, if the bracket for some \(x\) equals \(d\), hanging two shattered trees of depth \(d - 1\) under the root \(x\) gives a shattered tree of depth \(d\). \(\blacksquare\)

One more fact we will use: a subset can only lose dimension. If \(V' \subseteq V\), every tree shattered by \(V'\) is shattered by \(V\), so \(\mathrm{Ldim}(V') \le \mathrm{Ldim}(V)\).

## The Standard Optimal Algorithm

Halving splits the version space into the part that says junk and the part that says keep, and predicts with the bigger part. Bigger meant more hypotheses. What if “bigger” meant “higher Littlestone dimension” instead? That is the whole change, and the learner it gives is called the **Standard Optimal Algorithm**, SOA for short.

```
SOA(H):
  V_1 ← H
  for t = 1, 2, ...:
    receive x_t
    for r in {0, 1}: V_t^(r) ← {h in V_t : h(x_t) = r}
    predict p_t ← argmax over r of Ldim(V_t^(r))   (ties: 1)
    receive y_t
    V_{t+1} ← V_t^(y_t)
```

Why should this help? Halving fights the *number* of survivors. But the number is not what the adversary needs; it needs a deep shattered tree. SOA fights the dimension directly: it always predicts with the side that would leave the adversary the richer game, so that if it is wrong, the adversary is left with the poorer one.

**Lemma (SOA’s mistake bound).** \(M_{\text{SOA}}(\mathcal H) \le \mathrm{Ldim}(\mathcal H)\).

*Proof.* It is enough to show that every mistake lowers the dimension of the version space by at least one: \(\mathrm{Ldim}(V_{t+1}) \le \mathrm{Ldim}(V_t) - 1\) whenever \(p_t \neq y_t\). Since \(h^\star\) is always in the version space, the dimension never drops below 0, so it can drop at most \(\mathrm{Ldim}(\mathcal H)\) times.

Suppose round \(t\) is a mistake, and suppose for contradiction that the dimension did not drop. Write \(d = \mathrm{Ldim}(V_t)\). Since \(V_{t+1} = V_t^{(y_t)} \subseteq V_t\), its dimension is at most \(d\); not dropping means \(\mathrm{Ldim}(V_t^{(y_t)}) = d\). SOA predicted \(p_t\) because its side had the larger (or tied) dimension, so \(\mathrm{Ldim}(V_t^{(p_t)}) \ge \mathrm{Ldim}(V_t^{(y_t)}) = d\), and it is also at most \(d\) as a subset. So both parts have dimension exactly \(d\), and \(p_t\) and \(y_t\) are the two different labels, so these are \(V_t^{(0)}\) and \(V_t^{(1)}\).

Now use the root split with root \(x_t\): both parts are nonempty (they have dimension \(d \ge 0\)), so \(\mathrm{Ldim}(V_t) \ge 1 + \min\{d, d\} = d + 1\). That contradicts \(\mathrm{Ldim}(V_t) = d\). \(\blacksquare\)

Put the two lemmas side by side. Every learner can be forced into \(\mathrm{Ldim}(\mathcal H)\) mistakes, and SOA never makes more.

**Theorem (the Littlestone dimension characterizes online learnability).** For every class \(\mathcal H\), \(M_{\text{SOA}}(\mathcal H) = \mathrm{Ldim}(\mathcal H)\), and no online learner \(A\) has \(M_A(\mathcal H) < \mathrm{Ldim}(\mathcal H)\). In particular, \(\mathcal H\) is online learnable exactly when \(\mathrm{Ldim}(\mathcal H) < \infty\).

Does SOA ever beat Halving in practice? Here is a class built to show it. Take eight messages, \(x_0, u, v, s_1, \dots, s_5\). Part A of the class is five hypotheses, “junk exactly when the message is \(s_j\)”, one per \(j\): singletons, dimension 1. Part B is four hypotheses that say junk on \(x_0\), keep on every \(s_j\), and every one of the four possible label pairs on \(u\) and \(v\): dimension 2. When \(x_0\) arrives, part A (five hypotheses) says keep and part B (four) says junk. Halving counts heads and predicts keep; the adversary says junk, and is left with part B and two more forced mistakes. SOA sees that part B has the higher dimension and predicts junk; if the adversary says keep, it is left with the singletons and can force only one more mistake.

```python
from functools import lru_cache

A = [(0, 0, 0) + tuple(int(i == j) for i in range(5))
     for j in range(5)]
B = [(1, u, v) + (0,) * 5 for u in (0, 1) for v in (0, 1)]
part = lambda V, x, r: frozenset(h for h in V if h[x] == r)
moves = lambda V: [(x, y, W) for x in range(8) for y in (0, 1)
                   for W in [part(V, x, y)] if W and W != V]

@lru_cache(None)
def ldim(V):
    return max([1 + min(ldim(W), ldim(V - W))
                for x, y, W in moves(V)], default=0)

@lru_cache(None)
def worst(V, learn):
    return max([(learn(V, x) != y) + worst(W, learn)
                for x, y, W in moves(V)], default=0)

big = lambda f, V, x: int(f(part(V, x, 1)) >= f(part(V, x, 0)))
halving = lambda V, x: big(len, V, x)
soa = lambda V, x: big(ldim, V, x)
H = frozenset(A + B)
print(len(H), ldim(H), worst(H, halving), worst(H, soa))
```

It prints `9 2 3 2`: nine hypotheses, dimension 2, the best adversary forces 3 mistakes on Halving and only 2 on SOA. The only difference between the two learners is the function `big` compares the parts with: `len` for Halving, `ldim` for SOA.

A word on cost, since we promised to ignore it and should be honest about what we ignored. SOA needs the Littlestone dimension of two sets on every round, and computing it by the root split recursion visits a game tree that can be exponentially large. SOA is the optimal learner in mistakes, not in time.

## Littlestone dimension against VC dimension

Thresholds showed that online learning can be much harder than batch learning. Is it ever easier? No, and the proof is a picture.

**Theorem.** For every class \(\mathcal H\), \(\mathrm{VCdim}(\mathcal H) \le \mathrm{Ldim}(\mathcal H)\), and the gap can be arbitrarily large.

*Proof.* Recall from lesson 6 that a set \(\{x_1, \dots, x_d\}\) is shattered by \(\mathcal H\) when every one of the \(2^d\) labellings of it is given by some hypothesis, and \(\mathrm{VCdim}(\mathcal H)\) is the largest such \(d\). Take a shattered set of size \(d\) and build a complete tree of depth \(d\) whose nodes on level \(i\) all hold \(x_i\). Whatever the label path \((y_1, \dots, y_d)\), the messages visited are \(x_1, \dots, x_d\) in order, and the set is shattered, so some \(h\) gives \(h(x_i) = y_i\) for all \(i\). The tree is shattered, so \(\mathrm{Ldim}(\mathcal H) \ge d\). For the gap: thresholds on \([0, 1]\) have VC dimension 1 and Littlestone dimension \(\infty\). \(\blacksquare\)

For \(\mathcal H_1\): the set \(\{1100, 1010\}\) is shattered, since \(h_F, h_L, h_U, h_N\) give it the labels (1, 1), (1, 0), (0, 1), (0, 0). The tree with 1100 at the root and 1010 at both children is then shattered.

```python
from itertools import product

h = {n: (lambda x, k=k: int(x[k])) for k, n in enumerate("FLUN")}
x1, x2 = "1100", "1010"
print(sorted((h[n](x1), h[n](x2)) for n in h))
tree = {1: x1, 2: x2, 3: x2}
path = lambda ys: [1, 2 + ys[0]]
print(all(any(all(h[n](tree[i]) == y
                  for i, y in zip(path(ys), ys)) for n in h)
          for ys in product([0, 1], repeat=2)))
```

It prints `[(0, 0), (0, 1), (1, 0), (1, 1)]` (all four labellings of the set appear) and `True` (the tree built from the set is shattered). The VC dimension asks for a tree whose messages may not depend on the labels; the Littlestone dimension lets them depend on the labels, which is exactly the extra power an adversary has.

## When no hypothesis is perfect

Realizability was a comfortable assumption, and the spammers are not comfortable people. Real mail is noisy: some users report newsletters they subscribed to, a few junk messages look exactly like real ones, and no hypothesis in any class you can afford agrees with every label. Lesson 3 met the same problem in the batch world and answered it with **agnostic** learning: stop asking for a perfect hypothesis and ask instead to do almost as well as the best one in the class.

The online version of “almost as well as the best one” is called regret. The idea first, with numbers. Suppose that after 1,000 rounds your filter has made 130 mistakes. Looking back, you check every hypothesis in \(\mathcal H\) and find that the best of them, had you followed it from the first round, would have made 100. You are “sorry” about 30 mistakes. That 30 is your **regret** with respect to that hypothesis. If the worst-case regret grows more slowly than the number of rounds, say like \(\sqrt T\), then the regret per round goes to zero, and in the long run the filter is as good as the best hypothesis in hindsight, against any sequence at all.

In symbols, for an online learner \(A\), a hypothesis \(h\) and a number of rounds \(T\),

\[ \mathrm{Regret}_A(h, T) = \sup_{(x_1, y_1), \dots, (x_T, y_T)} \left[ \sum_{t=1}^T |p_t - y_t| - \sum_{t=1}^T |h(x_t) - y_t| \right], \qquad \mathrm{Regret}_A(\mathcal H, T) = \sup_{h \in \mathcal H} \mathrm{Regret}_A(h, T). \]

Read it from the inside out.

- \(|p_t - y_t|\) is the learner’s loss on round \(t\). With predictions and labels in \(\{0, 1\}\), it is 1 for a mistake and 0 otherwise.
- \(\sum_{t=1}^T\) adds that loss over the rounds \(1, \dots, T\): the first sum is the learner’s mistake count, and the second, with \(h(x_t)\) in place of \(p_t\), is the mistake count of the fixed hypothesis \(h\) on the same messages.
- The bracket is the difference: how many more mistakes the learner made than \(h\).
- \(\sup\) over the sequence \((x_1, y_1), \dots, (x_T, y_T)\) takes the worst sequence of \(T\) rounds for the learner. No realizability this time: any labels at all.
- \(\sup_{h \in \mathcal H}\) then takes the hypothesis that makes the learner look worst, which is the best hypothesis in hindsight.

In plain words: **the regret is how many more mistakes the learner makes than the best hypothesis in the class, on the worst sequence of \(T\) rounds.** With the numbers above, \(130 - 100 = 30\) on that sequence; the regret is the largest such difference any sequence can produce. The goal is a regret that grows **sublinearly** in \(T\), meaning \(\mathrm{Regret}_A(\mathcal H, T)/T \to 0\) as \(T \to \infty\).

## The flipping adversary wins again

Is sublinear regret possible? Not for any learner that decides deterministically, even for a class of two hypotheses. Take the smallest interesting class: \(h_0\) says keep on every message and \(h_1\) says junk on every message. The adversary plays the opening trick again: it waits for the prediction and announces the opposite label. The learner makes \(T\) mistakes in \(T\) rounds.

And the best hypothesis in hindsight? Out of the \(T\) labels, either at least half are 0 or at least half are 1. Whichever label is in the majority, the constant hypothesis that always says it is wrong on at most \(T/2\) rounds. So the regret is at least \(T - T/2 = T/2\), which is linear in \(T\). This is **Cover’s impossibility result**.

[[diagram:cover]]

```python
def follow_the_majority(labels):
    return int(2 * sum(labels) > len(labels))

T, labels, mistakes = 1000, [], 0
for t in range(T):
    p = follow_the_majority(labels)
    y = 1 - p
    mistakes += p != y
    labels.append(y)
best = min(sum(labels), T - sum(labels))
print(mistakes, best, mistakes - best)
coin = sum(abs(0.5 - y) for y in labels)
print(coin, coin - best)
```

It prints `1000 500 500`: a learner that predicts the majority label so far makes 1,000 mistakes, the better constant hypothesis makes 500, and the regret is 500, half of \(T\). The second line is a hint of the way out. A learner that ignores everything and predicts junk with probability one half would expect to lose half a mistake per round, 500 here, the same as the best constant. Randomness is something the flipping adversary cannot read in advance. (The plain coin is not the answer, though: if every label were junk, the coin would still lose 500 while \(h_1\) lost nothing.)

The same thing happens with more hypotheses. Here is a deterministic learner with ten **experts** to choose from (we define experts properly below; for now think of ten hypotheses) that follows whichever one has the fewest mistakes so far. The adversary makes the one it picked wrong, every round:

```python
d, T = 10, 2000
totals, paid = [0] * d, 0
for t in range(T):
    pick = min(range(d), key=lambda i: totals[i])
    v = [int(i == pick) for i in range(d)]
    paid += v[pick]
    totals = [s + vi for s, vi in zip(totals, v)]
print(paid, min(totals), paid - min(totals))
```

It prints `2000 200 1800`: the learner pays on every one of the 2,000 rounds, the punishment is spread over all ten experts so the best of them pays only 200, and the regret is 1,800, which is \(0.9\,T\). Following the leader is the most natural deterministic strategy there is, and it is the one the adversary eats first.

## Let the learner flip a coin

So we give the learner a coin, and we change one rule of the game: the adversary must choose the label \(y_t\) without seeing the outcome of the learner’s coin on round \(t\). It may know the learner’s code, its probabilities, and every coin flip of earlier rounds. It just cannot see this round’s flip before it commits. For a spam filter that is realistic: the spammer sends the message before the filter decides.

Now the learner announces a number \(p_t \in [0, 1]\), the probability with which it will say junk, and then draws its actual label \(\hat y_t\), which is 1 with probability \(p_t\). How often is it wrong? If the true label is 1, it is wrong when it draws 0, with probability \(1 - p_t\). If the true label is 0, it is wrong when it draws 1, with probability \(p_t\). Both cases are the same formula:

\[ \mathbb P[\hat y_t \neq y_t] = |p_t - y_t|. \]

- \(\hat y_t\) (“y hat”) is the label the learner actually outputs after flipping its coin; \(p_t\) is now the coin’s bias, not a label.
- \(\mathbb P[\cdot]\) is the probability of the event in brackets, over the learner’s own coin only. The message and label are fixed when the coin is flipped.

In plain words: **a learner that says junk with probability \(p_t\) is wrong with probability equal to the distance between \(p_t\) and the true label.** With \(p_t = 0.3\): if the message is junk, it misses with probability \(0.7\); if it is keep, it raises a false alarm with probability \(0.3\).

```python
import random

rng, p, n = random.Random(4), 0.3, 100_000
for y in (1, 0):
    wrong = sum((rng.random() < p) != y for _ in range(n))
    print(y, round(wrong / n, 3), abs(p - y))
```

It prints `1 0.699 0.7` and `0 0.301 0.3`: the measured error rates over 100,000 coin flips land on \(|p - y|\). So the regret formula above does not change at all. We simply let \(p_t\) be any number in \([0, 1]\) and read \(\sum_t |p_t - y_t|\) as the learner’s **expected** number of mistakes, the expectation being over its own coins.

## Weighted-Majority

The learner that takes advantage of the coin is called **Weighted-Majority**, and it is easiest to describe in a slightly more general setting called **prediction with expert advice**. There are \(d\) **experts**: advisers whose advice the learner may follow. On each round, the learner chooses a probability distribution over the experts, a vector of weights \(\mathbf w^{(t)} = (w_1^{(t)}, \dots, w_d^{(t)})\) with every \(w_i^{(t)} \ge 0\) and \(\sum_i w_i^{(t)} = 1\), and follows expert \(i\) with probability \(w_i^{(t)}\). Then the adversary reveals a **cost** for every expert, a vector \(\mathbf v_t \in [0, 1]^d\), where \(v_{t,i}\) is what following expert \(i\) would have cost this round. The learner’s expected cost is the weighted average of the experts’ costs, \(\sum_i w_i^{(t)} v_{t,i} = \langle \mathbf w^{(t)}, \mathbf v_t \rangle\), using the inner product of lesson 9.

For the filter, the experts will be hypotheses and the cost of an expert will be 1 if it got the message wrong and 0 if it got it right. But the general version is cleaner to analyse, so let’s stay with it.

The idea of Weighted-Majority, in words: trust every expert equally at first, and every time an expert costs something, shrink its trust by a factor that depends on how much it cost. Experts that keep being wrong fade away exponentially fast, while experts that are mostly right keep their weight. Because the learner always spreads its bets, the adversary cannot single out one choice to punish.

```
Weighted-Majority(d, T):
  η ← sqrt(2 log(d) / T)
  w̃^(1) ← (1, ..., 1)
  for t = 1, 2, ..., T:
    Z_t ← sum of w̃^(t)_i over i
    w^(t) ← w̃^(t) / Z_t
    choose expert i with probability w^(t)_i
    receive costs v_t in [0, 1]^d
    pay ⟨w^(t), v_t⟩
    for every i: w̃^(t+1)_i ← w̃^(t)_i · exp(−η v_{t,i})
```

The update is the heart of it:

\[ \tilde w_i^{(t+1)} = \tilde w_i^{(t)} e^{-\eta v_{t,i}}, \qquad \mathbf w^{(t)} = \frac{\tilde{\mathbf w}^{(t)}}{Z_t}, \quad Z_t = \sum_{i=1}^d \tilde w_i^{(t)}. \]

- \(\tilde w_i^{(t)}\) (“w tilde”) is expert \(i\)’s raw trust before round \(t\); it starts at 1.
- \(e^{-\eta v_{t,i}}\) is the shrink factor: 1 when the cost is 0, and \(e^{-\eta} \approx 1 - \eta\) when the cost is 1. The number \(\eta > 0\) is the **step size**, as in lessons 14 and 15, and it sets how hard one bad round is punished.
- \(Z_t\) is the total raw trust, and dividing by it turns the raw trusts into weights that add up to 1: \(\mathbf w^{(t)}\) is a probability distribution.

In plain words: **multiply each expert’s trust by \(e^{-\eta}\) raised to its cost, then rescale so the trusts add up to one.** With two experts, \(\eta = 0.5\), and a first round where expert 1 costs 1 and expert 2 costs 0: the raw trusts become \(e^{-0.5} \approx 0.607\) and 1, so the weights become \(0.607/1.607 \approx 0.38\) and \(1/1.607 \approx 0.62\).

Let’s watch it against a hostile adversary. Ten experts, 2,000 rounds. On every round, the adversary looks at the weights and charges cost 1 to the five experts the learner currently trusts most, and cost 0 to the other five. A deterministic learner would be destroyed by this, as we saw. Weighted-Majority:

```python
from math import exp, log, sqrt

d, T = 10, 2000
eta = sqrt(2 * log(d) / T)
w_tilde, paid, totals = [1.0] * d, 0.0, [0.0] * d
for t in range(T):
    Z = sum(w_tilde)
    w = [wi / Z for wi in w_tilde]
    trusted = sorted(range(d), key=lambda i: -w[i])[:d // 2]
    v = [1.0 if i in trusted else 0.0 for i in range(d)]
    paid += sum(wi * vi for wi, vi in zip(w, v))
    totals = [s + vi for s, vi in zip(totals, v)]
    w_tilde = [wi * exp(-eta * vi) for wi, vi in zip(w_tilde, v)]
bound = sqrt(2 * log(d) * T)
print(round(paid, 1), min(totals), round(paid - min(totals), 1),
      round(bound, 1))
```

It prints `1012.0 1000.0 12.0 96.0`. The learner’s expected cost is 1,012, the best expert in hindsight paid 1,000, so the regret is 12, well under the 96 that the theorem below promises. Each line of the loop is a line of the pseudocode: `Z` is \(Z_t\), `w` is \(\mathbf w^{(t)}\), `paid` accumulates \(\langle \mathbf w^{(t)}, \mathbf v_t \rangle\), and the last line is the update.

[[diagram:weights]]

**Theorem (Weighted-Majority’s regret).** If \(T > 2 \log d\), Weighted-Majority with \(\eta = \sqrt{2 \log(d)/T}\) satisfies, for every sequence of cost vectors,

\[ \sum_{t=1}^T \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_{i \in [d]} \sum_{t=1}^T v_{t,i} \le \sqrt{2 \log(d) \, T}. \]

- The first sum is the learner’s total expected cost over \(T\) rounds.
- \(\sum_t v_{t,i}\) is expert \(i\)’s total cost, and \(\min_{i \in [d]}\) picks the cheapest expert in hindsight.
- \(\log\) is the natural logarithm, as everywhere in this course.

In plain words: **over any \(T\) rounds, Weighted-Majority pays at most \(\sqrt{2 \log(d)\, T}\) more than the best expert, whatever the adversary does.** With ten experts and 2,000 rounds, \(\sqrt{2 \cdot 2.303 \cdot 2000} \approx 96\), so at most about 4.8% of the rounds, and that fraction keeps shrinking like \(1/\sqrt T\).

*Proof.* The plan: watch the total raw trust \(Z_t\). We show it must fall quickly whenever the learner pays a lot (an upper bound), and it cannot fall below the trust of the best expert (a lower bound). Squeezed between the two, the learner’s cost cannot run far ahead of the best expert’s.

*Step 1: one round of shrinking.* Divide the new total by the old one:

\[ \log \frac{Z_{t+1}}{Z_t} = \log \sum_i \frac{\tilde w_i^{(t)}}{Z_t} e^{-\eta v_{t,i}} = \log \sum_i w_i^{(t)} e^{-\eta v_{t,i}}. \]

The first equality writes \(Z_{t+1} = \sum_i \tilde w_i^{(t)} e^{-\eta v_{t,i}}\), which is the update, and divides each term by \(Z_t\). The second recognizes \(\tilde w_i^{(t)}/Z_t\) as the weight \(w_i^{(t)}\). In words: **the log of how much the total trust shrank this round is the log of the weighted average of the shrink factors.** In the snippet above, on round 1 every weight is 0.1 and five experts shrink by \(e^{-0.048} \approx 0.953\), so the average shrink factor is \(0.5 + 0.5 \cdot 0.953 = 0.977\).

*Step 2: a quadratic bound on the exponential.* For \(0 \le a \le 1\), \(e^{-a} \le 1 - a + a^2/2\). Why: the difference \(g(a) = 1 - a + a^2/2 - e^{-a}\) has \(g(0) = 0\) and slope \(g'(a) = -1 + a + e^{-a}\), which is at least 0 because \(e^{-a} \ge 1 - a\) (the exponential lies above its tangent line at 0, the convexity of lesson 12). So \(g\) never decreases from 0. Apply it with \(a = \eta v_{t,i}\), which is in \([0, 1]\) because \(v_{t,i} \in [0, 1]\) and \(\eta < 1\) (that is what \(T > 2 \log d\) buys: \(\eta^2 = 2 \log(d)/T < 1\)). Since the weights add up to 1,

\[ \log \frac{Z_{t+1}}{Z_t} \le \log \sum_i w_i^{(t)} \left( 1 - \eta v_{t,i} + \frac{\eta^2 v_{t,i}^2}{2} \right) = \log(1 - b), \qquad b = \sum_i w_i^{(t)} \left( \eta v_{t,i} - \frac{\eta^2 v_{t,i}^2}{2} \right). \]

Here \(\log\) is increasing, so a bigger inside gives a bigger log; and \(\sum_i w_i^{(t)} \cdot 1 = 1\) pulled the 1 out front. The new name \(b\) is just the weighted average of \(\eta v - \eta^2 v^2/2\). In words: **the shrink per round is at most \(\log(1 - b)\), where \(b\) is roughly \(\eta\) times the learner’s cost.**

*Step 3: \(\log(1 - b) \le -b\).* This holds for every \(b < 1\), because \(1 - b \le e^{-b}\) (the same tangent-line fact) and \(\log\) is increasing. Our \(b\) is less than 1 since each term \(\eta v - \eta^2 v^2/2 \le \eta < 1\). So

\[ \log \frac{Z_{t+1}}{Z_t} \le -\eta \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{\eta^2}{2} \sum_i w_i^{(t)} v_{t,i}^2 \le -\eta \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{\eta^2}{2}. \]

The first inequality expands \(-b\) into its two parts, the first being \(\eta\) times the learner’s expected cost. The second uses \(v_{t,i}^2 \le 1\) and weights that add up to 1. In words: **every round, the total trust shrinks (in log terms) by at least \(\eta\) times what the learner paid, minus a small \(\eta^2/2\).**

```python
from math import exp, log

grid = [k / 1000 for k in range(1001)]
print(all(exp(-a) <= 1 - a + a * a / 2 for a in grid))
print(all(log(1 - b) <= -b for b in grid[:-1] + [-0.5, -3]))
print(round(1 - 1 + 1 / 2 - exp(-1), 3))
```

It prints `True`, `True` and `0.132`: both inequalities hold on a fine grid (including some negative \(b\)), and at \(a = 1\) the quadratic is 0.132 above \(e^{-1}\), so the bound is not tight but is never wrong.

*Step 4: add up the rounds.* Summing Step 3 over \(t = 1, \dots, T\), the left side telescopes: \(\log(Z_2/Z_1) + \log(Z_3/Z_2) + \dots = \log Z_{T+1} - \log Z_1\). And \(Z_1 = d\), since every raw trust starts at 1. So

\[ \log Z_{T+1} - \log d \le -\eta \sum_{t=1}^T \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{T \eta^2}{2}. \]

In words: **the total trust has fallen by at least \(\eta\) times the learner’s total cost, minus \(T\eta^2/2\).**

*Step 5: the best expert holds the total up.* Unrolling the update, expert \(i\)’s raw trust after \(T\) rounds is \(\tilde w_i^{(T+1)} = e^{-\eta \sum_t v_{t,i}}\). The total is at least its largest term, so

\[ \log Z_{T+1} = \log \sum_i e^{-\eta \sum_t v_{t,i}} \ge \log \max_i e^{-\eta \sum_t v_{t,i}} = -\eta \min_i \sum_t v_{t,i}. \]

The last step: the largest \(e^{-\eta c}\) belongs to the smallest cost \(c\), and \(\log e^{-\eta c} = -\eta c\). In words: **the total trust is at least the trust of the best expert, which has only shrunk by its own cost.**

*Step 6: squeeze.* Steps 4 and 5 bound \(\log Z_{T+1}\) from both sides:

\[ -\eta \min_i \sum_t v_{t,i} - \log d \le -\eta \sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle + \frac{T \eta^2}{2}. \]

Move the learner’s cost to the left and divide by \(\eta > 0\):

\[ \sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_i \sum_t v_{t,i} \le \frac{\log d}{\eta} + \frac{\eta T}{2}. \]

This holds for *every* \(\eta \in (0, 1)\), and it is the form we will use in the lab. Read it as two forces: a small \(\eta\) learns slowly (the first term, the cost of starting with no idea which expert is good), and a large \(\eta\) overreacts (the second term). With \(\eta = \sqrt{2 \log(d)/T}\), the first term is \(\log d \cdot \sqrt{T/(2 \log d)} = \sqrt{T \log(d)/2}\) and the second is \(\frac T2 \sqrt{2 \log(d)/T} = \sqrt{T \log(d)/2}\), and together they make \(2\sqrt{T \log(d)/2} = \sqrt{2 \log(d)\, T}\). \(\blacksquare\)

Here are Steps 3 to 5 checked on the same hostile run as before:

```python
from math import exp, log, sqrt

d, T = 10, 2000
eta = sqrt(2 * log(d) / T)
w_t, ok, paid, totals = [1.0] * d, True, 0.0, [0.0] * d
for t in range(T):
    Z = sum(w_t)
    w = [wi / Z for wi in w_t]
    trusted = sorted(range(d), key=lambda i: -w[i])[:d // 2]
    v = [1.0 if i in trusted else 0.0 for i in range(d)]
    wv = sum(wi * vi for wi, vi in zip(w, v))
    w_t = [wi * exp(-eta * vi) for wi, vi in zip(w_t, v)]
    ok &= log(sum(w_t) / Z) <= -eta * wv + eta ** 2 / 2
    paid += wv
    totals = [s + vi for s, vi in zip(totals, v)]
log_Z = log(sum(w_t))
print(ok, round(log_Z - log(d), 2),
      round(-eta * paid + T * eta ** 2 / 2, 2))
print(round(log_Z, 2), round(-eta * min(totals), 2))
```

It prints `True -47.99 -46.26` and `-45.68 -47.99`. Step 3 held on every round (`True`); the telescoped total \(\log Z_{T+1} - \log d = -47.99\) sits below Step 4’s right side, \(-46.26\); and \(\log Z_{T+1} = -45.68\) sits above Step 5’s floor, \(-47.99\).

And the choice of \(\eta\) really does minimize the two-force bound:

```python
from math import log, sqrt

d, T = 10, 2000
bound = lambda eta: log(d) / eta + eta * T / 2
etas = [k / 10_000 for k in range(1, 10_000)]
best = min(etas, key=bound)
print(best, round(bound(best), 2))
eta_star = sqrt(2 * log(d) / T)
print(round(eta_star, 4), round(sqrt(2 * log(d) * T), 2))
```

It prints `0.048 95.97` twice: a grid search over \(\eta\) finds the same minimum as the formula, at \(\eta \approx 0.048\), with value about 96.

What if \(T \le 2 \log d\)? Then \(\sqrt{2 \log(d)\, T} \ge T\), and the bound is true for free: no learner can pay more than \(T\) above anyone, since each round costs between 0 and 1. The condition only matters for the proof, not for the truth of the statement.

## From experts to hypotheses

Now back to classification. Let \(\mathcal H = \{h_1, \dots, h_d\}\) be finite, and make each hypothesis an expert: expert \(i\) advises \(h_i(x_t)\), and its cost is \(v_{t,i} = |h_i(x_t) - y_t|\), 1 if it is wrong and 0 if it is right. The learner predicts the weighted vote, \(p_t = \sum_i w_i^{(t)} h_i(x_t)\), as its probability of saying junk. For example, with weights 0.5, 0.3, 0.2 on three hypotheses that say junk, keep, junk, the learner says junk with probability \(0.5 + 0.2 = 0.7\).

Is the learner’s expected mistake \(|p_t - y_t|\) the same as Weighted-Majority’s cost \(\langle \mathbf w^{(t)}, \mathbf v_t \rangle\)? It is, and the reason is a sign argument:

\[ |p_t - y_t| = \Big| \sum_i w_i^{(t)} \big( h_i(x_t) - y_t \big) \Big| = \sum_i w_i^{(t)} \big| h_i(x_t) - y_t \big| = \langle \mathbf w^{(t)}, \mathbf v_t \rangle. \]

The first equality uses \(\sum_i w_i^{(t)} = 1\) to write \(y_t = \sum_i w_i^{(t)} y_t\). For the second: if \(y_t = 1\), every \(h_i(x_t) - y_t\) is 0 or \(-1\), all of one sign; if \(y_t = 0\), every one is 0 or \(+1\). The absolute value of a sum of terms of one sign is the sum of their absolute values. In words: **because all the hypotheses err in the same direction, the vote’s distance from the label is exactly the weighted fraction of hypotheses that are wrong.** In the example, if the message is keep, \(|0.7 - 0| = 0.7\), and the hypotheses that are wrong (the two saying junk) carry weight \(0.5 + 0.2 = 0.7\).

```python
import random

rng = random.Random(3)
for trial in range(10_000):
    d = rng.randint(1, 6)
    w = [rng.random() for _ in range(d)]
    w = [wi / sum(w) for wi in w]
    h = [rng.randint(0, 1) for _ in range(d)]
    y = rng.randint(0, 1)
    p = sum(wi * hi for wi, hi in zip(w, h))
    rhs = sum(wi * abs(hi - y) for wi, hi in zip(w, h))
    assert abs(abs(p - y) - rhs) < 1e-12
print("identity holds in 10,000 random checks")
```

It prints `identity holds in 10,000 random checks`. And \(\sum_t v_{t,i}\) is just the number of mistakes of \(h_i\). So Weighted-Majority’s theorem becomes, directly:

**Corollary (finite classes).** For a finite \(\mathcal H\) and \(T > 2 \log |\mathcal H|\), the weighted vote learner has, on every sequence,

\[ \sum_{t=1}^T |p_t - y_t| - \min_{h \in \mathcal H} \sum_{t=1}^T |h(x_t) - y_t| \le \sqrt{2 \log(|\mathcal H|) \, T}. \]

In plain words: **with a finite class, the randomized filter’s expected mistakes exceed those of the best hypothesis in hindsight by at most \(\sqrt{2 \log(|\mathcal H|)\, T}\).** For lesson 1’s ten one-feature hypotheses over 2,000 messages, that is the 96 we computed: under 5% extra mistakes per round, against the worst spammers.

What about an infinite class? The rogue-sender class from earlier has one hypothesis per possible sender address, and there are infinitely many addresses, so \(\log |\mathcal H| = \infty\). But its Littlestone dimension is 1. Can we build a *finite* set of experts that, on whatever messages arrive, includes one that behaves exactly like the best hypothesis?

Here is the idea in words. SOA, run on messages labelled by some \(h\), makes at most \(L = \mathrm{Ldim}(\mathcal H)\) mistakes. If you knew *on which rounds* it would make them, you could reproduce \(h\) exactly: follow SOA’s guess, except flip it on those rounds. You don’t know the rounds, so create one expert for every possible set of at most \(L\) rounds. One of them has the right set.

```
Expert(i_1, ..., i_L), for rounds i_1 < ... < i_L:
  V_1 ← H
  for t = 1, 2, ..., T:
    receive x_t
    for r in {0, 1}: V_t^(r) ← {h in V_t : h(x_t) = r}
    guess ← argmax over r of Ldim(V_t^(r))   (ties: 0)
    if t is one of i_1, ..., i_L: predict 1 − guess
    else: predict guess
    V_{t+1} ← V_t^(prediction)
```

Each expert only needs the messages so far, never the labels, so it can advise in real time.

**Lemma (some expert copies each hypothesis).** Let \(\mathrm{Ldim}(\mathcal H) < \infty\) and fix any messages \(x_1, \dots, x_T\). For every \(h \in \mathcal H\) there are \(L \le \mathrm{Ldim}(\mathcal H)\) and rounds \(i_1 < \dots < i_L\) such that \(\mathrm{Expert}(i_1, \dots, i_L)\) predicts \(h(x_t)\) on every round \(t\).

*Proof.* Run SOA (with ties going to 0, as in the expert) on the sequence \((x_1, h(x_1)), \dots, (x_T, h(x_T))\). It is realizable, labelled by \(h\), so SOA makes at most \(\mathrm{Ldim}(\mathcal H)\) mistakes; let \(L\) be their number and \(i_1, \dots, i_L\) the rounds where they happen. Now compare SOA with \(\mathrm{Expert}(i_1, \dots, i_L)\), round by round. Suppose both have the same version space at the start of round \(t\) (true at \(t = 1\)). They then compute the same guess. On a round in the list, SOA’s guess was wrong, so \(h(x_t) = 1 - \text{guess}\), which is what the expert predicts; on any other round, SOA’s guess was right, so \(h(x_t) = \text{guess}\), which again is what the expert predicts. Either way the expert predicts \(h(x_t)\), and both then keep \(V_t^{(h(x_t))}\), so the version spaces agree at round \(t + 1\) too. By induction, the expert predicts \(h(x_t)\) on every round. \(\blacksquare\)

In plain words: **for any hypothesis, one of the experts is a perfect copy of it on the messages that actually arrived.** Here is the construction on \(\mathcal H_1\), for six messages: for each of the four hypotheses, find SOA’s mistake rounds, then check that the expert with those rounds reproduces the hypothesis.

```python
from functools import lru_cache
from itertools import product

X = ["".join(b) for b in product("01", repeat=4)]
H = frozenset(tuple(int(x[k]) for x in X) for k in range(4))
part = lambda V, i, r: frozenset(h for h in V if h[i] == r)

@lru_cache(None)
def ldim(V):
    splits = [(part(V, i, 0), part(V, i, 1)) for i in range(16)]
    return -1 if not V else max([1 + min(map(ldim, s))
                                 for s in splits if all(s)],
                                default=0)

def run(xs, flips=(), labels=None):
    V, out = H, []
    for t, x in enumerate(xs, 1):
        i = X.index(x)
        guess = int(ldim(part(V, i, 1)) > ldim(part(V, i, 0)))
        flip = t in flips
        out.append(labels[t - 1] if labels else guess ^ flip)
        flips = flips if out[-1] == guess else (*flips, t)
        V = part(V, i, out[-1])
    return out, flips

xs = ["1100", "0011", "1010", "0110", "1111", "0001"]
for h in sorted(H):
    labels = [h[X.index(x)] for x in xs]
    flips = run(xs, labels=labels)[1]
    print(flips, run(xs, flips)[0] == labels)
```

It prints `(1, 3) True`, `(1,) True`, `(3,) True` and `() True`: the four hypotheses need flips at rounds {1, 3}, {1}, {3} and none, never more than \(\mathrm{Ldim}(\mathcal H_1) = 2\) flips, and each expert copies its hypothesis exactly. The function `run` is SOA when given `labels` (it records where its guess was wrong) and the expert when given `flips`.

How many experts is that? One for each set of at most \(L = \mathrm{Ldim}(\mathcal H)\) rounds out of \(T\):

\[ d = \sum_{k=0}^{L} \binom{T}{k} \le \left( \frac{eT}{L} \right)^L \quad (1 \le L \le T). \]

- \(\binom{T}{k}\) (“\(T\) choose \(k\)”) is the number of ways to pick \(k\) rounds out of \(T\).
- The sum counts all the sets of size \(0, 1, \dots, L\).
- The inequality is the polynomial bound from Sauer’s lemma in lesson 6, with \(T\) in place of \(m\) and \(L\) in place of \(d\).

In plain words: **the number of experts grows only polynomially in \(T\), like \(T^L\), not exponentially.** For the rogue sender, \(L = 1\), and over \(T = 10{,}000\) rounds there are \(1 + 10{,}000 = 10{,}001\) experts, below \(e \cdot 10{,}000 \approx 27{,}183\). So \(\log d \le L \log(eT/L) \le L \log(eT)\), and Weighted-Majority over these experts, plus the lemma applied to the best hypothesis, gives the regret bound \(\sqrt{2 L \log(eT)\, T}\).

```python
from math import comb, e, log, sqrt

T, L = 10_000, 1
experts = sum(comb(T, k) for k in range(L + 1))
print(experts, round((e * T / L) ** L))
regret = sqrt(2 * log(experts) * T)
print(round(regret, 1), round(regret / T, 4))
print(round(sqrt(2 * L * log(e * T) * T), 1))
```

It prints `10001 27183`, then `429.2 0.0429` (the regret bound with the exact expert count, about 4.3% of the rounds), then `451.9` (the slightly looser \(\sqrt{2L\log(eT)\,T}\)). An infinite class, an adversary, and still a guarantee.

**Theorem (online learnability with regret).** For every class \(\mathcal H\) there is an online learner whose predictions are probabilities \(p_t \in [0, 1]\) and that satisfies, for every \(h \in \mathcal H\) and every sequence,

\[ \sum_{t=1}^T |p_t - y_t| - \sum_{t=1}^T |h(x_t) - y_t| \le \sqrt{2 \min\{\log |\mathcal H|, \; \mathrm{Ldim}(\mathcal H) \log(eT)\} \, T}. \]

Moreover, no learner can guarantee an expected regret smaller than a constant times \(\sqrt{\mathrm{Ldim}(\mathcal H)\, T}\).

In plain words: **with a coin, every class of finite Littlestone dimension can be learned online against any sequence, with regret growing like \(\sqrt T\); and that is the best possible rate.** The upper bound is the two constructions above: the finite one, and the Littlestone experts (when \(\mathrm{Ldim} = 0\), the class has a single behaviour and the bound is 0). We prove the upper bound only; the matching lower bound is a research result we state without proof. So the Littlestone dimension decides learnability online in both settings, realizable and not, just as the VC dimension did for the batch world in lesson 6.

## Online convex optimization

Weighted-Majority needs a list of experts, which for big classes is a big list. In lessons 12 and 14 you met a different way to make learning tractable: convex losses, minimized by stepping against a subgradient. Does that survive an adversary? Yes, almost word for word.

The setting first, in words. The learner holds a weight vector. On each round, it commits to its vector, the adversary reveals an example, and the learner pays the loss of its vector on that example. The loss must be convex in the vector, as in lesson 12, and the set of allowed vectors must be convex. The adversary picks the examples, adapting to everything the learner did.

**Online convex optimization.** A hypothesis class \(\mathcal H \subseteq \mathbb R^d\) that is a convex set, a domain \(Z\) of examples, and a loss \(\ell : \mathcal H \times Z \to \mathbb R\) with \(\ell(\cdot, z)\) convex for every \(z\). For \(t = 1, \dots, T\): the learner announces \(\mathbf w^{(t)} \in \mathcal H\), the environment answers with \(z_t \in Z\), and the learner pays \(\ell(\mathbf w^{(t)}, z_t)\).

Regret is measured against any fixed vector \(\mathbf w^\star\):

\[ \mathrm{Regret}_A(\mathbf w^\star, T) = \sum_{t=1}^T \ell(\mathbf w^{(t)}, z_t) - \sum_{t=1}^T \ell(\mathbf w^\star, z_t), \qquad \mathrm{Regret}_A(\mathcal H, T) = \sup_{\mathbf w^\star \in \mathcal H} \mathrm{Regret}_A(\mathbf w^\star, T). \]

- The first sum is the learner’s total loss with its changing vectors; the second is the total loss of the single vector \(\mathbf w^\star\), held fixed, on the same examples.
- \(\sup_{\mathbf w^\star \in \mathcal H}\) picks the comparison vector that makes the learner look worst: the best fixed vector in hindsight.

In plain words: **the regret is how much more loss the learner paid than the best single weight vector it could have used throughout.** For example, if over 100 rounds the learner pays 60 and the best fixed vector would have paid 52, the regret against it is 8.

The learner is **Online Gradient Descent**: lesson 14’s gradient descent with projection, run on the losses as they arrive.

```
OnlineGradientDescent(η):
  w^(1) ← 0
  for t = 1, 2, ..., T:
    predict w^(t)
    receive z_t; let f_t(w) = ℓ(w, z_t)
    choose a subgradient v_t of f_t at w^(t)
    w^(t+½) ← w^(t) − η v_t
    w^(t+1) ← the point of H closest to w^(t+½)
```

The half step moves against the subgradient, and the projection pulls the vector back into \(\mathcal H\) if it stepped outside. Recall from lesson 14 that a **subgradient** \(\mathbf v\) of a convex \(f\) at \(\mathbf w\) is the slope of a supporting plane: \(f(\mathbf u) \ge f(\mathbf w) + \langle \mathbf u - \mathbf w, \mathbf v \rangle\) for every \(\mathbf u\); that \(f\) is **\(\rho\)-Lipschitz** exactly when every subgradient has norm at most \(\rho\); and the **projection lemma**: projecting onto a convex set never increases the distance to any point of the set.

[[diagram:ogd]]

**Theorem (Online Gradient Descent’s regret).** For every \(\mathbf w^\star \in \mathcal H\),

\[ \mathrm{Regret}_A(\mathbf w^\star, T) \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

If every \(f_t\) is \(\rho\)-Lipschitz and \(\eta = 1/\sqrt T\), then \(\mathrm{Regret}_A(\mathbf w^\star, T) \le \frac12 (\|\mathbf w^\star\|^2 + \rho^2) \sqrt T\). If moreover every vector in \(\mathcal H\) has norm at most \(B\) and \(\eta = B/(\rho \sqrt T)\), then \(\mathrm{Regret}_A(\mathcal H, T) \le B \rho \sqrt T\).

- \(\|\mathbf w^\star\|\) is the length of the comparison vector: how far it is from the starting point \(\mathbf 0\).
- \(\|\mathbf v_t\|^2\) is the squared length of the step direction on round \(t\).
- The two terms are lesson 14’s two forces again: a timid step size pays for the distance to cover, and a bold one pays for the size of its steps.

In plain words: **against any sequence of convex losses, Online Gradient Descent pays at most \(B\rho\sqrt T\) more than the best fixed vector, the same bound gradient descent had in lesson 14, now with no distribution at all.** With \(B = 1\), \(\rho = 1\) and \(T = 10{,}000\) rounds, the regret is at most 100, one hundredth of a loss per round.

*Proof.* Fix \(\mathbf w^\star \in \mathcal H\). Track the squared distance to it, \(\|\mathbf w^{(t)} - \mathbf w^\star\|^2\), and see how much one round changes it. Add and subtract the half step:

\[ \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 = \Big( \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 \Big) + \Big( \|\mathbf w^{(t+\frac12)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 \Big). \]

The first bracket is at most 0 by the projection lemma, since \(\mathbf w^\star \in \mathcal H\). For the second, \(\mathbf w^{(t+\frac12)} = \mathbf w^{(t)} - \eta \mathbf v_t\), and expanding \(\|\mathbf a - \eta\mathbf v\|^2 = \|\mathbf a\|^2 - 2\eta\langle \mathbf a, \mathbf v \rangle + \eta^2\|\mathbf v\|^2\) with \(\mathbf a = \mathbf w^{(t)} - \mathbf w^\star\) gives \(-2\eta \langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle + \eta^2 \|\mathbf v_t\|^2\). Finally, the subgradient inequality at \(\mathbf w^{(t)}\) with \(\mathbf u = \mathbf w^\star\) reads \(f_t(\mathbf w^\star) \ge f_t(\mathbf w^{(t)}) + \langle \mathbf w^\star - \mathbf w^{(t)}, \mathbf v_t \rangle\), that is, \(\langle \mathbf w^{(t)} - \mathbf w^\star, \mathbf v_t \rangle \ge f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star)\). Together:

\[ \|\mathbf w^{(t+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(t)} - \mathbf w^\star\|^2 \le -2\eta \big( f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \big) + \eta^2 \|\mathbf v_t\|^2. \]

In words: **every round, the distance to the comparison vector shrinks by at least \(2\eta\) times the regret of that round, up to a penalty of \(\eta^2\|\mathbf v_t\|^2\).**

Sum over \(t = 1, \dots, T\). The left side telescopes to \(\|\mathbf w^{(T+1)} - \mathbf w^\star\|^2 - \|\mathbf w^{(1)} - \mathbf w^\star\|^2\). With \(\mathbf w^{(1)} = \mathbf 0\), moving terms across and dividing by \(2\eta\):

\[ \sum_{t=1}^T \big( f_t(\mathbf w^{(t)}) - f_t(\mathbf w^\star) \big) \le \frac{\|\mathbf w^\star\|^2 - \|\mathbf w^{(T+1)} - \mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2 \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

The left side is the regret, since \(f_t(\mathbf w) = \ell(\mathbf w, z_t)\), and dropping the subtracted squared distance only makes the right side bigger. That is the first bound. If each \(f_t\) is \(\rho\)-Lipschitz, then \(\|\mathbf v_t\| \le \rho\), so the sum is at most \(T\rho^2\): with \(\eta = 1/\sqrt T\) the bound is \(\frac{\sqrt T}{2}\|\mathbf w^\star\|^2 + \frac{\sqrt T}{2}\rho^2\), and with \(\|\mathbf w^\star\| \le B\) and \(\eta = B/(\rho\sqrt T)\) it is \(\frac{B\rho\sqrt T}{2} + \frac{B\rho\sqrt T}{2} = B\rho\sqrt T\). \(\blacksquare\)

The proof is lesson 14’s step lemma with one thing missing: there is no expectation anywhere. Lesson 14 needed fresh random examples to turn the step lemma into a statement about the true risk. Here the step lemma itself is the result, and it holds for every sequence, because it never cared where the \(\mathbf v_t\) came from.

Let’s check it on a one-dimensional fight. The class is \(\mathcal H = [-1, 1]\), the loss is \(\ell(w, z) = |w - z|\) (convex and 1-Lipschitz in \(w\)), and the adversary always picks \(z_t\) at the end of the interval farthest from the learner’s current \(w\).

```python
from math import sqrt

B, rho, T = 1.0, 1.0, 10_000
eta = B / (rho * sqrt(T))
w, paid, zs = 0.0, 0.0, []
for t in range(T):
    z = -1.0 if w > 0 else 1.0
    paid += abs(w - z)
    v = 1.0 if w > z else -1.0
    w = max(-B, min(B, w - eta * v))
    zs.append(z)
grid = [k / 100 - 1 for k in range(201)]
best = min(sum(abs(u - z) for z in zs) for u in grid)
print(round(paid, 1), round(best, 1), round(paid - best, 1),
      B * rho * sqrt(T))
```

It prints `10050.0 10000.0 50.0 100.0`: the learner pays 10,050, the best fixed point in hindsight pays 10,000, and the regret of 50 is inside the promised \(B\rho\sqrt T = 100\). The line `w = max(-B, min(B, w - eta * v))` is the half step and the projection onto \([-1, 1]\) in one go. And the per-round inequality of the proof, checked against \(w^\star = 0.5\):

```python
from math import sqrt

B, T, w_star = 1.0, 10_000, 0.5
eta = B / sqrt(T)
w, ok, total = 0.0, True, 0.0
for t in range(T):
    z = -1.0 if w > 0 else 1.0
    v = 1.0 if w > z else -1.0
    w_next = max(-B, min(B, w - eta * v))
    drop = (w_next - w_star) ** 2 - (w - w_star) ** 2
    gap = abs(w - z) - abs(w_star - z)
    ok &= drop <= -2 * eta * gap + eta ** 2 * v * v + 1e-12
    total += gap
    w = w_next
print(ok, round(total, 1), round(w_star ** 2 / (2 * eta)
                                  + eta * T / 2, 1))
```

It prints `True 50.0 62.5`: the per-round inequality held on all 10,000 rounds, and the regret against \(w^\star = 0.5\), 50, sits under the first bound \(\|w^\star\|^2/(2\eta) + \eta T\rho^2/2 = 12.5 + 50 = 62.5\).

## The online Perceptron

The filter of lessons 9 and 15 was a halfspace: a weight vector \(\mathbf w\) over word counts, predicting junk when the score \(\langle \mathbf w, \mathbf x \rangle\) is positive. For halfspaces we switch labels to \(y \in \{-1, +1\}\), as lesson 9 did, with \(+1\) for junk. Can an online filter learn halfspaces against an adversary?

Not with a mistake bound in general. In two or more dimensions, halfspaces contain the thresholds in disguise: take messages \((s, 1, 0, \dots, 0)\) whose first coordinate is a score \(s\), and weight vectors \((-1, a, 0, \dots, 0)\); then \(\langle \mathbf w, \mathbf x \rangle = a - s\), which is positive when \(s < a\), a threshold on \(s\). Binary search on \(s\) shatters trees of every depth, so the Littlestone dimension of halfspaces is infinite, and by the tree lower bound every online learner can be forced into a mistake on every round.

So we need to assume more about the sequence, and lessons 9 and 15 tell us what: a **margin**. If some \(\mathbf w^\star\) scores every message with \(y_t \langle \mathbf w^\star, \mathbf x_t \rangle \ge 1\), the messages sit at distance at least \(1/\|\mathbf w^\star\|\) from its boundary, and an adversary cannot binary-search ever closer to it.

The tool is the one lesson 12 introduced for exactly this purpose, a **convex surrogate loss**. The 0-1 loss, \(\mathbb 1[y\langle \mathbf w, \mathbf x \rangle \le 0]\) (1 on a mistake, 0 otherwise; \(\mathbb 1[\cdot]\) is 1 when the statement inside is true), is not convex. The **hinge loss** \(\max\{0, 1 - y\langle \mathbf w, \mathbf x \rangle\}\) is convex and is never below the 0-1 loss. Online, we may even pick a different surrogate on each round, because the analysis is worst case and never averages over rounds. So we use

\[ f_t(\mathbf w) = \begin{cases} \max\{0, 1 - y_t \langle \mathbf w, \mathbf x_t \rangle\} & \text{if } y_t \langle \mathbf w^{(t)}, \mathbf x_t \rangle \le 0, \\ 0 & \text{otherwise.} \end{cases} \]

- \(y_t \langle \mathbf w, \mathbf x_t \rangle\) is the **signed score** of lesson 9: positive when \(\mathbf w\) labels the message correctly.
- The case split looks at the learner’s *current* vector \(\mathbf w^{(t)}\): on a round where it errs (signed score at most 0), the surrogate is the hinge loss; on a round where it is right, the surrogate is the zero function.

In plain words: **charge the hinge loss on the rounds where the filter got the message wrong, and nothing on the rounds where it got it right.** Both cases are convex. And on a mistake round, \(f_t(\mathbf w^{(t)}) \ge 1 - y_t\langle \mathbf w^{(t)}, \mathbf x_t \rangle \ge 1\), while on other rounds \(f_t(\mathbf w^{(t)}) = 0\), so \(f_t(\mathbf w^{(t)})\) is always at least the 0-1 loss of the learner. For example, if \(\mathbf w^{(t)}\) gives a junk message the score \(-0.4\), the round is a mistake and \(f_t(\mathbf w^{(t)}) = 1.4 \ge 1\).

Now run Online Gradient Descent on \(f_1, f_2, \dots\) with \(\mathcal H = \mathbb R^d\) (no projection needed). On a correct round, \(f_t\) is zero and \(\mathbf v_t = \mathbf 0\). On a mistake round, the hinge part \(1 - y_t\langle \mathbf w, \mathbf x_t \rangle\) is at least 1 at \(\mathbf w^{(t)}\), so the max is taken by the linear piece, whose gradient is \(-y_t \mathbf x_t\). So the update is

\[ \mathbf w^{(t+1)} = \begin{cases} \mathbf w^{(t)} + \eta\, y_t \mathbf x_t & \text{if } y_t \langle \mathbf w^{(t)}, \mathbf x_t \rangle \le 0, \\ \mathbf w^{(t)} & \text{otherwise,} \end{cases} \]

which reads: **on a mistake, add \(\eta\) times the message, signed by its label; otherwise leave the weights alone.** That is lesson 9’s Perceptron update, with a step size. Does the step size matter? Write \(\mathcal M\) for the set of rounds with an update. Starting from \(\mathbf 0\), the weights are a sum of past updates, so the prediction on round \(t\) is

\[ \operatorname{sign}\langle \mathbf w^{(t)}, \mathbf x_t \rangle = \operatorname{sign}\Big( \eta \sum_{i \in \mathcal M,\, i < t} y_i \langle \mathbf x_i, \mathbf x_t \rangle \Big). \]

- \(\sum_{i \in \mathcal M, i < t}\) runs over the earlier rounds on which the learner updated.
- Each term is the past message’s similarity to the new one, \(\langle \mathbf x_i, \mathbf x_t \rangle\), signed by its label.
- \(\operatorname{sign}\) returns \(+1\) for a positive number and \(-1\) otherwise.

In plain words: **the prediction is a vote of the past mistakes, each weighted by how similar it is to the new message, and the step size \(\eta > 0\) multiplies every vote equally, so it never changes the outcome.** So the set of mistake rounds is the same for every \(\eta > 0\), and we may as well take \(\eta = 1\). That is the online **Perceptron**.

```python
import random

dot = lambda a, b: sum(ai * bi for ai, bi in zip(a, b))
sign = lambda s: 1 if s > 0 else -1
for eta in (1.0, 0.01):
    rng, w, M, signs = random.Random(8), [0.0] * 3, [], []
    for t in range(200):
        x = [rng.uniform(-1, 1) for _ in range(3)]
        y = sign(x[0] - x[1] + 0.3 * x[2])
        dual = eta * sum(yi * dot(xi, x) for xi, yi in M)
        assert abs(dot(w, x) - dual) < 1e-9
        signs.append(sign(dot(w, x)))
        if y * dot(w, x) <= 0:
            M.append((x, y))
            w = [wi + eta * y * xi for wi, xi in zip(w, x)]
    print(eta, len(M), sum(signs))
```

It prints `1.0 19 -2` and `0.01 19 -2`: on the same 200 messages, \(\eta = 1\) and \(\eta = 0.01\) update on the same 19 rounds and make identical predictions, and the `assert` confirms on every round that the score equals the vote over past mistakes.

**Theorem (the online Perceptron’s mistake bound).** Run the Perceptron on any sequence \((\mathbf x_1, y_1), \dots, (\mathbf x_T, y_T)\), let \(R = \max_t \|\mathbf x_t\|\), and let \(\mathcal M\) be the rounds on which it updates. Then for every \(\mathbf w^\star\),

\[ |\mathcal M| \le \sum_{t \in \mathcal M} [1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+ + R\|\mathbf w^\star\| \sqrt{\sum_{t \in \mathcal M} [1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+} + R^2\|\mathbf w^\star\|^2. \]

In particular, if \(y_t \langle \mathbf w^\star, \mathbf x_t \rangle \ge 1\) for every \(t\), then \(|\mathcal M| \le R^2 \|\mathbf w^\star\|^2\).

- \([a]_+ = \max\{0, a\}\), so \([1 - y_t\langle \mathbf w^\star, \mathbf x_t \rangle]_+\) is the hinge loss of the comparison vector on round \(t\): 0 when \(\mathbf w^\star\) scores the message with margin, positive when it falls short.
- The sum over \(t \in \mathcal M\) adds that shortfall over the Perceptron’s mistake rounds; call it \(C\).
- \(R\|\mathbf w^\star\|\) is lesson 9’s \(RB\): message size times comparison size.

In plain words: **the Perceptron’s mistakes are at most the comparison vector’s total hinge shortfall, plus a correction, plus \((R\|\mathbf w^\star\|)^2\); and if the comparison vector separates every message with margin, the mistakes never exceed \((R\|\mathbf w^\star\|)^2\), however the adversary orders them.** With \(R = 1\), \(\|\mathbf w^\star\| = 4\) and a shortfall of \(C = 20\): at most \(20 + 4\sqrt{20} + 16 \approx 53.9\) mistakes.

*Proof.* Apply Online Gradient Descent’s first bound to the surrogates \(f_t\), with \(\mathbf v_t = -y_t\mathbf x_t\) on update rounds and \(\mathbf 0\) otherwise:

\[ \sum_{t=1}^T f_t(\mathbf w^{(t)}) - \sum_{t=1}^T f_t(\mathbf w^\star) \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|\mathbf v_t\|^2. \]

Three substitutions. First, \(f_t(\mathbf w^{(t)}) \ge 1\) on each update round and 0 elsewhere, so the first sum is at least \(|\mathcal M|\). Second, \(f_t(\mathbf w^\star)\) is the hinge shortfall on update rounds and 0 elsewhere, so the second sum is \(C\). Third, \(\|\mathbf v_t\|^2 = \|\mathbf x_t\|^2 \le R^2\) on update rounds and 0 elsewhere, so the last sum is at most \(|\mathcal M| R^2\). So, for every \(\eta > 0\),

\[ |\mathcal M| - C \le \frac{\|\mathbf w^\star\|^2}{2\eta} + \frac{\eta}{2} |\mathcal M| R^2. \]

The run, and therefore \(|\mathcal M|\), is the same for every \(\eta\), so we may choose \(\eta\) after the fact (if \(|\mathcal M| = 0\) there is nothing to prove). The choice \(\eta = \|\mathbf w^\star\|/(R\sqrt{|\mathcal M|})\) makes both terms equal to \(\frac12 R\|\mathbf w^\star\|\sqrt{|\mathcal M|}\), so

\[ |\mathcal M| - R\|\mathbf w^\star\| \sqrt{|\mathcal M|} - C \le 0. \]

In words: **the mistake count, minus a square-root correction, minus the shortfall, is not positive.**

It remains to solve this for \(|\mathcal M|\). Write \(x = |\mathcal M|\), \(b = R\|\mathbf w^\star\|\), \(c = C\), all nonnegative, so \(x - b\sqrt x - c \le 0\). Put \(y = \sqrt x\): then \(y^2 - by - c \le 0\), and a parabola that opens upward is at most 0 only between its roots, so \(y \le \frac{b + \sqrt{b^2 + 4c}}{2}\). Squaring, \(x \le \frac{2b^2 + 4c + 2b\sqrt{b^2 + 4c}}{4} = \frac{b^2}{2} + c + \frac b2 \sqrt{b^2 + 4c}\). Finally \(\sqrt{b^2 + 4c} \le b + 2\sqrt c\) (square both sides: \(b^2 + 4c \le b^2 + 4b\sqrt c + 4c\)), so \(x \le \frac{b^2}{2} + c + \frac{b^2}{2} + b\sqrt c = c + b^2 + b\sqrt c\). That is the theorem. If \(\mathbf w^\star\) has margin on every message, \(c = 0\) and \(x \le b^2 = R^2\|\mathbf w^\star\|^2\). \(\blacksquare\)

```python
import random
from math import sqrt

rng = random.Random(5)
for trial in range(10_000):
    b, c = rng.uniform(0, 10), rng.uniform(0, 10)
    root = ((b + sqrt(b * b + 4 * c)) / 2) ** 2
    assert abs(root - b * sqrt(root) - c) < 1e-9
    assert root <= c + b * b + b * sqrt(c) + 1e-9
print("x <= c + b^2 + b*sqrt(c) in 10,000 random checks")
```

It prints `x <= c + b^2 + b*sqrt(c) in 10,000 random checks`: for random \(b\) and \(c\), the largest \(x\) with \(x - b\sqrt x - c \le 0\) (the square of the parabola’s right root) is below the claimed bound.

Now the whole chain on a run where the labels come from \(\mathbf u = (0.6, 0.8)\), the messages are drawn from the square \([-1, 1]^2\) with no margin promised, and the comparison vector is \(\mathbf w^\star = \mathbf u / 0.1\), so \(\|\mathbf w^\star\| = 10\):

```python
import random
from math import hypot, sqrt

rng, gamma, u = random.Random(2), 0.1, (0.6, 0.8)
dot = lambda a, b: a[0] * b[0] + a[1] * b[1]
w_star = (u[0] / gamma, u[1] / gamma)
w, M, R, hinge = (0.0, 0.0), 0, 0.0, 0.0
while M < 40:
    x = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    y = 1 if dot(u, x) > 0 else -1
    if y * dot(w, x) <= 0:
        M, R = M + 1, max(R, hypot(*x))
        hinge += max(0.0, 1 - y * dot(w_star, x))
        w = (w[0] + y * x[0], w[1] + y * x[1])
norm = hypot(*w_star)
b, c = R * norm, hinge
print(M, round(c, 1), round(M - b * sqrt(M) - c, 1))
print(round(c + b * sqrt(c) + b * b, 1))
```

It prints `40 22.0 -66.3` and `261.9`: after 40 mistakes, \(\mathbf w^\star\)’s shortfall on those rounds is 22.0, the quantity \(|\mathcal M| - R\|\mathbf w^\star\|\sqrt{|\mathcal M|} - C\) is \(-66.3\), negative as the proof requires, and the theorem’s bound is 261.9, far above 40. (Here \(R\) is the largest norm among the update rounds, which is all the proof uses.)

And against an adversary that always sends the message the current Perceptron gets most wrong, from a pool of 5,000 messages that \(\mathbf u\) separates with margin \(\gamma = 0.1\) (every \(|\langle \mathbf u, \mathbf x \rangle| \ge 0.1\), \(\|\mathbf x\| \le 1\)):

```python
import random
from math import hypot

rng, gamma, u = random.Random(2), 0.1, (0.6, 0.8)
dot = lambda a, b: a[0] * b[0] + a[1] * b[1]
pool = []
while len(pool) < 5000:
    x = (rng.uniform(-1, 1), rng.uniform(-1, 1))
    if hypot(*x) <= 1 and abs(dot(u, x)) >= gamma:
        pool.append((x, 1 if dot(u, x) > 0 else -1))

def perceptron(eta):
    w, M = (0.0, 0.0), 0
    while True:
        bad = [(y * dot(w, x), x, y) for x, y in pool
               if y * dot(w, x) <= 0]
        if not bad:
            return M
        s, x, y = min(bad)
        M += 1
        w = (w[0] + eta * y * x[0], w[1] + eta * y * x[1])

print(perceptron(1.0), perceptron(0.01), round(1 / gamma ** 2))
```

It prints `28 28 100`. With \(\mathbf w^\star = \mathbf u/\gamma\), every message has \(y\langle \mathbf w^\star, \mathbf x \rangle \ge 1\), \(R \le 1\) and \(\|\mathbf w^\star\| = 1/\gamma = 10\), so the theorem promises at most \(R^2\|\mathbf w^\star\|^2 = (R/\gamma)^2 \le 100\) mistakes. This adversary manages 28, the same for both step sizes, and once no message in the pool is misclassified it has nothing left to send.

[[diagram:surrogate]]

The bound is honest about its price. The Perceptron can still make many mistakes on a sequence that some halfspace labels perfectly, because the Littlestone dimension is infinite; the theorem only helps when the comparison vector’s total hinge shortfall is not too large. Margin is the assumption that replaces realizability.

## The learners as code

Here are the six learners side by side, with what each costs per round. Write \(n\) for the number of features of a message and \(d\) for the number of experts.

```
Consistent(H):
  V ← H
  each round: receive x; h ← any member of V; predict h(x)
              receive y; V ← {h in V : h(x) = y}

Halving(H):
  V ← H
  each round: receive x; predict 1 if at least half of V says 1,
              else 0; receive y; V ← {h in V : h(x) = y}

SOA(H):
  V ← H
  each round: receive x; split V into V^(0), V^(1) by h(x);
              predict the r with larger Ldim(V^(r)) (ties: 1)
              receive y; V ← V^(y)

WeightedMajority(d, T):
  η ← sqrt(2 log(d) / T); w̃ ← (1, ..., 1)
  each round: w ← w̃ / sum(w̃); follow expert i with prob. w_i
              receive costs v; w̃_i ← w̃_i · exp(−η v_i) for all i

OnlineGradientDescent(η):
  w ← 0
  each round: predict w; receive z
              v ← a subgradient of ℓ(·, z) at w
              w ← projection onto H of (w − η v)

Perceptron:
  w ← 0
  each round: receive x; predict sign⟨w, x⟩; receive y
              if y⟨w, x⟩ ≤ 0: w ← w + y x
```

**Consistent** and **Halving** keep the version space: \(O(|\mathcal H|)\) space, and \(O(|\mathcal H|)\) hypothesis evaluations per round, so \(O(T|\mathcal H|)\) in total. For lesson 1’s ten one-feature hypotheses that is nothing; for halfspaces it is impossible, since the class is infinite. **SOA** needs the Littlestone dimension of two classes per round. Through the root split recursion, that is a search over a game tree whose size can grow exponentially with \(|\mathcal H|\) and the number of messages; SOA is optimal in mistakes and can be hopeless in time. **Weighted-Majority** stores \(d\) weights and spends \(O(d)\) time per round (plus \(d\) hypothesis evaluations when the experts are hypotheses); with the Littlestone experts, \(d\) is about \(T^L\) and each expert runs SOA, so that construction proves learnability but is not a practical filter. **Online Gradient Descent** and the **Perceptron** keep one vector: \(O(n)\) space and \(O(n)\) time per round (\(O(s)\) for a message with \(s\) nonzero word counts), plus the projection, which for a ball of radius \(B\) is a rescaling in \(O(n)\). These last two are the ones a real mail system can afford at millions of messages a day.

**Build it yourself.** Using only the Python standard library:

```python
from itertools import product
from math import exp, log, sqrt

X = list(product([0, 1], repeat=4))
H = {name: tuple(x[j] for x in X)
     for j, name in enumerate("FLUN")}
stream = [((1, 0, 0, 0), 0), ((0, 1, 0, 0), 0), ((0, 0, 1, 0), 0),
          ((0, 0, 0, 1), 1), ((1, 1, 1, 1), 1), ((0, 0, 0, 0), 0)]
costs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]] * 50
points = [((1, 0), 1), ((0, 1), -1), ((1, 1), 1),
          ((-1, 0.5), -1), ((0.5, -1), 1), ((0.2, 0.9), -1)]

def consistent(H, X, stream): ...
def halving(H, X, stream): ...
def ldim(V, X): ...
def weighted_majority(costs, eta): ...
def perceptron(points, eta=1.0): ...

if __name__ == "__main__":
    assert consistent(H, X, stream) == 3
    assert halving(H, X, stream) == 1
    assert ldim(frozenset(H.values()), X) == 2
    paid, best = weighted_majority(costs, sqrt(2 * log(3) / 200))
    assert paid - best <= sqrt(2 * log(3) * 200)
    assert round(paid, 2) == 61.48 and best == 50
    assert perceptron(points) == perceptron(points, 0.01) == 4
    print("all asserts pass")
```

1. A hypothesis is the tuple of its labels on the 16 messages of `X`, in order, and `H` maps the names F, L, U, N to those tuples. `consistent` predicts with the first survivor in the order F, L, U, N and returns its number of mistakes on `stream`; `halving` votes, with ties going to junk, and returns its number of mistakes.
2. `ldim(V, X)` returns the Littlestone dimension of a set of label tuples, through the root split (use \(-1\) for the empty set).
3. `weighted_majority(costs, eta)` runs the algorithm on the list of cost vectors and returns the learner’s total expected cost and the best expert’s total cost.
4. `perceptron(points, eta)` presents the points in order, over and over, until a full pass makes no update, and returns the number of updates. Check that the answer does not depend on `eta`.
5. Then play adversary: write `worst(V, learner)` as in “Halving: let the survivors vote” and confirm that it returns 3 for Consistent, 2 for Halving, and 2 for SOA on \(\mathcal H_1\).

## Fix the bug

Back to the spammers. You now have learners whose promises do not depend on where the mail comes from, and the lab below puts them against an adversary that plays as well as it can: in every round it computes the message and label that hurt your learner most, while obeying whatever rules the round sets.

Each week has a rule and a budget. In some weeks the junk follows one of the ten one-feature hypotheses (realizable), and you may make at most a fixed number of mistakes, ever. In others, no hypothesis is perfect, and your regret against the best expert over the week must stay under a budget. In the last, the filter is a halfspace over word counts, the spammers can only promise a margin, and you must certify the number of mistakes. For each week you choose the learner and its parameters, and the lab first checks that the bound you just proved, with your parameters, is within the budget. Only then does it run the week. A filter that happens to do well against a lazy adversary earns nothing; the adversary here is not lazy, and the certificate is what ships.

Before you start, work out what each budget allows. How many mistakes can Consistent be forced into with ten hypotheses, and how many Halving? If the adversary sees your prediction before choosing the label, what is the least regret it can force on any deterministic learner over 2,000 rounds? Which \(\eta\) makes \(\log(d)/\eta + \eta T/2\) smallest?

[[lab-act2]]

## What people get wrong

**“Online learning is batch learning done one example at a time.”** The difference is not the order of the examples but who chooses them. Lesson 14’s SGD also sees one example per step, but its examples are fresh draws from \(\mathcal D\). Online learning assumes nothing about where they come from, and that is why thresholds, easy in the batch world, become impossible online.

**“Any consistent hypothesis is a fine choice.”** It was in lesson 2. Online, Consistent can be forced into \(|\mathcal H| - 1\) mistakes while Halving makes at most \(\log_2|\mathcal H|\). Which survivor you follow matters exponentially.

**“The Littlestone dimension is just the VC dimension with a different name.”** It is at least the VC dimension and can be infinitely larger. The VC dimension asks for one set of messages that every labelling fits; the Littlestone dimension lets each next message depend on the labels so far, which is exactly the power of an adversary.

**“Randomizing just adds noise to the filter.”** Without it, no learner has sublinear regret even against two constant hypotheses: the adversary reads the prediction and flips it. The coin works because the adversary must commit to the label before the coin is flipped, and the guarantee is about the expected number of mistakes.

**“The Weighted-Majority bound needs the adversary to be random.”** The bound holds for every sequence of costs, including ones chosen after looking at the weights. The only expectation in it is over the learner’s own coin.

**“The Perceptron learns any separable stream with few mistakes.”** Only with a margin. Without one, the Littlestone dimension of halfspaces is infinite, and an adversary can binary-search the boundary forever. The bound \(R^2\|\mathbf w^\star\|^2\) is large exactly when the margin \(1/\|\mathbf w^\star\|\) is small.

## Recap and the next bug

1. In online learning, messages arrive one at a time, chosen by an adversary; the learner predicts, sees the label, and updates. With no restriction the adversary forces a mistake on every round.
2. Under realizability, a learner is judged by its mistake bound \(M_A(\mathcal H)\). Consistent makes at most \(|\mathcal H| - 1\) mistakes and Halving at most \(\log_2 |\mathcal H|\), because each of its mistakes halves the version space.
3. The adversary’s strategy is a binary tree of messages. The Littlestone dimension is the depth of the deepest tree the class shatters. Every learner can be forced into \(\mathrm{Ldim}(\mathcal H)\) mistakes, and SOA, which predicts with the side of higher dimension, never makes more. So a class is online learnable exactly when its Littlestone dimension is finite. It is always at least the VC dimension, and thresholds show the gap can be infinite.
4. Without realizability, we measure regret against the best hypothesis in hindsight. Deterministic learners can be forced to regret \(T/2\) (Cover’s impossibility). With a coin the adversary cannot see, Weighted-Majority has regret at most \(\sqrt{2 \log(d)\, T}\), proved by squeezing the total trust \(Z_t\) between the learner’s cost and the best expert’s; with experts built from SOA, every class of finite Littlestone dimension has regret \(O(\sqrt{\mathrm{Ldim}(\mathcal H)\, T \log T})\).
5. For convex losses, Online Gradient Descent has regret at most \(B\rho\sqrt T\) against any sequence: lesson 14’s step lemma, with no expectations.
6. The Perceptron is Online Gradient Descent on hinge surrogates charged only on mistake rounds. Its mistakes are at most \(C + R\|\mathbf w^\star\|\sqrt C + R^2\|\mathbf w^\star\|^2\), and at most \(R^2\|\mathbf w^\star\|^2\) when \(\mathbf w^\star\) separates with margin.

The filter holds its ground against the spammers now, with a promise written down before each week starts. Then a different problem lands on your desk. The team of people who label messages has quit, all of them, on the same day. Thousands of messages keep arriving and not one of them has a label: no junk, no keep, nothing to count mistakes against. And the company still wants the mail sorted into kinds, newsletters with newsletters, invoices with invoices, whatever the kinds turn out to be. Every learner in this course so far has needed labels. What can a learner possibly do with none? That is the next lesson.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. Weighted-Majority’s theorem assumes \(T > 2 \log d\). Point to the step of the proof that uses it, and say what goes wrong there without it. Then explain why the final inequality \(\sum_t \langle \mathbf w^{(t)}, \mathbf v_t \rangle - \min_i \sum_t v_{t,i} \le \sqrt{2 \log(d)\, T}\) is still true when \(T \le 2 \log d\).
2. In the proof of SOA’s mistake bound, where exactly is the prediction rule used? Suppose a variant of SOA predicts with the side of *smaller* Littlestone dimension. Which step of the proof breaks? Give a class (any small one from this lesson will do) on which an adversary forces this variant into more than \(\mathrm{Ldim}(\mathcal H)\) mistakes.
3. A week of mail has \(d = 100\) experts and \(T = 5{,}000\) rounds. Which \(\eta\) does Weighted-Majority use, what regret does it guarantee, and what is that per round? In a later week the filter is a halfspace with messages of norm at most \(R = 1\), and a comparison vector with \(\|\mathbf w^\star\| = 4\) has total hinge shortfall 9 over the stream. At most how many mistakes can the Perceptron make?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (Halving can be pushed to its bound).** Find a hypothesis class \(\mathcal H\) and a sequence of examples, realizable by \(\mathcal H\), on which Halving makes exactly \(\log_2 |\mathcal H|\) mistakes. Can you do it for every power of two?
- **Problem 2 (not knowing the horizon).** Weighted-Majority and Online Gradient Descent both choose their step size from the number of rounds \(T\), which a mail filter running forever does not know. Suppose an algorithm has regret at most \(\alpha\sqrt T\) whenever it is told \(T\). Run it afresh on rounds \(1\), then \(2, 3\), then \(4, \dots, 7\), and in general on the \(2^m\) rounds \(t = 2^m, \dots, 2^{m+1} - 1\), each time telling it the length of the period. Show that the total regret after any \(T\) rounds is at most \(\frac{\sqrt 2}{\sqrt 2 - 1}\,\alpha\sqrt T\).
- **Problem 3 (from online to batch).** Suppose an online learner \(A\) has a mistake bound \(M_A(\mathcal H) < \infty\), and on round \(t\) it predicts with a hypothesis \(h_t\) built from the earlier rounds. Now feed it \(T\) messages drawn i.i.d. from a distribution \(\mathcal D\) and labelled by some \(h^\star \in \mathcal H\), and pick a round \(r\) uniformly at random from \([T]\). Show that \(\mathbb E[L_{\mathcal D}(h_r)] \le M_A(\mathcal H)/T\), where the expectation is over the messages and over \(r\).
