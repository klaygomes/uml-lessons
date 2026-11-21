# Online Mistake Bounds

## The spammers are watching

Lesson 20 ended with the triangle filter, certificate and all, losing ground the very next morning. Look closer at what the spammers do now. They no longer send a big batch of junk and wait. They send one message, see whether it got through, and then decide what to send next. If “FREE” in the subject gets caught, the next message says “F.R.E.E”. If the filter lets a message with a link through, the next hundred all have links. Every decision the filter makes is information for them, and they use it straight away.

Your monitoring shows the damage. The filter is retrained every night on the day’s labelled mail, and every morning it starts losing again, because whatever it learned overnight is exactly what the spammers stop doing. The mail today depends on what the filter did yesterday, and on what it did five minutes ago. Every guarantee from lessons 2 to 20 began with one promise: the messages are drawn independently from one fixed distribution \(\mathcal D\), and the learner gets a whole training sequence \(S \sim \mathcal D^m\) before it has to say anything. An opponent who reacts to you is about as far from that promise as anything can be, so not one of those guarantees applies.

So what can a learner promise when the mail is chosen by someone who reads its decisions? It sounds hopeless. It is not, and this lesson and the next, lesson 21b, are about what you can still prove. This one handles the case where some rule behind the junk is perfect; lesson 21b drops even that.

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

A bound \(M_A(\mathcal H) \le B < \infty\) is a **mistake bound** (here \(B\) is just a number, not the norm bound of lessons 12b to 15b). A class \(\mathcal H\) is **online learnable** if some online learner \(A\) has a mistake bound for it. For now we ignore running time completely, as lesson 3 did at first with PAC learning: the learners in this lesson may be slow, and the efficient ones arrive in lesson 21b.

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

That is the idea from above at work: every mistake at least halves the version space while the truth stays, and starting from \(|\mathcal H|\) you can halve only about \(\log_2 |\mathcal H|\) times before you are down to one. The prediction rule, written out, is

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

Halving’s bound depends on \(|\mathcal H|\), the number of hypotheses. Lesson 6a taught us to distrust counts: the class of thresholds is infinite, yet it is learnable from a few labels. Is there a better measure of how hard a class is online? To find it, look at the game from the adversary’s side.

The adversary wants to force a mistake on every one of the first few rounds. That part is easy: it waits for the prediction and announces the opposite label. The hard part is staying honest, so that at the end some hypothesis in \(\mathcal H\) agrees with every label it announced. Which message should it send on round 2? That depends on what happened on round 1, which depends on the learner. So the adversary’s plan is not a list of messages. It is a branching plan: “send \(v_1\) first; if the label turns out 0, send this next; if 1, send that.”

A branching plan is a **binary tree**. Each node holds a message. The adversary starts at the top node, the **root**, and sends its message. When the label is revealed, it moves to the left child if the label was 0 and to the right child if it was 1, and sends that node’s message next. The **depth** of the tree is the number of nodes on a path from the root down to the bottom, one for each round the plan covers. A tree in which every path has the same depth \(d\) and every node above the bottom has two children is **complete**, and it has \(2^d - 1\) nodes that hold messages: 1 for depth 1, 3 for depth 2.

Here is one for \(\mathcal H_1\), of depth 2. The root holds 1100, on which \(h_F\) and \(h_L\) say junk, \(h_U\) and \(h_N\) keep. Its left child, used after the label 0, holds 0010: the survivors are then \(h_U\) and \(h_N\), and 0010 splits them. Its right child, used after the label 1, holds 1000: the survivors are \(h_F\) and \(h_L\), and 1000 splits them. Every one of the four label paths (0 then 0, 0 then 1, 1 then 0, 1 then 1) is followed by exactly one hypothesis. So whatever the learner predicts, the adversary can announce the opposite label twice and still have a hypothesis that agrees. That is the depth-2 game from the Halving table, drawn as a plan.

[[diagram:shattered-tree]]

To write this down we need names for the nodes. Number them like a heap in a computer science course: the root is node 1, and node \(i\) has left child \(2i\) and right child \(2i + 1\). So the messages are \(v_1, v_2, \dots, v_{2^d - 1}\). If \(i_t\) is the node used on round \(t\), then \(i_1 = 1\) and \(i_{t+1} = 2 i_t + y_t\): **double to go down a level, and add the label to go right on a 1.** With labels \(y_1 = 1\), \(y_2 = 0\): \(i_1 = 1\), and \(i_2 = 2 \cdot 1 + 1 = 3\), the right child, holding 1000 in our tree.

```python
from itertools import product

def path(ys):
    i = [1]
    for y in ys[:-1]:
        i.append(2 * i[-1] + y)
    return i

print(path((1, 0)))
v = {1: "1100", 2: "0010", 3: "1000"}
h = {n: (lambda x, k=k: int(x[k])) for k, n in enumerate("FLUN")}
print(all(any(all(h[n](v[i]) == y for i, y in zip(path(ys), ys))
              for n in h) for ys in product([0, 1], repeat=2)))
```

It prints `[1, 3]` (the path for labels 1 then 0) and `True` (our depth-2 tree for \(\mathcal H_1\) works: for each of the four label paths, some hypothesis agrees with all of it).

That last check is the property we need, and it gets a name.

**Definition (shattered tree).** A complete binary tree of depth \(d\) with messages \(v_1, \dots, v_{2^d - 1}\) is **shattered** by \(\mathcal H\) if for every sequence of labels \((y_1, \dots, y_d) \in \{0, 1\}^d\) there is some \(h \in \mathcal H\) with \(h(v_{i_t}) = y_t\) for every \(t \in [d]\), where \(i_t\) is the node the labels lead to.

In plain words: **whichever way the labels go down the tree, some hypothesis in the class agrees with all of them.** The word is borrowed from lesson 6a, where a set of messages was shattered when the class could label it in every possible way; here it is a tree instead of a set, so later messages may depend on earlier labels.

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

**Thresholds on a score.** Lesson 6a met thresholds on a spam score in \([0, 1]\): for each cut-off \(\theta\) (“theta”), the hypothesis \(h_\theta(x) = \mathbb 1[x > \theta]\) says junk when the score \(x\) is above \(\theta\). (The indicator \(\mathbb 1[\cdot]\) is 1 when the statement inside is true and 0 when it is false, so \(h_{0.5}(0.7) = 1\), junk, and \(h_{0.5}(0.2) = 0\), keep.) Its VC dimension is 1, so in the batch world it is learnable from a handful of labels. Online, the adversary plays binary search. It sends score \(1/2\). Whatever the learner predicts, it announces the opposite. If it said junk, the threshold must be below \(1/2\), so it next sends \(1/4\); if keep, the threshold is at least \(1/2\), so it sends \(3/4\). And so on, halving the interval of thresholds that agree with everything so far. Every round is a mistake, and because the interval never becomes empty, some threshold agrees with all the labels, at every depth. So \(\mathrm{Ldim} = \infty\), and *no* online learner has a mistake bound for thresholds.

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

*Proof.* The paragraph before the lemma, in both directions: a shattered tree of depth \(d \ge 1\) with root \(x\) makes the bracket for \(x\) at least \(d\), and a bracket equal to \(d\) for some \(x\) builds a shattered tree of depth \(d\) under the root \(x\). \(\blacksquare\)

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

*Proof.* Recall from lesson 6a that a set \(\{x_1, \dots, x_d\}\) is shattered by \(\mathcal H\) when every one of the \(2^d\) labellings of it is given by some hypothesis, and \(\mathrm{VCdim}(\mathcal H)\) is the largest such \(d\). Take a shattered set of size \(d\) and build a complete tree of depth \(d\) whose nodes on level \(i\) all hold \(x_i\). Whatever the label path \((y_1, \dots, y_d)\), the messages visited are \(x_1, \dots, x_d\) in order, and the set is shattered, so some \(h\) gives \(h(x_i) = y_i\) for all \(i\). The tree is shattered, so \(\mathrm{Ldim}(\mathcal H) \ge d\). For the gap: thresholds on \([0, 1]\) have VC dimension 1 and Littlestone dimension \(\infty\). \(\blacksquare\)

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


## The learners as code

Here are the three learners of this lesson side by side, with what each costs per round.

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
```

**Consistent** and **Halving** keep the version space: \(O(|\mathcal H|)\) space, and \(O(|\mathcal H|)\) hypothesis evaluations per round, so \(O(T|\mathcal H|)\) in total. For lesson 1’s ten one-feature hypotheses that is nothing; for a class with millions of members it is already heavy, and for an infinite class, like the thresholds, you cannot even write the version space down as a list. **SOA** keeps the same version space, and on top of that needs the Littlestone dimension of two classes per round, a game-tree search that can grow exponentially with \(|\mathcal H|\) and the number of messages. All three are learners you would run on a small class to know the truth about it, not filters you would run on millions of messages a day. The cheap ones come in lesson 21b.

**Build it yourself.** Using only the Python standard library:

```python
from itertools import product

X = list(product([0, 1], repeat=4))
H = {name: tuple(x[j] for x in X)
     for j, name in enumerate("FLUN")}
stream = [((1, 0, 0, 0), 0), ((0, 1, 0, 0), 0), ((0, 0, 1, 0), 0),
          ((0, 0, 0, 1), 1), ((1, 1, 1, 1), 1), ((0, 0, 0, 0), 0)]
scores = list(range(15))
thresholds = {t: tuple(int(x > t) for x in scores)
              for t in range(-1, 15)}

def consistent(H, X, stream): ...
def halving(H, X, stream): ...
def ldim(V, X): ...
def soa(H, X, stream): ...

if __name__ == "__main__":
    assert consistent(H, X, stream) == 3
    assert halving(H, X, stream) == 1
    assert ldim(frozenset(H.values()), X) == 2
    assert ldim(frozenset(thresholds.values()), scores) == 4
    assert soa(H, X, stream) == 1
    print("all asserts pass")
```

1. A hypothesis is the tuple of its labels on the 16 messages of `X`, in order, and `H` maps the names F, L, U, N to those tuples. `consistent` predicts with the first survivor in the order F, L, U, N and returns its number of mistakes on `stream`; `halving` votes, with ties going to junk, and returns its number of mistakes.
2. `ldim(V, X)` returns the Littlestone dimension of a set of label tuples, through the root split (use \(-1\) for the empty set). The second `ldim` assert runs it on the 16 thresholds over 15 grid scores, where each message is just its position in `scores`.
3. `soa` predicts with the side of larger Littlestone dimension, ties going to junk, and returns its number of mistakes on `stream`. Work out by hand, before you run it, on which round its one mistake falls.
4. Then play adversary: write `worst(V, learner)` as in “Halving: let the survivors vote” and confirm that it returns 3 for Consistent, 2 for Halving, and 2 for SOA on \(\mathcal H_1\).

## Fix the bug

Back to the spammers. You now have learners whose promises do not depend on where the mail comes from, and the lab below puts them against an adversary that plays as well as it can: in every round it computes the message and label that hurt your learner most, while keeping the labels honest, so that at the end some hypothesis in the week’s class agrees with all of them.

Each week has a class and a mistake budget: the most mistakes the filter may make, ever, however long the week runs. In the first week the junk follows one of the ten one-feature hypotheses, and the budget is 3. In the second the class is the nine-hypothesis class of “The Standard Optimal Algorithm”, parts A and B together, and the budget is 2. In the third, the junk is a threshold on a score that takes 15 grid values, and the budget is 4. For each week you choose the learner, and the lab first checks that the mistake bound you proved for it, on that class, is within the budget. Only then does it run the week. A filter that happens to do well against a lazy adversary earns nothing; the adversary here is not lazy, and the proved bound is what ships.

Before you start, work out what each budget allows. How many mistakes can Consistent be forced into with ten hypotheses, and what does Halving’s bound promise there, once you remember that a number of mistakes is a whole number? On the nine-hypothesis class, what does Halving’s bound promise, and what does SOA’s? And on the thresholds, what are \(|\mathcal H|\) and \(\mathrm{Ldim}(\mathcal H)\)?

[[lab-act2]]

## What people get wrong

**“Online learning is batch learning done one example at a time.”** The difference is not the order of the examples but who chooses them. Lesson 14a’s SGD also sees one example per step, but its examples are fresh draws from \(\mathcal D\). Online learning assumes nothing about where they come from, and that is why thresholds, easy in the batch world, become impossible online.

**“Any consistent hypothesis is a fine choice.”** It was in lesson 2. Online, Consistent can be forced into \(|\mathcal H| - 1\) mistakes while Halving makes at most \(\log_2|\mathcal H|\). Which survivor you follow matters exponentially.

**“The Littlestone dimension is just the VC dimension with a different name.”** It is at least the VC dimension and can be infinitely larger. The VC dimension asks for one set of messages that every labelling fits; the Littlestone dimension lets each next message depend on the labels so far, which is exactly the power of an adversary.

## Recap and the next bug

1. In online learning, messages arrive one at a time, chosen by an adversary; the learner predicts, sees the label, and updates. With no restriction the adversary forces a mistake on every round.
2. Under realizability, a learner is judged by its mistake bound \(M_A(\mathcal H)\). Consistent makes at most \(|\mathcal H| - 1\) mistakes and Halving at most \(\log_2 |\mathcal H|\), because each of its mistakes halves the version space.
3. The adversary’s strategy is a binary tree of messages. The Littlestone dimension is the depth of the deepest tree the class shatters. Every learner can be forced into \(\mathrm{Ldim}(\mathcal H)\) mistakes, and SOA, which predicts with the side of higher dimension, never makes more. So a class is online learnable exactly when its Littlestone dimension is finite.
4. The Littlestone dimension is always at least the VC dimension, and thresholds show the gap can be infinite: a class that is easy in the batch world can be impossible online.

The filter ships with a proved mistake bound, and for a few days the spammers get nowhere. Then, on Wednesday morning, a user reports a newsletter she subscribed to, a plain daytime message from a sender she knows, as junk. An hour later her colleague receives the very same newsletter and keeps it. One message, two labels. No hypothesis in any class can agree with both, so the version space is now empty, and every learner in this lesson was built on the promise that it never would be. What does Halving do when there is no one left to vote? And what can any filter promise once no hypothesis is perfect? That is lesson 21b.

## Concept check

Write your answers down before moving on to the next lesson, with your reasoning, not just the answer.

1. In the proof of SOA’s mistake bound, where exactly is the prediction rule used? Suppose a variant of SOA predicts with the side of *smaller* Littlestone dimension. Which step of the proof breaks? Give a class (any small one from this lesson will do) on which an adversary forces this variant into more than \(\mathrm{Ldim}(\mathcal H)\) mistakes.
2. The junk score now takes 31 grid values, and the class is the 32 thresholds on them. What is its Littlestone dimension? What do Consistent’s and Halving’s bounds promise, and what is the most mistakes the best adversary can force on SOA? With a mistake budget of 5, which of the three learners can you ship, and why is the answer different on the whole interval \([0, 1]\) of scores?

## Problems

Once the concept check is done, try these. Hints come one at a time in the chat, and no full solutions.

- **Problem 1 (Halving can be pushed to its bound).** Find a hypothesis class \(\mathcal H\) and a sequence of examples, realizable by \(\mathcal H\), on which Halving makes exactly \(\log_2 |\mathcal H|\) mistakes. Can you do it for every power of two?
- **Problem 2 (from online to batch).** Suppose an online learner \(A\) has a mistake bound \(M_A(\mathcal H) < \infty\), and on round \(t\) it predicts with a hypothesis \(h_t\) built from the earlier rounds. Now feed it \(T\) messages drawn i.i.d. from a distribution \(\mathcal D\) and labelled by some \(h^\star \in \mathcal H\), and pick a round \(r\) uniformly at random from \([T]\). Show that \(\mathbb E[L_{\mathcal D}(h_r)] \le M_A(\mathcal H)/T\), where the expectation is over the messages and over \(r\).
