# Understanding Machine Learning, the mail room series

Source: Shai Shalev-Shwartz and Shai Ben-David, *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, 2014. Chapters 1 to 31 and Appendices A to C. Notation follows the book's Section 1.6.
Audience: experienced software engineers with strong analytical and programming skills and no machine learning background.
Language: English.

## The world

You are building the learner behind a company's mail filter. Every incoming message is described by yes/no features (does the subject say "FREE", does it have a link, is the sender unknown, was it sent at night), and the filter has to label it junk or keep. Your users label some messages by hand; the learner has to label everything else. Every lesson opens on a bug your users notice in the filter, and ends when you fix it yourself.

## Lessons

The course follows the book in order. Each lesson assumes only the lessons before it. Lesson numbers match chapter numbers. Lesson pages never cite the book: they read as an original course, and book references live only in each lesson's 1-sheet.md.

| # | Lesson | Bug | Fix | New terms | Page | Status |
|---|---|---|---|---|---|---|
| 1 | Learning needs a bias | the first filter is perfect on every message your users labelled, and no better than a coin on every message it has not seen | give the learner an inductive bias: let it choose only among one-feature hypotheses, a small class that still contains the truth, and give it enough examples that no coincidence survives | example, training sequence, hypothesis, learner, empirical risk, true risk, consistent, memorizer, hypothesis class, inductive bias, supervised, unsupervised, active, passive, online, batch | 01-learning-needs-a-bias/3-page/3-page.html | page ready; no video in this series |

## Course plan

Every lesson of the course, delivered or not. `_kit/index.py` reads this table for titles and `TASKS.md` for status, so the index always lists all of them.

| # | Title | Question |
|---|---|---|
| 1 | Learning needs a bias | Why does a filter that is perfect on every labelled message still guess like a coin on new ones? |
| 2 | A gentle start | With a finite class and a few hundred labels, how sure can you be that a perfect fit is good? |
| 3 | A formal learning model | What exactly does it mean for a class to be learnable, and what if no rule is perfect? |
| 4 | Uniform convergence | When does every hypothesis's training error track its true error at once? |
| 5 | No free lunch | Why can no learner win on every problem, and how much error comes from the class versus the sample? |
| 6a | The VC dimension | Which infinite classes are learnable, and what measures how rich a class is? |
| 6b | The growth function | Why does a finite VC dimension make every rule's training error track its true error, and how many noisy labels does that take? |
| 7a | Nonuniform learnability | Can you learn from a class too rich for one sample size, by paying more for longer lists? |
| 7b | Description length and Occam's razor | Can a list pay for what it says, not for the size class it lives in? |
| 8 | The runtime of learning | When is learning possible in principle but too slow to compute? |
| 9a | Halfspaces and the Perceptron | How can a filter that weighs every word be learned when its class is infinite, and how many updates can the Perceptron take? |
| 9b | Linear and logistic regression | How do you find, and prove you found, the best line for a junk score, and how does the same line give a probability of junk? |
| 10a | AdaBoost | Can a pile of rules that are each barely better than a coin be turned into one that is right on nearly every labelled message? |
| 10b | Boosting and fresh mail | A boosted vote is nearly perfect on its labels, so what can you promise about fresh mail, and how many rounds should you run? |
| 11 | Model selection | How do you pick among models without fooling yourself with the same data? |
| 12a | Convexity | Which losses can an optimiser minimise reliably, and how does a walk prove it has reached the bottom? |
| 12b | Convex learnability and surrogates | A convex loss can be minimised, but can it be learned from a finite label budget, and how does it become a promise about mistakes? |
| 13a | Regularization and stability | Why does a penalty on the weights steady a jumpy learner, and why does a steady learner not overfit? |
| 13b | Strong convexity and the fitting-stability tradeoff | Can you prove a penalized learner is stable for every distribution, and pick the penalty that certifies its error? |
| 14a | From gradient descent to SGD | How can a learner that reads one message per step still reach a filter it can guarantee? |
| 14b | SGD variants and faster rates | Can SGD solve a fixed regularized objective fast, and certify how close it gets? |
| 15 | Support vector machines | What does it buy you to separate the mail with the widest possible margin? |
| 16 | Kernel methods | How do you use a huge feature space without ever writing it down? |
| 17 | Multiclass and ranking | What changes when the answer is one of many folders, or an ordering of the inbox? |
| 18 | Decision trees | How do you grow a readable rule from questions, and when do you stop? |
| 19 | Nearest neighbour | Can you label a message just by looking at the most similar ones you have seen? |
| 20 | Neural networks | Can the filter learn its own features from raw messages? |
| 21a | Online mistake bounds | If an adversary picks every message but some rule in the class is perfect, how few mistakes can a filter promise? |
| 21b | Online regret | When no rule is perfect and the adversary reads every decision, how can a filter promise to be nearly as good as the best rule? |
| 22 | Clustering | How do you group messages when nobody has labelled any of them? |
| 23 | Dimensionality reduction | How do you squeeze thousands of features into a few without losing what matters? |
| 24 | Generative models | What if you model how each kind of message is produced, then ask which one made this? |
| 25 | Feature selection and generation | Which features should the filter keep, and which new ones should it build? |
| 26 | Rademacher complexity | Can you measure a class's richness on the very data you have? |
| 27 | Covering numbers | How many representative rules does it take to cover a whole class? |
| 28 | Proving the fundamental theorem | Why do VC dimension and learnability really match, in full? |
| 29 | Multiclass learnability | Which multiclass problems are learnable, and what replaces the VC dimension? |
| 30 | Compression bounds | If a few examples are enough to rebuild the rule, why does it generalise? |
| 31 | PAC-Bayes | How does a prior belief over rules turn into a guarantee? |

## Conventions every lesson keeps

- A message is a vector of yes/no features; the label is 1 for junk and 0 for keep.
- The four features of lesson 1, in this order: F (the subject says "FREE"), L (it has a link), U (the sender is unknown), N (it was sent at night). A message is written as four bits, FLUN, so 1011 is "FREE", no link, unknown sender, at night.
- Lessons are written as an original course: no "the book", no chapter, section, theorem or page numbers in the prose, figures, labs or page. Results are named by what they say, other lessons as "lesson N". The source (Shalev-Shwartz and Ben-David, *Understanding Machine Learning*, 2014) is credited in NOTICE and the README, and cited with pages only in each 1-sheet.md.
- Notation: accuracy is \(\varepsilon\) (written `\varepsilon`, never `\epsilon`), confidence \(\delta\), sample size \(m\), true risk \(L_{\mathcal D}\), empirical risk \(L_S\).
- Hard math gets a short runnable Python snippet beside it (standard library, prints or asserts the number the prose states), run before the lesson ships.

## Glossary

- example: one message together with its label, written \(z = (x, y)\) (lesson 1)
- training sequence: the \(m\) examples the learner gets to see, \(S = (z_1, \dots, z_m)\) (lesson 1)
- hypothesis: a rule that gives a label to every possible message, a function \(h: \mathcal X \to \mathcal Y\) (lesson 1)
- learner: an algorithm that takes a training sequence and returns a hypothesis, \(h_S = A(S)\) (lesson 1)
- empirical risk: the fraction of the training examples a hypothesis gets wrong, \(L_S(h)\) (lesson 1)
- true risk: the fraction of all messages, weighted by how often each arrives, that a hypothesis gets wrong, \(L_{\mathcal D}(h)\) (lesson 1)
- consistent: a hypothesis with empirical risk 0; it agrees with every example (lesson 1)
- hypothesis class: the set \(\mathcal H\) of hypotheses a learner is allowed to return, fixed before it sees any example (lesson 1)
- inductive bias: the prior knowledge a learner brings, built in by choosing its hypothesis class before seeing data (lesson 1)

## Next bug

The real mail system has 30 features, and labels are expensive: you can afford a few hundred. Which hypothesis classes can you afford, and when the learner returns a hypothesis that fits all your examples, how sure can you be that its true risk is small? (lesson 2)
