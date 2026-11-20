# Understanding Machine Learning: navigation map

- Title: Understanding Machine Learning: From Theory to Algorithms (Cambridge University Press, 2014)
- Authors: Shai Shalev-Shwartz and Shai Ben-David
- PDF: `/Users/cleiton.loiola/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/understanding-machine-learning-theory-algorithms.pdf` (449 pages)
- Page-offset rule: PDF page = printed page, offset 0 throughout the whole book (verified at every chapter start, the appendices, Notes, References and Index; all 325 table-of-contents entries were checked against the heading on that PDF page). Front matter uses roman numerals but is also aligned (Preface = vii = PDF 7).
- Extract a range (N to M inclusive):

```bash
pdftotext -layout -f N -l M "/Users/cleiton.loiola/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/understanding-machine-learning-theory-algorithms.pdf" -
```

Conventions: all page numbers below are PDF pages. Starred sections (*) are marked advanced in the book. In exercise lists, the star column shows a star on the whole exercise, or names the starred parts when only parts are starred ((**) means harder). Blank pages: 18, 30, 32, 114, 116, 284, 286, 374, 436.

Front matter: Preface vii-viii (PDF 7-8), Contents (PDF 9-17).

### Chapter 1: Introduction (PDF 19-29)

Sections:

- 1.1 What Is Learning?: 19
- 1.2 When Do We Need Machine Learning?: 21
- 1.3 Types of Learning: 22
- 1.4 Relations to Other Fields: 24
- 1.5 How to Read This Book: 25
  - 1.5.1 Possible Course Plans Based on This Book: 26
- 1.6 Notation: 27

Bibliographic remarks: none

Exercises: none (verified: Chapter 1 has no Exercises section; PDF 19-29 has no Exercises heading)

## Part I: Foundations (part title page PDF 31)

### Chapter 2: A Gentle Start (PDF 33-42)

Sections:

- 2.1 A Formal Model – The Statistical Learning Framework: 33
- 2.2 Empirical Risk Minimization: 35
  - 2.2.1 Something May Go Wrong – Overfitting: 35
- 2.3 Empirical Risk Minimization with Inductive Bias: 36
  - 2.3.1 Finite Hypothesis Classes: 37
- 2.4 Exercises: 41

Bibliographic remarks: none

Exercises: PDF 41-42, 3 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show the memorizing predictor of Eq. (2.3) is a thresholded polynomial, so ERM over thresholded polynomials can overfit. |
| 2 |  | Show the expected empirical risk of a fixed h over random samples equals its true risk. |
| 3 |  | Axis aligned rectangles (realizable): tightest enclosing rectangle is ERM, prove its PAC sample bound via four margin strips and a union bound, extend to R^d, show polynomial runtime. |

### Chapter 3: A Formal Learning Model (PDF 43-53)

Sections:

- 3.1 PAC Learning: 43
- 3.2 A More General Learning Model: 44
  - 3.2.1 Releasing the Realizability Assumption – Agnostic PAC Learning: 45
  - 3.2.2 The Scope of Learning Problems Modeled: 47
- 3.3 Summary: 49
- 3.4 Bibliographic Remarks: 50
- 3.5 Exercises: 50

Bibliographic remarks: PDF 50

Exercises: PDF 50-53, 9 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show the PAC sample complexity is nonincreasing in epsilon and in delta. |
| 2 |  | Singletons class: give an ERM algorithm in the realizable case and prove PAC learnability with a sample bound. |
| 3 |  | Concentric circles in the plane: prove PAC learnable with sample complexity at most log(1/delta)/epsilon. |
| 4 |  | Boolean conjunctions over d variables: prove PAC learnable, bound sample complexity, give a poly(d m) ERM algorithm. |
| 5 |  | Independent but non-identical instance distributions: bound the probability that a bad consistent hypothesis exists w.r.t. the average distribution (hint: AM-GM). |
| 6 |  | Show agnostic PAC learnability implies PAC learnability (same learner works). |
| 7 | * | Prove the Bayes optimal predictor minimizes 0-1 risk among all classifiers. |
| 8 | * | Bayes predictor also beats probabilistic predictors; for each D some algorithm is best; no algorithm is best for every D. |
| 9 | parts 1 (*), 2 (**) | Two-oracle PAC model (separate positive and negative samples): show it is equivalent to the standard model (second direction assumes all-plus and all-minus are in H). |

### Chapter 4: Learning via Uniform Convergence (PDF 54-59)

Sections:

- 4.1 Uniform Convergence Is Sufficient for Learnability: 54
- 4.2 Finite Classes Are Agnostic PAC Learnable: 55
- 4.3 Summary: 58
- 4.4 Bibliographic Remarks: 58
- 4.5 Exercises: 58

Bibliographic remarks: PDF 58

Exercises: PDF 58-59, 2 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show (epsilon, delta) convergence of the risk is equivalent to the expected risk tending to 0, for losses in [0,1]. |
| 2 |  | Bounded losses in [a,b]: derive the finite-class sample complexity with a (b-a)^2 factor. |

### Chapter 5: The Bias-Complexity Tradeoff (PDF 60-66)

Sections:

- 5.1 The No-Free-Lunch Theorem: 61
  - 5.1.1 No-Free-Lunch and Prior Knowledge: 63
- 5.2 Error Decomposition: 64
- 5.3 Summary: 65
- 5.4 Bibliographic Remarks: 66
- 5.5 Exercises: 66

Bibliographic remarks: PDF 66

Exercises: PDF 66, 3 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show Eq. (5.2) implies P[L_D(A(S)) >= 1/8] >= 1/7 (via Lemma B.1). |
| 2 |  | Heart attack prediction: discuss pros and cons of rectangles on 2 features vs 5 features, and how sample size affects the choice. |
| 3 |  | Strengthen the No-Free-Lunch lower bound from 1/4 to 1/2 - 1/(2k) when |X| >= km. |

### Chapter 6: The VC-Dimension (PDF 67-82)

Sections:

- 6.1 Infinite-Size Classes Can Be Learnable: 67
- 6.2 The VC-Dimension: 68
- 6.3 Examples: 70
  - 6.3.1 Threshold Functions: 70
  - 6.3.2 Intervals: 71
  - 6.3.3 Axis Aligned Rectangles: 71
  - 6.3.4 Finite Classes: 72
  - 6.3.5 VC-Dimension and the Number of Parameters: 72
- 6.4 The Fundamental Theorem of PAC learning: 72
- 6.5 Proof of Theorem 6.7: 73
  - 6.5.1 Sauer’s Lemma and the Growth Function: 73
  - 6.5.2 Uniform Convergence for Classes of Small Effective Size: 75
- 6.6 Summary: 78
- 6.7 Bibliographic remarks: 78
- 6.8 Exercises: 78

Bibliographic remarks: PDF 78

Exercises: PDF 78-82, 12 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show VC dimension is monotone under class inclusion. |
| 2 |  | Find and prove the VC dimension of 'exactly k ones' and 'at most k ones or at most k zeros' classes on a finite domain. |
| 3 |  | Find the VC dimension of parity functions on {0,1}^n. |
| 4 |  | Give examples where each of the two Sauer's lemma inequalities is strict or tight (all four combinations). |
| 5 |  | Prove axis aligned rectangles in R^d have VC dimension 2d. |
| 6 | part 4 (**) | Boolean conjunctions: size bound 3^d+1, VC <= d log 3, unit vectors shattered, VC <= d via pigeonhole, monotone conjunctions have VC = d. |
| 7 |  | Give an infinite class on [0,1] with VC dimension 1, and a finite class with VC = floor(log2 |H|). |
| 8 | * | Prove the one-parameter class x -> ceil(sin(theta x)) has infinite VC dimension. |
| 9 |  | Compute the VC dimension of signed intervals. |
| 10 |  | Show VC >= d forces expected excess error (d-m)/(2d) for some D, and conclude PAC learnable implies finite VC. |
| 11 | part 2 (*) | VC of a union of r classes: bound by 4d log(2d) + 2 log r; for r = 2 show at most 2d+1. |
| 12 | part 2 (**) | Dudley classes: VC of POS(F+g) equals VC of POS(F) equals linear dimension of F; apply to halfspaces, balls, polynomial classifiers. |

### Chapter 7: Nonuniform Learnability (PDF 83-99)

Sections:

- 7.1 Nonuniform Learnability: 83
  - 7.1.1 Characterizing Nonuniform Learnability: 84
- 7.2 Structural Risk Minimization: 85
- 7.3 Minimum Description Length and Occam’s Razor: 89
  - 7.3.1 Occam’s Razor: 91
- 7.4 Other Notions of Learnability – Consistency: 92
- 7.5 Discussing the Different Notions of Learnability: 93
  - 7.5.1 The No-Free-Lunch Theorem Revisited: 95
- 7.6 Summary: 96
- 7.7 Bibliographic Remarks: 97
- 7.8 Exercises: 97

Bibliographic remarks: PDF 97

Exercises: PDF 97-99, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Bound the VC dimension of a finite class by twice its max description length (or once, if prefix-free). |
| 2 |  | Show a countable class cannot get summable weights that are monotonically nondecreasing. |
| 3 | second bullet (*) | Build a weighting from n(h) and |H_n| for a union of finite classes; then for countable H_n. |
| 4 |  | Prove an oracle inequality for the MDL rule against the best hypothesis of description length at most B. |
| 5 |  | No-Free-Lunch for nonuniform learning: nonuniform learnability means countable union of finite-VC classes; build classes that are nonuniform but not PAC learnable, and not nonuniform learnable. |
| 6 |  | Show Memorize is a consistent learner over any countable domain (step by step via mass of unseen points). |

### Chapter 8: The Runtime of Learning (PDF 100-113)

Sections:

- 8.1 Computational Complexity of Learning: 101
  - 8.1.1 Formal Definition*: 102
- 8.2 Implementing the ERM Rule: 103
  - 8.2.1 Finite Classes: 104
  - 8.2.2 Axis Aligned Rectangles: 105
  - 8.2.3 Boolean Conjunctions: 106
  - 8.2.4 Learning 3-Term DNF: 107
- 8.3 Efficiently Learnable, but Not by a Proper ERM: 107
- 8.4 Hardness of Learning*: 108
- 8.5 Summary: 110
- 8.6 Bibliographic Remarks: 110
- 8.7 Exercises: 110

Bibliographic remarks: PDF 110

Exercises: PDF 110-113, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Give an O(m^2) agnostic ERM for intervals on the line (dynamic programming). |
| 2 |  | If realizable ERM depends on O(n) examples, get unrealizable ERM in time O(m n m^O(n)). |
| 3 |  | Prove ERM is NP-hard for halfspaces (reduction from Max FS) and for intersections of k >= 3 halfspaces (reduction from graph k-coloring). |
| 4 |  | Show efficient proper PAC learnability puts ERM in RP, so NP-hard ERM rules out efficient proper learning unless NP = RP. |

## Part II: From Theory to Algorithms (part title page PDF 115)

### Chapter 9: Linear Predictors (PDF 117-129)

Sections:

- 9.1 Halfspaces: 118
  - 9.1.1 Linear Programming for the Class of Halfspaces: 119
  - 9.1.2 Perceptron for Halfspaces: 120
  - 9.1.3 The VC Dimension of Halfspaces: 122
- 9.2 Linear Regression: 123
  - 9.2.1 Least Squares: 124
  - 9.2.2 Linear Regression for Polynomial Regression Tasks: 125
- 9.3 Logistic Regression: 126
- 9.4 Summary: 128
- 9.5 Bibliographic Remarks: 128
- 9.6 Exercises: 128

Bibliographic remarks: PDF 128

Exercises: PDF 128-129, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Write least absolute deviation regression as a linear program. |
| 2 |  | Show the matrix A of Eq. (9.6) is invertible iff the x_i span R^d. |
| 3 |  | Show the Perceptron bound of Theorem 9.1 is tight (m updates, using unit vectors). |
| 4 | * | Find a sequence in R^3 where the Perceptron bound equals m and m mistakes occur. |
| 5 |  | Show scaling the Perceptron update by eta changes neither the number of iterations nor the output direction. |
| 6 | part 2 (*) | VC dimension of balls in R^d: lift to halfspaces in R^(d+1) for an upper bound, find d+1 shattered points, conclude d+1 <= VC <= d+2. |

### Chapter 10: Boosting (PDF 130-143)

Sections:

- 10.1 Weak Learnability: 131
  - 10.1.1 Efficient Implementation of ERM for Decision Stumps: 133
- 10.2 AdaBoost: 134
- 10.3 Linear Combinations of Base Hypotheses: 137
  - 10.3.1 The VC-Dimension of L(B, T ): 139
- 10.4 AdaBoost for Face Recognition: 140
- 10.5 Summary: 141
- 10.6 Bibliographic Remarks: 141
- 10.7 Exercises: 142

Bibliographic remarks: PDF 141-142

Exercises: PDF 142-143, 5 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Boosting the confidence: turn a fixed-confidence learner into an agnostic PAC learner via k chunks plus a validation chunk. |
| 2 |  | Show the function h of Eq. (10.5) equals the piecewise constant function with the same thresholds. |
| 3 |  | Show h_t has error exactly 1/2 under the next AdaBoost weights D^(t+1). |
| 4 |  | VC of L(B,T): a class where it does not grow with T; decision stumps in R^d have VC between log d and 5 + 2 log d; L(B_d,T) has VC >= 0.5 T log d. |
| 5 |  | Integral image: compute in linear time, then evaluate any Viola-Jones feature in constant time. |

### Chapter 11: Model Selection and Validation (PDF 144-155)

Sections:

- 11.1 Model Selection Using SRM: 145
- 11.2 Validation: 146
  - 11.2.1 Hold Out Set: 146
  - 11.2.2 Validation for Model Selection: 147
  - 11.2.3 The Model-Selection Curve: 148
  - 11.2.4 k-Fold Cross Validation: 149
  - 11.2.5 Train-Validation-Test Split: 150
- 11.3 What to Do If Learning Fails: 151
- 11.4 Summary: 154
- 11.5 Exercises: 154

Bibliographic remarks: none

Exercises: PDF 154-155, 2 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show leave-one-out fails by exactly 1/2 for a label-parity learner on random labels. |
| 2 |  | Compare ERM over a union of classes vs model selection by validation: when is each better? |

### Chapter 12: Convex Learning Problems (PDF 156-170)

Sections:

- 12.1 Convexity, Lipschitzness, and Smoothness: 156
  - 12.1.1 Convexity: 156
  - 12.1.2 Lipschitzness: 160
  - 12.1.3 Smoothness: 162
- 12.2 Convex Learning Problems: 163
  - 12.2.1 Learnability of Convex Learning Problems: 164
  - 12.2.2 Convex-Lipschitz/Smooth-Bounded Learning Problems: 166
- 12.3 Surrogate Loss Functions: 167
- 12.4 Summary: 168
- 12.5 Bibliographic Remarks: 169
- 12.6 Exercises: 169

Bibliographic remarks: PDF 169

Exercises: PDF 169-170, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Construct a sample where the 0-1 loss has a local minimum that is not global. |
| 2 |  | Show logistic regression is convex-Lipschitz-bounded and convex-smooth-bounded; give the constants. |
| 3 |  | Show the hinge loss over a radius-R ball is R-Lipschitz. |
| 4 | * | Convexity is not enough for efficiency: a convex-Lipschitz-bounded problem over Turing machines that no computable learner can learn. |

### Chapter 13: Regularization and Stability (PDF 171-183)

Sections:

- 13.1 Regularized Loss Minimization: 171
  - 13.1.1 Ridge Regression: 172
- 13.2 Stable Rules Do Not Overfit: 173
- 13.3 Tikhonov Regularization as a Stabilizer: 174
  - 13.3.1 Lipschitz Loss: 176
  - 13.3.2 Smooth and Nonnegative Loss: 177
- 13.4 Controlling the Fitting-Stability Tradeoff: 178
- 13.5 Summary: 180
- 13.6 Bibliographic Remarks: 180
- 13.7 Exercises: 181

Bibliographic remarks: PDF 180-181

Exercises: PDF 181-183, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | From a bound on expected risk to agnostic PAC: Markov's inequality, then a k-chunk-plus-validation procedure. |
| 2 |  | Learnability without uniform convergence: a masked center-of-mass problem learnable by RLM but whose uniform convergence rate grows with d. |
| 3 |  | Prove Theorem 13.12: on-average-replace-one stability plus asymptotic ERM implies learning. |
| 4 | parts 2 and 4 (*) | Strong convexity under general norms: which items of Lemma 13.5 survive, a counterexample norm, stability of RLM, and an l_q regularizer that is strongly convex w.r.t. l1. |

### Chapter 14: Stochastic Gradient Descent (PDF 184-201)

Sections:

- 14.1 Gradient Descent: 185
  - 14.1.1 Analysis of GD for Convex-Lipschitz Functions: 186
- 14.2 Subgradients: 188
  - 14.2.1 Calculating Subgradients: 189
  - 14.2.2 Subgradients of Lipschitz Functions: 190
  - 14.2.3 Subgradient Descent: 190
- 14.3 Stochastic Gradient Descent (SGD): 191
  - 14.3.1 Analysis of SGD for Convex-Lipschitz-Bounded Functions: 191
- 14.4 Variants: 193
  - 14.4.1 Adding a Projection Step: 193
  - 14.4.2 Variable Step Size: 194
  - 14.4.3 Other Averaging Techniques: 195
  - 14.4.4 Strongly Convex Functions*: 195
- 14.5 Learning with SGD: 196
  - 14.5.1 SGD for Risk Minimization: 196
  - 14.5.2 Analyzing SGD for Convex-Smooth Learning Problems: 198
  - 14.5.3 SGD for Regularized Loss Minimization: 199
- 14.6 Summary: 200
- 14.7 Bibliographic Remarks: 200
- 14.8 Exercises: 201

Bibliographic remarks: PDF 200-201

Exercises: PDF 201, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Prove Claim 14.10 (extend the proof of Lemma 13.5). |
| 2 |  | Prove Corollary 14.14. |
| 3 |  | Derive the Perceptron as subgradient descent on max_i (1 - y_i <w,x_i>) and compare to Batch Perceptron. |
| 4 | * | Prove an analog of Theorem 14.8 for SGD with step size eta_t = B/(rho sqrt(t)). |

### Chapter 15: Support Vector Machines (PDF 202-214)

Sections:

- 15.1 Margin and Hard-SVM: 202
  - 15.1.1 The Homogenous Case: 205
  - 15.1.2 The Sample Complexity of Hard-SVM: 205
- 15.2 Soft-SVM and Norm Regularization: 206
  - 15.2.1 The Sample Complexity of Soft-SVM: 208
  - 15.2.2 Margin and Norm-Based Bounds versus Dimension: 208
  - 15.2.3 The Ramp Loss*: 209
- 15.3 Optimality Conditions and “Support Vectors”*: 210
- 15.4 Duality*: 211
- 15.5 Implementing Soft-SVM Using SGD: 212
- 15.6 Summary: 213
- 15.7 Bibliographic Remarks: 213
- 15.8 Exercises: 214

Bibliographic remarks: PDF 213

Exercises: PDF 214, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show two formulations of the hard-SVM rule are equivalent. |
| 2 |  | Bound Batch Perceptron updates by (rho/gamma)^2 for margin gamma and radius rho. |
| 3 |  | Prove or refute: some lambda makes soft-SVM equal hard-SVM on every separable sample. |
| 4 |  | Prove weak duality: min max f >= max min f. |

### Chapter 16: Kernel Methods (PDF 215-226)

Sections:

- 16.1 Embeddings into Feature Spaces: 215
- 16.2 The Kernel Trick: 217
  - 16.2.1 Kernels as a Way to Express Prior Knowledge: 221
  - 16.2.2 Characterizing Kernel Functions*: 222
- 16.3 Implementing Soft-SVM with Kernels: 222
- 16.4 Summary: 224
- 16.5 Bibliographic Remarks: 225
- 16.6 Exercises: 225

Bibliographic remarks: PDF 225

Exercises: PDF 225-226, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show each substring-detector in the class is a unit-norm linear classifier over psi(x) with margin 1. |
| 2 |  | Kernelize the Perceptron. |
| 3 |  | Kernel ridge regression: reduce to the Gram matrix objective and find the closed form for alpha. |
| 4 |  | Show K(x,x') = min{x,x'} on {1..N} is a valid kernel by giving the feature map. |
| 5 |  | Supermarket baby-detection: design a learner polynomial in s, k, 1/epsilon (a k-item disjunction). |
| 6 |  | Closest-class-mean classifier: write it as a halfspace and express it via the kernel only. |

### Chapter 17: Multiclass, Ranking, and Complex Prediction Problems (PDF 227-249)

Sections:

- 17.1 One-versus-All and All-Pairs: 227
- 17.2 Linear Multiclass Predictors: 230
  - 17.2.1 How to Construct Ψ: 230
  - 17.2.2 Cost-Sensitive Classification: 232
  - 17.2.3 ERM: 232
  - 17.2.4 Generalized Hinge Loss: 233
  - 17.2.5 Multiclass SVM and SGD: 234
- 17.3 Structured Output Prediction: 236
- 17.4 Ranking: 238
  - 17.4.1 Linear Predictors for Ranking: 240
- 17.5 Bipartite Ranking and Multivariate Performance Measures: 243
  - 17.5.1 Linear Predictors for Bipartite Ranking: 245
- 17.6 Summary: 247
- 17.7 Bibliographic Remarks: 247
- 17.8 Exercises: 248

Bibliographic remarks: PDF 247-248

Exercises: PDF 248-249, 5 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show well-separated class balls make the multiclass multivector construction realizable with zero loss. |
| 2 |  | Prove the Multiclass Batch Perceptron stops within (R ||w*||)^2 iterations (Theorem 17.5). |
| 3 |  | Generalize the Section 17.3 dynamic program to the loss-augmented maximization used by multiclass SGD. |
| 4 |  | Prove Equation (17.7). |
| 5 |  | Show the two definitions of pi (Eqs. 17.12 and 17.13) agree for all multivariate performance measures. |

### Chapter 18: Decision Trees (PDF 250-257)

Sections:

- 18.1 Sample Complexity: 251
- 18.2 Decision Tree Algorithms: 252
  - 18.2.1 Implementations of the Gain Measure: 253
  - 18.2.2 Pruning: 254
  - 18.2.3 Threshold-Based Splitting Rules for Real-Valued Features: 255
- 18.3 Random Forests: 255
- 18.4 Summary: 256
- 18.5 Bibliographic Remarks: 256
- 18.6 Exercises: 256

Bibliographic remarks: PDF 256

Exercises: PDF 256-257, 2 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Any Boolean function on {0,1}^d is a decision tree of height d+1, so the VC dimension of such trees is 2^d. |
| 2 |  | Suboptimality of ID3: on a given 4-example set depth-2 ID3 has training error >= 1/4, yet some depth-2 tree has zero error. |

### Chapter 19: Nearest Neighbor (PDF 258-267)

Sections:

- 19.1 k Nearest Neighbors: 258
- 19.2 Analysis: 259
  - 19.2.1 A Generalization Bound for the 1-NN Rule: 260
  - 19.2.2 The “Curse of Dimensionality”: 263
- 19.3 Efficient Implementation*: 264
- 19.4 Summary: 264
- 19.5 Bibliographic Remarks: 264
- 19.6 Exercises: 265

Bibliographic remarks: PDF 264-265

Exercises: PDF 265-267, 4 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Prove Lemma 19.6: expected mass of cells hit by fewer than k sample points is at most 2rk/m. |
| 2 |  | Prove Lemma 19.7: majority of k Bernoulli draws is at most (1 + sqrt(8/k)) times worse than the Bayes rule on p. |
| 3 |  | Show P_{y~p}[y != y'] <= P_{y~p'}[y != y'] + |p - p'|. |
| 4 |  | Combine the lemmas to finish the k-NN generalization bound (Theorem 19.5). |

### Chapter 20: Neural Networks (PDF 268-283)

Sections:

- 20.1 Feedforward Neural Networks: 269
- 20.2 Learning Neural Networks: 270
- 20.3 The Expressive Power of Neural Networks: 271
  - 20.3.1 Geometric Intuition: 273
- 20.4 The Sample Complexity of Neural Networks: 274
- 20.5 The Runtime of Learning Neural Networks: 276
- 20.6 SGD and Backpropagation: 277
- 20.7 Summary: 281
- 20.8 Bibliographic Remarks: 281
- 20.9 Exercises: 282

Bibliographic remarks: PDF 281-282

Exercises: PDF 282-283, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Universal approximation: build a sigmoid network approximating any Lipschitz function on [-1,1]^n. |
| 2 |  | Prove Theorem 20.5 (some Lipschitz functions need exponentially large networks). |
| 3 |  | Growth function of a product class is at most the product of growth functions. |
| 4 |  | Growth function of a composition class is at most the product of growth functions. |
| 5 |  | Build a sigmoidal network with O(n) weights and VC dimension n^2 (so Omega(|E|^2)). |
| 6 |  | Prove Theorem 20.7 (hardness of learning networks), via the intersection-of-halfspaces reduction. |

## Part III: Additional Learning Models (part title page PDF 285)

### Chapter 21: Online Learning (PDF 287-306)

Sections:

- 21.1 Online Classification in the Realizable Case: 288
  - 21.1.1 Online Learnability: 290
- 21.2 Online Classification in the Unrealizable Case: 294
  - 21.2.1 Weighted-Majority: 295
- 21.3 Online Convex Optimization: 300
- 21.4 The Online Perceptron Algorithm: 301
- 21.5 Summary: 304
- 21.6 Bibliographic Remarks: 305
- 21.7 Exercises: 305

Bibliographic remarks: PDF 305

Exercises: PDF 305-306, 5 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Find a class and sequence on which Consistent makes |H| - 1 mistakes. |
| 2 |  | Find a class and sequence on which the Halving mistake bound is tight. |
| 3 |  | Compute exactly the Halving mistake bound for the singletons class on {1..d}. |
| 4 |  | Doubling trick: remove the dependence on the horizon T at a constant-factor regret cost. |
| 5 |  | Online-to-batch: a mistake bound M gives expected risk M/T for a randomly chosen round hypothesis. |

### Chapter 22: Clustering (PDF 307-322)

Sections:

- 22.1 Linkage-Based Clustering Algorithms: 310
- 22.2 k-Means and Other Cost Minimization Clusterings: 311
  - 22.2.1 The k-Means Algorithm: 313
- 22.3 Spectral Clustering: 315
  - 22.3.1 Graph Cut: 315
  - 22.3.2 Graph Laplacian and Relaxed Graph Cuts: 315
  - 22.3.3 Unnormalized Spectral Clustering: 317
- 22.4 Information Bottleneck*: 317
- 22.5 A High Level View of Clustering: 318
- 22.6 Summary: 320
- 22.7 Bibliographic Remarks: 320
- 22.8 Exercises: 320

Bibliographic remarks: PDF 320

Exercises: PDF 320-322, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show k-means can return a solution at least t times worse than optimal, for any t. |
| 2 |  | Show k-means can converge to a point that is not a local minimum (points 1..4, centers 2 and 4). |
| 3 |  | Prove farthest-first traversal is a 2-approximation for the k-diameter objective. |
| 4 |  | Show k-diameter clustering is not a center-based clustering function. |
| 5 |  | Which of scale invariance, richness, consistency does single linkage satisfy under each stopping rule; any pair is achievable. |
| 6 |  | Show some clustering function satisfies scale invariance, k-richness and consistency. |

### Chapter 23: Dimensionality Reduction (PDF 323-341)

Sections:

- 23.1 Principal Component Analysis (PCA): 324
  - 23.1.1 A More Efficient Solution for the Case d >> m: 326
  - 23.1.2 Implementation and Demonstration: 326
- 23.2 Random Projections: 329
- 23.3 Compressed Sensing: 330
  - 23.3.1 Proofs*: 333
- 23.4 PCA or Compressed Sensing?: 338
- 23.5 Summary: 338
- 23.6 Bibliographic Remarks: 339
- 23.7 Exercises: 339

Bibliographic remarks: PDF 339

Exercises: PDF 339-341, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show exact recovery from linear compression with n < d is impossible in general. |
| 2 |  | Show the max of sum alpha_j beta_j over beta in [0,1]^d with ||beta||_1 <= n is the sum of the top n alphas. |
| 3 |  | Kernel PCA: show it runs in time polynomial in m and n using only kernel evaluations. |
| 4 |  | PCA as variance maximization: first and second principal components. |
| 5 |  | Reprove Theorem 23.2 (PCA solution) using the SVD. |
| 6 | part 2 (*) | Random projections preserve inner products in the unit ball; and preserve margin separability (gamma to gamma/2) with n = c/gamma^2. |

### Chapter 24: Generative Models (PDF 342-356)

Sections:

- 24.1 Maximum Likelihood Estimator: 343
  - 24.1.1 Maximum Likelihood Estimation for Continuous Random Variables: 344
  - 24.1.2 Maximum Likelihood and Empirical Risk Minimization: 345
  - 24.1.3 Generalization Analysis: 345
- 24.2 Naive Bayes: 347
- 24.3 Linear Discriminant Analysis: 347
- 24.4 Latent Variables and the EM Algorithm: 348
  - 24.4.1 EM as an Alternate Maximization Algorithm: 350
  - 24.4.2 EM for Mixture of Gaussians (Soft k-Means): 352
- 24.5 Bayesian Reasoning: 353
- 24.6 Summary: 355
- 24.7 Bibliographic Remarks: 355
- 24.8 Exercises: 356

Bibliographic remarks: PDF 355

Exercises: PDF 356, 3 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Prove the maximum likelihood variance estimator of a Gaussian is biased. |
| 2 |  | Regularized ML for a Bernoulli parameter equals adding two pseudo-examples; bound |theta_hat - theta*| and the true risk. |
| 3 |  | Show the soft k-means M step weight problem is solved by normalizing nu (via relative entropy). |

### Chapter 25: Feature Selection and Generation (PDF 357-372)

Sections:

- 25.1 Feature Selection: 358
  - 25.1.1 Filters: 359
  - 25.1.2 Greedy Selection Approaches: 360
  - 25.1.3 Sparsity-Inducing Norms: 363
- 25.2 Feature Manipulation and Normalization: 365
  - 25.2.1 Examples of Feature Transformations: 367
- 25.3 Feature Learning: 368
  - 25.3.1 Dictionary Learning Using Auto-Encoders: 368
- 25.4 Summary: 370
- 25.5 Bibliographic Remarks: 371
- 25.6 Exercises: 371

Bibliographic remarks: PDF 371

Exercises: PDF 371-372, 3 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Prove the equality in Equation (25.1). |
| 2 |  | Show Equation (25.7) solves Equation (25.6). |
| 3 |  | AdaBoost as forward greedy selection: gradient of the exponential loss is 2 eps_j - 1, and each step decreases R by log sqrt(1 - 4 gamma^2). |

## Part IV: Advanced Theory (part title page PDF 373)

### Chapter 26: Rademacher Complexities (PDF 375-387)

Sections:

- 26.1 The Rademacher Complexity: 375
  - 26.1.1 Rademacher Calculus: 379
- 26.2 Rademacher Complexity of Linear Classes: 382
- 26.3 Generalization Bounds for SVM: 383
- 26.4 Generalization Bounds for Predictors with Low l1 Norm: 386
- 26.5 Bibliographic Remarks: 386

Bibliographic remarks: PDF 386-387

Exercises: none (no Exercises section in this chapter)

### Chapter 27: Covering Numbers (PDF 388-391)

Sections:

- 27.1 Covering: 388
  - 27.1.1 Properties: 388
- 27.2 From Covering to Rademacher Complexity via Chaining: 389
- 27.3 Bibliographic Remarks: 391

Bibliographic remarks: PDF 391

Exercises: none (no Exercises section in this chapter)

### Chapter 28: Proof of the Fundamental Theorem of Learning Theory (PDF 392-401)

Sections:

- 28.1 The Upper Bound for the Agnostic Case: 392
- 28.2 The Lower Bound for the Agnostic Case: 393
  - 28.2.1 Showing That m(eps, delta) >= 0.5 log(1/(4 delta))/eps^2: 393
  - 28.2.2 Showing That m(eps, 1/8) >= 8d/eps^2: 395
- 28.3 The Upper Bound for the Realizable Case: 398
  - 28.3.1 From eps-Nets to PAC Learnability: 401

Bibliographic remarks: none

Exercises: none (no Exercises section in this chapter)

### Chapter 29: Multiclass Learnability (PDF 402-409)

Sections:

- 29.1 The Natarajan Dimension: 402
- 29.2 The Multiclass Fundamental Theorem: 403
  - 29.2.1 On the Proof of Theorem 29.3: 403
- 29.3 Calculating the Natarajan Dimension: 404
  - 29.3.1 One-versus-All Based Classes: 404
  - 29.3.2 General Multiclass-to-Binary Reductions: 405
  - 29.3.3 Linear Multiclass Predictors: 405
- 29.4 On Good and Bad ERMs: 406
- 29.5 Bibliographic Remarks: 408
- 29.6 Exercises: 409

Bibliographic remarks: PDF 408

Exercises: PDF 409, 6 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | Show a binary class of VC dimension d whose one-versus-all extension has Natarajan dimension d. |
| 2 |  | Prove Lemma 29.6. |
| 3 |  | Prove Natarajan's lemma (induction over a removed point). |
| 4 |  | Prove the agnostic multiclass upper bound C (d log(kd/eps) + log(1/delta))/eps^2. |
| 5 |  | Prove the agnostic multiclass lower bound C (d + log(1/delta))/eps^2. |
| 6 |  | Show one-versus-all over halfspaces in R^d has Natarajan dimension at least (d-1)(k-1). |

### Chapter 30: Compression Bounds (PDF 410-414)

Sections:

- 30.1 Compression Bounds: 410
- 30.2 Examples: 412
  - 30.2.1 Axis Aligned Rectangles: 412
  - 30.2.2 Halfspaces: 412
  - 30.2.3 Separating Polynomials: 413
  - 30.2.4 Separation with Margin: 414
- 30.3 Bibliographic Remarks: 414

Bibliographic remarks: PDF 414

Exercises: none (no Exercises section in this chapter)

### Chapter 31: PAC-Bayes (PDF 415-418)

Sections:

- 31.1 PAC-Bayes Bounds: 415
- 31.2 Bibliographic Remarks: 417
- 31.3 Exercises: 417

Bibliographic remarks: PDF 417

Exercises: PDF 417-418, 2 exercises

| # | Star | Asks |
|---|------|------|
| 1 |  | If P[X >= eps] <= exp(-2 m eps^2), show E[exp(2(m-1)X^2)] <= m. |
| 2 |  | PAC-Bayes with uniform prior and point-mass posterior on a finite class; compare to uniform convergence and derive an Occam-like bound. |

## Appendices

### Appendix A: Technical Lemmas (PDF 419-421)

No numbered sections (a list of lemmas A.1 onward). No exercises.

### Appendix B: Measure Concentration (PDF 422-429)

- B.1 Markov's Inequality: 422
- B.2 Chebyshev's Inequality: 423
- B.3 Chernoff's Bounds: 423
- B.4 Hoeffding's Inequality: 425
- B.5 Bennet's and Bernstein's Inequalities: 426
  - B.5.1 Application: 427
- B.6 Slud's Inequality: 428
- B.7 Concentration of Chi-squared Variables: 428

No exercises.

### Appendix C: Linear Algebra (PDF 430-434)

- C.1 Basic Definitions: 430
- C.2 Eigenvalues and Eigenvectors: 431
- C.3 Positive definite matrices: 431
- C.4 Singular Value Decomposition (SVD): 431

No exercises.

## Back matter

- Notes: PDF 435 (a blank page titled Notes for the reader; 436 is blank). The book has no end-notes section; per-chapter commentary lives in each chapter's Bibliographic Remarks.
- References: PDF 437-446
- Index: PDF 447-449

## Bibliographic remarks quick index

| Chapter | PDF |
|---|---|
| 1 | none |
| 2 | none |
| 3 | 50 |
| 4 | 58 |
| 5 | 66 |
| 6 | 78 |
| 7 | 97 |
| 8 | 110 |
| 9 | 128 |
| 10 | 141-142 |
| 11 | none |
| 12 | 169 |
| 13 | 180-181 |
| 14 | 200-201 |
| 15 | 213 |
| 16 | 225 |
| 17 | 247-248 |
| 18 | 256 |
| 19 | 264-265 |
| 20 | 281-282 |
| 21 | 305 |
| 22 | 320 |
| 23 | 339 |
| 24 | 355 |
| 25 | 371 |
| 26 | 386-387 |
| 27 | 391 |
| 28 | none |
| 29 | 408 |
| 30 | 414 |
| 31 | 417 |

## Exercise counts

| Chapter | Exercises | PDF |
|---|---|---|
| 1 | 0 | - |
| 2 | 3 | 41-42 |
| 3 | 9 | 50-53 |
| 4 | 2 | 58-59 |
| 5 | 3 | 66 |
| 6 | 12 | 78-82 |
| 7 | 6 | 97-99 |
| 8 | 4 | 110-113 |
| 9 | 6 | 128-129 |
| 10 | 5 | 142-143 |
| 11 | 2 | 154-155 |
| 12 | 4 | 169-170 |
| 13 | 4 | 181-183 |
| 14 | 4 | 201 |
| 15 | 4 | 214 |
| 16 | 6 | 225-226 |
| 17 | 5 | 248-249 |
| 18 | 2 | 256-257 |
| 19 | 4 | 265-267 |
| 20 | 6 | 282-283 |
| 21 | 5 | 305-306 |
| 22 | 6 | 320-322 |
| 23 | 6 | 339-341 |
| 24 | 3 | 356 |
| 25 | 3 | 371-372 |
| 26 | 0 | - |
| 27 | 0 | - |
| 28 | 0 | - |
| 29 | 6 | 409 |
| 30 | 0 | - |
| 31 | 2 | 417-418 |
| Total | 122 | |
