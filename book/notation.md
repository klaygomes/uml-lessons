# Notation reference

Source: Shalev-Shwartz and Ben-David, *Understanding Machine Learning* (2014). Page numbers are the printed ones, which equal the PDF page numbers. Meanings are paraphrased.

## Conventions from Section 1.6 (p. 27)

- Scalars and abstract objects are lowercase (\(x\), \(\lambda\)). Vectors are boldface lowercase when the vector nature matters. The \(i\)th element of \(\mathbf x\) is \(x_i\).
- Uppercase letters denote matrices, sets and sequences; context decides which.
- \(z\) is an abstract example and \(S = z_1, \dots, z_m\) is a sequence of \(m\) examples. The book notes that \(S\) is "often referred to as a training set" but always treats it as a sequence, not a set.
- \(\mathcal D\) is a distribution over some set (for example \(\mathcal Z\)). Measurability details (\(\sigma\)-algebras) are relegated to footnotes that the reader may skip (p. 27, footnote 2).
- \(\mathcal D^m\) is the distribution over \(\mathcal Z^m\) obtained by drawing each of \(z_1, \dots, z_m\) from \(\mathcal D\) independently (announced p. 27, defined in Chapter 2, p. 38).
- Asymptotic notation is avoided where possible (p. 27 to 28). The book writes \(\min\) and \(\max\) even where \(\inf\) and \(\sup\) would be more precise (p. 29).

## Table 1.1, Summary of notation (p. 28)

| Symbol | Meaning | Page |
|---|---|---|
| \(\mathbb R\) | real numbers | 28 |
| \(\mathbb R^d\) | vectors of length \(d\) with real entries | 28 |
| \(\mathbb R_+\) | non-negative reals | 28 |
| \(\mathbb N\) | natural numbers | 28 |
| \(O, o, \Theta, \omega, \Omega, \tilde O\) | asymptotic notation, defined in the text | 28 to 29 |
| \(\mathbb 1[\text{Boolean expression}]\) | indicator: 1 when the expression holds, 0 otherwise | 28 |
| \([a]_+\) | \(\max\{0, a\}\) | 28 |
| \([n]\) | the set \(\{1, \dots, n\}\) for \(n \in \mathbb N\) | 28 |
| \(\mathbf x, \mathbf v, \mathbf w\) | column vectors | 28 |
| \(x_i, v_i, w_i\) | the \(i\)th entry of a vector | 28 |
| \(\langle \mathbf x, \mathbf v \rangle\) | inner product, \(\sum_{i=1}^d x_i v_i\) | 28 |
| \(\lVert \mathbf x \rVert_2\) or \(\lVert \mathbf x \rVert\) | Euclidean (\(\ell_2\)) norm, \(\sqrt{\langle \mathbf x, \mathbf x \rangle}\) | 28 |
| \(\lVert \mathbf x \rVert_1\) | \(\ell_1\) norm, sum of absolute entries | 28 |
| \(\lVert \mathbf x \rVert_\infty\) | \(\ell_\infty\) norm, largest absolute entry | 28 |
| \(\lVert \mathbf x \rVert_0\) | count of nonzero entries | 28 |
| \(A \in \mathbb R^{d,k}\) | a real \(d \times k\) matrix | 28 |
| \(A^\top\) | transpose of \(A\) | 28 |
| \(A_{i,j}\) | entry in row \(i\), column \(j\) | 28 |
| \(\mathbf x \mathbf x^\top\) | the \(d \times d\) matrix with entries \(x_i x_j\) | 28 |
| \(\mathbf x_1, \dots, \mathbf x_m\) | a sequence of \(m\) vectors | 28 |
| \(x_{i,j}\) | entry \(j\) of the \(i\)th vector in the sequence | 28 |
| \(\mathbf w^{(1)}, \dots, \mathbf w^{(T)}\) | successive values of \(\mathbf w\) in an iterative algorithm | 28 |
| \(w_i^{(t)}\) | entry \(i\) of \(\mathbf w^{(t)}\) | 28 |
| \(\mathcal X\) | instances domain (a set) | 28 |
| \(\mathcal Y\) | labels domain (a set) | 28 |
| \(\mathcal Z\) | examples domain (a set) | 28 |
| \(\mathcal H\) | hypothesis class (a set) | 28 |
| \(\ell : \mathcal H \times \mathcal Z \to \mathbb R_+\) | loss function | 28 |
| \(\mathcal D\) | a distribution, usually over \(\mathcal Z\) or \(\mathcal X\) | 28 |
| \(\mathcal D(A)\) | probability that \(\mathcal D\) gives to a set \(A \subseteq \mathcal Z\) | 28 |
| \(z \sim \mathcal D\) | \(z\) is drawn according to \(\mathcal D\) | 28 |
| \(S = z_1, \dots, z_m\) | a sequence of \(m\) examples | 28 |
| \(S \sim \mathcal D^m\) | the \(m\) examples are drawn i.i.d. from \(\mathcal D\) | 28 |
| \(\mathbb P, \mathbb E\) | probability and expectation of a random variable | 28 |
| \(\mathbb P_{z \sim \mathcal D}[f(z)]\) | \(\mathcal D(\{z : f(z) = \text{true}\})\) for a Boolean \(f\) on \(\mathcal Z\) | 28 |
| \(\mathbb E_{z \sim \mathcal D}[f(z)]\) | expected value of a real-valued \(f\) on \(\mathcal Z\) | 28 |
| \(\mathcal N(\mu, C)\) | Gaussian with mean \(\mu\) and covariance \(C\) | 28 |
| \(f'(x)\), \(f''(x)\) | first and second derivative of \(f : \mathbb R \to \mathbb R\) at \(x\) | 28 |
| \(\frac{\partial f(\mathbf w)}{\partial w_i}\) | partial derivative of \(f : \mathbb R^d \to \mathbb R\) at \(\mathbf w\) along \(w_i\) | 28 |
| \(\nabla f(\mathbf w)\) | gradient of \(f\) at \(\mathbf w\) | 28 |
| \(\partial f(\mathbf w)\) | differential set of \(f\) at \(\mathbf w\) | 28 |
| \(\min_{x \in C} f(x)\), \(\max_{x \in C} f(x)\) | smallest or largest value of \(f\) on \(C\) | 28 |
| \(\operatorname{argmin}_{x \in C} f(x)\), \(\operatorname{argmax}_{x \in C} f(x)\) | the set of points of \(C\) where \(f\) reaches its min or max | 28 |
| \(\log\) | natural logarithm | 28 |

## Chapter 2 terms and symbols (pp. 33 to 38)

| Term or symbol | Meaning | Page |
|---|---|---|
| domain set \(\mathcal X\) | any set of objects we may want to label; its points are also called instances, and \(\mathcal X\) the instance space | 33 |
| label set \(\mathcal Y\) | for now a two-element set, usually \(\{0, 1\}\) or \(\{-1, +1\}\) | 33 |
| training data \(S = ((x_1, y_1), \dots, (x_m, y_m))\) | a finite sequence of labeled points in \(\mathcal X \times \mathcal Y\), the only input the learner sees; its pairs are training examples | 33 to 34 |
| training set | a name the book sometimes uses for \(S\); a footnote stresses \(S\) is still a sequence, so repeats and order can matter | 34 |
| learner's output \(h : \mathcal X \to \mathcal Y\) | a prediction rule, also called a predictor, a hypothesis or a classifier | 34 |
| \(A(S)\) | the hypothesis that learning algorithm \(A\) returns on training sequence \(S\) | 34 |
| \(\mathcal D\) (data-generation model) | an unknown probability distribution over \(\mathcal X\) that produces the instances; it may be arbitrary | 34 |
| \(f : \mathcal X \to \mathcal Y\) | the unknown "correct" labeling function, with \(y_i = f(x_i)\) for all \(i\) (relaxed in Chapter 3) | 34 |
| \(\mathbb P_{x \sim \mathcal D}[\pi(x)]\) | shorthand for \(\mathcal D(A)\) when the event \(A\) is given by \(\pi : \mathcal X \to \{0, 1\}\) | 34 |
| \(L_{\mathcal D, f}(h)\), also \(L_{(\mathcal D, f)}(h)\) | error of \(h\): the probability under \(\mathcal D\) of drawing \(x\) with \(h(x) \neq f(x)\) (eq. 2.1); synonyms: generalization error, risk, true error | 34 to 35 |
| \(L_{\mathcal D}(h)\) | the same error with the subscript \(f\) dropped when context makes it clear | 35, used on 36 |
| \(h_S\) | a predictor output on \(S\); the subscript marks its dependence on \(S\) | 35 |
| \(L_S(h)\) | training error, \(\lvert\{i \in [m] : h(x_i) \neq y_i\}\rvert / m\) (eq. 2.2); synonyms: empirical error, empirical risk | 35 |
| \([m]\) | \(\{1, \dots, m\}\) | 35 |
| ERM | Empirical Risk Minimization: output a predictor that minimizes \(L_S(h)\) | 35 |
| overfitting | fitting the training data "too well" while doing badly on \(\mathcal D\) | 36 |
| hypothesis class \(\mathcal H\) | a set of predictors fixed before seeing data | 36 |
| \(\mathrm{ERM}_{\mathcal H}(S) \in \operatorname{argmin}_{h \in \mathcal H} L_S(h)\) | ERM restricted to \(\mathcal H\) | 36 to 37 |
| inductive bias | the restriction to \(\mathcal H\), ideally chosen from prior knowledge | 37 |
| realizability (Definition 2.1) | some \(h^\star \in \mathcal H\) has \(L_{(\mathcal D, f)}(h^\star) = 0\); then \(L_S(h^\star) = 0\) with probability 1 | 38 |
| i.i.d. assumption, \(S \sim \mathcal D^m\) | each \(x_i\) is drawn fresh from \(\mathcal D\), independently of the others, then labeled by \(f\); \(\mathcal D^m\) is the induced distribution on \(m\)-tuples | 38 |
| \(\delta\), \(1 - \delta\) | probability of a nonrepresentative sample, and the confidence parameter | 38 |
