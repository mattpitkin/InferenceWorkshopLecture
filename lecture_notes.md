latex input:	mmd-beamer-header-11pt
Title:		Parameter estimation in gravitational-wave astronomy
Date:		14 May 2019
Author:		Matthew Pitkin
Affiliation:	University of Glasgow
LaTeX xslt:	beamer
latex mode:	beamer
Theme:		m
Event:		PyCBC Inference Workshop
latex input:	mmd-beamer-begin-doc
latex footer:	mmd-beamer-footer

<!--
% Lecture notes of parameter estimation for the PyCBC Inference Workshop
%
% Note: comments can be included in the LaTeX file by surrounding them with html style comment
% blocks and a % sign
-->


### Overview ###

* Introduction
  * Rules of probability
  * Bayes' theorem
* Gravitational-wave inference
  * the likelihood function
  * the priors
  * parameter estimation examples
  * model selection
  * hierarchical inference


### Introduction ###

There are many textbooks on statistics (and Bayesian statistics in particular), but three I can 
recommend are:

<!--\begin{columns}[T]
\begin{column}{0.33\textwidth}-->
![][Sivia]
<!--\end{column}
\begin{column}{0.33\textwidth}-->
![][Gregory]
<!--\end{column}
\begin{column}{0.33\textwidth}-->
![][MacKay]
<!--\end{column}
\end{columns}-->

[Sivia]: figures/sivia.jpg "Sivia" height="140px" width="105px"
[Gregory]: figures/gregory.jpg "Gregory" height="140px" width="105px"
[MacKay]: figures/mackay.jpg "MacKay" height="140px" width="105px"


### Papers to read ###

LALInference paper [][#2015PhRvD..91d2003V] and PyCBC Inference paper [][#Biwer:2018osg].


### Rules of probability ###

The rules for probabilities of propositions are inherited from classical logic and Boolean algebra:

* _Law of Excluded Middle_ $P(A\text{ or not}(A)) = 1$
* _Law of Non-contradiction_ $P(A\text{ and not}(A)) = 0$
    * i.e. $P(A) + P(\text{not }A) = 1$ (the **sum rule**)
* _Association_
    * $P(A,[B,C]) = P([A,B],C)$
    * $P(A \text{ or } [B \text{ or }C]) = P([A \text{ or } B] \text{ or }C)$
* _Distribution_
    * $P(A,[B\text{ or }C]) = P(A,B\text{ or }A,C)$
    * $P(A\text{ or }[B,C]) = P([A\text{ or }B],[A\text{ or }C])$


### Rules of probability ###

* _Commutation_
    * $P(A,B) = P(B,A)$
    * $P(A \text{ or } B) = P(B \text{ or } A)$
* _Duality_ (De Morgan's Theorem)
    * $P(\text{not }[A,B]) = P(\text{not}(A) \text{ or } \text{not}(B))$
    * $P(\text{not }[A\text{ or }B]) = P(\text{not}(A),\text{not}(B))$


### Rules of probability ###

Note that you may see other notation for probabilities expressed with Boolean logic (this list is 
not exhaustive)

* Negation ($A$ is false)
    * $P(\text{not }A)$, or $P(\bar{A})$, or $P(\lnot A)$
* Logical product (both A _and_ B are true)
    * $P(A,B)$, or $P(AB)$, or $P(A\text{ and }B)$, or $P(A\land B)$
* Logical sum (at least one of A _or_ B is true)
    * $P(A+B)$, or $P(A\text{ or }B)$, or $P(A\lor B)$


### Rules of probability ###
    
From these axioms we can derive:

* The **(Extended) Sum Rule**
    * $P(A\text{ or }B) = P(A) + P(B) - P(A\text{ and }B)$
* The **Product Rule**
    * $P(A\text{ and }B) \equiv p(A,B) = P(A)P(B|A) = P(B)P(A|B)$, where $P(x|y)$ is the probability that $x$ is true given $y$ is true. 

These rules apply to probabilities $P$ and also probability density functions (pdfs) $p$.


### Rules of probability ###

_Simple demonstration of the extended sum rule._

What is the probability that a card drawn from a standard deck of cards is a spade _or_ an ace?

We have $P(\spadesuit) = 13/52 = 1/4$ and $P(\text{ace}) = 4/52 = 1/13$, and $P(\spadesuit\text{ 
and ace}) = 1/(4\times 13) = 1/52$. It is reasonably obvious that for $P(\spadesuit\text{ or ace})$ 
we want to sum the probabilities for both cases, however they both contain the case where 
$P(\spadesuit\text{ and ace})$, so we have to remove one of those instances

\\[
P(\spadesuit\text{ or ace}) = \frac{13 + 4 - 1}{52} = \frac{16}{52}
\\]


### Bayes' theorem ###

From the product rule comes
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
    P(B|A,I) = \frac{P(A|B,I)P(B|I)}{P(A|I)}.
\end{empheq}-->

This is <!--{\color{red} {\bf Bayes' theorem}}-->.

From now on we will stick to using $P(A,B|I)$ to denoted the 
probability of $A$ _and_ $B$, where we have also explicitly added the conditioning on background 
information $I$.


### Bayes' theorem ###

Bayes theorem can be cast in terms of a **model** and some observations, or **data**. It tells us 
how to update our degree of belief about our model based on new data.
\\[
\redub{P(\text{model}|\text{data},I)}_{\mathclap{\text{Posterior}}} = 
\frac{\redob{P(\text{data}|\text{model},I)}^{\mathclap{\text{Likelihood}}} 
\redob{p(\text{model}|I)}^{\mathclap{\text{Prior}}}}{\redub{P(\text{data}|I)}_{\mathclap{\text{
Evidence } } } }
\\]

We can often calculate the these terms (e.g., analytically or numerically on a computer). 


### Bayes' theorem ###

* **Prior**: what we knew, or our degree of belief, about our model before taking data
* **Likelihood**: the influence of the data in updating our degree of belief 
* **Evidence**: the ``_evidence_'' for the data, or the likelihood for the data _marginalised_ over 
the model (we'll explore this later, but at the moment note it as a constant normalisation factor 
for the posterior)
* **Posterior**: our new degree of belief about our model in light of the data


### A Bayesian example: is a coin fair? ###

How can we determine if a coin is fair[^fnsvia]? We can consider a large number of contiguous propositions over
the range in which the bias weighting $H$ of the coin might lie:

* $H = 0$ coin produces a tail every time
* $H = 1$ coin produces a head every time
* $H = 0.5$ is a 'fair' coin with 50:50 chance of head or tail
* continuum of probabilities $0 \le H \le 1$

Given some data (an observed number of coin tosses) we can assess how much we believe each of
these propositions (e.g. $0 \le H < 0.01$, $0.01 \le H < 0.02$, and so on) to be true, e.g.
\\[
\text{Prob}(0 \le H < 0.01|d)
\\]

[^fnsvia]: See e.g. Chap. 2 of Sivia[][#Sivia].


### A Bayesian example: is a coin fair? ###

In the limiting case where our propositions each lie in the infinitesimal range ${\rm d}H$ 
our inference about the bias weighting is summarised by the pdf for the conditional probability
$p(H|d,I)$, i.e. the _posterior_. We can use Bayes' theorem to calculate it.

For coin flips, assuming that they are independent events, the probability of obtaining `$r$ heads in
$n$ tosses' is given by the binomial distribution, so our _likelihood_ is:
\\[
p(d|H,I) \propto H^r(1-H)^{n-r}
\\]
But, what should we use as our _prior_?


### A Bayesian example: is a coin fair? ###

But, what should we use as our _prior_?

Assuming we have no knowledge about the provenance of the coin, or the person tossing it, and want to
reflect total ignorance of the possible bias, then a simple probability reflecting this is a **uniform**,
or **flat**, pdf:
\\[
p(H|I) =
\begin{cases}
1, \text{if } 0 \le H \le 1, \\
0, \text{otherwise}.
\end{cases}
\\]
Using these we can calculate our posterior, $p(H|d,I)$, as we obtain more data (counting $r$ as the
number of coin tosses, $n$, increases).[^fncoinscript]

[^fncoinscript]: See <!--\href{https://github.com/mattpitkin/InferenceWorkshopLecture/blob/master/figures/scripts/coin\_toss.py}{\tt coin\_toss.py} -->


### A Bayesian example: is a coin fair? ###

![][coin_toss]

[coin_toss]: figures/coin_toss.pdf "Coin toss example" height="210px"


### A Bayesian example: is a coin fair? ###

<!--\begin{columns}
    \begin{column}{0.35\textwidth}-->
As the number of coin tosses increases the posterior evolves from the uniform prior to a tight range in $H$
with the most probable value being $H=0.3$.
<!--\end{column}
\begin{column}{0.65\textwidth}-->
![][coin_toss]
<!--\end{column}
\end{columns}-->


### A Bayesian example: is a coin fair? ###

What about a _different_ prior?

We know that coins are generally fair, so what if we assume this one is too? 

We can assign a Gaussian prior distribution that focusses the probability around the
expected 'fair coin' value
\\[
p(H|I) \propto \exp{\left(-\frac{1}{2}\frac{(H-\mu_H)^2}{\sigma_H^2}\right)},
\\]
with $\sigma_H = 0.05$ and $\mu_H = 0.5$.[^fncoinscript2]

[^fncoinscript2]: See <!--\href{https://github.com/mattpitkin/InferenceWorkshopLecture/blob/master/figures/scripts/coin\_toss\_2.py}{\tt coin\_toss\_2.py} -->


### A Bayesian example: is a coin fair? ###

![][coin_toss_2]

[coin_toss_2]: figures/coin_toss_2.pdf "Coin toss another example" height="210px"


### A Bayesian example ###

What do we learn from this?

* As our data improve (i.e. we gather more samples), the posterior pdf narrows and becomes
less sensitive to our choice of prior (i.e. the likelihood starts to dominate)
* The posterior conveys our (evolving) degree of belief in different values of $H$ given our
data
* If we want to express our belief as a **single number** we can adopt e.g., the mean, median or mode
* It is very straightforward to define _Bayesian confidence intervals_ (more correctly termed
**credible intervals**), to quantify our uncertainty on $H$.


### Bayesian credible interval ###

We define a **credible interval** $[\theta_a, \theta_b]$ as a (_non-unique_) range that a
contains a certain amount of posterior probability, $X$,
\\[
X = \int_{\theta_a}^{\theta_b} p(\theta|d,I) {\rm d}\theta.
\\]
If $X=0.95$ then we can find $[\theta_a, \theta_b]$ that e.g. gives the minimum range containing
95% of the probability.

The meaning of this is simple: _we are 95% sure that $\theta$ lies between $\theta_a$ and $\theta_b$._

This is just based on the data at hand and requires no assumptions about a frequency of measuring a
statistic over multiple trials.


### The Gaussian/Normal Likelihood ###

In many situations in physics/astronomy our data consists of the signal with some additive noise, e.g. considering a single data point

\\[
d_1 = s_1 + n_1.
\\]

We are interested in the _inverse problem_ of inferring the properties of the signal given the data.
To do this, and define a likelihood, we need to make sum assumptions about the noise.


### The Gaussian/Normal likelihood ###

If the noise generating process can be thought of as the sum of independent random processes then by the
<!--\href{https://en.wikipedia.org/wiki/Central_limit_theorem}{\it central limit theorem}--> it will tend
towards a <!--\href{https://en.wikipedia.org/wiki/Normal_distribution}{\it normal distribution}-->. So, we
often assume $n \sim N(0, \sigma^2)$ ("_$n$ is drawn from a Normal distribution with mean of zero and
variance $\sigma^2$_")

Also, for a process were we have the expectation value $\mu$ and variance $\sigma^2$, the distribution
that <!--\href{https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution\#Specified_variance:_the_normal_distribution}{maximises the entropy}-->, is the least informative (most conservative), is the normal distribution:

<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(x|\mu,\sigma,I) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2} \right)}
\end{empheq}-->


### The Gaussian/Normal likelihood ###

For our data $d_1 = s_1 + n_1$, if we have a model of our signal parameterised by $\vec{\theta}$, such that
$s_1 \equiv s_1(\vec{\theta})$, then due to the additive nature of the noise and signal, the expectation value
$\mu = s_1(\vec{\theta})$, and we have:

<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(d_1|\vec{\theta},\sigma,I) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{\left(d_1-s_1(\vec{\theta})\right)^2}{2\sigma^2} \right)}
\end{empheq}-->


### The joint likelihood ###

Often we have more than one data point! If the noise in the data is <!--\href{https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables}{\it independent and identically distributed}--> (_i.i.d._)
you can just multiply the likelihoods for each data point for for the _joint_ likelihood for all the data
$\mathbf{d} = \{d_1, d_2, \dots, d_N\}$:

<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{align*}
p(\mathbf{d}|\vec{\theta},\sigma,I) &= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{2\sigma^2} \right)}, \nonumber \\
&= \left(2\pi\sigma^2\right)^{-n/2}\exp{\left(-\frac{1}{2}\sum_{i=1}^n\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{\sigma^2} \right)}
\end{empheq}-->


### The joint likelihood: non-_i.i.d._ ###

If the noise process for points is Gaussian, but the noise is _not_
correlated, i.e., each point (or a subset of points) has a different _known_ variance $\bm{\sigma} = \{\sigma_1, \dots, \sigma_n\}$, we have:

<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(\mathbf{d}|\vec{\theta},\bm{\sigma},I) = \left(\prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_i^2}}\right)\exp{\left(-\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{2\sigma_i^2} \right)}.
\end{empheq}-->


### The joint likelihood: non-_i.i.d._ ###

If the noise process for points is Gaussian, but the noise is
correlated, but the covariance matrix $\bm{\Sigma}$ is known, we have a <!-- \href{https://en.wikipedia.org/wiki/Multivariate\_normal\_distribution}{\it multivariate normal distribution}-->:

<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(\mathbf{d}|\vec{\theta}, \bm{\Sigma}, I) = \left(2\pi\right)^{n/2}\left|\bm{\Sigma}\right|^{-1/2}\exp{\left(-\frac{1}{2}\left(\mathbf{d} - \mathbf{s}(\vec{\theta})\right)'\bm{\Sigma}^{-1}\left(\mathbf{d} - \mathbf{s}(\vec{\theta})\right)\right)},
\end{empheq}-->

where $\mathbf{s}(\vec{\theta}) = \{s_1(\vec{\theta)}, \dots, s_n(\vec{\theta)}\}$. This becomes the previous case if $\bm{\Sigma}$ is diagonal.

The covariance is ...


### Inference for gravitational-wave astronomy ###

Gravitational wave detectors produce a time series of strain measurement $h(t)$ ("_$h$ of $t$_")


### Anatomy of the GW likelihood function ###

We've seen the "standard" Gaussian likelihood function, but in GW papers you
might (see, e.g., Equations 3 and 4 of [][#Biwer:2018osg], assuming a single detector,
and ignoring the normalisation):
<!--\begin{empheq}[box={\borderedmathbox[scale=0.6, left=10mm, right=10mm]}]{align*}
p(d|\vec{\theta}, I) \propto&  \exp{\left(-\frac{1}{2} \redob{\langle \tilde{d}(f) - \tilde{s}(f;\vec{\theta}) | \tilde{d}(f) - \tilde{s}(f;\vec{\theta}) \rangle}^{\mathclap{\text{noise weighted innner product}}}\right)} \nonumber \\
\equiv & \exp{\left(-\frac{1}{2} \left[ 4\Re \int_0^\infty \frac{\left(\tilde{d}(f) - \tilde{s}(f;\vec{\theta})\right)\left(\tilde{d}(f) - \tilde{s}(f;\vec{\theta})\right)^*}{S_n} {\rm d}f \right]\right)},
\end{empheq}-->
where $d$ is the data, $\vec{\theta}$ is a set of parameters defining a waveform model $s$, the tilde
represents the Fourier transform, and $S_n$ is the one-sided power spectral density of the noise in the data.

Let's see how this relates to our earlier equation.


### Anatomy of the GW likelihood function ###

First, we actually work with discrete data, and discrete Fourier transforms, so
<!--\begin{empheq}[box={\borderedmathbox[scale=0.6]}]{equation*}
p(d|\vec{\theta}, I) = \exp{\left(-\frac{1}{2} \left[ 4\Re \int_0^\infty \frac{\left(\tilde{d}(f) - \tilde{s}(f;\vec{\theta})\right)\left(\tilde{d}(f) - \tilde{s}(f;\vec{\theta})\right)^*}{S_n}(f) {\rm d}f \right]\right)},
\end{empheq}-->
becomes
<!--\begin{empheq}[box={\borderedmathbox[scale=0.75]}]{equation*}
p(d|\vec{\theta}, I) = \exp{\left(-\frac{1}{2} \left[ 4\Re \textcolor{red}{\sum_{i=j}^k} \frac{\left(\tilde{d}_i - \tilde{s}_i(\vec{\theta})\right)\left(\tilde{d}_i - \tilde{s}_i(\vec{\theta})\right)^*}{\textcolor{red}{T} S_n^{(i)}} \right]\right)},
\end{empheq}-->
due to $\int \dots {\rm d}f \approx \sum \dots \Delta f$, and $\Delta f = 1/T$ for data of length $T$ seconds,
and $i$ in the index over frequency bins from bin $j$ to $k$.


### The power spectral density ###


### Gravitational wave parameters ###

CBC signals are defined by 15 intrinsic parameters (only 9 for a non-spinning system):

<!--\scalebox{0.8}{\begin{columns}[T]
\begin{column}{0.5\textwidth}-->

Spinning/non-spinning:

 * component masses ($m_i$ where $m_1 > m_2$)
 * a reference time, e.g., the coalescence time $t_c$
 * the orbital phase at $t_c$, $\phi_c$
 * sky position (right ascension $\alpha$ and declination $\delta$)
 * luminosity distance $d_L$
 * polarisation angle $\psi$

<!--\end{column}
\begin{column}{0.5\textwidth}-->

Spinning sources:

 * dimensionless spin magnitudes $a_i = |\vec{s}_i| c / G m_i^2$ where $\vec{s}_i$ is the spin vector.
 * two angles defining $\vec{s}_i$ specifying the orientation with respect to the plane defined by the line of sight and the initial orbital angular momentum.
 
<!--\end{column}
\end{columns}}-->


### Gravitational wave parameters ###


You're free to choose whatever priors you want, but commonly we use priors that are (see [][#2015PhRvD..91d2003V]):

<!--\begin{columns}\begin{column}{0.6\textwidth} -->
 * <!-- {\scriptsize uniform in component masses with the constraints that $m_1 > m_2$ and $m_1 + m_2 < M_{\rm max}$}-->
 * <!-- {\scriptsize isotropic in orientation of the binary and sky position, so:}-->
    * <!-- {\scriptsize $p(\iota, \psi, \phi_c) \propto \sin{\iota}$, and}-->
    * <!-- {\scriptsize $p(\alpha, \delta) \propto \sin{\delta}$}-->
 * <!-- {\scriptsize uniform in volume (for the local universe), so $p(d_L) \propto d_L^2$ (from $p(d_L) = p(V)\left|\frac{{\rm d}V}{{\rm d}d_L}\right|)$, with $p(V) \propto 1$ and $V \propto d_L^3$).}-->

<!--\end{column}\begin{column}{0.4\textwidth}

\begin{figure}[htbp]
\centering
\includegraphics[keepaspectratio, width=\textwidth]{figures/massprior.png}
\caption*{{\tiny Example of uniform mass prior from \cite{2015PhRvD..91d2003V}}.}
\label{massprior}
\end{figure}

\end{column}\end{columns}-->


<!--% Bibliography slide -->
### Bibliography ###

<!--
\bibliographystyle{unsrt}
\bibliography{lecture_notes}
-->
