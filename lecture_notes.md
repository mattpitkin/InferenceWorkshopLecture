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
    * Bayes' theorem
    * Marginalisation
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

Some useful papers to read are:

* The ``LALInference`` paper <!--\citep{2015PhRvD..91d2003V}-->.
* The <!--\textsc{PyCBC Inference} paper \citep{Biwer:2018osg}-->.
* The <!--\textsc{Bilby} paper \citep{Ashton:2019}-->.


### Bayes' theorem ###

From the product rule comes
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
    P(B|A,I) = \frac{P(A|B,I)P(B|I)}{P(A|I)}.
\end{empheq}-->

This is <!--\href{https://en.wikipedia.org/wiki/Bayes\%27\_theorem}{{\color{violet}{\bf Bayes' theorem}}}-->.

From now on we will stick to using $P(A,B|I)$ to denoted the 
probability of $A$ _and_ $B$, where we have also explicitly added the conditioning on background 
information $I$.


### Bayes' theorem ###

Bayes theorem can be cast in terms of a **model** and some observations, or **data**. It tells us 
how to update our degree of belief about our model based on new data.
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
\redub{P(\text{model}|\text{data},I)}_{\mathclap{\text{Posterior}}} = 
\frac{\redob{P(\text{data}|\text{model},I)}^{\mathclap{\text{Likelihood}}} 
\redob{p(\text{model}|I)}^{\mathclap{\text{Prior}}}}{\redub{P(\text{data}|I)}_{\mathclap{\text{
Evidence } } } }
\end{empheq}-->
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
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
\text{Prob}(0 \le H < 0.01|d)
\end{empheq}-->

[^fnsvia]: See e.g. Chap. 2 of <!--\citet{Sivia}-->.


### A Bayesian example: is a coin fair? ###

In the limiting case where our propositions each lie in the infinitesimal range ${\rm d}H$ 
our inference about the bias weighting is summarised by the pdf for the conditional probability
$p(H|d,I)$, i.e. the _posterior_. We can use Bayes' theorem to calculate it.

For coin flips, assuming that they are independent events, the probability of obtaining `$r$ heads in
$n$ tosses' is given by the binomial distribution, so our _likelihood_ is:
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
p(d|H,I) \propto H^r(1-H)^{n-r}
\end{empheq}-->
But, what should we use as our _prior_?


### A Bayesian example: is a coin fair? ###

But, what should we use as our _prior_?

Assuming we have no knowledge about the provenance of the coin, or the person tossing it, and want to
reflect total ignorance of the possible bias, then a simple probability reflecting this is a **uniform**,
or **flat**, pdf:
<!--\begin{empheq}[box={\borderedmathbox}]{equation*}
p(H|I) =
\begin{cases}
1, \text{if } 0 \le H \le 1, \\
0, \text{otherwise}.
\end{cases}
\end{empheq}-->
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
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(H|I) \propto \exp{\left(-\frac{1}{2}\frac{(H-\mu_H)^2}{\sigma_H^2}\right)},
\end{empheq}-->
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


### Probability mass and probability density ###

The probability distribution for a _discrete_ parameter is called the <!--\href{https://en.wikipedia.org/wiki/Probability\_mass\_function}{{\bf {\color{violet} probability mass function}} (PMF)}-->. The value of the PMF at a
particular value of the parameter (say $H$) is the _probability_ for that value, and we must have:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
\sum_{i=1}^N P(H| I) = 1
\end{empheq}-->


### Probability mass and probability density ###

For a continuous parameter, the probability distribution is called the
<!--\href{https://en.wikipedia.org/wiki/Probability\_density\_function}{{\bf {\color{violet} probability density function}} (PDF)}-->. The value of the PDF
at a particular parameter value is _not_ the probability for the value, it is a probability _density_, and the probability
can be calculated only for some range of the allowed parameter values, e.g.
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
P(h_1 \leq H \leq h_2)  = \int_{h_1}^{h_2} p(H| I) {\rm d}H,
\end{empheq}-->
provided the probability distribution is properly normalised, such that
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
\int_{-\infty}^{\infty} p(H| I) {\rm d}H = 1.
\end{empheq}-->


### Marginalisation ###

Posteriors probability distributions have the same dimensionality as the number of parameters you want to infer in your
probabilistic model, e.g., if we need to infer the gradient and $y$-intercept of a straight line ($m$ and $c$) from some data ($\bm{d}$),
then you have two parameters and your posterior will be two-dimensional: $p(m, c|\bm{d}, I)$.

You may only be interested in the distribution of one (or some subset) of the parameters, so you <!--\href{https://en.wikipedia.org/wiki/Marginal\_distribution}{\bf \color{violet}{marginalise}}--> (i.e., integrate) over the _nuisance_ parameters. E.g., if we're only interested in the gradient of the line then:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(m|\bm{d},I) = \int_{-\infty}^{\infty} p(m, c|\bm{d}, I) {\rm d}c.
\end{empheq}-->


### Marginalisation ###

For higher dimensional problems multiple integrals may be required:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(x|\bm{d},I) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p(x, y, z|\bm{d}, I) {\rm d}y {\rm d}z.
\end{empheq}-->
In some cases marginalisation can be performed analytically. For low dimensional problems, with a well localised
posterior (i.e., it doesn't have support out to $\pm \infty$), the posterior can be evaluated on a grid and
marginalisation can be performed numerically. For high-dimensional problems this is not viable and we must use
stochastic sampling methods, e.g., <!--\href{https://en.wikipedia.org/wiki/Markov\_chain\_Monte\_Carlo}{\bf \color{violet}{Markov chain Monte Carlo}}--> (see talk be Vivien Raymond) or <!--\href{https://en.wikipedia.org/wiki/Nested\_sampling\_algorithm}{\bf \color{violet}{nested sampling}}--> (see talk by John Veitch).


### Evidence ###

<!--{\footnotesize The normalisation constant for the posterior probability is often called the Bayesian \href{https://en.wikipedia.org/wiki/Marginal\_likelihood}{{\bf {\color{violet} evidence}}, or {\bf {\color{violet} marginal likelihood}}}}-->,
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(\bm{d}|\mathcal{H},I) = \int^{\bm{\theta}} p(\bm{d}| \bm{\theta}, \mathcal{H}, I) p(\bm{\theta}|\mathcal{H}, I) {\rm d}\bm{\theta},
\end{empheq}-->
<!--{\footnotesize where the integral is multi-dimensional over all parameters $\bm{\theta} = \{\theta_1, \theta_2, \dots, \theta_n\}$. Here we been explicitly stated that the likelihood and prior are conditional on a given model, or hypothesis, $\mathcal{H}$. Hence, the evidence is the {\it likelihood} of observing the data for a given model $\mathcal{H}$.}

{\footnotesize The multi-dimensional integral may difficult/impossible to compute analytically or using standard numerical integration methods, so the \href{https://en.wikipedia.org/wiki/Nested\_sampling\_algorithm}{\bf \color{violet}{nested sampling}} algorithm may be required (see talk by John Veitch).}-->

### Model comparison ###

<!--{\footnotesize If you are purely interested in marginal posterior distributions the normalisation is not important and can must often
be ignored. However, the evidence allows you to compare different models (say $\mathcal{H}_1$ and $\mathcal{H}_2$) given the same data. The model odds is:}
\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
\mathcal{O}_{12} \equiv \redub{\frac{p(\mathcal{H}_1|\bm{d}, I)}{p(\mathcal{H}_2|\bm{d}, I)}}_{\mathclap{\text{Model Odds}}} = \redob{\frac{\bm{d}| p(\mathcal{H}_1, I)}{p(\bm{d}|\mathcal{H}_2, I)}}^{\mathclap{\text{Bayes Factor}}} \redob{\frac{p(\mathcal{H}_1| I)}{p(\mathcal{H}_2| I)}}^{\mathclap{\text{Prior Odds}}}.
\end{empheq}
{\footnotesize \href{https://en.wikipedia.org/wiki/Bayes\_factor}{\bf \color{violet}{Bayes factor}} is the ratio of the evidences for the two hypotheses. The prior odds defines the {\it a priori} relative degree-of-belief about each model, which in practice is often set to unity, i.e., neither model is preferred {\it a priori}.}-->


### Bayesian credible interval ###

We define a <!--\href{https://en.wikipedia.org/wiki/Credible\_interval}{\color{violet}{\bf credible interval}}--> $[\theta_a, \theta_b]$ as a (_non-unique_) range that a
contains a certain amount of posterior probability, $X$,
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
X = \int_{\theta_a}^{\theta_b} p(\theta|d,I) {\rm d}\theta.
\end{empheq}-->
If $X=0.95$ then we can find $[\theta_a, \theta_b]$ that e.g. gives the minimum range containing
95% of the probability.

The meaning of this is simple: _we are 95% sure that $\theta$ lies between $\theta_a$ and $\theta_b$._


### The Gaussian/Normal Likelihood ###

In many situations in physics/astronomy our data consists of the signal with some additive noise, e.g. considering a single data point
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
d_1 = s_1 + n_1.
\end{empheq}-->
We are interested in the _inverse problem_ of inferring the properties of the signal given the data.
To do this, and define a likelihood, we need to make some assumptions about the noise properties.


### The Gaussian/Normal likelihood ###

If the noise generating process can be thought of as the sum of independent random processes then by the
<!--\href{https://en.wikipedia.org/wiki/Central_limit_theorem}{\it central limit theorem}--> it will tend
towards a <!--\href{https://en.wikipedia.org/wiki/Normal_distribution}{\textbf{\color{violet}{normal (or Gaussian) distribution}}}-->. So, we
often assume $n \sim N(0, \sigma^2)$ ("_$n$ is drawn from a Normal distribution with mean of zero and
variance $\sigma^2$_")

Also, for a process where we have the expectation value $\mu$ and variance $\sigma^2$, the distribution
that <!--\href{https://en.wikipedia.org/wiki/Maximum\_entropy\_probability\_distribution\#Specified\_variance:\_the\_normal\_distribution}{maximises the entropy}-->, is the least informative (most conservative), is the normal distribution:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(x|\mu,\sigma,I) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2} \right)}
\end{empheq}-->


### The Gaussian/Normal likelihood ###

For our single point of data $d_1 = s_1 + n_1$, if we have a model of our signal parameterised by $\vec{\theta}$, such that
$s_1 \equiv s_1(\vec{\theta})$, then due to the additive nature of the noise and signal, the expectation value
$\mu = s_1(\vec{\theta})$, and we have:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(d_1|\vec{\theta},\sigma,I) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{\left(d_1-s_1(\vec{\theta})\right)^2}{2\sigma^2} \right)}
\end{empheq}-->


### The joint likelihood ###

Often we have more than one data point! If the noise in the data is <!--\href{https://en.wikipedia.org/wiki/Independent\_and\_identically\_distributed\_random\_variables}{\bf \color{violet}{independent and identically distributed}}--> (_i.i.d._)
you can multiply the likelihoods for each data point to give the _joint_ likelihood for all the data
$\mathbf{d} = \{d_1, d_2, \dots, d_N\}$:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{align*}
p(\mathbf{d}|\vec{\theta},\sigma,I) &= \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{2\sigma^2} \right)}, \nonumber \\
&= \left(2\pi\sigma^2\right)^{-N/2}\exp{\left(-\frac{1}{2}\sum_{i=1}^N\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{\sigma^2} \right)}
\end{empheq}-->

This is a <!--\href{https://en.wikipedia.org/wiki/Stationary\_process}{strictly (or strongly) \color{violet}{\bf stationary process}}-->.


### The joint likelihood: non-_i.i.d._ ###

If the noise process is Gaussian, but the noise is _not_
correlated, i.e., each point (or a subset of points) has a different _known_ variance $\bm{\sigma} = \{\sigma_1, \dots, \sigma_N\}$, we have:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(\mathbf{d}|\vec{\theta},\bm{\sigma},I) = \left(\prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_i^2}}\right)\exp{\left(-\frac{\left(d_i-s_i(\vec{\theta})\right)^2}{2\sigma_i^2} \right)}.
\end{empheq}-->


### The joint likelihood: non-_i.i.d._ ###

If the noise process is Gaussian, but the noise is
correlated and can be defined by a known (or estimatable) covariance matrix $\bm{\Sigma}$ (a <!--\href{https://en.wikipedia.org/wiki/Stationary\_process\#Weak\_or\_wide-sense_stationarity}{\color{violet}{\bf weakly stationary process}}-->), we have a <!-- \href{https://en.wikipedia.org/wiki/Multivariate\_normal\_distribution}{\bf \color{violet}{multivariate normal distribution}}-->:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation}\label{eq:gauss}
p(\mathbf{d}|\vec{\theta}, \bm{\Sigma}, I) = \left(2\pi\right)^{n/2}\left|\bm{\Sigma}\right|^{-1/2}\exp{\left(-\frac{1}{2}\left(\mathbf{d} - \mathbf{s}(\vec{\theta})\right)'\bm{\Sigma}^{-1}\left(\mathbf{d} - \mathbf{s}(\vec{\theta})\right)\right)},
\end{empheq}-->
where $\mathbf{s}(\vec{\theta}) = \{s_1(\vec{\theta)}, \dots, s_N(\vec{\theta)}\}$. This becomes the previous case if $\bm{\Sigma}$ is diagonal.


### Estimating the covariance matrix ###

The covariance matrix can be estimated by finding the
<!--\href{https://en.wikipedia.org/wiki/Autocovariance}{\textbf{\color{violet}{autocovariance}}}--> of the noise (ideally from
some data that is drawn purely from the noise process, i.e., contains not signal). If we assume $N$
evenly sampled noise data points $\bm{n}$, then the autocovariance is:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
\gamma_j = \frac{1}{N-1}\sum_{i=0}^{N-j} \left(n_{i+j} - \bar{\bm{n}}\right) \left(n_{i+1} - \bar{\bm{n}}\right),
\end{empheq}-->
with $j$ indices starting at 1, and $\bar{\bm{n}} = (1/N)\sum_{i=1}^N n_i$. This could be estimated from $M$ multiple stretches of data and averaged, e.g., $\bar{\gamma}_j = (1/M) \sum_{i=1}^M \gamma_{ij}$.


### Estimating the covariance matrix ###

The <!--\href{https://en.wikipedia.org/wiki/Autocorrelation}{\textbf{\color{violet}{autocorrelation function}}}--> is then defined as $\rho_j = \frac{\gamma_j}{\gamma_1}$ and, setting $\gamma_1 \equiv \sigma^2$, the covariance matrix is:
<!-- \begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
\bm{\Sigma} = \sigma^2\left(\begin{array}{ccccc}
1      & \rho_2 & \rho_3 & \cdots & \rho_n \\
\rho_2 & 1      & \rho_2 & \cdots & \rho_{n-1} \\
\rho_3 & \rho_2 & 1      & \ddots & \vdots \\
\vdots & \vdots & \ddots & \ddots & \rho_2 \\
\rho_n & \rho_{n-1} & \cdots & \rho_2 & 1
\end{array} \right).
\end{empheq}
-->

### Example: fitting a line ###

As an example, we'll examine the problem of fitting a line $\bm{y} = m\bm{x}+c$ to data, $\bm{d}$ (<!--\href{https://en.wikipedia.org/wiki/Linear_regression}{\color{violet}{\bf linear regression}}-->). We can write the posterior for
the parameters
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
p(m,c|\bm{d},I) \propto \redub{p(\bm{d}|m,c,I)}_{\mathclap{\text{Likelihood}}} \times \redub{p(m,c|I)}_{\mathclap{\text{Prior}}}.
\end{empheq}-->
If the prior on the parameters is uniform and independent, so
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
p(m,c|I) = p(m|I)p(c|I) = \text{constant},
\end{empheq}-->
then the posterior is
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
p(m,c|\bm{d},I) \propto p(\bm{d}|m,c,I).
\end{empheq}-->
We could, in this case, use the machinery of _maximum likelihood_ to estimate the parameters.


### Example: fitting a line ###

However, we will use this to show to general concept of fitting any model (see <!--\href{https://github.com/mattpitkin/InferenceWorkshopLecture/blob/master/figures/scripts/bayesian\_line\_fittingy}{\scriptsize{\tt bayesian\_line\_fitting.py}}-->). If
the likelihood is Gaussian, with known values of $\sigma_i$, then
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
p(m,c|\bm{d},I) = p(m,c|I)\left(\frac{1}{2\pi \sigma_i^2}\right)^{n/2}\exp{\left(-\sum_{i=1}^n\frac{[d_i-(m x_i + c)]^2}{2\sigma_i^2}\right)}
\end{empheq}-->
and we can evaluate the posterior over a grid in the parameters $m$ and $c$.

We can also compute the marginal posteriors on $m$ and $c$ as, e.g.
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(m|\bm{d},I) = \int_{-\infty}^{\infty} p(m,c|d,I) {\rm d}c.
\end{empheq}-->


### Example: fitting a line ###

![][bayesian_line_fitting]

[bayesian_line_fitting]: figures/bayesian_line_fitting.pdf "Bayesian line fitting example" height="210px"


### Example: fitting a line ###

In the above example, in practice, when $p(m,c|I) = \text{constant}$ and $\sigma_i = \sigma$ are constant, we can just calculate the posterior[^fnlogspace] over a grid in $m$ and $c$
<!--\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
\ell(m_{i_m},c_{i_c}) = \ln{p(m_{i_m},c_{i_c}|d,I)} = -\sum_{i=1}^n\frac{[d_i-(m_{i_m} x_i + c_{i_c})]^2}{2\sigma^2}
\end{empheq}-->
and get the marginal posteriors through numerical integration, e.g.
<!--\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
p(m_{i_m}|d,I) \propto \sum_{i_c}^{n_c} \exp{\left(\ell(m_{i_m},c_{i_c}) - \text{max}\ell(m,c)\right)} \Delta c
\end{empheq}-->
where $\Delta c$ are the grid step sizes in $c$ (or you could use the trapezium rule for more accuracy).

[^fnlogspace]: We generally work in natural logarithm space due to numerical precision issues.


### Inference for gravitational-wave astronomy ###

<!--{\scriptsize Gravitational-wave detectors produce a {\it real} time series of strain measurements $h(t)$ (``$h$ {\it of} $t$'').
This is the linear combination of noise (assumed to be produced by a weakly stationary process, i.e.,
\href{https://en.wikipedia.org/wiki/Colors\_of\_noise}{\bf \color{violet}{coloured Gaussian noise}}) and the signal as projected onto the detector via its response function:}
\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
h(t) = n(t) + s(t; \bm{\theta}),
\end{empheq}
{\scriptsize where}
\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
s(t; \bm{\theta}) = F_+^D(t; \alpha, \delta, \psi) h_+(t; \bm{\theta}') + F_{\times}^D(t; \alpha, \delta, \psi) h_{\times}(t; \bm{\theta}').
\end{empheq}-->

<!--{\tiny $F_{+/\times}^D$ are the `plus' and `cross' polarisation responses of detector $D$ to a source at a sky position given by right ascension $\alpha$ and declination $\delta$ and with polarisation angle $\psi$. $h_{+/\times}$ are the source amplitudes at the Earth defined by the parameters $\bm{\theta}'$, where $\bm{\theta} = \{\alpha, \delta, \psi, \bm{\theta}'\}$.}-->


### Inference for gravitational-wave astronomy ###

<!--{\footnotesize We are interested in using $h(t)$ (we'll use $\bm{d}$ for the vector of observed time series {\it data} points instead of $h$ from now on) to infer the marginal probability distributions of the parameters
$\bm{\theta}$ (or some subset of them). E.g., the posteriors on the sky position of the source}
\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(\alpha, \delta|\bm{d}, I) \propto \int^{\bm{\theta}_{\not\in \{\alpha, \delta\}}} p(\bm{d}|\bm{\theta},I) p(\bm{\theta}|I) {\rm d}\bm{\theta}_{\not\in \{\alpha, \delta\}},
\end{empheq}
{\footnotesize or the source masses for a compact binary coalescence event}
\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(m_1, m_2|\bm{d}, I) \propto \int^{\bm{\theta}_{\not\in \{m_1, m_2\}}} p(\bm{d}|\bm{\theta},I) p(\bm{\theta}|I) {\rm d}\bm{\theta}_{\not\in \{m_1, m_2\}}.
\end{empheq}
{\footnotesize So, we need to have a likelihood for the data and a prior for the source parameters.}-->


### Inference in gravitational-wave astronomy ###

<!--\begin{columns}
\begin{column}{0.5\textwidth}
\begin{figure}[htbp]
\centering
\includegraphics[keepaspectratio, width=\textwidth]{figures/skyloc_GW150914.png}
\caption*{{\tiny Sky location posterior for GW150914 \citep{Abbott:2016}}.}
\label{skyloc}
\end{figure}
\end{column}
\begin{column}{0.5\textwidth}
\begin{figure}[htbp]
\centering
\includegraphics[keepaspectratio, width=\textwidth]{figures/masses_GW150914.png}
\caption*{{\tiny Source mass posteriors for GW150914 \citep{Abbott:2016}}.}
\label{skyloc}
\end{figure}
\end{column}
\end{columns}-->


### Examples of inference in gravitational-wave astronomy ###

A non-exhaustive list examples of where Bayesian inference has been used in (ground-based) gravitational-wave astronomy:

* Searches for continuous (monochromatic) gravitational waves from known pulsars
* Source parameter estimation for CBC signals
* Rapid CBC source sky and distance localisation (<!--{\textsc{BayeStar}}-->)
* Unmodelled burst event trigger generator (<!--{\textsc{BlockNormal}}-->)
* Unmodelled burst waveform reconstruction, glitch reconstruction, and power spectrum estimation (<!--{\textsc{BayesWave}}-->)
* Unmodelled burst parameter estimation (oLIB)
* Supernova signal model comparison (SMEE)
* Hierarchical inference of CBC mass and spin distributions


### Anatomy of the GW likelihood function ###

We've seen the "standard" Gaussian likelihood function, but in GW papers you
might see <!--\citep[e.g., Equations 3 and 4 of][assuming a single detector, and ignoring the normalisation]{Biwer:2018osg}-->:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.6, left=10mm, right=10mm]}]{align*}
p(\bm{d}|\bm{\theta}, I) \propto&  \exp{\left(-\frac{1}{2} \redob{\langle \tilde{d}(f) - \tilde{s}(f;\bm{\theta}) | \tilde{d}(f) - \tilde{s}(f;\bm{\theta}) \rangle}^{\mathclap{\text{noise weighted innner product}}}\right)} \nonumber \\
\equiv & \exp{\left(-\frac{1}{2} \left[ 4\Re \int_0^\infty \frac{\left(\tilde{d}(f) - \tilde{s}(f;\bm{\theta})\right)\left(\tilde{d}(f) - \tilde{s}(f;\bm{\theta})\right)^*}{S_n} {\rm d}f \right]\right)},
\end{empheq}-->
where $d$ is the data, $\bm{\theta}$ is a set of parameters defining a waveform model $s$, the tilde
represents the Fourier transform, and $S_n$ is the one-sided power spectral density of the noise in the data.

Let's see how this relates to our earlier equation.


### Anatomy of the GW likelihood function ###

First, we actually work with discrete data, and discrete Fourier transforms, so
<!--\begin{empheq}[box={\borderedmathbox[scale=0.6]}]{equation*}
p(\bm{d}|\bm{\theta}, I) \propto \exp{\left(-\frac{1}{2} \left[ 4\Re \int_0^\infty \frac{\left(\tilde{d}(f) - \tilde{s}(f;\bm{\theta})\right)\left(\tilde{d}(f) - \tilde{s}(f;\bm{\theta})\right)^*}{S_n}(f) {\rm d}f \right]\right)},
\end{empheq}-->
becomes
<!--\begin{empheq}[box={\borderedmathbox[scale=0.75]}]{equation*}
p(\bm{d}|\bm{\theta}, I) \propto \exp{\left(-\frac{1}{2} \left[ 4\Re \textcolor{red}{\sum_{i=j}^k} \frac{\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)^*}{\textcolor{red}{T} S_n^{(i)}} \right]\right)},
\end{empheq}-->
due to $\int \dots {\rm d}f \approx \sum \dots \Delta f$, and $\Delta f = 4/T$ for data of length $T$ seconds,
and $i$ in the index over frequency bins from bin $j$ to $k$.


### Anatomy of the GW likelihood function ###

Given that for complex $x = a + i b$, $x x^* = \left(a + i b\right)\left(a - i b\right) = a^2 + b^2$,
we get
<!--\begin{empheq}[box={\borderedmathbox[scale=0.75]}]{align*}
p(\bm{d}|\bm{\theta}, I)  & \propto \exp{\left(-\frac{1}{2} \left[ 4 \sum_{i=j}^k \frac{\textcolor{red}{\Re}\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)^2 + \textcolor{red}{\Im}\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)^2}{T S_n^{(i)}} \right]\right)}, \nonumber \\
& \equiv \exp{\left(-\frac{1}{2} \left[ 4 \sum_{i=j}^k \frac{\textcolor{red}{\left|\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right|^2}}{T S_n^{(i)}} \right]\right)}
\end{empheq}-->
This is the same as starting from the assumption that the noise in the real and imaginary parts of the Fourier transform are independent
(but drawn from the same noise process), and writing the _joint_ the likelihood of these independent data sets.


### Anatomy of the GW likelihood function ###

For numerical reasons we generally work with the natural logarithm of the likelihood (and other
probability densities), so
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
\ln{p(\bm{d}|\bm{\theta}, I)} = -\frac{1}{2} \left[ 4 \sum_{i=j}^k \frac{\Re\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)^2 + \Im\left(\tilde{d}_i - \tilde{s}_i(\bm{\theta})\right)^2}{T S_n^{(i)}} \right] + C,
\end{empheq}-->
where $C$ is the normalisation term given by (see Equation 12 of <!--\citet{2015PhRvD..91d2003V}-->)
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
C = -\frac{1}{2}\sum_{i=j}^k \ln{\left(\pi T S_n^{(i)} / 2 \right)}.
\end{empheq}-->


### Anatomy of the GW likelihood function ###

If we expand out the quadratic terms we get:
<!--
\begin{itemize}
\item the {\it null} log-likelihood (noise-only log {\it evidence}):
\end{itemize}
\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
\ln{p(\bm{d}|\bm{s}(\bm{\theta})=0, I)} \equiv \mathcal{L}_n = -(2/T) \sum_i \left(\Re{(d_i)^2} + \Im{(d_i)^2}\right)/S_n^{(i)} + C,\end{empheq}
\begin{itemize}
\item the {\it optimal} signal-to-noise:
\end{itemize}
\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
\rho^2_{\rm opt} = (4/T) \sum_i \left(\Re{(s_i)^2} + \Im{(s_i)^2}\right)/S_n^{(i)},
\end{empheq}
\begin{itemize}
\item the {\it matched filter} signal-to-noise:
\end{itemize}
\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
\rho^2_{\rm mf} = (4/T) \sum_i \left(\Re{(d_i)}\Re{(s_i)} + \Im{(d_i)}\Im{(s_i)}\right)/S_n^{(i)}.\end{empheq}
-->


### Anatomy of the GW likelihood function ###

So, we can write the log-likelihood in terms of:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation}\label{eq:loglike}
\ln{p(d|\bm{\theta}, I)} = -\frac{1}{2}\left(\rho^2_{\rm opt}(\bm{\theta}) - 2\rho^2_{\rm mf}(\bm{\theta})\right) + \mathcal{L}_n.
\end{empheq}-->
The log of the likelihood ratio $p(\bm{d}|\bm{\theta}, I) / p(\bm{d}|\bm{s}(\bm{\theta})=0, I)$ is therefore:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
\mathcal{L}(\bm{\theta}) - \mathcal{L}_n = -\frac{1}{2}\left(\rho^2_{\rm opt}(\bm{\theta}) - 2\rho^2_{\rm mf}(\bm{\theta})\right).
\end{empheq}-->
Evaluating the likelihood over the parameter space effectively required evaluating these $\rho^2$ terms.


### The power spectral density ###

In the above likelihood it assumes the noise in each frequency bin is independent and Gaussian, with
a variance defined using the _one-sided_ <!--\href{https://en.wikipedia.org/wiki/Spectral\_density\#Power\_spectral\_density}{\bf \color{violet}{power spectral density}}--> (PSD), $S_n(f)$, given by[^fnpsd]
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
S_n(f) = \frac{2}{N^2\Delta f}|\tilde{n}(f)|^2,\text{~with~}\tilde{n}(f_k) = \sum_j n_j e^{-2\pi i j k/N},
\end{empheq}-->
where $N$ is the number of data points, and $\Delta f = 1/T = 1/(N\Delta t)$ for observation time $T$ <!--\citep[see, e.g. Appendix of][]{Veitch:2010}-->

The variance for the real and imaginary components of each frequency bin is given by $\sigma_i^2 = (T/4)S_n^{(i)}$,
which when substituted into <!--Equation~(\ref{eq:loglike})--> gives the standard Gaussian log-likelihood.

[^fnpsd]: This is equivalent to the Fourier transform of the noise autocorrelation function.

### The power spectral density ###

In practice we estimate the PSD using, e.g., <!--\href{https://en.wikipedia.org/wiki/Welch\%27s\_method}{\bf \color{violet}{Welch's method}}-->. A stretch of noise-only data is chosen and
divided into $M$ overlapping segments (with fractional overlap $\alpha$) each of the same length, which is
the same length $N$ as the data segment to be analysed. Each segments is multiplied by a window, Fourier
transformed, and the average power from all segments is used:
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{equation*}
S_n(f) = \frac{2}{MN^2\Delta f} \sum_{i=0}^{M-1} \left|{\rm FFT}(d(t_{1+i\alpha N: N(1+i\alpha)})w(t))\right|^2.
\end{empheq}-->
Windowing is _vital_ (when Fourier tranforming the analysis segment and for the PSR estimation) to prevent
<!--\href{https://en.wikipedia.org/wiki/Spectral\_leakage}{\bf \color{violet}{spectral leakage}}--> and add in correlations to the data.

### Time domain likelihood ###

For broadband signals, like those from CBC, the coloured nature of the noise generally means it's easier
to work in the frequency domain; in the frequency domain we just have vector-vector dot products of the data/signal and PSD rather than a vector-matrix product of the data/signal and the correlation matrix as in <!--Equation~\ref{eq:gauss}--> (i.e., its just a multiplication in the frequency domain rather than a <!--\href{https://en.wikipedia.org/wiki/Convolution\_theorem}{\bf \color{violet}{convolution}}-->).

But, some advantage of the time-domain are:

* you do not have to worry about windowing(!);
* time delays are just time delays rather than frequency dependent phase shifts;
* the start and end of signals can be simply defined.


### Multiple detectors ###

If you have $M$ detectors, assuming the noise in each is independent, you can coherently combine them
by taking the product of the likelihoods for each, so
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{align*}
p(\bm{\mathcal{D}}|\bm{\theta},I) & \propto \prod_{j=1}^M p(\bm{d}_j|\bm{\theta},I) \nonumber \\
& = \exp{\left(-\frac{1}{2}\left[4\sum_{j=1}^M\sum_{i=1}^{N_j}\frac{\left|\tilde{d}_{ij} - \tilde{s}_{ij}(\bm{\theta}) \right|^2}{T_j S_{n_j}^{(i)}}\right]\right)},
\end{empheq}-->
where $\bm{\mathcal{D}} = \{\bm{d}_1, \bm{d}_2, \dots, \bm{d}_M\}$ is the combined data
from all detectors.


### Analytical marginalisation ###

For the standard GW likelihood function and a certain forms of the signal model, it
is possible to analytically marginalise out certain parameters. For example, if the signal
consists of a sinusoidal term with an initial phase, e.g., $s \propto e^{i\phi_0}$, then
<!--\begin{empheq}[box={\borderedmathbox[scale=0.8]}]{align*}
p(\bm{d}|\bm{\theta}', I) &\propto \int_0^{2\pi} p(\bm{d}|\bm{\theta}) p(\phi_0|I) {\rm d}\phi_0, \nonumber \\
& = \exp{\left(-\frac{2}{T} \sum_i \frac{|s_i(\bm{\theta}')|^2 + |d_i|^2)}{S_n^{(i)}}\right)} I_0\left(\frac{4}{T} \left|\sum_i \frac{s_i(\bm{\theta}')d_i^*}{S_n^{(i)}}\right|\right),
\end{empheq}-->
where $\bm{\theta}'$ contains all the parameters of $\bm{\theta}$ except $\phi_0$ <!--\citep[see Equation (20) of][]{2015PhRvD..91d2003V}-->. 


### Hierarchical (multi-level) inference ###

Add one or two slides on this!

<!--% Bibliography slide -->
### Bibliography ###

<!--
\bibliographystyle{unsrtnat}

\begin{thebibliography}{99}

\bibitem[{{Veitch} {et~al.}(2015){Veitch}, \& et~al.}]{2015PhRvD..91d2003V}
{\scriptsize J.~{Veitch} et~al.
\newblock {\href{https://ui.adsabs.harvard.edu/abs/2015PhRvD..91d2003V/abstract}{\color{blue}{Parameter estimation for compact binaries with ground-based
  gravitational-wave observations using the LALInference software library}}}.
\newblock {\em Phys.\ Rev.\ D}, 91, 042003, 2015.}

\bibitem[{{Biwer} {et~al.}(2019){Biwer}, \& et~al.}]{Biwer:2018osg}
{\scriptsize C.~M. Biwer et~al.
\newblock {\href{https://ui.adsabs.harvard.edu/abs/2019PASP..131b4503B/abstract}{\color{blue}{PyCBC Inference: A Python-based parameter estimation toolkit for
  compact binary coalescence signals}}}.
\newblock {\em Publ. Astron. Soc. Pac.}, 131, 024503, 2019.}

\bibitem[{{Ashton} {et~al.}(2019){Ashton}, \& et~al.}]{Ashton:2019}
{\scriptsize G.\ Ashton et~al.
\newblock{\href{https://ui.adsabs.harvard.edu/abs/2019ApJS..241...27A/abstract}{\color{blue}{BILBY: A User-friendly Bayesian Inference Library for Gravitational-wave Astronomy}}}
\newblock{\em Astrophys. J. Supplement Series}, 241, 27, 2019.}

\bibitem[{{Sivia}(2006){Sivia}}]{Sivia}
{\scriptsize D.~S. {Sivia}.
\newblock {\em Data analysis: A Bayesian Tutorial}.
\newblock Oxford University Press, 2006.}

\bibitem[{{Abbott} {et~al.}(2016){Abbott}, \& et~al.}]{Abbott:2016}
{\scriptsize B.~P. Abbott et~al.
\newblock{\href{https://ui.adsabs.harvard.edu/abs/2016PhRvL.116x1102A/abstract}{\color{blue}{Properties of the Binary Black Hole Merger GW150914}}}
\newblock{\em Phys.\ Rev.\ Lett.}, 116, 241102, 2016.}

\bibitem[{{Veitch} \& {Vecchio}(2006){Veitch}, \& {Vecchio}}]{Veitch:2010}
{\scriptsize J.\ Veitch \& A.\ Vecchio
\newblock {\href{https://ui.adsabs.harvard.edu/abs/2010PhRvD..81f2003V/abstract}{\color{blue}{Bayesian coherent analysis of in-spiral gravitational wave signals with a detector network}}}.
\newblock {\em Phys.\ Rev.\ D}, 81, 062003, 2010.}

\end{thebibliography}
-->


### Appendix: Rules of probability ###

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


### Appendix: Rules of probability ###

* _Commutation_
    * $P(A,B) = P(B,A)$
    * $P(A \text{ or } B) = P(B \text{ or } A)$
* _Duality_ (De Morgan's Theorem)
    * $P(\text{not }[A,B]) = P(\text{not}(A) \text{ or } \text{not}(B))$
    * $P(\text{not }[A\text{ or }B]) = P(\text{not}(A),\text{not}(B))$


### Appendix: Rules of probability ###

Note that you may see other notation for probabilities expressed with Boolean logic (this list is 
not exhaustive)

* Negation ($A$ is false)
    * $P(\text{not }A)$, or $P(\bar{A})$, or $P(\lnot A)$
* Logical product (both A _and_ B are true)
    * $P(A,B)$, or $P(AB)$, or $P(A\text{ and }B)$, or $P(A\land B)$
* Logical sum (at least one of A _or_ B is true)
    * $P(A+B)$, or $P(A\text{ or }B)$, or $P(A\lor B)$


### Appendix: Rules of probability ###
    
From these axioms we can derive:

* The **(Extended) Sum Rule**
    * $P(A\text{ or }B) = P(A) + P(B) - P(A\text{ and }B)$
* The **Product Rule**
    * $P(A\text{ and }B) \equiv p(A,B) = P(A)P(B|A) = P(B)P(A|B)$, where $P(x|y)$ is the probability that $x$ is true given $y$ is true. 

These rules apply to probabilities $P$ and also probability density functions (pdfs) $p$.


### Appendix: Rules of probability ###

_Simple demonstration of the extended sum rule._

What is the probability that a card drawn from a standard deck of cards is a spade _or_ an ace?

We have $P(\spadesuit) = 13/52 = 1/4$ and $P(\text{ace}) = 4/52 = 1/13$, and $P(\spadesuit\text{ 
and ace}) = 1/(4\times 13) = 1/52$. It is reasonably obvious that for $P(\spadesuit\text{ or ace})$ 
we want to sum the probabilities for both cases, however they both contain the case where 
$P(\spadesuit\text{ and ace})$, so we have to remove one of those instances
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
P(\spadesuit\text{ or ace}) = \frac{13 + 4 - 1}{52} = \frac{16}{52}
\end{empheq}-->


### Appendix: fitting a line (unknown $\sigma$) ###

What if we don't know $\sigma$?

In this case we can treat $\sigma$ as another unknown variable and marginalise over it, e.g.
<!--\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
p(m,c|\bm{d},I) = p(m,c|I) \int_0^{\infty} p(d|m,c,\sigma,I)p(\sigma|I) {\rm d}\sigma
\end{empheq}-->
If the likelihood is Gaussian and we assume a flat prior on all parameters, e.g.
<!--\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
p(\sigma|I) = \begin{cases}C, \sigma > 0 \\ 0, \sigma \le 0\end{cases}
\end{empheq}-->
Then we have
<!--\begin{empheq}[box={\borderedmathbox[scale=0.7]}]{equation*}
p(m,c|\bm{d},I) \propto \int_0^{\infty} \sigma^{-n} \exp{\left(-\sum_{i=1}^n\frac{[d_i-(m x_i + c)]^2}{2\sigma^2}\right)} {\rm d}\sigma
\end{empheq}-->


### Appendix: fitting a line (unknown $\sigma$) ###

This integral is analytic, and through some substitution <!--\citep[see, e.g., Chap.\ 3 of][]{Sivia}-->, becomes
<!--\begin{empheq}[box={\borderedmathbox[scale=0.9]}]{equation*}
p(m,c|\bm{d},I) \propto \left( \sum_{i=1}^n [d_i-(m x_i + c)]^2 \right)^{-(n-1)/2}
\end{empheq}-->
This is essentially a <!--\href{https://en.wikipedia.org/wiki/Student\%27s\_t-distribution}{\bf \color{violet}{Student's $t$-distribution}}--> with $\nu = (n-2)$ degrees of freedom.

Note: if we were instead to use a prior on $\sigma$ of $p(\sigma|I) \propto 1/\sigma$ it would
lead to a Student's $t$-distribution with $\nu = n-1$ degrees of freedom.


### Appendix: fitting a line (unknown $\sigma$) ###

![][bayesian_line_fitting_2]

[bayesian_line_fitting_2]: figures/bayesian_line_fitting_2.pdf "Bayesian line fitting example" height="210px"

