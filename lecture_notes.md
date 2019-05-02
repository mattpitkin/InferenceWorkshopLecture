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

**Part 1**: _An introduction to statistics and inference_

* Rules of probability
* Bayes' theorem
* Important probability density functions (pdf)
* Moments of pdfs


### Introduction ###

This lecture course is based heavily on the <!--\href{http://www.astro.gla.ac.uk/users/martin/supa-da.html}{SUPA Advanced Data Analysis}--> course by Prof. Martin Hendry. 

There are many textbooks on statistics (and Bayesian statistics in particular), but three I can 
recommend are:

<!--\begin{columns}
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

[Sivia]: figures/sivia.jpg "Sivia" height="150px"
[Gregory]: figures/gregory.jpg "Gregory" height="150px"
[MacKay]: figures/mackay.jpg "MacKay" height="150px"

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
    * $P(A\text{ and }B) = P(A)P(B|A) = P(B)P(A|B)$

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

We can calculate the these terms (e.g., analytically or numerically on a computer). 


### Bayes' theorem ###

* **Prior**: what we knew, or our degree of belief, about our model before taking data
* **Likelihood**: the influence of the data in updating our degree of belief 
* **Evidence**: the ``_evidence_'' for the data, or the likelihood for the data _marginalised_ over 
the model (we'll explore this later, but at the moment note it as a constant normalisation factor 
for the posterior)
* **Posterior**: our new degree of belief about our model in light of the data


### Anatomy of the GW likelihood function ###

We've seen the "standard" Gaussian likelihood function, but in GW papers you
might (see, e.g., Equations 3 and 4 of [][#Biwer:2018osg], assuming a single detector,
and ignoring the normalisation):
<!--\begin{empheq}[box={\borderedmathbox[scale=0.6]}]{align*}
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

<!--% Bibliography slide -->
### Bibliography ###

<!--
\bibliographystyle{unsrt}
\bibliography{lecture_notes}
-->
