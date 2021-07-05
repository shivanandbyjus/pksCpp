\name{glimCpp}
\alias{glimCpp}

\title{Generalized Local Independence Model (GLIM)}
\description{Fits a generalized local independence model (GLIM) for probabilistic
  knowledge structures by maximum likelihood estimation.
}
\usage{
glimCpp(K, N.R, Weights, tol = 1e-07, maxiter = 100000, fdb = FALSE)

}

\arguments{
  \item{K}{a state-by-problem indicator matrix representing the knowledge
    structure.  An element is one if the problem is contained in the state,
    and else zero.}
  \item{N.R}{a (named) vector of absolute frequencies of response patterns.}
  \item{Weights}{state-by-problem matrix representing weighting factors to be applied to the problem-specific error rates to model their dependence on the states.}
  \item{tol}{tolerance, stopping criterion for iteration.}
  \item{maxiter}{the maximum number of iterations.}
  \item{fdb}{feedback during the optimization.}

}

\details{
  See Doignon and Falmagne (1999) for details on the basic local independence
  model (BLIM) for probabilistic knowledge structures and ... for the generalization.

 ....

}

\value{
  An object of class \code{glim} having (at least) the following components:
  \item{discrepancy}{the mean minimum discrepancy between response patterns
    and knowledge states.}
  \item{P.K}{the vector of estimated parameter values for probabilities of
    knowledge states.}
  \item{beta}{the vector of estimated parameter values for probabilities of
    a careless error.}
  \item{eta}{the vector of estimated parameter values for probabilities of a
    lucky guess.}
  \item{disc.tab}{the minimum discrepancy distribution.}
  \item{K}{the knowledge structure.}
  \item{N.R}{the vector of frequencies of response patterns.}
  \item{nitems}{the number of items.}
  \item{nstates}{the number of knowledge states.}
  \item{npatterns}{the number of response patterns.}
  \item{ntotal}{the number of respondents.}
  \item{nerror}{the number of response errors.}
  \item{npar}{the number of parameters.}
  \item{method}{the parameter estimation method.}
  \item{iter}{the number of iterations needed.}
  \item{loglik}{the log-likelihood.}
  \item{fitted.values}{the fitted response frequencies.}
  \item{goodness.of.fit}{the goodness of fit statistic including the
    likelihood ratio fitted vs. saturated model (G2), the degrees of
    freedom, and the p-value of the corresponding chi-square distribution.
    (See Details.)}
}

\references{
  Doignon, J.-P., & Falmagne, J.-C. (1999).
  \emph{Knowledge spaces}. Berlin: Springer.

  Heller, J., & Wickelmaier, F. (2013).
  Minimum discrepancy estimation in probabilistic knowledge structures.
  \emph{Electronic Notes in Discrete Mathematics}, \bold{42}, 49--56.
  \doi{10.1016/j.endm.2013.05.145}

}

\seealso{
  \code{\link{simulate.blim}}, \code{\link{plot.blim}},
  \code{\link{residuals.blim}}, \code{\link{logLik.blim}},
  \code{\link{delineate}}, \code{\link{jacobian}}, \code{\link{endm}},
  \code{\link{probability}}, \code{\link{chess}}.
}

\examples{
data(DoignonFalmagne7)
K   <- DoignonFalmagne7$K         # knowledge structure
N.R <- DoignonFalmagne7$N.R       # frequencies of response patterns

w <- exp(- 0.5 * 
   matrix(c(1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 3, 
3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 
3, 2, 1, 2, 1, 1, 1) - 1, 9, 5))      # weighting factors

## Fit basic local independence model (BLIM) 
glimCpp(K, N.R, w)
}
