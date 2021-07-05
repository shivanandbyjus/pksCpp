# Name: GlimCpp.R
# Description: Estimates parameters of GLIM using MLE
#         
# Last mod: 05/Juli/2021, Julian Mollenhauer
##########################################

glimCpp <- function(K, N.R, Weights, tol = 1e-07, maxiter = 10000, fdb = FALSE){

  K     <- as.matrix(K)
  R     <- as.binmat(N.R, uniq = TRUE)
  Weights <- as.matrix(Weights)
  
  # input checks
  if(ncol(K) != ncol(R)) stop("Matrix K and R must have the same number of columns")
  if(ncol(K) != ncol(Weights)) stop("Matrix K and Weights must have the same number of columns")
  if(nrow(K) != nrow(Weights)) stop("Matrix K and Weights must have the same number of rows")



  N       <- sum(N.R) # sample size
  nitems  <- ncol(K)   # number of items q in Q
  nstates <- nrow(K)   # number of states k in K
  npat    <- nrow(R)   # number of unique response pattern

  ## set initial values
  P.K  <- rep(1/nstates, nstates) # probability vecotr P(K)
  beta <- rep(0.1, nitems)        # initital parameter estimations 
  eta  <- rep(0.1, nitems)

  W <- (R == 0) # W: wrong pattern
  R <- (R == 1) # R: right pattern

  iter <- 1
  maxdiff <- 2 * tol

  # convert to double for c++ function
  mode(R) <- mode(W) <- mode(K) <- mode(N.R) <- mode(P.K) <- mode(beta) <- 
             mode(eta) <- mode(Weights) <- "double"

  # EM Loop
  para <- emGLIMcpp(Weights, R, W, K, N.R, P.K, beta, eta, maxiter, tol, fdb)

  # warning if maxiter was reached
  if(para$maxiter) warning(paste("Maximum number of", maxiter, 
      " iterations was reached!"))
  if(!para$converged) warning(paste("EM-algorithm did NOT converge!"))

  # set names
  names(para$beta) <- names(para$eta) <- colnames(K)
  colnames(para$P.R)  <- names(N.R)
  para$P.R <- drop(para$P.R)
  rownames(para$PK.R) <- rownames(K)
  colnames(para$PK.R) <- rownames(R)


  # add information to output
  para$K <- K
  para$nitems <- nitems
  para$nstates <- nstates
  para$npatterns <- npat
  para$ntotal <- N
  para$N.R <- N.R

  # additional stats for glim print method
  if (sum(para$P.R) < 1)
        para$P.R <- para$P.R/sum(para$P.R)

  
  ## Mean number of errors
  P.Kq <- numeric(nitems)
  for(j in seq_len(nitems)){
    P.Kq[j] <- sum(para$P.K[which(K[,j] == 1)])
  }
  para$nerror <- c("careless error" = sum(para$beta * P.Kq),
    "lucky guess" = sum( para$eta * (1 - P.Kq)))


  
  ## Assigning state K given response R
  d.RK  <- apply(K, 1, function(k) colSums(xor(t(R), k)))
  d.min <- apply(d.RK, 1, min, na.rm = TRUE)             # minimum discrepancy
  i.RK  <- (d.RK <= (d.min)) & !is.na(d.RK)

  ## Minimum discrepancy distribution
  disc.tab <- xtabs(N.R ~ d.min)
  disc     <- as.numeric(names(disc.tab)) %*% disc.tab / N

  para$discrepancy <-  disc
  para$disc.tab <- disc.tab
  
  para$loglik <- sum(log(para$P.R) * N.R, na.rm = TRUE)
  para$fitted.values <- setNames(N * para$P.R, names(N.R))
  G2 <- 2 * sum(N.R * log(N.R/para$fitted.values), na.rm = TRUE)
  para$method = "ML"
  npar <- nstates - 1 + 2 * nitems
  df <- min(2^nitems - 1, N) - npar
  para$goodness.of.fit <- c(G2 = G2, df = df, pval = 1 - pchisq(G2, df))

  class(para) <- "glim"
  return(para)
}



