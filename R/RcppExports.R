# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

emBLIMcpp <- function(R, W, K, NR, PKr, betar, etar, maxiter = 10000L, tol = 1e-07, fdb = TRUE) {
    .Call('_pksCpp_emBLIMcpp', PACKAGE = 'pksCpp', R, W, K, NR, PKr, betar, etar, maxiter, tol, fdb)
}

emGLIMcpp <- function(weights, R, W, K, NRr, PKr, betar, etar, maxiter = 1000000L, tol = 1e-07, fdb = TRUE) {
    .Call('_pksCpp_emGLIMcpp', PACKAGE = 'pksCpp', weights, R, W, K, NRr, PKr, betar, etar, maxiter, tol, fdb)
}

