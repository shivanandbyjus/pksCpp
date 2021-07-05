#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;

// [[Rcpp::export]]
Rcpp::List emGLIMcpp(
    const Eigen::Map < Eigen::MatrixXd > weights, //weights
    const Eigen::Map < Eigen::MatrixXd > R,       // right answers
    const Eigen::Map < Eigen::MatrixXd > W,       // wrong answers (1 - R)
    const Eigen::Map < Eigen::MatrixXd > K,
    const Eigen::Map < Eigen::VectorXd > NRr,
    const Eigen::Map < Eigen::VectorXd > PKr,
    const Eigen::Map < Eigen::VectorXd > betar,
    const Eigen::Map < Eigen::VectorXd > etar,
    const int maxiter = 1000000,
    const double tol = 1e-07,
    const bool fdb = true

    ) {

  //copy mapped input
  Eigen::VectorXd PK = PKr;
  Eigen::VectorXd eta = etar;
  Eigen::VectorXd beta = betar;
  Eigen::VectorXd NR = NRr;

        
  Eigen::MatrixXd PRK;
  Eigen::MatrixXd PR;
  Eigen::MatrixXd MKR;
  //Eigen::MatrixXd W  = (1 - R.array()).matrix(); // wrong items; 1 - R
  Eigen::MatrixXd nK = (1 - K.array()).matrix(); // items not in K; 1 - K

  Eigen::MatrixXd lbeta_w;
  Eigen::MatrixXd lrbeta_w; 
  Eigen::MatrixXd leta_w;   
  Eigen::MatrixXd lreta_w; 

  Eigen::MatrixXd diff(3, 1);

  Eigen::VectorXd PKold;
  Eigen::VectorXd etaold;
  Eigen::VectorXd betaold;

  int iter = 0;
  double eps = 1e-09;
  bool converged = false;


  // EM-LOOP
  do {

    // E-Step
    //prevent log zero // maybe .array() <= eps is faster....
    for (int i = 0; i < eta.rows(); i++) {
      if (eta(i) < eps) {
        eta(i) = eps;
      } else if (eta(i) > (1 - eps)) {
        eta(i) = 1 - eps;
      }

      if (beta(i) < eps) {
        beta(i) = eps;
      } else if (beta(i) > (1 - eps)) {
        beta(i) = 1 - eps;
      }
    }


    // PRK
  lbeta_w   = ((beta.replicate(1, weights.rows()).transpose()
        ).cwiseProduct(weights)).array().log().matrix();
  lrbeta_w  = (1 - ((beta.replicate(1, weights.rows()).transpose()
          ).cwiseProduct(weights)).array()).log().matrix();
  leta_w    = ((eta.replicate(1, weights.rows()).transpose()
        ).cwiseProduct(weights)).array().log().matrix();
  lreta_w   = (1 - ((eta.replicate(1, weights.rows()).transpose()
          ).cwiseProduct(weights)).array()).log().matrix();
  
    PRK = (
        (K.cwiseProduct(lbeta_w) * ((1 - R.array()).matrix()).transpose()) +
        (K.cwiseProduct(lrbeta_w) * (R).transpose()) +
        (((1 - K.array()).matrix()).cwiseProduct(leta_w) * (R).transpose()) +
        (((1 - K.array()).matrix()).cwiseProduct(lreta_w) * ((1 - R.array()).matrix()).transpose())  
        ).array().exp().matrix().transpose();

    PR = PRK * PK;

    MKR = (((PK * PR.cwiseInverse().transpose()).cwiseProduct(PRK.transpose())
           ).transpose()).cwiseProduct(NR.replicate(1, K.rows()));


    // M - Step

    // save old estimates
    PKold = PK;
    etaold = eta;
    betaold = beta;

    // PK
    PK = (MKR.colwise().sum());
    PK /= NR.sum();

    // beta, eta
    beta = 
        (((MKR * K).cwiseProduct(W)).colwise().sum()         // beta.num
        ).cwiseQuotient(
        (MKR * (K.cwiseProduct(weights))).colwise().sum());  // beta.denom
  
    eta = 
        (((MKR * nK).cwiseProduct(R)).colwise().sum()         // eta.num
        ).cwiseQuotient(
        (MKR * (nK.cwiseProduct(weights))).colwise().sum());  // eta.denom


    diff(0, 0) = ((PKold - PK).cwiseAbs().maxCoeff()) < tol;
    diff(1, 0) = ((etaold - eta).cwiseAbs().maxCoeff()) < tol;
    diff(2, 0) = ((betaold - beta).cwiseAbs().maxCoeff()) < tol;

    iter++;

    if (fdb) {
      if (iter % 50 == 0) {
        Rcpp::Rcout << ".  " << "Iteration #: " << iter << std::endl;
        Rcpp::checkUserInterrupt();
      } else {
        Rcpp::Rcout << ".";
      }
    }
  }
  while (diff.sum() < 3 && iter < maxiter);

  if (diff.sum() >= 3) {
    converged = true;
  }

  if (fdb) {
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << "     **** DONE ****" << std::endl;

    if (iter >= maxiter) {
      Rcpp::Rcout << "Maximum # of " << maxiter << " Iterations reached." << std::endl;
    }

    if (converged == false) {
      Rcpp::Rcout << "EM-Algorithm did NOT converged!" << std::endl;
    }

  }



  // compute PKR with final estimates
  for (int i = 0; i < eta.rows(); i++) {
    if (eta(i) < eps) {
      eta(i) += eps;
    } else if (eta(i) > (1 - eps)) {
      eta(i) -= eps;
    }

    if (beta(i) < eps) {
      beta(i) += eps;
    } else if (beta(i) > (1 - eps)) {
      beta(i) -= eps;
    }
  } 



   //PRK
  lbeta_w   = ((beta.replicate(1, weights.rows()).transpose()
        ).cwiseProduct(weights)).array().log().matrix();
  lrbeta_w  = (1 - ((beta.replicate(1, weights.rows()).transpose()
          ).cwiseProduct(weights)).array()).log().matrix();
  leta_w    = ((eta.replicate(1, weights.rows()).transpose()
        ).cwiseProduct(weights)).array().log().matrix();
  lreta_w   = (1 - ((eta.replicate(1, weights.rows()).transpose()
          ).cwiseProduct(weights)).array()).log().matrix();

    PRK = (
        (K.cwiseProduct(lbeta_w) * ((1 - R.array()).matrix()).transpose()) +
        (K.cwiseProduct(lrbeta_w) * (R).transpose()) +
        (((1 - K.array()).matrix()).cwiseProduct(leta_w) * (R).transpose()) +
        (((1 - K.array()).matrix()).cwiseProduct(lreta_w) * ((1 - R.array()).matrix()).transpose())  
        ).array().exp().matrix().transpose();


  PR = PRK * PK;

  MKR = (((PK * PR.cwiseInverse().transpose()).cwiseProduct(PRK.transpose())).transpose()).cwiseProduct(NR.replicate(1, K.rows()));

      // PK
    PK = (MKR.colwise().sum());
    PK /= NR.sum();

  


  return Rcpp::List::create(Rcpp::Named("P.K") = PK,
      Rcpp::Named("beta") = beta,
      Rcpp::Named("eta") = eta,
      Rcpp::Named("iter") = iter,
      Rcpp::Named("maxiter") = iter >= maxiter,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("PK.R") = MKR.cwiseQuotient(NR.replicate(1, K.rows())).transpose(),
      Rcpp::Named("P.R") = PR.transpose()


      );

}
