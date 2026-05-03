#pragma once

#include "polca/types.h"

namespace polca {

/**
 * Fit a latent class model via EM (and Newton-Raphson for covariates).
 *
 * @param data      Input data (responses + covariates).
 * @param nclass    Number of latent classes (R >= 1).
 * @param maxiter   Maximum EM iterations.
 * @param tol       Convergence tolerance for log-likelihood change.
 * @param verbose   If true, print iteration progress to stdout.
 * @param probs_start  Optional starting values for vecprobs (can be empty).
 * @param beta_start   Optional starting values for beta (can be empty).
 * @return          Fitted Results struct.
 */
Results fit_em(const Data& data,
               int nclass,
               int maxiter = 1000,
               double tol = 1e-10,
               bool verbose = false,
               const Eigen::VectorXd& probs_start = Eigen::VectorXd(),
               const Eigen::VectorXd& beta_start = Eigen::VectorXd());

} // namespace polca
