#pragma once

#include "pypolca/types.h"

namespace pypolca {

/**
 * Fit a latent class model via EM with optional Newton-Raphson covariate updates.
 *
 * The algorithm alternates between:
 *   1. E-step: compute posterior membership from current parameters.
 *   2. M-step: update response probabilities from weighted responses.
 *   3. If covariates present: Newton-Raphson update of beta coefficients.
 *
 * If nrep > 1, the best fit (highest log-likelihood) across random starts is returned.
 *
 * @param data         Input data container (responses and optional covariates).
 * @param nclass       Number of latent classes (≥ 1).
 * @param maxiter      Maximum EM iterations (default: 1000).
 * @param tol          Convergence tolerance on log-likelihood change (default: 1e-10).
 * @param probs_start  Optional starting vecprobs vector (empty = random init).
 * @param beta_start   Optional starting beta vector (empty = zeros or random).
 * @param seed         Random seed for reproducibility (default: 42).
 * @param calc_se      Whether to compute standard errors at convergence (default: true).
 * @return             Fitted Results containing parameters, posterior, diagnostics, and SEs.
 */
Results fit_em(const Data &data, int nclass, int maxiter = 1000, double tol = 1e-10,
               const Eigen::VectorXd &probs_start = Eigen::VectorXd(),
               const Eigen::VectorXd &beta_start = Eigen::VectorXd(), const unsigned int seed = 42,
               bool calc_se = true);

}  // namespace pypolca
