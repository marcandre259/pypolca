#pragma once

#include "pypolca/types.h"
#include <Eigen/Dense>
#include <utility>

namespace pypolca {

/**
 * Compute class-conditional response likelihoods P(y_i | class = r).
 *
 * Returns: N x nclass matrix.  Each entry is the product over items j of
 * the probability of observing y[i,j] given class r.
 */
Eigen::MatrixXd compute_ylik(const Data& data, const Params& p, int nclass);

/**
 * E-step: compute posterior class membership probabilities.
 *
 * Returns: N x nclass matrix of posterior probabilities.
 */
Eigen::MatrixXd e_step(const Data& data, const Params& p,
                       const Eigen::MatrixXd& prior, int nclass);

/**
 * M-step for response probabilities (no covariates).
 *
 * Returns: updated vecprobs vector.
 */
Eigen::VectorXd m_step_probs(const Data& data,
                             const Eigen::MatrixXd& posterior,
                             const std::vector<int>& num_choices,
                             int nclass);

/**
 * Compute gradient and observed information for the beta parameters.
 *
 * Returns: {gradient vector, negative Hessian (observed information matrix)}.
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
compute_beta_derivatives(const Data& data,
                         const Eigen::MatrixXd& posterior,
                         const Eigen::MatrixXd& prior,
                         const Eigen::VectorXd& beta,
                         int nclass);

/**
 * Update beta via one Newton-Raphson step and recompute priors.
 *
 * Returns: {new_beta, new_prior_matrix}.
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
update_beta(const Data& data,
            const Eigen::MatrixXd& posterior,
            const Eigen::MatrixXd& prior,
            const Eigen::VectorXd& beta,
            int nclass);

/**
 * Build prior matrix from beta via multinomial logit (softmax).
 *
 * beta: flat vector of length S*(R-1).
 * Returns: N x R matrix where each row sums to 1.
 */
Eigen::MatrixXd compute_prior_from_beta(const Eigen::MatrixXd& x,
                                        const Eigen::VectorXd& beta,
                                        int nclass);

} // namespace pypolca
