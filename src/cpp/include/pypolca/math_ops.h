#pragma once

#include <Eigen/Dense>
#include <utility>

#include "pypolca/types.h"

namespace pypolca {

/**
 * Compute the log of class-conditional response likelihoods.
 *
 * For each observation i and class r, computes:
 *   log P(y_i | class = r) = sum_j log(P(y_{ij} | class = r))
 * with proper handling of missing values (coded as 0).
 *
 * @param data    Input data container.
 * @param p       Current parameter estimates.
 * @param nclass  Number of latent classes.
 * @return        N × nclass matrix of log-likelihood contributions.
 */
Eigen::MatrixXd compute_log_ylik(const Data &data, const Params &p, int nclass);

/**
 * E-step: compute posterior class membership probabilities.
 *
 * Uses Bayes' rule with current parameters and priors to update
 * posterior = P(class = r | y_i) for each observation.
 *
 * @param data    Input data container.
 * @param p       Current parameter estimates.
 * @param prior   N × nclass prior probabilities (from beta or uniform).
 * @param nclass  Number of latent classes.
 * @return        {posterior (N × nclass), total log-likelihood}.
 */
std::pair<Eigen::MatrixXd, double> e_step(const Data &data, const Params &p,
                                          const Eigen::MatrixXd &prior, int nclass);

/**
 * Numerically stable log-sum-exp of a vector.
 *
 * Computes log(sum(exp(x))) by shifting by max(x) to avoid overflow.
 *
 * @param x  Input vector.
 * @return   log(sum(exp(x))).
 */
double compute_logsumexp(const Eigen::VectorXd &x);

/**
 * M-step for class-conditional response probabilities.
 *
 * Updates the multinomial probabilities p_{jkr} = P(y_j = k | class = r)
 * by weighting observed responses by posterior membership.
 *
 * @param data         Input data container.
 * @param posterior    N × nclass posterior matrix from the E-step.
 * @param num_choices  Number of categories per item.
 * @param nclass       Number of latent classes.
 * @return             Updated flattened vecprobs vector.
 */
Eigen::VectorXd m_step_probs(const Data &data, const Eigen::MatrixXd &posterior,
                             const std::vector<int> &num_choices, int nclass);

/**
 * Compute gradient and observed information for covariate coefficients (beta).
 *
 * Used in the Newton-Raphson update for the multinomial logit model
 * on the class priors.
 *
 * @param data       Input data container.
 * @param posterior  N × nclass posterior matrix.
 * @param prior      N × nclass prior matrix.
 * @param beta       Current beta vector (flat, length S*(nclass-1)).
 * @param nclass     Number of latent classes.
 * @return           {gradient vector, negative Hessian (observed information)}.
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> compute_beta_derivatives(
    const Data &data, const Eigen::MatrixXd &posterior, const Eigen::MatrixXd &prior,
    const Eigen::VectorXd &beta, int nclass);

/**
 * One Newton-Raphson step for beta and corresponding prior update.
 *
 * Solves H * delta = grad and updates beta_new = beta + delta.
 * Recomputes prior from the new beta via multinomial logit.
 *
 * @param data       Input data container.
 * @param posterior  N × nclass posterior matrix.
 * @param prior      N × nclass prior matrix.
 * @param beta       Current beta vector.
 * @param nclass     Number of latent classes.
 * @return           {updated beta vector, updated prior matrix}.
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> update_beta(const Data &data,
                                                        const Eigen::MatrixXd &posterior,
                                                        const Eigen::MatrixXd &prior,
                                                        const Eigen::VectorXd &beta, int nclass);

/**
 * Build prior class-membership probabilities from covariates via multinomial logit.
 *
 * For each observation i and class r (except the reference class), computes:
 *   prior[i, r] = exp(x_i * beta_r) / (1 + sum_s exp(x_i * beta_s))
 * The reference class (last) gets the residual probability.
 *
 * @param x       N × S covariate matrix.
 * @param beta    Flat vector of coefficients, column-major S × (nclass-1).
 * @param nclass  Number of latent classes.
 * @return        N × nclass prior matrix (each row sums to 1).
 */
Eigen::MatrixXd compute_prior_from_beta(const Eigen::MatrixXd &x, const Eigen::VectorXd &beta,
                                        int nclass);

/**
 * Container for standard error estimates.
 */
struct SEs {
    /// Standard errors for vecprobs (same layout as Params::vecprobs).
    Eigen::VectorXd vecprobs_se;

    /// Standard errors for class population shares.
    Eigen::VectorXd P_se;

    /// Standard errors for beta coefficients.
    Eigen::VectorXd beta_se;

    /// Covariance matrix of beta coefficients.
    Eigen::MatrixXd beta_V;
};

/**
 * Compute standard errors from the observed information matrix.
 *
 * Inverts the observed Fisher information at the MLE to obtain
 * asymptotic covariance matrices and standard errors.
 *
 * @param data       Input data container.
 * @param params     Parameter estimates at convergence.
 * @param posterior  N × nclass posterior matrix.
 * @param prior      N × nclass prior matrix.
 * @param nclass     Number of latent classes.
 * @return           SEs struct with all standard error estimates.
 */
SEs compute_standard_errors(const Data &data, const Params &params,
                            const Eigen::MatrixXd &posterior, const Eigen::MatrixXd &prior,
                            int nclass);

}  // namespace pypolca
