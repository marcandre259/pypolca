#pragma once

#include <Eigen/Dense>
#include <vector>

namespace pypolca {

/**
 * Input data container for latent class analysis.
 *
 * Holds response matrix, covariate design matrix, and per-item category counts.
 * Responses are 1-based with 0 indicating missing values.
 */
struct Data {
    /// N × J response matrix (1-based integers, 0 = missing).
    Eigen::MatrixXi y;

    /// N × S covariate/design matrix. Include an intercept column manually.
    Eigen::MatrixXd x;

    /// Number of response categories for each item (length J).
    std::vector<int> num_choices;

    /// Number of observations (rows of y).
    int n_obs() const {
        return static_cast<int>(y.rows());
    }

    /// Number of items (columns of y).
    int n_items() const {
        return static_cast<int>(y.cols());
    }

    /// Number of covariates (columns of x).
    int n_covariates() const {
        return static_cast<int>(x.cols());
    }
};

/**
 * Latent class model parameters.
 *
 * Encapsulates class-conditional response probabilities and covariate
 * coefficients (beta) in flattened vector layouts suitable for Eigen.
 */
struct Params {
    /** Flattened class-conditional response probabilities.
     *  Layout: [class 0: item 0 cat 0..K, item 1 cat 0..K, ...], [class 1: ...].
     *  Total length = nclass * sum_j(num_choices[j]).
     */
    Eigen::VectorXd vecprobs;

    /** Flattened covariate coefficients (multinomial logit parametrization).
     *  Layout: column-major S × (nclass-1) matrix stored as a single vector.
     *  The last class is the reference category (coefficients fixed at 0).
     */
    Eigen::VectorXd beta;
};

/**
 * Complete output from a latent class model fit.
 *
 * Contains parameter estimates, posterior membership probabilities,
 * convergence diagnostics, and (optionally) standard errors.
 */
struct Results {
    /// Estimated model parameters (vecprobs and beta).
    Params params;

    /// N × nclass posterior class-membership probabilities.
    Eigen::MatrixXd posterior;

    /// N × nclass prior class probabilities per observation.
    Eigen::MatrixXd prior;

    /// Final log-likelihood.
    double loglik = 0.0;

    /// Number of EM iterations performed.
    int iterations = 0;

    /// Whether the EM algorithm converged before reaching maxiter.
    bool converged = false;

    /// Whether a fatal error occurred during fitting.
    bool error = false;

    /// Standard errors for vecprobs (same layout as Params::vecprobs).
    Eigen::VectorXd vecprobs_se;

    /// Standard errors for class population shares (length nclass).
    Eigen::VectorXd P_se;

    /// Standard errors for beta coefficients (same layout as Params::beta).
    Eigen::VectorXd beta_se;

    /// Covariance matrix of beta coefficients (S(nclass-1) × S(nclass-1)).
    Eigen::MatrixXd beta_V;
};

}  // namespace pypolca
