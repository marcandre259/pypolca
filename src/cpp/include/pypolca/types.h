#pragma once

#include <Eigen/Dense>
#include <vector>

namespace pypolca {

/**
 * Input data container.
 *   y:            N x J matrix of responses. Values are 1-based; 0 = missing.
 *   x:            N x S covariate/design matrix (include intercept manually).
 *   num_choices:  Length-J vector; num_choices[j] = number of categories for
 * item j.
 */
struct Data {
    Eigen::MatrixXi y;
    Eigen::MatrixXd x;
    std::vector<int> num_choices;

    int n_obs() const {
        return static_cast<int>(y.rows());
    }
    int n_items() const {
        return static_cast<int>(y.cols());
    }
    int n_covariates() const {
        return static_cast<int>(x.cols());
    }
};

/**
 * Model parameters.
 *   vecprobs:  Flattened class-conditional response probabilities.
 *              Layout: for each class r, then each item j, then each category
 * k. Total length = nclass * sum_j(num_choices[j]). beta:      Flattened
 * covariate coefficients. Layout: column-major S x (nclass-1) matrix stored as
 * vector.
 */
struct Params {
    Eigen::VectorXd vecprobs;
    Eigen::VectorXd beta;
};

/**
 * Fitted model results.
 */
struct Results {
    Params params;
    Eigen::MatrixXd posterior;  // N x nclass (class membership probabilities)
    Eigen::MatrixXd prior;      // N x nclass (prior class probs per observation)
    double loglik = 0.0;
    int iterations = 0;
    bool converged = false;
    bool error = false;

    // --- Standard errors (populate later) ---
    Eigen::VectorXd vecprobs_se;  // flat, same layout as Params::vecprobs
    Eigen::VectorXd P_se;         // SE of class population shares
    Eigen::VectorXd beta_se;      // SE of beta coefficients
    Eigen::MatrixXd beta_V;       // COV matrix of beta coefficients
};

}  // namespace pypolca
