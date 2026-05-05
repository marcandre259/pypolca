#include "pypolca/math_ops.h"
#include "Eigen/src/Core/Matrix.h"
#include <iostream>

namespace pypolca {

// P(y_i | class = r) for all i, r.
// Need to figure out how to deal with underflow: sum-log-exp trick?
Eigen::MatrixXd compute_ylik(const Data &data, const Params &p, int nclass) {
  const int N = data.n_obs();
  const int R = nclass;

  const int M = data.n_items();

  const Eigen::VectorXd vecprobs = p.vecprobs;

  int sum_choices = 0;
  for (int j = 0; j < M; j++) {
    sum_choices += data.num_choices[j];
  }

  Eigen::MatrixXd ylik(N, R);
  ylik.setOnes();

  for (int i = 0; i < N; i++) {
    for (int r = 0; r < R; r++) {
      // Defined across M and choices
      int current_choice_pos = 0;
      for (int j = 0; j < M; j++) {
        int obs_value = data.y(i, j);
        int cat_choices = data.num_choices[j];
        for (int k = 0; k < cat_choices; k++) {
          int idx = r * sum_choices + current_choice_pos;
          double prob_rjk = vecprobs(idx);
          if (obs_value == k + 1) {
            // Don't forget multiplication for each j of M
            ylik(i, r) *= prob_rjk;
          }
          current_choice_pos++;
        }
      }
    }
  }

  return ylik;
}

// Posterior class membership probabilities.
Eigen::MatrixXd e_step(const Data &data, const Params &p,
                       const Eigen::MatrixXd &prior, int nclass) {
  const int N = data.n_obs();
  const int R = nclass;

  Eigen::MatrixXd ylik = compute_ylik(data, p, nclass);

  Eigen::MatrixXd posterior(N, nclass);

  for (int i = 0; i < N; i++) {
    Eigen::VectorXd current_ylik = ylik.row(i);
    Eigen::VectorXd current_prior = prior.row(i);
    double denom = current_prior.dot(current_ylik);
    for (int r = 0; r < R; r++) {
      posterior(i, r) = current_prior(r) * current_ylik(r) / denom;
    }
  }

  return posterior;
}

// Update class-conditional response probabilities.
Eigen::VectorXd m_step_probs(const Data &data, const Eigen::MatrixXd &posterior,
                             const std::vector<int> &num_choices, int nclass) {
  int N = data.n_obs();
  int M = data.n_items();

  Eigen::MatrixXi y = data.y;

  int total_choices = 0;
  for (int k : num_choices) {
    total_choices += k;
  }

  Eigen::VectorXd vecprobs;
  vecprobs = Eigen::VectorXd::Zero(nclass * total_choices);

  for (int r = 0; r < nclass; r++) {
    int pos = 0;
    double sum_posteriors = posterior.col(r).sum();
    for (int j = 0; j < M; j++) {
      int current_choices = num_choices[j];
      for (int k = 0; k < current_choices; k++) {
        for (int i = 0; i < N; i++) {
          int value = y(i, j);
          if (value == k + 1) {
            vecprobs(r * total_choices + pos + k) += posterior(i, r);
          }
        }
        vecprobs(r * total_choices + pos + k) /= sum_posteriors;
      }
      pos += current_choices;
    }
  }

  return vecprobs;
}

// Multinomial logit (softmax) prior.
Eigen::MatrixXd compute_prior_from_beta(const Eigen::MatrixXd &x,
                                        const Eigen::VectorXd &beta,
                                        int nclass) {
  const int N = x.rows();
  const int M = x.cols();

  Eigen::Map<const Eigen::MatrixXd> beta_mat(beta.data(), M, nclass - 1);

  if (nclass == 1) {
    return Eigen::MatrixXd::Ones(N, 1);
  }

  Eigen::MatrixXd priors(N, nclass);

  for (int i = 0; i < N; i++) {
    Eigen::VectorXd eta = x.row(i) * beta_mat;
    double max_eta = fmax(0.0, eta.maxCoeff());

    double denom = exp(-max_eta);

    for (int r = 0; r < nclass - 1; r++) {
        denom += exp(eta[r] - max_eta);
    }

    priors(i, 0) = exp(-max_eta) / denom;

    for (int r = 1; r < nclass; r++) {
        priors(i, r) = exp(eta(r - 1) - max_eta) / denom;
    }
  }

  return priors;
}

// Gradient and observed information for Newton-Raphson.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
compute_beta_derivatives(const Data &data, const Eigen::MatrixXd &posterior,
                         const Eigen::MatrixXd &prior,
                         const Eigen::VectorXd &beta, int nclass) {

  const int S = data.n_covariates();
  const int rank = S * (nclass - 1);

  return {Eigen::VectorXd::Zero(rank), Eigen::MatrixXd::Identity(rank, rank)};
}

// TODO: Implement update_beta
// Single Newton-Raphson step.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
update_beta(const Data &data, const Eigen::MatrixXd &posterior,
            const Eigen::MatrixXd &prior, const Eigen::VectorXd &beta,
            int nclass) {
  (void)data;
  (void)posterior;
  (void)prior;
  (void)beta;
  (void)nclass;

  // STUB
  return {beta, prior};
}

} // namespace pypolca
