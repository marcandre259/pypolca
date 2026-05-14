#include "pypolca/math_ops.h"
#include <Eigen/Dense>
#include <limits>
#include <stdexcept>

namespace pypolca {

// log P(y_i | class = r) for all i, r.
Eigen::MatrixXd compute_log_ylik(const Data &data, const Params &p,
                                 int nclass) {
  const int N = data.n_obs();
  const int R = nclass;

  const int M = data.n_items();

  const Eigen::VectorXd &vecprobs = p.vecprobs;

  int sum_choices = 0;
  for (int j = 0; j < M; j++) {
    sum_choices += data.num_choices[j];
  }

  Eigen::MatrixXd log_ylik(N, R);

  for (int i = 0; i < N; i++) {
    for (int r = 0; r < R; r++) {
      // Defined across M and choices
      int current_choice_pos = 0;
      double ll = 0.0;
      for (int j = 0; j < M; j++) {
        int obs_value = data.y(i, j);
        int cat_choices = data.num_choices[j];
        if (obs_value < 1) {
          current_choice_pos += cat_choices;
          continue;
        }
        int idx = r * sum_choices + current_choice_pos + (obs_value - 1);
        double prob_rjk = vecprobs(idx);
        if (prob_rjk <= 0) {
          ll = -std::numeric_limits<double>::infinity();
        } else {
          ll += std::log(prob_rjk);
        }
        current_choice_pos += cat_choices;
      }
      log_ylik(i, r) = ll;
    }
  }

  return log_ylik;
}

double compute_logsumexp(const Eigen::VectorXd &x) {
  double max_x = x.maxCoeff();
  int N = x.size();
  double sum = 0.0;

  for (int i = 0; i < N; i++) {
    sum += exp(x(i) - max_x);
  }

  return max_x + std::log(sum);
}

// Posterior class membership probabilities.
std::pair<Eigen::MatrixXd, double> e_step(const Data &data, const Params &p,
                                          const Eigen::MatrixXd &prior,
                                          int nclass) {
  const int N = data.n_obs();
  const int R = nclass;

  Eigen::MatrixXd log_ylik = compute_log_ylik(data, p, nclass);

  Eigen::MatrixXd posterior(N, nclass);
  Eigen::VectorXd log_nums(nclass);

  double total_loglik = 0.0;

  for (int i = 0; i < N; i++) {
    for (int r = 0; r < R; r++) {
      double log_prior = std::log(prior(i, r));
      double log_num = log_prior + log_ylik(i, r);
      log_nums(r) = log_num;
    }
    double log_denom = compute_logsumexp(log_nums);
    total_loglik += log_denom;

    for (int r = 0; r < R; r++) {
      posterior(i, r) = std::exp(log_nums(r) - log_denom);
    }
  }

  return {posterior, total_loglik};
}

// Update class-conditional response probabilities.
Eigen::VectorXd m_step_probs(const Data &data, const Eigen::MatrixXd &posterior,
                             const std::vector<int> &num_choices, int nclass) {
  int N = data.n_obs();
  int M = data.n_items();

  const Eigen::MatrixXi &y = data.y;

  int total_choices = 0;
  for (int k : num_choices) {
    total_choices += k;
  }

  Eigen::VectorXd vecprobs(nclass * total_choices);
  vecprobs.setZero();

  // I don't think I even need to loop through choice, I can just assign and so
  // on
  for (int r = 0; r < nclass; r++) {
    int pos = 0;
    for (int j = 0; j < M; j++) {
      int K = num_choices[j];
      std::vector<double> acc(K, 0.0);
      double sum = 0.0;
      for (int i = 0; i < N; i++) {
        int value = y(i, j);
        if (value > 0) {
          double post = posterior(i, r);
          acc[value - 1] += post;
          sum += post;
        }
      }
      for (int k = 0; k < K; k++) {
        vecprobs(r * total_choices + pos + k) = acc[k] / sum;
      }
      pos += K;
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

  if (beta.size() != (nclass - 1) * M) {
    throw std::invalid_argument(
        "beta size must equal (nclass - 1) * n_covariates");
  }

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

  int N = data.n_obs();
  int M = data.n_covariates();
  int rank = M * (nclass - 1);

  const Eigen::MatrixXd &x = data.x;

  Eigen::VectorXd gradients(rank);
  gradients.setZero();
  Eigen::MatrixXd hessians(rank, rank);
  hessians.setZero();

  if (nclass == 1) {
    return {gradients, hessians};
  }

  for (int i = 0; i < N; i++) {
    for (int m = 0; m < M; m++) {
      for (int r = 1; r < nclass; r++) {
        int row = M * (r - 1) + m;
        gradients(row) += x(i, m) * (posterior(i, r) - prior(i, r));
        for (int l = 0; l < M; l++) {
          int col = M * (r - 1) + l;
          hessians(row, col) +=
              x(i, m) * x(i, l) *
              (-1.0 * posterior(i, r) * (1 - posterior(i, r)) +
               prior(i, r) * (1 - prior(i, r)));
          for (int s = 1; s < r; s++) {
            int col2 = M * (s - 1) + l;
            hessians(row, col2) += x(i, m) * x(i, l) *
                                   (-1.0 * prior(i, r) * prior(i, s) +
                                    posterior(i, r) * posterior(i, s));
          }
        }
      }
    }
  }

  hessians.triangularView<Eigen::StrictlyUpper>() = hessians.transpose();

  return {gradients, hessians};
}

// Single Newton-Raphson step.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
update_beta(const Data &data, const Eigen::MatrixXd &posterior,
            const Eigen::MatrixXd &prior, const Eigen::VectorXd &beta,
            int nclass) {
  const Eigen::MatrixXd &x = data.x;

  int M = data.n_covariates();

  if (beta.size() != (nclass - 1) * M) {
    throw std::invalid_argument(
        "beta size must equal (nclass - 1) * n_covariates");
  }

  auto [gradients, hessians] =
      compute_beta_derivatives(data, posterior, prior, beta, nclass);

  Eigen::VectorXd updated_beta(beta.size());

  updated_beta = beta + hessians.ldlt().solve(gradients);

  Eigen::MatrixXd updated_prior =
      compute_prior_from_beta(x, updated_beta, nclass);

  return {updated_beta, updated_prior};
}

} // namespace pypolca
