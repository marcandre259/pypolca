#include "pypolca/em_engine.h"
#include "pypolca/math_ops.h"
#include "pypolca/types.h"
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <random>

namespace pypolca {

static Eigen::VectorXd random_init_probs(const std::vector<int> &num_choices,
                                         int nclass, unsigned int seed) {
  Eigen::Index total = 0;

  for (int k : num_choices) {
    total += k;
  }

  std::mt19937 gen(seed);
  std::gamma_distribution<double> gamma(1.0, 1.0); // Exponential

  Eigen::VectorXd vecprobs(nclass * total);
  Eigen::Index J = num_choices.size();

  for (int r = 0; r < nclass; r++) {
    int pos = 0;
    for (int j = 0; j < J; j++) {
      int K = num_choices[j];
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        double draw = gamma(gen);
        vecprobs(r * total + pos + k) = draw;
        sum += draw;
      }
      for (int k = 0; k < K; k++) {
        vecprobs(r * total + pos + k) /= sum;
      }
      pos += K;
    }
  }

  return vecprobs;
}

Results fit_em(const Data &data, int nclass, int maxiter, double tol,
               const Eigen::VectorXd &probs_start,
               const Eigen::VectorXd &beta_start, unsigned int seed,
               bool calc_se = true) {
  const int N = data.n_obs();
  const int S = data.n_covariates();

  Results res;
  Params p;

  // --- Initialize response probabilities ---
  if (probs_start.size() > 0) {
    p.vecprobs = probs_start;
  } else {
    p.vecprobs = random_init_probs(data.num_choices, nclass, seed);
  }

  // --- Initialize beta / prior ---
  if (beta_start.size() > 0) {
    p.beta = beta_start;
  } else {
    p.beta = Eigen::VectorXd::Zero(S * (nclass - 1));
  }

  Eigen::MatrixXd prior;
  if (S > 1) {
    prior = compute_prior_from_beta(data.x, p.beta, nclass);
  } else {
    Eigen::VectorXd pi = Eigen::VectorXd::Constant(nclass, 1.0 / nclass);
    prior = pi.transpose().replicate(N, 1);
  }

  // EM loop: iterate until |dll| < tol, maxiter reached, or likelihood drops.
  double dll = std::numeric_limits<double>::infinity();
  double log_lik_latest = std::numeric_limits<double>::infinity();
  int n_iter = 0;
  bool converged = true;
  bool error = false;

  Eigen::MatrixXd posterior(N, nclass);

  {
    std::pair<Eigen::MatrixXd, double> result = e_step(data, p, prior, nclass);
    posterior = result.first;
    log_lik_latest = result.second;
  }

  while (std::abs(dll) >= tol) {
    n_iter += 1;
    double log_lik_prev = log_lik_latest;

    p.vecprobs = m_step_probs(data, posterior, data.num_choices, nclass);

    if (S > 1) {
      auto result = update_beta(data, posterior, prior, p.beta, nclass);
      p.beta = result.first;
      prior = result.second;
    } else {
      Eigen::VectorXd col_means(nclass);
      for (int r = 0; r < nclass; r++) {
        col_means(r) = posterior.col(r).mean();
      }
      prior = col_means.transpose().replicate(N, 1);
    }

    std::pair<Eigen::MatrixXd, double> result = e_step(data, p, prior, nclass);
    posterior = result.first;
    log_lik_latest = result.second;

    dll = log_lik_latest - log_lik_prev;

    if (S > 1 && dll < -1e-7) {
      converged = false;
      error = true;
      break;
    } else if (n_iter >= maxiter) {
      converged = false;
      break;
    }
  }

  res.converged = converged;
  res.loglik = log_lik_latest;
  res.iterations = n_iter;
  res.posterior = posterior;
  res.prior = prior;
  res.params = p;

  if (calc_se && !error) {
    auto ses = compute_standard_errors(data, p, posterior, prior, nclass);
    res.vecprobs_se = ses.vecprobs_se;
    res.P_se = ses.P_se;
    res.beta_se = ses.beta_se;
    res.beta_V = ses.beta_V;
  }

  return res;
}

} // namespace pypolca
