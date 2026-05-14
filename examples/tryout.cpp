#include "pypolca/math_ops.h"
#include "pypolca/types.h"
#include <iostream>

int main() {
  // ============================================================
  // Setup: small synthetic dataset
  //   N = 3 observations, M = 2 items
  //   Item 0: 2 choices, Item 1: 3 choices
  // ============================================================
  pypolca::Data data;
  data.y = Eigen::MatrixXi(3, 2);
  data.y << 1, 2,
            2, 3,
            1, 1;
  data.num_choices = {2, 3};

  int n_classes = 2;
  int total_choices = 2 + 3; // 5

  std::cout << "===== Data =====" << std::endl;
  std::cout << "y (N=" << data.n_obs() << ", M=" << data.n_items() << "):\n"
            << data.y << std::endl;
  std::cout << "num_choices: {2, 3}" << std::endl;
  std::cout << "n_classes: " << n_classes << std::endl;

  // ============================================================
  // Params: vecprobs layout = nclass * total_choices = 2 * 5 = 10
  //   For class 0: [p00, p01, p02, p03, p04]
  //   For class 1: [p10, p11, p12, p13, p14]
  // ============================================================
  pypolca::Params p;
  p.vecprobs = Eigen::VectorXd(10);
  // Class 0: item0(0.3, 0.7), item1(0.1, 0.7, 0.2)
  // Class 1: item0(0.9, 0.1), item1(0.3, 0.5, 0.2)
  p.vecprobs << 0.3, 0.7, 0.1, 0.7, 0.2,
                0.9, 0.1, 0.3, 0.5, 0.2;

  std::cout << "\n===== Params =====" << std::endl;
  std::cout << "vecprobs:\n" << p.vecprobs.transpose() << std::endl;

  // ============================================================
  // 1. compute_log_ylik
  // ============================================================
  std::cout << "\n===== 1. compute_log_ylik =====" << std::endl;
  Eigen::MatrixXd ylik = pypolca::compute_log_ylik(data, p, n_classes);
  std::cout << "ylik (N x nclass):\n" << ylik << std::endl;
  // Manual check for obs 0, class 0:
  //   item0: y=1 -> 0.3, item1: y=2 -> 0.7
  //   product = 0.3 * 0.7 = 0.21
  // Obs 0, class 1:
  //   item0: y=1 -> 0.9, item1: y=2 -> 0.5
  //   product = 0.9 * 0.5 = 0.45

  // ============================================================
  // 2. e_step
  // ============================================================
  std::cout << "\n===== 2. e_step =====" << std::endl;
  Eigen::VectorXd pi = Eigen::VectorXd::Constant(n_classes, 1.0 / n_classes);
  Eigen::MatrixXd prior = pi.transpose().replicate(data.n_obs(), 1);
  std::cout << "prior (uniform):\n" << prior << std::endl;

  auto [posterior, loglik] = pypolca::e_step(data, p, prior, n_classes);
  std::cout << "posterior:\n" << posterior << std::endl;
  std::cout << "loglik: " << loglik << std::endl;
  // With uniform prior, posterior(i,r) = ylik(i,r) / sum_r(ylik(i,r))

  // ============================================================
  // 3. m_step_probs
  // ============================================================
  std::cout << "\n===== 3. m_step_probs =====" << std::endl;
  Eigen::VectorXd updated_probs =
      pypolca::m_step_probs(data, posterior, data.num_choices, n_classes);
  std::cout << "updated vecprobs:\n" << updated_probs.transpose() << std::endl;

  // ============================================================
  // 4. compute_prior_from_beta
  // ============================================================
  std::cout << "\n===== 4. compute_prior_from_beta =====" << std::endl;
  // 3 covariates (including intercept), 2 classes -> beta length = 3 * 1 = 3
  data.x = Eigen::MatrixXd(3, 3);
  data.x << 1.0, 0.5, -0.2,
            1.0, 1.0,  0.0,
            1.0, 0.0,  0.8;
  Eigen::VectorXd beta(3);
  beta << 0.5, -0.3, 0.1; // one column for class 1 (class 0 is reference)

  std::cout << "x:\n" << data.x << std::endl;
  std::cout << "beta:\n" << beta.transpose() << std::endl;

  Eigen::MatrixXd new_prior =
      pypolca::compute_prior_from_beta(data.x, beta, n_classes);
  std::cout << "computed prior (rows sum to 1):\n" << new_prior << std::endl;
  std::cout << "row sums: " << new_prior.rowwise().sum().transpose() << std::endl;

  // ============================================================
  // 5. compute_beta_derivatives (stub)
  // ============================================================
  std::cout << "\n===== 5. compute_beta_derivatives =====" << std::endl;
  auto [grad, info] = pypolca::compute_beta_derivatives(
      data, posterior, new_prior, beta, n_classes);
  std::cout << "gradient (stub -> zeros):\n" << grad.transpose() << std::endl;
  std::cout << "info matrix (stub -> identity):\n" << info << std::endl;

  // ============================================================
  // 6. update_beta (stub)
  // ============================================================
  std::cout << "\n===== 6. update_beta =====" << std::endl;
  auto [new_beta, updated_prior] = pypolca::update_beta(
      data, posterior, new_prior, beta, n_classes);
  std::cout << "new_beta (stub -> same as input):\n"
            << new_beta.transpose() << std::endl;
  std::cout << "updated_prior (stub -> same as input):\n"
            << updated_prior << std::endl;

  std::cout << "\n===== Done =====" << std::endl;
  return 0;
}
