#include "pypolca/math_ops.h"
#include "pypolca/types.h"
#include <iostream>

int main() {
  pypolca::Data data;
  data.y = Eigen::MatrixXi(2, 2);
  data.y << 1, 2, 1, 3;
  data.num_choices = {2, 3};
  int n_classes = 2;

  std::cout << "y:\n" << data.y << std::endl;

  pypolca::Params p;
  p.vecprobs = Eigen::VectorXd(10);
  p.vecprobs << 0.3, 0.7, 0.1, 0.7, 0.2, 0.1, 0.9, 0.3, 0.5, 0.2;

  Eigen::VectorXd pi = Eigen::VectorXd::Constant(n_classes, 1.0 / n_classes);
  Eigen::MatrixXd prior = pi.transpose().replicate(data.n_obs(), 1);

  Eigen::MatrixXd posterior = pypolca::e_step(data, p, prior, n_classes);
  std::cout << "posterior:\n" << posterior << std::endl;

  Eigen::VectorXd updated_probs =
      pypolca::m_step_probs(
          data,
          posterior,
          data.num_choices,
          n_classes
      );

    std::cout << "updated probs:\n" << updated_probs << std::endl;

    return 0;
}
