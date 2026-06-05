#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pypolca/em_engine.h"
#include "pypolca/math_ops.h"
#include "pypolca/types.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for pypolca (Polytomous Variable Latent Class Analysis)";

    // --- Data struct ---
    py::class_<pypolca::Data>(m, "Data")
        .def(py::init<>())
        .def_readwrite("y", &pypolca::Data::y, "N x J response matrix (1-based, 0=missing)")
        .def_readwrite("x", &pypolca::Data::x, "N x S covariate matrix")
        .def_readwrite("num_choices", &pypolca::Data::num_choices, "Categories per item")
        .def("n_obs", &pypolca::Data::n_obs)
        .def("n_items", &pypolca::Data::n_items)
        .def("n_covariates", &pypolca::Data::n_covariates);

    // --- Params struct ---
    py::class_<pypolca::Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("vecprobs", &pypolca::Params::vecprobs)
        .def_readwrite("beta", &pypolca::Params::beta);

    // --- Results struct ---
    py::class_<pypolca::Results>(m, "Results")
        .def_readonly("params", &pypolca::Results::params)
        .def_readonly("posterior", &pypolca::Results::posterior, "N x nclass posterior probs")
        .def_readonly("prior", &pypolca::Results::prior, "N x nclass prior probs")
        .def_readonly("loglik", &pypolca::Results::loglik)
        .def_readonly("iterations", &pypolca::Results::iterations)
        .def_readonly("converged", &pypolca::Results::converged)
        .def_readonly("error", &pypolca::Results::error)
        .def_readonly("vecprobs_se", &pypolca::Results::vecprobs_se)
        .def_readonly("P_se", &pypolca::Results::P_se)
        .def_readonly("coeff_se", &pypolca::Results::beta_se)
        .def_readonly("beta_V", &pypolca::Results::beta_V);

    // --- fit_em ---
    m.def("fit_em", &pypolca::fit_em, py::arg("data"), py::arg("nclass"), py::arg("maxiter") = 1000,
          py::arg("tol") = 1e-10, py::arg("probs_start") = Eigen::VectorXd(),
          py::arg("beta_start") = Eigen::VectorXd(), py::arg("seed") = 42,
          py::arg("calc_se") = true,
          R"pbdoc(
          Fit a latent class model via EM.

          Parameters
          ----------
          data : pypolca.Data
              Input data container.
          nclass : int
              Number of latent classes.
          maxiter : int
              Maximum EM iterations.
          tol : float
              Convergence tolerance.
          probs_start : np.ndarray
              Optional starting values for response probabilities.
          beta_start : np.ndarray
              Optional starting values for covariate coefficients.

          Returns
          -------
          pypolca.Results
              Fitted model results.
          )pbdoc");

    // --- Expose math helpers for unit testing ---
    m.def("compute_ylik", &pypolca::compute_log_ylik, py::arg("data"), py::arg("params"),
          py::arg("nclass"), "Compute class-conditional likelihoods (N x nclass).");

    m.def("e_step", &pypolca::e_step, py::arg("data"), py::arg("params"), py::arg("prior"),
          py::arg("nclass"),
          "E-step: compute posterior class membership probabilities and "
          "log-likelihood. Returns (posterior, loglik).");

    m.def("m_step_probs", &pypolca::m_step_probs, py::arg("data"), py::arg("posterior"),
          py::arg("num_choices"), py::arg("nclass"),
          "M-step: update class-conditional response probabilities.");

    m.def("compute_beta_derivatives", &pypolca::compute_beta_derivatives, py::arg("data"),
          py::arg("posterior"), py::arg("prior"), py::arg("beta"), py::arg("nclass"),
          "Compute gradient and observed information for Newton-Raphson.");

    m.def("update_beta", &pypolca::update_beta, py::arg("data"), py::arg("posterior"),
          py::arg("prior"), py::arg("beta"), py::arg("nclass"),
          "Single Newton-Raphson step for beta. Returns (new_beta, new_prior).");

    m.def("compute_prior_from_beta", &pypolca::compute_prior_from_beta, py::arg("x"),
          py::arg("beta"), py::arg("nclass"), "Build prior matrix from beta via softmax.");
}
