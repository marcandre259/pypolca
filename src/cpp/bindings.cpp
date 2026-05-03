#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "polca/types.h"
#include "polca/em_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for poLCA (Polytomous Variable Latent Class Analysis)";

    // --- Data struct ---
    py::class_<polca::Data>(m, "Data")
        .def(py::init<>())
        .def_readwrite("y", &polca::Data::y, "N x J response matrix (1-based, 0=missing)")
        .def_readwrite("x", &polca::Data::x, "N x S covariate matrix")
        .def_readwrite("num_choices", &polca::Data::num_choices, "Categories per item")
        .def("n_obs", &polca::Data::n_obs)
        .def("n_items", &polca::Data::n_items)
        .def("n_covariates", &polca::Data::n_covariates);

    // --- Params struct ---
    py::class_<polca::Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("vecprobs", &polca::Params::vecprobs)
        .def_readwrite("beta", &polca::Params::beta);

    // --- Results struct ---
    py::class_<polca::Results>(m, "Results")
        .def_readonly("params", &polca::Results::params)
        .def_readonly("posterior", &polca::Results::posterior, "N x nclass posterior probs")
        .def_readonly("prior", &polca::Results::prior, "N x nclass prior probs")
        .def_readonly("loglik", &polca::Results::loglik)
        .def_readonly("iterations", &polca::Results::iterations)
        .def_readonly("converged", &polca::Results::converged);

    // --- fit_em ---
    m.def("fit_em",
          &polca::fit_em,
          py::arg("data"),
          py::arg("nclass"),
          py::arg("maxiter") = 1000,
          py::arg("tol") = 1e-10,
          py::arg("verbose") = false,
          py::arg("probs_start") = Eigen::VectorXd(),
          py::arg("beta_start") = Eigen::VectorXd(),
          R"pbdoc(
          Fit a latent class model via EM.

          Parameters
          ----------
          data : polca.Data
              Input data container.
          nclass : int
              Number of latent classes.
          maxiter : int
              Maximum EM iterations.
          tol : float
              Convergence tolerance.
          verbose : bool
              Print iteration progress.
          probs_start : np.ndarray
              Optional starting values for response probabilities.
          beta_start : np.ndarray
              Optional starting values for covariate coefficients.

          Returns
          -------
          polca.Results
              Fitted model results.
          )pbdoc");

    // --- Expose math helpers for unit testing ---
    m.def("compute_ylik",
          &polca::compute_ylik,
          py::arg("data"), py::arg("params"), py::arg("nclass"),
          "Compute class-conditional likelihoods (N x nclass).");

    m.def("compute_prior_from_beta",
          &polca::compute_prior_from_beta,
          py::arg("x"), py::arg("beta"), py::arg("nclass"),
          "Build prior matrix from beta via softmax.");
}
