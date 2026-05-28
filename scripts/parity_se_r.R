# Parity test: Run poLCA on cheating data, output JSON for pypolca comparison
library(poLCA)
library(jsonlite)

data(cheating)

# Formula: all 4 binary manifest variables, intercept-only (no covariates)
f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1

# Deterministic starting values to ensure same initialization
probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)

set.seed(42)
res <- poLCA(f, cheating, nclass=2, maxiter=1000, tol=1e-10,
             probs.start=probs_start, verbose=FALSE, calc.se=TRUE,
             nrep=1, graphs=FALSE)

cat("=== LOGLIK ===\n")
cat(res$llik, "\n")

cat("=== NPAR ===\n")
cat(res$npar, "\n")

cat("=== P (class shares) ===\n")
cat(res$P, "\n")

cat("=== PROBS ===\n")
for (j in 1:4) {
  cat(sprintf("Item %d:\n", j))
  print(res$probs[[j]])
}

cat("=== PROBS SE ===\n")
for (j in 1:4) {
  cat(sprintf("Item %d SE:\n", j))
  print(res$probs.se[[j]])
}

cat("=== P SE ===\n")
cat(res$P.se, "\n")

cat("=== POSTERIOR (first 5 rows) ===\n")
print(res$posterior[1:5,])

cat("=== PRIOR (first 5 rows) ===\n")
print(res$prior[1:5,])

cat("=== PREDCLASS (first 20) ===\n")
cat(res$predclass[1:20], "\n")
