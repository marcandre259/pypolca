# Parity with covariates: cheating data, ~ GPA
library(poLCA)
library(MASS)

data(cheating)

y <- as.matrix(cheating[,1:4])
N <- nrow(y)
J <- 4
K.j <- rep(2, 4)
R <- 2

# Same starting values for probs
probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)

f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
res <- poLCA(f, cheating, nclass=2, probs.start=probs_start,
             verbose=FALSE, calc.se=TRUE, nrep=1)

cat("=== R with covariates (GPA) ===\n")
cat("Loglik:", res$llik, "\n")
cat("Npar:", res$npar, "\n")
cat("P:", res$P, "\n")
cat("P.se:", res$P.se, "\n")

cat("\n=== coeff ===\n")
print(res$coeff)
cat("\n=== coeff.se ===\n")
print(res$coeff.se)
cat("\n=== coeff.V ===\n")
print(res$coeff.V)

cat("\n=== probs ===\n")
print(res$probs)
cat("\n=== probs.se ===\n")
print(res$probs.se)

cat("\n=== posterior[1:5,] ===\n")
print(res$posterior[1:5,])
cat("\n=== prior[1:5,] ===\n")
print(res$prior[1:5,])

# Get dimensions
x <- model.matrix(f, cheating)
cat("\nS (ncol(x)):", ncol(x), "\n")
cat("dim(x):", dim(x)[1], "x", dim(x)[2], "\n")

# Verify beta values
cat("\n=== full beta vector (flattened) ===\n")
# Manually reconstruct from poLCA internals
# R stores b as S x (R-1) matrix: here S=2, R-1=1
cat("\n")