# Parity: show beta_SE for S=1 case
library(poLCA)
library(MASS)

data(cheating)
y <- cheating[,1:4]
y <- as.matrix(y)
N <- nrow(y)
J <- ncol(y)
K.j <- rep(2, 4)
R <- 2
x <- matrix(1, nrow=N, ncol=1)  # intercept only

probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)

f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
res <- poLCA(f, cheating, nclass=2, probs.start=probs_start, 
             verbose=FALSE, calc.se=TRUE, nrep=1)

cat("=== R poLCA full results ===\n")
cat("Loglik:", res$llik, "\n")
cat("Npar:", res$npar, "\n")
cat("P:", res$P, "\n")
cat("P.se:", res$P.se, "\n")
cat("\nprobs:\n")
print(res$probs)
cat("\nprobs.se:\n")
print(res$probs.se)
cat("\nposterior[1:5,]:\n")
print(res$posterior[1:5,])
cat("\ncoeff:", res$coeff, "\n")
cat("coeff.se:", res$coeff.se, "\n")

# Now manually compute beta_SE for S=1 to verify
rgivy <- res$posterior
prior <- matrix(colMeans(rgivy), nrow=N, ncol=R, byrow=TRUE)
vp <- res$probs

# Replicate poLCA.se logic
s <- NULL
for (r in 1:R) {
  for (j in 1:J) {
    s <- cbind(s, rgivy[,r] * t(t(matrix(y[,j]==1:2, nrow=N, ncol=2)[,2:2,drop=FALSE]) - vp[[j]][r,2:2]))
  }
}
ppdiff <- rgivy - prior
if (R>1) for (r in 2:R) { s <- cbind(s, x * ppdiff[,r]) }
s[is.na(s)] <- 0
info <- t(s) %*% s
VCE <- ginv(info)

cat("\n=== Manual VCE dimensions ===\n")
cat("Dp:", sum(R*(K.j-1)), "\n")
cat("Dbeta:", ncol(x)*(R-1), "\n")
cat("Total:", dim(VCE)[1], "\n")

VCE.beta <- VCE[(1+sum(R*(K.j-1))):dim(VCE)[1], (1+sum(R*(K.j-1))):dim(VCE)[2], drop=FALSE]
cat("\nVCE.beta (should be 1x1):\n")
print(VCE.beta)
cat("se.beta (sqrt diag):", sqrt(diag(VCE.beta)), "\n")