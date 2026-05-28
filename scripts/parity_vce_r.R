# Full VCE comparison between R and pypolca
library(poLCA)
library(MASS)

data(cheating)

y <- cheating[,1:4]
y <- as.matrix(y)
N <- nrow(y)
J <- ncol(y)
K.j <- rep(2, 4)
R <- 2
x <- as.matrix(rep(1, N))  # intercept only, Nx1 matrix
S <- ncol(x)  # = 1

probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)

f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
res <- poLCA(f, cheating, nclass=2, probs.start=probs_start,
             verbose=FALSE, calc.se=TRUE, nrep=1)

cat("R results:\n")
cat("Loglik:", res$llik, "\n")
cat("Posterior sum by class:", colSums(res$posterior), "\n")

# Reconstruct prior from posterior (as pypolca does for S=1)
pbar <- colMeans(res$posterior)
prior <- matrix(pbar, nrow=N, ncol=R, byrow=TRUE)
cat("Pbar:", pbar, "\n")

# Build score matrix s exactly as poLCA.se does
ymat <- y
# 1-hot encode
y_hot <- list()
for (j in 1:J) {
  y_hot[[j]] <- matrix(0, nrow=N, ncol=K.j[j])
  for (i in 1:N) {
    if (ymat[i,j] > 0) y_hot[[j]][i, ymat[i,j]] <- 1
    else y_hot[[j]][i,] <- NA
  }
}

s <- NULL
for (r in 1:R) {
  for (j in 1:J) {
    s <- cbind(s, res$posterior[,r] *
      t(t(y_hot[[j]][,2:K.j[j]]) - res$probs[[j]][r, 2:K.j[j]]))
  }
}
ppdiff <- res$posterior - prior
if (R>1) for (r in 2:R) { s <- cbind(s, x * ppdiff[,r]) }
s[is.na(s)] <- 0

info <- t(s) %*% s
VCE_R <- ginv(info)

cat("\n=== R VCE (9x9) ===\n")
print(round(VCE_R, 6))

cat("\n=== R VCE.beta (bottom-right) ===\n")
print(round(VCE_R[9,9], 6))

cat("\n=== R vecprobs_se (raw sqrt diag from Jac * VCE_lo * t(Jac)) ===\n")
cat("From poLCA: probs.se[[1]]:\n")
print(res$probs.se[[1]])
cat("\nprobs.se[[2]]:\n")
print(res$probs.se[[2]])
cat("\nprobs.se[[3]]:\n")
print(res$probs.se[[3]])
cat("\nprobs.se[[4]]:\n")
print(res$probs.se[[4]])

cat("\n=== R P.se ===\n")
print(res$P.se)

# Also show Dp, Dbeta
Dp <- sum(R*(K.j-1))
Dbeta <- S * (R-1)
cat("\nDp:", Dp, " Dbeta:", Dbeta, " D:", Dp+Dbeta, "\n")
cat("S:", S, " R:", R, "\n")
cat("ncol(s):", ncol(s), "\n")