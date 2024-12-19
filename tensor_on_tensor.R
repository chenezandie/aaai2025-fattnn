rm(list=ls())

ctprod <- function(A,B,K){
  da <- dim(A)
  la <- length(da)
  db <- dim(B)
  lb <- length(db)
  Amat <- array(A, dim=c(prod(da[1:(la-K)]),prod(da[(la-K+1):la])))
  Bmat <- array(B, dim=c(prod(db[1:K]),prod(db[(K+1):lb])))
  Cmat <- Amat %*% Bmat
  C <- array(Cmat,dim=c(da[1:(la-K)],db[(K+1):lb]))
  return(C)
} 

rrr <- function(X,Y,R=1,lambda=0,annealIter=0,convThresh=10^(-5), seed=0){
  if(seed>0) set.seed(seed)
  lambdaFin = lambda
  L = length(dim(X))-1
  N=dim(X)[1]
  P = dim(X)[2:(L+1)]
  M = length(dim(Y))-1
  Q = dim(Y)[2:(M+1)]
  Xmat = array(X,dim=c(N,prod(P)))
  Ymat = array(Y,dim=c(N,prod(Q)))
  Yvec = as.vector(Y)
  
  U = list()
  for(l in 1:L) U[[l]] = matrix(rnorm(P[l]*R),ncol=R)
  V = list()
  for(l in 1:M) V[[l]] = matrix(rnorm(Q[l]*R),ncol=R)
  Vmat = matrix(nrow=prod(Q),ncol=R)
  for(r in 1:R){
    Vr = lapply(V, function(x) x[,r])
    Vmat[,r] = as.vector(array(apply(expand.grid(Vr), 1, prod), dim=Q))
  }
#  B = array(apply(expand.grid(U), 1, prod), dim=P)
  Bmat = matrix(nrow=R,ncol=prod(P))
  for(r in 1:R){
    Ur = lapply(U, function(x) x[,r])
    Br = array(apply(expand.grid(Ur), 1, prod), dim=P)  
    Bmat[r,] = array(Br,dim=c(1,prod(P)))
  }
  Y_pred = as.vector(Xmat%*%t(Bmat)%*%t(Vmat))
  Xarrays = list()
  for(l in 1:L){
    Xarrays[[l]] <- array(dim=c(N,P[l],prod(P[-l])))
    perm = c(l,c(1:L)[-l])
    for(i in 1:N){
      X_slice = array(Xmat[i,],dim=c(P))
      X_slice_perm = aperm(X_slice,perm)
      Xarrays[[l]][i,,] = array(X_slice_perm,dim=c(P[l],prod(P[-l])))
    }
  }  
  Ymats = list()
  for(m in 1:M){
    perm = c(m+1,c(1:(M+1))[-(m+1)])
    Y_perm = aperm(Y,perm) 
    Ymats[[m]] = array(Y_perm,dim=c(Q[m],N*prod(Q[-m])))
  }
  sse=c()
  sseSig = c()
  sseR = c()
  j=0
  conv=convThresh+1
  while(conv>convThresh){
    j=j+1
    if(j<=annealIter){
      lambda = 100*(annealIter-j)/annealIter + lambdaFin
    } else {
      lambda = lambdaFin # Ensure lambdaFin is not too small
    }
    for(l in 1:L){
      ###Matrix B
      B_red = matrix(nrow=prod(P[-l]),ncol=R)
      for(r in 1:R){
        Ured_r <- lapply(U[-l], function(x) x[,r]) 
        B_red[,r] = as.vector(array(apply(expand.grid(Ured_r), 1, prod), dim=P[-l])) 
      }
      X_red = matrix(nrow=N,ncol=P[l]*r)
      perm = c(l,c(1:L)[-l])
      for(i in 1:N){X_red[i,] = as.vector(Xarrays[[l]][i,,] %*% B_red)} 
      C = matrix(nrow=dim(X_red)[1]*prod(Q),ncol=dim(X_red)[2])
      for(r in 1:R){
        index = ((r-1)*P[l]+1):(r*P[l])
        C[,index] = kronecker(Vmat[,r],X_red[,index])
      }
      lambdaMat = kronecker(t(Vmat)%*%Vmat * t(B_red)%*%B_red, max(lambda, 1e-5)*diag(P[l]))
      CP <- crossprod(X_red)*kronecker(t(Vmat)%*%Vmat, matrix(rep(1,P[l]^2),nrow=P[l]))
      regMat = solve(CP+lambdaMat) 
      Ulvec <- regMat%*%(t(C)%*%Yvec)
      U[[l]] = matrix(Ulvec,nrow=P[l])
      Bmat = matrix(nrow=R,ncol=prod(P))
      for(r in 1:R){
        Ur = lapply(U, function(x) x[,r])
        Br = array(apply(expand.grid(Ur), 1, prod), dim=P)  
        Bmat[r,] = array(Br,dim=c(1,prod(P)))
      }
      Y_pred = as.vector(Xmat%*%t(Bmat)%*%t(Vmat))
      sse[(M+L)*(j-1)+l] = sum((Yvec-Y_pred)^2)
      sseR[(M+L)*(j-1)+l] = sum((Yvec-Y_pred)^2)+lambda*sum((t(Bmat)%*%t(Vmat))^2)
    }
    for(m in 1:M){
      BRmat = matrix(nrow = prod(c(P,Q[-m])),ncol=R)
      D = matrix(nrow=N*prod(Q[-m]),ncol=R)
      for(r in 1:R){
        Vecs <- lapply(c(U,V[-m]), function(x) x[,r])
        Br = array(apply(expand.grid(Vecs), 1, prod), dim=c(prod(P),prod(Q[-m])))
        BRmat[,r] = as.vector(Br)
        D[,r] = as.vector(Xmat %*% Br)
      }
      V[[m]] = Ymats[[m]]%*%D%*%solve(t(D)%*%D+lambda*t(BRmat)%*%BRmat) ##now for X, use Xorig
      Vmat = matrix(nrow=prod(Q),ncol=R)
      for(r in 1:R){
        Vr = lapply(V, function(x) x[,r])
        Vmat[,r] = as.vector(array(apply(expand.grid(Vr), 1, prod), dim=Q))
      }
      Y_pred = as.vector(Xmat%*%t(Bmat)%*%t(Vmat))
      sse[(M+L)*(j-1)+L+m] = sum((Yvec-Y_pred)^2)
      sseR[(M+L)*(j-1)+L+m] = sum((Yvec-Y_pred)^2)+lambda*sum((t(Bmat)%*%t(Vmat))^2)
    }
  if(j>1) conv = abs(sseR[length(sseR)]-sseR[length(sseR)-L-M])/sseR[length(sseR)] 
  }
  B = array(t(Bmat)%*%t(Vmat),dim = c(P,Q))
  return(list(U=U,V=V,B=B,sse=sse,sseR=sseR))
}

##################################################################################
##################################################################################
#Load data
X_train <- read.csv('X_train_data.csv')
Y_train <- read.csv('Y_train_data.csv')
X_test <- read.csv('X_test_data.csv')
Y_test <- read.csv('Y_test_data.csv')

X_train_matrix <- as.matrix(X_train)
Y_train_matrix <- as.matrix(Y_train)
X_test_matrix <- as.matrix(X_test)
Y_test_matrix <- as.matrix(Y_test)

X_train_tensor <- array(X_train_matrix, dim = c(352,2, 19, 19, 8))
Y_train_tensor <- array(Y_train_matrix, dim = c(352, 19, 19, 8))
X_test_tensor <- array(X_test_matrix, dim = c(152,2, 19, 19, 8))
Y_test_tensor <- array(Y_test_matrix, dim = c(152, 19, 19, 8))
##################################################################################
##################################################################################
set.seed(12)
start.time <- Sys.time()
Results <- rrr(X_train_tensor,Y_train_tensor,R=3) 
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
Y_pred <- ctprod(X_test_tensor,Results$B,4)  ##Array of fitted values
mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}
# Compute MSE
mse_value <- mse(Y_test_tensor, Y_pred) 
print(paste("MSE:", mse_value))
##################################################################################
##################################################################################
#for simulation, report average mse
perform_rrr <- function(seed) {
  set.seed(seed)
  
  # Perform Reduced Rank Regression (RRR)
  Results <- rrr(X_train_tensor, Y_train_tensor, R=2)
  
  # Predict using the results of RRR
  Y_pred <- ctprod(X_test_tensor, Results$B, 3)
  
  # Compute MSE
  mse_value <- mean((Y_test_tensor - Y_pred)^2)
  
  return(list(mse = mse_value))
}

# Initialize vectors to store the MSE and RMSE values
mse_values <- numeric()

# Repeat the process for seeds from 1 to 2
for(seed in 1:20) {
  results <- perform_rrr(seed)
  mse_values <- c(mse_values, results$mse)
}

# Compute average MSE and RMSE
average_mse <- mean(mse_values)

# Print the results
print(paste("Average MSE:", average_mse))
