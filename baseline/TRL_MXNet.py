#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import MXNet with it's Gluon Interface
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import Block
from mxnet import gluon

import numpy as np

import tensorly as tl
from tensorly.random import check_random_state

# Make the calculations reproducible
mx.random.seed(1)


# In[2]:


def inner(tensor1, tensor2, n_modes=None):
    """Generalised inner products between tensors

        Takes the inner product between the last (respectively first)
        `n_modes` of `tensor1` (respectively `tensor2`)

    Parameters
    ----------
    tensor1, tensor2 : tensor
    n_modes : int, default is None
        * if None, the traditional inner product is returned (i.e. a float)
        * otherwise, the product between the `n_modes` last modes of `tensor1`
            and the `n_modes` first modes of `tensor2` is returned.
            The resulting tensor's order is `ndim(tensor1) - n_modes`.

    Returns
    -------
    inner_product : float if n_modes is None, tensor otherwise
    """
    # Traditional inner product
    if n_modes is None:
        if tensor1.shape != tensor2.shape:
            raise ValueError(
                "Taking a generalised product between two tensors without specifying common modes"
                " is equivalent to taking inner product."
                "This requires tensor1.shape == tensor2.shape."
                f"However, got shapes {tensor1.shape} and {tensor2.shape}"
            )
        return T.sum(tensor1 * tensor2)

    # Inner product along `n_modes` common modes
    shape_t1 = list(T.shape(tensor1))
    shape_t2 = list(T.shape(tensor2))
    common_modes = shape_t1[len(shape_t1) - n_modes :]
    common_size = int(np.prod(common_modes))
    output_shape = shape_t1[:-n_modes] + shape_t2[n_modes:]

    if common_modes != shape_t2[:n_modes]:
        raise ValueError(
            f"Incorrect shapes for inner product along {n_modes} common modes."
            f"Shapes {tensor1.shape} and {tensor2.shape}"
        )
    inner_product = T.dot(
        T.reshape(tensor1, (-1, common_size)), T.reshape(tensor2, (common_size, -1))
    )
    return T.reshape(inner_product, output_shape)


# ## Set the backend and context

# First, let's instruct TensorLy to use the MXNet backend to perform all computations.

# In[3]:


tl.set_backend('mxnet')
ctx = mx.cpu()


# In[4]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
from tensorly.decomposition import tucker
from math import prod
from numpy.linalg import qr
from scipy.linalg import fractional_matrix_power
from numpy.linalg import qr
from tensorly.decomposition import parafac, tucker
from tensorly.tenalg import multi_mode_dot

def generate_random_orthogonal_matrix(size):
    H = np.random.randn(size, size)
    Q, R = qr(H)
    return Q

def kronecker_list(matrices):
    result = np.array([[1.0]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

def generate_A(dim, R, P, rho):
    A = []
    for p in range(P):
        A_p = []
        for r in range(R):
            A_p_r = []
            for k in range(len(dim)):
                size = dim[k]
                matrix = generate_random_orthogonal_matrix(size)
                A_p_r.append(matrix)
            A_p.append(A_p_r)
        A.append(A_p)
    return A

def tenAR_sim(t, dim, R, P, rho, cov='iid', A=None, Sig=None):
    if A is None:
        A = generate_A(dim, R, P, rho)
    e = np.random.normal(loc=0, scale=0.1, size=(t + 500, np.prod(dim)))
    K = len(A[0][0])
    phi = []
    for p in range(P):
        phi_p = 0
        for r in range(R):
            kronecker_product = kronecker_list(list(reversed(A[p][r])))
            phi_p += kronecker_product
        phi.append(phi_p)
    
    x = np.zeros((t + 500, np.prod(dim)))
    for p in range(P):
        x[p, :] = np.random.normal(size=np.prod(dim))
    
    for i in range(P, 500 + t):  # Burning number = 500
        temp = np.zeros(np.prod(dim))
        for p in range(P):
            temp += phi[p].dot(x[i - p - 1, :])
        x[i, :] = temp + e[i,:]
        #x[i, :] = temp
    
    return x[500:(500 + t), :].reshape((t, *dim))



def tenFM_sim(Ft, dims, lambda_val=1, A=None):
    r = Ft.shape[1:]  # Dimensions of the factor process
    t = Ft.shape[0]  # Length of output series
    dd = len(dims)
    
    # Ensure the input dimensions match
    if len(r) != dd and A is None:
        raise ValueError("Incorrect length K of input dims or A, for order K tensor time series.")
    
    # Initialize X with the transposed Ft to match R's aperm function
    X = np.transpose(Ft, axes=(1, 2, 3, 0))
    
    if A is None:
        A = []
        for i in range(dd):
            # Generate transformation matrix Ai for each mode of Ft
            Ai = np.random.normal(size=(dims[i], r[i]))
            Q, _ = np.linalg.qr(Ai)  # Ensure orthogonality
            # Apply transformation
            X = np.tensordot(Q, X, axes=([1], [0]))
            # Adjust the axes to match R's tensor function behavior
            X = np.moveaxis(X, 0, -1)
            A.append(Q)
    else:
        for i, Ai in enumerate(A):
            Q, _ = np.linalg.qr(Ai)
            X = np.tensordot(Q, X, axes=([1], [0]))
            X = np.moveaxis(X, 0, -1)
    E = np.random.normal(loc=0, scale=0.1,size=(t, *dims))
    # Apply lambda_val to adjust signal strength
    X = lambda_val * X + E
    #X = lambda_val * X
    
    # Reshape X to move the time dimension back to the first axis
    X = np.transpose(X, axes=(1,2,3,0))
    
    return X
# x = tenFM_sim(Ft=Ft, dims=dims, lambda_val=lambda_val, A = None)
# x.shape

def tensor_product_and_contract(A, B, alongA, alongB, n, h):
    """
    Perform a tensor product of A and B similar to the R tensor package, contracting over specified axes.
    
    Parameters:
    - A, B: Input arrays.
    - alongA, alongB: Axes in A and B to sum over after the product.
    - n: Size of the last dimension in the original tensor.
    - h: The current lag in the computation.
    
    Returns:
    - Resulting tensor after product and contraction.
    """
    # Perform the tensor product
    Omega = np.tensordot(A, B, axes=(alongA, alongB)) / (n - h)
    return Omega

def tipup_init_tensor_1(x, r, h0=1, oneside_true=False, norm_true=False):
    dd = x.shape
    #print(dd)
    #print(dims)
    d = len(dd)
    n = dd[-1]
    
    # Initialize placeholders for the results
    ans_M = []
    ans_Q = []
    ans_lambda = []
    
    k = 1 if oneside_true else d - 1
    
    
    for i in range(k):
        lst = list(range(k + 1))  # Generate list from 0 to k
        permutations = [[lst[j] for j in range(len(lst)) if j != i] for i in range(len(lst))]
        permutations = permutations[:-1]
        M_temp = np.zeros((dd[i], dd[i]))
        for h in range(1):  # To match your R loop, but note: range(1) will only iterate once with h=0
            x_left = x.reshape(dd[0], dd[1], dd[2], n)[:,:,:,:n-h].transpose(0, 1, 2, 3)  # Adjusted slicing and reshaping
            x_right = x.reshape(dd[0], dd[1], dd[2], n)[:,:,:,h:n].transpose(0, 1, 2, 3)  # Adjusted slicing and reshaping
            #print("x_left shape:", x_left.shape)
            #print("x_right shape:", x_right.shape)
            Omega = tensordot(x_left, x_right, permutations[i])
            #print("Omega:", Omega.shape)
            M_temp = np.dot(Omega, Omega.T)
            eig_vals, eig_vecs = np.linalg.eig(M_temp)
            ans_M.append(M_temp)
            ans_Q.append(eig_vecs[:, :r[i]])
            ans_lambda.append(eig_vals)
            #print(ans_Q)
    
    # Placeholder for norm calculation and x_hat computation
    norm_percent = None
    x_hat = None
    # print('x is:',x[1])
    if norm_true:
        modes = [0, 1, 2]  # Modes to contract on
        transposed_Q = [q.T for q in ans_Q]
        x_hat = multi_mode_dot(x, transposed_Q, modes)
        x_hat = multi_mode_dot(x_hat, ans_Q, modes=[0, 1, 2])
        norm_percent = tl.norm(x - x_hat, 2) / tl.norm(x, 2)
        pass
    
    return {"M": ans_M, "Q": ans_Q, "lambda": ans_lambda, "norm.percent": norm_percent, "x.hat": x_hat}


def _validate_contraction_modes(shape1, shape2, modes, batched_modes=False):
    """Takes in the contraction modes (for a tensordot) and validates them

    Parameters
    ----------
    modes : int or tuple[int] or (modes1, modes2)
    batched_modes : bool, default is False

    Returns
    -------
    modes1, modes2 : a list of modes for each contraction
    """
    if isinstance(modes, int):
        if batched_modes:
            modes1 = [modes]
            modes2 = [modes]
        else:
            modes1 = list(range(-modes, 0))
            modes2 = list(range(0, modes))
    else:
        try:
            modes1, modes2 = modes
        except ValueError:
            modes1 = modes
            modes2 = modes
    try:
        modes1 = list(modes1)
    except TypeError:
        modes1 = [modes1]
    try:
        modes2 = list(modes2)
    except TypeError:
        modes2 = [modes2]

    if len(modes1) != len(modes2):
        if batched_modes:
            message = f"Both tensors must have the same number of batched modes"
        else:
            message = (
                "Both tensors must have the same number of modes to contract along. "
            )
        raise ValueError(
            message + f"However, got modes={modes}, "
            f" i.e. {len(modes1)} modes for tensor 1 and {len(modes2)} mode for tensor 2"
            f"(modes1={modes1}, and modes2={modes2})"
        )
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    for i in range(len(modes1)):
        if shape1[modes1[i]] != shape2[modes2[i]]:
            if batched_modes:
                message = "Batch-dimensions must have the same dimensions in both tensors but got"
            else:
                message = "Contraction dimensions must have the same dimensions in both tensors but got"
            raise ValueError(
                message + f" mode {modes1[i]} of size {shape1[modes1[i]]} and "
                f" mode {modes2[i]} of size {shape2[modes2[i]]}."
            )
        if modes1[i] < 0:
            modes1[i] += ndim1
        if modes2[i] < 0:
            modes2[i] += ndim2

    return modes1, modes2

def tensordot(tensor1, tensor2, modes, batched_modes=()):
    """Batched tensor contraction between two tensors on specified modes

    Parameters
    ----------
    tensor1 : tl.tensor
    tensor2 : tl.tensor
    modes : int list or int
        modes on which to contract tensor1 and tensor2
    batched_modes : int or tuple[int]

    Returns
    -------
    contraction : tensor1 contracted with tensor2 on the specified modes
    """
    modes1, modes2 = _validate_contraction_modes(tensor1.shape, tensor2.shape, modes)
    batch_modes1, batch_modes2 = _validate_contraction_modes(
        tensor1.shape, tensor2.shape, batched_modes, batched_modes=True
    )

    contraction_shape = [s for (i, s) in enumerate(tl.shape(tensor1)) if i in modes1]
    contraction_dim = prod(contraction_shape)
    batch_shape = [s for (i, s) in enumerate(tl.shape(tensor1)) if i in batch_modes1]

    # Prepare to reorganize the modes afterwards by moving bactch size back to their place
    # (while ommiting modes contracted over)
    final_modes = []
    n_batches = len(batch_modes1)
    batch_counter = 0
    free_counter = 0
    for i in range(tl.ndim(tensor1)):
        if i in modes1:
            continue
        elif i in batch_modes1:
            final_modes.append(batch_counter)
            batch_counter += 1
        else:
            final_modes.append(free_counter + n_batches)
            free_counter += 1

    # We will reorganize tensor1 to (batch_modes, new_modes1, contraction_modes)
    new_modes1 = [i for i in range(tl.ndim(tensor1)) if i not in batch_modes1 + modes1]
    new_shape1 = [tl.shape(tensor1)[i] for i in new_modes1]
    tensor1 = tl.transpose(tensor1, batch_modes1 + new_modes1 + modes1)
    tensor1 = tl.reshape(tensor1, (*batch_shape, -1, contraction_dim))

    # Tensor2 will be (batch_modes, contraction_modes, new_modes2)
    new_modes2 = [i for i in range(tl.ndim(tensor2)) if i not in batch_modes2 + modes2]
    new_shape2 = [tl.shape(tensor2)[i] for i in new_modes2]
    tensor2 = tl.transpose(tensor2, batch_modes2 + modes2 + new_modes2)
    tensor2 = tl.reshape(tensor2, (*batch_shape, contraction_dim, -1))

    res = tl.matmul(tensor1, tensor2)
    res = tl.reshape(res, (*batch_shape, *new_shape1, *new_shape2))

    final_modes += [i for i in range(res.ndim) if i not in final_modes]

    if final_modes:
        res = tl.transpose(res, final_modes)

    return res

def tipup_init_tensor_2(x, r, h0=1, oneside_true=False, norm_true=False):
    dd = x.shape
    #print(dd)
    #print(dims)
    d = len(dd)
    n = dd[-1]
    
    # Initialize placeholders for the results
    ans_M = []
    ans_Q = []
    ans_lambda = []
    
    k = 1
    lst = list(range(k + 1))  # Generate list from 0 to k
    for i in range(k):
        M_temp = np.zeros((dd[i], dd[i]))
    h = 0
    x_left = x.reshape(dd[0], dd[1], dd[2], n)[:,:,:,:n-h].transpose(0, 1, 2, 3)  # Adjusted slicing and reshaping
    x_right = x.reshape(dd[0], dd[1], dd[2], n)[:,:,:,h:n].transpose(0, 1, 2, 3)  # Adjusted slicing and reshaping
    #print("x_left shape:", x_left.shape)
    #print("x_right shape:", x_right.shape)
    axes_to_contract_left = list(range(d))
    del axes_to_contract_left[i - 1]  # Adjust i to Python indexing
    Omega = tensordot(x_left, x_right, [1,2,3])/n
    #print("Omega:", Omega.shape)
    M_temp = np.dot(Omega, Omega.T)
    eig_vals, eig_vecs = np.linalg.eig(M_temp)
    ans_M.append(M_temp)
    ans_Q.append(eig_vecs[:, :r[i]])
    ans_lambda.append(eig_vals)
    #print(ans_Q)
    
    # Placeholder for norm calculation and x_hat computation
    norm_percent = None
    x_hat = None
    # print('x is:',x[1])
    if norm_true:
        modes = [0, 1, 2]  # Modes to contract on
        transposed_Q = [q.T for q in ans_Q]
        x_hat = multi_mode_dot(x, transposed_Q, modes)
        x_hat = multi_mode_dot(x_hat, ans_Q, modes=[0, 1, 2])
        norm_percent = tl.norm(x - x_hat, 2) / tl.norm(x, 2)
        pass
    
    return {"M": ans_M, "Q": ans_Q, "lambda": ans_lambda, "norm.percent": norm_percent, "x.hat": x_hat}

def reshape_x_new(x_new):
    """
    Reshape x_new from (3, m, 3, 100) or (3, 3, m, 100) to (m, 3, 3, 100) if necessary.
    
    Parameters:
    - x_new: numpy array to be potentially reshaped.
    
    Returns:
    - x_new reshaped if its initial shape was (3, m, 3, 100) or (3, 3, m, 100),
      otherwise x_new is returned unchanged.
    """
    # Known valid m values
    valid_m_values = dims
    
    # Initial check to avoid reshaping if already in desired shape
    if x_new.shape[0] in valid_m_values and x_new.shape[1] == r[1] and x_new.shape[2] == r[2]:
        #print("x_new is already in the expected shape.")
        return x_new

    # Check for (3, m, 3, 100) shape
    if x_new.shape[0] == r[0] and x_new.shape[2] == r[2] and x_new.shape[1] in valid_m_values and x_new.shape[3] == max(x_new.shape[0],x_new.shape[1],x_new.shape[2],x_new.shape[3]):
        x_new = x_new.transpose(1, 0, 2, 3)
        #print(f"x_new reshaped to {x_new.shape}.")
        return x_new
    
    # Check for (3, 3, m, 100) shape
    elif x_new.shape[0] == r[0] and x_new.shape[1] == r[1] and x_new.shape[2] in valid_m_values and x_new.shape[3] == max(x_new.shape[0],x_new.shape[1],x_new.shape[2],x_new.shape[3]):
        x_new = x_new.transpose(2, 0, 1, 3)
        #print(f"x_new reshaped to {x_new.shape}.")
        return x_new
    
    #print("x_new does not match any expected pattern for reshaping.")
    return x_new

def tenFM_est(x, r, h0=1, method='TIPUP', iter=True, tol=1e-4, maxiter=100):
    # Assuming x is a numpy array and already in the correct shape
    # If not, you might need to permute x similar to R's aperm function
    # x = x.transpose((1, 2, 0))  # Example if you need to permute
    
    dd = x.shape
    d = len(dd)
    n = dd[-1]
    x_tnsr = x  # In Python, you might not need to explicitly convert to tensor
    d_seq = list(range(1, d))
    ans_init = tipup_init_tensor_1(x, r, h0=1, oneside_true=False, norm_true=True)
    
    ddd = dd[:-1]  # This will give you (16, 18, 20)
    iiter = 1  # Initialize iteration counter
    dis = 1  # Initialize a variable (possibly for distance or discrepancy)
    fnorm_resid = np.zeros(maxiter + 2)
    
    ans_Q = ans_init['Q']
    transposed_Q = [q.T for q in ans_Q]
    x_hat = multi_mode_dot(x, transposed_Q, modes)
    x_hat = multi_mode_dot(x_hat, ans_Q, modes=[0, 1, 2])

    
    #x_hat = ttl(x_tnsr, [np.transpose(q) for q in ans_init['Q']], d_seq)
    #print(x_hat)
    #x_hat = ttl(x_hat, ans_init['Q'], d_seq)
    #print(x_hat)
    
    tnsr_norm = tl.norm(x_tnsr,2)
    fnorm_resid[0] = np.linalg.norm(x_tnsr - x_hat) / tnsr_norm
    # Update ans.Q from ans_init
    ans_Q = ans_init['Q']
    transposed_Q = [q.T for q in ans_Q]
    Ft = multi_mode_dot(x_tnsr, transposed_Q, [0, 1, 2])
    # Sum over the first d-1 dimensions of Ft
    Ft_all = np.sum(Ft, axis=tuple(range(d-1)))
    # Update fnorm_resid after some operations on x_hat (not shown here)
    fnorm_resid[iiter + 1] = tl.norm(x_tnsr - x_hat,2) / tnsr_norm
    
    iiter = 1

    if iter:
        while dis > tol and iiter < maxiter:
            for i in range(d-1): 
                x_new = multi_mode_dot(x_tnsr, [np.transpose(q) for idx, q in enumerate(ans_Q) if idx != i], modes=[idx for idx in range(d-1) if idx != i])
                #print(x_new.shape)
                dims_new = x_new.shape[:-1]  # Get the new dimensions, excluding the last dimension if it represents time or another dimension not affected by the contractions
                #print(dims_new)
                x_new = reshape_x_new(x_new)
                ans_iter = tipup_init_tensor_2(x_new, [r[i]] + [r[j] for j in range(len(r)) if j != i], h0=0, oneside_true=True, norm_true=False)
                ans_Q[i] = ans_iter['Q'][0]
                x_hat = multi_mode_dot(x_tnsr, [np.transpose(q) for q in ans_Q], modes=range(d-1))
                x_hat = multi_mode_dot(x_hat, ans_Q, modes=range(d-1))
                fnorm_resid[iiter] = np.linalg.norm(x_tnsr - x_hat) / tnsr_norm
                dis = abs(fnorm_resid[iiter] - fnorm_resid[iiter-1])
                
                if iiter == 1:
                    Qfirst = ans_Q.copy()
                    x_hat_first = x_hat  # Assuming x_hat is already in the desired data format
                iiter += 1
    else:
        iiter += 1
    
    # Finalize fnorm residuals and prepare the model output
    # Assuming fnorm_resid and x are already defined, along with dd (which represents the dimensions of x)
    fnorm_resid = np.array(fnorm_resid)  # Assuming fnorm_resid is some list or array
    fnorm_resid = fnorm_resid[fnorm_resid != 0]  # Filter out zeros
    fnorm_resid = fnorm_resid**2  # Square the elements
    x0 = x.reshape(-1, dd[-1])  # Flattens all dimensions except the last one
    x0 = x0.T  # Transpose (equivalent to t(x0) in R)
    x0 = x0.T  # Transpose back to original orientation (this undoes the previous transpose, thus has no net effect here)
    x0 = x0.reshape(dd)  # Reshape back to original dimensions
    
    # Placeholder for final model assembly
    model = {
        "Ft": Ft,  # Placeholder for Ft calculation
        "Ft.all": Ft_all,  # Placeholder for Ft.all calculation
        "Q": ans_Q,
        "x.hat": x_hat,
        "niter": iiter,
        "fnorm.resid": fnorm_resid[-1]
    }
    
    # Placeholder for tenFM object creation - adjust as needed
    tenFM = model  # Placeholder for actual model object creation
    
    return tenFM

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
from tensorly.decomposition import tucker
#from tenFM import *

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_shape=None):
        super(TemporalConvNet, self).__init__()
        self.output_shape = output_shape
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv3d(in_channels, out_channels, kernel_size,
                                 stride=1, dilation=dilation_size, padding=padding),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv3d(num_channels[-1], 1, kernel_size=1)  # Ensuring the final output has 1 channel
        self.adaptive_layer = nn.AdaptiveAvgPool3d((output_shape[0], output_shape[1], output_shape[2]))

    def forward(self, x):
        x = self.network(x)
        x = self.final_conv(x)
        x = self.adaptive_layer(x)
        x = x.squeeze(1)  # Remove the singleton channel dimension to match the target shape
        return x

def train_tcn_model(model, X_train, Y_train, epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        #print("Output shape:", output.shape)
        #print("Target shape:", Y_train.shape)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        
def function_m(x):
    """Element-wise ReLU applied to x."""
    return np.cos(x)

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(2)
np.random.seed(2)

def generate_random_matrix(rows, cols):
    """Generate a random matrix with normally distributed entries."""
    return np.random.randn(rows, cols)

def tensor_product_contract(A, B, modes):
    """Contracted tensor product of tensors A and B along specified modes."""
    #print(f"Shape of A: {A.shape}")
    #print(f"Shape of B: {B.shape}")
    #print(f"Modes: {modes}")
    return np.tensordot(A, B, axes=([1,2,3], [0,1,2]))

def generate_Y(Ft, dims, R, output_shape, noise_level=0.1):
    """Generate tensor Y from Ft using the contracted tensor product."""
    # Generate random matrices U_k and V_d
    U_matrices = [generate_random_matrix(dims[k], R) for k in range(len(dims))]
    V_matrices = [generate_random_matrix(output_shape[d], R) for d in range(len(output_shape))]
    
    # Form coefficient tensor B from U_matrices and V_matrices
    B_shape = list(dims) + list(output_shape)  # Convert both to lists before concatenation
    B = np.zeros(B_shape)
    for idx in range(R):
        outer_product = U_matrices[0][:, idx]
        for k in range(1, len(U_matrices)):
            outer_product = np.outer(outer_product, U_matrices[k][:, idx])
        outer_product = outer_product.flatten()
        for d in range(len(V_matrices)):
            outer_product = np.outer(outer_product, V_matrices[d][:, idx])
            outer_product = outer_product.flatten()
        B += outer_product.reshape(B_shape)
    
    #print(f"Shape of B: {B.shape}")
    
    # Compute the contracted tensor product of Ft and B
    Y = tensor_product_contract(function_m(Ft), B, list(range(len(dims))))
    
    #print(f"Shape of Y before adding noise: {Y.shape}")
    
    # Add noise
    noise = np.random.normal(0, noise_level, Y.shape)
    Y += noise
    
    return Y


# In[5]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
from tensorly.decomposition import tucker

df = pd.read_csv('tenFM_yellow_taxi_manhattan_14_all.csv')
df = df.drop(df.columns[0], axis=1)

values = df.values  # Extract values from the DataFrame
tensor_shape = (504, 69, 69, 24)  # Desired shape of the tensor

# Reshape the values into the tensor
X_train_np = values.reshape(tensor_shape)
Y_train_np = values.reshape(tensor_shape)


tensor_t = X_train_np.transpose(3,0,1,2)
tensor_t = tensor_t[6:14]
X_train_np = tensor_t.transpose(1,2,3,0)

tensor_t = Y_train_np.transpose(3,0,1,2)
tensor_t = tensor_t[14:22]
Y_train_np = tensor_t.transpose(1,2,3,0)
Y_train_np.shape


X_train_ts = torch.tensor(X_train_np, dtype=torch.float32)  # Shape: (T, 1, depth, height, width)
Y_train_ts = torch.tensor(Y_train_np, dtype=torch.float32)

# Given location IDs in the 69x69 dimensions
location_ids = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]
# Specific location IDs to create the subtensor
specific_ids = [4, 12, 13, 88, 87, 261, 209, 231, 45, 125, 144, 211, 148, 232, 158, 249, 114, 113, 79]
#specific_ids = [4, 12, 13, 88, 87, 261, 209, 231, 45, 125, 144, 211, 14, 232, 158, 249, 
                #114, 113, 79,246,68,90,100,186,234,164,170,107,224,137,233]
# Map specific_ids to their indices in location_ids
indices = [location_ids.index(id) for id in specific_ids if id in location_ids]

# Assume X_train_np is your original numpy array
# Now using the indices to slice the array
subtensor = X_train_ts[:, indices, :, :]
X = subtensor[:, :, indices, :]
subtensor2 = Y_train_ts[:, indices, :, :]
Y = subtensor2[:, :, indices, :]
print(X.shape)
print(Y.shape)


# In[6]:


import torch

# Assuming both X_train and Y_train are shaped [503, 69, 69, 24]

# Create a zero tensor for the initial lag
zero_pad = torch.zeros(1, *Y.shape[1:], device=Y.device)

# Create the lagged Y tensor
Y_lagged = torch.cat([zero_pad, Y[:-1]], dim=0)

# Ensure that Y_lagged is not adding an extra dimension
print("Shape of Y_lagged:", Y_lagged.shape)  # Should be [503, 69, 69, 24]

# Concatenate Y_lagged with X_train along a new fifth dimension if needed
# Since X_train and Y_lagged are both 4D, we need to add a dimension before concatenation to treat the spatial dimensions separately
X_train_with_lag = torch.cat([X.unsqueeze(1), Y_lagged.unsqueeze(1)], dim=1)

# Now X_train_with_lag should have the shape [503, 2, 69, 69, 24]
print("Shape of X_train with lag:", X_train_with_lag.shape)

# This new shape treats the two input types (X and Y_lagged) as separate channels of input,
# which should be compatible with a 3D convolutional layer expecting a channel dimension.
num_train = int(X_train_with_lag.shape[0]*0.7)
X_test = X_train_with_lag[num_train:]
X_train = X_train_with_lag[:num_train]
Y_test = Y[num_train:]
Y_train = Y[:num_train]


# In[7]:


print(X_test.shape)
print(Y_test.shape)
print(X_train.shape)
print(Y_train.shape)


# In[8]:


import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.data import ArrayDataset, DataLoader
torch.manual_seed(123)
np.random.seed(123)
# Convert PyTorch tensors to MXNet NDArray
X_train_mx = nd.array(X_train.numpy())
Y_train_mx = nd.array(Y_train.numpy())
X_test_mx = nd.array(X_test.numpy())
Y_test_mx = nd.array(Y_test.numpy())

# Create datasets
train_dataset = ArrayDataset(X_train_mx, Y_train_mx)
test_dataset = ArrayDataset(X_test_mx, Y_test_mx)

# Create data loaders
batch_size = 32
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[9]:


from mxnet.gluon import Block
from mxnet import nd
import mxnet as mx
import tensorly as tl
from tensorly.random import check_random_state

class TRL(Block):
    def __init__(self, ranks, n_outputs, input_shape=(2, 19, 19, 8), verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        with self.name_scope():
            self.ranks = tuple(ranks)
            self.n_outputs = n_outputs
            self.input_shape = input_shape
            self.weight_initializer = mx.init.Xavier(magnitude=2.24)

            # Core of the Tucker tensor
            self.core = self.params.get(
                'core', init=self.weight_initializer, 
                shape=self.ranks)

            # Factors of the Tucker tensors
            self.factors = [self.params.get(
                'factor_{}'.format(i), init=self.weight_initializer,
                allow_deferred_init=True,
                shape=(0, e)) for (i, e) in enumerate(ranks[:-1])]

            self.factors.append(self.params.get(
                'factor_{}'.format(len(ranks) - 1), init=self.weight_initializer,
                shape=(n_outputs, ranks[-1])))

            # Optional: add a bias term to the regression
            self.bias = self.params.get('bias', shape=(1, n_outputs))
            self.initialised_factors = False
            self.verbose = verbose
            self.displayed_info = False

    def forward(self, x):
        if not self.initialised_factors:
            if self.verbose:
                print('TRL: initializing factors') 
            for i in range(len(self.ranks) - 1):
                # Debug: Print each dimension being set
                print(f"Setting shape for factor {i} to {(x.shape[i+1], self.ranks[i])}")
                self.factors[i].shape = (x.shape[i+1], self.ranks[i])
                self.factors[i]._finish_deferred_init()
            self.initialised_factors = True
            if self.verbose:
                print('  ..done.')
        
        if self.verbose and not self.displayed_info:
            print('TRL on input of shape={} with batch size={}.'.format(
                            x.shape[1:], x.shape[0], self.n_outputs))
            print('    TRL-rank={} corresponding to ({} [inputs], {} [outputs])'.format(
                            self.core.shape, x.shape[1:], self.n_outputs))
            self.displayed_info = True
            
        with x.context:
            # Reshape input tensor
            x = x.reshape((x.shape[0], -1))

            regression_weights = tl.tucker_to_tensor((self.core.data(), [f.data() for f in self.factors]))
            result = nd.dot(x, nd.reshape(regression_weights, (-1, self.n_outputs)))

            return nd.broadcast_add(result, self.bias.data())
        
    def penalty(self, order=2):
        penalty = tl.norm(self.core.data(), order)
        for f in self.factors:
            penalty += tl.norm(f.data(), order)
        return penalty
    
    @property
    def regression_weights(self):
        return tl.tucker_to_tensor((self.core.data(), [f.data() for f in self.factors]))


# In[10]:


from mxnet.gluon import nn

# Context setup
ctx = mx.cpu()  # or mx.gpu() if GPU is available

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv3D(channels=20, kernel_size=(1, 1, 4), activation='relu'))
    net.add(nn.MaxPool3D(pool_size=1, strides=1))            
    net.add(nn.Conv3D(channels=50, kernel_size=(1, 1, 4), activation='relu'))
    net.add(nn.MaxPool3D(pool_size=1, strides=1))
    net.add(nn.Conv3D(channels=50, kernel_size=(1, 1, 2), activation='relu'))
    net.add(nn.MaxPool3D(pool_size=1, strides=1))
    
    # The ranks should be adapted to your specific case
    # Modify ranks and n_outputs according to your output shape (6, 8, 6)
    trl = TRL([12, 12, 12, 50], np.prod([19, 19, 8]), input_shape=(2, 19, 19, 8))
    net.add(trl)

# Initialize parameters
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

# Define the loss and optimizer
mse_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .0004})


# In[11]:


# Updated evaluation function for regression
def evaluate_regression(data_iterator, net):
    mse_metric = mx.metric.MSE()
    mae_metric = mx.metric.MAE()

    for i, batch in enumerate(data_iterator):
        data = batch[0].as_in_context(ctx)
        label = batch[1].as_in_context(ctx)

        # Forward pass
        output = net(data).reshape(label.shape)

        # Update metrics
        mse_metric.update([label], [output])
        mae_metric.update([label], [output])

    mse_value = mse_metric.get()[1]
    mae_value = mae_metric.get()[1]

    return mse_value, mae_value

# Training loop
n_epoch = 100
smoothing_constant = .001
rng = check_random_state(1)
C = 0.001

for e in range(n_epoch):
    for i, batch in enumerate(train_data):
        data = batch[0].as_in_context(ctx)
        label = batch[1].as_in_context(ctx)
        
        with autograd.record():
            prediction = net(data).reshape(label.shape)
            loss = mse_loss(prediction, label) + C*trl.penalty()
        
        loss.backward()
        trainer.step(data.shape[0])
        
        current_loss = nd.mean(loss).asscalar()
        moving_loss = (current_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * current_loss)
    
    test_mse, test_mae = evaluate_regression(test_data, net)
    train_mse, train_mae = evaluate_regression(train_data, net)
    print(f"Epoch {e}. Loss: {moving_loss}, Train MSE: {train_mse}, Train MAE: {train_mae}, Test MSE: {test_mse}, Test MAE: {test_mae}")


# In[ ]:




