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
    e = np.random.normal(loc=0, scale=0.05, size=(t + 500, np.prod(dim)))
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
    E = np.random.normal(loc=0, scale=0.05,size=(t, *dims))
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils


## Convolutional Tensor-Train LSTM Module
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the Conv-TT-LSTM 
        input_channels, hidden_channels,
        # convolutional tensor-train network
        order = 3, steps = 3, ranks = 8,
        # convolutional operations
        kernel_size = 2, bias = True):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the convolutional tensor-train format)
        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Convolutional tensor-train network
        self.steps = steps
        self.order = order

        self.lags = steps - order + 1

        ## Convolutional operations
        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        Conv3d = lambda in_channels, out_channels: nn.Conv3d(
            in_channels = in_channels, out_channels = out_channels, bias = bias,
            kernel_size = kernel_size + (self.lags, ), padding = padding + (0,))

        ## Convolutional layers
        self.layers  = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv3d(
                in_channels = hidden_channels, out_channels = ranks))

    def initialize(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """

        if first_step: self.initialize(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.stack(input_states, dim = -1)
            input_states = self.layers_[l](input_states)
            input_states = torch.squeeze(input_states, dim = -1)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)
                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps
        
        return outputs


## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size = 2, bias = True):
        """
        Construction of convolutional-LSTM cell.
        
        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels  = input_channels + hidden_channels, 
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)
        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of convolutional-LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states   = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

# Define ConvLSTMNet Model
class ConvLSTMNet(nn.Module):
    def __init__(self,
                 input_channels,
                 layers_per_block, hidden_channels, skip_stride=None,
                 scope="all", scope_params={},
                 cell="convlstm", cell_params={},
                 kernel_size=3, bias=True,
                 output_sigmoid=False,
                 output_size=(6, 8)):
        super(ConvLSTMNet, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Module type of convolutional LSTM layers
        if cell == "convlstm":  # standard convolutional LSTM
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                kernel_size=kernel_size, bias=bias)
        elif cell == "convttlstm":  # convolutional tensor-train LSTM
            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                order=cell_params.get("order", 3), steps=cell_params.get("steps", 5), ranks=cell_params.get("rank", 16),
                kernel_size=kernel_size, bias=bias)
        else:
            raise NotImplementedError

        ## Construction of convolutional tensor-train LSTM network
        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0:  # if l == 0 and b == 0:
                    channels = input_channels
                else:  # if l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b-1-self.skip_stride]

                lid = "b{}l{}".format(b, l)  # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1-self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, input_channels,
                                          kernel_size=1, padding=0, bias=True)

        # Adjust the output size
        self.output_size = output_size

    def forward(self, inputs, input_frames, future_frames, output_frames, 
                teacher_forcing=False, scheduled_sampling_ratio=0):
        # compute the teacher forcing mask 
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio * 
                                                   torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1, device=inputs.device))
        else:
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames: 
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t-1] * (1 - mask)

            first_step = (t == 0)

            queue = []  # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_ = self.layers[lid](input_, first_step=first_step)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)  # concat over the channels

            outputs[t] = self.layers["output"](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])

        outputs = outputs[-output_frames:]

        # Reshape the output to match the desired output size
        outputs = torch.stack([outputs[t] for t in range(output_frames)], dim=1)
        
        # Adjust spatial dimensions
        batch_size, output_frames, channels, height, width = outputs.size()
        outputs = outputs.view(-1, channels, height, width)  # Flatten output_frames and batch_size

        # Interpolate to the correct output size
        outputs = nn.functional.interpolate(outputs, size=self.output_size, mode='bilinear', align_corners=False)
        
        # Restore the output_frames dimension
        outputs = outputs.view(batch_size, output_frames, channels, self.output_size[0], self.output_size[1])

        return outputs

# Function to train ConvLSTM model
def train_conv_lstm_net(model, X_train, Y_train, input_frames, future_frames, output_frames, epochs=100, lr=0.005):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through the model
        output = model(X_train, input_frames, future_frames, output_frames)
        
        # Ensure the output shape matches the target shape
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


df = pd.read_csv('tenFM_yellow_taxi_manhattan_14_all.csv')
df = df.drop(df.columns[0], axis=1)
df1 = df1.drop(df.columns[0], axis=1)

values = df.values  # Extract values from the DataFrame
values1 = df1.values
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

########################################################################
########################################################################
########################################################################

import numpy as np

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

########################################################################
########################################################################
########################################################################

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

Y = Y.unsqueeze(1)
Y = torch.cat([Y, Y], dim=1)
num_train = int(X_train_with_lag.shape[0]*0.7)
X_test = X_train_with_lag[num_train:]
X_train = X_train_with_lag[:num_train]
Y_test = Y[num_train:]
Y_train = Y[:num_train]
X_train = X_train.permute(0,2,1,3,4)
Y_train = Y_train.permute(0,2,1,3,4)
X_test = X_test.permute(0,2,1,3,4)
Y_test = Y_test.permute(0,2,1,3,4)
########################################################################
########################################################################
########################################################################

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils


## Convolutional Tensor-Train LSTM Module
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the Conv-TT-LSTM 
        input_channels, hidden_channels,
        # convolutional tensor-train network
        order = 3, steps = 3, ranks = 8,
        # convolutional operations
        kernel_size = 2, bias = True):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the convolutional tensor-train format)
        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Convolutional tensor-train network
        self.steps = steps
        self.order = order

        self.lags = steps - order + 1

        ## Convolutional operations
        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        Conv3d = lambda in_channels, out_channels: nn.Conv3d(
            in_channels = in_channels, out_channels = out_channels, bias = bias,
            kernel_size = kernel_size + (self.lags, ), padding = padding + (0,))

        ## Convolutional layers
        self.layers  = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv3d(
                in_channels = hidden_channels, out_channels = ranks))

    def initialize(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """

        if first_step: self.initialize(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.stack(input_states, dim = -1)
            input_states = self.layers_[l](input_states)
            input_states = torch.squeeze(input_states, dim = -1)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)
                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps
        
        return outputs


## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size = 2, bias = True):
        """
        Construction of convolutional-LSTM cell.
        
        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels  = input_channels + hidden_channels, 
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)
        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of convolutional-LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states   = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states 

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

# Define ConvLSTMNet Model
class ConvLSTMNet(nn.Module):
    def __init__(self,
                 input_channels,
                 layers_per_block, hidden_channels, skip_stride=None,
                 scope="all", scope_params={},
                 cell="convlstm", cell_params={},
                 kernel_size=3, bias=True,
                 output_sigmoid=False):
        super(ConvLSTMNet, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Module type of convolutional LSTM layers
        if cell == "convlstm":  # standard convolutional LSTM
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                kernel_size=kernel_size, bias=bias)
        elif cell == "convttlstm":  # convolutional tensor-train LSTM
            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                order=cell_params.get("order", 3), steps=cell_params.get("steps", 5), ranks=cell_params.get("rank", 16),
                kernel_size=kernel_size, bias=bias)
        else:
            raise NotImplementedError

        ## Construction of convolutional tensor-train LSTM network
        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0:  # if l == 0 and b == 0:
                    channels = input_channels
                else:  # if l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b-1-self.skip_stride]

                lid = "b{}l{}".format(b, l)  # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1-self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, input_channels,
                                          kernel_size=1, padding=0, bias=True)

    def forward(self, inputs, input_frames, future_frames, output_frames, 
                teacher_forcing=False, scheduled_sampling_ratio=0):
        # compute the teacher forcing mask 
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio * 
                                                   torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1, device=inputs.device))
        else:
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames: 
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t-1] * (1 - mask)

            first_step = (t == 0)

            queue = []  # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_ = self.layers[lid](input_, first_step=first_step)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)  # concat over the channels

            outputs[t] = self.layers["output"](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])

        outputs = outputs[-output_frames:]

        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack([outputs[t] for t in range(output_frames)], dim=1)

        return outputs

# Initialize the ConvLSTMNet model
input_channels = 2  # Number of channels in the input video (e.g., 1 for grayscale, 3 for RGB)
layers_per_block = [2, 2, 2]  # Example: 3 blocks with 2 layers each
hidden_channels = [50, 50, 50]  # Example: Number of output channels in each block
skip_stride = 1  # Example: Skip connection every block
kernel_size = 3  # Kernel size for Conv2D operations
cell = "convttlstm"  # Using ConvTTLSTMCell
cell_params = {"order": 3, "steps": 5, "rank": 16}  # Parameters specific to ConvTTLSTMCell

conv_lstm_net = ConvLSTMNet(
    input_channels=input_channels,
    layers_per_block=layers_per_block,
    hidden_channels=hidden_channels,
    skip_stride=skip_stride,
    cell=cell,
    cell_params=cell_params,
    kernel_size=kernel_size
)

# Function to train ConvLSTM model
import random

def train_conv_lstm_net(model, X_train, Y_train, input_frames, future_frames, output_frames, batch_size=32, epochs=10, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    num_samples = X_train.size(0)
    
    for epoch in range(epochs):
        # Randomly select a subset of indices for this epoch
        indices = random.sample(range(num_samples), batch_size)
        
        # Select the corresponding samples
        X_batch = X_train[indices]
        Y_batch = Y_train[indices]
        
        optimizer.zero_grad()
        
        # Forward pass through the model
        output = model(X_batch, input_frames, future_frames, output_frames)
        
        # Compute loss
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()
        
        # Print the loss for the epoch
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
batch_size = 8
input_frames = 19  # Number of input frames to the model
future_frames = 19  # Number of future frames to predict
output_frames = 19  # Number of output frames to return
train_conv_lstm_net(conv_lstm_net, X_train, Y_train, input_frames, future_frames, output_frames, batch_size=batch_size, epochs=500, lr=0.0003)

import torch
import torch.nn as nn

def evaluate_model(model, X_test, Y_test, input_frames, future_frames, output_frames):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()  # Define the loss function
    
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(X_test, input_frames, future_frames, output_frames)
        
        # Calculate the MSE on the entire test set
        mse = criterion(output, Y_test)
    
    return mse.item()  # Return the MSE value

mse = evaluate_model(conv_lstm_net, X_test, Y_test, input_frames, future_frames, output_frames)
print(f"Test Set MSE: {mse}")
