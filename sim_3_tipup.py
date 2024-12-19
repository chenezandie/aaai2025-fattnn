

import numpy as np
from math import prod
from numpy.linalg import qr
from scipy.linalg import fractional_matrix_power
import tensorly as tl
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
    e = np.random.normal(loc=0, scale=1, size=(t + 500, np.prod(dim)))
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
    E = np.random.normal(size=(t, *dims))
    # Apply lambda_val to adjust signal strength
    X = lambda_val * X + E
    
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


# In[7]:


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
    return np.log1p(np.exp(x))


# In[8]:


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

dims_X = [12, 3, 12] 
r = [4, 3, 4] 
modes = [0, 1, 2]
t = 100
num_channels = [50, 50, 50]
kernel_size = 2
dropout = 0.2
lambda_val = np.sqrt(dims_X[0] * dims_X[1] * dims_X[2])
T = t
ranks = r
# Example usage:
dims = r 
R = 3  # Fixed R
output_shape = (3, 3, 3) 
rho = 0.9
lambda_val = np.sqrt(np.prod(dims))
Ft = tenAR_sim(t=t, dim=dims, R=1, P=1, rho=rho, cov='iid')
Y = generate_Y(Ft, dims, R, output_shape, noise_level=0.5)
X = tenFM_sim(Ft=Ft, dims=dims_X, lambda_val=lambda_val, A = None)
print(X.shape)
dims = dims_X 
results = tenFM_est(X,r,h0=0,iter=True,method='TIPUP')
cores = []
for i in range(T):
    abc = results['Ft'].transpose(3,0,1,2)
    core = abc[i]
    cores.append(core)
X = torch.tensor(cores, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
print(X.shape)
Y = torch.tensor(Y, dtype=torch.float32)
ds = 80
X_test = X[ds:]
X_train = X[:ds]
Y_test = Y[ds:]
Y_train = Y[:ds]





# Lists to store MSE and RPE values for each run
mse_values = []
for seed in range(1, 21):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Define your model here (if it's different for each seed, you need to redefine it)
    tcn_model = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, output_shape=output_shape)
    
    # Train the model
    train_tcn_model(tcn_model, X_train, Y_train, epochs=300, lr=0.003)
    
    # Set model to evaluation mode
    tcn_model.eval()
    
    # No gradient computation
    with torch.no_grad():
        predicted = tcn_model(X_test)
        
        # MSE calculation
        mse_loss = nn.MSELoss()
        mse = mse_loss(predicted, Y_test)
        mse_values.append(mse.item())

# Calculate average MSE and RPE
average_mse = sum(mse_values) / len(mse_values)

print(f"Average Test MSE over 20 runs: {average_mse}")

rmse = []
for i in mse_values:
    a = np.sqrt(i)
    rmse.append(a)
sum(rmse)/20  



