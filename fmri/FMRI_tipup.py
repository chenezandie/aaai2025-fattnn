#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nilearn import datasets

# Fetch sample dataset
haxby_dataset = datasets.fetch_haxby()
# fMRI data and anatomical image
fmri_filename = haxby_dataset.func[0]
# Load the fMRI data using nibabel
fmri_img_nii = nib.load(fmri_filename)
fmri_data = fmri_img_nii.get_fdata()

# Normalize the data
fmri_data_normalized = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)

class FMRISequenceDataset(Dataset):
    def __init__(self, fmri_volumes, num_input_slices, num_predict_slices):
        """
        Args:
            fmri_volumes (numpy.ndarray): 4D array of fMRI data (depth, height, width, time).
            num_input_slices (int): Number of slices in each input sequence.
            num_predict_slices (int): Number of slices to predict.
        """
        # Assuming that you only want to consider a specific depth slice across all times,
        # here using the middle slice for simplicity. Adjust index as needed.
        middle_slice_index = fmri_volumes.shape[0] // 2
        self.fmri_volumes = fmri_volumes[middle_slice_index]  # Shape now should be (64, 64, 1452)
        self.num_input_slices = num_input_slices
        self.num_predict_slices = num_predict_slices
        self.total_volumes = self.fmri_volumes.shape[2]  # The time dimension size

    def __len__(self):
        return self.total_volumes - self.num_input_slices - self.num_predict_slices + 1

    def __getitem__(self, idx):
        X = self.fmri_volumes[:, :, idx:idx + self.num_input_slices]  # (64, 64, num_input_slices)
        Y = self.fmri_volumes[:, :, idx + self.num_input_slices:idx + self.num_input_slices + self.num_predict_slices]  # (64, 64, num_predict_slices)
        return torch.tensor(X, dtype=torch.float32).permute(2, 0, 1), torch.tensor(Y, dtype=torch.float32).permute(2, 0, 1)

num_input_slices = 25
num_predict_slices = 1
dataset = FMRISequenceDataset(fmri_data_normalized, num_input_slices=num_input_slices, num_predict_slices=num_predict_slices)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

from sklearn.model_selection import train_test_split

# Split indices to create training and validation sets
indices = list(range(len(dataset)))  # get indices of the whole dataset
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create PyTorch data samplers and loaders:
from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)


# In[5]:


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
        x[i, :] = temp
    
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
    
    # Apply lambda_val to adjust signal strength
    X = lambda_val * X
    
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
        x_hat = multi_mode_dot(x, transposed_Q, modes = [0,1,2])
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
        print("x_new is already in the expected shape.")
        return x_new

    # Check for (3, m, 3, 100) shape
    if x_new.shape[0] == r[0] and x_new.shape[2] == r[2] and x_new.shape[1] in valid_m_values and x_new.shape[3] == max(x_new.shape[0],x_new.shape[1],x_new.shape[2],x_new.shape[3]):
        x_new = x_new.transpose(1, 0, 2, 3)
        print(f"x_new reshaped to {x_new.shape}.")
        return x_new
    
    # Check for (3, 3, m, 100) shape
    elif x_new.shape[0] == r[0] and x_new.shape[1] == r[1] and x_new.shape[2] in valid_m_values and x_new.shape[3] == max(x_new.shape[0],x_new.shape[1],x_new.shape[2],x_new.shape[3]):
        x_new = x_new.transpose(2, 0, 1, 3)
        print(f"x_new reshaped to {x_new.shape}.")
        return x_new
    
    print("x_new does not match any expected pattern for reshaping.")
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
    x_hat = multi_mode_dot(x, transposed_Q, modes = [0,1,2])
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


# In[6]:


def loader_to_tensor(data_loader):
    x_batches = []
    y_batches = []
    
    # Iterate over the DataLoader
    for x_batch, y_batch in data_loader:
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    
    # Concatenate all batches to form a single tensor
    X_tensor = torch.cat(x_batches, dim=0)  # Assuming dim=0 is your batch dimension
    Y_tensor = torch.cat(y_batches, dim=0)

    return X_tensor, Y_tensor

# Get tensors from loaders
X_train_tensor, Y_train_tensor = loader_to_tensor(train_loader)
X_test_tensor, Y_test_tensor = loader_to_tensor(val_loader)

X_train = np.array(X_train_tensor)
Y_train = np.array(Y_train_tensor)
X_test = np.array(X_test_tensor)
Y_test = np.array(Y_test_tensor)

X_train = X_train.transpose(1,2,3,0)
X_test = X_test.transpose(1,2,3,0)

modes = [0,1,2]
dims = [25, 64, 64]
r = [8, 23, 23]
ranks = [8, 23, 23]
h = 0
results = tenFM_est(X_train,r = ranks,h0=0,iter=True,method='TIPUP')
results_test = tenFM_est(X_test,r = ranks,h0=0,iter=True,method='TIPUP')


# In[7]:


X_train_np = X_train.transpose(3,0,1,2)
cores = []
factors_list = []
for i in range(X_train_tensor.shape[0]):
    abc = results['Ft'].transpose(3,0,1,2)
    core = abc[i]
    cores.append(core)
X_train_tensor = torch.tensor(cores, dtype=torch.float32)
X_train_tensor = X_train_tensor.unsqueeze(1)


# In[8]:


X_test_np = X_test.transpose(3,0,1,2)
cores = []
factors_list = []
for i in range(X_test_tensor.shape[0]):
    abc = results['Ft'].transpose(3,0,1,2)
    core = abc[i]
    cores.append(core)
X_test_tensor = torch.tensor(cores, dtype=torch.float32)
X_test_tensor = X_test_tensor.unsqueeze(1)


import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import tensorly as tl
from tensorly.decomposition import tucker

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        self.num_channels = num_channels
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride=1, padding=padding, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        # Adding a final convolutional layer to adjust channel dimensions
        self.final_conv = nn.Conv2d(num_channels[-1], 1, 1)  # Reduce to 1 channel
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))  # Ensure spatial dimensions are 72x72

    def forward(self, x):
        batch_size, channels, seq_length, height, width = x.size()
        x = x.view(batch_size, channels * seq_length, height, width)  # Reshape input
        x = self.network(x)
        x = self.final_conv(x)  # Reduce channel dimensions to 1
        x = self.adaptive_pool(x)  # Ensure output dimensions are 72x72
        return x

def train_tcn_model(model, X_train, Y_train, epochs=100, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Use MSE for image prediction
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# In[35]:


num_channels = [50,50,50]
kernel_size = 3
dropout = 0.2
torch.manual_seed(1)
np.random.seed(1)
tcn_model = TemporalConvNet2D(num_inputs=ranks[0], num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
start_time = time.time()
train_tcn_model(tcn_model, X_train_tensor,Y_train_tensor, epochs=200, lr=0.001)
end_time = time.time()
# Calculate and print the duration
duration = end_time - start_time
print(f"Training completed in {duration:.2f} seconds.")


tcn_model.eval()
with torch.no_grad():
    predicted = tcn_model(X_test_tensor)
    mse_loss = nn.MSELoss()
    mse = mse_loss(predicted, Y_test_tensor)
    print(f"Test MSE: {mse.item()}")
