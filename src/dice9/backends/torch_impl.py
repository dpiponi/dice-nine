import builtins
import functools
import collections
import atexit
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import dice9.config as config

# --- Global Stats ---
_stats = collections.defaultdict(lambda: [0.0, 0])

# --- Scalar Conversion ---
def to_py_scalar(x):
    if isinstance(x, (builtins.int, builtins.float, builtins.bool)):
        return x
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        raise ValueError(f"Expected 0-d tensor (scalar), got tensor with shape {tuple(x.shape)}")
    raise TypeError(f"Unsupported type: {type(x)}")

def convert_to_tuple(x):
    if isinstance(x, list):
        return tuple(convert_to_tuple(i) for i in x)
    elif isinstance(x, torch.Tensor):
        return convert_to_tuple(x.tolist())
    elif isinstance(x, dict):
        return {k: convert_to_tuple(v) for k, v in x.items()}
    else:
        return x

# --- Tensor Basics ---
constant = lambda x, dtype=None: torch.tensor(x, dtype=dtype)
Tensor = torch.Tensor
shape = lambda x: x.shape
rank = lambda x: x.ndim
to_python = convert_to_tuple

# --- Creation ---
reshape = torch.reshape
stack = torch.stack
unstack = torch.unbind
expand_dims = torch.unsqueeze
transpose = lambda x, perm=None: x.permute(*perm if perm else list(reversed(range(x.dim()))))
concat = lambda seq, axis, *args, **kwargs: torch.cat(seq, dim=axis, *args, **kwargs)

# --- Broadcasting ---
if hasattr(torch, "broadcast_to"):
    broadcast_to = torch.broadcast_to
else:
    broadcast_to = lambda x, shape: x.expand(shape)

# --- Reductions ---
def torch_reduce_sum(x, axis=None, keepdims=False, *args, **kwargs):
    if isinstance(axis, torch.Tensor):
        axis = axis.item()
    return torch.sum(x, dim=axis, keepdim=keepdims)

def torch_reduce_min(x, axis=None, keepdims=False, *args, **kwargs):
    if isinstance(axis, torch.Tensor):
        axis = axis.item()
    return torch.min(x, dim=axis, keepdim=keepdims).values

def torch_reduce_max(x, axis=None, keepdims=False, *args, **kwargs):
    if isinstance(axis, torch.Tensor):
        axis = axis.item()
    return torch.max(x, dim=axis, keepdim=keepdims).values

# --- Segmented Ops ---
def torch_unsorted_segment_sum(data, segment_ids, num_segments, axis=0):
    out_shape = list(data.shape)
    out_shape[axis] = num_segments
    out = torch.zeros(out_shape, dtype=data.dtype, device=data.device)
    idx = segment_ids.view(*([1] * axis + [-1] + [1] * (data.dim() - axis - 1))).expand_as(data)
    out.scatter_add_(axis, idx, data)
    return out

def torch_unsorted_segment_min(data, segment_ids, num_segments, axis=0):
    dtype = data.dtype
    init_val = float("inf") if dtype.is_floating_point else torch.iinfo(dtype).max
    out_shape = list(data.shape)
    out_shape[axis] = num_segments
    out = torch.full(out_shape, init_val, dtype=dtype, device=data.device)
    idx = segment_ids.view(*([1] * axis + [-1] + [1] * (data.dim() - axis - 1))).expand_as(data)
    out.scatter_reduce_(axis, idx, data, reduce="amin")
    return out

# --- Cumulative Ops ---
def torch_cumprod_exclusive(x, axis=0, *args, **kwargs):
    cp = torch.cumprod(x, dim=axis)
    truncated = cp.narrow(axis, 0, cp.size(axis) - 1)
    ones = torch.ones_like(x.select(axis, 0), dtype=torch.int64).unsqueeze(axis)
    return torch.cat([ones, truncated], dim=axis)

# --- Indexing ---
def torch_gather(params, indices, axis=0, **kwargs):
    return torch.index_select(params, dim=axis, index=indices)

def subscript(tensor: torch.Tensor, idx: torch.LongTensor) -> torch.Tensor:
    batch = torch.arange(tensor.size(0), device=idx.device)
    return tensor[batch, idx]

def scatter_update_op(x, slices, updates, op=None):
    num_worlds = x.shape[0]

    if isinstance(slices, int) or (isinstance(slices, torch.Tensor) and slices.ndim == 0):
        slices = torch.full((num_worlds,), slices, dtype=torch.long, device=x.device)

    row_indices = torch.arange(num_worlds, device=x.device)
    x_new = x.clone()

    if op is None:
        x_new[row_indices, slices] = updates
    else:
        x_new[row_indices, slices] = op(x_new[row_indices, slices], updates)

    return x_new

def scatter_update(x, slices, updates):
    return scatter_update_op(x, slices, updates, op=None)

def scatter_update_add(x, slices, updates):
    return scatter_update_op(x, slices, updates, op=torch.add)

def scatter_update_sub(x, slices, updates):
    return scatter_update_op(x, slices, updates, op=torch.sub)

def scatter_update_multiply(x, slices, updates):
    return scatter_update_op(x, slices, updates, op=torch.mul)


# --- Casting ---
def cast(x, dtype, device=None):
    return torch.as_tensor(x, dtype=dtype, device=device)

def bitcast(tensor, dtype):
    return tensor.view(dtype)

# --- Topk ---
def top_k(tensor, n):
    return tensor.topk(n)

# --- Repeats and Bincount ---
def rowwise_bincount(x, size):
    M, N = x.shape
    x_flat = x.reshape(-1)
    row_idx = torch.arange(M, device=x.device).repeat_interleave(N)
    valid = (x_flat >= 0) & (x_flat < size)
    x_valid = x_flat[valid]
    row_idx_valid = row_idx[valid]
    result = torch.zeros((M, size), dtype=torch.int64, device=x.device)
    result.index_put_((row_idx_valid, x_valid), torch.ones_like(x_valid), accumulate=True)
    return result

# --- Sampling ---
def dd_helper(nv):
    B = nv.size(0)
    device = nv.device
    dtype = torch.float64
    max_nv = int(nv.max().item())
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_nv)
    roll_vals = torch.arange(max_nv, device=device).unsqueeze(0).expand(B, max_nv)
    mask = roll_vals < nv.unsqueeze(1)
    indices = batch_idx[mask]
    rolls = roll_vals[mask]
    probs = (1.0 / nv.to(dtype)).unsqueeze(1).expand(B, max_nv)[mask]
    return probs, rolls, indices

def rolls(x):
    x = x.to(torch.int64)
    max_val = torch.max(x)
    base_sequence = torch.arange(max_val, dtype=torch.int64, device=x.device)
    mask = base_sequence.unsqueeze(0) < x.unsqueeze(1)
    tiled_sequence = base_sequence.unsqueeze(0).expand(x.size(0), -1)
    masked_sequence = torch.where(mask, tiled_sequence, torch.zeros_like(tiled_sequence))
    return masked_sequence[mask]

def repeat_with_indices(counts):
    indices = torch.arange(len(counts), device=counts.device)
    return torch.repeat_interleave(indices, counts)

def get_sample(ranges, roll):
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    cumsums = torch.cumsum(group_sizes, dim=1)
    cumsums_excl = torch.cat([torch.zeros(cumsums.size(0), 1, dtype=cumsums.dtype, device=cumsums.device), cumsums[:, :-1]], dim=1)
    group_index = torch.sum((cumsums_excl <= roll[:, None]), dim=1) - 1
    batch_indices = torch.arange(ranges.size(0), device=ranges.device)
    gather_indices = torch.stack([batch_indices, group_index], dim=1)
    selected_groups = ranges[gather_indices[:, 0], gather_indices[:, 1]]
    group_offsets = cumsums_excl[gather_indices[:, 0], gather_indices[:, 1]]
    return selected_groups[:, 0] + selected_groups[:, 2] * (roll - group_offsets)

def multiroll_helper(ranges, p):
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    total_sizes = group_sizes.sum(dim=1)
    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)
    new_ranges = ranges[indices_for_rolls]
    group_sizes = (new_ranges[:, :, 1] - new_ranges[:, :, 0]) // new_ranges[:, :, 2]
    cumsums = torch.cumsum(group_sizes, dim=1)
    cumsums_excl = torch.cat([torch.zeros(cumsums.size(0), 1, dtype=cumsums.dtype, device=cumsums.device), cumsums[:, :-1]], dim=1)
    group_index = torch.sum((cumsums_excl <= all_rolls[:, None]), dim=1) - 1
    batch_indices = torch.arange(group_index.size(0), device=ranges.device)
    gather_indices = torch.stack([batch_indices, group_index], dim=1)
    selected_groups = new_ranges[gather_indices[:, 0], gather_indices[:, 1]]
    group_offsets = cumsums_excl[gather_indices[:, 0], gather_indices[:, 1]]
    offset_within_group = all_rolls - group_offsets
    sample = selected_groups[:, 0].to(torch.int64) + selected_groups[:, 2].to(torch.int64) * offset_within_group.to(torch.int64)
    weights = p[indices_for_rolls] / total_sizes[indices_for_rolls].to(torch.float64)
    return sample, weights, indices_for_rolls

# --- Hash Collision ---
def check_hash_collision(keys: torch.Tensor, hashes: torch.Tensor) -> bool:
    sorted_idx = torch.argsort(hashes)
    hashes_sorted = hashes[sorted_idx]
    keys_sorted = keys[sorted_idx]
    same_hash = hashes_sorted[1:] == hashes_sorted[:-1]
    same_keys = (keys_sorted[1:] == keys_sorted[:-1]).all(dim=1)
    collision = same_hash & (~same_keys)
    any_collision = collision.any().item()
    if any_collision:
        idx = torch.nonzero(collision, as_tuple=False)[0, 0].item()
        print("!!!!!!!!!!!!!!!!!", collision)
        print(f"idx={idx}")
        print(f"{keys_sorted[idx]} -> {hashes_sorted[idx].item()}")
        print(f"{keys_sorted[idx+1]} -> {hashes_sorted[idx+1].item()}")
    return any_collision

# --- Dtypes ---
__name__ = "torch"
uint64 = torch.int64
int64 = torch.int64
int32 = torch.int32
uint32 = torch.uint32
float64 = torch.float64
float32 = torch.float32
complex128 = torch.complex128
complex64 = torch.complex64
bool = torch.bool
ptype = torch.float64

# --- Export API ---
exported = [
    ("det", lambda x: torch.round(torch.linalg.det(x.to(torch.float64))).to(torch.int64)),
    ("arange", torch.arange),
    ("bincount", rowwise_bincount),
    ("convert_to_tuple", convert_to_tuple),
    ("isin", torch.isin),
    ("to_list", lambda x: x.numpy().tolist()),
    ("abs", torch.abs),
    ("add", torch.add),
    ("argmax", torch.argmax),
    ("argmin", torch.argmin),
    ("argsort", torch.argsort),
    ("bitwise_xor", torch.bitwise_xor),
    ("cast", cast),
    ("convert_to_tensor", torch.as_tensor),
    ("cumprod_exclusive", torch_cumprod_exclusive),
    ("cumsum", torch.cumsum),
    ("divide", torch.div),
    ("equal", torch.eq),
    ("fft", torch.fft.fft),
    ("fill", lambda shape, v, dtype=None: torch.full(shape, v, dtype=dtype)),
    ("floordiv", torch.floor_divide),
    ("gather", torch_gather),
    ("greater_equal", torch.ge),
    ("greater", torch.gt),
    ("ifft", torch.fft.ifft),
    ("less_equal", torch.le),
    ("less", torch.lt),
    ("logical_and", torch.logical_and),
    ("logical_not", torch.logical_not),
    ("logical_or", torch.logical_or),
    ("maximum", torch.max),
    ("meshgrid", torch.meshgrid),
    ("minimum", torch.min),
    ("mod", torch.remainder),
    ("multiply", torch.mul),
    ("negative", torch.negative),
    ("not_equal", torch.ne),
    ("one_hot", torch.nn.functional.one_hot),
    ("ones", torch.ones),
    ("pad", F.pad),
    ("pow", torch.pow),
    ("range", torch.arange),
    ("reduce_all", torch.all),
    ("reduce_any", torch.any),
    ("reduce_max", torch_reduce_max),
    ("reduce_min", torch_reduce_min),
    ("reduce_sum", torch_reduce_sum),
    ("right_shift", torch.bitwise_right_shift),
    ("sort", lambda x, axis=None: torch.sort(x, dim=axis or 0)[0]),
    ("subtract", torch.sub),
    ("topk", top_k),
    ("unique", lambda x: torch.unique(x, return_inverse=True)),
    ("unsorted_segment_min", torch_unsorted_segment_min),
    ("unsorted_segment_sum", torch_unsorted_segment_sum),
    ("where", torch.where),
    ("zeros", torch.zeros),
    ("int64", torch.int64),
    ("scalar", to_py_scalar),
    ("flip", lambda input, axis: torch.flip(input, dims=[axis])),
]

for name, function in exported:
    globals()[name] = function

