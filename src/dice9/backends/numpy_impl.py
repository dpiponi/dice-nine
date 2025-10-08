import builtins
import numpy as np

# =============================================================================
# Scalar / Python interop
# =============================================================================


def to_py_scalar(x):
    """Convert numpy scalar/0-d array (or builtin scalar) to a Python scalar."""
    if isinstance(x, (builtins.int, builtins.float, builtins.bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        raise ValueError(f"Expected 0-d array, got shape {x.shape}")
    if isinstance(x, np.int64):
        return int(x)
    raise TypeError(f"Unsupported type: {type(x)}")


def _to_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(_to_tuple(v) for v in x)
    return to_py_scalar(x)


def numpy_to_tuple(arr):
    """Convert arrays (recursively) to tuples/lists of Python scalars."""
    if isinstance(arr, np.ndarray):
        return _to_tuple(arr.tolist())
    return _to_tuple(arr)


# =============================================================================
# Array/Tensor utilities
# =============================================================================


def tf_unstack(x, axis=0):
    """NumPy equivalent of tf.unstack."""
    return [
        np.squeeze(a, axis=axis) for a in np.split(x, x.shape[axis], axis=axis)
    ]


def tf_unique(x):
    """Return (unique_values, inverse_indices) like tf.unique (vector case)."""
    return np.unique(x, return_inverse=True)


def tf_unsorted_segment_sum(data, segment_ids, num_segments):
    """UnsortedSegmentSum over first dimension."""
    # print("data=", data, segment_ids, num_segments)
    output_shape = (num_segments,) + data.shape[1:]
    result = np.zeros(output_shape, dtype=data.dtype)
    # print("result=", result)
    np.add.at(result, segment_ids, data)
    return result


def tf_unsorted_segment_min(data, segment_ids, num_segments):
    """UnsortedSegmentMin over first dimension."""
    if np.issubdtype(data.dtype, np.floating):
        init_val = np.inf
    elif np.issubdtype(data.dtype, np.integer):
        init_val = np.iinfo(data.dtype).max
    else:
        raise TypeError("Unsupported dtype for unsorted_segment_min")

    out = np.full((num_segments,) + data.shape[1:], init_val, dtype=data.dtype)
    np.minimum.at(out, segment_ids, data)
    return out


def tf_cumprod_exclusive(a, axis=0):
    """Exclusive cumprod along axis (first element is multiplicative identity)."""
    a = np.asarray(a)
    if a.size == 0:
        return np.zeros_like(a)
    shifted = np.roll(a, shift=1, axis=axis)
    slicer = [slice(None)] * a.ndim
    slicer[axis] = 0
    shifted[tuple(slicer)] = 1
    return np.cumprod(shifted, axis=axis)


def subscript(tensor, idx):
    """Row-wise gather: tensor[i, idx[i], ...]."""
    batch = np.arange(tensor.shape[0])
    return tensor[batch, idx]


def one_hot(index, depth):
    """One-hot encode a 1-D index array to shape [len(index), depth]."""
    return np.eye(depth, dtype=np.int64)[index]


def bitcast(tensor, dtype):
    """Reinterpret the underlying bytes with a new dtype (no copy)."""
    return tensor.view(dtype)


def top_k(x, k, axis=-1):
    """
    Return the top-k values along `axis`, sorted descending (indices omitted).
    """
    idx_part = np.argpartition(x, -k, axis=axis)
    topk_idx = np.take(idx_part, range(-k, 0), axis=axis)
    topk_vals = np.take_along_axis(x, topk_idx, axis=axis)
    topk_vals.sort(axis=axis)
    return np.flip(topk_vals, axis=axis), None


def gather(tensor, index):
    """Simple fancy-index gather passthrough."""
    return tensor[index]


# =============================================================================
# Scatter updates (row-wise)
# =============================================================================

import operator


def scatter_update_op(x, slices, updates, op=operator.setitem):
    """
    Apply a point-wise op to x[row, slices] with `updates`.
    For op=setitem, it's assignment; otherwise uses the binary op.
    """
    x_new = x.copy()
    row_indices = np.arange(x.shape[0])
    if op == operator.setitem:
        x_new[row_indices, slices] = updates
    else:
        x_new[row_indices, slices] = op(x_new[row_indices, slices], updates)
    return x_new


def scatter_update(x, slices, updates):
    return scatter_update_op(x, slices, updates, operator.setitem)


def scatter_update_add(x, slices, updates):
    return scatter_update_op(x, slices, updates, operator.add)


def scatter_update_sub(x, slices, updates):
    return scatter_update_op(x, slices, updates, operator.sub)


def scatter_update_multiply(x, slices, updates):
    return scatter_update_op(x, slices, updates, operator.mul)


def rowwise_bincount(arr, maxlength):
    """Row-wise bincount for nonnegative ints in [0, maxlength)."""
    M, N = arr.shape
    row_indices = np.repeat(np.arange(M), N)
    vals = arr.ravel()
    mask = (vals >= 0) & (vals < maxlength)
    result = np.zeros((M, maxlength), dtype=np.int64)
    np.add.at(result, (row_indices[mask], vals[mask]), 1)
    return result


# =============================================================================
# Sampling helpers
# =============================================================================


def repeat_with_indices(counts: np.ndarray) -> np.ndarray:
    """Expand counts into repeated row indices: [0...0, 1...1, ...]."""
    return np.repeat(np.arange(len(counts)), counts)


def batched_searchsorted(a: np.ndarray,
                         v: np.ndarray,
                         side="left") -> np.ndarray:
    """
    Vectorized searchsorted over rows of `a` for each v[i].
    NOTE: This returns 0 if v[i] is greater than all a[i], which differs from
    NumPy's searchsorted that would return a.shape[1]. Use with care.
    """
    assert a.ndim == 2 and v.ndim == 1 and a.shape[0] == v.shape[0]
    v = v[:, None]
    cmp = (v <= a) if side == "left" else (v < a)
    return np.argmax(cmp, axis=1)


def rolls(x: np.ndarray) -> np.ndarray:
    """
    For per-row counts x[i], emit 0..x[i]-1 for each row, flattened.
    """
    x = x.astype(np.int64)
    max_val = np.max(x) if x.size else 0
    base = np.arange(max_val)
    mask = base < x[:, None]
    return np.where(mask, base, 0)[mask]


def get_sample(ranges: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """
    Given per-row 'ranges' [[start, stop, step], ...] segments and a flat 'roll'
    index per row into the concatenated discretization, return selected values.
    """
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    cumsums = np.cumsum(group_sizes, axis=1)
    cumsums_excl = np.roll(cumsums, 1, axis=1)
    cumsums_excl[:, 0] = 0

    group_index = np.searchsorted(cumsums_excl, roll[:, None],
                                  side="right")[:, 0] - 1
    selected_groups = ranges[np.arange(roll.shape[0]), group_index]
    group_offsets = cumsums_excl[np.arange(roll.shape[0]), group_index]
    offset_within_group = roll - group_offsets
    return selected_groups[:, 0] + selected_groups[:, 2] * offset_within_group


# --- uniform per-row sampling “helper” with flat outputs ---


def roll_helper(x: np.ndarray, p: np.ndarray):
    """
    x: [B, M] values; choose each column uniformly per row (M outcomes per row).
    p: [B] row masses; weight assigned equally across that row's outcomes.

    Returns (sample, weights, indices_for_rolls) flattened across rows.
    """
    B, M = x.shape
    total_sizes = np.full(B, M, dtype=np.int64)
    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)
    group_index = all_rolls
    sample = x[indices_for_rolls, group_index].astype(np.int64)
    weights = p[indices_for_rolls] / total_sizes[indices_for_rolls]
    return sample, weights, indices_for_rolls


def roll_helper_weighted(x: np.ndarray, w: np.ndarray, p: np.ndarray):
    """
    Weighted per-row outcomes:
      x: [B, M] values
      w: [B, M] nonnegative weights; P(x[i,j]) ∝ w[i,j] within row i
      p: [B]     row mass

    Returns:
      sample:            flattened values with w>0
      weights:           p[i] * w[i,j] / sum_j w[i,j]
      indices_for_rolls: row index for each emitted outcome
    """
    assert x.ndim == 2 and w.shape == x.shape
    assert p.ndim == 1 and p.shape[0] == x.shape[0]
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative")

    B, M = x.shape
    pos = w > 0
    total_sizes = pos.sum(axis=1).astype(np.int64)

    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)

    # Map k-th positive (per row) → original column
    gs = pos.astype(np.int64)
    cumsums = np.cumsum(gs, axis=1)
    cumsums_excl = np.roll(cumsums, 1, axis=1)
    cumsums_excl[:, 0] = 0
    col_index = batched_searchsorted(cumsums_excl, all_rolls, side="right") - 1

    sample = x[indices_for_rolls, col_index].astype(np.int64)

    row_sums = (w * pos).sum(axis=1)
    numer = w[indices_for_rolls, col_index] * p[indices_for_rolls]
    denom = row_sums[indices_for_rolls]
    weights = numer / denom

    return sample, weights, indices_for_rolls


def new_multiroll_helper(semiring, ranges: np.ndarray, p, mode="prob"):
    first, last, weight = np.moveaxis(ranges, -1, 0)
    group_sizes = last - first + 1
    total_sizes = np.sum(group_sizes, axis=-1)
    total_weight = np.sum(weight * group_sizes, -1)

    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)
    new_ranges = ranges[indices_for_rolls]
    new_first, new_last, new_weight = np.moveaxis(new_ranges, -1, 0)

    group_sizes = new_last - new_first + 1
    cumsums = np.cumsum(group_sizes, axis=1)
    cumsums_excl = np.roll(cumsums, 1, axis=1)
    cumsums_excl[:, 0] = 0

    group_index = batched_searchsorted(cumsums_excl, all_rolls,
                                       side="right") - 1
    selected_groups = new_ranges[np.arange(group_index.shape[0]), group_index]
    group_offsets = cumsums_excl[np.arange(group_index.shape[0]), group_index]
    offset_within_group = all_rolls - group_offsets

    sample = selected_groups[:, 0] + offset_within_group
    weights1 = semiring.promote(new_weight[np.arange(group_index.shape[0]),
                                           group_index])
    total_weights0 = total_weight[indices_for_rolls]
    total_weights1 = semiring.promote(total_weights0)
    temp = semiring.mul(p[indices_for_rolls], weights1)
    weights = semiring.divide(temp, total_weights1) if mode == "prob" else temp
    return sample.astype(np.int64), weights, indices_for_rolls


def dd_helper(nv):
    """
    Given nv[i] outcomes in row i, return:
      probs  = 1/nv[i] for each outcome (flattened),
      rolls  = per-row 0..nv[i]-1 indices (flattened),
      indices= row indices (flattened).
    """
    nv = np.asarray(nv, dtype=int)
    B = nv.shape[0]
    if B == 0 or nv.sum() == 0:
        return np.array([], float), np.array([], int), np.array([], int)

    max_nv = nv.max()
    batch_idx = np.arange(B)[:, None].repeat(max_nv, axis=1)
    roll_vals = np.arange(max_nv)[None, :].repeat(B, axis=0)
    mask = roll_vals < nv[:, None]

    indices = batch_idx[mask]
    rolls_out = roll_vals[mask]
    probs = (1.0 / nv[:, None].astype(float)).repeat(max_nv, axis=1)[mask]

    return probs, rolls_out, indices


# =============================================================================
# Hash collision diagnostics
# =============================================================================


def check_hash_collision(keys: np.ndarray, hashes: np.ndarray) -> bool:
    """
    Check for collisions where hashes match but keys differ (row-wise equality).
    Prints the first such collision for debugging; returns True if any found.
    """
    sorted_idx = np.argsort(hashes)
    hashes_sorted = hashes[sorted_idx]
    keys_sorted = keys[sorted_idx]

    same_hash = hashes_sorted[1:] == hashes_sorted[:-1]
    same_keys = np.all(keys_sorted[1:] == keys_sorted[:-1], axis=1)
    collision = same_hash & (~same_keys)

    if np.any(collision):
        idx = np.where(collision)[0][0]
        print("!!!!!!!!!!!!!!!!!", collision)
        print(f"idx={idx}")
        print(f"{keys_sorted[idx]} -> {hashes_sorted[idx]}")
        print(f"{keys_sorted[idx+1]} -> {hashes_sorted[idx+1]}")
        return True
    return False


# =============================================================================
# Log-sum-exp by unsorted segments
# =============================================================================


def np_unsorted_segment_logsumexp(values, segment_ids, num_segments):
    """
    Compute logsumexp within each segment id over the first axis.
    values:      [N, *rest]
    segment_ids: [N]
    returns:     [num_segments, *rest]
    """
    values = np.asarray(values)
    segment_ids = np.asarray(segment_ids)
    S = int(num_segments)

    rest = values.shape[1:]
    # 1) max per segment for numerical stability
    max_per_seg = np.full((S, *rest), -np.inf, dtype=values.dtype)
    np.maximum.at(max_per_seg, (segment_ids, *[slice(None)] * len(rest)),
                  values)

    # 2) sum exp(values - max)
    gathered_max = max_per_seg[segment_ids]  # [N, *rest]
    shifted = np.exp(values - gathered_max)  # [N, *rest]
    sum_exp = np.zeros((S, *rest), dtype=values.dtype)
    np.add.at(sum_exp, (segment_ids, *[slice(None)] * len(rest)), shifted)

    # 3) logsumexp = log(sum_exp) + max
    out = np.log(sum_exp) + max_per_seg  # [S, *rest]
    return out


# Linear time algorithm to fill table of modular inverses.
def inv_tables(primes):
    p_arr = np.atleast_1d(np.array(primes, dtype=np.int64))  # shape (K,)
    K = p_arr.shape[0]
    max_p = int(p_arr.max())

    inv = np.zeros((K, max_p), dtype=np.int64)
    inv[:, 1] = 1

    rows = np.arange(K)
    for i in range(2, max_p):
        r = p_arr % i
        q = p_arr // i
        inv[:, i] = (p_arr - (q * inv[rows, r] % p_arr)) % p_arr

    return inv

import numpy as np

def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes over which the sum is taken.
    keepdims : bool, optional
        If True, retains reduced dimensions with length 1.

    Returns
    -------
    res : ndarray
        The result, log(sum(exp(x))).
    """
    x = np.asarray(x)
    # Find max along the reduction axis for numerical stability
    xmax = np.max(x, axis=axis, keepdims=True)
    # Avoid overflow when all entries are -inf
    xmax = np.where(np.isfinite(xmax), xmax, 0)
    # Compute the shifted exponentials
    shifted = np.exp(x - xmax)
    # Sum them, take the log, and add back the shift
    s = np.sum(shifted, axis=axis, keepdims=True)
    out = np.log(s) + xmax
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

# =============================================================================
# Public aliases (formerly 'exported' table)
# =============================================================================

# Types
Tensor = np.ndarray

# Numpy ops & small shims
abs = np.abs
add = np.add
argmin = np.argmin
argmax = np.argmax
argsort = np.argsort
arange = np.arange
broadcast_to = np.broadcast_to
cast = lambda x, t, *args, **kwargs: np.asarray(x).astype(t)
complex128 = np.complex128
concat = np.concatenate
constant = np.array
cumprod_exclusive = tf_cumprod_exclusive
cumsum = np.cumsum
divide = np.divide
equal = np.equal
expand_dims = np.expand_dims
fft = np.fft.fft
fill = np.full
float32 = np.float32
float64 = np.float64
floordiv = np.floor_divide
# gather_alias = gather
gather = gather
greater = np.greater
greater_equal = np.greater_equal
ifft = np.fft.ifft
imag = np.imag
int32 = np.int32
int64 = np.int64
less = np.less
less_equal = np.less_equal
logical_and = np.logical_and
logical_not = np.logical_not
logical_or = np.logical_or
maximum = np.maximum
meshgrid = np.meshgrid
minimum = np.minimum
mod = np.mod
multiply = np.multiply
negative = np.negative
not_equal = np.not_equal
pad = np.pad
pow = np.power
range_ = np.arange  # 'range' shadowing is often undesirable; keep both if needed
range = np.arange
rank = lambda x, *args, **kwargs: len(np.asarray(x).shape)
real = np.real
reduce_all = np.all
reduce_any = np.any
reduce_max = np.max
reduce_min = np.min
reduce_sum = np.sum
reshape = np.reshape
right_shift = np.right_shift
shape = np.shape
sign = np.sign
sort = np.sort
stack = np.stack
subtract = np.subtract
to_python = numpy_to_tuple
# top_k_alias = top_k
top_k = top_k
transpose = np.transpose
uint32 = np.uint32
uint64 = np.uint64
unique = tf_unique
unsorted_segment_min = tf_unsorted_segment_min
unsorted_segment_sum = tf_unsorted_segment_sum
unstack = tf_unstack
where = np.where
zeros = np.zeros
ones = np.ones
convert_to_tensor = np.asarray
to_list = lambda x: x.tolist()
isin = np.isin
isnotin = lambda a, b: np.isin(a, b, invert=True)
convert_to_tuple = numpy_to_tuple
bincount = rowwise_bincount
flip = np.flip
bitwise_and = np.bitwise_and
bitwise_xor = np.bitwise_xor
bitwise_or = np.bitwise_or
invert = np.invert
left_shift = np.left_shift
unsorted_segment_logsumexp = np_unsorted_segment_logsumexp
reduce_logsumexp = logsumexp
logaddexp = np.logaddexp
convert_to_tensor = np.asarray
moveaxis = np.moveaxis

# Aliases
bool = np.bool_
# ptype = np.float64
