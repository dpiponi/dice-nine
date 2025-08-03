import builtins
import numpy as np

# ---------------------- Scalar and Type Conversion ----------------------

def to_py_scalar(x):
    if isinstance(x, (builtins.int, builtins.float, builtins.bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        else:
            raise ValueError(f"Expected 0-d array, got shape {x.shape}")
    raise TypeError(f"Unsupported type: {type(x)}")


def _to_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(_to_tuple(v) for v in x)
    return x


def numpy_to_tuple(arr):
    if isinstance(arr, np.ndarray):
        return _to_tuple(arr.tolist())
    return _to_tuple(arr)

# ---------------------- Tensor Ops ----------------------

def tf_unstack(x, axis=0):
    return [np.squeeze(a, axis=axis) for a in np.split(x, x.shape[axis], axis=axis)]


def tf_unique(x):
    return np.unique(x, return_inverse=True)


def tf_unsorted_segment_sum(data, segment_ids, num_segments):
    output_shape = (num_segments,) + data.shape[1:]
    result = np.zeros(output_shape, dtype=data.dtype)
    np.add.at(result, segment_ids, data)
    return result


def tf_unsorted_segment_min(data, segment_ids, num_segments):
    if np.issubdtype(data.dtype, np.floating):
        init_val = np.inf
    elif np.issubdtype(data.dtype, np.integer):
        init_val = np.iinfo(data.dtype).max
    else:
        raise TypeError("Unsupported dtype for unsorted_segment_min")

    output = np.full((num_segments,) + data.shape[1:], init_val, dtype=data.dtype)
    np.minimum.at(output, segment_ids, data)
    return output


def tf_cumprod_exclusive(a, axis=0):
    a = np.asarray(a)
    if a.size == 0:
        return np.zeros_like(a)
    shifted = np.roll(a, shift=1, axis=axis)
    shifted[(slice(None),) * axis + (0,)] = 1
    return np.cumprod(shifted, axis=axis)

# ---------------------- Core Functional Utilities ----------------------

def subscript(tensor, idx):
    batch = np.arange(tensor.shape[0])
    return tensor[batch, idx]


def one_hot(index, depth):
    return np.eye(depth, dtype=np.int64)[index]


def bitcast(tensor, dtype):
    return tensor.view(dtype)


def top_k(x, k, axis=-1):
    idx_part = np.argpartition(x, -k, axis=axis)
    topk_idx = np.take(idx_part, range(-k, 0), axis=axis)
    topk_vals = np.take_along_axis(x, topk_idx, axis=axis)
    topk_vals.sort(axis=axis)
    return np.flip(topk_vals, axis=axis), None


def gather(tensor, index):
    return tensor[index]

import operator

def scatter_update_op(x, slices, updates, op=operator.setitem):
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
    M, N = arr.shape
    row_indices = np.repeat(np.arange(M), N)
    vals = arr.ravel()
    mask = (vals >= 0) & (vals < maxlength)
    result = np.zeros((M, maxlength), dtype=np.int64)
    np.add.at(result, (row_indices[mask], vals[mask]), 1)
    return result

# ---------------------- Sampling Utilities ----------------------

def repeat_with_indices(counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(counts)), counts)


def batched_searchsorted(a: np.ndarray, v: np.ndarray, side="left") -> np.ndarray:
    assert a.ndim == 2 and v.ndim == 1 and a.shape[0] == v.shape[0]
    v = v[:, None]
    cmp = v <= a if side == "left" else v < a
    return np.argmax(cmp, axis=1)


def rolls(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    max_val = np.max(x)
    base_sequence = np.arange(max_val)
    mask = base_sequence < x[:, None]
    masked_sequence = np.where(mask, base_sequence, 0)
    return masked_sequence[mask]


def get_sample(ranges: np.ndarray, roll: np.ndarray) -> np.ndarray:
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    cumsums = np.cumsum(group_sizes, axis=1)
    cumsums_excl = np.roll(cumsums, shift=1, axis=1)
    cumsums_excl[:, 0] = 0

    group_index = np.searchsorted(cumsums_excl, roll[:, None], side="right")[:, 0] - 1
    selected_groups = ranges[np.arange(roll.shape[0]), group_index]
    group_offsets = cumsums_excl[np.arange(roll.shape[0]), group_index]
    offset_within_group = roll - group_offsets
    return selected_groups[:, 0] + selected_groups[:, 2] * offset_within_group


def multiroll_helper(ranges: np.ndarray, p: np.ndarray):
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    total_sizes = np.sum(group_sizes, axis=-1)

    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)
    new_ranges = ranges[indices_for_rolls]

    group_sizes = (new_ranges[:, :, 1] - new_ranges[:, :, 0]) // new_ranges[:, :, 2]
    cumsums = np.cumsum(group_sizes, axis=1)
    cumsums_excl = np.roll(cumsums, shift=1, axis=1)
    cumsums_excl[:, 0] = 0

    group_index = batched_searchsorted(cumsums_excl, all_rolls, side="right") - 1
    selected_groups = new_ranges[np.arange(group_index.shape[0]), group_index]
    group_offsets = cumsums_excl[np.arange(group_index.shape[0]), group_index]
    offset_within_group = all_rolls - group_offsets

    sample = selected_groups[:, 0] + selected_groups[:, 2] * offset_within_group
    weights = p[indices_for_rolls] / total_sizes[indices_for_rolls]
    return sample.astype(np.int64), weights, indices_for_rolls


def dd_helper(nv):
    nv = np.asarray(nv, dtype=int)
    B = nv.shape[0]
    if B == 0 or nv.sum() == 0:
        return np.array([], float), np.array([], int), np.array([], int)

    max_nv = nv.max()
    batch_idx = np.arange(B)[:, None].repeat(max_nv, axis=1)
    roll_vals = np.arange(max_nv)[None, :].repeat(B, axis=0)
    mask = roll_vals < nv[:, None]

    indices = batch_idx[mask]
    rolls = roll_vals[mask]
    probs = (1.0 / nv[:, None].astype(float)).repeat(max_nv, axis=1)[mask]

    return probs, rolls, indices

# ---------------------- Hash Collisions ----------------------

def check_hash_collision(keys: np.ndarray, hashes: np.ndarray) -> bool:
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

# ---------------------- Export Map ----------------------

exported = [
    ('Tensor', np.ndarray),
    ('abs', np.abs),
    ('add', np.add),
    ('argmin', np.argmin),
    ('argmax', np.argmax),
    ('argsort', np.argsort),
    ('bitwise_xor', np.bitwise_xor),
    ('broadcast_to', np.broadcast_to),
    ('cast', lambda x, t, *args, **kwargs: x.astype(t)),
    ('complex128', np.complex128),
    ('concat', np.concatenate),
    ('constant', np.array),
    ('cumprod_exclusive', tf_cumprod_exclusive),
    ('cumsum', np.cumsum),
    ('divide', np.divide),
    ('equal', np.equal),
    ('expand_dims', np.expand_dims),
    ('fft', np.fft.fft),
    ('fill', np.full),
    ('float32', np.float32),
    ('float64', np.float64),
    ('floordiv', np.floor_divide),
    ('gather', gather),
    ('greater', np.greater),
    ('greater_equal', np.greater_equal),
    ('ifft', np.fft.ifft),
    ('imag', np.imag),
    ('int32', np.int32),
    ('int64', np.int64),
    ('less', np.less),
    ('less_equal', np.less_equal),
    ('logical_and', np.logical_and),
    ('logical_not', np.logical_not),
    ('logical_or', np.logical_or),
    ('maximum', np.maximum),
    ('meshgrid', np.meshgrid),
    ('minimum', np.minimum),
    ('mod', np.mod),
    ('multiply', np.multiply),
    ('negative', np.negative),
    ('not_equal', np.not_equal),
    ('pad', np.pad),
    ('pow', np.power),
    ('range', np.arange),
    ('rank', lambda x, *args, **kwargs: len(x.shape)),
    ('real', np.real),
    ('reduce_all', np.all),
    ('reduce_any', np.any),
    ('reduce_max', np.max),
    ('reduce_min', np.min),
    ('reduce_sum', np.sum),
    ('reshape', np.reshape),
    ('right_shift', np.right_shift),
    ('shape', np.shape),
    ('sort', np.sort),
    ('stack', np.stack),
    ('subtract', np.subtract),
    ('to_python', numpy_to_tuple),
    ('top_k', top_k),
    ('transpose', np.transpose),
    ('uint32', np.uint32),
    ('uint64', np.uint64),
    ('unique', tf_unique),
    ('unsorted_segment_min', tf_unsorted_segment_min),
    ('unsorted_segment_sum', tf_unsorted_segment_sum),
    ('unstack', tf_unstack),
    ('where', np.where),
    ('zeros', np.zeros),
    ('ones', np.ones),
    ('convert_to_tensor', np.asarray),
    ('to_list', lambda x: x.tolist()),
    ('isin', np.isin),
    ('convert_to_tuple', numpy_to_tuple),
    ('bincount', rowwise_bincount),
    ('arange', np.arange),
    ('scalar', to_py_scalar),
    ('flip', lambda input, axis: np.flip(input, axis))
]

for name, function in exported:
    globals()[name] = function

# Aliases
bool = np.bool_
ptype = np.float64

