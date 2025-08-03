# tensorflow.py

# --- Environment setup to suppress TensorFlow logs ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"

# --- Imports ---
import builtins
import logging
import tensorflow as tf

# --- Logging ---
tf.get_logger().setLevel(logging.ERROR)
# tf.config.run_functions_eagerly(True)

# --- Scalar conversion utility ---
def to_py_scalar(x):
    if isinstance(x, (builtins.int, builtins.float, builtins.bool)):
        return x
    if isinstance(x, tf.Tensor):
        if not tf.rank(x) == 0:
            raise ValueError(f"Expected scalar tensor, got shape {x.shape}")
        val = x.numpy()
        if isinstance(val, (builtins.int, builtins.float, builtins.bool)):
            return val
        if hasattr(val, "item"):
            return val.item()
        return val
    raise TypeError(f"Unsupported type: {type(x)}")


# --- Tensor conversion utilities ---
def _to_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(_to_tuple(v) for v in x)
    else:
        return x

def numpy_to_tuple(arr):
    import numpy as np
    return _to_tuple(arr.tolist())

def tensorflow_to_tuple(tensor):
    if isinstance(tensor, tf.Tensor):
        return numpy_to_tuple(tensor.numpy())
    return _to_tuple(tensor)


# --- Backend utility functions ---
def subscript(tensor, idx):
    axis_size = tf.shape(tensor)[1]
    idx = tf.where(idx < 0, idx + tf.cast(axis_size, tf.int64), idx)
    return tf.gather(tensor, idx, axis=1, batch_dims=1)

def one_hot(index, depth):
    return tf.one_hot(index, depth, dtype=tf.int64)

def check_hash_collision(keys_tensor, hashes):
    sorted_idx = tf.argsort(hashes)
    hashes_sorted = tf.gather(hashes, sorted_idx)
    keys_sorted = tf.gather(keys_tensor, sorted_idx)
    same_hash = tf.equal(hashes_sorted[1:], hashes_sorted[:-1])
    same_keys = tf.reduce_all(tf.equal(keys_sorted[1:], keys_sorted[:-1]), axis=1)
    collision = tf.logical_and(same_hash, tf.logical_not(same_keys))
    if tf.reduce_any(collision):
        print("!!!!!!!!!!!!!!!!!", collision)
        idx = tf.where(collision)[0, 0]
        print(f"idx={idx}")
        print(f"{keys_sorted[idx]} -> {hashes_sorted[idx]}")
        print(f"{keys_sorted[idx+1]} -> {hashes_sorted[idx+1]}")
    return tf.reduce_any(collision)

def tf_isin(elements, test_elements, assume_unique=False, invert=False):
    elements = tf.convert_to_tensor(elements)
    test_elems = tf.convert_to_tensor(test_elements)
    test_elems = tf.reshape(test_elems, [-1])
    matches = tf.equal(tf.expand_dims(elements, -1), test_elems)
    isin_mask = tf.reduce_any(matches, axis=-1)
    return tf.logical_not(isin_mask) if invert else isin_mask


# --- Sampling utilities ---
def dd_helper(nv):
    indices = tf.repeat(tf.range(nv.shape[0]), nv)
    probs = tf.repeat(tf.constant(1, tf.float64) / tf.cast(nv, tf.float64), nv)
    rolls = tf.ragged.range(nv).flat_values
    return probs, rolls, indices

def rolls(x):
    x = tf.cast(x, tf.int64)
    max_val = tf.cast(tf.reduce_max(x), tf.int64)
    base_sequence = tf.range(max_val, dtype=tf.int64)
    mask = base_sequence[None, :] < tf.cast(x, tf.int64)[:, None]
    tiled_sequence = tf.tile(base_sequence[None, :], [tf.shape(x)[0], 1])
    masked_sequence = tf.where(
        mask, tiled_sequence, tf.zeros_like(tiled_sequence, dtype=tf.int64)
    )
    return tf.boolean_mask(masked_sequence, mask)

def repeat_with_indices(counts):
    indices = tf.range(tf.shape(counts)[0])
    return tf.repeat(indices, counts)

def get_sample(ranges, roll):
    group_sizes = (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2]
    cumsums = tf.cumsum(group_sizes, axis=1, exclusive=True)
    group_index = tf.searchsorted(cumsums, roll[:, None], side="right")[:, 0] - 1
    batch_indices = tf.range(tf.shape(group_index)[0])
    gather_indices = tf.stack([batch_indices, group_index], axis=1)
    selected_groups = tf.gather_nd(ranges, gather_indices)
    group_offsets = tf.gather_nd(cumsums, gather_indices)
    offset_within_group = roll - group_offsets
    sample = selected_groups[:, 0] + selected_groups[:, 2] * offset_within_group
    return sample

def multiroll_helper(ranges, p):
    total_sizes = tf.reduce_sum(
        (ranges[:, :, 1] - ranges[:, :, 0]) // ranges[:, :, 2], -1
    )
    all_rolls = rolls(total_sizes)
    indices_for_rolls = repeat_with_indices(total_sizes)
    new_ranges = tf.gather(ranges, indices_for_rolls)
    group_sizes = (new_ranges[:, :, 1] - new_ranges[:, :, 0]) // new_ranges[:, :, 2]
    cumsums = tf.cumsum(group_sizes, axis=1, exclusive=True)
    roll_indices = tf.cast(all_rolls[:, None], tf.int64)
    group_index = tf.searchsorted(cumsums, roll_indices, side="right")[:, 0] - 1
    batch_indices = tf.range(tf.shape(group_index)[0])
    gather_indices = tf.stack([batch_indices, group_index], axis=1)
    selected_groups = tf.gather_nd(new_ranges, gather_indices)
    group_offsets = tf.gather_nd(cumsums, gather_indices)
    offset_within_group = all_rolls - tf.cast(group_offsets, tf.int64)
    sample = tf.cast(selected_groups[:, 0], tf.int64) + tf.cast(
        selected_groups[:, 2], tf.int64
    ) * tf.cast(offset_within_group, tf.int64)
    return (
        sample,
        tf.gather(p / tf.cast(total_sizes, tf.float64), indices_for_rolls),
        indices_for_rolls,
    )

def scatter_update_op(x, slices, updates, op=None):
    num_worlds = tf.shape(x)[0]

    if isinstance(slices, int):
        slices = tf.fill([num_worlds], slices)
    elif isinstance(slices, tf.Tensor) and tf.rank(slices) == 0:
        slices = tf.fill([num_worlds], slices)

    row_indices = tf.range(num_worlds, dtype=slices.dtype)
    indices = tf.stack([row_indices, slices], axis=1)

    x_new = tf.identity(x)

    if op is None:
        return tf.tensor_scatter_nd_update(x_new, indices, updates)
    else:
        old_values = tf.gather_nd(x_new, indices)
        new_values = op(old_values, updates)
        return tf.tensor_scatter_nd_update(x_new, indices, new_values)

def scatter_update(x, slices, updates):
    return scatter_update_op(x, slices, updates, op=None)

def scatter_update_add(x, slices, updates):
    return scatter_update_op(x, slices, updates, tf.add)

def scatter_update_sub(x, slices, updates):
    return scatter_update_op(x, slices, updates, tf.subtract)

def scatter_update_multiply(x, slices, updates):
    return scatter_update_op(x, slices, updates, tf.multiply)



# --- Type aliases ---
bool = tf.bool
ptype = tf.float64

# --- Backend operation aliases and exports ---
reshape = tf.reshape
abs = tf.abs
concat = lambda tensors, axis: tf.concat(tensors, axis)
det = lambda x: tf.cast(tf.round(tf.linalg.det(tf.cast(x, tf.float64))), tf.int64)
meshgrid = tf.meshgrid

exported = [
    ("where", tf.where),
    ("argmin", tf.argmin),
    ("argmax", tf.argmax),
    ("rank", tf.rank),
    ("transpose", tf.transpose),
    ("broadcast_to", tf.broadcast_to),
    ("range", tf.range),
    ("constant", tf.constant),
    ("reduce_sum", tf.reduce_sum),
    ("shape", tf.shape),
    ("expand_dims", tf.expand_dims),
    ("cumprod_exclusive", lambda a, *args, **kwargs: tf.math.cumprod(a, exclusive=True)),
    ("fill", tf.fill),
    ("Tensor", tf.Tensor),
    ("uint64", tf.uint64),
    ("uint32", tf.uint32),
    ("int64", tf.int64),
    ("int32", tf.int32),
    ("float64", tf.float64),
    ("float32", tf.float32),
    ("complex128", tf.complex128),
    ("complex64", tf.complex64),
    ("cast", tf.cast),
    ("pad", tf.pad),
    ("fft", tf.signal.fft),
    ("ifft", tf.signal.ifft),
    ("pow", tf.pow),
    ("real", tf.math.real),
    ("imag", tf.math.imag),
    ("bitwise_xor", tf.bitwise.bitwise_xor),
    ("right_shift", tf.bitwise.right_shift),
    ("unsorted_segment_min", tf.math.unsorted_segment_min),
    ("unsorted_segment_sum", tf.math.unsorted_segment_sum),
    ("unique", tf.unique),
    ("argsort", tf.argsort),
    ("gather", tf.gather),
    ("equal", tf.equal),
    ("reduce_all", tf.reduce_all),
    ("reduce_any", tf.reduce_any),
    ("logical_and", tf.logical_and),
    ("logical_or", tf.logical_or),
    ("logical_not", tf.logical_not),
    ("sort", tf.sort),
    ("stack", tf.stack),
    ("unstack", tf.unstack),
    ("add", tf.add),
    ("mod", tf.math.floormod),
    ("subtract", tf.subtract),
    ("multiply", tf.multiply),
    ("divide", tf.divide),
    ("not_equal", tf.not_equal),
    ("less", tf.less),
    ("floordiv", tf.math.floordiv),
    ("greater", tf.greater),
    ("maximum", tf.maximum),
    ("minimum", tf.minimum),
    ("to_python", tensorflow_to_tuple),
    ("bitcast", tf.bitcast),
    ("top_k", tf.math.top_k),
    ("reduce_min", tf.reduce_min),
    ("reduce_max", tf.reduce_max),
    ("cumsum", tf.cumsum),
    ("greater_equal", tf.greater_equal),
    ("less_equal", tf.less_equal),
    ("negative", tf.negative),
    ("convert_to_tensor", tf.convert_to_tensor),
    ("zeros", tf.zeros),
    ("ones", tf.ones),
    ("to_list", lambda x: x.numpy().tolist()),
    ("isin", tf_isin),
    ("convert_to_tuple", tensorflow_to_tuple),
    ("bincount", lambda arr, maxlength: tf.math.bincount(
        arr, maxlength=maxlength, minlength=maxlength, dtype=tf.int64, axis=-1)),
    ("arange", tf.range),
    ("one_hot", one_hot),
    ("scalar", to_py_scalar),
    ("flip", lambda input, axis: tf.reverse(input, axis=[axis]))
]

for name, func in exported:
    globals()[name] = func

# --- Override module name ---
__name__ = "tf"

