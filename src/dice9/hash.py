import logging
import dice9.backends.numpy_impl as sx

def conform_to_64(tensor):
    dtype = tensor.dtype
    if dtype in (sx.bool, sx.int32, sx.uint32):
        return sx.cast(tensor, sx.uint64)
    if dtype in (sx.float64, sx.uint64, sx.int64):
        return sx.bitcast(tensor, sx.uint64)
    if dtype == sx.float32:
        return sx.cast(sx.bitcast(tensor, sx.int32), sx.uint64)
    return tensor

# Unused currently
def _pack_tensor(tensor):
    mins = sx.reduce_min(tensor, -2)
    maxs = sx.reduce_max(tensor, -2) + 1
    ranges = maxs - mins
    powers = sx.cumprod_exclusive(ranges)
    total_max = powers[-1] * ranges[-1]
    if total_max >= 2**63:
        raise ValueError(f"{total_max} too big!")

def flatten_tensors(tensors):
    num_rows = sx.shape(tensors[0])[0]
    blocks = [sx.reshape(t, [num_rows, -1]) for t in tensors]
    result = sx.concat(blocks, axis=1)
    logging.debug("Flattened to size %s.", sx.shape(result))
    return result

def _hash_tensor(tensor, prime=7371967656361):
    width = sx.shape(tensor)[1]
    powers = sx.constant(prime, dtype=sx.int64)
    weights = sx.cumprod_exclusive(sx.fill([width], powers), axis=0)
    tensor64 = sx.cast(tensor, sx.int64)
    acc = sx.reduce_sum(tensor64 * weights, axis=1)
    if sx.check_hash_collision(tensor64, acc):
        raise ValueError("Not an injection")
    return acc

def hash_tensors(tensors: sx.Tensor, prime=73716967656361):
    rows = flatten_tensors(tensors)
    return _hash_tensor(rows, prime)

def check_hash_injective(keys_tensor, hashes):
    sorted_idx = sx.argsort(hashes)
    hashes_sorted = sx.gather(hashes, sorted_idx)
    keys_sorted = sx.gather(keys_tensor, sorted_idx)
    same_hash = sx.equal(hashes_sorted[1:], hashes_sorted[:-1])
    same_keys = sx.reduce_all(sx.equal(keys_sorted[1:], keys_sorted[:-1]), axis=1)
    collision = sx.logical_and(same_hash, sx.logical_not(same_keys))
    return sx.reduce_any(collision)
