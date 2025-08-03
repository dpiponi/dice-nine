from dice9.config import sx


def conform_to_64(tensor):
    """
    Cast a tensor into a form suitable for hashing.

    Args:
        tensor: a tensor for hashing.

    Returns:
        The tensor mapped to sx.uint64 injectively.
    """
    if tensor.dtype == sx.bool or tensor.dtype == sx.int32 or tensor.dtype == sx.uint32:
        return sx.cast(tensor, sx.uint64)
    if tensor.dtype == sx.float64 or tensor.dtype == sx.uint64 or tensor.dtype == sx.int64:
        return sx.bitcast(tensor, sx.uint64)
    if tensor.dtype == sx.float32:
        return sx.cast(sx.bitcast(tensor, sx.int32), sx.uint64)

    return tensor

def _pack_tensor(tensor):
    mins = sx.reduce_min(tensor, -2)
    maxs = sx.reduce_max(tensor, -2) + 1
    ranges = maxs - mins
    powers = sx.cumprod_exclusive(ranges)
    total_max = powers[-1] * ranges[-1]
    print(ranges, powers, total_max)
    if total_max >= 2**63:
        raise ValueError(f"{total_max} too big!")

def flatten_tensors(tensors):
    """
    Flatten a list of tensors [t0, t1, ..., tn] so that each
    row [t0[i], ..., tn[i]] is packed into a row of the
    result.
    """
    num_rows = sx.shape(tensors[0])[0]
    blocks = [sx.reshape(t, [num_rows, -1]) for t in tensors]
    result =  sx.concat(blocks, axis=1)
    return result

def _hash_tensor(tensor, prime=7371967656361):
    width = sx.shape(tensor)[1]
    powers = sx.constant(prime, dtype=sx.int64)

    # exclusive cumprod gives [1, p, p^2, ...]
    weights = sx.cumprod_exclusive(sx.fill([width], powers), axis=0)

    tensor64 = sx.cast(tensor, sx.int64)
    acc = sx.reduce_sum(tensor64 * weights, axis=1)  # shape=(batch,)

    inj = sx.check_hash_collision(tensor64, acc)
    if inj:
        raise "not inj"

    return acc

def hash_tensors(tensors: sx.Tensor, prime=73716967656361):
    """
    Hash each individual row of a 2D tensor.
    """

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
