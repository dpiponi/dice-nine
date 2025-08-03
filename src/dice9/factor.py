import pandas as pd
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

from dice9.config import sx
import dice9.config as config
from dice9.hash import hash_tensors

constant_one = sx.constant([1.0], dtype=sx.ptype)


next_register = 20000

class Register:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return "register:" + str(self.number)

    def __eq__(self, other):
        if isinstance(other, Register):
            return self.number == other.number
        return NotImplemented

    def __hash__(self):
        return hash(self.number)

def check_is_reg(register):
    assert isinstance(register, Register)

def new_register_name():
    import sys
    global next_register
    new_register = next_register
    next_register += 1

    if logger.isEnabledFor(logging.DEBUG):
        caller_of_caller = sys._getframe(2)
        logging.debug(f"register {new_register} created at line {caller_of_caller.f_lineno!r} "
                      f"in {caller_of_caller.f_code.co_filename!r}")

    return Register(new_register)

        
def is_reg(register):
    return isinstance(register, Register)


def to_numpy(x):
    #import numpy as np
    #"""Convert a TensorFlow tensor to numpy, leave other types alone."""
    #if isinstance(x, sx.Tensor):
    #    return x.numpy()
    #else:
    #    return x
    return x
    
def print_table(p, values):
    """
    Print a table with the first column 'p' and additional columns
    from `values`. All vectors must have the same length.
    """
    # Convert p to a numpy-friendly array
    p_arr = to_numpy(p)

    # Build a dict for DataFrame construction
    data = {"p": p_arr}
    for name, col in values.items():
        data[name] = to_numpy(col)

    # Construct and print
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    

def is_reg(register):
    #return True
    return isinstance(register, Register)


def flatten_first_two(tensor):
    """
    Flattens first two axes of a rank >=2 tensor.
    """
    shape = tensor.shape
    head = shape[0] * shape[1]
    tail = shape[2:]
    new_shape = (head,) + tail
    return sx.reshape(tensor, new_shape)


def expand_to(tensor, axis, n):
    """
    Inserts a new axis at `axis` in `tensor` and then 'repeats'
    each slice along that axis `n` times via broadcasting.

    E.g. if tensor.shape == [A, B, C] and axis==1, n==4,
    then result.shape == [A, 4, B, C].
    """
    if not isinstance(tensor, sx.Tensor):
        tensor = sx.constant(tensor, dtype=sx.int64)
    orig_shape = tuple(tensor.shape)
    tensor = sx.expand_dims(tensor, axis=axis)  # shape [..., 1, ...]

    before = orig_shape[:axis]
    after = orig_shape[axis:]
    target_shape = before + (n,) + after

    return sx.broadcast_to(tensor, target_shape)


def dedupe_and_aggregate(tensors, p, hash_fn):
    """
    * tensors: list of [batch, …] tensors
    * p:       [batch] tensor of probabilities
    * hash_fn: callable(list of tensors) → [batch] tensor of integer hashes

    Returns:
      * deduped: list of tensors each of shape [num_hashes, …]
      * p_summed:  tensor of shape [num_hashes] where
         p_summed[k] = sum of all p[i]
         s.t. hash[i]==unique_hashes[k]
    """
    if len(tensors) == 0:
        return tensors, p
    if sx.shape(tensors[0])[0] == 0:
        return tensors, p

    hashes = hash_fn(tensors)

    unique_hashes, inv_idx = sx.unique(hashes)
    num_hashes = unique_hashes.shape[0]

    batch = hashes.shape[0]
    positions = sx.range(batch, dtype=inv_idx.dtype)
    first_pos = sx.unsorted_segment_min(
    	positions, inv_idx, num_segments=num_hashes)

    p_summed = sx.unsorted_segment_sum(
    	p, inv_idx, num_segments=num_hashes)

    deduped = [sx.gather(t, first_pos) for t in tensors]

    return deduped, p_summed


marginalize_count = 0

# Project onto kept_vars
def marginalize(p, kept_vars):
    global marginalize_count
    marginalize_count += 1
    orig_types = [v.dtype for v in kept_vars]
    kept_vars = [sx.cast(v, sx.int64) for v in kept_vars]

    vars_marg, p_marg = dedupe_and_aggregate(
    	kept_vars, p, hash_tensors)

    vars_marg = [
    	sx.cast(v, orig)
    	for v, orig in zip(vars_marg, orig_types)]

    if not vars_marg:
        p_marg = sx.reduce_sum(p, keepdims=True)

    return p_marg, vars_marg


@dataclass
class Factor:
    p: sx.Tensor
    _values: dict

    def __init__(self, p, values):
        assert p.dtype == sx.ptype
        self.p = p
        self._values = values

    def clone(self):
        return Factor(self.p, dict(self._values))

    def chop(self, eps=0):
        mask = self.p > eps
        self.p = self.p[mask]
        self._values = {key: value[mask] for key, value in self._values.items()}

    def __repr__(self):
        return (
        	f"Factor<p: {self.p.shape}, " +
        	", ".join(
            f"{k}: {v.shape}" for k, v in self._values.items()) +
            ">")

    def is_trivial(self):
        return (len(self._values) == 0 and sx.reduce_sum(self.p) == 1.0)

    def copy(self):
        return Factor(self.p, self._values.copy())

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        check_is_reg(key)
        del self._values[key]

    def __contains__(self, key):
        return key in self._values

    def marginalize(self):
        keys = [key for key, _ in self._values.items()]
        values = [value for _, value in self._values.items()]
        p, values = marginalize(self.p, values)
        self.p = p
        self._values = dict(zip(keys, values))
        return self

    def __mul__(self, other):
        p = self.p[:, None] * other.p[None, :]
        shape = p.shape

        new_vars1 = {
            k: flatten_first_two(expand_to(v, axis=1, n=shape[1]))
            for k, v in self._values.items()
        }
        new_vars2 = {
            k: flatten_first_two(expand_to(v, axis=0, n=shape[0]))
            for k, v in other._values.items()
        }

        vars_joined = {**new_vars1, **new_vars2}
        return Factor(sx.reshape(p, [-1]), vars_joined)

    def rejoin(self, other, pairs):
        p = sx.concat([self.p, other.p], axis=0)

        values = {}
        # Reusing register names from first factor but should maybe consider
        # fresh names.
        for register1, register2, register3 in pairs:
            # We only need to rejoin variables in the newly built factor.
            if register1 in self._values and register2 in other._values:
                #values[register1] = sx.concat(
                values[register3] = sx.concat(
                    [self._values[register1], other._values[register2]],
                    axis=0)
        return Factor(p, values)

    # "Deep" as new tensors and new dict.
    # Splits on condition.
    # Condition is a register.
    # Condition must be within the factor.
    # Note we have registers with same name but in different factors.
    # Those factors should always be kept in separate environments.
    def split(self, condition):
        condition = sx.cast(condition, sx.bool)
        p1, p2 = self.p[condition], self.p[~condition]
        factor1 = None
        factor2 = None
        if len(p1):
            registers1 = {key: value[condition] for key, value in self._values.items()}
            factor1 = Factor(p1, registers1)
        if len(p2):
            registers2 = {key: value[~condition] for key, value in self._values.items()}
            factor2 = Factor(p2, registers2)
        return factor1, factor2

    def semi_split(self, condition):
        condition = sx.cast(condition, sx.bool)
        p = self.p[condition]
        registers = {key: value[condition] for key, value in self._values.items()}
        return Factor(p, registers)

    def vars(self):
        return self._values.keys()

    def pretty_print(self):
        print_table(self.p, self._values)

    @staticmethod
    def allocate_factor_with_register_with_probability(new_register, p, value):
        check_is_reg(new_register)
        return Factor(p, {new_register: value})

    @staticmethod
    def allocate_factor_with_register(register_name, value):
        check_is_reg(register_name)
        return Factor(
            constant_one, {register_name: sx.constant([value], dtype=sx.int64)}
        )
