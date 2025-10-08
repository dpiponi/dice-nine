from functools import reduce
import logging
from dataclasses import dataclass
from typing import ClassVar

from dice9.hash import hash_tensors, conform_to_64
import dice9.backends.numpy_impl as sx
from dice9.algebra import Semiring, Real64, LogReal64, Int64, BigFraction, BigInteger, SemiringProduct, Complex128
import operator


@dataclass(frozen=True)
class Register:
    _number: int

    def __str__(self):
        return f"R({self._number})"

    _next_register: ClassVar[int] = 1

    @staticmethod
    def new():
        num = Register._next_register
        Register._next_register += 1
        return Register(num)


def is_reg(register):
    return isinstance(register, Register)



def flatten_first_two(tensor):
    """
    Convert a tensor of shape [a, b, ...]
    to a tensor of shape [a * b, ...].
    """
    shape = tensor.shape
    head = shape[0] * shape[1]
    tail = shape[2:]
    new_shape = (head,) + tail
    return sx.reshape(tensor, new_shape)


def expand_to(tensor, axis, n):
    """
    Expand a tensor by inserting a new axis at `axis` and
    broadcast it to size `n` along that axis.

    Eg. for axis = 2
    converts shape [a, b, c, ...] to
    shape [a, b, n, ...].
    """
    if not isinstance(tensor, sx.Tensor):
        # This cast is a bug
        tensor = sx.constant(tensor, dtype=sx.int64)
    orig_shape = tuple(tensor.shape)
    tensor = sx.expand_dims(tensor, axis=axis)
    before = orig_shape[:axis]
    after = orig_shape[axis:]
    target_shape = before + (n,) + after
    return sx.broadcast_to(tensor, target_shape)

# def broadcast_up(num_rows, tensor):
#     old_shape = sx.shape(tensor)
#     new_shape = (num_rows,) + sx.old_shape[1:]
#     return sx.broadcast_to(tensor, new_shape)


def dedupe_and_aggregate(tensors, p, hash_fn, semiring: Semiring):
    tensors = list(t for t in tensors if len(t) > 0)

    if len(tensors) == 0:
        return [], p

    num_rows = semiring.len(p)
    if num_rows == 1:
        return [0], p

    logging.debug("Marginalizing %s tensors in %s worlds.", len(tensors), semiring.len(p))

    hashes = hash_fn(tensors)
    unique_hashes, inv_idx = sx.unique(hashes)
    num_hashes = unique_hashes.shape[0]

    batch = hashes.shape[0]
    positions = sx.range(batch, dtype=inv_idx.dtype)
    first_pos = sx.unsorted_segment_min(positions, inv_idx, num_segments=num_hashes)

    p_summed = semiring.segment_sum(values=p, segment_ids=inv_idx, num_segments=num_hashes)

    return first_pos, p_summed


def marginalize(p, kept_vars, semiring: Semiring):
    orig_vars = kept_vars
    # This is wrong xxx
    #kept_vars = [sx.cast(v, sx.int64) for v in kept_vars]
    kept_vars = map(conform_to_64, kept_vars)

    s = semiring
    rep_index, p_marg = dedupe_and_aggregate(kept_vars, p, hash_tensors, s)
    vars_marg = [sx.gather(t, rep_index) for t in orig_vars]

    if not vars_marg:
        p_marg = s.add_reduce(p, keepdims=True)

    return p_marg, vars_marg


class Factor:

    def __init__(self, semiring, p, values):
        self.semiring = semiring
        self.p = p
        self._values = values

    def clone(self):
        return Factor(self.semiring, self.p, dict(self._values))

    def __repr__(self):
        return (f"Factor<p({id(self)}): {self.p.shape}, " +
                ", ".join(f"{k}: {v.shape}" for k, v in self._values.items()) +
                ">")

    def rich(self, names={}, max_rows=6):
        from rich.table import Table
        from rich import box

        reverse_names = {v: k for k, v in names.items()}

        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Probability")
        for k in self._values.keys():
            table.add_column(str(reverse_names.get(k, k)))
        num_rows = self.semiring.len(self.p)

        idx = self.semiring.argsort(self.p)[::-1]

        if num_rows > max_rows:
            for i in range(max_rows - 1):
                row = ([str(self.semiring.as_scalar(self.p[idx[i]]))]
                       + [str(v[idx[i]]) for v in self._values.values()])
                table.add_row(*row)

            row = [f"…{num_rows - max_rows} rows…"]
            for v in self._values.values():
                row = row + ["…"]
            table.add_row(*row)

            row = [str(self.semiring.as_scalar(self.p[0]))]
            for v in self._values.values():
                row = row + [str(v[idx[-1]])]
            table.add_row(*row)

        else:
            for i in range(num_rows):
                row = [str(self.semiring.as_scalar(self.p[idx[i]]))]
                for v in self._values.values():
                    row = row + [str(v[idx[i]])]
                table.add_row(*row)

        return table

    def show(self, names):
        from rich.console import Console
        # from rich.table import Table
        from rich.panel import Panel
        # from rich import box

        console = Console()
        table = self.rich(names)
        console.print(Panel(table, expand=False))

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]

    def __contains__(self, key):
        return key in self._values

    def marginalize(self):
        s = self.semiring
        num_worlds_before = s.len(self.p)
        values = self._values.values()

        p, values = marginalize(self.p, values, s)

        self.p = p
        self._values = dict(zip(self._values.keys(), values))
        logging.debug("Marginalized from %s worlds to %s.", num_worlds_before, s.len(self.p))

    def __mul__(self, other):
        s = self.semiring
        assert s is other.semiring, "mixing semirings is not supported"
        self_len = s.len(self.p)
        other_len = s.len(other.p)
        p_outer = s.kronecker(self.p, other.p)

        # EXPERIMENTAL
        # if other_len == 1 or self_len == 1:
        #     new_vars1 = {
        #         k: flatten_first_two(expand_to(v, axis=1, n=1))
        #         for k, v in self._values.items()
        #     }
        #     new_vars2 = {
        #         k: flatten_first_two(expand_to(v, axis=0, n=1))
        #         for k, v in other._values.items()
        #     }
        #     vars_joined = {**new_vars1, **new_vars2}
        #     return Factor(s, p_outer, vars_joined)

        new_vars1 = {
            k: flatten_first_two(expand_to(v, axis=1, n=other_len))
            for k, v in self._values.items()
        }
        new_vars2 = {
            k: flatten_first_two(expand_to(v, axis=0, n=self_len))
            for k, v in other._values.items()
        }
        vars_joined = {**new_vars1, **new_vars2}
        return Factor(s, p_outer, vars_joined)

    def split(self, condition):
        condition = sx.cast(condition, sx.bool)
        p1, p2 = self.p[condition], self.p[~condition]
        s = self.semiring

        def select(condition):
            return {key: value[condition] for key, value in self._values.items()}

        factor1 = Factor(s, p1, select(condition)) if p1.size else None
        factor2 = Factor(s, p2, select(~condition)) if p2.size else None
        return factor1, factor2

    def semi_split(self, condition):
        condition = sx.cast(condition, sx.bool)
        p = self.p[condition]
        registers = {
            key: value[condition] for key, value in self._values.items()
        }
        return Factor(self.semiring, p, registers)

    def vars(self):
        return self._values.keys()

    # def pretty_print(self):
    #     print_table(self.p, self._values)

    @staticmethod
    def allocate_factor_with_register_with_probability(semiring, new_register, p, value):
        return Factor(semiring, p, {new_register: value})

    def rejoin(self, other, pairs):
        p = self.semiring.concat([self.p, other.p], axis=0)

        values = {}
        for register1, register2, register3 in pairs:
            if register1 in self._values and register2 in other._values:
                values[register3] = sx.concat(
                    [self._values[register1], other._values[register2]], axis=0)
        return Factor(self.semiring, p, values)

    @staticmethod
    def rejoin_factors(handled, pairs, if_factors, orelse_factors):
        if_reduced = reduce(operator.mul, if_factors)
        orelse_reduced = reduce(operator.mul, orelse_factors)

        for register in set(if_reduced._values.keys()) & set(orelse_reduced._values.keys()):
            if register not in handled:
                pairs.append((register, register, register))
                logging.debug("Passing through %s in merge.", register)
                handled.add(register)

        return if_reduced.rejoin(orelse_reduced, pairs)

__all__ = [
    "Semiring", "Real64", "LogReal64", "Int64", "BigFraction", "BigInteger", "SemiringProduct", "Complex128"
            ]
