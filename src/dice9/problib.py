import builtins
import types
import ast
import dice9 as d9
import logging
import dice9.backends.numpy_impl as sx
from dice9.exceptions import InterpreterError


# def is_reg(register):
#     return isinstance(register, d9.Register)


def reshape_impl(interpreter, context, tensor_register, shape_register):
    value = interpreter.move_definite_value(shape_register)
    shape = (-1,) + sx.convert_to_tuple(value)
    return interpreter.un_op(tensor_register, lambda x: sx.reshape(x, shape))


def reshape(array, shape):
    r"""Reshapes an array.

Given `array`, this operation returns a new array that has the same
values as `array` in the same order, except with a new shape given by
`shape`.

>>>  @d9.dist
>>>  def main():
>>>      x = [d(2), d(2), d(2), d(2)]
>>>      return reshape(x, [2, 2])
>>>
>>>  print(main())
{((1, 1), (2, 2)): np.float64(0.0625), ... }

Args:
  array: An array.
  shape: An array.
    Defines the shape of the output array.

Returns:
  An `array`. Has the same type as `array`.
"""
    return __inline_impl__(reshape_impl, array, shape)


def abs_impl(interpreter, context, x_register):
    return interpreter.un_op(x_register, sx.abs)


def abs(x):
    return __inline_impl__(abs_impl, x)


def arange_impl(interpreter, context, start_register, limit_register, delta_register):
    start = interpreter.move_definite_value(start_register)
    limit = interpreter.move_definite_value(limit_register)
    delta = interpreter.move_definite_value(delta_register)
    return interpreter.env.allocate_register_with_definite_value(
        sx.arange(start, limit, delta)
    )


def arange(start, limit, delta):
    return __inline_impl__(arange_impl, start, limit, delta)


def sign_impl(interpreter, context, value_register):
    return interpreter.un_op(value_register, lambda x: sx.sign(x))


def sign(value):
    return __inline_impl__(sign_impl, value)


def int_impl(interpreter, context, value_register):
    return interpreter.un_op(value_register, lambda x: sx.cast(x, sx.int64))


def int(value):
    return __inline_impl__(int_impl, value)


def impl_axis_maker(op):
    def impl(interpreter, context, input_register, axis_register):
        axis = interpreter.move_definite_value(axis_register)
        return interpreter.un_op(
            input_register, lambda x: op(x, axis=d9.lift_axis(axis))
        )

    return impl


# Flip needs to work with bools too.
flip_impl = impl_axis_maker(lambda x, axis: sx.flip(x, axis=axis))

argsort_impl = impl_axis_maker(lambda x, axis: sx.argsort(x, axis=axis))
argmin_impl = impl_axis_maker(lambda x, axis: sx.argmin(x, axis=axis))
argmax_impl = impl_axis_maker(lambda x, axis: sx.argmax(x, axis=axis))

reduce_all_impl = impl_axis_maker(lambda x, axis: sx.reduce_all(x, axis=axis))
reduce_any_impl = impl_axis_maker(lambda x, axis: sx.reduce_any(x, axis=axis))
reduce_min_impl = impl_axis_maker(lambda x, axis: sx.reduce_min(x, axis=axis))
reduce_max_impl = impl_axis_maker(lambda x, axis: sx.reduce_max(x, axis=axis))
reduce_sum_impl = impl_axis_maker(lambda x, axis: sx.reduce_sum(x, axis=axis))


def reduce_all(input, axis=-1):
    return __inline_impl__(reduce_all_impl, input, axis)


def reduce_any(input, axis=-1):
    return __inline_impl__(reduce_any_impl, input, axis)


def reduce_min(input, axis=-1):
    return __inline_impl__(reduce_min_impl, input, axis)


def reduce_max(input, axis=-1):
    return __inline_impl__(reduce_max_impl, input, axis)


def reduce_sum(input, axis=-1):
    return __inline_impl__(reduce_sum_impl, input, axis)


def one_hot_impl(interpreter, context, index_register, depth_register):
    depth = interpreter.move_definite_value(depth_register)
    return interpreter.un_op(index_register, lambda x: sx.one_hot(x, depth))


def one_hot(tensor, depth):
    return __inline_impl__(one_hot_impl, tensor, depth)


def zeros_impl(interpreter, context, shape_register):
    shape = interpreter.move_definite_value(shape_register)
    # del interpreter.env[shape_register]

    zero_array = sx.zeros(sx.to_python(shape), dtype=sx.int64)

    return interpreter.env.allocate_register_with_definite_value(zero_array)


def zeros(shape):
    return __inline_impl__(zeros_impl, shape)


def flip(input, axis=-1):
    return __inline_impl__(flip_impl, move(input), move(axis))


def argmin(values, axis=-1):
    return __inline_impl__(argmin_impl, values, axis)


def argmax(values, axis=-1):
    return __inline_impl__(argmax_impl, move(values), move(axis))


def argsort(values, axis=-1):
    return __inline_impl__(argsort_impl, values, axis)


def bincount_impl(interpreter, context, arr_register, maxlength_register):
    maxlength = interpreter.move_definite_value(maxlength_register)
    return interpreter.un_op(arr_register, lambda x: sx.bincount(x, maxlength))


def bincount(arr, maxlength):
    return __inline_impl__(bincount_impl, arr, maxlength)


def top_k_impl(interpreter, context, input_register, k_register):
    k = interpreter.move_definite_value(k_register)
    return interpreter.un_op(input_register, lambda x: sx.top_k(x, k)[0])


def top_k(input, k):
    return __inline_impl__(top_k_impl, input, k)


def sort_impl(interpreter, context, values_register, axis_register):
    axis = interpreter.move_definite_value(axis_register)
    return interpreter.un_op(
        values_register,
        lambda x: sx.sort(x, axis=d9.lift_axis(axis)),
    )


def sort(values, axis=-1):
    return __inline_impl__(sort_impl, values, axis)


def ones_impl(interpreter, context, shape_register):
    shape = interpreter.move_definite_value(shape_register)

    one_array = sx.ones(sx.to_python(shape), dtype=sx.int64)

    return interpreter.env.allocate_register_with_definite_value(one_array)


def ones(shape):
    return __inline_impl__(ones_impl, shape)


def range_impl(interpreter, context, args):
    definite_args = [interpreter.move_definite_value(arg) for arg in args]
    return builtins.range(*definite_args)


def range(*args):
    return __inline_impl__(range_impl, args)


def list_impl(interpreter, context, args):
    def outer_product(xs):
        s = sx.stack(xs)
        rank = sx.rank(s)
        return sx.transpose(s, [1, 0] + builtins.list(builtins.range(2, rank)))

    gens = args
    element_registers = [g for gen in gens for g in builtins.list(gen)]
    return interpreter.lift(lambda *x: outer_product(x))(*element_registers)


def list(*args):
    return __inline_impl__(list_impl, move(args))


def sum_impl(interpreter, context, args):
    total_register = None
    arg_register = args[0]
    if builtins.len(args) == 2:
        total_register = args[1]

    gen = arg_register

    if type(gen) == types.GeneratorType:
        for i_register in gen:
            if total_register is None:
                total_register = i_register
            else:
                total_register = interpreter.bin_op(
                    total_register,
                    i_register,
                    lambda x, y: sx.add(x, y),
                )

        return total_register

    else:
        raise ValueError("sum->reduce_sum not implemented yet")


# def sum(*args):
#    return __inline_impl__(sum_impl, args)


def cumsum_impl(interpreter, context, input_register, axis):
    axis = interpreter.move_definite_value(axis)
    return interpreter.un_op(
        input_register,
        lambda x: sx.cumsum(x, axis=d9.lift_axis(axis)),
    )


def cumsum(input, axis=-1):
    return __inline_impl__(cumsum_impl, input, axis)


def print_impl(interpreter, context, args):
    stuff = []
    for arg in args:
        elt = interpreter.move_value(arg)
        stuff.append(elt)
    builtins.print(" ".join(map(str, stuff)))


def print(*args):
    return __inline_impl__(print_impl, args)


def max_impl(interpreter, context, a, b):
    return interpreter.bin_op(a, b, sx.maximum)


def max(a, b):
    return __inline_impl__(max_impl, a, b)


def len_impl(interpreter, context, arg):
    elt = interpreter.move_value(arg)
    if isinstance(elt, sx.Tensor):
        shape = sx.shape(elt)
        if builtins.len(shape) < 2:
            raise InterpreterError("Can't compute len() of scalar.")
        return shape[1]
    else:
        return builtins.len(elt)


def len(arg):
    return __inline_impl__(len_impl, arg)


# Some helpers


def lazy_sort(seq):
    result = []
    for x in seq:
        result = sort([x, *result])
    return result


def lazy_topk(seq, k):
    result = []
    for x in seq:
        result = sort([x, *result])
        if len(result) > k:
            result = result[1:]
    return result


def lazy_botk(seq, k):
    result = []
    for x in seq:
        result = sort([x, *result])
        if len(result) > k:
            result = result[:-1]
    return result


def lazy_bincount(seq, n):
    counts = zeros(n)
    for x in seq:
        counts[x] += 1
    return counts


def lazy_perm(n, m):
    counts = ones(n)
    for i in range(m):
        j = d(n - i)
        x = argmin(j > cumsum(counts))
        counts[x] = 0
        yield x


def lazy_sum(seq):
    return sum(*move(seq))


def lazy_all(seq):
    flag = True
    for x in seq:
        flag = flag and x
    return flag


def lazy_any(seq):
    flag = False
    for x in seq:
        flag = flag or x
    return flag


def lazy_max(seq):
    return max(*move(seq))


def lazy_min(seq):
    return min(*move(seq))


def lazy_kth(seq, k):
    return lazy_topk(seq, k)[0]


def choose(xs):
    return xs[d(len(xs)) - 1]


def weighted(xs, weights):
    n = reduce_sum(weights)
    return xs[argmin(d(n) > cumsum(weights))]


def first(xs):
    return argmax(xs)


def last(xs):
    return len(xs) - 1 - argmax(flip(xs))


# B(n, k / m)
def binomial(n, k, m):
    count = 0
    for i in range(__max__(n)):
        if i < n:
            count += d[0:0:k, 1 : 1 : m - k] == 0
    return count
