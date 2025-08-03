import builtins
import types
import ast
import dice9 as d9

def is_reg(register):
    return isinstance(register, d9.Register)

from dice9.config import sx

def reshape_impl(interpreter, context, tensor_register, shape_register):
    value = interpreter.env.move_definite_value(shape_register)
    shape = (-1,) + sx.convert_to_tuple(value)
    return interpreter.un_op(
        tensor_register, lambda x: sx.reshape(x, shape)
    )

def reshape(tensor, shape):
    return __inline_impl__(reshape_impl, tensor, shape)

def isin_impl(interpreter, context, element_register, test_elements_register):
    return interpreter.bin_op(element_register, test_elements_register, lambda t, e: sx.isin(t, e))

def isin(element, test_elements):
    return __inline_impl__(isin_impl, element, test_elements)

def abs_impl(interpreter, context, x_register):
    return interpreter.un_op(x_register, sx.abs)

def abs(x):
    return __inline_impl__(abs_impl, x)

def arange_impl(interpreter, context, start_register, limit_register, delta_register):
    start = interpreter.env.move_definite_value(start_register)
    limit = interpreter.env.move_definite_value(limit_register)
    delta = interpreter.env.move_definite_value(delta_register)
    return interpreter.env.allocate_register_with_definite_value(sx.arange(start, limit, delta))

def arange(start, limit, delta):
    return __inline_impl__(arange_impl, start, limit, delta)

def int_impl(interpreter, context, value_register):
    return interpreter.un_op(value_register, lambda x: sx.cast(x, sx.int64))

def int(value):
    return __inline_impl__(int_impl, value)

def impl_axis_maker(op):
    def impl(interpreter, context, input_register, axis_register):
        axis = interpreter.env.move_definite_value(axis_register)
        return interpreter.un_op(
            input_register,
            lambda x: op(x, axis=d9.lift_axis(axis))
        )
    return impl

# Flip needs to work with bools too.
flip_impl = impl_axis_maker(lambda x, axis: sx.flip(sx.cast(x, sx.int64), axis=axis))

argsort_impl = impl_axis_maker(lambda x, axis: sx.argsort(sx.cast(x, sx.int64), axis=axis))
argmin_impl = impl_axis_maker(lambda x, axis: sx.argmin(sx.cast(x, sx.int64), axis=axis))
argmax_impl = impl_axis_maker(lambda x, axis: sx.argmax(sx.cast(x, sx.int64), axis=axis))

reduce_all_impl = impl_axis_maker(lambda x, axis: sx.reduce_all(sx.cast(x, sx.bool), axis=axis))
reduce_any_impl = impl_axis_maker(lambda x, axis: sx.reduce_any(sx.cast(x, sx.bool), axis=axis))
reduce_min_impl = impl_axis_maker(lambda x, axis: sx.reduce_min(sx.cast(x, sx.int64), axis=axis))
reduce_max_impl = impl_axis_maker(lambda x, axis: sx.reduce_max(sx.cast(x, sx.int64), axis=axis))
reduce_sum_impl = impl_axis_maker(lambda x, axis: sx.reduce_sum(sx.cast(x, sx.int64), axis=axis))

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
    depth = interpreter.env.move_definite_value(depth_register)
    return interpreter.un_op(index_register,
      lambda x: sx.one_hot(x, depth))

def one_hot(tensor, depth):
    return __inline_impl__(one_hot_impl, tensor, depth)

def zeros_impl(interpreter, context, shape_register):
    shape = interpreter.env.move_definite_value(shape_register)
    #del interpreter.env[shape_register]

    zero_array = sx.zeros(sx.to_python(shape), dtype=sx.int64)

    return interpreter.env.allocate_register_with_definite_value(zero_array)

def zeros(shape):
    return __inline_impl__(zeros_impl, shape)

def flip(input, axis=-1):
    return __inline_impl__(flip_impl, input, axis)

#def argmin_impl(interpreter, context, values_register, axis_register):
#    axis = interpreter.env.move_definite_value(axis_register)
#    return interpreter.un_op(
#        values_register,
#        lambda x: sx.argmin(sx.cast(x, sx.int64), axis=d9.lift_axis(axis))
#    )

def argmin(values, axis=-1):
    return __inline_impl__(argmin_impl, values, axis)

#def argmax_impl(interpreter, context, values_register, axis_register):
#    axis = interpreter.env.move_definite_value(axis_register)
#    return interpreter.un_op(
#        values_register,
#        lambda x: sx.argmax(sx.cast(x, sx.int64), axis=d9.lift_axis(axis))
#    )

def argmax(values, axis=-1):
    return __inline_impl__(argmax_impl, values, axis)

def argsort(values, axis=-1):
    return __inline_impl__(argsort_impl, values, axis)

def bincount_impl(interpreter, context, arr_register, maxlength_register):
    maxlength = interpreter.env.move_definite_value(maxlength_register)
    return interpreter.un_op(
        arr_register,
        lambda x: sx.bincount(sx.cast(x, sx.int64), maxlength)
    )

def bincount(arr, maxlength):
    return __inline_impl__(bincount_impl, arr, maxlength)

def top_k_impl(interpreter, context, input_register, k_register):
    k = interpreter.env.move_definite_value(k_register)
    return interpreter.un_op(input_register, lambda x: sx.top_k(x, k)[0])

def top_k(input, k):
    return __inline_impl__(top_k_impl, input, k)

def sort_impl(interpreter, context, values_register, axis_register):
    axis = interpreter.env.move_definite_value(axis_register)
    return interpreter.un_op(
        values_register,
        lambda x: sx.sort(sx.cast(x, sx.int64), axis=d9.lift_axis(axis))
    )

def sort(values, axis=-1):
    return __inline_impl__(sort_impl, values, axis)

def ones_impl(interpreter, context, shape_register):
    shape = interpreter.env.move_definite_value(shape_register)

    one_array = sx.ones(sx.to_python(shape), dtype=sx.int64)

    return interpreter.env.allocate_register_with_definite_value(one_array)

def ones(shape):
    return __inline_impl__(ones_impl, shape)

def range_impl(interpreter, context, args):
    definite_args = [interpreter.env.move_definite_value(arg) for arg in args]
    return builtins.range(*definite_args)

def range(*args):
    return __inline_impl__(range_impl, args)

def list_impl(interpreter, context, args):
    def outer_product(xs):
        s = sx.stack(xs)
        rank = sx.rank(s)
        u = sx.transpose(s, [1, 0] + builtins.list(builtins.range(2, rank)))
        return u

    gens = args
    element_registers = [g for gen in gens for g in builtins.list(gen)]
    r = interpreter.multi_op_direct(
        element_registers, lambda *x: outer_product(x)
    )
    return r

def list(*args):
    return __inline_impl__(list_impl, args)

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
                    lambda x, y: sx.add(
                        sx.cast(x, sx.int64), sx.cast(y, sx.int64)
                    )
                )

        return total_register

    else:
        raise ValueError("sum->reduce_sum not implemented yet")

#def sum(*args):
#    return __inline_impl__(sum_impl, args)

def cumsum_impl(interpreter, context, input_register, axis):
    axis = interpreter.env.move_definite_value(axis)
    return interpreter.un_op(
        input_register,
        lambda x: sx.cumsum(sx.cast(x, sx.int64), axis=d9.lift_axis(axis))
    )

def cumsum(input, axis=-1):
    return __inline_impl__(cumsum_impl, input, axis)

def print_impl(interpreter, context, args):
    stuff = []
    for arg in args:
        elt = interpreter.env.move_value(arg)
        stuff.append(elt)
    builtins.print(" ".join(map(str, stuff)))

def print(*args):
    return __inline_impl__(print_impl, args)

def max_impl(interpreter, context, a, b):
    return interpreter.bin_op(a, b, sx.maximum)

def max(a, b):
    return __inline_impl__(max_impl, a, b)

if 0:
    def min_impl(interpreter, context, a):
        t = interpreter.visit(a[0], context)
        for i in a[1:]:
            if isinstance(i, ast.Starred):
                for j in self.visit(i.value, context):
                    t = interpreter.bin_op(t, interpreter.visit(j, context), sx.minimum)
            else:
                ii = interpreter.visit(i, context)
                t = interpreter.bin_op(t, ii, sx.minimum)
        return t

    def min(*a):
        return __inline__(min_impl, a)

def len_impl(interpreter, context, arg):
    elt = interpreter.env.move_value(arg)
    if isinstance(elt, sx.Tensor):
        return sx.shape(elt)[1]
    else:
        return builtins.len(elt)

def len(arg):
    return __inline_impl__(len_impl, arg)
