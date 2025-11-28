"""An interpreter for the probabilistic language dice-nine."""

import ast
import inspect
import operator
import logging
import dataclasses
import textwrap
import types
from typing import cast
import functools
import warnings
from collections.abc import Iterable

import numpy as np

import dice9.backends.numpy_impl as sx
from dice9 import problib  # pylint: disable=unused-import

from .factor import (
    Factor,
    Real64,
    LogReal64,
    Complex128,
    Int64,
    Semiring,
    BigFraction,
    BigInteger,
    Register,
    SemiringProduct,
)
from .environment import Environment, is_reg
from .frame import Frame
from .exceptions import InterpreterError, FoundReturn
from .analysis import move_analysis

from .utils import is_gen_fun, get_signature_from_functiondef, report_error, lift_axis

warnings.filterwarnings("ignore", category=SyntaxWarning)
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Context:
    frame: Frame


def tuple_broadcast(f, node):
    return (
        tuple(f(elt) for elt in node.elts) if isinstance(node, ast.Tuple) else f(node)
    )


def rowwise_indices(a, indices):
    idxs = [np.asarray(ix) for ix in indices]
    n = a.shape[0]
    batch = np.arange(n).reshape((n,) + (1,) * (idxs[0].ndim - 1))
    ref_shape = idxs[0].shape
    if any(ix.shape != ref_shape for ix in idxs):
        raise ValueError("All index arrays must have identical shapes.")
    if any(ix.shape[0] != n for ix in idxs):
        raise ValueError(
            "Leading dimension of all indices must equal a.shape[0]."
        )
    return idxs, batch


def updated_rowwise(a, b, *indices):
    """
    ∀i a[i, *indices] = b[i]
    """

    if not indices:
        raise ValueError("Provide at least one index array after `b`.")

    idxs, batch = rowwise_indices(a, indices)

    b = np.asarray(b)
    ref_shape = idxs[0].shape

    if b.shape != ref_shape:
        raise ValueError("b and all index arrays must have identical shapes.")

    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError(
            "Leading dimension of b and all indices must equal a.shape[0]."
        )

    a_copy = a.copy()
    a_copy[(batch, *idxs)] = b
    return a_copy


def updated_aug_rowwise(op, a, b, *indices):
    """
    ∀i a[i, *indices] op= b[i]
    """
    if not indices:
        raise ValueError("Provide at least one index array after `b`.")

    idxs, batch = rowwise_indices(a, indices)

    b = np.asarray(b)
    ref_shape = idxs[0].shape

    if b.shape != ref_shape:
        raise ValueError("b and all index arrays must have identical shapes.")

    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError(
            "Leading dimension of b and all indices must equal a.shape[0]."
        )

    a_copy = a.copy()
    a_copy[(batch, *idxs)] = op(a_copy[(batch, *idxs)], b)
    return a_copy


def read_rowwise(a, *indices):
    """
    return v such that ∀i v[i] = a[i, indices[0, i]], indices[1, i], ...
    """
    idxs, batch = rowwise_indices(a, indices)
    return a[(batch, *idxs)]


class Interpreter(ast.NodeVisitor):

    def __init__(
        self,
        parsed,
        globals,
        semiring,
        traceback=False,
        static_analyse=True,
        show_analysis=False,
    ):

        if not isinstance(semiring, Semiring):
            raise TypeError("The semiring should be in instance of type `Semiring`.")

        self.parsed = parsed
        self.node = None
        self.globals = globals
        self.semiring = semiring
        self.traceback = traceback
        self.static_analyse = static_analyse
        self.show_analysis = show_analysis
        self.env = Environment(semiring, [])
        self.analysis_cache = {}

        self.visit_dispatch_table = {
            "Pass": self.visit_pass,
            "Assert": self.visit_assert,
            "Assign": self.visit_assign,
            "AugAssign": self.visit_aug_assign,
            "Expr": self.visit_expr,
            "For": self.visit_for,
            "FunctionDef": self.visit_function_def,
            "GeneratorExp": self.visit_generator_exp,
            "List": self.visit_list,
            "ListComp": self.visit_list_comp,
            "Module": self.visit_module,
            "Name": self.visit_name,
            "Return": self.visit_return,
            "If": self.visit_if,
            "Constant": self.visit_constant,
            "IfExp": self.visit_if_exp,
            "Yield": self.visit_yield,
            "BinOp": self.visit_bin_op,
            "Compare": self.visit_compare,
            "BoolOp": self.visit_bool_op,
            "UnaryOp": self.visit_unary_op,
            "Call": self.visit_call,
            "Delete": self.visit_delete,
            "Subscript": self.visit_subscript,
            "Set": self.visit_set,
            "arguments": self.visit_arg_with_contextuments,
            "arg": self.visit_arg_with_context,
        }

        self.visit_as_generator_dispatch_table = {
            "Module": self.visit_as_generator_module,
            "FunctionDef": self.visit_as_generator_functiondef,
            "arguments": self.visit_as_generator_arguments,
            "arg": self.visit_as_generator_arg,
            "Yield": self.visit_as_generator_yield,
            "YieldFrom": self.visit_as_generator_yieldfrom,
            "Expr": self.visit_as_generator_expr,
            "For": self.visit_as_generator_for,
            "If": self.visit_as_generator_if,
            "AugAssign": self.visit_as_generator_augassign,
            "Assign": self.visit_as_generator_assign,
            "Delete": self.visit_as_generator_delete,
        }

        self.call_dispatch_table = {
            "move": self.call_move,
            "d": self.call_d,
            "dd": self.builtin_dd,
            "__dumpvars__": self.call_dumpvars,
            "__listvars__": self.call_listvars,
            "__inline_impl__": self.call_inline_impl,
            "min": self.call_min,
            "max": self.call_max,
            "sum": self.call_sum,
            "__max__": self.call__max__,
            "__min__": self.call__min__,
            "__any__": self.call__any__,
            "__all__": self.call__all__,
            "zip": self.call_zip,
            "importance": self.call_importance,
        }

    def run(self, node, context: Context):
        self.visit_with(node, context)

    def set_debug_loc(self, node, frame):
        self.node = node
        self.frame = frame

    def visit_with(self, node, context: Context):
        self.set_debug_loc(node, context.frame)
        name = node.__class__.__name__
        if name not in self.visit_dispatch_table:
            raise InterpreterError(
                f"{type(node).__name__} statement not supported.",
                node=node,
                frame=context.frame,
            )
        method = self.visit_dispatch_table[name]
        try:
            return method(node, context)
        except InterpreterError as exc:
            if not hasattr(exc, "node") and node:
                exc.node = node
                exc.frame = context.frame
            raise exc

    def visit_generator_function(self, node, context: Context):
        try:
            visitor = self.visit_as_generator_dispatch_table[node.__class__.__name__]
            yield from visitor(node, context)
        finally:
            logging.debug("Cleaning up generator function.")
            context.frame.delete_all(self.env)

    def visit_as_generator(self, node, context: Context):
        try:
            self.node = node
            result = self.visit_as_generator_dispatch_table[node.__class__.__name__](
                node, context
            )
            return result

        except InterpreterError:
            raise
        except Exception as e:
            raise InterpreterError(
                f"Error while interpreting {type(node).__name__}",
                node=node,
                frame=context.frame,
            ) from e

    def visit_as_generator_generic(self, node, context: Context):
        for child in ast.iter_child_nodes(node):
            yield from self.visit_as_generator(child, context)

    def visit_generic(self, node, context: Context):
        for child in ast.iter_child_nodes(node):
            self.visit_with(child, context)

    # This is necessary or else the default args will
    # be evaluated and left orphaned.
    # XXX do same for generators
    def visit_function_def(self, node, context: Context):
        self.visit_statements(node.body, context)

    visit_arg_with_contextuments = visit_generic
    visit_arg_with_context = visit_generic
    visit_as_generator_module = visit_as_generator_generic
    visit_as_generator_functiondef = visit_as_generator_generic
    visit_as_generator_arguments = visit_as_generator_generic
    visit_as_generator_arg = visit_as_generator_generic

    def visit_yield(self, node, context: Context):
        raise InterpreterError(
            "Can't yield from top level function.", node=node, frame=context.frame
        )

    def visit_as_generator_yield(self, node, context: Context):
        logging.debug("Yielding... %s", node.value)
        if context.frame.conditional_depth > 0:
            raise InterpreterError(
                "Can't yield from a branch of a conditional",
                node=node,
                frame=context.frame,
            )

        value = tuple_broadcast(lambda x: self.visit_with(x, context), node.value)

        logging.debug("Yielding %s.", value)
        yield value

    def visit_as_generator_yieldfrom(self, node, context: Context):
        yield from self.visit_with(node.value, context)

    def visit_as_generator_expr(self, node, context: Context):
        return self.visit_as_generator(node.value, context)

    def visit_list_comp(self, node, context: Context):
        expr = node.elt
        element_name = node.generators[0].target
        domain = node.generators[0].iter

        elts = list(self.comprehension(expr, element_name, domain, context))
        return self.lift(lambda *z: sx.stack(z, 1))(*elts)

    def comprehension(self, expr, var_expr, iterator_expr, context: Context):
        iter_values = self.visit_with(iterator_expr, context)
        name = var_expr.id

        for i in iter_values:
            context.frame.delete_var_if_defined(self.env, name)

            if isinstance(iter_values, range):
                bound_value = context.frame.allocate_register_with_definite_value(
                    self.env, i
                )
            else:
                bound_value = i

            context.frame.bind(name, bound_value)
            yield self.visit_with(expr, context)

        if context.frame.has_var(name):
            context.frame.delete_var_and_register(self.env, name)

    def visit_as_generator_for(self, node, context: Context):
        for _ in self.for_generator(node, context):
            yield from self.visit_as_generator_statements(node.body, context)

    def visit_list(self, node, context: Context):
        if not node.elts:
            return context.frame.allocate_register_with_definite_value(
                self.env, sx.zeros((0,), dtype=sx.int64)
            )

        element_registers = [
            (
                self.visit_with(element.value, context)
                if isinstance(element, ast.Starred)
                else self.un_op(self.visit_with(element, context), lambda x: x[:, None])
            )
            for element in node.elts
        ]

        return self.lift(lambda *x: sx.concat(x, 1))(*element_registers)

    def visit_pass(self, _node, _context: Context):
        # These dels are to shut up pyright.
        del _node, _context
        return

    def visit_return(self, node, context: Context):
        if context.frame.conditional_depth > 0:
            raise InterpreterError(
                "Can't return from a branch of a conditional",
                node=node,
                frame=context.frame,
            )

        registers = tuple_broadcast(lambda x: self.visit_with(x, context), node.value)
        logging.debug("visit_return: return register = %s.", registers)

        vars_to_delete = [key for key in context.frame._allocations]
        logging.debug("Deleting variables `%s` at function return.", vars_to_delete)
        context.frame.delete_all(self.env)
        raise FoundReturn(registers)

    def report_line(self, context: Context, stmt):
        start = stmt.lineno
        end = getattr(stmt, "end_lineno", start)
        lines = context.frame.source.splitlines()
        message = lines[start - 1]
        if end > start:
            message = message + " ..."
        logging.debug(
            "[bold yellow]Executing[/bold yellow]: [bold white]%s[/bold white]",
            textwrap.dedent(message),
        )

    def visit_statements(self, node, context: Context):
        for stmt in node:
            if LOGGER.isEnabledFor(logging.DEBUG) and hasattr(stmt, "lineno"):
                self.report_line(context, stmt)
            self.visit_with(stmt, context)

    def visit_as_generator_statements(self, node, context: Context):
        for stmt in node:
            logging.debug("Executing: %s.", ast.unparse(stmt))
            yield from self.visit_as_generator(stmt, context)

    def visit_module(self, node, context: Context):
        self.visit_statements(node.body, context)

    def visit_expr(self, node, context: Context):
        return self.visit_with(node.value, context)

    def visit_generator_exp(self, node, context: Context):
        element_name = node.generators[0].target
        domain = node.generators[0].iter

        return self.comprehension(node.elt, element_name, domain, context)

    def for_generator(self, node, context: Context):
        logging.debug("For")

        iterable = self.visit_with(node.iter, context)

        match iterable:
            case Register():
                name = node.target.id
                # Axis 0 is probability; Axis 1 is user-visible.
                num = sx.shape(self.env[iterable])[1]
                for i in range(num):
                    context.frame.delete_var_if_defined(self.env, name)
                    # Refetch each time: marginalizations may rearrange storage.
                    row = self.env[iterable][:, i]
                    new_register = context.frame.allocate_in_same_factor_as_register(
                        self.env, iterable, name, row
                    )
                    logging.debug(
                        "Created register %s in Env(%s) to contain loop var `%s`.",
                        new_register,
                        id(self.env),
                        name,
                    )
                    yield

            case Iterable():  # types.GeneratorType() | zip() | list():
                names = (
                    # Assuming we have simple `for <name1>, ... = ...`
                    tuple(cast(ast.Name, target).id for target in node.target.elts)
                    if isinstance(node.target, ast.Tuple)
                    else (node.target.id,)
                )
                for element in iterable:
                    elements = element if isinstance(element, tuple) else (element,)
                    for name, element in zip(names, elements):
                        context.frame.delete_var_if_defined(self.env, name)
                        register = context.frame.assign_move(self.env, name, element)
                        logging.debug(
                            "Created register %s in Env(%s) to contain loop var `%s`.",
                            register,
                            id(self.env),
                            name,
                        )
                    yield

            case _:
                raise InterpreterError(
                    f"You tried to iterate over something of type"
                    f"{type(iterable).__name__!r} which isn't iterable",
                    frame=context.frame,
                    node=node.iter,
                )

    def visit_for(self, node, context: Context):
        for _ in self.for_generator(node, context):
            self.visit_statements(node.body, context)

    def _handle_subscript_assignment(
        self,
        context,
        array_name,
        index_node,
        rhs,
        rowwise_fn
    ):
        array = context.frame[array_name]

        match index_node:

            case ast.Slice():
                raise InterpreterError(
                    "Slices not supported as assignment targets.",
                    frame=context.frame,
                    node=index_node,
                )

            case ast.Tuple(elts=elts):
                indices = [self.visit_with(e, context) for e in elts]
                result = self.lift(rowwise_fn)(array, rhs, *indices)

            case _:
                indices = self.visit_with(index_node, context)
                result = self.lift(rowwise_fn)(array, rhs, indices)

        context.frame.bind(array_name, result)

    def visit_assign(self, node, context: Context):
        if len(node.targets) != 1:
            raise TypeError("Only single assignments supported.")

        target = node.targets[0]
        rhs = self.visit_with(node.value, context)

        match target:
            case ast.Name(id=var_name):
                self.visit_simple_assign(context, var_name, rhs)
                logging.debug("Assigned %s = %s", var_name, rhs)

            case ast.Tuple(elts=elts):
                self.visit_assign_tuple(context, elts, rhs)

            case ast.Subscript(value=ast.Name(id=var_name), slice=index_node):
                self._handle_subscript_assignment(
                    context,
                    var_name,
                    index_node,
                    rhs,
                    updated_rowwise
                )

            case ast.Subscript():
                raise NotImplementedError(
                    "Only simple subscript assignment currently supported."
                )

            case _:
                raise NotImplementedError("Only simple assignments supported.")

    def visit_aug_assign(self, node, context: Context):
        rhs = self.visit_with(node.value, context)

        op_map = {
            ast.Add: sx.add,
            ast.Sub: sx.subtract,
            ast.Mult: sx.multiply,
            ast.Div: sx.divide,
            ast.FloorDiv: sx.floordiv,
            ast.BitAnd: sx.bitwise_and,
            ast.BitOr: sx.bitwise_or,
            ast.BitXor: sx.bitwise_xor
        }

        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError("Unsupported augmented op")

        op_fn = op_map[op_type]

        match node.target:
            case ast.Name(id=var_name):
                logging.debug("Assign `%s`.", var_name)
                old_register = context.frame[var_name]
                result = self.bin_op(old_register, rhs, op_fn)
                context.frame.bind(var_name, result)

            case ast.Subscript(value=ast.Name(id=var_name), slice=index_node):
                def rowwise_fn(o, v, *i):
                    return updated_aug_rowwise(op_fn, o, v, *i)

                self._handle_subscript_assignment(
                    context,
                    var_name,
                    index_node,
                    rhs,
                    rowwise_fn
                )

            case _:
                raise NotImplementedError("Unsupported augmented assignment target")

    def lift_as_gen(self, fn, node, context: Context):
        fn(node, context)
        yield from ()

    def visit_as_generator_augassign(self, node, context: Context):
        yield from self.lift_as_gen(self.visit_aug_assign, node, context)

    def visit_as_generator_assign(self, node, context: Context):
        yield from self.lift_as_gen(self.visit_with, node, context)

    def visit_as_generator_delete(self, node, context: Context):
        yield from self.lift_as_gen(self.visit_delete, node, context)

    def visit_as_generator_if(self, node, context: Context):
        yield from self.lift_as_gen(self.visit_if, node, context)

    def visit_simple_assign(self, context: Context, var_name, value):
        context.frame.assign(self.env, var_name, value, clobber=True)

    def visit_assign_tuple(self, context: Context, elts, values):
        for value, var in zip(values, elts):
            self.visit_simple_assign(context, var.id, value)

    def visit_name(self, node, context: Context):
        if isinstance(node.ctx, ast.Load):
            var_name = node.id

            try:
                result = context.frame.copy_of_var(self.env, var_name)
                if context.frame.conditional_depth > 0 and isinstance(
                    result, types.GeneratorType
                ):
                    raise InterpreterError(
                        "Can't use stream variable in branch of conditional."
                    )
                return result
            except KeyError as exc:
                raise InterpreterError(
                    f"Variable '[bold yellow]" f"{var_name}[/bold yellow]' not found.",
                    node=node,
                    frame=context.frame,
                ) from exc

        raise NotImplementedError("???")

    def visit_constant(self, node, _context: Context):
        del _context
        return node.value

    def bin_op(self, left, right, op):
        left, right = self.promote(left), self.promote(right)
        logging.debug("Computing binary operator op=%s(%s, %s)", op, left, right)
        destination = self.env.multi_op([left, right], op)
        del self.env[left], self.env[right]
        return destination

    def bin_op_raw(self, left, right, op):
        left, right = self.promote(left), self.promote(right)
        logging.debug("Computing binary operator op=%s(%s, %s)", op, left, right)
        destination = self.env.binary_op_direct(left, right, op)
        del self.env[left], self.env[right]
        return destination

    def un_op(self, source, op):
        source = self.promote(source)
        destination = self.env.unary_op(source, op)
        logging.debug(
            "Applied unop %s, result in % s, deleting %s.", op, destination, source
        )
        del self.env[source]
        return destination

    def multi_op_direct(self, op, sources):
        logging.debug(f"op={op}")
        sources = list(map(self.promote, sources))
        destination = self.env.multi_op_direct(sources, op)
        for register in sources:
            if is_reg(register):
                del self.env[register]
        return destination

    def lift(self, op):

        def lifted_op(*sources):
            sources = list(map(self.promote, sources))
            destination = self.env.multi_op_direct(sources, op)
            for register in sources:
                if is_reg(register):
                    del self.env[register]
            return destination

        return lifted_op

    def visit_if_exp(self, node, context: Context):
        test = self.visit_with(node.test, context)
        body = self.visit_with(node.body, context)
        orelse = self.visit_with(node.orelse, context)
        return self.lift(sx.where)(test, body, orelse)

    BIN_OP_DISPATCH = {
        ast.Add: sx.add,
        ast.Sub: sx.subtract,
        ast.Mult: sx.multiply,
        ast.Div: sx.divide,
        ast.FloorDiv: sx.floordiv,
        ast.Mod: sx.mod,
        ast.BitAnd: sx.bitwise_and,
        ast.BitOr: sx.bitwise_or,
        ast.BitXor: sx.bitwise_xor,
        ast.LShift: sx.left_shift,
        ast.RShift: sx.right_shift,
    }

    # XXX Need to define local Frame
    def matmult(self, n, gen, context: Context):
        for _ in range(n):
            yield self.visit_with(gen, context)

    def visit_set(self, node, context: Context):
        for element in node.elts:
            if isinstance(element, ast.Starred):
                yield from self.visit_with(element.value, context)
            else:
                yield self.visit_with(element, context)

    def move_value(self, value):
        if not is_reg(value):
            return value
        return self.env.move_register(value)

    def move_definite_value(self, value):
        if not is_reg(value):
            logging.debug("Moving already definite value %s.", value)
            return value
        return self.env.move_definite_register(value)

    def promote(self, value):
        if is_reg(value):
            return value
        if isinstance(value, types.GeneratorType):
            raise InterpreterError(
                "You are trying to use a sequence in an unexpected location.",
                node=self.node,
                frame=self.frame,
            )
        try:
            return self.env.promote(value)
        except Exception:
            raise InterpreterError(
                "Can't convert expression to numeric dice-nine type.",
                node=self.node,
                frame=self.frame,
            )

    def visit_bin_op(self, node, context: Context):
        self.set_debug_loc(node, context.frame)
        logging.debug("Evaluating binary operator `%s`.", type(node.op).__name__)

        left = self.visit_with(node.left, context)

        op_type = type(node.op)
        if op_type == ast.MatMult:
            try:
                n = self.move_definite_value(left)
            except ValueError:
                raise InterpreterError(
                    "The `@` operator needs a definite value for its first argument.",
                    node=node.left,
                    frame=context.frame,
                )
            gen_context = Context(context.frame.copy())
            return self.matmult(n, node.right, gen_context)

        right = self.visit_with(node.right, context)
        try:
            op_func = self.BIN_OP_DISPATCH[op_type]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported binary op {op_type}") from exc

        try:
            return self.bin_op(left, right, op_func)
        except TypeError:
            raise InterpreterError(
                "Can't evaluate binary operator.", node=node.op, frame=context.frame
            )

    COMPARE_OP_DISPATCH = {
        ast.Eq: sx.equal,
        ast.NotEq: sx.not_equal,
        ast.Lt: sx.less,
        ast.Gt: sx.greater,
        ast.GtE: sx.greater_equal,
        ast.LtE: sx.less_equal,
        ast.In: sx.isin,
        ast.NotIn: sx.isnotin,
    }

    def visit_compare(self, node, context: Context):
        if not node.ops:
            return self.visit_with(node.left, context)

        logging.debug("Performing a comparison using `%s`.", type(node.ops[0]).__name__)

        left = self.visit_with(node.left, context)
        result = None

        for op, rhs_node in zip(node.ops, node.comparators):
            current_op_type = type(op)
            op_func = self.COMPARE_OP_DISPATCH.get(current_op_type)
            if op_func is None:
                raise InterpreterError(
                    f"Unsupported compare op `{current_op_type.__name__}`",
                    node=rhs_node,
                    frame=context.frame,
                )

            right = self.visit_with(rhs_node, context)
            left, right = self.promote(left), self.promote(right)

            try:
                logging.debug("Comparing %s with %s.", left, right)
                cmp_result = self.env.multi_op([left, right], op_func)
            except TypeError as exc:
                raise InterpreterError(
                    "Error in comparison operator.",
                    node=op,
                    frame=context.frame,
                ) from exc

            result = (
                cmp_result
                if result is None
                else self.bin_op(result, cmp_result, sx.logical_and)
            )

            logging.debug("Deleting LHS of comparison %s.", left)
            del self.env[left]

            left = right

        del self.env[right]

        return result

    def visit_bool_op(self, node, context: Context):
        values = [self.visit_with(v, context) for v in node.values]

        op_map = {ast.And: sx.logical_and, ast.Or: sx.logical_or}

        try:

            def op(x, y):
                return self.bin_op(x, y, op_map[type(node.op)])

            return functools.reduce(op, values)
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported bool op {type(node.op)}") from exc

    def visit_unary_op(self, node, context: Context):
        operand = self.visit_with(node.operand, context)

        op_map = {
            ast.Not: sx.logical_not,
            ast.USub: sx.negative,
            #            ast.UAdd: sx.positive,
            ast.Invert: sx.invert,
        }

        try:
            op = op_map[type(node.op)]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported unary op {type(node.op)}") from exc

        return self.un_op(operand, op)

    def visit_assert(self, node, context: Context):
        condition_register = self.visit_with(node.test, context)

        frame, env = context.frame.semi_split(self.env, condition_register)

        context.frame = frame
        self.env = env

    def visit_if(self, node, context: Context):

        condition = self.visit_with(node.test, context)

        # Deterministic
        if not is_reg(condition):
            if condition:
                self.visit_statements(node.body, context)
            else:
                self.visit_statements(node.orelse, context)
            return

        frame1, env1, frame2, env2 = context.frame.split(self.env, condition)

        def run_branch(nodes, frame, env):
            context.frame = frame
            self.env = env
            self.visit_statements(nodes, context)
            return context.frame, self.env

        if frame1:
            frame1, env1 = run_branch(node.body, frame1, env1)

        if frame2:
            frame2, env2 = run_branch(node.orelse, frame2, env2)

        if not frame1 or not frame2:
            context.frame.conditional_depth -= 1
            return

        try:
            context.frame, env = Frame.rejoin_frames(frame1, env1, frame2, env2)
        except ValueError as exc:
            raise InterpreterError(
                "Some variables have different sizes in different branches of conditional.\n\n"
                + textwrap.fill('"' + str(exc) + '"', 60),
                node=node,
                frame=context.frame,
            ) from exc

        self.env = env

    def builtin_multiroll(self, ranges_reg, _context: Context, mode="prob"):
        del _context
        fi = self.env.find_factor_index(ranges_reg)
        factor = self.env.factors[fi]
        ranges = self.env[ranges_reg]

        rolls, new_p, idx = sx.new_multiroll_helper(
            self.semiring, ranges, factor.p, mode
        )
        new_values = {reg: sx.gather(val, idx) for reg, val in factor._values.items()}

        roll_reg = Register.new()
        new_values[roll_reg] = rolls

        self.env.factors[fi] = Factor(self.semiring, new_p, new_values)
        del self.env[ranges_reg]

        return roll_reg

    def builtin_dd(self, node, context: Context):
        sides_register = self.visit_with(node.args[0], context)
        return self.env.dd(sides_register)

    def call__reduce__(self, node, context: Context, op):
        reg = self.visit_with(node.args[0], context)
        if is_reg(reg):
            values = self.env[reg]
            del self.env[reg]
            return op(values, 0)

        return reg

    def call__max__(self, node, context: Context):
        return self.call__reduce__(node, context, sx.reduce_max)

    def call__min__(self, node, context: Context):
        return self.call__reduce__(node, context, sx.reduce_min)

    def call__all__(self, node, context: Context):
        return self.call__reduce__(node, context, sx.reduce_all)

    def call__any__(self, node, context: Context):
        return self.call__reduce__(node, context, sx.reduce_any)

    # Special because multiple args which themselves can be *arg.
    def call_extremum(self, node, context: Context, op, identity=None):
        self.set_debug_loc(node, context.frame)
        result = identity

        for arg in node.args:
            values = (
                self.visit_with(arg.value, context)
                if isinstance(arg, ast.Starred)
                else [self.visit_with(arg, context)]
            )
            for value in values:
                result = value if result is None else self.bin_op(result, value, op)

        return result

    def call_min(self, node, context: Context):
        return self.call_extremum(node, context, sx.minimum)

    def call_max(self, node, context: Context):
        return self.call_extremum(node, context, sx.maximum)

    def call_sum(self, node, context: Context):
        return self.call_extremum(node, context, operator.add, identity=0)

    def call_zip(self, node, context: Context):
        gen1 = self.visit_with(node.args[0], context)
        gen2 = self.visit_with(node.args[1], context)
        return zip(gen1, gen2)

    def call_move(self, node, context: Context):
        var_name = node.args[0].id
        result = context.frame.move(var_name)
        logging.debug("Moving var `%s` of value %s.", var_name, result)
        if context.frame.conditional_depth > 0 and isinstance(
            result, types.GeneratorType
        ):
            raise InterpreterError(
                "Can't read stream variable in a conditional branch.",
                frame=context.frame,
                node=node.args[0],
            )
        return result

    def call_d(self, node, context: Context):
        # Check len()
        sides_register = self.visit_with(node.args[0], context)

        sides = self.move_definite_value(sides_register)
        sides = sx.to_py_scalar(sides)

        # Division
        s = self.semiring
        pmf = s.const_ratio(1, sides, shape=(sides,))
        result = self.env.allocate_factor_with_register_with_probability(
            pmf,
            sx.cast(sx.range(1, sides + 1, dtype=sx.int64), sx.int64),
        )
        logging.debug(
            "Created register %s in env #%s to contain result of `d`",
            result,
            id(self.env),
        )
        return result

    # This is a little quirky. The `self.promote` is needed to_py_scalar
    # create a factor, any factor that the importance can be applied
    # to.
    def call_importance(self, node, context: Context):
        # Check len()
        sides_register = self.promote(self.visit_with(node.args[0], context))
        fi = self.env.find_factor_index(sides_register)
        factor = self.env.factors[fi]
        factor.p = factor.p * factor[sides_register]
        return

    def call_dumpvars(self, _node, context: Context):
        del _node
        context.frame.show(self.env)

    def call_listvars(self, _node, _context: Context):
        del _node, _context
        self.env.listvars()

    def call_inline_impl(self, node, context: Context):
        impl_name = node.args[0].id
        args = [self.visit_with(arg, context) for arg in node.args[1:]]
        return self.globals[impl_name](self, context, *args)

    # def call_inline(self, node, context):
    #     impl_name = node.args[0].id
    #     return self.globals[impl_name](self, context, *node.args[1:])

    def visit_call(self, node, context: Context):
        try:
            func_name = node.func.id  # we assume it's a simple name, not obj.method
        except Exception:
            raise InterpreterError("You called a function that is not a function.")
        logging.debug("Call to function with name `%s`.", func_name)

        if func_name in self.call_dispatch_table:
            return self.call_dispatch_table[func_name](node, context)

        # User function call
        if func_name in self.globals:
            logging.debug("Call to user function with name `%s`.", func_name)

            # Get the caller arguments.
            args = [self.visit_with(arg, context) for arg in node.args]
            kwargs = {
                kw.arg: self.visit_with(kw.value, context)
                for kw in node.keywords
                if kw.arg is not None
            }

            # Get the function body
            source = inspect.getsource(self.globals[func_name])

            source = textwrap.dedent(source)
            function_body = ast.parse(source)

            # visit_call
            if self.static_analyse:
                if func_name in self.analysis_cache:
                    function_body = self.analysis_cache[func_name]
                else:
                    function_body = move_analysis(function_body)
                    self.analysis_cache[func_name] = function_body
                    if self.show_analysis:
                        print("analysis =>", ast.unparse(function_body))

            function_def = function_body.body[0]
            new_frame = Frame(self.semiring, source, {}, 0)

            # Get callee signature.
            try:
                signature = get_signature_from_functiondef(function_def)
            except Exception as exc:
                raise InterpreterError(
                    f"Only literals can be used as default arguments.\n{exc}",
                    node=function_body.body[0],
                    frame=new_frame,
                )
            logging.debug("Got signature %s.", signature)

            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                logging.debug("Applied defaults to get %s.", bound)
            except Exception as exc:
                raise InterpreterError(
                    f"Problem binding argument to function\n{exc}",
                    frame=context.frame,
                    node=node,
                )
            full_arg_list = bound.arguments.items()

            logging.debug("Calling `%s` with arguments:", func_name)
            for arg_name, arg in full_arg_list:
                logging.debug("`%s` = %s", arg_name, arg)
                new_frame.bind(arg_name, arg)

            if is_gen_fun(function_def):
                # It's a generator function.
                logging.debug("Creating generator function.")
                try:
                    new_context = Context(new_frame)
                    gen = self.visit_generator_function(function_body, new_context)
                    return gen
                except FoundReturn as exc:
                    raise ValueError("return in generator") from exc

            # it's an ordinary function
            new_context = Context(new_frame)

            try:
                self.visit_with(function_body, new_context)
            except FoundReturn as e:
                logging.debug("Hit `return` in `%s`.", func_name)
                return e.value_node

            raise ValueError("Missing return")

        raise InterpreterError(
            f"Function '[bold yellow]{func_name}[/bold yellow]' not defined.",
            frame=context.frame,
            node=node.func,
        )

    def visit_delete(self, node, context: Context):
        for target in node.targets:
            match target:
                case ast.Name(id=var_name):
                    context.frame.delete_var_and_register(self.env, var_name)

                case ast.Attribute():
                    raise InterpreterError(
                        f"Deleting attribute '" f"{target.attr}' not supported yet.",
                        frame=context.frame,
                        node=node,
                    )

                case ast.Subscript():
                    raise InterpreterError(
                        "Deleting subscripted values isn't supported yet.",
                        frame=context.frame,
                        node=node,
                    )

    def multiroll_d(self, range_expr, context: Context, mode="prob"):

        def one_range(slice_expr):
            match slice_expr:
                case ast.Slice(lower=lower, upper=upper, step=step):
                    lower = self.visit_with(lower, context)
                    upper = self.visit_with(upper, context) if upper else lower
                    step = self.visit_with(step, context) if step else 1
                    return self.lift(lambda *x: sx.stack(x, 1))(lower, upper, step)

                case _:
                    r = self.visit_with(slice_expr, context)
                    return self.lift(lambda *x: sx.stack(x, 1))(r, r, 1)

        match range_expr:
            case ast.Tuple(elts=elts):
                ranges = [one_range(s) for s in elts]

                all_rolls = self.lift(lambda *x: sx.stack(x, 1))(*ranges)

                return self.builtin_multiroll(all_rolls, context, mode)

            case _:
                s = one_range(range_expr)
                s = self.lift(lambda x: sx.stack([x], 1))(s)
                return self.builtin_multiroll(s, context, mode)

    def visit_subscript(self, node, context: Context):
        # Special case of d[...]
        if isinstance(node.value, ast.Name) and node.value.id == "d":
            return self.multiroll_d(node.slice, context)

        obj = self.visit_with(node.value, context)
        slice_node = node.slice

        def eval_lit(n):
            if n is None:
                return None
            try:
                return ast.literal_eval(n)
            except Exception:
                raise InterpreterError(
                    "dice9 only supports literals on slice ranges so as to avoid divergence issues.",
                    frame=context.frame,
                    node=n,
                )

        def literal_slice(lower, upper, step):
            return slice(eval_lit(lower), eval_lit(upper), eval_lit(step))

        match slice_node:
            case ast.Slice(lower=lower, upper=upper, step=step):
                aslice = literal_slice(lower, upper, step)
                return self.un_op(obj, lambda x: x[:, aslice])

            case ast.Tuple(elts=elts):
                if any(isinstance(e, ast.Slice) for e in elts):
                    def to_slice(e):
                        match e:
                            case ast.Constant(value=value):
                                return value
                            case ast.Slice(lower=lower, upper=upper, step=step):
                                return literal_slice(lower, upper, step)
                            case _:
                                raise InterpreterError(
                                    "dice-nine currently only supports constant multidimensional indices.",
                                    node=slice_node,
                                    frame=context.frame,
                                )
                    slices = tuple(to_slice(e) for e in elts)
                    lifted_slices = (slice(None, None, None),) + slices
                    return self.un_op(obj, lambda x: x[lifted_slices])

                indices = [self.visit_with(e, context) for e in elts]
                try:
                    return self.lift(lambda o, *i: read_rowwise(o, *i))(obj, *indices)
                except Exception as exc:
                    raise InterpreterError(
                        f"Array subscript error when reading array.\n{exc}.",
                        node=slice_node,
                        frame=context.frame,
                    )

            case ast.Constant(value=value):
                return self.un_op(obj, lambda x: x[:, value])

            case _:
                slice_array = self.visit_with(slice_node, context)
                try:
                    return self.bin_op_raw(obj, slice_array, sx.subscript)
                except IndexError as exc:
                    raise InterpreterError(
                        f"Bounds error when reading array.\n{exc}.",
                        node=slice_node,
                        frame=context.frame,
                    )

    def analyse(self, source, show_analysis=False):
        try:
            self.parsed = move_analysis(self.parsed)
        except InterpreterError as exc:
            node = exc.node if exc.node else self.node
            report_error(exc, node, source)
            if self.traceback:
                raise
            raise InterpreterError(exc.args) from None

        if show_analysis:
            print("transformed code:")
            print(ast.unparse(self.parsed))


def dist(f=None, **options):

    defaults = {
        "normalize": False,
        "modules": [problib],
        "traceback": False,
        "semiring": Real64(),
        "static_analyse": True,
        "show_analysis": False,
    }
    merged_options = defaults | options

    def decorator(f):
        source = textwrap.dedent(inspect.getsource(f))
        parsed = ast.parse(source)

        function_def = parsed.body[0]
        assert isinstance(function_def, ast.FunctionDef)
        signature = get_signature_from_functiondef(function_def)

        @functools.wraps(f)
        def wrapped(*args, _options=None, **kwargs):
            logging.debug("Starting execution with function `%s(args=%s, kwargs=%s)`.", f.__name__, args, kwargs)
            nonlocal merged_options
            if _options:
                merged_options |= _options

            context = {}
            if merged_options:
                for module in merged_options["modules"]:
                    context.update(module.__dict__)
            context.update(f.__globals__)

            seen = set()

            def add_closure(fn):
                if fn in seen:
                    return
                seen.add(fn)
                if fn.__closure__:
                    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                        val = cell.cell_contents
                        context[name] = val
                        if callable(val) and hasattr(val, "__code__"):
                            add_closure(val)

            add_closure(f)

            interpreter = Interpreter(
                parsed,
                semiring=merged_options["semiring"],
                traceback=merged_options["traceback"],
                globals=context,
            )

            if merged_options["static_analyse"]:
                interpreter.analyse(source, show_analysis=merged_options["show_analysis"])

            frame = Frame(interpreter.semiring, source, {}, 0)
            ctx = Context(frame)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            logging.debug("Calling `%s` with arguments:", function_def.name)
            for arg_name, arg_value in bound.arguments.items():
                logging.debug("`%s` = %s", arg_name, arg_value)
                frame.bind(arg_name, arg_value)

            try:
                interpreter.run(parsed, ctx)
            except FoundReturn as e:
                values = e.value_node
                if isinstance(values, tuple):
                    values = tuple(interpreter.promote(v) for v in values)
                else:
                    values = interpreter.promote(values)
                logging.debug("return `%s`", values)
                return interpreter.env.distribution(values, merged_options["normalize"])
            except InterpreterError as exc:
                node = exc.node if exc.node else interpreter.node
                report_error(exc, node, exc.frame.source)
                if interpreter.traceback:
                    raise
                raise InterpreterError(exc.args) from None

            return interpreter.env

        return wrapped

    return decorator if f is None else decorator(f)

__all__ = [
    "LogReal64", "Complex128", "Int64", "BigFraction",
    "BigInteger", "SemiringProduct", "lift_axis", "dist", "Real64", "InterpreterError"
]
