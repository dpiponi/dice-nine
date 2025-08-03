from functools import reduce
from typing import Any, List
import ast
import inspect
import logging
import operator
import textwrap
import types

from dice9 import problib  # pylint: disable=unused-import
from dice9.config import sx
from .factor import Factor, new_register_name
from .environment import Environment, is_reg
from .frame import Frame


def lift_axis(axis):
    return axis + 1 if axis >= 0 else axis


def partition_by_version(x, y):
    """
    Given two lists of factors split them up into
    elements common to both and elements unique to each.
    """
    x_ids = {id(obj) for obj in x}
    y_ids = {id(obj) for obj in y}
    common_ids = x_ids & y_ids

    seen = set()
    common: List[Any] = []
    for obj in x + y:
        if id(obj) in common_ids and id(obj) not in seen:
            common.append(obj)
            seen.add(id(obj))

    x_unique = [obj for obj in x if id(obj) not in y_ids]
    y_unique = [obj for obj in y if id(obj) not in x_ids]

    return common, x_unique, y_unique


def to_numpy(x: Any) -> Any:
    if isinstance(x, sx.Tensor):
        return x.numpy()
    return x


def is_gen_fun(tree):
    for subnode in ast.walk(tree):
        if isinstance(subnode, (ast.Yield, ast.YieldFrom)):
            return True
    return False


class FoundReturn(Exception):
    def __init__(self, value_node):
        self.value_node = value_node


class Context:
    def __init__(self, globals, frame):
        self.globals = globals
        self.frame = frame

    def copy(self):
        return Context(self.globals, self.frame)


def get_signature_from_functiondef(fndef: ast.FunctionDef) -> inspect.Signature:
    def make_param(arg, kind, default=inspect._empty):
        return inspect.Parameter(arg.arg, kind, default=default)

    def try_literal_eval(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return inspect._empty

    args = fndef.args
    params = []

    # Positional-only (Python 3.8+)
    posonlyargs = getattr(args, "posonlyargs", [])
    for arg in posonlyargs:
        params.append(make_param(arg, inspect.Parameter.POSITIONAL_ONLY))

    # Positional-or-keyword
    for arg in args.args:
        params.append(make_param(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))

    # *args
    if args.vararg:
        params.append(make_param(args.vararg, inspect.Parameter.VAR_POSITIONAL))

    # Keyword-only
    for arg in args.kwonlyargs:
        params.append(make_param(arg, inspect.Parameter.KEYWORD_ONLY))

    # **kwargs
    if args.kwarg:
        params.append(make_param(args.kwarg, inspect.Parameter.VAR_KEYWORD))

    # === Apply defaults ===
    total_pos = len(posonlyargs) + len(args.args)

    # Defaults for positional-or-keyword
    for i, default_node in enumerate(args.defaults):
        index = total_pos - len(args.defaults) + i
        params[index] = params[index].replace(default=try_literal_eval(default_node))

    # Defaults for keyword-only
    kwonly_start = total_pos
    for i, default_node in enumerate(args.kw_defaults):
        if default_node is not None:
            index = kwonly_start + i
            params[index] = params[index].replace(
                default=try_literal_eval(default_node)
            )

    return inspect.Signature(params)


def outer_product(x):
    s = sx.stack(x)
    rank = sx.rank(s)
    u = sx.transpose(s, [1, 0] + list(range(2, rank)))
    return u


class InterpreterError(Exception):
    def __init__(self, message, node=None, frame=None, cause=None):
        super().__init__(message)
        self.node = node
        self.frame = frame
        self.__cause__ = cause  # for exception chaining


class Interpreter(ast.NodeVisitor):
    def __init__(self, parsed, env, modules=None):
        self.parsed = parsed
        self.env = env
        self.next = 0
        self.node = None
        self.modules = modules

        self.visit_dispatch_table = {
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
            "BinOp": self.visit_bin_op,
            "Compare": self.visit_compare,
            "BoolOp": self.visit_bool_op,
            "UnaryOp": self.visit_unary_op,
            "Call": self.visit_call,
            "Delete": self.visit_delete,
            "Subscript": self.visit_subscript,
            "Set": self.visit_set,
            "arguments": self.visit_arguments,
            "arg": self.visit_arg,
        }

        self.call_dispatch_table = {
            "move": self.call_move,
            "__chop__": self.call_chop,
            "d": self.call_d,
            "dd": self.builtin_dd,
            "concat": self.call_concat,
            "cast": self.call_cast,
            "__dumpvars__": self.call_dumpvars,
            "__listvars__": self.call_listvars,
            "__inline_impl__": self.call_inline_impl,
            "__inline__": self.call_inline,
            "min": self.call_min,
            "max": self.call_max,
            "sum": self.call_sum,
        }

    def visit(self, node, context):
        try:
            self.node = node
            result = self.visit_dispatch_table[node.__class__.__name__](node, context)
            return result
        except FoundReturn as f:
            raise
        except Exception as e:
            if isinstance(e, InterpreterError):
                raise  # already enriched
            raise InterpreterError(
                f"Error while interpreting {type(node).__name__}",
                node=node,
                frame=context.frame,
                cause=e,
            ) from e

    def visit_generator_function(self, node, context):
        logging.debug("visit_generator_function")
        method = "visit_as_generator_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit_as_generator)
        gen = visitor(node, context)
        logging.debug(f"gen={gen}")
        yield from gen
        logging.debug("Cleaning up generator fn.")
        # assert self.env == context.frame.environment
        context.frame.delete_all(self.env)

    def visit_as_generator(self, node, context):
        try:
            self.node = node
            method = "visit_as_generator_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit_as_generator)
            yield from visitor(node, context)
            logging.debug("Called `%s`", method)
        except Exception as e:
            if isinstance(e, InterpreterError):
                raise  # already enriched
            raise InterpreterError(
                f"Error while interpreting {type(node).__name__}",
                node=node,
                frame=context.frame,
                cause=e,
            ) from e

    def generic_visit(self, node, context):
        raise ValueError(
            f"I don't know how to evaluate a starred expression in this context."
        )

    def visit_arguments(self, node, context):
        for child in ast.iter_child_nodes(node):
            self.visit(child, context)

    def visit_function_def(self, node, context):
        assert context.frame
        for child in ast.iter_child_nodes(node):
            assert context.frame
            self.visit(child, context)
            assert context.frame

    def visit_arg(self, node, context):
        for child in ast.iter_child_nodes(node):
            self.visit(child, context)

    def generic_visit_as_generator(self, node, context):
        raise ValueError("I dont know how to visit `%s`.", node)

    def visit_as_generator_Module(self, node, context):
        logging.debug("visit_as_generator_Module")
        for child in ast.iter_child_nodes(node):
            yield from self.visit_as_generator(child, context)

    def visit_as_generator_FunctionDef(self, node, context):
        for child in ast.iter_child_nodes(node):
            yield from self.visit_as_generator(child, context)

    def visit_as_generator_arguments(self, node, context):
        for child in ast.iter_child_nodes(node):
            yield from self.visit_as_generator(child, context)

    def visit_as_generator_arg(self, node, context):
        for child in ast.iter_child_nodes(node):
            yield from self.visit_as_generator(child, context)

    # 'yield' returns the register, not the variable or value.
    def visit_as_generator_Yield(self, node, context):
        yield_register = self.visit(node.value, context)
        yield yield_register

    def visit_as_generator_YieldFrom(self, node, context):
        yield_register = self.visit(node.value, context)
        yield from yield_register

    def visit_as_generator_Expr(self, node, context):
        return self.visit_as_generator(node.value, context)

    def visit_list_comp(self, node, context):
        expr = node.elt
        element_name = node.generators[0].target
        domain = node.generators[0].iter

        z = list(self.comprehension(expr, element_name, domain, context))
        reg = self.multi_op_direct(z, lambda *z: outer_product(z))

        return reg

    def comprehension(self, expr, var_expr, iterator_expr, context):
        iter_values = self.visit(iterator_expr, context)
        name = var_expr.id

        for i in iter_values:
            context.frame.delete_var_if_defined(self.env, name)

            if isinstance(iter_values, range):
                context.frame.bind(
                    name, self.env.allocate_register_with_definite_value(i)
                )
            else:
                context.frame.bind(name, i)

            yield self.visit(expr, context)

        if context.frame.has_var(name):
            context.frame.delete_var_and_register(self.env, name)

    def visit_as_generator_For(self, node, context):
        logging.debug("For")

        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("General for loops not supported yet")

        name = node.target.id
        rr = self.visit(node.iter, context)

        if is_reg(rr):
            iterable_register = rr
            r = self.env[iterable_register]  # context.frame.get(iter)
            num = sx.shape(r)[1]
            for i in range(num):
                context.frame.delete_var_if_defined(self.env, name)
                row = self.env[iterable_register][:, i]
                logging.debug(f"row value = {row}")
                context.frame.allocate_in_same_factor_as_register(
                    self.env, iterable_register, name, row
                )
                for stmt in node.body:
                    logging.debug("Executing:", ast.unparse(stmt))
                    yield from self.visit_as_generator(stmt, context)
        else:
            iterable = rr  # context.frame.allocations[rr]

            if type(iterable) == types.GeneratorType:
                for element in iterable:
                    context.frame.delete_var_if_defined(self.env, name)
                    # if context.frame.has_var(name):
                    #    context.frame.delete_var_and_register(self.env, name)
                    context.frame.assign_move(self.env, name, element)

                    for stmt in node.body:
                        logging.debug("Executing:", ast.unparse(stmt))
                        yield from self.visit_as_generator(stmt, context)
                    logging.debug("Loop end")

            elif type(iterable) == range:
                for i in iterable:
                    context.frame.delete_var_if_defined(self.env, name)
                    reg = context.frame.allocate(self.env, name, i)
                    logging.debug(f"iterating range with {name} in register {reg}")

                    for stmt in node.body:
                        logging.debug("Executing: %s", ast.unparse(stmt))
                        yield from self.visit_as_generator(stmt, context)

    def visit_list(self, node, context):
        if len(node.elts) == 0:
            return self.env.allocate_register_with_definite_value(
                sx.zeros((0,), dtype=sx.int64)
            )

        element_registers = [self.visit(element, context) for element in node.elts]
        r = self.multi_op_direct(element_registers, lambda *x: outer_product(x))
        return r

    def visit_return(self, node, context):
        if isinstance(node.value, ast.Tuple):
            registers = [self.visit(x, context) for x in node.value.elts]
        else:
            return_value = self.visit(node.value, context)
            logging.debug(f"visit_return: return register = {return_value}")
            registers = [return_value]
        logging.debug("Deleting variables at function return.")
        context.frame.delete_all(self.env)
        raise FoundReturn(registers)

    def visit_module(self, node, context):
        assert context.frame
        for stmt in node.body:
            logging.debug(f"Executing: {ast.unparse(stmt)}")
            self.visit(stmt, context)

    def visit_expr(self, node, context):
        return self.visit(node.value, context)

    def visit_generator_exp(self, node, context):
        expr = node.elt
        element_name = node.generators[0].target
        domain = node.generators[0].iter

        return self.comprehension(expr, element_name, domain, context)

    def visit_for(self, node, context):
        logging.debug("For")

        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("General for loops not supported yet")

        name = node.target.id
        iterable = self.visit(node.iter, context)
        logging.debug(f"iterablr={iterable}")

        if is_reg(iterable):
            # Axis 0 is the probability axis.
            # Axis 1 looks like axis 0 to user.
            num = sx.shape(self.env[iterable])[1]
            for i in range(num):
                context.frame.delete_var_if_defined(self.env, name)
                # Note that we have to refetch the iterable as some rearrangement
                # can take place as a result of marginaliztions.
                row = self.env[iterable][:, i]
                logging.debug(f"row value = {row}")
                context.frame.allocate_in_same_factor_as_register(
                    self.env, iterable, name, row
                )
                for stmt in node.body:
                    logging.debug("Executing: %s", ast.unparse(stmt))
                    self.visit(stmt, context)

        else:
            if isinstance(iterable, types.GeneratorType):
                for element in iterable:
                    context.frame.delete_var_if_defined(self.env, name)
                    context.frame.assign_move(self.env, name, element)
                    for stmt in node.body:
                        logging.debug("Executing: %s", ast.unparse(stmt))
                        self.visit(stmt, context)
                    logging.debug("Loop end")

            elif isinstance(iterable, range):
                for i in iterable:
                    context.frame.delete_var_if_defined(self.env, name)
                    reg = context.frame.allocate(self.env, name, i)
                    logging.debug(f"iterating range with {name} in register {reg}")
                    for stmt in node.body:
                        logging.debug("Executing: %s", ast.unparse(stmt))
                        self.visit(stmt, context)

    def visit_aug_assign(self, node, context):
        register = self.visit(node.value, context)

        op_map = {
            ast.Add: (sx.add, sx.scatter_update_add),
            ast.Sub: (sx.subtract, sx.scatter_update_sub),
            ast.Mult: (sx.multiply, sx.scatter_update_multiply),
        }

        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError("Unsupported augmented op")

        op_fn, scatter_fn = op_map[op_type]

        match node.target:
            case ast.Name(id=var_name):
                logging.debug(f"Assign {var_name}")
                old_register = context.frame.allocations[var_name]
                new_register = self.bin_op(old_register, register, op_fn)
                context.frame.bind(var_name, new_register)

            case ast.Subscript(value=ast.Name(id=var_name), slice=index_node):
                old_register = context.frame.allocations[var_name]

                match index_node:
                    case ast.Constant():
                        index = index_node.value
                        result = self.bin_op_raw(
                            old_register, register, lambda x, v: scatter_fn(x, index, v)
                        )
                        logging.debug("Assigned %s[%s] op= %s", var_name, index, result)

                    case ast.Slice():
                        raise InterpreterError("Slices not supported as assignment targets.", frame=context.frame, node=index_node)

                    case ast.Tuple():
                        raise InterpreterError("Multidimensional indices not supported as assignment targets.", frame=context.frame, node=index_node)
                    case _:
                        index = self.visit(index_node, context)
                        result = self.multi_op_direct(
                            [old_register, index, register], scatter_fn
                        )
                context.frame.bind(var_name, result)

            case _:
                raise NotImplementedError("Unsupported augmented assignment target", node=index_node)

    def visit_as_generator_AugAssign(self, node, context):
        self.visit_aug_assign(node, context)
        yield from ()

    def visit_as_generator_Assign(self, node, context):
        self.visit_assign(node, context)
        yield from ()

    def visit_assign(self, node, context):
        if len(node.targets) != 1:
            raise TypeError("Only single assignments supported.")

        target = node.targets[0]
        value = self.visit(node.value, context)

        match target:
            case ast.Name(id=var_name):
                logging.debug(f"Assign {var_name} = ...")
                context.frame.delete_var_if_defined(self.env, var_name)

                if is_reg(value):
                    context.frame.assign_move(self.env, var_name, value)
                    logging.debug(f"Assigned {var_name} = reg {value}")
                else:
                    if context.frame.conditional_depth > 0:
                        context.frame.bind(var_name, self.env.promote(value))
                    else:
                        context.frame.bind(var_name, value)
                        logging.debug(f"Assigned {var_name} = pyvalue {value}")

            case ast.Subscript(value=ast.Name(id=var_name), slice=index_node):
                old_register = context.frame.allocations[var_name]

                if isinstance(index_node, ast.Constant):
                    index = index_node.value
                    result = self.bin_op_raw(
                        old_register, value, lambda x, v: sx.scatter_update(x, index, v)
                    )
                    logging.debug("Assigned %s[%s] = %s", var_name, index, result)
                else:
                    try:
                        index = self.visit(index_node, context)
                        result = self.multi_op_direct(
                            [old_register, index, value], sx.scatter_update
                        )
                    except Exception as e:
                        raise ValueError("Error during subscript assignment.") from e

                context.frame.bind(var_name, result)

            case ast.Subscript():
                raise NotImplementedError(
                    "Only simple subscript assignment currently supported."
                )

            case _:
                raise NotImplementedError("Only simple assignments supported.")

    def visit_name(self, node, context):
        if isinstance(node.ctx, ast.Load):
            var_name = node.id
            if context.frame.has_var(var_name):
                register = context.frame.allocations[var_name]
                if is_reg(register):
                    new_register = self.env.duplicate_register(
                        context.frame.allocations[var_name]
                    )
                    logging.debug(
                        "Reading from variable `%s` (register %s) into register %s.",
                        var_name,
                        register,
                        new_register,
                    )
                    return new_register
                else:
                    logging.debug("Reading %s reg=%s", var_name, register)
                    return register
            raise NameError(f"Variable '{var_name}' not found.")
        else:
            raise NotImplementedError("???")

    def visit_constant(self, node, context):
        return node.value

    def bin_op(self, left_register, right_register, op):
        left_register = self.env.promote(left_register)
        right_register = self.env.promote(right_register)
        logging.debug(
            "Computing binary operator op=%s(%s, %s)", op, left_register, right_register
        )
        destination_register = self.env.multi_op([left_register, right_register], op)
        del self.env[left_register]
        del self.env[right_register]
        return destination_register

    def bin_op_raw(self, left_register, right_register, op):
        left_register = self.env.promote(left_register)
        right_register = self.env.promote(right_register)
        destination_register = self.env.binary_op_direct(
            left_register, right_register, op
        )
        del self.env[left_register]
        del self.env[right_register]
        return destination_register

    def un_op(self, source_register, op):
        source_register = self.env.promote(source_register)
        destination_register = self.env.unary_op(source_register, op)
        del self.env[source_register]
        return destination_register

    def multi_op_direct(self, source_registers, op):
        logging.debug(f"op={op}")
        source_registers = [self.env.promote(register) for register in source_registers]
        destination_register = self.env.multi_op_direct(source_registers, op)
        for register in source_registers:
            if is_reg(register):
                del self.env[register]
        return destination_register

    def visit_if_exp(self, node, context):
        body = self.visit(node.body, context)
        test = self.visit(node.test, context)
        orelse = self.visit(node.orelse, context)
        return self.multi_op_direct([test, body, orelse], sx.where)

    BIN_OP_DISPATCH = {
        ast.Add: sx.add,
        ast.Sub: sx.subtract,
        ast.Mult: sx.multiply,
        ast.Div: sx.divide,
        ast.FloorDiv: sx.floordiv,
        ast.Mod: sx.mod,
    }

    def matmult(self, n_register, gen, context):
        n = self.env.move_definite_value(n_register)

        for i in range(n):
            yield self.visit(gen, context)

    def visit_set(self, node, context):
        for element in node.elts:
            if isinstance(element, ast.Starred):
                yield from self.visit(element.value, context)
            else:
                yield self.visit(element, context)

    def visit_bin_op(self, node, context):
        logging.debug(f"Evaluating binary operator {node.op}")

        left_register = self.visit(node.left, context)

        op_type = type(node.op)
        if op_type == ast.MatMult:
            return self.matmult(left_register, node.right, context)
        right_register = self.visit(node.right, context)
        try:
            op_func = self.BIN_OP_DISPATCH[op_type]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported binary op {op_type}") from exc

        return self.bin_op(left_register, right_register, op_func)

    COMPARE_OP_DISPATCH = {
        ast.Eq: sx.equal,
        ast.NotEq: sx.not_equal,
        ast.Lt: sx.less,
        ast.Gt: sx.greater,
        ast.GtE: sx.greater_equal,
        ast.LtE: sx.less_equal,
        ast.In: sx.isin,
    }

    def visit_compare(self, node, context):
        logging.debug(f"Binop {node.ops[0]}")
        left = self.visit(node.left, context)
        right = self.visit(node.comparators[0], context)

        op_type = type(node.ops[0])
        try:
            op_func = self.COMPARE_OP_DISPATCH[op_type]
        except KeyError:
            raise NotImplementedError(f"Unsupported compare op {op_type}")

        return self.bin_op(left, right, op_func)

    def visit_bool_op(self, node, context):
        values = [self.visit(v, context) for v in node.values]
        left = values[0]
        right = values[1]
        if isinstance(node.op, ast.And):
            return self.bin_op(left, right, sx.logical_and)
        if isinstance(node.op, ast.Or):
            return self.bin_op(left, right, sx.logical_or)
        raise NotImplementedError(f"Unsupported bool op {type(node.op)}")

    UNARY_OP_DISPATCH = {ast.Not: sx.logical_not, ast.USub: sx.negative}

    def visit_unary_op(self, node, context):
        operand = self.visit(node.operand, context)
        try:
            op_func = self.UNARY_OP_DISPATCH[type(node.op)]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported unary op {type(node.op)}") from exc
        return self.un_op(operand, op_func)

    def rejoin_factors(self, handled, pairs, if_factors, orelse_factors):
        common_factors, unique_to_if, unique_to_orelse = partition_by_version(
            if_factors, orelse_factors
        )

        b_reduced = reduce(operator.mul, unique_to_if)
        c_reduced = reduce(operator.mul, unique_to_orelse)
        for register in b_reduced._values.keys():
            if register in c_reduced._values.keys():
                if register not in handled:
                    pairs.append((register, register, register))
                    logging.debug(f"Passing through {register} in merge.")
                    handled.add(register)

        joined = b_reduced.rejoin(c_reduced, pairs)
        new_factors = common_factors + [joined]
        return new_factors

    def rejoin_vars(self, handled, allocations1, env1, allocations2, env2):
        allocations = {}
        pairs = []

        for var in allocations1:
            val1 = allocations1[var]
            val2 = allocations2.get(var)

            if val2 is None:
                logging.warning(
                    "`%s` is in the True branch of a condition but not the False branch.",
                    var,
                )
                continue

            is_reg1 = is_reg(val1)
            is_reg2 = is_reg(val2)

            if is_reg1 and is_reg2:
                src1, src2 = val1, val2
            elif is_reg1 and not is_reg2:
                src1 = val1
                src2 = env2.allocate_register_with_definite_value(val2)
            elif not is_reg1 and is_reg2:
                src1 = env1.allocate_register_with_definite_value(val1)
                src2 = val2
            else:
                allocations[var] = val1
                continue

            new_reg = new_register_name()
            pairs.append((src1, src2, new_reg))
            logging.debug("Merging %s and %s into %s", src1, src2, new_reg)
            handled.add(src1)
            allocations[var] = new_reg

        return pairs, allocations

    def rejoin_frames(self, frame1, env1, frame2, env2):
        handled = set()
        pairs, allocations = self.rejoin_vars(
            handled, frame1.allocations, env1, frame2.allocations, env2
        )
        new_factors = self.rejoin_factors(handled, pairs, env1.factors, env2.factors)
        new_env = Environment(new_factors)
        assert frame1.conditional_depth == frame2.conditional_depth
        return (
            Frame(frame1.source, new_env, allocations, frame1.conditional_depth - 1),
            new_env,
        )

    def visit_assert(self, node, context):
        condition_register = self.visit(node.test, context)

        # Wasteful as it does construct frame2, env2
        frame, env = context.frame.semi_split(self.env, condition_register)

        context.frame = frame
        self.env = env

    def visit_if(self, node, context):
        old_depth = context.frame.conditional_depth

        condition_register = self.visit(node.test, context)
        # Deterministic
        if not is_reg(condition_register):
            if condition_register:
                for stmt in node.body:
                    self.visit(stmt, context)
            else:
                for stmt in node.orelse:
                    self.visit(stmt, context)
            return

        frame1, env1, frame2, env2 = context.frame.split(self.env, condition_register)
        if frame1 and frame2:
            assert frame1.conditional_depth == frame2.conditional_depth

        # Try 'if' branch
        if frame1:
            context.frame = frame1
            self.env = env1
            for stmt in node.body:
                old = context.frame.conditional_depth
                self.visit(stmt, context)
                assert context.frame.conditional_depth == old
            if not frame2:
                context.frame.conditional_depth -= 1
                return
            frame1 = context.frame
            env1 = self.env

        # assert frame1.conditional_depth == frame2.conditional_depth

        # Try 'else' branch
        if frame2:
            context.frame = frame2
            self.env = env2
            for stmt in node.orelse:
                self.visit(stmt, context)
            if not frame1:
                context.frame.conditional_depth -= 1
                return
            frame2 = context.frame
            env2 = self.env

        assert frame1.conditional_depth == frame2.conditional_depth
        context.frame, env = self.rejoin_frames(frame1, env1, frame2, env2)
        self.env = env
        self.env.tidy()

        new_depth = context.frame.conditional_depth
        assert old_depth == new_depth

    # sometimes use literal eval?
    def eval_as_python(self, node):
        expr = ast.Expression(node)
        codeobj = compile(
            ast.fix_missing_locations(expr), filename="<inline>", mode="eval"
        )
        caller_frame = inspect.currentframe().f_back
        g = caller_frame.f_globals
        l = caller_frame.f_locals
        return eval(codeobj, g, l)

    def builtin_d(self, args):
        if len(args) == 1:
            sides_register = args["arg0"]
            sides = self.env.move_definite_value(sides_register)
            sides = sx.to_py_scalar(sides)
            prob = 1 / sides
            pmf = sx.constant(sides * [prob], dtype=sx.ptype)
            result = self.env.allocate_factor_with_register_with_probability(
                pmf,
                sx.cast(sx.range(1, sides + 1, dtype=sx.int64), sx.int64),
            )
            logging.debug("Created register %s  to contain result of `d`", result)
            return result

        raise ValueError("Unknown args to d")

    def builtin_multiroll(self, args, _context):
        if len(args) == 1:
            ranges_register = args["ranges"]
            i = self.env.find_factor_index(ranges_register)
            factor = self.env.factors[i]
            if is_reg(ranges_register):
                ranges = self.env[ranges_register]
                p = factor.p
                rolls, new_p, indices = sx.multiroll_helper(ranges, p)
                new_values = {}
                for reg, value in factor._values.items():
                    new_values[reg] = sx.gather(value, indices)
                roll_register = new_register_name()
                new_values[roll_register] = rolls
                new_factor = Factor(new_p, new_values)
                self.env.factors[i] = new_factor
                # Can I delete earlier?
                del self.env[ranges_register]
                return roll_register
            raise NotImplementedError()

        raise ValueError("Unknown args to multiroll")

    def builtin_dd(self, node, context):
        sides_register = self.visit(node.args[0], context)
        return self.env.dd(sides_register)

    def get_and_eval_args(self, node, expected, context):
        args = node.args
        arg_registers = [self.visit(arg, context) for arg in node.args]
        keywords = node.keywords
        n_args = len(args)
        arg_dict = dict(zip(expected[:n_args], arg_registers)) | {
            keyword.arg: self.visit(keyword.value, context) for keyword in keywords
        }
        return arg_dict

    def call_extremum(self, node, context, op):
        # print(op.__name__, node)
        args = node.args
        result = None

        for arg in args:
            if isinstance(arg, ast.Starred):
                values = self.visit(arg.value, context)
                for value in values:
                    result = value if result is None else self.bin_op(result, value, op)
            else:
                value = self.visit(arg, context)
                result = value if result is None else self.bin_op(result, value, op)

        return result

    def call_min(self, node, context):
        return self.call_extremum(node, context, sx.minimum)

    def call_max(self, node, context):
        return self.call_extremum(node, context, sx.maximum)

    def call_sum(self, node, context):
        return self.call_extremum(node, context, lambda x, y: x + y)

    # DO call_sum XXX !!!

    def call_move(self, node, context):
        var_name = node.args[0].id
        return context.frame.move(var_name)

    def call_chop(self, _node, _context):
        self.env.chop(1e-10)

    def call_d(self, node, context):
        args = self.get_and_eval_args(node, ["arg0", "arg1"], context)
        return self.builtin_d(args)

    def call_concat(self, node, context):
        if len(node.args) != 2:
            raise TypeError("For now, concat only accepts two arguments.")
        args = node.args[0]
        axis = node.args[1]
        axis = self.eval_as_python(axis)
        if len(args.elts) != 2:
            raise ValueError("For now, concat only accepts two element lists.")
        arg0 = self.visit(args.elts[0], context)
        arg1 = self.visit(args.elts[1], context)
        return self.bin_op_raw(arg0, arg1, lambda x, y: sx.concat([x, y], axis))

    def call_cast(self, node, context):
        if len(node.args) != 2:
            raise TypeError("cast only accepts two arguments.")
        tensor_node = node.args[0]
        dtype = node.args[1]
        # dtype = self.eval_as_python(dtype)
        dtype = self.visit(dtype, context)
        tensor_var = self.visit(tensor_node, context)
        # Need to fix that -1 but that requires knowing # rows.
        tensor_types = {
            "int64": sx.int64,
            "int32": sx.int32,
            "uint64": sx.uint64,
            "uint32": sx.uint32,
            "bool": sx.bool,
        }
        return self.un_op(tensor_var, lambda x: sx.cast(x, tensor_types[dtype]))

    def call_dumpvars(self, _node, _context):
        self.env.dumpvars()
        return

    def call_listvars(self, _node, _context):
        self.env.listvars()
        return

    def call_inline_impl(self, node, context):
        impl_name = node.args[0].id
        args = [self.visit(arg, context) for arg in node.args[1:]]
        return context.globals[impl_name](self, context, *args)

    def call_inline(self, node, context):
        impl_name = node.args[0].id
        return context.globals[impl_name](self, context, *node.args[1:])

    def visit_call(self, node, context):
        func_name = node.func.id  # we assume it's a simple name, not obj.method
        logging.debug("Call to function `%s`.", func_name)

        if func_name in self.call_dispatch_table:
            return self.call_dispatch_table[func_name](node, context)

        # User function call
        if func_name in context.globals:
            source = inspect.getsource(context.globals[func_name])

            source = textwrap.dedent(source)
            function_body = ast.parse(source)

            if self.static_analyse:
                function_body = move_analysis(function_body, self.predefined)
                if self.show_analysis:
                    print(ast.unparse(function_body))

            function_def = function_body.body[0]

            signature = get_signature_from_functiondef(function_def)

            new_frame = Frame(source, self.env, {}, 0)
            args = [self.visit(arg, context) for arg in node.args]
            kwargs = {
                kw.arg: self.visit(kw.value, context)
                for kw in node.keywords
                if kw.arg is not None
            }
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            full_arg_list = bound.arguments.items()

            logging.debug("Calling `%s` with arguments:", func_name)
            for arg_name, arg in full_arg_list:
                logging.debug("%s = %s", arg_name, arg)
                new_frame.bind(arg_name, arg)

            if is_gen_fun(function_def):
                logging.debug("Creating generator function.")
                try:
                    new_context = context.copy()
                    new_context.frame = new_frame
                    gen = self.visit_generator_function(function_body, new_context)
                    return gen
                except FoundReturn as exc:
                    raise ValueError("return in generator") from exc

                raise ValueError("Missing return in gen")

            new_context = context.copy()
            new_context.frame = new_frame

            try:
                self.visit(function_body, new_context)
            except FoundReturn as e:
                context.frame.environment = self.env
                return_register = e.value_node[0]
                return return_register

            raise ValueError("Missing return")

        raise NameError(f"Function '{func_name}' not defined.")

    def visit_as_generator_Delete(self, node, context):
        self.visit_delete(node, context)
        yield from ()

    def visit_delete(self, node, context):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                context.frame.delete_var_and_register(self.env, var_name)

            elif isinstance(target, ast.Attribute):
                raise ValueError(
                    f"Deleting attribute '{target.attr}' not supported yet."
                )

            elif isinstance(target, ast.Subscript):
                raise ValueError("Deleting subscripted values isn't supported yet.")

    def multiroll_d(self, range_expr, context):
        def one_range(slice_expr):
            lower = self.visit(slice_expr.lower, context)
            upper = self.visit(slice_expr.upper, context)
            step = self.visit(slice_expr.step, context) if slice_expr.step else 1
            return self.multi_op_direct(
                [lower, upper, step], lambda *x: outer_product(x)[..., None, :]
            )

        if isinstance(range_expr, ast.Slice):
            range_ = one_range(range_expr)
            return self.builtin_multiroll({"ranges": range_}, context)

        if isinstance(range_expr, ast.Tuple):
            ranges = [
                self.multi_op_direct(
                    [
                        self.visit(s.lower, context),
                        self.visit(s.upper, context),
                        self.visit(s.step, context) if s.step else 1,
                    ],
                    lambda *x: outer_product(x),
                )
                for s in range_expr.elts
            ]

            all_rolls = self.multi_op_direct(ranges, lambda *x: outer_product(x))

            return self.builtin_multiroll({"ranges": all_rolls}, context)

        raise ValueError("Bad call to `multiroll_d`.")

    def visit_subscript(self, node, context):
        if isinstance(node.value, ast.Name) and node.value.id == "d":
            return self.multiroll_d(node.slice, context)

        obj = self.visit(node.value, context)
        slice_node = node.slice

        node = slice_node
        if isinstance(slice_node, ast.Slice):
            try:
                node = slice_node.lower
                lower = ast.literal_eval(node) if node else None
                node = slice_node.upper
                upper = ast.literal_eval(node) if node else None
                node = slice_node.step
                step = ast.literal_eval(node) if node else None
            except:
                raise InterpreterError(
                    "dice9 only supports literals on slice ranges so as to avoid divergence issues.",
                    frame=context.frame,
                    node=node,
                )
            aslice = slice(lower, upper, step)
            return self.un_op(obj, lambda x: x[:, aslice])

        if isinstance(slice_node, ast.Index):
            print(f"index {ast.unparse(slice_node)}")

        elif isinstance(slice_node, ast.ExtSlice):
            print(f"ext slice {ast.unparse(slice_node)}")

        elif isinstance(slice_node, ast.Tuple):
            print(f"tuple {ast.unparse(slice_node)}")
            ss = ()
            for i in slice_node.elts:
                if isinstance(i, ast.Constant):
                    ss += (i.value,)
                elif isinstance(i, ast.Slice):
                    try:
                        node = i.lower
                        lower = ast.literal_eval(node) if node else None
                        node = i.upper
                        upper = ast.literal_eval(node) if node else None
                        node = i.step
                        step = ast.literal_eval(node) if node else None
                    except:
                        raise InterpreterError(
                            "dice9 only supports literals on slice ranges so as to avoid divergence issues.",
                            frame=context.frame,
                            node=node,
                        )
                    aslice = slice(lower, upper, step)

                    ss += (aslice,)
            lifted_slices = (slice(None, None, None),) + ss
            return self.un_op(obj, lambda x: x[lifted_slices])

        elif isinstance(slice_node, ast.Constant):
            return self.un_op(obj, lambda x: x[:, slice_node.value])

        else:
            slice_array = self.visit(slice_node, context)
            result = self.bin_op_raw(obj, slice_array, sx.subscript)
            return result

        raise NotImplementedError("Subscript/index not supported yet")


class DupTracker(ast.NodeVisitor):
    def __init__(self, predefined):
        self.last_moves = {}  # list of moves that might block a read
        self.moves = {}  # is the assigment a move?
        self.predefined = predefined

    def needs(self, var_name):
        for move_node in self.last_moves[var_name]:
            self.moves[move_node] = False

    def provides(self, var_name):
        self.last_moves[var_name] = set()

    def visit_statements(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

    def visit_FunctionDef(self, node):
        if node.args.vararg or node.args.kwarg:
            logging.info("Can't perform static analysis of functions with vararg.")
        else:
            for arg in node.args.args:
                # indicate that arg is present but never needs to
                # have its move removed (as it has no move).
                self.last_moves[arg.arg] = {}

            self.visit_statements(node.body)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # print("visiting load at", node, ast.unparse(node))
            name = node.id
            if name in self.predefined:
                return
            if name in self.last_moves:
                if not self.last_moves[name]:
                    pass
                else:
                    # Need to fix move
                    self.needs(name)
            else:
                raise ValueError(f"Name {name} not assigned")
            # print("Adding",name,"at",node,"as a last move")
            self.last_moves[name] = {node}
            if node in self.moves:
                self.moves[node] = self.moves[node] & True
            else:
                self.moves[node] = True
                # print(name,"at",node,"is a move for now")

    def visit_Assign(self, node):
        self.visit(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.provides(target.id)
            elif isinstance(target, ast.Subscript):
                # In a statement like a[i] = ...
                # we read to a as well as writing it.
                if isinstance(target.value, ast.Name):
                    self.visit(target.slice)
                    self.needs(target.value.id)
                    self.provides(target.value.id)
                else:
                    raise NotImplementedError(
                        "Only simple subscript assignment supported."
                    )
            else:
                raise NotImplementedError("Only simple subscript assignment supported.")

    def visit_Return(self, node):
        self.visit(node.value)

    def visit_AugAssign(self, node):
        self.visit(node.value)

        if isinstance(node.target, ast.Name):
            # read first

            name = node.target.id
            if name in self.last_moves:
                if not self.last_moves[name]:
                    pass
                else:
                    # Need to fix move
                    self.needs(name)
            else:
                raise ValueError(f"Name {name} not assigned")

            # now write
            self.provides(name)

        # self.generic_visit(node)

    def visit_If(self, node):
        self.visit(node.test)

        last_moves_orelse = {k: set(v) for k, v in self.last_moves.items()}
        moves_orelse = dict(self.moves)

        self.visit_statements(node.body)

        last_moves_if = {k: set(v) for k, v in self.last_moves.items()}
        moves_if = dict(self.moves)

        self.last_moves = last_moves_orelse
        self.moves = moves_orelse

        self.visit_statements(node.orelse)

        ks = set(self.last_moves.keys()) | set(last_moves_if.keys())
        new_last_moves = {}
        for k in ks:
            if k in self.last_moves and k in last_moves_if:
                new_last_moves[k] = self.last_moves[k] | last_moves_if[k]
            elif k in self.last_moves:
                new_last_moves[k] = self.last_moves[k]
            elif k in last_moves_if:
                new_last_moves[k] = last_moves_if[k]
        self.last_moves = new_last_moves

        ks = set(self.moves.keys()) | set(moves_if.keys())
        new_moves = {}
        for k in ks:
            if k in self.moves and k in moves_if:
                new_moves[k] = self.moves[k] & moves_if[k]
            elif k in self.moves:
                new_moves[k] = self.moves[k]
            elif k in moves_if:
                new_moves[k] = moves_if[k]
        self.moves = new_moves

    def visit_GeneratorExp(self, node):
        self.visit(node.generators[0].iter)
        loop_var = node.generators[0].target.id

        # First iteration
        self.provides(loop_var)
        self.visit(node.elt)

        # Second iteration
        self.provides(loop_var)
        self.visit(node.elt)

    def visit_ListComp(self, node):
        self.visit(node.generators[0].iter)
        element_name = node.generators[0].target.id

        # First iteration
        self.provides(element_name)
        self.visit(node.elt)

        # Second iteration
        self.provides(element_name)
        self.visit(node.elt)

    def visit_For(self, node):
        self.visit(node.iter)

        loop_var = node.target.id
        self.provides(loop_var)

        # First iteration
        self.provides(loop_var)
        self.visit_statements(node.body)

        # Second iteration
        self.provides(loop_var)
        self.visit_statements(node.body)


class DupInserter(ast.NodeTransformer):
    def __init__(self, moves, predefined):
        self.moves = moves
        self.predefined = predefined

    def visit_Name(self, node):
        if node.id not in self.predefined:
            if (
                isinstance(node.ctx, ast.Load)
                and node in self.moves
                and self.moves[node]
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="move", ctx=ast.Load()),
                        args=[node],
                        keywords=[],
                    ),
                    node,
                )
        return node


def move_analysis(tree, predefined):
    try:
        tracker = DupTracker(predefined)
        tracker.visit(tree)

        inserter = DupInserter(tracker.moves, predefined)
        transformed_tree = inserter.visit(tree)
        ast.fix_missing_locations(transformed_tree)

        return transformed_tree
    except Exception as exc:
        raise ValueError("Error during move analysis.") from exc


def report_error(node, source):
    source_lines = source.splitlines()
    if node:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        if lineno is not None:
            line = source_lines[lineno - 1]
            logging.error(f"Exception while interpreting line %s:", lineno)
            logging.error(f"%s: %s", lineno, line.rstrip())
            if col_offset is not None:
                logging.error(" " * (len(str(lineno)) + 2 + col_offset) + "^")
    raise


def run(
    f,
    *args,
    normalize=False,
    static_analyse=True,
    squeeze=False,
    show_analysis=False,
    modules=None,
):
    """
    Interpret and execute a Python function using the dice9 interpreter framework.

    The function `f` should be written in a restricted subset of Python suitable for
    AST-based interpretation. This function extracts the source of `f`, parses it into
    an AST, applies static move analysis (if enabled), and then executes it using a
    custom interpreter.

    Parameters:
        f (function): The function to interpret.
        *args: Positional arguments to pass to the interpreted function.
        normalize (bool): If True, normalize the returned probability distribution.
        static_analyse (bool): If True, perform static move/dup analysis and transform the AST.
        squeeze (bool): If True, flatten single-element tuple return values.
        show_analysis (bool): If True, print the transformed AST after analysis.
        modules (list[module], optional): List of modules whose globals should be available
            to the interpreted function.

    Returns:
        dict or list:
            If the function returns registers, returns a dictionary mapping outcomes
            to probabilities (a PMF). If not, returns the raw Python values.

    Raises:
        TypeError: If the number of arguments doesn't match the function signature.
        InterpreterError: If interpretation fails due to unsupported constructs or runtime errors.
        ValueError: If static analysis fails or other invalid inputs are provided.
    """
    logging.debug("Starting execution with function `%s`.", f)
    source = inspect.getsource(f)
    source = textwrap.dedent(source)
    parsed = ast.parse(source)

    env = Environment([])
    interpreter = Interpreter(parsed, env, modules=modules)

    seen = set()
    context = {}
    if modules is None:
        modules = [problib]
    for module in modules:
        context = context | module.__dict__
    context = context | f.__globals__.copy()

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

    predefined = list(context.keys())
    predefined = predefined + [
        "d",
        "range",
        "constant",
        "concat",
        "sort",
        "max",
        "list",
        "sum",
        "cast",
        "cumsum",
        "dd",
        "__inline__",
        "min",
        "reduce_all",
        "reduce_any",
        "print",
        "reduce_sum",
        "reduce_max",
        "__inline_impl__",
        "__inline_with_prob__",
        "__inline_impl_star__",
        "__listvars__",
    ]
    interpreter.predefined = predefined
    interpreter.static_analyse = static_analyse
    interpreter.show_analysis = show_analysis

    if static_analyse:
        parsed = move_analysis(parsed, predefined)

        if show_analysis:
            print("transformed code:")
            print(ast.unparse(parsed))

    frame = Frame(source, interpreter.env, {}, 0)
    context = Context(context, frame)

    function_name = parsed.body[0].name
    expected_num_args = len(parsed.body[0].args.args)
    num_args = len(args)
    if expected_num_args > num_args:
        missing_argument = parsed.body[0].args.args[num_args].arg
        raise TypeError(
            f"Missing positional argument to function '{function_name}': '{missing_argument}'."
        )
    if expected_num_args < num_args:
        raise TypeError(
            f"Extra argument to '{function_name}' at position {expected_num_args + 1}."
        )
    for arg_name, user_arg in zip(parsed.body[0].args.args, args):
        logging.debug("Assign %s = %s", arg_name.arg, user_arg)
        frame.bind(arg_name.arg, user_arg)

    try:
        interpreter.visit(parsed, context)
    except FoundReturn as e:
        values = e.value_node

        if isinstance(values, list):
            log = any(is_reg(t) for t in values)
            if not log:
                return values

        logging.debug("return `%s`", values)

        pmf = interpreter.env.distribution(values, normalize)
        if squeeze:
            pmf = {key[0]: value for key, value in pmf.items()}
        return pmf
    except InterpreterError as exc:
        node = exc.node if exc.node else interpreter.node
        source = exc.frame.source
        report_error(node, source)

    return interpreter.env
