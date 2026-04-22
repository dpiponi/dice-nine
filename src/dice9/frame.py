import logging
import types

from .factor import Factor, is_reg, Register
from .environment import Environment
from .exceptions import InterpreterError


class Frame:

    """
    A `Frame` is essentially the mapping from variable names to
    values, usually indirected via registers.

    The `Frame` is also used to tie back together registers that
    are split in conditional branches.

    A `Frame` isn't tied to any particular `Environment` which is
    where the values of registers are stored.
    """

    def __init__(self, semiring, source, allocations, conditional_depth):
        self.semiring = semiring
        self.source = source
        self._allocations = allocations if allocations else {}
        self.conditional_depth = conditional_depth

    def copy(self):
        return Frame(
            self.semiring, self.source, self._allocations.copy(), self.conditional_depth
        )

    def bind(self, name, value):#, overwrite=False):
        # if overwrite:
        #     context.frame.delete_var_if_defined(name)
        logging.debug("Binding `%s` = %s.", name, value)
        self._allocations[name] = value

    def __repr__(self):
        return (
            "Frame<"
            + ", ".join(f"'{k}': {v}" for k, v in self._allocations.items())
            + ">"
        )

    def show(self, env):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        if not all(isinstance(v, Register) for v in self._allocations.values()):
            table = Table(box=box.SIMPLE_HEAVY)
            table.add_column("var")
            table.add_column("value")
            for k, v in self._allocations.items():
                if not isinstance(v, Register):
                    table.add_row(str(k), str(v))

            console = Console()
            console.print(Panel(table, expand=False))

        env.show(self._allocations)

    def __getitem__(self, var_name):
        return self._allocations[var_name]

    # Creates new independent variable.
    def allocate_register(self, env, new_var_name, value):
        """
        Create a new independent random variable in the specified environment.

        Args:
            env: The environment in which the new register lives.
            new_var_name: the name of the new variable.
            value: the value of the new register and variable.
        """
        
        reg = env.allocate_register_with_definite_value(value)
        self._allocations[new_var_name] = reg
        self.allocate_register
        return reg

    # Allocates new register to variable in same factor as previously
    # existing register.
    def allocate_in_same_factor_as_register(self, env, register, new_var_name, value):
        """
        Create a new variable in the specified environment in the same
        factor as the given variable.

        Args:
            env: The environment in which the new register lives.
            register: The register whose factor we want the new register in.
            new_var_name: the name of the new variable.
            value: the value of the new register and variable.
        """

        factor = env.find_factor(register)
        new_register = Register.new()
        factor[new_register] = value
        self._allocations[new_var_name] = new_register
        return new_register

    def allocate_register_with_definite_value(self, env, value):
        return env.allocate_register_with_definite_value(value)

    def delete_value(self, env, value):
        logging.debug("Deleting `%s`.", value)

        match value:
            case Register():
                del env[value]
            case tuple():
                for x in value:
                    self.delete_value(env, x)
            case types.GeneratorType():
                del value
            case _:
                pass

    def duplicate_value(self, env, value):
        match value:
            case Register():
                return env.duplicate_register(value)
            case tuple():
                return tuple(self.duplicate_value(env, x) for x in value)
            case types.GeneratorType():
                raise InterpreterError("Can't copy a stream.")
            case _:
                return value

    def copy_of_var(self, env, var_name):
        register = self[var_name]
        new_register = self.duplicate_value(env, register)
        logging.debug(
            "Reading from variable `%s` (%s) into %s.",
            var_name,
            register,
            new_register,
        )
        return new_register

    def split(self, env, condition_register, keep_condition=False):
        env1, env2 = env.split(condition_register, keep_condition=keep_condition)
        logging.debug(
            "Split Env(%s) into Env(%s) and Env(%s).",
            id(env) if env else None,
            id(env1) if env1 else None,
            id(env2) if env2 else None,
        )

        def new_frame(env):
            return (
                Frame(
                    self.semiring,
                    self.source,
                    self._allocations.copy(),
                    self.conditional_depth + 1,
                )
                if env
                else None
            )

        frame1 = new_frame(env1)
        frame2 = new_frame(env2)

        return frame1, env1, frame2, env2

    def semi_split(self, env, condition_register, keep_condition=False):
        env = env.semi_split(condition_register, keep_condition=keep_condition)
        frame = Frame(
            self.semiring, self.source, self._allocations, self.conditional_depth
        )
        return frame, env

    def has_var(self, var_name):
        return var_name in self._allocations

    def delete_var_and_register(self, env, var_name):
        value = self._allocations[var_name]
        logging.debug(
            "Deleting variable `%s` currently bound to `%s`.", var_name, value
        )
        self.delete_value(env, value)
        del self._allocations[var_name]

    def delete_var_if_defined(self, env, var_name):
        if self.has_var(var_name):
            self.delete_var_and_register(env, var_name)

    def move(self, var_name):
        if var_name in self._allocations:
            register = self._allocations[var_name]
            del self._allocations[var_name]
            return register
        raise InterpreterError(f"Variable `{var_name}` not found.")

    def delete_all(self, env):
        for var, register in self._allocations.items():
            logging.debug(
                "Deleting variable `%s` currently bound to %s.", var, register
            )
            self.delete_value(env, register)
        self._allocations = {}

    def get(self, env, var_name):
        return env[self._allocations[var_name]]

    def assign(self, env, var_name, value, clobber):
        if clobber:
            self.delete_var_if_defined(env, var_name)

        if is_reg(value):
            self.assign_move(env, var_name, value)
        else:
            self.bind(var_name, value)

        logging.debug(f"Assigned {var_name} = reg {value}")

    # Sets var to point at new register deleting old register
    # if var pointed at it.
    def assign_move(self, env, var_name, new_register):
        if var_name in self._allocations:
            old_register = self._allocations[var_name]
            del env[old_register]
        self._allocations[var_name] = new_register
        return new_register

    @staticmethod
    def rejoin_vars(handled, allocations1, env1, allocations2, env2):
        allocations = {}
        pairs = []

        common_vars = set(allocations1.keys()) & set(allocations2.keys())
        logging.debug("Variables to merge: %s.", common_vars)
        for var in common_vars:
            val1 = allocations1[var]
            val2 = allocations2[var]

            if val2 is None:
                logging.debug(
                    "`%s` is in the True branch of a condition but not the False branch.",
                    var,
                )
                continue

            is_reg1, is_reg2 = is_reg(val1), is_reg(val2)

            if not is_reg1 and not is_reg2 and val1 == val2:
                allocations[var] = val1
                continue

            src1 = env1.promote(val1)
            src2 = env2.promote(val2)

            new_reg = Register.new()
            pairs.append((src1, src2, new_reg))
            logging.debug(
                "Merging %s in env #%s and %s in env #%s into %s"    ,
                src1,
                id(env1),
                src2,
                id(env2),
                new_reg,
            )
            handled.add(src1)
            allocations[var] = new_reg

        return pairs, allocations

    @staticmethod
    def rejoin_frames(frame1, env1, frame2, env2):
        handled = set()

        # Rejoin named variables
        pairs, allocations = Frame.rejoin_vars(
            handled, frame1._allocations, env1, frame2._allocations, env2
        )
        # Rejoin registers not named in current frme
        new_factor = Factor.rejoin_factors(handled, pairs, env1.factors, env2.factors)

        new_env = Environment(frame1.semiring, [new_factor])
        return (
            Frame(
                frame1.semiring,
                frame1.source,
                allocations,
                frame1.conditional_depth - 1,
            ),
            new_env,
        )
