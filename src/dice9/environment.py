from functools import reduce
import logging
import operator
from typing import Any, Dict

import dice9.backends.numpy_impl as sx
from .algebra import Semiring, PartialField
from .factor import Factor, is_reg, Register


class Environment:

    def __init__(self, semiring : Semiring, factors):
        self.semiring = semiring
        self.factors = factors
        logging.debug("Created Env(%s).", id(self))

    def __repr__(self):
        return "Environment<" + ", ".join(f"{f}" for f in self.factors) + ">"

    def show(self, names={}):
        from rich.columns import Columns
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        c = Columns([Panel(f.rich(names)) for f in self.factors])
        console.print(Panel(c))


    def registers(self):
        return [factor._values.keys() for factor in self.factors]

    def promote(self, value):
        return (value if is_reg(value) else
                self.allocate_register_with_definite_value(value))

    def allocate_register_with_definite_value(self, value) -> Register:
        """
        Create a new register from specified value. It is created
        in a new factor so it becomes a new "independent" variable
        with probability 1.

        Args:
            value: the value of the new register.

        Returns:
            The new register.
        """

        new_register = Register.new()
        p = self.semiring.ones((1,))
        converted = sx.convert_to_tensor(value)
        tensor = sx.stack([converted])
        factor = Factor(self.semiring, p, {new_register: tensor})
        self.add_factor(factor)
        logging.debug("Created register %s in env #%s to contain `%s`.",
                      new_register, id(self), value)
        return new_register

    def allocate_factor_with_register_with_probability(self, p, value) -> Register:
        new_register = Register.new()
        factor = Factor(self.semiring, p, {new_register: value})
        self.add_factor(factor)
        return new_register

    def add_factor(self, factor: Factor):
        self.factors.append(factor)

    def __contains__(self, register: Register) -> bool:
        """
        Checks whether any factor in the environment contains the given
        register.

        Args:
            register: the register to check for.

        Returns:
            True or False depending on whether register is present.
        """

        return any(register in factor for factor in self.factors)

    def find_factor(self, register):
        for factor in self.factors:
            if register in factor:
                return factor
        raise ValueError(
            f"Register {register} not found in environment Env({id(self)})")

    def find_factor_index(self, register):
        for i, factor in enumerate(self.factors):
            if register in factor:
                return i
        raise ValueError(f"Register {register} not found in environment")

    def duplicate_register(self, register: Register) -> Register:
        new_register = Register.new()
        factor = self.find_factor(register)
        factor[new_register] = factor[register]
        return new_register

    def __delitem__(self, register):
        logging.debug("Deleting register %s and marginalizing.", register)
        factor = self.find_factor(register)
        if sx.shape(factor._values[register])[0] == 0:
            del factor[register]
            return
        min_value = sx.reduce_min(factor._values[register], 0)
        max_value = sx.reduce_max(factor._values[register], 0)
        del factor[register]
        if sx.reduce_all(max_value == min_value):
            logging.debug("Skipping marginalization.")
            return
        factor.marginalize()
        logging.debug("Deleted register %s and marginalized.", register)

    def __getitem__(self, register):
        factor = self.find_factor(register)
        return factor[register]

    def __setitem__(self, register, value):
        factor = self.find_factor(register)
        factor[register] = value

    def split(self, condition, keep_condition=False):
        env1 = Environment(self.semiring, [])
        env2 = Environment(self.semiring, [])
        for factor in self.factors:
            if condition in factor:
                f1, f2 = factor.split(factor[condition])
                if f1:
                    env1.factors.append(f1)
                    if not keep_condition:
                        del f1[condition]
                if f2:
                    env2.factors.append(f2)
                    if not keep_condition:
                        del f2[condition]
            else:
                env1.factors.append(factor)
                env2.factors.append(factor.clone())
        return env1 if f1 else None, env2 if f2 else None

    def semi_split(self, condition, keep_condition=False):
        env = Environment(self.semiring, [])
        for factor in self.factors:
            if condition in factor:
                split_factor = factor.semi_split(factor[condition])
                env.factors.append(split_factor)
                if not keep_condition:
                    del split_factor[condition]
            else:
                env.factors.append(factor)
        return env

    def listvars(self):
        print([[(key, value.shape, value)
                for key, value in factor._values.items()]
               for factor in self.factors])

    def distribution(self, vars, normalize=False) -> Dict:
        s = self.semiring

        vars_was_tuple = isinstance(vars, tuple)
        vars = vars if vars_was_tuple else (vars,)

        for factor in self.factors:
            factor_vars = list(factor._values.keys())
            for var in factor_vars:
                if var not in vars:
                    del factor[var]
            factor.marginalize()
        factor = reduce(operator.mul, self.factors)

        pdf = {}
        probs = factor.p
        if normalize:
            assert isinstance(s, PartialField)
            total_prob = s.add_reduce(probs)
            probs = s.divide(probs, total_prob)
        for i in range(s.len(probs)):
            if vars_was_tuple:
                t = tuple(sx.to_python(factor[var][i]) for var in vars)
            else:
                t = sx.to_python(factor[vars[0]][i])
            pdf[t] = s.as_scalar(s.get(probs, i))

        return pdf

    def common_factor(self, registers: list[Register]) -> Factor:
        shared_factors = []
        unshared_factors = []

        for factor in self.factors:
            if any(register in factor for register in registers):
                shared_factors.append(factor)
            else:
                unshared_factors.append(factor)

        shared_factor = reduce(operator.mul, shared_factors)
        self.factors = unshared_factors + [shared_factor]

        return shared_factor

    def move_definite_register(self, register: Register):
        factor = self.find_factor(register)
        values = factor[register]
        if len(values) == 1:
            logging.debug("Using definite value from %s.", values)
            del self[register]
            return values[0]

        min_value = sx.reduce_min(values, 0)
        max_value = sx.reduce_max(values, 0)
        if min_value == max_value:
            logging.debug("Using definite value from %s with zero range.",
                          values)
            del self[register]
            return min_value

        raise ValueError(f"Value {values} is not definite.")

    def move_register(self, register: Register) -> Any:
        factor = self.find_factor(register)
        values = factor[register]
        self.__delitem__(register)
        return values

    def multi_op_general(self, registers, pre_op, post_op, op):
        destination = Register.new()
        logging.debug("Created new register %s to contain result of %s",
                      destination, op)
        new_factor = self.common_factor(registers)
        values = [pre_op(new_factor[register]) for register in registers]
        new_factor[destination] = post_op(op(*values))
        return destination

    def multi_op(self, registers, op):
        def first_to_last(x):
            return sx.moveaxis(x, 0, -1)

        def last_to_first(x):
            return sx.moveaxis(x, -1, 0)

        return self.multi_op_general(registers, first_to_last, last_to_first,
                                     op)

    def multi_op_direct(self, registers, op):
        return self.multi_op_general(registers, lambda x: x, lambda x: x, op)

    def unary_op(self, source, op):
        return self.multi_op_direct([source], op)

    def binary_op_direct(self, left, right, op):
        return self.multi_op_direct([left, right], op)

    def dd(self, n):
        destination = Register.new()
        factor = self.find_factor(n)
        probs, rolls, indices = sx.dd_helper(factor[n])
        factor.p = self.semiring.mul(sx.gather(factor.p, indices), probs)
        for var in factor._values:
            factor._values[var] = sx.gather(factor._values[var], indices)
        factor._values[destination] = rolls
        return destination
