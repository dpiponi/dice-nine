from functools import reduce

from dice9.config import sx
import dice9.config as config
import logging

from .factor import Factor, new_register_name, Register, is_reg, check_is_reg


class Environment:
    def __init__(self, factors):
        self.factors = factors
        # register_to_factor is for when we switch to Frames
        self.register_to_factor = {}

    def chop(self, eps=0):
        for factor in self.factors:
            factor.chop(eps)

    def __repr__(self):
        return "Environment<" + ", ".join(f"{f}" for f in self.factors) + ">"

    def tidy(self):
        self.factors = list(filter(lambda f: not f.is_trivial(), self.factors))

    def registers(self):
        return [factor._values.keys() for factor in self.factors]

    def promote(self, value):
        return (
            value
            if is_reg(value)
            else self.allocate_register_with_definite_value(value)
        )

    def allocate_register_with_definite_value(self, value):
        new_register = new_register_name()
        p = sx.constant([1.0], dtype=sx.ptype)
        logging.debug(f"val={value}")
        # A bit gross
        if type(value) == str:
            tensor = sx.stack([sx.convert_to_tensor(value)])
        else:
            converted = sx.convert_to_tensor(value)#, dtype=sx.int64)
            # XXX come back to this as we need to handle more general types.
            converted = sx.cast(converted, sx.int64)
            tensor = sx.stack([converted])
        # tensor = sx.convert_to_tensor([value], dtype=sx.int64)
        factor = Factor(p, {new_register: tensor})
        self.add_factor(factor)
        return new_register

    def allocate_factor_with_register_with_probability(self, p, value):
        new_register = new_register_name()
        factor = Factor(p, {new_register: value})
        self.add_factor(factor)
        return new_register

    def add_factor(self, factor):
        self.factors.append(factor)
        self.register_to_factor |= {name: factor for name in factor._values.keys()}

    def push(self):
        for factor in self.factors:
            factor.push()

    def pop(self):
        for factor in self.factors:
            factor.pop()

    def __contains__(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register")
        for factor in self.factors:
            if register in factor:
                return True
        return False

    def find_factor(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register")
        for factor in self.factors:
            if register in factor:
                return factor
        raise ValueError(f"Register {register} not found in environment")

    def find_factor_index(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register")
        for i, factor in enumerate(self.factors):
            if register in factor:
                return i
        raise ValueError(f"Register {register} not found in environment")

    def duplicate_register(self, register):
        new_register = new_register_name()
        factor = self.find_factor(register)
        factor[new_register] = factor[register]
        self.register_to_factor[new_register] = factor
        return new_register

    # XXX Handle case when all variables deleted from a factor.
    # This requires replaced the p part with a single element
    # total probability.
    # `marginalize` can do this work.
    # @todo
    # Might want to make all marginalizations explicit.
    def __delitem__(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register")
        logging.debug(f"Deleting register {register} and marginalizing.")
        factor = self.find_factor(register)
        del factor[register]
        factor.marginalize()
        if factor.is_trivial():
            self.tidy()

    def raw_del(self, register):
        check_is_reg(register)
        factor = self.find_factor(register)
        del factor[register]

    def __getitem__(self, register):
        check_is_reg(register)
        factor = self.find_factor(register)
        return factor[register]

    def get_with_probability(self, register):
        check_is_reg(register)
        factor = self.find_factor(register)
        return factor.p, factor[register]

    def __setitem__(self, register, value):
        check_is_reg(register)
        factor = self.find_factor(register)
        factor[register] = value

    # Only the split factor is "deep"
    # Other factors are shared :(
    # Condition is a register.
    def split(self, condition, keep_condition=False):
        check_is_reg(condition)
        env1 = Environment([])
        env2 = Environment([])
        for i, factor in enumerate(self.factors):
            if condition in factor:  # use find factor
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
                # This is a bug! (maybe fixed now)
                env2.factors.append(factor.clone())
                # env2.factors.append(factor)
        return env1 if f1 else None, env2 if f2 else None

    def semi_split(self, condition, keep_condition=False):
        check_is_reg(condition)
        env = Environment([])
        for i, factor in enumerate(self.factors):
            if condition in factor:
                f = factor.semi_split(factor[condition])
                env.factors.append(f)
                if not keep_condition:
                    del f[condition]
            else:
                env.factors.append(factor)
        return env

    def pretty_print(self):
        for factor in self.factors:
            factor.pretty_print()

    def dumpvars(self):
        print("Variable list")
        print("-------------")
        factor_number = 0
        for factor in self.factors:
            col1 = ["var"]
            col2 = ["shape"]
            for name, value in factor._values.items():
                col1.append(str(var_name_and_frame(name)))
                col2.append(str(sx.shape(value)))

            width_a = max(len(str(x)) for x in col1)
            width_b = max(len(str(x)) for x in col2)

            # 2) Print each pair, right-justified in its column
            factor_heading = f"Factor {factor_number}"
            print(factor_heading)
            print(max(width_a + 1 + width_b, len(factor_heading)) * "=")
            for x, y in zip(col1, col2):
                print(f"{x:>{width_a}} {y:>{width_b}}")
            print(max(width_a + 1 + width_b, len(factor_heading)) * "=")
            factor_number += 1

    def listvars(self):
        print("env.listvars")
        print(
            [
                [(key, value.shape, value) for key, value in factor._values.items()]
                for factor in self.factors
            ]
        )

    def distribution(self, vars, normalize=False):
        factors = []
        for factor in self.factors:
            if set(vars) & set(factor._values.keys()):
                factors.append(factor)
        factor = reduce(lambda x, y: x * y, factors)
        keys = list(factor._values.keys())
        for var in keys:
            if var not in vars:
                del factor[var]
        factor.marginalize()

        pdf = {}
        probs = factor.p
        if normalize:
            total_prob = sx.reduce_sum(probs, 0)
            probs = probs / total_prob
        for i in range(len(probs)):
            t = tuple(sx.to_python(factor[var][i]) for var in vars)
            pdf[t] = sx.to_python(probs[i])

        return pdf

    def common_factor2(self, left_register, right_register):
        for i, factor in enumerate(self.factors):
            if left_register in factor:
                left_index = i
            if right_register in factor:
                right_index = i

        indices = sorted([left_index, right_index])
        new_factors = []

        if indices[1] > indices[0]:
            right_factor = self.factors.pop(indices[1])
            left_factor = self.factors.pop(indices[0])
            new_factor = left_factor * right_factor
            self.add_factor(new_factor)
            new_factors.append(new_factor)
        else:
            # They're conveniently already in same factor.
            new_factor = self.factors[indices[0]]
            new_factors.append(new_factor)

        for register in new_factor._values.keys():
            self.register_to_factor[register] = new_factor

        # self.factors = new_factors
        print(f"FACTORS: {self.factors},{new_factors}")
        return new_factor

    def common_factor(self, registers):
        shared_factors = []
        unshared_factors = []

        for factor in self.factors:
            if any(register in factor for register in registers):
                shared_factors.append(factor)
            else:
                unshared_factors.append(factor)

        shared_factor = reduce(lambda x, y: x * y, shared_factors)

        for register in shared_factor._values.keys():
            self.register_to_factor[register] = shared_factor

        self.factors = unshared_factors + [shared_factor]

        return shared_factor

    def get_value(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register.")
        factor = self.find_factor(register)
        return factor[register]

    def get_definite_value(self, register):
        if not is_reg(register):
            raise ValueError(f"{register} is not a register.")
        factor = self.find_factor(register)
        values = factor[register]
        if len(values) == 1:
            return values[0]
        else:
            raise ValueError(f"Value {values} is not definite.")

    def move_definite_value(self, register):
        if not is_reg(register):
            return register
        factor = self.find_factor(register)
        values = factor[register]
        if len(values) == 1:
            self.__delitem__(register)
            return values[0]
        else:
            raise ValueError(f"Value {values} is not definite.")

    def move_value(self, register):
        if not is_reg(register):
            return register
        factor = self.find_factor(register)
        values = factor[register]
        self.__delitem__(register)
        return values

    def copy_register(register):
        factor = self.find_factor(register)
        return factor.copy_register(register)

    # Environment
    def unary_op(self, source_register, op):
        return self.multi_op_direct([source_register], op)

    # Environment
    def binary_op_direct(self, left_register, right_register, op):
        if not is_reg(left_register):
            raise ValueError(f"{left_register} is not a register.")
        if not is_reg(right_register):
            raise ValueError(f"{right_register} is not a register.")
        destination_register = new_register_name()
        logging.debug(
            f"Created new register {destination_register} to contain result of {op}"
        )
        new_factor = self.common_factor([left_register, right_register])
        left_value = new_factor[left_register]
        right_value = new_factor[right_register]
        new_value = op(left_value, right_value)
        new_factor[destination_register] = new_value
        self.register_to_factor[destination_register] = new_factor
        return destination_register

    # Environment. Returns register.
    def binary_op(self, left_register, right_register, op):
        return self.multi_op([left_register, right_register], op)

    def multi_op(self, registers, op):
        def first_to_last(x):
            rank = sx.rank(x)
            perm = list(range(1, rank)) + [0]
            return sx.transpose(x, perm)

        def last_to_first(x):
            rank = sx.rank(x)
            # build permutation [rank-1, 0, 1, ..., rank-2]
            perm = [rank - 1] + list(range(0, rank - 1))
            return sx.transpose(x, perm)

        destination_register = new_register_name()
        logging.debug(
            f"Created new register {destination_register} to contain result of {op}"
        )
        new_factor = self.common_factor(registers)
        values = [first_to_last(new_factor[register]) for register in registers]
        new_value = op(*values)
        new_factor[destination_register] = last_to_first(new_value)
        self.register_to_factor[destination_register] = new_factor
        return destination_register

    def multi_op_direct(self, registers, op):
        destination_register = new_register_name()
        logging.debug(
            f"Created new register {destination_register} to contain result of {op}"
        )
        new_factor = self.common_factor(registers)
        values = [new_factor[register] for register in registers]
        new_value = op(*values)
        new_factor[destination_register] = new_value
        self.register_to_factor[destination_register] = new_factor
        return destination_register

    def dd(self, n_register):
        destination_register = new_register_name()
        factor = self.find_factor(n_register)
        probs, rolls, indices = sx.dd_helper(factor[n_register])
        factor.p = sx.gather(factor.p, indices) * probs
        for var in factor._values:
            factor._values[var] = sx.gather(factor._values[var], indices)
        factor._values[destination_register] = rolls
        return destination_register
