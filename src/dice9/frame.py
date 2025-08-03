from collections import defaultdict
from dice9.config import sx
import dice9.config as config
import logging

global_next_var = 1000000

from .factor import Factor, new_register_name, is_reg, Register, check_is_reg

# Variables also serve as the connections to retie branches of conditionals.
class Frame:
    def __init__(self, source, environment, allocations, conditional_depth):
        # Mapping from variable name to underlying register.
        self.source = source
        self.allocations = allocations if allocations else {}
        self.environment = environment
        self.conditional_depth = conditional_depth

    def bind(self, name, value):
        logging.debug(f"Binding {name} = {value}")
        self.allocations[name] = value

    def list_allocations(self, env):
        groups = defaultdict(list)
        print(f"frame.list_allocations = {self.allocations}")
        for var, register in self.allocations.items():
            if is_reg(register):
                try:
                    factor_id = id(env.find_factor(register))
                    groups[factor_id].append((var, register))
                except:
                    print(f"var {var}'s register {register} not found in environment")
        for group, vars in groups.items():
            print("[")
            for var, register in vars:
                value = env[register]
                print(f"{var}: {value.shape} = {value}")
            print("]")

    def __repr__(self):
        return "Frame<" + ", ".join(
            f"'{k}': {v}" for k, v in self.allocations.items()) + ">"
            
    # Frame
    def allocate(self, env, new_var_name, value):
        # if new_var_name in self.allocations:
        #    raise ValueError(f"Attempt to reallocate {new_var_name}")
        new_register = new_register_name()
        factor = Factor.allocate_factor_with_register(new_register, value)
        check_is_reg(new_register)
        self.allocations[new_var_name] = new_register
        env.add_factor(factor)
        return new_register

    # Frame
    def allocate_in_same_factor(self, env, old_var_name, new_var_name, value):
        # if new_var_name in self.allocations:
        #    raise ValueError(f"Attempt to reallocate {new_var_name}")
        old_register = self.allocations[old_var_name]
        factor = env.find_factor(old_register)
        new_register = new_register_name()
        factor[new_register] = value
        if not is_reg(new_register):
            raise (f"{new_register} is not a register.")
        self.allocations[new_var_name] = new_register

    def allocate_in_same_factor_as_register(self, env, register, new_var_name, value):
        # if new_var_name in self.allocations:
        #    raise ValueError(f"Attempt to reallocate {new_var_name}")
        factor = env.find_factor(register)
        new_register = new_register_name()
        factor[new_register] = value
        if not is_reg(new_register):
            raise (f"{new_register} is not a register.")
        self.allocations[new_var_name] = new_register

    # Frame
    def allocate_with_probability(self, env, new_var_name, p, value):
        new_register = env.allocate_factor_with_register_with_probability(
            p, value
        )
        if not is_reg(new_register):
            raise (f"{new_register} is not a register.")
        self.allocations[new_var_name] = new_register

    def split(self, env, condition_register, keep_condition=False):
        env1, env2 = env.split(condition_register, keep_condition=keep_condition)

        frame1 = None
        frame2 = None

        if env1:
            frame1 = Frame(self.source, env1, self.allocations.copy(), self.conditional_depth + 1)
        if env2:
            frame2 = Frame(self.source, env2, self.allocations.copy(), self.conditional_depth + 1)

        return frame1, env1, frame2, env2

    def semi_split(self, env, condition_register, keep_condition=False):
        env = env.semi_split(condition_register, keep_condition=keep_condition)
        frame = Frame(self.source, env, self.allocations, self.conditional_depth)
        return frame, env


    # Frame
    def has_var(self, var_name):
        return var_name in self.allocations

    def delete_var_and_register(self, env, var_name):
        register = self.allocations[var_name]
        logging.debug(f"Deleting variable '{var_name}' currently bound to {register}")
        if is_reg(register):
            del env[register]
        del self.allocations[var_name]

    def delete_var_if_defined(self, env, var_name):
        if self.has_var(var_name):
            self.delete_var_and_register(env, var_name)

    def move(self, var_name):
        if var_name in self.allocations:
            register = self.allocations[var_name]
            del self.allocations[var_name]
            return register
        else:
            raise NameError(f"Variable `{var_name}` not found.")

    def delete_all(self, env):
        for var, register in self.allocations.items():
            logging.debug(f"Deleting variable '{var}' currently bound to {register}")
            if is_reg(register):
                del env[register]
        self.allocations = {}
        
    if 0:
        def delete_if_temp(self, var_name):
            if var_name[0] == '_':
                self.delete_var_and_register(var_name)

    def delete_without_marginalisation(self, env, var_name):
        register = self.allocations[var_name]
        env.raw_del(register)
        del self.allocations[var_name]

    # Frame
    def get(self, env, var_name):
        register = self.allocations[var_name]
        return env[register]

    def get_register(self, var_name):
        return self.allocations[var_name]

    def get_definite(self, env, var_name):
        register = self.allocations[var_name]
        return env.get_definite_value(register)
        
    # The variable beging assigned to should not exist
    # at this point.
    # old_name = new_name
    def assign(self, env, old_name, new_name):
        old_register = new_register_name()
        self.allocations[old_name] = old_register
        new_register = self.allocations[new_name]
        factor = env.find_factor(new_register)
        factor[old_register] = factor[new_register]

    def assign_copy(self, env, var_name, register):
        new_register = env.duplicate_register(register)
        # XXX Should we delete old?
        # visit_Assign does delete so this is defensive
        if var_name in self.allocations:
            old_register = self.allocations[var_name]
            del env[old_register]
        self.allocations[var_name] = new_register

    def assign_move(self, env, var_name, new_register):
        if var_name in self.allocations:
            old_register = self.allocations[var_name]
            del env[old_register]
        self.allocations[var_name] = new_register

    # ???
    def assign_from_register(self, env, var_name, new_register):
        old_register = new_register_name()
        self.allocations[var_name] = old_register
        # new_register = self.allocations[new_name]
        if not is_reg(new_register):
            raise ValueError(f"{new_register} is not a register.")
        factor = env.find_factor(new_register)
        factor[old_register] = factor[new_register]

    # Frame
    def un_op2(self, env, source_var, op):
        destination_var = fresh_var_name()
        source_register = self.allocations[source_var]
        if not is_reg(source_register):
            raise ValueError(f"{source_register} is not a register.")
        destination_register = env.unary_op(source_register, op)
        if not is_reg(destination_register):
            raise ValueError(f"{destination_register} is not a register.")
        self.allocations[destination_var] = destination_register
        return destination_var
                        
    def bin_op2(self, env, left_var, right_var, op):
        destination_var = fresh_var_name()
        left_register = self.allocations[left_var]
        right_register = self.allocations[right_var]
        destination_register = env.binary_op(
            left_register, right_register, op
        )
        self.allocations[destination_var] = destination_register
        return destination_var
                
    def bin_op_direct2(self, env, left_var, right_var, op):
        destination_var = fresh_var_name()
        left_register = self.allocations[left_var]
        right_register = self.allocations[right_var]
        destination_register = env.binary_op_direct(
            left_register, right_register, op
        )
        self.allocations[destination_var] = destination_register
        return destination_var

    def multi_op_direct2(self, env, vars, op):
        destination_var = fresh_var_name()
        registers = [self.allocations[var] for var in vars]
        #left_register = self.allocations[left_var]
        #right_register = self.allocations[right_var]
        destination_register = env.multi_op_direct(
            registers, op
        )
        self.allocations[destination_var] = destination_register
        return destination_var
        
    if 0:
        # Frame
        def dd(self, var_name, n_var):
            destination_register = new_register_name()
            n_register = self.allocations[n_var]
            factor = self.environment.find_factor(n_register)
            probs, rolls, indices = sx.dd_helper(factor[n_register])
            factor.p = sx.gather(factor.p, indices) * probs
            for var in factor._values:
                factor._values[var] = sx.gather(factor._values[var], indices)
            factor._values[destination_register] = rolls
            self.allocations[var_name] = destination_register
            return var_name

#        def dump(self):
#            for name, value in self.allocations.items():
