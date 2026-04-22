import ast
from .exceptions import InterpreterError
import logging

class DupTracker(ast.NodeVisitor):
    def __init__(self):
        self.last_moves = {}
        self.moves = {}

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
            for arg in node.args.posonlyargs + node.args.args:
                self.last_moves[arg.arg] = {}

            self.visit_statements(node.body)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "move":
            self.needs(node.args[0].id)
            return

        for a in node.args:
            self.visit(a)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            name = node.id
            if name in self.last_moves:
                if self.last_moves[name]:
                    self.needs(name)
            else:
                return
            self.last_moves[name] = {node}
            self.moves[node] = self.moves.get(node, True)

    def visit_Assign(self, node):
        self.visit(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.provides(target.id)
            elif isinstance(target, ast.Tuple):
                for var in target.elts:
                    self.provides(var.id)
            elif isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name):
                    self.visit(target.slice)
                    self.needs(target.value.id)
                    self.provides(target.value.id)
                else:
                    raise NotImplementedError("Only simple subscript assignment supported.")
            else:
                raise NotImplementedError("Only simple subscript assignment supported.")

    def visit_Return(self, node):
        if node.value is None:
            raise InterpreterError("You must return something.", node=node)
        self.visit(node.value)

    def visit_AugAssign(self, node):
        self.visit(node.value)

        if isinstance(node.target, ast.Subscript):
            name = node.target.value.id
            if name in self.last_moves:
                if self.last_moves[name]:
                    self.needs(name)
            else:
                raise ValueError(f"Name {name} not assigned")

            self.visit(node.target.slice)
            self.provides(name)

        if isinstance(node.target, ast.Name):
            name = node.target.id
            if name in self.last_moves:
                if self.last_moves[name]:
                    self.needs(name)
            else:
                raise ValueError(f"Name {name} not assigned")

            self.provides(name)

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
        new_last_moves = {k: self.last_moves.get(k, set()) | last_moves_if.get(k, set()) for k in ks}
        self.last_moves = new_last_moves

        ks = set(self.moves.keys()) | set(moves_if.keys())
        new_moves = {k: self.moves.get(k, True) & moves_if.get(k, True) for k in ks}
        self.moves = new_moves

    def visit_GeneratorExp(self, node):
        self.visit(node.generators[0].iter)
        loop_var = node.generators[0].target.id

        self.provides(loop_var)
        self.visit(node.elt)

        self.provides(loop_var)
        self.visit(node.elt)

    def visit_ListComp(self, node):
        self.visit(node.generators[0].iter)
        element_name = node.generators[0].target.id

        self.provides(element_name)
        self.visit(node.elt)

        self.provides(element_name)
        self.visit(node.elt)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self.visit(node.left)
            self.visit(node.right)
            self.visit(node.right)
        else:
            self.visit(node.left)
            self.visit(node.right)

    def visit_For(self, node):
        self.visit(node.iter)

        loop_vars = [e.id for e in node.target.elts] if isinstance(node.target, ast.Tuple) else [node.target.id]
        for var in loop_vars:
            self.provides(var)

        for var in loop_vars:
            self.provides(var)
        self.visit_statements(node.body)

        for var in loop_vars:
            self.provides(var)
        self.visit_statements(node.body)


class DupInserter(ast.NodeTransformer):
    def __init__(self, moves):
        self.moves = moves

    def visit_Call(self, node: ast.Call):
        node.args = [self.visit(a) for a in node.args]
        for kw in node.keywords:
            kw.value = self.visit(kw.value)

        if not isinstance(node.func, ast.Name):
            node.func = self.visit(node.func)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node in self.moves and self.moves[node]:
            return ast.copy_location(
                ast.Call(func=ast.Name(id="move", ctx=ast.Load()), args=[node], keywords=[]),
                node,
            )
        return node


def move_analysis(tree):
    tracker = DupTracker()
    tracker.visit(tree)

    inserter = DupInserter(tracker.moves)
    transformed_tree = inserter.visit(tree)
    ast.fix_missing_locations(transformed_tree)

    return transformed_tree

def names_in_expr(node):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.names.add(node.id)

        def visit_Call(self, node: ast.Call):
            if not isinstance(node.func, ast.Name):
                self.visit(node.func)
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                if kw.value is not None:
                    self.visit(kw.value)

        def visit_Subscript(self, node: ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "d"):
                self.visit(node.value)
            self._visit_slice(node.slice)

        def _visit_slice(self, s):
            if isinstance(s, ast.AST):
                if isinstance(s, ast.Slice):
                    if s.lower: self.visit(s.lower)
                    if s.upper: self.visit(s.upper)
                    if s.step:  self.visit(s.step)
                elif isinstance(s, ast.ExtSlice):
                    for dim in s.dims:
                        self._visit_slice(dim)
                elif isinstance(s, ast.Index):
                    self.visit(s.value)
                else:
                    self.visit(s)

        def visit_ListComp(self, node: ast.ListComp):
            for gen in node.generators:
                self.visit(gen.iter)
                for cond in gen.ifs: self.visit(cond)
            self.visit(node.elt)

        def visit_SetComp(self, node: ast.SetComp):
            for gen in node.generators:
                self.visit(gen.iter)
                for cond in gen.ifs: self.visit(cond)
            self.visit(node.elt)

        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            for gen in node.generators:
                self.visit(gen.iter)
                for cond in gen.ifs: self.visit(cond)
            self.visit(node.elt)

        def visit_DictComp(self, node: ast.DictComp):
            for gen in node.generators:
                self.visit(gen.iter)
                for cond in gen.ifs:
                    self.visit(cond)
            self.visit(node.key)
            self.visit(node.value)

    if not isinstance(node, ast.AST):
        raise TypeError("names_in_expr expects an ast.AST (expression)")

    v = Visitor()
    v.visit(node)
    return v.names
