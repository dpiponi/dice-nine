import ast

sample_code = """
x = 1
y = 2
z = x + y
w = x + 2
j = 3
b = x + x
if w:
    a = x + z + 3 * j
    a -= 4
    for k in z:
        x = b
        w += j
        b = x + a
else:
    b = w * w
p = a
q = b
"""


# ----- Pass 1: Identify where dup is needed -----


class DupTracker(ast.NodeVisitor):
    def __init__(self):
        self.blockers = {}  # list of moves that might block a read
        self.moves = {}  # is the assigment a move?

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            name = node.id
            if name in self.blockers:
                if not self.blockers[name]:
                    pass
                else:
                    # Need to fix move
                    for move_node in self.blockers[name]:
                        self.moves[move_node] = False
            else:
                raise ValueError(f"Name {name} not assigned")
            self.blockers[name] = {node}
            self.moves[node] = True

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.blockers[target.id] = set()
        self.generic_visit(node)

    def visit_Move(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.blockers[target.id] = {node}
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            # read first

            name = node.target.id
            if name in self.blockers:
                if not self.blockers[name]:
                    pass
                else:
                    # Need to fix move
                    for move_node in self.blockers[name]:
                        self.moves[move_node] = False
            else:
                raise ValueError(f"Name {name} not assigned")

            # now write
            self.blockers[name] = {}

        self.generic_visit(node)

        def visit_If(self, node):
            self.visit(node.test)

            blockers_orelse = {k: set(v) for k, v in self.blockers.items()}
            moves_orelse = dict(self.moves)

            self.visit_statements(node.body)

            blockers_if = {k: set(v) for k, v in self.blockers.items()}
            moves_if = dict(self.moves)

            self.blockers = blockers_orelse
            self.moves = moves_orelse

            self.visit_statements(orelse.body)

            # merge
            self.blockers = {k: v and blockers_if[k] for k, v in self.blockers}
            self.moves = {k: v | moves_if[k] for k, v in self.blockers}

        def visit_For(self, node):
            self.visit(node.iter)

            loop_var = node.target

            # First iteration
            self.blockers[loop_var] = set()
            self.visit_statements(node.body)

            # Second iteration
            self.blockers[loop_var] = set()
            self.visit_statements(node.body)

    if 0:
        def visit_While(self, node):
            # Save pre-loop state
            pre_reads = self._copy_last_reads()
            pre_dups = set(self.dup_flags)

            # First iteration
            self.last_reads = self._copy_last_reads(pre_reads)
            self.dup_flags = set(pre_dups)
            self.visit(node.test)
            self.visit_statements(node.body)
            iter1_reads = self._copy_last_reads()
            iter1_dups = set(self.dup_flags)

            # Merge pre-loop and iter1
            merged_reads = self._merge_reads(pre_reads, iter1_reads)
            merged_dups = pre_dups | iter1_dups

            # Second iteration: use merged_reads so we can detect re-use
            self.last_reads = self._copy_last_reads(merged_reads)
            self.dup_flags = set(merged_dups)
            self.visit(node.test)
            self.visit_statements(node.body)
            iter2_reads = self._copy_last_reads()
            iter2_dups = set(self.dup_flags)

            # Final merge of all seen reads and dups
            self.last_reads = self._merge_reads(pre_reads, iter1_reads, iter2_reads)
            self.dup_flags = merged_dups | iter2_dups

        def visit_For(self, node):
            self.visit(node.iter)

            loop_var = getattr(node.target, "id", None)

            # Save pre-loop state
            pre_reads = self._copy_last_reads()
            pre_dups = set(self.dup_flags)

            # First iteration
            self.last_reads = self._copy_last_reads(pre_reads)
            self.dup_flags = set(pre_dups)
            if loop_var:
                self.last_reads.pop(loop_var, None)
            self.visit_statements(node.body)
            iter1_reads = self._copy_last_reads()
            iter1_dups = set(self.dup_flags)

            # Second iteration
            merged_reads = self._merge_reads(pre_reads, iter1_reads)
            merged_dups = pre_dups | iter1_dups
            self.last_reads = self._copy_last_reads(merged_reads)
            self.dup_flags = set(merged_dups)
            if loop_var:
                self.last_reads.pop(loop_var, None)
            self.visit_statements(node.body)
            iter2_reads = self._copy_last_reads()
            iter2_dups = set(self.dup_flags)

            # Final merge
            self.last_reads = self._merge_reads(iter1_reads, iter2_reads)
            self.dup_flags = iter1_dups | iter2_dups

    def visit_statements(self, stmts):
        for stmt in stmts:
            self.visit(stmt)


# ----- Pass 2: Insert dup where needed -----


class DupInserter(ast.NodeTransformer):
    def __init__(self, moves):
        self.moves = moves

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and self.moves[node]:
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="move", ctx=ast.Load()), args=[node], keywords=[]
                ),
                node,
            )
        return node


# ----- Parse and Transform -----

tree = ast.parse(sample_code)

tracker = DupTracker()
tracker.visit(tree)

inserter = DupInserter(tracker.moves)
transformed_tree = inserter.visit(tree)
ast.fix_missing_locations(transformed_tree)

# ----- Output -----

print("Original code:")
print(sample_code)

print("Transformed code with dup insertions:")
print(ast.unparse(transformed_tree))
