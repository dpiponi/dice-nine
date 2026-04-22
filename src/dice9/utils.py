import ast
import inspect
import logging
from rich.console import Console
from rich.panel import Panel

def lift_axis(axis):
    return axis + 1 if axis >= 0 else axis


def is_gen_fun(tree):
    return any(
        isinstance(subnode, (ast.Yield, ast.YieldFrom))
        for subnode in ast.walk(tree))

def get_signature_from_functiondef(fndef):
    def make_param(arg, kind, default=inspect._empty):
        return inspect.Parameter(arg.arg, kind, default=default)

    args = fndef.args
    params = []

    posonlyargs = getattr(args, "posonlyargs", [])
    for arg in posonlyargs:
        params.append(make_param(arg, inspect.Parameter.POSITIONAL_ONLY))

    for arg in args.args:
        params.append(make_param(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD))

    if args.vararg:
        params.append(make_param(args.vararg, inspect.Parameter.VAR_POSITIONAL))

    for arg in args.kwonlyargs:
        params.append(make_param(arg, inspect.Parameter.KEYWORD_ONLY))

    if args.kwarg:
        params.append(make_param(args.kwarg, inspect.Parameter.VAR_KEYWORD))

    # Apply defaults
    total_pos = len(posonlyargs) + len(args.args)

    # Defaults for positional-or-keyword.
    # Only literals supported.
    for i, default_node in enumerate(args.defaults):
        index = total_pos - len(args.defaults) + i
        params[index] = params[index].replace(
            default=ast.literal_eval(default_node))

    # Defaults for keyword-only
    # Only literals supported.
    kwonly_start = total_pos
    for i, default_node in enumerate(args.kw_defaults):
        if default_node is not None:
            index = kwonly_start + i
            params[index] = params[index].replace(
                default=ast.literal_eval(default_node))

    return inspect.Signature(params)

console = Console()


def report_error(exc, node, source):
    source_lines = source.splitlines()
    # Module has no line number but this is just a band-aid.
    if isinstance(node, ast.Module):
        node = node.body[0]
    if node:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)

        if lineno is not None:
            line = source_lines[lineno - 1].rstrip()
            prefix = f"{lineno}: "
            full_line = prefix + line
            caret_line = " " * (len(prefix) + (col_offset or 0)) + "^"

            message = "\n".join([
                f"[bold red]{exc}[/bold red]\n",
                full_line,
                caret_line if col_offset is not None else "",
            ])

            console.print(Panel(message, expand=False, border_style="red"))
    else:
        print("NO NODE")

    logging.error(str(exc))

