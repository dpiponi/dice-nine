"""Exceptions used by dice-nine interpreter"""

class InterpreterError(Exception):
    """Errors found during dice-nine interpretation"""

    def __init__(self, message, node=None, frame=None):
        super().__init__(message)
        if node:
            self.node = node
        if frame:
            self.frame = frame

class FoundReturn(Exception):
    """Used to escape out of interpreter when `return` found"""

    def __init__(self, value_node):
        self.value_node = value_node
