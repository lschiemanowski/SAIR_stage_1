#!/usr/bin/env python3
"""Compile a postfix equation into the equation-tensor representation.

This file is intentionally didactic rather than maximally optimized. It shows
how a postfix equation such as

    x = y x x * *

can be turned into the same kind of flat "program" described in
`syntax_encoder_math.pdf`:

    op   = [leaf, leaf, leaf, leaf, diamond, diamond]
    arg1 = [0,    1,    0,    0,    2,       1]
    arg2 = [-1,  -1,   -1,   -1,    3,       4]
    rL   = 0
    rR   = 5

The important point is that postfix notation already lists every operator
after its two arguments. That means we can compile it with a simple stack:

    variable token: create a leaf node and push its node index
    "*" token:      pop right and left child indices, create an operator node,
                    then push the new node index

The final stack item is the root node for that side of the equation.
"""

from __future__ import annotations

from dataclasses import dataclass


# These integer op codes are what a neural model would usually store in a
# tensor. The names are kept readable here, and the values are deliberately
# simple:
#
#   0 means "this node is a variable leaf"
#   1 means "this node applies the binary operation *"
#
# The value -1 is not an op code. It is used below as a sentinel in `arg2` for
# leaf nodes, because leaf nodes only need one argument: their variable id.
OP_LEAF = 0
OP_BINARY = 1
UNUSED = -1


@dataclass(frozen=True)
class EquationTensor:
    """Flat compiled representation of one equation.

    Attributes
    ----------
    op:
        `op[j]` says what kind of node `j` is. It is `OP_LEAF` for variable
        leaves and `OP_BINARY` for internal `*` nodes.

    arg1:
        If node `j` is a leaf, `arg1[j]` is the local variable id.
        If node `j` is an internal `*`, `arg1[j]` is the left child node index.

    arg2:
        If node `j` is a leaf, `arg2[j]` is `UNUSED` (-1).
        If node `j` is an internal `*`, `arg2[j]` is the right child node index.

    rL, rR:
        Root node indices for the left-hand side and right-hand side.

    variable_to_id:
        Mapping from variable names in the equation to local variable ids.
        The ids are assigned in order of first appearance while reading the
        postfix equation from left to right, left side before right side.
    """

    op: list[int]
    arg1: list[int]
    arg2: list[int]
    rL: int
    rR: int
    variable_to_id: dict[str, int]


def compile_postfix_equation(equation: str) -> EquationTensor:
    """Compile one postfix equation into `EquationTensor`.

    Parameters
    ----------
    equation:
        A single equation in postfix form, for example:

            "x = y x x * *"

        The only built-in operator is the binary operator `*`. Every other
        token is treated as a variable name.

    Returns
    -------
    EquationTensor
        The flat postorder program for the equation.

    Raises
    ------
    ValueError
        If the equation does not contain exactly one `=`, or if either side is
        not a well-formed postfix expression.
    """

    if equation.count("=") != 1:
        raise ValueError("A postfix equation must contain exactly one '='")

    lhs_text, rhs_text = equation.split("=")

    # These arrays are filled in exactly the order nodes are created. Since
    # postfix notation creates children before parents, this order is already
    # postorder, and every child index will be smaller than its parent index.
    op: list[int] = []
    arg1: list[int] = []
    arg2: list[int] = []

    # Variables are local to an equation. For example, the first distinct
    # variable encountered gets id 0, the next gets id 1, and so on. This means
    # "x" does not need to have a global id shared by every equation.
    variable_to_id: dict[str, int] = {}

    def variable_id(name: str) -> int:
        """Return the local id for `name`, creating one if needed."""
        if name not in variable_to_id:
            variable_to_id[name] = len(variable_to_id)
        return variable_to_id[name]

    def append_leaf(variable_name: str) -> int:
        """Append one variable leaf node and return its new node index."""
        node_index = len(op)

        op.append(OP_LEAF)
        arg1.append(variable_id(variable_name))
        arg2.append(UNUSED)

        return node_index

    def append_binary(left_child: int, right_child: int) -> int:
        """Append one internal `*` node and return its new node index."""
        node_index = len(op)

        op.append(OP_BINARY)
        arg1.append(left_child)
        arg2.append(right_child)

        return node_index

    def compile_side(side_text: str, side_name: str) -> int:
        """Compile one postfix side and return its root node index."""
        tokens = side_text.split()
        if not tokens:
            raise ValueError(f"The {side_name} side is empty")

        # The stack stores node indices, not values. When we see a variable,
        # we create a leaf node and push its index. When we see "*", we pop
        # two already-created node indices and create their parent node.
        stack: list[int] = []

        for token in tokens:
            if token == "*":
                if len(stack) < 2:
                    raise ValueError(
                        f"The {side_name} side has an operator without two operands"
                    )

                # In postfix notation, the left child was pushed before the
                # right child. Since the right child is on top, pop it first.
                right_child = stack.pop()
                left_child = stack.pop()

                parent = append_binary(left_child, right_child)
                stack.append(parent)
            else:
                leaf = append_leaf(token)
                stack.append(leaf)

        if len(stack) != 1:
            raise ValueError(
                f"The {side_name} side leaves {len(stack)} items on the stack; "
                "a well-formed postfix term leaves exactly one"
            )

        return stack[0]

    # Compile left first and then right. This mirrors the convention in the PDF
    # and gives a single node index space shared by both sides of the equation.
    rL = compile_side(lhs_text, "left-hand")
    rR = compile_side(rhs_text, "right-hand")

    return EquationTensor(
        op=op,
        arg1=arg1,
        arg2=arg2,
        rL=rL,
        rR=rR,
        variable_to_id=variable_to_id,
    )


if __name__ == "__main__":
    # Tiny demonstration for direct execution:
    #
    #     python3 postfix_equation_tensor.py
    #
    # This is not needed by other code; it is here only to make the file easy
    # to inspect manually.
    example = "x = y x x * *"
    tensor = compile_postfix_equation(example)

    print(f"equation: {example}")
    print(f"variable_to_id: {tensor.variable_to_id}")
    print(f"op:   {tensor.op}")
    print(f"arg1: {tensor.arg1}")
    print(f"arg2: {tensor.arg2}")
    print(f"rL:   {tensor.rL}")
    print(f"rR:   {tensor.rR}")
