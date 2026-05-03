#!/usr/bin/env python3
"""Didactic node-level encoder for compiled postfix equation tensors.

This file implements the recurrence from `syntax_encoder_math.pdf` for one
compiled equation at a time. It is deliberately written for clarity, not for
maximum batching speed.

The input is an `EquationTensor` from `postfix_equation_tensor.py`, containing
flat arrays:

    op[j], arg1[j], arg2[j]

for each node `j`. Because the tensor was compiled from postfix notation, child
nodes always appear before their parent nodes. That lets us evaluate the nodes
in order with a plain Python loop.

Mathematically, this module learns:

    V(variable_id)       -> vector in R^d
    C(left_vec, right_vec) -> vector in R^d

Then a symbolic term such as

    x * (y * z)

is encoded as

    C(V(x), C(V(y), V(z))).

That is the sense in which the symbolic operation `*` is mirrored by a learned
operation `C` in feature/embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from postfix_equation_tensor import (
    OP_BINARY,
    OP_LEAF,
    EquationTensor,
    compile_postfix_equation,
)


@dataclass(frozen=True)
class NodeLevelEncoding:
    """Output of the node-level encoder for one equation.

    Attributes
    ----------
    node_states:
        Tensor of shape `(num_nodes, feature_dim)`. Row `j` is the learned
        vector state h_j for node `j`.

    lhs_root:
        Tensor of shape `(feature_dim,)`. This is the vector for the left-hand
        side root term.

    rhs_root:
        Tensor of shape `(feature_dim,)`. This is the vector for the right-hand
        side root term.
    """

    node_states: torch.Tensor
    lhs_root: torch.Tensor
    rhs_root: torch.Tensor


class NodeLevelEncoder(nn.Module):
    """Encode a compiled equation tree into learned node vectors.

    Parameters
    ----------
    max_variables:
        Size of the variable embedding table. This must be at least the maximum
        number of distinct local variables in any equation you want to encode.
        For example, if an equation uses local variable ids 0, 1, 2, then
        `max_variables` must be at least 3.

    feature_dim:
        Dimension `d` of the learned vector space.

    composition_hidden_dim:
        Hidden size for the learned binary operation `C`. If omitted, this uses
        `2 * feature_dim`.
    """

    def __init__(
        self,
        max_variables: int,
        feature_dim: int,
        composition_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if max_variables <= 0:
            raise ValueError("max_variables must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        hidden_dim = composition_hidden_dim or 2 * feature_dim

        # V maps a local variable id to a learned vector in R^d. There is no
        # embedding for -1 here: -1 is only an unused-argument sentinel in the
        # compiled representation, not a real variable.
        self.variable_embedding = nn.Embedding(max_variables, feature_dim)

        # C is the learned version of the binary operation `*` in feature space.
        # It receives the concatenated left and right child vectors and returns
        # one vector with the same dimension as a variable embedding.
        self.composition = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    @property
    def feature_dim(self) -> int:
        """The dimension of each node vector."""
        return self.variable_embedding.embedding_dim

    @property
    def max_variables(self) -> int:
        """The number of local variable ids the encoder can embed."""
        return self.variable_embedding.num_embeddings

    def forward(self, equation_tensor: EquationTensor) -> NodeLevelEncoding:
        """Evaluate all node states for `equation_tensor`.

        This function follows the recurrence:

            h_j = V(arg1[j])                       if op[j] is leaf
            h_j = C(h_arg1[j], h_arg2[j])          if op[j] is binary

        Since postfix compilation guarantees children have smaller indices
        than their parent, `states[left_child]` and `states[right_child]` are
        already available when an internal node is evaluated.
        """

        self._validate_tensor(equation_tensor)

        # Keep newly created states in a Python list for readability. Each entry
        # is a rank-1 tensor of shape `(feature_dim,)`.
        states: list[torch.Tensor] = []

        # Create scalar index tensors on the same device as the model. This
        # keeps the demo working whether the model is on CPU or moved to GPU.
        device = self.variable_embedding.weight.device

        for node_index, op_code in enumerate(equation_tensor.op):
            if op_code == OP_LEAF:
                # For a leaf node, arg1 stores the local variable id and arg2 is
                # just -1. We intentionally ignore arg2.
                variable_id = equation_tensor.arg1[node_index]
                variable_id_tensor = torch.tensor(variable_id, device=device)
                node_state = self.variable_embedding(variable_id_tensor)

            elif op_code == OP_BINARY:
                # For an internal node, arg1 and arg2 are child node indices.
                # The order matters: C(left, right) need not equal C(right,
                # left), just as a magma operation need not be commutative.
                left_child_index = equation_tensor.arg1[node_index]
                right_child_index = equation_tensor.arg2[node_index]

                left_state = states[left_child_index]
                right_state = states[right_child_index]

                # Concatenate along the feature dimension to form a vector in
                # R^(2d), then apply the learned composition map C.
                child_pair = torch.cat([left_state, right_state], dim=0)
                node_state = self.composition(child_pair)

            else:
                raise ValueError(f"Unknown op code at node {node_index}: {op_code}")

            states.append(node_state)

        node_states = torch.stack(states, dim=0)

        return NodeLevelEncoding(
            node_states=node_states,
            lhs_root=node_states[equation_tensor.rL],
            rhs_root=node_states[equation_tensor.rR],
        )

    def _validate_tensor(self, equation_tensor: EquationTensor) -> None:
        """Check the structural assumptions used by `forward`.

        These checks are useful in a teaching implementation because they make
        mistakes in the compiled representation fail loudly. A production
        batched implementation would usually perform these checks once during
        preprocessing, not on every forward pass.
        """

        lengths = {
            len(equation_tensor.op),
            len(equation_tensor.arg1),
            len(equation_tensor.arg2),
        }
        if len(lengths) != 1:
            raise ValueError("op, arg1, and arg2 must have the same length")

        num_nodes = len(equation_tensor.op)
        if num_nodes == 0:
            raise ValueError("equation_tensor must contain at least one node")

        if not 0 <= equation_tensor.rL < num_nodes:
            raise ValueError(f"Left root index is out of range: {equation_tensor.rL}")
        if not 0 <= equation_tensor.rR < num_nodes:
            raise ValueError(f"Right root index is out of range: {equation_tensor.rR}")

        for node_index, op_code in enumerate(equation_tensor.op):
            first_arg = equation_tensor.arg1[node_index]
            second_arg = equation_tensor.arg2[node_index]

            if op_code == OP_LEAF:
                if not 0 <= first_arg < self.max_variables:
                    raise ValueError(
                        f"Leaf node {node_index} has variable id {first_arg}, "
                        f"but max_variables is {self.max_variables}"
                    )
                if second_arg != -1:
                    raise ValueError(
                        f"Leaf node {node_index} should have arg2 = -1, "
                        f"got {second_arg}"
                    )

            elif op_code == OP_BINARY:
                # Child indices must point backward. This is the key postorder
                # property that lets the encoder use one left-to-right pass.
                if not 0 <= first_arg < node_index:
                    raise ValueError(
                        f"Binary node {node_index} has invalid left child "
                        f"index {first_arg}"
                    )
                if not 0 <= second_arg < node_index:
                    raise ValueError(
                        f"Binary node {node_index} has invalid right child "
                        f"index {second_arg}"
                    )

            else:
                raise ValueError(f"Unknown op code at node {node_index}: {op_code}")


if __name__ == "__main__":
    # Small manual demo:
    #
    #     python3 node_level_encoder.py
    #
    # This creates random learned parameters, so the numeric vectors printed
    # below are not meaningful by themselves. The important part is the shape:
    # every syntax node receives one vector in the chosen feature dimension.
    equation = "x = y x x * *"
    tensor = compile_postfix_equation(equation)

    encoder = NodeLevelEncoder(max_variables=8, feature_dim=4)
    encoding = encoder(tensor)

    print(f"equation: {equation}")
    print(f"node_states shape: {tuple(encoding.node_states.shape)}")
    print(f"lhs_root shape: {tuple(encoding.lhs_root.shape)}")
    print(f"rhs_root shape: {tuple(encoding.rhs_root.shape)}")
