#!/usr/bin/env python3
"""Batched equation encoder for compiled postfix equations.

The earlier teaching files encode one equation at a time. That is useful for
understanding the recurrence, but it is slow when a training batch contains
thousands of implication pairs. This module keeps the same mathematical idea
while batching over equation indices:

    equation index -> compiled node table -> equation embedding

The compiled representation is still the simple postfix table from
`postfix_equation_tensor.py`. The only difference is that all equations are
padded into rectangular tensors once, then stored as non-trainable buffers in
the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from postfix_equation_tensor import OP_BINARY, OP_LEAF, EquationTensor, compile_postfix_equation


PAD_OP = -1


@dataclass(frozen=True)
class CompiledEquationTable:
    """Padded tensor form of a list of compiled equations."""

    op: torch.Tensor
    arg1: torch.Tensor
    arg2: torch.Tensor
    lhs_root: torch.Tensor
    rhs_root: torch.Tensor


def build_compiled_equation_table(
    equation_tensors: list[EquationTensor],
    max_variables: int,
) -> CompiledEquationTable:
    """Pad a list of `EquationTensor` objects into rectangular tensors."""

    if not equation_tensors:
        raise ValueError("equation_tensors must not be empty")
    if max_variables <= 0:
        raise ValueError("max_variables must be positive")

    max_nodes = max(len(equation.op) for equation in equation_tensors)
    num_equations = len(equation_tensors)

    op = torch.full((num_equations, max_nodes), PAD_OP, dtype=torch.long)
    arg1 = torch.full((num_equations, max_nodes), -1, dtype=torch.long)
    arg2 = torch.full((num_equations, max_nodes), -1, dtype=torch.long)
    lhs_root = torch.empty(num_equations, dtype=torch.long)
    rhs_root = torch.empty(num_equations, dtype=torch.long)

    for equation_index, equation in enumerate(equation_tensors):
        _validate_equation_tensor(equation, max_variables)

        length = len(equation.op)
        op[equation_index, :length] = torch.tensor(equation.op, dtype=torch.long)
        arg1[equation_index, :length] = torch.tensor(equation.arg1, dtype=torch.long)
        arg2[equation_index, :length] = torch.tensor(equation.arg2, dtype=torch.long)
        lhs_root[equation_index] = equation.rL
        rhs_root[equation_index] = equation.rR

    return CompiledEquationTable(
        op=op,
        arg1=arg1,
        arg2=arg2,
        lhs_root=lhs_root,
        rhs_root=rhs_root,
    )


class CompiledEquationEncoder(nn.Module):
    """Encode many equations by index.

    Parameters
    ----------
    equation_tensors:
        The postfix-compiled equations, in equation-id order.

    max_variables:
        Maximum number of local variables in any equation.

    feature_dim:
        Dimension of learned node vectors.

    equation_dim:
        Dimension of the final equation embedding. If omitted, this is the
        same as `feature_dim`.

    The two root vectors of one equation are aggregated by the symmetric feature
    map

        [hL + hR ; |hL - hR|] -> MLP.
    """

    def __init__(
        self,
        equation_tensors: list[EquationTensor],
        max_variables: int,
        feature_dim: int = 256,
        equation_dim: int | None = None,
        composition_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        output_dim = equation_dim or feature_dim
        if output_dim <= 0:
            raise ValueError("equation_dim must be positive")

        table = build_compiled_equation_table(equation_tensors, max_variables)
        self.register_buffer("op_tensor", table.op)
        self.register_buffer("arg1_tensor", table.arg1)
        self.register_buffer("arg2_tensor", table.arg2)
        self.register_buffer("lhs_root_tensor", table.lhs_root)
        self.register_buffer("rhs_root_tensor", table.rhs_root)

        self.feature_dim = feature_dim
        self.equation_dim = output_dim

        hidden_dim = composition_hidden_dim or feature_dim
        self.variable_embedding = nn.Embedding(max_variables, feature_dim)
        self.composition = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self.aggregation = nn.Sequential(
            nn.Linear(2 * feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

        self._initialize_parameters()

    @property
    def num_equations(self) -> int:
        """Number of equations compiled into this encoder."""

        return int(self.op_tensor.size(0))

    @property
    def max_nodes(self) -> int:
        """Maximum number of syntax nodes in one compiled equation."""

        return int(self.op_tensor.size(1))

    @property
    def max_variables(self) -> int:
        """Number of local variable ids supported by the embedding table."""

        return int(self.variable_embedding.num_embeddings)

    def _initialize_parameters(self) -> None:
        nn.init.normal_(self.variable_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_unique_indices(self, unique_indices: torch.Tensor) -> torch.Tensor:
        """Encode sorted unique equation indices."""

        op = self.op_tensor[unique_indices]
        arg1 = self.arg1_tensor[unique_indices]
        arg2 = self.arg2_tensor[unique_indices]

        batch_size, max_nodes = op.shape
        states = torch.zeros(
            batch_size,
            max_nodes,
            self.feature_dim,
            dtype=self.variable_embedding.weight.dtype,
            device=op.device,
        )

        for node_position in range(max_nodes):
            op_column = op[:, node_position]

            leaf_rows = torch.nonzero(op_column == OP_LEAF, as_tuple=False).squeeze(-1)
            if leaf_rows.numel() > 0:
                variable_ids = arg1[leaf_rows, node_position]
                states[leaf_rows, node_position] = self.variable_embedding(variable_ids)

            binary_rows = torch.nonzero(op_column == OP_BINARY, as_tuple=False).squeeze(-1)
            if binary_rows.numel() > 0:
                left_states = states[binary_rows, arg1[binary_rows, node_position]]
                right_states = states[binary_rows, arg2[binary_rows, node_position]]
                child_pair = torch.cat([left_states, right_states], dim=-1)
                states[binary_rows, node_position] = self.composition(child_pair)

        row_indices = torch.arange(batch_size, device=op.device)
        left_roots = self.lhs_root_tensor[unique_indices]
        right_roots = self.rhs_root_tensor[unique_indices]
        left_states = states[row_indices, left_roots]
        right_states = states[row_indices, right_roots]

        symmetric_features = torch.cat(
            [left_states + right_states, torch.abs(left_states - right_states)],
            dim=-1,
        )
        return self.aggregation(symmetric_features)

    def encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Return equation embeddings for a tensor of equation indices."""

        indices = indices.to(device=self.op_tensor.device, dtype=torch.long)
        original_shape = tuple(indices.shape)
        flat_indices = indices.reshape(-1)

        if flat_indices.numel() == 0:
            return torch.empty(
                (*original_shape, self.equation_dim),
                dtype=self.variable_embedding.weight.dtype,
                device=self.op_tensor.device,
            )

        unique_indices, inverse = torch.unique(
            flat_indices,
            sorted=True,
            return_inverse=True,
        )
        encoded_unique = self._encode_unique_indices(unique_indices)
        encoded = encoded_unique[inverse]
        return encoded.reshape(*original_shape, self.equation_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Alias for `encode_indices`."""

        return self.encode_indices(indices)


def _validate_equation_tensor(
    equation: EquationTensor,
    max_variables: int,
) -> None:
    lengths = {len(equation.op), len(equation.arg1), len(equation.arg2)}
    if len(lengths) != 1:
        raise ValueError("op, arg1, and arg2 must have the same length")
    if not equation.op:
        raise ValueError("equation tensors must contain at least one node")

    num_nodes = len(equation.op)
    if not 0 <= equation.rL < num_nodes:
        raise ValueError(f"left root index is out of range: {equation.rL}")
    if not 0 <= equation.rR < num_nodes:
        raise ValueError(f"right root index is out of range: {equation.rR}")

    for node_index, op_code in enumerate(equation.op):
        first_arg = equation.arg1[node_index]
        second_arg = equation.arg2[node_index]

        if op_code == OP_LEAF:
            if not 0 <= first_arg < max_variables:
                raise ValueError(
                    f"leaf node {node_index} has variable id {first_arg}, "
                    f"but max_variables is {max_variables}"
                )
            if second_arg != -1:
                raise ValueError(
                    f"leaf node {node_index} should have arg2 = -1, got {second_arg}"
                )

        elif op_code == OP_BINARY:
            if not 0 <= first_arg < node_index:
                raise ValueError(
                    f"binary node {node_index} has invalid left child {first_arg}"
                )
            if not 0 <= second_arg < node_index:
                raise ValueError(
                    f"binary node {node_index} has invalid right child {second_arg}"
                )

        else:
            raise ValueError(f"unknown op code at node {node_index}: {op_code}")


if __name__ == "__main__":
    examples = [
        "x = y x x * *",
        "x y * = y x *",
        "x y * z * = x y z * *",
    ]
    tensors = [compile_postfix_equation(equation) for equation in examples]
    max_variables = max(len(tensor.variable_to_id) for tensor in tensors)

    encoder = CompiledEquationEncoder(
        equation_tensors=tensors,
        max_variables=max_variables,
        feature_dim=8,
        equation_dim=8,
    )
    indices = torch.tensor([0, 1, 2, 1])
    embeddings = encoder.encode_indices(indices)

    print(f"indices: {indices.tolist()}")
    print(f"embeddings shape: {tuple(embeddings.shape)}")
    print(f"compiled equations: {encoder.num_equations}")
    print(f"max nodes: {encoder.max_nodes}")
