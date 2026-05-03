#!/usr/bin/env python3
"""Equation-level encoder with a learned symmetric bilinear aggregation.

The node-level encoder in `node_level_encoder.py` turns the two sides of an
equation into root vectors:

    hL, hR in R^d

The original note in `syntax_encoder_math.pdf` aggregates these vectors with

    [hL + hR ; |hL - hR|]

followed by an MLP. This file replaces that hand-chosen symmetric feature map
with a learned symmetric bilinear map:

    B(hL, hR) in R^k.

For each output coordinate `o`, the map has a learned matrix A_o and computes:

    B_o(hL, hR) = hL^T A_o hR.

To make the map symmetric in its two arguments, each A_o is forced to be a
symmetric matrix:

    A_o = A_o^T.

Then:

    hL^T A_o hR = hR^T A_o hL,

so swapping the left and right sides gives the same equation-level vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from node_level_encoder import NodeLevelEncoder, NodeLevelEncoding
from postfix_equation_tensor import EquationTensor, compile_postfix_equation


@dataclass(frozen=True)
class EquationLevelEncoding:
    """Output of the equation-level encoder for one equation.

    Attributes
    ----------
    node_encoding:
        The full node-level result. This keeps the intermediate node states and
        the two root vectors available for inspection.

    equation_embedding:
        Tensor of shape `(equation_dim,)`. This is B(hL, hR), the learned
        symmetric bilinear aggregation of the two root vectors.
    """

    node_encoding: NodeLevelEncoding
    equation_embedding: torch.Tensor


class SymmetricBilinearMap(nn.Module):
    """Learned symmetric bilinear map B: R^d x R^d -> R^k.

    This module stores unconstrained raw matrices W_o. During the forward pass,
    it uses their symmetric parts:

        A_o = (W_o + W_o^T) / 2.

    This is a simple way to guarantee symmetry without having to manually store
    only the upper triangular entries.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim

        # raw_weight[o] is the unconstrained matrix for output coordinate o.
        # The actual matrix used in forward() is its symmetric part.
        self.raw_weight = nn.Parameter(torch.empty(output_dim, input_dim, input_dim))

        # A small scale keeps the initial bilinear outputs from exploding. The
        # exact initialization is not conceptually important for the construction.
        nn.init.normal_(self.raw_weight, mean=0.0, std=1.0 / input_dim)

    def symmetric_weight(self) -> torch.Tensor:
        """Return the symmetric matrices actually used by the bilinear map."""
        return 0.5 * (self.raw_weight + self.raw_weight.transpose(-1, -2))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute B(left, right).

        `left` and `right` may be single vectors of shape `(input_dim,)`, or
        batched tensors with shape `(..., input_dim)`. The output shape is
        `(output_dim,)` for single vectors and `(..., output_dim)` for batched
        inputs.
        """

        if left.shape[-1] != self.input_dim:
            raise ValueError(
                f"left has last dimension {left.shape[-1]}, expected {self.input_dim}"
            )
        if right.shape[-1] != self.input_dim:
            raise ValueError(
                f"right has last dimension {right.shape[-1]}, expected {self.input_dim}"
            )

        weight = self.symmetric_weight()

        # Einstein notation for:
        #
        #   output[..., o] = sum_i sum_j left[..., i] * weight[o, i, j] * right[..., j]
        #
        # The "...i" and "...j" parts let the same code work for both one
        # equation and a batch of root-vector pairs.
        return torch.einsum("...i,oij,...j->...o", left, weight, right)


class EquationLevelEncoder(nn.Module):
    """Encode one equation tensor into an equation-level vector.

    The model first uses `NodeLevelEncoder` to compute hL and hR. It then uses
    `SymmetricBilinearMap` as the learned aggregation:

        equation_embedding = B(hL, hR).
    """

    def __init__(
        self,
        max_variables: int,
        feature_dim: int,
        equation_dim: int | None = None,
        composition_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        # If equation_dim is omitted, keep the equation embedding in the same
        # dimensionality as the term/node embeddings.
        output_dim = equation_dim or feature_dim

        self.node_encoder = NodeLevelEncoder(
            max_variables=max_variables,
            feature_dim=feature_dim,
            composition_hidden_dim=composition_hidden_dim,
        )
        self.aggregation = SymmetricBilinearMap(
            input_dim=feature_dim,
            output_dim=output_dim,
        )

    def forward(self, equation_tensor: EquationTensor) -> EquationLevelEncoding:
        """Encode `equation_tensor` into an equation-level embedding."""

        node_encoding = self.node_encoder(equation_tensor)
        equation_embedding = self.aggregation(
            node_encoding.lhs_root,
            node_encoding.rhs_root,
        )

        return EquationLevelEncoding(
            node_encoding=node_encoding,
            equation_embedding=equation_embedding,
        )


if __name__ == "__main__":
    # Manual demo:
    #
    #     python3 equation_level_encoder.py
    #
    # The numbers are random because the model is untrained. The symmetry check
    # is the important part: B(hL, hR) and B(hR, hL) should match up to floating
    # point roundoff.
    equation = "x = y x x * *"
    tensor = compile_postfix_equation(equation)

    encoder = EquationLevelEncoder(
        max_variables=8,
        feature_dim=4,
        equation_dim=6,
    )
    encoding = encoder(tensor)

    hL = encoding.node_encoding.lhs_root
    hR = encoding.node_encoding.rhs_root
    swapped = encoder.aggregation(hR, hL)
    max_symmetry_error = (encoding.equation_embedding - swapped).abs().max().detach()

    print(f"equation: {equation}")
    print(f"node_states shape: {tuple(encoding.node_encoding.node_states.shape)}")
    print(f"equation_embedding shape: {tuple(encoding.equation_embedding.shape)}")
    print(f"max symmetry error: {float(max_symmetry_error):.8f}")
