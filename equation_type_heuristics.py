#!/usr/bin/env python3
"""A hand-written four-type syntax oracle.

This is the deliberately simple syntax approximation to the collapsed
relation-cluster oracle discussed in the blogpost. It does not use a trained
model. It reads the surface shape of an equation and assigns one of four coarse
types:

    S, 2, 4, 5

The block-rate table below is the same kind of object as a learned cluster
oracle: source type and target type determine an empirical implication rate,
and rates above a threshold are predicted as implications.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from convert_equations_to_postfix import Parser, tokenize


CoarseType = Literal["S", "2", "4", "5"]

TYPE_ORDER: tuple[CoarseType, ...] = ("S", "2", "4", "5")

@dataclass(frozen=True)
class EquationTypeFeatures:
    equation: str
    lhs_is_variable: bool
    lhs_variable: str | None
    lhs_size: int
    lhs_leaf_names: tuple[str, ...]
    rhs_size: int
    rhs_depth: int
    rhs_leaf_names: tuple[str, ...]
    rhs_leftmost: str
    rhs_rightmost: str
    rhs_x_count: int
    rhs_y_count: int
    rhs_z_count: int
    rhs_w_count: int


COARSE_IMPLICATION_RATE_TABLE: dict[CoarseType, dict[CoarseType, float]] = {
    "S": {"S": 1.0, "2": 1.0, "4": 1.0, "5": 1.0},
    "2": {"S": 0.0, "2": 0.09348337107657685, "4": 0.0847224945017392, "5": 0.0},
    "4": {"S": 0.0, "2": 0.0, "4": 0.07064396050887559, "5": 0.0},
    "5": {"S": 0.0, "2": 0.0, "4": 1.0, "5": 1.0},
}


def _parse_term(term: str):
    return Parser(tokenize(term)).parse()


def _leaf_names(node) -> tuple[str, ...]:
    if node.left is None and node.right is None:
        return (node.token,)
    return _leaf_names(node.left) + _leaf_names(node.right)


def _term_size(node) -> int:
    if node.left is None and node.right is None:
        return 1
    return 1 + _term_size(node.left) + _term_size(node.right)


def _term_depth(node) -> int:
    if node.left is None and node.right is None:
        return 1
    return 1 + max(_term_depth(node.left), _term_depth(node.right))


@lru_cache(maxsize=None)
def extract_type_features(equation: str) -> EquationTypeFeatures:
    if equation.count("=") != 1:
        raise ValueError(f"expected exactly one '=' in equation: {equation!r}")

    lhs_text, rhs_text = equation.split("=")
    lhs = _parse_term(lhs_text)
    rhs = _parse_term(rhs_text)

    lhs_leaf_names = _leaf_names(lhs)
    rhs_leaf_names = _leaf_names(rhs)
    lhs_is_variable = lhs.left is None and lhs.right is None

    return EquationTypeFeatures(
        equation=equation,
        lhs_is_variable=lhs_is_variable,
        lhs_variable=lhs.token if lhs_is_variable else None,
        lhs_size=_term_size(lhs),
        lhs_leaf_names=lhs_leaf_names,
        rhs_size=_term_size(rhs),
        rhs_depth=_term_depth(rhs),
        rhs_leaf_names=rhs_leaf_names,
        rhs_leftmost=rhs_leaf_names[0],
        rhs_rightmost=rhs_leaf_names[-1],
        rhs_x_count=rhs_leaf_names.count("x"),
        rhs_y_count=rhs_leaf_names.count("y"),
        rhs_z_count=rhs_leaf_names.count("z"),
        rhs_w_count=rhs_leaf_names.count("w"),
    )


def _matches_type_5_signature(features: EquationTypeFeatures) -> bool:
    lhs_is_xy = features.lhs_size == 3 and features.lhs_leaf_names == ("x", "y")
    return lhs_is_xy and (
        (features.rhs_leftmost == "z" and features.rhs_rightmost != "y")
        or (
            features.rhs_leftmost == "y"
            and features.rhs_rightmost != "y"
            and features.rhs_z_count >= 1
        )
    )


def _matches_type_4_signature(features: EquationTypeFeatures) -> bool:
    return not features.lhs_is_variable


def _matches_type_2_signature(features: EquationTypeFeatures) -> bool:
    if not (features.lhs_is_variable and features.lhs_variable == "x"):
        return False

    boundary_x = features.rhs_leftmost == "x" or features.rhs_rightmost == "x"
    y_led_internal_x = (
        features.rhs_w_count == 0
        and features.rhs_x_count >= 1
        and features.rhs_y_count >= 2
        and features.rhs_z_count != 1
    )
    return boundary_x or y_led_internal_x


def classify_coarse_type(equation: str) -> CoarseType:
    """Classify an equation into the coarse S / 2 / 4 / 5 syntax taxonomy."""

    features = extract_type_features(equation)
    if _matches_type_5_signature(features):
        return "5"
    if _matches_type_4_signature(features):
        return "4"
    if _matches_type_2_signature(features):
        return "2"
    return "S"


def implication_rate_from_types(source_type: CoarseType, target_type: CoarseType) -> float:
    return COARSE_IMPLICATION_RATE_TABLE[source_type][target_type]


def predict_implication_from_types(
    source_type: CoarseType,
    target_type: CoarseType,
    threshold: float = 0.5,
) -> bool:
    return implication_rate_from_types(source_type, target_type) > threshold


def implication_rate_from_equations(source_equation: str, target_equation: str) -> float:
    return implication_rate_from_types(
        classify_coarse_type(source_equation),
        classify_coarse_type(target_equation),
    )


def predict_implication_from_equations(
    source_equation: str,
    target_equation: str,
    threshold: float = 0.5,
) -> bool:
    return implication_rate_from_equations(source_equation, target_equation) > threshold


def rate_matrix() -> list[list[float]]:
    return [
        [COARSE_IMPLICATION_RATE_TABLE[source][target] for target in TYPE_ORDER]
        for source in TYPE_ORDER
    ]


__all__ = [
    "COARSE_IMPLICATION_RATE_TABLE",
    "CoarseType",
    "EquationTypeFeatures",
    "TYPE_ORDER",
    "classify_coarse_type",
    "extract_type_features",
    "implication_rate_from_equations",
    "implication_rate_from_types",
    "predict_implication_from_equations",
    "predict_implication_from_types",
    "rate_matrix",
]
