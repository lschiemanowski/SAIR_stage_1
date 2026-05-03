#!/usr/bin/env python3
"""Convert infix equations to postfix notation.

The default input is `equations.txt`, with one equation per line. Each equation
must have exactly one `=` separating the left and right terms. Terms are parsed
over variables, parentheses, and the binary operator `*`.

The default output is `equations_postfix.txt`. Each output line keeps the same
equation delimiter, but writes each side in postfix token order:

    x * (y * z) = w

becomes:

    x y z * * = w
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("equations.txt")
DEFAULT_OUTPUT = Path("equations_postfix.txt")

TOKEN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|\*|\(|\))")


@dataclass(frozen=True)
class Node:
    token: str
    left: "Node | None" = None
    right: "Node | None" = None

    def postfix_tokens(self) -> list[str]:
        if self.left is None and self.right is None:
            return [self.token]
        if self.left is None or self.right is None:
            raise ValueError(f"Malformed operator node: {self!r}")
        return self.left.postfix_tokens() + self.right.postfix_tokens() + [self.token]


def tokenize(term: str) -> list[str]:
    tokens: list[str] = []
    position = 0

    while position < len(term):
        match = TOKEN_RE.match(term, position)
        if not match:
            if term[position:].strip() == "":
                break
            raise ValueError(
                f"Unexpected character at position {position + 1}: {term[position]!r}"
            )
        tokens.append(match.group(1))
        position = match.end()

    return tokens


class Parser:
    """Recursive-descent parser for terms built from variables and `*`."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> Node:
        if not self.tokens:
            raise ValueError("Empty term")

        node = self.parse_term()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token after complete term: {self.peek()!r}")
        return node

    def parse_term(self) -> Node:
        """Parse a left-associative chain of factors joined by `*`."""
        node = self.parse_factor()

        while self.peek() == "*":
            self.consume("*")
            right = self.parse_factor()
            node = Node("*", node, right)

        return node

    def parse_factor(self) -> Node:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of term")

        if token == "(":
            self.consume("(")
            node = self.parse_term()
            self.consume(")")
            return node

        if token in {"*", ")"}:
            raise ValueError(f"Expected variable or parenthesized term, got {token!r}")

        self.position += 1
        return Node(token)

    def peek(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self, expected: str) -> None:
        token = self.peek()
        if token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.position += 1


def term_to_postfix(term: str) -> str:
    tree = Parser(tokenize(term)).parse()
    return " ".join(tree.postfix_tokens())


def equation_to_postfix(line: str, line_number: int) -> str:
    if line.count("=") != 1:
        raise ValueError(f"Line {line_number}: expected exactly one '='")

    lhs, rhs = line.split("=")
    try:
        return f"{term_to_postfix(lhs)} = {term_to_postfix(rhs)}"
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: {exc}") from exc


def convert_file(input_path: Path, output_path: Path) -> int:
    converted_count = 0

    with input_path.open() as input_file, output_path.open("w") as output_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            output_file.write(equation_to_postfix(line, line_number))
            output_file.write("\n")
            converted_count += 1

    return converted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert infix equations to postfix notation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input equation file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output postfix equation file. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted_count = convert_file(args.input, args.output)
    print(f"Converted equations: {converted_count}")
    print(f"Wrote postfix equations: {args.output}")


if __name__ == "__main__":
    main()
