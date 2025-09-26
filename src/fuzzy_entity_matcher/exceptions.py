# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
Custom exceptions for fuzzy entity matcher.
"""


class FuzzyMatcherError(Exception):
    """Base exception for fuzzy matcher operations."""

    pass


class BenchmarkError(FuzzyMatcherError):
    """Exception raised during benchmark operations."""

    def __init__(self, operation: str, reason: str, retry_possible: bool = False):
        self.operation = operation
        self.reason = reason
        self.retry_possible = retry_possible
        super().__init__(f"Benchmark failed during {operation}: {reason}")


class DatabaseError(FuzzyMatcherError):
    """Exception raised during database operations."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Database operation '{operation}' failed: {reason}")


class ValidationError(FuzzyMatcherError):
    """Exception raised for input validation errors."""

    def __init__(self, parameter: str, value: any, expected: str):
        self.parameter = parameter
        self.value = value
        self.expected = expected
        super().__init__(f"Invalid {parameter}: got {value}, expected {expected}")
