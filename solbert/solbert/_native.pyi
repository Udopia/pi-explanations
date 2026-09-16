from collections.abc import Iterator, Sequence
from typing import TypeAlias

Formula: TypeAlias = Sequence[Sequence[int]]
Result: TypeAlias = list[list[int]] | str


def compute_prime_implicants(
    formula: Formula,
    inputs: Sequence[int],
    time_limit: int = ...,
    memory_limit: int = ...,
) -> Result: ...


def compute_prime_implicants2(
    formula: Formula,
    inputs: Sequence[int],
    time_limit: int = ...,
    memory_limit: int = ...,
) -> Result: ...


def enumerate_models(
    formula: Formula,
    inputs: Sequence[int],
    time_limit: int = ...,
    memory_limit: int = ...,
) -> Result: ...


class model_iterator(Iterator[list[int]]):
    def __init__(self, formula: Formula, inputs: Sequence[int]) -> None: ...
    def __next__(self) -> list[int]: ...


class monotonic_circuit:
    def __init__(self, formula: Formula, inputs: Sequence[int]) -> None: ...
    def append_root(self, inputs: Sequence[int]) -> None: ...
    def update_prime_implicants(self) -> None: ...
    def get_primp(self) -> list[list[int]]: ...