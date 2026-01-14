"""Common methods for vector scaling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType


@lazydispatch
def _vector_factory(base: object, num_classes: int, device: object) -> type[Any]:
    message = f"No Vector Scaling implementation for base={type(base)}, device={type(device)}"
    raise NotImplementedError(message)


def register_vector_factory(key: LazyType) -> Callable:
    """Returns a decorator to register a class in the vector factory."""
    return _vector_factory.register(key)


def vector_scaling(base: object, num_classes: int, device: object) -> object:
    """Dispatches different implementations for vector scaling."""
    return _vector_factory(base, num_classes, device)
