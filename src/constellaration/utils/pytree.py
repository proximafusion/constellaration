from collections.abc import Iterable
from typing import Any

import jax.tree_util as jtu
import pydantic


def pydantic_flatten(
    something: pydantic.BaseModel,
    meta_fields: list[str] | None = None,
) -> tuple[
    tuple,
    tuple[type[pydantic.BaseModel], tuple[str, ...], tuple[str, ...], tuple[Any, ...]],
]:
    """A jax pytree compatible implementation of flattening pydantic objects.

    A general pydantic.BaseModel is flattened into a tuple of children and aux_data. aux_data is
    used to reconstruct the same type of Pydantic in unflatten. meta_fields are used
    to specify fields that should not be visible to pytree operations. See
    https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_dataclass.html
    for details.
    """

    if meta_fields is None:
        meta_fields = []

    data_fields = [
        field for field in something.model_fields if field not in meta_fields
    ]

    aux_data = (
        type(something),
        tuple(data_fields),
        tuple(meta_fields),
        tuple(getattr(something, field) for field in meta_fields),
    )
    children = tuple(
        (jtu.GetAttrKey(field), getattr(something, field)) for field in data_fields
    )
    return children, aux_data


def pydantic_unflatten(
    aux_data: tuple[
        type[pydantic.BaseModel], tuple[str, ...], tuple[str, ...], tuple[Any, ...]
    ],
    children: Iterable[Any],
) -> pydantic.BaseModel:
    """A jax pytree compatible implementation of un-flattening Pydantic objects.

    pydantic_unflatten is the inverse of pydantic_flatten. Using the type annotation
    and children-meta_fields split in aux_data, it constructs a new Pydantic object.

    Note that this object will typically not validate against the Pydantic
    specification. Pytrees are used to manipulate the data stored in leafs and thus can
    contain anything. Pytrees only make statements about the structure of the data, not
    the content. An example of data manipulation might be to filter data in a pytree by
    type:

    ```
    my_data = MyModel(...)
    strings = jax.tree_map(lambda x: x if isinstance(x, str) else None, my_data)
    ```

    See
    https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    for details.
    """
    cls, data_fields, meta_fields, metas = aux_data
    kwargs = {
        **dict(zip(data_fields, children)),
        **dict(zip(meta_fields, metas)),
    }
    return cls.model_construct(**kwargs)


def register_pydantic_data(cls: type, meta_fields: list[str] | None = None) -> type:
    """Register a pydantic.BaseModel class for jax pytree compatibility.

    Args:
        cls: The pydantic.BaseModel class to register.
        meta_fields: Fields that should be part of aux_data rather becoming children.
            Defaults to None.

    Returns:
        The registered class, to enable the function to be used as a decorator.
    """
    jtu.register_pytree_with_keys(
        cls,
        lambda x: pydantic_flatten(x, meta_fields),
        pydantic_unflatten,
    )

    return cls
