import dataclasses
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pydantic

from constellaration.utils.types import NpOrJaxArray


def pydantic_flatten(
    something: pydantic.BaseModel,
    meta_fields: list[str] | None = None,
) -> tuple[
    tuple,
    tuple[type[pydantic.BaseModel], tuple[str, ...], tuple[str, ...], tuple[Any, ...]],
]:
    """A jax pytree compatible implementation of flattening pydantic objects.

    A general pydantic.BaseModel is flattened into a tuple of children and aux_data.
    aux_data is used to reconstruct the same type of Pydantic in unflatten. meta_fields
    are used to specify fields that should not be visible to pytree operations. See
    https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_dataclass.html
    for details.
    """

    if meta_fields is None:
        meta_fields = []

    data_fields = [
        field for field in type(something).model_fields if field not in meta_fields
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


PytreeT = TypeVar("PytreeT")
LeafT = TypeVar("LeafT")


@dataclasses.dataclass
class _LeafInfo:
    leaf: NpOrJaxArray | float
    mask: NpOrJaxArray | None
    is_scalar: bool

    @property
    def is_masked(self) -> bool:
        return self.mask is not None


class _UnravelFn:
    """A picklable callable that reconstructs a pytree from a flat parameter vector.

    It stores all necessary info (the original pytree treedef and a list of per-leaf
    info) to reassemble the structure.
    """

    _leaves_info: list[_LeafInfo]
    _treedef: jax.tree_util.PyTreeDef

    def __init__(
        self, leaves_info: list[_LeafInfo], treedef: jax.tree_util.PyTreeDef
    ) -> None:
        self._leaves_info = leaves_info
        self._treedef = treedef

    def __call__(self, flat: NpOrJaxArray) -> Any:
        new_leaves: list[NpOrJaxArray | float] = []
        pos = 0
        for leaf_info in self._leaves_info:
            if leaf_info.is_masked:
                assert leaf_info.mask is not None
                # Determine how many entries were extracted for this leaf.
                n_true = int(leaf_info.mask.sum())
                segment = flat[pos : pos + n_true]
                pos += n_true
                # Find the indices where the mask is True.
                idx = jnp.nonzero(leaf_info.mask)
                if leaf_info.is_scalar:
                    # If the leaf is a scalar, `segment` has a single entry.
                    new_leaf = segment[0].item()
                else:
                    assert isinstance(leaf_info.leaf, NpOrJaxArray)
                    # Replace the masked positions with the new parameters.
                    new_leaf = jnp.asarray(leaf_info.leaf).at[idx].set(segment)
                new_leaves.append(new_leaf)
            else:
                new_leaves.append(leaf_info.leaf)
        if pos != len(flat):
            raise ValueError(
                f"Expected to consume {len(flat)} values, but only consumed {pos}."
            )
        return jax.tree_util.tree_unflatten(self._treedef, new_leaves)


def mask_and_ravel(
    pytree: PytreeT,
    mask: PytreeT,
) -> tuple[NpOrJaxArray, Callable[[NpOrJaxArray], PytreeT]]:
    """Ravel a pytree but only include entries where the corresponding mask is True.
    Returns a 1D array and a picklable unravel function that reconstructs the pytree
    by inserting new values into the masked locations while keeping other
    entries unchanged.

    Args:
        pytree: The pytree to be flattened.
        mask: A pytree of booleans with the same structure as `pytree`. True indicates
            the entries to be flattened.

    Returns:
        A tuple of the flat array and the unravel function.
    """

    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    mask_leaves, _ = jax.tree_util.tree_flatten(mask)

    flat_segments: list[NpOrJaxArray] = []
    leaves_info: list[_LeafInfo] = []
    for leaf, leaf_mask in zip(leaves, mask_leaves):
        if isinstance(leaf_mask, NpOrJaxArray) and leaf_mask.dtype == jnp.bool_:
            selected = leaf[leaf_mask]
            flat_segments.append(jnp.atleast_1d(selected.ravel()))
            leaves_info.append(_LeafInfo(mask=leaf_mask, leaf=leaf, is_scalar=False))
        elif isinstance(leaf_mask, float):
            if leaf_mask not in (0.0, 1.0):
                raise ValueError("Only 0.0 and 1.0 are supported as float masks.")
            is_scalar = True
            if leaf_mask == 1.0:
                flat_segments.append(jnp.atleast_1d(leaf))
                leaves_info.append(
                    _LeafInfo(mask=jnp.array([True]), leaf=leaf, is_scalar=is_scalar)
                )
            else:
                leaves_info.append(_LeafInfo(leaf=leaf, mask=None, is_scalar=is_scalar))
        else:
            raise ValueError(
                f"Unsupported mask type {type(leaf_mask)} for leaf {leaf}."
            )

    if flat_segments:
        flat = jnp.concatenate(flat_segments)
    else:
        flat = jnp.array([])

    unravel_fn = _UnravelFn(leaves_info, treedef)
    return flat, unravel_fn
