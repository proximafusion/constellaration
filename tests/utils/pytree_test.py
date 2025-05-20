import jax.tree_util as jtu
import numpy as np
import pydantic
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.utils import pytree, types


class MyData(pydantic.BaseModel):
    a: str
    b: int
    c: float


class MyOtherData(pydantic_numpy.BaseModelWithNumpy):
    x: float
    y: float
    z: np.ndarray
    d: dict[str, float]


pytree.register_pydantic_data(MyData, meta_fields=["b"])
pytree.register_pydantic_data(MyOtherData)


def test_flatten() -> None:
    data = MyData(a="a", b=1, c=2.3)

    children, aux_data = pytree.pydantic_flatten(data, ["b"])
    assert children == (
        (jtu.GetAttrKey("a"), "a"),
        (jtu.GetAttrKey("c"), 2.3),
    )
    assert aux_data == (
        MyData,
        ("a", "c"),
        ("b",),
        (1,),
    )


def test_unflatten() -> None:
    aux_data = (
        MyData,
        ("a", "c"),
        ("b",),
        (1,),
    )
    children = ("a", 2.3)

    data = pytree.pydantic_unflatten(aux_data, children)
    assert data == MyData(a="a", b=1, c=2.3)


def test_mask_and_ravel_roundtrip_2d_array() -> None:
    data = MyOtherData(
        x=1.0,
        y=2.0,
        z=np.array([[3.0, 4.0], [5.0, 6.0]]),
        d={"a": 6.0, "b": 7.0},
    )
    mask = MyOtherData(
        x=True,
        y=False,
        z=np.array([[True, False], [True, True]]),
        d={"a": True, "b": False},
    )
    flat, unravel_fn = pytree.mask_and_ravel(data, mask)
    assert isinstance(flat, types.NpOrJaxArray)
    assert len(flat.shape) == 1
    data_unraveled = unravel_fn(flat)
    assert jtu.tree_structure(data_unraveled) == jtu.tree_structure(data)
    assert jtu.tree_all(jtu.tree_map(np.allclose, data_unraveled, data))
