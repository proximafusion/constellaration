import jax.tree_util as jtu
import numpy as np
import pydantic
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.utils import pytree


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


def test_flatten():
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


def test_unflatten():
    aux_data = (
        MyData,
        ("a", "c"),
        ("b",),
        (1,),
    )
    children = ("a", 2.3)

    data = pytree.pydantic_unflatten(aux_data, children)
    assert data == MyData(a="a", b=1, c=2.3)
