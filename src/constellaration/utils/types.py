import jaxtyping as jt
import numpy as np

NpOrJaxArray = np.ndarray | jt.Array
"""Annotation for either a numpy or JAX array.

Use this for function arguments if your function can handle both array types.

Use this as a return type ONLY if all your callers need to be able to handle JAX arrays.
Otherwise, use `NpOrJaxArrayT` instead for both the input arguments and the return type,
so callers that are not aware of JAX can still use numpy functions.
"""

ScalarFloat = float | jt.Float[NpOrJaxArray, " "]
"""Either a normal Python float or a scalar NpOrJaxArray.

Useful for functions that used to take Python floats and are now being jaxified.
"""
