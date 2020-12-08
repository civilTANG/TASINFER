import numpy as np
from numpy import ndarray
from typing import Union,Iterable,Any,Optional

def norm(x: Union[ndarray, Iterable, int, float],
         ord: Union[int, str, None] = None,
         axis: Optional[int] = None,
         keepdims: Optional[bool] = False) -> float:...