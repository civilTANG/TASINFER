import numpy as np
from numpy import ndarray
from typing import Union,Iterable,Any,Optional

def where(condition: Union[ndarray, Iterable, int, float, bool],
          x: Union[ndarray, Iterable, int, float] = None,
          y: Union[ndarray, Iterable, int, float] = None) -> ndarray:...

def concatenate(arrays: Any,
                axis: Optional[int] = None,
                out: Optional[ndarray] = None) -> ndarray:...

def dot(a: Union[ndarray, Iterable, int, float],
        b: Union[ndarray, Iterable, int, float],
        out: Optional[ndarray] = None) -> ndarray: ...

def arange(start: Union[int, float, complex, None] = None,
           *args: Any,
           **kwargs: Any) -> ndarray:...