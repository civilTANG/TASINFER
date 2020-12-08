import numpy as np
from numpy import ndarray
from typing import Union,Iterable,Any,Optional

def repeat(a: Union[ndarray, Iterable, int, float],
           repeats: Union[int, ndarray, Iterable, float[int]],
           axis: Optional[int] = None) -> Any: ...

def sum(a: Union[ndarray, Iterable, int, float],
        axis: Union[None, int, Iterable, tuple[int]] = None,
        dtype: Optional[object] = None,
        out: Optional[ndarray] = None,
        keepdims: Optional[bool] = np._NoValue,
        initial: Union[int, float, complex, None] = np._NoValue) -> Any:...

def cumprod(a: Union[ndarray, Iterable, int, float],
            axis: Optional[int] = None,
            dtype: Optional[object] = None,
            out: Optional[ndarray] = None) -> Any:...

def nonzero(a: Union[ndarray, Iterable, int, float]) -> tuple:...

def transpose(a: Union[ndarray, Iterable, int, float],
              axes: Optional[Iterable[int]] = None) -> Any: ...

def sort(a: Union[ndarray, Iterable, int, float],
         axis: Optional[int] = -1,
         kind: Optional[str] = 'quicksort',
         order: Union[str, Iterable[str], None] = None) -> Any: ...

def argmax(a: Union[ndarray, Iterable, int, float],
           axis: Optional[int] = None,
           out: Union[ndarray, Iterable, int, float, None] = None)-> Any:...

def mean(a: Union[ndarray, Iterable, int, float],
         axis: Union[None, int, Iterable, tuple[int]] = None,
         dtype: Optional[object] = None,
         out: Optional[ndarray] = None,
         keepdims: Optional[bool] = np._NoValue) -> Any: ...


def std(a: Union[ndarray, Iterable, int, float],
        axis: Union[None, int, Iterable, tuple[int]] = None,
        dtype: Optional[object] = None,
        out: Optional[ndarray] = None,
        ddof: Optional[int] = 0,
        keepdims: Optional[bool] = np._NoValue) -> ndarray:...

def argmax(a: Union[ndarray, Iterable, int, float],
           axis: Optional[int] = None,
           out: Union[ndarray, Iterable, int, float, None] = None)-> Union[ndarray, int]: ...