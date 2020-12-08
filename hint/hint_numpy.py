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

def einsum(*operands: Union[Iterable[ndarray], Iterable, int, float],
           **kwargs: Any) -> Any: ...

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


def fill(self, value: Union[int, float, complex]) -> None: ...

def sum(self, axis: Any = None, dtype: Any = None, out: Any = None,keepdims: bool = False) -> None:...

def dot(self,
        b: Any,
        out: Any = None) -> None: ...

def astype(self,
           dtype: Union[str, object],
           order: Optional[str] = 'K',
           casting: Optional[str] = 'unsafe',
           subok: Optional[bool] = True,
           copy: Optional[bool] = True) -> ndarray: ...

def reshape(self,
            shape: Any,
            order: str = 'C') -> None: ...

def tolist(self) -> list: ...

def full(shape: Union[int, Iterable[int]],
         fill_value: Union[int, float, complex],
         dtype: Optional[object] = None,
         order: Optional[str] = 'C') -> ndarray: ...

def ones(shape: Union[int, Iterable[int]],
         dtype: Optional[object] = None,
         order: Optional[str] = 'C') -> ndarray: ...

def tensordot(a: Union[ndarray, Iterable, int, float],
              b: Union[ndarray, Iterable, int, float],
              axes: Any = 2) -> Optional[Any]:...
def tile(A: Union[ndarray, Iterable, int, float],
         reps: Union[ndarray, Iterable, int, float]) -> Any: ...

def column_stack(tup: Any) -> Any:...

def vstack(tup: Iterable) -> Any:...

def hstack(tup: Iterable) -> Any:...

def atleast_2d(*arys: Any) -> list:...

def norm(x: Union[ndarray, Iterable, int, float],
         ord: Union[int, str, None] = None,
         axis: Optional[int] = None,
         keepdims: Optional[bool] = False) -> float:...

def empty(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C') -> ndarray: ...

def frompyfunc(func: Any, nin: int, nout: int) -> Any:...


def zeros(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C') -> ndarray: ...

def sqrt(x: Union[float,int, ndarray, Iterable],
         *args: Any,
         **kwargs: Any) -> ndarray:...

def count_nonzero(a: Union[ndarray, Iterable, int, float],
                  axis: Union[int, Iterable, tuple, None] = None) -> int: ...

def array(p_object: Union[ndarray, Iterable, int, float],
          dtype: Optional[object] = None,
          copy: Optional[bool] = True,
          order: Optional[str] = 'K',
          subok: Optional[bool] = False,
          ndmin: Optional[int] = 0) -> ndarray:...

def fromiter(iterable: Any,
             dtype: object,
             count: Optional[int] = -1) -> ndarray:...

def log(x: Union[int,float, ndarray, Iterable],
        *args: Any,
        **kwargs: Any) -> ndarray:...

def absolute(x: Union[int,float, ndarray, Iterable],
             *args: Any,
             **kwargs: Any) -> ndarray: ...

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
         keepdims: Optional[bool] = np._NoValue) -> ndarray: ...

def linspace(start:[float,int],stop:[float,int],num:Optional[int] = 50,
             endpoint:Optional[bool]  = True,
             retstep:Optional[bool] = False,
             dtype:Optional[object] = None,axis:Optional[int] = 0 )-> ndarray: ...

def std(a: Union[ndarray, Iterable, int, float],
        axis: Union[None, int, Iterable, tuple[int]] = None,
        dtype: Optional[object] = None,
        out: Optional[ndarray] = None,
        ddof: Optional[int] = 0,
        keepdims: Optional[bool] = np._NoValue) -> ndarray:...