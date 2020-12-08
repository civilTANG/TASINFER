from numpy import ndarray
from typing import Union, Iterable, Any, Optional

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