from numpy import ndarray
from typing import Union,Iterable,Any,Optional

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
