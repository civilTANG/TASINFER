from numpy import ndarray
from typing import Union, Iterable, Any, Optional

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