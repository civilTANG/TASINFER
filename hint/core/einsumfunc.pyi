from numpy import ndarray
from typing import Union, Iterable, Any, Optional

def einsum(*operands: Union[Iterable[ndarray], Iterable, int, float],
           **kwargs: Any) -> Any: ...
