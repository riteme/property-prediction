from typing import Callable, Dict, Tuple, Any, Optional

import functools

TFunction = Callable[..., Any]

def memcached(_fn: Optional[TFunction] = None, *, ignore_self: bool = False):
    def decorator(fn: TFunction):
        mem: Dict[Tuple[Any, ...], Any] = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = args[1:] if ignore_self else args
            if key not in mem:
                mem[key] = fn(*args, **kwargs)
            return mem[key]

        return wrapper

    if _fn is None:
        return decorator
    else:
        return decorator(_fn)
