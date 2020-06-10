from typing import Callable, Dict, Tuple, Any, Optional, Text
TFunction = Callable[..., Any]

import pickle
import functools

import log


# _PROVIDERS can be set by external code via `register_provider`
_PROVIDERS: Dict[Text, Text] = {}

def register_provider(fn: TFunction, fp: Text):
    qualname = fn.__qualname__
    _PROVIDERS[qualname] = fp

def memcached(_fn: Optional[TFunction] = None, *, ignore_self: bool = False):
    def decorator(fn: TFunction):
        checked = False
        mem: Dict[Tuple[Any, ...], Any] = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal checked
            nonlocal mem

            # load disk cache at the first time
            if not checked:
                checked = True
                fn_qualname = fn.__qualname__
                if fn_qualname in _PROVIDERS:
                    with open(_PROVIDERS[fn_qualname], 'rb') as fp:
                        mem = pickle.load(fp)
                    log.debug(f'cache loaded: "{fp.name}" for function "{fn_qualname}".')
                else:
                    log.debug('no cache loaded.')

            key = args[1:] if ignore_self else args
            if key not in mem:
                mem[key] = fn(*args, **kwargs)

            return mem[key]

        return wrapper

    if _fn is None:
        return decorator
    else:
        return decorator(_fn)
