from typing import (
    Callable, Dict, Tuple, Any,
    Optional, Text, TypeVar
)

import sys

if sys.version_info.minor >= 8:  # Protocol is only available in Python 3.8+
    from typing import Protocol
    # from interface import ModelInterface  # circular import
    class _ModelInterface(Protocol):
        def decode_data(self, data: Any, **kwargs) -> Any: ...
        def process(self, smiles: Text, **kwargs) -> Any: ...

    # https://stackoverflow.com/a/59406717: accept subclasses in callable arguments
    # however, it seems that TypeVar simply shuts up mypy about `TProcessFn`, even
    # if ModelInterface does not match the signature of the protocol `_ModelInterface`.
    TModelInterface = TypeVar('TModelInterface', bound=_ModelInterface)

# ModelInterface.process
TProcessFn = Callable[[TModelInterface, Text], Any]

import pickle
import functools

import log


# cache file paths stored in _PROVIDERS. For compatibility with "spawn" method.
# _PROVIDERS can be set by external code via `register_provider`
_PROVIDERS: Dict[Text, Text] = {}

def register_provider(fn: Callable, fp: Text):
    qualname = fn.__qualname__
    _PROVIDERS[qualname] = fp

def memcached(_fn: Optional[Callable] = None, *, ignore_self: bool = False):
    '''
    ignore_self: for class member functions: the first argument `self`
        will not be part of the key.
    '''

    def decorator(fn: Callable):
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

def smiles_cache(fn: TProcessFn):
    cached_fn = memcached(fn, ignore_self=True)
    mem: Dict[Text, Any] = {}

    @functools.wraps(fn)
    def wrapper(self: TModelInterface, smiles: Text, **kwargs):
        nonlocal cached_fn
        nonlocal mem

        if smiles not in mem:
            data = cached_fn(self, smiles, **kwargs)
            mem[smiles] = self.decode_data(data, **kwargs)
        return mem[smiles]

    return wrapper