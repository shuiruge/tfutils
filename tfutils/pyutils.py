import os
import time
import inspect
import functools
from typing import Iterable, List


def inheritdocstring(cls):
    """Decorator for inherting docstring from parent class, for both the class
    and its methods.

    Examples:
    >>> class A:
    >>>     '''This is A.'''
    >>>     pass
    >>>
    >>> @inheritdocstring
    >>> class B(A):
    >>>     '''This is B.'''
    >>>     pass
    >>>
    >>> print(B.__doc__)
    """
    base_classes = inspect.getmro(cls)[1:]  # the first is the `cls` itself.

    # Inherit docstring of class
    cls.__doc__ = ''  # initialize.
    for base in base_classes:
        if base.__name__ in ('abc.ABC', 'ABC', 'object'):
            continue
        if base.__doc__ is not None:
            cls.__doc__ += ('\n\nInheriting from class {0} --\n\n{1}'
                            .format(base.__name__, base.__doc__))

    methods = [attr_name for attr_name in dir(cls)
               if is_method(cls, attr_name)]

    # Inherit docstrings for methods
    def get_method_docstring(method):
        # Initialize
        docstring = getattr(cls, method).__doc__
        if docstring is None:
            docstring = ''

        for base in base_classes:
            if not hasattr(base, method):
                continue
            base_docstring = getattr(base, method).__doc__
            if base_docstring and base_docstring != docstring:
                docstring = base_docstring + '\n\n' + docstring
        return docstring

    for method in methods:
        getattr(cls, method).__doc__ = get_method_docstring(method)

    return cls


def is_method(cls, attr_name):
    """Auxillary function for `inheritdocstring()`. Returns if the attribute
    with name `attr_name` is a method (i.e. callable) of the class `cls`.

    Args:
        cls: A class.
        attr_name: String.

    Returns:
        Boolean.
    """
    if attr_name.startswith('__'):  # collect no intrinsic method.
        return False
    attr = getattr(cls, attr_name)
    if hasattr(attr, '__call__'):  # thus being a method.
        return True
    return False


def lazy_property(function):
    """Decorator for lazy-property.

    Forked from: https://danijar.com/structuring-your-tensorflow-models/
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def ensure_directory(path_to_dir):
    """Creates the direcotry in path `path_to_dir` if not exists."""
    try:
        os.makedirs(path_to_dir)
    except FileExistsError:
        # Directory already exists, then noting to do.
        pass


Batch = List[object]


def chunck(size: int, elems: Iterable[object]) -> Iterable[Batch]:
    """Yields batch of size `size` of the elements `elems`.

    The last batch may have size smaller than `size` if and only if
    `len(elems) // size != 0`.
    """
    batch, batch_size = [], 0
    for elem in elems:
        batch.append(elem)
        batch_size += 1
        if batch_size == size:
            batch_to_yield = batch[:]
            batch, batch_size = [], 0
            yield batch_to_yield
    yield batch


class Timer(object):

    def __init__(self, name='', verbose=True):
        self._verbose = verbose
        self._name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self):
        self._end = time.time()
        self._interval = self._end - self._start
        if self._verbose:
            if self._name:
                print('=> {} costs {} secs.'
                      .format(self._name, self._interval))
            else:
                print('=> Costs {} secs.'.format(self._interval))
