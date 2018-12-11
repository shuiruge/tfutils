import inspect


def inheritdocstring(cls):
    """Decorator for inherting docstring from parent class.

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
    for base in inspect.getmro(cls)[1:]:  # the first is the `cls` itself.
        if base.__name__ in ('abc.ABC', 'ABC', 'object'):
            continue
        if base.__doc__ is not None:
            cls.__doc__ += ('\n\n  Inheriting from class {0}:\n\n{1}'
                            .format(base.__name__, base.__doc__))
    return cls
