from tfutils.pyutils import inheritdocstring


# Test `inheritdocstring()`

class A(object):
    """This is A."""

    def method_1(self):
        """This is method_1."""
        pass


@inheritdocstring
class B(A):
    """This is B."""
    pass


@inheritdocstring
class C(A):
    """This is C."""
    def method_1(self):
        """Override."""
        pass

    def method_2(self):
        """New method."""


if __name__ == '__main__':

    print(B.__doc__)
    print('\n\n\n')

    print(B.method_1.__doc__)
    print('\n\n\n')

    print(C.method_1.__doc__)
    print('\n\n\n')

    print(C.method_2.__doc__)
    print('\n\n\n')
