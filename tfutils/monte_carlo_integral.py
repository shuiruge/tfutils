import numpy as np
import tensorflow as tf


EPSILON = 1e-8


class MonteCarloIntegral(object):
  """
  Args:
    value: Tensor.
    variance: Tensor, with the same shape and dtype as `value`.

  Raises:
    ValueError: The shape or the dtype of `value` and `variance` is
      not the same.
  """

  def __init__(self, value, variance):
    if value.shape != variance.shape or value.dtype != variance.dtype:
      raise ValueError('The shape or the dtype of `value` and '
                       '`variance` is not the same.')

    self._value = value
    self._variance = variance
    self._error = tf.sqrt(variance)
    self._relative_error = self._error / (value + EPSILON)

  @property
  def value(self):
    return self._value

  @property
  def variance(self):
    return self._variance

  @property
  def error(self):
    return self._error

  @property
  def relative_error(self):
    return self._relative_error

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype

  def __add__(self, other):
    """
    C.f.: https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables  # noqa: E501

    Args:
      other: A `MonteCarloIntegral` instance.

    Returns:
      A (new) `MonteCarloIntegral` instance.

    Raises:
      TypeError: If `other` is not a `MonteCarloIntegral` instance.
    """
    if not isinstance(other, MonteCarloIntegral):
      raise TypeError('Arg `other` should be an instance of '
                      '`MonteCarloIntegral`, but being a {}'
                      .format(type(other)))

    return MonteCarloIntegral(value=(self.value + other.value),
                              variance=(self.variance + other.variance))


def monte_carlo_integrate(
        integrands,
        axes=[0],
        n_samples=None,
        name='monte_carlo_integrate'):
  r"""
  Definition:
    ```math
    Let $x \in \mathcal{X}$ sampled from measurement $P_X$. The integral
    $\int_{\mathcal{X}} dx p(x) f(x)$ can be approximated by Monte-Carlo
    integral

    \begin{equaiton}
        \int_{\mathcal{X}} dx p(x) f(x) \approx
            \mathbb{E}_{x \sim P_X} \left[ f(x) \right],
    \end{equation}

    which carries an relative error

    $$\sqrt{\frac{\textrm{Var}_{x \sim P_X} \left[ f(x) \right]}{N}},$$

    where $N$ the number of samples. The set $\{ f(x) \mid x \sim P_X \}$
    is called "integrands" herein.
    ```

  Examples:
    >>> # Example 1
    >>> # To integrate f(x) := x^2
    >>> xs = tf.random_uniform(shape=(128), dtype='float32')
    >>> integrands = tf.square(xs)
    >>> mc_int = monte_carlo_integrate(integrands, axes=[0])
    >>> #
    >>> # Example 2
    >>> # To integrate f(x) := x^2, but with non-trivial `axes`
    >>> xs = tf.random_uniform(shape=(16, 8), dtype='float32')
    >>> integrands = tf.square(xs)
    >>> mc_int = monte_carlo_integrate(integrands, axes=[0, 1])

  Args:
    integrand: Tensor.
    axes: Iterable of non-negative integers, as the axes to be integrated
      over.
    n_samples: Positive integer, or `None`, as the `N` in the formula.
      If the sample-shape is static, then put it as `None` and the number
      of samples will be automatically computed. However, if the sample-shape
      is dynamical, like being constructed by placeholder or by variable,
      this argument must be fulfilled as an integer.
    name: String.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):

    if n_samples is None:  # compute the `n_samples` automatically.
      integrands_shape = integrands.get_shape().as_list()
      sample_shape = [integrands_shape[axis] for axis in axes]
      if None in sample_shape:
        raise NonStaticSampleShapeError(
            'Ensure that the sample-shape is static, not being placeholder '
            'nor variable. Otherwise, you should set the argument `n_samples` '
            'manually. The "integrand-samples" is {}'.format(integrands))
      n_samples = np.product(sample_shape)

    mean, var = tf.nn.moments(integrands, axes)
    n_samples = tf.cast(n_samples, dtype=integrands.dtype)
    return MonteCarloIntegral(value=mean, variance=(var / n_samples))


class NonStaticSampleShapeError(Exception):
  """Auxillary exception."""
  pass
