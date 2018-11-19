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
    value_errors = (value.shape != variance.shape,
                    value.dtype != variance.dtype)
    if any(value_errors):
      raise ValueError('The shape or the dtype of `value` and '
                       '`variance` is not the same.')

    self.value = value
    self.variance = variance

    self.shape = self.value.shape
    self.dtype = self.value.dtype
    self.error = tf.sqrt(self.variance)
    self.relative_error = self.error / (self.value + EPSILON)

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
    name: String.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):
    with tf.name_scope('n_samples'):
      integrands_shape = integrands.get_shape().as_list()

      sample_shape = []
      for axis in axes:
        sample_shape.append(integrands_shape[axis])

      n_samples = np.product(sample_shape)
      n_samples = tf.constant(n_samples, dtype=integrands.dtype)

    mean, var = tf.nn.moments(integrands, axes)
    return MonteCarloIntegral(value=mean, variance=(var / n_samples))
