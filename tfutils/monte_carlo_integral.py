import numpy as np
import tensorflow as tf


class MonteCarloIntegral(object):
  """
  Args:
    value: Tensor.
    error: Tensor, with the same shape and dtype as `value`.

  Raises:
    ValueError: The shape or the dtype of `value` and `error` is not the same.
  """

  def __init__(self, value, error):
    if value.shape != error.shape or value.dtype != error.dtype:
      raise ValueError('The shape or the dtype of `value` and '
                       '`error` is not the same.')

    self.value = value
    self.error = error

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype


def mc_mean_with_nan(samples):
  """Returns the Monte-Carlo-mean of `samples` wherein there may be NaN.

  This function is useful at the starting epoch of training, wherein the
  initialized distributions may be far from compatible with some of the
  provided data.

  However, this function may hide the numerical instability that is
  intrinsic to the model to be trained. So, it is NOT suggested to use
  this function!

  Args:
    samples: Tensor with shape `[n_samples] + rest_shapes`, for any
      `rest_shapes`.

  Returns:
    Tensor with shape `rest_shapes`.
  """
  # [n_samples] + rest_shape
  finite_sample_mask = tf.where(tf.is_nan(samples),
                                tf.zeros_like(samples),
                                tf.ones_like(samples))
  # [1] + rest_shape
  finite_sample_count = tf.reduce_sum(finite_sample_mask,
                                      axis=0, keepdims=True)
  # [n_samples] + rest_shape
  finite_samples = tf.where(tf.is_nan(samples),
                            tf.zeros_like(samples),
                            samples)
  return tf.reduce_sum(finite_samples / (finite_sample_count + 1e-8),
                       axis=0)


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
    A `GANLoss` instance.
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
    return MonteCarloIntegral(value=mean, error=tf.sqrt(var / n_samples))
