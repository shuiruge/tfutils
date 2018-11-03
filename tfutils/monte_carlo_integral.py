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
