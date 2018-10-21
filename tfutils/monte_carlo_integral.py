
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
