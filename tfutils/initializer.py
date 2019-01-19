import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling


class GlorotInitializer(VarianceScaling):
  """C.f. https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/init_ops.py#L1120  # noqa:E501

  Comparing to the original `tf.GlorotInitializer`, we add `scale` argument to
  `self.__init__()`.
  
  Args:
    scale: Float.
    distribution: String, like "truncated_normal", "uniform".
    seed: Integer.
    dtype: Dtype.
  """

  def __init__(self,
               scale=1.0,
               distribution='truncated_normal',
               seed=None,
               dtype=tf.float32):
    super().__init__(scale=scale,
                     mode="fan_avg",
                     distribution=distribution,
                     seed=seed,
                     dtype=dtype)
