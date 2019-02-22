import tensorflow as tf


def variable_summaries(var, name='summaries'):
  """Attaches a lot of summaries to a Tensor `var` (for TensorBoard
  visualization).

  Forked from: https://www.tensorflow.org/guide/summaries_and_tensorboard

  Args:
    var: Tensor.
    name: String.
  """
  with tf.name_scope(name):
    tf.summary.histogram('histogram', var)

    shape = var.get_shape().as_list()
    if not shape:  # scalar.
      tf.summary.scalar('mean', var)

    else:  # multi-component tensor
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
