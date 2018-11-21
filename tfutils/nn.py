import tensorflow as tf


def residual_block_wrapper(inputs, outputs):
  with tf.variable_scope('residual_block'):
    if inputs.shape == outputs.shape:
      return outputs + inputs

    def get_shape(x):
      return x.get_shape().as_list()

    ishape, oshape = [get_shape(_) for _ in (inputs, outputs)]
    irank, orank = [len(_) for _ in (ishape, oshape)]

    for i in range(min(irank, orank)):
      if ishape[i] != oshape[i]:
        max_batch_axis = i - 1
        break
    else:
      max_batch_axis = i

    # TODO: Determine the shape of weights.
    weight_shape = ...
    weights = tf.get_variable(
        name='weights', shape=weight_shape, dtype=inputs.dtype)
    return outputs + tf.matmul(weights, inputs)
