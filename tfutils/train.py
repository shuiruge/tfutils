import os
import tensorflow as tf


def clip(gradient, min_clip_value=-1e+0,  max_clip_value=1e+0):
  """
  Examples:
  >>> loss = ...
  >>> optimizer = tf.train.AdamOptimizer()
  >>> gvs = optimizer.compute_gradients(loss)
  >>> gvs = [(clip(grad), var) for grad, var in gvs]
  >>> train_op = optimizer.apply_gradients(gvs)

  Args:
    gradient: Tensor.
    min_clip_value: Negative float.
    max_clip_value: Positive float.

  Returns:
    Tensor.
  """
  assert (min_clip_value < 0 and max_clip_value > 0)
  return tf.clip_by_value(gradient, min_clip_value, max_clip_value)


def save_variables(session, scope, save_dir):
    """Saves the trained variables within the scope from session
    to disk.
    
    Args:
        session: An instance of `tf.Session`.
        scope: String.
        save_dir: String.
    """
    pretrained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=scope)
    saver = tf.train.Saver(pretrained_vars)

    ckpt_path = os.path.join(save_dir, '{}_scope.ckpt'.format(scope))
    saver.save(session, ckpt_path)


def restore_variables(session, scope, save_dir):
    """Restores the pre-trained variables within the scope from disk
    to session. Cooperating with the function `save_vars`.
    
    Args:
        session: An instance of `tf.Session`.
        scope: String.
        save_dir: String.
    """
    pretrained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=scope)
    saver = tf.train.Saver(pretrained_vars)

    ckpt_path = os.path.join(save_dir, '{}_scope.ckpt'.format(scope))
    saver.restore(session, ckpt_path)
