import tensorflow as tf
from tfutils.monte_carlo_integral import monte_carlo_integrate


def get_entropy(distribution, n_samples=32, name='entropy'):
  """Returns the entropy of the distribution `distribution` as a Monte-Carlo
  integral.

  Args:
    distribution: A `tfp.Distribution` instance.
    n_samples: Positive integer.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):
    samples = distribution.sample(n_samples)
    # shape: # [n_samples] + batch-shape
    integrands = - distribution.log_prob(samples)
    # shape: batch-shape
    return monte_carlo_integrate(integrands, axes=[0])


def get_kl_divergence(p, q, n_samples, name='KL_divergence'):
  """
  Args:
    p: A `tfp.Distribution` instance.
    q: A `tfp.Distribution` instance.
    n_samples: Positive integers.
    name: String.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):
    p_samples = p.sample(n_samples)
    integrands = p.log_prob(p_samples) - q.log_prob(p_samples)
    return monte_carlo_integrate(integrands)


def get_jensen_shannon(p, q, n_samples, name='Jensen_Shannon'):
  """
  Args:
    p: A `tfp.Distribution` instance.
    q: A `tfp.Distribution` instance.
    n_samples: Positive integers.
    name: String.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):
    return (get_kl_divergence(p, q, n_samples) +
            get_kl_divergence(q, p, n_samples)) * 0.5
