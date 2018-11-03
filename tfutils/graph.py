import collections
import numpy as np
import tensorflow as tf
from IPython.display import display, HTML


def get_dependent_variables(tensor):
  """Returns all variables that the tensor `tensor` depends on.

  Forked from: https://stackoverflow.com/a/42861919/1218716

  Args:
    tensor: Tensor.

  Returns:
    List of variables.
  """
  # Initialize
  starting_op = tensor.op
  dependent_vars = []
  queue = collections.deque()
  queue.append(starting_op)
  op_to_var = {var.op: var for var in tf.trainable_variables()}
  visited = set([starting_op])

  while queue:
    op = queue.popleft()
    try:
      dependent_vars.append(op_to_var[op])
    except KeyError:
      # `op` is not a variable, so search its inputs (if any).
      for op_input in op.inputs:
        if op_input.op not in visited:
          queue.append(op_input.op)
          visited.add(op_input.op)

  return dependent_vars


def strip_consts(graph_def, max_const_size=32):
  """Strip large constant values from graph_def. The auxillary function of
  the `show_graph()`."""
  strip_def = tf.GraphDef()
  for n0 in graph_def.node:
    n = strip_def.node.add()
    n.MergeFrom(n0)
    if n.op == 'Const':
      tensor = n.attr['value'].tensor
      size = len(tensor.tensor_content)
      if size > max_const_size:
        tensor.tensor_content = bytes('<stripped %d bytes>' % size,
                                      'utf-8')
  return strip_def


def show_graph(graph_def, max_const_size=32):
  """Visualize TensorFlow graph within jupyter-notebook.

  Fork from: https://stackoverflow.com/a/38192374/1218716

  Args:
    graph_def: An instance of `tf.GraphDef`.

  Examples:
  >>> # To visualize current graph
  >>> show_graph(tf.get_default_graph().as_graph_def())
  >>> # If your graph is saved as pbtxt, you could do
  >>> gdef = tf.GraphDef()
  >>> from google.protobuf import text_format
  >>> text_format.Merge(open("tf_persistent.pbtxt").read(), gdef)
  >>> show_graph(gdef)
  >>> # which shows the graph as in the TensorBoard.
  """
  if hasattr(graph_def, 'as_graph_def'):
    graph_def = graph_def.as_graph_def()
  strip_def = strip_consts(graph_def, max_const_size=max_const_size)
  code = """
      <script>
        function load() {{
          document.getElementById("{id}").pbtxt = {data};
        }}
      </script>
      <link rel="import"
        href="https://tensorboard.appspot.com/tf-graph-basic.build.html"
        onload=load()>
      <div style="height:600px">
        <tf-graph-basic id="{id}"></tf-graph-basic>
      </div>
  """.format(data=repr(str(strip_def)),
             id=('graph' + str(np.random.rand())))

  iframe = """
      <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}">
      </iframe>
  """.format(code.replace('"', '&quot;'))
  display(HTML(iframe))
