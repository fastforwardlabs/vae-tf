import tensorflow as tf


class Layer():
    """Neural network base class layer"""
    @staticmethod
    def wbVars(fan_in: int, fan_out: int, normal=True):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/pdf/1502.01852v1.pdf
        (distribution defaults to truncated Normal; else Uniform)
        """
        # (int, int, bool) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = (
            # tf.truncated_normal([fan_in, fan_out], stddev=stddev) if normal else
            tf.random_normal([fan_in, fan_out], stddev=stddev) if normal else
            tf.random_uniform([fan_in, fan_out], -stddev, stddev) # (range therefore not truly stddev)
        )
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))


class Dense(Layer):
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, float or tf.Variable, tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying - i.e. to appy specified layer to any input tensor `x`"""
        # tf.Tensor -> tf.op
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if layer already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = Layer.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)
