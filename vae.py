import numpy as np
import tensorflow as tf


ARCHITECTURE = [784, # MNIST = 28*28
                128,
                #500, 500, # intermediate encoding
                2] # latent space dims
# (and symmetrically back out again)


#@ops.RegisterGradient("PlaceholderWithDefault")
#def _PlaceholderWithDefault(op, grad):
    #default = op.inputs[0]
    #tf.gradients()

class VAE():
    """Variational Autoencoder"""

    DEFAULTS = {
        "batch_size": 100,
        "epsilon_std": 0.0001,
        "learning_rate": 0.01
    }

    def __init__(self, architecture=ARCHITECTURE, d_hyperparams={},
                 save_graph_def=True):

        self.architecture = architecture

        self.hyperparams = VAE.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        (self.x_in, self.latent_in, self.assign_op, self.x_decoded_mean,
         self.cost, self.train_op) = self._buildGraph()
        if save_graph_def:
            logger = tf.train.SummaryWriter('.', self.sesh.graph)
            logger.flush()
            logger.close()

    def _buildGraph(self):

        def print_(var, name, first_n = 3, summarize = 5):
            """Util for debugging by printing values during training"""
            # tf.Print is identity fn with side effect of printing requested [vals]
            try:
                return tf.Print(var, [var], '{}: '.format(name), first_n=first_n,
                                summarize=summarize)
            except(TypeError):
                return tf.Print(var, var, '{}: '.format(name), first_n=first_n,
                                summarize=summarize)

        def wbVars(fan_in: int, fan_out: int, normal=True):
            """Helper to initialize weights and biases, via He's adaptation
            of Xavier init for ReLUs: https://arxiv.org/pdf/1502.01852v1.pdf
            (distribution defaults to truncated Normal; else Uniform)
            """
            stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

            initial_w = (
                tf.truncated_normal([fan_in, fan_out], stddev=stddev) if normal#,
                                    #name="xavier_truncated_normal") if normal
                else tf.random_uniform([fan_in, fan_out], -stddev, stddev))#, # (range therefore not truly stddev)
                                       #name="xavier_uniform"))
            initial_b = tf.zeros([fan_out])#, name="zeros")

            return (tf.Variable(initial_w, trainable=True, name="weights"),
                    tf.Variable(initial_b, trainable=True, name="biases"))

        def dense(scope="dense_layer", size=None, nonlinearity=tf.identity):
            """Dense layer currying - i.e. to appy specified layer to any input tensor"""
            assert size, "Must specify layer size (num nodes)"
            def _dense(tensor_in):
                with tf.name_scope(scope):
                    w, b = wbVars(tensor_in.get_shape()[1].value, size)
                    return nonlinearity(tf.matmul(tensor_in, w) + b)
            return _dense

        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")
        # encoding
        xs = [x_in]
        for hidden_size in self.architecture[1:-1]:
            h = dense("encoding", hidden_size, tf.nn.relu)(xs[-1])
            xs.append(h)
        #h_encoded = tf.identity(xs[-1], name="h_encoded")
        h_encoded = xs[-1]

        # latent space based on encoded output
        z_mean = dense(self.architecture[-1], "z_mean")(h_encoded)
        z_log_sigma = dense(self.architecture[-1], "z_log_sigma")(h_encoded)
        z = VAE.sampleGaussian(z_mean, z_log_sigma, "latent_sampling")

        # nodes to take points on latent manifold and generate reconstructed outputs
        latent_in = tf.placeholder(tf.float32, name="latent_in",
                                   shape=[None, self.architecture[-1]])
        assign_op = tf.assign(z, latent_in)

        # decoding
        hs = [z]
        # iterate backwards through symmetric hidden architecture
        for hidden_size in self.architecture[1:-1:-1]:
            h = dense(hidden_size, "decoding", tf.nn.relu)(hs[-1])
            hs.append(h)
        #h_decoded = dense(self.architecture.hidden_size, "h_decoder", tf.nn.relu)(z)
        h_decoded = tf.identity(hs[-1], name="h_decoded")
        x_decoded_mean = dense(self.architecture[0], "x_decoded", tf.sigmoid)(h_decoded)

        # training ops
        with tf.name_scope("cost"):
            #     #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(x_decoded_mean, x_in)
            # with tf.name_scope("kullback_leibler_divergence"):
            #     kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean)
            #                                     - tf.exp(z_log_sigma), reduction_indices=1)
            # cost = cross_entropy + kl_loss
            cross_entropy = VAE.crossEntropy(x_decoded_mean, x_in)
            kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)
            cost = cross_entropy + kl_loss

        train_op = (tf.train.AdamOptimizer(self.hyperparams["learning_rate"])
                            .minimize(cost))

        return (x_in, latent_in, latent_assign, x_decoded_mean, cost, train_op)

    @staticmethod
    def sampleGaussian(mu, log_sigma, scope="sampling"):
        """Draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            epsilon = tf.random_normal(tf.shape(mu), mean=0, stddev=
                                       self.hyperparams['epsilon_std'],
                                       name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)

    @staticmethod
    def crossEntropy(observed, actual):
        with tf.name_scope("cross_entropy"):
            # bound obs by clipping to avoid NaN
            return -tf.reduce_mean(actual * tf.log(tf.clip_by_value(
                observed, 1e-12, 1.0)), reduction_indices=1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        with tf.name_scope("kullback_leibler_divergence"):
            return -0.5 * tf.reduce_mean(1 + log_sigma - tf.square(mu)
                                         - tf.exp(log_sigma), reduction_indices=1)

    def encoder(self, x):
        """Encoder from inputs to latent space"""
        with tf.Session() as sesh:
            out = tf.run(self.z_mean, {self.x_in: x})
        return out

    def decoder(self, latent_pt):
        """Generator from latent space to reconstructed inputs"""
        with tf.Session() as sesh:
            _, generator = tf.run([self.assign_op, self.x_decoded_mean],
                                  {self.latent_in: latent_pt})
        return generator

    def vae(self, x, train=False):
        """End-to-end autoencoder"""
        to_compute = ([self.x_decoded_mean, self.cost, self.train_op] if train
                      else [self.x_decoded_mean])
        with tf.Session() as sesh:
            out = tf.run(to_compute, {self.x_in: x})
        return out

    def train(self, x, verbose=True):
        i = 0
        while True:
            x_decoded, cost, _ = self.vae(x, train=True)
            i += 1

            if i%10 == 0 and verbose:
                print "x_in: ", x
                print "x_decoded: ", x_decoded

            if i%100 == 0 and verbose:
                print "cost: ", cost
