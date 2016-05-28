import matplotlib.pyplot as plt
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
        "epsilon_std": 0.001,
        "learning_rate": 0.002
    }

    def __init__(self, architecture=ARCHITECTURE, d_hyperparams={},
                 save_graph_def=True):

        self.architecture = architecture

        self.hyperparams = VAE.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        (self.x_in, self.z_mean, self.latent_in, self.z_assign,
         self.x_decoded_mean, self.cost, self.train_op) = self._buildGraph()

        self.sesh = tf.Session()
        self.sesh.run(tf.initialize_all_variables())

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
            h = print_(h, "h")
            xs.append(h)
        #h_encoded = tf.identity(xs[-1], name="h_encoded")
        h_encoded = xs[-1]

        # latent space based on encoded output
        z_mean = dense("z_mean", self.architecture[-1])(h_encoded)
        z_log_sigma = dense("z_log_sigma", self.architecture[-1])(h_encoded)
        z = self.sampleGaussian(z_mean, z_log_sigma)
        z = print_(z, "z")

        #z = tf.placeholder_with_default(self.sampleGaussian(z_mean, z_log_sigma),
                                        #[None, self.architecture[-1]])
        #z = tf.Variable(tf.zeros([47, self.architecture[-1]]),
                        #trainable=False) # dummy init to make z assignable
        ## latent sampling
        #with tf.name_scope("latent_sampling"):
            #z = tf.assign(z, self.sampleGaussian(z_mean, z_log_sigma), validate_shape=False)
        ## direct exploration of latent manifold to generate reconstructed outputs
        #with tf.name_scope("latent_feed"):
            #z_in = tf.placeholder(tf.float32, name="z_in",
                                    #shape=[None, self.architecture[-1]])
            #z_ = tf.assign(z, latent_in)

        # decoding
        hs = [z]
        # iterate backwards through symmetric hidden architecture
        for hidden_size in reversed(self.architecture[1:-1]): # aka [-2:0:-1]
            h = dense("decoding", hidden_size, tf.nn.relu)(hs[-1])
            hs.append(h)
        h_decoded = hs[-1]
        #h_decoded = tf.identity(hs[-1], name="h_decoded")

        # reconstructed output
        x_decoded_mean = tf.identity(
            dense("x_decoding", self.architecture[0], tf.sigmoid)(h_decoded),
            name = "x_reconstructed")

        # loss
        cross_entropy = VAE.crossEntropy(x_decoded_mean, x_in) # reconstruction loss
        cross_entropy = print_(cross_entropy, "ce")
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma) # mismatch b/w learned latent dist and prior
        kl_loss = print_(kl_loss, "kl")
        cost = tf.add(cross_entropy, kl_loss, name="cost")

        # optimization
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.hyperparams["learning_rate"])
            tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
            #train_op = optimizer.apply_gradients(zip(grads, tvars))
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            #global_norm = tf.global_norm(tvars)
            clipped = [(tf.clip_by_value(grad, -1, 1), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, name="minimize_cost")
            #train_op = (tf.train.AdamOptimizer(self.hyperparams["learning_rate"])
                                #.minimize(cost))

        latent_in, z_assign = 47, 47
        return (x_in, z_mean, latent_in, z_assign, x_decoded_mean, cost, train_op)

    def sampleGaussian(self, mu, log_sigma):
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
                observed, 1e-10, 1.0)))

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_mean(1 + log_sigma - mu**2 - tf.exp(log_sigma))

    def encode(self, x):
        """Encoder from inputs to latent space"""
        encoded = self.sesh.run(self.z_mean, feed_dict={self.x_in: x})
        return encoded

    # def decode(self, latent_pt):
    #     """Generator from latent space to reconstructed inputs"""
    #     #with tf.Session() as sesh:
    #         #sesh.run(tf.initialize_all_variables())
    #         #_, generator = sesh.run([self.assign_op, self.x_decoded_mean],
    #                                 #feed_dict={self.latent_in: latent_pt})
    #     #return generator
    #     _, generator = self.sesh.run([self.z_assign, self.x_decoded_mean],
    #                                  feed_dict={self.latent_in: latent_pt})
    #     return generator

    def vae(self, x, train=False):
        """End-to-end autoencoder"""
        fetches = ([self.x_decoded_mean, self.cost, self.train_op] if train
                   else [self.x_decoded_mean])
        out = self.sesh.run(fetches, feed_dict={self.x_in: x})
        return out

    def train(self, X, verbose=True):
        i = 0
        while True:
            try:
                x, labels = X.train.next_batch(self.hyperparams['batch_size'])
                x_reconstructed, cost, _ = self.vae(x, train=True)
                i += 1

                # if i%100 == 0 and verbose:
                #     print("x_in: ", x)
                #     print("x_reconstructed: ", x_reconstructed)

                if i%100 == 0 and verbose:
                    print("cost: ", cost)

                if i%1000 == 0 and verbose:
                    across = int(np.sqrt(self.hyperparams["batch_size"]))
                    down = across
                    #down = int(self.hyperparams["batch_size"] / across) + self.hyperparams["batch_size"] % across
                    dims = [int(inputs * self.architecture[0]**0.5) for inputs in (across, down)]
                    plt.subplot(211)
                    plt.imshow(x.reshape(dims), cmap='Greys')
                    plt.subplot(212)
                    plt.imshow(x_reconstructed.reshape(dims), cmap='Greys')
                    plt.savefig('blkwht_{}.png'.format(i))
                    from IPython import embed; embed()
                    plt.show()

            except(KeyboardInterrupt):
                plt.show()


def test_mnist():
    import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    #imgs, labels = mnist.train.next_batch(100)

    vae = VAE()
    vae.train(mnist)

if __name__ == "__main__":
    test_mnist()
