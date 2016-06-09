import datetime
import functools
import os

from functional import compose, partial
import numpy as np
import tensorflow as tf

import plot


# TODO: prettytensor ?
ARCHITECTURE = [784, # MNIST = 28*28
                #128, # intermediate encoding
                500, 500,
                2] # latent space dims
# (and symmetrically back out again)

def composeAll(*args):
    """Util for multiple function composition"""
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)

def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging by printing values during training"""
    # tf.Print is identity fn with side effect of printing requested [vals]
    try:
        return tf.Print(var, [var], '{}: '.format(name), first_n=first_n,
                        summarize=summarize)
    except(TypeError):
        return tf.Print(var, var, '{}: '.format(name), first_n=first_n,
                        summarize=summarize)

class Layer():
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
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        """Fully-connected layer"""
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
                try:
                    # reuse weights if layer already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = Layer.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/pdf/1312.6114v10.pdf)
    """

    DEFAULTS = {
        "batch_size": 128,
        "epsilon_std": 1E-3,#1.
        "learning_rate": 1E-4,
        "dropout": 0.9
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=ARCHITECTURE, d_hyperparams={},
                 save_graph_def=True, plots_outdir="./png", meta_graph=None):
        # YYMMDD_HHMM
        self.datetime = "".join(c for c in str(datetime.datetime.today())
                                if c.isdigit() or c.isspace())[2:13].replace(" ", "_")
        self.plots_outdir = os.path.abspath(plots_outdir)

        self.architecture = architecture
        self.hyperparams = VAE.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        tf.reset_default_graph()
        self.sesh = tf.Session()

        if not meta_graph:
            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)

            # handles for tensor ops to feed or fetch
            (self.x_in, self.dropout, self.z_mean, self.z_log_sigma,
             self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.global_step, self.train_op) = handles

            self.sesh.run(tf.initialize_all_variables())

        else:
            # rebuild graph
            self.datetime = "{}_reloaded".format(os.path.basename(meta_graph)[:11])
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)

            # restore handles for tensor ops to feed or fetch
            (self.x_in, self.dropout, self.z_mean, self.z_log_sigma,
             self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.global_step, self.train_op) = (
                 self.sesh.graph.get_collection(VAE.RESTORE_KEY))

        if save_graph_def:
            self.logger = tf.train.SummaryWriter("./log", self.sesh.graph)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")

        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x)
        # approximation of true posterior p(z|x) -- intractable to calculate
        encoding = [Dense("encoding", hidden_size, dropout, tf.nn.elu)
                    # hidden layers reversed for fn composition s.t. list reads outer -> inner
                    for hidden_size in reversed(self.architecture[1:-1])]
        h_encoded = composeAll(encoding)(x_in)

        # latent distribution Z from which X is generated, parameterized based on hidden encoding
        z_mean = Dense("z_mean", self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", self.architecture[-1], dropout)(h_encoded)

        # let z ~ N(z_mean, np.exp(z_log_sigma)**2)
        # probabilistic decoder - given z, can observe distribution over corresponding x!
        # kingma & welling: only 1 draw per datapoint necessary as long as minibatch is large enough (>100)
        z = self.sampleGaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        # assumes symmetric hidden architecture
        decoding = [Dense("decoding", hidden_size, dropout, tf.nn.elu)
                    for hidden_size in self.architecture[1:-1]]
        # prepend final reconstruction as outermost fn --> restore original dims, squash outputs [0, 1]
        decoding.insert(0, Dense("x_decoding", self.architecture[0], dropout, tf.nn.sigmoid))
        x_reconstructed = tf.identity(composeAll(decoding)(z), name="x_reconstructed")

        # optimization
        # goal: find variational & generative parameters that best reconstruct x
        # i.e. maximize log likelihood over observed datapoints
        # do this by maximizing (variational) lower bound on each marginal log likelihood
        # goal: increase (variational) lower bound on marginal log likelihood
        # loss
        # reconstruction loss, modeled as Bernoulli (i.e. with binary cross-entropy) / log likelihood
        rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
        rec_loss = print_(rec_loss, "ce")
        # Kullback-Leibler divergence: mismatch b/w approximate vs. true posterior
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)
        kl_loss = print_(kl_loss, "kl")
        cost = tf.reduce_mean(rec_loss + kl_loss, name="cost")
        #cost = tf.add(rec_loss, kl_loss, name="cost")
        cost = print_(cost, "cost")

        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.hyperparams["learning_rate"])
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -1, 1), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")
            #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
            #global_norm = tf.global_norm(tvars)
            #train_op = optimizer.apply_gradients(zip(grads, tvars))
            #train_op = (tf.train.AdamOptimizer(self.hyperparams["learning_rate"])
                                #.minimize(cost))

        # ops to directly explore latent space
        z_ = tf.placeholder(tf.float32, shape=[None, self.architecture[-1]], name="latent_in")
        x_reconstructed_ = composeAll(decoding)(z_)

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed, z_,
                x_reconstructed_, cost, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        """Draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # sampling / reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), mean=0,
                                       stddev=self.hyperparams['epsilon_std'],
                                       name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)

    @staticmethod
    def crossEntropy(observed, actual, offset=1e-10):#45):
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs = tf.clip_by_value(observed, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs) +
                                  (1 - actual) * tf.log(1 - obs), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma), 1)
            # return -0.5 * tf.reduce_sum(1 + log_sigma - mu**2 - tf.exp(log_sigma), 1)

    def encode(self, x):
        """Encoder from inputs to latent distribution parameters"""
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, latent_pt):
        """Generative decoder from latent space to reconstructions of input space"""
        # np.array -> np.array
        feed_dict = {self.z_: latent_pt}
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sesh.run(self.sampleGaussian(*self.encode(x))))

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=False, outdir="./out"):
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            #err_cv = 0
            while True:
                x, _ = X.train.next_batch(self.hyperparams["batch_size"])
                feed_dict = {self.x_in: x,
                             self.dropout: self.hyperparams["dropout"]}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if i%500 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i%2000 == 0 and verbose:
                    plot.plotSubset(self, x, x_reconstructed, n=10, name="train")

                    if cross_validate:
                        x, _ = X.validation.next_batch(self.hyperparams["batch_size"])
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

                        #err_cv += cost
                        print("round {} --> CV cost: ".format(i), cost)

                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cv")

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    break

        except(KeyboardInterrupt):
            pass

        finally:
            print("final avg cost (@ step {} = epoch {}): ".format(
                i, X.train.epochs_completed, err_train / i))
            if save:
                outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                    self.datetime, "_".join(map(str, self.architecture))))
                saver.save(self.sesh, outfile, global_step=self.step)
            try:
                self.logger.flush()
                self.logger.close()
            except(AttributeError):
                return
