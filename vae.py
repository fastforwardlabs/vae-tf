import datetime
import functools
import itertools
import os
import sys

from functional import compose, partial
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# TODO: prettytensor ?
ARCHITECTURE = [784, # MNIST = 28*28
                #128, # intermediate encoding
                500, #500,
                2] # latent space dims
# (and symmetrically back out again)

def composeAll(*args):
    """Util for multiple function composition"""
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)

def print_(var, name: str, first_n = 10, summarize = 5):
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
    def wbVars(fan_in, fan_out, normal=True):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/pdf/1502.01852v1.pdf
        (distribution defaults to truncated Normal; else Uniform)
        """
        # (int, int, bool) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = (
            #tf.truncated_normal([fan_in, fan_out], stddev=stddev) if normal else
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
            # reuse weights if layer already initialized
            while True:
                try:
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = Layer.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)
            # return self.nonlinearity(tf.matmul(tensor_in, w) + b)


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

    def __init__(self, architecture=ARCHITECTURE, d_hyperparams={},
                 save_graph_def=True, plots_outdir="./png"):
        # YYMMDD_HHMM
        self.datetime = "".join(c for c in str(datetime.datetime.today())
                                if c.isdigit() or c.isspace())[2:13].replace(" ", "_")
        self.plots_outdir = os.path.abspath(plots_outdir)

        self.architecture = architecture
        self.hyperparams = VAE.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        # handles for tensor ops to feed or fetch
        (self.x_in, self.dropout, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = self._buildGraph()

        self.sesh = tf.Session()
        self.sesh.run(tf.initialize_all_variables())

        if save_graph_def:
            self.logger = tf.train.SummaryWriter("./log", self.sesh.graph)
            # logger.flush()
            # logger.close()

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
        # prepend final reconstruction as outermost fn
        # restore original dims and squash vals [0, 1]
        decoding.insert(0, Dense("x_decoding", self.architecture[0], dropout, tf.nn.sigmoid))
        x_reconstructed = tf.identity(composeAll(decoding)(z), name="x_reconstructed")

        # optimization
        # goal: find variational & generative parameters that best reconstruct x
        # i.e. maximize log likelihood over observed datapoints
        # do this by maximizing (variational) lower bound on each marginal log likelihood
        # goal: increase (variationl) lower bound on marginal log likelihood
        # loss
        # reconstruction loss, modeled as Bernoulli (i.e. with binary cross-entropy) / log likelihood
        rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
        rec_loss = print_(rec_loss, "ce")
        # Kullback-Leibler divergence: mismatch b/w approximate vs. true posterior
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)
        kl_loss = print_(kl_loss, "kl")
        cost = tf.reduce_mean(rec_loss + kl_loss, name="cost")
        #cost = tf.add(rec_loss, kl_loss, name="cost")
        #cost = print_(cost, "cost")

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
            # sampling / reparametrization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), mean=0, stddev=
                                       self.hyperparams['epsilon_std'], # TODO: 1. ?
                                       name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)

    @staticmethod
    def crossEntropy(observed, actual, offset = 1e-10):#45):
        with tf.name_scope("binary_cross_entropy"):
            # bound by clipping to avoid NaN
            obs = tf.clip_by_value(observed, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs) +
                                  (1 - actual) * tf.log(1 - obs))#, 0)

            # clip = functools.partial(tf.clip_by_value, clip_value_min=offset,
            #                          clip_value_max=np.inf)
            # return -tf.reduce_sum(actual * tf.log(clip(observed)) +
            #                        (1 - actual) * tf.log(clip(1 - observed)))

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_sum(1 + log_sigma - mu**2 - tf.exp(log_sigma))#, 0)

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
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True, verbose=True):
        try:
            err_train = 0
            #err_cv = 0
            while True:
                x, labels = X.train.next_batch(self.hyperparams["batch_size"])
                feed_dict = {self.x_in: x,
                             self.dropout: self.hyperparams["dropout"]}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if i%500 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i%2000 == 0 and verbose:
                    self.plotSubset(x, x_reconstructed, n=10, name="train")

                    if cross_validate:
                        x, labels = X.validation.next_batch(self.hyperparams["batch_size"])
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

                        #err_cv += cost
                        print("round {} --> CV cost: ".format(i), cost)

                        self.plotSubset(x, x_reconstructed, n=10, name="cv")

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final cost: ", cost)
                    break

        except(KeyboardInterrupt):
            return

    def plotSubset(self, x_in, x_reconstructed, n=10, save=True, name="subset"):
        """Util to plot subset of inputs and reconstructed outputs"""
        plt.figure(figsize = (n * 2, 4))
        plt.title("round {}: {}".format(self.step, name))
        # assume square images
        dim = int(self.architecture[0]**0.5)

        #for idx in range(1, n+1):
        for idx in range(n):
            # display original
            #ax = plt.subplot(2, n, idx)
            ax = plt.subplot(2, n, idx + 1)
            plt.imshow(x_in[idx].reshape([dim, dim]), cmap="Greys")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            #ax = plt.subplot(2, n, idx + n)
            ax = plt.subplot(2, n, idx + n + 1)
            plt.imshow(x_reconstructed[idx].reshape([dim, dim]), cmap="Greys")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if save:
            title = "{}_vae_{}_round_{}_{}.png".format(
                self.datetime, "_".join(map(str, self.architecture)), self.step, name)
            plt.savefig(os.path.join(self.plots_outdir, title))

        plt.show()

    def plotInLatent(self, x_in, labels=np.array([]), save=True, name="data"):
        """Util to plot points in 2-D latent space"""
        assert self.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
        mus, log_sigmas = self.encode(x_in)
        xs, ys = mus.T

        plt.figure()
        plt.title("round {}: {} in latent space".format(self.step, name))
        ax = plt.subplot(111)

        if labels.any():
            classes = set(labels)
            colormap = cm.rainbow(np.linspace(0, 1, len(classes)))
            plt.scatter(xs, ys, alpha=0.8, c=[colormap[i] for i in labels])

            # make room for legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                       for i, class_ in enumerate(classes)]
            ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                      fancybox=True, loc='center left')

        else:
            plt.scatter(xs, ys, alpha=0.8)

        plt.show()
        if save:
            title = "{}_latent_{}_round_{}_{}.png".format(
                self.datetime, "_".join(map(str, self.architecture)), self.step, name)
            plt.savefig(os.path.join(self.plots_outdir, title))

    def exploreLatent(self, nx = 20, ny = 20, save=True):
        """Util to explore low-dimensional manifold of latent space"""
        assert self.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"

        # z_xs = np.linspace(-3, 3, nx)
        # z_ys = np.linspace(-3, 3, ny)

        # inspired by https://jmetzen.github.io/2015-11-27/vae.html
        dim = int(self.architecture[0]**0.5)
        canvas = np.empty([dim * ny, dim * nx])

        # for i, z_xi in enumerate(z_xs):
        #     for j, z_yj in enumerate(z_ys):
        #         x_reconstructed = self.decode([z_xi, z_yj])
        #         canvas[(nx-i-1) * dim : (nx-i) * dim,
        #                j * dim : (j+1) * dim] = x_reconstructed.reshape([dim, dim])
        # complex number steps act to replace np.linspace
        X, Y = np.mgrid[-3:3:nx*1j, -3:3:ny*1j]
        for i, j in itertools.product(range(nx), range(ny)):
            x_reconstructed = self.decode([[X[i,j], Y[i,j]]])
            canvas[(i*dim):((i+1)*dim-1),
                   (j*dim):((j+1)*dim-1)] = x_reconstructed.reshape([dim, dim])
        plt.figure(figsize=(8, 10))
        plt.imshow(canvas)#, origin="upper")
        #plt.tight_layout()

        if save:
            title = "{}_latent_{}_round_{}_explore.png".format(
                self.datetime, "_".join(map(str, self.architecture)), self.step)
            plt.savefig(os.path.join(self.plots_outdir, title))

    def interpolate(self, latent_1, latent_2, n=20):
        """Interpolate between two points in arbitrary-dimensional latent space"""
        # TODO
        interpolations = [np.linspace(start, end, n)
                          for start, end in zip(latent_1, latent_2)]
        zs = np.array([[interp[i] for interp in interpolations] for i in range(n)])
        xs_reconstructed = self.decode(zs)

        plt.figure((20, 4))
        dim = int(self.architecture[0]**0.5)

        for idx in range(1, n+1):
            ax = plt.subplot(2, n, idx)
            plt.imshow(xs_reconstructed[idx].reshape([dim, dim]), cmap="Greys")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def test_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data")

    vae = VAE()
    #vae.train(mnist, max_iter=10000, verbose=True)
    vae.train(mnist, max_iter=4000, verbose=True)

    vae.plotInLatent(mnist.train.images, mnist.train.labels, name="train")
    vae.exploreLatent(nx=20, ny=20)

    mus, sigmas = vae.encode(mnist.test.next_batch(2)[0])
    vae.interpolate(*mus)#np.random.rand(2), np.random.rand(2))

    vae.logger.flush()
    vae.logger.close()

if __name__ == "__main__":
    test_mnist()
