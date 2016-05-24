from __future__ import division
from collections import namedtuple
from functools import partial

import numpy as np
import tensorflow as tf


# Layers = namedtuple("Layers", ("input_size", "hidden_size", "latent_size"))
#ARCHITECTURE = Layers(100, 50, 20)
# input -> intermediate/s -> latent
# (and symmetrically back out again)
ARCHITECTURE = [100, 50, 20]

class VAE():
    """Variational Autoencoder"""

    DEFAULTS = {
        "batch_size": 100,
        "epsilon_std": 0.1,
        "learning_rate": 0.05
    }

    def __init__(self, architecture=ARCHITECTURE, d_hyperparams={}):

        self.architecture = architecture

        self.hyperparams = VAE.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        (self.x_in, self.latent_in, self.assign_op, self.x_decoded_mean,
         self.cost, self.train_op) = self._buildGraph()

    def _buildGraph(self):

        def wbVars(nodes_in, nodes_out):
            """Helper to initialize weights and biases"""
            initial_w = tf.truncated_normal([nodes_in, nodes_out],
                                            #stddev = (2/nodes_in)**0.5)
                                            stddev = nodes_in**-0.5,
                                            name="truncated_normal")
            initial_b = tf.random_normal([nodes_out], name="random_normal")

            return (tf.Variable(initial_w, trainable=True, name="weights"),
                    tf.Variable(initial_b, trainable=True, name="biases"))

        def denseLayer(tensor_in, size, scope="dense", nonlinearity=tf.identity):
            """Densely connected layer, with given nonlinearity (default: none)"""
            _, m = tensor_in.get_shape()
            with tf.name_scope(scope):
                w, b = wbVars(m, size)
                return nonlinearity(tf.matmul(tensor_in, w) + b)

        def dense(size, scope, nonlinearity=tf.identity):
            """Dense layer currying, e.g. to appy specified layer to any input tensor"""
            return functools.partial(denseLayer(size=size, scope=scope,
                                                nonlinearity=nonlinearity))

        x_in = tf.placeholder(tf.float32, name="x",
                              # None dim enables variable batch size
                              shape=[None, self.architecture[0]])
        xs = [x_in]

        # encoding
        for hidden_size in self.architecture[1:-1]:
            h = dense(hidden_size, "encoding", tf.nn.relu)(xs[-1])
            xs.append(h)
        h_encoded = tf.identity(xs[-1], name="h_encoded")

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
        n, m = mu.get_shape()
        with tf.name_scope(scope):
            epsilon = tf.random_normal([n, m], mean=0, stddev=
                                       self.hyperparams["epsilon_std"])
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
