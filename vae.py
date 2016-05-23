from __future__ import division
from collections import namedtuple
from functools import partial

import numpy as np
import tensorflow as tf


Layers = namedtuple('Layers', ('input_size', 'hidden_size', 'latent_size'))
# input -> intermediate -> latent
# (and symmetrically back out again)
LAYERS = Layers(100, 50, 20)

class Model():

    DEFAULTS = {
        'batch_size': 100,
        'epsilon_std': 0.1,
        'learning_rate': 0.05
    }

    def __init__(self, layers=LAYERS, d_hyperparams={}):

        self.layers = layers

        self.hyperparams = Model.DEFAULTS.copy()
        self.hyperparams.update(**d_hyperparams)

        (self.x_in, self.latent_in, self.latent_assign, self.x_decoded_mean,
         self.cost, self.train_op) = self._buildGraph()

    def _buildGraph():

        def wbVars(nodes_in, nodes_out):
            """Helper to initialize weights and biases"""
            initial_w = tf.truncated_normal([nodes_in, nodes_out],
                                            #stddev = (2/nodes_in)**0.5)
                                            stddev = nodes_in**-0.5,
                                            name='truncated_normal')
            initial_b = tf.random_normal([nodes_out], name='random_normal')

            return (tf.Variable(initial_w, trainable=True, name='weights'),
                    tf.Variable(initial_b, trainable=True, name='biases'))

        def denseLayer(tensor_in, size, scope='dense', nonlinearity=None):
            """Densely connected layer"""
            _, m = tensor_in.get_shape()
            with tf.name_scope(scope):
                w, b = wbVars(m, size)
                return nonlinearity(tf.matmul(tensor_in, w) + b)

        def dense(size, scope, nonlinearity):
            """Currying to appy given dense layer to any input tensor"""
            return functools.partial(denseLayer(size=size, scope=scope,
                                                nonlinearity=nonlinearity))

        def sample(mu, sigma, scope='sampling'):
            n, m = mu.get_shape()
            with tf.name_scope(scope):
                epsilon = tf.random_normal([n, m], mean=0, stddev=self.epsilon_std)
                return mu + epsilon * tf.exp(z_log_sigma)

        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.layers[0]], name='x')
        # encoding
        h = dense(self.layers[1], 'encoding', tf.nn.relu)(x_in)

        # latent space
        z_mean = dense(self.layers[2], 'z_mean')(h)
        z_log_sigma = dense(self.layers.latent, 'z_log_sigma')(h)
        z = sample(z_mean, z_log_sigma, 'latent_sampling')

        # nodes to take points on latent manifold and generate reconstructed outputs
        latent_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                      self.latent_dims], name='latent_in')
        latent_assign = tf.assign(z, latent_in)

        # decoding
        h_decoded = dense(self.hyperparamsintermediate_dims, 'h_decoder', tf.nn.relu)(z)
        x_decoded_mean = dense(self.input_dims, 'x_decoder', tf.nn.sigmoid)(h_decoded)

        # training ops
        with tf.name_scope('cost'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(x_decoded_mean)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), reduction_indices=1, name='KL_divergence')
            cost = cross_entropy + kl_loss

        train_op = tf.train.AdamOptimizer(self.hyperparams['learning_rate'])\
                                          .minimize(cost)

        return (x_in, latent_in, latent_assign, x_decoded_mean, cost, train_op)

    def encoder(x):
        # encoder from inputs to latent space
        with tf.Session() as sesh:
            out = tf.run(self.z_mean, {self.x_in: x})
        return out

    def decoder(latent_pt):
        # generator from latent space to reconstructed inputs
        with tf.Session() as sesh:
            _, generator = tf.run([self.latent_assign, self.x_decoded_mean],
                                  {self.latent_in: latent_pt})
        return generator

    def vae(x, train=False):
        # end-to-end autoencoder
        to_compute = ([self.x_decoded_mean, self.cost, self.train_op] if train
                      else [self.x_decoded_mean])
        with tf.Session() as sesh:
            out = tf.run(to_compute, {self.x_in: x})
        return out

    def train(x, verbose=True):
        i = 0
        while True:
            x_decoded, cost, _ = self.vae(x, train=True)
            i += 1

            if i%10 == 0 and verbose:
                print "x_in: ", x
                print "x_decoded: ", x_decoded

            if i%100 == 0 and verbose:
                print "cost: ", cost
