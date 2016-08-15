import functools
import os

from functional import compose, partial
import tensorflow as tf


def composeAll(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)

def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging by printing values during training"""
    # tf.Print is identity fn with side-effect of printing requested [vals]
    try:
        return tf.Print(var, [var], '{}: '.format(name), first_n=first_n,
                        summarize=summarize)
    except(TypeError):
        return tf.Print(var, var, '{}: '.format(name), first_n=first_n,
                        summarize=summarize)
