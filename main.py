import os

import numpy as np
import tensorflow as tf

import plot
from utils import get_mnist
import vae


IMG_DIM = 28

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                500, 500, # intermediate encoding
                2] # latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,#1E-3,
    "dropout": 0.9,#0.8,
    "lambda_l2_reg": 1E-5,#1E-4,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid,
}

MAX_ITER = 2**16#20000#1E5#20000
MAX_EPOCHS = np.inf#100

LOG_DIR = "./log/mnist" # "./log"
METAGRAPH_DIR = "./out/mnist" # "./out"
PLOTS_DIR = "./png/mnist" # "./plots"


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./data/MNIST_data") # "./mnist_data"

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)
        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)
        for n in (24, 30, 60, 100):
            plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=PLOTS_DIR,
                               name="explore_ppf{}".format(n))

    print("Interpolating...")
    interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, mnist)

    print("Morphing...")
    morph_numbers(model, mnist)

    # print("Plotting 10 MNIST digits...")
    # for i in range(10):
    #     plot.justMNIST(*mnist.train.next_batch(1), outdir=PLOTS_DIR)
    #     plot.justMNIST(get_mnist(i), name=str(i), outdir=PLOTS_DIR)

def plot_all_in_latent(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        plot.plotInLatent(model, dataset.images, dataset.labels, name=name,
                          outdir=PLOTS_DIR)

def interpolate_digits(model, mnist):
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
    plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
        *(labels[i] for i in idxs)), outdir=PLOTS_DIR)

def plot_all_end_to_end(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        x, _ = dataset.next_batch(10)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name,
                        outdir=PLOTS_DIR)

def morph_numbers(model, mnist, ns=None):
    if not ns:
        import random
        ns = random.sample(range(10), 10) # non-in-place shuffle

    xs = np.squeeze([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    plot.morph(model, mus, n_per_morph=10, outdir=PLOTS_DIR,
               name="morph_{}".format("".join(str(n) for n in ns)))

def test_mnist(to_reload=None):
    mnist = load_mnist()

    if to_reload:
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else:
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(mnist, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
                verbose=False,#True,
                save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=True)
        print("Trained!")

    #all_plots(v, mnist)
    # morph_numbers(v, mnist, [4,7,3,0,8,1,6,9,5,2])


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    # test_mnist()
    # test_mnist(to_reload="./out/mnist/160816_1754_vae_784_500_500_50-65536")
    test_mnist(to_reload="./out/mnist/160816_1813_vae_784_500_500_2-65536")
