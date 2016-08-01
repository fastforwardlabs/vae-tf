import os

import numpy as np
import tensorflow as tf

import plot
import vae


IMG_DIM = 28

ARCHITECTURE = [IMG_DIM**2, # 784 pixels
                500, 500, # intermediate encoding
                # 1024, 1024,
                #2] # latent space dims
                50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 1E-3,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    # "nonlinearity": tf.nn.tanh,
    "squashing": tf.nn.sigmoid,
}

MAX_ITER = 20000#1E5#20000
MAX_EPOCHS = np.inf#100

LOG_DIR = "./log/mnist"
METAGRAPH_DIR = "./out/mnist"
PLOTS_DIR = "./png/mnist"


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./data/MNIST_data")

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)
        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)

    print("Interpolating...")
    interpolate_digits(model, mnist)

    # print("Latent vector arithmetic...")
    # ORIG, TARGET = "A", "X"
    # from_font, to_font = fonts.train.random(2)
    # chars = (to_font[CHAR2ORD[ORIG]], from_font[CHAR2ORD[ORIG]], from_font[CHAR2ORD[TARGET]])
    # # chars[0] - chars[1] + chars[2]
    # plot.latent_arithmetic(model, *[np.expand_dims(c, 0) for c in chars], name=
    #                        "{}-{}+{}".format(ORIG, ORIG, TARGET), outdir=PLOTS_DIR)

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
        #feed_dict = {model.x_in: x}
        #fetches = model.x_reconstructed
        #x_reconstructed = model.sesh.run(fetches, feed_dict)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name, outdir=PLOTS_DIR)

def test_mnist(to_reload=None):
    mnist = load_mnist()

    if to_reload:
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else:
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(mnist, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
                verbose=True, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR)
        print("Trained!")

    plot.freeAssociate(v, outdir=PLOTS_DIR)
    # plot_all_end_to_end(v, mnist)
    #all_plots(v, mnist)
    #plot.randomWalk(v)


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    # test_mnist()
    test_mnist(to_reload="./out/mnist/160801_1234_vae_784_500_500_50-20000")
