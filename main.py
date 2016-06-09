import numpy as np

import plot
import vae


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data")

def all_plots(model, mnist=None):
    if not mnist:
        mnist = load_mnist()
    print("Plotting in latent space...")
    plot_all_in_latent(model, mnist)
    print("Exploring latent...")
    plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4))
    print("Interpolating...")
    interpolate_digits(model, mnist)

def plot_all_in_latent(model, mnist=None):
    if not mnist:
        mnist = load_mnist()
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        plot.plotInLatent(model, dataset.images, dataset.labels, name=name)

def interpolate_digits(model, mnist=None):
    if not mnist:
        mnist = load_mnist()
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
    plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
        *(labels[i] for i in idxs)))

def test_mnist():
    mnist = load_mnist()
    v = vae.VAE()
    v.train(mnist, max_epochs=100, verbose=False, save=True)
    print("Trained!")
    all_plots(v, mnist)

def reload(meta_graph="./out/160608_1414_vae_784_500_500_2-20164"):
    v = vae.VAE(meta_graph=meta_graph)
    print("Loaded!")
    return v
    #plot.randomWalk(v)
    #all_plots(v)


if __name__ == "__main__":
    test_mnist()
    #reload()
