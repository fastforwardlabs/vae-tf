import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import moviepy.editor as movie
import numpy as np


def plotSubset(model, x_in, x_reconstructed, n=10, cols=None, outlines=True,
               save=True, name="subset", outdir="."):
    """Util to plot subset of inputs and reconstructed outputs"""
    n = min(n, x_in.shape[0])
    cols = (cols if cols else n)
    rows = 2 * int(np.ceil(n / cols)) # doubled b/c input & reconstruction

    plt.figure(figsize = (cols * 2, rows * 2))
    # plt.title("round {}: {}".format(model.step, name))
    dim = int(model.architecture[0]**0.5) # assume square images

    for i, x in enumerate(x_in[:n], 1):
        # display original
        ax = plt.subplot(rows, cols, i) # rows, cols, subplot numbered from 1
        plt.imshow(x.reshape([dim, dim]), cmap="Greys")
        if outlines:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_axis_off()

    for i, x in enumerate(x_reconstructed[:n], 1):
        # display reconstruction
        ax = plt.subplot(rows, cols, i + cols * (rows / 2))
        plt.imshow(x.reshape([dim, dim]), cmap="Greys")
        if outlines:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_axis_off()

    plt.show()
    if save:
        title = "{}_batch_{}_round_{}_{}.png".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def plotInLatent(model, x_in, labels=[], save=True, name="data", outdir="."):
    """Util to plot points in 2-D latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    mus, _ = model.encode(x_in)
    ys, xs = mus.T

    plt.figure()
    plt.title("round {}: {} in latent space".format(model.step, name))
    kwargs = {'alpha': 0.8}

    classes = set(labels)
    if classes:
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs['c'] = [colormap[i] for i in labels]

        # make room for legend
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                    for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                    fancybox=True, loc='center left')

    plt.scatter(xs, ys, **kwargs)

    plt.show()
    if save:
        title = "{}_latent_{}_round_{}_{}".format(
            model.datetime, "_".join(map(str, model.architecture)),
            model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def exploreLatent(model, nx=20, ny=20, range_=(-4, 4), save=True, name="explore",
                  outdir="."):
    """Util to explore low-dimensional manifold of latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    dim = int(model.architecture[0]**0.5)
    min_, max_ = range_

    # complex number steps act like np.linspace
    # row, col indices (i, j) correspond to graph coords (y, x)
    # rollaxis enables iteration over latent space 2-tuples
    zs = np.rollaxis(np.mgrid[max_:min_:ny*1j, min_:max_:nx*1j], 0, 3)
    canvas = np.vstack([np.hstack([x.reshape([dim, dim]) for x in
                                    model.decode(z_row)]) for z_row in iter(zs)])

    plt.figure(figsize=(10, 10))
    # `extent` sets axis labels corresponding to latent space coords
    plt.imshow(canvas, cmap="Greys", aspect="auto", extent=(range_ * 2))
    plt.tight_layout()

    plt.show()
    if save:
        title = "{}_latent_{}_round_{}_{}.png".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def interpolate(model, latent_1, latent_2, n=20, save=True, name="interpolate", outdir="."):
    """Util to interpolate between two points in n-dimensional latent space"""
    zs = np.array([np.linspace(start, end, n) # interpolate across every z dimension
                    for start, end in zip(latent_1, latent_2)]).T
    xs_reconstructed = model.decode(zs)

    dim = int(model.architecture[0]**0.5)
    canvas = np.hstack([x.reshape([dim, dim]) for x in xs_reconstructed])

    plt.figure(figsize = (n, 2))
    plt.imshow(canvas, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()

    plt.show()
    if save:
        title = "{}_latent_{}_round_{}_{}".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def latent_arithmetic(model, a, b, c, save=True, name="arithmetic", outdir="."):
    """Util to implement vector math in latent space equivalent to (a - b + c)"""
    inputs = (a, b, c)
    a_, b_, c_ = (model.sampleGaussian(*model.encode(vec)) for vec in inputs)
    d = model.decode(a_ - b_ + c_)

    plt.figure(figsize = (5, 4))
    plt.title("a + b - c = ...")
    # assume square images
    dim = int(model.architecture[0]**0.5)

    for i, img in enumerate(inputs):
        # inputs to latent vector arithmetic
        ax = plt.subplot(2, 3, i + 1) # rows, cols, subplot numbered from 1
        plt.imshow(img.reshape([dim, dim]), cmap="Greys")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # display output
    ax = plt.subplot(2, 3, 5)
    plt.imshow(d.reshape([dim, dim]), cmap="Greys")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()
    if save:
        title = "{}_latent_{}_round_{}_{}.png".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def randomWalk(model, starting_pt=np.array([]), step_size=20, steps_till_turn=10,
               save=True, outdir="."):
    # TODO: random walk gif in latent space!
    dim = int(model.architecture[0]**0.5)

    def iterWalk(start):
        """Yield points on random walk"""
        def step():
            """Equally sized step in random direction"""
            # random normal in each dimension
            direction = np.random.randn(starting_pt.size)
            return step_size * (direction / np.linalg.norm(direction))

        here = start
        yield here
        while True:
            next_step = step()
            for i in range(steps_till_turn):
                here += next_step
                yield here

    if not starting_pt.any():
        # if not specified, pick randomly from latent space
        starting_pt = 4 * np.random.randn(model.architecture[-1])
    walk = iterWalk(starting_pt)

    def to_rgb(im):
        # c/o http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
        return np.dstack([im.astype(np.uint8)] * 3)

    def make_frame(t):
        z = next(walk)
        x_reconstructed = model.decode([z]).reshape([dim, dim])
        return to_rgb(x_reconstructed)
    # TODO: recursive ?

    clip = movie.VideoClip(make_frame, duration=20)

    if save:
        title = "{}_random_walk_{}_round_{}.mp4".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step)
        clip.write_videofile(os.path.join(outdir, title), fps=10)



def freeAssociate(model, starting_pt=np.array([]), step_size=2, steps_till_turn=10,
                  save=True, outdir="."):
    # TODO: random walk gif in latent space!
    dim = int(model.architecture[0]**0.5)

    def iterWalk(start):
        """Yield points on random walk"""
        def step():
            """Equally sized step in random direction"""
            # random normal in each dimension
            direction = np.random.randn(starting_pt.size)
            return step_size * (direction / np.linalg.norm(direction))

        here = start
        yield here
        while True:
            next_step = step()
            for i in range(steps_till_turn):
                here += next_step
                yield here

    if not starting_pt.any():
        # if not specified, sample randomly from latent space
        starting_pt = model.sesh.run(model.z_)

    walk = iterWalk(starting_pt)
    for i in range(100):
        z = next(walk)
        x_reconstructed = model.decode(z).reshape([dim, dim])
        plt.figure(figsize = (2, 2))
        plt.imshow(x_reconstructed, cmap="Greys")
        plt.axis("off")
        plt.tight_layout()

        plt.show()
        if save:
            title = "{}_random_walk_{}_round_{}.{}.png".format(
                model.datetime, "_".join(map(str, model.architecture)), model.step, i)
            plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
