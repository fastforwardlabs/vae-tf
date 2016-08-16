import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plotSubset(model, x_in, x_reconstructed, n=10, cols=None, outlines=True,
               save=True, name="subset", outdir="."):
    """Util to plot subset of inputs and reconstructed outputs"""
    n = min(n, x_in.shape[0])
    cols = (cols if cols else n)
    rows = 2 * int(np.ceil(n / cols)) # doubled b/c input & reconstruction

    plt.figure(figsize = (cols * 2, rows * 2))
    dim = int(model.architecture[0]**0.5) # assume square images

    def drawSubplot(x_, ax_):
        plt.imshow(x_.reshape([dim, dim]), cmap="Greys")
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for i, x in enumerate(x_in[:n], 1):
        # display original
        ax = plt.subplot(rows, cols, i) # rows, cols, subplot numbered from 1
        drawSubplot(x, ax)

    for i, x in enumerate(x_reconstructed[:n], 1):
        # display reconstruction
        ax = plt.subplot(rows, cols, i + cols * (rows / 2))
        drawSubplot(x, ax)

    plt.show()
    if save:
        title = "{}_batch_{}_round_{}_{}.png".format(
            model.datetime, "_".join(map(str, model.architecture)), model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def plotInLatent(model, x_in, labels=[], range_=None, save=True, title=None,
                 name="data", outdir="."):
    """Util to plot points in 2-D latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    title = (title if title else name)
    mus, _ = model.encode(x_in)
    ys, xs = mus.T

    plt.figure()
    plt.title("round {}: {} in latent space".format(model.step, title))
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

    if range_:
        plt.xlim(*range_)
        plt.ylim(*range_)

    plt.show()
    if save:
        title = "{}_latent_{}_round_{}_{}.png".format(
            model.datetime, "_".join(map(str, model.architecture)),
            model.step, name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def exploreLatent(model, nx=20, ny=20, range_=(-4, 4), ppf=False,
                  save=True, name="explore", outdir="."):
    """Util to explore low-dimensional manifold of latent space"""
    assert model.architecture[-1] == 2, "2-D plotting only works for latent space in R2!"
    # linear range; else ppf (percent point function) == inverse CDF [0, 1]
    range_ = ((0, 1) if ppf else range_)
    min_, max_ = range_
    dim = int(model.architecture[0]**0.5)

    # complex number steps act like np.linspace
    # row, col indices (i, j) correspond to graph coords (y, x)
    # rollaxis enables iteration over latent space 2-tuples
    zs = np.rollaxis(np.mgrid[max_:min_:ny*1j, min_:max_:nx*1j], 0, 3)

    if ppf: # sample from prior ~ N(0, 1)
        from scipy.stats import norm
        DELTA = 1E-16 # delta to avoid +/- inf at 0, 1 boundaries
        # ppf == percent point function == inverse cdf
        zs = np.array([norm.ppf(np.clip(z, DELTA, 1 - DELTA)) for z in zs])

    canvas = np.vstack([np.hstack([x.reshape([dim, dim]) for x in
                                    model.decode(z_row)])
                        for z_row in iter(zs)])

    plt.figure(figsize=(nx / 2, ny / 2))
    # `extent` sets axis labels corresponding to latent space coords
    plt.imshow(canvas, cmap="Greys", aspect="auto", extent=(range_ * 2))
    if ppf: # no axes
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")
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


def justMNIST(x, name="digit", outdir="."):
    """Plot individual pixel-wise MNIST digit x"""
    DIM = 28
    TICK_SPACING = 4

    fig, ax = plt.subplots(1,1)
    plt.imshow(x.reshape([DIM, DIM]), cmap="Greys",
               extent=((0, DIM) * 2), interpolation="none")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))

    plt.show()
    if save:
        title = "mnist_{}.png".format(name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def morph(model, zs=[None, None], n_per_morph=10, sinusoid=False, name="morph",
          save=True, outdir="."):
    '''
    returns a list of img_data to represent morph between z1 and z2
    default to linear morph, but can try sinusoid for more time near the anchor pts
    n_total_frame must be >= 2, since by definition there's one frame for z1 and z2

    list of zs (or random from prior if None)
    '''
    assert len(zs) > 1, "Must specify at least two latent pts for morph!"
    dim = int(model.architecture[0]**0.5) # assume square images

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        # via https://docs.python.org/dev/library/itertools.html
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    all_xs = []
    for z1, z2 in pairwise(zs):
        zs_morph = np.array([np.linspace(start, end, n_per_morph)
                             # interpolate across every z dimension
                             for start, end in zip(z1, z2)]).T
        xs_reconstructed = model.decode(zs_morph)
        all_xs.extend(xs_reconstructed)

    for i, x in enumerate(all_xs):
        plt.figure(figsize = (5, 5))
        plt.imshow(x.reshape([dim, dim]), cmap="Greys")

        # axes off
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")

        plt.show()
        if save:
            title = "{}_latent_{}_round_{}_{}.{}.png".format(
                model.datetime, "_".join(map(str, model.architecture)),
                model.step, name, i)
            plt.savefig(os.path.join(outdir, title), bbox_inches="tight")

    # https://github.com/hardmaru/cppn-gan-vae-tensorflow/blob/master/sampler.py
    # delta_z = 1.0 / (n_total_frame-1)
    # diff_z = (z2-z1)
    # img_data_array = []
    # for i in range(n_total_frame):
    #   percentage = delta_z * float(i)
    #   factor = percentage
    #   if sinusoid == True:
    #     factor = np.sin(percentage*np.pi/2)
    #   z = z1 + diff_z*factor
    #   print "processing image ", i
    #   img_data_array.append(self.generate(z, x_dim, y_dim, scale))


# def randomWalk(model, starting_pt=np.array([]), step_size=20, steps_till_turn=10,
#                save=True, outdir="."):

#     # TODO: random walk gif in latent space!
#     import moviepy.editor as movie
#     dim = int(model.architecture[0]**0.5)

#     def iterWalk(start):
#         """Yield points on random walk"""
#         def step():
#             """Equally sized step in random direction"""
#             # random normal in each dimension
#             direction = np.random.randn(starting_pt.size)
#             return step_size * (direction / np.linalg.norm(direction))

#         here = start
#         yield here
#         while True:
#             next_step = step()
#             for i in range(steps_till_turn):
#                 here += next_step
#                 yield here

#     if not starting_pt.any():
#         # if not specified, pick randomly from latent space
#         starting_pt = 4 * np.random.randn(model.architecture[-1])
#     walk = iterWalk(starting_pt)

#     def to_rgb(im):
#         # c/o http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
#         return np.dstack([im.astype(np.uint8)] * 3)

#     def make_frame(t):
#         z = next(walk)
#         x_reconstructed = model.decode([z]).reshape([dim, dim])
#         return to_rgb(x_reconstructed)
#     # TODO: recursive ?

#     clip = movie.VideoClip(make_frame, duration=20)

#     if save:
#         title = "{}_random_walk_{}_round_{}.mp4".format(
#             model.datetime, "_".join(map(str, model.architecture)), model.step)
#         clip.write_videofile(os.path.join(outdir, title), fps=10)


# def freeAssociate(model, starting_pt=np.array([]), step_size=20, steps_till_turn=10,
#                   save=True, outdir="."):
#     """TODO: util"""
#     dim = int(model.architecture[0]**0.5)

#     if not starting_pt.any():
#         # if not specified, sample randomly from latent space
#         starting_pt = model.sesh.run(model.z_)

#     def iterWalk(start=starting_pt):
#         """Yield points on random walk"""
#         def step():
#             """Equally sized step in random direction"""
#             # random normal in each dimension
#             direction = np.random.randn(starting_pt.size)
#             return step_size * (direction / np.linalg.norm(direction))

#         here = start
#         yield here

#         while True:
#             next_step = step()
#             for i in range(steps_till_turn):
#                 here += next_step
#                 yield here

#     walk = iterWalk()
#     for i in range(100):
#         z = next(walk)
#         x_reconstructed = model.decode(z).reshape([dim, dim])
#         plt.figure(figsize = (2, 2))
#         plt.imshow(x_reconstructed, cmap="Greys")
#         plt.axis("off")
#         plt.tight_layout()

#         plt.show()
#         if save:
#             title = "{}_dream_{}_round_{}.{}.png".format(
#                 model.datetime, "_".join(map(str, model.architecture)), model.step, i)
#             plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
