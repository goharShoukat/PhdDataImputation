import matplotlib.pyplot as plt


def plot(*args, **kwargs):
    fig = plt.figure()
    for arg, kwarg in zip(args, kwargs.values()):
        plt.plot(arg, label=kwarg)
    plt.legend()
    return fig
