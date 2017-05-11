import matplotlib.pyplot as plt


def plot(spike_times, min_time=0.0, max_time=1.0, color="k"):
    """Raster plot based on spike arrival times

    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """

    for i, spikes in enumerate(spike_times):
        plt.vlines(spikes, i, i + 1.0, color=color)

    if min_time < 0.0:
        plt.vlines([0.0], 0.0, len(spike_times), color="r", linestyle="--")

    plt.ylim(0.0, len(spike_times))
    plt.xlim(min_time, max_time)

