import matplotlib.pyplot as plt


def plot(spike_times, min_time=0.0, max_time=1.0, color="k"):
    """Raster plot based on spike arrival times
   
    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """

    for i, spikes in enumerate(spike_times):
        plt.vlines(spikes, i + 0.5, i + 1.5, color=color)

    if min_time < 0.0:
        plt.vlines([0.0], 0.5, len(spike_times) + 0.5, color="r", linestyle="--")

    plt.ylim(0.5, len(spike_times) + 0.5)
    plt.xlim(min_time, max_time)
    plt.yticks([])

