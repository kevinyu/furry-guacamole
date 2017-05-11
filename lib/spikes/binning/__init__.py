import numpy as np


# FIXME: need to allow for non-ms binning!
def bin_spikes(spike_times, min_time=None, max_time=None, t_align="start"):
    """Convert arrays of spike times into a single binary array

    Converts spike times into binary arrays with 1 ms time bins

    Args:
    spike_times (list, N_units)
        A list of arrays in which each array contains the spike times
        (in seconds) for a single unit/trial
    min_time (float, default=None)
        Specify what time range to include (set the array width).
        If None, will use the floor of the earliest and ceil of the latest
        spike in all rows (in seconds)
    max_time (float, default=None)
    t_align (str, default="start", choices=["start", "mid", "end"])
        Where to align the timestamps for each bin (start of the bin,
        midpoint of the time bin, or end of time bin)

    Returns:
    t_arr (N_timesteps)
        1D array representing the time bins
    spikes (N_units x N_timesteps)
        2D array in which each row is a single unit/trial and each column
        represents a one millisecond timestep
    """
    # map to milliseconds
    spike_times = [row * 1e3 for row in spike_times]
    rows = len(spike_times)

    if min_time is None:
        min_time = int(min(np.floor(row[0]) for row in spike_times if len(row)))
    else:
        min_time = int(1e3 * min_time)

    if max_time is None:
        max_time = int(max(np.ceil(row[-1]) for row in spike_times if len(row)))
    else:
        max_time = int(1e3 * max_time)

    # generate output array with correct dimensions
    spikes = np.zeros((rows, int(max_time - min_time)))

    for i, row in enumerate(spike_times):
        for spike_time in row[(row >= min_time) & (row <= max_time)]:
            spikes[i, int(np.floor(spike_time)) - min_time] = 1

    t_arr = np.linspace(min_time * 1e-3, max_time * 1e-3, spikes.shape[1])

    return t_arr, spikes

