import util


def gaussian_filter(spikes, std, n=3.0):
    return util.conv(spikes, util.gaussian_convolver, std, n=n)


def exp_filter(spikes, tau, n=4.0):
    return util.conv(spikes, util.exponential_convolver, tau, n=n)

