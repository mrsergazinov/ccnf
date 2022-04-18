import numpy as np

def random_spaced(low, high, delta, n, size=None):
    """
    Choose n random values between low and high, with minimum spacing delta.

    If size is None, one sample is returned.
    Set size=m (an integer) to return m samples.

    The values in each sample returned by random_spaced are in increasing
    order.
    """
    empty_space = high - low - (n-1)*delta
    if empty_space < 0:
        raise ValueError("not possible")

    if size is None:
        u = np.random.rand(n)
    else:
        u = np.random.rand(size, n)
    x = empty_space * np.sort(u, axis=-1)
    return low + x + delta * np.arange(n)