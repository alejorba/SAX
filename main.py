import numpy as np

def normalize(series: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalize a time series to have zero mean and unit variance.
    If the standard deviation is below `epsilon`, return a constant time series.

    Args:
        series (np.ndarray): the time series data, a one-dimensional NumPy array.
        epsilon (float): threshold for standard deviation.

    Return:
        np.ndarray: a one-dimensional NumPy array of the same shape as the input time series, but normalized.
    """

    mean = np.mean(series)
    std = np.std(series)

    if std < epsilon:
        return np.full_like(series, 0.5)
    
    return (series - mean) / std
    

def paa(series: np.ndarray, w: int) -> np.ndarray:
    """
    Perform Piecewise Aggregate Approximation (PAA) on a time series.

    Args:
        series (np.ndarray): the time series data, a one-dimensional NumPy array.
        w (int): the number of segments to represent the time series. Must be a positive integer less than the time series's length.
        
    Returns:
        np.ndarray: a one-dimensional NumPy array of shape (w,), the PAA representation of the time series.
    """

    normalized_series = normalize(series)

    print(np.array([np.mean(l) for l in np.array_split(normalized_series, w)]))



# SAX (w)

# LookUpTable(a)

# MINDist(x, y, LookUp)

# EuclideanDist(x,y)

# Paper Recommends 5 <= a <= 8

if __name__ == "__main__":

    data_path = './data/'

    cc_path = data_path + 'CC/synthetic_control.data'
    cbf_path = data_path + 'CBF/'

    # Clustering Benchmark
    cc = np.loadtxt(cc_path)

    cc_normal = cc[np.random.choice(np.arange(0, 100), size=3, replace=False)]
    cc_decreasing = cc[np.random.choice(np.arange(300, 400), size=3, replace=False)]
    cc_upward = cc[np.random.choice(np.arange(400, 500), size=3, replace=False)]

    print(cc_normal.shape, cc_decreasing.shape, cc_upward.shape)


    paa(np.array([1,2,3,4,5,6,7,8,9,10]), 3)