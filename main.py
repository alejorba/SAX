import numpy as np
import scipy.stats as stats

class SAX:
    """
    Symbolic Aggregate approXimation (SAX) for time series.
    """

    def __init__(self, w: int , a: int, epsilon: float = 1e-6):
        """
        Initializes the SAX transformer.

        Args:
            w (int): the number of segments to represent the time series. Must be a positive integer less than the time series's length.
            a (int): the number of discrete symbols to use in the SAX representation.
            epsilon (float): threshold for standard deviation.
        """
        self.w = w
        self.a = a

        self.epsilon = epsilon

        self.breakpoints = self._compute_breakpoints()
        self.alphabet = np.array([chr(97 + i) for i in range(a)])

        self.distance_table = self._generate_distance_table()


    def _generate_breakpoints(self) -> np.ndarray:
        """
        Generate breakpoints that equally divide the area under a standard normal distribution, to use for mapping PAA coefficients to SAX symbols.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape (a-1, ), containing the breakpoints for this vocabulary size.
        """

        p = np.linspace(0, 1, self.a + 1)[1:-1]

        return stats.norm.ppf(p)

    def _generate_distance_table(self) -> np.ndarray:
        """
        Generates the MINDist lookup table for computing distances bewteen two time series.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape ((a - 1) * (a - 2) // 2,) storing the "upper triangular" lookup table in row-major order.
            Actually, stricly speaking, the symbol distance lookup table is not upper triangular, it is even more sparse; the distance between the same symbol is zero as well as the distance between "neighboring" symbols (a and b, b and c, etc.).
        """
        table = []

        for i in range(self.a - 2):
            for j in range(i + 2, self.a):
                table.append(self.breakpoints[j - 1] - self.breakpoints[i])

        return np.array(table)

    def _normalize(self, series: np.ndarray) -> np.ndarray:
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

        if std < self.epsilon:
            return np.full_like(series, 0.5)
        
        return (series - mean) / std

    def _paa(self, series: np.ndarray) -> np.ndarray:
        """
        Perform Piecewise Aggregate Approximation (PAA) on a time series.

        Args:
            series (np.ndarray): the time series data, a one-dimensional NumPy array.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape (w,), the PAA representation of the time series.
        """

        normalized_series = self._normalize(series)

        return np.array([np.mean(l) for l in np.array_split(normalized_series, self.w)])
    
    # def _symbol_distance(self):

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Convert a time series into its SAX representation.

        Args:
            series (np.ndarray): the time series data, a one-dimensional NumPy array.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape (w,), the SAX representation of the time series.
        """

        return np.array([self.alphabet[np.searchsorted(self.breakpoints, cbar, side='right')] for cbar in self._paa(series)])
        
    def transform_multiple(self, serieses: np.ndarray):
        """
        Converts multiple time series into their SAX representations.

        Args:
            serieses (np.ndarray): the time series data, a two-dimensional NumPy array of shape (n, N).

        Returns:
            np.ndarray: a two-dimensional NumPy array of shape (w, N), the SAX representation of the time series.
        """

        return np.array([self.transform(series) for series in serieses])

    # def mindist(self, )


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