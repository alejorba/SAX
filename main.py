import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.stats import norm


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

        self.breakpoints = self._generate_breakpoints()
        self.alphabet = np.array([chr(97 + i) for i in range(a)])

        self.distance_table = self._generate_distance_table()


    def _generate_breakpoints(self) -> np.ndarray:
        """
        Generate breakpoints that equally divide the area under a standard normal distribution, to use for mapping PAA coefficients to SAX symbols.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape (a-1, ), containing the breakpoints for this vocabulary size.
        """

        p = np.linspace(0, 1, self.a + 1)[1:-1]

        return norm.ppf(p)

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

        Args:
            series (np.ndarray): the time series data, a one-dimensional NumPy array.

        Return:
            np.ndarray: a one-dimensional NumPy array of the same shape as the input time series, but normalized.
        """

        mean = np.mean(series)
        std = np.std(series, ddof=1)
        
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

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Convert a time series into its SAX representation.
        If the standard deviation is below `epsilon`, return a SAX representation with only the middle value of the alphabet.

        Args:
            series (np.ndarray): the time series data, a one-dimensional NumPy array.

        Returns:
            np.ndarray: a one-dimensional NumPy array of shape (w,), the SAX representation of the time series.
        """

        if np.std(series) < self.epsilon:
            return np.full(self.w, chr(97 + self.a // 2))

        return np.array([self.alphabet[np.searchsorted(self.breakpoints, cbar, side='right')] for cbar in self._paa(series)])
        
    def transform_multiple(self, serieses: np.ndarray):
        """
        Converts multiple time series into their SAX representations.

        Args:
            serieses (np.ndarray): the time series data, a two-dimensional NumPy array of shape (n, N).

        Returns:
            np.ndarray: a two-dimensional NumPy array of shape (N, w), the SAX representation of the time series.
        """

        return np.array([self.transform(series) for series in serieses])

    def mindist(self, q: np.ndarray, c: np.ndarray) -> float:
        """
        Computes the MINDist between the SAX representations of two time series, q and c.

        Args:
            c (np.ndarray): SAX representation of the first time series.
            q (np.ndarray): sax representation of the second time series.

        Returns:
            float: computed distance.
        """

        d = 0.0

        for sq, sc in zip(q, c):
            i, j = ord(sq) - 97, ord(sc) - 97

            if np.abs(i - j) > 1:
                if i > j:
                    i, j = j, i
                
                index = np.sum(self.a - np.arange(i) - 2) + (j - (i + 2))
                d += self.distance_table[index]**2

        return np.sqrt(d)
        # When comparing distances between sequences of the same original length and the same "word" length, multiplying by the factor sqrt(n/w) makes little sense (all of the distance values will be scales by the same coefficient!).
        # In any case, I believe the paper is either inconsistent with its notation or made a mistake.
        # As it was presented, MINDIST is calculated based on the symbolic representation of two sequences Q_hat and C_hat (see Eq. 5).
        # Therefore, MINDIST should not depend on the orignal sequence's length, since it is a parameter of the original sequences Q and C and not of their reprsentations Q_hat and C_hat.

def euclideandist(q: np.ndarray, c: np.ndarray) -> float:
    """
    Computes the euclidean distance between two time series q and c.

    Args:
            c (np.ndarray): SAX representation of the first time series.
            q (np.ndarray): sax representation of the second time series.

        Returns:
            float: computed distance.
    """

    return np.sqrt(np.mean(np.subtract(q, c) ** 2))

def plot_dendrograms_besides_time_series(euclidean_linkage, sax_linkage, time_series):
    """
    Wrapper function to recreate Figure 11 of the paper.

    Args:
        linkage_matrix
        time_series
    """

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(9, 4, width_ratios=[1, 2, 1, 2])

    ax_dendro_euclidean = plt.subplot(gs[:, 1])
    ax_series_euclidean = [plt.subplot(gs[i, 0]) for i in range(len(cc_samples))]

    ax_dendro_sax = plt.subplot(gs[:, 3])
    ax_series_sax = [plt.subplot(gs[i, 2]) for i in range(len(cc_samples))]

    euclidean_dendrogram = hierarchy.dendrogram(euclidean_linkage, orientation='right', ax=ax_dendro_euclidean, no_labels=True, color_threshold=0, above_threshold_color='black')

    leaf_order = euclidean_dendrogram['leaves']

    for i, idx in enumerate(leaf_order):
        ax_series_euclidean[i].plot(time_series[idx], color='black', lw=1)
        ax_series_euclidean[i].set_yticks([])
        ax_series_euclidean[i].set_xticks([])
        ax_series_euclidean[i].spines['top'].set_visible(False)
        ax_series_euclidean[i].spines['bottom'].set_visible(False)
        ax_series_euclidean[i].spines['left'].set_visible(False)
        ax_series_euclidean[i].spines['right'].set_visible(False)

    ax_dendro_euclidean.set_xticks([])
    ax_dendro_euclidean.spines['top'].set_visible(False)
    ax_dendro_euclidean.spines['bottom'].set_visible(False)
    ax_dendro_euclidean.spines['left'].set_visible(False)
    ax_dendro_euclidean.spines['right'].set_visible(False)
    ax_dendro_euclidean.set_title('Euclidean')

    sax_dendrogram = hierarchy.dendrogram(sax_linkage, orientation='right', ax=ax_dendro_sax, no_labels=True, color_threshold=0, above_threshold_color='black')

    leaf_order = sax_dendrogram['leaves']

    for i, idx in enumerate(leaf_order):
        ax_series_sax[i].plot(time_series[idx], color='black', lw=1)
        ax_series_sax[i].set_yticks([])
        ax_series_sax[i].set_xticks([])
        ax_series_sax[i].spines['top'].set_visible(False)
        ax_series_sax[i].spines['bottom'].set_visible(False)
        ax_series_sax[i].spines['left'].set_visible(False)
        ax_series_sax[i].spines['right'].set_visible(False)

    ax_dendro_sax.set_xticks([])
    ax_dendro_sax.spines['top'].set_visible(False)
    ax_dendro_sax.spines['bottom'].set_visible(False)
    ax_dendro_sax.spines['left'].set_visible(False)
    ax_dendro_sax.spines['right'].set_visible(False)
    ax_dendro_sax.set_title('SAX')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = './data/'

    cc_path = data_path + 'CC/synthetic_control.data'
    cbf_path_train = data_path + 'CBF/CBF_TRAIN.tsv'
    cbf_path_test = data_path + 'CBF/CBF_TEST.tsv'

    # Clustering Benchmark
    clustering_params = {'w': 16, 'a': 10}

    sax_clustering = SAX(**clustering_params)

    cc = np.loadtxt(cc_path)

    cc_normal = cc[np.random.choice(np.arange(0, 100), size=3, replace=False)]
    cc_decreasing = cc[np.random.choice(np.arange(300, 400), size=3, replace=False)]
    cc_upward = cc[np.random.choice(np.arange(400, 500), size=3, replace=False)]

    cc_samples = np.vstack((cc_normal, cc_decreasing, cc_upward))
    cc_samples_symbolic = sax_clustering.transform_multiple(cc_samples)

    euclidean_linkage = hierarchy.linkage(pdist(cc_samples, metric=euclideandist), method='complete')
    sax_linkage = hierarchy.linkage(pdist(cc_samples_symbolic, metric=sax_clustering.mindist), method='complete')

    plot_dendrograms_besides_time_series(euclidean_linkage, sax_linkage, cc_samples)

    

    # Classification Benchmark
    # w = n / 4
    # alphabet size np.arange(5, 11)
    # classification_params = {'w': 32, 'a': 10}

    # cc_indices = np.random.permutation(cc.shape[0])
    # cc_train = cc[cc_indices[:int(np.floor(0.8 * cc.shape[0]))]]
    # cc_test = cc[cc_indices[int(np.ceil(0.8 * cc.shape[0])):]]

    # # TODO: This is raw data, where would the labels be?

    # cbf_train = np.loadtxt(cbf_path_train, delimiter="\t")
    # cbf_test = np.loadtxt(cbf_path_test, delimiter="\t")

    # print(cc_train.shape, cc_test.shape, cbf_train.shape, cbf_test.shape)
