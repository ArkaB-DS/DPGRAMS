import numpy as np

class QuantileScaler:
    """
    Robust per-feature scaling using quantiles (typically 1st and 99th).
    Transforms features to approximately [0, 1] range and stores parameters for inverse transform.
    """

    def __init__(self, lower_q=1.0, upper_q=99.0):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.q_low = None
        self.q_high = None

    def fit(self, data):
        """
        Compute quantiles from data.
        
        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)
        """
        self.q_low = np.percentile(data, self.lower_q, axis=0)
        self.q_high = np.percentile(data, self.upper_q, axis=0)
        # Prevent division by zero
        self.q_high = np.where(self.q_high == self.q_low, self.q_high + 1e-8, self.q_high)

    def transform(self, data):
        """
        Scale data to [0, 1] using stored quantiles.
        
        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Scaled data
        """
        if self.q_low is None or self.q_high is None:
            raise RuntimeError("QuantileScaler not fitted.")
        return (data - self.q_low) / (self.q_high - self.q_low)

    def inverse_transform(self, data_scaled):
        """
        Undo scaling.
        
        Args:
            data_scaled (np.ndarray): Scaled data in [0, 1]

        Returns:
            np.ndarray: Original scale data
        """
        return data_scaled * (self.q_high - self.q_low) + self.q_low
