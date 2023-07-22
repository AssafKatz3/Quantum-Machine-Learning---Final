import numpy as np

class QAOAResult:
    def __init__(self, title, p, nfev, S, qaoa_data):
        """
        Class to store QAOA results for a specific graph type and p-value.

        Parameters:
            title (str): The title of the QAOA result.
            p (int): The p-value used in the QAOA.
            nfev (int): The number of function evaluations (total).
            S (int): The number of function evaluations per update.
            qaoa_data (numpy.ndarray or list of dict): QAOA data for different stages of optimization.
                Each entry in qaoa_data should correspond to a specific stage and contain gamma and beta values.
        """
        self.title = title
        self.p = p
        self.nfev = nfev
        self.S = S
        self.qaoa_data = np.array(qaoa_data)

        # Calculate the total number of samples obtained during the optimization process
        self.total_samples = len(qaoa_data) * S

        # Initialize additional data for the "Equal Probability" case
        self.equal_probability_data = np.full((len(qaoa_data), 2*p), 0.5/(2**p))

    def add_intermediate_data(self, intermediate_data):
        """
        Add intermediate data to the QAOA result.

        Parameters:
            intermediate_data (numpy.ndarray or list of dict): Intermediate QAOA data to add.
        """
        intermediate_data = np.array(intermediate_data)
        self.qaoa_data = np.concatenate((self.qaoa_data, intermediate_data), axis=0)

    def add_optimized_data(self, optimized_data):
        """
        Add optimized data to the QAOA result.

        Parameters:
            optimized_data (numpy.ndarray or list of dict): Optimized QAOA data to add.
        """
        optimized_data = np.array(optimized_data)
        self.qaoa_data = np.concatenate((self.qaoa_data, optimized_data), axis=0)

    def add_equal_probability_data(self):
        """
        Add data for the "Equal Probability" case to the QAOA result.
        """
        self.qaoa_data = np.concatenate((self.qaoa_data, self.equal_probability_data), axis=0)

    # Other methods (if any)
