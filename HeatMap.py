import numpy as np
import matplotlib.pyplot as plt

class HeatMap:

    def __init__(self, p):
        self.p = p
        self.parameter_occurrences = {}


    @staticmethod
    def plot_heatmap(heatmaps_data, digits=2):
        parameter_occurrences = {}
        
        # Loop through the list of HeatMap instances
        for heatmap in heatmaps_data:
            for param_set, occurrences_matrix in heatmap.parameter_occurrences.items():
                beta_values, gamma_values = param_set
                for beta_index, beta in enumerate(beta_values):
                    for gamma_index, gamma in enumerate(gamma_values):
                        rounded_beta = round(beta, digits)
                        rounded_gamma = round(gamma, digits)
                        if (rounded_beta, rounded_gamma) in parameter_occurrences:
                            parameter_occurrences[rounded_beta, rounded_gamma] += occurrences_matrix[beta_index, gamma_index]
                        else:
                            parameter_occurrences[rounded_beta, rounded_gamma] = occurrences_matrix[beta_index, gamma_index]

        # # Create sorted lists of beta and gamma values
        # sorted_beta_values = sorted(set([key[0] for key in parameter_occurrences.keys()]))
        # sorted_gamma_values = sorted(set([key[1] for key in parameter_occurrences.keys()]))

        # # Create dictionaries to map beta and gamma values to indices
        # beta_index_map = {beta: index for index, beta in enumerate(sorted_beta_values)}
        # gamma_index_map = {gamma: index for index, gamma in enumerate(sorted_gamma_values)}

        # # Create a 2D numpy array to store the occurrences
        # occurrences_matrix = np.zeros((len(sorted_gamma_values), len(sorted_beta_values)))

        # # Populate the occurrences_matrix
        # for (beta, gamma), occurrences in parameter_occurrences.items():
        #     beta_idx = beta_index_map[beta]
        #     gamma_idx = gamma_index_map[gamma]
        #     occurrences_matrix[gamma_idx, beta_idx] = occurrences

        plt.figure(figsize=(10, 8))
        plt.imshow(np.asarray(parameter_occurrences), cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Occurrences')
        plt.xlabel('$\\beta$ Index')
        plt.ylabel('$\\gamma$ Index')
        plt.title('$\\beta$-$\\gamma$ Occurrences Heatmap')
        # plt.xticks(np.arange(len(sorted_beta_values)), sorted_beta_values)
        # plt.yticks(np.arange(len(sorted_gamma_values)), sorted_gamma_values)
        plt.show()

    def callback(self, xk):
        beta = xk[:self.p]
        gamma = xk[self.p:]
        param_set = (tuple(beta), tuple(gamma))
        
        if param_set not in self.parameter_occurrences:
            self.parameter_occurrences[param_set] = np.zeros((self.p, len(gamma)))
        
        # Increment the occurrence count for the current parameter set
        for i in range(self.p):
            for j in range(len(gamma)):
                if beta[i] == xk[i] and gamma[j] == xk[self.p + j]:
                    self.parameter_occurrences[param_set][i, j] += 1
