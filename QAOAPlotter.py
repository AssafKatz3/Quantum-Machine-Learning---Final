import numpy as np
import matplotlib.pyplot as plt

class QAOAPlotter:
    def __init__(self, qaoa_results):
        """
        Class to plot the expected number of repetitions for QAOA results obtained from APOSMM.

        Parameters:
            qaoa_results (dict): A dictionary containing QAOAResult instances for different graph types and p-values.
                                 The dictionary should have the format: {graph_type: {p: QAOAResult}}.
        """
        self.qaoa_results = qaoa_results

    def plot_expected_repetitions(self):
        """
        Plot the expected number of repetitions for QAOA results obtained from APOSMM.
        """
        # Calculate the number of graph types for nrows in subplots
        num_graph_types = len(self.qaoa_results)
        # Calculate the number of columns for subplots
        ncols = (num_graph_types + 1) // 2
        # Calculate figsize based on the number of rows (nrows) and columns (ncols)
        figsize = (ncols * 6, 8)

        # Create subplots based on the number of rows (nrows) and columns (ncols)
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=figsize)

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for idx, (graph_type, p_results) in enumerate(self.qaoa_results.items()):
            ax = axes[idx]

            for p, qaoa_instance in p_results.items():
                blue_line_mean = np.mean(qaoa_instance.intermediate_data, axis=0)
                blue_line_std = np.std(qaoa_instance.intermediate_data, axis=0)

                orange_line_mean = np.mean(qaoa_instance.optimized_data, axis=0)
                orange_line_std = np.std(qaoa_instance.optimized_data, axis=0)

                # Plot the blue line for the distribution at an intermediate stage of the optimization
                ax.errorbar(np.arange(0, qaoa_instance.nfev, qaoa_instance.S), blue_line_mean, yerr=blue_line_std, color='blue', label=f'Intermediate (p={p})', alpha=0.7)

                # Plot the orange line for the distribution for the optimized parameters
                ax.errorbar(np.arange(0, qaoa_instance.nfev, qaoa_instance.S), orange_line_mean, yerr=orange_line_std, color='orange', label=f'Optimized (p={p})', alpha=0.7)

            # Plot the green line for the case of γₖ=βₖ=0 for each k=1,2,...p
            ax.axhline(y=0.5/qaoa_instance.N, color='green', linestyle='--', label='Equal Probability', alpha=0.7)

            ax.set_title(graph_type.capitalize())
            ax.set_xlabel('Function Evaluations (nfev)')
            ax.set_ylabel('Probability')
            ax.legend()

        # Adjust layout to prevent overlapping labels
        plt.tight_layout()

        # Show the plot
        plt.show()
