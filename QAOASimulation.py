import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from operator import itemgetter
from scipy.optimize import minimize
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit_algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
from qiskit.providers.fake_provider.backends import FakeJakarta, FakeNairobi, FakePerth
from qiskit_aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram
from QAOACircuit import QAOACircuit
from HeatMap import HeatMap
import time
import multiprocessing

import random
from enum import Enum


# Define an enumeration for the simulation types
class SimType(Enum):
    STATE_VECTOR = 1
    IDEAL_SIMULATOR = 2
    NOISY_SIMULATOR = 3

class QAOASimulation:
    def __init__(self, sim_type, graph):
        """
        Class to define the simulation parameters for the circuits.

        Parameters:
            sim_type (Enum): Simulator type parameter.
        """
        self.type = sim_type
        self.backend = self._get_backend()
        self.noise_model = self._noise_setup()
        self.graph = graph

    def _get_backend(self):
        """
        Returns the backend for the simulation type.
        """
        if self.type == SimType.STATE_VECTOR:
            return Aer.get_backend('statevector_simulator')
        elif self.type == SimType.IDEAL_SIMULATOR:
            return Aer.get_backend('qasm_simulator')
        elif self.type == SimType.NOISY_SIMULATOR:
            return FakeJakarta()
        
    def _noise_setup(self):
        """
        Returns the noise model for the simulation type.
        """
        noise_model = None
        if self.type == SimType.NOISY_SIMULATOR:
            noise_model = NoiseModel.from_backend(self.backend)
        return noise_model

    def _invert_counts(self, counts):
        """
        Inverts the counts dictionary.
        """
        return {k[::-1]:v for k, v in counts.items()}
        

    def _maxcut_obj(self, x, G):
        """
        Compute the cut size for a given cut x of graph G.
        """
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:
                # the edge is cut
                cut -= 1
        return cut

    def _compute_maxcut_energy(self, counts, G, maxcut_vals):
        """
        Compute the maxcut energy for a given set of counts.
        """
        energy = 0
        total_counts = 0
        for meas, meas_count in counts.items():
            obj_for_meas = self._maxcut_obj(meas, G)
            maxcut_vals.append(obj_for_meas)
            energy += obj_for_meas * meas_count
            total_counts += meas_count
        return energy / total_counts

    def _get_black_box_objective(self, G, p, shots_amt=4096):
        """
        Returns the black box objective function for the given graph.
        """
        def f(theta):
            # let's assume first half is betas, second half is gammas
            beta = theta[:p]
            gamma = theta[p:]
            qc = QAOACircuit(G.graph, beta, gamma)
            result = execute(qc.qaoa_circuit, self.backend,
                       noise_model=self.noise_model,
                       shots=shots_amt).result()
            counts = result.get_counts()
            # return the energy
            return self._compute_maxcut_energy(self._invert_counts(counts), G.graph, [])
        return f
    
    def get_opt_params(self, G, init_point, p=7, optimizer='COBYLA', opt_options={'maxiter': 500, 'disp': True}):
        """
        Returns the optimal parameters for the given graph.

        Parameters:
            G: The input graph for the optimization.
            init_point: The initial point for the optimization.
            p: The number of QAOA alternating operators (default is 7).
            optimizer: The optimization method to use (default is 'COBYLA').
            opt_options: Optimization options (default is {'maxiter': 500, 'disp': True}).

        Returns:
            res_sample: The optimization result object.
            heatmap: The HeatMap object used during optimization.
        """
        # p is the number of QAOA alternating operators
        obj = self._get_black_box_objective(G, p)

        # Create an instance of HeatMap
        heatmap = HeatMap(p)
                
        # Perform the optimization with the specified callback
        res_sample = minimize(obj, init_point, method=optimizer, options=opt_options, callback=heatmap.callback)
        
        return res_sample, heatmap



    
    def run_circuit_optimal_params(self, res_sample, G, p=7):
        """
        Runs the circuit with the optimal parameters.
        """
        optimal_beta = res_sample['x'][:p]  # Extract beta values
        optimal_gamma = res_sample['x'][p:]  # Extract gamma values
        
        qc = QAOACircuit(G.graph, optimal_beta, optimal_gamma)
        counts = self._invert_counts(self.backend.run(qc.qaoa_circuit, shots=4096).result().get_counts())

        return counts

    def analyze_results(self, optimal_counts_list, optimal_params_array, split_index=7):
        # Extract count values from dictionaries
        count_values = np.array([sum(count_dict.values()) for count_dict in optimal_counts_list])
        total_count = np.sum(count_values)
        prob_values = count_values / total_count

        # Extract beta and gamma values from optimal_params_array
        optimal_beta_values = np.array([params[:split_index] for params in optimal_params_array])
        optimal_gamma_values = np.array([params[split_index:] for params in optimal_params_array])

        return prob_values, optimal_beta_values, optimal_gamma_values

    def plot_results(self, prob_values, optimal_beta_values, optimal_gamma_values):
        # Calculate the maximum number of bins that can fit on the x-axis
        max_bins = int(np.ceil(np.sqrt(len(optimal_beta_values))))

        # Calculate histogram bins for Beta and Gamma
        beta_bins = np.linspace(0, np.pi, max_bins + 1)
        gamma_bins = np.linspace(-np.pi, np.pi, max_bins + 1)

        # Initialize arrays to store binned probabilities
        binned_beta_probs = np.zeros(max_bins)
        binned_gamma_probs = np.zeros(max_bins)

        # Loop over bins and calculate probabilities for Beta and Gamma
        for i in range(len(beta_bins) - 1):
            beta_mask = (optimal_beta_values[:, 0] >= beta_bins[i]) & (optimal_beta_values[:, 0] < beta_bins[i + 1])
            gamma_mask = (optimal_gamma_values[:, 0] >= gamma_bins[i]) & (optimal_gamma_values[:, 0] < gamma_bins[i + 1])
            binned_beta_probs[i] = np.sum(prob_values[beta_mask])
            binned_gamma_probs[i] = np.sum(prob_values[gamma_mask])

        # Create side by side histograms with different colors for Beta and Gamma
        plt.figure(figsize=(10, 5))
        plt.bar(beta_bins[:-1], binned_beta_probs, width=np.pi / max_bins, align='edge', color='blue', label=r'$P(\beta)$')
        plt.bar(gamma_bins[:-1], binned_gamma_probs, width=np.pi / max_bins, align='edge', color='orange', label=r'$P(\gamma)$')
        plt.xlabel('Angles')
        plt.ylabel('Probabilities')
        plt.title('Histogram of Probabilities for Beta and Gamma')
        plt.legend()
        plt.show()

    def best_solution(self, G, counts):
        """
        Returns the best solution for the given graph.
        """
        best_cut, best_solution = min([(self._maxcut_obj(x, G.graph),x) for x in counts.keys()], key=itemgetter(0))
        print(f"Best string: {best_solution} with cut: {-best_cut}")
        return best_cut, best_solution
    
    def optimize(self, init_point):
         # Print the start time before optimization
        start_time = time.time()
        print(f"Thread {multiprocessing.current_process().name} started at {start_time:.2f} seconds")

       # Perform optimization for a given init_point
        opt_res, heatmap_data = self.get_opt_params(self.graph, init_point)
        # Print the time after optimization
        end_time = time.time()
        print(f"Thread {multiprocessing.current_process().name} finished at {end_time:.2f} seconds")

        return opt_res, heatmap_data

