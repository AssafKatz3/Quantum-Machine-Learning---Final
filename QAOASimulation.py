import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
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

import random
from enum import Enum


# Define an enumeration for the simulation types
class SimType(Enum):
    STATE_VECTOR = "State Vector"
    IDEAL_SIMULATOR = "Ideal Simulator"
    NOISY_SIMULATOR = "Noisy Simulator"

class QAOASimulation:
    """
    Class to define the simulation parameters and methods for Quantum Approximate Optimization Algorithm (QAOA) circuits.

    Parameters:
        sim_type (SimType): The type of simulator to use (STATE_VECTOR, IDEAL_SIMULATOR, NOISY_SIMULATOR).
        shot_amt (int): Number of shots for the simulation (default is 4096).
        noise_multiplier (float): Multiplier for noise probabilities (default is None).

    Attributes:
        type (SimType): The type of simulator being used.
        backend (qiskit.providers.BaseBackend): The backend for the simulation type.
        noise_model (qiskit.providers.aer.noise.noise_model.NoiseModel): The noise model for the simulation type.
        shot_amt (int): Number of shots for the simulation.

    Methods:
        _get_backend(): Returns the backend for the simulation type.
        _noise_setup(multiplier): Returns the noise model for the simulation type.
        _multiply_noise_probabilities(data, multiplier): Multiply probabilities of noise model data while normalizing.
        _normalize_multiplied_probabilities(probabilities, multiplier): Normalize probabilities while applying a multiplier.
        _invert_counts(counts): Inverts a counts dictionary.
        maxcut_obj(x, G): Computes the cut size for a given cut x of graph G.
        compute_maxcut_energy(counts, G, maxcut_vals): Compute the MaxCut energy for a set of measurement counts.
        _get_black_box_objective(G, p): Returns the black box objective function for the given graph.
        get_opt_params(G, p, optimizer, opt_options): Get the optimal parameters for the given graph.
        run_circuit_optimal_params(res_sample, G, p, analysis): Runs the circuit with the optimal parameters.
        best_solution(G, counts): Returns the best solution for the given graph.
    """
    def __init__(self, sim_type, shot_amt, noise_multiplier=None):
        """
        Class to define the simulation parameters for the circuits.

        Parameters:
            sim_type (Enum): Simulator type parameter.
            shot_amt (int): Number of shots for the simulation
        """
        self.type = sim_type
        self.backend = self._get_backend()
        self.noise_model = self._noise_setup(noise_multiplier)
        self.shot_amt = shot_amt

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
        
    def _noise_setup(self, multiplier):
        """
        Returns the noise model for the simulation type.
        """
        noise_model = None
        if self.type == SimType.NOISY_SIMULATOR:
            noise_model = NoiseModel.from_backend(self.backend)
            if multiplier != None:
                noise_model = NoiseModel.from_dict(self._multiply_noise_probabilities(noise_model.to_dict(), multiplier))
        return noise_model

    def _multiply_noise_probabilities(self, data, multiplier):
        """
        Multiply probabilities of noise model data while normalizing.

        Parameters:
            data (dict): Data to be multiplied and normalized.
            multiplier (float): Multiplier for probabilities.

        Returns:
            dict: Multiplied and normalized data.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "probabilities" and isinstance(value, list):
                    self._normalize_multiplied_probabilities(value, multiplier)
                if isinstance(value, dict):
                    self._multiply_noise_probabilities(value, multiplier)
                elif isinstance(value, list):
                    for item in value:
                        self._multiply_noise_probabilities(item, multiplier)
        elif isinstance(data, list):
            for item in data:
                self._multiply_noise_probabilities(item, multiplier)

        return data

    def _normalize_multiplied_probabilities(self, probabilities, multiplier):
        """
        Normalize probabilities while applying a multiplier.

        Parameters:
            probabilities (list): List of probabilities.
            multiplier (float): Multiplier for probabilities.
        """
        if isinstance(probabilities, list):
            if isinstance(probabilities[0], list):
                for sublist in probabilities:
                    self._normalize_multiplied_probabilities(sublist, multiplier)
            else:
                new_values = [val * multiplier if val * multiplier <= 1 else val for val in probabilities]
                sum_values = sum(new_values)
                if sum_values > 0:
                    probabilities[:] = [val / sum_values for val in new_values]

    def _invert_counts(self, counts):
        """
        Inverts the counts dictionary.

        Parameters:
            counts (dict): The dictionary containing measurement outcomes and their corresponding counts.

        Returns:
            dict: The inverted dictionary where measurement outcomes are reversed and counts are preserved.
        """
        return {k[::-1]:v for k, v in counts.items()}
        

    def maxcut_obj(self, x, G):
        """
        Compute the cut size for a given cut x of graph G.

        Parameters:
            x (list): A list representing the node assignment to partitions in the cut.
            G (networkx.Graph): The graph for which the cut size is computed.

        Returns:
            int: The computed cut size.
        """
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:
                # the edge is cut
                cut -= 1
        return cut

    def compute_maxcut_energy(self, counts, G, maxcut_vals):
        """
        Compute the MaxCut energy for a given set of measurement counts.

        Parameters:
            counts (dict): A dictionary of measurement outcomes and their corresponding counts.
            G (networkx.Graph): The graph for which MaxCut energy is computed.
            maxcut_vals (list): A list to store individual MaxCut values.

        Returns:
            float: The computed MaxCut energy.
        """
        energy = 0
        total_counts = 0
        for meas, meas_count in counts.items():
            obj_for_meas = self.maxcut_obj(meas, G)
            maxcut_vals.append(obj_for_meas)
            energy += obj_for_meas * meas_count
            total_counts += meas_count
        return energy / total_counts

    def _get_black_box_objective(self, G, p):
        """
        Returns the black box objective function for the given graph.
        """
        def f(theta):
            # let's assume first half is betas, second half is gammas
            beta = theta[:p]
            gamma = theta[p:]
            self.qc = QAOACircuit(G.graph, beta, gamma)
            result = execute(self.qc.qaoa_circuit, self.backend,
                       noise_model=self.noise_model,
                       shots=self.shot_amt).result()
            counts = result.get_counts()
            # return the energy
            return self.compute_maxcut_energy(self._invert_counts(counts), G.graph, [])
        return f
    
    def get_opt_params(self, G, p, optimizer='COBYLA', opt_options={'maxiter': 500, 'disp': True}):
        """
        Get the optimal parameters for the given graph.

        Parameters:
            G (networkx.Graph): The graph for which to find the optimal parameters.
            p (int): The number of QAOA alternating operators.
            optimizer (str): The optimization method to use (default is 'COBYLA').
            opt_options (dict): Options to pass to the optimization method (default is {'maxiter': 500, 'disp': True}).

        Returns:
            scipy.optimize.OptimizeResult: The optimization result containing the optimal parameters.
        
        Prints:
            LaTeX source for circuit drawing
        """
        obj = self._get_black_box_objective(G, p)

        # Initialize point to zero        
        init_point = np.zeros(2*p)
        
        # Perform the optimization
        res_sample = minimize(obj, init_point, method=optimizer, options=opt_options)
        print(self.qc.draw_circuit())
        return res_sample
    
    def run_circuit_optimal_params(self, res_sample, G, p, analysis=False):
        """
        Runs the circuit with the optimal parameters.

        Parameters:
            res_sample (scipy.optimize.OptimizeResult): The optimization result containing the optimal parameters.
            G (GraphData): The graph for which to run the circuit.
            p (int): The number of QAOA alternating operators.
            analysis (bool): Flag indicating whether to perform analysis (default is False).

        Returns:
            dict: A dictionary containing measurement counts.
        """
        optimal_theta = res_sample['x']
        qc = QAOACircuit(G.graph, optimal_theta[:7], optimal_theta[7:])
        counts = self._invert_counts(self.backend.run(qc.qaoa_circuit, shots=self.shot_amt).result().get_counts())

        return counts
    
    def best_solution(self, G, counts):
        """
        Returns the best solution for the given graph.

        Parameters:
            G (GraphData): The graph for which to find the best solution.
            counts (dict): A dictionary containing measurement counts.

        Returns:
            tuple: A tuple containing the best cut value and the corresponding solution.
        """
        best_cut, best_solution = min([(self.maxcut_obj(x, G.graph),x) for x in counts.keys()], key=itemgetter(0))
        return best_cut, best_solution
