import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from scipy.optimize import minimize
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp, SummedOp
from qiskit.opflow.primitive_ops import PrimitiveOp
from qiskit.providers.fake_provider.backends import FakeJakarta, FakeNairobi, FakePerth
from qiskit_aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram
from QAOACircuit import QAOACircuit

import random
from enum import Enum


# Define an enumeration for the simulation types
class SimType(Enum):
    STATE_VECTOR = 1
    IDEAL_SIMULATOR = 2
    NOISY_SIMULATOR = 3

class QAOASimulation:
    def __init__(self, sim_type):
        """
        Class to define the simulation parameters for the circuits.

        Parameters:
            sim_type (Enum): Simulator type parameter.
        """
        self.type = sim_type
        self.backend = self._get_backend()
        self.noise_model = self._noise_setup()

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
    
    def get_opt_params(self, G, p=7, optimizer='COBYLA', opt_options={'maxiter':500, 'disp': True}):
        """
        Returns the optimal parameters for the given graph.
        """
        # p is the number of QAOA alternating operators
        obj = self._get_black_box_objective(G, p)

        init_point = np.array([0.81069872, 2.2067517 , 0.83830696, 2.15579759, 0.37060699,
            2.42068091, 6.1575306 , 2.2453419 , 3.85060091, 6.137845,6.1575306 , 2.2453419 , 3.85060091, 6.137845  ])
        # We are going to limit the number of iterations to 2500
        res_sample = minimize(obj, init_point, method=optimizer, options=opt_options)
        return res_sample
    
    def run_circuit_optimal_params(self, res_sample, G, p=7, analysis=True):
        """
        Runs the circuit with the optimal parameters.
        """
        optimal_theta = res_sample['x']
        qc = QAOACircuit(G.graph, optimal_theta[:p], optimal_theta[p:])
        counts = self._invert_counts(self.backend.run(qc.qaoa_circuit, shots=4096).result().get_counts())

        if analysis:
            energies = defaultdict(int)
            for k, v in counts.items():
                energies[self._maxcut_obj(k, G.graph)] += v
            x,y = zip(*energies.items())
            plt.bar(x,y)
            plt.xlabel('Energy')
            plt.ylabel('Counts')
            plt.title('Energy histogram for MaxCut\nSimulator Type: {}\nGraph: {}'.format(self.type.name, G.name))

        return counts
    
    def best_solution(self, G, counts):
        """
        Returns the best solution for the given graph.
        """
        best_cut, best_solution = min([(self._maxcut_obj(x, G.graph),x) for x in counts.keys()], key=itemgetter(0))
        print(f"Best string: {best_solution} with cut: {-best_cut}")
        return best_cut, best_solution