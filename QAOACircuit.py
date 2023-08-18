import numpy as np
import networkx as nx
from collections import defaultdict
from operator import itemgetter
from qiskit import QuantumCircuit

class QAOACircuit:
    def __init__(self, G, beta:list, gamma:list):
        """
        Class to define the circuit for QAOA.

        Parameters:
            qaoa_circuit (Quantum Circuit): A qiskit quantum QAOA circuit for the given graph.
            beta (list): A list of beta parameters for the QAOA circuit.
            gamma (list): A list of gamma parameters for the QAOA circuit.
        """
        self.graph = G
        self.vertex_count = G.number_of_nodes()
        self.beta_list = beta
        self.gamma_list = gamma

        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.qaoa_circuit = QuantumCircuit(self.vertex_count,self.vertex_count)
        # first, apply a layer of Hadamards
        self.qaoa_circuit.h(range(self.vertex_count))
        # second, apply p alternating operators
        for i in range(p):
            self.qaoa_circuit = self.qaoa_circuit.compose(self._get_cost_operator_circuit(self.gamma_list[i]))
            self.qaoa_circuit.barrier(range(self.vertex_count))
            self.qaoa_circuit = self.qaoa_circuit.compose(self._get_mixer_operator_circuit(self.beta_list[i]))
        # finally, do not forget to measure the result!
        self.qaoa_circuit.barrier(range(self.vertex_count))
        self.qaoa_circuit.measure(range(self.vertex_count), range(self.vertex_count))

    def _append_zz_term(self, qc, q1, q2, gamma):
        """
        Append the ZZ term to the cost operator circuit.
        """
        qc.cx(q1, q2)
        qc.rz(-gamma, q2)
        qc.cx(q1, q2)
    
    def _append_x_term(self, qc, q1, beta):
        """
        Append the X term to the mixer operator circuit.
        """
        qc.rx(2*beta, q1)

    def _get_cost_operator_circuit(self, gamma):
        """
        Generate the cost operator circuit.
        """
        qc = QuantumCircuit(self.vertex_count, self.vertex_count)
        for i, j in self.graph.edges():
            self._append_zz_term(qc, i, j, gamma)
        return qc

    def _get_mixer_operator_circuit(self, beta):
        """
        Generate the mixer operator circuit.
        """
        qc = QuantumCircuit(self.vertex_count, self.vertex_count)
        for n in self.graph.nodes():
            self._append_x_term(qc, n, beta)
        return qc
    
    def draw_circuit(self):
        """
        Draw the circuit.
        """
        return self.qaoa_circuit.draw('mpl')