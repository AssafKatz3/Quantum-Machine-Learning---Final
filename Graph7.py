import networkx as nx

class Graph7:
    def __init__(self, G, name):
        """
        Class to define the Graph objects.

        Parameters:
            qaoa_circuit (Quantum Circuit): A qiskit quantum QAOA circuit for the given graph.
            beta (list): A list of beta parameters for the QAOA circuit.
            gamma (list): A list of gamma parameters for the QAOA circuit.
        """
        self.graph = G
        self.name = name