from enum import Enum
import networkx as nx

class GraphType(Enum):
    RANDOM_PARTITION = "Random Partition"
    ERDOS_RENYI = "Erdős–Rényi"
    RANDOM_GRAPH = "Random Graph"
    FAKE_COMPUTER = "Fake Computer"

class GraphData:
    def __init__(self, graph_type, G, name, layers):
        """
        Class to define the Graph objects.

        Parameters:
            graph_type (GraphType): Type of the graph (enum value).
            G (NetworkX Graph): The graph object.
            name (str): Name of the graph.
            layers (int): Number of QAOA layers.
        """
        self.graph_type = graph_type
        self.graph = G
        self.name = name
        self.layers = layers

