from enum import Enum
import networkx as nx

class GraphType(Enum):
    RANDOM_PARTITION = "Random Partition"
    ERDOS_RENYI = "Erdős–Rényi"
    RANDOM_GRAPH = "Random Graph"
    FAKE_COMPUTER = "Fake Computer"

class GraphData:
    def __init__(self, graph_type, G, name, layers, noise_multiplier=None):
        """
        Class to define the Graph objects.

        Parameters:
            graph_type (GraphType): Type of the graph (enum value).
            G (NetworkX Graph): The graph object.
            name (str): Name of the graph.
            layers (int): Number of QAOA layers.
            noise_multiplier (int): The noise multiplier to include in the new names (defualt=None).
        """
        self.graph_type = graph_type
        self.graph = G
        self.layers = layers
        self.short_name = name
        self.name_without_noise = f"{name} #Layers {layers}"
        if noise_multiplier == None:
            self.name = self.name_without_noise
        else:
            self.name = f'{self.name_without_noise} Noise ×{noise_multiplier}'
        self.noise_multiplier = noise_multiplier

    @staticmethod
    def duplicate_graph_objs_with_noise_multiplier(graph_objs, noise_multiplier):
        """
        Duplicate a list of GraphData objects with modified names including noise multiplier.

        Parameters:
            graph_objs (list[GraphData]): List of GraphData objects to duplicate.
            noise_multiplier (int): The noise multiplier to include in the new names.

        Returns:
            list[GraphData]: List of duplicated GraphData objects with modified names.
        """
        graph_objs_with_name = []
        for graph_obj in graph_objs:
            new_graph_obj = GraphData(graph_obj.graph_type, graph_obj.graph, graph_obj.name, graph_obj.layers, noise_multiplier)
            graph_objs_with_name.append(new_graph_obj)
        return graph_objs_with_name
