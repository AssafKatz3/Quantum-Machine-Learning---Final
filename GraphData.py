import networkx as nx

class GraphData:
    def __init__(self, G, layers, probability):
        """
        Class to define the Graph objects.

        Parameters:
            G (NetworkX Graph): The graph object.
            layers (int): Number of QAOA layers.
            probability (float): Probability parameter.
        """
        self.graph = G
        self.layers = layers
        self.probability = probability
        self.name = f'Graph #Layers={layers}, probability={probability:.2f}'
