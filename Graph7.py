class Graph7:
    def __init__(self, G, name):
        """
        Class to define the Graph objects.

        Parameters:
            G (nx.Graph): A NetworkX graph object.
            name (str): Name of the graph.
        """
        self.graph = G
        self.name = name

    def get_graph(self):
        return self.graph

    def get_title(self):
        return self.name
