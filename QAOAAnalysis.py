import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate
from collections import defaultdict
import matplotlib
from GraphData import GraphType

class QAOAAnalysis:
    """
    Class to analyze and visualize Quantum Approximate Optimization Algorithm (QAOA) results.

    Parameters:
        graph_objs (list): List of GraphData objects.
        simulator (QAOASimulation): The simulator instance to use for analysis.
        cut_values (dict): A dictionary containing cut values for each graph.

    Attributes:
        _simulator (QAOASimulation): The simulator instance.
        _cut_values (dict): A dictionary containing cut values (deterministic) for each graph.
        _best_solutions (dict): A dictionary containing best solutions for each graph.
        _graph_objs (list): List of GraphData objects.
        _name (str): The name of the first graph in graph_objs.
    """

    def __init__(self, graph_objs, simulator, cut_values):
        self._simulator = simulator
        self._cut_values = cut_values
        self._best_solutions = {}
        self.graph_objs = graph_objs 

    @property
    def graph_objs(self):
        return self._graph_objs

    @graph_objs.setter
    def graph_objs(self, value):
        self._graph_objs = value
        self._name = value[0].name 
        self._name_without_noise = value[0].name_without_noise
        self.noise_multiplier = value[0].noise_multiplier

    @property
    def name(self):
        return self._name

    @property
    def simulator(self):
        return self._simulator


    @property
    def cut_values(self):
        return self._cut_values

    def run_optimal_params(self):
        """
        Run optimal parameter simulation for each graph and store results.

        Returns:
        dict: Dictionary containing optimal results counts for each graph object.
        """
        opt_res_counts_dict = {}  # Create a dictionary to store results

        print(f"#Shots {self._simulator.shot_amt}:")
        for graph_obj in self._graph_objs:
            opt_res = self._simulator.get_opt_params(graph_obj, graph_obj.layers)
            opt_res_counts = self._simulator.run_circuit_optimal_params(opt_res, graph_obj, graph_obj.layers)
            opt_res_counts_dict[graph_obj.name] = opt_res_counts

            print(f"Graph: {graph_obj.name}")
            print("Optimal Parameters:", opt_res)
            print("Optimal Results Counts:", opt_res_counts)

        return opt_res_counts_dict

    def find_best_solutions(self, opt_res_counts_dict):
        """
        Find the best solutions for each graph using optimal results counts.

        Parameters:
        opt_res_counts_dict (dict): Dictionary containing optimal results counts.

        Prints:
        str: Table containing best solutions for each graph object.
        """
        # Find the best solutions for each graph_obj using opt_res_counts
        for graph_obj in self._graph_objs:
            opt_res_counts = opt_res_counts_dict[graph_obj.name]
            best_cut, best_solution = self._simulator.best_solution(graph_obj, opt_res_counts)
            self._best_solutions[graph_obj.name] = (best_cut, best_solution)

        # Prepare data for the table
        table_data = []
        for graph_obj_name, (best_cut, best_solution) in self._best_solutions.items():
            table_data.append([graph_obj_name, -best_cut, best_solution, self._simulator.shot_amt])

        # Define table headers
        headers = ["Graph Name", "Best Cut", "Best Solution", "#Shots"]

        # Display table using tabulate
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        print(table)

    def plot_maxcut_graphs(self):
        """
        Plot graphs with colored nodes and edges based on best solutions.

        Displays:
        Plots of graphs with colored nodes and edges indicating the best solutions.
        """
        # Loop over graph_objs
        for graph_obj in self._graph_objs:
            # Get the best solution for the current graph_obj
            best_solution = self._best_solutions[graph_obj.name][1]

            # Color the graph nodes by part
            G = graph_obj.graph
            colors = ['r' if best_solution[node] == '0' else 'b' for node in G]

            # Identify the cut edges
            cut_edges = [(u, v) for u, v in G.edges() if ((best_solution[u] == '0' and best_solution[v] != '0') or (best_solution[u] != '0' and best_solution[v] == '0'))]

            cut_edge_color = 'red'
            non_cut_edge_color = 'black'

            # Assign edge colors based on whether they are cut edges or not
            for edge in G.edges():
                if edge in cut_edges:
                    G[edge[0]][edge[1]]['color'] = cut_edge_color
                else:
                    G[edge[0]][edge[1]]['color'] = non_cut_edge_color

            edge_colors = [G[edge[0]][edge[1]]['color'] for edge in G.edges()]

            # Draw the graph with colored nodes and edges
            nx.draw(G, node_color=colors, with_labels=True, font_weight='bold', edge_color=edge_colors)
            plt.title(f'Graph: {graph_obj.name}, #Shots: {self._simulator.shot_amt}')
            plt.show()

    def plot_approximate_rate(self):
        """
        Plot approximate rate vs. number of layers.

        Displays:
        Plots of approximate rate vs. number of layers.
        """
        # Separate the data for plotting
        approximately_rate_layers_data = {}  # Dictionary to store approximately_rate data per name
        for graph_obj in self._graph_objs:
            name = graph_obj.short_name
            layers = graph_obj.layers
            if name not in approximately_rate_layers_data:
                approximately_rate_layers_data[name] = {}
            approximately_rate = -self._best_solutions[graph_obj.name][0] / self._cut_values[graph_obj.name]
            approximately_rate_layers_data[name][layers] = approximately_rate

        # Extract unique layers and names from graph_objs
        unique_layers = sorted(set(graph_obj.layers for graph_obj in self._graph_objs))
        unique_names = sorted(set(graph_obj.short_name for graph_obj in self._graph_objs))
        
        line_styles = ['-', '--', '-.', ':']

        # Plot Approximate Rate vs. Number of Layers for each graph
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(unique_names):
            approximately_rate_list = list(dict(sorted(approximately_rate_layers_data[name].items())).values())
            linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
            
            # Pad or truncate approximately_rate_list to match the length of unique_layers
            approximately_rate_list_padded = approximately_rate_list + [None] * (len(unique_layers) - len(approximately_rate_list))
            ax.plot(unique_layers, approximately_rate_list_padded, linestyle=linestyle, label=name)
            
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Approximate Rate (Optimal Counts / Cut Values)')
        ax.set_title(f'Approximate Rate vs. Number of Layers\n#Shots: {self._simulator.shot_amt}')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_energy_histogram_maxcut(self, counts, G):
        """
        Plot the energy histogram for MaxCut.

        Parameters:
            counts (dict): A dictionary containing measurement counts.
            G (GraphData): The graph for which to plot the energy histogram.
        """
        energies = defaultdict(int)
        for k, v in counts.items():
            energies[self._simulator._maxcut_obj(k, G.graph)] += v
        x,y = zip(*energies.items())
        plt.bar(x, y)
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.title('Energy histogram for MaxCut\nSimulator Type: {}\nGraph: {}'.format(self._simulator.type.name, G.name))

    @staticmethod
    def plot_relative_rate(reference_qaoa_analysis, *comparison_qaoa_analyses):
        """
        Plot relative rate vs. number of layers and name for different QAOAAnalysis instances.

        Parameters:
        - reference_qaoa_analysis (QAOAAnalysis): The reference QAOAAnalysis instance for calculating relative rates.
        - *comparison_qaoa_analyses (QAOAAnalysis): Variable number of QAOAAnalysis instances to compare with the reference.

        Displays:
        Plots of relative rate vs. number of layers and name.
        """
        # Create dictionaries to store the data
        relative_rate_layers_data = {}
        reference_best_solutions = {}

        for graph_obj in reference_qaoa_analysis.graph_objs:
            reference_best_solutions[(graph_obj.short_name, graph_obj.layers)] = reference_qaoa_analysis._best_solutions[graph_obj.name][0]

        # Loop over the given QAOAAnalysis instances except the reference (all comparsion between same graph instances)
        for qaoa_analysis in comparison_qaoa_analyses:
            if qaoa_analysis.name == reference_qaoa_analysis.name: # #Shots
                comparison_text = f'{qaoa_analysis.simulator.shot_amt}/{reference_qaoa_analysis.simulator.shot_amt} #Shots'
            elif qaoa_analysis.noise_multiplier == None: # Simulator types
                comparison_text = f'{qaoa_analysis.short_name}/{reference_qaoa_analysis.short_name}'
            else: # Noise level
                comparison_text = f'{qaoa_analysis.noise_multiplier}/{reference_qaoa_analysis.noise_multiplier} xNoise'
            # Create dictionaries to store the data
            relative_rate_layers_data = {}

            for graph_obj in qaoa_analysis.graph_objs:
                name = graph_obj.short_name
                layers = graph_obj.layers
                graph_type = graph_obj.graph_type

                if name not in relative_rate_layers_data:
                    relative_rate_layers_data[name] = {}

                relative_rate = qaoa_analysis._best_solutions[graph_obj.name][0] / reference_best_solutions[(name, layers)]
                relative_rate_layers_data[name][layers] = relative_rate

            line_styles = ['-', '--', '-.', ':']
            
            # Extract unique layers and names from graph_objs
            unique_layers = sorted(set(graph_obj.layers for graph_obj in qaoa_analysis.graph_objs))
            unique_names = sorted(set(graph_obj.short_name for graph_obj in qaoa_analysis.graph_objs))

            # Plot Relative Rate vs. Number of Layers and name
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, name in enumerate(unique_names):
                relative_rate_list = list(dict(sorted(relative_rate_layers_data[name].items())).values())
                linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
                
                # Pad or truncate relative_rate_list to match the length of unique_layers
                relative_rate_list_padded = relative_rate_list + [None] * (len(unique_layers) - len(relative_rate_list))
                
                ax.plot(unique_layers, relative_rate_list_padded, linestyle=linestyle, label=name)
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Relative Rate')
            ax.set_title(f'Relative Rate vs. Number of Layers\n{comparison_text}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()
