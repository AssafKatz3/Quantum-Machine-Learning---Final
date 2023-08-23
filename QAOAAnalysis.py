import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate
from collections import defaultdict
import matplotlib
from GraphData import GraphType

class QAOAAnalysis:
    """
    Class to perform QAOA analysis and visualization.

    Parameters:
    graph_objs (list): List of graph objects for analysis.
    simulator: QAOASimulation object for simulation and analysis.
    cut_values (dict): Dictionary to store cut values for graph objects.
    """

    def __init__(self, graph_objs, simulator, cut_values):
        self.graph_objs = graph_objs
        self.simulator = simulator
        self.cut_values = cut_values
        self.best_solutions = {}

    def run_optimal_params(self):
        """
        Run optimal parameter simulation for each graph and store results.

        Returns:
        dict: Dictionary containing optimal results counts for each graph object.
        """
        opt_res_counts_dict = {}  # Create a dictionary to store results

        for graph_obj in self.graph_objs:
            opt_res = self.simulator.get_opt_params(graph_obj, graph_obj.layers)
            opt_res_counts = self.simulator.run_circuit_optimal_params(opt_res, graph_obj, graph_obj.layers)
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
        for graph_obj in self.graph_objs:
            opt_res_counts = opt_res_counts_dict[graph_obj.name]
            best_cut, best_solution = self.simulator.best_solution(graph_obj, opt_res_counts)
            self.best_solutions[graph_obj.name] = (best_cut, best_solution)

        # Prepare data for the table
        table_data = []
        for graph_obj_name, (best_cut, best_solution) in self.best_solutions.items():
            table_data.append([graph_obj_name, -best_cut, best_solution])

        # Define table headers
        headers = ["Graph Name", "Best Cut", "Best Solution"]

        # Display table using tabulate
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        print(table)

    def plot_graphs(self):
        """
        Plot graphs with colored nodes and edges based on best solutions.

        Displays:
        Plots of graphs with colored nodes and edges indicating the best solutions.
        """
        # Loop over graph_objs
        for graph_obj in self.graph_objs:
            # Get the best solution for the current graph_obj
            best_solution = self.best_solutions[graph_obj.name][1]

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
            plt.title(f'Graph: {graph_obj.name}')
            plt.show()

    def plot_approximate_rate(self):
        """
        Plot approximate rate vs. number of layers and probability.

        Displays:
        Plots of approximate rate vs. number of layers and probability for each graph type.
        """
        # Separate the data for plotting
        approximately_rate_layers_data = {}  # Dictionary to store approximately_rate data per probability
        approximately_rate_prob_data = {}  # Dictionary to store approximately_rate data per layer
        for graph_obj in self.graph_objs:
            prob = graph_obj.probability
            layers = graph_obj.layers
            graph_type = graph_obj.graph_type  # Add graph_type to GraphData class
            if prob not in approximately_rate_layers_data:
                approximately_rate_layers_data[prob] = {}
            if layers not in approximately_rate_prob_data:
                approximately_rate_prob_data[layers] = {}
            if graph_type not in approximately_rate_layers_data[prob]:
                approximately_rate_layers_data[prob][graph_type] = []
            if graph_type not in approximately_rate_prob_data[layers]:
                approximately_rate_prob_data[layers][graph_type] = []
            approximately_rate = -self.best_solutions[graph_obj.name][0] / self.cut_values[graph_obj.name]
            approximately_rate_layers_data[prob][graph_type].append(approximately_rate)
            approximately_rate_prob_data[layers][graph_type].append(approximately_rate)

        # Extract unique layers and probabilities from graph_objs
        unique_layers = sorted(set(graph_obj.layers for graph_obj in self.graph_objs))
        unique_probs = sorted(set(graph_obj.probability for graph_obj in self.graph_objs))
        
        line_styles = ['-', '--', '-.', ':']
        all_graph_types = list(GraphType)  # List of all enum values

        # Plot Approximate Rate vs. Number of Layers for each graph type
        for graph_type in all_graph_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, prob in enumerate(unique_probs):
                approximately_rate_list = approximately_rate_layers_data[prob].get(graph_type, [])
                linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
                
                # Pad or truncate approximately_rate_list to match the length of unique_layers
                approximately_rate_list_padded = approximately_rate_list + [None] * (len(unique_layers) - len(approximately_rate_list))
                ax.plot(unique_layers, approximately_rate_list_padded, linestyle=linestyle, label=f'{prob:.1%}')
                
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Approximate Rate (Optimal Counts / Cut Values)')
            ax.set_title(f'Approximate Rate vs. Number of Layers - Graph Type: {graph_type.name}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        # Plot Approximate Rate vs. Probability for each graph type
        for graph_type in all_graph_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, layers in enumerate(unique_layers):
                approximately_rate_list = approximately_rate_prob_data[layers].get(graph_type, [])
                linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
                
                # Pad or truncate approximately_rate_list to match the length of unique_probs
                approximately_rate_list_padded = approximately_rate_list + [None] * (len(unique_probs) - len(approximately_rate_list))
                
                ax.plot(unique_probs, approximately_rate_list_padded, linestyle=linestyle, label=f'{layers} Layers')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Approximate Rate (Optimal Counts / Cut Values)')
            ax.set_title(f'Approximate Rate vs. Probability - Graph Type: {graph_type.name}')
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
            energies[self.simulator._maxcut_obj(k, G.graph)] += v
        x,y = zip(*energies.items())
        plt.bar(x, y)
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.title('Energy histogram for MaxCut\nSimulator Type: {}\nGraph: {}'.format(self.simulator.type.name, G.name))

    @staticmethod
    def plot_relative_rate(reference_qaoa_analysis, *comparison_qaoa_analyses):
        """
        Plot relative rate vs. number of layers and probability for different QAOAAnalysis instances.

        Parameters:
        - reference_qaoa_analysis (QAOAAnalysis): The reference QAOAAnalysis instance for calculating relative rates.
        - *comparison_qaoa_analyses (QAOAAnalysis): Variable number of QAOAAnalysis instances to compare with the reference.

        Displays:
        Plots of relative rate vs. number of layers and probability for each graph type.
        """
        reference_name = reference_qaoa_analysis.simulator.type.name.replace("_", " ").title()

        # Create dictionaries to store the data
        relative_rate_layers_data = {}
        relative_rate_prob_data = {}
        reference_best_solutions = {}

        for graph_obj in reference_qaoa_analysis.graph_objs:
            reference_best_solutions[(graph_obj.probability, graph_obj.layers)] = reference_qaoa_analysis.best_solutions[graph_obj.name][0]

        # Loop over the given QAOAAnalysis instances except the reference
        for qaoa_analysis in comparison_qaoa_analyses:
            comparison_name = qaoa_analysis.simulator.type.name.replace("_", " ").title()
            # Create dictionaries to store the data
            relative_rate_layers_data = {}
            relative_rate_prob_data = {}

            for graph_obj in qaoa_analysis.graph_objs:
                prob = graph_obj.probability
                layers = graph_obj.layers
                graph_type = graph_obj.graph_type

                if prob not in relative_rate_layers_data:
                    relative_rate_layers_data[prob] = {}
                if layers not in relative_rate_prob_data:
                    relative_rate_prob_data[layers] = {}
                if graph_type not in relative_rate_layers_data[prob]:
                    relative_rate_layers_data[prob][graph_type] = []
                if graph_type not in relative_rate_prob_data[layers]:
                    relative_rate_prob_data[layers][graph_type] = []

                relative_rate = qaoa_analysis.best_solutions[graph_obj.name][0] / reference_best_solutions[(prob, layers)]
                relative_rate_layers_data[prob][graph_type].append(relative_rate)
                relative_rate_prob_data[layers][graph_type].append(relative_rate)

            line_styles = ['-', '--', '-.', ':']
            all_graph_types = list(GraphType)  # List of all enum values
            
            # Extract unique layers and probabilities from graph_objs
            unique_layers = sorted(set(graph_obj.layers for graph_obj in qaoa_analysis.graph_objs))
            unique_probs = sorted(set(graph_obj.probability for graph_obj in qaoa_analysis.graph_objs))

            # Plot Relative Rate vs. Number of Layers and Probability for each graph type
            for graph_type in all_graph_types:
                # Plot Relative Rate vs. Probability
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, layers in enumerate(unique_layers):
                    relative_rate_list = relative_rate_prob_data[layers].get(graph_type, [])
                    linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
                    
                    # Pad or truncate relative_rate_list to match the length of unique_probs
                    relative_rate_list_padded = relative_rate_list + [None] * (len(unique_probs) - len(relative_rate_list))
                    
                    ax.plot(unique_probs, relative_rate_list_padded, linestyle=linestyle, label=f'{layers} Layers')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Relative Rate')
                ax.set_title(f'Relative Rate vs. Probability - {comparison_name}/{reference_name} - Graph Type: {graph_type.name}')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()

                # Plot Relative Rate vs. Layers
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, prob in enumerate(unique_probs):
                    relative_rate_list = relative_rate_layers_data[prob].get(graph_type, [])
                    linestyle = line_styles[i % len(line_styles)]  # Cycle through line styles
                    
                    # Pad or truncate relative_rate_list to match the length of unique_layers
                    relative_rate_list_padded = relative_rate_list + [None] * (len(unique_layers) - len(relative_rate_list))
                    
                    ax.plot(unique_layers, relative_rate_list_padded, linestyle=linestyle, label=f'{prob:.1%}')
                ax.set_xlabel('Number of Layers')
                ax.set_ylabel('Relative Rate')
                ax.set_title(f'Relative Rate vs. Number of Layers - {comparison_name}/{reference_name} - Graph Type: {graph_type.name}')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()
