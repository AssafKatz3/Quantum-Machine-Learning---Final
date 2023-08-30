import multiprocessing

# Function to run optimal parameters for a QAOA analysis instance
def run_optimal_params(analysis):
    """
    Executes the optimal parameter run for a given QAOA analysis instance.

    Parameters:
        analysis (QAOAAnalysis): An instance of the QAOA analysis containing simulation and optimization parameters.

    Returns:
        Tuple (shot_amt, opt_res_counts): The number of shots used for simulation and the results of the optimal parameter run.
    """
    shot_amt = analysis.simulator.shot_amt
    opt_res_counts = analysis.run_optimal_params()
    return shot_amt, opt_res_counts

# Function to process a list of QAOA analysis instances in parallel
def process_analysis_list(analysis_list):
    """
    Executes the optimal parameter runs for a list of QAOA analysis instances in parallel using multiprocessing.

    Parameters:
        analysis_list (List[QAOAAnalysis]): A list of QAOA analysis instances to be processed in parallel.

    Returns:
        opt_res_counts_dict_ns (dict): A dictionary containing the simulation results of optimal parameter runs for different shot amounts.
    """
    opt_res_counts_dict_ns = {}  # Create a dictionary to store results for this analysis_list
    for qaoa_analysis in analysis_list:
        opt_res_counts_dict_ns[qaoa_analysis.simulator.shot_amt] = run_optimal_params(qaoa_analysis)
    return opt_res_counts_dict_ns
