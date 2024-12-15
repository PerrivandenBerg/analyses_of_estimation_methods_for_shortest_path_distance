import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd
import math
import random
import sys
from typing import Dict, List, Tuple

# Distances --------------------------------------------------------------------

## Gets all the shortest paths from all points to all landparks.
def all_landmark_distances(G: nx.Graph, landmarks: List[int]
                           ) -> Dict[int, Dict[int, float]]:
    distances = {}
    for l in landmarks:
        distances[l] = nx.single_source_dijkstra_path_length(G, l)
    return distances

# Closeness Centrallity Estimation ---------------------------------------------

## Approximate closeness centrality for nodes in a large network using sampling.
def approximate_closeness_centrality(G: nx.Graph, n=100) -> List[Tuple]:
   # Randomly sample seed nodes
    nodes = list(G.nodes())
    seed_nodes = random.sample(nodes, min(n, len(nodes)))

    # Initialize distance sum for each node
    dist_sum = {node: 0 for node in nodes}
    reach = {node: 0 for node in nodes}

    for s in seed_nodes:
        # Perform BFS/SSSP from each seed
        shortest_paths = nx.single_source_shortest_path_length(G, s)
        
        for t, dist in shortest_paths.items():
            dist_sum[t] += dist
            reach[t] += 1

    # Compute cc for each node.
    cc_approx = []
    for node in nodes:
        if reach[node] > 0:  # Avoid division by zero
            avg_dist = dist_sum[node] / reach[node]
            cc_approx.append((node, (reach[node] / len(nodes)) / avg_dist))
        else:
            cc_approx.append((node, 0))

    return cc_approx



# Estimation Methods -----------------------------------------------------------

def estimate_bound(U: int, L: int, type: int) -> float:
    if type == 1:
        return estimate_bound_1(U, L)
    if type == 2:
        return estimate_bound_2(U, L)
    if type == 3:
        return estimate_bound_3(U, L)
    if type == 4:
        return estimate_bound_4(U, L)
    return 0.0

def estimate_bound_1(U: int, L: int) -> float:
    '''Returns U.'''
    return U

def estimate_bound_2(U: int, L: int) -> float:
    '''Returns L.'''
    return L

def estimate_bound_3(U: int, L: int) -> float:
    '''Returns average.'''
    return (U + L)/2

def estimate_bound_4(U: int, L: int) -> float:
    '''Returns square root of U and L.'''
    return math.sqrt(U+L)

def estimate_bound_5(U: int, L: int, alpha: float) -> float:
    '''Returns weighted average.
    alpha: weight of U, with range [0, 1].'''
    return U*alpha + L*(1-alpha)

# Selection Methods ------------------------------------------------------------

## Random
def rand_landmarks(G: nx.Graph, k: int) -> List[int]:
    return random.sample(list(G.nodes), k)

## Degree-based
def deg_landmarks(G: nx.Graph, k: int) -> List[int]:
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:k]]

## Closeness Centrality-based (select lowest centrality)
def cc_landmarks(G: nx.Graph, k: int, nodes: List[Tuple]) -> List[int]:
    sorted_nodes = sorted(nodes, key=lambda x: x[1])
    return [node for node, _ in sorted_nodes[:k]]

## Degree-based where landmarks cannot be neighbors.
def deg_1_landmarks(G: nx.Graph, k: int) -> List[int]:
    p = 1  # Best value according to the paper.
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    selected = []
    for node, _ in sorted_nodes:
        if len(selected) >= k:
            break
        if all(nx.shortest_path_length(G, node, other) > p for other in selected):
            selected.append(node)
    return selected

## Closeness Centrality-based where landmarks cannot be neighbors.
## Note: Doesn't strictly return k nodes.
def cc_1_landmarks(G: nx.Graph, k: int, nodes: List[Tuple]) -> List[int]:
    p = 1  # Best value according to the paper.
    sorted_nodes = sorted(nodes, key=lambda x: x[1])
    selected = []
    for node, _ in sorted_nodes:
        if len(selected) >= k:
            break
        if all(nx.shortest_path_length(G, node, other) > p for other in selected):
            selected.append(node)
    return selected


## P/Border-based: P}ick nodes close to the border of each partition (louvain).
def border_landmarks(G: nx.Graph, k: int) -> List[int]:

    # Detect partitions
    comm = list(nx.algorithms.community.greedy_modularity_communities(G))
    pars = {i: list(comm) for i, comm in enumerate(comm)} # Partitions

    # Compute border scores using the partitions and the b(u) formula.
    border_list = {}
    for node in G.nodes:

        # Determine the partition of the node.
        par = next((p for p, nodes in pars.items() if node in nodes), None)
        if par is None:
            continue

        # Compute b(u) = \sum{ i ∈ P, u ∈ p, i != p }{ d_i(u) · d_p(u) }.
        b_u = sum(
            G.degree[other] * G.degree[node]
            for other_partition, other_nodes in pars.items()
            if other_partition != par
            for other in other_nodes
        )
        border_list[node] = b_u
    
    # Select top-k nodes.
    sorted_nodes = sorted(border_list.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:k]]



# UPPER / LOWER BOUNDS ---------------------------------------------------------

# Function to calculate lower and upper bounds based on landmark distances.
def get_bounds(s: int, t: int, marks: List[int]
               , mark_dists: Dict[int, Dict[int, float]]
               ) -> Tuple[float, float]:
    '''Returns two ints, the lower and upper bounds.'''
    lower = sys.float_info.min
    upper = sys.float_info.max
    
    for l in marks:
        d_s_l = mark_dists[l].get(s, sys.float_info.max)
        d_t_l = mark_dists[l].get(t, sys.float_info.max)
        
        lower = max(lower, abs(d_s_l - d_t_l))
        upper = min(upper, d_s_l + d_t_l)

    return lower, upper

# CODE -------------------------------------------------------------------------

name = "facebook"  # Name of dataset
threshold = 44  # Degree threshold for binning
n = 10000  # Number of experiments.

# Import dataset

print(f"DEBUG: Starting...")

df = pd.read_csv(f'{name}.csv', sep=' ', header=None, names=['node1', 'node2'])

G_all = nx.Graph()
for _, row in df.iterrows():
    n1 = row['node1']
    n2 = row['node2']
    G_all.add_edge(n1, n2, weight=1)

G_largest = max(nx.connected_components(G_all), key=len)
G = G_all.subgraph(G_largest)

# Important info about graph:
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

degree_sequence = list(dict(G.degree()).values())
print(f"Degree Statistics:")
print(f"  - Minimum degree: {min(degree_sequence)}")
print(f"  - Maximum degree: {max(degree_sequence)}")
print(f"  - Average degree: {np.mean(degree_sequence):.2f}")

# Sparsity
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
max_possible_edges = num_nodes * (num_nodes - 1) / 2
sparsity = (2 * num_edges) / max_possible_edges
print(f"Sparsity of the graph: {sparsity:.4e}")




# Number of landmarks in the graph.
for marks_n in [20, 100]:

    print(f"DEBUG [{marks_n}]: Now starting marks_n.")

    # Approximate Closeness Centrallity for each node
    cc_nodes = approximate_closeness_centrality(G, 1000)

    # Selection
    marks_rand  = rand_landmarks(G, marks_n)
    marks_deg   = deg_landmarks(G, marks_n)
    marks_cc    = cc_landmarks(G, marks_n, cc_nodes)
    marks_deg_1 = deg_1_landmarks(G, marks_n)
    marks_cc_1  = cc_1_landmarks(G, marks_n, cc_nodes)

    # Store distances of landmarks and points.
    marks_dis_rand  = all_landmark_distances(G, marks_rand)
    marks_dis_deg   = all_landmark_distances(G, marks_deg)
    marks_dis_cc    = all_landmark_distances(G, marks_cc)
    marks_dis_deg_1 = all_landmark_distances(G, marks_deg_1)
    marks_dis_cc_1  = all_landmark_distances(G, marks_cc_1)

    print(f"DEBUG [{marks_n}]: Landmarks calculated.")

    # FROM HERE ON IT SHOULD BE VERY FAST! -----------------------------------------

    # Storage of the data.
    methods_list = ['rand', 'deg', 'cc', 'deg_1', 'cc_1']

    # Error
    error_dict = {
        'rand': {1: [], 2: [], 3: [], 4: []},
        'deg': {1: [], 2: [], 3: [], 4: []},
        'cc': {1: [], 2: [], 3: [], 4: []},
        'deg_1': {1: [], 2: [], 3: [], 4: []},
        'cc_1': {1: [], 2: [], 3: [], 4: []},
    }

    # The degree-based Lower / Upperbound estimation.
    deg_LU_est = {
        'rand': {1: [], 2: [], 3: [], 4: []},
        'deg': {1: [], 2: [], 3: [], 4: []},
        'cc': {1: [], 2: [], 3: [], 4: []},
        'deg_1': {1: [], 2: [], 3: [], 4: []},
        'cc_1': {1: [], 2: [], 3: [], 4: []},
    }

    # Perform distance estimation experiments.
    for i in range(n):
        s, t = random.choice(list(G.nodes)), random.choice(list(G.nodes))
        deg_s, deg_t = G.degree[s], G.degree[t]

        if nx.has_path(G, s, t):
            # Compute bounds for different landmark selection methods.
            lb_rand, ub_rand = get_bounds(s, t, marks_rand, marks_dis_rand)
            lb_deg, ub_deg = get_bounds(s, t, marks_deg, marks_dis_deg)
            lb_cc, ub_cc = get_bounds(s, t, marks_cc, marks_dis_cc)
            lb_deg_1, ub_deg_1 = get_bounds(s, t, marks_deg_1, marks_dis_deg_1)
            lb_cc_1, ub_cc_1 = get_bounds(s, t, marks_cc_1, marks_dis_cc_1)
            
            actual_result = nx.shortest_path_length(G, s, t)

            # Go over all combinations.
            for method, (lb, ub) in zip(methods_list, 
                            [(lb_rand, ub_rand), (lb_deg, ub_deg), (lb_cc, ub_cc),
                            (lb_deg_1, ub_deg_1), (lb_cc_1, ub_cc_1)]):
                for est_method in range(1, 5):
                    estimated = estimate_bound(ub, lb, est_method)

                    # Part 1: Store the errors for each method and estimation.
                    error_dict[method][est_method].append(1/n * (estimated - actual_result))

                    # Part 2: Degree based nodes.
                    deg_LU_est[method][est_method].append((deg_s, deg_t, actual_result, estimated))

    # Preparing Data for Plotting
    boxplot_data = []
    positions = []
    labels = []
    methods = ['Random', 'Degree', 'Closeness Centrality', "Degree (1)", 
            "Closeness Centrality (1)"]
    estimations = ['Upperbound', 'Lowerbound', 'Average', 'Square Root']

    width = 1.2  # Group width
    offset = 0.25  # Distance between boxplots within each group
    group_spacing = 1.5  # Distance between groups

    print(f"DEBUG [{marks_n}]: Part 1.")

    # PART 1 -----------------------------------------------------------------------

    # Collect data for boxplots grouped by method and estimation method
    for i, method in enumerate(methods_list):
        method_position = i * group_spacing  # Start of each group
        for j in range(4):  # Four estimation methods
            est_data = error_dict[method][j + 1]  # Error data for the estimation method
            boxplot_data.append(est_data)
            positions.append(method_position + j * offset)
        labels.append(methods[i])  # Add method label

    # Plot Setup
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create Boxplots
    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=offset / 1.5, showmeans=True, meanline=True)

    # Use distinct colors for each estimation method
    palette = sns.color_palette("pastel", 4)  # Use pastel shades for consistency
    for patch, color in zip(bp['boxes'], palette * len(methods_list)):
        patch.set_facecolor(color)

    # Overlay the mean values as markers
    for pos, data in zip(positions, boxplot_data):
        mean_value = np.mean(data)
        ax.scatter(pos, mean_value, color='black', marker='o', label='Mean', s=30, zorder=3)

    # Add legend for estimation methods
    legend_handles = [mpatches.Patch(color=palette[idx], label=estimations[idx]) for idx in range(4)]
    ax.legend(handles=legend_handles, title="Estimation Methods", loc="upper right", fontsize=12)

    # Add labels and titles
    ax.set_xticks([i * group_spacing + 1.5 * offset for i in range(len(methods_list))])
    ax.set_xticklabels(labels, fontsize=11, ha='right')
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Error Comparison Across Graph Selection Methods', fontsize=16)

    # Enhance grid and layout
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    # Save and Display Plot
    plt.savefig(f'error_{name}_{marks_n}.eps', format='eps')


    # PART 2 -----------------------------------------------------------------------

    print(f"DEBUG [{marks_n}]: Part 2.")

    for est in range(1, 5):

        categories = ["L-L", "L-H", "H-L", "H-H"]

        # Prepare data for bar plot
        grouped_data = {method: {cat: [] for cat in categories} for method in methods_list}

        for method in methods_list:
            for deg_s, deg_t, actual, estimated in deg_LU_est[method][est]:
                error = abs(estimated - actual)
                if deg_s < threshold and deg_t < threshold:
                    grouped_data[method]["L-L"].append(error)
                elif deg_s < threshold and deg_t >= threshold:
                    grouped_data[method]["L-H"].append(error)
                elif deg_s >= threshold and deg_t < threshold:
                    grouped_data[method]["H-L"].append(error)
                elif deg_s >= threshold and deg_t >= threshold:
                    grouped_data[method]["H-H"].append(error)

        # Compute means for the grouped bar plot
        means = {method: [np.mean(grouped_data[method][cat]) if grouped_data[method][cat] else 0
                        for cat in categories] for method in methods_list}

        # Create grouped bar plot
        x = np.arange(len(categories))  # Category positions
        width = 0.15  # Width of each bar

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, method in enumerate(methods_list):
            ax.bar(x + i * width, means[method], width, label=methods[i])

        # Formatting
        ax.set_title(f"{estimations[est-1]} Errors Based on Degree", fontsize=16)
        ax.set_xticks(x + width * (len(methods_list) - 1) / 2)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylabel('Absolute Mean Error', fontsize=12)
        ax.legend(title="Methods", fontsize=10)
        plt.tight_layout()

        # Save and display
        plt.savefig(f'degree_error_{name}_{marks_n}_{est}.eps', format='eps')


    print(f"DEBUG [{marks_n}]: Completed.")

print(f"DEBUG: Completed.")
