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

## Closeness Centrality-based
def cc_landmarks(G: nx.Graph, k: int) -> List[int]:
    centrality = nx.closeness_centrality(G) # TODO: Maybe something faster?
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:k]]

## Degree-based where we have landmarks be at least 'p' nodes apart.
def deg_p_landmarks(G: nx.Graph, k: int, p: int) -> List[int]:
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    selected = []
    for node, _ in sorted_nodes:
        if len(selected) >= k:
            break
        if all(nx.shortest_path_length(G, node, other) > p for other in selected):
            selected.append(node)
    return selected

## Closeness Centrality-based where we have landmarks be at least 'p' nodes apart.
## Note: Doesn't strictly return k nodes.
def cc_p_landmarks(G: nx.Graph, k: int, p: int) -> List[int]:
    centrality = nx.closeness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
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

# TODO: Find good datasets for our research.
# # Use a network stored as CSV and import it.
# df = pd.read_csv('file.csv', sep=',', header=None, names=['node1', 'node2', 
#                  'weight'])
# 
# G = nx.DiGraph()
# for _, row in df.iterrows():
#     n1 = row['node1']
#     n2 = row['node2']
#     G.add_edge(n1, n2, weight=1)

# TODO: Import the real-world graph, random for now.
G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=False)

# Number of landmarks in the graph.
marks_n = 5

# Selection
marks_rand  = rand_landmarks(G, marks_n)
marks_deg   = deg_landmarks(G, marks_n)
marks_cc    = cc_landmarks(G, marks_n)
marks_deg_p = deg_p_landmarks(G, marks_n, 1)
marks_cc_p  = cc_p_landmarks(G, marks_n, 1)
marks_bord  = border_landmarks(G, marks_n)

# Store distances of landmarks and points.
marks_dis_rand  = all_landmark_distances(G, marks_rand)
marks_dis_deg   = all_landmark_distances(G, marks_deg)
marks_dis_cc    = all_landmark_distances(G, marks_cc)
marks_dis_deg_p = all_landmark_distances(G, marks_deg_p)
marks_dis_cc_p  = all_landmark_distances(G, marks_cc_p)
marks_dis_bord  = all_landmark_distances(G, marks_bord)


# FROM HERE ON IT SHOULD BE VERY FAST! -----------------------------------------


# Do a lot of experiments of distance estimation between 2 points and average
# the error margine compared to the actual answer.
n = 1000  # Number of experiments.

# Storage of the data.
methods_list = ['rand', 'deg', 'cc', 'deg_p', 'cc_p', 'bord']
error_dict = {
    'rand': {1: [], 2: [], 3: [], 4: []},
    'deg': {1: [], 2: [], 3: [], 4: []},
    'cc': {1: [], 2: [], 3: [], 4: []},
    'deg_p': {1: [], 2: [], 3: [], 4: []},
    'cc_p': {1: [], 2: [], 3: [], 4: []},
    'bord': {1: [], 2: [], 3: [], 4: []},
}

# Perform distance estimation experiments.
for i in range(n):
    s, t = random.choice(list(G.nodes)), random.choice(list(G.nodes))

    if nx.has_path(G, s, t):
        # Compute bounds for different landmark selection methods.
        lb_rand, ub_rand = get_bounds(s, t, marks_rand, marks_dis_rand)
        lb_deg, ub_deg = get_bounds(s, t, marks_deg, marks_dis_deg)
        lb_cc, ub_cc = get_bounds(s, t, marks_cc, marks_dis_cc)
        lb_deg_p, ub_deg_p = get_bounds(s, t, marks_deg_p, marks_dis_deg_p)
        lb_cc_p, ub_cc_p = get_bounds(s, t, marks_cc_p, marks_dis_cc_p)
        lb_bord, ub_bord = get_bounds(s, t, marks_bord, marks_dis_bord)
        
        actual_result = nx.dijkstra_path_length(G, s, t)

        # Store the errors for each method and estimation.
        for method, (lb, ub) in zip(methods_list, 
                        [(lb_rand, ub_rand), (lb_deg, ub_deg), (lb_cc, ub_cc),
                         (lb_deg_p, ub_deg_p), (lb_cc_p, ub_cc_p), (lb_bord, ub_bord)]):
            for est_method in range(1, 5):
                estimated = estimate_bound(lb, ub, est_method)
                error_dict[method][est_method].append(estimated - actual_result)

# Calculate the mean and std.
means = {}
stds = {}
for method in methods_list:
    means[method] = [np.mean(error_dict[method][i]) for i in range(1, 5)]
    stds[method] = [np.std(error_dict[method][i]) for i in range(1, 5)]

# Plotting
methods = ['Random', 'Degree', 'Closeness Centrality', "Degree (P)", 
           "Closeness Centrality (P)", "Border"]
estimations = ['Upperbound', 'Lowerbound', 'Average', 'Square Root']
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

fig, ax = plt.subplots(figsize=(14, 8))

# Collect data for boxplots
boxplot_data = []
positions = []
labels = []

width = 0.7  # Width of each group of boxplots
offset = 0.15  # Spacing between boxplots within a group

# Build grouped boxplots for each method
for i, method in enumerate(methods_list):
    method_position = i * (width + offset)  # Starting position for the group
    for j in range(4):  # Loop over estimation methods
        est_data = error_dict[method][j + 1]  # Estimation method data
        boxplot_data.append(est_data)
        positions.append(method_position + j * offset)
    labels.append(methods[i])  # Add group label (once per method)

# Create the boxplots
bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=offset / 1.5, notch=True)

# Color the boxplots based on the estimation methods
for patch, color in zip(bp['boxes'], colors * len(methods_list)):
    patch.set_facecolor(color)

# Add a legend for the colors
legend_handles = [mpatches.Patch(color=color, label=estimations[idx]) for idx, color in enumerate(colors)]
ax.legend(handles=legend_handles, title="Estimation Methods", loc="upper right", fontsize=10)

# Add labels and title
ax.set_ylabel('Absolute Error', fontsize=12)
ax.set_xlabel('Graph Selection Methods', fontsize=12)
ax.set_title('Boxplot of Estimation Errors for Various Methods', fontsize=14)
ax.set_xticks([i * (width + offset) + offset for i in range(len(methods_list))])
ax.set_xticklabels(labels, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('error_boxplot.eps', format='eps')
plt.show()