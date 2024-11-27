import networkx as nx
import matplotlib.pyplot as plt
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
G = nx.erdos_renyi_graph(100, 0.1, seed=42, directed=False)

# Number of landmarks in the graph.
marks_n = 5 

# Selection
# marks_rand = rand_landmarks(G, marks_n)
# marks_deg  = deg_landmarks(G, marks_n)
marks_cc   = cc_p_landmarks(G, marks_n, 1)

# Store distances of landmarks and points.
# marks_dis_rand = all_landmark_distances(G, marks_rand)
# marks_dis_deg  = all_landmark_distances(G, marks_deg)
marks_dis_cc   = all_landmark_distances(G, marks_cc)


# FROM HERE ON IT SHOULD BE VERY FAST! -----------------------------------------


# We do a lot of experiments and see how the degree of a note pair influences 
# the estimation method that we could best use for the estimating the distance
# between a node pair.
n = 1000  # Number of experiments.

print("Calculations go brrrr")

# Collect the data.
results = {method: [] for method in range(1, 5)}
for _ in range(n):
    # Randomly select two nodes
    s, t = random.sample(G.nodes, 2)
    deg_s, deg_t = G.degree[s], G.degree[t]
    
    # Random landmark selection
    # lb_rand, ub_rand = get_bounds(s, t, marks_rand, marks_dis_rand)
    # lb_deg, ub_deg = get_bounds(s, t, marks_deg, marks_dis_deg)
    lb, ub = get_bounds(s, t, marks_cc, marks_dis_cc)

    actual_result = nx.dijkstra_path_length(G, s, t)

    # Store results
    for est_method in range(1, 5):
        estimated = estimate_bound(lb, ub, est_method)
        results[est_method].append((deg_s, deg_t, actual_result, estimated))

print("We do be plotting")

# Plotting
deg_thres = 10  # Degree threshold for binning
methods = ['Upperbound', 'Lowerbound', 'Average', 'Square Root']
file = ['upper', 'lower', 'avg', 'sqrt']

# Prepare a 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten for easier iteration over axes

for method, data in results.items():
    degree_bins = {"high-high": [], "high-low": [], "low-high": [], "low-low": []}
    
    for deg1, deg2, actual, estimate in data:
        bin_label = (
            "high-high" if deg1 > deg_thres and deg2 > deg_thres else
            "high-low" if deg1 > deg_thres and deg2 <= deg_thres else
            "low-high" if deg1 <= deg_thres and deg2 > deg_thres else
            "low-low"
        )
        degree_bins[bin_label].append(abs(actual - estimate))  # Absolute error
    
    # Average errors for each bin
    avg_errors = {label: np.mean(errors) if errors else 0 for label, errors in degree_bins.items()}
    
    # Plot on the corresponding subplot
    ax = axes[method - 1]
    ax.bar(avg_errors.keys(), avg_errors.values())
    ax.set_title(f"Performance of {methods[method-1]}")
    ax.set_ylabel("Average Error")
    ax.set_xlabel("Degree Combination")

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('performance_comparison.eps', format='eps')
plt.show()
