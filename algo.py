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
marks_rand = rand_landmarks(G, marks_n)
marks_deg  = deg_landmarks(G, marks_n)
marks_cc   = cc_landmarks(G, marks_n)

# Store distances of landmarks and points.
marks_dis_rand = all_landmark_distances(G, marks_rand)
marks_dis_deg  = all_landmark_distances(G, marks_deg)
marks_dis_cc   = all_landmark_distances(G, marks_cc)


# FROM HERE ON IT SHOULD BE VERY FAST! -----------------------------------------


# Do a lot of experiments of distance estimation between 2 points and average
# the error margine compared to the actual answer.
n = 1000  # Number of experiments.

# Storage of the data.
error_dict = {
    'rand': {1: [], 2: [], 3: [], 4: []},
    'deg': {1: [], 2: [], 3: [], 4: []},
    'cc': {1: [], 2: [], 3: [], 4: []}
}

# Perform distance estimation experiments.
for i in range(n):
    s, t = random.choice(list(G.nodes)), random.choice(list(G.nodes))

    if nx.has_path(G, s, t):
        # Compute bounds for different landmark selection methods.
        lb_rand, ub_rand = get_bounds(s, t, marks_rand, marks_dis_rand)
        lb_deg, ub_deg = get_bounds(s, t, marks_deg, marks_dis_deg)
        lb_cc, ub_cc = get_bounds(s, t, marks_cc, marks_dis_cc)
        
        actual_result = nx.dijkstra_path_length(G, s, t)

        # Store the errors for each method and estimation.
        for method, (lb, ub) in zip(['rand', 'deg', 'cc'], 
                        [(lb_rand, ub_rand), (lb_deg, ub_deg), (lb_cc, ub_cc)]):
            for est_method in range(1, 5):
                estimated = estimate_bound(lb, ub, est_method)
                error_dict[method][est_method].append(estimated - actual_result)

# Calculate the mean and std.
means = {}
stds = {}
for method in ['rand', 'deg', 'cc']:
    means[method] = [np.mean(error_dict[method][i]) for i in range(1, 5)]
    stds[method] = [np.std(error_dict[method][i]) for i in range(1, 5)]

# Plotting
methods = ['Random', 'Degree', 'Closeness Centrality']
estimations = ['Upperbound', 'Lowerbound', 'Average', 'Square Root']
bar_width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

# Set the bar positions for grouped bars.
x = np.arange(len(methods))  # x locations, one for each method.
offsets = np.array([i * bar_width for i in range(4)])

# Plot each estimation method for each graph selection method.
for i in range(4):
    ax.bar(x + offsets[i], [means[method][i] for method in ['rand', 'deg', 'cc']],
           yerr=[stds[method][i] for method in ['rand', 'deg', 'cc']],
           width=bar_width, label=estimations[i], capsize=5)

# Add labels, title, and legend.
ax.set_ylabel('Mean Absolute Error', fontsize=12)
ax.set_xlabel('Estimation Methods', fontsize=12)
ax.set_title('Comparison of Estimation Methods', fontsize=14)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(title='Estimation Methods', fontsize=10)

# Annotate bars with their heights.
for i in range(4):
    for j, method in enumerate(['rand', 'deg', 'cc']):
        height = means[method][i]
        if height >= 0:
            ax.text(x[j] + offsets[i], height + 0.01, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        else:
            ax.text(x[j] + offsets[i], height - 0.15, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

# Plotting
ax.grid(axis='y', linestyle='-', alpha=0.7)
plt.tight_layout()

plt.savefig('error.eps', format='eps')
plt.show()