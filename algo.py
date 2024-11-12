import networkx as nx
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
    '''Returns U.'''
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

# Function to calculate lower and upper bounds based on landmark distances
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

# TODO: Import the graph, random for now.
G = nx.erdos_renyi_graph(100, 0.05, seed=42, directed=False)

# Number of landmarks in the graph.
marks_n = 5 

# Selection
marks_rand = rand_landmarks(G, marks_n)
marks_deg  = deg_landmarks(G, marks_n)
marks_cc   = cc_landmarks(G, marks_n)

# Store distances of landmarks and points
marks_dis_rand = all_landmark_distances(G, marks_rand)
marks_dis_deg  = all_landmark_distances(G, marks_deg)
marks_dis_cc   = all_landmark_distances(G, marks_cc)


# FROM HERE ON IT SHOULD BE VERY FAST! -----------------------------------------

# Choose two nodes for distance estimation
s, t = random.choice(list(G.nodes)), random.choice(list(G.nodes))

# Compute bounds for different landmark selection methods
lb_rand, ub_rand = get_bounds(s, t, marks_rand, marks_dis_rand)
lb_deg , ub_deg  = get_bounds(s, t, marks_deg , marks_dis_deg )
lb_cc  , ub_cc   = get_bounds(s, t, marks_cc  , marks_dis_cc  )

# Display results
print(f"Actual Distance: {nx.dijkstra_path_length(G, s, t)}")
print(f"Random Landmarks: Lower Bound = {lb_rand}, Upper Bound = {ub_rand}")
print(f"Degree Landmarks: Lower Bound = {lb_deg}, Upper Bound = {ub_deg}")
print(f"Centrality Landmarks: Lower Bound = {lb_cc}, Upper Bound = {ub_cc}")


# TODO: Make a thing to test it on a lot of random points and return some graph.

# TODO: Test the 4 different estimation methods.

