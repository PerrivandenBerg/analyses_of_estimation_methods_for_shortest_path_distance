## Findings during the project

- The path estimation struggles with a graph which has many small connected 
components. We might need to consider the largest connected component or make
something so we can be sure that there is no path between 2 points using the
data from the landmarks.

- On random Erdos Renyi graphs, the square root method seems to work the best,
with the "Only Upperbound" and "Only Lowerbound" estimation methods performing 
the worst.