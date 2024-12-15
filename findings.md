## Findings during the project
Here are the findings (in short details) that we found after the application
on the mentioned datasets:
   - We found that the Degree and Deg-1 landmark selection methods performed the
   best with all the estimation methods.
   - We found that taking the Upperbound estimation method works best for the Degree and Deg-1 landmark selection methods.
   - We found that CC and CC-1 performed the least optimal.
   - The Random landmark selection method performed best in sparse graphs with a low medium degree.

During testing, we did not recalculate the landmarks for the Random method. The other methods were deterministic. Tiebreakers were handled by Python's sorting function. 