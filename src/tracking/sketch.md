## Evaluate model's trajectories

1. Match predicted trajectories with ground truth trajectories
2. Compute HOTA score for each predicted trajectory using each predicted trajectory's matching ground truth trajectory


match_trajs -- compute_traj_matrix -- compute_similarity
compute_score

---
## Correlate loops with beta cells

1. Fetch prediction tracks
2. Fetch loops
3. Compare

Loops have unique pixel intensities in each 3D image.

**Task**: Determine if a beta cell inside a loop.

*Approach 1*
- For each cell B:
  - For each slice S of B:
    - For each loop L:
        - Is 

*Approach 2*
- For each cell B:
  - Get outer contour of B
  - Approximate loop by sampling from its coordinates
  - Find if contour intersects plane inside of a loop
    - If, for some loop, we cannot find two opposing coordinates in the loop for which at least one of the cell coordinates are not between them, then the cell is probably inside that loop.
      - If the cell is between two opposing coordinates, then, if we take the difference between loop coords 1 and cell coords, and then loop coords 2 and cell coords, then the sign of these two differences is going to be different.

**Debug locally** by using available tif file (made sparse on HPC)
    - 

**Questions for Pia and Silja**
- What point of intersection is required for the cell to be considered to be inside the loop? I.e. does the intersection have to be near the middle of the cell, or is it enough for the cell to barely touch it?

- I think I should keep the point of intersection.