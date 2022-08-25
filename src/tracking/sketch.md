Evaluate model's trajectories

1. Match predicted trajectories with ground truth trajectories
2. Compute HOTA score for each predicted trajectory using each predicted trajectory's matching ground truth trajectory


match_trajs -- compute_traj_matrix -- compute_similarity
compute_score

---
Correlate loops with beta cells

1. Fetch prediction tracks
2. Fetch loops
3. Compare

Loops have unique pixel intensities in each 3D image.

**Task**: Determine if a beta cell inside a loop.
- For each cell B:
  - For each slice S of B:
    - For each loop L:
        - Is 

- For each cell B:
  - Find outer contour of B
  - Find if contour intersects area inside of a loop

**Questions for Pia and Silja**
- What amount of intersection is required for the cell to be considered to be inside the loop?