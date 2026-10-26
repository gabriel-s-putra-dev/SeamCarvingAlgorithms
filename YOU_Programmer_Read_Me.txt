What we need to do:

DynamicProgramming.cpp & DynamicProgramming.h

1. Implement Seam carving using Dynamic programming (for both vertical and
horizontal seams). The main steps are:
a) Calculate the Energy Map: For each pixel in the image, compute its energy
value, which represents the disruption measure. The energy is typically
calculated based on the gradient (the difference in intensity) between adjacent
pixels, or by using filters like the Sobel filter to detect edges and transitions.
b) Use Dynamic Programming to Find the Minimum Seam: Apply dynamic
programming to find the seam (both vertically and horizontally) with the
minimum cumulative energy. This seam represents the path through the image
with the least disruption, as measured by the sum of the energy values along the
seam.
c) Remove the Seam and Repeat: After identifying the minimum energy seam,
remove it from the image. Then recalculate the energy map, and repeat the
process until the image is resized to the desired dimensions.
 
GreedyAlgorithm.cpp & GreedyAlgorithm.h

3. Implement using Greedy Algorithm. As an alternative to dynamic programming,
implement the seam carving using a Greedy Algorithm. In this approach, select the
seam by making a local, row-by-row decision, choosing the minimum-energy pixel at
each step, without considering the entire image. Does the Greedy Algorithm provide
results that are comparable to dynamic programming in terms of visual quality?
Compare the seams, visual quality and running time with your Dynamic Programming
implementation.

GraphCutting.cpp & GraphCutting.h

4. Implement using Graph Cut (min-cut/max-flow). As another alternative to dynamic
programming, reformulate seam finding as a minimum s–t cut problem and solve it
with max-flow. This basic idea is: for a vertical seam, build a graph with one node per
pixel, connect the source to every pixel in the first column and every pixel in the last
column to the sink with infinite capacity, and give each edge from pixel img[i, j], to its
right neighbour img [i, j+1] a capacity equal to the cost of removing pixel img[i, j], for
example its energy e[i, j] or the forward energy |I[i, j+1] – I[i, j−1]|. Add infinite-capacity
backward edges from [i, j+1] to [i, j], and diagonal edges from [i, j] to [i−1, j−1] and [i+1,
j−1], so that every finite cut crosses each row exactly once and forms a connected
seam. Run a max-flow algorithm (e.g. Ford–Fulkerson or Edmonds–Karp), then find
the source side S of the minimum cut by a BFS/DFS from the source in the residual
graph; in each row, the rightmost pixel in S is the seam pixel to remove. By the max-
flow min-cut theorem the capacity of this cut equals the total cost of the seam, so the
seam is optimal. Horizontal seams are handled the same way on the transposed
image. Compare the seams, visual quality and running time with your Dynamic
Programming implementation.

main.cpp

The plan is to let the user choose which algorithm to do image cutting. You can choose to use 
third-party libraries, including image libraries like OpenCV for image processing (reading/writing 
images, Seam carving Cropping Scaling computing gradient magnitude or Sobel filter) and math 
libraries for math computations. However, you cannot use existing functions of content-aware image 
resizing in the image libraries.