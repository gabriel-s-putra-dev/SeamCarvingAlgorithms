/*!************************************************************************
\file GraphCutting.h
\author GYQ-SIT
\par DP email: 2301516@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This header file contains the declarations for seam carving by Graph
Cut.

Seam carving by Graph Cut (minimum s-t cut solved with max-flow).

Seam finding is reformulated as a minimum s-t cut and solved with max-flow
(Dinic or Edmonds-Karp, both implemented from scratch). OpenCV is only used
for image I/O, colour conversion and the Sobel filter.

The energy map, seam removal, the carving loop and the seam overlay are
shared with DP and Greedy (SeamCarving.h); this file only finds seams.

Graph for a VERTICAL seam on an m x n cost map c[i][j]
  nodes : one per pixel (i, j), j = 0..n-1, plus a VIRTUAL column j = n,
          plus source s and sink t                    -> V = m(n+1) + 2
  s -> (i, 0)            capacity INF  (first column is on the source side)
  (i, n) -> t            capacity INF  (virtual column is on the sink side)
  (i, j) -> (i, j+1)     capacity c[i][j]   "cutting it removes pixel (i, j)"
  (i, j+1) -> (i, j)     capacity INF       backward edge
  (i, j) -> (i-1, j-1)   capacity INF       diagonal edge
  (i, j) -> (i+1, j-1)   capacity INF       diagonal edge

Why the virtual column: if the last real column were wired straight to t
(as in the brief), the cut could never fall to its right, so pixels in the
last column could never be removed. With the extra column, every column is
removable and the seam space is exactly the same as the DP's.

Why every finite cut is exactly one connected seam (S = source side):
  * backward INF edges: (i, j+1) in S  =>  (i, j) in S, so row i of S is a
    prefix {0..b_i} and exactly ONE forward edge per row is cut;
  * diagonal INF edges: (i, b_i) in S  =>  (i-1, b_i-1) and (i+1, b_i-1)
    in S, so |b_i - b_{i+1}| <= 1 (8-connected seam);
  * cut capacity = sum_i c[i][b_i] = seam cost. By the max-flow min-cut
    theorem, max flow = min cut = optimal seam cost.
- FlowNetwork
residual graph in CSR form with the Edmonds-Karp and Dinic max-flow
solvers
- SeamResult
a graph-cut seam, its cost, the flow value and solver counters
- BuildSeamGraph(cost, s, t)
builds the flow network for a vertical seam
- FindVerticalSeam(cost, algo)
returns the optimal vertical seam from a minimum s-t cut
- FindHorizontalSeam(cost, algo)
returns the optimal horizontal seam on the transposed cost map
- Carve(img, newWidth, newHeight, algo, energy, stats, recordSeams)
shrinks an image by repeatedly removing the graph-cut seam
- EnergyType, ComputeEnergy, RemoveVerticalSeam, RemoveHorizontalSeam, CarveStats, DrawSeams
re-exported from SeamCarving.h
**************************************************************************/

#pragma once

// ----- Headers ------------------------------------------------------- //
#include "SeamCarving.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace GraphCut
{
    enum class MaxFlowAlgorithm { Dinic, EdmondsKarp };

    // Shared energy map, removal, stats and overlay (see SeamCarving.h).
    using EnergyType = SeamCarving::EnergyType;
    using SeamCarving::ComputeEnergy;
    using SeamCarving::RemoveVerticalSeam;
    using SeamCarving::RemoveHorizontalSeam;
    using SeamCarving::CarveStats;
    using SeamCarving::DrawSeams;

    // ------------------------------------------------------------------------
    // Residual graph in CSR (compressed sparse row) form.
    // Arcs leaving node u are stored at indices start[u] .. start[u+1]-1.
    // Every logical edge (u, v, c, rc) becomes two arcs: u->v with capacity c
    // and v->u with capacity rc; rev[a] is the index of the paired arc.
    // Pushing f units along arc a: cap[a] -= f, cap[rev[a]] += f.
    // ------------------------------------------------------------------------
    class FlowNetwork
    {
    public:
        struct Edge { int64_t from, to, cap, revCap; };

        FlowNetwork(int64_t numNodes, const std::vector<Edge>& edges);

        int64_t EdmondsKarp(int64_t s, int64_t t);   // O(V E^2)
        int64_t Dinic(int64_t s, int64_t t);         // O(V^2 E)
        int64_t Solve(int64_t s, int64_t t, MaxFlowAlgorithm algo);

        // After a max-flow run: true for nodes reachable from s in the
        // residual graph = source side S of a minimum cut.
        std::vector<uint8_t> SourceSide(int64_t s) const;

        /************************************************************************/
        /*!
          \brief
        	gets the number of nodes in the flow network
          \return int64_t
          64-bit integer
          returns the number of nodes, V
        */
        /************************************************************************/
        int64_t NumNodes() const { return n; }
        /************************************************************************/
        /*!
          \brief
        	gets the number of arcs in the residual graph (two per logical edge)
          \return int64_t
          64-bit integer
          returns the number of arcs
        */
        /************************************************************************/
        int64_t NumArcs() const { return static_cast<int64_t>(to.size()); }
        /************************************************************************/
        /*!
          \brief
        	gets how many augmenting paths max-flow has pushed flow along on this
        	network
          \return int64_t
          64-bit integer
          returns the number of augmentations
        */
        /************************************************************************/
        int64_t Augmentations() const { return augmentations; }
        /************************************************************************/
        /*!
          \brief
        	gets how many Dinic phases (level graphs built by BFS) max-flow has used
        	on this network; stays 0 for Edmonds-Karp
          \return int64_t
          64-bit integer
          returns the number of phases
        */
        /************************************************************************/
        int64_t Phases() const { return phases; }
        size_t MemoryBytes() const;   // CSR arrays + solver work arrays

    private:
        int64_t n;
        std::vector<int64_t> start, to, cap, rev;
        int64_t augmentations = 0, phases = 0;
    };

    struct SeamResult
    {
        std::vector<int> seam;     // vertical: seam[row] = column; horizontal: seam[col] = row
        int64_t cost = 0;          // sum of costs along the seam (== max-flow value)
        int64_t flow = 0;
        int64_t nodes = 0, arcs = 0, augmentations = 0, phases = 0;
    };

    // Build the flow network for a vertical seam (exposed for tests / analysis).
    FlowNetwork BuildSeamGraph(const cv::Mat& cost, int64_t& s, int64_t& t);

    // Optimal seam on a non-empty CV_32SC1 cost map with no negative values
    // (otherwise std::invalid_argument).
    //
    // NOTE on FORWARD energy: it compares left and right neighbours, so it is
    // the right cost for VERTICAL seams only. For a horizontal seam with
    // forward energy use FindVerticalSeam(ComputeEnergy(img.t(), Forward)) on
    // the transposed image (Carve does this). Backward energy works either way.
    SeamResult FindVerticalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic);
    SeamResult FindHorizontalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic);

    // Shrink img to newWidth x newHeight: vertical seams first, then horizontal.
    // The energy is recomputed after every removal. Runs SeamCarving::CarveWith
    // with the graph-cut FindVerticalSeam; see there for stats, recordSeams
    // and errors.
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);
}
