#pragma once
// ============================================================================
// Seam carving by Graph Cut (minimum s-t cut solved with max-flow).
//
// Seam finding is reformulated as a minimum s-t cut and solved with max-flow
// (Dinic or Edmonds-Karp, both implemented from scratch). OpenCV is only used
// for image I/O, colour conversion and the Sobel filter.
//
// Graph for a VERTICAL seam on an m x n cost map c[i][j]
//   nodes : one per pixel (i, j), j = 0..n-1, plus a VIRTUAL column j = n,
//           plus source s and sink t                    -> V = m(n+1) + 2
//   s -> (i, 0)            capacity INF  (first column is on the source side)
//   (i, n) -> t            capacity INF  (virtual column is on the sink side)
//   (i, j) -> (i, j+1)     capacity c[i][j]   "cutting it removes pixel (i, j)"
//   (i, j+1) -> (i, j)     capacity INF       backward edge
//   (i, j) -> (i-1, j-1)   capacity INF       diagonal edge
//   (i, j) -> (i+1, j-1)   capacity INF       diagonal edge
//
// Why the virtual column: if the last real column were wired straight to t
// (as in the brief), the cut could never fall to its right, so pixels in the
// last column could never be removed. With the extra column, every column is
// removable and the seam space is exactly the same as the DP's.
//
// Why every finite cut is exactly one connected seam (S = source side):
//   * backward INF edges: (i, j+1) in S  =>  (i, j) in S, so row i of S is a
//     prefix {0..b_i} and exactly ONE forward edge per row is cut;
//   * diagonal INF edges: (i, b_i) in S  =>  (i-1, b_i-1) and (i+1, b_i-1)
//     in S, so |b_i - b_{i+1}| <= 1 (8-connected seam);
//   * cut capacity = sum_i c[i][b_i] = seam cost. By the max-flow min-cut
//     theorem, max flow = min cut = optimal seam cost.
// ============================================================================
#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace GraphCut
{
    enum class MaxFlowAlgorithm { Dinic, EdmondsKarp };
    enum class EnergyType { Backward, Forward };

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

        int64_t NumNodes() const { return n; }
        int64_t NumArcs() const { return static_cast<int64_t>(to.size()); }
        int64_t Augmentations() const { return augmentations; }
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

    // Energy / cost map, returned as CV_32S (exact integer capacities).
    //   Backward: |Sobel_x| + |Sobel_y| on the grayscale image.
    //   Forward : the brief's SIMPLIFIED per-pixel forward cost
    //             |I[i][j+1] - I[i][j-1]| (not the transition-dependent version).
    cv::Mat ComputeEnergy(const cv::Mat& img, EnergyType type = EnergyType::Backward);

    // Build the flow network for a vertical seam (exposed for tests / analysis).
    FlowNetwork BuildSeamGraph(const cv::Mat& cost, int64_t& s, int64_t& t);

    // Optimal seam on a CV_32S cost map.
    SeamResult FindVerticalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic);
    SeamResult FindHorizontalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic);

    // Remove one pixel per row (vertical) / per column (horizontal). Any Mat type.
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam);
    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam);

    struct CarveStats
    {
        double seamSeconds = 0.0;        // time spent finding seams (energy + max-flow)
        int64_t totalCost = 0;           // sum of all removed seam costs
        std::vector<int64_t> seamCosts;  // cost of each removed seam, in order
        // Original-image (x = col, y = row) coordinates of every removed seam,
        // filled only when recordSeams is true. Used by DrawSeams.
        std::vector<std::vector<cv::Point>> verticalSeams, horizontalSeams;
    };

    // Shrink img to newWidth x newHeight: vertical seams first, then horizontal.
    // The energy is recomputed after every removal.
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  MaxFlowAlgorithm algo = MaxFlowAlgorithm::Dinic,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);

    // Original image with all removed seams drawn (red = vertical, green = horizontal).
    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats);
}
