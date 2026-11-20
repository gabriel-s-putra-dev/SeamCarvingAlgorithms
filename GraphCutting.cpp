#include "GraphCutting.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace GraphCut
{
    // ========================================================================
    // FlowNetwork
    // ========================================================================
    FlowNetwork::FlowNetwork(int64_t numNodes, const std::vector<Edge>& edges)
        : n(numNodes), start(numNodes + 1, 0)
    {
        // Counting sort of the 2E arcs by their tail node -> CSR layout.
        for (const Edge& e : edges)
        {
            ++start[e.from + 1];
            ++start[e.to + 1];
        }
        for (int64_t u = 0; u < n; ++u)
            start[u + 1] += start[u];

        const size_t numArcs = 2 * edges.size();
        to.resize(numArcs);
        cap.resize(numArcs);
        rev.resize(numArcs);
        std::vector<int64_t> next(start.begin(), start.end() - 1);  // next free slot per node
        for (const Edge& e : edges)
        {
            const int64_t a = next[e.from]++;   // forward arc  u -> v
            const int64_t b = next[e.to]++;     // reverse arc  v -> u
            to[a] = e.to;   cap[a] = e.cap;    rev[a] = b;
            to[b] = e.from; cap[b] = e.revCap; rev[b] = a;
        }
    }

    // Edmonds-Karp: BFS for the shortest augmenting path, push its bottleneck,
    // repeat until t is unreachable. Each augmentation costs one O(E) BFS.
    int64_t FlowNetwork::EdmondsKarp(int64_t s, int64_t t)
    {
        std::vector<int64_t> parentArc(n), queue(n);
        int64_t flow = 0;
        while (true)
        {
            std::fill(parentArc.begin(), parentArc.end(), -1);
            parentArc[s] = -2;
            int64_t head = 0, tail = 0;
            queue[tail++] = s;
            while (head < tail && parentArc[t] == -1)
            {
                const int64_t u = queue[head++];
                for (int64_t a = start[u]; a < start[u + 1]; ++a)
                {
                    const int64_t v = to[a];
                    if (cap[a] > 0 && parentArc[v] == -1)
                    {
                        parentArc[v] = a;
                        queue[tail++] = v;
                    }
                }
            }
            if (parentArc[t] == -1)
                return flow;  // no augmenting path left: flow is maximum

            int64_t f = -1;   // bottleneck capacity along the path
            for (int64_t v = t; v != s; v = to[rev[parentArc[v]]])
            {
                const int64_t a = parentArc[v];
                if (f < 0 || cap[a] < f) f = cap[a];
            }
            for (int64_t v = t; v != s; v = to[rev[parentArc[v]]])
            {
                const int64_t a = parentArc[v];
                cap[a] -= f;
                cap[rev[a]] += f;
            }
            flow += f;
            ++augmentations;
        }
    }

    // Dinic: one BFS builds the level graph, then a blocking flow is sent with
    // an iterative DFS (current-arc pointers, dead-end pruning). Repeat until t
    // is unreachable. The DFS is iterative because seam paths are thousands of
    // nodes long.
    int64_t FlowNetwork::Dinic(int64_t s, int64_t t)
    {
        std::vector<int64_t> level(n), it(n), queue(n), path;
        int64_t flow = 0;
        while (true)
        {
            // 1) BFS: level[v] = shortest distance from s in the residual graph.
            std::fill(level.begin(), level.end(), -1);
            level[s] = 0;
            int64_t head = 0, tail = 0;
            queue[tail++] = s;
            while (head < tail)
            {
                const int64_t u = queue[head++];
                for (int64_t a = start[u]; a < start[u + 1]; ++a)
                {
                    const int64_t v = to[a];
                    if (cap[a] > 0 && level[v] < 0)
                    {
                        level[v] = level[u] + 1;
                        queue[tail++] = v;
                    }
                }
            }
            if (level[t] < 0)
                return flow;
            ++phases;
            for (int64_t u = 0; u < n; ++u)
                it[u] = start[u];

            // 2) Blocking flow along arcs with level[v] == level[u] + 1.
            path.clear();
            int64_t u = s;
            while (true)
            {
                if (u == t)
                {
                    int64_t f = cap[path[0]];
                    for (int64_t a : path) f = std::min(f, cap[a]);
                    for (int64_t a : path)
                    {
                        cap[a] -= f;
                        cap[rev[a]] += f;
                    }
                    flow += f;
                    ++augmentations;
                    // Retreat to the tail of the first saturated arc and continue.
                    size_t k = 0;
                    while (cap[path[k]] > 0) ++k;
                    path.resize(k);
                    u = path.empty() ? s : to[path.back()];
                    continue;
                }
                // Advance along the next admissible arc of u.
                const int64_t end = start[u + 1];
                const int64_t nextLevel = level[u] + 1;
                int64_t a = it[u];
                while (a < end && (cap[a] <= 0 || level[to[a]] != nextLevel)) ++a;
                it[u] = a;
                if (a < end)
                {
                    path.push_back(a);
                    u = to[a];
                }
                else
                {
                    // Dead end: remove u from the level graph and step back.
                    if (u == s) break;
                    level[u] = -1;
                    a = path.back();
                    path.pop_back();
                    u = to[rev[a]];
                    ++it[u];
                }
            }
        }
    }

    int64_t FlowNetwork::Solve(int64_t s, int64_t t, MaxFlowAlgorithm algo)
    {
        return algo == MaxFlowAlgorithm::Dinic ? Dinic(s, t) : EdmondsKarp(s, t);
    }

    std::vector<uint8_t> FlowNetwork::SourceSide(int64_t s) const
    {
        std::vector<uint8_t> inS(n, 0);
        std::vector<int64_t> stack{ s };
        inS[s] = 1;
        while (!stack.empty())
        {
            const int64_t u = stack.back();
            stack.pop_back();
            for (int64_t a = start[u]; a < start[u + 1]; ++a)
            {
                const int64_t v = to[a];
                if (cap[a] > 0 && !inS[v])
                {
                    inS[v] = 1;
                    stack.push_back(v);
                }
            }
        }
        return inS;
    }

    size_t FlowNetwork::MemoryBytes() const
    {
        const size_t csr = (start.size() + to.size() + cap.size() + rev.size()) * sizeof(int64_t);
        const size_t work = 3 * static_cast<size_t>(n) * sizeof(int64_t);  // level, it, queue
        return csr + work;
    }

    // The energy map is SeamCarving::ComputeEnergy (shared with DP and Greedy).

    // ========================================================================
    // Seam finding
    // ========================================================================
    FlowNetwork BuildSeamGraph(const cv::Mat& cost, int64_t& s, int64_t& t)
    {
        const int64_t m = cost.rows, n = cost.cols;
        const int64_t W = n + 1;                    // columns incl. the virtual one
        auto node = [W](int64_t i, int64_t j) { return i * W + j; };
        s = m * W;
        t = s + 1;

        int64_t INF = 1;                            // larger than any finite cut
        for (int64_t i = 0; i < m; ++i)
        {
            const int* row = cost.ptr<int>(static_cast<int>(i));
            for (int64_t j = 0; j < n; ++j) INF += row[j];
        }

        std::vector<FlowNetwork::Edge> edges;
        edges.reserve(2 * m + m * n + 2 * std::max<int64_t>(m - 1, 0) * std::max<int64_t>(n - 1, 0));
        for (int64_t i = 0; i < m; ++i)
        {
            const int* row = cost.ptr<int>(static_cast<int>(i));
            edges.push_back({ s, node(i, 0), INF, 0 });          // source -> first column
            edges.push_back({ node(i, n), t, INF, 0 });          // virtual column -> sink
            for (int64_t j = 0; j < n; ++j)                      // forward c[i][j], backward INF
                edges.push_back({ node(i, j), node(i, j + 1), row[j], INF });
            for (int64_t j = 1; j < n; ++j)                      // diagonal INF edges
            {
                if (i > 0)     edges.push_back({ node(i, j), node(i - 1, j - 1), INF, 0 });
                if (i < m - 1) edges.push_back({ node(i, j), node(i + 1, j - 1), INF, 0 });
            }
        }
        return FlowNetwork(m * W + 2, edges);
    }

    SeamResult FindVerticalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo)
    {
        if (cost.type() != CV_32SC1 || cost.empty())
            throw std::invalid_argument("FindVerticalSeam: cost must be a non-empty CV_32SC1 Mat");
        double minVal;
        cv::minMaxLoc(cost, &minVal);
        if (minVal < 0)
            throw std::invalid_argument("FindVerticalSeam: costs must be non-negative");

        const int m = cost.rows, n = cost.cols;
        SeamResult result;
        result.seam.assign(m, 0);
        if (n == 1)
        {
            for (int i = 0; i < m; ++i) result.cost += cost.at<int>(i, 0);
            result.flow = result.cost;
            return result;
        }

        int64_t s, t;
        FlowNetwork net = BuildSeamGraph(cost, s, t);
        result.flow = net.Solve(s, t, algo);

        // Min cut: S = nodes reachable from s. Each row of S is a prefix, so the
        // rightmost pixel of row i in S is (number of S pixels in row i) - 1.
        const std::vector<uint8_t> inS = net.SourceSide(s);
        const int64_t W = n + 1;
        for (int i = 0; i < m; ++i)
        {
            int count = 0;
            for (int j = 0; j < W; ++j) count += inS[i * W + j];
            result.seam[i] = count - 1;
            result.cost += cost.at<int>(i, result.seam[i]);
        }
        result.nodes = net.NumNodes();
        result.arcs = net.NumArcs();
        result.augmentations = net.Augmentations();
        result.phases = net.Phases();

        // Max-flow min-cut theorem: flow value == cut capacity == seam cost.
        if (result.flow != result.cost)
            throw std::logic_error("GraphCut: max-flow " + std::to_string(result.flow) +
                                   " != seam cost " + std::to_string(result.cost));
        return result;
    }

    SeamResult FindHorizontalSeam(const cv::Mat& cost, MaxFlowAlgorithm algo)
    {
        cv::Mat transposed;
        cv::transpose(cost, transposed);
        return FindVerticalSeam(transposed, algo);   // seam[col] = row
    }

    // ========================================================================
    // Carving: the shared remove-and-repeat loop with the graph-cut seam finder
    // ========================================================================
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight, MaxFlowAlgorithm algo,
                  EnergyType energy, CarveStats* stats, bool recordSeams)
    {
        return SeamCarving::CarveWith(img, newWidth, newHeight, energy,
                                      [algo](const cv::Mat& e) { return FindVerticalSeam(e, algo).seam; },
                                      stats, recordSeams, "GraphCut::Carve");
    }
}
