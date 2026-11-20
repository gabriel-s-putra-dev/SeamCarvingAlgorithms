// Correctness tests + quick benchmark for GraphCutting.cpp.
// Build and run from the repo root:
//   cmake --build --preset vs2022-release --target gc_tests
//   out\build\vs2022\Release\gc_tests.exe     (or: ctest --preset vs2022-release)
// or with g++:
//   g++ -std=c++14 -O2 -I. tests/GraphCuttingTests.cpp SeamCarving.cpp GraphCutting.cpp DynamicProgramming.cpp -o gc_tests `pkg-config --cflags --libs opencv4`
//
// Key check: by the max-flow min-cut theorem, the graph-cut seam must have
// EXACTLY the same total cost as the DP optimum on the same cost map. The
// optimum comes from the team's DP (TeamDPCost in tests/Check.h), which is
// itself checked against brute force in tests/SeamTests.cpp.
#include "Check.h"
#include "GraphCutting.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <string>
#include <vector>

using namespace GraphCut;

static bool ValidSeam(const std::vector<int>& seam, int n)
{
    for (size_t i = 0; i < seam.size(); ++i)
    {
        if (seam[i] < 0 || seam[i] >= n) return false;
        if (i > 0 && std::abs(seam[i] - seam[i - 1]) > 1) return false;
    }
    return true;
}

static const MaxFlowAlgorithm ALGOS[] = { MaxFlowAlgorithm::Dinic, MaxFlowAlgorithm::EdmondsKarp };

// ---------------------------------------------------------------------------
static void TestMaxFlowHandExamples()
{
    // CLRS Figure 26.1: max flow 23.
    const std::vector<FlowNetwork::Edge> clrs = {
        {0, 1, 16, 0}, {0, 2, 13, 0}, {2, 1, 4, 0}, {1, 3, 12, 0}, {3, 2, 9, 0},
        {2, 4, 14, 0}, {4, 3, 7, 0}, {3, 5, 20, 0}, {4, 5, 4, 0} };
    for (auto algo : ALGOS)
    {
        FlowNetwork net(6, clrs);
        CHECK(net.Solve(0, 5, algo) == 23);
        auto S = net.SourceSide(0);
        int64_t cut = 0;
        for (const auto& e : clrs) if (S[e.from] && !S[e.to]) cut += e.cap;
        CHECK(S[0] && !S[5] && cut == 23);

        CHECK(FlowNetwork(2, { {0, 1, 7, 0} }).Solve(0, 1, algo) == 7);
        CHECK(FlowNetwork(3, { {0, 1, 5, 0} }).Solve(0, 2, algo) == 0);
        CHECK(FlowNetwork(4, { {0,1,10,0},{1,3,3,0},{0,2,4,0},{2,3,9,0} }).Solve(0, 3, algo) == 7);
        // needs a cancelling push along a reverse arc
        CHECK(FlowNetwork(4, { {0,1,1,0},{0,2,1,0},{1,2,1,0},{1,3,1,0},{2,3,1,0} }).Solve(0, 3, algo) == 2);
    }
}

static void TestAlgorithmsAgreeOnRandomGraphs()
{
    std::mt19937 rng(1);
    for (int trial = 0; trial < 200; ++trial)
    {
        const int n = 2 + rng() % 10, E = 1 + rng() % 40;
        std::vector<FlowNetwork::Edge> edges;
        for (int k = 0; k < E; ++k)
            edges.push_back({ int64_t(rng() % n), int64_t(rng() % n), int64_t(rng() % 10), 0 });
        FlowNetwork a(n, edges), b(n, edges);
        CHECK(a.Dinic(0, n - 1) == b.EdmondsKarp(0, n - 1));
    }
}

static void TestRandomCostMapsMatchDP()
{
    std::mt19937 rng(42);
    for (auto algo : ALGOS)
        for (int trial = 0; trial < 300; ++trial)
        {
            const int m = 1 + rng() % 11, n = 1 + rng() % 11;
            cv::Mat c(m, n, CV_32S);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) c.at<int>(i, j) = rng() % 50;
            const SeamResult r = FindVerticalSeam(c, algo);
            CHECK(r.cost == TeamDPCost(c));
            CHECK(ValidSeam(r.seam, n));
            CHECK(r.flow == r.cost);
        }
}

static void TestEdgeCases()
{
    const std::vector<cv::Mat> cases = {
        FromRows({ {5} }),
        FromRows({ {3, 1, 2} }),
        FromRows({ {4}, {2}, {7} }),
        cv::Mat::zeros(5, 6, CV_32S),
        cv::Mat(4, 4, CV_32S, cv::Scalar(9)),
        FromRows({ {1, 9}, {9, 1}, {1, 9} }) };
    for (const cv::Mat& c : cases)
    {
        const SeamResult r = FindVerticalSeam(c);
        CHECK(r.cost == TeamDPCost(c));
        CHECK(ValidSeam(r.seam, c.cols));
    }
}

static void TestFirstAndLastColumnRemovable()
{
    cv::Mat c(6, 5, CV_32S, cv::Scalar(10));
    c.col(4).setTo(0);
    SeamResult r = FindVerticalSeam(c);
    CHECK(r.cost == 0);
    for (int j : r.seam) CHECK(j == 4);

    c.setTo(10);
    c.col(0).setTo(0);
    r = FindVerticalSeam(c);
    CHECK(r.cost == 0);
    for (int j : r.seam) CHECK(j == 0);
}

static void TestConnectivityEnforced()
{
    cv::Mat c(4, 6, CV_32S, cv::Scalar(100));
    c.at<int>(0, 0) = c.at<int>(1, 5) = c.at<int>(2, 0) = c.at<int>(3, 5) = 0;
    const SeamResult r = FindVerticalSeam(c);
    CHECK(ValidSeam(r.seam, 6));
    CHECK(r.cost == TeamDPCost(c) && r.cost > 0);
}

static void TestGraphSize()
{
    const int m = 7, n = 9;
    int64_t s, t;
    FlowNetwork net = BuildSeamGraph(cv::Mat::ones(m, n, CV_32S), s, t);
    CHECK(net.NumNodes() == m * (n + 1) + 2);
    CHECK(net.NumArcs() == 2 * (2 * m + m * n + 2 * (m - 1) * (n - 1)));
}

static void TestHorizontalSeamAndRemoval()
{
    cv::Mat c(5, 4, CV_32S, cv::Scalar(10));
    c.row(2).setTo(0);
    const SeamResult r = FindHorizontalSeam(c);
    CHECK(r.cost == 0 && r.seam.size() == 4);
    for (int i : r.seam) CHECK(i == 2);

    cv::Mat a = (cv::Mat_<int>(3, 4) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    RemoveVerticalSeam(a, { 0, 1, 3 });
    const cv::Mat expect = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 6, 7, 8, 9, 10);
    CHECK(cv::countNonZero(a != expect) == 0);
}

static cv::Mat TestImage()
{
    std::mt19937 rng(3);
    cv::Mat img(24, 30, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(rng() % 256, rng() % 256, rng() % 256);
    img.colRange(12, 16).setTo(cv::Scalar(255, 255, 255));
    return img;
}

static void TestCarveLoop()
{
    const cv::Mat img = TestImage();
    for (auto energy : { EnergyType::Backward, EnergyType::Forward })
    {
        const cv::Mat c = ComputeEnergy(img, energy);
        CHECK(c.type() == CV_32SC1 && c.size() == img.size());
        CHECK(FindVerticalSeam(c).cost == TeamDPCost(c));
    }
    CarveStats st;
    const cv::Mat out = Carve(img, 25, 20, MaxFlowAlgorithm::Dinic, EnergyType::Backward, &st, true);
    CHECK(out.cols == 25 && out.rows == 20 && out.type() == img.type());
    CHECK(st.verticalSeams.size() == 5 && st.horizontalSeams.size() == 4);
    // Every original pixel is removed at most once and the counts add up.
    cv::Mat removed = cv::Mat::zeros(img.size(), CV_8U);
    size_t total = 0;
    for (const auto* list : { &st.verticalSeams, &st.horizontalSeams })
        for (const auto& seam : *list)
            for (const cv::Point& p : seam) { removed.at<uchar>(p) += 1; ++total; }
    double maxCount;
    cv::minMaxLoc(removed, nullptr, &maxCount);
    CHECK(maxCount <= 1 && total + 25 * 20 == size_t(24 * 30));

    bool threw = false;
    try { Carve(img, 0, 10); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { FindVerticalSeam(FromRows({ {-1, 2} })); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Bad seams throw std::invalid_argument (like DP and Greedy) and leave the
    // image unchanged.
    const cv::Mat e = FromRows({ { 0, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 9, 10, 11 } });
    cv::Mat m = e.clone();
    CHECK(Throws([&] { RemoveVerticalSeam(m, { 0, 1 }); }));          // wrong length
    CHECK(Throws([&] { RemoveVerticalSeam(m, { 0, 1, 4 }); }));       // out of range
    CHECK(Throws([&] { RemoveVerticalSeam(m, { 0, -1, 0 }); }));      // negative
    CHECK(Throws([&] { RemoveHorizontalSeam(m, { 0, 1, 2 }); }));     // wrong length
    CHECK(Throws([&] { RemoveHorizontalSeam(m, { 0, 1, 3, 0 }); }));  // out of range
    CHECK(m.size() == e.size() && cv::countNonZero(m != e) == 0);            // untouched
    cv::Mat thin(3, 1, CV_32S, cv::Scalar(0)), flat(1, 3, CV_32S, cv::Scalar(0)), empty;
    CHECK(Throws([&] { RemoveVerticalSeam(thin, { 0, 0, 0 }); }));
    CHECK(Throws([&] { RemoveHorizontalSeam(flat, { 0, 0, 0 }); }));
    CHECK(Throws([&] { RemoveVerticalSeam(empty, {}); }));

    // DrawSeams rejects seams that do not belong to the image.
    CHECK(DrawSeams(img, st).size() == img.size());
    const cv::Mat small(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    CHECK(Throws([&] { DrawSeams(small, st); }));                     // stats from a bigger image
    CHECK(Throws([&] { DrawSeams(cv::Mat(), st); }));
    CHECK(Throws([&] { DrawSeams(cv::Mat(24, 30, CV_32FC3), st); })); // not 8-bit
}

// Same image as the Python prototype: first seam cost must be 2446. Then carve
// 80 seams with graph cut and, at EVERY step, check that the graph-cut seam cost
// equals the DP optimum on the same current image. (When several seams tie, the
// two methods may pick different optimal seams; that is fine. Graph cut always
// returns the minimal source side, i.e. the leftmost optimal seam.)
static void TestTowerImage()
{
    const cv::Mat img = cv::imread("images/tower.png");
    if (img.empty()) { std::printf("  (skipped: images/tower.png not found)\n"); return; }
    const cv::Mat c = ComputeEnergy(img);
    const SeamResult r = FindVerticalSeam(c);
    CHECK(r.cost == 2446);
    CHECK(r.cost == TeamDPCost(c));

    cv::Mat cur = img.clone();
    int identical = 0, costMatches = 0;
    for (int k = 0; k < 80; ++k)
    {
        const cv::Mat e = ComputeEnergy(cur);
        std::vector<int> dpSeam;
        const int64_t dpCost = TeamDPCost(e, &dpSeam);
        const SeamResult g = FindVerticalSeam(e);
        costMatches += (g.cost == dpCost);
        identical += (dpSeam == g.seam);
        RemoveVerticalSeam(cur, g.seam);
    }
    std::printf("  tower: GC cost == DP cost at %d/80 steps; identical seams %d/80\n", costMatches, identical);
    CHECK(costMatches == 80);
}

// ---------------------------------------------------------------------------
static void Benchmark()
{
    const cv::Mat img = cv::imread("images/tower.png");
    if (img.empty()) return;
    std::printf("\nOne-seam timing on images/tower.png (%dx%d):\n", img.cols, img.rows);
    const cv::Mat c = ComputeEnergy(img);
    for (auto algo : ALGOS)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const SeamResult r = FindVerticalSeam(c, algo);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %-12s %8.1f ms  cost=%lld  V=%lld  arcs=%lld  augmentations=%lld  phases=%lld\n",
                    algo == MaxFlowAlgorithm::Dinic ? "Dinic" : "EdmondsKarp", ms,
                    (long long)r.cost, (long long)r.nodes, (long long)r.arcs,
                    (long long)r.augmentations, (long long)r.phases);
    }
    const auto t0 = std::chrono::steady_clock::now();
    TeamDPCost(c);
    std::printf("  %-12s %8.2f ms  (team DP)\n", "DP",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

int main()
{
    const int result = RunTests({
        { "max-flow hand examples", TestMaxFlowHandExamples },
        { "Dinic == Edmonds-Karp on random graphs", TestAlgorithmsAgreeOnRandomGraphs },
        { "random cost maps: GC cost == DP cost", TestRandomCostMapsMatchDP },
        { "edge cases", TestEdgeCases },
        { "first and last column removable", TestFirstAndLastColumnRemovable },
        { "connectivity enforced", TestConnectivityEnforced },
        { "graph size", TestGraphSize },
        { "horizontal seam and removal", TestHorizontalSeamAndRemoval },
        { "carve loop", TestCarveLoop },
        { "tower image vs DP", TestTowerImage },
    });
    Benchmark();
    return result;
}
