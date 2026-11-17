// Cross-check: Graph Cut (GraphCutting.cpp) against the TEAM's DP
// (DynamicProgramming.cpp), not the small reference DP inside
// GraphCuttingTests.cpp.
//
// Build from the repo root:
//   g++ -std=c++14 -O2 -I. tests/GraphCutVsTeamDPTests.cpp GraphCutting.cpp DynamicProgramming.cpp -o gc_vs_dp `pkg-config --cflags --libs opencv4`
//   ./gc_vs_dp            (run from the repo root so images/ is found)
//
// What is checked:
//   1. Both modules compute the SAME energy map (backward and forward).
//   2. Single seams on random cost maps: graph-cut cost == DP optimum, for
//      Dinic and Edmonds-Karp, vertical and horizontal. The GC seam is also
//      re-scored with DP::SeamEnergy, which throws if the seam is invalid.
//   3. Full carves on real images: GC::Carve and DP::Carve remove seams of
//      equal cost at every step and produce pixel-identical output.
//
// By the max-flow min-cut theorem, step 2 must hold exactly. Equal seams and
// identical images (step 3) additionally rely on both sides breaking ties
// towards the leftmost seam; costs would match even without that.
#include "GraphCutting.h"
#include "DynamicProgramming.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0, g_passed = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) ++g_passed;                                                       \
        else { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

static bool SameMat(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size() || a.type() != b.type()) return false;
    if (a.empty()) return true;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

static cv::Mat RandomCost(std::mt19937& rng, int rows, int cols, int maxVal)
{
    std::uniform_int_distribution<int> v(0, maxVal);
    cv::Mat c(rows, cols, CV_32S);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) c.at<int>(i, j) = v(rng);
    return c;
}

static cv::Mat RandomImage(std::mt19937& rng, int rows, int cols)
{
    std::uniform_int_distribution<int> v(0, 255);
    cv::Mat img(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(v(rng)), uchar(v(rng)), uchar(v(rng)));
    return img;
}

// ---------------------------------------------------------------------------
static void TestEnergyMapsAgree(const std::vector<cv::Mat>& images)
{
    std::printf("[energy maps: GraphCut::ComputeEnergy == DP::ComputeEnergy]\n");
    for (const cv::Mat& img : images)
    {
        CHECK(SameMat(GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Backward),
                      DP::ComputeEnergy(img, DP::EnergyType::Backward)));
        CHECK(SameMat(GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Forward),
                      DP::ComputeEnergy(img, DP::EnergyType::Forward)));
    }
    std::printf("  ok\n");
}

static void TestRandomSeams()
{
    std::printf("[random cost maps: GC seam cost == team DP optimum]\n");
    std::mt19937 rng(20261117);
    std::uniform_int_distribution<int> dim(1, 14);
    int cases = 0;
    for (int t = 0; t < 400; ++t)
    {
        const int rows = dim(rng), cols = dim(rng);
        // Small value ranges create many ties; large ones test big sums.
        const int maxVal = (t % 3 == 0) ? 3 : (t % 3 == 1) ? 255 : 2000000;
        const cv::Mat c = RandomCost(rng, rows, cols, maxVal);

        const std::vector<int> dpV = DP::FindVerticalSeam(c);
        const std::vector<int> dpH = DP::FindHorizontalSeam(c);
        const long long dpVCost = DP::SeamEnergy(c, dpV, true);
        const long long dpHCost = DP::SeamEnergy(c, dpH, false);

        for (auto algo : { GraphCut::MaxFlowAlgorithm::Dinic, GraphCut::MaxFlowAlgorithm::EdmondsKarp })
        {
            const GraphCut::SeamResult v = GraphCut::FindVerticalSeam(c, algo);
            const GraphCut::SeamResult h = GraphCut::FindHorizontalSeam(c, algo);
            CHECK(v.cost == dpVCost);
            CHECK(h.cost == dpHCost);
            CHECK(v.flow == v.cost);
            CHECK(h.flow == h.cost);
            // Re-score with the team's validator (throws on an invalid seam).
            try
            {
                CHECK(DP::SeamEnergy(c, v.seam, true) == v.cost);
                CHECK(DP::SeamEnergy(c, h.seam, false) == h.cost);
            }
            catch (const std::exception& e)
            {
                CHECK(!"GC seam rejected by DP::SeamEnergy");
                std::printf("    %s\n", e.what());
            }
            // Both prefer the leftmost (topmost) optimum, so seams match too.
            CHECK(v.seam == dpV);
            CHECK(h.seam == dpH);
        }
        ++cases;
    }
    std::printf("  %d random maps x {Dinic, EK} x {vertical, horizontal}: ok\n", cases);
}

static void TestCarveAgrees(const std::string& name, const cv::Mat& img, int newW, int newH,
                            GraphCut::EnergyType gcE, DP::EnergyType dpE, const char* eName)
{
    GraphCut::CarveStats gs;
    DP::CarveStats ds;
    const cv::Mat gOut = GraphCut::Carve(img, newW, newH, GraphCut::MaxFlowAlgorithm::Dinic, gcE, &gs, true);
    const cv::Mat dOut = DP::Carve(img, newW, newH, dpE, &ds, true);

    const size_t steps = gs.seamCosts.size();
    CHECK(steps == ds.seamCosts.size());
    int equalCost = 0;
    for (size_t k = 0; k < steps && k < ds.seamCosts.size(); ++k)
        if (gs.seamCosts[k] == ds.seamCosts[k]) ++equalCost;
    CHECK(equalCost == static_cast<int>(steps));
    CHECK(gs.totalCost == ds.totalCost);
    CHECK(gs.verticalSeams == ds.verticalSeams);
    CHECK(gs.horizontalSeams == ds.horizontalSeams);
    CHECK(SameMat(gOut, dOut));
    std::printf("  %-10s %-8s %dx%d -> %dx%d: equal cost %d/%zu steps, total GC %lld / DP %lld, image %s\n",
                name.c_str(), eName, img.cols, img.rows, newW, newH, equalCost, steps,
                (long long)gs.totalCost, ds.totalCost, SameMat(gOut, dOut) ? "identical" : "DIFFERENT");
}

int main()
{
    const cv::Mat tower = cv::imread("images/tower.png");
    const cv::Mat synthetic = cv::imread("images/synthetic.png");
    if (tower.empty() || synthetic.empty())
    {
        std::printf("Run from the repository root (images/ not found)\n");
        return 1;
    }

    std::mt19937 rng(7);
    std::vector<cv::Mat> images = { tower, synthetic };
    for (int k = 0; k < 20; ++k) images.push_back(RandomImage(rng, 3 + k, 4 + 2 * k));
    TestEnergyMapsAgree(images);

    TestRandomSeams();

    std::printf("[full carve: GraphCut::Carve vs DP::Carve, energy recomputed each step]\n");
    cv::Mat towerSmall;
    cv::resize(tower, towerSmall, cv::Size(137, 93), 0, 0, cv::INTER_AREA);
    for (int e = 0; e < 2; ++e)
    {
        const auto gcE = e ? GraphCut::EnergyType::Forward : GraphCut::EnergyType::Backward;
        const auto dpE = e ? DP::EnergyType::Forward : DP::EnergyType::Backward;
        const char* en = e ? "forward" : "backward";
        TestCarveAgrees("synthetic", synthetic, synthetic.cols - 30, synthetic.rows - 20, gcE, dpE, en);
        TestCarveAgrees("tower/2", towerSmall, towerSmall.cols - 30, towerSmall.rows - 15, gcE, dpE, en);
    }
    TestCarveAgrees("tower", tower, tower.cols - 40, tower.rows, GraphCut::EnergyType::Backward,
                    DP::EnergyType::Backward, "backward");

    std::printf("\n%d checks passed, %d failed\n", g_passed, g_failed);
    return g_failed ? 1 : 0;
}
