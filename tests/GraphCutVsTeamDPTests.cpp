// Cross-check: Graph Cut (GraphCutting.cpp) against the TEAM's DP
// (DynamicProgramming.cpp).
//
// Build and run from the repo root:
//   cmake --build --preset vs2022-release --target gc_vs_dp
//   out\build\vs2022\Release\gc_vs_dp.exe     (or: ctest --preset vs2022-release)
// or with g++:
//   g++ -std=c++14 -O2 -I. tests/GraphCutVsTeamDPTests.cpp SeamCarving.cpp GraphCutting.cpp DynamicProgramming.cpp -o gc_vs_dp `pkg-config --cflags --libs opencv4`
//
// What is checked:
//   1. Single seams on random cost maps: graph-cut cost == DP optimum, for
//      Dinic and Edmonds-Karp, vertical and horizontal. The GC seam is also
//      re-scored with DP::SeamEnergy, which throws if the seam is invalid.
//   2. Full carves on real images: GC::Carve and DP::Carve remove seams of
//      equal cost at every step and produce pixel-identical output.
// (Both modules use the one shared SeamCarving::ComputeEnergy, so their
// energy maps are identical by construction; see tests/DynamicProgrammingTests.cpp.)
//
// By the max-flow min-cut theorem, step 1 must hold exactly. Equal seams and
// identical images (step 2) additionally rely on both sides breaking ties
// towards the leftmost seam; costs would match even without that.
#include "Check.h"
#include "GraphCutting.h"
#include "DynamicProgramming.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
static void TestRandomSeams()
{
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
    std::printf("  %d random maps x {Dinic, EK} x {vertical, horizontal}\n", cases);
}

static void TestCarveAgrees(const std::string& name, const cv::Mat& img, int newW, int newH,
                            SeamCarving::EnergyType energy, const char* eName)
{
    SeamCarving::CarveStats gs, ds;
    const cv::Mat gOut = GraphCut::Carve(img, newW, newH, GraphCut::MaxFlowAlgorithm::Dinic, energy, &gs, true);
    const cv::Mat dOut = DP::Carve(img, newW, newH, energy, &ds, true);

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
                (long long)gs.totalCost, (long long)ds.totalCost, SameMat(gOut, dOut) ? "identical" : "DIFFERENT");
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

    return RunTests({
        { "random cost maps: GC seam cost == team DP optimum", TestRandomSeams },
        { "full carve: GraphCut::Carve vs DP::Carve, energy recomputed each step", [&] {
              using SeamCarving::EnergyType;
              cv::Mat towerSmall;
              cv::resize(tower, towerSmall, cv::Size(137, 93), 0, 0, cv::INTER_AREA);
              for (EnergyType e : { EnergyType::Backward, EnergyType::Forward })
              {
                  const char* en = e == EnergyType::Forward ? "forward" : "backward";
                  TestCarveAgrees("synthetic", synthetic, synthetic.cols - 30, synthetic.rows - 20, e, en);
                  TestCarveAgrees("tower/2", towerSmall, towerSmall.cols - 30, towerSmall.rows - 15, e, en);
              }
              TestCarveAgrees("tower", tower, tower.cols - 40, tower.rows, EnergyType::Backward, "backward");
          } },
    });
}
