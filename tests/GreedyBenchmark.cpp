// Benchmark + raw comparison data for GreedyAlgorithm.cpp. Build from the repo root
// (CMake installs OpenCV itself through vcpkg on the first configure):
//   Windows:       cmake --preset vs2022
//                  cmake --build --preset vs2022-release --target greedy_bench
//                  out\build\vs2022\Release\greedy_bench.exe
//   Linux / macOS: cmake --preset unix-release
//                  cmake --build --preset unix-release --target greedy_bench
//                  out/build/unix-release/greedy_bench
// Run it from the repository root (writes docs/Greedy/results/).
// Then draw the timing chart:  python tools/plot_greedy_timing.py
//
// "DP (ref)" below is a small REFERENCE dynamic-programming seam finder so the
// greedy seams can be compared before DynamicProgramming.cpp step (b) exists.
// When DP::FindVerticalSeam is available, replace RefDP with it. Graph Cut
// (Dinic) is included as an independent check: it is optimal too, so its seam
// cost must equal DP's.
//
// Outputs (docs/Greedy/results/):
//   single_seam_timing.csv  time to find ONE vertical seam vs image size
//   seam_cost.csv           first-seam cost: greedy vs optimal, both energies
//   full_resize.csv         time + total cost of a full resize per method
//   seam_costs.csv          cost of every removed seam, per method (for plots)
//   compare_<img>.png       original | Greedy | DP | Graph Cut results
//   seams_<img>.png         removed seams drawn on the original, per method
//   greedy_both_directions.png  greedy, vertical + horizontal seams
#include "DynamicProgramming.h"
#include "GraphCutting.h"
#include "GreedyAlgorithm.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

static const std::string OUT = "docs/Greedy/results/";

struct Seam { std::vector<int> cols; int64_t cost; };
using SeamFinder = std::function<Seam(const cv::Mat&)>;

// ---------------------------------------------------------------------------
// Seam finders (vertical seam on a CV_32S energy map)
// ---------------------------------------------------------------------------
static Seam GreedySeam(const cv::Mat& c)
{
    Greedy::SeamResult r = Greedy::FindVerticalSeam(c);
    return Seam{ r.seam, r.cost };
}

static Seam RefDP(const cv::Mat& c)
{
    const int m = c.rows, n = c.cols;
    std::vector<int64_t> M(static_cast<size_t>(m) * n);
    std::vector<int> back(static_cast<size_t>(m) * n, 0);
    for (int j = 0; j < n; ++j) M[j] = c.at<int>(0, j);
    for (int i = 1; i < m; ++i)
    {
        const int* row = c.ptr<int>(i);
        for (int j = 0; j < n; ++j)
        {
            int best = -1;   // ties -> leftmost
            for (int k = std::max(j - 1, 0); k <= std::min(j + 1, n - 1); ++k)
                if (best < 0 || M[(i - 1) * n + k] < M[(i - 1) * n + best]) best = k;
            M[i * n + j] = row[j] + M[(i - 1) * n + best];
            back[i * n + j] = best;
        }
    }
    int j = 0;
    for (int k = 1; k < n; ++k)
        if (M[(m - 1) * n + k] < M[(m - 1) * n + j]) j = k;
    Seam s{ std::vector<int>(m), M[(m - 1) * n + j] };
    for (int i = m - 1; i >= 0; --i) { s.cols[i] = j; j = back[i * n + j]; }
    return s;
}

static Seam GCSeam(const cv::Mat& c)
{
    GraphCut::SeamResult r = GraphCut::FindVerticalSeam(c, GraphCut::MaxFlowAlgorithm::Dinic);
    return Seam{ r.seam, r.cost };
}

// ---------------------------------------------------------------------------
static double Seconds(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static cv::Mat Shrink(const cv::Mat& img, int w)
{
    const int h = static_cast<int>(std::lround(double(w) * img.rows / img.cols));
    cv::Mat small;
    cv::resize(img, small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    return small;
}

static const char* EnergyName(DP::EnergyType e)
{
    return e == DP::EnergyType::Forward ? "forward" : "backward";
}

// ---------------------------------------------------------------------------
// 1. Time to find one seam (energy map excluded) vs image size
// ---------------------------------------------------------------------------
static void SingleSeamTiming(const cv::Mat& img)
{
    struct M { const char* name; SeamFinder f; int maxWidth; double minSeconds; };
    // Each method is repeated until at least minSeconds have passed and the
    // AVERAGE per call is reported, so microsecond greedy times are stable.
    const std::vector<M> methods = {
        { "Greedy", GreedySeam, 100000, 0.05 },
        { "DP (ref)", RefDP, 100000, 0.05 },
        { "GC Dinic", GCSeam, 600, 0.0 },
    };
    FILE* f = std::fopen((OUT + "single_seam_timing.csv").c_str(), "w");
    std::fprintf(f, "method,width,height,pixels,seconds,seam_cost\n");
    std::printf("Single-seam timing (energy map excluded)\n");
    for (int w : { 30, 45, 68, 100, 150, 225, 274, 400, 600, 900, 1350, 2000 })
    {
        const cv::Mat cost = DP::ComputeEnergy(Shrink(img, w));
        const int h = cost.rows;
        for (const M& m : methods)
        {
            if (w > m.maxWidth) continue;
            int reps = 0;
            int64_t sc = 0;
            const auto t0 = std::chrono::steady_clock::now();
            do { sc = m.f(cost).cost; ++reps; } while (Seconds(t0) < m.minSeconds);
            const double per = Seconds(t0) / reps;
            std::fprintf(f, "%s,%d,%d,%d,%.9f,%lld\n", m.name, w, h, w * h, per, (long long)sc);
            std::printf("  %-10s %4dx%-4d %12.4f ms  cost=%lld\n", m.name, w, h, per * 1e3, (long long)sc);
        }
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// 2. First-seam cost: how far from optimal is the greedy seam?
// ---------------------------------------------------------------------------
static void FirstSeamCost(const std::vector<std::pair<std::string, cv::Mat>>& images)
{
    FILE* f = std::fopen((OUT + "seam_cost.csv").c_str(), "w");
    std::fprintf(f, "image,energy,greedy_cost,dp_cost,gc_cost,greedy_over_optimal\n");
    std::printf("First-seam cost (greedy vs optimal)\n");
    for (const auto& item : images)
        for (DP::EnergyType e : { DP::EnergyType::Backward, DP::EnergyType::Forward })
        {
            const cv::Mat c = DP::ComputeEnergy(item.second, e);
            const int64_t g = GreedySeam(c).cost, d = RefDP(c).cost, gc = GCSeam(c).cost;
            const double ratio = d > 0 ? double(g) / d : 1.0;
            std::fprintf(f, "%s,%s,%lld,%lld,%lld,%.4f\n", item.first.c_str(), EnergyName(e),
                         (long long)g, (long long)d, (long long)gc, ratio);
            std::printf("  %-10s %-8s greedy %8lld  DP %8lld  GC %8lld  (greedy = %.2fx optimal)\n",
                        item.first.c_str(), EnergyName(e), (long long)g, (long long)d, (long long)gc, ratio);
            if (g < d) std::printf("  !! greedy cheaper than DP - one of them is wrong\n");
            if (d != gc) std::printf("  !! DP and Graph Cut disagree on the optimal cost\n");
        }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// 3. Full resize: remove k vertical seams with each method on the same image
// ---------------------------------------------------------------------------
struct CarveResult
{
    cv::Mat out;
    double seamSeconds = 0.0, totalSeconds = 0.0;
    int64_t totalCost = 0;
    std::vector<int64_t> seamCosts;
    std::vector<std::vector<cv::Point>> seams;
};

// Generic carving loop for a SeamFinder (used for the reference DP).
static CarveResult CarveWith(const cv::Mat& img, int k, const SeamFinder& finder)
{
    const auto start = std::chrono::steady_clock::now();
    CarveResult r;
    r.out = img.clone();
    cv::Mat coords(img.rows, img.cols, CV_32SC2);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) coords.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
    for (int step = 0; step < k; ++step)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Seam s = finder(DP::ComputeEnergy(r.out));
        r.seamSeconds += Seconds(t0);
        r.totalCost += s.cost;
        r.seamCosts.push_back(s.cost);
        std::vector<cv::Point> pts(r.out.rows);
        for (int i = 0; i < r.out.rows; ++i)
        {
            const cv::Vec2i p = coords.at<cv::Vec2i>(i, s.cols[i]);
            pts[i] = cv::Point(p[0], p[1]);
        }
        r.seams.push_back(pts);
        Greedy::RemoveVerticalSeam(r.out, s.cols);
        Greedy::RemoveVerticalSeam(coords, s.cols);
    }
    r.totalSeconds = Seconds(start);
    return r;
}

static CarveResult CarveGreedy(const cv::Mat& img, int k)
{
    Greedy::CarveStats st;
    CarveResult r;
    r.out = Greedy::Carve(img, img.cols - k, img.rows, Greedy::EnergyType::Backward, &st, true);
    r.seamSeconds = st.seamSeconds;
    r.totalSeconds = st.totalSeconds;
    r.totalCost = st.totalCost;
    r.seamCosts = st.seamCosts;
    r.seams = st.verticalSeams;
    return r;
}

static CarveResult CarveGC(const cv::Mat& img, int k)
{
    const auto start = std::chrono::steady_clock::now();
    GraphCut::CarveStats st;
    CarveResult r;
    r.out = GraphCut::Carve(img, img.cols - k, img.rows, GraphCut::MaxFlowAlgorithm::Dinic,
                            GraphCut::EnergyType::Backward, &st, true);
    r.seamSeconds = st.seamSeconds;
    r.totalSeconds = Seconds(start);
    r.totalCost = st.totalCost;
    r.seamCosts = st.seamCosts;
    r.seams = st.verticalSeams;
    return r;
}

static cv::Mat Label(const cv::Mat& img, const std::string& text)
{
    cv::Mat pad(22, img.cols, img.type(), cv::Scalar(255, 255, 255));
    cv::putText(pad, text, cv::Point(4, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    cv::Mat out;
    cv::vconcat(pad, img, out);
    return out;
}

static cv::Mat SideBySide(const std::vector<cv::Mat>& images, int gap = 8)
{
    int h = 0, w = 0;
    for (const cv::Mat& m : images) { h = std::max(h, m.rows); w += m.cols + gap; }
    cv::Mat canvas(h, w - gap, CV_8UC3, cv::Scalar(255, 255, 255));
    int x = 0;
    for (const cv::Mat& m : images)
    {
        m.copyTo(canvas(cv::Rect(x, 0, m.cols, m.rows)));
        x += m.cols + gap;
    }
    return canvas;
}

static void FullResize(const std::string& name, const cv::Mat& img, int k, FILE* csv, FILE* perSeam)
{
    struct M { const char* name; std::function<CarveResult()> run; };
    const std::vector<M> methods = {
        { "Greedy", [&] { return CarveGreedy(img, k); } },
        { "DP (ref)", [&] { return CarveWith(img, k, RefDP); } },
        { "GC Dinic", [&] { return CarveGC(img, k); } },
    };
    std::vector<cv::Mat> results{ Label(img, "original") }, overlays;
    std::vector<std::vector<std::vector<cv::Point>>> allSeams;
    for (const M& m : methods)
    {
        const CarveResult r = m.run();
        std::fprintf(csv, "%s,%s,%dx%d,%d,%.6f,%.6f,%lld\n", name.c_str(), m.name, img.cols, img.rows, k,
                     r.seamSeconds, r.totalSeconds, (long long)r.totalCost);
        for (size_t s = 0; s < r.seamCosts.size(); ++s)
            std::fprintf(perSeam, "%s,%s,%zu,%lld\n", name.c_str(), m.name, s + 1, (long long)r.seamCosts[s]);
        std::printf("  %-10s %-9s %d seams: %9.4f s finding seams, %9.4f s total, total cost=%lld\n",
                    name.c_str(), m.name, k, r.seamSeconds, r.totalSeconds, (long long)r.totalCost);
        results.push_back(Label(r.out, m.name));
        cv::Mat vis = img.clone();
        for (const auto& seam : r.seams)
            for (const cv::Point& p : seam) vis.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
        overlays.push_back(Label(vis, std::string(m.name) + ": all " + std::to_string(k) + " seams"));
        allSeams.push_back(r.seams);
    }
    cv::imwrite(OUT + "compare_" + name + ".png", SideBySide(results));
    cv::imwrite(OUT + "seams_" + name + ".png", SideBySide(overlays));

    // How many removed seams does greedy share exactly with DP?
    // (Each method carves its own image, so after one divergence they drift.)
    int identical = 0;
    for (int s = 0; s < k; ++s) identical += (allSeams[0][s] == allSeams[1][s]);
    std::printf("  %-10s Greedy vs DP: %d/%d seams identical\n", name.c_str(), identical, k);
}

int main()
{
    const cv::Mat tower = cv::imread("images/tower.png");
    const cv::Mat synthetic = cv::imread("images/synthetic.png");
    if (tower.empty() || synthetic.empty())
    {
        std::printf("run from the repo root (images/tower.png, images/synthetic.png)\n");
        return 1;
    }
    if (FILE* probe = std::fopen((OUT + "single_seam_timing.csv").c_str(), "w")) std::fclose(probe);
    else { std::printf("cannot write to %s (create the folder first)\n", OUT.c_str()); return 1; }

    SingleSeamTiming(tower);
    FirstSeamCost({ { "tower", tower }, { "synthetic", synthetic } });

    std::printf("Full resize (vertical seams, backward energy)\n");
    FILE* f = std::fopen((OUT + "full_resize.csv").c_str(), "w");
    FILE* ps = std::fopen((OUT + "seam_costs.csv").c_str(), "w");
    std::fprintf(f, "image,method,size,seams_removed,seam_seconds,total_seconds,total_cost\n");
    std::fprintf(ps, "image,method,seam_index,seam_cost\n");
    FullResize("tower", tower, 80, f, ps);
    FullResize("synthetic", synthetic, 50, f, ps);
    std::fclose(f);
    std::fclose(ps);

    // Greedy in both directions, for the README figure.
    const int w = tower.cols * 7 / 10, h = tower.rows * 8 / 10;
    Greedy::CarveStats st;
    const cv::Mat both = Greedy::Carve(tower, w, h, Greedy::EnergyType::Backward, &st, true);
    cv::imwrite(OUT + "greedy_both_directions.png",
                SideBySide({ Label(Greedy::DrawSeams(tower, st),
                                   std::to_string(tower.cols - w) + " vertical (red), " +
                                   std::to_string(tower.rows - h) + " horiz. (green)"),
                             Label(both, "greedy " + std::to_string(w) + "x" + std::to_string(h)) }));
    std::printf("Greedy both directions %dx%d -> %dx%d: %.3f s finding seams, total cost %lld\n",
                tower.cols, tower.rows, w, h, st.seamSeconds, (long long)st.totalCost);
    std::printf("done -> %s\n", OUT.c_str());
    return 0;
}
