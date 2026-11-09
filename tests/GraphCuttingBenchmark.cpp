// Benchmark + visual comparison for GraphCutting.cpp. Build from the repo root:
//   g++ -std=c++14 -O2 -I. tests/GraphCuttingBenchmark.cpp GraphCutting.cpp -o gc_bench `pkg-config --cflags --libs opencv4`
//   ./gc_bench            (writes docs/GraphCutting/results/)
// Then draw the timing chart:  python tools/plot_graphcut_timing.py
//
// DP and Greedy below are small REFERENCE versions so that Graph Cut can be
// compared on identical cost maps before the team's DynamicProgramming.cpp /
// GreedyAlgorithm.cpp are ready. Swap them in via the SeamFinder type.
#include "GraphCutting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace GraphCut;

static const std::string OUT = "docs/GraphCutting/results/";

struct Seam { std::vector<int> cols; int64_t cost; };
using SeamFinder = std::function<Seam(const cv::Mat&)>;

// ---------------------------------------------------------------------------
// Reference finders (vertical seam on a CV_32S cost map)
// ---------------------------------------------------------------------------
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

static Seam RefGreedy(const cv::Mat& c)
{
    const int m = c.rows, n = c.cols;
    Seam s{ std::vector<int>(m), 0 };
    int j = 0;
    for (int k = 1; k < n; ++k)
        if (c.at<int>(0, k) < c.at<int>(0, j)) j = k;
    s.cols[0] = j;
    s.cost = c.at<int>(0, j);
    for (int i = 1; i < m; ++i)
    {
        int best = std::max(j - 1, 0);
        for (int k = best + 1; k <= std::min(j + 1, n - 1); ++k)
            if (c.at<int>(i, k) < c.at<int>(i, best)) best = k;
        j = best;
        s.cols[i] = j;
        s.cost += c.at<int>(i, j);
    }
    return s;
}

static SeamFinder GC(MaxFlowAlgorithm algo)
{
    return [algo](const cv::Mat& c) {
        SeamResult r = FindVerticalSeam(c, algo);
        return Seam{ r.seam, r.cost };
    };
}

// ---------------------------------------------------------------------------
static double Seconds(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void SingleSeamTiming(const cv::Mat& img)
{
    struct M { const char* name; SeamFinder f; int maxWidth; int reps; };
    const std::vector<M> methods = {
        { "DP", RefDP, 100000, 5 },
        { "Greedy", RefGreedy, 100000, 5 },
        { "GC Dinic", GC(MaxFlowAlgorithm::Dinic), 100000, 1 },
        { "GC Edmonds-Karp", GC(MaxFlowAlgorithm::EdmondsKarp), 400, 1 },
    };
    FILE* f = std::fopen((OUT + "single_seam_timing.csv").c_str(), "w");
    std::fprintf(f, "method,width,height,pixels,seconds,seam_cost\n");
    std::printf("Single-seam timing\n");
    for (int w : { 30, 45, 68, 100, 150, 225, 274, 400, 600, 900 })
    {
        const int h = static_cast<int>(std::lround(double(w) * img.rows / img.cols));
        cv::Mat small;
        cv::resize(img, small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        const cv::Mat cost = ComputeEnergy(small);
        for (const M& m : methods)
        {
            if (w > m.maxWidth) continue;
            double best = 1e18;
            int64_t sc = 0;
            for (int r = 0; r < m.reps; ++r)
            {
                const auto t0 = std::chrono::steady_clock::now();
                sc = m.f(cost).cost;
                best = std::min(best, Seconds(t0));
            }
            std::fprintf(f, "%s,%d,%d,%d,%.9f,%lld\n", m.name, w, h, w * h, best, (long long)sc);
            std::printf("  %-16s %4dx%-4d %11.3f ms  cost=%lld\n", m.name, w, h, best * 1e3, (long long)sc);
        }
    }
    std::fclose(f);
}

static void MemoryUsage(const cv::Mat& img)
{
    FILE* f = std::fopen((OUT + "memory_usage.csv").c_str(), "w");
    std::fprintf(f, "width,height,pixels,nodes,arcs,gc_bytes,gc_bytes_per_pixel,dp_bytes,dp_bytes_per_pixel\n");
    std::printf("Memory usage\n");
    for (int w : { 68, 150, 274, 600, 1200, 1920 })
    {
        const int h = static_cast<int>(std::lround(double(w) * img.rows / img.cols));
        cv::Mat small;
        cv::resize(img, small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        int64_t s, t;
        const FlowNetwork net = BuildSeamGraph(ComputeEnergy(small), s, t);
        const double px = double(w) * h;
        const size_t gc = net.MemoryBytes();
        const size_t dp = static_cast<size_t>(px) * (sizeof(int64_t) + sizeof(int));  // M table + back-pointers
        std::fprintf(f, "%d,%d,%.0f,%lld,%lld,%zu,%.1f,%zu,%.1f\n", w, h, px, (long long)net.NumNodes(),
                     (long long)net.NumArcs(), gc, gc / px, dp, dp / px);
        std::printf("  %4dx%-4d V=%8lld arcs=%9lld  GC %8.2f MiB (%.1f B/px)  DP %6.2f MiB (%.1f B/px)\n", w, h,
                    (long long)net.NumNodes(), (long long)net.NumArcs(), gc / 1048576.0, gc / px,
                    dp / 1048576.0, dp / px);
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Full resize: remove k vertical seams with each method on the same image.
// ---------------------------------------------------------------------------
struct CarveResult { cv::Mat out; double seconds; int64_t totalCost; std::vector<std::vector<cv::Point>> seams; };

static CarveResult CarveWith(const cv::Mat& img, int k, const SeamFinder& finder)
{
    CarveResult r{ img.clone(), 0.0, 0, {} };
    cv::Mat coords(img.rows, img.cols, CV_32SC2);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) coords.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
    for (int step = 0; step < k; ++step)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Seam s = finder(ComputeEnergy(r.out));
        r.seconds += Seconds(t0);
        r.totalCost += s.cost;
        std::vector<cv::Point> pts(r.out.rows);
        for (int i = 0; i < r.out.rows; ++i)
        {
            const cv::Vec2i p = coords.at<cv::Vec2i>(i, s.cols[i]);
            pts[i] = cv::Point(p[0], p[1]);
        }
        r.seams.push_back(pts);
        RemoveVerticalSeam(r.out, s.cols);
        RemoveVerticalSeam(coords, s.cols);
    }
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

static void FullResize(const std::string& name, const cv::Mat& img, int k, FILE* csv)
{
    struct M { const char* name; SeamFinder f; };
    const std::vector<M> methods = { { "DP", RefDP }, { "Greedy", RefGreedy },
                                     { "GC Dinic", GC(MaxFlowAlgorithm::Dinic) } };
    std::vector<cv::Mat> results{ Label(img, "original") }, overlays;
    std::vector<std::vector<std::vector<cv::Point>>> allSeams;
    for (const M& m : methods)
    {
        const CarveResult r = CarveWith(img, k, m.f);
        std::fprintf(csv, "%s,%s,%dx%d,%d,%.6f,%lld\n", name.c_str(), m.name, img.cols, img.rows, k,
                     r.seconds, (long long)r.totalCost);
        std::printf("  %-10s %-10s %d seams: %8.3f s  total cost=%lld\n", name.c_str(), m.name, k,
                    r.seconds, (long long)r.totalCost);
        results.push_back(Label(r.out, m.name));
        cv::Mat vis = img.clone();
        for (size_t s = 0; s < r.seams.size() && s < 30; ++s)
            for (const cv::Point& p : r.seams[s]) vis.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
        overlays.push_back(Label(vis, std::string(m.name) + ": first 30 seams"));
        allSeams.push_back(r.seams);
    }
    cv::imwrite(OUT + "compare_" + name + ".png", SideBySide(results));
    cv::imwrite(OUT + "seams_" + name + ".png", SideBySide(overlays));

    // How many of the k seams are identical between DP and graph cut?
    // (Each method carves its own image, so after a divergence they may differ.)
    int identical = 0;
    for (int s = 0; s < k; ++s) identical += (allSeams[0][s] == allSeams[2][s]);
    std::printf("  %-10s DP vs GC: %d/%d seams identical\n", name.c_str(), identical, k);
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
    MemoryUsage(tower);

    std::printf("First-seam cost\n");
    FILE* f = std::fopen((OUT + "seam_cost.csv").c_str(), "w");
    std::fprintf(f, "image,method,seam_cost,augmentations,phases\n");
    for (const auto& item : { std::make_pair(std::string("tower"), tower), std::make_pair(std::string("synthetic"), synthetic) })
    {
        const cv::Mat c = ComputeEnergy(item.second);
        const SeamResult d = FindVerticalSeam(c, MaxFlowAlgorithm::Dinic);
        const SeamResult e = FindVerticalSeam(c, MaxFlowAlgorithm::EdmondsKarp);
        std::fprintf(f, "%s,DP,%lld,,\n", item.first.c_str(), (long long)RefDP(c).cost);
        std::fprintf(f, "%s,Greedy,%lld,,\n", item.first.c_str(), (long long)RefGreedy(c).cost);
        std::fprintf(f, "%s,GC Dinic,%lld,%lld,%lld\n", item.first.c_str(), (long long)d.cost,
                     (long long)d.augmentations, (long long)d.phases);
        std::fprintf(f, "%s,GC Edmonds-Karp,%lld,%lld,\n", item.first.c_str(), (long long)e.cost,
                     (long long)e.augmentations);
        std::printf("  %-10s DP %lld  Greedy %lld  GC %lld (Dinic: %lld aug, %lld phases; EK: %lld aug)\n",
                    item.first.c_str(), (long long)RefDP(c).cost, (long long)RefGreedy(c).cost,
                    (long long)d.cost, (long long)d.augmentations, (long long)d.phases, (long long)e.augmentations);
    }
    std::fclose(f);

    std::printf("Full resize\n");
    f = std::fopen((OUT + "full_resize.csv").c_str(), "w");
    std::fprintf(f, "image,method,size,seams_removed,seconds,total_cost\n");
    FullResize("tower", tower, 80, f);
    FullResize("synthetic", synthetic, 50, f);
    std::fclose(f);

    // Both directions with graph cut, for the README figure.
    CarveStats st;
    const cv::Mat both = Carve(tower, 194, 150, MaxFlowAlgorithm::Dinic, EnergyType::Backward, &st, true);
    cv::imwrite(OUT + "tower_both_directions.png",
                SideBySide({ Label(DrawSeams(tower, st), "80 vertical (red), 36 horiz. (green)"),
                             Label(both, "result 194x150") }));
    std::printf("Both directions 274x186 -> 194x150: %.2f s, total cost %lld\n", st.seamSeconds,
                (long long)st.totalCost);
    std::printf("done -> %s\n", OUT.c_str());
    return 0;
}
