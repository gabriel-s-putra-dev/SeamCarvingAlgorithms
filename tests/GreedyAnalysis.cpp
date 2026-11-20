// Seam-cost and visual-quality analysis for the Greedy write-up (Person B).
// Complements tests/GreedyBenchmark.cpp (timing, first-seam cost, side-by-side
// results). Build and run from the repository root:
//   Windows:       cmake --build --preset vs2022-release --target greedy_analysis
//                  out\build\vs2022\Release\greedy_analysis.exe
//   Linux / macOS: cmake --build --preset unix-release --target greedy_analysis
//                  out/build/unix-release/greedy_analysis
// Then draw the charts:  python tools/plot_greedy_analysis.py
//
// Outputs (docs/Greedy/results/):
//   random_maps.csv          greedy / optimal seam cost on random cost maps, by size
//   quality.csv              how much "important" content each method removes
//   zoom_tower.png           the tower region of each result, enlarged 2x
//   compare_tower_forward.png  Greedy vs DP with the forward energy
//
// "Optimal" is a small reference DP (same recurrence as the team's DP and the
// same cost as graph cut, see tests/GraphCuttingTests.cpp).
#include "DynamicProgramming.h"
#include "GreedyAlgorithm.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

static const std::string OUT = "docs/Greedy/results/";

// Opens OUT + name for writing. Exits with a clear message instead of crashing
// if the file cannot be opened (e.g. the CSV is open in Excel, which locks it).
static FILE* OpenCsv(const std::string& name)
{
    FILE* f = std::fopen((OUT + name).c_str(), "w");
    if (!f)
    {
        std::printf("cannot write %s%s (is it open in another program?)\n", OUT.c_str(), name.c_str());
        std::exit(1);
    }
    return f;
}

// ---------------------------------------------------------------------------
// Reference DP seam (leftmost optimum on ties) on a CV_32S cost map.
// ---------------------------------------------------------------------------
static Greedy::SeamResult RefDP(const cv::Mat& c)
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
            int best = -1;
            for (int k = std::max(j - 1, 0); k <= std::min(j + 1, n - 1); ++k)
                if (best < 0 || M[(i - 1) * n + k] < M[(i - 1) * n + best]) best = k;
            M[i * n + j] = row[j] + M[(i - 1) * n + best];
            back[i * n + j] = best;
        }
    }
    int j = 0;
    for (int k = 1; k < n; ++k)
        if (M[(m - 1) * n + k] < M[(m - 1) * n + j]) j = k;
    Greedy::SeamResult r;
    r.cost = M[(m - 1) * n + j];
    r.seam.resize(m);
    for (int i = m - 1; i >= 0; --i) { r.seam[i] = j; j = back[i * n + j]; }
    return r;
}

// ---------------------------------------------------------------------------
// 1. Random cost maps: how far from optimal is one greedy seam?
// ---------------------------------------------------------------------------
// Two kinds of map:
//   uniform - every pixel independent in [0, 255]: no structure at all;
//   smooth  - uniform noise blurred, so cheap pixels cluster in regions like
//             in a real energy map.
static void RandomMaps()
{
    std::mt19937 rng(7);
    FILE* f = OpenCsv("random_maps.csv");
    std::fprintf(f, "map,size,trials,mean_ratio,median_ratio,max_ratio,pct_optimal\n");
    std::printf("Random maps: greedy cost / optimal cost (first seam)\n");
    for (const char* kind : { "uniform", "smooth" })
        for (int s : { 4, 8, 16, 32, 64, 128, 256 })
        {
            const int trials = 300;
            std::vector<double> ratios;
            int optimal = 0;
            for (int t = 0; t < trials; ++t)
            {
                cv::Mat noise(s, s, CV_32F);
                for (int i = 0; i < s; ++i)
                    for (int j = 0; j < s; ++j) noise.at<float>(i, j) = float(rng() % 256);
                if (std::string(kind) == "smooth")
                {
                    cv::GaussianBlur(noise, noise, cv::Size(0, 0), std::max(1.0, s / 16.0));
                    cv::normalize(noise, noise, 0, 255, cv::NORM_MINMAX);
                }
                cv::Mat c;
                noise.convertTo(c, CV_32S);
                const int64_t g = Greedy::FindVerticalSeam(c).cost, d = RefDP(c).cost;
                ratios.push_back(d > 0 ? double(g) / d : 1.0);
                optimal += (g == d);
            }
            std::sort(ratios.begin(), ratios.end());
            double sum = 0;
            for (double r : ratios) sum += r;
            const double mean = sum / trials, median = ratios[trials / 2], mx = ratios.back();
            const double pct = 100.0 * optimal / trials;
            std::fprintf(f, "%s,%d,%d,%.4f,%.4f,%.4f,%.1f\n", kind, s, trials, mean, median, mx, pct);
            std::printf("  %-7s %3dx%-3d mean %.2fx  median %.2fx  max %.2fx  optimal %5.1f%%\n",
                        kind, s, s, mean, median, mx, pct);
        }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// 2. Visual quality: what does each method cut through?
// ---------------------------------------------------------------------------
struct Carved
{
    cv::Mat out;
    int64_t totalCost = 0;
    std::vector<std::vector<cv::Point>> seams;   // original coordinates
};

using Finder = std::function<Greedy::SeamResult(const cv::Mat&)>;

// Remove k vertical seams with the given seam finder, recording each seam in
// original-image coordinates (same bookkeeping as Greedy::Carve).
static Carved CarveVertical(const cv::Mat& img, int k, DP::EnergyType energy, const Finder& find)
{
    Carved r;
    r.out = img.clone();
    cv::Mat coords(img.rows, img.cols, CV_32SC2);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) coords.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
    for (int s = 0; s < k; ++s)
    {
        const Greedy::SeamResult seam = find(DP::ComputeEnergy(r.out, energy));
        r.totalCost += seam.cost;
        std::vector<cv::Point> pts(r.out.rows);
        for (int i = 0; i < r.out.rows; ++i)
        {
            const cv::Vec2i p = coords.at<cv::Vec2i>(i, seam.seam[i]);
            pts[i] = cv::Point(p[0], p[1]);
        }
        r.seams.push_back(pts);
        Greedy::RemoveVerticalSeam(r.out, seam.seam);
        Greedy::RemoveVerticalSeam(coords, seam.seam);
    }
    return r;
}

// Energy value at the given percentile (0..100) of the map.
static int Percentile(const cv::Mat& energy, double pct)
{
    std::vector<int> v(energy.begin<int>(), energy.end<int>());
    const size_t idx = std::min(v.size() - 1, size_t(pct / 100.0 * v.size()));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

// Quality metrics for a set of removed pixels, measured on the ORIGINAL image's
// backward energy (so all methods and both energy types use the same yardstick):
//   mean_energy   - average original energy of the removed pixels;
//   pct_top10     - % of removed pixels that were in the top 10 % of energy
//                   (edges / detail: the content seam carving should keep);
//   pct_edge      - % of removed pixels on a strong edge, energy >= 100. For a
//                   step edge Sobel gives |Gx| = 4h, so 100 is a step of ~25
//                   grey levels. Unlike pct_top10 this does not depend on how
//                   much of the image is flat (the synthetic image is ~90 % flat);
//   cols_touched  - how many original columns lost at least one pixel. Low =
//                   seams stack in a few low-energy bands; high = spread out.
static void Quality(FILE* f, const std::string& image, const char* energyName, const char* method,
                    const cv::Mat& original, const Carved& r)
{
    const cv::Mat e = DP::ComputeEnergy(original, DP::EnergyType::Backward);
    const int top10 = Percentile(e, 90.0);
    int64_t sum = 0, n = 0, high = 0, edge = 0;
    std::vector<char> touched(original.cols, 0);
    for (const auto& seam : r.seams)
        for (const cv::Point& p : seam)
        {
            const int v = e.at<int>(p);
            sum += v;
            high += (v >= top10);
            edge += (v >= 100);
            touched[p.x] = 1;
            ++n;
        }
    int cols = 0;
    for (char t : touched) cols += t;
    const double mean = double(sum) / n, pct = 100.0 * high / n, pctEdge = 100.0 * edge / n;
    std::fprintf(f, "%s,%s,%s,%zu,%lld,%.2f,%.2f,%.2f,%d\n", image.c_str(), energyName, method, r.seams.size(),
                 (long long)r.totalCost, mean, pct, pctEdge, cols);
    std::printf("  %-9s %-8s %-6s total cost %8lld  mean energy %6.1f  top-10%% %5.1f%%  edge %5.2f%% (%lld px)"
                "  cols touched %d\n",
                image.c_str(), energyName, method, (long long)r.totalCost, mean, pct, pctEdge,
                (long long)edge, cols);
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

static cv::Mat Zoom(const cv::Mat& img, const cv::Rect& roi, int scale)
{
    cv::Mat z;
    cv::resize(img(roi & cv::Rect(0, 0, img.cols, img.rows)), z, cv::Size(), scale, scale, cv::INTER_NEAREST);
    return z;
}

static void VisualQuality(const cv::Mat& tower, const cv::Mat& synthetic)
{
    const Finder greedy = [](const cv::Mat& c) { return Greedy::FindVerticalSeam(c); };
    const Finder dp = RefDP;

    FILE* f = OpenCsv("quality.csv");
    std::fprintf(f, "image,energy,method,seams,total_cost,mean_energy_removed,pct_top10_removed,pct_edge_removed,cols_touched\n");
    std::printf("Visual quality (metrics on the original image's backward energy)\n");

    struct Job { std::string name; cv::Mat img; int k; };
    for (const Job& job : { Job{ "tower", tower, 80 }, Job{ "synthetic", synthetic, 50 } })
        for (DP::EnergyType energy : { DP::EnergyType::Backward, DP::EnergyType::Forward })
        {
            const char* en = energy == DP::EnergyType::Forward ? "forward" : "backward";
            const Carved g = CarveVertical(job.img, job.k, energy, greedy);
            const Carved d = CarveVertical(job.img, job.k, energy, dp);
            Quality(f, job.name, en, "Greedy", job.img, g);
            Quality(f, job.name, en, "DP", job.img, d);

            if (job.name == "tower" && energy == DP::EnergyType::Backward)
            {
                // The tower sits in the right half; after 80 seams it is at
                // roughly x = 70..194 in every result. Crop and enlarge 2x.
                const cv::Rect roi(70, 10, 124, 170);
                cv::imwrite(OUT + "zoom_tower.png",
                            SideBySide({ Label(Zoom(g.out, roi, 2), "Greedy (backward), 2x"),
                                         Label(Zoom(d.out, roi, 2), "DP (backward), 2x") }));
            }
            if (job.name == "tower" && energy == DP::EnergyType::Forward)
            {
                cv::imwrite(OUT + "compare_tower_forward.png",
                            SideBySide({ Label(job.img, "original"),
                                         Label(g.out, "Greedy (forward)"),
                                         Label(d.out, "DP (forward)") }));
            }
        }
    std::fclose(f);
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
    if (FILE* probe = std::fopen((OUT + "random_maps.csv").c_str(), "w")) std::fclose(probe);
    else { std::printf("cannot write to %s (create the folder first)\n", OUT.c_str()); return 1; }

    RandomMaps();
    VisualQuality(tower, synthetic);
    std::printf("done -> %s\n", OUT.c_str());
    return 0;
}
