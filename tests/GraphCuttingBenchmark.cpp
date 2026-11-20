/*!************************************************************************
\file GraphCuttingBenchmark.cpp
\author GYQ-SIT
\par DP email: 2301516@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Benchmark + visual comparison for GraphCutting.cpp. Build from the repo root
(CMake target gc_bench), or:
  g++ -std=c++14 -O2 -I. tests/GraphCuttingBenchmark.cpp SeamCarving.cpp GraphCutting.cpp DynamicProgramming.cpp GreedyAlgorithm.cpp -o gc_bench `pkg-config --cflags --libs opencv4`
  ./gc_bench            (writes docs/GraphCutting/results/)
Then draw the timing chart:  python tools/plot_graphcut_timing.py

DP and Greedy are the TEAM's implementations (DynamicProgramming.cpp,
GreedyAlgorithm.cpp), run on the same energy map as Graph Cut.
- OpenCsv(name)
opens a results CSV for writing or exits with a message
- TeamDP(c), TeamGreedy(c), GC(algo)
the three seam finders wrapped as SeamFinder
- Seconds(t0)
returns the seconds elapsed since a time point
- SingleSeamTiming(img)
times one seam per method across image sizes
- PhaseScaling(img)
measures Dinic phases and time per phase across sizes
- MemoryUsage(img)
counts the bytes used by the graph-cut network and the DP table
- CarveDP(img, k, st), CarveGreedy(img, k, st), CarveGC(img, k, st)
remove k vertical seams per method
- Label(img, text)
adds a caption above an image
- SideBySide(images, gap)
places images next to each other
- FullResize(name, img, k, csv)
carves one image with every method and saves the comparison
- main(argc, argv)
runs the experiments and writes docs/GraphCutting/results/
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "GraphCutting.h"
#include "DynamicProgramming.h"
#include "GreedyAlgorithm.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace GraphCut;

static const std::string OUT = "docs/GraphCutting/results/";

/************************************************************************/
/*!
  \brief
	Opens OUT + name for writing. Exits with a clear message instead of crashing
	if the file cannot be opened (e.g. the CSV is open in Excel, which locks it).
  \param name
  const std::string&
	The file name inside docs/GraphCutting/results/
  \return FILE*
  C file handle
  returns the open file (the program exits if it cannot be opened)
*/
/************************************************************************/
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

struct Seam { std::vector<int> cols; int64_t cost; };
using SeamFinder = std::function<Seam(const cv::Mat&)>;

// ---------------------------------------------------------------------------
// Seam finders (vertical seam on a CV_32S cost map)
// ---------------------------------------------------------------------------

/************************************************************************/
/*!
  \brief
	Team DP (DynamicProgramming.cpp) wrapped as a SeamFinder.
  \param c
  const cv::Mat&
	The CV_32SC1 cost map
  \return Seam
  seam
  returns the seam (one column per row) and its total cost
*/
/************************************************************************/
static Seam TeamDP(const cv::Mat& c)
{
    Seam s{ DP::FindVerticalSeam(c), 0 };
    s.cost = DP::SeamEnergy(c, s.cols, true);
    return s;
}

/************************************************************************/
/*!
  \brief
	Team Greedy (GreedyAlgorithm.cpp) wrapped as a SeamFinder.
  \param c
  const cv::Mat&
	The CV_32SC1 cost map
  \return Seam
  seam
  returns the seam (one column per row) and its total cost
*/
/************************************************************************/
static Seam TeamGreedy(const cv::Mat& c)
{
    Greedy::SeamResult r = Greedy::FindVerticalSeam(c);
    return Seam{ std::move(r.seam), r.cost };
}

/************************************************************************/
/*!
  \brief
	wraps the graph-cut seam finder with a chosen max-flow algorithm as a
	SeamFinder
  \param algo
  MaxFlowAlgorithm
	The max-flow solver to use
  \return SeamFinder
  seam finder function
  returns a function returning the graph-cut seam and its cost
*/
/************************************************************************/
static SeamFinder GC(MaxFlowAlgorithm algo)
{
    return [algo](const cv::Mat& c) {
        SeamResult r = FindVerticalSeam(c, algo);
        return Seam{ r.seam, r.cost };
    };
}

// ---------------------------------------------------------------------------

/************************************************************************/
/*!
  \brief
	measures how much time has passed since a given time point
  \param t0
  std::chrono::steady_clock::time_point
	The starting time point
  \return double
  double
  returns the elapsed time in seconds
*/
/************************************************************************/
static double Seconds(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/************************************************************************/
/*!
  \brief
	times one vertical seam with DP, Greedy, Graph Cut (Dinic) and Graph Cut
	(Edmonds-Karp, up to 400 wide) on the tower image scaled to widths from
	30 to 900, keeping the best of several runs, and writes
	single_seam_timing.csv
  \param img
  const cv::Mat&
	The tower image; smaller sizes are made from it by scaling
*/
/************************************************************************/
static void SingleSeamTiming(const cv::Mat& img)
{
    struct M { const char* name; SeamFinder f; int maxWidth; int reps; };
    const std::vector<M> methods = {
        { "DP", TeamDP, 100000, 5 },
        { "Greedy", TeamGreedy, 100000, 5 },
        { "GC Dinic", GC(MaxFlowAlgorithm::Dinic), 100000, 1 },
        { "GC Edmonds-Karp", GC(MaxFlowAlgorithm::EdmondsKarp), 400, 1 },
    };
    FILE* f = OpenCsv("single_seam_timing.csv");
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

/************************************************************************/
/*!
  \brief
	Why graph cut slows down sharply at large sizes: Dinic phase count and
	time per phase for one vertical seam (results/phase_scaling.csv).
  \param img
  const cv::Mat&
	The tower image; smaller sizes are made from it by scaling
*/
/************************************************************************/
static void PhaseScaling(const cv::Mat& img)
{
    FILE* f = OpenCsv("phase_scaling.csv");
    std::fprintf(f, "width,height,pixels,arcs,graph_mib,seconds,phases,augmentations,ms_per_phase,seam_cost\n");
    std::printf("Dinic phase scaling\n");
    for (int w : { 274, 400, 600, 750, 900 })
    {
        const int h = static_cast<int>(std::lround(double(w) * img.rows / img.cols));
        cv::Mat small;
        cv::resize(img, small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        const cv::Mat cost = ComputeEnergy(small);
        int64_t s, t;
        const double mib = BuildSeamGraph(cost, s, t).MemoryBytes() / 1048576.0;
        const auto t0 = std::chrono::steady_clock::now();
        const SeamResult r = FindVerticalSeam(cost, MaxFlowAlgorithm::Dinic);
        const double sec = Seconds(t0);
        std::fprintf(f, "%d,%d,%d,%lld,%.2f,%.6f,%lld,%lld,%.3f,%lld\n", w, h, w * h, (long long)r.arcs, mib, sec,
                     (long long)r.phases, (long long)r.augmentations, 1e3 * sec / r.phases, (long long)r.cost);
        std::printf("  %4dx%-4d arcs=%8lld graph=%6.2f MiB  %8.3f s  phases=%4lld  aug=%5lld  %6.2f ms/phase\n",
                    w, h, (long long)r.arcs, mib, sec, (long long)r.phases, (long long)r.augmentations,
                    1e3 * sec / r.phases);
    }
    std::fclose(f);
}

/************************************************************************/
/*!
  \brief
	counts, from the real objects, the bytes of the graph-cut flow network
	and of the DP cumulative table plus seam, for widths from 68 to 1920,
	and writes memory_usage.csv
  \param img
  const cv::Mat&
	The tower image; smaller sizes are made from it by scaling
*/
/************************************************************************/
static void MemoryUsage(const cv::Mat& img)
{
    FILE* f = OpenCsv("memory_usage.csv");
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
        // Team DP (DynamicProgramming.cpp): one CV_64F cumulative table M, plus the
        // seam (one int per row). The seam is backtracked through M, so there are
        // no back-pointers. Counted from the real objects, not from a formula.
        const cv::Mat M = DP::CumulativeEnergy(ComputeEnergy(small));
        const size_t dp = M.total() * M.elemSize() + static_cast<size_t>(h) * sizeof(int);
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
// Each module's Carve runs the same shared loop (SeamCarving::CarveWith), so
// only the seam finder differs between the methods.
using CarveFn = std::function<cv::Mat(const cv::Mat& img, int k, CarveStats* st)>;

/************************************************************************/
/*!
  \brief
	removes k vertical seams with DP::Carve (backward energy), recording the
	seams
  \param img
  const cv::Mat&
	The image to carve
  \param k
  int
	The number of vertical seams to remove
  \param st
  CarveStats*
	Receives the time, costs and recorded seams
  \return cv::Mat
  OpenCV matrix
  returns the carved image, k columns narrower
*/
/************************************************************************/
static cv::Mat CarveDP(const cv::Mat& img, int k, CarveStats* st)
{
    return DP::Carve(img, img.cols - k, img.rows, EnergyType::Backward, st, true);
}

/************************************************************************/
/*!
  \brief
	removes k vertical seams with Greedy::Carve (backward energy), recording
	the seams
  \param img
  const cv::Mat&
	The image to carve
  \param k
  int
	The number of vertical seams to remove
  \param st
  CarveStats*
	Receives the time, costs and recorded seams
  \return cv::Mat
  OpenCV matrix
  returns the carved image, k columns narrower
*/
/************************************************************************/
static cv::Mat CarveGreedy(const cv::Mat& img, int k, CarveStats* st)
{
    return Greedy::Carve(img, img.cols - k, img.rows, EnergyType::Backward, st, true);
}

/************************************************************************/
/*!
  \brief
	removes k vertical seams with GraphCut::Carve (Dinic, backward energy),
	recording the seams
  \param img
  const cv::Mat&
	The image to carve
  \param k
  int
	The number of vertical seams to remove
  \param st
  CarveStats*
	Receives the time, costs and recorded seams
  \return cv::Mat
  OpenCV matrix
  returns the carved image, k columns narrower
*/
/************************************************************************/
static cv::Mat CarveGC(const cv::Mat& img, int k, CarveStats* st)
{
    return Carve(img, img.cols - k, img.rows, MaxFlowAlgorithm::Dinic, EnergyType::Backward, st, true);
}

/************************************************************************/
/*!
  \brief
	adds a white strip with a caption above an image
  \param img
  const cv::Mat&
	The image to label
  \param text
  const std::string&
	The caption
  \return cv::Mat
  OpenCV matrix
  returns the image with the caption strip on top
*/
/************************************************************************/
static cv::Mat Label(const cv::Mat& img, const std::string& text)
{
    cv::Mat pad(22, img.cols, img.type(), cv::Scalar(255, 255, 255));
    cv::putText(pad, text, cv::Point(4, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    cv::Mat out;
    cv::vconcat(pad, img, out);
    return out;
}

/************************************************************************/
/*!
  \brief
	places images next to each other on a white canvas, top-aligned
  \param images
  const std::vector<cv::Mat>&
	The CV_8UC3 images, left to right
  \param gap
  int
	The number of white pixels between neighbouring images, default 8
  \return cv::Mat
  OpenCV matrix
  returns the combined CV_8UC3 image
*/
/************************************************************************/
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

/************************************************************************/
/*!
  \brief
	removes k vertical seams from one image with DP, Greedy and Graph Cut,
	writes the time and total cost of each to the CSV, saves the results and
	the first 30 seams side by side, and prints how many seams DP and Graph
	Cut share
  \param name
  const std::string&
	The image name used in the CSV and file names
  \param img
  const cv::Mat&
	The image to carve
  \param k
  int
	The number of vertical seams to remove
  \param csv
  FILE*
	The open full_resize.csv file
*/
/************************************************************************/
static void FullResize(const std::string& name, const cv::Mat& img, int k, FILE* csv)
{
    struct M { const char* name; CarveFn carve; };
    const std::vector<M> methods = { { "DP", CarveDP }, { "Greedy", CarveGreedy }, { "GC Dinic", CarveGC } };
    std::vector<cv::Mat> results{ Label(img, "original") }, overlays;
    std::vector<std::vector<std::vector<cv::Point>>> allSeams;
    for (const M& m : methods)
    {
        CarveStats r;
        const cv::Mat out = m.carve(img, k, &r);
        std::fprintf(csv, "%s,%s,%dx%d,%d,%.6f,%lld\n", name.c_str(), m.name, img.cols, img.rows, k,
                     r.seamSeconds, (long long)r.totalCost);
        std::printf("  %-10s %-10s %d seams: %8.3f s  total cost=%lld\n", name.c_str(), m.name, k,
                    r.seamSeconds, (long long)r.totalCost);
        results.push_back(Label(out, m.name));
        cv::Mat vis = img.clone();
        for (size_t s = 0; s < r.verticalSeams.size() && s < 30; ++s)
            for (const cv::Point& p : r.verticalSeams[s]) vis.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
        overlays.push_back(Label(vis, std::string(m.name) + ": first 30 seams"));
        allSeams.push_back(r.verticalSeams);
    }
    cv::imwrite(OUT + "compare_" + name + ".png", SideBySide(results));
    cv::imwrite(OUT + "seams_" + name + ".png", SideBySide(overlays));

    // How many of the k seams are identical between DP and graph cut?
    // (Each method carves its own image, so after a divergence they may differ.)
    int identical = 0;
    for (int s = 0; s < k; ++s) identical += (allSeams[0][s] == allSeams[2][s]);
    std::printf("  %-10s DP vs GC: %d/%d seams identical\n", name.c_str(), identical, k);
}

/************************************************************************/
/*!
  \brief
	Usage: gc_bench            run every experiment
	       gc_bench memory     only the counted-memory table (memory_usage.csv);
	                           these are exact byte counts, not timings, so they
	                           can be regenerated without re-running the timings
	Otherwise it runs the single-seam timing, phase scaling, memory count,
	first-seam cost, full resize and both-directions experiments.
  \param argc
  int
	The number of command-line arguments
  \param argv
  char**
	The command-line arguments
  \return int
  integer
  returns 0 on success, 1 if the sample images or the results folder
  cannot be used
*/
/************************************************************************/
int main(int argc, char** argv)
{
    const bool memoryOnly = argc > 1 && std::string(argv[1]) == "memory";
    const cv::Mat tower = cv::imread("images/tower.png");
    const cv::Mat synthetic = cv::imread("images/synthetic.png");
    if (tower.empty() || synthetic.empty())
    {
        std::printf("run from the repo root (images/tower.png, images/synthetic.png)\n");
        return 1;
    }
    // Check the results folder is writable. "a" (append) does not truncate, so
    // "gc_bench memory" leaves the existing timing CSV untouched.
    if (FILE* probe = std::fopen((OUT + "single_seam_timing.csv").c_str(), "a")) std::fclose(probe);
    else { std::printf("cannot write to %s (create the folder first)\n", OUT.c_str()); return 1; }

    if (memoryOnly)
    {
        MemoryUsage(tower);
        return 0;
    }
    SingleSeamTiming(tower);
    PhaseScaling(tower);
    MemoryUsage(tower);

    std::printf("First-seam cost\n");
    FILE* f = OpenCsv("seam_cost.csv");
    std::fprintf(f, "image,method,seam_cost,augmentations,phases\n");
    for (const auto& item : { std::make_pair(std::string("tower"), tower), std::make_pair(std::string("synthetic"), synthetic) })
    {
        const cv::Mat c = ComputeEnergy(item.second);
        const SeamResult d = FindVerticalSeam(c, MaxFlowAlgorithm::Dinic);
        const SeamResult e = FindVerticalSeam(c, MaxFlowAlgorithm::EdmondsKarp);
        std::fprintf(f, "%s,DP,%lld,,\n", item.first.c_str(), (long long)TeamDP(c).cost);
        std::fprintf(f, "%s,Greedy,%lld,,\n", item.first.c_str(), (long long)TeamGreedy(c).cost);
        std::fprintf(f, "%s,GC Dinic,%lld,%lld,%lld\n", item.first.c_str(), (long long)d.cost,
                     (long long)d.augmentations, (long long)d.phases);
        std::fprintf(f, "%s,GC Edmonds-Karp,%lld,%lld,\n", item.first.c_str(), (long long)e.cost,
                     (long long)e.augmentations);
        std::printf("  %-10s DP %lld  Greedy %lld  GC %lld (Dinic: %lld aug, %lld phases; EK: %lld aug)\n",
                    item.first.c_str(), (long long)TeamDP(c).cost, (long long)TeamGreedy(c).cost,
                    (long long)d.cost, (long long)d.augmentations, (long long)d.phases, (long long)e.augmentations);
    }
    std::fclose(f);

    std::printf("Full resize\n");
    f = OpenCsv("full_resize.csv");
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
    f = OpenCsv("both_directions.csv");
    std::fprintf(f, "image,from,to,vertical_seams,horizontal_seams,seconds,total_cost\n");
    std::fprintf(f, "tower,274x186,194x150,%zu,%zu,%.6f,%lld\n", st.verticalSeams.size(),
                 st.horizontalSeams.size(), st.seamSeconds, (long long)st.totalCost);
    std::fclose(f);
    std::printf("done -> %s\n", OUT.c_str());
    return 0;
}
