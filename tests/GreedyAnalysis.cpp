/*!************************************************************************
\file GreedyAnalysis.cpp
\author XIONG YANG
\par DP email: xiong.yang@digipen.edu
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Seam-cost and visual-quality analysis for the Greedy write-up (Person B).
Complements tests/GreedyBenchmark.cpp (timing, first-seam cost, side-by-side
results). Build and run from the repository root:
  Windows:       cmake --build --preset vs2022-release --target greedy_analysis
                 out\build\vs2022\Release\greedy_analysis.exe
  Linux / macOS: cmake --build --preset unix-release --target greedy_analysis
                 out/build/unix-release/greedy_analysis
Then draw the charts:  python tools/plot_greedy_analysis.py

Outputs (docs/Greedy/results/):
  random_maps.csv          greedy / optimal seam cost on random cost maps, by size
  quality.csv              how much "important" content each method removes
  zoom_tower.png           the tower region of each result, enlarged 2x
  compare_tower_forward.png  Greedy vs DP with the forward energy

"Optimal" is the team's DP (DynamicProgramming.cpp), which gives the same
cost as graph cut (see tests/GraphCutVsTeamDPTests.cpp). Both methods carve
with the shared SeamCarving::CarveWith loop via their module's Carve.
- OpenCsv(name)
opens a results CSV for writing or exits with a message
- RandomMaps()
greedy / optimal seam cost on random cost maps
- CarveVertical(img, k, energy, greedy)
removes k vertical seams with Greedy or DP
- Percentile(energy, pct)
returns an energy percentile
- Quality(f, image, energyName, method, original, r)
writes the quality metrics of one carve
- Label(img, text)
adds a caption above an image
- SideBySide(images, gap)
places images next to each other
- Zoom(img, roi, scale)
crops and enlarges part of an image
- VisualQuality(tower, synthetic)
compares what Greedy and DP remove and saves the figures
- main()
runs both analyses
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
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

/************************************************************************/
/*!
  \brief
	Opens OUT + name for writing. Exits with a clear message instead of crashing
	if the file cannot be opened (e.g. the CSV is open in Excel, which locks it).
  \param name
  const std::string&
	The file name inside docs/Greedy/results/
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

// ---------------------------------------------------------------------------
// 1. Random cost maps: how far from optimal is one greedy seam?
// ---------------------------------------------------------------------------

/************************************************************************/
/*!
  \brief
	for each map kind and sizes 4 to 256, generates 300 random maps,
	compares the first greedy seam's cost with the optimal (team DP) cost,
	and writes the mean, median and maximum ratio and the share of optimal
	seams to random_maps.csv.
	Two kinds of map:
	  uniform - every pixel independent in [0, 255]: no structure at all;
	  smooth  - uniform noise blurred, so cheap pixels cluster in regions like
	            in a real energy map.
*/
/************************************************************************/
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
                const int64_t g = Greedy::FindVerticalSeam(c).cost;
                const int64_t d = DP::SeamEnergy(c, DP::FindVerticalSeam(c), true);
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
    SeamCarving::CarveStats st;   // st.verticalSeams: removed seams, original coordinates
};

/************************************************************************/
/*!
  \brief
	Remove k vertical seams with Greedy or DP (both run the shared carving loop,
	SeamCarving::CarveWith, and record each seam in original-image coordinates).
  \param img
  const cv::Mat&
	The image to carve
  \param k
  int
	The number of vertical seams to remove
  \param energy
  SeamCarving::EnergyType
	The energy to use
  \param greedy
  bool
	true to use Greedy, false to use DP
  \return Carved
  carve result
  returns the carved image and its statistics, including the recorded
  seams
*/
/************************************************************************/
static Carved CarveVertical(const cv::Mat& img, int k, SeamCarving::EnergyType energy, bool greedy)
{
    Carved r;
    r.out = greedy ? Greedy::Carve(img, img.cols - k, img.rows, energy, &r.st, true)
                   : DP::Carve(img, img.cols - k, img.rows, energy, &r.st, true);
    return r;
}

/************************************************************************/
/*!
  \brief
	Energy value at the given percentile (0..100) of the map.
  \param energy
  const cv::Mat&
	The CV_32SC1 energy map
  \param pct
  double
	The percentile, 0 to 100
  \return int
  integer
  returns the energy value at that percentile
*/
/************************************************************************/
static int Percentile(const cv::Mat& energy, double pct)
{
    std::vector<int> v(energy.begin<int>(), energy.end<int>());
    const size_t idx = std::min(v.size() - 1, size_t(pct / 100.0 * v.size()));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

/************************************************************************/
/*!
  \brief
	Quality metrics for a set of removed pixels, measured on the ORIGINAL image's
	backward energy (so all methods and both energy types use the same yardstick):
	  mean_energy   - average original energy of the removed pixels;
	  pct_top10     - % of removed pixels that were in the top 10 % of energy
	                  (edges / detail: the content seam carving should keep);
	  pct_edge      - % of removed pixels on a strong edge, energy >= 100. For a
	                  step edge Sobel gives |Gx| = 4h, so 100 is a step of ~25
	                  grey levels. Unlike pct_top10 this does not depend on how
	                  much of the image is flat (the synthetic image is ~90 % flat);
	  cols_touched  - how many original columns lost at least one pixel. Low =
	                  seams stack in a few low-energy bands; high = spread out.
	Writes one CSV row and prints one summary line.
  \param f
  FILE*
	The open quality.csv file
  \param image
  const std::string&
	The image name
  \param energyName
  const char*
	The energy used for carving
  \param method
  const char*
	The method name (Greedy or DP)
  \param original
  const cv::Mat&
	The original image
  \param r
  const Carved&
	The carve result with its recorded seams
*/
/************************************************************************/
static void Quality(FILE* f, const std::string& image, const char* energyName, const char* method,
                    const cv::Mat& original, const Carved& r)
{
    const cv::Mat e = DP::ComputeEnergy(original, DP::EnergyType::Backward);
    const int top10 = Percentile(e, 90.0);
    int64_t sum = 0, n = 0, high = 0, edge = 0;
    std::vector<char> touched(original.cols, 0);
    for (const auto& seam : r.st.verticalSeams)
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
    std::fprintf(f, "%s,%s,%s,%zu,%lld,%.2f,%.2f,%.2f,%d\n", image.c_str(), energyName, method, r.st.verticalSeams.size(),
                 (long long)r.st.totalCost, mean, pct, pctEdge, cols);
    std::printf("  %-9s %-8s %-6s total cost %8lld  mean energy %6.1f  top-10%% %5.1f%%  edge %5.2f%% (%lld px)"
                "  cols touched %d\n",
                image.c_str(), energyName, method, (long long)r.st.totalCost, mean, pct, pctEdge,
                (long long)edge, cols);
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
	crops a region of an image (clipped to the image) and enlarges it with
	nearest-neighbour scaling
  \param img
  const cv::Mat&
	The image
  \param roi
  const cv::Rect&
	The region to crop
  \param scale
  int
	The enlargement factor
  \return cv::Mat
  OpenCV matrix
  returns the enlarged crop
*/
/************************************************************************/
static cv::Mat Zoom(const cv::Mat& img, const cv::Rect& roi, int scale)
{
    cv::Mat z;
    cv::resize(img(roi & cv::Rect(0, 0, img.cols, img.rows)), z, cv::Size(), scale, scale, cv::INTER_NEAREST);
    return z;
}

/************************************************************************/
/*!
  \brief
	carves the tower (80 seams) and synthetic (50 seams) images with Greedy
	and DP, both energies, writes the quality metrics to quality.csv, and
	saves zoom_tower.png and compare_tower_forward.png
  \param tower
  const cv::Mat&
	The tower sample image
  \param synthetic
  const cv::Mat&
	The synthetic sample image
*/
/************************************************************************/
static void VisualQuality(const cv::Mat& tower, const cv::Mat& synthetic)
{
    FILE* f = OpenCsv("quality.csv");
    std::fprintf(f, "image,energy,method,seams,total_cost,mean_energy_removed,pct_top10_removed,pct_edge_removed,cols_touched\n");
    std::printf("Visual quality (metrics on the original image's backward energy)\n");

    struct Job { std::string name; cv::Mat img; int k; };
    for (const Job& job : { Job{ "tower", tower, 80 }, Job{ "synthetic", synthetic, 50 } })
        for (DP::EnergyType energy : { DP::EnergyType::Backward, DP::EnergyType::Forward })
        {
            const char* en = energy == DP::EnergyType::Forward ? "forward" : "backward";
            const Carved g = CarveVertical(job.img, job.k, energy, true);
            const Carved d = CarveVertical(job.img, job.k, energy, false);
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

/************************************************************************/
/*!
  \brief
	loads the sample images, checks that the results folder is writable,
	then runs the random-map and visual-quality analyses
  \return int
  integer
  returns 0 on success, 1 if the images or the results folder cannot be
  used
*/
/************************************************************************/
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
