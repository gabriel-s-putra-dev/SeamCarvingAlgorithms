// Seam carving: choose an algorithm and shrink an image to a new size.
//
//   seamcarve <input> <output> <newWidth> <newHeight> <dp|gc-dinic|gc-ek> [backward|forward]
//   seamcarve                     (no arguments: interactive menu)
//
// Writes the resized image to <output> and the removed seams drawn on the
// original (red = vertical, green = horizontal) to <output-name>_seams.png.
// Greedy will be added to the menu once GreedyAlgorithm is implemented.
#include "DynamicProgramming.h"
#include "GraphCutting.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

enum class Algorithm { DP, GraphCutDinic, GraphCutEdmondsKarp };

struct Options
{
    std::string input, output;
    int newWidth = 0, newHeight = 0;
    Algorithm algorithm = Algorithm::DP;
    bool forward = false;   // energy type: false = backward (Sobel), true = forward
};

struct RunResult
{
    cv::Mat resized, seams;
    long long totalCost = 0;
    size_t seamsRemoved = 0;
    double seamSeconds = 0.0;
};

static const char* AlgorithmName(Algorithm a)
{
    switch (a)
    {
    case Algorithm::DP:                  return "Dynamic Programming";
    case Algorithm::GraphCutDinic:       return "Graph Cut (Dinic)";
    case Algorithm::GraphCutEdmondsKarp: return "Graph Cut (Edmonds-Karp)";
    }
    return "?";
}

static bool ParseAlgorithm(const std::string& s, Algorithm& out)
{
    if (s == "dp" || s == "1")       { out = Algorithm::DP;                  return true; }
    if (s == "gc-dinic" || s == "2") { out = Algorithm::GraphCutDinic;       return true; }
    if (s == "gc-ek" || s == "3")    { out = Algorithm::GraphCutEdmondsKarp; return true; }
    return false;
}

static bool ParseEnergy(const std::string& s, bool& forward)
{
    if (s == "backward" || s == "1") { forward = false; return true; }
    if (s == "forward" || s == "2")  { forward = true;  return true; }
    return false;
}

static bool ParseInt(const std::string& s, int& out)
{
    try
    {
        size_t used = 0;
        out = std::stoi(s, &used);
        return used == s.size();
    }
    catch (const std::exception&) { return false; }
}

// "out/result.png" -> "out/result_seams.png"
static std::string SeamsPath(const std::string& output)
{
    const size_t slash = output.find_last_of("/\\");
    const size_t dot = output.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return output + "_seams.png";
    return output.substr(0, dot) + "_seams" + output.substr(dot);
}

static std::string Ask(const std::string& prompt)
{
    std::cout << prompt;
    std::string line;
    std::getline(std::cin, line);
    return line;
}

// Interactive menu; returns false if the user gave up / input ended.
static bool AskOptions(Options& o, cv::Mat& img)
{
    std::cout << "=== Seam carving ===\n";
    for (;;)
    {
        o.input = Ask("Input image path: ");
        if (!std::cin) return false;
        img = cv::imread(o.input, cv::IMREAD_COLOR);
        if (!img.empty()) break;
        std::cout << "  could not read that image, try again\n";
    }
    std::cout << "  size: " << img.cols << " x " << img.rows << "\n";

    while (!ParseInt(Ask("New width  (1-" + std::to_string(img.cols) + "): "), o.newWidth)
           || o.newWidth < 1 || o.newWidth > img.cols)
        if (!std::cin) return false; else std::cout << "  out of range\n";
    while (!ParseInt(Ask("New height (1-" + std::to_string(img.rows) + "): "), o.newHeight)
           || o.newHeight < 1 || o.newHeight > img.rows)
        if (!std::cin) return false; else std::cout << "  out of range\n";

    std::cout << "Algorithm:\n  1) Dynamic Programming\n  2) Graph Cut (Dinic)\n  3) Graph Cut (Edmonds-Karp)\n";
    while (!ParseAlgorithm(Ask("Choice: "), o.algorithm))
        if (!std::cin) return false; else std::cout << "  enter 1, 2 or 3\n";

    std::cout << "Energy:\n  1) Backward (Sobel gradient)\n  2) Forward (simplified)\n";
    while (!ParseEnergy(Ask("Choice: "), o.forward))
        if (!std::cin) return false; else std::cout << "  enter 1 or 2\n";

    o.output = Ask("Output path [output.png]: ");
    if (o.output.empty()) o.output = "output.png";
    return true;
}

static RunResult Run(const cv::Mat& img, const Options& o)
{
    RunResult r;
    if (o.algorithm == Algorithm::DP)
    {
        DP::CarveStats st;
        r.resized = DP::Carve(img, o.newWidth, o.newHeight,
                              o.forward ? DP::EnergyType::Forward : DP::EnergyType::Backward, &st, true);
        r.seams = DP::DrawSeams(img, st);
        r.totalCost = st.totalCost;
        r.seamsRemoved = st.seamCosts.size();
        r.seamSeconds = st.seamSeconds;
    }
    else
    {
        GraphCut::CarveStats st;
        r.resized = GraphCut::Carve(img, o.newWidth, o.newHeight,
                                    o.algorithm == Algorithm::GraphCutDinic ? GraphCut::MaxFlowAlgorithm::Dinic
                                                                            : GraphCut::MaxFlowAlgorithm::EdmondsKarp,
                                    o.forward ? GraphCut::EnergyType::Forward : GraphCut::EnergyType::Backward,
                                    &st, true);
        r.seams = GraphCut::DrawSeams(img, st);
        r.totalCost = static_cast<long long>(st.totalCost);
        r.seamsRemoved = st.seamCosts.size();
        r.seamSeconds = st.seamSeconds;
    }
    return r;
}

static void PrintUsage(const char* exe)
{
    std::printf("usage: %s <input> <output> <newWidth> <newHeight> <dp|gc-dinic|gc-ek> [backward|forward]\n"
                "       %s                (interactive menu)\n", exe, exe);
}

int main(int argc, char** argv)
{
    Options o;
    cv::Mat img;

    if (argc == 1)
    {
        if (!AskOptions(o, img)) return 1;
    }
    else
    {
        if (argc < 6 || argc > 7 || !ParseInt(argv[3], o.newWidth) || !ParseInt(argv[4], o.newHeight)
            || !ParseAlgorithm(argv[5], o.algorithm) || (argc == 7 && !ParseEnergy(argv[6], o.forward)))
        {
            PrintUsage(argv[0]);
            return 1;
        }
        o.input = argv[1];
        o.output = argv[2];
        img = cv::imread(o.input, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::printf("could not read %s\n", o.input.c_str());
            return 1;
        }
        if (o.newWidth < 1 || o.newWidth > img.cols || o.newHeight < 1 || o.newHeight > img.rows)
        {
            std::printf("target size must be between 1x1 and %dx%d (shrinking only)\n", img.cols, img.rows);
            return 1;
        }
    }

    const int seams = (img.cols - o.newWidth) + (img.rows - o.newHeight);
    if (o.algorithm != Algorithm::DP && seams > 0 && img.total() > 50000)
        std::printf("note: Graph Cut builds a max-flow graph for every seam and is roughly 1000x slower\n"
                    "      than DP at this size; %d seams may take a while.\n", seams);

    try
    {
        const auto t0 = std::chrono::steady_clock::now();
        const RunResult r = Run(img, o);
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const std::string seamsPath = SeamsPath(o.output);
        if (!cv::imwrite(o.output, r.resized) || !cv::imwrite(seamsPath, r.seams))
        {
            std::printf("could not write %s / %s\n", o.output.c_str(), seamsPath.c_str());
            return 1;
        }

        std::printf("\n%s, %s energy\n", AlgorithmName(o.algorithm), o.forward ? "forward" : "backward");
        std::printf("  %dx%d -> %dx%d  (%d vertical + %d horizontal seams)\n", img.cols, img.rows,
                    r.resized.cols, r.resized.rows, img.cols - o.newWidth, img.rows - o.newHeight);
        std::printf("  total energy removed : %lld\n", r.totalCost);
        std::printf("  energy + seam finding: %.3f s\n", r.seamSeconds);
        std::printf("  total time           : %.3f s\n", total);
        std::printf("  wrote %s and %s\n", o.output.c_str(), seamsPath.c_str());
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    return 0;
}
