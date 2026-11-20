/*!************************************************************************
\file main.cpp
\author Gabriel Sebastian Putra, Asus, XIONG YANG
\par DP email: gabrielsebastian.p@digipen.edu, ngylethan@gmail.com, xiong.yang@digipen.edu
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This source file contains the seam-carving program: it reads an image,
lets the user choose the target size, the algorithm (Dynamic
Programming, Graph Cut with Dinic or Edmonds-Karp, or Greedy) and the
energy type, either from the command line or from an interactive menu,
then writes the resized image and the removed seams.

Seam carving: choose an algorithm and shrink an image to a new size.

  seamcarve <input> <output> <newWidth> <newHeight> <dp|gc-dinic|gc-ek|greedy> [backward|forward]
  seamcarve                     (no arguments: interactive menu)

A width or height of 0 (or pressing Enter in the menu) keeps that dimension
at its original size, e.g. "seamcarve in.png out.png 194 0 dp" shrinks the
width only.

Writes the resized image to <output> and the removed seams drawn on the
original (red = vertical, green = horizontal) to <output-name>_seams.png.
- AlgorithmName(a)
returns the display name of an algorithm
- ParseAlgorithm(s, out)
reads an algorithm from its name or menu number
- ParseEnergy(s, forward)
reads an energy type from its name or menu number
- ParseInt(s, out)
reads a whole string as an int
- SeamsPath(output)
builds the file name of the seam overlay image
- Ask(prompt)
prints a prompt and reads one line
- AskSize(label, original, out)
asks for a new width or height
- AskOptions(o, img)
runs the interactive menu
- Run(img, o)
carves the image with the chosen algorithm and draws the removed seams
- PrintUsage(exe)
prints the command-line usage
- main(argc, argv)
reads the options, runs the carve, writes the images and prints a
summary
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "DynamicProgramming.h"
#include "GraphCutting.h"
#include "GreedyAlgorithm.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

enum class Algorithm { DP, GraphCutDinic, GraphCutEdmondsKarp, Greedy };

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

/************************************************************************/
/*!
  \brief
	gets the name of an algorithm as shown to the user
  \param a
  Algorithm
	The algorithm
  \return const char*
  C string
  returns the display name, or "?" for an unknown value
*/
/************************************************************************/
static const char* AlgorithmName(Algorithm a)
{
    switch (a)
    {
    case Algorithm::DP:                  return "Dynamic Programming";
    case Algorithm::GraphCutDinic:       return "Graph Cut (Dinic)";
    case Algorithm::GraphCutEdmondsKarp: return "Graph Cut (Edmonds-Karp)";
    case Algorithm::Greedy:              return "Greedy";
    }
    return "?";
}

/************************************************************************/
/*!
  \brief
	reads an algorithm choice: dp or 1, gc-dinic or 2, gc-ek or 3, greedy or
	4
  \param s
  const std::string&
	The text to read
  \param out
  Algorithm&
	Receives the algorithm if s is valid
  \return bool
  boolean
  returns true if s named an algorithm, false otherwise
*/
/************************************************************************/
static bool ParseAlgorithm(const std::string& s, Algorithm& out)
{
    if (s == "dp" || s == "1")       { out = Algorithm::DP;                  return true; }
    if (s == "gc-dinic" || s == "2") { out = Algorithm::GraphCutDinic;       return true; }
    if (s == "gc-ek" || s == "3")    { out = Algorithm::GraphCutEdmondsKarp; return true; }
    if (s == "greedy" || s == "4")   { out = Algorithm::Greedy;              return true; }
    return false;
}

/************************************************************************/
/*!
  \brief
	reads an energy choice: backward or 1, forward or 2
  \param s
  const std::string&
	The text to read
  \param forward
  bool&
	Receives true for forward energy, false for backward
  \return bool
  boolean
  returns true if s named an energy type, false otherwise
*/
/************************************************************************/
static bool ParseEnergy(const std::string& s, bool& forward)
{
    if (s == "backward" || s == "1") { forward = false; return true; }
    if (s == "forward" || s == "2")  { forward = true;  return true; }
    return false;
}

/************************************************************************/
/*!
  \brief
	converts a string to an int, accepting it only if the whole string is a
	number
  \param s
  const std::string&
	The text to read
  \param out
  int&
	Receives the number
  \return bool
  boolean
  returns true if the whole string was a valid int, false otherwise
*/
/************************************************************************/
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

/************************************************************************/
/*!
  \brief
	builds the path of the seam overlay image by adding _seams before the
	file extension (or _seams.png if there is none):
	"out/result.png" -> "out/result_seams.png"
  \param output
  const std::string&
	The path of the resized output image
  \return std::string
  string
  returns the path of the seam overlay image
*/
/************************************************************************/
static std::string SeamsPath(const std::string& output)
{
    const size_t slash = output.find_last_of("/\\");
    const size_t dot = output.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return output + "_seams.png";
    return output.substr(0, dot) + "_seams" + output.substr(dot);
}

/************************************************************************/
/*!
  \brief
	prints a prompt and reads one line from the console
  \param prompt
  const std::string&
	The text to print before reading
  \return std::string
  string
  returns the line entered, without the newline (empty if input ended)
*/
/************************************************************************/
static std::string Ask(const std::string& prompt)
{
    std::cout << prompt;
    std::string line;
    std::getline(std::cin, line);
    return line;
}

/************************************************************************/
/*!
  \brief
	Asks for a new width/height in 1..original. Enter or 0 keeps the original.
	Invalid input is asked for again.
  \param label
  const char*
	The name of the dimension shown in the prompt
  \param original
  int
	The original size, the largest value allowed
  \param out
  int&
	Receives the chosen size
  \return bool
  boolean
  returns true once a size was chosen, false if input ended
*/
/************************************************************************/
static bool AskSize(const char* label, int original, int& out)
{
    const std::string range = std::to_string(original);
    for (;;)
    {
        const std::string s = Ask(std::string(label) + " (1-" + range + ", Enter = keep " + range + "): ");
        if (!std::cin) return false;
        int v = 0;
        if (s.empty() || (ParseInt(s, v) && v == 0)) { out = original; return true; }
        if (ParseInt(s, v) && v >= 1 && v <= original) { out = v; return true; }
        std::cout << "  enter a number from 1 to " << range << ", or press Enter to keep it\n";
    }
}

/************************************************************************/
/*!
  \brief
	asks for the input image (until one can be read), the new width and
	height, the algorithm, the energy type and the output path.
	Interactive menu; returns false if the user gave up / input ended.
  \param o
  Options&
	Receives the chosen options
  \param img
  cv::Mat&
	Receives the input image
  \return bool
  boolean
  returns true if every option was chosen, false if input ended
*/
/************************************************************************/
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

    if (!AskSize("New width ", img.cols, o.newWidth)) return false;
    if (!AskSize("New height", img.rows, o.newHeight)) return false;

    std::cout << "Algorithm:\n  1) Dynamic Programming\n  2) Graph Cut (Dinic)\n  3) Graph Cut (Edmonds-Karp)\n"
                 "  4) Greedy\n";
    while (!ParseAlgorithm(Ask("Choice: "), o.algorithm))
        if (!std::cin) return false; else std::cout << "  enter 1, 2, 3 or 4\n";

    std::cout << "Energy:\n  1) Backward (Sobel gradient)\n  2) Forward (simplified)\n";
    while (!ParseEnergy(Ask("Choice: "), o.forward))
        if (!std::cin) return false; else std::cout << "  enter 1 or 2\n";

    o.output = Ask("Output path [output.png]: ");
    if (o.output.empty()) o.output = "output.png";
    return true;
}

/************************************************************************/
/*!
  \brief
	carves the image to the chosen size with the chosen algorithm and
	energy, recording the removed seams, and draws them on the original
	image.
	All three methods fill the same SeamCarving::CarveStats, so only the call
	that chooses the seam finder differs.
  \param img
  const cv::Mat&
	The input image
  \param o
  const Options&
	The chosen size, algorithm and energy type
  \return RunResult
  run result
  returns the resized image, the seam overlay, the total energy removed,
  the number of seams and the time spent finding seams
  \exception std::invalid_argument
	thrown if the target size is not between 1 and the original size
*/
/************************************************************************/
static RunResult Run(const cv::Mat& img, const Options& o)
{
    using SeamCarving::EnergyType;
    const EnergyType energy = o.forward ? EnergyType::Forward : EnergyType::Backward;
    SeamCarving::CarveStats st;
    RunResult r;
    switch (o.algorithm)
    {
    case Algorithm::DP:
        r.resized = DP::Carve(img, o.newWidth, o.newHeight, energy, &st, true);
        break;
    case Algorithm::Greedy:
        r.resized = Greedy::Carve(img, o.newWidth, o.newHeight, energy, &st, true);
        break;
    case Algorithm::GraphCutDinic:
    case Algorithm::GraphCutEdmondsKarp:
        r.resized = GraphCut::Carve(img, o.newWidth, o.newHeight,
                                    o.algorithm == Algorithm::GraphCutDinic ? GraphCut::MaxFlowAlgorithm::Dinic
                                                                            : GraphCut::MaxFlowAlgorithm::EdmondsKarp,
                                    energy, &st, true);
        break;
    }
    r.seams = SeamCarving::DrawSeams(img, st);
    r.totalCost = static_cast<long long>(st.totalCost);
    r.seamsRemoved = st.seamCosts.size();
    r.seamSeconds = st.seamSeconds;
    return r;
}

/************************************************************************/
/*!
  \brief
	prints how to run the program from the command line
  \param exe
  const char*
	The program name, argv[0]
*/
/************************************************************************/
static void PrintUsage(const char* exe)
{
    std::printf("usage: %s <input> <output> <newWidth> <newHeight> <dp|gc-dinic|gc-ek|greedy> [backward|forward]\n"
                "       %s                (interactive menu)\n"
                "       a width or height of 0 keeps the original size\n", exe, exe);
}

/************************************************************************/
/*!
  \brief
	runs the program. With no arguments it shows the interactive menu;
	otherwise it reads the input, output, new width, new height, algorithm
	and optional energy type from the command line, where 0 keeps a
	dimension. It then carves the image, writes the result and the seam
	overlay, and prints the size change, total energy removed and timings.
	Graph Cut on a large image prints a warning first because it is slow.
  \param argc
  int
	The number of command-line arguments
  \param argv
  char**
	The command-line arguments
  \return int
  integer
  returns 0 on success, 1 on invalid arguments, unreadable input, a failed
  write or an error while carving
*/
/************************************************************************/
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
        if (o.newWidth < 0 || o.newWidth > img.cols || o.newHeight < 0 || o.newHeight > img.rows)
        {
            std::printf("target size must be between 1x1 and %dx%d (shrinking only; 0 = keep)\n",
                        img.cols, img.rows);
            return 1;
        }
        if (o.newWidth == 0) o.newWidth = img.cols;     // 0 = keep the original width
        if (o.newHeight == 0) o.newHeight = img.rows;   // 0 = keep the original height
    }

    const int seams = (img.cols - o.newWidth) + (img.rows - o.newHeight);
    const bool graphCut = o.algorithm == Algorithm::GraphCutDinic || o.algorithm == Algorithm::GraphCutEdmondsKarp;
    if (graphCut && seams > 0 && img.total() > 50000)
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
