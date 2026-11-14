// Seam carving driver: the user chooses which algorithm resizes the image.
//
//   seam_carve <input> <output.png>
//       -> asks for the algorithm, energy type and target size
//   seam_carve <input> <output.png> <greedy|graphcut|dp> <newWidth> <newHeight> [backward|forward]
//   seam_carve <input> <output.png> energy [backward|forward]
//       -> saves the energy map as a viewable 8-bit picture (bright = edges)
//
// For the carving algorithms a second picture, <output>_seams.png, shows every
// removed seam on the original image (red = vertical, green = horizontal).
#include "DynamicProgramming.h"
#include "GraphCutting.h"
#include "GreedyAlgorithm.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

static void Usage(const char* prog)
{
    std::printf("usage: %s <input> <output.png>\n"
                "       %s <input> <output.png> <greedy|graphcut|dp> <newWidth> <newHeight> [backward|forward]\n"
                "       %s <input> <output.png> energy [backward|forward]\n",
                prog, prog, prog);
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

static int SaveEnergy(const cv::Mat& img, const std::string& input, const std::string& output,
                      const std::string& mode)
{
    const cv::Mat energy = DP::ComputeEnergy(
        img, mode == "forward" ? DP::EnergyType::Forward : DP::EnergyType::Backward);
    double minVal, maxVal;
    cv::minMaxLoc(energy, &minVal, &maxVal);
    cv::Mat vis;
    cv::normalize(energy, vis, 0, 255, cv::NORM_MINMAX, CV_8U);   // stretch to 0..255 for viewing
    if (!cv::imwrite(output, vis))
    {
        std::printf("could not write %s\n", output.c_str());
        return 1;
    }
    std::printf("%s energy of %s (%dx%d): min %.0f, max %.0f -> %s\n", mode.c_str(), input.c_str(),
                img.cols, img.rows, minVal, maxVal, output.c_str());
    return 0;
}

static int RunCarve(const cv::Mat& img, const std::string& output, const std::string& algo,
                    int newWidth, int newHeight, const std::string& mode)
{
    if (newWidth < 1 || newHeight < 1 || newWidth > img.cols || newHeight > img.rows)
    {
        std::printf("target size must be between 1x1 and %dx%d (seam carving only shrinks)\n",
                    img.cols, img.rows);
        return 1;
    }
    const bool forward = mode == "forward";

    cv::Mat result, seams;
    double seamSeconds = 0.0;
    long long totalCost = 0;
    if (algo == "greedy")
    {
        Greedy::CarveStats st;
        result = Greedy::Carve(img, newWidth, newHeight,
                               forward ? Greedy::EnergyType::Forward : Greedy::EnergyType::Backward,
                               &st, true);
        seams = Greedy::DrawSeams(img, st);
        seamSeconds = st.seamSeconds;
        totalCost = st.totalCost;
    }
    else if (algo == "graphcut")
    {
        GraphCut::CarveStats st;
        result = GraphCut::Carve(img, newWidth, newHeight, GraphCut::MaxFlowAlgorithm::Dinic,
                                 forward ? GraphCut::EnergyType::Forward : GraphCut::EnergyType::Backward,
                                 &st, true);
        seams = GraphCut::DrawSeams(img, st);
        seamSeconds = st.seamSeconds;
        totalCost = st.totalCost;
    }
    else   // "dp"
    {
        std::printf("Dynamic Programming seam carving is not implemented yet (only the energy map, step a).\n");
        return 1;
    }

    const std::string seamsPath = SeamsPath(output);
    if (!cv::imwrite(output, result) || !cv::imwrite(seamsPath, seams))
    {
        std::printf("could not write %s / %s\n", output.c_str(), seamsPath.c_str());
        return 1;
    }
    std::printf("%s (%s energy): %dx%d -> %dx%d, %d seams, %.3f s finding seams, total seam cost %lld\n",
                algo.c_str(), mode.c_str(), img.cols, img.rows, newWidth, newHeight,
                (img.cols - newWidth) + (img.rows - newHeight), seamSeconds, totalCost);
    std::printf("result -> %s\nseams  -> %s\n", output.c_str(), seamsPath.c_str());
    return 0;
}

// Interactive menu used when only <input> and <output> are given.
static bool AskOptions(const cv::Mat& img, std::string& algo, std::string& mode, int& w, int& h)
{
    std::printf("Image is %dx%d. Choose an algorithm:\n"
                "  1) Greedy\n"
                "  2) Graph Cut (min-cut / max-flow)\n"
                "  3) Dynamic Programming\n"
                "  4) Energy map only\n"
                "> ",
                img.cols, img.rows);
    int choice = 0;
    if (!(std::cin >> choice) || choice < 1 || choice > 4) return false;
    const char* names[] = { "greedy", "graphcut", "dp", "energy" };
    algo = names[choice - 1];

    std::printf("Energy: 1) backward (Sobel)  2) forward\n> ");
    int e = 0;
    if (!(std::cin >> e) || (e != 1 && e != 2)) return false;
    mode = e == 2 ? "forward" : "backward";

    if (algo != "energy")
    {
        std::printf("Target width and height (max %d %d):\n> ", img.cols, img.rows);
        if (!(std::cin >> w >> h)) return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        Usage(argv[0]);
        return 1;
    }
    const std::string input = argv[1], output = argv[2];

    try
    {
        const cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::printf("could not read %s\n", input.c_str());
            return 1;
        }

        std::string algo, mode = "backward";
        int w = img.cols, h = img.rows;
        if (argc == 3)
        {
            if (!AskOptions(img, algo, mode, w, h))
            {
                std::printf("invalid choice\n");
                return 1;
            }
        }
        else
        {
            algo = argv[3];
            if (algo == "energy")
            {
                if (argc > 4) mode = argv[4];
            }
            else if (algo == "greedy" || algo == "graphcut" || algo == "dp")
            {
                if (argc < 6)
                {
                    Usage(argv[0]);
                    return 1;
                }
                w = std::stoi(argv[4]);
                h = std::stoi(argv[5]);
                if (argc > 6) mode = argv[6];
            }
            else
            {
                Usage(argv[0]);
                return 1;
            }
        }
        if (mode != "backward" && mode != "forward")
        {
            std::printf("energy type must be 'backward' or 'forward'\n");
            return 1;
        }

        if (algo == "energy") return SaveEnergy(img, input, output, mode);
        return RunCarve(img, output, algo, w, h, mode);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}
