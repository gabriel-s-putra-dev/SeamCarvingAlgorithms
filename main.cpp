// Temporary driver: computes the DP energy map of an image and saves it as a
// viewable 8-bit picture (bright = high energy = edges).
//   ./energy <input image> <output.png> [backward|forward]
// Will be replaced by the algorithm-selection menu once seam finding exists.
#include "DynamicProgramming.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <input image> <output.png> [backward|forward]\n", argv[0]);
        return 1;
    }
    const std::string mode = argc > 3 ? argv[3] : "backward";
    if (mode != "backward" && mode != "forward")
    {
        std::printf("energy type must be 'backward' or 'forward'\n");
        return 1;
    }

    try
    {
        const cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::printf("could not read %s\n", argv[1]);
            return 1;
        }
        const cv::Mat energy = DP::ComputeEnergy(
            img, mode == "forward" ? DP::EnergyType::Forward : DP::EnergyType::Backward);

        double minVal, maxVal;
        cv::minMaxLoc(energy, &minVal, &maxVal);
        cv::Mat vis;
        cv::normalize(energy, vis, 0, 255, cv::NORM_MINMAX, CV_8U);   // stretch to 0..255 for viewing
        if (!cv::imwrite(argv[2], vis))
        {
            std::printf("could not write %s\n", argv[2]);
            return 1;
        }
        std::printf("%s energy of %s (%dx%d): min %.0f, max %.0f -> %s\n", mode.c_str(), argv[1],
                    img.cols, img.rows, minVal, maxVal, argv[2]);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    return 0;
}
