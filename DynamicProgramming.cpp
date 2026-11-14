#include "DynamicProgramming.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace DP
{
    // ========================================================================
    // (a) Energy map
    // ========================================================================
    static cv::Mat ToGray(const cv::Mat& img)
    {
        cv::Mat gray;
        if (img.channels() == 3)      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4) cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
        else                          gray = img;
        return gray;
    }

    // e = |dI/dx| + |dI/dy|, gradients from a 3x3 Sobel filter.
    static cv::Mat BackwardEnergy(const cv::Mat& gray)
    {
        cv::Mat g, gx, gy;
        gray.convertTo(g, CV_64F);
        cv::Sobel(g, gx, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT);
        cv::Sobel(g, gy, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT);
        const cv::Mat mag = cv::abs(gx) + cv::abs(gy);
        cv::Mat energy;
        mag.convertTo(energy, CV_32S);   // Sobel of integer pixels is integer: exact
        return energy;
    }

    // c[i][j] = |I[i][j+1] - I[i][j-1]|, out-of-range neighbours replicated.
    static cv::Mat ForwardEnergy(const cv::Mat& gray)
    {
        cv::Mat g;
        gray.convertTo(g, CV_32S);
        cv::Mat energy(g.size(), CV_32S);
        const int cols = g.cols;
        for (int i = 0; i < g.rows; ++i)
        {
            const int* row = g.ptr<int>(i);
            int* out = energy.ptr<int>(i);
            for (int j = 0; j < cols; ++j)
            {
                const int left = row[std::max(j - 1, 0)];
                const int right = row[std::min(j + 1, cols - 1)];
                out[j] = std::abs(right - left);
            }
        }
        return energy;
    }

    cv::Mat ComputeEnergy(const cv::Mat& img, EnergyType type)
    {
        if (img.empty())
            throw std::invalid_argument("DP::ComputeEnergy: image is empty");
        const cv::Mat gray = ToGray(img);
        return type == EnergyType::Backward ? BackwardEnergy(gray) : ForwardEnergy(gray);
    }
}
