#pragma once
// ============================================================================
// Seam carving by Dynamic Programming.
//
// Step (a): energy map.  (Steps (b) DP minimum seam and (c) remove & repeat
// are not implemented yet.)
//
// OpenCV is only used for colour conversion and the Sobel filter; no
// content-aware resizing function from any library is used.
//
// The energy formulas are the same as GraphCut::ComputeEnergy so that DP,
// Greedy and Graph Cut can later be compared on identical cost maps.
// ============================================================================
#include <opencv2/core.hpp>

namespace DP
{
    enum class EnergyType { Backward, Forward };

    // Energy / cost map of img, returned as CV_32SC1 (same size, values >= 0).
    //   Backward: e[i][j] = |Sobel_x| + |Sobel_y| on the grayscale image
    //             (3x3 kernel, reflected borders).
    //   Forward : simplified per-pixel forward cost
    //             c[i][j] = |I[i][j+1] - I[i][j-1]|  (border pixels replicated),
    //             i.e. the new intensity edge created when (i, j) is removed
    //             and its left and right neighbours meet.
    // Accepts 1-, 3- (BGR) or 4-channel (BGRA) 8-bit images.
    // Throws std::invalid_argument on an empty image.
    cv::Mat ComputeEnergy(const cv::Mat& img, EnergyType type = EnergyType::Backward);
}
