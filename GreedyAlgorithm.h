#pragma once
// ============================================================================
// Seam carving by a Greedy Algorithm.
//
// Instead of the DP's globally optimal seam, the greedy seam is built by a
// purely local, row-by-row (or column-by-column) decision:
//   * VERTICAL seam: start at the minimum-energy pixel of row 0; in every
//     following row step to whichever of the three neighbours below
//     (down-left, down, down-right) has the lowest energy.
//   * HORIZONTAL seam: the same on columns - start at the minimum-energy pixel
//     of column 0, then step right / up-right / down-right.
// No look-ahead and no cumulative table, so one seam costs O(rows) (vertical)
// or O(cols) (horizontal) after the energy map, versus O(rows * cols) for DP.
// The price: the seam is NOT guaranteed optimal - its cost is always >= the
// DP / Graph Cut seam cost, and it can be trapped by a cheap pixel that leads
// into an expensive region.
//
// Tie-breaking (fixed so results are deterministic and testable):
//   * start pixel : lowest index among equal minima;
//   * each step   : prefer straight (down / right), then the smaller index
//                   (left / up), then the larger index (right / down).
//
// The energy map is DP::ComputeEnergy, so Greedy, DP and Graph Cut are
// compared on identical cost maps. OpenCV is only used for image I/O, colour
// conversion and the Sobel filter; no content-aware resizing function from
// any library is used.
// ============================================================================
#include "DynamicProgramming.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace Greedy
{
    // Same energy options as DP (Backward = Sobel, Forward = |I[i][j+1] - I[i][j-1]|).
    using EnergyType = DP::EnergyType;

    struct SeamResult
    {
        std::vector<int> seam;   // vertical: seam[row] = column; horizontal: seam[col] = row
        int64_t cost = 0;        // sum of energy values along the seam
    };

    // Greedy seam on an energy map of type CV_32SC1 (as returned by
    // DP::ComputeEnergy). The seam is 8-connected: consecutive entries differ
    // by at most 1. Throws std::invalid_argument on an empty or non-CV_32SC1 map.
    SeamResult FindVerticalSeam(const cv::Mat& energy);
    SeamResult FindHorizontalSeam(const cv::Mat& energy);

    // Remove one pixel per row (vertical) / per column (horizontal), in place.
    // Works on any Mat type (image or energy map). img shrinks by one column /
    // row. Throws std::invalid_argument if the seam length or any index does
    // not match img.
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam);
    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam);

    struct CarveStats
    {
        // Times and costs ACCUMULATE if the same stats object is passed to
        // several Carve calls (same behaviour as GraphCut::CarveStats).
        double seamSeconds = 0.0;        // time spent finding seams (energy + greedy walk)
        double totalSeconds = 0.0;       // whole Carve call, including removals
        int64_t totalCost = 0;           // sum of all removed seam costs
        std::vector<int64_t> seamCosts;  // cost of each removed seam, in removal order
        // Original-image (x = col, y = row) coordinates of every removed seam,
        // filled only when recordSeams is true. Used by DrawSeams.
        std::vector<std::vector<cv::Point>> verticalSeams, horizontalSeams;
    };

    // Shrink img to newWidth x newHeight: vertical seams first, then
    // horizontal. The energy map is recomputed after every removal.
    // Requires 1 <= newWidth <= img.cols and 1 <= newHeight <= img.rows,
    // otherwise throws std::invalid_argument. stats may be nullptr.
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);

    // Copy of the original image with all removed seams drawn
    // (red = vertical, green = horizontal). Needs stats from a Carve call
    // made with recordSeams = true.
    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats);
}
