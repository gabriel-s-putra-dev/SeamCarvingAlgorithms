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
// The energy map, seam removal, the carving loop and the seam overlay are
// shared with DP and Graph Cut (SeamCarving.h), so the three methods are
// compared on identical cost maps and differ only in how each seam is chosen.
// OpenCV is only used for image I/O, colour conversion and the Sobel filter;
// no content-aware resizing function from any library is used.
// ============================================================================
#include "SeamCarving.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace Greedy
{
    // Shared energy map, removal, stats and overlay (see SeamCarving.h).
    using EnergyType = SeamCarving::EnergyType;   // Backward = Sobel, Forward = |I[i][j+1] - I[i][j-1]|
    using SeamCarving::ComputeEnergy;
    using SeamCarving::RemoveVerticalSeam;
    using SeamCarving::RemoveHorizontalSeam;
    using SeamCarving::CarveStats;
    using SeamCarving::DrawSeams;

    struct SeamResult
    {
        std::vector<int> seam;   // vertical: seam[row] = column; horizontal: seam[col] = row
        int64_t cost = 0;        // sum of energy values along the seam
    };

    // Greedy seam on an energy map of type CV_32SC1 (as returned by
    // DP::ComputeEnergy). The seam is 8-connected: consecutive entries differ
    // by at most 1. Throws std::invalid_argument on an empty or non-CV_32SC1
    // map, or one with negative values.
    //
    // NOTE on FORWARD energy: it compares left and right neighbours, so it is
    // the right cost for VERTICAL seams only. For a horizontal seam with
    // forward energy use FindVerticalSeam(ComputeEnergy(img.t(), Forward))
    // on the transposed image (Carve does this). See SeamCarving::ComputeEnergy.
    SeamResult FindVerticalSeam(const cv::Mat& energy);
    SeamResult FindHorizontalSeam(const cv::Mat& energy);

    // Shrink img to newWidth x newHeight: vertical seams first, then
    // horizontal. The energy map is recomputed after every removal.
    // Runs SeamCarving::CarveWith with the greedy FindVerticalSeam; see there
    // for stats (which accumulate over calls), recordSeams and errors.
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);
}
