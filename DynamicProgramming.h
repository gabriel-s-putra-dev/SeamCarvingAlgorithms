#pragma once
// ============================================================================
// Seam carving by Dynamic Programming.
//
// Step (a): energy map                -> SeamCarving::ComputeEnergy (shared)
// Step (b): DP minimum seam (vertical and horizontal)   -> this file
// Step (c): remove the seam, recompute the energy map, repeat until the
//           image has the requested size  -> DP::Carve, which runs the shared
//           SeamCarving::CarveWith loop with the DP seam finder
//
// Steps (a) and (c) are shared with Greedy and Graph Cut (SeamCarving.h), so
// all three methods use the same energy map, removal and carving order and
// differ only in how each seam is chosen. The shared names are re-exported
// below, so DP::ComputeEnergy, DP::CarveStats, DP::DrawSeams, ... still work.
//
// OpenCV is only used for colour conversion, the Sobel filter and transpose;
// no content-aware resizing function from any library is used.
// ============================================================================
#include "SeamCarving.h"

#include <opencv2/core.hpp>

#include <vector>

namespace DP
{
    // ------------------------------------------------------------------------
    // (a) Energy map, and the shared parts of (c) - see SeamCarving.h
    // ------------------------------------------------------------------------
    using EnergyType = SeamCarving::EnergyType;
    using SeamCarving::ComputeEnergy;
    using SeamCarving::RemoveVerticalSeam;
    using SeamCarving::RemoveHorizontalSeam;
    using SeamCarving::CarveStats;
    using SeamCarving::DrawSeams;

    // ------------------------------------------------------------------------
    // (b) Minimum seam by dynamic programming
    // ------------------------------------------------------------------------

    // Cumulative minimum energy table M (CV_64FC1, same size as energy) for a
    // VERTICAL seam:
    //   M[0][j] = e[0][j]
    //   M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])
    // (neighbours outside the image are ignored).
    // The table is 64-bit so that sums cannot overflow, matching SeamEnergy.
    // A double holds every whole number up to 2^53 exactly, so the values are
    // exact integers for any realistic image size. (CV_64S is not used because
    // it only exists in OpenCV >= 4.12.)
    // energy must be a non-empty CV_32SC1 map with no negative values,
    // otherwise std::invalid_argument (this applies to every function in (b)).
    cv::Mat CumulativeEnergy(const cv::Mat& energy);

    // Minimum-energy VERTICAL seam: returns a vector of size energy.rows where
    // seam[i] is the column removed in row i. Consecutive entries differ by
    // at most 1. Ties are broken towards the left.
    std::vector<int> FindVerticalSeam(const cv::Mat& energy);

    // Minimum-energy HORIZONTAL seam: returns a vector of size energy.cols
    // where seam[j] is the row removed in column j. Consecutive entries differ
    // by at most 1. Implemented by transposing the energy map and reusing
    // FindVerticalSeam.
    // With FORWARD energy, use the transposed image instead (see the note at
    // SeamCarving::ComputeEnergy); Carve already does this.
    std::vector<int> FindHorizontalSeam(const cv::Mat& energy);

    // Total energy of a seam (useful for testing / comparing with Greedy and
    // Graph Cut). vertical = true: seam[i] is a column per row;
    // vertical = false: seam[j] is a row per column.
    // Throws std::invalid_argument if energy is empty / not CV_32SC1 / has
    // negative values, if the seam has the wrong length, if any index is
    // outside the image, or if the seam is not connected (consecutive entries
    // differing by more than 1).
    long long SeamEnergy(const cv::Mat& energy, const std::vector<int>& seam, bool vertical);

    // ------------------------------------------------------------------------
    // (c) Remove the seam and repeat
    // ------------------------------------------------------------------------

    // Shrinks img to newWidth x newHeight by repeatedly removing the minimum
    // energy (DP) seam. The energy map is recomputed from the current image
    // after every removal. Vertical seams (width) are removed first, then
    // horizontal seams (height). Runs SeamCarving::CarveWith with
    // FindVerticalSeam; see there for stats, recordSeams and errors.
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);
}
