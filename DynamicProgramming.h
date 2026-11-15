#pragma once
// ============================================================================
// Seam carving by Dynamic Programming.
//
// Step (a): energy map.
// Step (b): DP minimum seam (vertical and horizontal).
// Step (c): remove the seam, recompute the energy map, repeat until the
//           image has the requested size (Carve).
//
// OpenCV is only used for colour conversion and the Sobel filter; no
// content-aware resizing function from any library is used.
//
// The energy formulas are the same as GraphCut::ComputeEnergy so that DP,
// Greedy and Graph Cut can later be compared on identical cost maps.
// ============================================================================
#include <opencv2/core.hpp>

#include <vector>

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
    // energy must be non-empty CV_32SC1, otherwise std::invalid_argument.
    cv::Mat CumulativeEnergy(const cv::Mat& energy);

    // Minimum-energy VERTICAL seam: returns a vector of size energy.rows where
    // seam[i] is the column removed in row i. Consecutive entries differ by
    // at most 1. Ties are broken towards the left.
    // energy must be non-empty CV_32SC1, otherwise std::invalid_argument.
    std::vector<int> FindVerticalSeam(const cv::Mat& energy);

    // Minimum-energy HORIZONTAL seam: returns a vector of size energy.cols
    // where seam[j] is the row removed in column j. Consecutive entries differ
    // by at most 1. Implemented by transposing the energy map and reusing
    // FindVerticalSeam.
    // energy must be non-empty CV_32SC1, otherwise std::invalid_argument.
    std::vector<int> FindHorizontalSeam(const cv::Mat& energy);

    // Total energy of a seam (useful for testing / comparing with Greedy and
    // Graph Cut). vertical = true: seam[i] is a column per row;
    // vertical = false: seam[j] is a row per column.
    // Throws std::invalid_argument if energy is empty / not CV_32SC1, if the
    // seam has the wrong length, if any index is outside the image, or if the
    // seam is not connected (consecutive entries differing by more than 1).
    long long SeamEnergy(const cv::Mat& energy, const std::vector<int>& seam, bool vertical);

    // ------------------------------------------------------------------------
    // (c) Remove the seam and repeat
    // ------------------------------------------------------------------------

    // Removes one pixel per row: seam[i] is the column removed in row i.
    // Works for any Mat type (image, energy map, coordinate map, ...).
    // Throws std::invalid_argument if img is empty or only 1 column wide,
    // if seam.size() != img.rows, or if any index is outside [0, cols).
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam);

    // Removes one pixel per column: seam[j] is the row removed in column j.
    // Same rules as RemoveVerticalSeam with rows and columns swapped.
    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam);

    // Carve ADDS to these fields, so use a fresh CarveStats for every run.
    struct CarveStats
    {
        double seamSeconds = 0.0;          // time spent on energy + seam finding
                                           // (same meaning as GraphCut::CarveStats)
        long long totalCost = 0;           // sum of the energies of all removed seams
        std::vector<long long> seamCosts;  // energy of each removed seam, in order
        // Original-image (x = col, y = row) coordinates of every removed seam,
        // filled only when recordSeams is true. Used by DrawSeams.
        std::vector<std::vector<cv::Point>> verticalSeams, horizontalSeams;
    };

    // Shrinks img to newWidth x newHeight by repeatedly removing the minimum
    // energy seam. The energy map is recomputed from the current image after
    // every removal. Vertical seams (width) are removed first, then horizontal
    // seams (height), the same order as GraphCut::Carve.
    // Throws std::invalid_argument if img is empty or the target is not within
    // 1..original size in both directions (enlarging is not supported).
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight,
                  EnergyType energy = EnergyType::Backward,
                  CarveStats* stats = nullptr, bool recordSeams = false);

    // Original image with all recorded seams drawn on it
    // (red = vertical, green = horizontal). Needs Carve(..., stats, true).
    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats);
}
