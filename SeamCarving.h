#pragma once
// ============================================================================
// Shared seam-carving pipeline, used by all three seam-finding methods
// (DynamicProgramming, GreedyAlgorithm, GraphCutting):
//
//   * the energy map (step a),
//   * removing a seam from an image (step c),
//   * the "remove the seam and repeat" loop (CarveWith), which takes the
//     seam finder as a parameter, so DP, Greedy and Graph Cut only differ in
//     how they choose each seam,
//   * the statistics of a carve (CarveStats) and the seam overlay (DrawSeams).
//
// Because every method uses this one copy, all three are compared on exactly
// the same energy map, the same removal and the same carving order.
//
// OpenCV is only used for colour conversion, the Sobel filter and transpose;
// no content-aware resizing function from any library is used.
// ============================================================================
#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <vector>

namespace SeamCarving
{
    enum class EnergyType { Backward, Forward };

    // Energy / cost map of img, returned as CV_32SC1 (same size, values >= 0).
    //   Backward: e[i][j] = |Sobel_x| + |Sobel_y| on the grayscale image
    //             (3x3 kernel, reflected borders).
    //   Forward : simplified per-pixel forward cost
    //             c[i][j] = |I[i][j+1] - I[i][j-1]|  (border pixels replicated),
    //             i.e. the new intensity edge created when (i, j) is removed
    //             and its left and right neighbours meet.
    //
    // NOTE on FORWARD energy: it compares LEFT and RIGHT neighbours, so it is
    // the right cost for VERTICAL seams only. For a horizontal seam with
    // forward energy, compute the energy on the TRANSPOSED image and find a
    // vertical seam there:
    //     std::vector<int> seam = DP::FindVerticalSeam(ComputeEnergy(img.t(), EnergyType::Forward));
    //     // seam[j] = row removed in column j of img
    // CarveWith already does this. Backward (Sobel) energy is the same in both
    // directions, so FindHorizontalSeam(ComputeEnergy(img)) is fine for it.
    //
    // Accepts 1-, 3- (BGR) or 4-channel (BGRA) 8-bit images.
    // Throws std::invalid_argument on an empty image.
    cv::Mat ComputeEnergy(const cv::Mat& img, EnergyType type = EnergyType::Backward);

    // Removes one pixel per row: seam[i] is the column removed in row i.
    // Works for any Mat type (image, energy map, coordinate map, ...).
    // Throws std::invalid_argument if img is empty or only 1 column wide,
    // if seam.size() != img.rows, or if any index is outside [0, cols);
    // img is left unchanged when it throws.
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam);

    // Removes one pixel per column: seam[j] is the row removed in column j.
    // Same rules as RemoveVerticalSeam with rows and columns swapped.
    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam);

    // Statistics of a carve. CarveWith ADDS to these fields, so use a fresh
    // CarveStats for every run (or reuse one on purpose to accumulate).
    struct CarveStats
    {
        double seamSeconds = 0.0;         // energy map + seam finding
        double totalSeconds = 0.0;        // whole carve, including removals
        int64_t totalCost = 0;            // sum of the energies of all removed seams
        std::vector<int64_t> seamCosts;   // energy of each removed seam, in order
        // Original-image (x = col, y = row) coordinates of every removed seam,
        // filled only when recordSeams is true. Used by DrawSeams.
        std::vector<std::vector<cv::Point>> verticalSeams, horizontalSeams;
    };

    // Finds a VERTICAL seam on a CV_32SC1 energy map: returns seam[row] = column.
    using VerticalSeamFinder = std::function<std::vector<int>(const cv::Mat& energy)>;

    // Shrinks img to newWidth x newHeight by repeatedly removing a seam:
    //   energy of the CURRENT image -> findSeam -> remove -> repeat.
    // Vertical seams (width) are removed first. Horizontal seams (height) are
    // then removed as vertical seams of the transposed image, so findSeam only
    // ever needs to find vertical seams.
    // The cost of each seam (sum of its energies) is added to stats.
    // Seams are recorded into stats->verticalSeams / horizontalSeams only when
    // recordSeams is true AND stats is not null; the map of original pixel
    // coordinates needed for that is only built in that case.
    // Throws std::invalid_argument if img is empty or the target is not within
    // 1..original size in both directions (enlarging is not supported). who
    // is the function name used in error messages (e.g. "DP::Carve").
    cv::Mat CarveWith(const cv::Mat& img, int newWidth, int newHeight, EnergyType energy,
                      const VerticalSeamFinder& findSeam, CarveStats* stats = nullptr,
                      bool recordSeams = false, const char* who = "SeamCarving::CarveWith");

    // Copy of the original image (as BGR) with all recorded seams drawn on it
    // (red = vertical, green = horizontal). Needs stats from a carve made with
    // recordSeams = true. Throws std::invalid_argument if original is empty /
    // not 8-bit, or a seam point lies outside it (stats from a different image).
    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats);
}
