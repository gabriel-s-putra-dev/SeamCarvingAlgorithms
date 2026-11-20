/*!************************************************************************
\file DynamicProgramming.cpp
\author Gabriel Sebastian Putra, Ho Jun
\par DP email: 2400735@sit.singaporetech.edu.sg, 2401700@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This source file contains the definitions for step (b) of seam carving
by Dynamic Programming: the cumulative minimum energy table, the minimum
vertical and horizontal seams found from it, the energy of a seam, and
DP::Carve, which runs the shared remove-and-repeat loop
(SeamCarving::CarveWith) with the DP seam finder.
- CheckEnergy(energy, who)
validates an energy map and throws with the caller's name if it is
invalid
- BuildCumulative(energy)
builds the cumulative minimum energy table M without validating
- CumulativeEnergy(energy)
validates the energy map and returns its cumulative table M
- FindVerticalSeam(energy)
finds the leftmost minimum-energy vertical seam by backtracking through
M
- FindHorizontalSeam(energy)
finds the minimum-energy horizontal seam on the transposed energy map
- SeamEnergy(energy, seam, vertical)
validates a seam and returns the sum of its energies
- Carve(img, newWidth, newHeight, energy, stats, recordSeams)
shrinks an image by repeatedly removing the DP minimum seam
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "DynamicProgramming.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Step (a), the energy map, and the removal / repeat loop of step (c) are in
// SeamCarving.cpp, shared with Greedy and Graph Cut.

namespace DP
{
    // ========================================================================
    // (b) Minimum seam by dynamic programming
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	checks that an energy map is non-empty, of type CV_32SC1 and has no
    	negative values.
    	Every public function validates with its own name, so an error message
    	always points at the function the caller actually used.
      \param energy
      const cv::Mat&
    	The energy map to validate
      \param who
      const char*
    	The name of the calling function, used at the start of the error message
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    static void CheckEnergy(const cv::Mat& energy, const char* who)
    {
        if (energy.empty())
            throw std::invalid_argument(std::string(who) + ": energy map is empty");
        if (energy.type() != CV_32SC1)
            throw std::invalid_argument(std::string(who) + ": energy map must be CV_32SC1");
        // Energies are non-negative by definition (ComputeEnergy never produces
        // negatives); rejected here so DP, Greedy and Graph Cut accept the
        // same inputs.
        double minVal;
        cv::minMaxLoc(energy, &minVal);
        if (minVal < 0)
            throw std::invalid_argument(std::string(who) + ": energy values must be non-negative");
    }

    /************************************************************************/
    /*!
      \brief
    	builds the cumulative minimum energy table M of an energy map.
    	Builds the table without validating (callers have already checked).
    	M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1]).
    	Each cell stores the cheapest total energy of any vertical path that
    	starts in row 0 and ends at (i, j). Overall cost O(rows * cols).
    	Stored as 64-bit (double holds integers up to 2^53 exactly) so the sums
    	cannot overflow, consistent with SeamEnergy.
      \param energy
      const cv::Mat&
    	The energy map, already validated by the caller
      \return cv::Mat
      OpenCV matrix
      returns the CV_64FC1 cumulative table M, the same size as energy
    */
    /************************************************************************/
    static cv::Mat BuildCumulative(const cv::Mat& energy)
    {
        const int rows = energy.rows;
        const int cols = energy.cols;

        cv::Mat M;
        energy.convertTo(M, CV_64F);         // row 0 is just the energy itself

        for (int i = 1; i < rows; ++i)
        {
            const double* prev = M.ptr<double>(i - 1);
            double* cur = M.ptr<double>(i);
            for (int j = 0; j < cols; ++j)
            {
                double best = prev[j];                                // straight up
                if (j > 0)        best = std::min(best, prev[j - 1]); // up-left
                if (j < cols - 1) best = std::min(best, prev[j + 1]); // up-right
                cur[j] += best;                                       // cur[j] held e[i][j]
            }
        }
        return M;
    }

    /************************************************************************/
    /*!
      \brief
    	validates the energy map, then builds its cumulative minimum energy
    	table M for a vertical seam, where M[i][j] is the cheapest total energy
    	of any seam from row 0 ending at (i, j)
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \return cv::Mat
      OpenCV matrix
      returns the CV_64FC1 cumulative table M, the same size as energy
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    cv::Mat CumulativeEnergy(const cv::Mat& energy)
    {
        CheckEnergy(energy, "DP::CumulativeEnergy");
        return BuildCumulative(energy);
    }

    /************************************************************************/
    /*!
      \brief
    	finds the minimum-energy vertical seam by dynamic programming.
    	Backtracking: start at the cheapest cell of the last row, then walk up,
    	each time moving to whichever of the 3 cells above has the smallest M.
    	Ties are broken towards the left, both in the bottom row and on the way
    	up, so the result is the leftmost of all minimum-energy seams.
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \return std::vector<int>
      vector of int
      returns the seam, one entry per row: seam[i] is the column removed in
      row i
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    std::vector<int> FindVerticalSeam(const cv::Mat& energy)
    {
        CheckEnergy(energy, "DP::FindVerticalSeam");

        const cv::Mat M = BuildCumulative(energy);
        const int rows = M.rows;
        const int cols = M.cols;

        std::vector<int> seam(rows);

        // End of the seam: minimum of the bottom row (leftmost on ties).
        const double* last = M.ptr<double>(rows - 1);
        int j = 0;
        for (int c = 1; c < cols; ++c)
            if (last[c] < last[j]) j = c;
        seam[rows - 1] = j;

        // Walk upwards.
        for (int i = rows - 2; i >= 0; --i)
        {
            const double* row = M.ptr<double>(i);
            const int lo = std::max(j - 1, 0);
            const int hi = std::min(j + 1, cols - 1);
            int bestCol = lo;
            for (int c = lo + 1; c <= hi; ++c)
                if (row[c] < row[bestCol]) bestCol = c;
            j = bestCol;
            seam[i] = j;
        }
        return seam;
    }

    /************************************************************************/
    /*!
      \brief
    	finds the minimum-energy horizontal seam.
    	A horizontal seam in the image is a vertical seam in the transposed
    	image, so: transpose, find vertical seam, and read the result as
    	"row index for every column".
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \return std::vector<int>
      vector of int
      returns the seam, one entry per column: seam[j] is the row removed in
      column j
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    std::vector<int> FindHorizontalSeam(const cv::Mat& energy)
    {
        CheckEnergy(energy, "DP::FindHorizontalSeam");
        cv::Mat energyT;
        cv::transpose(energy, energyT);
        return FindVerticalSeam(energyT);
    }

    /************************************************************************/
    /*!
      \brief
    	checks that a seam fits the energy map and is connected, then adds up
    	the energy of every pixel on it. Everything is validated before the
    	matrix is read, because at() is only bounds-checked in Debug builds
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \param seam
      const std::vector<int>&
    	The seam: a column per row if vertical, a row per column if not
      \param vertical
      bool
    	true for a vertical seam, false for a horizontal seam
      \return long long
      64-bit integer
      returns the total energy of the seam
      \exception std::invalid_argument
    	thrown if energy is invalid, the seam has the wrong length, an index is
    	outside the image, or two consecutive entries differ by more than 1
    */
    /************************************************************************/
    long long SeamEnergy(const cv::Mat& energy, const std::vector<int>& seam, bool vertical)
    {
        CheckEnergy(energy, "DP::SeamEnergy");

        // vertical:   one entry per row,    each entry is a column in [0, cols)
        // horizontal: one entry per column, each entry is a row    in [0, rows)
        const int length = vertical ? energy.rows : energy.cols;
        const int limit = vertical ? energy.cols : energy.rows;

        if (static_cast<int>(seam.size()) != length)
            throw std::invalid_argument(vertical
                ? "DP::SeamEnergy: seam size must equal rows"
                : "DP::SeamEnergy: seam size must equal cols");

        // Validate everything BEFORE reading the matrix: at() is only
        // bounds-checked in Debug builds, so an out-of-range index would be
        // undefined behaviour in Release.
        for (int k = 0; k < length; ++k)
        {
            if (seam[k] < 0 || seam[k] >= limit)
                throw std::invalid_argument("DP::SeamEnergy: seam index out of range");
            if (k > 0 && std::abs(seam[k] - seam[k - 1]) > 1)
                throw std::invalid_argument("DP::SeamEnergy: seam is not connected");
        }

        long long total = 0;
        for (int k = 0; k < length; ++k)
            total += vertical ? energy.at<int>(k, seam[k])
                              : energy.at<int>(seam[k], k);
        return total;
    }

    // ========================================================================
    // (c) Remove the seam and repeat: the shared loop with the DP seam finder
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	shrinks an image to newWidth x newHeight by repeatedly removing the
    	minimum energy seam found by dynamic programming. It runs the shared
    	SeamCarving::CarveWith loop with FindVerticalSeam as the seam finder:
    	vertical seams first, then horizontal seams on the transposed image,
    	recomputing the energy map after every removal.
      \param img
      const cv::Mat&
    	The image to shrink (it is not modified)
      \param newWidth
      int
    	The target width, between 1 and img.cols
      \param newHeight
      int
    	The target height, between 1 and img.rows
      \param energy
      EnergyType
    	The energy to use: Backward (Sobel) or Forward
      \param stats
      CarveStats*
    	Where timing, seam costs and recorded seams are added; may be nullptr
      \param recordSeams
      bool
    	Whether to record every removed seam in original-image coordinates (only
    	when stats is not null)
      \return cv::Mat
      OpenCV matrix
      returns the resized image, the same type as img
      \exception std::invalid_argument
    	thrown if img is empty or the target size is not between 1 and the
    	original size
    */
    /************************************************************************/
    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight, EnergyType energy,
                  CarveStats* stats, bool recordSeams)
    {
        return SeamCarving::CarveWith(img, newWidth, newHeight, energy,
                                      [](const cv::Mat& e) { return FindVerticalSeam(e); },
                                      stats, recordSeams, "DP::Carve");
    }
}
