/*!************************************************************************
\file GreedyAlgorithm.cpp
\author Ethan Ng Yong Le
\par DP email: 2400557@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This source file contains the definitions for seam carving by a Greedy
Algorithm: the greedy vertical and horizontal seams, which make a purely
local choice row by row (or column by column), and Greedy::Carve, which
runs the shared remove-and-repeat loop (SeamCarving::CarveWith) with the
greedy seam finder.
- CheckEnergy(energy, who)
validates an energy map and throws with the caller's name if it is
invalid
- ArgMin(n, get)
returns the index of the smallest value, lowest index on ties
- Step(p, n, get)
returns the cheapest next index among p, p-1 and p+1
- FindVerticalSeam(energy)
builds the greedy vertical seam row by row
- FindHorizontalSeam(energy)
builds the greedy horizontal seam column by column
- Carve(img, newWidth, newHeight, energy, stats, recordSeams)
shrinks an image by repeatedly removing the greedy seam
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "GreedyAlgorithm.h"

#include <stdexcept>
#include <string>

namespace Greedy
{
    // ========================================================================
    // Greedy seam finding
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	checks that an energy map is non-empty, of type CV_32SC1 and has no
    	negative values, the same rule as DP and Graph Cut
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
        // Same rule as DP and Graph Cut: energies are non-negative.
        double minVal;
        cv::minMaxLoc(energy, &minVal);
        if (minVal < 0)
            throw std::invalid_argument(std::string(who) + ": energy values must be non-negative");
    }

    /************************************************************************/
    /*!
      \brief
    	Index of the smallest value in [0, n); ties -> lowest index.
      \tparam Get
    	A callable taking an index (int) and returning the value at that index
      \param n
      int
    	The number of values, indices 0 to n - 1
      \param get
      Get
    	Returns the value at an index
      \return int
      integer
      returns the index of the smallest value (the lowest index among equal
      minima)
    */
    /************************************************************************/
    template <typename Get>
    static int ArgMin(int n, Get get)
    {
        int best = 0;
        for (int k = 1; k < n; ++k)
            if (get(k) < get(best)) best = k;
        return best;
    }

    /************************************************************************/
    /*!
      \brief
    	chooses the next position of a greedy seam.
    	Next index from position p among {p, p-1, p+1} (clipped to [0, n)).
    	Ties: straight first, then the smaller index, then the larger one -
    	a candidate only wins if it is STRICTLY cheaper than the current best.
      \tparam Get
    	A callable taking an index (int) and returning the value at that index
      \param p
      int
    	The current position (column or row index)
      \param n
      int
    	The number of positions; candidates outside [0, n) are skipped
      \param get
      Get
    	Returns the energy at an index in the next row (or column)
      \return int
      integer
      returns the index of the cheapest candidate
    */
    /************************************************************************/
    template <typename Get>
    static int Step(int p, int n, Get get)
    {
        int best = p;
        if (p - 1 >= 0 && get(p - 1) < get(best)) best = p - 1;
        if (p + 1 < n && get(p + 1) < get(best)) best = p + 1;
        return best;
    }

    /************************************************************************/
    /*!
      \brief
    	builds a greedy vertical seam: it starts at the minimum-energy pixel of
    	row 0 and, in every following row, steps to whichever of the three
    	pixels below (down-left, down, down-right) has the lowest energy. There
    	is no look-ahead, so the seam is not guaranteed to be optimal.
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \return SeamResult
      seam result
      returns the seam (seam[i] = column removed in row i) and its total
      energy
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    SeamResult FindVerticalSeam(const cv::Mat& energy)
    {
        CheckEnergy(energy, "Greedy::FindVerticalSeam");
        const int m = energy.rows, n = energy.cols;
        SeamResult r;
        r.seam.resize(m);

        const int* row0 = energy.ptr<int>(0);
        int j = ArgMin(n, [row0](int k) { return row0[k]; });
        r.seam[0] = j;
        r.cost = row0[j];
        for (int i = 1; i < m; ++i)
        {
            const int* row = energy.ptr<int>(i);
            j = Step(j, n, [row](int k) { return row[k]; });
            r.seam[i] = j;
            r.cost += row[j];
        }
        return r;
    }

    /************************************************************************/
    /*!
      \brief
    	builds a greedy horizontal seam: it starts at the minimum-energy pixel
    	of column 0 and, in every following column, steps to whichever of right,
    	up-right and down-right has the lowest energy (ties: straight, then up,
    	then down).
      \param energy
      const cv::Mat&
    	The energy map, a non-empty CV_32SC1 matrix with no negative values
      \return SeamResult
      seam result
      returns the seam (seam[j] = row removed in column j) and its total
      energy
      \exception std::invalid_argument
    	thrown if energy is empty, is not CV_32SC1, or contains a negative value
    */
    /************************************************************************/
    SeamResult FindHorizontalSeam(const cv::Mat& energy)
    {
        CheckEnergy(energy, "Greedy::FindHorizontalSeam");
        const int m = energy.rows, n = energy.cols;
        SeamResult r;
        r.seam.resize(n);

        int i = ArgMin(m, [&energy](int k) { return energy.at<int>(k, 0); });
        r.seam[0] = i;
        r.cost = energy.at<int>(i, 0);
        for (int j = 1; j < n; ++j)
        {
            i = Step(i, m, [&energy, j](int k) { return energy.at<int>(k, j); });
            r.seam[j] = i;
            r.cost += energy.at<int>(i, j);
        }
        return r;
    }

    // ========================================================================
    // Carving: the shared remove-and-repeat loop with the greedy seam finder
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	shrinks an image to newWidth x newHeight by repeatedly removing the
    	greedy seam. It runs the shared SeamCarving::CarveWith loop with
    	FindVerticalSeam as the seam finder: vertical seams first, then
    	horizontal seams on the transposed image, recomputing the energy map
    	after every removal. On the transposed image a greedy vertical seam is
    	exactly a greedy horizontal seam of the original (same tie-breaking).
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
    	Where timing, seam costs and recorded seams are added; they accumulate
    	over calls; may be nullptr
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
        // On the transposed image a greedy vertical seam is exactly a greedy
        // horizontal seam of the original (same tie-breaking: up before down).
        return SeamCarving::CarveWith(img, newWidth, newHeight, energy,
                                      [](const cv::Mat& e) { return FindVerticalSeam(e).seam; },
                                      stats, recordSeams, "Greedy::Carve");
    }
}
