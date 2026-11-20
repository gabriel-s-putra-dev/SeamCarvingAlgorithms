/*!************************************************************************
\file SeamCarving.cpp
\author Gabriel Sebastian Putra
\par DP email: 2400735@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This source file contains the definitions for the seam-carving pipeline
shared by Dynamic Programming, Greedy and Graph Cut: the energy map
(step a), seam removal and the remove-and-repeat loop (step c), and the
seam overlay. Each method only supplies its own seam finder to
CarveWith.
- ToGray(img)
converts a BGR or BGRA image to grayscale
- BackwardEnergy(gray)
computes the Sobel gradient energy |Gx| + |Gy|
- ForwardEnergy(gray)
computes the simplified forward energy |I[i][j+1] - I[i][j-1]|
- ComputeEnergy(img, type)
computes the backward or forward energy map of an image
- CheckRemoval(img, seam, length, limit, who)
validates a seam before it is removed
- RemoveVerticalUnchecked(img, seam)
removes a vertical seam without validating it
- RemoveVerticalSeam(img, seam)
validates and removes one pixel per row
- RemoveHorizontalSeam(img, seam)
validates and removes one pixel per column
- SecondsSince(t0)
returns the seconds elapsed since a time point
- RemoveSeams(img, coords, k, energyType, findSeam, stats, record, who)
removes k vertical seams one at a time, recomputing the energy each time
- CarveWith(img, newWidth, newHeight, energy, findSeam, stats, recordSeams, who)
shrinks an image to a target size with the given seam finder
- DrawSeams(original, stats)
draws all recorded seams on a copy of the original image
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "SeamCarving.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace SeamCarving
{
    // ========================================================================
    // (a) Energy map
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	converts an image to a single-channel grayscale image. 3-channel images
    	are treated as BGR and 4-channel images as BGRA; any other image is
    	returned as it is.
      \param img
      const cv::Mat&
    	The input image: 1-, 3- (BGR) or 4-channel (BGRA), 8-bit
      \return cv::Mat
      OpenCV matrix
      returns the grayscale image (the input itself if it was already single-
      channel)
    */
    /************************************************************************/
    static cv::Mat ToGray(const cv::Mat& img)
    {
        cv::Mat gray;
        if (img.channels() == 3)      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4) cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
        else                          gray = img;
        return gray;
    }

    /************************************************************************/
    /*!
      \brief
    	computes the backward energy map of a grayscale image.
    	e = |dI/dx| + |dI/dy|, gradients from a 3x3 Sobel filter.
    	The 3x3 Sobel filters use reflected borders. The result is exact because
    	the Sobel of integer pixels is an integer.
      \param gray
      const cv::Mat&
    	The single-channel grayscale image
      \return cv::Mat
      OpenCV matrix
      returns the CV_32SC1 energy map, the same size as gray
    */
    /************************************************************************/
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

    /************************************************************************/
    /*!
      \brief
    	computes the simplified forward energy map of a grayscale image: the new
    	intensity edge created when pixel (i, j) is removed and its left and
    	right neighbours meet.
    	c[i][j] = |I[i][j+1] - I[i][j-1]|, out-of-range neighbours replicated.
      \param gray
      const cv::Mat&
    	The single-channel grayscale image
      \return cv::Mat
      OpenCV matrix
      returns the CV_32SC1 energy map, the same size as gray
    */
    /************************************************************************/
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

    /************************************************************************/
    /*!
      \brief
    	computes the energy map of an image on its grayscale version, using
    	either the backward (Sobel) or the simplified forward energy. This is
    	step (a), shared by all three methods.
      \param img
      const cv::Mat&
    	The input image: 1-, 3- (BGR) or 4-channel (BGRA), 8-bit
      \param type
      EnergyType
    	EnergyType::Backward or EnergyType::Forward
      \return cv::Mat
      OpenCV matrix
      returns the CV_32SC1 energy map, the same size as img, with values >= 0
      \exception std::invalid_argument
    	thrown if img is empty
    */
    /************************************************************************/
    cv::Mat ComputeEnergy(const cv::Mat& img, EnergyType type)
    {
        if (img.empty())
            throw std::invalid_argument("SeamCarving::ComputeEnergy: image is empty");
        const cv::Mat gray = ToGray(img);
        return type == EnergyType::Backward ? BackwardEnergy(gray) : ForwardEnergy(gray);
    }

    // ========================================================================
    // (c) Remove the seam
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	checks that a seam can be removed from an image.
    	length = pixels in the seam (rows for vertical, cols for horizontal),
    	limit  = size of the dimension being shrunk (cols / rows).
    	Everything is validated BEFORE img is touched.
      \param img
      const cv::Mat&
    	The image the seam will be removed from
      \param seam
      const std::vector<int>&
    	The seam to check
      \param length
      int
    	The number of pixels the seam must have (rows for vertical, cols for
    	horizontal)
      \param limit
      int
    	The size of the dimension being shrunk; every index must be in [0,
    	limit)
      \param who
      const char*
    	The name of the calling function, used at the start of the error message
      \exception std::invalid_argument
    	thrown if img is empty, limit is below 2, the seam has the wrong length,
    	or an index is out of range
    */
    /************************************************************************/
    static void CheckRemoval(const cv::Mat& img, const std::vector<int>& seam,
                             int length, int limit, const char* who)
    {
        if (img.empty())
            throw std::invalid_argument(std::string(who) + ": image is empty");
        if (limit < 2)
            throw std::invalid_argument(std::string(who) + ": image is too small to remove a seam");
        if (static_cast<int>(seam.size()) != length)
            throw std::invalid_argument(std::string(who) + ": seam has the wrong length");
        for (int k = 0; k < length; ++k)
            if (seam[k] < 0 || seam[k] >= limit)
                throw std::invalid_argument(std::string(who) + ": seam index out of range");
    }

    /************************************************************************/
    /*!
      \brief
    	removes one pixel per row without validating the seam (callers have
    	already checked it).
    	Copies each row into a (cols - 1)-wide image, skipping the seam pixel.
    	Uses raw bytes (elemSize), so any type and channel count works.
      \param img
      cv::Mat&
    	The matrix to shrink (any type and channel count); replaced by the
    	result
      \param seam
      const std::vector<int>&
    	The seam: seam[i] is the column removed in row i
    */
    /************************************************************************/
    static void RemoveVerticalUnchecked(cv::Mat& img, const std::vector<int>& seam)
    {
        cv::Mat out(img.rows, img.cols - 1, img.type());
        const size_t es = img.elemSize();
        for (int i = 0; i < img.rows; ++i)
        {
            const int c = seam[i];
            const uchar* src = img.ptr(i);
            uchar* dst = out.ptr(i);
            std::memcpy(dst, src, c * es);                                          // left of seam
            std::memcpy(dst + c * es, src + (c + 1) * es, (img.cols - 1 - c) * es); // right of seam
        }
        img = out;
    }

    /************************************************************************/
    /*!
      \brief
    	validates a vertical seam, then removes it, so the image becomes one
    	column narrower.
      \param img
      cv::Mat&
    	The matrix to shrink (any type and channel count); replaced by the
    	result
      \param seam
      const std::vector<int>&
    	The seam: seam[i] is the column removed in row i
      \exception std::invalid_argument
    	thrown if img is empty or too small to remove a seam, the seam has the
    	wrong length, or an index is out of range; img is left unchanged
    */
    /************************************************************************/
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam)
    {
        CheckRemoval(img, seam, img.rows, img.cols, "SeamCarving::RemoveVerticalSeam");
        RemoveVerticalUnchecked(img, seam);
    }

    /************************************************************************/
    /*!
      \brief
    	validates a horizontal seam, then removes it, so the image becomes one
    	row shorter.
    	A horizontal seam of img is a vertical seam of its transpose.
    	The image is transposed, the seam is removed as a vertical seam, and the
    	result is transposed back.
      \param img
      cv::Mat&
    	The matrix to shrink (any type and channel count); replaced by the
    	result
      \param seam
      const std::vector<int>&
    	The seam: seam[j] is the row removed in column j
      \exception std::invalid_argument
    	thrown if img is empty or too small to remove a seam, the seam has the
    	wrong length, or an index is out of range; img is left unchanged
    */
    /************************************************************************/
    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam)
    {
        CheckRemoval(img, seam, img.cols, img.rows, "SeamCarving::RemoveHorizontalSeam");
        cv::Mat t;
        cv::transpose(img, t);
        RemoveVerticalUnchecked(t, seam);
        cv::transpose(t, img);
    }

    // ========================================================================
    // (c) ... and repeat
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	measures how much time has passed since a given time point
      \param t0
      std::chrono::steady_clock::time_point
    	The starting time point
      \return double
      double
      returns the elapsed time in seconds
    */
    /************************************************************************/
    static double SecondsSince(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    /************************************************************************/
    /*!
      \brief
    	removes k vertical seams from an image, adding each seam's cost and time
    	to stats and recording each seam in original-image coordinates when
    	asked.
    	Removes k vertical seams from img, one at a time:
    	  energy of the CURRENT image -> findSeam -> remove -> repeat.
    	coords (optional) holds the original (x, y) of every pixel and is
    	carved in step with img, so removed seams can be reported in
    	original-image coordinates.
      \param img
      cv::Mat&
    	The image to shrink; it loses k columns
      \param coords
      cv::Mat*
    	The CV_32SC2 map of original (x, y) positions carved alongside img, or
    	nullptr when seams are not recorded
      \param k
      int
    	The number of seams to remove
      \param energyType
      EnergyType
    	The energy recomputed before every seam
      \param findSeam
      const VerticalSeamFinder&
    	The method that finds each vertical seam
      \param stats
      CarveStats*
    	Where the seam time and costs are added; may be nullptr
      \param record
      std::vector<std::vector<cv::Point>>*
    	Where each removed seam is stored in original coordinates; may be
    	nullptr
      \param who
      const char*
    	The name of the public caller, used in error messages
      \exception std::invalid_argument
    	thrown if findSeam returns an invalid seam
    */
    /************************************************************************/
    static void RemoveSeams(cv::Mat& img, cv::Mat* coords, int k, EnergyType energyType,
                            const VerticalSeamFinder& findSeam, CarveStats* stats,
                            std::vector<std::vector<cv::Point>>* record, const char* who)
    {
        for (int step = 0; step < k; ++step)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const cv::Mat energy = ComputeEnergy(img, energyType);   // recomputed every time
            const std::vector<int> seam = findSeam(energy);
            const double seconds = SecondsSince(t0);

            // Validates the seam before it is used to read energy / coords.
            CheckRemoval(img, seam, img.rows, img.cols, who);

            if (stats)
            {
                int64_t cost = 0;
                for (int i = 0; i < img.rows; ++i) cost += energy.at<int>(i, seam[i]);
                stats->seamSeconds += seconds;
                stats->totalCost += cost;
                stats->seamCosts.push_back(cost);
            }
            if (record && coords)
            {
                std::vector<cv::Point> pts(img.rows);
                for (int i = 0; i < img.rows; ++i)
                {
                    const cv::Vec2i p = coords->at<cv::Vec2i>(i, seam[i]);
                    pts[i] = cv::Point(p[0], p[1]);
                }
                record->push_back(std::move(pts));
            }

            RemoveVerticalUnchecked(img, seam);
            if (coords) RemoveVerticalUnchecked(*coords, seam);
        }
    }

    /************************************************************************/
    /*!
      \brief
    	shrinks an image to newWidth x newHeight by repeatedly removing a seam
    	found by findSeam. Vertical seams (width) are removed first; the image
    	is then transposed once so the same loop removes the horizontal seams
    	(height), and transposed back. The energy map is recomputed after every
    	removal. The map of original pixel coordinates is only built when seams
    	are recorded.
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
      \param findSeam
      const VerticalSeamFinder&
    	The seam finder (DP, Greedy or Graph Cut)
      \param stats
      CarveStats*
    	Where times, seam costs and recorded seams are added; may be nullptr
      \param recordSeams
      bool
    	Whether to record every removed seam (only when stats is not null)
      \param who
      const char*
    	The name of the public caller, used in error messages
      \return cv::Mat
      OpenCV matrix
      returns the resized image, the same type as img
      \exception std::invalid_argument
    	thrown if img is empty, the target size is not between 1 and the
    	original size, no seam finder is given, or the seam finder returns an
    	invalid seam
    */
    /************************************************************************/
    cv::Mat CarveWith(const cv::Mat& img, int newWidth, int newHeight, EnergyType energy,
                      const VerticalSeamFinder& findSeam, CarveStats* stats, bool recordSeams,
                      const char* who)
    {
        if (img.empty())
            throw std::invalid_argument(std::string(who) + ": image is empty");
        if (newWidth < 1 || newWidth > img.cols || newHeight < 1 || newHeight > img.rows)
            throw std::invalid_argument(std::string(who) + ": target size must be between 1 and the original size");
        if (!findSeam)
            throw std::invalid_argument(std::string(who) + ": no seam finder given");

        const auto start = std::chrono::steady_clock::now();
        cv::Mat out = img.clone();
        const bool rec = recordSeams && stats;

        // Original (x, y) of every pixel; only needed to record seams.
        cv::Mat coords;
        if (rec)
        {
            coords.create(img.rows, img.cols, CV_32SC2);
            for (int y = 0; y < img.rows; ++y)
                for (int x = 0; x < img.cols; ++x)
                    coords.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
        }

        // 1. Vertical seams reduce the width.
        RemoveSeams(out, rec ? &coords : nullptr, img.cols - newWidth, energy, findSeam, stats,
                    rec ? &stats->verticalSeams : nullptr, who);

        // 2. Horizontal seams reduce the height: transpose once, remove
        //    vertical seams of the transposed image, transpose back.
        if (newHeight < img.rows)
        {
            cv::Mat outT, coordsT;
            cv::transpose(out, outT);
            if (rec) cv::transpose(coords, coordsT);
            RemoveSeams(outT, rec ? &coordsT : nullptr, img.rows - newHeight, energy, findSeam, stats,
                        rec ? &stats->horizontalSeams : nullptr, who);
            cv::transpose(outT, out);
        }

        if (stats) stats->totalSeconds += SecondsSince(start);
        return out;
    }

    // ========================================================================
    // Seam overlay
    // ========================================================================

    /************************************************************************/
    /*!
      \brief
    	draws every recorded seam on a BGR copy of the original image: vertical
    	seams in red, horizontal seams in green.
      \param original
      const cv::Mat&
    	The original 8-bit image the seams were recorded on
      \param stats
      const CarveStats&
    	The statistics of a carve made with recordSeams = true
      \return cv::Mat
      OpenCV matrix
      returns a CV_8UC3 copy of original with the seams drawn on it
      \exception std::invalid_argument
    	thrown if original is empty or not 8-bit, or a seam point lies outside
    	it (stats from a different image)
    */
    /************************************************************************/
    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats)
    {
        if (original.empty() || original.depth() != CV_8U)
            throw std::invalid_argument("SeamCarving::DrawSeams: original must be a non-empty 8-bit image");

        cv::Mat vis;
        if (original.channels() == 1)      cv::cvtColor(original, vis, cv::COLOR_GRAY2BGR);
        else if (original.channels() == 4) cv::cvtColor(original, vis, cv::COLOR_BGRA2BGR);
        else                               vis = original.clone();

        const cv::Rect bounds(0, 0, vis.cols, vis.rows);
        auto draw = [&](const std::vector<std::vector<cv::Point>>& seams, const cv::Vec3b& colour)
        {
            for (const auto& seam : seams)
                for (const cv::Point& p : seam)
                {
                    if (!bounds.contains(p))
                        throw std::invalid_argument("SeamCarving::DrawSeams: seam point outside the image "
                                                    "(stats from a different image?)");
                    vis.at<cv::Vec3b>(p) = colour;
                }
        };
        draw(stats.verticalSeams, cv::Vec3b(0, 0, 255));     // red   (BGR)
        draw(stats.horizontalSeams, cv::Vec3b(0, 255, 0));   // green
        return vis;
    }
}
