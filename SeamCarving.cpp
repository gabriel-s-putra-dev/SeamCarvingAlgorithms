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
            throw std::invalid_argument("SeamCarving::ComputeEnergy: image is empty");
        const cv::Mat gray = ToGray(img);
        return type == EnergyType::Backward ? BackwardEnergy(gray) : ForwardEnergy(gray);
    }

    // ========================================================================
    // (c) Remove the seam
    // ========================================================================

    // length = pixels in the seam (rows for vertical, cols for horizontal),
    // limit  = size of the dimension being shrunk (cols / rows).
    // Everything is validated BEFORE img is touched.
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

    // Copies each row into a (cols - 1)-wide image, skipping the seam pixel.
    // Uses raw bytes (elemSize), so any type and channel count works.
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

    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam)
    {
        CheckRemoval(img, seam, img.rows, img.cols, "SeamCarving::RemoveVerticalSeam");
        RemoveVerticalUnchecked(img, seam);
    }

    // A horizontal seam of img is a vertical seam of its transpose.
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
    static double SecondsSince(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // Removes k vertical seams from img, one at a time:
    //   energy of the CURRENT image -> findSeam -> remove -> repeat.
    // coords (optional) holds the original (x, y) of every pixel and is
    // carved in step with img, so removed seams can be reported in
    // original-image coordinates.
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
