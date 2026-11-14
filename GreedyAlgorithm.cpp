#include "GreedyAlgorithm.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace Greedy
{
    // ========================================================================
    // Greedy seam finding
    // ========================================================================
    static void CheckEnergy(const cv::Mat& energy, const char* who)
    {
        if (energy.empty())
            throw std::invalid_argument(std::string(who) + ": energy map is empty");
        if (energy.type() != CV_32SC1)
            throw std::invalid_argument(std::string(who) + ": energy map must be CV_32SC1");
    }

    // Index of the smallest value in [0, n); ties -> lowest index.
    template <typename Get>
    static int ArgMin(int n, Get get)
    {
        int best = 0;
        for (int k = 1; k < n; ++k)
            if (get(k) < get(best)) best = k;
        return best;
    }

    // Next index from position p among {p, p-1, p+1} (clipped to [0, n)).
    // Ties: straight first, then the smaller index, then the larger one -
    // a candidate only wins if it is STRICTLY cheaper than the current best.
    template <typename Get>
    static int Step(int p, int n, Get get)
    {
        int best = p;
        if (p - 1 >= 0 && get(p - 1) < get(best)) best = p - 1;
        if (p + 1 < n && get(p + 1) < get(best)) best = p + 1;
        return best;
    }

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
    // Seam removal
    // ========================================================================
    void RemoveVerticalSeam(cv::Mat& img, const std::vector<int>& seam)
    {
        if (img.cols < 2)
            throw std::invalid_argument("Greedy::RemoveVerticalSeam: image must be at least 2 columns wide");
        if (static_cast<int>(seam.size()) != img.rows)
            throw std::invalid_argument("Greedy::RemoveVerticalSeam: seam length must equal the number of rows");

        cv::Mat out(img.rows, img.cols - 1, img.type());
        const size_t es = img.elemSize();
        for (int i = 0; i < img.rows; ++i)
        {
            const int c = seam[i];
            if (c < 0 || c >= img.cols)
                throw std::invalid_argument("Greedy::RemoveVerticalSeam: seam column out of range");
            const uchar* src = img.ptr(i);
            uchar* dst = out.ptr(i);
            std::memcpy(dst, src, c * es);                                            // left of seam
            std::memcpy(dst + c * es, src + (c + 1) * es, (img.cols - 1 - c) * es);   // right of seam
        }
        img = out;
    }

    void RemoveHorizontalSeam(cv::Mat& img, const std::vector<int>& seam)
    {
        if (img.rows < 2)
            throw std::invalid_argument("Greedy::RemoveHorizontalSeam: image must be at least 2 rows tall");
        if (static_cast<int>(seam.size()) != img.cols)
            throw std::invalid_argument("Greedy::RemoveHorizontalSeam: seam length must equal the number of columns");
        cv::Mat t;
        cv::transpose(img, t);
        RemoveVerticalSeam(t, seam);   // seam[col] = row becomes seam[row'] = col'
        cv::transpose(t, img);
    }

    // ========================================================================
    // Carving loop
    // ========================================================================
    // Removes k vertical seams from (img, coords). coords holds the original
    // (x, y) of every pixel so removed seams can be drawn on the input image.
    static void RemoveSeams(cv::Mat& img, cv::Mat& coords, int k, EnergyType energy,
                            CarveStats* stats, std::vector<std::vector<cv::Point>>* record)
    {
        for (int step = 0; step < k; ++step)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const SeamResult r = FindVerticalSeam(DP::ComputeEnergy(img, energy));
            const auto t1 = std::chrono::steady_clock::now();
            if (stats)
            {
                stats->seamSeconds += std::chrono::duration<double>(t1 - t0).count();
                stats->totalCost += r.cost;
                stats->seamCosts.push_back(r.cost);
            }
            if (record)
            {
                std::vector<cv::Point> pts(img.rows);
                for (int i = 0; i < img.rows; ++i)
                {
                    const cv::Vec2i p = coords.at<cv::Vec2i>(i, r.seam[i]);
                    pts[i] = cv::Point(p[0], p[1]);
                }
                record->push_back(std::move(pts));
            }
            RemoveVerticalSeam(img, r.seam);
            RemoveVerticalSeam(coords, r.seam);
        }
    }

    cv::Mat Carve(const cv::Mat& img, int newWidth, int newHeight, EnergyType energy,
                  CarveStats* stats, bool recordSeams)
    {
        if (img.empty() || newWidth < 1 || newHeight < 1 || newWidth > img.cols || newHeight > img.rows)
            throw std::invalid_argument("Greedy::Carve: target size must be between 1 and the original size");

        const auto start = std::chrono::steady_clock::now();
        cv::Mat out = img.clone();
        cv::Mat coords(img.rows, img.cols, CV_32SC2);
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x)
                coords.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);

        const bool rec = recordSeams && stats;
        // Vertical seams reduce the width.
        RemoveSeams(out, coords, img.cols - newWidth, energy, stats,
                    rec ? &stats->verticalSeams : nullptr);
        // Horizontal seams: transpose, remove vertical seams, transpose back.
        // On the transposed image a greedy vertical seam is exactly a greedy
        // horizontal seam of the original (same tie-breaking: up before down).
        if (newHeight < img.rows)
        {
            cv::Mat outT, coordsT;
            cv::transpose(out, outT);
            cv::transpose(coords, coordsT);
            RemoveSeams(outT, coordsT, img.rows - newHeight, energy, stats,
                        rec ? &stats->horizontalSeams : nullptr);
            cv::transpose(outT, out);
        }
        if (stats)
            stats->totalSeconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return out;
    }

    cv::Mat DrawSeams(const cv::Mat& original, const CarveStats& stats)
    {
        cv::Mat vis;
        if (original.channels() == 1)      cv::cvtColor(original, vis, cv::COLOR_GRAY2BGR);
        else if (original.channels() == 4) cv::cvtColor(original, vis, cv::COLOR_BGRA2BGR);
        else                               vis = original.clone();
        for (const auto& seam : stats.verticalSeams)
            for (const cv::Point& p : seam) vis.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
        for (const auto& seam : stats.horizontalSeams)
            for (const cv::Point& p : seam) vis.at<cv::Vec3b>(p) = cv::Vec3b(0, 255, 0);
        return vis;
    }
}
