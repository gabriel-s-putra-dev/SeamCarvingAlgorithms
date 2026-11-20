#include "GreedyAlgorithm.h"

#include <stdexcept>
#include <string>

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
        // Same rule as DP and Graph Cut: energies are non-negative.
        double minVal;
        cv::minMaxLoc(energy, &minVal);
        if (minVal < 0)
            throw std::invalid_argument(std::string(who) + ": energy values must be non-negative");
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
    // Carving: the shared remove-and-repeat loop with the greedy seam finder
    // ========================================================================
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
