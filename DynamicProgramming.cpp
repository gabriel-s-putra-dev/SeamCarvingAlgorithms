#include "DynamicProgramming.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace DP
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
            throw std::invalid_argument("DP::ComputeEnergy: image is empty");
        const cv::Mat gray = ToGray(img);
        return type == EnergyType::Backward ? BackwardEnergy(gray) : ForwardEnergy(gray);
    }

    // ========================================================================
    // (b) Minimum seam by dynamic programming
    // ========================================================================
    static void CheckEnergy(const cv::Mat& energy, const char* who)
    {
        if (energy.empty())
            throw std::invalid_argument(std::string(who) + ": energy map is empty");
        if (energy.type() != CV_32SC1)
            throw std::invalid_argument(std::string(who) + ": energy map must be CV_32SC1");
    }

    // M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1]).
    // Each cell stores the cheapest total energy of any vertical path that
    // starts in row 0 and ends at (i, j). Overall cost O(rows * cols).
    cv::Mat CumulativeEnergy(const cv::Mat& energy)
    {
        CheckEnergy(energy, "DP::CumulativeEnergy");

        const int rows = energy.rows;
        const int cols = energy.cols;

        cv::Mat M = energy.clone();          // row 0 is just the energy itself

        for (int i = 1; i < rows; ++i)
        {
            const int* prev = M.ptr<int>(i - 1);
            int* cur = M.ptr<int>(i);
            for (int j = 0; j < cols; ++j)
            {
                int best = prev[j];                                   // straight up
                if (j > 0)        best = std::min(best, prev[j - 1]); // up-left
                if (j < cols - 1) best = std::min(best, prev[j + 1]); // up-right
                cur[j] += best;                                       // cur[j] held e[i][j]
            }
        }
        return M;
    }

    // Backtracking: start at the cheapest cell of the last row, then walk up,
    // each time moving to whichever of the 3 cells above has the smallest M.
    std::vector<int> FindVerticalSeam(const cv::Mat& energy)
    {
        const cv::Mat M = CumulativeEnergy(energy);
        const int rows = M.rows;
        const int cols = M.cols;

        std::vector<int> seam(rows);

        // End of the seam: minimum of the bottom row (leftmost on ties).
        const int* last = M.ptr<int>(rows - 1);
        int j = 0;
        for (int c = 1; c < cols; ++c)
            if (last[c] < last[j]) j = c;
        seam[rows - 1] = j;

        // Walk upwards.
        for (int i = rows - 2; i >= 0; --i)
        {
            const int* row = M.ptr<int>(i);
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

    // A horizontal seam in the image is a vertical seam in the transposed
    // image, so: transpose, find vertical seam, and read the result as
    // "row index for every column".
    std::vector<int> FindHorizontalSeam(const cv::Mat& energy)
    {
        CheckEnergy(energy, "DP::FindHorizontalSeam");
        cv::Mat energyT;
        cv::transpose(energy, energyT);
        return FindVerticalSeam(energyT);
    }

    long long SeamEnergy(const cv::Mat& energy, const std::vector<int>& seam, bool vertical)
    {
        CheckEnergy(energy, "DP::SeamEnergy");
        long long total = 0;
        if (vertical)
        {
            if (static_cast<int>(seam.size()) != energy.rows)
                throw std::invalid_argument("DP::SeamEnergy: seam size must equal rows");
            for (int i = 0; i < energy.rows; ++i)
                total += energy.at<int>(i, seam[i]);
        }
        else
        {
            if (static_cast<int>(seam.size()) != energy.cols)
                throw std::invalid_argument("DP::SeamEnergy: seam size must equal cols");
            for (int j = 0; j < energy.cols; ++j)
                total += energy.at<int>(seam[j], j);
        }
        return total;
    }
}
