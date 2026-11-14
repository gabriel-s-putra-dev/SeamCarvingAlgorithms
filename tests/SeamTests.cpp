// ============================================================================
// Tests for part (b): DP minimum seam (vertical and horizontal).
// No test framework needed: build it, run it, and it prints PASS / FAIL.
//
// Build example (adapt to your CMake setup):
//   g++ -std=c++17 tests/SeamTests.cpp DynamicProgramming.cpp \
//       $(pkg-config --cflags --libs opencv4) -o seam_tests
// ============================================================================
#include "../DynamicProgramming.h"   // adjust the path if your layout differs

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::cout << "  FAIL line " << __LINE__ << ": " #cond << "\n";   \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------- helpers --
// Build a CV_32SC1 matrix from a flat row-major list of values.
static cv::Mat MakeEnergy(int rows, int cols, const std::vector<int>& v)
{
    cv::Mat m(rows, cols, CV_32SC1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m.at<int>(i, j) = v[i * cols + j];
    return m;
}

// A valid vertical seam: one column per row, in range, steps of at most 1.
static bool IsValidSeam(const std::vector<int>& seam, int length, int limit)
{
    if (static_cast<int>(seam.size()) != length) return false;
    for (int k = 0; k < length; ++k)
    {
        if (seam[k] < 0 || seam[k] >= limit) return false;
        if (k > 0 && std::abs(seam[k] - seam[k - 1]) > 1) return false;
    }
    return true;
}

// Brute force: try EVERY vertical path (3^rows paths, fine for tiny inputs)
// and return the cheapest total energy. This is the "ground truth".
static long long BestFrom(const cv::Mat& e, int i, int j)
{
    long long here = e.at<int>(i, j);
    if (i == e.rows - 1) return here;
    long long best = std::numeric_limits<long long>::max();
    for (int dj = -1; dj <= 1; ++dj)
    {
        const int nj = j + dj;
        if (nj < 0 || nj >= e.cols) continue;
        best = std::min(best, BestFrom(e, i + 1, nj));
    }
    return here + best;
}

static long long BruteForceVertical(const cv::Mat& e)
{
    long long best = std::numeric_limits<long long>::max();
    for (int j = 0; j < e.cols; ++j)
        best = std::min(best, BestFrom(e, 0, j));
    return best;
}

static cv::Mat Transposed(const cv::Mat& m)
{
    cv::Mat t;
    cv::transpose(m, t);
    return t;
}

// ------------------------------------------------------------------ tests --

// 1. A 3x3 example worked out by hand.
static void TestHandWorkedExample()
{
    std::cout << "Hand-worked 3x3 example\n";
    const cv::Mat e = MakeEnergy(3, 3, { 1, 2, 3,
                                         4, 1, 6,
                                         7, 8, 1 });

    // M[i][j] = e[i][j] + min of the (up to) three cells above.
    const cv::Mat M = DP::CumulativeEnergy(e);
    const cv::Mat expected = MakeEnergy(3, 3, { 1,  2, 3,
                                                5,  2, 8,
                                                9, 10, 3 });
    CHECK(cv::countNonZero(M != expected) == 0);

    const std::vector<int> seam = DP::FindVerticalSeam(e);
    CHECK((seam == std::vector<int>{ 0, 1, 2 }));
    CHECK(DP::SeamEnergy(e, seam, true) == 3);
}

// 2. Horizontal seam of a matrix == vertical seam of its transpose.
static void TestHorizontalMatchesTranspose()
{
    std::cout << "Horizontal seam matches vertical seam of transpose\n";
    const cv::Mat e = MakeEnergy(3, 3, { 1, 2, 3,
                                         4, 1, 6,
                                         7, 8, 1 });
    const cv::Mat t = Transposed(e);

    const std::vector<int> hSeam = DP::FindHorizontalSeam(e);   // row per column
    CHECK((hSeam == std::vector<int>{ 0, 1, 2 }));
    CHECK(hSeam == DP::FindVerticalSeam(t));
    CHECK(DP::SeamEnergy(e, hSeam, false) == 3);
}

// 3. The seam must avoid a wall of high energy, and ties go to the left.
static void TestAvoidsHighEnergyColumn()
{
    std::cout << "Avoids a high-energy column / tie-breaking\n";
    cv::Mat e = cv::Mat::zeros(5, 6, CV_32SC1);
    for (int i = 0; i < 5; ++i) e.at<int>(i, 3) = 100;

    const std::vector<int> v = DP::FindVerticalSeam(e);
    CHECK(DP::SeamEnergy(e, v, true) == 0);
    CHECK((v == std::vector<int>(5, 0)));        // all zeros: leftmost wins ties

    // Same idea rotated: a high-energy row, horizontal seam must avoid it.
    const cv::Mat t = Transposed(e);
    const std::vector<int> h = DP::FindHorizontalSeam(t);
    CHECK(DP::SeamEnergy(t, h, false) == 0);
}

// 4. Seam shape is always valid on random input.
static void TestSeamValidityRandom()
{
    std::cout << "Seam validity on random matrices\n";
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dim(1, 12), val(0, 255);

    for (int trial = 0; trial < 200; ++trial)
    {
        const int r = dim(rng), c = dim(rng);
        cv::Mat e(r, c, CV_32SC1);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                e.at<int>(i, j) = val(rng);

        CHECK(IsValidSeam(DP::FindVerticalSeam(e), r, c));
        CHECK(IsValidSeam(DP::FindHorizontalSeam(e), c, r));
    }
}

// 5. The most important test: DP must match brute force on small matrices.
static void TestOptimalityVsBruteForce()
{
    std::cout << "DP vs brute force (optimality)\n";
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> rowsD(1, 7), colsD(1, 7), val(0, 20);

    for (int trial = 0; trial < 300; ++trial)
    {
        const int r = rowsD(rng), c = colsD(rng);
        cv::Mat e(r, c, CV_32SC1);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                e.at<int>(i, j) = val(rng);   // small range => many ties

        const std::vector<int> v = DP::FindVerticalSeam(e);
        CHECK(DP::SeamEnergy(e, v, true) == BruteForceVertical(e));

        // Horizontal: brute force on the transpose.
        const std::vector<int> h = DP::FindHorizontalSeam(e);
        CHECK(DP::SeamEnergy(e, h, false) == BruteForceVertical(Transposed(e)));
    }
}

// 6. Tiny and degenerate shapes.
static void TestEdgeCases()
{
    std::cout << "Edge cases (1x1, 1xN, Nx1)\n";

    const cv::Mat one = MakeEnergy(1, 1, { 7 });
    CHECK((DP::FindVerticalSeam(one) == std::vector<int>{ 0 }));
    CHECK((DP::FindHorizontalSeam(one) == std::vector<int>{ 0 }));

    // One row: vertical seam is just the cheapest pixel.
    const cv::Mat rowOnly = MakeEnergy(1, 5, { 9, 4, 7, 4, 8 });
    CHECK((DP::FindVerticalSeam(rowOnly) == std::vector<int>{ 1 }));   // leftmost tie
    CHECK((DP::FindHorizontalSeam(rowOnly) == std::vector<int>(5, 0)));

    // One column: vertical seam has no choice.
    const cv::Mat colOnly = MakeEnergy(5, 1, { 9, 4, 7, 4, 8 });
    CHECK((DP::FindVerticalSeam(colOnly) == std::vector<int>(5, 0)));
    CHECK((DP::FindHorizontalSeam(colOnly) == std::vector<int>{ 1 }));
}

// 7. Bad input must throw instead of crashing.
static void TestInvalidInput()
{
    std::cout << "Invalid input throws\n";

    bool threw = false;
    try { DP::FindVerticalSeam(cv::Mat()); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { DP::FindHorizontalSeam(cv::Mat::zeros(3, 3, CV_8UC1)); }   // wrong type
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { DP::SeamEnergy(cv::Mat::zeros(3, 3, CV_32SC1), { 0, 1 }, true); }  // wrong size
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// 8. Integration with part (a): real ComputeEnergy output on a tiny image.
static void TestWithComputeEnergy()
{
    std::cout << "Integration with DP::ComputeEnergy\n";

    // 8x8 gray image: a bright stripe in columns 3-4, black elsewhere.
    cv::Mat img = cv::Mat::zeros(8, 8, CV_8UC1);
    img.colRange(3, 5).setTo(255);

    for (DP::EnergyType type : { DP::EnergyType::Backward, DP::EnergyType::Forward })
    {
        const cv::Mat e = DP::ComputeEnergy(img, type);
        const std::vector<int> v = DP::FindVerticalSeam(e);
        CHECK(IsValidSeam(v, 8, 8));
        CHECK(DP::SeamEnergy(e, v, true) == 0);          // flat background exists
        for (int col : v)                                // never cuts through the stripe
            CHECK(col < 2 || col > 5);
    }
}

int main()
{
    TestHandWorkedExample();
    TestHorizontalMatchesTranspose();
    TestAvoidsHighEnergyColumn();
    TestSeamValidityRandom();
    TestOptimalityVsBruteForce();
    TestEdgeCases();
    TestInvalidInput();
    TestWithComputeEnergy();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
