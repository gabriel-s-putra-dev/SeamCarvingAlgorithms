/*!************************************************************************
\file SeamTests.cpp
\author Ho Jun
\par DP email: 2401700@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Tests for part (b): DP minimum seam (vertical and horizontal).
No test framework needed (see tests/Check.h): build it, run it, and it
prints any failures and "N checks passed, M failed".

CMake target: seam_tests (see CMakeLists.txt). Without CMake:
  g++ -std=c++14 -I. tests/SeamTests.cpp SeamCarving.cpp DynamicProgramming.cpp \
      $(pkg-config --cflags --libs opencv4) -o seam_tests
Restored from commit a5cf885 (it had been overwritten by a copy of
DynamicProgramming.cpp) and updated for the 64-bit cumulative table.
- MakeEnergy(rows, cols, v)
builds a CV_32SC1 matrix from a flat list of values
- IsValidSeam(seam, length, limit)
checks the shape of a vertical seam
- BestFrom(e, i, j)
brute-force cheapest path from (i, j) to the bottom row
- BruteForceVertical(e)
brute-force cheapest vertical seam
- Transposed(m)
returns the transpose of a matrix
- TestHandWorkedExample() ... TestNoOverflow()
the nine test groups listed in main
- main()
runs every test and prints the summary
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "Check.h"
#include "DynamicProgramming.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------- helpers --

/************************************************************************/
/*!
  \brief
	Build a CV_32SC1 matrix from a flat row-major list of values.
  \param rows
  int
	The number of rows
  \param cols
  int
	The number of columns
  \param v
  const std::vector<int>&
	The rows * cols values, row by row
  \return cv::Mat
  OpenCV matrix
  returns the CV_32SC1 matrix
*/
/************************************************************************/
static cv::Mat MakeEnergy(int rows, int cols, const std::vector<int>& v)
{
    cv::Mat m(rows, cols, CV_32SC1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m.at<int>(i, j) = v[i * cols + j];
    return m;
}

/************************************************************************/
/*!
  \brief
	A valid vertical seam: one column per row, in range, steps of at most 1.
  \param seam
  const std::vector<int>&
	The seam to check
  \param length
  int
	The length the seam must have
  \param limit
  int
	Every entry must be in [0, limit)
  \return bool
  boolean
  returns true if the seam is valid, false otherwise
*/
/************************************************************************/
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

/************************************************************************/
/*!
  \brief
	Brute force: try EVERY vertical path (3^rows paths, fine for tiny inputs)
	and return the cheapest total energy. This is the "ground truth".
	Recursive helper: the cheapest path starting at (i, j) and going down to
	the last row.
  \param e
  const cv::Mat&
	The CV_32SC1 energy map
  \param i
  int
	The row of the starting pixel
  \param j
  int
	The column of the starting pixel
  \return long long
  64-bit integer
  returns the cheapest total energy from (i, j) to the bottom row
*/
/************************************************************************/
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

/************************************************************************/
/*!
  \brief
	finds the cheapest vertical seam by brute force, starting BestFrom at
	every pixel of row 0
  \param e
  const cv::Mat&
	The CV_32SC1 energy map
  \return long long
  64-bit integer
  returns the minimum seam energy
*/
/************************************************************************/
static long long BruteForceVertical(const cv::Mat& e)
{
    long long best = std::numeric_limits<long long>::max();
    for (int j = 0; j < e.cols; ++j)
        best = std::min(best, BestFrom(e, 0, j));
    return best;
}

/************************************************************************/
/*!
  \brief
	returns the transpose of a matrix
  \param m
  const cv::Mat&
	The matrix to transpose
  \return cv::Mat
  OpenCV matrix
  returns the transposed copy
*/
/************************************************************************/
static cv::Mat Transposed(const cv::Mat& m)
{
    cv::Mat t;
    cv::transpose(m, t);
    return t;
}

// ------------------------------------------------------------------ tests --

/************************************************************************/
/*!
  \brief
	1. A 3x3 example worked out by hand.
*/
/************************************************************************/
static void TestHandWorkedExample()
{
    const cv::Mat e = MakeEnergy(3, 3, { 1, 2, 3,
                                         4, 1, 6,
                                         7, 8, 1 });

    // M[i][j] = e[i][j] + min of the (up to) three cells above.
    // The table is CV_64F (64-bit so sums cannot overflow).
    const cv::Mat M = DP::CumulativeEnergy(e);
    CHECK(M.type() == CV_64FC1 && M.size() == e.size());
    cv::Mat expected;
    MakeEnergy(3, 3, { 1,  2, 3,
                       5,  2, 8,
                       9, 10, 3 }).convertTo(expected, CV_64F);
    CHECK(cv::countNonZero(M != expected) == 0);

    const std::vector<int> seam = DP::FindVerticalSeam(e);
    CHECK((seam == std::vector<int>{ 0, 1, 2 }));
    CHECK(DP::SeamEnergy(e, seam, true) == 3);
}

/************************************************************************/
/*!
  \brief
	2. Horizontal seam of a matrix == vertical seam of its transpose.
*/
/************************************************************************/
static void TestHorizontalMatchesTranspose()
{
    const cv::Mat e = MakeEnergy(3, 3, { 1, 2, 3,
                                         4, 1, 6,
                                         7, 8, 1 });
    const cv::Mat t = Transposed(e);

    const std::vector<int> hSeam = DP::FindHorizontalSeam(e);   // row per column
    CHECK((hSeam == std::vector<int>{ 0, 1, 2 }));
    CHECK(hSeam == DP::FindVerticalSeam(t));
    CHECK(DP::SeamEnergy(e, hSeam, false) == 3);
}

/************************************************************************/
/*!
  \brief
	3. The seam must avoid a wall of high energy, and ties go to the left.
*/
/************************************************************************/
static void TestAvoidsHighEnergyColumn()
{
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

/************************************************************************/
/*!
  \brief
	4. Seam shape is always valid on random input.
*/
/************************************************************************/
static void TestSeamValidityRandom()
{
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

/************************************************************************/
/*!
  \brief
	5. The most important test: DP must match brute force on small matrices.
*/
/************************************************************************/
static void TestOptimalityVsBruteForce()
{
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

/************************************************************************/
/*!
  \brief
	6. Tiny and degenerate shapes.
*/
/************************************************************************/
static void TestEdgeCases()
{

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

/************************************************************************/
/*!
  \brief
	7. Bad input must throw instead of crashing.
*/
/************************************************************************/
static void TestInvalidInput()
{

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

    // Further SeamEnergy validation and the shared input rules.
    const cv::Mat z = cv::Mat::zeros(3, 3, CV_32SC1);
    CHECK(Throws([&] { DP::SeamEnergy(z, { 0, 2, 1 }, true); }));    // not connected
    CHECK(Throws([&] { DP::SeamEnergy(z, { 0, 1, 3 }, true); }));    // index out of range
    CHECK(Throws([&] { DP::SeamEnergy(z, { -1, 0, 0 }, false); }));  // negative index
    CHECK(Throws([&] { DP::SeamEnergy(cv::Mat(), { 0 }, true); }));  // empty map
    CHECK(Throws([&] { DP::SeamEnergy(cv::Mat::zeros(3, 3, CV_8UC1), { 0, 0, 0 }, true); }));
    CHECK(Throws([&] { DP::CumulativeEnergy(cv::Mat()); }));
    CHECK(Throws([&] { DP::CumulativeEnergy(cv::Mat::zeros(3, 3, CV_64FC1)); }));

    // Negative energies are rejected (same rule as Greedy and Graph Cut).
    const cv::Mat neg = MakeEnergy(2, 2, { 1, -1,
                                           2,  3 });
    CHECK(Throws([&] { DP::CumulativeEnergy(neg); }));
    CHECK(Throws([&] { DP::FindVerticalSeam(neg); }));
    CHECK(Throws([&] { DP::FindHorizontalSeam(neg); }));
    CHECK(Throws([&] { DP::SeamEnergy(neg, { 0, 0 }, true); }));
}

/************************************************************************/
/*!
  \brief
	9. The 64-bit table: sums beyond INT_MAX stay exact (no overflow).
*/
/************************************************************************/
static void TestNoOverflow()
{
    const int rows = 10, big = INT_MAX / 2;
    const cv::Mat e(rows, 2, CV_32SC1, cv::Scalar(big));
    const long long expected = static_cast<long long>(big) * rows;   // > INT_MAX

    const cv::Mat M = DP::CumulativeEnergy(e);
    CHECK(static_cast<long long>(M.at<double>(rows - 1, 0)) == expected);
    CHECK(static_cast<long long>(M.at<double>(rows - 1, 1)) == expected);

    const std::vector<int> seam = DP::FindVerticalSeam(e);
    CHECK((seam == std::vector<int>(rows, 0)));                       // tie -> leftmost
    CHECK(DP::SeamEnergy(e, seam, true) == expected);
}

/************************************************************************/
/*!
  \brief
	8. Integration with part (a): real ComputeEnergy output on a tiny image.
*/
/************************************************************************/
static void TestWithComputeEnergy()
{

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

/************************************************************************/
/*!
  \brief
	runs every test group and prints the summary
  \return int
  integer
  returns 0 if every check passed, 1 otherwise
*/
/************************************************************************/
int main()
{
    return RunTests({
        { "hand-worked 3x3 example", TestHandWorkedExample },
        { "horizontal seam == vertical seam of transpose", TestHorizontalMatchesTranspose },
        { "avoids a high-energy column / tie-breaking", TestAvoidsHighEnergyColumn },
        { "seam validity on random matrices", TestSeamValidityRandom },
        { "DP vs brute force (optimality)", TestOptimalityVsBruteForce },
        { "edge cases (1x1, 1xN, Nx1)", TestEdgeCases },
        { "invalid input throws", TestInvalidInput },
        { "integration with ComputeEnergy", TestWithComputeEnergy },
        { "64-bit table: no overflow", TestNoOverflow },
    });
}
