/*!************************************************************************
\file CarveTests.cpp
\author Gabriel Sebastian Putra
\par DP email: 2400735@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
This source file contains the tests for step (c), removing the seam and
repeating.

Correctness tests for step (c): remove the seam and repeat
(RemoveVerticalSeam, RemoveHorizontalSeam, Carve, DrawSeams), through the DP
module. The removal, loop and overlay live in SeamCarving.cpp and are shared
with Greedy and Graph Cut.
Build and run from the repo root:
  cmake --build --preset vs2022-release --target carve_tests
  out\build\vs2022\Release\carve_tests.exe     (or: ctest --preset vs2022-release)
or with g++:
  g++ -std=c++14 -O2 -I. tests/CarveTests.cpp SeamCarving.cpp DynamicProgramming.cpp GraphCutting.cpp -o carve_tests `pkg-config --cflags --libs opencv4`

GraphCutting.cpp is linked to check that DP::Carve and GraphCut::Carve give
pixel-identical results: both return the leftmost optimal seam.
- SameMat(a, b)
checks whether two matrices are identical
- MakeU8(rows, cols, v)
builds a small 8-bit image from a list of values
- RandomBGR(rows, cols, rng)
builds a random colour image
- LoadImage(path)
reads a sample image, with a hint if it is missing
- TestRemoveVertical() ... TestForwardCarve()
the nine test groups listed in main
- Timing()
prints how long a full carve of the tower image takes
- main()
runs every test and prints the summary
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "Check.h"
#include "DynamicProgramming.h"
#include "GraphCutting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/************************************************************************/
/*!
  \brief
	checks whether two matrices have the same size, type and values (any
	channel count)
  \param a
  const cv::Mat&
	The first matrix
  \param b
  const cv::Mat&
	The second matrix
  \return bool
  boolean
  returns true if the matrices are identical, false otherwise
*/
/************************************************************************/
static bool SameMat(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size() || a.type() != b.type()) return false;
    if (a.empty()) return true;
    cv::Mat diff;
    cv::absdiff(a.reshape(1), b.reshape(1), diff);
    return cv::countNonZero(diff) == 0;
}

/************************************************************************/
/*!
  \brief
	builds a single-channel 8-bit image from a flat row-major list of values
  \param rows
  int
	The number of rows
  \param cols
  int
	The number of columns
  \param v
  const std::vector<int>&
	The rows * cols pixel values, row by row
  \return cv::Mat
  OpenCV matrix
  returns the CV_8UC1 image
*/
/************************************************************************/
static cv::Mat MakeU8(int rows, int cols, const std::vector<int>& v)
{
    cv::Mat m(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) m.at<uchar>(i, j) = static_cast<uchar>(v[i * cols + j]);
    return m;
}

/************************************************************************/
/*!
  \brief
	builds a colour image with uniformly random pixel values
  \param rows
  int
	The number of rows
  \param cols
  int
	The number of columns
  \param rng
  std::mt19937&
	The random number generator to draw the pixel values from
  \return cv::Mat
  OpenCV matrix
  returns the CV_8UC3 image
*/
/************************************************************************/
static cv::Mat RandomBGR(int rows, int cols, std::mt19937& rng)
{
    std::uniform_int_distribution<int> d(0, 255);
    cv::Mat img(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(d(rng)), static_cast<uchar>(d(rng)),
                                                static_cast<uchar>(d(rng)));
    return img;
}

/************************************************************************/
/*!
  \brief
	reads a colour image and prints a hint to run from the repository root
	if it cannot be read
  \param path
  const char*
	The path of the image
  \return cv::Mat
  OpenCV matrix
  returns the image, or an empty matrix if it could not be read
*/
/************************************************************************/
static cv::Mat LoadImage(const char* path)
{
    const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) std::printf("  (could not read %s; run from the repo root)\n", path);
    return img;
}

/************************************************************************/
/*!
  \brief
	1. RemoveVerticalSeam on hand-made examples.
*/
/************************************************************************/
static void TestRemoveVertical()
{
    // 1  2  3  4          2  3  4
    // 5  6  7  8   -->    5  7  8      seam = {0, 1, 2}
    // 9 10 11 12          9 10 12
    cv::Mat img = MakeU8(3, 4, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    DP::RemoveVerticalSeam(img, { 0, 1, 2 });
    CHECK(SameMat(img, MakeU8(3, 3, { 2, 3, 4, 5, 7, 8, 9, 10, 12 })));

    // First and last column.
    cv::Mat a = MakeU8(2, 3, { 1, 2, 3, 4, 5, 6 });
    DP::RemoveVerticalSeam(a, { 0, 0 });
    CHECK(SameMat(a, MakeU8(2, 2, { 2, 3, 5, 6 })));
    cv::Mat b = MakeU8(2, 3, { 1, 2, 3, 4, 5, 6 });
    DP::RemoveVerticalSeam(b, { 2, 2 });
    CHECK(SameMat(b, MakeU8(2, 2, { 1, 2, 4, 5 })));

    // Colour image: every channel of the seam pixel goes, nothing else changes.
    cv::Mat c(2, 3, CV_8UC3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            c.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(10 * i + j), 100, static_cast<uchar>(200 + j));
    DP::RemoveVerticalSeam(c, { 1, 2 });
    CHECK(c.rows == 2 && c.cols == 2 && c.type() == CV_8UC3);
    CHECK(c.at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 100, 200));
    CHECK(c.at<cv::Vec3b>(0, 1) == cv::Vec3b(2, 100, 202));
    CHECK(c.at<cv::Vec3b>(1, 0) == cv::Vec3b(10, 100, 200));
    CHECK(c.at<cv::Vec3b>(1, 1) == cv::Vec3b(11, 100, 201));

    // Two-channel int map (the coordinate map Carve uses).
    cv::Mat m(2, 3, CV_32SC2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) m.at<cv::Vec2i>(i, j) = cv::Vec2i(j, i);
    DP::RemoveVerticalSeam(m, { 0, 1 });
    CHECK(m.at<cv::Vec2i>(0, 0) == cv::Vec2i(1, 0) && m.at<cv::Vec2i>(0, 1) == cv::Vec2i(2, 0));
    CHECK(m.at<cv::Vec2i>(1, 0) == cv::Vec2i(0, 1) && m.at<cv::Vec2i>(1, 1) == cv::Vec2i(2, 1));
}

/************************************************************************/
/*!
  \brief
	2. RemoveHorizontalSeam on a hand-made example.
*/
/************************************************************************/
static void TestRemoveHorizontal()
{
    // 1 2 3                         4 2 3
    // 4 5 6   seam = {0, 1, 2}  --> 7 8 6
    // 7 8 9   (row removed in each column)
    cv::Mat img = MakeU8(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    DP::RemoveHorizontalSeam(img, { 0, 1, 2 });
    CHECK(SameMat(img, MakeU8(2, 3, { 4, 2, 3, 7, 8, 6 })));

    // Same as transpose -> remove vertical -> transpose.
    std::mt19937 rng(3);
    const cv::Mat src = RandomBGR(7, 9, rng);
    const std::vector<int> seam = { 0, 1, 2, 3, 4, 5, 6, 6, 5 };
    cv::Mat h = src.clone();
    DP::RemoveHorizontalSeam(h, seam);
    cv::Mat t, back;
    cv::transpose(src, t);
    DP::RemoveVerticalSeam(t, seam);
    cv::transpose(t, back);
    CHECK(SameMat(h, back));
}

/************************************************************************/
/*!
  \brief
	3. Invalid removals throw instead of corrupting memory.
*/
/************************************************************************/
static void TestInvalidRemoval()
{
    const cv::Mat img = MakeU8(3, 4, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    CHECK(Throws([&] { cv::Mat m = img.clone(); DP::RemoveVerticalSeam(m, { 0, 1 }); }));        // too short
    CHECK(Throws([&] { cv::Mat m = img.clone(); DP::RemoveVerticalSeam(m, { 0, 1, -1 }); }));    // negative
    CHECK(Throws([&] { cv::Mat m = img.clone(); DP::RemoveVerticalSeam(m, { 0, 1, 4 }); }));     // == cols
    CHECK(Throws([&] { cv::Mat m(3, 1, CV_8UC1, cv::Scalar(0)); DP::RemoveVerticalSeam(m, { 0, 0, 0 }); }));  // width 1
    CHECK(Throws([&] { cv::Mat m; DP::RemoveVerticalSeam(m, {}); }));                            // empty

    CHECK(Throws([&] { cv::Mat m = img.clone(); DP::RemoveHorizontalSeam(m, { 0, 1, 2 }); }));   // needs 4 entries
    CHECK(Throws([&] { cv::Mat m = img.clone(); DP::RemoveHorizontalSeam(m, { 0, 1, 2, 3 }); })); // 3 == rows
    CHECK(Throws([&] { cv::Mat m(1, 3, CV_8UC1, cv::Scalar(0)); DP::RemoveHorizontalSeam(m, { 0, 0, 0 }); }));  // height 1

    // A failed call leaves the image unchanged.
    cv::Mat m = img.clone();
    Throws([&] { DP::RemoveVerticalSeam(m, { 0, 1, 9 }); });
    CHECK(SameMat(m, img));
}

/************************************************************************/
/*!
  \brief
	4. Carve output sizes and argument checks.
*/
/************************************************************************/
static void TestCarveSizes()
{
    std::mt19937 rng(4);
    const cv::Mat img = RandomBGR(15, 20, rng);   // 20 wide, 15 high
    const int targets[][2] = { { 12, 15 }, { 20, 9 }, { 12, 9 }, { 1, 15 }, { 20, 1 }, { 1, 1 } };
    for (const auto& t : targets)
    {
        const cv::Mat out = DP::Carve(img, t[0], t[1]);
        CHECK(out.cols == t[0] && out.rows == t[1] && out.type() == img.type());
    }

    // Same size: an unchanged copy, not the same buffer.
    const cv::Mat same = DP::Carve(img, 20, 15);
    CHECK(SameMat(same, img) && same.data != img.data);

    CHECK(Throws([&] { DP::Carve(img, 0, 15); }));
    CHECK(Throws([&] { DP::Carve(img, 21, 15); }));   // enlarging is not supported
    CHECK(Throws([&] { DP::Carve(img, 20, 0); }));
    CHECK(Throws([&] { DP::Carve(img, 20, 16); }));
    CHECK(Throws([&] { DP::Carve(cv::Mat(), 1, 1); }));

    // Grayscale and BGRA images work too.
    cv::Mat gray, bgra;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
    CHECK(DP::Carve(gray, 14, 11).size() == cv::Size(14, 11));
    CHECK(DP::Carve(bgra, 14, 11).size() == cv::Size(14, 11));
}

/************************************************************************/
/*!
  \brief
	5. Carve does exactly "energy -> seam -> remove" with the energy RECOMPUTED
	   from the current image at every step.
	   (Backward energy only: |Gx| + |Gy| is symmetric, so the energy of the
	   transposed image is the transpose of the energy and FindHorizontalSeam
	   gives the same seam Carve finds on the transposed image. The simplified
	   forward energy only looks left/right, so it is not symmetric.)
*/
/************************************************************************/
static void TestCarveMatchesManualLoop()
{
    std::mt19937 rng(5);
    const cv::Mat img = RandomBGR(18, 24, rng);
    const int newW = 17, newH = 13;

    cv::Mat cur = img.clone();
    std::vector<long long> costs;
    for (int k = 0; k < img.cols - newW; ++k)
    {
        const cv::Mat e = DP::ComputeEnergy(cur);
        const std::vector<int> s = DP::FindVerticalSeam(e);
        costs.push_back(DP::SeamEnergy(e, s, true));
        DP::RemoveVerticalSeam(cur, s);
    }
    for (int k = 0; k < img.rows - newH; ++k)
    {
        const cv::Mat e = DP::ComputeEnergy(cur);
        const std::vector<int> s = DP::FindHorizontalSeam(e);
        costs.push_back(DP::SeamEnergy(e, s, false));
        DP::RemoveHorizontalSeam(cur, s);
    }

    DP::CarveStats st;
    const cv::Mat out = DP::Carve(img, newW, newH, DP::EnergyType::Backward, &st);
    CHECK(SameMat(out, cur));
    CHECK(st.seamCosts == costs);
    long long sum = 0;
    for (long long c : costs) sum += c;
    CHECK(st.totalCost == sum);
    CHECK(st.seamSeconds >= 0.0);
    CHECK(st.verticalSeams.empty() && st.horizontalSeams.empty());   // recordSeams = false
}

/************************************************************************/
/*!
  \brief
	6. Recorded seams are in original-image coordinates, never overlap, and
	   account for every removed pixel. DrawSeams marks exactly those pixels.
*/
/************************************************************************/
static void TestRecordedSeams()
{
    std::mt19937 rng(6);
    const cv::Mat img = RandomBGR(20, 30, rng);   // 30 wide, 20 high
    const int newW = 22, newH = 14;

    DP::CarveStats st;
    const cv::Mat out = DP::Carve(img, newW, newH, DP::EnergyType::Backward, &st, true);
    CHECK(static_cast<int>(st.verticalSeams.size()) == img.cols - newW);
    CHECK(static_cast<int>(st.horizontalSeams.size()) == img.rows - newH);
    CHECK(st.seamCosts.size() == st.verticalSeams.size() + st.horizontalSeams.size());

    cv::Mat used(img.size(), CV_8UC1, cv::Scalar(0));
    int removed = 0;
    bool inside = true, overlap = false, rowsOk = true, lengthsOk = true;
    for (const auto& seam : st.verticalSeams)
    {
        lengthsOk = lengthsOk && static_cast<int>(seam.size()) == img.rows;
        for (int i = 0; i < static_cast<int>(seam.size()); ++i)
            rowsOk = rowsOk && seam[i].y == i;   // vertical seams: one pixel in every original row
    }
    for (const auto& seam : st.horizontalSeams)
        lengthsOk = lengthsOk && static_cast<int>(seam.size()) == newW;
    for (const auto* list : { &st.verticalSeams, &st.horizontalSeams })
        for (const auto& seam : *list)
            for (const cv::Point& p : seam)
            {
                if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) { inside = false; continue; }
                if (used.at<uchar>(p)) overlap = true;
                used.at<uchar>(p) = 1;
                ++removed;
            }
    CHECK(lengthsOk);
    CHECK(rowsOk);
    CHECK(inside);
    CHECK(!overlap);
    CHECK(removed == img.cols * img.rows - out.cols * out.rows);

    // DrawSeams: same size as the original, seam pixels coloured.
    const cv::Mat vis = DP::DrawSeams(img, st);
    CHECK(vis.size() == img.size() && vis.type() == CV_8UC3);
    bool coloured = true;
    for (const auto& seam : st.verticalSeams)
        for (const cv::Point& p : seam) coloured = coloured && vis.at<cv::Vec3b>(p) == cv::Vec3b(0, 0, 255);
    for (const auto& seam : st.horizontalSeams)
        for (const cv::Point& p : seam) coloured = coloured && vis.at<cv::Vec3b>(p) == cv::Vec3b(0, 255, 0);
    CHECK(coloured);

    // Stats from a bigger image cannot be drawn on a smaller one.
    CHECK(Throws([&] { DP::DrawSeams(out, st); }));
}

/************************************************************************/
/*!
  \brief
	7. Reference totals (docs/GraphCutting/README.md §3.1, same energy, same
	   leftmost tie-break).
*/
/************************************************************************/
static void TestReferenceTotals()
{
    const cv::Mat tower = LoadImage("images/tower.png");
    const cv::Mat synthetic = LoadImage("images/synthetic.png");
    CHECK(!tower.empty() && !synthetic.empty());
    if (tower.empty() || synthetic.empty()) return;

    DP::CarveStats a, b, c;
    CHECK(DP::Carve(tower, tower.cols - 80, tower.rows, DP::EnergyType::Backward, &a).cols == tower.cols - 80);
    CHECK(a.totalCost == 430676);
    DP::Carve(synthetic, synthetic.cols - 50, synthetic.rows, DP::EnergyType::Backward, &b);
    CHECK(b.totalCost == 16400);
    const cv::Mat both = DP::Carve(tower, 194, 150, DP::EnergyType::Backward, &c);
    CHECK(both.cols == 194 && both.rows == 150);
    CHECK(c.totalCost == 668244);
    std::printf("  tower 80 vertical: %lld | synthetic 50 vertical: %lld | tower -> 194x150: %lld\n",
                a.totalCost, b.totalCost, c.totalCost);
}

/************************************************************************/
/*!
  \brief
	8. DP and Graph Cut produce pixel-identical results (both energies).
*/
/************************************************************************/
static void TestMatchesGraphCut()
{
    std::mt19937 rng(8);
    std::vector<cv::Mat> images;
    for (int k = 0; k < 4; ++k) images.push_back(RandomBGR(15, 20, rng));
    const cv::Mat tower = LoadImage("images/tower.png");
    if (!tower.empty()) images.push_back(tower(cv::Rect(100, 60, 40, 30)).clone());   // real-image crop

    const std::pair<DP::EnergyType, GraphCut::EnergyType> energies[] = {
        { DP::EnergyType::Backward, GraphCut::EnergyType::Backward },
        { DP::EnergyType::Forward, GraphCut::EnergyType::Forward },
    };
    for (const cv::Mat& img : images)
        for (const auto& e : energies)
        {
            const int newW = img.cols * 3 / 4, newH = img.rows * 4 / 5;
            DP::CarveStats ds;
            GraphCut::CarveStats gs;
            const cv::Mat dp = DP::Carve(img, newW, newH, e.first, &ds);
            const cv::Mat gc = GraphCut::Carve(img, newW, newH, GraphCut::MaxFlowAlgorithm::Dinic, e.second, &gs);
            CHECK(SameMat(dp, gc));
            CHECK(ds.totalCost == static_cast<long long>(gs.totalCost));
        }
}

/************************************************************************/
/*!
  \brief
	9. Forward energy carving.
*/
/************************************************************************/
static void TestForwardCarve()
{
    std::mt19937 rng(9);
    const cv::Mat img = RandomBGR(16, 21, rng);
    DP::CarveStats st;
    const cv::Mat out = DP::Carve(img, 13, 10, DP::EnergyType::Forward, &st);
    CHECK(out.cols == 13 && out.rows == 10);
    CHECK(st.seamCosts.size() == static_cast<size_t>((21 - 13) + (16 - 10)));
    bool nonNegative = true;
    for (long long c : st.seamCosts) nonNegative = nonNegative && c >= 0;
    CHECK(nonNegative);
}

/************************************************************************/
/*!
  \brief
	carves the tower image to 194x150 with backward energy and prints the
	total time, the time spent finding seams and the total energy removed.
	Not a check; skipped if the image cannot be read.
*/
/************************************************************************/
static void Timing()
{
    const cv::Mat tower = cv::imread("images/tower.png", cv::IMREAD_COLOR);
    if (tower.empty()) return;
    DP::CarveStats st;
    const auto t0 = std::chrono::steady_clock::now();
    DP::Carve(tower, 194, 150, DP::EnergyType::Backward, &st);
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nDP carve, tower %dx%d -> 194x150 (%zu seams): %.3f s total, %.3f s energy + seam finding\n",
                tower.cols, tower.rows, st.seamCosts.size(), total, st.seamSeconds);
}

/************************************************************************/
/*!
  \brief
	runs every test group, prints the summary, then prints the timing
  \return int
  integer
  returns 0 if every check passed, 1 otherwise
*/
/************************************************************************/
int main()
{
    const int result = RunTests({
        { "remove vertical seam", TestRemoveVertical },
        { "remove horizontal seam", TestRemoveHorizontal },
        { "invalid removal throws", TestInvalidRemoval },
        { "carve output sizes / argument checks", TestCarveSizes },
        { "carve == manual energy->seam->remove loop", TestCarveMatchesManualLoop },
        { "recorded seams and DrawSeams", TestRecordedSeams },
        { "reference totals (tower, synthetic)", TestReferenceTotals },
        { "DP carve == Graph Cut carve", TestMatchesGraphCut },
        { "forward energy carve", TestForwardCarve },
    });
    Timing();
    return result;
}
