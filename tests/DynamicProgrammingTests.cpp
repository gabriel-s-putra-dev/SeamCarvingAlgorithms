// Correctness tests for DynamicProgramming.cpp — step (a) energy map.
// Self-contained (no test framework). Build from the repo root:
//   g++ -std=c++14 -O2 -I. tests/DynamicProgrammingTests.cpp DynamicProgramming.cpp GraphCutting.cpp -o dp_tests `pkg-config --cflags --libs opencv4`
//   ./dp_tests            (run from the repo root so images/ is found)
//
// GraphCutting.cpp is linked only to check that DP::ComputeEnergy produces
// exactly the same cost map as GraphCut::ComputeEnergy, so all methods are
// compared on identical energies.
#include "DynamicProgramming.h"
#include "GraphCutting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0, g_passed = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) ++g_passed;                                                       \
        else { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

static const DP::EnergyType TYPES[] = { DP::EnergyType::Backward, DP::EnergyType::Forward };

static bool SameMat(const cv::Mat& a, const cv::Mat& b)
{
    return a.size() == b.size() && a.type() == b.type() && cv::countNonZero(a != b) == 0;
}

static cv::Mat RandomGray(int rows, int cols, std::mt19937& rng)
{
    cv::Mat img(rows, cols, CV_8UC1);
    std::uniform_int_distribution<int> d(0, 255);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) img.at<uchar>(i, j) = static_cast<uchar>(d(rng));
    return img;
}

// Output is CV_32SC1, same size as the input, and never negative.
static void TestTypeSizeNonNegative()
{
    std::mt19937 rng(1);
    const int sizes[][2] = { { 1, 1 }, { 1, 7 }, { 7, 1 }, { 2, 2 }, { 5, 9 }, { 33, 20 } };
    for (const auto& s : sizes)
        for (DP::EnergyType t : TYPES)
        {
            const cv::Mat e = DP::ComputeEnergy(RandomGray(s[0], s[1], rng), t);
            CHECK(e.type() == CV_32SC1);
            CHECK(e.rows == s[0] && e.cols == s[1]);
            double minVal;
            cv::minMaxLoc(e, &minVal);
            CHECK(minVal >= 0);
        }
}

// A flat image has no gradient anywhere.
static void TestConstantImageIsZero()
{
    const cv::Mat img(6, 9, CV_8UC3, cv::Scalar(37, 120, 201));
    for (DP::EnergyType t : TYPES)
        CHECK(cv::countNonZero(DP::ComputeEnergy(img, t)) == 0);
}

// 5x8 image, columns 0..3 = 0, columns 4..7 = 100.
// Backward: Sobel_x = 1*(100) + 2*(100) + 1*(100) = 400 at columns 3 and 4,
// 0 elsewhere (reflected borders keep the outer columns flat); Sobel_y = 0.
// Forward : |I[j+1] - I[j-1]| = 100 at columns 3 and 4, 0 elsewhere.
static void TestVerticalStepEdge()
{
    cv::Mat img(5, 8, CV_8UC1, cv::Scalar(0));
    img.colRange(4, 8).setTo(100);
    const cv::Mat eb = DP::ComputeEnergy(img, DP::EnergyType::Backward);
    const cv::Mat ef = DP::ComputeEnergy(img, DP::EnergyType::Forward);
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            const bool edge = (j == 3 || j == 4);
            CHECK(eb.at<int>(i, j) == (edge ? 400 : 0));
            CHECK(ef.at<int>(i, j) == (edge ? 100 : 0));
        }
}

// Same edge rotated 90 degrees: only Sobel_y responds, so Backward is
// symmetric; Forward is purely horizontal, so it sees nothing.
static void TestHorizontalStepEdge()
{
    cv::Mat img(8, 5, CV_8UC1, cv::Scalar(0));
    img.rowRange(4, 8).setTo(100);
    const cv::Mat eb = DP::ComputeEnergy(img, DP::EnergyType::Backward);
    const cv::Mat ef = DP::ComputeEnergy(img, DP::EnergyType::Forward);
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            const bool edge = (i == 3 || i == 4);
            CHECK(eb.at<int>(i, j) == (edge ? 400 : 0));
            CHECK(ef.at<int>(i, j) == 0);
        }
}

// Hand-checked forward costs, including both borders.
//   row      : 10  20  50  40  40
//   expected : |20-10| |50-10| |40-20| |40-50| |40-40| = 10 40 20 10 0
static void TestForwardHandValues()
{
    const cv::Mat img = (cv::Mat_<uchar>(1, 5) << 10, 20, 50, 40, 40);
    const cv::Mat e = DP::ComputeEnergy(img, DP::EnergyType::Forward);
    const int expected[] = { 10, 40, 20, 10, 0 };
    for (int j = 0; j < 5; ++j) CHECK(e.at<int>(0, j) == expected[j]);

    // Each row is independent in the forward cost.
    const cv::Mat img2 = (cv::Mat_<uchar>(3, 4) <<
        0,   0,   0,   0,
        0, 255,   0, 255,
        9,   3,   7,   1);
    const cv::Mat e2 = DP::ComputeEnergy(img2, DP::EnergyType::Forward);
    const int expected2[3][4] = { { 0, 0, 0, 0 }, { 255, 0, 0, 255 }, { 6, 2, 2, 6 } };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) CHECK(e2.at<int>(i, j) == expected2[i][j]);
}

// 1-, 3- and 4-channel inputs with the same gray level give the same energy.
static void TestChannelCounts()
{
    std::mt19937 rng(7);
    const cv::Mat gray = RandomGray(12, 15, rng);
    cv::Mat bgr, bgra;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gray, bgra, cv::COLOR_GRAY2BGRA);
    for (DP::EnergyType t : TYPES)
    {
        const cv::Mat e1 = DP::ComputeEnergy(gray, t);
        CHECK(SameMat(e1, DP::ComputeEnergy(bgr, t)));
        CHECK(SameMat(e1, DP::ComputeEnergy(bgra, t)));
    }
}

static void TestEmptyThrows()
{
    for (DP::EnergyType t : TYPES)
    {
        bool threw = false;
        try { DP::ComputeEnergy(cv::Mat(), t); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}

// DP and Graph Cut must see exactly the same cost map.
static void TestMatchesGraphCutEnergy()
{
    std::mt19937 rng(42);
    for (int k = 0; k < 50; ++k)
    {
        cv::Mat img;
        cv::cvtColor(RandomGray(1 + k % 13, 1 + (k * 7) % 17, rng), img, cv::COLOR_GRAY2BGR);
        CHECK(SameMat(DP::ComputeEnergy(img, DP::EnergyType::Backward),
                      GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Backward)));
        CHECK(SameMat(DP::ComputeEnergy(img, DP::EnergyType::Forward),
                      GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Forward)));
    }
    for (const char* path : { "images/tower.png", "images/synthetic.png" })
    {
        const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        CHECK(!img.empty());
        if (img.empty()) { std::printf("  (could not read %s; run from the repo root)\n", path); continue; }
        CHECK(SameMat(DP::ComputeEnergy(img, DP::EnergyType::Backward),
                      GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Backward)));
        CHECK(SameMat(DP::ComputeEnergy(img, DP::EnergyType::Forward),
                      GraphCut::ComputeEnergy(img, GraphCut::EnergyType::Forward)));
    }
}

int main()
{
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        { "type, size, non-negative", TestTypeSizeNonNegative },
        { "constant image -> zero energy", TestConstantImageIsZero },
        { "vertical step edge", TestVerticalStepEdge },
        { "horizontal step edge", TestHorizontalStepEdge },
        { "forward energy hand values", TestForwardHandValues },
        { "1/3/4-channel inputs", TestChannelCounts },
        { "empty image throws", TestEmptyThrows },
        { "matches GraphCut::ComputeEnergy", TestMatchesGraphCutEnergy },
    };
    for (const auto& test : tests)
    {
        const int before = g_failed;
        std::printf("[%s]\n", test.first);
        test.second();
        if (g_failed == before) std::printf("  ok\n");
    }
    std::printf("\n%d checks passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
