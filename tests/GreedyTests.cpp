/*!************************************************************************
\file GreedyTests.cpp
\author XIONG YANG
\par DP email: 2400647@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Correctness tests for GreedyAlgorithm.cpp (Person B: check and explain).
No test framework (shared helpers in tests/Check.h). Build from the repo root:
  Windows:       cmake --preset vs2022
                 cmake --build --preset vs2022-release --target greedy_tests
                 out\build\vs2022\Release\greedy_tests.exe
  Linux / macOS: cmake --preset unix-release
                 cmake --build --preset unix-release --target greedy_tests
                 out/build/unix-release/greedy_tests
Run it from the repository root so images/ is found.

What is checked:
  1. Hand-checked maps: seams and costs worked out on paper (see README §3.1).
  2. Tie-breaking and image borders, exactly as documented in GreedyAlgorithm.h.
  3. The "greedy trap": a cheap start pixel that leads into an expensive
     region, so greedy pays 80x the optimum.
  4. Valid seams on 2 000 random maps: right length, inside the image,
     8-connected, cost = sum of the pixels, every step is the locally
     cheapest neighbour, and cost >= the optimal (DP) cost.
  5. The optimum used for comparison is itself checked against brute force
     (every one of the up to n * 3^(m-1) seams enumerated on tiny maps).
  6. Seam removal (vertical and horizontal, any Mat type) and its errors.
  7. The Carve loop: output size, seam bookkeeping, every original pixel
     removed at most once, stats add up, Carve == manual find/remove loop.
  8. DrawSeams colours exactly the removed pixels.
  9. The tower image: greedy >= optimal at every one of 80 carving steps.
The "optimal" yardstick is the team's DP (TeamDPCost in tests/Check.h);
GraphCuttingTests.cpp checks that graph cut gives the same optimum.
- RandomMap(rng, m, n, maxValue)
builds a random cost map
- BruteForce(c, i, j, sum, best, count)
enumerates every seam from (i, j) down
- BruteForceMin(c, count)
returns the optimal seam cost by brute force
- ValidSeam(seam, rows, cols)
checks the shape of a vertical seam
- SumAlong(c, seam)
adds up the costs along a seam
- FollowsGreedyRule(c, seam)
checks a seam against an independent greedy rule
- TestHandCheckedMaps() ... TestTowerImage()
the seven test groups listed in main
- TestImage()
builds the image used by TestCarve
- main()
runs every test and prints the summary
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "Check.h"
#include "DynamicProgramming.h"
#include "GreedyAlgorithm.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Greedy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/************************************************************************/
/*!
  \brief
	builds a cost map with random values in [0, maxValue)
  \param rng
  std::mt19937&
	The random number generator
  \param m
  int
	The number of rows
  \param n
  int
	The number of columns
  \param maxValue
  int
	One more than the largest value
  \return cv::Mat
  OpenCV matrix
  returns the CV_32S cost map
*/
/************************************************************************/
static cv::Mat RandomMap(std::mt19937& rng, int m, int n, int maxValue)
{
    cv::Mat c(m, n, CV_32S);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) c.at<int>(i, j) = static_cast<int>(rng() % maxValue);
    return c;
}

/************************************************************************/
/*!
  \brief
	Brute force: enumerate EVERY connected vertical seam. Returns the minimum
	cost and (via count) how many seams exist. Only for tiny maps.
	Recursive helper: follows every path from (i, j) down to the last row.
  \param c
  const cv::Mat&
	The CV_32SC1 cost (energy) map
  \param i
  int
	The current row
  \param j
  int
	The current column
  \param sum
  int64_t
	The cost of the path so far, before (i, j)
  \param best
  int64_t&
	The cheapest complete seam found so far, updated in place
  \param count
  int64_t&
	The number of complete seams found so far, updated in place
*/
/************************************************************************/
static void BruteForce(const cv::Mat& c, int i, int j, int64_t sum, int64_t& best, int64_t& count)
{
    sum += c.at<int>(i, j);
    if (i == c.rows - 1)
    {
        ++count;
        best = std::min(best, sum);
        return;
    }
    for (int d = -1; d <= 1; ++d)
        if (j + d >= 0 && j + d < c.cols) BruteForce(c, i + 1, j + d, sum, best, count);
}

/************************************************************************/
/*!
  \brief
	finds the optimal vertical seam cost by brute force, starting BruteForce
	at every pixel of row 0
  \param c
  const cv::Mat&
	The CV_32SC1 cost (energy) map
  \param count
  int64_t*
	Receives the number of seams; may be nullptr
  \return int64_t
  64-bit integer
  returns the minimum seam cost
*/
/************************************************************************/
static int64_t BruteForceMin(const cv::Mat& c, int64_t* count = nullptr)
{
    int64_t best = INT64_MAX, n = 0;
    for (int j = 0; j < c.cols; ++j) BruteForce(c, 0, j, 0, best, n);
    if (count) *count = n;
    return best;
}

/************************************************************************/
/*!
  \brief
	A vertical seam is valid if it has one entry per row, every entry is inside
	the image, and consecutive entries differ by at most one column.
  \param seam
  const std::vector<int>&
	The seam to check
  \param rows
  int
	The number of rows
  \param cols
  int
	The number of columns
  \return bool
  boolean
  returns true if the seam is valid, false otherwise
*/
/************************************************************************/
static bool ValidSeam(const std::vector<int>& seam, int rows, int cols)
{
    if (static_cast<int>(seam.size()) != rows) return false;
    for (size_t i = 0; i < seam.size(); ++i)
    {
        if (seam[i] < 0 || seam[i] >= cols) return false;
        if (i > 0 && std::abs(seam[i] - seam[i - 1]) > 1) return false;
    }
    return true;
}

/************************************************************************/
/*!
  \brief
	adds up the cost of every pixel on a vertical seam
  \param c
  const cv::Mat&
	The CV_32SC1 cost (energy) map
  \param seam
  const std::vector<int>&
	The seam: seam[i] is the column in row i
  \return int64_t
  64-bit integer
  returns the total cost of the seam
*/
/************************************************************************/
static int64_t SumAlong(const cv::Mat& c, const std::vector<int>& seam)
{
    int64_t s = 0;
    for (int i = 0; i < c.rows; ++i) s += c.at<int>(i, seam[i]);
    return s;
}

/************************************************************************/
/*!
  \brief
	checks that a seam is exactly the greedy seam.
	The greedy rule written independently of GreedyAlgorithm.cpp:
	  row 0: the leftmost minimum;
	  row i: the cheapest of {p, p-1, p+1}, ties taken in exactly that order.
  \param c
  const cv::Mat&
	The CV_32SC1 cost (energy) map
  \param seam
  const std::vector<int>&
	The seam to check
  \return bool
  boolean
  returns true if every step follows the greedy rule, false otherwise
*/
/************************************************************************/
static bool FollowsGreedyRule(const cv::Mat& c, const std::vector<int>& seam)
{
    const int* row0 = c.ptr<int>(0);
    const int start = static_cast<int>(std::min_element(row0, row0 + c.cols) - row0);
    if (seam[0] != start) return false;
    for (int i = 1; i < c.rows; ++i)
    {
        const int p = seam[i - 1];
        int lo = INT32_MAX;
        for (int k = p - 1; k <= p + 1; ++k)
            if (k >= 0 && k < c.cols) lo = std::min(lo, c.at<int>(i, k));
        int expected = -1;
        for (int k : { p, p - 1, p + 1 })
            if (k >= 0 && k < c.cols && c.at<int>(i, k) == lo) { expected = k; break; }
        if (seam[i] != expected) return false;
    }
    return true;
}

/************************************************************************/
/*!
  \brief
	1. Hand-checked maps
*/
/************************************************************************/
static void TestHandCheckedMaps()
{
    // Worked by hand in README §3.1:
    //   row 0: minimum is 2 at col 1                       -> seam[0] = 1, cost 2
    //   row 1: from col 1, neighbours {6, 4, 1} at cols 1,0,2 -> col 2, cost 3
    //   row 2: from col 2, neighbours {5, 3, 2} at cols 2,1,3 -> col 3, cost 5
    //   row 3: from col 3, neighbours {6, 4}    at cols 3,2   -> col 2, cost 9
    // DP optimum: 2 (col 1) -> 1 (col 2) -> 3 (col 1) -> 1 (col 0) = 7.
    // Greedy took the cheap 2 in row 2 (col 3) and then had to pay 4 in row 3,
    // instead of the 3 in row 2 (col 1) that leads to the 1 in row 3.
    const cv::Mat c = FromRows({ { 5, 2, 7, 3 },
                                 { 4, 6, 1, 8 },
                                 { 9, 3, 5, 2 },
                                 { 1, 7, 4, 6 } });
    const SeamResult v = FindVerticalSeam(c);
    CHECK((v.seam == std::vector<int>{ 1, 2, 3, 2 }));
    CHECK(v.cost == 9);
    CHECK(TeamDPCost(c) == 7);
    CHECK(BruteForceMin(c) == 7);
    CHECK(v.cost > TeamDPCost(c));   // greedy is NOT optimal here

    // The same map transposed: the horizontal seam must be the same path.
    const SeamResult h = FindHorizontalSeam(c.t());
    CHECK((h.seam == std::vector<int>{ 1, 2, 3, 2 }));
    CHECK(h.cost == 9);

    // A horizontal map worked directly. Column 0 minimum is 1 at row 2; then
    // col 1 from row 2: {0 at row 2} straight wins; col 2 from row 2:
    // {7 (row 2), 3 (row 1), -- } -> row 1; col 3 from row 1: {2, 6, 9} -> row 1.
    const cv::Mat hm = FromRows({ { 4, 5, 9, 6 },
                                  { 8, 3, 3, 2 },
                                  { 1, 0, 7, 9 } });
    const SeamResult h2 = FindHorizontalSeam(hm);
    CHECK((h2.seam == std::vector<int>{ 2, 2, 1, 1 }));
    CHECK(h2.cost == 1 + 0 + 3 + 2);

    // A map where greedy happens to be optimal: one column of zeros.
    cv::Mat z(6, 5, CV_32S, cv::Scalar(7));
    z.col(3).setTo(0);
    const SeamResult zr = FindVerticalSeam(z);
    CHECK(zr.cost == 0);
    for (int j : zr.seam) CHECK(j == 3);
}

/************************************************************************/
/*!
  \brief
	2. Tie-breaking and borders (documented in GreedyAlgorithm.h)
*/
/************************************************************************/
static void TestTieBreakingAndBorders()
{
    // Start: equal minima -> lowest index.
    CHECK(FindVerticalSeam(FromRows({ { 3, 1, 1, 1 } })).seam[0] == 1);
    CHECK(FindVerticalSeam(cv::Mat(3, 4, CV_32S, cv::Scalar(5))).seam ==
          (std::vector<int>{ 0, 0, 0 }));

    // Step: all three equal -> straight down.
    CHECK((FindVerticalSeam(FromRows({ { 9, 0, 9 }, { 2, 2, 2 } })).seam == std::vector<int>{ 1, 1 }));
    // Step: left and right equal and cheaper than straight -> left (smaller index).
    CHECK((FindVerticalSeam(FromRows({ { 9, 0, 9 }, { 1, 5, 1 } })).seam == std::vector<int>{ 1, 0 }));
    // Step: straight equal to the cheaper side -> straight.
    CHECK((FindVerticalSeam(FromRows({ { 9, 0, 9 }, { 1, 1, 4 } })).seam == std::vector<int>{ 1, 1 }));
    // Step: only the right is strictly cheaper -> right.
    CHECK((FindVerticalSeam(FromRows({ { 9, 0, 9 }, { 3, 3, 2 } })).seam == std::vector<int>{ 1, 2 }));

    // Left border: from col 0 only cols 0 and 1 are reachable (not col 2, even though it is 0).
    CHECK((FindVerticalSeam(FromRows({ { 0, 9, 9 }, { 5, 1, 0 } })).seam == std::vector<int>{ 0, 1 }));
    // Right border: from the last col only the last two cols are reachable.
    CHECK((FindVerticalSeam(FromRows({ { 9, 9, 0 }, { 0, 1, 5 } })).seam == std::vector<int>{ 2, 1 }));

    // Horizontal ties: up before down.
    CHECK((FindHorizontalSeam(FromRows({ { 9, 1 }, { 0, 5 }, { 9, 1 } })).seam == std::vector<int>{ 1, 0 }));

    // Degenerate shapes.
    const SeamResult oneCol = FindVerticalSeam(FromRows({ { 4 }, { 2 }, { 7 } }));
    CHECK((oneCol.seam == std::vector<int>{ 0, 0, 0 }) && oneCol.cost == 13);
    const SeamResult oneRow = FindVerticalSeam(FromRows({ { 3, 1, 2 } }));
    CHECK((oneRow.seam == std::vector<int>{ 1 }) && oneRow.cost == 1);
    const SeamResult single = FindHorizontalSeam(FromRows({ { 5 } }));
    CHECK((single.seam == std::vector<int>{ 0 }) && single.cost == 5);

    // Invalid energy maps.
    CHECK(Throws([] { FindVerticalSeam(cv::Mat()); }));
    CHECK(Throws([] { FindHorizontalSeam(cv::Mat()); }));
    CHECK(Throws([] { FindVerticalSeam(cv::Mat::zeros(3, 3, CV_8U)); }));
    CHECK(Throws([] { FindHorizontalSeam(cv::Mat::zeros(3, 3, CV_32F)); }));
    // Negative energies are rejected (same rule as DP and Graph Cut).
    CHECK(Throws([] { FindVerticalSeam(FromRows({ { 1, -1 }, { 2, 3 } })); }));
    CHECK(Throws([] { FindHorizontalSeam(FromRows({ { 1, -1 }, { 2, 3 } })); }));
}

/************************************************************************/
/*!
  \brief
	3. The greedy trap
*/
/************************************************************************/
static void TestGreedyTrap()
{
    // Row 0 has its cheapest pixel (0) at the far right, but that pixel leads
    // into a 2-column wall of 100s. Greedy commits to it and can never escape:
    // from cols 3-4 it can only reach cols 2-4, and the cheapest of those it
    // can see is always 100 (straight wins ties). DP sees the column of 1s.
    //    1 1 1 1   0
    //    1 1 1 100 100
    //    1 1 1 100 100  ... 5 rows
    cv::Mat c(5, 5, CV_32S, cv::Scalar(1));
    c.at<int>(0, 4) = 0;
    c(cv::Rect(3, 1, 2, 4)).setTo(100);
    // From col 4 the reachable cols are 3 and 4 only: greedy stays at col 4.
    // (Col 2 would be reachable from col 3, but greedy never goes to col 3
    // because straight wins the 100 vs 100 tie.)
    const SeamResult r = FindVerticalSeam(c);
    CHECK((r.seam == std::vector<int>{ 4, 4, 4, 4, 4 }));
    CHECK(r.cost == 400);
    CHECK(TeamDPCost(c) == 5);
    CHECK(BruteForceMin(c) == 5);
    std::printf("  trap: greedy %lld vs optimal %lld (%.0fx)\n", (long long)r.cost,
                (long long)TeamDPCost(c), double(r.cost) / TeamDPCost(c));

    // The trap gets worse with height: greedy grows by 100 per row, DP by 1.
    for (int m : { 10, 50, 200 })
    {
        cv::Mat t(m, 5, CV_32S, cv::Scalar(1));
        t.at<int>(0, 4) = 0;
        t(cv::Rect(3, 1, 2, m - 1)).setTo(100);
        CHECK(FindVerticalSeam(t).cost == 100LL * (m - 1));
        CHECK(TeamDPCost(t) == m);
    }

    // The same trap, horizontally (transpose).
    const SeamResult rh = FindHorizontalSeam(c.t());
    CHECK(rh.cost == 400 && TeamDPCost(c) == 5);
}

/************************************************************************/
/*!
  \brief
	4. + 5. Random maps: valid seams, greedy rule, greedy >= optimal (= brute force)
*/
/************************************************************************/
static void TestRandomMaps()
{
    std::mt19937 rng(2024);
    int optimal = 0, total = 0;
    for (int trial = 0; trial < 2000; ++trial)
    {
        const int m = 1 + rng() % 15, n = 1 + rng() % 15;
        const cv::Mat c = RandomMap(rng, m, n, 1 + rng() % 100);
        const SeamResult v = FindVerticalSeam(c);
        const SeamResult h = FindHorizontalSeam(c);
        const int64_t best = TeamDPCost(c);

        CHECK(ValidSeam(v.seam, m, n));
        CHECK(v.cost == SumAlong(c, v.seam));
        CHECK(FollowsGreedyRule(c, v.seam));
        CHECK(v.cost >= best);

        // Horizontal on c == vertical on c transposed (same tie rule: up before down).
        const SeamResult vt = FindVerticalSeam(c.t());
        CHECK(h.seam == vt.seam && h.cost == vt.cost);
        CHECK(ValidSeam(h.seam, n, m));
        CHECK(h.cost >= TeamDPCost(cv::Mat(c.t())));

        optimal += (v.cost == best);
        ++total;
    }
    std::printf("  greedy found an optimal seam on %d/%d random maps\n", optimal, total);

    // The yardstick is itself correct: DP == brute force over every seam.
    for (int trial = 0; trial < 300; ++trial)
    {
        const int m = 1 + rng() % 6, n = 1 + rng() % 6;
        const cv::Mat c = RandomMap(rng, m, n, 20);
        CHECK(TeamDPCost(c) == BruteForceMin(c));
        CHECK(FindVerticalSeam(c).cost >= BruteForceMin(c));
    }

    // Question 2a(i): the number of seams grows at least exponentially in m.
    // For n = 2 every row has 2 choices whatever came before: exactly 2^m seams.
    for (int m = 1; m <= 10; ++m)
    {
        int64_t count = 0;
        BruteForceMin(cv::Mat::zeros(m, 2, CV_32S), &count);
        CHECK(count == (int64_t(1) << m));
    }
}

/************************************************************************/
/*!
  \brief
	6. Seam removal
*/
/************************************************************************/
static void TestRemoval()
{
    // Vertical, by hand: remove cols {0, 1, 3} from rows 0, 1, 2.
    cv::Mat a = FromRows({ { 0, 1, 2, 3 },
                           { 4, 5, 6, 7 },
                           { 8, 9, 10, 11 } });
    RemoveVerticalSeam(a, { 0, 1, 3 });
    const cv::Mat expectV = FromRows({ { 1, 2, 3 },
                                       { 4, 6, 7 },
                                       { 8, 9, 10 } });
    CHECK(a.size() == expectV.size() && cv::countNonZero(a != expectV) == 0);

    // Horizontal, by hand: remove rows {2, 0, 1} from cols 0, 1, 2.
    cv::Mat b = FromRows({ { 0, 1, 2 },
                           { 3, 4, 5 },
                           { 6, 7, 8 } });
    RemoveHorizontalSeam(b, { 2, 0, 1 });
    const cv::Mat expectH = FromRows({ { 0, 4, 2 },
                                       { 3, 7, 8 } });
    CHECK(b.size() == expectH.size() && cv::countNonZero(b != expectH) == 0);

    // Colour image: pixels to the right of the seam shift left by one, the
    // rest are untouched, the type is kept.
    cv::Mat img(4, 5, CV_8UC3);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 5; ++x) img.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(x), uchar(y), uchar(10 * y + x));
    const std::vector<int> seam{ 2, 3, 4, 3 };
    cv::Mat cut = img.clone();
    RemoveVerticalSeam(cut, seam);
    CHECK(cut.rows == 4 && cut.cols == 4 && cut.type() == CV_8UC3);
    bool ok = true;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            ok &= cut.at<cv::Vec3b>(y, x) == img.at<cv::Vec3b>(y, x < seam[y] ? x : x + 1);
    CHECK(ok);

    // Removing along a horizontal seam keeps the type and the other pixels too.
    cv::Mat cutH = img.clone();
    RemoveHorizontalSeam(cutH, { 0, 1, 2, 3, 3 });
    CHECK(cutH.rows == 3 && cutH.cols == 5 && cutH.type() == CV_8UC3);
    CHECK(cutH.at<cv::Vec3b>(0, 0) == img.at<cv::Vec3b>(1, 0));   // row 0 removed in col 0
    CHECK(cutH.at<cv::Vec3b>(0, 4) == img.at<cv::Vec3b>(0, 4));   // above the seam: unchanged

    // Errors: wrong length, out of range, nothing left to remove.
    cv::Mat e = cv::Mat::zeros(3, 4, CV_32S);
    CHECK(Throws([&] { RemoveVerticalSeam(e, { 0, 1 }); }));
    CHECK(Throws([&] { RemoveVerticalSeam(e, { 0, 1, 4 }); }));
    CHECK(Throws([&] { RemoveVerticalSeam(e, { 0, -1, 0 }); }));
    CHECK(Throws([&] { RemoveHorizontalSeam(e, { 0, 1, 2 }); }));
    CHECK(Throws([&] { RemoveHorizontalSeam(e, { 0, 1, 3, 0 }); }));
    cv::Mat thin = cv::Mat::zeros(3, 1, CV_32S), flat = cv::Mat::zeros(1, 3, CV_32S);
    CHECK(Throws([&] { RemoveVerticalSeam(thin, { 0, 0, 0 }); }));
    CHECK(Throws([&] { RemoveHorizontalSeam(flat, { 0, 0, 0 }); }));
    CHECK(e.size() == cv::Size(4, 3));   // a failed removal leaves the image alone
}

// ---------------------------------------------------------------------------
// 7. + 8. The carving loop and DrawSeams
// ---------------------------------------------------------------------------

/************************************************************************/
/*!
  \brief
	builds a 30x24 gray-noise colour image (channels equal, values up to
	200) with a bright bar in columns 12-15, so no pixel is pure red or
	green before DrawSeams
  \return cv::Mat
  OpenCV matrix
  returns the CV_8UC3 test image
*/
/************************************************************************/
static cv::Mat TestImage()
{
    // Gray noise (channels equal, values <= 200) so no pixel is pure red or
    // green before DrawSeams paints the seams; a bright bar adds structure.
    std::mt19937 rng(3);
    cv::Mat img(24, 30, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
        {
            const uchar g = uchar(rng() % 200);
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(g, g, g);
        }
    img.colRange(12, 16).setTo(cv::Scalar(200, 200, 200));
    return img;
}

/************************************************************************/
/*!
  \brief
	checks Greedy::Carve with both energies: output size and type, seam
	bookkeeping, every original pixel removed at most once, stats that add
	up, DrawSeams painting exactly the removed pixels, and Carve equal to a
	hand-written find/remove loop; then accumulated stats, extreme and
	invalid sizes, gray and BGRA inputs, and DrawSeams on mismatched stats
*/
/************************************************************************/
static void TestCarve()
{
    const cv::Mat img = TestImage();

    for (EnergyType energy : { EnergyType::Backward, EnergyType::Forward })
    {
        CarveStats st;
        const cv::Mat out = Carve(img, 25, 20, energy, &st, true);
        CHECK(out.cols == 25 && out.rows == 20 && out.type() == img.type());
        CHECK(st.verticalSeams.size() == 5 && st.horizontalSeams.size() == 4);
        CHECK(st.seamCosts.size() == 9);
        int64_t sum = 0;
        for (int64_t s : st.seamCosts) sum += s;
        CHECK(sum == st.totalCost);
        CHECK(st.totalSeconds >= st.seamSeconds && st.seamSeconds >= 0.0);

        // Vertical seams are removed before any row, so each one has one point
        // per original row, in row order. Only the FIRST seam is guaranteed to
        // be connected in original x: later seams were found on a narrower
        // image, where neighbours may be several original columns apart.
        for (const auto& seam : st.verticalSeams)
        {
            CHECK(static_cast<int>(seam.size()) == img.rows);
            bool rowsInOrder = true;
            for (int i = 0; i < img.rows; ++i) rowsInOrder &= seam[i].y == i;
            CHECK(rowsInOrder);
        }
        for (const auto& seam : st.horizontalSeams) CHECK(seam.size() == 25u);
        {
            std::vector<int> first;
            for (const cv::Point& p : st.verticalSeams[0]) first.push_back(p.x);
            CHECK(ValidSeam(first, img.rows, img.cols));
        }

        // Every original pixel is removed at most once and the counts add up.
        cv::Mat removed = cv::Mat::zeros(img.size(), CV_8U);
        size_t total = 0;
        for (const auto* list : { &st.verticalSeams, &st.horizontalSeams })
            for (const auto& seam : *list)
                for (const cv::Point& p : seam) { removed.at<uchar>(p) += 1; ++total; }
        double maxCount;
        cv::minMaxLoc(removed, nullptr, &maxCount);
        CHECK(maxCount <= 1 && total + 25 * 20 == size_t(24 * 30));

        // DrawSeams paints exactly the removed pixels (red = vertical, green = horizontal).
        const cv::Mat vis = DrawSeams(img, st);
        int red = 0, green = 0;
        for (int y = 0; y < vis.rows; ++y)
            for (int x = 0; x < vis.cols; ++x)
            {
                red += vis.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255);
                green += vis.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 255, 0);
            }
        CHECK(red == 5 * 24 && green == 4 * 25);

        // Carve == the obvious loop "energy -> greedy seam -> remove" by hand.
        // A horizontal seam's energy is the energy of the transposed image,
        // transposed back. For Backward (Sobel) that is the same map; for
        // Forward it uses |I[i+1][j] - I[i-1][j]|, the edge created when the
        // pixels above and below the removed one meet.
        cv::Mat manual = img.clone();
        for (int k = 0; k < 5; ++k)
            RemoveVerticalSeam(manual, FindVerticalSeam(DP::ComputeEnergy(manual, energy)).seam);
        for (int k = 0; k < 4; ++k)
        {
            const cv::Mat eT = DP::ComputeEnergy(cv::Mat(manual.t()), energy);
            RemoveHorizontalSeam(manual, FindHorizontalSeam(cv::Mat(eT.t())).seam);
        }
        CHECK(manual.size() == out.size() && cv::norm(manual, out, cv::NORM_INF) == 0);
    }

    // Stats accumulate over calls (documented in GreedyAlgorithm.h).
    CarveStats acc;
    Carve(img, 28, 24, EnergyType::Backward, &acc);
    Carve(img, 27, 24, EnergyType::Backward, &acc);
    CHECK(acc.seamCosts.size() == 5);

    // No-op, extreme and invalid sizes.
    CHECK(cv::norm(Carve(img, img.cols, img.rows), img, cv::NORM_INF) == 0);
    CHECK(Carve(img, 1, 1).size() == cv::Size(1, 1));
    CHECK(Carve(cv::Mat(img, cv::Rect(0, 0, 5, 4)).clone(), 2, 3).size() == cv::Size(2, 3));
    CHECK(Throws([&] { Carve(img, 0, 10); }));
    CHECK(Throws([&] { Carve(img, 10, 0); }));
    CHECK(Throws([&] { Carve(img, img.cols + 1, 10); }));
    CHECK(Throws([&] { Carve(img, 10, img.rows + 1); }));
    CHECK(Throws([] { Carve(cv::Mat(), 1, 1); }));

    // Grayscale and BGRA inputs work too; DrawSeams always returns BGR.
    cv::Mat gray, bgra;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
    CarveStats sg, sa;
    CHECK(Carve(gray, 26, 22, EnergyType::Backward, &sg, true).type() == CV_8UC1);
    CHECK(Carve(bgra, 26, 22, EnergyType::Backward, &sa, true).type() == CV_8UC4);
    CHECK(DrawSeams(gray, sg).type() == CV_8UC3 && DrawSeams(bgra, sa).type() == CV_8UC3);

    // DrawSeams rejects seams that do not belong to the image.
    const cv::Mat small(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    CHECK(Throws([&] { DrawSeams(small, sg); }));                       // stats from a bigger image
    CHECK(Throws([&] { DrawSeams(cv::Mat(), sg); }));
    CHECK(Throws([&] { DrawSeams(cv::Mat(24, 30, CV_32FC3), sg); }));   // not 8-bit
}

/************************************************************************/
/*!
  \brief
	9. Real image: greedy vs optimal at every carving step
*/
/************************************************************************/
static void TestTowerImage()
{
    const cv::Mat img = cv::imread("images/tower.png");
    if (img.empty()) { std::printf("  (skipped: images/tower.png not found)\n"); return; }

    const cv::Mat c = DP::ComputeEnergy(img);
    const SeamResult first = FindVerticalSeam(c);
    CHECK(ValidSeam(first.seam, c.rows, c.cols));
    CHECK(TeamDPCost(c) == 2446);   // same optimum as GraphCuttingTests.cpp
    CHECK(first.cost >= 2446);
    std::printf("  tower first seam: greedy %lld, optimal %lld (%.2fx)\n",
                (long long)first.cost, (long long)TeamDPCost(c), double(first.cost) / TeamDPCost(c));

    // Carve 80 seams greedily; at every step the greedy seam on the CURRENT
    // image must be valid and cost at least the optimum on that same image.
    cv::Mat cur = img.clone();
    int atLeastOptimal = 0, equalToOptimal = 0;
    for (int k = 0; k < 80; ++k)
    {
        const cv::Mat e = DP::ComputeEnergy(cur);
        const SeamResult g = FindVerticalSeam(e);
        const int64_t best = TeamDPCost(e);
        CHECK(ValidSeam(g.seam, e.rows, e.cols));
        atLeastOptimal += (g.cost >= best);
        equalToOptimal += (g.cost == best);
        RemoveVerticalSeam(cur, g.seam);
    }
    std::printf("  tower: greedy >= optimal at %d/80 steps, equal at %d/80\n", atLeastOptimal, equalToOptimal);
    CHECK(atLeastOptimal == 80);
    CHECK(cur.cols == img.cols - 80);
}

// ---------------------------------------------------------------------------

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
        { "hand-checked maps", TestHandCheckedMaps },
        { "tie-breaking and borders", TestTieBreakingAndBorders },
        { "greedy trap", TestGreedyTrap },
        { "random maps: valid seams, greedy rule, greedy >= optimal", TestRandomMaps },
        { "seam removal", TestRemoval },
        { "carve loop and DrawSeams", TestCarve },
        { "tower image vs optimal", TestTowerImage },
    });
}
