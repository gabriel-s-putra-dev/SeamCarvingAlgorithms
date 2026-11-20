/*!************************************************************************
\file Check.h
\author Gabriel Sebastian Putra
\par DP email: 2400735@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Minimal shared test support for the test programs in tests/ (no framework):

  CHECK(cond)          count a check; print "FAIL file:line cond" if false
  Throws(fn)           true if fn() throws std::invalid_argument
  FromRows({{..}})     small CV_32SC1 matrix from rows of ints
  TeamDPCost(e, &seam) optimal vertical seam cost via the team's DP
                       (the "optimal" yardstick for Greedy and Graph Cut)
  RunTests({...})      run named test functions, print "[name]" / "ok" and
                       "N checks passed, M failed"; returns the exit code

Include it in ONE .cpp per program (each test program is a single file).
**************************************************************************/

#pragma once

// ----- Headers ------------------------------------------------------- //
#include "DynamicProgramming.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

static int g_failed = 0, g_passed = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) ++g_passed;                                                       \
        else { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

/************************************************************************/
/*!
  \brief
	Runs f and reports whether it threw std::invalid_argument (any other
	exception counts as "did not throw the expected error").
  \param f
  const std::function<void()>&
	The function to run
  \return bool
  boolean
  returns true if f threw std::invalid_argument, false otherwise
*/
/************************************************************************/
inline bool Throws(const std::function<void()>& f)
{
    try { f(); }
    catch (const std::invalid_argument&) { return true; }
    catch (...) { return false; }
    return false;
}

/************************************************************************/
/*!
  \brief
	builds a small energy map from a list of rows
  \param rows
  const std::vector<std::vector<int>>&
	The values, one inner vector per row; every row must have the same
	length
  \return cv::Mat
  OpenCV matrix
  returns a CV_32SC1 matrix holding the values
*/
/************************************************************************/
inline cv::Mat FromRows(const std::vector<std::vector<int>>& rows)
{
    cv::Mat c(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()), CV_32SC1);
    for (int i = 0; i < c.rows; ++i)
        for (int j = 0; j < c.cols; ++j) c.at<int>(i, j) = rows[i][j];
    return c;
}

/************************************************************************/
/*!
  \brief
	Cost of the optimal (minimum-energy) vertical seam, from the team's DP
	(DynamicProgramming.cpp, itself checked against brute force in
	tests/SeamTests.cpp). Ties go to the leftmost seam; seamOut receives it.
  \param energy
  const cv::Mat&
	The energy map, a non-empty CV_32SC1 matrix with no negative values
  \param seamOut
  std::vector<int>*
	Receives the leftmost optimal seam; may be nullptr
  \return int64_t
  64-bit integer
  returns the total energy of the optimal vertical seam
  \exception std::invalid_argument
	thrown if energy is empty, is not CV_32SC1, or contains a negative value
*/
/************************************************************************/
inline int64_t TeamDPCost(const cv::Mat& energy, std::vector<int>* seamOut = nullptr)
{
    std::vector<int> seam = DP::FindVerticalSeam(energy);
    const int64_t cost = DP::SeamEnergy(energy, seam, true);
    if (seamOut) *seamOut = std::move(seam);
    return cost;
}

struct TestCase
{
    const char* name;
    std::function<void()> run;
};

/************************************************************************/
/*!
  \brief
	runs each test in order, printing [name] before it and ok after it if
	none of its checks failed, then prints "N checks passed, M failed"
  \param tests
  std::initializer_list<TestCase>
	The named test functions to run
  \return int
  integer
  returns 0 if every check passed, 1 otherwise (the program's exit code)
*/
/************************************************************************/
inline int RunTests(std::initializer_list<TestCase> tests)
{
    for (const TestCase& t : tests)
    {
        const int before = g_failed;
        std::printf("[%s]\n", t.name);
        t.run();
        if (g_failed == before) std::printf("  ok\n");
    }
    std::printf("\n%d checks passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
