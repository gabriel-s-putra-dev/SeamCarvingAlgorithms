/*!************************************************************************
\file GraphCuttingMemory.cpp
\author GYQ-SIT
\par DP email: 2301516@sit.singaporetech.edu.sg
\par Course: csd3131
\par Algorithm Analysis
\date 20-11-2026
\brief
Memory experiment: MEASURED peak memory of finding ONE vertical seam.

The table in README §2.2 counts the bytes of the final data structures.
This program measures what the process actually uses, including temporary
buffers (e.g. the edge list BuildSeamGraph builds before the CSR graph)
and allocator overhead.

One run = one (method, width) pair in a FRESH process, so the peak of one run
cannot hide inside the peak of an earlier one. Use the driver:
  python tools/run_graphcut_memory.py          (writes results/memory_peak.csv + chart)

Build from the repo root:
  g++ -std=c++14 -O2 -I. tests/GraphCuttingMemory.cpp SeamCarving.cpp GraphCutting.cpp DynamicProgramming.cpp -o gc_mem `pkg-config --cflags --libs opencv4`
Single run:
  ./gc_mem <dp|gc-dinic|gc-ek> <width>
  -> prints one CSV line: method,width,height,pixels,baseline_bytes,peak_bytes,delta_bytes,seconds,seam_cost,note
     (note = "below_resolution" if delta < 1 MiB: too small to measure reliably)

Measurement: delta = peak resident memory (high-water mark) after the seam
             minus resident memory just before it. Setup (image load, resize,
             energy map) happens before the baseline, so delta is the extra
             memory needed by the seam finder alone.
  Linux  : VmRSS / VmHWM from /proc/self/status. The peak is reset just
           before the seam (write "5" to /proc/self/clear_refs), and glibc is
           told to return freed memory to the OS and to mmap every large
           block, so the seam finder cannot hide inside memory left over
           from setup.
  Windows: WorkingSetSize / PeakWorkingSetSize (link Psapi.lib). The peak
           cannot be reset there, so run one size per process (the driver does).
           Not tested on Windows.
- ReadMem()
reads the current and peak resident memory of the process
- ResetPeak()
resets the peak-memory counter (Linux only)
- main(argc, argv)
measures the extra memory one vertical seam needs and prints one CSV
line
**************************************************************************/

// ----- Headers ------------------------------------------------------- //
#include "GraphCutting.h"
#include "DynamicProgramming.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__GLIBC__)
#  include <malloc.h>
#endif
#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#  include <psapi.h>
#  pragma comment(lib, "Psapi.lib")
#endif

struct Mem { long long current = -1, peak = -1; };

/************************************************************************/
/*!
  \brief
	reads the resident memory of this process: VmRSS / VmHWM from
	/proc/self/status on Linux, WorkingSetSize / PeakWorkingSetSize on
	Windows
  \return Mem
  memory reading
  returns the current and peak resident memory in bytes, or -1 for values
  that are not available on this OS
*/
/************************************************************************/
static Mem ReadMem()
{
    Mem m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        m.current = static_cast<long long>(pmc.WorkingSetSize);
        m.peak = static_cast<long long>(pmc.PeakWorkingSetSize);
    }
#elif defined(__linux__)
    if (FILE* f = std::fopen("/proc/self/status", "r"))
    {
        char line[256];
        while (std::fgets(line, sizeof line, f))
        {
            long long kb;
            if (std::sscanf(line, "VmRSS: %lld kB", &kb) == 1) m.current = kb * 1024;
            if (std::sscanf(line, "VmHWM: %lld kB", &kb) == 1) m.peak = kb * 1024;
        }
        std::fclose(f);
    }
#endif
    return m;
}

/************************************************************************/
/*!
  \brief
	Linux only: reset the resident-memory high-water mark (VmHWM).
  \return bool
  boolean
  returns true if the peak was reset, false otherwise (always false
  outside Linux)
*/
/************************************************************************/
static bool ResetPeak()
{
#if defined(__linux__)
    if (FILE* f = std::fopen("/proc/self/clear_refs", "w"))
    {
        const bool ok = std::fputs("5", f) >= 0;
        return (std::fclose(f) == 0) && ok;
    }
#endif
    return false;
}

/************************************************************************/
/*!
  \brief
	scales the tower image to the given width, computes its energy map, then
	measures the peak resident memory and time of finding one vertical seam
	with DP, Graph Cut (Dinic) or Graph Cut (Edmonds-Karp), and prints one
	CSV line. Peaks under 1 MiB are flagged as below the measurement
	resolution.
  \param argc
  int
	The number of command-line arguments (must be 3)
  \param argv
  char**
	The arguments: the method (dp, gc-dinic or gc-ek) and the width
  \return int
  integer
  returns 0 on success, 1 if the image or the memory counters are not
  available, 2 on invalid arguments
*/
/************************************************************************/
int main(int argc, char** argv)
{
#if defined(__GLIBC__)
    mallopt(M_MMAP_THRESHOLD, 64 * 1024);   // fixed: large blocks always mmap'd (freed = unmapped)
    mallopt(M_TRIM_THRESHOLD, 64 * 1024);   // return freed heap memory to the OS promptly
#endif
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <dp|gc-dinic|gc-ek> <width>\n", argv[0]);
        return 2;
    }
    const std::string method = argv[1];
    const int w = std::atoi(argv[2]);

    const cv::Mat tower = cv::imread("images/tower.png");
    if (tower.empty()) { std::fprintf(stderr, "run from the repo root (images/tower.png)\n"); return 1; }
    const int h = static_cast<int>(std::lround(double(w) * tower.rows / tower.cols));
    cv::Mat img;
    cv::resize(tower, img, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    const cv::Mat cost = GraphCut::ComputeEnergy(img);   // the shared SeamCarving::ComputeEnergy

#if defined(__GLIBC__)
    malloc_trim(0);
#endif
    ResetPeak();
    const Mem before = ReadMem();
    if (before.current < 0) { std::fprintf(stderr, "memory counters not available on this OS\n"); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    long long seamCost = 0;
    if (method == "dp")
        seamCost = DP::SeamEnergy(cost, DP::FindVerticalSeam(cost), true);
    else if (method == "gc-dinic")
        seamCost = GraphCut::FindVerticalSeam(cost, GraphCut::MaxFlowAlgorithm::Dinic).cost;
    else if (method == "gc-ek")
        seamCost = GraphCut::FindVerticalSeam(cost, GraphCut::MaxFlowAlgorithm::EdmondsKarp).cost;
    else { std::fprintf(stderr, "unknown method %s\n", method.c_str()); return 2; }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Mem after = ReadMem();

    // Below ~1 MiB the delta is dominated by page granularity (4 KiB) and heap
    // bookkeeping, so such rows are flagged and not used for B/pixel figures.
    const long long delta = after.peak - before.current;
    std::printf("%s,%d,%d,%d,%lld,%lld,%lld,%.6f,%lld,%s\n", method.c_str(), w, h, w * h, before.current,
                after.peak, delta, secs, seamCost, delta < (1LL << 20) ? "below_resolution" : "");
    return 0;
}
