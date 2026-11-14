# Seam carving: Dynamic Programming, Greedy and Graph Cut.
#
# Visual Studio 2022: File > Open > Folder... on the repo root, pick the
# "x64 Release" configuration, then choose a target (e.g. seam_carve.exe) from
# the start-up dropdown. Programs run from the repo root (see .vs/launch.vs.json
# or set the working directory yourself) so images/ and docs/ are found.
#
# Command line (any shell; picks 64-bit MSVC itself):
#   cmake --preset vs2022
#   cmake --build --preset vs2022-release
#   -> out/build/vs2022/Release/*.exe
# The Ninja presets (x64-release/x64-debug) only work from an
# "x64 Native Tools Command Prompt for VS 2022".
# Linux / macOS: cmake --preset unix-release && cmake --build --preset unix-release
#
# OpenCV is installed automatically: vcpkg.json lists it, and the first
# configure downloads and builds it with vcpkg (takes a while once, then cached
# in vcpkg's binary cache). Uses $VCPKG_ROOT if set (Visual Studio ships one),
# otherwise clones vcpkg into out/vcpkg. Needs git on PATH.
# To use an OpenCV you already have instead, configure with
#   -DSEAM_USE_VCPKG=OFF -DOpenCV_DIR=<folder containing OpenCVConfig.cmake>
cmake_minimum_required(VERSION 3.21)   # 3.21+ for $<TARGET_RUNTIME_DLLS>

option(SEAM_USE_VCPKG "Install OpenCV automatically with vcpkg (see vcpkg.json)" ON)

# The vcpkg toolchain has to be set before project().
if(SEAM_USE_VCPKG AND NOT CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(SEAM_VCPKG_ROOT "$ENV{VCPKG_ROOT}")
    else()
        set(SEAM_VCPKG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/out/vcpkg")
        if(NOT EXISTS "${SEAM_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
            find_program(SEAM_GIT git REQUIRED)
            message(STATUS "Cloning vcpkg into ${SEAM_VCPKG_ROOT} (one-time) ...")
            execute_process(
                COMMAND "${SEAM_GIT}" clone https://github.com/microsoft/vcpkg.git "${SEAM_VCPKG_ROOT}"
                RESULT_VARIABLE seam_clone_result)
            if(NOT seam_clone_result EQUAL 0)
                message(FATAL_ERROR "Cloning vcpkg failed. Install vcpkg and set VCPKG_ROOT, "
                                    "or use -DSEAM_USE_VCPKG=OFF with your own OpenCV.")
            endif()
        endif()
    endif()
    message(STATUS "Using vcpkg from ${SEAM_VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "${SEAM_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
endif()

project(SeamCarvingAlgorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT SEAM_USE_VCPKG)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(FATAL_ERROR
            "32-bit compiler detected, but the OpenCV Windows prebuilt is 64-bit only.\n"
            "Use 'cmake --preset vs2022' (works from any shell), or run the x64-* presets "
            "from an 'x64 Native Tools Command Prompt for VS 2022'. "
            "Delete out/build/<preset> after switching.")
    endif()
    if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    endif()
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

if(MSVC)
    add_compile_options(/W3 /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)   # fopen in the benchmarks
endif()

# The three algorithms, shared by every program below.
add_library(seam_carving STATIC
    DynamicProgramming.cpp DynamicProgramming.h
    GreedyAlgorithm.cpp    GreedyAlgorithm.h
    GraphCutting.cpp       GraphCutting.h)
target_include_directories(seam_carving PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(seam_carving PUBLIC ${OpenCV_LIBS})

function(seam_program name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE seam_carving)
    # Run from the repo root when started from a Visual Studio solution.
    set_target_properties(${name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # Copy the OpenCV DLLs next to the .exe so it starts even without PATH set.
    if(WIN32)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${name}> $<TARGET_FILE_DIR:${name}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()

seam_program(seam_carve   main.cpp)
seam_program(greedy_bench tests/GreedyBenchmark.cpp)
seam_program(gc_bench     tests/GraphCuttingBenchmark.cpp)
seam_program(gc_tests     tests/GraphCuttingTests.cpp)
seam_program(dp_tests     tests/DynamicProgrammingTests.cpp)
