cmake_minimum_required(VERSION 3.16)
project(perf)

set(CMAKE_CXX_STANDARD 14)

add_executable(perf main.cpp performance.cpp)