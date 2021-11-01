cmake_minimum_required(VERSION 3.16)
project(gpugraph LANGUAGES CUDA)

include(CheckLanguage)
check_language(CUDA)

add_executable(gpugraph main3.cu graph.cu vector.cu queue_vec.cu tree_covers.cu shortest_paths.cu)

set_target_properties(gpugraph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(gpugraph PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)