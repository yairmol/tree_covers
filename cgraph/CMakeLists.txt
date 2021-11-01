cmake_minimum_required(VERSION 3.16)
project(graph LANGUAGES C)

set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS "-pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
#set(CMAKE_CUDA_STANDARD 14)
# include(CheckLanguage)
# check_language(CUDA)

#find_package(igraph REQUIRED)
#add_executable(learncpp main.cpp Graph.h GraphAlgorithms.h Timer.h graph.c)
add_executable(
        graph main.c utils/linked_list.c utils/queue.c 
        graph.c graph_algorithms.c utils/utils.c
        series_parallel.c embeddings.c
)

add_executable(
        graph2 main2.c utils/vector.c utils/vector_queue.c graph_vec.c graph_algorithms2.c embeddings.c
)

# add_executable(gpugraph cuda/graph.cu)
# add_executable(main cuda/main2.cu)
#target_link_libraries(learncpp PUBLIC igraph::igraph)