cmake_minimum_required(VERSION 3.16)
project(graph)

set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS "-pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# set(CMAKE_CUDA_STANDARD 14)
# include(CheckLanguage)
# check_language(CUDA)

# add_executable(
#     graph main.c
#     graph/graph.c graph/graph_algorithms.c graph/series_parallel.c
#     utils/utils.c utils/linked_list.c utils/queue.c 
#     embeddings/embeddings.c
# )

add_executable(
    graph2 main2.c
    graph/graph_vec.c graph/graph_algorithms2.c
    utils/vector.c utils/vector_queue.c
    embeddings/embeddings.c
)

# add_executable(
#     gpugraph cuda/main.cu 
#     cuda/linked_list.cu
#     cuda/queue.cu
#     cuda/graph.cu
# )


add_executable(main3 graph/graph_vec.c utils/vector.c main3.cpp)