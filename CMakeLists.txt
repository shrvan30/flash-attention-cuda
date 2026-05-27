cmake_minimum_required(VERSION 3.18)

project(FlashAttentionCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CXX_STANDARD 14)

add_executable(
    attention_engine

    src/main.cu
    src/naive_attention.cu
    src/tiled_attention.cu
    src/softmax_stable.cu
    src/fused_attention.cu
)

set_target_properties(
    attention_engine
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
