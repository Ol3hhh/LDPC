cmake_minimum_required(VERSION 3.17)

project(LDPC_Decoder LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES native)

set(EXECUTABLE_NAME ldpc_decoder)

add_executable(${EXECUTABLE_NAME}
    main.cpp
    decoder_gpu.cu
)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED YES
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

target_link_libraries(${EXECUTABLE_NAME} PRIVATE CUDA::cudart)

message(STATUS "CMake skonfigurowany dla ${EXECUTABLE_NAME}")
message(STATUS "Aby zbudować, uruchom w katalogu 'build':")
message(STATUS "make")
