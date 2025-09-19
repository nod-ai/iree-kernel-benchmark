cmake_minimum_required(VERSION 3.16)
project(hipblaslt_gemm_benchmark LANGUAGES CXX)

# Set up ROCm environment variables
if(NOT DEFINED ENV{ROCM_PATH})
    set(ENV{ROCM_PATH} "/opt/rocm")
endif()

# Add ROCm to PATH
set(ENV{PATH} "$ENV{ROCM_PATH}/bin:$ENV{PATH}")

# Set up LD_LIBRARY_PATH for ROCm libraries
set(ENV{LD_LIBRARY_PATH} "$ENV{ROCM_PATH}/lib:$ENV{ROCM_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")

# Set the C++ compiler to hipcc
set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
if(NOT EXISTS ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
endif()

# Find HIP
find_package(HIP REQUIRED)

# Find hipBLASLt
find_path(HIPBLASLT_INCLUDE_DIR hipblaslt/hipblaslt.h
    HINTS ${ROCM_PATH}/include ${HIP_PATH}/include /opt/rocm/include
)

find_library(HIPBLASLT_LIBRARY hipblaslt
    HINTS ${ROCM_PATH}/lib ${HIP_PATH}/lib /opt/rocm/lib
)

if(NOT HIPBLASLT_INCLUDE_DIR OR NOT HIPBLASLT_LIBRARY)
    message(FATAL_ERROR "hipBLASLt not found!")
endif()

# Create executable
add_executable(gemm_benchmark gemm_benchmark.cpp)

# Set C++ standard
set_target_properties(gemm_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(gemm_benchmark PRIVATE 
    ${HIP_INCLUDE_DIRS}
    ${HIPBLASLT_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(gemm_benchmark 
    hip::device
    ${HIPBLASLT_LIBRARY}
)

# For gfx950, explicitly set the architecture
set(GPU_TARGETS "gfx950" CACHE STRING "GPU architectures to compile for")
set_property(TARGET gemm_benchmark PROPERTY HIP_ARCHITECTURES ${GPU_TARGETS})
