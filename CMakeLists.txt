cmake_minimum_required(VERSION 3.18)
project(fft_cuda_coursework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Подхват CUDA toolkit (обычно лежит в /opt/cuda на Arch/Manjaro)
# Если CMake не находит CUDA автоматически — скажешь, дам команду с CUDAToolkit_ROOT.
find_package(CUDAToolkit REQUIRED)

add_executable(fft_cuda main.cu)
target_link_libraries(fft_cuda PRIVATE CUDA::cufft)
