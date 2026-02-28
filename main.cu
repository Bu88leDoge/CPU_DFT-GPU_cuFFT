#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

//check-up CUDA
static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static void checkCufft(cufftResult r, const char* msg) {
    if (r != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << msg << " (code " << r << ")\n";
        std::exit(1);
    }
}

// Простая DFT O(N^2) для проверки корректности (не быстрая, но честная)
static std::vector<std::complex<float>> dft_cpu(const std::vector<std::complex<float>>& x) {
    const int N = (int)x.size();
    std::vector<std::complex<float>> X(N);
    const float twoPi = 2.0f * std::acos(-1.0f);

    for (int k = 0; k < N; k++) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < N; n++) {
            float angle = -twoPi * k * n / N;
            std::complex<float> w(std::cos(angle), std::sin(angle));
            sum += x[n] * w;
        }
        X[k] = sum;
    }
    return X;
}

// Безопасная ошибка: если увидели NaN/Inf — сразу сигнализируем
static float max_abs_error(const std::vector<std::complex<float>>& a,
                           const std::vector<std::complex<float>>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float da = std::abs(a[i] - b[i]);
        if (!std::isfinite(da)) return std::numeric_limits<float>::infinity();
        if (da > m) m = da;
    }
    return m;
}

int main() {
    const int N = 16384;
    std::cout << "N = " << N << "\n";

    // Генерим входные комплексные данные на CPU (host)
    std::vector<std::complex<float>> h_in(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N; i++) {
        h_in[i] = {dist(rng), dist(rng)};
    }

    std::cout << "\n=== INPUT SIGNAL x[n] ===\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "x[" << i << "] = "
                  << h_in[i].real() << " + "
                  << h_in[i].imag() << "i\n";
    }
    std::cout << "...\n";
    for (int i = N - 5; i < N; i++) {
        std::cout << "x[" << i << "] = "
                  << h_in[i].real() << " + "
                  << h_in[i].imag() << "i\n";
    }

    // ===== CPU: DFT (для проверки правильности) =====
    auto t0 = std::chrono::high_resolution_clock::now();
    auto h_cpu = dft_cpu(h_in);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "\nCPU DFT time: " << cpu_ms << " ms\n";

    // ===== GPU: cuFFT =====
    // Делаем out-of-place FFT: d_in -> d_work.
    // Это убирает лишние D2D-копии перед каждым прогоном.

    cufftComplex* d_in = nullptr;
    cufftComplex* d_work = nullptr;

    checkCuda(cudaMalloc((void**)&d_in,   sizeof(cufftComplex) * N), "cudaMalloc d_in");
    checkCuda(cudaMalloc((void**)&d_work, sizeof(cufftComplex) * N), "cudaMalloc d_work");

    // H2D вход один раз
    checkCuda(cudaMemcpy(d_in, h_in.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D (d_in)");

    cufftHandle plan;
    checkCufft(cufftPlan1d(&plan, N, CUFFT_C2C, 1), "cufftPlan1d");

    // Прогрев: 1 FFT
    checkCufft(cufftExecC2C(plan, d_in, d_work, CUFFT_FORWARD), "cufftExecC2C warmup");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    // Замер 1: чистое время FFT (без H2D/D2H)
    const int ITERS = 50;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < ITERS; i++) {
        checkCufft(cufftExecC2C(plan, d_in, d_work, CUFFT_FORWARD), "cufftExecC2C");
    }
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    std::vector<std::complex<float>> h_gpu(N);
    float gpu_fft_ms_total = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_fft_ms_total, start, stop), "cudaEventElapsedTime");
    float gpu_fft_ms = gpu_fft_ms_total / ITERS;
    std::cout << "GPU cuFFT time (avg, FFT only): " << gpu_fft_ms << " ms\n";

    // Замер 2: полная стоимость одного прогона (H2D + FFT + D2H)
    auto pipe_t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        checkCuda(cudaMemcpy(d_in, h_in.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D (pipeline)");
        checkCufft(cufftExecC2C(plan, d_in, d_work, CUFFT_FORWARD), "cufftExecC2C pipeline");
        checkCuda(cudaMemcpy(h_gpu.data(), d_work, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H (pipeline)");
    }
    auto pipe_t1 = std::chrono::high_resolution_clock::now();
    double gpu_pipeline_ms_total =
        std::chrono::duration<double, std::milli>(pipe_t1 - pipe_t0).count();
    double gpu_pipeline_ms = gpu_pipeline_ms_total / ITERS;
    std::cout << "GPU pipeline time (avg, H2D+FFT+D2H): " << gpu_pipeline_ms << " ms\n";

    // ===== Проверка корректности =====
    float err = max_abs_error(h_cpu, h_gpu);
    std::cout << "Max abs error CPU vs GPU: " << err << "\n";

    std::cout << "\n=== FFT OUTPUT X[k] (GPU) ===\n";
    for (int k = 0; k < 5; k++) {
        float re = h_gpu[k].real();
        float im = h_gpu[k].imag();
        float mag = std::sqrt(re * re + im * im);

        std::cout << "X[" << k << "] = "
                  << re << " + " << im << "i"
                  << " |X|=" << mag << "\n";
    }

    std::cout << "...\n";
    for (int k = N - 5; k < N; k++) {
        float re = h_gpu[k].real();
        float im = h_gpu[k].imag();
        float mag = std::sqrt(re * re + im * im);

        std::cout << "X[" << k << "] = "
                  << re << " + " << im << "i"
                  << " |X|=" << mag << "\n";
    }

    // ===== CUDA DEVICE INFO =====
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int driverVersion = 0;
    int runtimeVersion = 0;
    checkCuda(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion");
    checkCuda(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion");

    auto formatVersion = [](int v) {
        std::ostringstream oss;
        oss << (v / 1000) << "." << (v % 1000) / 10;
        return oss.str();
    };

    auto printRow = [](const std::string& key, const std::string& value) {
        std::cout << "  " << std::left << std::setw(26) << key << " : " << value << "\n";
    };

    std::cout << "\n=== CUDA DEVICE INFO ===\n";
    std::cout << "-----------------------------------------------\n";
    printRow("Using device", std::to_string(device) + " / " + std::to_string(deviceCount - 1));
    printRow("Name", prop.name);
    printRow("Compute capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    printRow("CUDA driver/runtime", formatVersion(driverVersion) + " / " + formatVersion(runtimeVersion));
    printRow("Global memory", std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB");
    printRow("Memory bus width", std::to_string(prop.memoryBusWidth) + " bit");
    printRow("L2 cache size", std::to_string(prop.l2CacheSize / 1024) + " KB");
    printRow("SM count", std::to_string(prop.multiProcessorCount));
    printRow("Warp size", std::to_string(prop.warpSize));
    printRow("Max threads per block", std::to_string(prop.maxThreadsPerBlock));
    printRow("Max threads dim",
             "[" + std::to_string(prop.maxThreadsDim[0]) + ", "
                 + std::to_string(prop.maxThreadsDim[1]) + ", "
                 + std::to_string(prop.maxThreadsDim[2]) + "]");
    printRow("Max grid size",
             "[" + std::to_string(prop.maxGridSize[0]) + ", "
                 + std::to_string(prop.maxGridSize[1]) + ", "
                 + std::to_string(prop.maxGridSize[2]) + "]");
    printRow("Shared memory per block", std::to_string(prop.sharedMemPerBlock / 1024) + " KB");
    printRow("Registers per block", std::to_string(prop.regsPerBlock));
    std::cout << "-----------------------------------------------\n\n";

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_work);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
