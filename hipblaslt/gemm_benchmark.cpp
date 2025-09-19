#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = call;                                          \
        if (err != hipSuccess)                                          \
        {                                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl;  \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Helper macro for hipBLASLt error checking
#define HIPBLASLT_CHECK(call)                                                 \
    do                                                                        \
    {                                                                         \
        hipblasStatus_t err = call;                                           \
        if (err != HIPBLAS_STATUS_SUCCESS)                                    \
        {                                                                     \
            std::cerr << "hipBLASLt error at " << __FILE__ << ":" << __LINE__ \
                      << " - Error code: " << err << std::endl;               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Structure to hold benchmark parameters
struct GemmParams
{
    int M, N, K;
    bool tA, tB;
    hipDataType input_dtype;
    hipDataType output_dtype;
    hipblasComputeType_t compute_type;
    hipDataType scale_dtype; // Type for alpha/beta
};

// Helper function to get data type size in bytes
size_t getDatatypeSize(hipDataType dtype)
{
    switch (dtype)
    {
    case HIP_R_16F:
        return 2;
    case HIP_R_16BF:
        return 2;
    case HIP_R_32F:
        return 4;
    case HIP_R_64F:
        return 8;
    case HIP_R_8I:
        return 1;
    case HIP_R_32I:
        return 4;
    default:
        return 4;
    }
}

// Helper function to determine compute type and scale type based on input/output types
void getComputeAndScaleType(hipDataType input_dtype, hipDataType output_dtype,
                            hipblasComputeType_t &compute_type, hipDataType &scale_type)
{
    if (input_dtype == HIP_R_16F || input_dtype == HIP_R_16BF)
    {
        compute_type = HIPBLAS_COMPUTE_32F;
        scale_type = HIP_R_32F;
    }
    else if (input_dtype == HIP_R_32F)
    {
        compute_type = HIPBLAS_COMPUTE_32F;
        scale_type = HIP_R_32F;
    }
    else if (input_dtype == HIP_R_64F)
    {
        compute_type = HIPBLAS_COMPUTE_64F;
        scale_type = HIP_R_64F;
    }
    else if (input_dtype == HIP_R_8I)
    {
        compute_type = HIPBLAS_COMPUTE_32I;
        scale_type = HIP_R_32I;
    }
    else
    {
        compute_type = HIPBLAS_COMPUTE_32F;
        scale_type = HIP_R_32F;
    }
}

// Function to parse string to data type
hipDataType parseDatatype(const std::string &dtype_str)
{
    if (dtype_str == "fp16" || dtype_str == "f16")
        return HIP_R_16F;
    if (dtype_str == "bf16")
        return HIP_R_16BF;
    if (dtype_str == "fp32" || dtype_str == "f32")
        return HIP_R_32F;
    if (dtype_str == "fp64" || dtype_str == "f64")
        return HIP_R_64F;
    if (dtype_str == "int8" || dtype_str == "i8")
        return HIP_R_8I;
    if (dtype_str == "fp8" || dtype_str == "f8")
        return HIP_R_8F_E4M3_FNUZ;
    if (dtype_str == "int32" || dtype_str == "i32")
        return HIP_R_32I;

    std::cerr << "Unknown data type: " << dtype_str << std::endl;
    exit(1);
}

// Function to get string representation of data type
std::string datatypeToString(hipDataType dtype)
{
    switch (dtype)
    {
    case HIP_R_16F:
        return "fp16";
    case HIP_R_16BF:
        return "bf16";
    case HIP_R_32F:
        return "fp32";
    case HIP_R_64F:
        return "fp64";
    case HIP_R_8I:
        return "int8";
    case HIP_R_8F_E4M3_FNUZ:
        return "fp8";
    case HIP_R_32I:
        return "int32";
    default:
        return "unknown";
    }
}

// Function to calculate TFLOPS
double calculateTflops(int M, int N, int K, double time_us)
{
    double flops = 2.0 * M * N * K;
    return (flops / time_us) / 1e6; // TFLOPS (flops per microsecond / 1e6)
}

// Main benchmarking function
void benchmarkGemm(const GemmParams &params, int warmup_runs = 10, int benchmark_runs = 100)
{
    // Initialize hipBLASLt
    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    // Set up matrix dimensions
    int lda = params.tA ? params.K : params.M;
    int ldb = params.tB ? params.N : params.K;
    int ldc = params.M;

    // Calculate matrix sizes
    size_t sizeA = lda * (params.tA ? params.M : params.K) * getDatatypeSize(params.input_dtype);
    size_t sizeB = ldb * (params.tB ? params.K : params.N) * getDatatypeSize(params.input_dtype);
    size_t sizeC = ldc * params.N * getDatatypeSize(params.output_dtype);

    // Allocate device memory
    void *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeA));
    HIP_CHECK(hipMalloc(&d_B, sizeB));
    HIP_CHECK(hipMalloc(&d_C, sizeC));

    // Initialize matrices with random data (simplified - just filling with pattern)
    HIP_CHECK(hipMemset(d_A, 1, sizeA));
    HIP_CHECK(hipMemset(d_B, 2, sizeB));
    HIP_CHECK(hipMemset(d_C, 0, sizeC));

    // Create matrix descriptors
    hipblasLtMatrixLayout_t matA, matB, matC;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matA, params.input_dtype,
                                                params.tA ? params.K : params.M, params.tA ? params.M : params.K, lda));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matB, params.input_dtype,
                                                params.tB ? params.N : params.K, params.tB ? params.K : params.N, ldb));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&matC, params.output_dtype, params.M, params.N, ldc));

    // Create matrix multiply descriptor with compute type and scale type
    hipblasLtMatmulDesc_t matmulDesc;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc, params.compute_type, params.scale_dtype));

    // Set transposition
    hipblasOperation_t transA = params.tA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t transB = params.tB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &transA, sizeof(transA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &transB, sizeof(transB)));

    // Set up preference for algorithm selection
    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 32 * 1024 * 1024; // 32 MB workspace
    void *workspace;
    HIP_CHECK(hipMalloc(&workspace, workspaceSize));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Find best algorithm
    const int maxAlgos = 32;
    hipblasLtMatmulHeuristicResult_t heuristicResult[maxAlgos];
    int returnedAlgoCount = 0;

    // Set alpha and beta based on scale type
    void *alpha;
    void *beta;
    float alpha_f32 = 1.0f, beta_f32 = 0.0f;
    double alpha_f64 = 1.0, beta_f64 = 0.0;
    int32_t alpha_i32 = 1, beta_i32 = 0;

    if (params.scale_dtype == HIP_R_32F)
    {
        alpha = &alpha_f32;
        beta = &beta_f32;
    }
    else if (params.scale_dtype == HIP_R_64F)
    {
        alpha = &alpha_f64;
        beta = &beta_f64;
    }
    else if (params.scale_dtype == HIP_R_32I)
    {
        alpha = &alpha_i32;
        beta = &beta_i32;
    }
    else
    {
        // Default to float
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, matA, matB, matC, matC,
                                                    pref, maxAlgos, heuristicResult, &returnedAlgoCount));

    if (returnedAlgoCount == 0)
    {
        std::cerr << "No suitable algorithm found!" << std::endl;
        exit(1);
    }

    // Use the best algorithm
    hipblasLtMatmulAlgo_t algo = heuristicResult[0].algo;

    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup runs
    for (int i = 0; i < warmup_runs; i++)
    {
        HIPBLASLT_CHECK(hipblasLtMatmul(handle, matmulDesc,
                                        alpha, d_A, matA, d_B, matB, beta, d_C, matC, d_C, matC,
                                        &algo, workspace, workspaceSize, 0));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark runs
    std::vector<float> times_us; // Store times in microseconds
    for (int i = 0; i < benchmark_runs; i++)
    {
        HIP_CHECK(hipEventRecord(start));

        HIPBLASLT_CHECK(hipblasLtMatmul(handle, matmulDesc,
                                        alpha, d_A, matA, d_B, matB, beta, d_C, matC, d_C, matC,
                                        &algo, workspace, workspaceSize, 0));

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float milliseconds = 0;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        times_us.push_back(milliseconds * 1000.0f); // Convert to microseconds
    }

    // Calculate statistics
    std::sort(times_us.begin(), times_us.end());
    float min_time_us = times_us[0];
    float max_time_us = times_us[times_us.size() - 1];
    float median_time_us = times_us[times_us.size() / 2];
    float avg_time_us = 0;
    for (float t : times_us)
        avg_time_us += t;
    avg_time_us /= times_us.size();

    // Calculate TFLOPS
    double peak_tflops = calculateTflops(params.M, params.N, params.K, min_time_us);
    double avg_tflops = calculateTflops(params.M, params.N, params.K, avg_time_us);

    // Print results
    std::cout << "\n========== Benchmark Results ==========" << std::endl;
    std::cout << "Matrix dimensions: M=" << params.M << ", N=" << params.N << ", K=" << params.K << std::endl;
    std::cout << "Transpose A: " << (params.tA ? "Yes" : "No") << ", Transpose B: " << (params.tB ? "Yes" : "No") << std::endl;
    std::cout << "Input dtype: " << datatypeToString(params.input_dtype) << std::endl;
    std::cout << "Output dtype: " << datatypeToString(params.output_dtype) << std::endl;
    std::cout << "Compute type: ";
    switch (params.compute_type)
    {
    case HIPBLAS_COMPUTE_16F:
        std::cout << "FP16";
        break;
    case HIPBLAS_COMPUTE_32F:
        std::cout << "FP32";
        break;
    case HIPBLAS_COMPUTE_64F:
        std::cout << "FP64";
        break;
    case HIPBLAS_COMPUTE_32I:
        std::cout << "INT32";
        break;
    default:
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Scale type: " << datatypeToString(params.scale_dtype) << std::endl;
    std::cout << "Number of runs: " << benchmark_runs << std::endl;
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "  Min time: " << std::fixed << std::setprecision(2) << min_time_us << " μs" << std::endl;
    std::cout << "  Max time: " << std::fixed << std::setprecision(2) << max_time_us << " μs" << std::endl;
    std::cout << "  Median time: " << std::fixed << std::setprecision(2) << median_time_us << " μs" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avg_time_us << " μs" << std::endl;
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Peak TFLOPS: " << std::fixed << std::setprecision(3) << peak_tflops << std::endl;
    std::cout << "  Average TFLOPS: " << std::fixed << std::setprecision(3) << avg_tflops << std::endl;
    std::cout << "======================================\n"
              << std::endl;

    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(workspace));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(matA));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(matB));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(matC));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmulDesc));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    HIPBLASLT_CHECK(hipblasLtDestroy(handle));
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " -M <m> -N <n> -K <k> [-tA] [-tB] "
              << "-input_dtype <dtype> -output_dtype <dtype> [-warmup <n>] [-runs <n>]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -M <m>              : Number of rows in matrix C (required)" << std::endl;
    std::cout << "  -N <n>              : Number of columns in matrix C (required)" << std::endl;
    std::cout << "  -K <k>              : Shared dimension (required)" << std::endl;
    std::cout << "  -tA                 : Transpose matrix A (optional, default: false)" << std::endl;
    std::cout << "  -tB                 : Transpose matrix B (optional, default: false)" << std::endl;
    std::cout << "  -input_dtype <type> : Input data type (required)" << std::endl;
    std::cout << "  -output_dtype <type>: Output data type (required)" << std::endl;
    std::cout << "  -warmup <n>         : Number of warmup runs (optional, default: 10)" << std::endl;
    std::cout << "  -runs <n>           : Number of benchmark runs (optional, default: 100)" << std::endl;
    std::cout << "\nSupported data types: fp16, bf16, fp32, fp64, int8, int32" << std::endl;
    std::cout << "\nExample: " << prog << " -M 1024 -N 1024 -K 1024 -input_dtype fp16 -output_dtype fp32" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 11)
    {
        printUsage(argv[0]);
        return 1;
    }

    // Parse command line arguments
    GemmParams params;
    params.tA = false;
    params.tB = false;
    int warmup_runs = 10;
    int benchmark_runs = 100;
    bool has_M = false, has_N = false, has_K = false;
    bool has_input_dtype = false, has_output_dtype = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc)
        {
            params.M = std::stoi(argv[++i]);
            has_M = true;
        }
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
        {
            params.N = std::stoi(argv[++i]);
            has_N = true;
        }
        else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc)
        {
            params.K = std::stoi(argv[++i]);
            has_K = true;
        }
        else if (strcmp(argv[i], "-tA") == 0)
        {
            params.tA = true;
        }
        else if (strcmp(argv[i], "-tB") == 0)
        {
            params.tB = true;
        }
        else if (strcmp(argv[i], "-input_dtype") == 0 && i + 1 < argc)
        {
            params.input_dtype = parseDatatype(argv[++i]);
            has_input_dtype = true;
        }
        else if (strcmp(argv[i], "-output_dtype") == 0 && i + 1 < argc)
        {
            params.output_dtype = parseDatatype(argv[++i]);
            has_output_dtype = true;
        }
        else if (strcmp(argv[i], "-warmup") == 0 && i + 1 < argc)
        {
            warmup_runs = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc)
        {
            benchmark_runs = std::stoi(argv[++i]);
        }
    }

    // Validate required parameters
    if (!has_M || !has_N || !has_K || !has_input_dtype || !has_output_dtype)
    {
        std::cerr << "Error: Missing required parameters!" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Set compute type and scale type based on input/output types
    getComputeAndScaleType(params.input_dtype, params.output_dtype,
                           params.compute_type, params.scale_dtype);

    // Run benchmark
    benchmarkGemm(params, warmup_runs, benchmark_runs);

    return 0;
}
