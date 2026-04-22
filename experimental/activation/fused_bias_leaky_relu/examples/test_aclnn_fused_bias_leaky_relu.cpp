/**
 * @file test_aclnn_fused_bias_leaky_relu.cpp
 * @brief ACLNN calling example for FusedBiasLeakyRelu custom operator.
 *
 * Demonstrates the two-phase ACLNN interface:
 *   Phase 1: aclnnFusedBiasLeakyReluGetWorkspaceSize - compute workspace size, create executor
 *   Phase 2: aclnnFusedBiasLeakyRelu - execute computation on NPU
 *
 * Formula:
 *   t = x + bias
 *   y = (t >= 0) ? t * scale : t * negativeSlope * scale
 *
 * Build and run:
 *   bash run.sh --eager
 *
 * Prerequisites:
 *   - CANN toolkit installed (set ASCEND_HOME_PATH)
 *   - Operator package installed (bash build.sh --soc=ascend910b && install .run)
 *   - NPU device available
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_fused_bias_leaky_relu.h"

// ============================================================================
// Macros
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_ACL(expr)                                          \
    do {                                                         \
        auto _ret = (expr);                                      \
        if (_ret != ACL_SUCCESS) {                               \
            LOG_PRINT("[ERROR] %s failed, ret=%d", #expr, _ret); \
            return _ret;                                         \
        }                                                        \
    } while (0)

// ============================================================================
// Helper: compute shape total size
// ============================================================================
static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// ============================================================================
// Helper: compute contiguous strides
// ============================================================================
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ============================================================================
// Helper: create aclTensor from host data
// ============================================================================
template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

// ============================================================================
// CPU golden reference for verification
// ============================================================================
static void ComputeGolden(const float* x, const float* bias, float* output,
                          size_t size, double negativeSlope, double scale)
{
    for (size_t i = 0; i < size; ++i) {
        double t = static_cast<double>(x[i]) + static_cast<double>(bias[i]);
        if (t >= 0.0) {
            output[i] = static_cast<float>(t * scale);
        } else {
            output[i] = static_cast<float>(t * negativeSlope * scale);
        }
    }
}

// ============================================================================
// Precision comparison
// ============================================================================
static bool CompareResults(const float* golden, const float* actual, size_t size)
{
    const double atol = 1e-4;
    const double rtol = 1e-4;
    int mismatch = 0;

    for (size_t i = 0; i < size; ++i) {
        double expected_val = static_cast<double>(golden[i]);
        double actual_val = static_cast<double>(actual[i]);
        double diff = std::abs(actual_val - expected_val);
        double denom = std::max(std::abs(expected_val), atol);
        double rel_error = diff / denom;

        if (rel_error >= rtol) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  mismatch [%zu]: expected=%.6f, actual=%.6f, rel_error=%.6e",
                          i, expected_val, actual_val, rel_error);
            }
        }
    }

    double fail_ratio = (size > 0) ? static_cast<double>(mismatch) / static_cast<double>(size) : 0.0;
    bool pass = fail_ratio < 0.0001; // 0.01%

    LOG_PRINT("  %zu elements, %d mismatch (%.4f%%) => %s",
              size, mismatch, fail_ratio * 100.0, pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================================
// Main: ACLNN calling example
// ============================================================================
int main()
{
    LOG_PRINT("========================================");
    LOG_PRINT("FusedBiasLeakyRelu ACLNN Calling Example");
    LOG_PRINT("========================================");

    // ---- Step 1: Initialize ACL ----
    LOG_PRINT("\n[Step 1] Initialize ACL...");
    int32_t deviceId = 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ---- Step 2: Prepare input data ----
    LOG_PRINT("\n[Step 2] Prepare input data...");

    // Example: a [2, 4] tensor with mixed positive/negative values
    std::vector<int64_t> shape = {2, 4};
    int64_t totalSize = GetShapeSize(shape);

    // Input x: mixed positive and negative values
    std::vector<float> xHost = {1.0f, -1.0f, 2.0f, -2.0f,
                                 0.5f, -0.5f, 3.0f, -3.0f};
    // Input bias: constant offset
    std::vector<float> biasHost = {0.5f, 0.5f, -0.5f, -0.5f,
                                    0.0f, 0.0f,  1.0f,  1.0f};

    // Attributes (double type, as required by ACLNN interface)
    double negativeSlope = 0.2;              // LeakyReLU negative slope
    double scale = 1.414213562373;           // sqrt(2) scaling factor

    LOG_PRINT("  Shape: [%ld, %ld], Total elements: %ld",
              (long)shape[0], (long)shape[1], (long)totalSize);
    LOG_PRINT("  negativeSlope = %.6f", negativeSlope);
    LOG_PRINT("  scale = %.12f", scale);

    // ---- Step 3: Create device tensors ----
    LOG_PRINT("\n[Step 3] Create device tensors...");

    void* xDev = nullptr;
    void* biasDev = nullptr;
    void* outDev = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* biasTensor = nullptr;
    aclTensor* outTensor = nullptr;

    CHECK_RET(CreateAclTensor(xHost, shape, &xDev, ACL_FLOAT, &xTensor) == ACL_SUCCESS,
              return -1);
    CHECK_RET(CreateAclTensor(biasHost, shape, &biasDev, ACL_FLOAT, &biasTensor) == ACL_SUCCESS,
              return -1);

    // Allocate output tensor (pre-allocated with zeros)
    std::vector<float> outHost(totalSize, 0.0f);
    CHECK_RET(CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outTensor) == ACL_SUCCESS,
              return -1);

    // ---- Step 4: Call ACLNN Phase 1 - GetWorkspaceSize ----
    LOG_PRINT("\n[Step 4] Call aclnnFusedBiasLeakyReluGetWorkspaceSize...");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnFusedBiasLeakyReluGetWorkspaceSize(
        xTensor, biasTensor, negativeSlope, scale,
        outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("  GetWorkspaceSize failed: %d", ret); return -1);

    LOG_PRINT("  workspaceSize = %lu bytes", workspaceSize);

    // Allocate workspace on device (if needed)
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // ---- Step 5: Call ACLNN Phase 2 - Execute ----
    LOG_PRINT("\n[Step 5] Call aclnnFusedBiasLeakyRelu (execute on NPU)...");

    ret = aclnnFusedBiasLeakyRelu(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("  Execute failed: %d", ret); return -1);

    // Wait for completion
    CHECK_ACL(aclrtSynchronizeStream(stream));
    LOG_PRINT("  Execution completed.");

    // ---- Step 6: Copy result back to host and verify ----
    LOG_PRINT("\n[Step 6] Verify results...");

    std::vector<float> npuOutput(totalSize);
    CHECK_ACL(aclrtMemcpy(npuOutput.data(), totalSize * sizeof(float),
                           outDev, totalSize * sizeof(float),
                           ACL_MEMCPY_DEVICE_TO_HOST));

    // Compute CPU golden reference
    std::vector<float> golden(totalSize);
    ComputeGolden(xHost.data(), biasHost.data(), golden.data(),
                  totalSize, negativeSlope, scale);

    // Print results
    LOG_PRINT("\n  Results comparison:");
    LOG_PRINT("  %-6s %-10s %-10s %-10s %-12s %-12s", "Index", "x", "bias", "t=x+bias", "Golden", "NPU Output");
    for (int64_t i = 0; i < totalSize; ++i) {
        float t = xHost[i] + biasHost[i];
        LOG_PRINT("  %-6ld %-10.4f %-10.4f %-10.4f %-12.6f %-12.6f",
                  (long)i, xHost[i], biasHost[i], t, golden[i], npuOutput[i]);
    }

    // Precision comparison
    LOG_PRINT("\n  Precision check (float32, rtol=1e-4, fail_ratio < 0.01%%):");
    bool passed = CompareResults(golden.data(), npuOutput.data(), totalSize);

    // ---- Step 7: Cleanup ----
    LOG_PRINT("\n[Step 7] Cleanup...");

    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(biasTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(xDev);
    aclrtFree(biasDev);
    aclrtFree(outDev);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // ---- Result ----
    LOG_PRINT("\n========================================");
    LOG_PRINT("Result: %s", passed ? "PASS" : "FAIL");
    LOG_PRINT("========================================\n");

    return passed ? 0 : 1;
}
