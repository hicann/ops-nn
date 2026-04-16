/**
 * @file test_aclnn_fast_gelu_v2.cpp
 * @brief ACLNN two-phase invocation example for FastGeluV2 custom operator.
 *
 * This example demonstrates the standard ACLNN two-phase calling pattern:
 *   Phase 1: aclnnFastGeluV2GetWorkspaceSize  -- compute required workspace
 *   Phase 2: aclnnFastGeluV2                   -- execute the operator
 *
 * Prerequisites:
 *   - CANN Toolkit installed (source set_env.sh)
 *   - FastGeluV2 custom operator package installed (build.sh --soc=ascend950, then run .run file)
 *   - NPU device available
 *
 * Build & run:
 *   cd examples && bash run.sh --eager
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn_fast_gelu_v2.h"

// ============================================================================
// Macros
// ============================================================================
#define CHECK_ACL(expr)                                                 \
    do {                                                                \
        auto _ret = (expr);                                             \
        if (_ret != ACL_SUCCESS) {                                      \
            std::cerr << "[ERROR] " << #expr << " failed: " << _ret     \
                      << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                     \
            return 1;                                                   \
        }                                                               \
    } while (0)

// ============================================================================
// Helper: compute strides from shape (row-major contiguous)
// ============================================================================
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ============================================================================
// Helper: create aclTensor from host data
// ============================================================================
static int CreateAclTensor(const void* hostData, size_t dataBytes,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor) {
    auto ret = aclrtMalloc(deviceAddr, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc failed: " << ret << std::endl;
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, dataBytes, hostData, dataBytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy H2D failed: " << ret << std::endl;
        aclrtFree(*deviceAddr);
        return ret;
    }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        std::cerr << "[ERROR] aclCreateTensor returned nullptr" << std::endl;
        aclrtFree(*deviceAddr);
        return -1;
    }
    return ACL_SUCCESS;
}

// ============================================================================
// CPU golden: FastGeluV2(x) using float32 precision
// ============================================================================
static void ComputeGolden(const float* x, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float xi = x[i];
        float eps = 1e-12f;
        float x_eps = xi + eps;
        float abs_x_eps = std::fabs(x_eps);
        float sgn_x = x_eps / abs_x_eps;
        float abs_x = std::fabs(xi);
        float scaled_x = 0.7071f * abs_x;
        float clipped_x = std::min(scaled_x, 1.769f);
        float diff = clipped_x - 1.769f;
        float sq = diff * diff;
        float inner = -0.1444f * sq + 0.5f;
        float bracket = sgn_x * inner + 0.5f;
        output[i] = xi * bracket;
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FastGeluV2 ACLNN Example" << std::endl;
    std::cout << "========================================" << std::endl;

    // -----------------------------------------------------------------------
    // Step 1: Initialize ACL runtime
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 1] Initializing ACL..." << std::endl;
    CHECK_ACL(aclInit(nullptr));

    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    std::cout << "  ACL initialized, device=" << deviceId << std::endl;

    // -----------------------------------------------------------------------
    // Step 2: Prepare input data on host
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 2] Preparing input data..." << std::endl;

    std::vector<int64_t> shape = {2, 8};  // 2D tensor, 16 elements total
    int64_t totalElements = 1;
    for (auto d : shape) totalElements *= d;

    // Generate test data: linearly spaced from -5.0 to 5.0
    std::vector<float> hostInput(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
        hostInput[i] = -5.0f + 10.0f * static_cast<float>(i) /
                       static_cast<float>(totalElements - 1);
    }

    size_t dataBytes = totalElements * sizeof(float);

    std::cout << "  Shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    std::cout << "  Elements: " << totalElements << std::endl;
    std::cout << "  Dtype: float32" << std::endl;
    std::cout << "  Input range: [" << hostInput.front() << ", "
              << hostInput.back() << "]" << std::endl;

    // -----------------------------------------------------------------------
    // Step 3: Create aclTensors (allocate device memory, copy H2D)
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 3] Creating aclTensors..." << std::endl;

    void* xDevAddr = nullptr;
    aclTensor* xTensor = nullptr;
    if (CreateAclTensor(hostInput.data(), dataBytes, shape,
                        &xDevAddr, ACL_FLOAT, &xTensor) != ACL_SUCCESS) {
        std::cerr << "  Failed to create input tensor" << std::endl;
        return 1;
    }

    // Allocate output tensor (initialized to zero)
    std::vector<float> hostOutput(totalElements, 0.0f);
    void* outDevAddr = nullptr;
    aclTensor* outTensor = nullptr;
    if (CreateAclTensor(hostOutput.data(), dataBytes, shape,
                        &outDevAddr, ACL_FLOAT, &outTensor) != ACL_SUCCESS) {
        std::cerr << "  Failed to create output tensor" << std::endl;
        aclDestroyTensor(xTensor);
        aclrtFree(xDevAddr);
        return 1;
    }
    std::cout << "  Input and output tensors created on device" << std::endl;

    // -----------------------------------------------------------------------
    // Step 4: Phase 1 -- GetWorkspaceSize
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 4] Calling aclnnFastGeluV2GetWorkspaceSize..." << std::endl;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    CHECK_ACL(aclnnFastGeluV2GetWorkspaceSize(xTensor, outTensor,
                                              &workspaceSize, &executor));
    std::cout << "  workspaceSize = " << workspaceSize << " bytes" << std::endl;

    // -----------------------------------------------------------------------
    // Step 5: Allocate workspace on device
    // -----------------------------------------------------------------------
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        std::cout << "\n[Step 5] Allocating workspace..." << std::endl;
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
        std::cout << "  Workspace allocated: " << workspaceSize
                  << " bytes" << std::endl;
    } else {
        std::cout << "\n[Step 5] No workspace needed (size=0)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 6: Phase 2 -- Execute the operator
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 6] Calling aclnnFastGeluV2..." << std::endl;
    CHECK_ACL(aclnnFastGeluV2(workspace, workspaceSize, executor, stream));

    // Synchronize to ensure execution completes
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::cout << "  Operator execution completed" << std::endl;

    // -----------------------------------------------------------------------
    // Step 7: Copy result from device to host
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 7] Copying results D2H..." << std::endl;
    CHECK_ACL(aclrtMemcpy(hostOutput.data(), dataBytes, outDevAddr, dataBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    // -----------------------------------------------------------------------
    // Step 8: Compare with CPU golden
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 8] Verifying results against CPU golden..." << std::endl;

    std::vector<float> golden(totalElements);
    ComputeGolden(hostInput.data(), golden.data(), totalElements);

    // Print first few results
    int printCount = std::min(static_cast<int64_t>(8), totalElements);
    std::cout << "\n  Index | Input     | Golden    | NPU Output | Match" << std::endl;
    std::cout << "  ------+-----------+-----------+------------+------" << std::endl;

    double maxRelErr = 0.0;
    double sumRelErr = 0.0;
    const double eps = 1e-7;

    for (int64_t i = 0; i < totalElements; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(hostOutput[i]);
        double relErr = std::abs(a - g) / (std::abs(g) + eps);
        if (relErr > maxRelErr) maxRelErr = relErr;
        sumRelErr += relErr;

        if (i < printCount) {
            const char* matchStr = (relErr < 1.22e-4) ? "OK" : "MISMATCH";
            printf("  %5ld | %9.4f | %9.6f | %10.6f | %s\n",
                   (long)i, hostInput[i], golden[i], hostOutput[i], matchStr);
        }
    }

    double meanRelErr = sumRelErr / static_cast<double>(totalElements);
    double threshold = 1.22e-4;  // 2^-13 for float32

    std::cout << "\n  MERE (mean relative error) = " << meanRelErr << std::endl;
    std::cout << "  MARE (max relative error)  = " << maxRelErr << std::endl;
    std::cout << "  Threshold                  = " << threshold << std::endl;

    bool pass = (meanRelErr < threshold) && (maxRelErr < 10.0 * threshold);

    // -----------------------------------------------------------------------
    // Step 9: Cleanup
    // -----------------------------------------------------------------------
    std::cout << "\n[Step 9] Cleaning up..." << std::endl;

    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(xDevAddr);
    aclrtFree(outDevAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "  Resources released" << std::endl;

    // -----------------------------------------------------------------------
    // Final result
    // -----------------------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    if (pass) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return pass ? 0 : 1;
}
