/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_aclnn_hard_sigmoid.cpp
 * \brief HardSigmoid aclnn example (Ascend950 / arch35)
 *
 * Tests FP32, FP16, and INT32 dtype paths via aclnnHardsigmoid two-stage API.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn_hardsigmoid.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i <= 0) {
            return 0;
        }
        int64_t prev = shapeSize;
        shapeSize *= i;
        if (shapeSize / i != prev) {
            return -1;
        }
    }
    return shapeSize;
}

static void CpuGoldenFloat(const std::vector<float>& x, std::vector<float>& y)
{
    constexpr float kAlpha = 1.0f / 6.0f;
    constexpr float kBeta = 0.5f;
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = kAlpha * x[i] + kBeta;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        y[i] = v;
    }
}

static void CpuGoldenInt32(const std::vector<int32_t>& x, std::vector<int32_t>& y)
{
    constexpr float kAlpha = 1.0f / 6.0f;
    constexpr float kBeta = 0.5f;
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = kAlpha * static_cast<float>(x[i]) + kBeta;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        y[i] = static_cast<int32_t>(v);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

static int RunHardsigmoid(aclTensor* self, aclTensor* out, void* outDeviceAddr,
                           int64_t elemCount, size_t elemSize, aclrtStream stream,
                           void* resultBuf)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnHardsigmoidGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardsigmoidGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnHardsigmoid(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardsigmoid failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resultBuf, elemCount * elemSize, outDeviceAddr, elemCount * elemSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

static int TestFloat32(aclrtStream stream)
{
    LOG_PRINT("\n===== Test FP32 =====\n");
    std::vector<int64_t> shape = {4, 8};
    std::vector<float> inputData = {
        -5.0f, -3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f,
         1.5f,  2.0f,  2.5f,  3.0f,  3.5f, 4.0f, 5.0f, 6.0f,
        -4.0f, -3.5f, -2.5f, -1.5f, -0.1f, 0.1f, 0.3f, 0.7f,
         1.2f,  1.8f,  2.2f,  2.8f,  3.2f, 3.8f, 4.5f, 5.5f
    };
    int64_t elemCount = GetShapeSize(shape);

    aclTensor* self = nullptr;
    void* selfDev = nullptr;
    auto ret = CreateAclTensor(inputData, shape, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == 0, return ret);

    aclTensor* out = nullptr;
    void* outDev = nullptr;
    std::vector<float> outHost(elemCount, 0.0f);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == 0, return ret);

    std::vector<float> resultData(elemCount);
    ret = RunHardsigmoid(self, out, outDev, elemCount, sizeof(float), stream, resultData.data());
    CHECK_RET(ret == 0, return ret);

    std::vector<float> golden;
    CpuGoldenFloat(inputData, golden);

    int errors = 0;
    for (size_t i = 0; i < resultData.size(); ++i) {
        float diff = std::fabs(resultData[i] - golden[i]);
        float thr = 1e-4f + 1e-4f * std::fabs(golden[i]);
        bool ok = diff <= thr;
        LOG_PRINT("hardsigmoid input[%zu]=%8.4f, npu=%8.6f, golden=%8.6f, diff=%8.6f %s\n",
                  i, inputData[i], resultData[i], golden[i], diff, ok ? "OK" : "FAIL");
        if (!ok) ++errors;
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDev);
    aclrtFree(outDev);

    LOG_PRINT("[FP32] %s (%zu/%zu)\n", errors == 0 ? "PASS" : "FAIL",
              resultData.size() - errors, resultData.size());
    return errors;
}

static int TestFloat16(aclrtStream stream)
{
    LOG_PRINT("\n===== Test FP16 =====\n");
    std::vector<int64_t> shape = {4, 4};
    std::vector<float> inputFloats = {
        -5.0f, -3.0f, -1.0f, 0.0f,
         0.5f,  1.0f,  2.0f, 3.0f,
         3.5f,  4.0f,  5.0f, 6.0f,
        -0.5f, -2.5f,  1.5f, 2.5f
    };
    int64_t elemCount = GetShapeSize(shape);

    std::vector<aclFloat16> inputFp16(elemCount);
    for (int64_t i = 0; i < elemCount; ++i) {
        inputFp16[i] = aclFloatToFloat16(inputFloats[i]);
    }

    aclTensor* self = nullptr;
    void* selfDev = nullptr;
    auto ret = CreateAclTensor(inputFp16, shape, &selfDev, ACL_FLOAT16, &self);
    CHECK_RET(ret == 0, return ret);

    aclTensor* out = nullptr;
    void* outDev = nullptr;
    std::vector<aclFloat16> outHost(elemCount, aclFloatToFloat16(0.0f));
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT16, &out);
    CHECK_RET(ret == 0, return ret);

    std::vector<aclFloat16> resultFp16(elemCount);
    ret = RunHardsigmoid(self, out, outDev, elemCount, sizeof(aclFloat16), stream, resultFp16.data());
    CHECK_RET(ret == 0, return ret);

    std::vector<float> golden;
    CpuGoldenFloat(inputFloats, golden);

    int errors = 0;
    for (int64_t i = 0; i < elemCount; ++i) {
        float npuVal = aclFloat16ToFloat(resultFp16[i]);
        float diff = std::fabs(npuVal - golden[i]);
        float thr = 1e-2f + 1e-2f * std::fabs(golden[i]);
        bool ok = diff <= thr;
        LOG_PRINT("hardsigmoid input[%ld]=%8.4f, npu=%8.6f, golden=%8.6f, diff=%8.6f %s\n",
                  (long)i, inputFloats[i], npuVal, golden[i], diff, ok ? "OK" : "FAIL");
        if (!ok) ++errors;
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDev);
    aclrtFree(outDev);

    LOG_PRINT("[FP16] %s (%ld/%ld)\n", errors == 0 ? "PASS" : "FAIL",
              elemCount - errors, elemCount);
    return errors;
}

static int TestInt32(aclrtStream stream)
{
    LOG_PRINT("\n===== Test INT32 =====\n");
    std::vector<int64_t> shape = {4, 4};
    std::vector<int32_t> inputData = {
        -10, -5, -3, -1,
          0,  1,  2,  3,
          4,  5,  6, 10,
         -6, -4,  7, 12
    };
    int64_t elemCount = GetShapeSize(shape);

    aclTensor* self = nullptr;
    void* selfDev = nullptr;
    auto ret = CreateAclTensor(inputData, shape, &selfDev, ACL_INT32, &self);
    CHECK_RET(ret == 0, return ret);

    aclTensor* out = nullptr;
    void* outDev = nullptr;
    std::vector<int32_t> outHost(elemCount, 0);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &out);
    CHECK_RET(ret == 0, return ret);

    std::vector<int32_t> resultData(elemCount);
    ret = RunHardsigmoid(self, out, outDev, elemCount, sizeof(int32_t), stream, resultData.data());
    CHECK_RET(ret == 0, return ret);

    std::vector<int32_t> golden;
    CpuGoldenInt32(inputData, golden);

    int errors = 0;
    for (int64_t i = 0; i < elemCount; ++i) {
        bool ok = (resultData[i] == golden[i]);
        LOG_PRINT("hardsigmoid input[%ld]=%4d, npu=%d, golden=%d %s\n",
                  (long)i, inputData[i], resultData[i], golden[i], ok ? "OK" : "FAIL");
        if (!ok) ++errors;
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDev);
    aclrtFree(outDev);

    LOG_PRINT("[INT32] %s (%ld/%ld)\n", errors == 0 ? "PASS" : "FAIL",
              elemCount - errors, elemCount);
    return errors;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalErrors = 0;
    totalErrors += TestFloat32(stream);
    totalErrors += TestFloat16(stream);
    totalErrors += TestInt32(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (totalErrors == 0) {
        LOG_PRINT("\n[RESULT] aclnnHardsigmoid ALL PASS\n");
        return 0;
    }
    LOG_PRINT("\n[RESULT] aclnnHardsigmoid FAIL (%d total mismatches)\n", totalErrors);
    return 1;
}
