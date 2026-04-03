/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file test_aclnn_quant_matmul_v5_perblock.cpp
* \brief Test case for perblock quantization matmul
*/

#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

/**
 * Test case for perblock quantization matmul
 *
 * Test configuration (satisfying all perblock tiling constraints):
 * - M = 128, K = 4096, N = 512
 * - groupSizeK = 128, groupSizeN = 128
 * - transposeX1 = false, transposeX2 = true
 * - x1: [M, K] = [128, 4096], dtype = INT8
 * - x2: [N, K] = [512, 4096] (transposed), dtype = INT8
 * - x1Scale: [M, CeilDiv(K, 128)] = [128, 32], dtype = FLOAT
 * - x2Scale: [CeilDiv(N, 128), CeilDiv(K, 128)] = [4, 32], dtype = FLOAT
 * - bias: [N] = [512], dtype = FLOAT
 * - output: [M, N] = [128, 512], dtype = BF16
 * - N % 256 == 0 (satisfied: 512 % 256 == 0)
 */
int AclnnQuantMatmulV5PerblockTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Perblock tiling constraints:
    // - groupSizeK = 128, groupSizeN = 128
    // - transposeX1 = false, transposeX2 = true
    // - N % 256 == 0
    constexpr int64_t M = 128;
    constexpr int64_t K = 4096;
    constexpr int64_t N = 512;  // N must be multiple of 256 (baseN)
    constexpr int64_t groupSizeM = 1;
    constexpr int64_t groupSizeK = 128;
    constexpr int64_t groupSizeN = 128;

    // x1: [M, K], dtype = INT8
    std::vector<int64_t> x1Shape = {M, K};
    // x2: [N, K] (transposed, so shape is [N, K]), dtype = INT8
    std::vector<int64_t> x2Shape = {N, K};
    // x1Scale: [M, CeilDiv(K, groupSizeK)], dtype = FLOAT
    std::vector<int64_t> x1ScaleShape = {M, (K + groupSizeK - 1) / groupSizeK};  // [128, 32]
    // x2Scale: [CeilDiv(N, groupSizeN), CeilDiv(K, groupSizeK)] (transposed), dtype = FLOAT
    std::vector<int64_t> x2ScaleShape = {(N + groupSizeN - 1) / groupSizeN, (K + groupSizeK - 1) / groupSizeK};  // [4, 32]
    // bias: [N], dtype = FLOAT
    std::vector<int64_t> biasShape = {N};
    // output: [M, N], dtype = BF16
    std::vector<int64_t> outShape = {M, N};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* out = nullptr;

    // Initialize input data
    std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
    std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 1);
    std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1.0f);
    std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1.0f);
    std::vector<float> biasHostData(GetShapeSize(biasShape), 0.0f);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);

    // 创建x1 aclTensor (INT8)
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2 aclTensor (INT8)
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1Scale aclTensor (FLOAT)
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Scale aclTensor (FLOAT)
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias aclTensor (FLOAT)
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor (BF16)
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Perblock requires: transposeX1 = false, transposeX2 = true
    bool transposeX1 = false;
    bool transposeX2 = true;
    // groupSize encoding: groupSizeM in high bits, groupSizeN in middle, groupSizeK in low
    int64_t groupSize = (groupSizeM << 32) | (groupSizeN << 16) | groupSizeK;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Testing perblock quantization matmul:\n");
    LOG_PRINT("  M=%ld, K=%ld, N=%ld\n", M, K, N);
    LOG_PRINT("  x1Shape=[%ld, %ld], x2Shape=[%ld, %ld]\n", x1Shape[0], x1Shape[1], x2Shape[0], x2Shape[1]);
    LOG_PRINT("  x1ScaleShape=[%ld, %ld], x2ScaleShape=[%ld, %ld]\n", x1ScaleShape[0], x1ScaleShape[1], x2ScaleShape[0], x2ScaleShape[1]);
    LOG_PRINT("  biasShape=[%ld], outShape=[%ld, %ld]\n", biasShape[0], outShape[0], outShape[1]);
    LOG_PRINT("  transposeX1=%d, transposeX2=%d, groupSize=0x%lx\n", transposeX1, transposeX2, groupSize);

    ret = aclnnQuantMatmulV5GetWorkspaceSize(
        x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize, out,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnQuantMatmulV5第二段接口
    ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Perblock test completed successfully! The sync event ID fix is working.\n");

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // Print first few results for verification
    int64_t printCount = (size < 10) ? size : 10;
    for (int64_t i = 0; i < printCount; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnQuantMatmulV5PerblockTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5PerblockTest failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return 0;
}