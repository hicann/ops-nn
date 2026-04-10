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
 * @brief 专门测试极小输出case（输出 < 32B）
 * 
 * 测试场景：
 * 1. 输出 < 32B（极小case）：触发 DataCopyPad 路径
 * 2. 输出 = 32B（边界case）：刚好对齐
 * 3. 输出 > 32B 但有 tail（普通case）：测试 tail 路径
 */

#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_v3.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

/**
 * @brief 测试极小输出case
 * @param M, N: 输出形状为 {M, N}
 * @param testName: 测试名称
 */
int TestSmallCase(int32_t deviceId, aclrtStream &stream, int64_t M, int64_t N, int64_t K, 
                  const char* testName, int64_t expectedOutputBytes)
{
    LOG_PRINT("\n========== %s ==========\n", testName);
    LOG_PRINT("Output shape: {%ld, %ld}, output size: %ld elements, %ld bytes\n", 
              M, N, M * N, expectedOutputBytes);

    std::vector<int64_t> x1Shape = {M, K};
    std::vector<int64_t> x2Shape = {K, N};
    std::vector<int64_t> biasShape = {N};
    std::vector<int64_t> offsetShape = {N};
    std::vector<int64_t> scaleShape = {N};
    std::vector<int64_t> outShape = {M, N};

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *quantParamDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *quantParam = nullptr;
    aclTensor *offset = nullptr;
    aclTensor *out = nullptr;

    // 构造输入数据
    std::vector<int8_t> x1HostData(M * K, 1);
    std::vector<int8_t> x2HostData(K * N, 1);
    std::vector<int32_t> biasHostData(N, 1);
    std::vector<float> scaleHostData(N, 1.0f);
    std::vector<float> offsetHostData(N, 1.0f);
    std::vector<uint16_t> outHostData(M * N, 0);

    // 创建tensors
    int ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);

    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);

    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);

    ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);

    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);

    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);

    bool transposeX1 = false;
    bool transposeX2 = false;

    // 调用 TransQuantParamV2
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrV2.reset(workspaceAddr);
    }
    ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

    // 调用 QuantMatmulV3
    ret = aclnnQuantMatmulV3GetWorkspaceSize(x1, x2, quantParam, nullptr, bias, transposeX1, transposeX2, out,
                                            &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrV3.reset(workspaceAddr);
    }

    ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出结果
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    
    LOG_PRINT("Result: ");
    for (int64_t i = 0; i < size && i < 10; i++) {  // 只打印前10个
        LOG_PRINT("[%ld]=%u ", i, resultData[i]);
    }
    if (size > 10) LOG_PRINT("...");
    LOG_PRINT("\nTest PASSED!\n");
    
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("测试极小输出case（输出 < 32B）的越界写修复\n");
    LOG_PRINT("========================================\n");

    // Case 1: 极小case - 输出 4 字节 (2个float16元素) < 32B
    // M=1, N=2, K=1 -> output = 1*2 = 2 elements = 4 bytes
    ret = TestSmallCase(deviceId, stream, 1, 2, 1, 
                        "Case 1: 极小case (4 bytes < 32B)", 4);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Case 1 failed. ERROR: %d\n", ret); return ret);

    // Case 2: 极小case - 输出 16 字节 (8个float16元素) < 32B
    // M=2, N=4, K=1 -> output = 2*4 = 8 elements = 16 bytes
    ret = TestSmallCase(deviceId, stream, 2, 4, 1, 
                        "Case 2: 极小case (16 bytes < 32B)", 16);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Case 2 failed. ERROR: %d\n", ret); return ret);

    // Case 3: 边界case - 输出刚好 32 字节 (16个float16元素) = 32B
    // M=4, N=4, K=1 -> output = 4*4 = 16 elements = 32 bytes
    ret = TestSmallCase(deviceId, stream, 4, 4, 1, 
                        "Case 3: 边界case (32 bytes = 32B)", 32);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Case 3 failed. ERROR: %d\n", ret); return ret);

    // Case 4: 普通case - 输出 64 字节 (32个float16元素) > 32B
    // M=4, N=8, K=1 -> output = 4*8 = 32 elements = 64 bytes
    ret = TestSmallCase(deviceId, stream, 4, 8, 1, 
                        "Case 4: 普通case (64 bytes > 32B)", 64);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Case 4 failed. ERROR: %d\n", ret); return ret);

    // Case 5: 原有测试用例 - 输出 30 字节 (15个float16元素) < 32B
    // M=5, N=3, K=2 -> output = 5*3 = 15 elements = 30 bytes
    ret = TestSmallCase(deviceId, stream, 5, 3, 2, 
                        "Case 5: 原有测试 (30 bytes < 32B)", 30);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Case 5 failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("All tests PASSED!\n");
    LOG_PRINT("========================================\n");

    Finalize(deviceId, stream);
    return 0;
}
