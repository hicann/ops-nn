/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gemm_syrk.h"

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
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 将FP16的uint16_t表示转换为float表示
float Fp16ToFloat(uint16_t h)
{
    int s = (h >> 15) & 0x1;  // sign
    int e = (h >> 10) & 0x1F; // exponent
    int f = h & 0x3FF;        // fraction
    if (e == 0) {
        if (f == 0) {
            return s ? -0.0f : 0.0f;
        }
        float sig = f / 1024.0f;
        float result = sig * pow(2, -24);
        return s ? -result : result;
    } else if (e == 31) {
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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

// GemmSyrk: C = alpha * (A @ A^T) + beta * C, C输入输出同地址原地更新。
// A为(128, 64)全1.0，C为(128, 128)全2.0（对称），alpha=beta=1.0时
// 结果为 64 * 1.0 + 2.0 = 66.0。
int AclnnGemmSyrkTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> aShape = {128, 64};
    std::vector<int64_t> cShape = {128, 128};
    void* aDeviceAddr = nullptr;
    void* cDeviceAddr = nullptr;
    aclTensor* a = nullptr;
    aclTensor* cRef = nullptr;
    std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0b0011110000000000); // fp16的1.0
    std::vector<uint16_t> cHostData(GetShapeSize(cShape), 0b0100000000000000); // fp16的2.0
    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> aTensorPtr(a, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> aDeviceAddrPtr(aDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // cRef既是输入也是输出（原地更新）
    ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT16, &cRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> cTensorPtr(cRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> cDeviceAddrPtr(cDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float alphaValue = 1.0f;
    float betaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    aclScalar* beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> alphaPtr(alpha, aclDestroyScalar);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> betaPtr(beta, aclDestroyScalar);
    CHECK_RET(alpha != nullptr && beta != nullptr, LOG_PRINT("aclCreateScalar failed\n");
              return ACL_ERROR_INTERNAL_ERROR);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGemmSyrkGetWorkspaceSize(a, cRef, alpha, beta, false, "full", &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnGemmSyrk(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrk failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(cShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    bool allMatch = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::fabs(Fp16ToFloat(resultData[i]) - 66.0f) > 1e-3f) {
            LOG_PRINT("result[%ld] is: %f, expected 66.0\n", i, Fp16ToFloat(resultData[i]));
            allMatch = false;
        }
    }
    LOG_PRINT("GemmSyrk in-place result check: %s\n", allMatch ? "PASS" : "FAIL");
    return allMatch ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// transpose_x场景：a以转置的(64, 128)存储（k=64, m=128），计算
// C = alpha * (A^T @ A) + beta * C。a、C同为全1.0/全2.0，结果同样为66.0。
int AclnnGemmSyrkTransTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> aShape = {64, 128};
    std::vector<int64_t> cShape = {128, 128};
    void* aDeviceAddr = nullptr;
    void* cDeviceAddr = nullptr;
    aclTensor* a = nullptr;
    aclTensor* cRef = nullptr;
    std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0b0011110000000000); // fp16的1.0
    std::vector<uint16_t> cHostData(GetShapeSize(cShape), 0b0100000000000000); // fp16的2.0
    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> aTensorPtr(a, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> aDeviceAddrPtr(aDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT16, &cRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> cTensorPtr(cRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> cDeviceAddrPtr(cDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float alphaValue = 1.0f;
    float betaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    aclScalar* beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> alphaPtr(alpha, aclDestroyScalar);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> betaPtr(beta, aclDestroyScalar);
    CHECK_RET(alpha != nullptr && beta != nullptr, LOG_PRINT("aclCreateScalar failed\n");
              return ACL_ERROR_INTERNAL_ERROR);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGemmSyrkGetWorkspaceSize(a, cRef, alpha, beta, true, "full", &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnGemmSyrk(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrk failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(cShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    bool allMatch = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::fabs(Fp16ToFloat(resultData[i]) - 66.0f) > 1e-3f) {
            LOG_PRINT("result[%ld] is: %f, expected 66.0\n", i, Fp16ToFloat(resultData[i]));
            allMatch = false;
        }
    }
    LOG_PRINT("GemmSyrk transpose_x in-place result check: %s\n", allMatch ? "PASS" : "FAIL");
    return allMatch ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnGemmSyrkTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnGemmSyrkTest failed. ERROR: %d\n", ret); return ret);

    ret = AclnnGemmSyrkTransTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnGemmSyrkTransTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
