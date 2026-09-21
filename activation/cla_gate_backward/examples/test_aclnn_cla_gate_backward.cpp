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
 * \file test_aclnn_cla_gate_backward.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cla_gate_backward.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: ClaGateBackward only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

// 将 float 转换为 bfloat16 的 uint16_t 表示。
uint16_t FloatToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    return static_cast<uint16_t>(bits >> 16);
}

// 将 bfloat16 的 uint16_t 位模式还原为 float。
float Bf16ToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
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
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return ACL_SUCCESS;
    }

    // token=8, head=64, dim=256：三路 TND 输入 [T, N, D]，两路 gate logits [T, N]。
    int64_t tokenCount = 8;
    int64_t headNum = 64;
    int64_t headDim = 256;
    std::vector<int64_t> tndShape = {tokenCount, headNum, headDim};
    std::vector<int64_t> logitsShape = {tokenCount, headNum};

    void* gradMergedDeviceAddr = nullptr;
    void* globalAttnDeviceAddr = nullptr;
    void* localAttnDeviceAddr = nullptr;
    void* globalGateLogitsDeviceAddr = nullptr;
    void* localGateLogitsDeviceAddr = nullptr;
    void* gradGlobalAttnOutDeviceAddr = nullptr;
    void* gradLocalAttnOutDeviceAddr = nullptr;
    void* gradGlobalGateLogitsOutDeviceAddr = nullptr;
    void* gradLocalGateLogitsOutDeviceAddr = nullptr;

    aclTensor* gradMerged = nullptr;
    aclTensor* globalAttn = nullptr;
    aclTensor* localAttn = nullptr;
    aclTensor* globalGateLogits = nullptr;
    aclTensor* localGateLogits = nullptr;
    aclTensor* gradGlobalAttnOut = nullptr;
    aclTensor* gradLocalAttnOut = nullptr;
    aclTensor* gradGlobalGateLogitsOut = nullptr;
    aclTensor* gradLocalGateLogitsOut = nullptr;

    // 输入用 BF16 存储（uint16_t 承载）；输出 host 缓冲仅用于回读。
    std::vector<uint16_t> gradMergedHostData(GetShapeSize(tndShape), FloatToBf16(0.5f));
    std::vector<uint16_t> globalAttnHostData(GetShapeSize(tndShape), FloatToBf16(1.0f));
    std::vector<uint16_t> localAttnHostData(GetShapeSize(tndShape), FloatToBf16(0.5f));
    std::vector<uint16_t> globalGateLogitsHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
    std::vector<uint16_t> localGateLogitsHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
    std::vector<uint16_t> gradGlobalAttnOutHostData(GetShapeSize(tndShape), FloatToBf16(0.0f));
    std::vector<uint16_t> gradLocalAttnOutHostData(GetShapeSize(tndShape), FloatToBf16(0.0f));
    std::vector<uint16_t> gradGlobalGateLogitsOutHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
    std::vector<uint16_t> gradLocalGateLogitsOutHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));

    ret = CreateAclTensor(gradMergedHostData, tndShape, &gradMergedDeviceAddr, ACL_BF16, &gradMerged);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(globalAttnHostData, tndShape, &globalAttnDeviceAddr, ACL_BF16, &globalAttn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(localAttnHostData, tndShape, &localAttnDeviceAddr, ACL_BF16, &localAttn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(globalGateLogitsHostData, logitsShape, &globalGateLogitsDeviceAddr, ACL_BF16,
                          &globalGateLogits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(localGateLogitsHostData, logitsShape, &localGateLogitsDeviceAddr, ACL_BF16, &localGateLogits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradGlobalAttnOutHostData, tndShape, &gradGlobalAttnOutDeviceAddr, ACL_BF16,
                          &gradGlobalAttnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradLocalAttnOutHostData, tndShape, &gradLocalAttnOutDeviceAddr, ACL_BF16, &gradLocalAttnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradGlobalGateLogitsOutHostData, logitsShape, &gradGlobalGateLogitsOutDeviceAddr, ACL_BF16,
                          &gradGlobalGateLogitsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradLocalGateLogitsOutHostData, logitsShape, &gradLocalGateLogitsOutDeviceAddr, ACL_BF16,
                          &gradLocalGateLogitsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // input_attn_layout：当前仅支持 "TND"。
    const char* inputAttnLayout = "TND";

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnClaGateBackwardGetWorkspaceSize(
        gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout, gradGlobalAttnOut,
        gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnClaGateBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto tndSize = GetShapeSize(tndShape);
    auto logitsSize = GetShapeSize(logitsShape);
    std::vector<uint16_t> gradGlobalAttnOutData(tndSize, 0);
    std::vector<uint16_t> gradLocalAttnOutData(tndSize, 0);
    std::vector<uint16_t> gradGlobalGateLogitsOutData(logitsSize, 0);
    std::vector<uint16_t> gradLocalGateLogitsOutData(logitsSize, 0);

    ret = aclrtMemcpy(gradGlobalAttnOutData.data(), tndSize * sizeof(uint16_t), gradGlobalAttnOutDeviceAddr,
                      tndSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradGlobalAttnOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(gradLocalAttnOutData.data(), tndSize * sizeof(uint16_t), gradLocalAttnOutDeviceAddr,
                      tndSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradLocalAttnOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(gradGlobalGateLogitsOutData.data(), logitsSize * sizeof(uint16_t),
                      gradGlobalGateLogitsOutDeviceAddr, logitsSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradGlobalGateLogitsOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(gradLocalGateLogitsOutData.data(), logitsSize * sizeof(uint16_t),
                      gradLocalGateLogitsOutDeviceAddr, logitsSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradLocalGateLogitsOut failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < 8 && i < tndSize; i++) {
        LOG_PRINT("gradGlobalAttnOut[%ld] is: %f\n", i, static_cast<double>(Bf16ToFloat(gradGlobalAttnOutData[i])));
    }
    for (int64_t i = 0; i < 8 && i < logitsSize; i++) {
        LOG_PRINT("gradGlobalGateLogitsOut[%ld] is: %f\n", i,
                  static_cast<double>(Bf16ToFloat(gradGlobalGateLogitsOutData[i])));
    }

    aclDestroyTensor(gradMerged);
    aclDestroyTensor(globalAttn);
    aclDestroyTensor(localAttn);
    aclDestroyTensor(globalGateLogits);
    aclDestroyTensor(localGateLogits);
    aclDestroyTensor(gradGlobalAttnOut);
    aclDestroyTensor(gradLocalAttnOut);
    aclDestroyTensor(gradGlobalGateLogitsOut);
    aclDestroyTensor(gradLocalGateLogitsOut);

    aclrtFree(gradMergedDeviceAddr);
    aclrtFree(globalAttnDeviceAddr);
    aclrtFree(localAttnDeviceAddr);
    aclrtFree(globalGateLogitsDeviceAddr);
    aclrtFree(localGateLogitsDeviceAddr);
    aclrtFree(gradGlobalAttnOutDeviceAddr);
    aclrtFree(gradLocalAttnOutDeviceAddr);
    aclrtFree(gradGlobalGateLogitsOutDeviceAddr);
    aclrtFree(gradLocalGateLogitsOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
