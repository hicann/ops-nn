/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_hi_f4_quant.h"

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
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d, msg: %s\n", ret, aclGetRecentErrMsg());
              return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // x1/x2: shape [2, 128], FP16
    // y_out: shape [2, 128], HIFLOAT4 (4bit packed, actual bytes = elements / 2)
    // x_out: shape [2, 128], FP16
    // scale_out: shape [2, 2, 1], HIFLOAT4_SCALE (float32 per element)
    // rstd_out: shape [2, 1], FLOAT32
    std::vector<int64_t> x1Shape = {2, 128};
    std::vector<int64_t> x2Shape = {2, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yOutShape = {2, 128};
    std::vector<int64_t> xOutShape = {2, 128};
    std::vector<int64_t> scaleOutShape = {2, 2, 1};
    std::vector<int64_t> rstdOutShape = {2, 1};

    int64_t xTotal = GetShapeSize(x1Shape);
    int64_t gammaTotal = GetShapeSize(gammaShape);

    // Host data
    std::vector<aclFloat16> x1HostData(xTotal);
    std::vector<aclFloat16> x2HostData(xTotal);
    std::vector<aclFloat16> gammaHostData(gammaTotal);
    for (int64_t i = 0; i < xTotal; i++) {
        x1HostData[i] = aclFloat16(1.0f);
        x2HostData[i] = aclFloat16(1.0f);
    }
    for (int64_t i = 0; i < gammaTotal; i++) {
        gammaHostData[i] = aclFloat16(1.0f);
    }

    // y: HIFLOAT4, 4bit packed, bytes = elementCount / 2
    int64_t yElementNum = GetShapeSize(yOutShape);
    size_t yBytes = yElementNum / 2;
    std::vector<uint8_t> yOutHostData(yBytes, 0);

    // xOut: FP16
    size_t xOutBytes = GetShapeSize(xOutShape) * sizeof(aclFloat16);
    std::vector<aclFloat16> xOutHostData(GetShapeSize(xOutShape));

    // scale: HIFLOAT4_SCALE, float32 per element
    int64_t scaleTotal = GetShapeSize(scaleOutShape);
    size_t scaleBytes = scaleTotal * sizeof(float);
    std::vector<float> scaleOutHostData(scaleTotal, 0.0f);

    // rstd: FLOAT32
    int64_t rstdTotal = GetShapeSize(rstdOutShape);
    size_t rstdBytes = rstdTotal * sizeof(float);
    std::vector<float> rstdOutHostData(rstdTotal, 0.0f);

    // Device memory and tensors
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* xOutDeviceAddr = nullptr;
    void* scaleOutDeviceAddr = nullptr;
    void* rstdOutDeviceAddr = nullptr;

    size_t xBytes = xTotal * sizeof(aclFloat16);
    size_t gammaBytes = gammaTotal * sizeof(aclFloat16);

    ret = aclrtMalloc(&x1DeviceAddr, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x1 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(x1DeviceAddr, xBytes, x1HostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&x2DeviceAddr, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x2 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(x2DeviceAddr, xBytes, x2HostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&gammaDeviceAddr, gammaBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc gamma failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(gammaDeviceAddr, gammaBytes, gammaHostData.data(), gammaBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy gamma failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&yOutDeviceAddr, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(yOutDeviceAddr, yBytes, yOutHostData.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yOut failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&xOutDeviceAddr, xOutBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(xOutDeviceAddr, xOutBytes, xOutHostData.data(), xOutBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xOut failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&scaleOutDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc scaleOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(scaleOutDeviceAddr, scaleBytes, scaleOutHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy scaleOut failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&rstdOutDeviceAddr, rstdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc rstdOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(rstdOutDeviceAddr, rstdBytes, rstdOutHostData.data(), rstdBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy rstdOut failed. ERROR: %d\n", ret); return ret);

    // Create aclTensor
    std::vector<int64_t> xStrides = {128, 1};
    std::vector<int64_t> gammaStrides = {1};
    std::vector<int64_t> yStrides = {128, 1};
    std::vector<int64_t> scaleStrides = {2, 1, 1};
    std::vector<int64_t> rstdStrides = {1};

    aclTensor* x1 = aclCreateTensor(x1Shape.data(), x1Shape.size(), aclDataType::ACL_FLOAT16, xStrides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, x1Shape.data(), x1Shape.size(), x1DeviceAddr);
    CHECK_RET(x1 != nullptr, LOG_PRINT("aclCreateTensor x1 failed.\n"); return 1);

    aclTensor* x2 = aclCreateTensor(x2Shape.data(), x2Shape.size(), aclDataType::ACL_FLOAT16, xStrides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, x2Shape.data(), x2Shape.size(), x2DeviceAddr);
    CHECK_RET(x2 != nullptr, LOG_PRINT("aclCreateTensor x2 failed.\n"); return 1);

    aclTensor* gamma = aclCreateTensor(gammaShape.data(), gammaShape.size(), aclDataType::ACL_FLOAT16,
                                       gammaStrides.data(), 0, aclFormat::ACL_FORMAT_ND, gammaShape.data(),
                                       gammaShape.size(), gammaDeviceAddr);
    CHECK_RET(gamma != nullptr, LOG_PRINT("aclCreateTensor gamma failed.\n"); return 1);

    aclTensor* yOut = aclCreateTensor(yOutShape.data(), yOutShape.size(), aclDataType::ACL_HIFLOAT4, yStrides.data(), 0,
                                      aclFormat::ACL_FORMAT_ND, yOutShape.data(), yOutShape.size(), yOutDeviceAddr);
    CHECK_RET(yOut != nullptr, LOG_PRINT("aclCreateTensor yOut failed.\n"); return 1);

    aclTensor* xOut = aclCreateTensor(xOutShape.data(), xOutShape.size(), aclDataType::ACL_FLOAT16, xStrides.data(), 0,
                                      aclFormat::ACL_FORMAT_ND, xOutShape.data(), xOutShape.size(), xOutDeviceAddr);
    CHECK_RET(xOut != nullptr, LOG_PRINT("aclCreateTensor xOut failed.\n"); return 1);

    aclTensor* scaleOut = aclCreateTensor(scaleOutShape.data(), scaleOutShape.size(), aclDataType::ACL_HIFLOAT4_SCALE,
                                          scaleStrides.data(), 0, aclFormat::ACL_FORMAT_ND, scaleOutShape.data(),
                                          scaleOutShape.size(), scaleOutDeviceAddr);
    CHECK_RET(scaleOut != nullptr, LOG_PRINT("aclCreateTensor scaleOut failed.\n"); return 1);

    aclTensor* rstdOut = aclCreateTensor(rstdOutShape.data(), rstdOutShape.size(), aclDataType::ACL_FLOAT,
                                         rstdStrides.data(), 0, aclFormat::ACL_FORMAT_ND, rstdOutShape.data(),
                                         rstdOutShape.size(), rstdOutDeviceAddr);
    CHECK_RET(rstdOut != nullptr, LOG_PRINT("aclCreateTensor rstdOut failed.\n"); return 1);

    double epsilon = 1e-6;
    bool outputRstd = true;
    int64_t scaleAlg = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddRmsNormHiF4QuantGetWorkspaceSize(x1, x2, gamma, nullptr, epsilon, outputRstd, scaleAlg, yOut, xOut,
                                                   scaleOut, rstdOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormHiF4QuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAddRmsNormHiF4Quant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormHiF4Quant failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Copy results
    std::vector<uint8_t> yResultData(yBytes, 0);
    ret = aclrtMemcpy(yResultData.data(), yBytes, yOutDeviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut failed. ERROR: %d\n", ret); return ret);

    std::vector<aclFloat16> xOutResultData(GetShapeSize(xOutShape));
    ret = aclrtMemcpy(xOutResultData.data(), xOutBytes, xOutDeviceAddr, xOutBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy xOut failed. ERROR: %d\n", ret); return ret);

    std::vector<float> scaleResultData(scaleTotal, 0.0f);
    ret = aclrtMemcpy(scaleResultData.data(), scaleBytes, scaleOutDeviceAddr, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scaleOut failed. ERROR: %d\n", ret); return ret);

    std::vector<float> rstdResultData(rstdTotal, 0.0f);
    ret = aclrtMemcpy(rstdResultData.data(), rstdBytes, rstdOutDeviceAddr, rstdBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstdOut failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("---- y result (HIFLOAT4 packed as uint8) ----\n");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(yBytes), int64_t(16)); i++) {
        LOG_PRINT("y[%ld] = 0x%02x\n", i, yResultData[i]);
    }
    LOG_PRINT("---- xOut result (FP16) ----\n");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(GetShapeSize(xOutShape)), int64_t(16)); i++) {
        LOG_PRINT("xOut[%ld] = 0x%04x\n", i, static_cast<uint16_t>(xOutResultData[i]));
    }
    LOG_PRINT("---- scale result (HIFLOAT4_SCALE, float32) ----\n");
    for (int64_t i = 0; i < scaleTotal; i++) {
        LOG_PRINT("scale[%ld] = %f\n", i, scaleResultData[i]);
    }
    LOG_PRINT("---- rstd result (FLOAT32) ----\n");
    for (int64_t i = 0; i < rstdTotal; i++) {
        LOG_PRINT("rstd[%ld] = %f\n", i, rstdResultData[i]);
    }

    // Cleanup
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(yOut);
    aclDestroyTensor(xOut);
    aclDestroyTensor(scaleOut);
    aclDestroyTensor(rstdOut);

    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yOutDeviceAddr);
    aclrtFree(xOutDeviceAddr);
    aclrtFree(scaleOutDeviceAddr);
    aclrtFree(rstdOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();

    return 0;
}
