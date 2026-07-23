/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_compress.h"

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

namespace {
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint8_t HexDigit(char value) { return static_cast<uint8_t>(value <= '9' ? value - '0' : value - 'a' + 10); }

std::vector<uint8_t> DecodeHex(const char* value, size_t length)
{
    std::vector<uint8_t> result(length / 2);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<uint8_t>((HexDigit(value[2 * i]) << 4) | HexDigit(value[2 * i + 1]));
    }
    return result;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        initialized_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        stream_ = *stream;
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

    AclRuntimeGuard(const AclRuntimeGuard&) = delete;
    AclRuntimeGuard& operator=(const AclRuntimeGuard&) = delete;

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

using DevicePtr = std::unique_ptr<void, aclError (*)(void*)>;
using TensorPtr = std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)>;

/*
 * CreateAclTensor即使在Memcpy或aclCreateTensor阶段失败，也会把已申请的地址返回给调用者；
 * 调用者必须在检查返回码前接管deviceAddr和tensor。
 */
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int PrintResult(const std::vector<int64_t>& shape, void* deviceAddr)
{
    const auto size = static_cast<size_t>(GetShapeSize(shape));
    std::vector<aclFloat16> resultData(size);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), deviceAddr,
                           resultData.size() * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Copy result to host failed. ERROR: %d\n", ret); return ret);

    const auto printCount = std::min<size_t>(size, 10);
    for (size_t i = 0; i < printCount; ++i) {
        LOG_PRINT("result[%zu] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
    return ACL_SUCCESS;
}
} // namespace

int main()
{
    // 由官方CompressFcOp对[64, 64]的FP16零权重执行W16A16S压缩得到，不依赖外部data目录。
    const char compressedWeightHex
        [] = "88100000000000afa1aeadacabaaa9a8a7a6a5a4a3a2bfb0b1b2b3b4b5b6b7b888100000000000afa1aeadacabaaa9a8a7a6a5a4a"
             "3a2bfb0b1b2b3b4b5b6b7b888100000000000afa1aeadacabaaa9a8a7a6a5a4a3a2bfb0b1b2b3b4b5b6b7b8"
             "88100000000000afa1aeadacabaaa9a8a7a6a5a4a3a2bfb0b1b2b3b4b5b6b7b8b9babbbcbdbec0904551144551000000455114455"
             "10000004551144551000000b9babbbcbdbec090455114455100000045511445510000004551144551000000"
             "b9babbbcbdbec090455114455100000045511445510000004551144551000000b9babbbcbdbec0904551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "455114455100000045511445510000004551144551000000455114455100000045511445510000004551144551000000455114455"
             "100000045511445510000004551144551000000455114455100000045511445510000004551144551000000"
             "4551144551e03e000000000000000000000000000000000000000000000000004551144551e03e000000000000000000000000000"
             "000000000000000000000004551144551e03e00000000000000000000000000000000000000000000000000"
             "4551144551e03e00000000000000000000000000000000000000000000000000";
    const char compressedIndexHex[] = "0a80000000000000";
    const auto weightHostData = DecodeHex(compressedWeightHex, sizeof(compressedWeightHex) - 1);
    const auto compressIndexHostData = DecodeHex(compressedIndexHex, sizeof(compressedIndexHex) - 1);

    const std::vector<int64_t> xShape = {16, 64};
    const std::vector<int64_t> weightShape = {static_cast<int64_t>(weightHostData.size())};
    const std::vector<int64_t> biasShape = {64};
    const std::vector<int64_t> compressIndexShape = {static_cast<int64_t>(compressIndexHostData.size())};
    const std::vector<int64_t> outShape = {16, 64};
    const std::vector<aclFloat16> xHostData(GetShapeSize(xShape), aclFloatToFloat16(1.0F));
    const std::vector<float> biasHostData(GetShapeSize(biasShape), 0.0F);
    const std::vector<aclFloat16> outHostData(GetShapeSize(outShape), aclFloatToFloat16(0.0F));

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard runtimeGuard(deviceId);
    auto ret = runtimeGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* compressIndexDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* compressIndex = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
    DevicePtr xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    TensorPtr xTensorPtr(x, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, ACL_INT8, &weight);
    DevicePtr weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    TensorPtr weightTensorPtr(weight, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, ACL_FLOAT, &bias);
    DevicePtr biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    TensorPtr biasTensorPtr(bias, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(compressIndexHostData, compressIndexShape, &compressIndexDeviceAddr, ACL_INT8,
                          &compressIndex);
    DevicePtr compressIndexDeviceAddrPtr(compressIndexDeviceAddr, aclrtFree);
    TensorPtr compressIndexTensorPtr(compressIndex, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_FLOAT16, &out);
    DevicePtr outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    TensorPtr outTensorPtr(out, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    DevicePtr workspaceAddrPtr(nullptr, aclrtFree);
    ret = aclnnMatmulCompressGetWorkspaceSize(x, weight, bias, compressIndex, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        workspaceAddrPtr.reset(workspaceAddr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMatmulCompress(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompress failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    ret = PrintResult(outShape, outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}
