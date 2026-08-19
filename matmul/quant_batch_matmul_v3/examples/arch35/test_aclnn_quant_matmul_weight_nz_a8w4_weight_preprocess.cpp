/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/float4_e2m1.h"
#include "aclnnop/aclnn_quant_matmul_weight_nz.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

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

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (aclInited_) {
            aclFinalize();
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                    const std::vector<int64_t>& viewStrides, const std::vector<int64_t>& storageShape,
                    aclDataType dataType, aclFormat format, void** deviceAddr, aclTensor** tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    const int64_t* strides = viewStrides.empty() ? nullptr : viewStrides.data();
    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides, 0, format, storageShape.data(),
                              storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

std::vector<uint8_t> PackFp4(const std::vector<float>& data)
{
    std::vector<uint8_t> packedData((data.size() + 1) / 2, 0);
    for (size_t i = 0; i < data.size(); i += 2) {
        uint8_t low = op::Float4E2M1(data[i]).value;
        uint8_t high = i + 1 < data.size() ? op::Float4E2M1(data[i + 1]).value : 0;
        packedData[i / 2] = static_cast<uint8_t>((high << 4) | low);
    }
    return packedData;
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int AclnnQuantMatmulWeightNzA8W4PreprocessTest(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    AclRuntimeGuard runtime(deviceId);
    auto ret = runtime.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t m = 5;
    const int64_t k = 64;
    const int64_t n = 128;
    const int64_t kGroupSize = 32;
    const int64_t scaleK = CEIL_DIV(k, 64);
    const bool transposeX1 = false;
    const bool transposeX2 = false;

    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x1Strides = {k, 1};
    std::vector<int64_t> weightViewShape = {k, n};
    std::vector<int64_t> weightStorageShape = {n, k};
    std::vector<int64_t> weightStrides = {1, k};
    std::vector<int64_t> x1ScaleShape = {m, scaleK, 2};
    std::vector<int64_t> x1ScaleStrides = {scaleK * 2, 2, 1};
    std::vector<int64_t> weightScaleViewShape = {scaleK, n, 2};
    std::vector<int64_t> weightScaleStorageShape = {n, scaleK, 2};
    std::vector<int64_t> weightScaleStrides = {2, scaleK * 2, 1};
    std::vector<int64_t> outWeightStorageShape = {CEIL_DIV(k, 32), CEIL_DIV(n, 16), 16, 32};
    std::vector<int64_t> outShape = {m, n};
    std::vector<int64_t> outStrides = {n, 1};

    std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape), 0b00111000); // FLOAT8_E4M3FN 1.0
    std::vector<float> weightFloatData(GetShapeSize(weightStorageShape), 1.0f);
    std::vector<uint8_t> weightHostData = PackFp4(weightFloatData);
    std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 0b01111111); // FLOAT8_E8M0 1.0
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleStorageShape),
                                             0b10000101); // FLOAT8_E8M0 64.0
    std::vector<uint8_t> outWeightHostData(GetShapeSize(outWeightStorageShape) / 2, 0);
    std::vector<uint8_t> outWeightScaleHostData(GetShapeSize(weightScaleStorageShape), 0);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);

    void* x1DeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* outWeightDeviceAddr = nullptr;
    void* outWeightScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* outWeight = nullptr;
    aclTensor* outWeightScale = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(x1HostData, x1Shape, x1Strides, x1Shape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &x1DeviceAddr,
                          &x1);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Ptr(x1, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightViewShape, weightStrides, weightStorageShape, ACL_FLOAT4_E2M1,
                          ACL_FORMAT_ND, &weightDeviceAddr, &weight);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPtr(weight, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, x1ScaleStrides, x1ScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                          &x1ScaleDeviceAddr, &x1Scale);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScalePtr(x1Scale, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                          ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleDeviceAddr, &weightScale);
    std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScalePtr(weightScale, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 保留weight的转置stride，使QuantMatmul在transposeX2=false时能从tensor元数据识别转置布局。
    ret = CreateAclTensor(outWeightHostData, weightViewShape, weightStrides, outWeightStorageShape, ACL_FLOAT4_E2M1,
                          ACL_FORMAT_FRACTAL_NZ_C0_32, &outWeightDeviceAddr, &outWeight);
    std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightPtr(outWeight, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outWeightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                          ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &outWeightScaleDeviceAddr, &outWeightScale);
    std::unique_ptr<void, aclError (*)(void*)> outWeightScaleDeviceAddrPtr(outWeightScaleDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScalePtr(outWeightScale, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, outStrides, outShape, ACL_BF16, ACL_FORMAT_ND, &outDeviceAddr, &out);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outPtr(out, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, nullptr, nullptr, ACL_FLOAT8_E4M3FN,
                                                     ACL_FLOAT8_E8M0, kGroupSize, outWeight, outWeightScale, nullptr,
                                                     nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* preprocessWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> preprocessWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&preprocessWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate preprocess workspace failed. ERROR: %d\n", ret); return ret);
        preprocessWorkspacePtr.reset(preprocessWorkspaceAddr);
    }

    ret = aclnnWeightQuantPreprocess(preprocessWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocess failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after preprocess failed. ERROR: %d\n", ret);
              return ret);

    workspaceSize = 0;
    executor = nullptr;
    // WeightQuantPreprocess的输出已满足QuantMatmul输入要求，直接作为x2和x2Scale使用。
    ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, outWeight, x1Scale, outWeightScale, nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, transposeX1, transposeX2, kGroupSize, out,
                                                   &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* matmulWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> matmulWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&matmulWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate matmul workspace failed. ERROR: %d\n", ret); return ret);
        matmulWorkspacePtr.reset(matmulWorkspaceAddr);
    }

    ret = aclnnQuantMatmulWeightNz(matmulWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after matmul failed. ERROR: %d\n", ret);
              return ret);

    std::vector<uint16_t> resultData(GetShapeSize(outShape), 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (size_t i = 0; i < resultData.size(); ++i) {
        LOG_PRINT("result[%zu] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
    }
    return ACL_SUCCESS;
}

int main()
{
    // WeightQuantPreprocess当前仅支持Ascend 950PR/Ascend 950DT。
    int32_t deviceId = 0;
    auto ret = AclnnQuantMatmulWeightNzA8W4PreprocessTest(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulWeightNzA8W4PreprocessTest failed. ERROR: %d\n", ret);
              return ret);
    return 0;
}
