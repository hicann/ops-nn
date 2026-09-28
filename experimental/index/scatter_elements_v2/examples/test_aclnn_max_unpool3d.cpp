/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d.h"

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

// bfloat16 占用 2 字节，等于 fp32 的高 16 位（截断），用于在纯 C++ 中构造/打印 bf16 数据
static uint16_t FloatToBf16(float x)
{
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}
static float Bf16ToFloat(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
template <typename T>
float ToFloat(T v)
{
    return static_cast<float>(v);
}
template <>
float ToFloat<uint16_t>(uint16_t v)
{ // uint16_t 存储视为 bf16
    return Bf16ToFloat(v);
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

// 通用单组调用：self(dtype=selfDt, 存储类型 SelfStoreT) + indices(dtype=idxDt, 类型 IdxT)
template <typename SelfStoreT, typename IdxT>
int RunMaxUnpool3d(const char* tag, const std::vector<SelfStoreT>& selfData, aclDataType selfDt,
                   const std::vector<IdxT>& idxData, aclDataType idxDt, const std::vector<int64_t>& selfShape,
                   const std::vector<int64_t>& outShape, const std::vector<int64_t>& outputSizeVec,
                   const std::vector<int64_t>& strideVec, const std::vector<int64_t>& paddingVec, aclrtStream stream)
{
    void* selfAddr = nullptr;
    void* idxAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* out = nullptr;
    std::vector<SelfStoreT> outData(GetShapeSize(outShape), static_cast<SelfStoreT>(0));

    auto ret = CreateAclTensor(selfData, selfShape, &selfAddr, selfDt, &self);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(idxData, selfShape, &idxAddr, idxDt, &indices);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(outData, outShape, &outAddr, selfDt, &out);
    CHECK_RET(ret == 0, return ret);
    const aclIntArray* outputSize = aclCreateIntArray(outputSizeVec.data(), outputSizeVec.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    const aclIntArray* stride = aclCreateIntArray(strideVec.data(), strideVec.size());
    CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    const aclIntArray* padding = aclCreateIntArray(paddingVec.data(), paddingVec.size());
    CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMaxUnpool3dGetWorkspaceSize(self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclnnMaxUnpool3dGetWorkspaceSize failed. ERROR: %d\n", tag, ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMaxUnpool3d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclnnMaxUnpool3d failed. ERROR: %d\n", tag, ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(SelfStoreT), outAddr, size * sizeof(SelfStoreT),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[%s] out =", tag);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT(" %.1f", ToFloat<SelfStoreT>(outData[i]));
    }
    LOG_PRINT("\n");

    aclDestroyTensor(self);
    aclDestroyTensor(indices);
    aclDestroyTensor(out);
    aclDestroyIntArray(outputSize);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(padding);
    aclrtFree(selfAddr);
    aclrtFree(idxAddr);
    aclrtFree(outAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 公共 shape：self [1,1,2,2](N,D,H,W) -> out [1,1,4,4]；stride/padding 为预留参数
    std::vector<int64_t> selfShape = {1, 1, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4};
    std::vector<int64_t> outputSize = {1, 4, 4};
    std::vector<int64_t> stride = {1, 2, 3};
    std::vector<int64_t> padding = {1, 2, 3};
    std::vector<int64_t> idx64 = {3, 8, 11, 13};
    std::vector<int32_t> idx32 = {3, 8, 11, 13};

    // 3. 多组 aclnn 调用：覆盖 fp32 与新增 bf16，indices 覆盖 int64 / int32
    RunMaxUnpool3d<float, int64_t>("fp32 + int64", {1, 2, 3, 4}, ACL_FLOAT, idx64, ACL_INT64, selfShape, outShape,
                                   outputSize, stride, padding, stream);
    RunMaxUnpool3d<float, int32_t>("fp32 + int32", {1, 2, 3, 4}, ACL_FLOAT, idx32, ACL_INT32, selfShape, outShape,
                                   outputSize, stride, padding, stream);
    std::vector<uint16_t> bf16Self = {FloatToBf16(1.0f), FloatToBf16(2.0f), FloatToBf16(3.0f), FloatToBf16(4.0f)};
    RunMaxUnpool3d<uint16_t, int64_t>("bf16 + int64", bf16Self, ACL_BF16, idx64, ACL_INT64, selfShape, outShape,
                                      outputSize, stride, padding, stream);
    RunMaxUnpool3d<uint16_t, int32_t>("bf16 + int32", bf16Self, ACL_BF16, idx32, ACL_INT32, selfShape, outShape,
                                      outputSize, stride, padding, stream);

    // 4. 释放 device 资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
