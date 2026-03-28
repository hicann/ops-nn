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
#include <memory>
#include <vector>
#include <cstdint>

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

float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    return *reinterpret_cast<float*>(&bits);
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
    auto size = hostData.size() * sizeof(T);
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

int AclnnQuantMatmulV5Int4Int4Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 1024;
    int64_t k = 1024;
    int64_t n = 1024;

    // K-CG量化模式：x1和x2为INT4，x1Scale和x2Scale为FLOAT32，x2Offset为FLOAT16，out为BFLOAT16
    aclDataType x1Dtype = aclDataType::ACL_INT4;
    aclDataType x2Dtype = aclDataType::ACL_INT4;
    aclDataType x1ScaleDtype = aclDataType::ACL_FLOAT;
    aclDataType x2ScaleDtype = aclDataType::ACL_FLOAT;
    aclDataType x2OffsetDtype = aclDataType::ACL_FLOAT16;
    aclDataType outDtype = aclDataType::ACL_BF16;

    // 形状设置
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {n, k}; // transposeX2为true时是(n, k)
    std::vector<int64_t> x1ScaleShape = {m, 1};
    std::vector<int64_t> x2ScaleShape = {k / 256, n};
    std::vector<int64_t> x2OffsetShape = {k / 256, n};
    std::vector<int64_t> outShape = {m, n};

    // 设备内存地址
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* x2OffsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    // 张量指针
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* x2Offset = nullptr;
    aclTensor* yScale = nullptr;   // 不使用
    aclTensor* x1Offset = nullptr; // 不使用
    aclTensor* yOffset = nullptr;  // 不使用
    aclTensor* bias = nullptr;     // 不使用
    aclTensor* out = nullptr;

    // 构造输入数据
    // x1/x2: INT4，每个int8存储2个int4值（低4bit + 高4bit）
    auto PackInt4Pair = [](int8_t low, int8_t high) -> int8_t {
        uint8_t lowNibble = static_cast<uint8_t>(low) & 0x0F;
        uint8_t highNibble = (static_cast<uint8_t>(high) & 0x0F) << 4;
        return static_cast<int8_t>(highNibble | lowNibble);
    };

    std::vector<int8_t> x1HostData(m * k / 2, 0);
    std::vector<int8_t> x2HostData(n * k / 2, 0);

    // x1 按 0 -> 7 循环递增
    for (int64_t i = 0; i < m * k / 2; ++i) {
        int64_t idx0 = 2 * i;
        int64_t idx1 = idx0 + 1;
        int8_t v0 = static_cast<int8_t>(idx0 % 8);
        int8_t v1 = static_cast<int8_t>(idx1 % 8);
        x1HostData[i] = PackInt4Pair(v0, v1);
    }

    // x2 按 7 -> 0 循环递减
    for (int64_t i = 0; i < n * k / 2; ++i) {
        int64_t idx0 = 2 * i;
        int64_t idx1 = idx0 + 1;
        int8_t v0 = static_cast<int8_t>(7 - (idx0 % 8));
        int8_t v1 = static_cast<int8_t>(7 - (idx1 % 8));
        x2HostData[i] = PackInt4Pair(v0, v1);
    }
    // x1Scale: FLOAT32，全1
    std::vector<float> x1ScaleHostData(m, 1.0f);
    // x2Scale: FLOAT32，全1
    std::vector<float> x2ScaleHostData((k / 256) * n, 1.0f);
    // x2Offset: FLOAT16，全0
    std::vector<uint16_t> x2OffsetHostData((k / 256) * n, 0);
    // out: BFLOAT16，用于接收结果
    std::vector<uint16_t> outHostData(m * n, 0);

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, x1Dtype, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, x2Dtype, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1Scale aclTensor
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, x1ScaleDtype, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Scale aclTensor
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, x2ScaleDtype, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Offset aclTensor
    ret = CreateAclTensor(x2OffsetHostData, x2OffsetShape, &x2OffsetDeviceAddr, x2OffsetDtype, &x2Offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2OffsetTensorPtr(x2Offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2OffsetDeviceAddrPtr(x2OffsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outDtype, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 转置设置
    bool transposeX1 = false;
    bool transposeX2 = true; // K-CG量化模式下x2需要转置

    // groupSize设置：K-CG量化模式下为[0, 0, 256]
    int64_t groupSize = 256; // groupSizeK

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnQuantMatmulV5第一段接口
    ret = aclnnQuantMatmulV5GetWorkspaceSize(
        x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2, groupSize, out,
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

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0); // C语言中无法直接打印bfloat16的数据，需要用uint16读出来
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印部分结果
    LOG_PRINT("First 10 results:\n");
    for (int64_t i = 0; i < std::min<int64_t>(10, size); i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main(int argc, char* argv[])
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnQuantMatmulV5Int4Int4Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5Int4Int4Test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
