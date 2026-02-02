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

#include "acl/acl.h"
#include "aclnnop/aclnn_dual_level_quant_matmul_nz.h"
#include "aclnnop/aclnn_npu_format_cast.h"

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
int CreateAclTensorWithFormat(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t* storageShape, uint64_t storageShapeSize,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
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

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape, storageShapeSize, *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> x1Shape = {256, 1024};
    std::vector<int64_t> x2Shape = {2048, 512}; // 一个int8的数据包含两个4bit数据在
    std::vector<int64_t> x2StorageShape = {16, 128, 16, 32};
    std::vector<int64_t> biasShape = {2048};
    std::vector<int64_t> x1Level0ScaleShape = {256, 2};
    std::vector<int64_t> x1Level1ScaleShape = {256, 16, 2};
    std::vector<int64_t> x2Level0ScaleShape = {2, 2048};
    std::vector<int64_t> x2Level1ScaleShape = {2048, 16, 2};
    std::vector<int64_t> outShape = {256, 2048};
    int32_t level0GroupSize = 512;
    int32_t level1GroupSize = 32;

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* x1Level0ScaleDeviceAddr = nullptr;
    void* x1Level1ScaleDeviceAddr = nullptr;
    void* x2Level0ScaleDeviceAddr = nullptr;
    void* x2Level1ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* x1Level0Scale = nullptr;
    aclTensor* x1Level1Scale = nullptr;
    aclTensor* x2Level0Scale = nullptr;
    aclTensor* x2Level1Scale = nullptr;
    aclTensor* out = nullptr;

    std::vector<int8_t> x1HostData(256 * 1024, 1);
    std::vector<int8_t> x2HostData(512 * 2048, 1);
    std::vector<int32_t> biasHostData(2048, 1);
    std::vector<float> x1Level0ScaleHostData(256 * 2, 1);
    std::vector<float> x1Level1ScaleHostData(256 * 32, 1);
    std::vector<float> x2Level0ScaleHostData(2048 * 2, 1);
    std::vector<float> x2Level1ScaleHostData(2048 * 32, 1);
    std::vector<uint16_t> outHostData(256 * 2048, 1); // 实际上是float16半精度方式
    // 创建x1 aclTensor
    ret = CreateAclTensorWithFormat(
        x1HostData, x1Shape, x1Shape.data(), x1Shape.size(), &x1DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x1,
        aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensorWithFormat(
        x2HostData, x2Shape, x2StorageShape.data(), x2StorageShape.size(), &x2DeviceAddr, aclDataType::ACL_INT8, &x2,
        aclFormat::ACL_FORMAT_FRACTAL_NZ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x1Level0Scale aclTensor
    ret = CreateAclTensorWithFormat(
        x1Level0ScaleHostData, x1Level0ScaleShape, x1Level0ScaleShape.data(), x1Level0ScaleShape.size(),
        &x1Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Level0Scale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x1Level1Scale aclTensor
    ret = CreateAclTensorWithFormat(
        x1Level1ScaleHostData, x1Level1ScaleShape, x1Level1ScaleShape.data(), x1Level1ScaleShape.size(),
        &x1Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Level1Scale, aclFormat::ACL_FORMAT_NCL);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2Level0Scale aclTensor
    ret = CreateAclTensorWithFormat(
        x2Level0ScaleHostData, x2Level0ScaleShape, x2Level0ScaleShape.data(), x2Level0ScaleShape.size(),
        &x2Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Level0Scale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2Level1Scale aclTensor
    ret = CreateAclTensorWithFormat(
        x2Level1ScaleHostData, x2Level1ScaleShape, x2Level1ScaleShape.data(), x2Level1ScaleShape.size(),
        &x2Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Level1Scale, aclFormat::ACL_FORMAT_NCL);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensorWithFormat(
        biasHostData, biasShape, biasShape.data(), biasShape.size(), &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias,
        aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensorWithFormat(
        outHostData, outShape, outShape.data(), outShape.size(), &outDeviceAddr, aclDataType::ACL_FLOAT16, &out,
        aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    bool transposeX1 = false;
    bool transposeX2 = true;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSizeDualLevel = 0;
    aclOpExecutor* executorDualLevel = nullptr;
    // 调用aclnnDualLevelQuantMatmulWeightNz第一段接口
    ret = aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
        x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, bias, transposeX1, transposeX2,
        level0GroupSize, level1GroupSize, out, &workspaceSizeDualLevel, &executorDualLevel);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddrDualLevel = nullptr;
    if (workspaceSizeDualLevel > 0) {
        ret = aclrtMalloc(&workspaceAddrDualLevel, workspaceSizeDualLevel, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnDualLevelQuantMatmulWeightNz第二段接口
    ret = aclnnDualLevelQuantMatmulWeightNz(workspaceAddrDualLevel, workspaceSizeDualLevel, executorDualLevel, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(
        size, 0); // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(bias);
    aclDestroyTensor(x1Level0Scale);
    aclDestroyTensor(x1Level1Scale);
    aclDestroyTensor(x2Level0Scale);
    aclDestroyTensor(x2Level1Scale);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(x1Level0ScaleDeviceAddr);
    aclrtFree(x1Level1ScaleDeviceAddr);
    aclrtFree(x2Level0ScaleDeviceAddr);
    aclrtFree(x2Level1ScaleDeviceAddr);
    aclrtFree(outDeviceAddr);

    Finalize(deviceId, stream);
    return 0;
}