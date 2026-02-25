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
#include "aclnnop/aclnn_cast.h"

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

template <typename T>
void PrintMat(std::vector<T> resultData, std::vector<int64_t> resultShape)
{
    int64_t m = resultShape[0];
    int64_t n = resultShape[1];
    for (size_t i = 0; i < m; i++) {
        printf(i == 0 ? "[[" : " [");
        for (size_t j = 0; j < n; j++) {
            std::cout << resultData[i * n + j] << (j == n - 1 ? "" : ", ");
            if (j == 2 && j + 3 < n) {
                printf("..., ");
                j = n - 4;
            }
        }
        printf(i < m - 1 ? "],\n" : "]]\n");
        if (i == 2 && i + 3 < m) {
            printf(" ... \n");
            i = m - 4;
        }
    }
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, const int64_t* storageShape,
    int64_t storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor,
    aclFormat format = aclFormat::ACL_FORMAT_ND)
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
        shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape, storageShapeSize, *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnDualLevelQuantMatmulWeightNz(int32_t deviceId, aclrtStream stream)
{
    int ret = 0;
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    constexpr int64_t B4_IN_B8_NUMS = 2L;
    constexpr int64_t B8_IN_B16_NUMS = 2L;
    int64_t m = 256;
    int64_t k = 1024;
    int64_t n = 512;
    int64_t level0GroupSize = 512;
    int64_t level1GroupSize = 32;
    bool transposeX1 = false;
    bool transposeX2 = true;
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {n, k};
    std::vector<int64_t> biasShape = {n};
    std::vector<int64_t> x1Level0ScaleShape = {m, k / level0GroupSize};
    std::vector<int64_t> x1Level1ScaleShape = {m, k / level1GroupSize / B8_IN_B16_NUMS, B8_IN_B16_NUMS};
    std::vector<int64_t> x2Level0ScaleShape = {k / level0GroupSize, n};
    std::vector<int64_t> x2Level1ScaleShape = {n, k / level1GroupSize / B8_IN_B16_NUMS, B8_IN_B16_NUMS};
    std::vector<int64_t> outShape = {m, n};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x2NzDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* x1Level0ScaleDeviceAddr = nullptr;
    void* x1Level1ScaleDeviceAddr = nullptr;
    void* x2Level0ScaleDeviceAddr = nullptr;
    void* x2Level1ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* outFp32DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x2Nz = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* x1Level0Scale = nullptr;
    aclTensor* x1Level1Scale = nullptr;
    aclTensor* x2Level0Scale = nullptr;
    aclTensor* x2Level1Scale = nullptr;
    aclTensor* out = nullptr;
    aclTensor* outFp32 = nullptr;

    // fp4的1.0二进制表示,0b0010，每个uint8_t表示两个fp4，用uint8_t承载fp4
    std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape) / 2, 0b0010'0010);
    std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape), 0b0010'0010);
    std::vector<float> biasHostData(n, 1002.0f);
    std::vector<float> x1Level0ScaleHostData(GetShapeSize(x1Level0ScaleShape), 1.0f);
    std::vector<float> x2Level0ScaleHostData(GetShapeSize(x2Level0ScaleShape), 1.0f);
    // fp8e8m0的1.0二进制表示，0b0111'1111
    std::vector<uint8_t> x1Level1ScaleHostData(GetShapeSize(x1Level1ScaleShape), 0b0111'1111);
    std::vector<uint8_t> x2Level1ScaleHostData(GetShapeSize(x2Level1ScaleShape), 0b0111'1111);
    // 输出用0填充
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<float> outFp32HostData(GetShapeSize(outShape), 0.0f);

    // 创建x1 aclTensor
    ret = CreateAclTensor(
        x1HostData, x1Shape, x1Shape.data(), x1Shape.size(), &x1DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2 aclTensor
    // NpuFormatCast无法转换B4类型，使用B8代替，因此内轴减半
    x2Shape[1] /= 2;
    ret =
        CreateAclTensor(x2HostData, x2Shape, x2Shape.data(), x2Shape.size(), &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 使用NpuFormatCast接口转换为实际tensor，或者直接构造weightNZ数据
    int64_t* x2NzShape = nullptr;
    uint64_t x2NzShapeSize = 0;
    int x2NzFormat;
    // 计算目标tensor的shape和format
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(
        x2, static_cast<int>(aclFormat::ACL_FORMAT_FRACTAL_NZ), static_cast<int>(aclDataType::ACL_INT8), &x2NzShape,
        &x2NzShapeSize, &x2NzFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);
    ret = CreateAclTensor(
        x2HostData, x2Shape, x2NzShape, x2NzShapeSize, &x2NzDeviceAddr, aclDataType::ACL_INT8, &x2Nz,
        static_cast<aclFormat>(x2NzFormat));
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzTensorPtr(x2Nz, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2NzDeviceAddrPtr(x2NzDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    // 创建x1Level0Scale aclTensor
    ret = CreateAclTensor(
        x1Level0ScaleHostData, x1Level0ScaleShape, x1Level0ScaleShape.data(), x1Level0ScaleShape.size(),
        &x1Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level0ScaleTensorPtr(
        x1Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level0ScaleDeviceAddrPtr(x1Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1Level1Scale aclTensor
    ret = CreateAclTensor(
        x1Level1ScaleHostData, x1Level1ScaleShape, x1Level1ScaleShape.data(), x1Level1ScaleShape.size(),
        &x1Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level1ScaleTensorPtr(
        x1Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level1ScaleDeviceAddrPtr(x1Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Level0Scale aclTensor
    ret = CreateAclTensor(
        x2Level0ScaleHostData, x2Level0ScaleShape, x2Level0ScaleShape.data(), x2Level0ScaleShape.size(),
        &x2Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level0ScaleTensorPtr(
        x2Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level0ScaleDeviceAddrPtr(x2Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Level1Scale aclTensor
    ret = CreateAclTensor(
        x2Level1ScaleHostData, x2Level1ScaleShape, x2Level1ScaleShape.data(), x2Level1ScaleShape.size(),
        &x2Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level1ScaleTensorPtr(
        x2Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level1ScaleDeviceAddrPtr(x2Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias aclTensor
    ret = CreateAclTensor(
        biasHostData, biasShape, biasShape.data(), biasShape.size(), &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(
        outHostData, outShape, outShape.data(), outShape.size(), &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outFp32 aclTensor
    ret = CreateAclTensor(
        outFp32HostData, outShape, outShape.data(), outShape.size(), &outFp32DeviceAddr, aclDataType::ACL_FLOAT, &outFp32);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outFp32TensorPtr(outFp32, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outFp32DeviceAddrPtr(outFp32DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

    // 首先使用NpuFormatCast将x2转为NZ格式
    // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
    ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2Nz, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceNpuFormatCastAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceNpuFormatCastAddrPtr(nullptr, aclrtFree);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceNpuFormatCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceNpuFormatCastAddrPtr.reset(workspaceNpuFormatCastAddr);
    }
    // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
    ret = aclnnNpuFormatCast(workspaceNpuFormatCastAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnDualLevelQuantMatmulWeightNz第一段接口
    ret = aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
        x1, x2Nz, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, bias, transposeX1, transposeX2,
        level0GroupSize, level1GroupSize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnDualLevelQuantMatmulWeightNz第二段接口
    ret = aclnnDualLevelQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果转为FP32拷贝至host侧，需要根据具体API的接口定义修改
    ret = aclnnCastGetWorkspaceSize(out, aclDataType::ACL_FLOAT, outFp32, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    ret = aclrtMemcpy(
        outFp32HostData.data(), size * sizeof(outFp32HostData[0]), outFp32DeviceAddr, size * sizeof(outFp32HostData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    PrintMat(outFp32HostData, outShape);
    return ret;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 执行测试过程，并释放内存
    AclnnDualLevelQuantMatmulWeightNz(deviceId, stream);
    Finalize(deviceId, stream);
    return 0;
}