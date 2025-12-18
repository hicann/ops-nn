/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_sparse4to2quant_matmul_weight_nz.h"

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

#define CREATE_TENSOR(hostData, shape, deviceAddr, dtype, tensor)                                        \
    ret = CreateAclTensor(hostData, shape, &deviceAddr, dtype, &tensor);                                 \
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
    std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(xDeviceAddr, aclrtFree);                  \
    CHECK_RET(ret == ACL_SUCCESS, return ret)

#define CREATE_SPARSE_TENSOR(hostData, weightShape, storageShape, deviceAddr, dataType, tensor)          \
    ret = CreateSparseTensor(hostData, weightShape, storageShape, &deviceAddr, dataType, &tensor);       \
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
    std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(xDeviceAddr, aclrtFree);                  \
    CHECK_RET(ret == ACL_SUCCESS, return ret)

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
int CreateSparseTensor(
    const T* sparseWeightData, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<uint64_t>(GetShapeSize(storageShape)) * sizeof(T);

    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, sparseWeightData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(viewShape.size(), 1);
    for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
        strides[i] = viewShape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
        storageShape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    if (hostData.size() > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

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

void GenRandomMask(std::vector<size_t>& masks)
{
    masks[0] = random() % 4;
    masks[1] = random() % 4;
    while (masks[1] == masks[0]) {
        masks[1] = random() % 4;
    }
}

void GenRandomSparseData(std::vector<int8_t>& weightHostData)
{
    srandom(233U);
    std::vector<size_t> masks(2, 0UL);
    constexpr size_t step = 4UL;
    for (size_t i = 0; i < weightHostData.size(); i += step) {
        GenRandomMask(masks);
        for (auto mask : masks) {
            weightHostData[i + mask] = 0;
        }
    }
}

std::vector<int64_t> GenStorageShape(int64_t* dims, uint64_t dimsNum)
{
    std::vector<int64_t> storageShape;
    for (uint64_t i = 0UL; i < dimsNum; i++) {
        storageShape.push_back(dims[i]);
    }
    return storageShape;
}

int aclnnSparse4to2QuantMatmulTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 64L;
    int64_t k = 512L;
    int64_t n = 128L;
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {n, k};
    std::vector<int64_t> indexShape = {n, (k + 7) / 8};
    std::vector<int64_t> biasShape = {n};
    std::vector<int64_t> xScaleShape = {m};
    std::vector<int64_t> weightScaleShape = {n};
    std::vector<int64_t> outShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* sparseWeightDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* sparseWeight = nullptr;
    aclTensor* index = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
    std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), 1); // 实际上是bfloat16半精度方式
    std::vector<float> xScaleHostData(GetShapeSize(xScaleShape), 1);
    std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape), 1);
    GenRandomSparseData(weightHostData);

    int8_t* sparseWeightHostData = nullptr;
    uint8_t* indexHostData = nullptr;
    int64_t* sparseWeightDims = nullptr;
    uint64_t sparseWeightDimsNum = 0UL;
    int64_t* indexDims = nullptr;
    uint64_t indexDimsNum = 0UL;
    aclIntArray* weightShapeArray = aclCreateIntArray(weightShape.data(), weightShape.size());
    ret = aclnnTransSparse4to2Para(
        weightHostData.data(), weightShapeArray, &sparseWeightHostData, &sparseWeightDims, &sparseWeightDimsNum,
        &indexHostData, &indexDims, &indexDimsNum);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransSparse4to2Para failed. ERROR: %d\n", ret); return ret);
    std::unique_ptr<int8_t[]> sparseWeightHostDataPtr(sparseWeightHostData);
    std::unique_ptr<uint8_t[]> indexHostDataPtr(indexHostData);
    std::unique_ptr<int64_t[]> sparseWeightDimsPtr(sparseWeightDims);
    std::unique_ptr<int64_t[]> indexDimsPtr(indexDims);

    CREATE_TENSOR(xHostData, xShape, xDeviceAddr, aclDataType::ACL_INT8, x);

    weightShape.back() = (weightShape.back() + 7) / 8 * 8 / 2; // 4选2后 K轴向上8对齐后减半
    auto sparseWeightStorageShape = GenStorageShape(sparseWeightDims, sparseWeightDimsNum);
    CREATE_SPARSE_TENSOR(
        sparseWeightHostData, weightShape, sparseWeightStorageShape, sparseWeightDeviceAddr, aclDataType::ACL_INT8,
        sparseWeight);

    auto indexStorageShape = GenStorageShape(indexDims, indexDimsNum);
    CREATE_SPARSE_TENSOR(indexHostData, indexShape, indexStorageShape, indexDeviceAddr, aclDataType::ACL_UINT8, index);
    CREATE_TENSOR(biasHostData, biasShape, biasDeviceAddr, aclDataType::ACL_BF16, bias);
    CREATE_TENSOR(xScaleHostData, xScaleShape, xScaleDeviceAddr, aclDataType::ACL_FLOAT, xScale);
    CREATE_TENSOR(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT, weightScale);
    CREATE_TENSOR(std::vector<uint16_t>(), outShape, outDeviceAddr, aclDataType::ACL_BF16, out);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 调用aclnnSparse4to2QuantMatmul第一段接口
    ret = aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
        x, sparseWeight, index, xScale, weightScale, bias, out, &workspaceSize, &executor);

    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrTrans.reset(workspaceAddr);
    }
    // 调用aclnnSparse4to2QuantMatmul第二段接口
    ret = aclnnSparse4to2QuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    // C语言中无法直接打印bf16的数据，需要用uint16读出来，自行通过二进制转成bf16
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnSparse4to2QuantMatmulTest(deviceId, stream);
    CHECK_FREE_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}