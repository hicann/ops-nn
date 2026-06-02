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
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"
#include "aclnnop/aclnn_convert_weight_to_int4_pack.h"

#define CHECK_RET(cond, return_expr)     \
    do {                                 \
        if (!(cond)) {                   \
            return_expr;                 \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)          \
    do {                                 \
        printf(message, ##__VA_ARGS__);  \
    } while (0)

#define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
#define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
    if (format == aclFormat::ACL_FORMAT_ND) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
    } else {
        std::vector<int64_t> nzShape;
        if (dataType == aclDataType::ACL_INT4) {
            nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 64};
        } else {
            nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 8};
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = true;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 72;
    int64_t n = 17;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
        weightDim0 = n;
        weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
        weightInt4PackShape = {weightDim0, weightDim1};
    } else {
        weightInt4PackShape = {weightDim0, weightDim1 / 8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightInt4PackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
        std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); //一个int8数据存放2个int4数据，所以这里除以2
        if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
            weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1 / 2, 32) * CEIL_ALIGN(weightDim0, 16), 0);
        }
        // 创建weightInt4Pack aclTensor
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
            weightInt4PackDtype, &weightInt4Pack, weightFormat);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
        std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); //一个int32数据存放8个int4数据，所以这里除以8
        if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
            weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1 / 8, 8) * CEIL_ALIGN(weightDim0, 16), 0);
            ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                weightInt4PackDtype, &weightInt4Pack, weightFormat);
        } else {
            // 创建weightInt4Pack aclTensor
            ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                weightInt4PackDtype, &weightInt4Pack);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightInt4PackTensorPtr(weightInt4Pack, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightInt4PackDeviceAddrPtr(weightInt4PackDeviceAddr, aclrtFree);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp16TensorPtr(antiquantScaleFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp16DeviceAddrPtr(antiquantScaleFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 对weight做int32转int4pack
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(nullptr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // weight为转置场景，且weightInt4Pack shape为NZ时，需要调用aclInitTensor转换为非连续的tensor
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
        for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
            strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
        }
        std::swap(strides[0], strides[1]);
        std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
        std::vector<int64_t> nzShape = {CEIL_DIV(k, 64), CEIL_DIV(n, 16), 16, 8};
        if (weightInt4PackDtype == aclDataType::ACL_INT4) {
            nzShape[3] = 64;
        }
        aclInitTensor(weightInt4Pack, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
            weightFormat, nzShape.data(), nzShape.size(), weightInt4PackDeviceAddr);
    }

    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    void* workspaceAddrCastX = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddrCastX, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrCastXPtr(workspaceAddrCastX, aclrtFree);
    ret = aclnnCast(workspaceAddrCastX, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    void* workspaceAddrCastScale = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddrCastScale, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrCastScalePtr(workspaceAddrCastScale, aclrtFree);
    ret = aclnnCast(workspaceAddrCastScale, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    void* workspaceAddrMatmul = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddrMatmul, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrMatmulPtr(workspaceAddrMatmul, aclrtFree);
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddrMatmul, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    void* workspaceAddrCastY = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddrCastY, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrCastYPtr(workspaceAddrCastY, aclrtFree);
    ret = aclnnCast(workspaceAddrCastY, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. aclTensor和device资源由unique_ptr自动释放
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}