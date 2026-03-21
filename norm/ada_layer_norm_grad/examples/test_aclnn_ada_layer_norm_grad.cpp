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
#include "aclnnop/aclnn_ada_layer_norm_backward.h"

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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
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
    std::vector<int64_t> xShape = {2, 2, 2};
    std::vector<int64_t> meanShape = {2, 2, 1};
    std::vector<int64_t> scaleShape = {2, 2};
    std::vector<int64_t> shiftShape = {2, 2};
    std::vector<int64_t> normShape = {2};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* dtDeviceAddr = nullptr;
    void* dwDeviceAddr = nullptr;
    void* dbDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* ds = nullptr;
    aclTensor* dt = nullptr;
    aclTensor* dw = nullptr;
    aclTensor* db = nullptr;
    std::vector<float> dyHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<float> xHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<int64_t> normData = {2};
    std::vector<float> meanHostData = {2, 3, 4, 5};
    std::vector<float> rstdHostData = {4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 1, 2, 2};
    std::vector<float> shiftHostData = {2, 2, 1, 1};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> outHostData(8, 0);
    std::vector<float> dsHostData(4, 0);
    std::vector<float> dtHostData(4, 0);
    std::vector<float> dwHostData(2, 0);
    std::vector<float> dbHostData(2, 0);

    // 创建dy aclTensor
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建normalizedShape aclIntArray
    norm = aclCreateIntArray(normData.data(), 1);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建shift aclTensor
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ds aclTensor
    ret = CreateAclTensor(dsHostData, scaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT, &ds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dt aclTensor
    ret = CreateAclTensor(dtHostData, shiftShape, &dtDeviceAddr, aclDataType::ACL_FLOAT, &dt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dw aclTensor
    ret = CreateAclTensor(dwHostData, normShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建db aclTensor
    ret = CreateAclTensor(dbHostData, normShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAdaLayerNormBackward第一段接口
    ret = aclnnAdaLayerNormBackwardGetWorkspaceSize(
        dy, x, norm, rstd, mean, scale, shift, weight, bias, out, ds, dt, dw, db, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnAdaLayerNormBackward第二段接口
    ret = aclnnAdaLayerNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(scaleShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dsDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("ds result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(shiftShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dtDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("dt result[%ld] is: %f\n", i, resultData2[i]);
    }

    auto size3 = GetShapeSize(normShape);
    std::vector<float> resultData3(size3, 0);
    ret = aclrtMemcpy(
        resultData3.data(), resultData3.size() * sizeof(resultData3[0]), dwDeviceAddr, size3 * sizeof(resultData3[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size3; i++) {
        LOG_PRINT("dw result[%ld] is: %f\n", i, resultData3[i]);
    }

    auto size4 = GetShapeSize(normShape);
    std::vector<float> resultData4(size4, 0);
    ret = aclrtMemcpy(
        resultData4.data(), resultData4.size() * sizeof(resultData4[0]), dbDeviceAddr, size4 * sizeof(resultData4[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size4; i++) {
        LOG_PRINT("db result[%ld] is: %f\n", i, resultData4[i]);
    }

    // 6. 释放aclTensor、aclIntArray和aclBoolArray，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(ds);
    aclDestroyTensor(dt);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);

    // 7. 释放device 资源
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(dtDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(dbDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}