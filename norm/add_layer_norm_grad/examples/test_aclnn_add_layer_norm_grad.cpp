/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_add_layer_norm_grad.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> x1Shape = {3, 1, 4};
    std::vector<int64_t> x2Shape = {3, 1, 4};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> dsumOptionalShape = {3, 1, 4};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgammaShape = {4};
    std::vector<int64_t> outputpdbetaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* dsumOptionalDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* dsumOptional = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgamma = nullptr;
    aclTensor* outputpdbeta = nullptr;
    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x1HostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x2HostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<int32_t> rstdHostData = {0, 1, 2};
    std::vector<int32_t> meanHostData = {0, 1, 2};
    std::vector<int32_t> gammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> dsumOptionalHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdgammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> outputpdbetaHostData = {0, 1, 2, 3};

    // 创建self aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        dsumOptionalHostData, dsumOptionalShape, &dsumOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dsumOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnAddLayerNormGrad接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnAddLayerNormGrad第一段接口
    LOG_PRINT("\nUse aclnnAddLayerNormGrad Port.");
    ret = aclnnAddLayerNormGradGetWorkspaceSize(
        dy, x1, x2, rstd, mean, gamma, dsumOptional, outputpdx, outputpdgamma, outputpdbeta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNormGrad第二段接口
    ret = aclnnAddLayerNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto outputpdxsize = GetShapeSize(outputpdxShape);
    std::vector<float> resultDataPdx(outputpdxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdx.data(), resultDataPdx.size() * sizeof(resultDataPdx[0]), outputpdxDeviceAddr,
        outputpdxsize * sizeof(resultDataPdx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputpdxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdx[i]);
    }
    auto outputpdgammasize = GetShapeSize(outputpdgammaShape);
    std::vector<float> resultDataPdGamma(outputpdgammasize, 0);
    ret = aclrtMemcpy(
        resultDataPdGamma.data(), resultDataPdGamma.size() * sizeof(resultDataPdGamma[0]), outputpdgammaDeviceAddr,
        outputpdgammasize * sizeof(resultDataPdGamma[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgamma output");
    for (int64_t i = 0; i < outputpdgammasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdGamma[i]);
    }
    auto outputpdbetasize = GetShapeSize(outputpdbetaShape);
    std::vector<float> resultDataPdBeta(outputpdbetasize, 0);
    ret = aclrtMemcpy(
        resultDataPdBeta.data(), resultDataPdBeta.size() * sizeof(resultDataPdBeta[0]), outputpdbetaDeviceAddr,
        outputpdbetasize * sizeof(resultDataPdBeta[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdbeta output");
    for (int64_t i = 0; i < outputpdbetasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdBeta[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(rstd);
    aclDestroyTensor(mean);
    aclDestroyTensor(gamma);
    aclDestroyTensor(dsumOptional);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgamma);
    aclDestroyTensor(outputpdbeta);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(dyDeviceAddr);
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(dsumOptionalDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}