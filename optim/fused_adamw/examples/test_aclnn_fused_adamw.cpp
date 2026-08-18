/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_fused_adamw.cpp
 * \brief
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_adamw.h"
#include <iostream>
#include <vector>

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> gradsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsqsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> maxexpavgsqsHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> stepsHostData1 = {3};

    std::vector<float> gradScaleOptionalHostData = {2};

    std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
    std::vector<float> gradsHostData2 = {0.9, 1.0, 1.1, 1.2};
    std::vector<float> expavgsHostData2 = {-1, -2, -3, -4};
    std::vector<float> expavgsqsHostData2 = {1, 2, 3, 4};
    std::vector<float> maxexpavgsqsHostData2 = {2, 2, 2, 2};
    std::vector<float> stepsHostData2 = {4};

    std::vector<int64_t> inputShape1 = {2, 2, 2};
    std::vector<int64_t> inputShape2 = {2, 2};
    std::vector<int64_t> scalarShape = {1};

    void* paramsRef1DeviceAddr = nullptr;
    void* grads1DeviceAddr = nullptr;
    void* expavgs1DeviceAddr = nullptr;
    void* expavgsqs1DeviceAddr = nullptr;
    void* maxexpavgsqs1DeviceAddr = nullptr;
    void* steps1DeviceAddr = nullptr;

    void* paramsRef2DeviceAddr = nullptr;
    void* grads2DeviceAddr = nullptr;
    void* expavgs2DeviceAddr = nullptr;
    void* expavgsqs2DeviceAddr = nullptr;
    void* maxexpavgsqs2DeviceAddr = nullptr;
    void* steps2DeviceAddr = nullptr;

    void* gradScaleOptionalDeviceAddr = nullptr;

    aclTensor* paramsRef1 = nullptr;
    aclTensor* grads1 = nullptr;
    aclTensor* expavgs1 = nullptr;
    aclTensor* expavgsqs1 = nullptr;
    aclTensor* maxexpavgsqs1 = nullptr;
    aclTensor* steps1 = nullptr;

    aclTensor* paramsRef2 = nullptr;
    aclTensor* grads2 = nullptr;
    aclTensor* expavgs2 = nullptr;
    aclTensor* expavgsqs2 = nullptr;
    aclTensor* maxexpavgsqs2 = nullptr;
    aclTensor* steps2 = nullptr;

    aclTensor* gradScaleOptional = nullptr;

    aclTensor* foundInfOptional = nullptr;

    ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData1, inputShape1, &grads1DeviceAddr, aclDataType::ACL_FLOAT, &grads1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData1, inputShape1, &expavgs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData1, inputShape1, &expavgsqs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData1, inputShape1, &maxexpavgsqs1DeviceAddr, aclDataType::ACL_FLOAT,
                          &maxexpavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData1, inputShape1, &steps1DeviceAddr, aclDataType::ACL_FLOAT, &steps1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData2, inputShape2, &grads2DeviceAddr, aclDataType::ACL_FLOAT, &grads2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData2, inputShape2, &expavgs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData2, inputShape2, &expavgsqs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData2, inputShape2, &maxexpavgsqs2DeviceAddr, aclDataType::ACL_FLOAT,
                          &maxexpavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData2, inputShape2, &steps2DeviceAddr, aclDataType::ACL_FLOAT, &steps2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradScaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
    std::vector<aclTensor*> gradsListData = {grads1, grads2};
    std::vector<aclTensor*> expavgsListData = {expavgs1, expavgs2};
    std::vector<aclTensor*> expavgsqsListData = {expavgsqs1, expavgsqs2};
    std::vector<aclTensor*> maxexpavgsqsData = {maxexpavgsqs1, maxexpavgsqs2};
    std::vector<aclTensor*> stepsListData = {steps1, steps2};
    aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
    aclTensorList* gradsList = aclCreateTensorList(gradsListData.data(), gradsListData.size());
    aclTensorList* expavgsList = aclCreateTensorList(expavgsListData.data(), expavgsListData.size());
    aclTensorList* expavgsqsListList = aclCreateTensorList(expavgsqsListData.data(), expavgsqsListData.size());
    aclTensorList* maxexpavgsqsList = aclCreateTensorList(maxexpavgsqsData.data(), maxexpavgsqsData.size());
    aclTensorList* stepsList = aclCreateTensorList(stepsListData.data(), stepsListData.size());

    double lr = 0.001f;
    double beta1 = 0.9f;
    double beta2 = 0.999f;
    double weightDecay = 0.0f;
    double eps = 1e-8;
    bool amsgrad = true;
    bool maximize = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnFusedAdamwGetWorkspaceSize(paramsRefList, gradsList, expavgsList, expavgsqsListList, maxexpavgsqsList,
                                          stepsList, gradScaleOptional, foundInfOptional, lr, beta1, beta2, weightDecay,
                                          eps, amsgrad, maximize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamwGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFusedAdamw(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamw failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("====== Tensor 1 paramsRef1 results ======\n");
    PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
    LOG_PRINT("====== Tensor 2 expavgs1 results ======\n");
    PrintOutResult(inputShape1, &expavgs1DeviceAddr);
    LOG_PRINT("====== Tensor 3 expavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &expavgsqs1DeviceAddr);
    LOG_PRINT("===== Tensor 4 maxexpavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &maxexpavgsqs1DeviceAddr);

    LOG_PRINT("====== Tensor 5 paramsRef results ======\n");
    PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
    LOG_PRINT("====== Tensor 6 expavgs2 results ======\n");
    PrintOutResult(inputShape2, &expavgs2DeviceAddr);
    LOG_PRINT("====== Tensor 7 expavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &expavgsqs2DeviceAddr);
    LOG_PRINT("====== Tensor 8 maxexpavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &maxexpavgsqs2DeviceAddr);

    aclDestroyTensorList(paramsRefList);
    aclDestroyTensorList(gradsList);
    aclDestroyTensorList(expavgsList);
    aclDestroyTensorList(expavgsqsListList);
    aclDestroyTensorList(maxexpavgsqsList);
    aclDestroyTensorList(stepsList);
    aclDestroyTensor(gradScaleOptional);
    aclDestroyTensor(foundInfOptional);

    aclrtFree(paramsRef1DeviceAddr);
    aclrtFree(grads1DeviceAddr);
    aclrtFree(expavgs1DeviceAddr);
    aclrtFree(expavgsqs1DeviceAddr);
    aclrtFree(maxexpavgsqs1DeviceAddr);
    aclrtFree(steps1DeviceAddr);

    aclrtFree(paramsRef2DeviceAddr);
    aclrtFree(grads2DeviceAddr);
    aclrtFree(expavgs2DeviceAddr);
    aclrtFree(expavgsqs2DeviceAddr);
    aclrtFree(maxexpavgsqs2DeviceAddr);
    aclrtFree(steps2DeviceAddr);

    aclrtFree(gradScaleOptionalDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    ret = aclrtDestroyStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed. ERROR: %d\n", ret); return ret);
    ret = aclrtResetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", ret); return ret);
    ret = aclFinalize();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", ret); return ret);
    return 0;
}
