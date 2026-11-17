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
#include "aclnnop/aclnn_foreach_minimum_scalar_list.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape1 = {2, 3};
    std::vector<int64_t> xShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclScalar* scalar1 = nullptr;
    aclScalar* scalar2 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    std::vector<float> x1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> x2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);
    float scalar1Value = 4.2f;
    float scalar2Value = 8.2f;
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape1, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape2, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scalar1 aclScalar
    scalar1 = aclCreateScalar(&scalar1Value, aclDataType::ACL_FLOAT);
    CHECK_RET(scalar1 != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // 创建scalar2 aclScalar
    scalar2 = aclCreateScalar(&scalar2Value, aclDataType::ACL_FLOAT);
    CHECK_RET(scalar2 != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<aclTensor*> tempX{x1, x2};
    aclTensorList* tensorListX = aclCreateTensorList(tempX.data(), tempX.size());
    std::vector<aclTensor*> tempOutput{out1, out2};
    aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());
    std::vector<aclScalar*> tempscalar{scalar1, scalar2};
    aclScalarList* scalarlist = aclCreateScalarList(tempscalar.data(), tempscalar.size());

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnForeachMinimumScalarList第一段接口
    ret = aclnnForeachMinimumScalarListGetWorkspaceSize(tensorListX, scalarlist, tensorListOutput, &workspaceSize,
                                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMinimumScalarListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnForeachMinimumScalarList第二段接口
    ret = aclnnForeachMinimumScalarList(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMinimumScalarList failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape1);
    std::vector<float> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
    }

    size = GetShapeSize(outShape2);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(tensorListX);
    aclDestroyTensorList(tensorListOutput);
    aclDestroyScalarList(scalarlist);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(out1DeviceAddr);
    aclrtFree(out2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
