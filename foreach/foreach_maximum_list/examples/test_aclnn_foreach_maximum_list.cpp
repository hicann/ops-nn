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
#include "aclnnop/aclnn_foreach_maximum_list.h"

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
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        // memcpy失败时释放刚申请的device内存，避免调用方无从回收
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

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

// 统一持有device资源，析构时按申请情况释放，避免错误分支遗漏释放。
struct AclResources {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool inited = false;
    void* workspaceAddr = nullptr;
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* other1DeviceAddr = nullptr;
    void* other2DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    aclTensorList* tensorListInput1 = nullptr;
    aclTensorList* tensorListInput2 = nullptr;
    aclTensorList* tensorListOutput = nullptr;

    ~AclResources()
    {
        if (tensorListInput1 != nullptr) {
            aclDestroyTensorList(tensorListInput1);
        }
        if (tensorListInput2 != nullptr) {
            aclDestroyTensorList(tensorListInput2);
        }
        if (tensorListOutput != nullptr) {
            aclDestroyTensorList(tensorListOutput);
        }
        if (input1DeviceAddr != nullptr) {
            aclrtFree(input1DeviceAddr);
        }
        if (input2DeviceAddr != nullptr) {
            aclrtFree(input2DeviceAddr);
        }
        if (other1DeviceAddr != nullptr) {
            aclrtFree(other1DeviceAddr);
        }
        if (other2DeviceAddr != nullptr) {
            aclrtFree(other2DeviceAddr);
        }
        if (out1DeviceAddr != nullptr) {
            aclrtFree(out1DeviceAddr);
        }
        if (out2DeviceAddr != nullptr) {
            aclrtFree(out2DeviceAddr);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (inited) {
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
        }
    }
};

int main()
{
    AclResources res;
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    res.deviceId = 0;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* other1 = nullptr;
    aclTensor* other2 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;

    auto ret = Init(res.deviceId, &res.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    res.inited = true;

    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> otherShape1 = {2, 3};
    std::vector<int64_t> otherShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> other1HostData = {6, 5, 4, 3, 2, 1};
    std::vector<float> other2HostData = {9, 8, 7};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);
    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, &res.input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, &res.input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other1 aclTensor
    ret = CreateAclTensor(other1HostData, otherShape1, &res.other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other2 aclTensor
    ret = CreateAclTensor(other2HostData, otherShape2, &res.other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, &res.out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, &res.out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> tempInput1{input1, input2};
    res.tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
    std::vector<aclTensor*> tempInput2{other1, other2};
    res.tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
    std::vector<aclTensor*> tempOutput{out1, out2};
    res.tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnForeachMaximumList第一段接口
    ret = aclnnForeachMaximumListGetWorkspaceSize(res.tensorListInput1, res.tensorListInput2, res.tensorListOutput,
                                                  &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnForeachMaximumList第二段接口
    ret = aclnnForeachMaximumList(res.workspaceAddr, workspaceSize, executor, res.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumList failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(res.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape1);
    std::vector<float> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), res.out1DeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
    }

    size = GetShapeSize(outShape2);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), res.out2DeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
    }

    // 6.资源由AclResources析构统一释放
    return 0;
}
