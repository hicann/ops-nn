/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_mul_list.h"

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

void DestroyTensorList(aclTensorList* tensorList, std::vector<aclTensor*>& tensors)
{
    if (tensorList != nullptr) {
        aclDestroyTensorList(tensorList);
        tensors.clear();
        return;
    }
    for (auto tensor : tensors) {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
    tensors.clear();
}

void FreeDeviceAddrs(std::vector<void*>& deviceAddrs)
{
    for (auto addr : deviceAddrs) {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
    deviceAddrs.clear();
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
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return 0;
}

template <typename T>
bool IsSameValue(T actual, T expected)
{
    return actual == expected;
}

template <>
bool IsSameValue<float>(float actual, float expected)
{
    return std::fabs(actual - expected) <= 1e-4F;
}

template <typename T>
int CheckOutput(const std::string& caseName, const std::vector<std::vector<T>>& actual,
                const std::vector<std::vector<T>>& expected)
{
    for (size_t tensorIdx = 0; tensorIdx < expected.size(); ++tensorIdx) {
        for (size_t elemIdx = 0; elemIdx < expected[tensorIdx].size(); ++elemIdx) {
            if (!IsSameValue(actual[tensorIdx][elemIdx], expected[tensorIdx][elemIdx])) {
                LOG_PRINT("%s failed at tensor %zu element %zu, actual %d, expected %d.\n", caseName.c_str(), tensorIdx,
                          elemIdx, static_cast<int>(actual[tensorIdx][elemIdx]),
                          static_cast<int>(expected[tensorIdx][elemIdx]));
                return ACL_ERROR_FAILURE;
            }
        }
    }
    LOG_PRINT("%s passed.\n", caseName.c_str());
    return ACL_SUCCESS;
}

template <>
int CheckOutput<float>(const std::string& caseName, const std::vector<std::vector<float>>& actual,
                       const std::vector<std::vector<float>>& expected)
{
    for (size_t tensorIdx = 0; tensorIdx < expected.size(); ++tensorIdx) {
        for (size_t elemIdx = 0; elemIdx < expected[tensorIdx].size(); ++elemIdx) {
            if (!IsSameValue(actual[tensorIdx][elemIdx], expected[tensorIdx][elemIdx])) {
                LOG_PRINT("%s failed at tensor %zu element %zu, actual %f, expected %f.\n", caseName.c_str(), tensorIdx,
                          elemIdx, actual[tensorIdx][elemIdx], expected[tensorIdx][elemIdx]);
                return ACL_ERROR_FAILURE;
            }
        }
    }
    LOG_PRINT("%s passed.\n", caseName.c_str());
    return ACL_SUCCESS;
}

template <typename T>
T CastMulResult(T lhs, T rhs)
{
    return static_cast<T>(lhs * rhs);
}

template <>
int8_t CastMulResult<int8_t>(int8_t lhs, int8_t rhs)
{
    int32_t product = static_cast<int32_t>(lhs) * static_cast<int32_t>(rhs);
    uint32_t lowByte = static_cast<uint32_t>(product) & 0xFFU;
    int32_t wrappedValue = (lowByte >= 0x80U) ? static_cast<int32_t>(lowByte) - 0x100 : static_cast<int32_t>(lowByte);
    return static_cast<int8_t>(wrappedValue);
}

template <>
uint8_t CastMulResult<uint8_t>(uint8_t lhs, uint8_t rhs)
{
    uint32_t product = static_cast<uint32_t>(lhs) * static_cast<uint32_t>(rhs);
    return static_cast<uint8_t>(product & 0xFFU);
}

template <typename T>
std::vector<std::vector<T>> CalcExpected(const std::vector<std::vector<T>>& x1, const std::vector<std::vector<T>>& x2)
{
    std::vector<std::vector<T>> expected(x1.size());
    for (size_t tensorIdx = 0; tensorIdx < x1.size(); ++tensorIdx) {
        expected[tensorIdx].resize(x1[tensorIdx].size());
        for (size_t elemIdx = 0; elemIdx < x1[tensorIdx].size(); ++elemIdx) {
            expected[tensorIdx][elemIdx] = CastMulResult<T>(x1[tensorIdx][elemIdx], x2[tensorIdx][elemIdx]);
        }
    }
    return expected;
}

template <typename T>
int RunForeachMulListCase(const std::string& caseName, aclrtStream stream, aclDataType dataType,
                          const std::vector<std::vector<int64_t>>& shapes, const std::vector<std::vector<T>>& x1Data,
                          const std::vector<std::vector<T>>& x2Data)
{
    std::vector<void*> input1DeviceAddrs;
    std::vector<void*> input2DeviceAddrs;
    std::vector<void*> outputDeviceAddrs;
    std::vector<aclTensor*> input1Tensors;
    std::vector<aclTensor*> input2Tensors;
    std::vector<aclTensor*> outputTensors;
    aclTensorList* input1TensorList = nullptr;
    aclTensorList* input2TensorList = nullptr;
    aclTensorList* outputTensorList = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    auto expected = CalcExpected(x1Data, x2Data);
    for (size_t i = 0; i < shapes.size(); ++i) {
        void* input1DeviceAddr = nullptr;
        void* input2DeviceAddr = nullptr;
        void* outputDeviceAddr = nullptr;
        aclTensor* input1Tensor = nullptr;
        aclTensor* input2Tensor = nullptr;
        aclTensor* outputTensor = nullptr;
        std::vector<T> outputData(x1Data[i].size(), static_cast<T>(0));

        auto ret = CreateAclTensor(x1Data[i], shapes[i], &input1DeviceAddr, dataType, &input1Tensor);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
        ret = CreateAclTensor(x2Data[i], shapes[i], &input2DeviceAddr, dataType, &input2Tensor);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
        ret = CreateAclTensor(outputData, shapes[i], &outputDeviceAddr, dataType, &outputTensor);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

        input1DeviceAddrs.push_back(input1DeviceAddr);
        input2DeviceAddrs.push_back(input2DeviceAddr);
        outputDeviceAddrs.push_back(outputDeviceAddr);
        input1Tensors.push_back(input1Tensor);
        input2Tensors.push_back(input2Tensor);
        outputTensors.push_back(outputTensor);
    }

    input1TensorList = aclCreateTensorList(input1Tensors.data(), input1Tensors.size());
    input2TensorList = aclCreateTensorList(input2Tensors.data(), input2Tensors.size());
    outputTensorList = aclCreateTensorList(outputTensors.data(), outputTensors.size());
    CHECK_RET(input1TensorList != nullptr && input2TensorList != nullptr && outputTensorList != nullptr,
              LOG_PRINT("aclCreateTensorList failed.\n");
              goto cleanup);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    {
        // 调用aclnnForeachMulList第一段接口
        auto ret = aclnnForeachMulListGetWorkspaceSize(input1TensorList, input2TensorList, outputTensorList,
                                                       &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("%s aclnnForeachMulListGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
                  goto cleanup);
    }

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }

    {
        // 调用aclnnForeachMulList第二段接口
        auto ret = aclnnForeachMulList(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnForeachMulList failed. ERROR: %d\n", caseName.c_str(), ret);
                  goto cleanup);
        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto cleanup);
    }

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    {
        std::vector<std::vector<T>> actual(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            actual[i].resize(expected[i].size());
            auto copySize = static_cast<int64_t>(actual[i].size() * sizeof(T));
            auto ret = aclrtMemcpy(actual[i].data(), copySize, outputDeviceAddrs[i], copySize,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                      goto cleanup);
        }
        auto ret = CheckOutput(caseName, actual, expected);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    // 7.释放device资源，需要根据具体API的接口定义修改
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensorList(input1TensorList, input1Tensors);
    DestroyTensorList(input2TensorList, input2Tensors);
    DestroyTensorList(outputTensorList, outputTensors);
    FreeDeviceAddrs(input1DeviceAddrs);
    FreeDeviceAddrs(input2DeviceAddrs);
    FreeDeviceAddrs(outputDeviceAddrs);
    return ACL_SUCCESS;

cleanup:
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensorList(input1TensorList, input1Tensors);
    DestroyTensorList(input2TensorList, input2Tensors);
    DestroyTensorList(outputTensorList, outputTensors);
    FreeDeviceAddrs(input1DeviceAddrs);
    FreeDeviceAddrs(input2DeviceAddrs);
    FreeDeviceAddrs(outputDeviceAddrs);
    return ACL_ERROR_FAILURE;
}

int RunCases(aclrtStream stream)
{
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 3}, {1, 3}};
        std::vector<std::vector<float>> x1 = {{1, 2, 3, 4, 5, 6}, {7, 8, 9}};
        std::vector<std::vector<float>> x2 = {{6, 5, 4, 3, 2, 1}, {9, 8, 7}};
        auto ret = RunForeachMulListCase("float32_multi_tensor", stream, ACL_FLOAT, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 3}, {1, 3}};
        std::vector<std::vector<int16_t>> x1 = {{301, 301, 301, 301, 301, 301}, {12, -12, 7}};
        std::vector<std::vector<int16_t>> x2 = {{100, 100, 100, 100, 100, 100}, {3, 3, -4}};
        auto ret = RunForeachMulListCase("int16_multi_tensor", stream, ACL_INT16, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 3}, {1, 3}};
        std::vector<std::vector<int8_t>> x1 = {{7, 7, 7, 7, 7, 7}, {2, -2, 3}};
        std::vector<std::vector<int8_t>> x2 = {{-3, -3, -3, -3, -3, -3}, {4, 4, -5}};
        auto ret = RunForeachMulListCase("int8_multi_tensor", stream, ACL_INT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 3}, {1, 3}};
        std::vector<std::vector<uint8_t>> x1 = {{7, 7, 7, 7, 7, 7}, {2, 3, 4}};
        std::vector<std::vector<uint8_t>> x2 = {{3, 3, 3, 3, 3, 3}, {4, 5, 6}};
        auto ret = RunForeachMulListCase("uint8_multi_tensor", stream, ACL_UINT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 4}, {1, 4}};
        std::vector<std::vector<int16_t>> x1 = {{257, -257, 511, -511, 1023, -1023, 181, -181}, {123, -123, 255, -255}};
        std::vector<std::vector<int16_t>> x2 = {{121, 121, 63, 63, 31, 31, 181, 181}, {257, 257, 127, 127}};
        auto ret = RunForeachMulListCase("int16_large_precision_tensor", stream, ACL_INT16, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 4}, {1, 4}};
        std::vector<std::vector<int8_t>> x1 = {{127, 127, -128, -128, 123, -127, 100, -100}, {9, -9, 64, -64}};
        std::vector<std::vector<int8_t>> x2 = {{127, -128, 127, -128, 123, -127, -100, 100}, {-127, -127, 4, 4}};
        auto ret = RunForeachMulListCase("int8_overflow_tensor", stream, ACL_INT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{2, 4}, {1, 4}};
        std::vector<std::vector<uint8_t>> x1 = {{255, 255, 254, 253, 251, 240, 200, 129}, {9, 127, 128, 255}};
        std::vector<std::vector<uint8_t>> x2 = {{255, 254, 253, 251, 249, 240, 200, 129}, {229, 3, 3, 2}};
        auto ret = RunForeachMulListCase("uint8_overflow_tensor", stream, ACL_UINT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{1}, {1, 1}, {2, 3}};
        std::vector<std::vector<int16_t>> x1 = {{10}, {-11}, {1, 2, 3, 4, 5, 6}};
        std::vector<std::vector<int16_t>> x2 = {{-3}, {-2}, {6, 5, 4, 3, 2, 1}};
        auto ret = RunForeachMulListCase("int16_small_shape", stream, ACL_INT16, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{1}, {1, 1}, {2, 3}};
        std::vector<std::vector<int8_t>> x1 = {{10}, {-11}, {1, 2, 3, 4, 5, 6}};
        std::vector<std::vector<int8_t>> x2 = {{-3}, {-2}, {6, 5, 4, 3, 2, 1}};
        auto ret = RunForeachMulListCase("int8_small_shape", stream, ACL_INT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<std::vector<int64_t>> shapes = {{1}, {1, 1}, {2, 3}};
        std::vector<std::vector<uint8_t>> x1 = {{10}, {11}, {1, 2, 3, 4, 5, 6}};
        std::vector<std::vector<uint8_t>> x2 = {{3}, {2}, {6, 5, 4, 3, 2, 1}};
        auto ret = RunForeachMulListCase("uint8_small_shape", stream, ACL_UINT8, shapes, x1, x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = RunCases(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
