/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant.h"

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

// 在正常返回和错误返回时释放已创建的资源。
struct ScopeCleanup {
    std::function<void()> release;
    ~ScopeCleanup() { release(); }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);

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

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); aclrtFree(*deviceAddr); *deviceAddr = nullptr;
              return ACL_ERROR_FAILURE);
    return 0;
}

template <typename T>
float OutputValue(T value)
{
    return static_cast<float>(value);
}

float OutputValue(short value) { return aclFloat16ToFloat(static_cast<aclFloat16>(value)); }

template <typename T>
bool CheckOutput(const char* name, void* deviceAddr, std::vector<T>& hostData, float expected, float tolerance = 0.0f)
{
    auto bytes = hostData.size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData.data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s failed. ERROR: %d\n", name, ret); return false);
    for (size_t i = 0; i < hostData.size(); ++i) {
        float actual = OutputValue(hostData[i]);
        if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
            LOG_PRINT("%s[%zu]: actual=%g, expected=%g, tolerance=%g\n", name, i, actual, expected, tolerance);
            return false;
        }
    }
    LOG_PRINT("%s: checked %zu elements, expected=%g, PASS\n", name, hostData.size(), expected);
    return true;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> reduceShape = {2};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* smooth1DeviceAddr = nullptr;
    void* smooth2DeviceAddr = nullptr;

    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* smooth1 = nullptr;
    aclTensor* smooth2 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* scale2 = nullptr;

    void* workspaceAddr = nullptr;
    ScopeCleanup cleanup{[&]() {
        // 释放设备内存前等待已提交的异步任务结束。
        aclrtSynchronizeStream(stream);
        for (auto* tensor : {x1, x2, gamma, smooth1, smooth2, y1, y2, x, scale1, scale2}) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
            }
        }
        for (auto* addr :
             {x1DeviceAddr, x2DeviceAddr, gammaDeviceAddr, smooth1DeviceAddr, smooth2DeviceAddr, y1DeviceAddr,
              y2DeviceAddr, xDeviceAddr, scale1DeviceAddr, scale2DeviceAddr, workspaceAddr}) {
            if (addr != nullptr) {
                aclrtFree(addr);
            }
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t reduceShapeSize = GetShapeSize(reduceShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth1HostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth2HostData(gammaShapeSize, 0x3e00);

    std::vector<int8_t> y1HostData(xShapeSize, 0);
    std::vector<int8_t> y2HostData(xShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);
    std::vector<float> scale1HostData(reduceShapeSize, 0.0f);
    std::vector<float> scale2HostData(reduceShapeSize, 0.0f);

    float epsilon = 1e-6;

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 smooth1 aclTensor
    ret = CreateAclTensor(smooth1HostData, gammaShape, &smooth1DeviceAddr, aclDataType::ACL_FLOAT16, &smooth1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 smooth2 aclTensor
    ret = CreateAclTensor(smooth2HostData, gammaShape, &smooth2DeviceAddr, aclDataType::ACL_FLOAT16, &smooth2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y1 aclTensor
    ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2 aclTensor
    ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outScale1 aclTensor
    ret = CreateAclTensor(scale1HostData, reduceShape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outScale2 aclTensor
    ret = CreateAclTensor(scale2HostData, reduceShape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAddRmsNormDynamicQuant第一段接口
    ret = aclnnAddRmsNormDynamicQuantGetWorkspaceSize(x1, x2, gamma, smooth1, smooth2, epsilon, y1, y2, x, scale1,
                                                      scale2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnAddRmsNormDynamicQuant第二段接口
    ret = aclnnAddRmsNormDynamicQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 拷回并检查全部有效输出。当前输入为x1=x2=0.5，gamma=smooth1=smooth2=1.5。
    // 因此x=1，两路量化结果均为127，scale=2.25/sqrt(1+epsilon)/127。
    float expectedScale = 2.25f / std::sqrt(1.0f + epsilon) / 127.0f;
    float scaleTolerance = expectedScale / 8192.0f;
    bool outputsMatch = CheckOutput("y1", y1DeviceAddr, y1HostData, 127.0f);
    outputsMatch &= CheckOutput("y2", y2DeviceAddr, y2HostData, 127.0f);
    outputsMatch &= CheckOutput("x", xDeviceAddr, xHostData, 1.0f);
    outputsMatch &= CheckOutput("scale1", scale1DeviceAddr, scale1HostData, expectedScale, scaleTolerance);
    outputsMatch &= CheckOutput("scale2", scale2DeviceAddr, scale2HostData, expectedScale, scaleTolerance);

    return outputsMatch ? 0 : 1;
}
