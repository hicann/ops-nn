/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_hard_shrink.cpp
 * @brief HardShrink 算子 aclnn 两段式调用示例
 *
 * 功能：对输入张量执行 HardShrink 激活函数
 * 公式：HardShrink(x) = x, if x > lambd
 *                        x, if x < -lambd
 *                        0, otherwise
 *
 * 本示例展示如何通过 aclnn 接口调用自定义 HardShrink 算子。
 * 资源管理：使用 std::unique_ptr + 自定义 deleter（RAII）封装 ACL 资源，
 * 函数任意路径 return 时保证 tensor/device memory/stream/device/acl 正确释放。
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn_hard_shrink.h"

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

// ---------- RAII 封装：自定义 deleter + unique_ptr ----------
struct AclTensorDeleter {
    void operator()(aclTensor* p) const { if (p != nullptr) { aclDestroyTensor(p); } }
};
struct DeviceMemDeleter {
    void operator()(void* p) const { if (p != nullptr) { aclrtFree(p); } }
};
struct StreamDeleter {
    void operator()(aclrtStream s) const { if (s != nullptr) { aclrtDestroyStream(s); } }
};
using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;
using DeviceMemPtr = std::unique_ptr<void, DeviceMemDeleter>;
using StreamPtr    = std::unique_ptr<std::remove_pointer_t<aclrtStream>, StreamDeleter>;

// deviceId/aclInit 需按顺序释放，封装成带析构的 guard
struct AclDeviceGuard {
    int32_t deviceId{-1};
    bool aclInitOk{false};
    ~AclDeviceGuard()
    {
        if (deviceId >= 0) {
            aclrtResetDevice(deviceId);
        }
        if (aclInitOk) {
            aclFinalize();
        }
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, DeviceMemPtr& deviceAddr,
    aclDataType dataType, AclTensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    void* rawAddr = nullptr;
    auto ret = aclrtMalloc(&rawAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawAddr);

    ret = aclrtMemcpy(rawAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    aclTensor* rawTensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), rawAddr);
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor returned nullptr\n"); return -1);
    tensor.reset(rawTensor);
    return 0;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void PrintResult(const std::vector<int64_t>& shape, void* deviceAddr,
                 const std::vector<float>& inputData, double lambd)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(float), deviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);

    LOG_PRINT("HardShrink results (lambd=%.2f):\n", lambd);
    for (int64_t i = 0; i < size && i < 16; i++) {
        LOG_PRINT("  input[%ld] = %f  =>  output[%ld] = %f\n", i, inputData[i], i, resultData[i]);
    }
    if (size > 16) {
        LOG_PRINT("  ... (%ld elements total)\n", size);
    }
}

int main()
{
    // 1. ACL 初始化（由 AclDeviceGuard 在函数退出时统一释放）
    AclDeviceGuard aclGuard;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    aclGuard.aclInitOk = true;

    int32_t deviceId = 0;
    aclrtStream rawStream = nullptr;
    ret = Init(deviceId, &rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclGuard.deviceId = deviceId;
    StreamPtr stream(rawStream);

    // 2. 构造输入和输出
    std::vector<int64_t> selfShape = {4, 4};
    std::vector<float> selfHostData = {
         1.0f,  -1.0f,  0.3f,  -0.3f,
         0.5f,  -0.5f,  0.0f,   2.0f,
        -2.0f,   0.1f, -0.1f,  10.0f,
         0.49f, -0.49f, 0.51f, -0.51f
    };

    AclTensorPtr self;
    DeviceMemPtr selfDeviceAddr;
    ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    AclTensorPtr out;
    DeviceMemPtr outDeviceAddr;
    std::vector<float> outHostData(16, 0.0f);
    ret = CreateAclTensor(outHostData, selfShape, outDeviceAddr, aclDataType::ACL_FLOAT, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double lambd = 0.5;

    // 3. 调用 aclnnHardShrink 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHardShrinkGetWorkspaceSize(self.get(), lambd, out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnHardShrinkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 申请 workspace
    DeviceMemPtr workspaceAddr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWs = nullptr;
        ret = aclrtMalloc(&rawWs, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWs);
    }

    // 5. 调用 aclnnHardShrink 第二段接口
    ret = aclnnHardShrink(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardShrink failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待计算完成
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 打印输出结果
    PrintResult(selfShape, outDeviceAddr.get(), selfHostData, lambd);

    // 8. 资源由 unique_ptr / AclDeviceGuard 自动释放
    return 0;
}
