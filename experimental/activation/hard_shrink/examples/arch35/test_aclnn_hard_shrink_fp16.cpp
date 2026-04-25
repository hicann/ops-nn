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
 * @file test_aclnn_hard_shrink_fp16.cpp
 * @brief HardShrink 算子 FP16 数据类型验证示例
 *
 * 覆盖场景：ACL_FLOAT16 输入/输出，lambd=0.5
 * 校验方式：与 CPU 参考值比对，输出 PASS/FAIL
 * 资源管理：使用 std::unique_ptr + 自定义 deleter（RAII），任意路径 return 都安全释放
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
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

int main()
{
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

    // shape=[4, 4]，FP16 数据类型
    std::vector<int64_t> selfShape = {4, 4};
    std::vector<float> selfFloat = {
         1.0f,  -1.0f,  0.3f,  -0.3f,
         0.5f,  -0.5f,  0.0f,   2.0f,
        -2.0f,   0.1f, -0.1f,  10.0f,
         0.49f, -0.49f, 0.51f, -0.51f
    };
    std::vector<aclFloat16> selfHostData(selfFloat.size());
    for (size_t i = 0; i < selfFloat.size(); i++) {
        selfHostData[i] = aclFloatToFloat16(selfFloat[i]);
    }

    AclTensorPtr self;
    DeviceMemPtr selfDeviceAddr;
    ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT16, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    AclTensorPtr out;
    DeviceMemPtr outDeviceAddr;
    std::vector<aclFloat16> outHostData(selfHostData.size(), aclFloatToFloat16(0.0f));
    ret = CreateAclTensor(outHostData, selfShape, outDeviceAddr, aclDataType::ACL_FLOAT16, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double lambd = 0.5;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHardShrinkGetWorkspaceSize(self.get(), lambd, out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnHardShrinkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    DeviceMemPtr workspaceAddr;
    if (workspaceSize > 0) {
        void* rawWs = nullptr;
        ret = aclrtMalloc(&rawWs, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWs);
    }

    ret = aclnnHardShrink(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardShrink failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷回结果并校验
    auto size = GetShapeSize(selfShape);
    std::vector<aclFloat16> resultData(size);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16),
                      outDeviceAddr.get(), size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("HardShrink FP16 results (lambd=%.2f):\n", lambd);
    int64_t passCount = 0;
    for (int64_t i = 0; i < size; i++) {
        float x = selfFloat[i];
        float expected = (std::fabs(x) > static_cast<float>(lambd)) ? x : 0.0f;
        float actual = aclFloat16ToFloat(resultData[i]);
        // FP16 精度容差
        bool ok = std::fabs(actual - expected) <= 1e-2f;
        if (ok) { passCount++; }
        LOG_PRINT("  input[%ld]=%7.4f  =>  output=%7.4f  expected=%7.4f  [%s]\n",
                  i, x, actual, expected, ok ? "PASS" : "FAIL");
    }
    LOG_PRINT("Summary: %ld/%ld passed\n", passCount, size);

    return (passCount == size) ? 0 : 1;
}
