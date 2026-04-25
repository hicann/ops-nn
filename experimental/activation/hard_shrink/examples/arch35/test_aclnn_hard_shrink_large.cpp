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
 * @file test_aclnn_hard_shrink_large.cpp
 * @brief HardShrink 算子大 Tensor 验证示例
 *
 * 覆盖场景：shape=[1024, 1024] 共 1M 元素，FP32 数据类型
 *   - 触发多核切分（totalNum / coreNum > 1）
 *   - 触发 UB 分片多次循环（ubFactor < totalNum/coreNum）
 *   - 触发双缓冲路径（BUFFER_MODE=1，totalLength > 1024）
 * 校验方式：与 CPU 参考值比对，统计 PASS/FAIL
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

    // shape=[1024, 1024]，共 1M 元素
    std::vector<int64_t> selfShape = {1024, 1024};
    int64_t totalNum = GetShapeSize(selfShape);
    double lambd = 0.3;

    // 生成确定性输入：x[i] = (i % 1000 - 500) / 500.0  ∈ [-1.0, 1.0)
    // 既覆盖 |x| > lambd 又覆盖 |x| <= lambd
    std::vector<float> selfHostData(totalNum);
    for (int64_t i = 0; i < totalNum; i++) {
        selfHostData[i] = static_cast<float>((i % 1000) - 500) / 500.0f;
    }

    AclTensorPtr self;
    DeviceMemPtr selfDeviceAddr;
    ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    AclTensorPtr out;
    DeviceMemPtr outDeviceAddr;
    std::vector<float> outHostData(totalNum, 0.0f);
    ret = CreateAclTensor(outHostData, selfShape, outDeviceAddr, aclDataType::ACL_FLOAT, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

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

    std::vector<float> resultData(totalNum, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                      outDeviceAddr.get(), totalNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("HardShrink Large Tensor (shape=[%ld,%ld], total=%ld, lambd=%.2f):\n",
              selfShape[0], selfShape[1], totalNum, lambd);

    int64_t failCount = 0;
    int64_t nonZeroCount = 0;
    int64_t firstFailIdx = -1;
    for (int64_t i = 0; i < totalNum; i++) {
        float x = selfHostData[i];
        float expected = (std::fabs(x) > static_cast<float>(lambd)) ? x : 0.0f;
        float actual = resultData[i];
        if (std::fabs(actual - expected) > 1e-6f) {
            if (firstFailIdx < 0) { firstFailIdx = i; }
            failCount++;
        }
        if (actual != 0.0f) { nonZeroCount++; }
    }

    LOG_PRINT("  non-zero outputs: %ld / %ld (expect roughly 40%% > lambd=0.3)\n", nonZeroCount, totalNum);
    if (failCount == 0) {
        LOG_PRINT("  PASS: all %ld elements match reference\n", totalNum);
    } else {
        LOG_PRINT("  FAIL: %ld elements mismatch, first at idx=%ld (input=%f, actual=%f, expected=%f)\n",
                  failCount, firstFailIdx, selfHostData[firstFailIdx],
                  resultData[firstFailIdx],
                  (std::fabs(selfHostData[firstFailIdx]) > static_cast<float>(lambd))
                      ? selfHostData[firstFailIdx] : 0.0f);
    }

    return (failCount == 0) ? 0 : 1;
}
