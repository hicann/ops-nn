/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i != 0 && shapeSize > INT64_MAX / i) {
            return -1;
        }
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    *stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

class AclEnvironmentGuard {
public:
    AclEnvironmentGuard(int32_t deviceId, aclrtStream stream) : deviceId_(deviceId), stream_(stream) {}

    ~AclEnvironmentGuard()
    {
        aclrtDestroyStream(stream_);
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

private:
    int32_t deviceId_;
    aclrtStream stream_;
};

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const int64_t shapeSize = GetShapeSize(shape);
    if (shapeSize <= 0 || hostData.size() != static_cast<size_t>(shapeSize) ||
        static_cast<uint64_t>(shapeSize) > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const size_t size = static_cast<size_t>(shapeSize) * sizeof(T);
    *deviceAddr = nullptr;
    *tensor = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

int CopyAndPrintOutput(const char* name, std::vector<uint8_t>& hostData, void* deviceAddr)
{
    auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(uint8_t), deviceAddr,
                           hostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    constexpr size_t PRINT_COUNT = 10;
    std::printf("%s first 10 elements:\n", name);
    for (size_t i = 0; i < PRINT_COUNT && i < hostData.size(); i++) {
        std::printf("%s[%zu] = %u\n", name, i, static_cast<unsigned int>(hostData[i]));
    }
    return ACL_SUCCESS;
}

int aclnnGroupedDynamicMxQuantWithDualAxisTest(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    AclEnvironmentGuard environmentGuard(deviceId, stream);

    // x shape: [M, N] = [256, 1024], N must be 64-aligned
    constexpr int64_t M = 256;
    constexpr int64_t N = 1024;
    constexpr int64_t BLOCK_SIZE = 32;
    constexpr int64_t SCALE_PAIR = 2;
    constexpr int64_t SCALE_AXIS_UNIT = BLOCK_SIZE * SCALE_PAIR;
    constexpr int64_t G = 4; // 4 groups, each has 64 rows

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> groupIndexShape = {G};
    std::vector<int64_t> y1ScaleOutShape = {M, N / SCALE_AXIS_UNIT, SCALE_PAIR};
    std::vector<int64_t> y2ScaleOutShape = {M / SCALE_AXIS_UNIT + G, N, SCALE_PAIR};

    // groupIndex: cumsum, last element must equal M
    std::vector<int64_t> groupIndexHostData = {64, 128, 192, 256};

    // x data: BF16 (uint16_t storage), fill with deterministic pattern
    std::vector<uint16_t> xHostData(M * N, 0);
    for (int64_t i = 0; i < M * N; i++) {
        xHostData[i] = static_cast<uint16_t>((i % 200) - 100);
    }

    // output buffers
    std::vector<uint8_t> y1OutHostData(M * N, 0);
    std::vector<uint8_t> y2OutHostData(M * N, 0);
    std::vector<uint8_t> y1ScaleOutHostData(GetShapeSize(y1ScaleOutShape), 0);
    std::vector<uint8_t> y2ScaleOutHostData(GetShapeSize(y2ScaleOutShape), 0);

    // attributes
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t scaleAlg = 1;  // Ceil rounding algorithm
    int64_t dstDtype = 36; // FLOAT8_E4M3FN
    double maxDtypeValue = 0.0;

    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* y1ScaleOutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* y2ScaleOutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* y1ScaleOut = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* y2ScaleOut = nullptr;

    // create x (BF16)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create groupIndex (INT64)
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y1OutHostData, xShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y1ScaleOutHostData, y1ScaleOutShape, &y1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y1ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1ScaleOutTensorPtr(y1ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1ScaleOutDeviceAddrPtr(y1ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y2OutHostData, xShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y2ScaleOutHostData, y2ScaleOutShape, &y2ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y2ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2ScaleOutTensorPtr(y2ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2ScaleOutDeviceAddrPtr(y2ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // first-stage API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, roundModeOptional, scaleAlg, dstDtype,
                                                                 maxDtypeValue, y1Out, y1ScaleOut, y2Out, y2ScaleOut,
                                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // allocate workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // second-stage API
    ret = aclnnGroupedDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // sync
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CopyAndPrintOutput("y1Out", y1OutHostData, y1OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y1ScaleOut", y1ScaleOutHostData, y1ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2Out", y2OutHostData, y2OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2ScaleOut", y2ScaleOutHostData, y2ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

int main() { return aclnnGroupedDynamicMxQuantWithDualAxisTest(0); }
