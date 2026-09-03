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
 * \file test_aclnn_huber_loss.cpp
 * \brief aclnn calling sample: one call per reduction, checked against a host reference
 *
 * The inputs -2, -1, -0.5, 0, 0.5, 1, 2 with delta = 1.0 cover both branches
 * of the definition and the knee at |e| = delta.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_huber_loss.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(fmt, ...)              \
    do {                                 \
        std::printf(fmt, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

int CreateAclTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclTensor** tensor)
{
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

float GoldenElem(float p, float t, float delta)
{
    const float a = std::fabs(p - t);
    return (a <= delta) ? (0.5f * a * a) : (delta * (a - 0.5f * delta));
}

// One call per reduction. reduction=0 gives an output shaped like the input;
// reduction=1/2 gives a rank-0 scalar -- an empty shape holding one element.
// The first-stage interface checks this, so getting it wrong is refused
// rather than silently written past the end of the allocation.
int RunOne(aclrtStream stream, int64_t reduction, const char* name, const std::vector<float>& inputHost,
           const std::vector<float>& targetHost, double delta)
{
    const std::vector<int64_t> shape = {static_cast<int64_t>(inputHost.size())};
    const bool scalarOut = (reduction != REDUCTION_NONE);
    const std::vector<int64_t> outShape = scalarOut ? std::vector<int64_t>{} : shape;
    const size_t outElems = scalarOut ? 1U : inputHost.size();

    void* inputDevice = nullptr;
    void* targetDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* out = nullptr;
    void* workspace = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> outHost(outElems, 0.0f);
    int ret = 0;

    do {
        ret = CreateAclTensor(inputHost, shape, &inputDevice, &input);
        CHECK_RET(ret == ACL_SUCCESS, break);
        ret = CreateAclTensor(targetHost, shape, &targetDevice, &target);
        CHECK_RET(ret == ACL_SUCCESS, break);
        ret = CreateAclTensor(outHost, outShape, &outDevice, &out);
        CHECK_RET(ret == ACL_SUCCESS, break);

        ret = aclnnHuberLossGetWorkspaceSize(input, target, reduction, delta, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHuberLossGetWorkspaceSize failed: %d\n", ret); break);

        // The reduced modes need workspace for the cross-core partial sums and
        // reduction=0 does not, but the returned size also carries the
        // framework's reserved region, so it is non-zero either way. Use what
        // the first stage returns rather than inferring it from reduction.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, break);
        }
        ret = aclnnHuberLoss(workspace, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHuberLoss failed: %d\n", ret); break);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, break);

        const size_t bytes = outElems * sizeof(float);
        ret = aclrtMemcpy(outHost.data(), bytes, outDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, break);
    } while (false);

    if (ret == ACL_SUCCESS) {
        double refSum = 0.0;
        for (size_t i = 0; i < inputHost.size(); ++i) {
            refSum += GoldenElem(inputHost[i], targetHost[i], static_cast<float>(delta));
        }
        LOG_PRINT("reduction=%s workspace=%llu out=[", name, static_cast<unsigned long long>(workspaceSize));
        for (size_t i = 0; i < outHost.size(); ++i) {
            LOG_PRINT(i + 1 == outHost.size() ? "%.6f" : "%.6f, ", outHost[i]);
        }
        if (scalarOut) {
            const double ref = (reduction == REDUCTION_MEAN) ? refSum / static_cast<double>(inputHost.size()) : refSum;
            LOG_PRINT("] ref=%.6f\n", ref);
        } else {
            LOG_PRINT("] ref=[");
            for (size_t i = 0; i < inputHost.size(); ++i) {
                const double r = GoldenElem(inputHost[i], targetHost[i], static_cast<float>(delta));
                LOG_PRINT(i + 1 == inputHost.size() ? "%.6f" : "%.6f, ", r);
            }
            LOG_PRINT("]\n");
        }
    }

    aclDestroyTensor(input);
    aclDestroyTensor(target);
    aclDestroyTensor(out);
    aclrtFree(inputDevice);
    aclrtFree(targetDevice);
    aclrtFree(outDevice);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}

} // namespace

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);

    // Spans both branches: |e| = 0.5 is quadratic, |e| = 2 is linear, and
    // |e| = 1 sits exactly on the knee, which aten's z < delta test puts on
    // the linear side. Both branches agree there.
    const std::vector<float> inputHost = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> targetHost(inputHost.size(), 0.0f);
    constexpr double delta = 1.0;

    int rc = 0;
    rc |= RunOne(stream, REDUCTION_NONE, "none", inputHost, targetHost, delta);
    rc |= RunOne(stream, REDUCTION_MEAN, "mean", inputHost, targetHost, delta);
    rc |= RunOne(stream, REDUCTION_SUM, "sum", inputHost, targetHost, delta);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc == ACL_SUCCESS ? 0 : 1;
}
