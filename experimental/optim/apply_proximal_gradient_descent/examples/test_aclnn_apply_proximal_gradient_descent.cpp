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
 * @file test_aclnn_apply_proximal_gradient_descent.cpp
 * @brief ApplyProximalGradientDescent 算子 aclnn 两段式调用示例
 *
 * 公式：
 *   prox_v = var - alpha * delta
 *   varOut = sign(prox_v) / (1 + alpha * l2) * max(|prox_v| - alpha * l1, 0)
 *
 * 本示例：
 *   1. 构造一个 [2, 3] 的 FP32 var/delta，alpha=0.01, l1=0.001, l2=0.01；
 *   2. 调用 aclnn 两段式接口在 NPU 上执行；
 *   3. 把 varOut 拷回 Host 与 CPU Golden 比对；
 *   4. 打印结果，简单数值检查 (atol=1e-5)。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_apply_proximal_gradient_descent.h"

#define CHECK_RET(cond, msg)                                              \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "[FAIL] " << msg << std::endl;                   \
            return -1;                                                    \
        }                                                                 \
    } while (0)

#define CHECK_ACL(expr, msg)                                              \
    do {                                                                  \
        auto _ret = (expr);                                               \
        if (_ret != ACL_SUCCESS) {                                        \
            std::cerr << "[FAIL] " << msg << ", ret=" << _ret             \
                      << std::endl;                                       \
            return -1;                                                    \
        }                                                                 \
    } while (0)

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
    if (shape.empty()) return {};
    std::vector<int64_t> s(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i) {
        s[i] = shape[i + 1] * s[i + 1];
    }
    return s;
}

template <typename T>
static int CreateAclTensor(const std::vector<T> &host,
                           const std::vector<int64_t> &shape,
                           aclDataType dtype, void **devAddr,
                           aclTensor **tensor) {
    size_t bytes = host.size() * sizeof(T);
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(*devAddr, bytes, host.data(), bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ret;
    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                              strides.empty() ? nullptr : strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *devAddr);
    return ACL_SUCCESS;
}

static void GoldenFp32(const float *var, const float *delta, float alpha,
                       float l1, float l2, size_t n, float *out) {
    const double a = (double)alpha, l1d = (double)l1, l2d = (double)l2;
    const double denom = 1.0 + a * l2d;
    for (size_t i = 0; i < n; ++i) {
        double prox = (double)var[i] - a * (double)delta[i];
        double sgn = (prox > 0.0) - (prox < 0.0);
        double shrink = std::fabs(prox) - a * l1d;
        if (shrink < 0.0) shrink = 0.0;
        out[i] = (float)(sgn * shrink / denom);
    }
}

int main() {
    // ----------------------------------------------------------------------
    // 1. 初始化 ACL
    // ----------------------------------------------------------------------
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(deviceId), "aclrtSetDevice");
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    // ----------------------------------------------------------------------
    // 2. 构造输入数据
    //    var [2,3], delta [2,3], alpha/l1/l2 shape=[1] 标量
    // ----------------------------------------------------------------------
    std::vector<int64_t> varShape = {2, 3};
    std::vector<int64_t> scalarShape = {1};
    std::vector<float> varHost   = { 0.5f, -0.3f,  0.1f, -0.8f,  0.2f,  0.4f};
    std::vector<float> deltaHost = { 0.1f,  0.1f,  0.1f,  0.1f,  0.1f,  0.1f};
    std::vector<float> alphaHost = {0.01f};
    std::vector<float> l1Host    = {0.001f};
    std::vector<float> l2Host    = {0.01f};
    std::vector<float> varOutHost(varHost.size(), 0.f);

    // ----------------------------------------------------------------------
    // 3. 创建 device tensor
    // ----------------------------------------------------------------------
    void *varDev = nullptr, *deltaDev = nullptr, *varOutDev = nullptr;
    void *alphaDev = nullptr, *l1Dev = nullptr, *l2Dev = nullptr;
    aclTensor *varT = nullptr, *deltaT = nullptr, *varOutT = nullptr;
    aclTensor *alphaT = nullptr, *l1T = nullptr, *l2T = nullptr;

    CHECK_ACL(CreateAclTensor(varHost,    varShape,    ACL_FLOAT, &varDev,    &varT),
              "create var");
    CHECK_ACL(CreateAclTensor(deltaHost,  varShape,    ACL_FLOAT, &deltaDev,  &deltaT),
              "create delta");
    CHECK_ACL(CreateAclTensor(varOutHost, varShape,    ACL_FLOAT, &varOutDev, &varOutT),
              "create varOut");
    CHECK_ACL(CreateAclTensor(alphaHost,  scalarShape, ACL_FLOAT, &alphaDev,  &alphaT),
              "create alpha");
    CHECK_ACL(CreateAclTensor(l1Host,     scalarShape, ACL_FLOAT, &l1Dev,     &l1T),
              "create l1");
    CHECK_ACL(CreateAclTensor(l2Host,     scalarShape, ACL_FLOAT, &l2Dev,     &l2T),
              "create l2");

    // ----------------------------------------------------------------------
    // 4. aclnn 两段式调用
    //    第一段: 计算 workspace 大小 + 构造 executor
    //    第二段: 提交 stream 执行
    // ----------------------------------------------------------------------
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnApplyProximalGradientDescentGetWorkspaceSize(
                  varT, alphaT, l1T, l2T, deltaT, varOutT, &wsSize, &executor),
              "aclnnApplyProximalGradientDescentGetWorkspaceSize");

    void *wsDev = nullptr;
    if (wsSize > 0) {
        CHECK_ACL(aclrtMalloc(&wsDev, wsSize, ACL_MEM_MALLOC_HUGE_FIRST),
                  "alloc workspace");
    }

    CHECK_ACL(aclnnApplyProximalGradientDescent(wsDev, wsSize, executor, stream),
              "aclnnApplyProximalGradientDescent");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    // ----------------------------------------------------------------------
    // 5. 回拷 varOut 到 Host
    // ----------------------------------------------------------------------
    size_t varBytes = varHost.size() * sizeof(float);
    CHECK_ACL(aclrtMemcpy(varOutHost.data(), varBytes, varOutDev, varBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "D2H varOut");

    // ----------------------------------------------------------------------
    // 6. 计算 CPU Golden 并打印对比
    // ----------------------------------------------------------------------
    std::vector<float> golden(varHost.size());
    GoldenFp32(varHost.data(), deltaHost.data(), alphaHost[0], l1Host[0],
               l2Host[0], varHost.size(), golden.data());

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "=========================================================="
              << std::endl;
    std::cout << "ApplyProximalGradientDescent example (var[2,3], FP32)"
              << std::endl;
    std::cout << "alpha=" << alphaHost[0] << ", l1=" << l1Host[0]
              << ", l2=" << l2Host[0] << std::endl;
    std::cout << "----------------------------------------------------------"
              << std::endl;
    std::cout << "idx |       var |     delta |    golden |    npuOut |  diff"
              << std::endl;
    bool ok = true;
    const double atol = 1e-5;
    for (size_t i = 0; i < varHost.size(); ++i) {
        double diff = std::fabs((double)varOutHost[i] - (double)golden[i]);
        if (diff > atol) ok = false;
        std::cout << std::setw(3) << i << " | " << std::setw(9) << varHost[i]
                  << " | " << std::setw(9) << deltaHost[i] << " | "
                  << std::setw(9) << golden[i] << " | " << std::setw(9)
                  << varOutHost[i] << " | " << std::scientific
                  << std::setprecision(2) << diff << std::fixed
                  << std::setprecision(7) << std::endl;
    }
    std::cout << "=========================================================="
              << std::endl;
    std::cout << "Result: " << (ok ? "PASS" : "FAIL") << " (atol=" << atol
              << ")" << std::endl;

    // ----------------------------------------------------------------------
    // 7. 释放资源
    // ----------------------------------------------------------------------
    aclDestroyTensor(varT);
    aclDestroyTensor(deltaT);
    aclDestroyTensor(varOutT);
    aclDestroyTensor(alphaT);
    aclDestroyTensor(l1T);
    aclDestroyTensor(l2T);
    aclrtFree(varDev);
    aclrtFree(deltaDev);
    aclrtFree(varOutDev);
    aclrtFree(alphaDev);
    aclrtFree(l1Dev);
    aclrtFree(l2Dev);
    if (wsDev) aclrtFree(wsDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return ok ? 0 : 1;
}
