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
 * @file test_aclnn_apply_proximal_adagrad.cpp
 * @brief ApplyProximalAdagrad 算子 ACLNN 调用示例（两段式接口）
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "aclnn_apply_proximal_adagrad.h"

#define CHECK_ACL(expr)                                                                 \
    do {                                                                                \
        auto _ret = (expr);                                                             \
        if (_ret != ACL_SUCCESS) {                                                      \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret                 \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
            goto cleanup;                                                               \
        }                                                                               \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// CPU Golden: 与算子语义完全一致
//   accum' = accum + grad^2
//   eta    = lr / sqrt(accum')
//   prox   = var - eta * grad
//   var'   = (l1==0) ? prox / (1 + eta*l2)
//                    : sign(prox) / (1 + eta*l2) * max(|prox| - eta*l1, 0)
// ---------------------------------------------------------------------------
inline void CpuGoldenStep(float& var, float& accum, float grad, float lr, float l1, float l2)
{
    accum = accum + grad * grad;
    float eta  = lr / std::sqrt(accum);
    float prox = var - eta * grad;
    float denom = 1.0f + eta * l2;
    if (l1 == 0.0f) {
        var = prox / denom;
    } else {
        float sgn  = (prox > 0.0f) ? 1.0f : ((prox < 0.0f) ? -1.0f : 0.0f);
        float mag  = std::fabs(prox) - eta * l1;
        if (mag < 0.0f) mag = 0.0f;
        var = sgn / denom * mag;
    }
}

inline bool ApproxEq(float a, float b, float atol = 1e-4f, float rtol = 1e-4f)
{
    float diff = std::fabs(a - b);
    return diff <= atol + rtol * std::fabs(b);
}

}  // namespace

int main()
{
    // =========================================================================
    // 1. 参数设置：shape=[16]、dtype=float32
    // =========================================================================
    constexpr int64_t ELEM_COUNT = 16;
    const int64_t shape[]   = {ELEM_COUNT};
    const int64_t strides[] = {1};
    constexpr int64_t ndim  = 1;

    // 标量 (lr, l1, l2) 用 numel=1 的 1-D Tensor 表示
    const int64_t scalarShape[]   = {1};
    const int64_t scalarStrides[] = {1};

    float hostVar[ELEM_COUNT] = {
        // 覆盖 +/- / 0 / 较大值
         1.00f, -1.00f,  0.50f, -0.50f,
         0.10f, -0.10f,  0.00f,  2.00f,
        -2.00f,  0.05f, -0.05f,  3.50f,
        -3.50f,  0.01f, -0.01f,  0.25f,
    };
    float hostAccum[ELEM_COUNT] = {
        // 必须非负（调用方保证）。覆盖小值 / 中等值。
        0.10f, 0.10f, 0.20f, 0.20f,
        0.05f, 0.05f, 0.10f, 1.00f,
        1.00f, 0.01f, 0.01f, 0.50f,
        0.50f, 0.30f, 0.30f, 0.40f,
    };
    float hostGrad[ELEM_COUNT] = {
         0.10f,  0.10f, -0.20f,  0.20f,
         0.30f, -0.30f,  0.10f, -0.50f,
         0.50f,  0.05f, -0.05f,  1.00f,
        -1.00f,  0.40f, -0.40f,  0.15f,
    };
    float hostLr[1] = {0.1f};
    float hostL1[1] = {0.01f};   // l1 > 0 → 走 HAS_L1=1 分支（含 abs/max 阈值收缩）
    float hostL2[1] = {0.0f};

    // CPU Golden（独立副本，避免污染输入）
    float goldVar[ELEM_COUNT];
    float goldAccum[ELEM_COUNT];
    std::memcpy(goldVar,   hostVar,   sizeof(hostVar));
    std::memcpy(goldAccum, hostAccum, sizeof(hostAccum));
    for (int i = 0; i < ELEM_COUNT; ++i) {
        CpuGoldenStep(goldVar[i], goldAccum[i], hostGrad[i], hostLr[0], hostL1[0], hostL2[0]);
    }

    // =========================================================================
    // 2. ACL 初始化
    // =========================================================================
    int32_t ret = 1;
    aclrtStream stream = nullptr;
    void *devVar = nullptr, *devAccum = nullptr;
    void *devLr = nullptr, *devL1 = nullptr, *devL2 = nullptr;
    void *devGrad = nullptr;
    void *workspace = nullptr;
    aclTensor *tVar = nullptr, *tAccum = nullptr;
    aclTensor *tLr = nullptr, *tL1 = nullptr, *tL2 = nullptr;
    aclTensor *tGrad = nullptr;
    aclTensor *tVarOut = nullptr, *tAccumOut = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&stream));

    // =========================================================================
    // 3. 设备内存分配 & 输入数据拷贝 Host→Device
    // =========================================================================
    {
        const size_t vecBytes    = ELEM_COUNT * sizeof(float);
        const size_t scalarBytes = 1          * sizeof(float);

        CHECK_ACL(aclrtMalloc(&devVar,   vecBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devAccum, vecBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devGrad,  vecBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devLr,    scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devL1,    scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devL2,    scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST));

        CHECK_ACL(aclrtMemcpy(devVar,   vecBytes, hostVar,   vecBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devAccum, vecBytes, hostAccum, vecBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devGrad,  vecBytes, hostGrad,  vecBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devLr,    scalarBytes, hostLr, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devL1,    scalarBytes, hostL1, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(devL2,    scalarBytes, hostL2, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        // =====================================================================
        // 4. 创建 aclTensor
        //    var/accum/grad: FLOAT, ND, [16]
        //    lr/l1/l2:        FLOAT, ND, [1]   (numel=1 1-D 标量)
        //    varOutOut / accumOutOut: 绑定与 var / accum 同一 Device 存储 → 观察 inplace
        // =====================================================================
        tVar    = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                  ACL_FORMAT_ND, shape, ndim, devVar);
        tAccum  = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                  ACL_FORMAT_ND, shape, ndim, devAccum);
        tGrad   = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                  ACL_FORMAT_ND, shape, ndim, devGrad);
        tLr     = aclCreateTensor(scalarShape, 1, ACL_FLOAT, scalarStrides, 0,
                                  ACL_FORMAT_ND, scalarShape, 1, devLr);
        tL1     = aclCreateTensor(scalarShape, 1, ACL_FLOAT, scalarStrides, 0,
                                  ACL_FORMAT_ND, scalarShape, 1, devL1);
        tL2     = aclCreateTensor(scalarShape, 1, ACL_FLOAT, scalarStrides, 0,
                                  ACL_FORMAT_ND, scalarShape, 1, devL2);
        // 占位输出复用同一 Device buffer，调用结束后通过 devVar / devAccum 直接读取
        tVarOut   = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                    ACL_FORMAT_ND, shape, ndim, devVar);
        tAccumOut = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                    ACL_FORMAT_ND, shape, ndim, devAccum);

        if (!tVar || !tAccum || !tLr || !tL1 || !tL2 || !tGrad || !tVarOut || !tAccumOut) {
            std::cerr << "aclCreateTensor failed" << std::endl;
            goto cleanup;
        }

        // =====================================================================
        // 5. 调用 aclnnApplyProximalAdagrad（两段式接口）
        // =====================================================================
        uint64_t       workspaceSize = 0;
        aclOpExecutor *executor      = nullptr;

        CHECK_ACL(aclnnApplyProximalAdagradGetWorkspaceSize(
            tVar, tAccum, tLr, tL1, tL2, tGrad, tVarOut, tAccumOut,
            &workspaceSize, &executor));

        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        CHECK_ACL(aclnnApplyProximalAdagrad(workspace, workspaceSize, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // =====================================================================
        // 6. 读取 inplace 结果（var / accum 已被更新）& 精度验证
        // =====================================================================
        float npuVar[ELEM_COUNT]   = {};
        float npuAccum[ELEM_COUNT] = {};
        const size_t vecBytesOut = ELEM_COUNT * sizeof(float);
        CHECK_ACL(aclrtMemcpy(npuVar,   vecBytesOut, devVar,   vecBytesOut, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(npuAccum, vecBytesOut, devAccum, vecBytesOut, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "ApplyProximalAdagrad Example (shape: [16], dtype: float32)" << std::endl;
        std::cout << "  scalars: lr=" << hostLr[0]
                  << "  l1=" << hostL1[0]
                  << "  l2=" << hostL2[0] << std::endl;
        std::cout << "  formula: accum' = accum + grad^2" << std::endl;
        std::cout << "           eta    = lr / sqrt(accum')" << std::endl;
        std::cout << "           prox   = var - eta * grad" << std::endl;
        std::cout << "           var'   = sign(prox) / (1 + eta*l2) * max(|prox| - eta*l1, 0)" << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::printf("  %4s | %10s | %10s | %10s | %10s | %10s | %s\n",
                    "Idx", "var", "accum", "grad", "var_npu", "var_gold", "Status");
        std::cout << "-------------------------------------------------------------------------------" << std::endl;

        int passVar = 0, passAccum = 0;
        for (int i = 0; i < ELEM_COUNT; ++i) {
            bool okV = ApproxEq(npuVar[i],   goldVar[i]);
            bool okA = ApproxEq(npuAccum[i], goldAccum[i]);
            if (okV) ++passVar;
            if (okA) ++passAccum;
            std::printf("  %4d | %10.5f | %10.5f | %10.5f | %10.5f | %10.5f | %s/%s\n",
                        i, hostVar[i], hostAccum[i], hostGrad[i],
                        npuVar[i], goldVar[i],
                        okV ? "PASS" : "FAIL",
                        okA ? "PASS" : "FAIL");
        }

        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << "Result (var):   " << passVar   << "/" << ELEM_COUNT << " passed" << std::endl;
        std::cout << "Result (accum): " << passAccum << "/" << ELEM_COUNT << " passed" << std::endl;
        if (passVar == ELEM_COUNT && passAccum == ELEM_COUNT) {
            std::cout << "ALL PASS" << std::endl;
            ret = 0;
        } else {
            std::cout << "FAILED" << std::endl;
            ret = 1;
        }
    }

    // =========================================================================
    // 7. 资源释放
    // =========================================================================
cleanup:
    if (tVar)      aclDestroyTensor(tVar);
    if (tAccum)    aclDestroyTensor(tAccum);
    if (tLr)       aclDestroyTensor(tLr);
    if (tL1)       aclDestroyTensor(tL1);
    if (tL2)       aclDestroyTensor(tL2);
    if (tGrad)     aclDestroyTensor(tGrad);
    if (tVarOut)   aclDestroyTensor(tVarOut);
    if (tAccumOut) aclDestroyTensor(tAccumOut);
    if (workspace) aclrtFree(workspace);
    if (devVar)    aclrtFree(devVar);
    if (devAccum)  aclrtFree(devAccum);
    if (devGrad)   aclrtFree(devGrad);
    if (devLr)     aclrtFree(devLr);
    if (devL1)     aclrtFree(devL1);
    if (devL2)     aclrtFree(devL2);
    if (stream)    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ret;
}
