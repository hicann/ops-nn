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

// Minimal two-phase ACLNN call demo for aclnnApplyCenteredRMSProp.
//
// Scenario: shape=[16], dtype=float32. Performs one Centered RMSProp step
// in-place on (var, mg, ms, mom) and verifies against a CPU golden.
//
// Build & run via examples/run.sh.
//
// NOTE: The autogen ACLNN signature requires 4 placeholder "out" tensors
// (varOutOut/mgOutOut/msOutOut/momOutOut). To observe the in-place update,
// each placeholder MUST be created on the SAME device buffer as its Ref input.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_apply_centered_rms_prop.h"

namespace {

#define CHECK_ACL(expr)                                                       \
    do {                                                                      \
        aclError __err = (expr);                                              \
        if (__err != ACL_SUCCESS) {                                           \
            std::fprintf(stderr, "ACL error %d at %s:%d: %s\n", __err,        \
                         __FILE__, __LINE__, #expr);                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define CHECK_ACLNN(expr)                                                     \
    do {                                                                      \
        aclnnStatus __s = (expr);                                             \
        if (__s != ACL_SUCCESS) {                                             \
            std::fprintf(stderr, "ACLNN error %d at %s:%d: %s\n", __s,        \
                         __FILE__, __LINE__, #expr);                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

void* DevAlloc(size_t bytes) {
    void* p = nullptr;
    CHECK_ACL(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return p;
}

aclTensor* MakeTensor(void* ptr, const std::vector<int64_t>& shape,
                      aclDataType dt) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(
        shape.empty() ? nullptr : shape.data(), shape.size(), dt,
        strides.empty() ? nullptr : strides.data(), 0, ACL_FORMAT_ND,
        shape.empty() ? nullptr : shape.data(), shape.size(), ptr);
}

// CPU golden for one Centered RMSProp step (fp32).
struct Golden {
    std::vector<float> var, mg, ms, mom;
};
Golden ComputeGolden(std::vector<float> var, std::vector<float> mg,
                     std::vector<float> ms,  std::vector<float> mom,
                     const std::vector<float>& grad,
                     float lr, float rho, float momentum, float epsilon) {
    size_t n = var.size();
    for (size_t i = 0; i < n; ++i) {
        mg[i]  = rho * mg[i] + (1.0f - rho) * grad[i];
        ms[i]  = rho * ms[i] + (1.0f - rho) * grad[i] * grad[i];
        float denom = std::sqrt(ms[i] - mg[i] * mg[i] + epsilon);
        mom[i] = momentum * mom[i] + lr * grad[i] / denom;
        var[i] = var[i] - mom[i];
    }
    return {std::move(var), std::move(mg), std::move(ms), std::move(mom)};
}

bool Compare(const std::vector<float>& actual, const std::vector<float>& expect,
             const char* name) {
    constexpr float kRtol = 1e-4f;
    constexpr float kAtol = 1e-5f;
    int passed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual[i] - expect[i]);
        float thr  = kAtol + kRtol * std::fabs(expect[i]);
        if (diff <= thr) ++passed;
    }
    std::printf("Result (%s): %d/%zu passed\n", name, passed, actual.size());
    return passed == static_cast<int>(actual.size());
}

}  // namespace

int main() {
    // ---- 1. ACL init / device / stream ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ---- 2. Prepare host data ----
    const int64_t N = 16;
    std::vector<int64_t> shape = {N};

    std::vector<float> var(N), mg(N), ms(N), mom(N), grad(N);
    for (int64_t i = 0; i < N; ++i) {
        var[i]  = 0.10f * (i + 1);
        mg[i]   = 0.01f * (i + 1);
        ms[i]   = 0.20f + 0.01f * i;       // > 0
        mom[i]  = 0.05f * (i + 1);
        grad[i] = 0.02f * ((i % 5) - 2);   // small mixed-sign
    }
    float lr = 1e-2f, rho = 0.9f, momentum = 0.9f, epsilon = 1e-6f;

    // CPU golden BEFORE H2D (var/mg/ms/mom are in-place updated on device).
    auto golden = ComputeGolden(var, mg, ms, mom, grad, lr, rho, momentum, epsilon);

    // ---- 3. Device buffers ----
    size_t main_bytes = N * sizeof(float);
    size_t scal_bytes = sizeof(float);
    void* d_var  = DevAlloc(main_bytes);
    void* d_mg   = DevAlloc(main_bytes);
    void* d_ms   = DevAlloc(main_bytes);
    void* d_mom  = DevAlloc(main_bytes);
    void* d_grad = DevAlloc(main_bytes);
    void* d_lr   = DevAlloc(scal_bytes);
    void* d_rho  = DevAlloc(scal_bytes);
    void* d_momc = DevAlloc(scal_bytes);
    void* d_eps  = DevAlloc(scal_bytes);

    CHECK_ACL(aclrtMemcpy(d_var,  main_bytes, var.data(),  main_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_mg,   main_bytes, mg.data(),   main_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_ms,   main_bytes, ms.data(),   main_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_mom,  main_bytes, mom.data(),  main_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_grad, main_bytes, grad.data(), main_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_lr,   scal_bytes, &lr,         scal_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_rho,  scal_bytes, &rho,        scal_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_momc, scal_bytes, &momentum,   scal_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_eps,  scal_bytes, &epsilon,    scal_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 4. Build aclTensors ----
    std::vector<int64_t> scalar_shape;  // 0-D
    aclTensor* t_var  = MakeTensor(d_var,  shape,        ACL_FLOAT);
    aclTensor* t_mg   = MakeTensor(d_mg,   shape,        ACL_FLOAT);
    aclTensor* t_ms   = MakeTensor(d_ms,   shape,        ACL_FLOAT);
    aclTensor* t_mom  = MakeTensor(d_mom,  shape,        ACL_FLOAT);
    aclTensor* t_lr   = MakeTensor(d_lr,   scalar_shape, ACL_FLOAT);
    aclTensor* t_rho  = MakeTensor(d_rho,  scalar_shape, ACL_FLOAT);
    aclTensor* t_momc = MakeTensor(d_momc, scalar_shape, ACL_FLOAT);
    aclTensor* t_eps  = MakeTensor(d_eps,  scalar_shape, ACL_FLOAT);
    aclTensor* t_grad = MakeTensor(d_grad, shape,        ACL_FLOAT);
    // Out placeholders alias SAME device storage (in-place semantics).
    aclTensor* t_var_o = MakeTensor(d_var, shape, ACL_FLOAT);
    aclTensor* t_mg_o  = MakeTensor(d_mg,  shape, ACL_FLOAT);
    aclTensor* t_ms_o  = MakeTensor(d_ms,  shape, ACL_FLOAT);
    aclTensor* t_mom_o = MakeTensor(d_mom, shape, ACL_FLOAT);

    // ---- 5. Two-phase ACLNN call ----
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnApplyCenteredRMSPropGetWorkspaceSize(
        t_var, t_mg, t_ms, t_mom, t_lr, t_rho, t_momc, t_eps, t_grad,
        t_var_o, t_mg_o, t_ms_o, t_mom_o, &ws_size, &executor));

    void* ws_ptr = nullptr;
    if (ws_size > 0) ws_ptr = DevAlloc(ws_size);

    CHECK_ACLNN(aclnnApplyCenteredRMSProp(ws_ptr, ws_size, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ---- 6. D2H + compare ----
    std::vector<float> out_var(N), out_mg(N), out_ms(N), out_mom(N);
    CHECK_ACL(aclrtMemcpy(out_var.data(), main_bytes, d_var, main_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(out_mg.data(),  main_bytes, d_mg,  main_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(out_ms.data(),  main_bytes, d_ms,  main_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(out_mom.data(), main_bytes, d_mom, main_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool ok = true;
    ok &= Compare(out_var, golden.var, "var");
    ok &= Compare(out_mg,  golden.mg,  "mg");
    ok &= Compare(out_ms,  golden.ms,  "ms");
    ok &= Compare(out_mom, golden.mom, "mom");
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");

    // ---- 7. Cleanup ----
    for (aclTensor* t : {t_var, t_mg, t_ms, t_mom, t_lr, t_rho, t_momc, t_eps,
                          t_grad, t_var_o, t_mg_o, t_ms_o, t_mom_o}) {
        if (t) aclDestroyTensor(t);
    }
    if (ws_ptr) aclrtFree(ws_ptr);
    for (void* p : {d_var, d_mg, d_ms, d_mom, d_grad, d_lr, d_rho, d_momc, d_eps}) {
        aclrtFree(p);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return ok ? 0 : 1;
}
