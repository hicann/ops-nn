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

/*!
 * @file test_aclnn_apply_rms_prop.cpp
 * @brief ApplyRmsProp 算子 aclnn 两段式调用示例（Inplace 语义）
 *
 * 功能：演示 aclnnApplyRmsProp 在 Ascend 950 上对 var / ms / mom 做 inplace 更新。
 *
 * 公式（逐元素）：
 *   ms_new  = ms + (1 - rho) * (grad^2 - ms)
 *   mom_new = momentum * mom + lr * grad / sqrt(epsilon + ms_new)
 *   var_new = var - mom_new
 *
 * 编译运行：参见同目录 run.sh（bash run.sh --eager）
 *
 * 运行前置条件：
 *   1. 已执行 `bash ../build.sh --soc=ascend950` 构建自定义算子包
 *   2. 已安装 custom_opp_*.run（vendor 名：apply_rms_prop_custom）
 *   3. 已 source CANN 环境（ASCEND_HOME_PATH 可用）
 *
 * Mock 模式（USE_MOCK_ACLNN）：跳过 NPU 调用，仅演示 CPU Golden 计算与数据打印，
 * 便于在无 NPU 环境下快速验证示例代码逻辑。
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_apply_rms_prop.h"
#endif

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

// -----------------------------------------------------------------------------
// 工具函数
// -----------------------------------------------------------------------------
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape) size *= d;
    return size;
}

// CPU Golden（fp32，用于与 NPU 结果比对，或 Mock 模式下直接演示计算）
void ComputeApplyRmsPropGoldenFP32(
    const std::vector<float>& var_in,
    const std::vector<float>& ms_in,
    const std::vector<float>& mom_in,
    float lr, float rho, float momentum, float epsilon,
    const std::vector<float>& grad,
    std::vector<float>& var_out,
    std::vector<float>& ms_out,
    std::vector<float>& mom_out)
{
    size_t n = var_in.size();
    var_out.resize(n); ms_out.resize(n); mom_out.resize(n);
    const double one_minus_rho = 1.0 - static_cast<double>(rho);
    for (size_t i = 0; i < n; ++i) {
        double g = grad[i];
        double ms_new = ms_in[i] + one_minus_rho * (g * g - ms_in[i]);
        double denom = std::sqrt(epsilon + ms_new);
        double inv_sqrt = (denom > 0.0) ? (1.0 / denom) : 0.0;
        double mom_new = static_cast<double>(momentum) * mom_in[i] +
                         static_cast<double>(lr) * g * inv_sqrt;
        double var_new = var_in[i] - mom_new;
        ms_out[i] = static_cast<float>(ms_new);
        mom_out[i] = static_cast<float>(mom_new);
        var_out[i] = static_cast<float>(var_new);
    }
}

#ifndef USE_MOCK_ACLNN
// -----------------------------------------------------------------------------
// Real 模式：ACL 初始化
// -----------------------------------------------------------------------------
int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 通用 aclTensor 创建（fp32 示例）
int CreateAclTensor(const std::vector<float>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(float);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // strides（rank=0 时 strides 为空）
    std::vector<int64_t> strides;
    if (!shape.empty()) {
        strides.assign(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.empty() ? nullptr : strides.data(),
                              0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
#endif  // !USE_MOCK_ACLNN

// -----------------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------------
int main()
{
    LOG_PRINT("============================================\n");
    LOG_PRINT("ApplyRmsProp aclnn 调用示例 (Inplace, fp32)\n");
    LOG_PRINT("============================================\n");

    // ------------------ 1. 构造输入数据（fp32） ------------------
    std::vector<int64_t> shape = {16};  // 小 shape 便于打印
    int64_t n = GetShapeSize(shape);

    std::vector<float> var_host(n), ms_host(n), mom_host(n), grad_host(n);
    for (int64_t i = 0; i < n; ++i) {
        var_host[i]  = static_cast<float>(1.0 + 0.1 * i);      // 1.0 ~ 2.5
        ms_host[i]   = static_cast<float>(0.1 + 0.01 * i);     // 0.1 ~ 0.25
        mom_host[i]  = 0.0f;
        grad_host[i] = static_cast<float>(0.5 - 0.05 * (i % 8));
    }
    const float lr = 0.01f;
    const float rho = 0.9f;
    const float momentum = 0.0f;
    const float epsilon = 1e-7f;

    LOG_PRINT("输入 shape = [%ld]，dtype = fp32\n", static_cast<long>(shape[0]));
    LOG_PRINT("scalar 参数：lr=%.4f rho=%.4f momentum=%.4f epsilon=%.0e\n\n",
              lr, rho, momentum, epsilon);

    // ------------------ 2. 计算 CPU Golden（用于对比 / Mock 展示） ------------------
    std::vector<float> var_gold, ms_gold, mom_gold;
    ComputeApplyRmsPropGoldenFP32(var_host, ms_host, mom_host,
                                   lr, rho, momentum, epsilon, grad_host,
                                   var_gold, ms_gold, mom_gold);

#ifdef USE_MOCK_ACLNN
    // --- Mock 模式：直接打印 CPU Golden，验证示例代码逻辑 ---
    LOG_PRINT("[Mock 模式] 未调用 NPU，直接展示 CPU Golden 结果：\n");
    for (int64_t i = 0; i < n; ++i) {
        LOG_PRINT("  i=%ld  var=%.6f -> %.6f   ms=%.6f -> %.6f   mom=%.6f -> %.6f\n",
                  static_cast<long>(i),
                  var_host[i], var_gold[i],
                  ms_host[i],  ms_gold[i],
                  mom_host[i], mom_gold[i]);
    }
    LOG_PRINT("\n[Mock 模式] 示例流程演示完成。\n");
    return 0;
#else
    // ------------------ 3. ACL 初始化 ------------------
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // ------------------ 4. 构造 aclTensor（方案 A：4 tensor + 4 double attr + 3 输出 tensor） ------------------
    aclTensor *var_t = nullptr, *ms_t = nullptr, *mom_t = nullptr, *grad_t = nullptr;
    aclTensor *var_out_t = nullptr, *ms_out_t = nullptr, *mom_out_t = nullptr;
    void *var_dev = nullptr, *ms_dev = nullptr, *mom_dev = nullptr, *grad_dev = nullptr;
    void *var_out_dev = nullptr, *ms_out_dev = nullptr, *mom_out_dev = nullptr;

    ret = CreateAclTensor(var_host, shape, &var_dev, ACL_FLOAT, &var_t);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(ms_host, shape, &ms_dev, ACL_FLOAT, &ms_t);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(mom_host, shape, &mom_dev, ACL_FLOAT, &mom_t);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(grad_host, shape, &grad_dev, ACL_FLOAT, &grad_t);
    CHECK_RET(ret == 0, return ret);

    // 3 个输出 tensor（用 zero 初始化）
    std::vector<float> out_zero(n, 0.0f);
    ret = CreateAclTensor(out_zero, shape, &var_out_dev, ACL_FLOAT, &var_out_t);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(out_zero, shape, &ms_out_dev,  ACL_FLOAT, &ms_out_t);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(out_zero, shape, &mom_out_dev, ACL_FLOAT, &mom_out_t);
    CHECK_RET(ret == 0, return ret);

    // ------------------ 5. 调用 aclnnApplyRmsProp 第一段接口 ------------------
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnApplyRmsPropGetWorkspaceSize(
        var_t, ms_t, mom_t, grad_t,
        static_cast<double>(lr), static_cast<double>(rho),
        static_cast<double>(momentum), static_cast<double>(epsilon),
        var_out_t, ms_out_t, mom_out_t,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnApplyRmsPropGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // ------------------ 6. 根据 workspaceSize 申请 device 内存 ------------------
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // ------------------ 7. 调用 aclnnApplyRmsProp 第二段接口（执行 Kernel） ------------------
    ret = aclnnApplyRmsProp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnApplyRmsProp failed. ERROR: %d\n", ret); return ret);

    // ------------------ 8. Stream 同步 ------------------
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // ------------------ 9. 将输出拷回 Host（方案 A：从 var_out_dev/ms_out_dev/mom_out_dev） ------------------
    std::vector<float> var_out(n), ms_out(n), mom_out(n);
    aclrtMemcpy(var_out.data(), n * sizeof(float), var_out_dev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(ms_out.data(),  n * sizeof(float), ms_out_dev,  n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(mom_out.data(), n * sizeof(float), mom_out_dev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // ------------------ 10. 打印结果并与 CPU Golden 简单比对 ------------------
    LOG_PRINT("[Real 模式] inplace 更新后的 var / ms / mom（NPU 结果，与 CPU Golden 对照）：\n");
    int mismatch = 0;
    for (int64_t i = 0; i < n; ++i) {
        double dv = std::abs(var_out[i] - var_gold[i]);
        double dm = std::abs(ms_out[i]  - ms_gold[i]);
        double dn = std::abs(mom_out[i] - mom_gold[i]);
        if (dv > 1e-3 || dm > 1e-3 || dn > 1e-3) mismatch++;
        LOG_PRINT("  i=%ld  var=%.6f (gold %.6f)  ms=%.6f (gold %.6f)  mom=%.6f (gold %.6f)\n",
                  static_cast<long>(i),
                  var_out[i], var_gold[i],
                  ms_out[i],  ms_gold[i],
                  mom_out[i], mom_gold[i]);
    }
    LOG_PRINT("\n不一致元素数 (|delta| > 1e-3): %d / %ld\n",
              mismatch, static_cast<long>(n));

    // ------------------ 11. 资源释放 ------------------
    aclDestroyTensor(var_t); aclDestroyTensor(ms_t);
    aclDestroyTensor(mom_t); aclDestroyTensor(grad_t);
    aclDestroyTensor(var_out_t); aclDestroyTensor(ms_out_t); aclDestroyTensor(mom_out_t);
    aclrtFree(var_dev); aclrtFree(ms_dev); aclrtFree(mom_dev); aclrtFree(grad_dev);
    aclrtFree(var_out_dev); aclrtFree(ms_out_dev); aclrtFree(mom_out_dev);
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\nApplyRmsProp aclnn 调用示例执行完成。\n");
    return mismatch == 0 ? 0 : 1;
#endif  // USE_MOCK_ACLNN
}
